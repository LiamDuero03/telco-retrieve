"""
Evaluation orchestrator for Open Telco RAG.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from open_telco_rag.evaluation._claude import claude_call
from open_telco_rag.evaluation.questions import EvalQuestion, load_questions
from open_telco_rag.evaluation.results import (
    QuestionResult, CorpusResult, aggregate, save_results, print_summary,
)
from open_telco_rag.evaluation.metrics.generation import (
    faithfulness, answer_relevancy,
)
from open_telco_rag.evaluation.metrics.retrieval import (
    cosine_sim_stats, source_coverage, canonical_source_hit,
)
from open_telco_rag.evaluation.metrics.ground_truth import (
    context_recall, answer_correctness,
)

def _build_generator(llm_call: Callable[[str, int], str], model_name: str | None = None):
    """Wrap an llm_call callable as a generator for pipeline.generate()."""
    from open_telco_rag.generators.openrouter import OpenRouterGenerator
    
    # Use default if none provided
    if not model_name:
        model_name = "meta-llama/llama-3.1-8b-instruct"
        
    return OpenRouterGenerator(model=model_name)

def _build_baseline_generator(llm_call: Callable[[str, int], str]):
    """Generator that answers from LLM knowledge only — no context chunks."""
    class _BaselineGenerator:
        def generate(self, question: str, results: list, source_type: str = "telecom") -> dict:
            prompt = (
                f"You are a {source_type} technical expert. "
                "Answer the following question based on your knowledge. "
                "Be precise and technical.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            answer = llm_call(prompt, max_tokens=1000)
            return {
                "answer":           answer,
                "sources":          [],
                "chunks_used":      0,
                "retrieved_chunks": [],
            }
    return _BaselineGenerator()

def evaluate_question(
    q:             EvalQuestion,
    store:         Any,
    embedder:      Any,
    top_k:         int   = 5,
    sim_threshold: float = 0.5,
    llm_call:      Callable[[str, int], str] = claude_call,
    rag_llm_call:  Callable[[str, int], str] | None = None,
    baseline:      bool  = False,
    reranker:      Any   = None,
    rerank_top_k:  int   = 5,
    debug:         bool  = False,
    rag_model_name: str | None = None,
    inject_metadata: bool = False, # NEW PARAMETER
) -> QuestionResult:
    """Run all applicable metrics for one question."""
    from open_telco_rag.core.pipeline import search, generate
    
    # Rewrite the question if metadata injection is enabled and a source exists
    active_question = q.question
    if inject_metadata and q.canonical_source:
        active_question = f"According to {q.corpus.upper()} document {q.canonical_source}, {q.question}"

    result = QuestionResult(
        question=active_question, corpus=q.corpus,
        answer="", sources=[], chunks_used=0,
    )

    try:
        start_time = time.perf_counter()
        if baseline:
            retrieved = []
            generator = _build_baseline_generator(rag_llm_call or llm_call)
            # Use active_question instead of q.question
            gen       = generate(active_question, retrieved,
                                 source_type=q.corpus,
                                 generator=generator)
        else:
            # Use active_question instead of q.question
            retrieved = search(active_question, store=store, embedder=embedder,
                                top_k=top_k, reranker=reranker, rerank_top_k=rerank_top_k)
            generator = _build_generator(rag_llm_call or llm_call, model_name=rag_model_name)
            # Use active_question instead of q.question
            gen       = generate(active_question, retrieved,
                                 source_type=q.corpus,
                                 generator=generator)
        
        # Extract metrics from generation result
        result.latency = time.perf_counter() - start_time
        usage = gen.get("usage", {})
        
        result.prompt_tokens = usage.get("prompt_tokens", 0)
        result.completion_tokens = usage.get("completion_tokens", 0)
        result.total_tokens = usage.get("total_tokens", result.prompt_tokens + result.completion_tokens)

        # Extract nested prompt details (Cached input tokens)
        prompt_details = usage.get("prompt_tokens_details", {}) or {}
        result.cached_tokens = prompt_details.get("cached_tokens", 0)

        # Extract nested completion details (DeepSeek R1 / o1 thinking tokens)
        comp_details = usage.get("completion_tokens_details", {}) or {}
        result.reasoning_tokens = comp_details.get("reasoning_tokens", 0)

        # Extract exact OpenRouter cost
        result.cost = usage.get("cost", 0.0)

        answer         = gen.get("answer", "")
        result.answer  = answer
        result.sources = gen.get("sources", [])
        result.chunks_used = gen.get("chunks_used", len(retrieved))

        if debug:
            mode = "BASELINE" if baseline else "RAG"
            print(f"\n  [DEBUG] [{mode}] question: {active_question[:80]}")
            print(f"  [DEBUG] [{mode}] answer preview: {answer[:200]}")

        # Use active_question for relevancy scoring
        result.faithfulness     = faithfulness(answer, retrieved,
                                               llm_call=llm_call, debug=debug)
        result.answer_relevancy = answer_relevancy(active_question, answer, embedder,
                                                   llm_call=llm_call, debug=debug)

        if not baseline:
            mean_s, min_s, max_s  = cosine_sim_stats(retrieved)
            result.mean_similarity = mean_s
            result.min_similarity  = min_s
            result.max_similarity  = max_s
            result.source_coverage = source_coverage(retrieved)
            result.canonical_hit   = canonical_source_hit(retrieved, q.canonical_source)

        if q.reference_answer:
            result.context_recall     = context_recall(
                q.reference_answer, retrieved, llm_call=llm_call, debug=debug)
            result.answer_correctness = answer_correctness(
                answer, q.reference_answer, embedder,
                llm_call=llm_call, debug=debug)

    except Exception as e:
        result.error = str(e)
        result.latency = 0.0
        result.prompt_tokens = 0
        result.completion_tokens = 0

    return result



def evaluate_corpus(
    corpus:        str,
    store:         Any,
    embedder:      Any,
    questions:     list[EvalQuestion] | None = None,
    model:         str   = "minilm",
    top_k:         int   = 5,
    sim_threshold: float = 0.5,
    delay:         float = 1.0,
    output_dir:    Path | None = None,
    llm_call:      Callable[[str, int], str] = claude_call,
    rag_llm_call:  Callable[[str, int], str] | None = None,
    include_baseline: bool = False,
    reranker:      Any   = None,
    rerank_top_k:  int   = 5,
    debug:         bool  = False,
    rag_generator_model: str | None = None,
    inject_metadata: bool = False,
) -> tuple[CorpusResult, CorpusResult | None]:
    """Run the full evaluation suite for one corpus.

    Returns:
        (rag_result, baseline_result) — baseline_result is None if
        include_baseline=False.
    """
    # 1. Track the source questions path or use the default
    # You can infer or pass down the name of the file loaded:
    questions_source_name = "default"
    if questions:
        # If questions were passed manually
        questions_source_name = "custom_list"
    else:
        # If default questions file was loaded (e.g., configs/eval_questions/3gpp.json)
        questions_source_name = f"{corpus}.json"
        
    qs = questions or load_questions(corpus)

    print(f"\nEvaluating {corpus.upper()} — {len(qs)} questions | "
          f"model={model} | top-k={top_k}"
          + (" | +BASELINE" if include_baseline else "")
          + (" | DEBUG ON" if debug else ""))
    
    # Default fallback string if none is provided down the pipe
    resolved_rag_model = rag_generator_model or "meta-llama/llama-3.1-8b-instruct"

    # ── RAG pass ──────────────────────────────────────────────────────────────
    rag_results = []
    for i, q in enumerate(qs, 1):
        display_q = q.question
        if inject_metadata and q.canonical_source:
            display_q = f"According to {corpus.upper()} document {q.canonical_source}, {q.question}"
        print(f"  [RAG {i}/{len(qs)}] {display_q[:65]}...")
        qr = evaluate_question(q, store, embedder, top_k, sim_threshold,
                               llm_call=llm_call, rag_llm_call=rag_llm_call,
                               baseline=False, reranker=reranker,
                               rerank_top_k=rerank_top_k, debug=debug,
                               rag_model_name=resolved_rag_model,
                               inject_metadata=inject_metadata)
        rag_results.append(qr)
        if qr.error:
            print(f"    [ERROR] {qr.error}")
        else:
            print(f"    faithful={qr.faithfulness:.2f}  "
                  f"relevancy={qr.answer_relevancy:.2f}  "
                  f"sim={qr.mean_similarity:.3f}  "
                  f"recall={qr.context_recall:.2f}  "
                  f"correctness={qr.answer_correctness:.2f}")
        if delay > 0 and i < len(qs):
            time.sleep(delay)

    rag_result = CorpusResult(
        corpus=corpus, model=model, top_k=top_k,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        questions_file=questions_source_name,
        question_results=rag_results,
        **aggregate(rag_results),
    )

    print_summary(rag_result)

    if output_dir:
        save_results(rag_result, Path(output_dir))

    # ── Baseline pass (optional) ──────────────────────────────────────────────
    baseline_result = None
    if include_baseline:
        print(f"\n  Running BASELINE pass ({len(qs)} questions, no retrieval)…")
        bl_results = []
        for i, q in enumerate(qs, 1):
            print(f"  [BL  {i}/{len(qs)}] {q.question[:65]}...")
            qr = evaluate_question(q, store, embedder, top_k, sim_threshold,
                                   llm_call=llm_call, rag_llm_call=rag_llm_call,
                                   baseline=True, debug=debug)
            bl_results.append(qr)
            if qr.error:
                print(f"    [ERROR] {qr.error}")
            else:
                print(f"    faithful={qr.faithfulness:.2f}  "
                      f"relevancy={qr.answer_relevancy:.2f}  "
                      f"recall={qr.context_recall:.2f}  "
                      f"correctness={qr.answer_correctness:.2f}")
            if delay > 0 and i < len(qs):
                time.sleep(delay)

        baseline_result = CorpusResult(
            corpus=corpus, model=model, top_k=0,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            question_results=bl_results,
            **aggregate(bl_results),
        )

        print(f"\n  BASELINE summary:")
        print_summary(baseline_result)

        if output_dir:
            save_results(baseline_result, Path(output_dir) / "baseline")

    return rag_result, baseline_result