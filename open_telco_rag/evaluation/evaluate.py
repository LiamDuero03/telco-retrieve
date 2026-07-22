"""
Evaluation orchestrator for Open Telco RAG.
"""

from __future__ import annotations

import time
import tiktoken
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

def _local_token_count(text: str) -> int:
    """Calculates tokens locally using the cl100k_base encoding."""
    if not text: return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback approximation if tiktoken fails
        return int(len(str(text).split()) * 1.3)

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
    inject_metadata: bool = False, 
    include_keywords: bool = False,  
    search_method: str = "dense",
    resolve_parents:  bool = False,
) -> QuestionResult:
    """Run all applicable metrics for one question."""
    from open_telco_rag.core.pipeline import search, generate
    
    # Rewrite the question if metadata injection is enabled and a source exists
    active_question = q.question
    if inject_metadata and q.canonical_source:
        active_question = f"According to {q.corpus.upper()} document {q.canonical_source}, {q.question}"

    # Prepare search query (appends keywords if flag is enabled)
    search_query = active_question
    if include_keywords and getattr(q, "keywords", None):
        kw_str = " ".join(q.keywords) if isinstance(q.keywords, list) else str(q.keywords)
        if kw_str.strip():
            search_query = f"{active_question} Keywords: {kw_str}"

    result = QuestionResult(
        question=active_question, corpus=q.corpus,
        answer="", sources=[], chunks_used=0,
    )

    try:
        start_time = time.perf_counter()
        if baseline:
            retrieved = []
            generator = _build_baseline_generator(rag_llm_call or llm_call)
            
            gen_start = time.perf_counter()
            gen       = generate(active_question, retrieved, source_type=q.corpus, generator=generator)
            result.gen_latency = time.perf_counter() - gen_start
            result.search_latency = 0.0
        else:
            # 1. Time the base search retrieval pass (WITHOUT reranker)
            search_start = time.perf_counter()
            retrieved_candidates = search(
                search_query, 
                store=store, 
                embedder=embedder,
                top_k=top_k, 
                reranker=None,  
                rerank_top_k=top_k,
                search_method=search_method,
                resolve_parents=resolve_parents
            )
            result.search_latency = time.perf_counter() - search_start
            
            # 2. Time the Reranking pass & calculate its tokens
            result.rerank_latency = 0.0
            result.rerank_tokens = 0
            
            if reranker:
                rerank_start = time.perf_counter()
                
                # Reranker processes the query + all retrieved candidate chunks
                rerank_payload = search_query + " ".join([c.text for c in retrieved_candidates])
                result.rerank_tokens = _local_token_count(rerank_payload)
                
                # Execute reranking explicitly
                try:
                    retrieved = reranker.rerank(active_question, retrieved_candidates, top_k=rerank_top_k)
                except AttributeError:
                    # Fallback if your reranker wrapper uses a different method name
                    retrieved = retrieved_candidates[:rerank_top_k]
                    
                result.rerank_latency = time.perf_counter() - rerank_start
            else:
                retrieved = retrieved_candidates[:rerank_top_k]
            
            # 3. Time the text generation pass
            gen_start = time.perf_counter()
            generator = _build_generator(rag_llm_call or llm_call, model_name=rag_model_name)
            gen       = generate(active_question, retrieved, source_type=q.corpus, generator=generator)
            result.gen_latency = time.perf_counter() - gen_start
        
        result.latency = time.perf_counter() - start_time
        
        # 4. Local Token Counting (Replaces API Usage)
        answer = gen.get("answer", "")
        
        # Generator processes the active question + the final filtered chunks
        prompt_payload = active_question + " ".join([c.text for c in retrieved])
        
        result.prompt_tokens = _local_token_count(prompt_payload)
        result.completion_tokens = _local_token_count(answer)
        result.total_tokens = result.prompt_tokens + result.completion_tokens

        # Zero out API-dependent advanced metrics since we are bypassing the API report
        result.cached_tokens = 0
        result.reasoning_tokens = 0
        result.cost = 0.0  # Optional: You can recalculate cost here using result.total_tokens if desired

        result.answer  = answer
        result.sources = gen.get("sources", [])
        result.chunks_used = gen.get("chunks_used", len(retrieved))

        if debug:
            mode = "BASELINE" if baseline else "RAG"
            print(f"\n  [DEBUG] [{mode}] question: {active_question[:80]}")
            print(f"  [DEBUG] [{mode}] search query: {search_query[:80]}")
            print(f"  [DEBUG] [{mode}] answer preview: {answer[:200]}")

        if q.answer_type in ("mc", "tf"):
            # Clean answers for exact matching
            clean_ans = answer.strip().strip(".*`\"'").upper()
            clean_ref = str(q.reference_answer).strip().upper()
            
            is_correct = 0.0
            if clean_ref == clean_ans or f"ANSWER IS {clean_ref}" in clean_ans or clean_ans.startswith(clean_ref):
                is_correct = 1.0
                
            result.faithfulness     = is_correct
            result.answer_relevancy = 1.0
            if q.reference_answer:
                result.context_recall     = 1.0 if not baseline else 0.0
                result.answer_correctness = is_correct
        else:
            # Free-text evaluation
            result.faithfulness     = faithfulness(answer, retrieved, llm_call=llm_call, debug=debug)
            result.answer_relevancy = answer_relevancy(active_question, answer, embedder, llm_call=llm_call, debug=debug)
            if q.reference_answer:
                result.context_recall     = context_recall(q.reference_answer, retrieved, llm_call=llm_call, debug=debug)
                result.answer_correctness = answer_correctness(answer, q.reference_answer, embedder, llm_call=llm_call, debug=debug)

        if not baseline:
            mean_s, min_s, max_s  = cosine_sim_stats(retrieved)
            result.mean_similarity = mean_s
            result.min_similarity  = min_s
            result.max_similarity  = max_s
            result.source_coverage = source_coverage(retrieved)
            result.canonical_hit   = canonical_source_hit(retrieved, q.canonical_source)

    except Exception as e:
        result.error = str(e)
        result.latency = 0.0
        result.prompt_tokens = 0
        result.completion_tokens = 0
        result.search_latency = 0.0 
        result.gen_latency = 0.0

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
    include_keywords: bool = False,
    search_method: str = "dense",
    resolve_parents:  bool = False,
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
                               inject_metadata=inject_metadata,
                               include_keywords=include_keywords,
                               search_method=search_method,
                               resolve_parents=resolve_parents,)
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