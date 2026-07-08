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


def _build_generator(llm_call: Callable[[str, int], str]):
    """Wrap an llm_call callable as a generator for pipeline.generate()."""
    class _CallableGenerator:
        def generate(self, question: str, results: list, source_type: str = "telecom") -> dict:
            if not results:
                return {
                    "answer":           "No relevant chunks found.",
                    "sources":          [],
                    "chunks_used":      0,
                    "retrieved_chunks": [],
                }
            context = "\n\n---\n\n".join(r.text for r in results)
            sources = list({r.metadata.get("source", "") for r in results})
            prompt  = (
                f"You are a {source_type} technical expert. "
                "Answer the question using only the context provided. "
                "If the answer is not in the context, say so clearly. "
                "Be precise and technical.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            answer = llm_call(prompt, max_tokens=1000)
            return {
                "answer":           answer,
                "sources":          sources,
                "chunks_used":      len(results),
                "retrieved_chunks": [r.text for r in results],
            }
    return _CallableGenerator()


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
) -> QuestionResult:
    """Run all applicable metrics for one question.

    Args:
        baseline: if True, skip retrieval and generate answer from LLM only.
                  Retrieval metrics (similarity, source_coverage) will be 0.
    """
    from open_telco_rag.core.pipeline import search, generate

    result = QuestionResult(
        question=q.question, corpus=q.corpus,
        answer="", sources=[], chunks_used=0,
    )

    try:
        if baseline:
            # No retrieval — answer from LLM knowledge only
            retrieved = []
            generator = _build_baseline_generator(rag_llm_call or llm_call)
            gen       = generate(q.question, retrieved,
                                 source_type=q.corpus,
                                 generator=generator)
        else:
            retrieved = search(q.question, store=store, embedder=embedder,
                                top_k=top_k, reranker=reranker, rerank_top_k=rerank_top_k)
            generator = _build_generator(rag_llm_call or llm_call)
            gen       = generate(q.question, retrieved,
                                 source_type=q.corpus,
                                 generator=generator)

        answer         = gen.get("answer", "")
        result.answer  = answer
        result.sources = gen.get("sources", [])
        result.chunks_used = gen.get("chunks_used", len(retrieved))

        if debug:
            mode = "BASELINE" if baseline else "RAG"
            print(f"\n  [DEBUG] [{mode}] question: {q.question[:80]}")
            print(f"  [DEBUG] [{mode}] answer preview: {answer[:200]}")

        # Tier 1 — generation
        result.faithfulness     = faithfulness(answer, retrieved,
                                               llm_call=llm_call, debug=debug)
        result.answer_relevancy = answer_relevancy(q.question, answer, embedder,
                                                   llm_call=llm_call, debug=debug)

        # Tier 1 — retrieval (0.0 in baseline mode — no chunks retrieved)
        if not baseline:
            mean_s, min_s, max_s  = cosine_sim_stats(retrieved)
            result.mean_similarity = mean_s
            result.min_similarity  = min_s
            result.max_similarity  = max_s
            result.source_coverage = source_coverage(retrieved)
            result.canonical_hit   = canonical_source_hit(retrieved, q.canonical_source)

        # Tier 2 — ground truth
        if q.reference_answer:
            result.context_recall     = context_recall(
                q.reference_answer, retrieved, llm_call=llm_call, debug=debug)
            result.answer_correctness = answer_correctness(
                answer, q.reference_answer, embedder,
                llm_call=llm_call, debug=debug)

    except Exception as e:
        result.error = str(e)

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
) -> tuple[CorpusResult, CorpusResult | None]:
    """Run the full evaluation suite for one corpus.

    Returns:
        (rag_result, baseline_result) — baseline_result is None if
        include_baseline=False.
    """
    qs = questions or load_questions(corpus)

    print(f"\nEvaluating {corpus.upper()} — {len(qs)} questions | "
          f"model={model} | top-k={top_k}"
          + (" | +BASELINE" if include_baseline else "")
          + (" | DEBUG ON" if debug else ""))

    # ── RAG pass ──────────────────────────────────────────────────────────────
    rag_results = []
    for i, q in enumerate(qs, 1):
        print(f"  [RAG {i}/{len(qs)}] {q.question[:65]}...")
        qr = evaluate_question(q, store, embedder, top_k, sim_threshold,
                               llm_call=llm_call, rag_llm_call=rag_llm_call,
                               baseline=False, reranker=reranker,
                               rerank_top_k=rerank_top_k, debug=debug)
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