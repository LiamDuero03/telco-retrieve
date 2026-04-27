"""
Evaluation orchestrator for Open Telco RAG.

Wires together questions, metrics, and results. Contains no metric logic
and no question data — those live in their own modules.

Usage:
    from open_telco_rag.evaluation.evaluate import evaluate_corpus
    from open_telco_rag.evaluation.questions import load_questions
    from open_telco_rag.evaluation._claude import make_llm_call

    questions = load_questions("oran")

    # Default — Claude
    result = evaluate_corpus("oran", store, embedder, questions=questions)

    # OpenRouter
    llm_call = make_llm_call("openrouter", "meta-llama/llama-3.3-70b-instruct")
    result   = evaluate_corpus("oran", store, embedder, questions=questions,
                               llm_call=llm_call)

    # Debug mode — logs raw LLM responses for every metric call
    result   = evaluate_corpus("oran", store, embedder, questions=questions,
                               llm_call=llm_call, debug=True)
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
    keyword_score, faithfulness, answer_relevancy,
)
from open_telco_rag.evaluation.metrics.retrieval import (
    cosine_sim_stats, source_coverage, canonical_source_hit, coverage_ratio,
)
from open_telco_rag.evaluation.metrics.ground_truth import (
    context_recall, answer_correctness,
)


def evaluate_question(
    q: EvalQuestion,
    store: Any,
    embedder: Any,
    top_k: int = 5,
    sim_threshold: float = 0.5,
    llm_call: Callable[[str, int], str] = claude_call,
    debug: bool = False,
) -> QuestionResult:
    """
    Run all applicable metrics for one question.

    Args:
        llm_call: callable (prompt, max_tokens) -> str for all metric LLM calls.
        debug:    if True, logs raw LLM responses for every metric call to stdout.
    """
    from open_telco_rag.core.pipeline import search, generate

    result = QuestionResult(
        question=q.question, corpus=q.corpus,
        answer="", sources=[], chunks_used=0,
    )

    try:
        retrieved      = search(q.question, store=store, embedder=embedder,
                                top_k=top_k)
        gen            = generate(q.question, retrieved, source_type=q.corpus)
        answer         = gen.get("answer", "")
        result.answer  = answer
        result.sources = gen.get("sources", [])
        result.chunks_used = gen.get("chunks_used", len(retrieved))

        if debug:
            print(f"\n  [DEBUG] question: {q.question[:80]}")
            print(f"  [DEBUG] answer preview: {answer[:200]}")

        # Tier 1 — generation
        result.keyword_score    = keyword_score(answer, q.keywords)
        result.faithfulness     = faithfulness(answer, retrieved,
                                               llm_call=llm_call, debug=debug)
        result.answer_relevancy = answer_relevancy(q.question, answer, embedder,
                                                   llm_call=llm_call, debug=debug)

        # Tier 1 — retrieval (pure math, no llm_call)
        mean_s, min_s, max_s   = cosine_sim_stats(retrieved)
        result.mean_similarity  = mean_s
        result.min_similarity   = min_s
        result.max_similarity   = max_s
        result.source_coverage  = source_coverage(retrieved)
        result.canonical_hit    = canonical_source_hit(retrieved, q.canonical_source)
        result.coverage_ratio   = coverage_ratio(retrieved, sim_threshold)

        # Tier 2 — ground truth (only if reference answer provided)
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
    corpus: str,
    store: Any,
    embedder: Any,
    questions: list[EvalQuestion] | None = None,
    model: str = "minilm",
    top_k: int = 5,
    sim_threshold: float = 0.5,
    delay: float = 1.0,
    output_dir: Path | None = None,
    llm_call: Callable[[str, int], str] = claude_call,
    debug: bool = False,
) -> CorpusResult:
    """
    Run the full evaluation suite for one corpus.

    Args:
        corpus:        source type string (e.g. "oran", "3gpp")
        store:         ChromaStore for this corpus
        embedder:      SentenceTransformerEmbedder
        questions:     override default questions (defaults to load_questions(corpus))
        model:         embedding model key — recorded in results metadata
        top_k:         chunks to retrieve per question
        sim_threshold: threshold for coverage_ratio metric (default 0.5)
        delay:         seconds between questions — rate limits API calls
        output_dir:    if provided, saves results JSON automatically
        llm_call:      callable (prompt, max_tokens) -> str for all metric LLM calls.
        debug:         if True, logs raw LLM responses for every metric call to stdout.
    """
    qs = questions or load_questions(corpus)

    print(f"\nEvaluating {corpus.upper()} — {len(qs)} questions | "
          f"model={model} | top-k={top_k}"
          + (" | DEBUG ON" if debug else ""))

    question_results = []
    for i, q in enumerate(qs, 1):
        print(f"  [{i}/{len(qs)}] {q.question[:70]}...")
        qr = evaluate_question(q, store, embedder, top_k, sim_threshold,
                               llm_call=llm_call, debug=debug)
        question_results.append(qr)
        if qr.error:
            print(f"    [ERROR] {qr.error}")
        else:
            print(f"    keyword={qr.keyword_score:.2f}  "
                  f"faithful={qr.faithfulness:.2f}  "
                  f"relevancy={qr.answer_relevancy:.2f}  "
                  f"sim={qr.mean_similarity:.3f}")
        if delay > 0 and i < len(qs):
            time.sleep(delay)

    agg = aggregate(question_results)

    result = CorpusResult(
        corpus=corpus,
        model=model,
        top_k=top_k,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        question_results=question_results,
        **agg,
    )

    print_summary(result)

    if output_dir:
        save_results(result, Path(output_dir))

    return result