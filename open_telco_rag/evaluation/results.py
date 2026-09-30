"""
Result dataclasses, serialisation, and display for evaluation runs.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


# Methodology note surfaced in print_summary()/report.md, not a computed metric.
COVERAGE_RATIO_CALIBRATION_NOTE = (
    "coverage_ratio's default threshold (0.5) is empirically uninformative "
    "on this corpus -- it clears ~100% of real retrieval across every "
    "combination, even though the metric itself correctly discriminates "
    "genuinely bad retrieval under direct testing. Treat mean_coverage_ratio "
    "as an uncalibrated signal, not a trustworthy comparison, until the "
    "threshold is deliberately recalibrated."
)

# answer_correctness above which an open/fill_blank answer counts as
# "correct" for weighted_score_pct.
OE_CORRECTNESS_THRESHOLD = 0.7


@dataclass
class QuestionResult:
    """All metric scores for one question."""
    question:    str
    corpus:      str
    answer:      str
    sources:     list[str]
    chunks_used: int

    # Drives which metrics apply; also used by aggregate() to group.
    answer_type: str = "open"

    # opt-in: copied from EvalQuestion.difficulty
    difficulty: str | None = None

    # ── EFFICIENCY METRICS  ───────────────────────────────────────
    latency:           float = 0.0
    search_latency:    float = 0.0   # embedding + vector search
    rerank_latency:    float = 0.0   # reranking retrieved chunks
    gen_latency:       float = 0.0   # answer generation
    prompt_tokens:     int   = 0
    rerank_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0

    # "api": generator's real usage. "estimated": tiktoken fallback. "": never evaluated.
    token_source:      str   = ""

    # None = not computed; a real value (incl. 0) means known.
    cached_tokens:     int | None = None
    reasoning_tokens:  int | None = None
    cost:              float | None = None

    # ── Judge/metric-call token usage ─────────────────────────────
    # Separate from generation tokens: judge cost is ~constant across chunking
    # strategies, so folding it in would conflate the comparison. 0 for mc/tf.
    judge_prompt_tokens:     int = 0
    judge_completion_tokens: int = 0

    # opt-in (record_judge_calls): [{"metric", "prompt", "raw_response"}, ...] in call order
    judge_calls: list[dict] | None = None

    # ── Deterministic scoring (mc/tf only; -1.0 = not applicable) ────────────
    # accuracy is mc/tf's own field; the four OE metrics stay at -1.0 for mc/tf.
    accuracy: float = -1.0

    # Three-way outcome behind `accuracy`: "correct" / "incorrect" /
    # "abstained". None for non-mc/tf.
    mc_tf_outcome: str | None = None

    # n_generations>1: how many generations were made for this question, and
    # the vote breakdown {predicted_value_or_"abstained/unparsed": count}.
    # None/0 when n_generations==1.
    self_consistency_n:     int  = 0
    self_consistency_votes: dict | None = None
    # winning_votes / self_consistency_n. -1.0 unless mc/tf with n_generations>1.
    self_consistency_agreement: float = -1.0

    # ── Tier 1: no ground truth (open/fill_blank only; -1.0 = not applicable) ─
    faithfulness:     float = -1.0
    answer_relevancy: float = -1.0

    # candidate_* = pre-rerank pool, rank order. used_* = what the generator saw.
    # [] for baseline. Text travels with the id since the corpus can be re-chunked.
    candidate_chunk_ids:   list[str] = field(default_factory=list)
    candidate_chunk_texts: list[str] = field(default_factory=list)
    used_chunk_ids:        list[str] = field(default_factory=list)
    used_chunk_texts:      list[str] = field(default_factory=list)

    # 1-indexed rank of the correct chunk in candidate/used_chunk_ids; 0 = not
    # found, -1 = not applicable. Same sentinel convention as exact_chunk_hit.
    candidate_chunk_rank: int = -1
    used_chunk_rank:      int = -1

    # ── span hit (opt-in): None without an evidence span; -1 = no retrieved chunk
    # carries offsets (legacy collection)
    span_hit_at_k:       int | None = None
    span_hit_in_context: int | None = None

    # ── raw retrieval scores (opt-in): store distances aligned with candidate_chunk_ids / used_chunk_ids
    candidate_distances: list | None = None
    used_distances:      list | None = None

    # ── budgeted context (opt-in): set only with context_token_budget
    context_tokens: int | None = None    # tokens in the chunks actually used
    n_chunks_used:  int | None = None    # chunks fill_context_budget() kept
    budget:         int | None = None    # the context_token_budget for this question
    fill_ratio:     float | None = None  # context_tokens / budget
    # fill outcome: candidates, kept, skipped, truncated, stopped_by_policy, pool_exhausted, empty
    budget_stats:   dict | None = None

    # ── Retrieval metrics: computed for every non-baseline question ──────────
    mean_similarity:  float = 0.0
    min_similarity:   float = 0.0
    max_similarity:   float = 0.0
    # Raw count of distinct source docs (0..top_k), not a ratio.
    distinct_sources_retrieved: int = 0
    canonical_hit:    int   = -1   # -1 = not applicable, 0 = miss, 1 = hit
    # None = no canonical_source; False = rejected as too generic
    # (is_discriminating_source()); True = a real identifier was checked.
    canonical_source_discriminating: bool | None = None
    # Stronger than canonical_hit: the exact source chunk, not just its
    # document (see metrics.retrieval.exact_chunk_hit()). -1/0/1.
    exact_chunk_hit:  int   = -1
    # Fraction of retrieved chunks meeting the --threshold floor. Uncalibrated
    # on this corpus (see CorpusResult).
    coverage_ratio:   float = -1.0

    # ── Tier 2: ground truth required (open/fill_blank only; -1.0 = not computed) ─
    context_recall:     float = -1.0
    answer_correctness: float = -1.0

    error: str = ""
    error_type: str = ""   # type(exception).__name__, "" if error is also ""


@dataclass
class CorpusResult:
    """Aggregate results across all questions for one corpus."""
    corpus:    str
    model:     str = ""   # embedding model key (e.g. "bge"), NOT the generator
    top_k:     int = 5
    timestamp: str = ""

    # The LLM that produced the RAG answers (distinct from `model`, the
    # embedder), so a pushed result.json is self-describing. "" for older results.
    rag_generator_model: str = ""
    rag_generator_name:  str = ""   # "local" / "openrouter" / "claude"

    rag_generator_temperature: float = 0.0
    rag_generator_top_p:       float | None = None

    questions_file: str = ""

    n_questions: int = 0

    # ── Deterministic (mc/tf) aggregate, -1.0 = no mc/tf questions in this run ──
    mean_accuracy: float = -1.0

    # Three-way mc/tf breakdown behind mean_accuracy; all -1.0 with no mc/tf
    # questions.
    #   mean_accuracy            = correct / n_mc_tf              (unchanged, all mc/tf)
    #   mean_accuracy_attempted  = correct / (correct + incorrect)  (abstentions excluded)
    #   abstention_rate          = abstained / n_mc_tf
    #   wrong_rate               = incorrect / n_mc_tf
    mean_accuracy_attempted: float = -1.0
    abstention_rate:         float = -1.0
    wrong_rate:               float = -1.0

    # ── Tier 1 aggregates (open/fill_blank only now - see QuestionResult) ──────
    mean_faithfulness:     float = -1.0
    mean_answer_relevancy: float = -1.0

    # ── Retrieval aggregates (every non-baseline question) ──────────────────
    mean_similarity:       float = 0.0
    # Mean of the raw distinct-source count; can be > 1.0 (it's a count).
    mean_distinct_sources_retrieved: float = 0.0
    canonical_hit_rate:    float = -1.0
    # Count of questions whose canonical_source was rejected as too generic;
    # if all are, canonical_hit_rate is -1.0 rather than a meaningless 1.000.
    n_canonical_source_non_discriminating: int = 0
    # Stronger companion to canonical_hit_rate. -1.0 with no chunk provenance.
    exact_chunk_hit_rate:  float = -1.0
    mean_reciprocal_rank: float = -1.0   # mean of 1/used_chunk_rank; -1.0 with no provenance
    # span hit rates (opt-in): only with evidence spans
    span_hit_at_k_rate:       float | None = None
    span_hit_in_context_rate: float | None = None
    mean_coverage_ratio:   float = -1.0

    # ── reproducibility record (see evaluation/provenance.py); absent in older files
    run_record: dict | None = None

    # ── failed questions: only present when at least one question failed
    n_errors:       int | None = None
    error_rate:     float | None = None
    errors_by_type: dict | None = None

    # ── run-level component stats (opt-in): only when the reranker/embedder exposes counters
    reranker_stats: dict | None = None
    embedder_stats: dict | None = None

    # ── budgeted context aggregates (opt-in)
    mean_context_tokens: float | None = None
    mean_fill_ratio:     float | None = None

    # ── Tier 2 aggregates (-1.0 = not computed) ───────────────────────────────
    mean_context_recall:     float = -1.0
    mean_answer_correctness: float = -1.0

    # ── Efficiency aggregates (sum/mean per combination) ────────────────────
    mean_latency:            float = 0.0
    max_latency:             float = 0.0
    p95_latency:             float = 0.0
    mean_search_latency:     float = 0.0
    mean_rerank_latency:     float = 0.0
    mean_gen_latency:        float = 0.0

    total_wall_clock_seconds: float = 0.0
    max_concurrency:          int   = 1

    total_prompt_tokens:     int   = 0
    total_completion_tokens: int   = 0
    total_tokens:            int   = 0
    max_total_tokens:        int   = 0
    p95_total_tokens:        float = 0.0
    mean_prompt_tokens:      float = 0.0
    mean_completion_tokens:  float = 0.0
    mean_total_tokens:       float = 0.0

    total_reasoning_tokens: int   | None = None   # None = never reported, not zero
    mean_reasoning_tokens:  float | None = None
    total_cached_tokens:    int   | None = None
    mean_cached_tokens:     float | None = None

    # Judge-call tokens, separate from generation totals above.
    total_judge_prompt_tokens:     int = 0
    total_judge_completion_tokens: int = 0
    mean_judge_prompt_tokens:      float = 0.0
    mean_judge_completion_tokens:  float = 0.0

    # None only when no question had a known cost; distinguishes "unknown"
    # from "$0.00".
    total_cost:   float | None = None
    mean_cost:    float | None = None
    n_cost_known: int = 0

    # {"api": n, "estimated": n} - see QuestionResult.token_source.
    token_source_counts: dict[str, int] = field(default_factory=dict)

    # mc/tf correct = 1pt, open/fill_blank correct = 5pts. -1.0 with no
    # scorable questions. See _weighted_score_pct().
    weighted_score_pct: float = -1.0

    # Per-answer_type metrics dict, so mc vs tf can be compared directly.
    by_format: dict[str, dict] = field(default_factory=dict)

    question_results: list[QuestionResult] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    """Mean of non-negative values; excludes the -1.0 "not applicable"
    sentinel rather than treating it as a real zero score."""
    valid = [v for v in values if v >= 0]
    return statistics.mean(valid) if valid else -1.0


def _mc_tf_outcome_rates(results: list[QuestionResult]) -> dict:
    """mean_accuracy_attempted/abstention_rate/wrong_rate from mc_tf_outcome
    (see CorpusResult for what each means). `results` may be mixed-format -
    only non-None mc_tf_outcome entries count. -1.0 for all three if no
    mc/tf questions.
    """
    outcomes = [r.mc_tf_outcome for r in results if r.mc_tf_outcome is not None]
    n = len(outcomes)
    if n == 0:
        return {
            "mean_accuracy_attempted": -1.0,
            "abstention_rate": -1.0,
            "wrong_rate": -1.0,
        }
    n_correct   = sum(1 for o in outcomes if o == "correct")
    n_incorrect = sum(1 for o in outcomes if o == "incorrect")
    n_abstained = sum(1 for o in outcomes if o == "abstained")
    n_attempted = n_correct + n_incorrect
    return {
        "mean_accuracy_attempted": (n_correct / n_attempted) if n_attempted > 0 else -1.0,
        "abstention_rate": n_abstained / n,
        "wrong_rate": n_incorrect / n,
    }


def _aggregate_one_format(results: list[QuestionResult]) -> dict:
    """Per-answer_type metrics dict for CorpusResult.by_format - mc/tf get
    accuracy + outcome breakdown; open/fill_blank get the four OE metrics.
    Retrieval metrics are reported for every format.
    """
    canonical = [r for r in results if r.canonical_hit >= 0]
    exact_chunk = [r for r in results if r.exact_chunk_hit >= 0]
    span_ak = [r for r in results if r.span_hit_at_k is not None and r.span_hit_at_k >= 0]
    span_ic = [r for r in results if r.span_hit_in_context is not None and r.span_hit_in_context >= 0]
    entry = {
        "n":                    len(results),
        "mean_similarity":      _mean([r.mean_similarity for r in _retrieval_population(results)]),
        "mean_distinct_sources_retrieved": _mean(
            [float(r.distinct_sources_retrieved) for r in _retrieval_population(results)]),
        "canonical_hit_rate":   (sum(r.canonical_hit for r in canonical) / len(canonical)
                                  if canonical else -1.0),
        "n_canonical_source_non_discriminating": sum(
            1 for r in results if r.canonical_source_discriminating is False
        ),
        "exact_chunk_hit_rate": (sum(r.exact_chunk_hit for r in exact_chunk) / len(exact_chunk)
                                  if exact_chunk else -1.0),
        "mean_coverage_ratio":  _mean([r.coverage_ratio for r in _retrieval_population(results)]),
    }
    entry.update({k: v for k, v in _error_summary(results).items() if k != "errors_by_type"})
    # opt-in keys: only when some question in this group has an evidence span
    if any(r.span_hit_at_k is not None for r in results):
        entry["span_hit_at_k_rate"] = (sum(r.span_hit_at_k for r in span_ak) / len(span_ak)
                                       if span_ak else -1.0)
        entry["span_hit_in_context_rate"] = (sum(r.span_hit_in_context for r in span_ic) / len(span_ic)
                                             if span_ic else -1.0)
    fmt = results[0].answer_type if results else "open"
    if fmt in ("mc", "tf"):
        entry["accuracy"] = _mean([r.accuracy for r in results])
        entry.update(_mc_tf_outcome_rates(results))
    else:
        entry["faithfulness"]      = _mean([r.faithfulness      for r in results])
        entry["answer_relevancy"]  = _mean([r.answer_relevancy  for r in results])
        entry["context_recall"]    = _mean([r.context_recall    for r in results])
        entry["answer_correctness"] = _mean([r.answer_correctness for r in results])
    return entry


def _mean_or_zero(values: list[float]) -> float:
    """Plain mean, 0.0 for empty - unlike _mean(), efficiency fields don't
    use -1.0 as a sentinel, so nothing needs filtering."""
    return statistics.mean(values) if values else 0.0


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in [0, 100]), 0.0 for empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def _aggregate_efficiency(results: list[QuestionResult]) -> dict:
    """Latency/token/judge-token/cost aggregates for CorpusResult. Straight
    sum/mean over every result, no sentinel filtering - an errored question's
    0.0/0/None defaults still count as a data point, same as elsewhere.
    """
    latencies          = [r.latency for r in results]
    prompt_tokens      = [r.prompt_tokens for r in results]
    completion_tokens  = [r.completion_tokens for r in results]
    total_tokens       = [r.total_tokens for r in results]
    judge_prompt       = [r.judge_prompt_tokens for r in results]
    judge_completion   = [r.judge_completion_tokens for r in results]

    known_costs = [r.cost for r in results if r.cost is not None]

    known_reasoning = [r.reasoning_tokens for r in results if r.reasoning_tokens is not None]
    known_cached    = [r.cached_tokens    for r in results if r.cached_tokens    is not None]

    token_source_counts: dict[str, int] = {}
    for r in results:
        if r.token_source:
            token_source_counts[r.token_source] = token_source_counts.get(r.token_source, 0) + 1

    return dict(
        mean_latency=        _mean_or_zero(latencies),
        max_latency=         max(latencies) if latencies else 0.0,
        p95_latency=         _percentile(latencies, 95),
        mean_search_latency= _mean_or_zero([r.search_latency for r in results]),
        mean_rerank_latency= _mean_or_zero([r.rerank_latency for r in results]),
        mean_gen_latency=    _mean_or_zero([r.gen_latency for r in results]),

        total_prompt_tokens=     sum(prompt_tokens),
        total_completion_tokens= sum(completion_tokens),
        total_tokens=             sum(total_tokens),
        max_total_tokens=        max(total_tokens) if total_tokens else 0,
        p95_total_tokens=        _percentile(total_tokens, 95),
        mean_prompt_tokens=      _mean_or_zero(prompt_tokens),
        mean_completion_tokens=  _mean_or_zero(completion_tokens),
        mean_total_tokens=       _mean_or_zero(total_tokens),

        total_reasoning_tokens= sum(known_reasoning) if known_reasoning else None,
        mean_reasoning_tokens=  (sum(known_reasoning) / len(known_reasoning)) if known_reasoning else None,
        total_cached_tokens=    sum(known_cached) if known_cached else None,
        mean_cached_tokens=     (sum(known_cached) / len(known_cached)) if known_cached else None,

        total_judge_prompt_tokens=     sum(judge_prompt),
        total_judge_completion_tokens= sum(judge_completion),
        mean_judge_prompt_tokens=      _mean_or_zero(judge_prompt),
        mean_judge_completion_tokens=  _mean_or_zero(judge_completion),

        total_cost=   sum(known_costs) if known_costs else None,
        mean_cost=    (sum(known_costs) / len(known_costs)) if known_costs else None,
        n_cost_known= len(known_costs),

        token_source_counts=token_source_counts,
    )


def _weighted_score_pct(question_results: list[QuestionResult]) -> float:
    """Weighted overall performance score, as a percentage:

        score_pct = (mc_tf_correct + 5*oe_correct)
                    / (mc_tf_total  + 5*oe_total) * 100

    oe weights 5x since it tests generation more strongly. Abstention scores
    0. oe questions with no reference_answer are excluded entirely. -1.0 when
    the denominator is 0.
    """
    mc_tf = [r for r in question_results if r.mc_tf_outcome is not None]
    mc_tf_correct = sum(1 for r in mc_tf if r.mc_tf_outcome == "correct")
    mc_tf_total = len(mc_tf)

    oe = [
        r for r in question_results
        if r.answer_type in ("open", "fill_blank") and r.answer_correctness >= 0
    ]
    oe_correct = sum(1 for r in oe if r.answer_correctness >= OE_CORRECTNESS_THRESHOLD)
    oe_total = len(oe)

    numerator   = mc_tf_correct + 5 * oe_correct
    denominator = mc_tf_total   + 5 * oe_total
    if denominator == 0:
        return -1.0
    return (numerator / denominator) * 100


def _error_summary(results: list[QuestionResult]) -> dict:
    """{} when nothing failed, else count / rate / breakdown by exception type"""
    errored = [r for r in results if r.error]
    if not errored:
        return {}
    by_type: dict[str, int] = {}
    for r in errored:
        by_type[r.error_type or "Unknown"] = by_type.get(r.error_type or "Unknown", 0) + 1
    return {"n_errors": len(errored), "error_rate": len(errored) / len(results),
            "errors_by_type": dict(sorted(by_type.items()))}


def _retrieval_population(results: list[QuestionResult]) -> list[QuestionResult]:
    """questions whose retrieval metrics are real numbers. a failed question keeps
    the 0.0 defaults of its retrieval fields, which are not scores and must not
    dilute the means; identical to `results` when nothing failed"""
    return [r for r in results if not r.error]


def aggregate(question_results: list[QuestionResult]) -> dict:
    """Compute aggregate metrics from a list of QuestionResults.

    Failed questions are counted in n_errors / error_rate / errors_by_type and
    left out of the retrieval means."""
    retrieval_qs = _retrieval_population(question_results)

    canonical_qs = [r for r in question_results if r.canonical_hit >= 0]
    exact_chunk_qs = [r for r in question_results if r.exact_chunk_hit >= 0]
    rank_qs = [r for r in question_results if r.used_chunk_rank >= 0]
    budgeted_qs = [r for r in question_results if r.budget is not None]
    span_ak_qs = [r for r in question_results if r.span_hit_at_k is not None and r.span_hit_at_k >= 0]
    span_ic_qs = [r for r in question_results
                  if r.span_hit_in_context is not None and r.span_hit_in_context >= 0]
    has_spans = any(r.span_hit_at_k is not None for r in question_results)

    by_format: dict[str, dict] = {}
    for fmt in sorted({r.answer_type for r in question_results}):
        by_format[fmt] = _aggregate_one_format(
            [r for r in question_results if r.answer_type == fmt]
        )

    outcome_rates = _mc_tf_outcome_rates(question_results)

    return dict(
        n_questions=             len(question_results),
        mean_accuracy=           _mean([r.accuracy           for r in question_results]),
        mean_accuracy_attempted= outcome_rates["mean_accuracy_attempted"],
        abstention_rate=         outcome_rates["abstention_rate"],
        wrong_rate=              outcome_rates["wrong_rate"],
        mean_faithfulness=       _mean([r.faithfulness       for r in question_results]),
        mean_answer_relevancy=   _mean([r.answer_relevancy   for r in question_results]),
        mean_similarity=         _mean([r.mean_similarity    for r in retrieval_qs]),
        mean_distinct_sources_retrieved=
                                  _mean([float(r.distinct_sources_retrieved) for r in retrieval_qs]),
        canonical_hit_rate=     (sum(r.canonical_hit for r in canonical_qs) /
                                  len(canonical_qs) if canonical_qs else -1.0),
        n_canonical_source_non_discriminating=
                                  sum(1 for r in question_results if r.canonical_source_discriminating is False),
        exact_chunk_hit_rate=   (sum(r.exact_chunk_hit for r in exact_chunk_qs) /
                                  len(exact_chunk_qs) if exact_chunk_qs else -1.0),
        # rank==0 (not found) contributes 0, not 1/0 -- standard MRR convention.
        mean_reciprocal_rank=   (sum(1.0 / r.used_chunk_rank if r.used_chunk_rank > 0 else 0.0
                                      for r in rank_qs) / len(rank_qs) if rank_qs else -1.0),
        mean_coverage_ratio=     _mean([r.coverage_ratio     for r in retrieval_qs]),
        mean_context_recall=     _mean([r.context_recall     for r in question_results]),
        mean_answer_correctness= _mean([r.answer_correctness for r in question_results]),
        weighted_score_pct=      _weighted_score_pct(question_results),
        by_format=by_format,
        **_aggregate_efficiency(question_results),
        **_error_summary(question_results),
        # opt-in keys: only present when the run has evidence spans
        **({
            "span_hit_at_k_rate": (sum(r.span_hit_at_k for r in span_ak_qs) /
                                   len(span_ak_qs) if span_ak_qs else -1.0),
            "span_hit_in_context_rate": (sum(r.span_hit_in_context for r in span_ic_qs) /
                                         len(span_ic_qs) if span_ic_qs else -1.0),
        } if has_spans else {}),
        # opt-in keys: only present when at least one question used a budget
        **({
            "mean_context_tokens": _mean_or_zero([float(r.context_tokens) for r in budgeted_qs]),
            "mean_fill_ratio":     _mean([r.fill_ratio for r in budgeted_qs]),
        } if budgeted_qs else {}),
    )


# opt-in fields: dropped from serialized output while None, so default runs
# serialize exactly as before
_OPT_IN_QUESTION_FIELDS = ("context_tokens", "n_chunks_used", "budget", "fill_ratio", "budget_stats",
                           "candidate_distances", "used_distances",
                           "span_hit_at_k", "span_hit_in_context", "difficulty", "judge_calls")
_OPT_IN_CORPUS_FIELDS = ("run_record", "n_errors", "error_rate", "errors_by_type", "mean_context_tokens", "mean_fill_ratio",
                         "reranker_stats", "embedder_stats",
                         "span_hit_at_k_rate", "span_hit_in_context_rate")


def question_to_dict(result: QuestionResult) -> dict:
    d = asdict(result)
    for key in _OPT_IN_QUESTION_FIELDS:
        if d.get(key) is None:
            d.pop(key, None)
    return d


def result_to_dict(result: CorpusResult) -> dict:
    d = asdict(result)
    for key in _OPT_IN_CORPUS_FIELDS:
        if d.get(key) is None:
            d.pop(key, None)
    for q in d.get("question_results", []):
        for key in _OPT_IN_QUESTION_FIELDS:
            if q.get(key) is None:
                q.pop(key, None)
    return d


def save_results(result: CorpusResult, output_dir: Path) -> Path:
    """Serialise CorpusResult to a timestamped JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts       = result.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{result.corpus}_{result.model}_{ts}.json"
    path     = output_dir / filename
    path.write_text(json.dumps(result_to_dict(result), indent=2, default=str),
                    encoding="utf-8")
    print(f"  Saved -> {path}")
    return path


def print_summary(result: CorpusResult) -> None:
    """Print a formatted summary table to stdout."""
    w = 62
    print(f"\n{'='*w}")
    print(f"  {result.corpus.upper()}  |  model={result.model}  |  "
          f"top-k={result.top_k}  |  n={result.n_questions}")
    print(f"{'='*w}")
    print(f"  {'Metric':<32} {'Score':>8}  {'Type'}")
    print(f"  {'-'*50}")

    rows = [
        ("Accuracy (mc/tf)",      result.mean_accuracy,           "Exact-match"),
        ("Accuracy, attempted (mc/tf)", result.mean_accuracy_attempted, "Exact-match"),
        ("Abstention rate (mc/tf)", result.abstention_rate,       "Exact-match"),
        ("Wrong rate (mc/tf)",    result.wrong_rate,              "Exact-match"),
        ("Faithfulness (open)",   result.mean_faithfulness,       "Generation"),
        ("Answer relevancy (open)", result.mean_answer_relevancy, "Generation"),
        ("Weighted score % (1/1/5)", result.weighted_score_pct,   "Composite"),
        ("Mean similarity",       result.mean_similarity,         "Retrieval"),
        ("Distinct sources (mean count)", result.mean_distinct_sources_retrieved, "Retrieval"),
        ("Coverage ratio (mean)", result.mean_coverage_ratio,     "Retrieval"),
        ("Canonical hit rate",    result.canonical_hit_rate,      "Retrieval"),
        ("Exact chunk hit rate",  result.exact_chunk_hit_rate,    "Retrieval"),
        ("Context recall (open)", result.mean_context_recall,     "Retrieval+GT"),
        ("Answer correctness (open)", result.mean_answer_correctness, "Generation+GT"),
    ]
    if result.span_hit_at_k_rate is not None:
        rows[12:12] = [
            ("Span hit@k rate",       result.span_hit_at_k_rate,      "Retrieval"),
            ("Span hit-in-context rate", result.span_hit_in_context_rate, "Retrieval"),
        ]
    for label, value, kind in rows:
        if value < 0:
            print(f"  {label:<32} {'N/A':>8}  {kind}")
        else:
            fmt = ".1f" if label == "Distinct sources (mean count)" else ".3f"
            print(f"  {label:<32} {value:>8{fmt}}  {kind}")

    if result.n_errors:
        types = ", ".join(f"{t} x{n}" for t, n in (result.errors_by_type or {}).items())
        print(f"  {'':32} FAILED questions: {result.n_errors}/{result.n_questions} "
              f"({result.error_rate:.0%})" + (f" -- {types}" if types else ""))
        print(f"  {'':32}   they are excluded from the retrieval means above and from every rate.")

    if result.n_canonical_source_non_discriminating > 0:
        print(f"  {'':32} {result.n_canonical_source_non_discriminating} question(s) had a "
              f"canonical_source too generic to trust (see canonical_source_discriminating) --")
        print(f"  {'':32}   canonical_hit_rate above is computed over fewer real checks than n={result.n_questions} suggests.")

    if result.mean_coverage_ratio >= 0:
        print(f"  {'':32} {COVERAGE_RATIO_CALIBRATION_NOTE}")

    if result.by_format:
        print(f"\n  Per-format breakdown:")
        print(f"  {'-'*50}")
        for fmt, m in sorted(result.by_format.items()):
            print(f"  [{fmt}]  n={m['n']}")
            if "accuracy" in m:
                metric_keys = [
                    ("accuracy", "accuracy"),
                    ("mean_accuracy_attempted", "accuracy (attempted)"),
                    ("abstention_rate", "abstention_rate"),
                    ("wrong_rate", "wrong_rate"),
                ]
            else:
                metric_keys = [
                    ("faithfulness",       "faithfulness"),
                    ("answer_relevancy",   "answer_relevancy"),
                    ("context_recall",     "context_recall"),
                    ("answer_correctness", "answer_correctness"),
                ]
            metric_keys += [
                ("mean_similarity",     "mean_similarity"),
                ("mean_distinct_sources_retrieved", "distinct_sources (mean)"),
                ("mean_coverage_ratio", "coverage_ratio"),
                ("canonical_hit_rate",  "canonical_hit_rate"),
                ("exact_chunk_hit_rate", "exact_chunk_hit_rate"),
            ]
            for key, label2 in metric_keys:
                v = m.get(key, -1.0)
                shown = f"{v:.3f}" if v >= 0 else "N/A"
                print(f"      {label2:<20} {shown}")

    print(f"\n  Efficiency:")
    print(f"  {'-'*50}")
    print(f"  {'Latency, mean (s)':<32} total={result.mean_latency:.3f}  "
          f"search={result.mean_search_latency:.3f}  "
          f"rerank={result.mean_rerank_latency:.3f}  "
          f"gen={result.mean_gen_latency:.3f}")
    print(f"  {'Generation tokens':<32} total={result.total_tokens}  "
          f"(prompt={result.total_prompt_tokens}, completion={result.total_completion_tokens}, "
          f"mean/q={result.mean_total_tokens:.1f})")
    print(f"  {'Judge tokens':<32} total={result.total_judge_prompt_tokens + result.total_judge_completion_tokens}  "
          f"(prompt={result.total_judge_prompt_tokens}, completion={result.total_judge_completion_tokens})")
    if result.total_cost is None:
        print(f"  {'Cost':<32} N/A (not computed for any question)")
    else:
        print(f"  {'Cost':<32} total=${result.total_cost:.4f}  "
              f"mean/q=${result.mean_cost:.4f}  "
              f"({result.n_cost_known}/{result.n_questions} questions had a known cost)")
    if result.token_source_counts:
        src = ", ".join(f"{k}={v}" for k, v in sorted(result.token_source_counts.items()))
        print(f"  {'Token source':<32} {src}")

    print(f"{'='*w}\n")