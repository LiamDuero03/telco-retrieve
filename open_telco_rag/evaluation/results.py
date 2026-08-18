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
# Default threshold (0.5) is empirically uninformative on this corpus — clears
# ~100% of retrieval across 24 grid combos, though the metric mechanism itself
# works correctly. Recalibrate threshold deliberately; don't silently change it.
COVERAGE_RATIO_CALIBRATION_NOTE = (
    "coverage_ratio's default threshold (0.5) is empirically uninformative "
    "on this corpus -- it clears ~100% of real retrieval across every "
    "combination, even though the metric itself correctly discriminates "
    "genuinely bad retrieval under direct testing. Treat mean_coverage_ratio "
    "as an uncalibrated signal, not a trustworthy comparison, until the "
    "threshold is deliberately recalibrated."
)

# Threshold above which an open/fill_blank answer_correctness counts as
# "correct" for weighted_score_pct binarization (see _weighted_score_pct).
# Judgment call, not derived — kept as a named constant, not a magic literal.
OE_CORRECTNESS_THRESHOLD = 0.7


@dataclass
class QuestionResult:
    """All metric scores for one question."""
    question:    str
    corpus:      str
    answer:      str
    sources:     list[str]
    chunks_used: int

    # Drives which metrics below apply to this question (evaluate_question()'s
    # branch). Stored here too so aggregate() can group by it.
    answer_type: str = "open"

    # ── EFFICIENCY METRICS  ───────────────────────────────────────
    latency:           float = 0.0
    search_latency:    float = 0.0   # embedding + vector search
    rerank_latency:    float = 0.0   # reranking retrieved chunks
    gen_latency:       float = 0.0   # answer generation
    prompt_tokens:     int   = 0
    rerank_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0

    # "api": from the generator's real usage block. "estimated": no usage
    # returned, evaluate_question() fell back to tiktoken approximation.
    # "": never run through evaluate_question() (e.g. hand-built fixture).
    # Downstream analysis filters/groups on this to avoid treating an
    # approximation as measured.
    token_source:      str   = ""

    # None = not computed; a real value (incl. 0) means it IS known — e.g.
    # local.py's self-hosted 0.0 vs. claude.py's None (no pricing table yet).
    # Distinguishes "genuinely free" from "never computed".
    cached_tokens:     int | None = None
    reasoning_tokens:  int | None = None
    cost:              float | None = None

    # ── Judge/metric-call token usage ─────────────────────────────
    # Tracked separately from prompt_tokens/completion_tokens (RAG generation
    # only) because judge cost is ~constant across chunking strategies for a
    # fixed judge model — folding it in would conflate RAG efficiency with a
    # cost that doesn't vary with what's being compared. 0 for mc/tf (no judge
    # calls) and for any llm_call that doesn't track usage.
    judge_prompt_tokens:     int = 0
    judge_completion_tokens: int = 0

    # ── Deterministic scoring (mc/tf only; -1.0 = not applicable) ────────────
    # mc/tf correctness used to be written into faithfulness/answer_relevancy/
    # context_recall/answer_correctness below — those are OE-specific metrics
    # with no mc/tf analogue, so blending exact-match into them made mixed
    # mc+tf+open aggregates uninterpretable. accuracy is mc/tf's own field;
    # the four OE metrics stay at -1.0 for mc/tf questions.
    accuracy: float = -1.0

    # Three-way outcome behind `accuracy`: "correct" / "incorrect" /
    # "abstained" (generator honestly said context was insufficient — see
    # evaluate.py's _match_deterministic_answer()). None for non-mc/tf.
    mc_tf_outcome: str | None = None

    # ── Tier 1: no ground truth (open/fill_blank only; -1.0 = not applicable) ─
    faithfulness:     float = -1.0
    answer_relevancy: float = -1.0

    # ── Retrieval metrics: computed for every non-baseline question ──────────
    mean_similarity:  float = 0.0
    min_similarity:   float = 0.0
    max_similarity:   float = 0.0
    # Raw count of distinct source docs among retrieved chunks, in [0, top_k]
    # — NOT a 0-1 ratio (renamed from source_coverage, which implied one).
    distinct_sources_retrieved: int = 0
    canonical_hit:    int   = -1   # -1 = not applicable, 0 = miss, 1 = hit
    # Why canonical_hit is -1: None = no canonical_source for this question;
    # False = canonical_source given but is_discriminating_source() rejected
    # it as too generic to trust (e.g. "raw.md", a converter default filename
    # common across this corpus's doc_ids); True = a real identifier was
    # checked. Separate from canonical_hit's -1 because these are different
    # problems (question-authoring gap vs. corpus/metadata quality gap).
    canonical_source_discriminating: bool | None = None
    # Stronger than canonical_hit: did the EXACT chunk this question was
    # generated from come back, not just any chunk from the right document —
    # see metrics.retrieval.exact_chunk_hit(). Same -1/0/1 convention;
    # -1 when the question has no recorded chunk_id/source_text at all.
    exact_chunk_hit:  int   = -1
    # Fraction of retrieved chunks meeting the --threshold similarity floor.
    # Previously advertised in CLI help but never computed — --threshold was
    # a silent no-op. Metric mechanism itself works (0.0 for off-topic, 1.0
    # for relevant, verified against real embeddings), but the default
    # threshold (0.5) is empirically uninformative on this corpus — flat
    # 1.000 across all 24 grid combinations. See CorpusResult's comment;
    # left unchanged pending a deliberate threshold-calibration decision.
    coverage_ratio:   float = -1.0

    # ── Tier 2: ground truth required (open/fill_blank only; -1.0 = not computed) ─
    context_recall:     float = -1.0
    answer_correctness: float = -1.0

    error: str = ""


@dataclass
class CorpusResult:
    """Aggregate results across all questions for one corpus."""
    corpus:    str
    model:     str = ""
    top_k:     int = 5
    timestamp: str = ""

    questions_file: str = ""

    n_questions: int = 0

    # ── Deterministic (mc/tf) aggregate, -1.0 = no mc/tf questions in this run ──
    mean_accuracy: float = -1.0

    # Three-way mc/tf breakdown behind mean_accuracy — see
    # QuestionResult.mc_tf_outcome. Distinguishes a retrieval problem (high
    # abstention) from a generation/reasoning problem (low abstention, high
    # wrong_rate) that a single accuracy number can't tell apart. All -1.0
    # when there are no mc/tf questions.
    #   mean_accuracy            = correct / n_mc_tf              (unchanged, all mc/tf)
    #   mean_accuracy_attempted  = correct / (correct + incorrect)  (abstentions excluded)
    #   abstention_rate          = abstained / n_mc_tf
    #   wrong_rate               = incorrect / n_mc_tf
    mean_accuracy_attempted: float = -1.0
    abstention_rate:         float = -1.0
    wrong_rate:               float = -1.0

    # ── Tier 1 aggregates (open/fill_blank only now — see QuestionResult) ──────
    mean_faithfulness:     float = -1.0
    mean_answer_relevancy: float = -1.0

    # ── Retrieval aggregates — computed across every non-baseline question,
    # ── regardless of answer_type ────────────────────────────────────────────
    mean_similarity:       float = 0.0
    # Mean of the raw distinct-source-document count — legitimately > 1.0,
    # since it averages an integer count in [0, top_k], not a ratio.
    mean_distinct_sources_retrieved: float = 0.0
    canonical_hit_rate:    float = -1.0
    # Count of questions whose canonical_source was rejected as too generic
    # by is_discriminating_source() (see QuestionResult.
    # canonical_source_discriminating). >0 means canonical_hit_rate is
    # computed over fewer real checks than n_questions suggests; if all
    # questions are rejected, canonical_hit_rate falls back to -1.0 instead
    # of silently reporting a meaningless 1.000.
    n_canonical_source_non_discriminating: int = 0
    # Stronger companion to canonical_hit_rate — see QuestionResult.
    # exact_chunk_hit. -1.0 when no question in this run has chunk provenance.
    exact_chunk_hit_rate:  float = -1.0
    mean_coverage_ratio:   float = -1.0

    # ── Tier 2 aggregates (-1.0 = not computed) ───────────────────────────────
    mean_context_recall:     float = -1.0
    mean_answer_correctness: float = -1.0

    # ── Efficiency aggregates — sum/mean per combination, so grid comparisons
    # ── don't require digging into question_results by hand ─────────────────
    mean_latency:            float = 0.0
    mean_search_latency:     float = 0.0
    mean_rerank_latency:     float = 0.0
    mean_gen_latency:        float = 0.0

    total_prompt_tokens:     int   = 0
    total_completion_tokens: int   = 0
    total_tokens:            int   = 0
    mean_prompt_tokens:      float = 0.0
    mean_completion_tokens:  float = 0.0
    mean_total_tokens:       float = 0.0

    # Judge-call tokens, kept separate from generation totals above (see
    # QuestionResult.judge_prompt_tokens).
    total_judge_prompt_tokens:     int = 0
    total_judge_completion_tokens: int = 0
    mean_judge_prompt_tokens:      float = 0.0
    mean_judge_completion_tokens:  float = 0.0

    # None only when no question in this run had a known cost (e.g. all ran
    # through claude.py, no pricing table wired up) — distinguishes "unknown"
    # from "$0.00, genuinely free" (local vLLM), same as QuestionResult.cost.
    total_cost:   float | None = None
    mean_cost:    float | None = None
    n_cost_known: int = 0

    # {"api": n, "estimated": n} — see QuestionResult.token_source.
    token_source_counts: dict[str, int] = field(default_factory=dict)

    # Weighted score %: mc/tf correct = 1pt, open/fill_blank correct
    # (answer_correctness >= OE_CORRECTNESS_THRESHOLD) = 5pts, since OE
    # questions test generation quality more strongly. -1.0 when there are
    # no scorable questions. See _weighted_score_pct() for the full formula.
    weighted_score_pct: float = -1.0

    # Per-answer_type breakdown ("mc"/"tf"/"open"/"fill_blank" -> metrics
    # dict), so mc vs. tf (or per-format counts) can be compared directly
    # instead of only via the combined flat means above.
    by_format: dict[str, dict] = field(default_factory=dict)

    question_results: list[QuestionResult] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    """Mean of non-negative values; excludes the -1.0 "not applicable"
    sentinel rather than treating it as a real zero score."""
    valid = [v for v in values if v >= 0]
    return statistics.mean(valid) if valid else -1.0


def _mc_tf_outcome_rates(results: list[QuestionResult]) -> dict:
    """mean_accuracy_attempted/abstention_rate/wrong_rate from mc_tf_outcome
    (see CorpusResult for what each means). `results` may be mixed-format —
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
    """Per-answer_type metrics dict for CorpusResult.by_format — mc/tf get
    accuracy + outcome breakdown; open/fill_blank get the four OE metrics.
    Retrieval metrics are reported for every format.
    """
    canonical = [r for r in results if r.canonical_hit >= 0]
    exact_chunk = [r for r in results if r.exact_chunk_hit >= 0]
    entry = {
        "n":                    len(results),
        "mean_similarity":      _mean([r.mean_similarity for r in results]),
        "mean_distinct_sources_retrieved": _mean([float(r.distinct_sources_retrieved) for r in results]),
        "canonical_hit_rate":   (sum(r.canonical_hit for r in canonical) / len(canonical)
                                  if canonical else -1.0),
        "n_canonical_source_non_discriminating": sum(
            1 for r in results if r.canonical_source_discriminating is False
        ),
        "exact_chunk_hit_rate": (sum(r.exact_chunk_hit for r in exact_chunk) / len(exact_chunk)
                                  if exact_chunk else -1.0),
        "mean_coverage_ratio":  _mean([r.coverage_ratio for r in results]),
    }
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
    """Plain mean, 0.0 for empty — unlike _mean(), efficiency fields don't
    use -1.0 as a sentinel, so nothing needs filtering."""
    return statistics.mean(values) if values else 0.0


def _aggregate_efficiency(results: list[QuestionResult]) -> dict:
    """Latency/token/judge-token/cost aggregates for CorpusResult. Straight
    sum/mean over every result, no sentinel filtering — an errored question's
    0.0/0/None defaults still count as a data point, same as elsewhere.
    """
    prompt_tokens     = [r.prompt_tokens for r in results]
    completion_tokens = [r.completion_tokens for r in results]
    total_tokens       = [r.total_tokens for r in results]
    judge_prompt       = [r.judge_prompt_tokens for r in results]
    judge_completion   = [r.judge_completion_tokens for r in results]

    known_costs = [r.cost for r in results if r.cost is not None]

    token_source_counts: dict[str, int] = {}
    for r in results:
        if r.token_source:
            token_source_counts[r.token_source] = token_source_counts.get(r.token_source, 0) + 1

    return dict(
        mean_latency=        _mean_or_zero([r.latency for r in results]),
        mean_search_latency= _mean_or_zero([r.search_latency for r in results]),
        mean_rerank_latency= _mean_or_zero([r.rerank_latency for r in results]),
        mean_gen_latency=    _mean_or_zero([r.gen_latency for r in results]),

        total_prompt_tokens=     sum(prompt_tokens),
        total_completion_tokens= sum(completion_tokens),
        total_tokens=             sum(total_tokens),
        mean_prompt_tokens=      _mean_or_zero(prompt_tokens),
        mean_completion_tokens=  _mean_or_zero(completion_tokens),
        mean_total_tokens=       _mean_or_zero(total_tokens),

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

        score_pct = (1*mc_correct + 1*tf_correct + 5*oe_correct)
                    / (1*mc_total  + 1*tf_total  + 5*oe_total)
                    * 100

    Open-ended questions weight 5x an mc/tf question, since they test
    generation quality more strongly than exact-match. mc+tf combined into
    one bucket (mathematically equivalent, since both carry weight 1).

    mc_correct/tf_correct: mc_tf_outcome == "correct" — abstention counts as
    0 points (same treatment `accuracy` gives it): an honest "I don't know"
    is useful info, but shouldn't outscore a genuine wrong attempt here.

    oe_correct: answer_correctness >= OE_CORRECTNESS_THRESHOLD. A question
    with no reference_answer (answer_correctness left at -1.0) is excluded
    from both numerator and oe_total — no reference means correctness can't
    be judged at all, a data gap rather than a performance signal like
    abstention.

    -1.0 when the weighted denominator is 0 (no scorable questions at all).
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


def aggregate(question_results: list[QuestionResult]) -> dict:
    """Compute aggregate metrics from a list of QuestionResults."""

    canonical_qs = [r for r in question_results if r.canonical_hit >= 0]
    exact_chunk_qs = [r for r in question_results if r.exact_chunk_hit >= 0]

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
        mean_similarity=         _mean([r.mean_similarity    for r in question_results]),
        mean_distinct_sources_retrieved=
                                  _mean([float(r.distinct_sources_retrieved) for r in question_results]),
        canonical_hit_rate=     (sum(r.canonical_hit for r in canonical_qs) /
                                  len(canonical_qs) if canonical_qs else -1.0),
        n_canonical_source_non_discriminating=
                                  sum(1 for r in question_results if r.canonical_source_discriminating is False),
        exact_chunk_hit_rate=   (sum(r.exact_chunk_hit for r in exact_chunk_qs) /
                                  len(exact_chunk_qs) if exact_chunk_qs else -1.0),
        mean_coverage_ratio=     _mean([r.coverage_ratio     for r in question_results]),
        mean_context_recall=     _mean([r.context_recall     for r in question_results]),
        mean_answer_correctness= _mean([r.answer_correctness for r in question_results]),
        weighted_score_pct=      _weighted_score_pct(question_results),
        by_format=by_format,
        **_aggregate_efficiency(question_results),
    )


def save_results(result: CorpusResult, output_dir: Path) -> Path:
    """Serialise CorpusResult to a timestamped JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts       = result.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{result.corpus}_{result.model}_{ts}.json"
    path     = output_dir / filename
    path.write_text(json.dumps(asdict(result), indent=2, default=str),
                    encoding="utf-8")
    print(f"  Saved → {path}")
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
    for label, value, kind in rows:
        if value < 0:
            print(f"  {label:<32} {'N/A':>8}  {kind}")
        else:
            fmt = ".1f" if label == "Distinct sources (mean count)" else ".3f"
            print(f"  {label:<32} {value:>8{fmt}}  {kind}")

    if result.n_canonical_source_non_discriminating > 0:
        print(f"  {'':32} ⚠ {result.n_canonical_source_non_discriminating} question(s) had a "
              f"canonical_source too generic to trust (see canonical_source_discriminating) --")
        print(f"  {'':32}   canonical_hit_rate above is computed over fewer real checks than n={result.n_questions} suggests.")

    if result.mean_coverage_ratio >= 0:
        print(f"  {'':32} ⚠ {COVERAGE_RATIO_CALIBRATION_NOTE}")

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