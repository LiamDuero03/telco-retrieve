"""
Result dataclasses, serialisation, and display for evaluation runs.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class QuestionResult:
    """All metric scores for one question."""
    question:    str
    corpus:      str
    answer:      str
    sources:     list[str]
    chunks_used: int

    # answer_type ("mc" / "tf" / "open" / "fill_blank") drives which of the
    # metrics below are actually applicable to THIS question — see
    # evaluate_question()'s branch. Recorded here (not just on the source
    # EvalQuestion) so aggregate() can group by it after the fact.
    answer_type: str = "open"

    # ── EFFICIENCY METRICS  ───────────────────────────────────────
    latency:           float = 0.0
    search_latency:    float = 0.0   # Time spent query embedding + vector search
    rerank_latency:    float = 0.0   # Time spent reranking retrieved chunks
    gen_latency:       float = 0.0   # Time spent generating answer
    prompt_tokens:     int   = 0
    rerank_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0

    # "api" when prompt_tokens/completion_tokens/total_tokens above came
    # from the generator's own real usage block (local.py/openrouter.py/
    # claude.py all return one now — see each module's own comment),
    # "estimated" when no usage was returned at all and evaluate_question()
    # fell back to a local tiktoken approximation instead. "" only for a
    # QuestionResult that was never run through evaluate_question() (e.g.
    # a hand-built test fixture) — downstream analysis (grid reports,
    # leaderboard) can filter/group on this to avoid silently treating an
    # approximation as a measured number.
    token_source:      str   = ""

    # None = not computed at all; a real value (including a genuine 0)
    # means it IS known — e.g. local.py's self-hosted 0.0 (no per-token
    # billing) vs. claude.py's None (no pricing table wired up here yet).
    # Previously these were unconditionally zeroed regardless of which
    # generator ran, which made "genuinely free" and "we never computed
    # this" look identical.
    cached_tokens:     int | None = None
    reasoning_tokens:  int | None = None
    cost:              float | None = None

    # ── Judge/metric-call token usage ─────────────────────────────
    # faithfulness/answer_relevancy/context_recall/answer_correctness each
    # make their own llm_call(s) against the EVAL judge model — tracked
    # SEPARATELY from prompt_tokens/completion_tokens above (which cover
    # RAG *generation* only), because judge cost is roughly constant
    # across chunking strategies for a fixed judge model: folding it into
    # generation tokens would conflate "how efficient is this RAG config"
    # with a cost that doesn't vary with the thing being compared. 0 for
    # mc/tf questions (no judge calls are made at all — see
    # evaluate_question()) and for any llm_call that doesn't track usage
    # (see _claude.py's claude_call/_make_local_call/_make_openrouter_call
    # — a plain callable without those attributes just contributes 0).
    judge_prompt_tokens:     int = 0
    judge_completion_tokens: int = 0

    # ── Deterministic scoring (mc/tf only; -1.0 = not applicable) ────────────
    # Real fix: mc/tf correctness used to be written into faithfulness/
    # answer_relevancy/context_recall/answer_correctness below (with
    # answer_relevancy hardcoded to 1.0 regardless of anything) — those are
    # OE-specific metrics (grounding-in-context, on-topic-via-reverse-
    # question-embedding, etc.) with no meaningful mc/tf analogue. Blending
    # exact-match correctness into them made a mixed mc+tf+open run's
    # aggregate faithfulness/relevancy numbers an uninterpretable average of
    # two different things. accuracy is mc/tf's own dedicated field instead;
    # the four OE metrics below are left at their -1.0 "not applicable"
    # default for mc/tf questions, matching the existing Tier 2 convention.
    accuracy: float = -1.0

    # Three-way classification behind `accuracy` above, mc/tf only:
    # "correct" / "incorrect" / "abstained" (generator honestly said the
    # context was insufficient — see evaluate.py's _match_deterministic_
    # answer() docstring for why this is tracked separately rather than
    # folded into "incorrect"). None for non-mc/tf questions.
    mc_tf_outcome: str | None = None

    # ── Tier 1: no ground truth (open/fill_blank only; -1.0 = not applicable) ─
    faithfulness:     float = -1.0
    answer_relevancy: float = -1.0

    # ── Retrieval metrics: computed for every non-baseline question ──────────
    mean_similarity:  float = 0.0
    min_similarity:   float = 0.0
    max_similarity:   float = 0.0
    source_coverage:  int   = 0
    canonical_hit:    int   = -1   # -1 = not applicable, 0 = miss, 1 = hit
    # Fraction of retrieved chunks meeting the --threshold similarity floor
    # (see metrics.retrieval.coverage_ratio) — previously advertised in the
    # CLI help text ("Similarity threshold for coverage_ratio metric") but
    # never actually computed anywhere; --threshold was silently a no-op.
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

    # Three-way mc/tf breakdown behind mean_accuracy above — see
    # QuestionResult.mc_tf_outcome / evaluate.py's _match_deterministic_
    # answer() docstring for the full "why" (distinguishing a retrieval
    # problem — high abstention — from a generation/reasoning problem —
    # low abstention, high wrong_rate — that a single accuracy number
    # can't tell apart). All -1.0 when there are no mc/tf questions.
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
    mean_source_coverage:  float = 0.0
    canonical_hit_rate:    float = -1.0
    mean_coverage_ratio:   float = -1.0

    # ── Tier 2 aggregates (-1.0 = not computed) ───────────────────────────────
    mean_context_recall:     float = -1.0
    mean_answer_correctness: float = -1.0

    # ── Efficiency aggregates — same per-combination sum/mean treatment as
    # ── the accuracy/faithfulness/etc. aggregates above, so a grid
    # ── comparison across configs doesn't require digging into
    # ── question_results by hand (see _aggregate_efficiency() below). ────────
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

    # Judge-call tokens — see QuestionResult.judge_prompt_tokens' own
    # comment for why these are kept separate from the generation totals
    # above rather than folded in.
    total_judge_prompt_tokens:     int = 0
    total_judge_completion_tokens: int = 0
    mean_judge_prompt_tokens:      float = 0.0
    mean_judge_completion_tokens:  float = 0.0

    # None only when NOT ONE question in this run had a known cost (e.g.
    # every question ran through claude.py, which has no pricing table
    # wired up) — distinguishes "cost entirely unknown" from "$0.00,
    # genuinely free" (a local vLLM run) the same way QuestionResult.cost
    # does per-question. n_cost_known / n_questions tells a reader what
    # fraction of questions actually contributed a real cost figure.
    total_cost:   float | None = None
    mean_cost:    float | None = None
    n_cost_known: int = 0

    # {"api": n, "estimated": n} — how many questions' token counts came
    # from a real usage block vs. the tiktoken fallback (see
    # QuestionResult.token_source). Missing/empty for a run with no
    # question_results at all.
    token_source_counts: dict[str, int] = field(default_factory=dict)

    # Per-answer_type breakdown ("mc"/"tf"/"open"/"fill_blank" -> metrics dict)
    # — the actual fix for running one mixed mc+tf+open set: the flat means
    # above already correctly separate deterministic accuracy from OE
    # metrics (see QuestionResult), but a user comparing e.g. mc vs. tf
    # specifically, or just wanting per-format counts, needs this finer
    # breakdown rather than a single "mc+tf combined" accuracy number.
    by_format: dict[str, dict] = field(default_factory=dict)

    question_results: list[QuestionResult] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    """Mean of non-negative values only — -1.0 (the shared "not
    applicable/not computed" sentinel throughout this module) is excluded,
    not treated as a real zero score. Module-level so both aggregate() and
    _aggregate_one_format() below share exactly one implementation."""
    valid = [v for v in values if v >= 0]
    return statistics.mean(valid) if valid else -1.0


def _mc_tf_outcome_rates(results: list[QuestionResult]) -> dict:
    """mean_accuracy_attempted/abstention_rate/wrong_rate from a list of
    QuestionResults' mc_tf_outcome field — see CorpusResult's own comment
    for exactly what each measures and why. `results` may be mixed-format
    (only entries with a non-None mc_tf_outcome are counted) or already
    mc/tf-only (_aggregate_one_format's per-format call). All three are
    -1.0 ("not applicable") when there are no mc/tf questions at all.
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
    their own accuracy figure (plus the three-way outcome breakdown);
    open/fill_blank get the four OE metrics. Retrieval metrics are
    reported for every format, since those are computed regardless of
    answer_type.
    """
    canonical = [r for r in results if r.canonical_hit >= 0]
    entry = {
        "n":                    len(results),
        "mean_similarity":      _mean([r.mean_similarity for r in results]),
        "mean_source_coverage": _mean([float(r.source_coverage) for r in results]),
        "canonical_hit_rate":   (sum(r.canonical_hit for r in canonical) / len(canonical)
                                  if canonical else -1.0),
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
    """Plain mean, 0.0 for an empty list — unlike _mean() above, none of
    the efficiency fields use -1.0 as an "not applicable" sentinel (every
    question that didn't error has a real, if possibly estimated, value
    for latency/tokens), so there's nothing to filter out here."""
    return statistics.mean(values) if values else 0.0


def _aggregate_efficiency(results: list[QuestionResult]) -> dict:
    """Latency/token/judge-token/cost aggregates for CorpusResult — see its
    own comment for what each field means. A straight sum/mean over every
    result (no sentinel filtering, unlike _mean()): an errored question
    contributes its exception-handler defaults (0.0/0/None), which dilutes
    the mean toward zero rather than being excluded — the same "still
    counts as a data point" treatment errored questions already get
    everywhere else aggregate() doesn't special-case them.
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


def aggregate(question_results: list[QuestionResult]) -> dict:
    """Compute aggregate metrics from a list of QuestionResults."""

    canonical_qs = [r for r in question_results if r.canonical_hit >= 0]

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
        mean_source_coverage=    _mean([float(r.source_coverage) for r in question_results]),
        canonical_hit_rate=     (sum(r.canonical_hit for r in canonical_qs) /
                                  len(canonical_qs) if canonical_qs else -1.0),
        mean_coverage_ratio=     _mean([r.coverage_ratio     for r in question_results]),
        mean_context_recall=     _mean([r.context_recall     for r in question_results]),
        mean_answer_correctness= _mean([r.answer_correctness for r in question_results]),
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
        ("Mean similarity",       result.mean_similarity,         "Retrieval"),
        ("Source coverage (mean)",result.mean_source_coverage,    "Retrieval"),
        ("Coverage ratio (mean)", result.mean_coverage_ratio,     "Retrieval"),
        ("Canonical hit rate",    result.canonical_hit_rate,      "Retrieval"),
        ("Context recall (open)", result.mean_context_recall,     "Retrieval+GT"),
        ("Answer correctness (open)", result.mean_answer_correctness, "Generation+GT"),
    ]
    for label, value, kind in rows:
        if value < 0:
            print(f"  {label:<32} {'N/A':>8}  {kind}")
        else:
            fmt = ".1f" if label == "Source coverage (mean)" else ".3f"
            print(f"  {label:<32} {value:>8{fmt}}  {kind}")

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
                ("mean_coverage_ratio", "coverage_ratio"),
                ("canonical_hit_rate",  "canonical_hit_rate"),
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