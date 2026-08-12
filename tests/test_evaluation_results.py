"""Tests for open_telco_rag.evaluation.results — the fix for a real design
bug: mc/tf exact-match correctness used to be written into faithfulness/
answer_relevancy/context_recall/answer_correctness (OE-specific metrics,
with answer_relevancy even hardcoded to 1.0 regardless of anything). That
made a mixed mc+tf+open evaluation run's aggregate faithfulness/relevancy
numbers an uninterpretable blend of two different things. accuracy is now
mc/tf's own dedicated field; the four OE metrics stay at their -1.0 "not
applicable" default for mc/tf questions.
"""
from __future__ import annotations

import pytest

from open_telco_rag.evaluation.results import QuestionResult, aggregate


def _mc(accuracy: float, sim: float = 0.8, outcome: str | None = None) -> QuestionResult:
    return QuestionResult(
        question="q", corpus="3gpp", answer="A", sources=[], chunks_used=3,
        answer_type="mc", accuracy=accuracy, mc_tf_outcome=outcome, mean_similarity=sim,
        source_coverage=2, canonical_hit=1, coverage_ratio=0.7,
    )


def _tf(accuracy: float, sim: float = 0.75, outcome: str | None = None) -> QuestionResult:
    return QuestionResult(
        question="q", corpus="3gpp", answer="True", sources=[], chunks_used=3,
        answer_type="tf", accuracy=accuracy, mc_tf_outcome=outcome, mean_similarity=sim,
        source_coverage=1, canonical_hit=0, coverage_ratio=0.6,
    )


def _open(faithfulness: float, relevancy: float, sim: float = 0.6) -> QuestionResult:
    return QuestionResult(
        question="q", corpus="3gpp", answer="some answer", sources=[], chunks_used=5,
        answer_type="open", faithfulness=faithfulness, answer_relevancy=relevancy,
        context_recall=0.9, answer_correctness=0.85, mean_similarity=sim,
        source_coverage=3, canonical_hit=-1, coverage_ratio=0.5,
    )


def test_mixed_mc_tf_open_run_does_not_blend_accuracy_into_oe_metrics():
    """The exact scenario the fix targets: one run with all three formats."""
    results = [
        _mc(1.0), _mc(0.0), _mc(1.0),      # 2/3 mc correct
        _tf(1.0), _tf(1.0),                # 2/2 tf correct
        _open(0.8, 0.9), _open(0.6, 0.7),  # OE metrics only
    ]
    agg = aggregate(results)

    # mc/tf correctness lands in mean_accuracy, computed over ALL 5 mc/tf
    # items together (not blended with the 2 open items' OE scores).
    assert agg["mean_accuracy"] == pytest.approx((1.0 + 0.0 + 1.0 + 1.0 + 1.0) / 5)

    # OE metrics are computed ONLY over the 2 open items, not diluted by
    # the 5 mc/tf items that would previously have contributed 1.0/is_correct
    # values into these same fields.
    assert agg["mean_faithfulness"] == pytest.approx((0.8 + 0.6) / 2)
    assert agg["mean_answer_relevancy"] == pytest.approx((0.9 + 0.7) / 2)


def test_by_format_breakdown_separates_mc_tf_and_open():
    results = [_mc(1.0), _mc(0.0), _tf(1.0), _open(0.8, 0.9)]
    agg = aggregate(results)

    assert set(agg["by_format"].keys()) == {"mc", "tf", "open"}

    mc_entry = agg["by_format"]["mc"]
    assert mc_entry["n"] == 2
    assert mc_entry["accuracy"] == pytest.approx(0.5)
    assert "faithfulness" not in mc_entry  # mc has no OE-metric keys at all

    tf_entry = agg["by_format"]["tf"]
    assert tf_entry["n"] == 1
    assert tf_entry["accuracy"] == pytest.approx(1.0)

    open_entry = agg["by_format"]["open"]
    assert open_entry["n"] == 1
    assert "accuracy" not in open_entry  # open has no accuracy key at all
    assert open_entry["faithfulness"] == pytest.approx(0.8)
    assert open_entry["answer_relevancy"] == pytest.approx(0.9)


def test_mean_accuracy_is_not_applicable_for_an_all_open_run():
    agg = aggregate([_open(0.8, 0.9), _open(0.6, 0.7)])
    assert agg["mean_accuracy"] == -1.0


def test_mean_faithfulness_is_not_applicable_for_an_all_mc_tf_run():
    agg = aggregate([_mc(1.0), _tf(0.0)])
    assert agg["mean_faithfulness"] == -1.0
    assert agg["mean_answer_relevancy"] == -1.0
    assert agg["mean_context_recall"] == -1.0
    assert agg["mean_answer_correctness"] == -1.0


def test_coverage_ratio_aggregates_correctly():
    """Regression test for a real dead-flag bug: --threshold was
    advertised as driving a coverage_ratio metric that was never actually
    computed anywhere. mean_coverage_ratio must now be real."""
    agg = aggregate([_mc(1.0), _tf(1.0)])  # coverage_ratio=0.7 and 0.6
    assert agg["mean_coverage_ratio"] == pytest.approx((0.7 + 0.6) / 2)


def test_empty_question_list_does_not_crash():
    agg = aggregate([])
    assert agg["n_questions"] == 0
    assert agg["mean_accuracy"] == -1.0
    assert agg["by_format"] == {}


# ── mc_tf_outcome three-way breakdown ───────────────────────────────────────
#
# Design requirement, not just a bug fix: a generator honestly abstaining
# ("the context doesn't contain enough information") is a DIFFERENT failure
# than confidently picking the wrong answer -- one points at retrieval/
# chunking (nothing relevant was ever surfaced), the other at generation/
# reasoning (something relevant WAS surfaced, but the model still got it
# wrong). A single accuracy number can't distinguish these; that's what
# mean_accuracy_attempted/abstention_rate/wrong_rate are for.

def test_mc_tf_outcome_rates_computed_across_correct_incorrect_abstained():
    results = [
        _mc(1.0, outcome="correct"),
        _mc(0.0, outcome="correct"),   # accuracy/outcome deliberately independent in this fixture
        _mc(0.0, outcome="incorrect"),
        _mc(0.0, outcome="abstained"),
        _tf(1.0, outcome="correct"),
    ]
    agg = aggregate(results)

    # 3 correct, 1 incorrect, 1 abstained out of 5 total mc/tf questions.
    assert agg["abstention_rate"] == pytest.approx(1 / 5)
    assert agg["wrong_rate"] == pytest.approx(1 / 5)
    # accuracy_attempted excludes the abstained one from the denominator:
    # 3 correct / (3 correct + 1 incorrect) = 3/4, NOT 3/5.
    assert agg["mean_accuracy_attempted"] == pytest.approx(3 / 4)


def test_mc_tf_outcome_rates_not_applicable_when_no_outcomes_tracked():
    """Existing QuestionResults built without mc_tf_outcome (accuracy set
    directly, the pre-this-fix convention) must not silently masquerade
    as "0% abstention" -- -1.0 ("not applicable"), same as every other
    not-computed sentinel in this module."""
    agg = aggregate([_mc(1.0), _mc(0.0)])  # no outcome= passed
    assert agg["mean_accuracy_attempted"] == -1.0
    assert agg["abstention_rate"] == -1.0
    assert agg["wrong_rate"] == -1.0


def test_mc_tf_outcome_rates_not_applicable_for_an_all_open_run():
    agg = aggregate([_open(0.8, 0.9)])
    assert agg["mean_accuracy_attempted"] == -1.0
    assert agg["abstention_rate"] == -1.0
    assert agg["wrong_rate"] == -1.0


def test_by_format_breakdown_includes_outcome_rates_for_mc_and_tf():
    results = [
        _mc(1.0, outcome="correct"),
        _mc(0.0, outcome="abstained"),
        _tf(0.0, outcome="incorrect"),
        _tf(1.0, outcome="correct"),
    ]
    agg = aggregate(results)

    mc_entry = agg["by_format"]["mc"]
    assert mc_entry["abstention_rate"] == pytest.approx(0.5)
    assert mc_entry["mean_accuracy_attempted"] == pytest.approx(1.0)  # 1 correct / (1 correct + 0 incorrect)

    tf_entry = agg["by_format"]["tf"]
    assert tf_entry["wrong_rate"] == pytest.approx(0.5)
    assert tf_entry["mean_accuracy_attempted"] == pytest.approx(0.5)

    # open entries never get the mc/tf-only outcome keys at all.
    assert "abstention_rate" not in agg["by_format"].get("open", {})


# ── Efficiency aggregation (latency/tokens/judge-tokens/cost) ──────────────
#
# Requirement: CorpusResult.aggregate() sums/means these the same way the
# accuracy/faithfulness/etc. metrics already are, against a small fixture
# with known per-question values -- computed by hand below so the
# assertions aren't just re-deriving aggregate()'s own formula.

def _eff(
    latency=1.0, search_latency=0.2, rerank_latency=0.1, gen_latency=0.7,
    prompt_tokens=100, completion_tokens=20, total_tokens=120,
    judge_prompt_tokens=10, judge_completion_tokens=2,
    cost=None, token_source="api",
) -> QuestionResult:
    return QuestionResult(
        question="q", corpus="3gpp", answer="some answer", sources=[], chunks_used=1,
        answer_type="open", faithfulness=0.5, answer_relevancy=0.5, mean_similarity=0.5,
        latency=latency, search_latency=search_latency, rerank_latency=rerank_latency,
        gen_latency=gen_latency, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=total_tokens, judge_prompt_tokens=judge_prompt_tokens,
        judge_completion_tokens=judge_completion_tokens, cost=cost, token_source=token_source,
    )


def test_efficiency_aggregates_sum_and_mean_correctly_against_known_fixture():
    results = [
        _eff(latency=1.0, prompt_tokens=100, completion_tokens=20, total_tokens=120,
             judge_prompt_tokens=10, judge_completion_tokens=2, cost=0.01, token_source="api"),
        _eff(latency=3.0, prompt_tokens=200, completion_tokens=40, total_tokens=240,
             judge_prompt_tokens=30, judge_completion_tokens=6, cost=0.03, token_source="api"),
    ]
    agg = aggregate(results)

    # Sums.
    assert agg["total_prompt_tokens"] == 300
    assert agg["total_completion_tokens"] == 60
    assert agg["total_tokens"] == 360
    assert agg["total_judge_prompt_tokens"] == 40
    assert agg["total_judge_completion_tokens"] == 8
    assert agg["total_cost"] == pytest.approx(0.04)

    # Means -- (1.0 + 3.0) / 2 = 2.0, (100 + 200) / 2 = 150, etc.
    assert agg["mean_latency"] == pytest.approx(2.0)
    assert agg["mean_prompt_tokens"] == pytest.approx(150.0)
    assert agg["mean_total_tokens"] == pytest.approx(180.0)
    assert agg["mean_judge_prompt_tokens"] == pytest.approx(20.0)
    assert agg["mean_cost"] == pytest.approx(0.02)

    assert agg["n_cost_known"] == 2
    assert agg["token_source_counts"] == {"api": 2}


def test_cost_none_when_no_question_has_a_known_cost():
    """A run where every question's cost is unknown (e.g. every question
    ran through claude.py, which has no pricing table) must report
    total_cost/mean_cost as None -- NOT 0.0, which would be
    indistinguishable from a genuinely free local-vLLM run."""
    results = [_eff(cost=None), _eff(cost=None)]
    agg = aggregate(results)

    assert agg["total_cost"] is None
    assert agg["mean_cost"] is None
    assert agg["n_cost_known"] == 0


def test_cost_aggregation_distinguishes_genuinely_free_from_partially_unknown():
    """A mix of a genuinely-free ($0.00) local question and an
    unknown-cost question must sum/average only the KNOWN ones -- the
    known $0.00 contributes a real zero, the unknown one is excluded
    entirely (not treated as an implicit zero)."""
    results = [_eff(cost=0.0), _eff(cost=None)]
    agg = aggregate(results)

    assert agg["total_cost"] == pytest.approx(0.0)
    assert agg["mean_cost"] == pytest.approx(0.0)
    assert agg["n_cost_known"] == 1


def test_token_source_counts_tracks_api_vs_estimated_mix():
    results = [
        _eff(token_source="api"), _eff(token_source="api"), _eff(token_source="estimated"),
    ]
    agg = aggregate(results)
    assert agg["token_source_counts"] == {"api": 2, "estimated": 1}


def test_efficiency_aggregates_default_to_zero_for_empty_question_list():
    agg = aggregate([])
    assert agg["total_tokens"] == 0
    assert agg["mean_latency"] == 0.0
    assert agg["total_cost"] is None
    assert agg["token_source_counts"] == {}


def test_mc_tf_outcome_rates_when_every_mc_tf_question_abstained():
    """Edge case: every mc/tf question abstained -- accuracy_attempted has
    an empty denominator (0 correct + 0 incorrect), must report -1.0
    ("not applicable"), not a ZeroDivisionError or a misleading 0.0."""
    agg = aggregate([_mc(0.0, outcome="abstained"), _tf(0.0, outcome="abstained")])
    assert agg["abstention_rate"] == pytest.approx(1.0)
    assert agg["wrong_rate"] == pytest.approx(0.0)
    assert agg["mean_accuracy_attempted"] == -1.0
