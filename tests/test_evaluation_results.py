"""Tests for the fix to a design bug: mc/tf exact-match correctness used to
be written into the OE-only metrics (faithfulness/answer_relevancy/
context_recall/answer_correctness), blending two different things together
in a mixed mc+tf+open run. accuracy is now mc/tf's own field; the four OE
metrics stay at their -1.0 "not applicable" default for mc/tf questions.
"""
from __future__ import annotations

import pytest

from open_telco_rag.evaluation.results import QuestionResult, aggregate


def _mc(accuracy: float, sim: float = 0.8, outcome: str | None = None) -> QuestionResult:
    return QuestionResult(
        question="q", corpus="3gpp", answer="A", sources=[], chunks_used=3,
        answer_type="mc", accuracy=accuracy, mc_tf_outcome=outcome, mean_similarity=sim,
        distinct_sources_retrieved=2, canonical_hit=1, coverage_ratio=0.7,
    )


def _tf(accuracy: float, sim: float = 0.75, outcome: str | None = None) -> QuestionResult:
    return QuestionResult(
        question="q", corpus="3gpp", answer="True", sources=[], chunks_used=3,
        answer_type="tf", accuracy=accuracy, mc_tf_outcome=outcome, mean_similarity=sim,
        distinct_sources_retrieved=1, canonical_hit=0, coverage_ratio=0.6,
    )


def _open(faithfulness: float, relevancy: float, sim: float = 0.6) -> QuestionResult:
    return QuestionResult(
        question="q", corpus="3gpp", answer="some answer", sources=[], chunks_used=5,
        answer_type="open", faithfulness=faithfulness, answer_relevancy=relevancy,
        context_recall=0.9, answer_correctness=0.85, mean_similarity=sim,
        distinct_sources_retrieved=3, canonical_hit=-1, coverage_ratio=0.5,
    )


def test_mixed_mc_tf_open_run_does_not_blend_accuracy_into_oe_metrics():
    """The exact scenario the fix targets: one run with all three formats."""
    results = [
        _mc(1.0), _mc(0.0), _mc(1.0),      # 2/3 mc correct
        _tf(1.0), _tf(1.0),                # 2/2 tf correct
        _open(0.8, 0.9), _open(0.6, 0.7),  # OE metrics only
    ]
    agg = aggregate(results)

    # mc/tf correctness lands in mean_accuracy over all 5 mc/tf items, not blended with open.
    assert agg["mean_accuracy"] == pytest.approx((1.0 + 0.0 + 1.0 + 1.0 + 1.0) / 5)

    # OE metrics computed only over the 2 open items, not diluted by mc/tf.
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
    """Regression: --threshold was advertised as driving coverage_ratio,
    but it was never actually computed."""
    agg = aggregate([_mc(1.0), _tf(1.0)])  # coverage_ratio=0.7 and 0.6
    assert agg["mean_coverage_ratio"] == pytest.approx((0.7 + 0.6) / 2)


def test_empty_question_list_does_not_crash():
    agg = aggregate([])
    assert agg["n_questions"] == 0
    assert agg["mean_accuracy"] == -1.0
    assert agg["by_format"] == {}


# ── mc_tf_outcome three-way breakdown ───────────────────────────────────────
# Design requirement: honest abstention (retrieval gap) is a different failure
# than a confident wrong answer (generation error) - a single accuracy number
# can't distinguish them; hence mean_accuracy_attempted/abstention_rate/wrong_rate.

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
    """Old-convention QuestionResults (accuracy set directly, no outcome)
    must not silently read as "0% abstention"; must be -1.0 like other
    sentinels."""
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
# Sums/means computed by hand against a known fixture, so assertions don't
# just re-derive aggregate()'s own formula.

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

    assert agg["total_prompt_tokens"] == 300
    assert agg["total_completion_tokens"] == 60
    assert agg["total_tokens"] == 360
    assert agg["total_judge_prompt_tokens"] == 40
    assert agg["total_judge_completion_tokens"] == 8
    assert agg["total_cost"] == pytest.approx(0.04)

    # Means: (1.0+3.0)/2=2.0, (100+200)/2=150, etc.
    assert agg["mean_latency"] == pytest.approx(2.0)
    assert agg["mean_prompt_tokens"] == pytest.approx(150.0)
    assert agg["mean_total_tokens"] == pytest.approx(180.0)
    assert agg["mean_judge_prompt_tokens"] == pytest.approx(20.0)
    assert agg["mean_cost"] == pytest.approx(0.02)

    assert agg["n_cost_known"] == 2
    assert agg["token_source_counts"] == {"api": 2}


def test_cost_none_when_no_question_has_a_known_cost():
    """All-unknown-cost run (e.g. claude.py, no pricing table) must report
    None, not 0.0 (which would look like a genuinely free local-vLLM run)."""
    results = [_eff(cost=None), _eff(cost=None)]
    agg = aggregate(results)

    assert agg["total_cost"] is None
    assert agg["mean_cost"] is None
    assert agg["n_cost_known"] == 0


def test_cost_aggregation_distinguishes_genuinely_free_from_partially_unknown():
    """Mix of genuinely-free ($0.00) and unknown-cost: only known costs
    are summed/averaged; unknown is excluded, not treated as an implicit zero."""
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
    """All-abstained edge case: empty denominator must give -1.0, not
    ZeroDivisionError or a misleading 0.0."""
    agg = aggregate([_mc(0.0, outcome="abstained"), _tf(0.0, outcome="abstained")])
    assert agg["abstention_rate"] == pytest.approx(1.0)
    assert agg["wrong_rate"] == pytest.approx(0.0)
    assert agg["mean_accuracy_attempted"] == -1.0


# ── weighted_score_pct - the new composite metric ───────────────────────────
#
# score_pct = (1*mc_correct + 1*tf_correct + 5*oe_correct)
#             / (1*mc_total  + 1*tf_total  + 5*oe_total)
#             * 100
# mc/tf: outcome == "correct" counts, abstained scores 0 (like accuracy itself).
# open: answer_correctness >= OE_CORRECTNESS_THRESHOLD (0.7) counts; no
# reference_answer (-1.0) is excluded from both numerator and denominator.

def _open_scored(answer_correctness: float) -> QuestionResult:
    """Like _open(), but exposes answer_correctness directly since _open()
    hardcodes 0.85."""
    return QuestionResult(
        question="q", corpus="3gpp", answer="some answer", sources=[], chunks_used=5,
        answer_type="open", faithfulness=0.7, answer_relevancy=0.7,
        context_recall=0.9, answer_correctness=answer_correctness, mean_similarity=0.6,
        distinct_sources_retrieved=3, canonical_hit=-1, coverage_ratio=0.5,
    )


def _open_no_reference_answer() -> QuestionResult:
    """No reference_answer -> answer_correctness stays at its -1.0 default;
    must be excluded from oe_total entirely, not counted as wrong."""
    return QuestionResult(
        question="q", corpus="3gpp", answer="some answer", sources=[], chunks_used=5,
        answer_type="open", faithfulness=0.7, answer_relevancy=0.7,
        mean_similarity=0.6, distinct_sources_retrieved=3, canonical_hit=-1, coverage_ratio=0.5,
    )


def test_weighted_score_pct_matches_hand_calculated_value_from_a_known_fixture():
    """3 mc (2 correct, 1 incorrect) + 2 tf (1 correct, 1 abstained) +
    3 open (0.8/0.75 >= 0.7 threshold -> correct, 0.4 -> not) + 1 open
    with no reference_answer (excluded).

    mc_tf_correct=3, mc_tf_total=5, oe_correct=2, oe_total=3
    numerator=3+5*2=13, denominator=5+5*3=20 -> 13/20*100=65.0
    """
    results = [
        _mc(1.0, outcome="correct"),
        _mc(1.0, outcome="correct"),
        _mc(0.0, outcome="incorrect"),
        _tf(1.0, outcome="correct"),
        _tf(0.0, outcome="abstained"),
        _open_scored(0.8),
        _open_scored(0.75),
        _open_scored(0.4),
        _open_no_reference_answer(),
    ]
    agg = aggregate(results)
    assert agg["weighted_score_pct"] == pytest.approx(65.0)


def test_weighted_score_pct_abstained_mc_tf_counts_as_not_correct():
    """Abstained mc/tf scores 0 points like a wrong answer, unlike the
    exclusion acc_attempted gives it."""
    results = [_mc(0.0, outcome="abstained"), _mc(0.0, outcome="abstained")]
    agg = aggregate(results)
    assert agg["weighted_score_pct"] == pytest.approx(0.0)


def test_weighted_score_pct_open_question_below_threshold_scores_zero_points():
    results = [_open_scored(0.69)]  # just under the 0.7 default threshold
    agg = aggregate(results)
    assert agg["weighted_score_pct"] == pytest.approx(0.0)


def test_weighted_score_pct_open_question_at_exactly_threshold_counts_as_correct():
    results = [_open_scored(0.7)]  # >= threshold, per the spec's own ">="
    agg = aggregate(results)
    assert agg["weighted_score_pct"] == pytest.approx(100.0)


def test_weighted_score_pct_not_applicable_when_denominator_is_zero():
    """All-excluded run (e.g. no-reference-answer opens only): must be
    -1.0, not ZeroDivisionError or a misleading 0.0."""
    results = [_open_no_reference_answer(), _open_no_reference_answer()]
    agg = aggregate(results)
    assert agg["weighted_score_pct"] == -1.0


def test_weighted_score_pct_open_weighs_five_x_an_mc_tf_question():
    """Proof of the 5x weighting: one wrong open question pulls the score
    down much further than one wrong mc/tf would."""
    mc_tf_only = aggregate([_mc(1.0, outcome="correct")])
    assert mc_tf_only["weighted_score_pct"] == pytest.approx(100.0)

    with_wrong_open = aggregate([_mc(1.0, outcome="correct"), _open_scored(0.0)])
    # numerator = 1 + 5*0 = 1, denominator = 1 + 5*1 = 6 -> 1/6*100
    assert with_wrong_open["weighted_score_pct"] == pytest.approx(100 / 6)

    with_wrong_mc = aggregate([_mc(1.0, outcome="correct"), _mc(0.0, outcome="incorrect")])
    # numerator=1, denominator=2 -> 50.0, a much smaller penalty, confirming the 5x weight.
    assert with_wrong_mc["weighted_score_pct"] == pytest.approx(50.0)
