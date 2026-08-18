"""Tests for open_telco_rag.evaluation.metrics.generation.

test_answer_relevancy_empty_reverse_questions_returns_zero_not_nan: regression
test — a validly-empty reverse-questions list used to reach np.mean([]) ->
NaN instead of a clean 0.0; faithfulness() already guarded this,
answer_relevancy() didn't.

test_*_exception_path_logs_a_warning: locks in that exception handlers now
always log a warning instead of silently falling back to 0.0 (previously
indistinguishable from a genuinely bad answer unless debug=True).
"""
from __future__ import annotations

import logging
import math

import pytest

from open_telco_rag.evaluation.metrics.generation import (
    _is_not_found_answer, _is_not_found_answer_mc_tf, answer_relevancy, faithfulness,
)


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeChunk:
    def __init__(self, text: str):
        self.text = text


def test_answer_relevancy_empty_reverse_questions_returns_zero_not_nan():
    llm_call = lambda prompt, max_tokens=256: "[]"  # validly parses, zero items

    score = answer_relevancy("What does the AMF handle?", "Some answer.", _FakeEmbedder(), llm_call=llm_call)

    assert score == 0.0
    assert not math.isnan(score)


def test_answer_relevancy_computes_a_real_score_for_nonempty_reverse_questions():
    llm_call = lambda prompt, max_tokens=256: '["What does the AMF do?", "What handles registration?"]'

    score = answer_relevancy("What does the AMF handle?", "The AMF handles registration.",
                              _FakeEmbedder(), llm_call=llm_call)

    # Identical fake embeddings -> cosine similarity ~1.0 (epsilon keeps it just under 1.0).
    assert score == pytest.approx(1.0)


def test_faithfulness_exception_path_logs_a_warning_without_debug(caplog):
    def _raising_llm_call(prompt, max_tokens=256):
        raise RuntimeError("connection refused")

    with caplog.at_level(logging.WARNING, logger="open_telco_rag.evaluation.metrics.generation"):
        score = faithfulness("The AMF handles registration.", [_FakeChunk("some context")],
                              llm_call=_raising_llm_call)  # debug left at its False default

    assert score == 0.0
    assert any("decompose_claims raised" in r.message for r in caplog.records)
    assert any("RuntimeError" in r.message for r in caplog.records)


def test_answer_relevancy_exception_path_logs_a_warning_without_debug(caplog):
    def _raising_llm_call(prompt, max_tokens=256):
        raise RuntimeError("connection refused")

    with caplog.at_level(logging.WARNING, logger="open_telco_rag.evaluation.metrics.generation"):
        score = answer_relevancy("What does the AMF handle?", "Some answer.", _FakeEmbedder(),
                                  llm_call=_raising_llm_call)

    assert score == 0.0
    assert any("reverse_questions raised" in r.message for r in caplog.records)


# ── _is_not_found_answer vs _is_not_found_answer_mc_tf split ────────────────
#
# Regression: five phrases ("cannot determine", "insufficient information",
# etc.) were added for mc/tf abstention detection, but the SAME phrase list
# also gated faithfulness()/answer_relevancy() -- so a substantively
# correct open-ended answer that happens to use one of them mid-explanation
# got force-zeroed. Split into a narrower open-question list and a broader
# mc/tf-only one.

def test_is_not_found_answer_does_not_match_mc_tf_only_phrases():
    answer = ("There is insufficient information to characterize every edge case, "
              "but the core procedure is X.")
    assert not _is_not_found_answer(answer)
    assert _is_not_found_answer_mc_tf(answer)  # still recognized for mc/tf abstention


def test_is_not_found_answer_still_matches_the_original_retrieval_failure_phrases():
    """The split must not weaken genuine not-found detection for open
    questions -- only the new mc/tf-only additions are excluded."""
    assert _is_not_found_answer("The context does not contain this information.")
    assert _is_not_found_answer_mc_tf("The context does not contain this information.")


def test_faithfulness_not_force_zeroed_by_an_mc_tf_only_phrase_mid_sentence():
    """The regression, end to end: an open-ended answer that's
    substantively correct but happens to say "insufficient information"
    while hedging must be scored normally, not force-zeroed."""
    answer = ("There is insufficient information to characterize every edge case, "
              "but the AMF handles registration and mobility management.")
    responses = iter([
        '["The AMF handles registration and mobility management."]',  # decompose_claims
        '[true]',                                                     # verify_claims
    ])
    llm_call = lambda prompt, max_tokens=512: next(responses)

    score = faithfulness(
        answer, [_FakeChunk("The AMF handles registration and mobility management.")],
        llm_call=llm_call,
    )

    assert score == 1.0


def test_faithfulness_still_force_zeroed_by_a_genuine_not_found_phrase():
    """Confirms the short-circuit for a REAL not-found answer still fires
    before any LLM call -- the split didn't weaken this."""
    def _unreachable(prompt, max_tokens=512):
        pytest.fail("faithfulness should short-circuit before calling llm_call")

    score = faithfulness("The context does not contain this information.",
                          [_FakeChunk("irrelevant")], llm_call=_unreachable)

    assert score == 0.0


def test_answer_relevancy_not_force_zeroed_by_an_mc_tf_only_phrase_mid_sentence():
    answer = ("There is insufficient information to characterize every edge case, "
              "but the AMF handles registration and mobility management.")
    llm_call = lambda prompt, max_tokens=256: '["What does the AMF handle?", "What manages mobility?"]'

    score = answer_relevancy("What does the AMF handle?", answer, _FakeEmbedder(), llm_call=llm_call)

    assert score == pytest.approx(1.0)
