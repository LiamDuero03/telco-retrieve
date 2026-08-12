"""Tests for open_telco_rag.evaluation.metrics.generation.

test_answer_relevancy_empty_reverse_questions_returns_zero_not_nan is a
regression test for a real bug found via testing evaluate_question(): an
LLM response that validly parses to an EMPTY list (not malformed, just
zero reverse-questions) passed the `isinstance(reverse_qs, list)` check
and reached np.mean([]) further down, which returns NaN (with a
RuntimeWarning) instead of raising or returning a clean fallback --
silently poisoning that question's answer_relevancy score. faithfulness()
already had the equivalent empty-list guard for its own analogous
empty-claims case; answer_relevancy() didn't.

The test_*_exception_path_logs_a_warning tests cover cleanup step 4
(Production Auditor): every except Exception block in this module used to
fall back to 0.0 silently unless debug=True (opt-in, off by default) --
indistinguishable from a genuinely low-quality answer in a real (non-debug)
grid run. They now always log a warning first; these tests lock that in
via caplog rather than requiring debug=True to see it.
"""
from __future__ import annotations

import logging
import math

import pytest

from open_telco_rag.evaluation.metrics.generation import answer_relevancy, faithfulness


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

    # Same fake embedding for every text -> cosine similarity of ~1.0 for
    # each reverse question against the original (the metric's own
    # 1e-10 epsilon in the denominator keeps it just under exactly 1.0).
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
