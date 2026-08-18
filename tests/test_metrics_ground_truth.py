"""Tests for open_telco_rag.evaluation.metrics.ground_truth (previously
uncovered). Exception handlers used to fall back to 0.0 silently unless
debug=True — indistinguishable from a genuinely bad answer in a non-debug
grid run — and now always log a warning first, verified here via caplog.
"""
from __future__ import annotations

import logging

import pytest

from open_telco_rag.evaluation.metrics.ground_truth import answer_correctness, context_recall


class _FakeChunk:
    def __init__(self, text: str):
        self.text = text


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _raising_llm_call(prompt, max_tokens=256):
    raise RuntimeError("connection refused")


# ── context_recall ───────────────────────────────────────────────────────────

def test_context_recall_computes_a_real_score():
    llm_call = lambda prompt, max_tokens=256: '{"total": 4, "supported": 3}'

    score = context_recall("The AMF handles registration and mobility.",
                            [_FakeChunk("The AMF handles registration.")], llm_call=llm_call)

    assert score == pytest.approx(0.75)


def test_context_recall_caps_at_one_even_if_supported_exceeds_total():
    # Guards against a judge miscounting; score must not exceed 1.0.
    llm_call = lambda prompt, max_tokens=256: '{"total": 2, "supported": 5}'

    score = context_recall("ref", [_FakeChunk("ctx")], llm_call=llm_call)

    assert score == 1.0


def test_context_recall_exception_path_logs_a_warning_without_debug(caplog):
    with caplog.at_level(logging.WARNING, logger="open_telco_rag.evaluation.metrics.ground_truth"):
        score = context_recall("ref", [_FakeChunk("ctx")], llm_call=_raising_llm_call)

    assert score == 0.0
    assert any("context_recall raised" in r.message for r in caplog.records)
    assert any("RuntimeError" in r.message for r in caplog.records)


# ── answer_correctness ───────────────────────────────────────────────────────

def test_answer_correctness_blends_f1_and_semantic_similarity():
    llm_call = lambda prompt, max_tokens=256: '{"correct": 2, "incorrect": 0, "missing": 0}'

    # Identical embeddings -> sem~1.0; correct=2/incorrect=0/missing=0 -> f1=1.0.
    score = answer_correctness("The AMF handles registration.", "The AMF handles registration.",
                                _FakeEmbedder(), llm_call=llm_call)

    assert score == pytest.approx(1.0)


def test_answer_correctness_f1_exception_path_logs_a_warning_and_still_scores_semantic(caplog):
    with caplog.at_level(logging.WARNING, logger="open_telco_rag.evaluation.metrics.ground_truth"):
        score = answer_correctness("answer", "reference", _FakeEmbedder(),
                                    llm_call=_raising_llm_call)

    # f1 and semantic halves use separate try/except; f1 fails, sem still computes (weight=0.5).
    assert score == pytest.approx(0.5 * 1.0, abs=1e-6)
    assert any("answer_correctness: f1 raised" in r.message for r in caplog.records)


def test_answer_correctness_embedding_exception_path_logs_a_warning(caplog):
    class _RaisingEmbedder:
        def embed(self, texts):
            raise RuntimeError("embedder unavailable")

    llm_call = lambda prompt, max_tokens=256: '{"correct": 1, "incorrect": 0, "missing": 0}'

    with caplog.at_level(logging.WARNING, logger="open_telco_rag.evaluation.metrics.ground_truth"):
        score = answer_correctness("answer", "reference", _RaisingEmbedder(), llm_call=llm_call)

    # f1=1.0, sem falls back to 0.0 -> score == 0.5*1.0 + 0.5*0
    assert score == pytest.approx(0.5, abs=1e-6)
    assert any("answer_correctness: embedding raised" in r.message for r in caplog.records)
