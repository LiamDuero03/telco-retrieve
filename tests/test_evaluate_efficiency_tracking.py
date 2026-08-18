"""Tests for evaluate_question()'s token/cost tracking.

Fix: this used to always estimate tokens via tiktoken and zero out cost, even
though generators return real "usage"/"cost" in their response dict. Confirms:
(1) real usage is preferred (token_source="api"); (2) tiktoken estimate is the
fallback when usage is absent; (3) judge-call tokens are tracked separately as
judge_prompt_tokens/judge_completion_tokens; (4) cost (including a genuine 0.0)
survives instead of being zeroed.

search()/generate() are patched on pipeline_module because evaluate_question()
imports them locally — patching the module attribute affects that import at
call time (same pattern as test_evaluate_question_mc_tf.py).
"""
from __future__ import annotations

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.evaluation.evaluate import evaluate_question
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.types import SearchResult


def _fake_chunks():
    return [SearchResult(text="The AMF handles registration.", distance=0.2,
                          metadata={"source": "23501-j00.md"}, doc_id="c1")]


def _open_question() -> EvalQuestion:
    return EvalQuestion(
        question="What does the AMF handle?", keywords=[], corpus="3gpp",
        answer_type="open", reference_answer="Registration and mobility management.",
    )


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _judge_call_returning_empty_json_with_usage(prompt_per_call: int, completion_per_call: int):
    """Fake llm_call matching _claude.py's usage-tracking convention; "[]"
    short-circuits judge functions to 0.0 without a real LLM (same trick as
    test_evaluate_question_mc_tf.py)."""
    def _call(prompt, max_tokens=1024):
        _call.total_prompt_tokens += prompt_per_call
        _call.total_completion_tokens += completion_per_call
        return "[]"

    _call.total_prompt_tokens = 0
    _call.total_completion_tokens = 0
    return _call


# ── 1. Real usage preferred over the local estimate ─────────────────────────

def test_real_generator_usage_is_used_instead_of_estimating(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
        "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
        "cost": 0.0021,
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json_with_usage(0, 0),
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.error == ""
    assert result.token_source == "api"
    # Real numbers, NOT a tiktoken approximation of the prompt/answer text.
    assert result.prompt_tokens == 123
    assert result.completion_tokens == 45
    assert result.total_tokens == 168
    assert result.cost == 0.0021


def test_falls_back_to_local_estimate_when_generator_returns_no_usage(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
        # No "usage" key — e.g. generators before this fix, or any custom/baseline generator.
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json_with_usage(0, 0),
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.error == ""
    assert result.token_source == "estimated"
    # Estimate must be > 0 — a silent 0 would be the old bug in disguise.
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens
    # No usage/cost data available — cost stays "not computed".
    assert result.cost is None


def test_partial_usage_missing_completion_tokens_falls_back_to_estimate(monkeypatch):
    """Partial usage (e.g. missing completion_tokens) must not be treated as real —
    falls back to estimating both, never a real+estimated mix."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
        "usage": {"prompt_tokens": 123},  # completion_tokens missing
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json_with_usage(0, 0),
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.token_source == "estimated"
    assert result.prompt_tokens != 123  # not the partial "real" value


# ── 2. Judge-call token tracking ─────────────────────────────────────────────

def test_judge_calls_populate_judge_token_fields_separately_from_generation(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    })
    # Each of the 4 OE judge calls adds prompt/completion_per_call to the running totals.
    judge_llm_call = _judge_call_returning_empty_json_with_usage(prompt_per_call=10, completion_per_call=2)

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=judge_llm_call,
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.error == ""
    # Judge tokens must be > 0 and not bleed into the generation fields.
    assert result.judge_prompt_tokens > 0
    assert result.judge_completion_tokens > 0
    assert result.judge_prompt_tokens % 10 == 0
    assert result.judge_completion_tokens % 2 == 0
    # Generation-side tokens are untouched by judge-call bookkeeping.
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20


def test_judge_tokens_are_zero_for_mc_tf_questions_which_make_no_judge_calls(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "B", "sources": [], "chunks_used": 1,
    })
    judge_llm_call = _judge_call_returning_empty_json_with_usage(prompt_per_call=10, completion_per_call=2)

    q = EvalQuestion(
        question="Which interface?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B",
    )
    result = evaluate_question(
        q, store=object(), embedder=object(), llm_call=judge_llm_call,
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.judge_prompt_tokens == 0
    assert result.judge_completion_tokens == 0


def test_judge_tokens_are_zero_when_llm_call_does_not_track_usage(monkeypatch):
    """Plain callable with no usage attributes must degrade to (0, 0), never AttributeError."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=lambda prompt, max_tokens=1024: "[]",
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.error == ""
    assert result.judge_prompt_tokens == 0
    assert result.judge_completion_tokens == 0


# ── 3. cost is not silently zeroed ───────────────────────────────────────────

def test_genuine_zero_cost_from_local_generator_is_preserved_not_overwritten(monkeypatch):
    """The exact bug: cost was hardcoded to 0.0. A genuine 0.0 cost must pass
    through as a float, not become None."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "cost": 0.0,  # genuinely free -- self-hosted, no per-token billing
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json_with_usage(0, 0),
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.cost == 0.0
    assert result.cost is not None


def test_nonzero_cost_from_generator_is_preserved(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "cost": 0.045,
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json_with_usage(0, 0),
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.cost == 0.045
