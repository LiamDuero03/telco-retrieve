"""Tests for evaluate_question()'s token/cost tracking — the fix for a
real design bug: this used to ALWAYS compute prompt/completion tokens via
a local tiktoken cl100k_base approximation and unconditionally zero
cached_tokens/reasoning_tokens/cost, even though local.py/openrouter.py/
claude.py's generators all return a REAL "usage" (and openrouter.py/
local.py a real "cost") in their response dict. These tests confirm:

  1. The real gen["usage"] values are used (token_source="api") when the
     generator returns them, instead of always estimating.
  2. The tiktoken estimate is still used as a fallback (token_source=
     "estimated") when a generator genuinely doesn't return usage.
  3. Judge/metric-call token usage (faithfulness/answer_relevancy/
     context_recall/answer_correctness) is tracked separately as
     judge_prompt_tokens/judge_completion_tokens, via llm_call's own
     cumulative usage attributes (see _claude.py).
  4. cost is no longer unconditionally zeroed — a real value (including a
     genuine 0.0 for a self-hosted local generator) survives through to
     QuestionResult.cost, and None ("not computed") is distinguishable
     from it.

search()/generate() are patched at their source module
(open_telco_rag.core.pipeline) since evaluate_question() imports them
locally inside the function body — patching the module attribute affects
that local import at call time, same pattern as
test_evaluate_question_mc_tf.py.
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
    """A tracked fake llm_call matching _claude.py's own convention
    (total_prompt_tokens/total_completion_tokens as plain attributes on
    the callable, accumulated on every call) — decompose_claims/
    verify_claims/reverse_questions all expect a JSON array back; "[]"
    short-circuits faithfulness()/answer_relevancy() to 0.0 without
    needing a real LLM, same trick test_evaluate_question_mc_tf.py uses.
    """
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
        # No "usage" key at all -- e.g. ClaudeGenerator/OpenRouterGenerator
        # before this fix, or any custom/baseline generator.
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json_with_usage(0, 0),
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.error == ""
    assert result.token_source == "estimated"
    # tiktoken estimate over non-empty prompt/answer text must be > 0 --
    # not the bug's old behavior in a different guise (a silent 0).
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens
    # No usage/cost data was available at all -- cost stays "not computed".
    assert result.cost is None


def test_partial_usage_missing_completion_tokens_falls_back_to_estimate(monkeypatch):
    """A generator that returns a malformed/partial usage block (e.g.
    prompt_tokens present but completion_tokens missing) must not be
    treated as "real" -- either both are present and real, or the local
    estimate is used for both, never a mix of one real + one estimated
    number that would misreport total_tokens."""
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
    # Each of the 4 OE judge calls (faithfulness x2, answer_relevancy,
    # context_recall, answer_correctness) adds prompt_per_call/
    # completion_per_call to the tracked callable's running totals.
    judge_llm_call = _judge_call_returning_empty_json_with_usage(prompt_per_call=10, completion_per_call=2)

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=judge_llm_call,
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.error == ""
    # At least one judge call was made (faithfulness alone makes 2) --
    # judge tokens must be > 0 and NOT bleed into the generation fields.
    assert result.judge_prompt_tokens > 0
    assert result.judge_completion_tokens > 0
    assert result.judge_prompt_tokens % 10 == 0
    assert result.judge_completion_tokens % 2 == 0
    # Generation-side tokens are the real usage values, untouched by the
    # judge-call bookkeeping.
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
    """A plain callable (e.g. a bare lambda, matching Callable[[str, int],
    str] exactly but with no total_prompt_tokens/total_completion_tokens
    attributes) must degrade to a (0, 0) delta, never an AttributeError."""
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
    """The exact bug this fix targets: cost used to be hardcoded to 0.0
    regardless of what the generator returned. Confirm a real (already-
    zero) cost value from a self-hosted local generator passes through
    unchanged, distinguishable from "not computed" by staying a real
    float rather than becoming None."""
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
