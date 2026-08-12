"""Tests for _claude.py's judge-call usage tracking — the fix for a real
gap: claude_call/_make_local_call/_make_openrouter_call/make_llm_call's
custom-Claude closure all discarded the API response's "usage" block
entirely, returning only the extracted text. That meant every judge/
metric call (faithfulness, answer_relevancy, context_recall,
answer_correctness) was completely invisible to token/cost accounting.

Each callable now accumulates total_prompt_tokens/total_completion_tokens
as plain attributes on itself (see each function's own comment for why:
preserving the exact Callable[[str, int], str] signature every metrics
function already calls, rather than threading a richer return type
through faithfulness()/answer_relevancy()/etc). These tests confirm the
accumulation itself, independent of evaluate_question()'s snapshot/delta
usage of it (see test_evaluate_efficiency_tracking.py for that).
"""
from __future__ import annotations

import requests

from open_telco_rag.evaluation._claude import _make_local_call, _make_openrouter_call


class _FakeResponse:
    def __init__(self, content, usage=None):
        self._content = content
        self._usage = usage or {}

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}], "usage": self._usage}


def test_local_call_accumulates_usage_across_multiple_calls(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **kw: _FakeResponse("ok", usage={"prompt_tokens": 50, "completion_tokens": 10}),
    )

    call = _make_local_call("openai/gpt-oss-120b", base_url="http://localhost:8000/v1")

    assert call.total_prompt_tokens == 0
    assert call.total_completion_tokens == 0

    call("first prompt", 64)
    assert call.total_prompt_tokens == 50
    assert call.total_completion_tokens == 10

    call("second prompt", 64)
    assert call.total_prompt_tokens == 100
    assert call.total_completion_tokens == 20


def test_local_call_missing_usage_block_does_not_crash_and_stays_zero(monkeypatch):
    """A server that doesn't return "usage" at all (e.g. an older vLLM
    build) must not crash the caller -- just contribute nothing."""
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse("ok", usage=None))

    call = _make_local_call("openai/gpt-oss-120b", base_url="http://localhost:8000/v1")
    call("prompt", 64)

    assert call.total_prompt_tokens == 0
    assert call.total_completion_tokens == 0


def test_two_local_call_instances_track_usage_independently(monkeypatch):
    """Each make_llm_call()-style factory call returns its OWN tracked
    callable -- usage on one must never leak into another (e.g. two
    different judge configs in the same process, or two questions'
    baseline-vs-judge generators)."""
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **kw: _FakeResponse("ok", usage={"prompt_tokens": 5, "completion_tokens": 1}),
    )

    call_a = _make_local_call("model-a", base_url="http://localhost:8000/v1")
    call_b = _make_local_call("model-b", base_url="http://localhost:8001/v1")

    call_a("prompt", 64)
    call_a("prompt", 64)

    assert call_a.total_prompt_tokens == 10
    assert call_b.total_prompt_tokens == 0


def test_openrouter_call_accumulates_usage(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **kw: _FakeResponse("ok", usage={"prompt_tokens": 30, "completion_tokens": 8}),
    )

    call = _make_openrouter_call("some/model")
    call("prompt", 64)

    assert call.total_prompt_tokens == 30
    assert call.total_completion_tokens == 8
