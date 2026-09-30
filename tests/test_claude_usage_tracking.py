"""Tests for _claude.py's judge-call usage tracking.

Fix: LLM call closures discarded the response's usage block, so judge/metric
calls were invisible to token/cost accounting. Usage now accumulates as
attributes (total_prompt_tokens/total_completion_tokens) to preserve the
Callable[[str, int], str] signature metrics functions expect. See
test_evaluate_efficiency_tracking.py for evaluate_question()'s use of this.
"""
from __future__ import annotations

import requests

from open_telco_rag.evaluation._claude import _make_local_call, _make_openrouter_call


class _FakeResponse:
    status_code = 200  # read by the OpenRouter retry helper
    headers: dict = {}

    def __init__(self, content, usage=None):
        self._content = content
        self._usage = usage or {}

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}], "usage": self._usage}


def test_local_call_accumulates_usage_across_multiple_calls(monkeypatch):
    monkeypatch.setattr(
        requests.Session, "post",
        lambda self, *a, **kw: _FakeResponse("ok", usage={"prompt_tokens": 50, "completion_tokens": 10}),
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
    """Server omitting "usage" (e.g. older vLLM) must not crash — just contribute nothing."""
    monkeypatch.setattr(requests.Session, "post", lambda self, *a, **kw: _FakeResponse("ok", usage=None))

    call = _make_local_call("openai/gpt-oss-120b", base_url="http://localhost:8000/v1")
    call("prompt", 64)

    assert call.total_prompt_tokens == 0
    assert call.total_completion_tokens == 0


def test_two_local_call_instances_track_usage_independently(monkeypatch):
    """Each factory call returns its own tracked callable; usage must not leak
    between instances (e.g. two judge configs in the same process)."""
    monkeypatch.setattr(
        requests.Session, "post",
        lambda self, *a, **kw: _FakeResponse("ok", usage={"prompt_tokens": 5, "completion_tokens": 1}),
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
        requests.Session, "post",
        lambda self, *a, **kw: _FakeResponse("ok", usage={"prompt_tokens": 30, "completion_tokens": 8}),
    )

    call = _make_openrouter_call("some/model")
    call("prompt", 64)

    assert call.total_prompt_tokens == 30
    assert call.total_completion_tokens == 8
