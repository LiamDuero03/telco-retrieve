"""Tests for _make_local_call()/_make_openrouter_call()'s response parsing.

Regression: reasoning models (e.g. gpt-oss-120b) can return content=null when
max_tokens is exhausted by hidden reasoning; old code crashed on .strip().
"""
from __future__ import annotations

import requests

from open_telco_rag.evaluation._claude import _make_local_call, _make_openrouter_call


class _FakeResponse:
    status_code = 200  # read by the OpenRouter retry helper

    def __init__(self, content):
        self.headers = {}
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_local_call_returns_empty_string_not_a_crash_when_content_is_null(monkeypatch):
    monkeypatch.setattr(requests.Session, "post", lambda self, *a, **kw: _FakeResponse(None))

    call = _make_local_call("openai/gpt-oss-120b", base_url="http://localhost:8000/v1")
    result = call("Reply with just the word: ok", 5)

    assert result == ""


def test_local_call_still_strips_and_returns_real_content(monkeypatch):
    monkeypatch.setattr(requests.Session, "post", lambda self, *a, **kw: _FakeResponse("  ok  "))

    call = _make_local_call("openai/gpt-oss-120b", base_url="http://localhost:8000/v1")
    result = call("Reply with just the word: ok", 64)

    assert result == "ok"


def test_openrouter_call_returns_empty_string_not_a_crash_when_content_is_null(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(requests.Session, "post", lambda self, *a, **kw: _FakeResponse(None))

    call = _make_openrouter_call("some/reasoning-model")
    result = call("Reply with just the word: ok", 5)

    assert result == ""
