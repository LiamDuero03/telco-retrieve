"""Tests for _make_local_call()/_make_openrouter_call()'s response parsing
— a real, confirmed-in-production bug: reasoning models (e.g. openai/
gpt-oss-120b) can spend their entire max_tokens budget on hidden reasoning
tokens and return a well-formed HTTP 200 response with "content": null when
nothing is left over for visible text. The old code did
`response.json()["choices"][0]["message"]["content"].strip()` directly,
which crashed with a bare, undiagnosable `'NoneType' object has no
attribute 'strip'` — hit by master_grid_eval.py's own pre-flight endpoint
check (using a too-small max_tokens=5) even though the endpoint itself was
completely healthy.
"""
from __future__ import annotations

import requests

from open_telco_rag.evaluation._claude import _make_local_call, _make_openrouter_call


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_local_call_returns_empty_string_not_a_crash_when_content_is_null(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(None))

    call = _make_local_call("openai/gpt-oss-120b", base_url="http://localhost:8000/v1")
    result = call("Reply with just the word: ok", 5)

    assert result == ""


def test_local_call_still_strips_and_returns_real_content(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse("  ok  "))

    call = _make_local_call("openai/gpt-oss-120b", base_url="http://localhost:8000/v1")
    result = call("Reply with just the word: ok", 64)

    assert result == "ok"


def test_openrouter_call_returns_empty_string_not_a_crash_when_content_is_null(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(None))

    call = _make_openrouter_call("some/reasoning-model")
    result = call("Reply with just the word: ok", 5)

    assert result == ""
