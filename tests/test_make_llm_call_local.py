"""Tests for make_llm_call()'s "local" backend, added so eval metrics can
run against a self-hosted GPU server (previously only claude/openrouter).
"""
from __future__ import annotations

from open_telco_rag.evaluation._claude import make_llm_call


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_local_llm_call_posts_to_the_given_base_url(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    captured = {}

    def _fake_post(self, url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse("local model response")

    monkeypatch.setattr("requests.Session.post", _fake_post)

    llm_call = make_llm_call("local", model="Qwen/Qwen3-32B", base_url="http://localhost:8001/v1")
    result = llm_call("some prompt", 256)

    assert result == "local model response"
    assert captured["url"] == "http://localhost:8001/v1/chat/completions"
    assert captured["json"]["model"] == "Qwen/Qwen3-32B"
    assert captured["json"]["max_tokens"] == 256
    assert "Authorization" not in captured["headers"]  # no api_key given


def test_local_llm_call_sends_authorization_header_when_api_key_given(monkeypatch):
    captured = {}

    def _fake_post(self, url, headers=None, json=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse("ok")

    monkeypatch.setattr("requests.Session.post", _fake_post)

    llm_call = make_llm_call(
        "local", model="local-model", base_url="http://localhost:8000/v1", api_key="secret-key",
    )
    llm_call("prompt")

    assert captured["headers"]["Authorization"] == "Bearer secret-key"


def test_local_llm_call_falls_back_to_env_vars(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "env-key")
    captured = {}

    def _fake_post(self, url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeResponse("ok")

    monkeypatch.setattr("requests.Session.post", _fake_post)

    llm_call = make_llm_call("local", model="local-model")
    llm_call("prompt")

    assert captured["url"] == "http://localhost:9000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer env-key"
