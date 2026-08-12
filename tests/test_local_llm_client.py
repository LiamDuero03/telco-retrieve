"""Unit tests for LocalLLMClient/_LocalCompletionAPI's Authorization header
behavior and cli/main.py's _get_llm_client() threading of api_key.

No real network calls anywhere here — requests.post is mocked throughout,
so nothing actually hits a local/vLLM server.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from open_telco_rag.generators.local_llm import LocalLLMClient


def _mock_response(content: str = "OK"):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


# ── LocalLLMClient / _LocalCompletionAPI header behavior ─────────────────────

def test_no_api_key_omits_authorization_header(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    assert client.api_key is None

    with patch("requests.post", return_value=_mock_response()) as mock_post:
        client.completion.create(model="m", messages=[{"role": "user", "content": "hi"}])

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {}


def test_explicit_api_key_sends_bearer_header(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    client = LocalLLMClient(base_url="http://localhost:8000/v1", api_key="secret-key")
    assert client.api_key == "secret-key"

    with patch("requests.post", return_value=_mock_response()) as mock_post:
        client.completion.create(model="m", messages=[{"role": "user", "content": "hi"}])

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer secret-key"}


def test_env_var_fallback_when_api_key_not_passed(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "env-key")
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    assert client.api_key == "env-key"

    with patch("requests.post", return_value=_mock_response()) as mock_post:
        client.completion.create(model="m", messages=[{"role": "user", "content": "hi"}])

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer env-key"}


def test_explicit_api_key_overrides_env_var(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "env-key")
    client = LocalLLMClient(base_url="http://localhost:8000/v1", api_key="explicit-key")
    assert client.api_key == "explicit-key"


# ── cli/main.py's _get_llm_client() threading ─────────────────────────────────

def test_get_llm_client_local_threads_api_key(monkeypatch):
    from cli.main import _get_llm_client

    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    client = _get_llm_client(backend="local", base_url="http://localhost:8000/v1", api_key="k")
    assert client.api_key == "k"


def test_get_llm_client_local_falls_back_to_env_var(monkeypatch):
    from cli.main import _get_llm_client

    monkeypatch.setenv("LOCAL_LLM_API_KEY", "env-key")
    client = _get_llm_client(backend="local", base_url="http://localhost:8000/v1")
    assert client.api_key == "env-key"
