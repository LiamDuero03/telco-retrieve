"""Tests for OpenRouterEmbedder's query_prefix/document_prefix handling and
real-usage accounting (configs/embedding_models.json's qwen3-embedding-8b
entry, added alongside these).

Mirrors tests/test_embedder_prefixes.py's approach for the local
(SentenceTransformerEmbedder) side: no real HTTP call is made -- `.session`
is swapped for a fake recording what was posted and returning a canned
response, the same way that file bypasses SentenceTransformer._load().
"""
from __future__ import annotations

import json

from open_telco_rag.embedders.openrouter_embedder import OpenRouterEmbedder


class _FakeResponse:
    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:
    """Records every post() call; returns one all-zero vector per input text."""

    def __init__(self, dimensions: int, usage: dict | None = None):
        self.dimensions = dimensions
        self.usage = usage
        self.last_payload: dict = {}
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, **kwargs):
        self.last_payload = json
        self.calls.append(json)
        data = [
            {"object": "embedding", "index": i, "embedding": [0.0] * self.dimensions}
            for i in range(len(json["input"]))
        ]
        payload = {"object": "list", "data": data, "model": json["model"]}
        if self.usage is not None:
            payload["usage"] = self.usage
        return _FakeResponse(payload)


def _wire_fake_session(embedder: OpenRouterEmbedder, dimensions: int = 8, usage=None) -> _FakeSession:
    fake = _FakeSession(dimensions, usage=usage)
    embedder.session = fake
    return fake


def test_qwen3_embedding_8b_applies_instruction_style_query_prefix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    embedder = OpenRouterEmbedder(model_key="qwen3-embedding-8b")
    fake = _wire_fake_session(embedder, dimensions=4096)

    embedder.embed(["what does the AMF do"], is_query=True)
    assert fake.last_payload["input"][0].startswith("Instruct:")
    assert fake.last_payload["input"][0].endswith("Query: what does the AMF do")

    embedder.embed(["a document chunk"], is_query=False)
    assert fake.last_payload["input"] == ["a document chunk"]  # no document prefix needed


def test_or_bge_m3_unaffected_by_is_query_no_prefix_configured(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    embedder = OpenRouterEmbedder(model_key="or-bge-m3")
    fake = _wire_fake_session(embedder)

    embedder.embed(["some text"], is_query=True)
    as_query = list(fake.last_payload["input"])
    embedder.embed(["some text"], is_query=False)
    as_document = list(fake.last_payload["input"])

    assert as_query == as_document == ["some text"]


def test_default_is_query_false_matches_ingest_document_path(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    embedder = OpenRouterEmbedder(model_key="qwen3-embedding-8b")
    fake = _wire_fake_session(embedder, dimensions=4096)

    embedder.embed(["a document chunk"])
    assert fake.last_payload["input"] == ["a document chunk"]


def test_usage_accumulates_real_tokens_and_cost_from_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    embedder = OpenRouterEmbedder(model_key="qwen3-embedding-8b", batch_size=1, max_workers=2)
    _wire_fake_session(
        embedder, dimensions=4096,
        usage={"prompt_tokens": 12, "total_tokens": 12, "cost": 1.2e-07},
    )

    assert embedder.total_prompt_tokens == 0
    assert embedder.total_cost_usd == 0.0

    embedder.embed(["chunk one", "chunk two", "chunk three"])

    # batch_size=1 forces 3 separate batched requests, each contributing usage.
    assert embedder.total_prompt_tokens == 36
    assert abs(embedder.total_cost_usd - 3.6e-07) < 1e-12


def test_max_workers_param_is_honored():
    import os
    os.environ.setdefault("OPENROUTER_API_KEY", "fake-key-for-test")
    embedder = OpenRouterEmbedder(model_key="qwen3-embedding-8b", max_workers=5)
    assert embedder.max_workers == 5


def test_new_registry_entry_present_with_expected_id_and_dimensions():
    from pathlib import Path
    models_file = Path(__file__).resolve().parents[1] / "configs" / "embedding_models.json"
    available = json.loads(models_file.read_text(encoding="utf-8"))

    assert available["qwen3-embedding-8b"]["model_name"] == "qwen/qwen3-embedding-8b"
    assert available["qwen3-embedding-8b"]["dimensions"] == 4096
    assert available["qwen3-embedding-8b"]["query_prefix"].startswith("Instruct:")
    assert "document_prefix" not in available["qwen3-embedding-8b"]


def test_new_entry_constructs_without_any_network_call(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    embedder = OpenRouterEmbedder(model_key="qwen3-embedding-8b")
    assert embedder.dimensions == 4096
    assert embedder.model_key == "qwen3-embedding-8b"


# ── backward compatibility for existing OpenRouter keys and defaults ──────────

def test_existing_openrouter_keys_send_text_unchanged(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    for key in ("or-3-small", "or-bge-m3"):
        embedder = OpenRouterEmbedder(model_key=key)
        fake = _wire_fake_session(embedder, dimensions=embedder.dimensions)
        embedder.embed(["some text"], is_query=True)
        assert fake.last_payload["input"] == ["some text"], key
        embedder.embed(["some text"])
        assert fake.last_payload["input"] == ["some text"], key


def test_qwen3_8b_document_side_has_no_prefix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    embedder = OpenRouterEmbedder(model_key="qwen3-embedding-8b")
    fake = _wire_fake_session(embedder, dimensions=4096)
    embedder.embed(["a document chunk"])
    assert fake.last_payload["input"] == ["a document chunk"]


def test_default_concurrency_is_the_previous_hardcoded_value(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    assert OpenRouterEmbedder(model_key="or-3-small").max_workers == 10


def test_cli_passes_max_workers_only_when_asked(monkeypatch):
    import cli.main as cli_main
    import open_telco_rag.embedders.openrouter_embedder as mod

    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.clear()
            seen.update(kw)

    monkeypatch.setattr(mod, "OpenRouterEmbedder", Fake)
    cli_main._get_embedder("or-3-small", backend="openrouter")
    assert seen == {"model_key": "or-3-small"}
    cli_main._get_embedder("or-3-small", backend="openrouter", max_workers=4)
    assert seen == {"model_key": "or-3-small", "max_workers": 4}
