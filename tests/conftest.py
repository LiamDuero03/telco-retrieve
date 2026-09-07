"""Shared pytest fixtures for telco-retrieve's test suite.

Fixtures here are deterministic and offline: FakeLLMClient never makes a
network call, and the fixture Chroma+FTS index pair is a small hand-built
store in a temp dir — no real embedding model, no GPU, no API key required.
The one exception is the Phase 4 ingest() integration test, which loads the
real (small, CPU-only) all-MiniLM-L6-v2 model via sentence-transformers.
"""
from __future__ import annotations

import hashlib

import pytest


# ── Fake LLM client ───────────────────────────────────────────────────────────
# Mimics LocalLLMClient/OpenRouterLLMClient's .completion.create(...) response shape.

class _FakeMessage:
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    __slots__ = ("message",)

    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    __slots__ = ("choices",)

    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletionAPI:
    def __init__(self, outer: "FakeLLMClient"):
        self._outer = outer

    def create(self, model: str, messages: list[dict], **kwargs) -> _FakeCompletionResponse:
        prompt = messages[-1]["content"] if messages else ""
        self._outer.calls.append(prompt)
        for needle, reply in self._outer.responses.items():
            if needle in prompt:
                return _FakeCompletionResponse(reply)
        return _FakeCompletionResponse(self._outer.default)


class FakeLLMClient:
    """Drop-in for LocalLLMClient/OpenRouterLLMClient. Responses are matched
    by substring against the last message's content — first match in
    `responses` wins, in insertion order — falling back to `default`
    ("OK" unless overridden). Every prompt seen is recorded in `.calls`.
    """

    def __init__(self, responses: dict[str, str] | None = None, default: str = "OK"):
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []
        self.completion = _FakeCompletionAPI(self)


@pytest.fixture
def fake_llm_client() -> FakeLLMClient:
    return FakeLLMClient()


# ── Fixture Chroma + FTS store pair ───────────────────────────────────────────

FIXTURE_CHUNKS = [
    {
        "id": "doc1::0",
        "text": "The Near-RT RIC exposes the E2 interface to O-RAN CUs and DUs for near-real-time control.",
        "metadata": {"source": "doc1.md", "heading": "E2 Interface", "source_type": "oran"},
    },
    {
        "id": "doc1::1",
        "text": "O-RAN WG3 defines the E2 Application Protocol (E2AP) used between the Near-RT RIC and E2 nodes.",
        "metadata": {"source": "doc1.md", "heading": "E2AP", "source_type": "oran"},
    },
    {
        "id": "doc2::0",
        "text": "3GPP TS 23.501 defines the 5G System architecture including the AMF, SMF and UPF network functions.",
        "metadata": {"source": "doc2.md", "heading": "5GS Architecture", "source_type": "3gpp"},
    },
    {
        "id": "doc2::1",
        "text": "The AMF (Access and Mobility Management Function) handles registration, connection and mobility management.",
        "metadata": {"source": "doc2.md", "heading": "AMF", "source_type": "3gpp"},
    },
    {
        "id": "doc3::0",
        "text": "ETSI ZSM defines closed-loop automation for zero-touch network and service management.",
        "metadata": {"source": "doc3.md", "heading": "ZSM Closed Loop", "source_type": "etsi"},
    },
    {
        "id": "doc3::1",
        "text": "Closed-loop automation requires a monitor-analyze-decide-execute cycle driven by policy.",
        "metadata": {"source": "doc3.md", "heading": "MAPE-K", "source_type": "etsi"},
    },
    {
        "id": "doc4::0",
        "text": "TM Forum Open APIs define standardized REST interfaces for telecom OSS/BSS integration.",
        "metadata": {"source": "doc4.md", "heading": "TMF Open APIs", "source_type": "tmforum"},
    },
    {
        "id": "doc4::1",
        "text": "TMF632 is the Party Management API, covering individuals and organizations as parties.",
        "metadata": {"source": "doc4.md", "heading": "TMF632", "source_type": "tmforum"},
    },
]


def _fake_embedding(text: str, dims: int = 8) -> list[float]:
    """Deterministic pseudo-embedding for exercising store plumbing without a real model."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [(h[i] / 255.0) * 2 - 1 for i in range(dims)]


@pytest.fixture
def fixture_store(tmp_path):
    """ChromaStore + FTSStore pair over 8 fixture chunks, mirroring real ingest() output."""
    from open_telco_rag.stores.chroma import ChromaStore
    from open_telco_rag.stores.fts import FTSStore

    db_path = tmp_path / "chroma_db"
    chroma_store = ChromaStore(
        path=db_path, collection_name="test_oran_minilm", metadata={"hnsw:space": "cosine"}
    )
    fts_store = FTSStore(path=db_path, collection_name="test_oran_minilm")

    ids = [c["id"] for c in FIXTURE_CHUNKS]
    docs = [c["text"] for c in FIXTURE_CHUNKS]
    chroma_store.add(
        ids=ids,
        embeddings=[_fake_embedding(t) for t in docs],
        documents=docs,
        metadatas=[c["metadata"] for c in FIXTURE_CHUNKS],
    )
    fts_store.add(ids=ids, documents=docs)
    return chroma_store, fts_store
