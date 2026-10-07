"""Fake embedder and collection builders for tests that need real Chroma + FTS stores but no
model, no GPU and no network."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pytest

from open_telco_rag.evaluation.provenance import index_settings_file
from open_telco_rag.stores.chroma import ChromaStore
from open_telco_rag.stores.fts import FTSStore


class FakeEmbedder:
    """hashed bag of words; records every embed call as (texts, is_query)"""

    def __init__(self, model_key: str = "fake64", dimensions: int = 64, query_prefix: str = ""):
        self.model_key = model_key
        self.dimensions = dimensions
        self.query_prefix = query_prefix
        self.calls: list[tuple[list[str], bool]] = []

    def embed(self, texts, is_query=False, apply_prefix=True):
        self.calls.append((list(texts), is_query))
        out = []
        for text in texts:
            if is_query:
                text = self.query_prefix + text
            vec = [0.0] * self.dimensions
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


def make_collection(db_path, name: str, rows: list[dict], embedder: FakeEmbedder,
                    space: str = "l2", sidecar_model: str | None = "default") -> ChromaStore:
    """write rows ({id, text, source, source_type, ...extra metadata}) to a Chroma collection and
    its BM25 index; sidecar_model 'default' records the embedder's key, None writes no sidecar"""
    store = ChromaStore(path=db_path, collection_name=name, metadata={"hnsw:space": space})
    fts = FTSStore(path=db_path, collection_name=name)
    if rows:
        texts = [r["text"] for r in rows]
        metas = [{"source": r.get("source", "doc.md"), "source_type": r.get("source_type", "3gpp"),
                  "heading": r.get("heading", ""), **r.get("meta", {})} for r in rows]
        store.add([r["id"] for r in rows], embedder.embed(texts), texts, metas)
        fts.add([r["id"] for r in rows], texts)
    if sidecar_model is not None:
        model = embedder.model_key if sidecar_model == "default" else sidecar_model
        entry = {"fingerprint": "f" * 64, "settings": {"collection": name, "model_key": model},
                 "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        index_settings_file(db_path, name).write_text(
            json.dumps({"version": 1, "latest": entry, "history": [entry]}), encoding="utf8")
    return store


@pytest.fixture(autouse=True)
def in_memory_chroma(monkeypatch):
    """Chroma 1.5 writes its index to disk in the background. A client that reads collections written
    moments ago in the same process can fail for good with 'Nothing found on disk' (about 1 run in 5
    for these tests); a real search runs in a process of its own, long after the ingest, and the
    existing tests only ever read through the client that wrote. These tests are about our logic, so
    they use one in-memory Chroma instead of the disk (the BM25 index and sidecars stay on disk)."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.EphemeralClient(settings=Settings(allow_reset=True, anonymized_telemetry=False))
    client.reset()
    monkeypatch.setattr(chromadb, "PersistentClient", lambda *args, **kwargs: client)
    yield client
    client.reset()


def registry(path: Path, entries: dict) -> Path:
    path.write_text(json.dumps(entries), encoding="utf8")
    return path
