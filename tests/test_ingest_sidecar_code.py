"""The ingest settings sidecar also records which code version ran the ingest."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from open_telco_rag.core.pipeline import ingest
from open_telco_rag.evaluation import provenance
from open_telco_rag.evaluation.provenance import (
    collect_run_record,
    index_settings_file,
    read_index_settings,
)
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore
from tests.test_retrieval_methods import DOCS, BagOfWordsEmbedder


def _ingest(tmp_path, collection="coll_a"):
    data = tmp_path / "data"
    if not data.exists():
        data.mkdir()
        for name, text in DOCS.items():
            (data / name).write_text(text, encoding="utf-8")
    store = ChromaStore(path=tmp_path / "db", collection_name=collection, metadata={"hnsw:space": "cosine"})
    ingest(source=TPPSource(), store=store, embedder=BagOfWordsEmbedder(), data_dir=str(data), resume=False,
           log_dir=str(tmp_path / "logs"), strategy="text_baseline", chunk_size=300, overlap=0, min_length=10)
    return store


def test_the_sidecar_records_the_code_version(tmp_path):
    store = _ingest(tmp_path)
    info = read_index_settings(store.path, "coll_a")
    assert info["code"]["name"] == "open-telco-rag"
    assert {"git_commit", "git_dirty", "version"} <= set(info["code"])


def test_the_code_version_is_unknown_without_a_git_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_PACKAGE_DIR", tmp_path / "not-a-checkout")
    store = _ingest(tmp_path)
    code = read_index_settings(store.path, "coll_a")["code"]
    assert code["git_commit"] == "unknown" and code["git_dirty"] == "unknown"


def test_a_sidecar_written_before_the_field_existed_still_loads(tmp_path):
    store = _ingest(tmp_path)
    path = index_settings_file(store.path, "coll_a")
    data = json.loads(Path(path).read_text(encoding="utf8"))
    for entry in [data["latest"], *data["history"]]:
        entry.pop("code")
    Path(path).write_text(json.dumps(data), encoding="utf8")
    info = read_index_settings(store.path, "coll_a")
    assert "code" not in info and info["settings"]["chunking"]["strategy"] == "text_baseline"
    record = collect_run_record(params={}, embedder=BagOfWordsEmbedder(), reranker=None, store=store,
                                questions=[], questions_name="q", started_at=datetime.now(timezone.utc))
    assert "code" not in record["index"]["ingest_settings"]


def test_reingesting_updates_the_entry_and_keeps_one_history_item(tmp_path):
    _ingest(tmp_path)
    store = _ingest(tmp_path)
    data = json.loads(Path(index_settings_file(store.path, "coll_a")).read_text(encoding="utf8"))
    assert len(data["history"]) == 1 and data["latest"]["code"]["name"] == "open-telco-rag"
