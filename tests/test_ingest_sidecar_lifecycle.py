"""The ingest settings sidecar must not disturb listing, and delete cleans it up."""
from __future__ import annotations

import argparse
import gc

import chromadb

import cli.main as cli_main
from open_telco_rag.core.pipeline import ingest
from open_telco_rag.evaluation.provenance import index_settings_file
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore
from tests.test_retrieval_methods import DOCS, BagOfWordsEmbedder


def _ingest(tmp_path, collection):
    data = tmp_path / "data"
    if not data.exists():
        data.mkdir()
        for name, text in DOCS.items():
            (data / name).write_text(text, encoding="utf-8")
    store = ChromaStore(path=tmp_path / "db", collection_name=collection, metadata={"hnsw:space": "cosine"})
    ingest(source=TPPSource(), store=store, embedder=BagOfWordsEmbedder(), data_dir=str(data), resume=False,
           log_dir=str(tmp_path / "logs"), strategy="text_baseline", chunk_size=300, overlap=0, min_length=10)
    return store


def test_the_sidecar_does_not_disturb_listing_or_reopening(tmp_path, capsys):
    _ingest(tmp_path, "coll_a")
    assert index_settings_file(tmp_path / "db", "coll_a").exists()
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    assert [c.name for c in client.list_collections()] == ["coll_a"]
    cli_main._run_info(argparse.Namespace(db_path=str(tmp_path / "db"), collection=None))
    assert "coll_a" in capsys.readouterr().out
    assert ChromaStore(path=tmp_path / "db", collection_name="coll_a").count() > 0


def test_delete_collection_removes_its_sidecar_and_only_its_own(tmp_path):
    _ingest(tmp_path, "coll_a")
    _ingest(tmp_path, "coll_b")
    gc.collect()                                     # release the ingest's SQLite handle (Windows file lock)
    cli_main._run_delete_collection(argparse.Namespace(db_path=str(tmp_path / "db"), collection="coll_a"))
    assert not index_settings_file(tmp_path / "db", "coll_a").exists()
    assert index_settings_file(tmp_path / "db", "coll_b").exists()
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    assert [c.name for c in client.list_collections()] == ["coll_b"]


def test_deleting_a_collection_without_a_sidecar_still_works(tmp_path):
    store = _ingest(tmp_path, "coll_a")
    index_settings_file(store.path, "coll_a").unlink()
    gc.collect()
    cli_main._run_delete_collection(argparse.Namespace(db_path=str(tmp_path / "db"), collection="coll_a"))
    assert chromadb.PersistentClient(path=str(tmp_path / "db")).list_collections() == []


def test_the_sidecar_is_removed_even_if_the_fts_step_fails(tmp_path, monkeypatch):
    import open_telco_rag.stores.fts as fts

    _ingest(tmp_path, "coll_a")

    def fail(self):
        raise PermissionError("locked")

    monkeypatch.setattr(fts.FTSStore, "delete", fail)
    cli_main._run_delete_collection(argparse.Namespace(db_path=str(tmp_path / "db"), collection="coll_a"))
    assert not index_settings_file(tmp_path / "db", "coll_a").exists()
