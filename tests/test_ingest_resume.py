"""Ingest resume: the progress file belongs to one (database, collection, chunking,
enrichment) configuration; anything else starts fresh instead of skipping files."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import open_telco_rag.core.pipeline as pipeline
from open_telco_rag.core.chunking import ChunkingStrategy
from open_telco_rag.core.pipeline import ingest
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore
from tests.test_retrieval_methods import DOCS, BagOfWordsEmbedder

CHUNK = dict(strategy="text_baseline", chunk_size=300, overlap=0, min_length=10)


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


def _ingest(tmp_path, data, collection="coll_a", db="db", resume=True, **over):
    store = ChromaStore(path=tmp_path / db, collection_name=collection, metadata={"hnsw:space": "cosine"})
    kw = {**CHUNK, **over}
    stats = ingest(source=TPPSource(), store=store, embedder=BagOfWordsEmbedder(), data_dir=str(data),
                   resume=resume, log_dir=str(tmp_path / "logs"), **kw)
    return stats, store


def _progress_files(tmp_path):
    return sorted(p.name for p in (tmp_path / "logs").glob(".progress*") if not p.name.endswith(".bak"))


def test_rerunning_the_same_configuration_skips_finished_files(tmp_path, data):
    first, _ = _ingest(tmp_path, data)
    second, _ = _ingest(tmp_path, data)
    assert first.files_processed == len(DOCS) and second.files_processed == 0
    assert second.skipped_files == len(DOCS) and second.chunks_stored == 0


def test_a_new_collection_is_ingested_not_skipped(tmp_path, data):
    _ingest(tmp_path, data, collection="coll_a")
    stats, store = _ingest(tmp_path, data, collection="coll_b")
    assert stats.files_processed == len(DOCS) and stats.skipped_files == 0
    assert store.count() == stats.chunks_stored > 0
    assert len(_progress_files(tmp_path)) == 2            # one progress file per configuration


def test_a_new_database_is_ingested_not_skipped(tmp_path, data):
    _ingest(tmp_path, data, db="db1")
    stats, store = _ingest(tmp_path, data, db="db2")
    assert stats.skipped_files == 0 and store.count() > 0


@pytest.mark.parametrize("change", [dict(chunk_size=500), dict(strategy="hierarchical_markdown"),
                                    dict(min_length=20), dict(enrichment="metadata_tagging"),
                                    dict(target_tokens=40)])
def test_a_different_chunking_configuration_is_not_resumed(tmp_path, data, change):
    _ingest(tmp_path, data, collection="coll_a")
    kw = {**CHUNK, **change}
    if "target_tokens" in change:
        kw.pop("chunk_size"), kw.pop("overlap")
    store = ChromaStore(path=tmp_path / "db", collection_name="coll_b", metadata={"hnsw:space": "cosine"})
    stats = ingest(source=TPPSource(), store=store, embedder=BagOfWordsEmbedder(), data_dir=str(data),
                   log_dir=str(tmp_path / "logs"), **kw)
    assert stats.skipped_files == 0 and stats.files_processed == len(DOCS)


def test_same_collection_new_config_warns_about_mixing(tmp_path, data, capsys):
    _ingest(tmp_path, data)
    capsys.readouterr()
    stats, _ = _ingest(tmp_path, data, chunk_size=500)
    out = capsys.readouterr().out
    assert stats.files_processed == len(DOCS)
    assert "No resume record matches these settings" in out and "already holds" in out
    assert "--no-resume" in out and "--collection" in out
    assert len(_progress_files(tmp_path)) == 2            # the first configuration's record is left alone


def test_a_fresh_collection_gives_no_warning(tmp_path, data, capsys):
    _ingest(tmp_path, data)
    assert "[WARN]" not in capsys.readouterr().out


def test_a_foreign_fingerprint_inside_the_file_warns_and_keeps_a_bak(tmp_path, data, capsys):
    _ingest(tmp_path, data)
    (progress,) = list((tmp_path / "logs").glob(".progress*.json"))
    doc = json.loads(progress.read_text(encoding="utf8"))
    doc["fingerprint"] = "0" * 64
    progress.write_text(json.dumps(doc), encoding="utf8")
    capsys.readouterr()
    stats, _ = _ingest(tmp_path, data)
    out = capsys.readouterr().out
    assert stats.files_processed == len(DOCS)
    assert "different collection, database or chunking" in out and "Starting fresh" in out
    assert progress.with_name(progress.name + ".bak").exists()


def test_a_corrupt_progress_file_warns_starts_fresh_and_is_kept_as_bak(tmp_path, data, capsys):
    _ingest(tmp_path, data)
    (progress,) = [p for p in (tmp_path / "logs").glob(".progress*.json")]
    progress.write_text("{not json", encoding="utf8")
    capsys.readouterr()
    stats, _ = _ingest(tmp_path, data)
    out = capsys.readouterr().out
    assert stats.files_processed == len(DOCS) and "could not be read" in out
    assert progress.with_name(progress.name + ".bak").read_text(encoding="utf8") == "{not json"
    assert json.loads(progress.read_text(encoding="utf8"))["completed"]     # a fresh, valid file


def test_an_old_format_file_restarts_from_scratch_with_a_clear_message(tmp_path, data, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    legacy = logs / ".progress_bow.json"
    legacy.write_text(json.dumps(sorted(DOCS)), encoding="utf8")            # the old list-of-files format
    stats, _ = _ingest(tmp_path, data)
    out = capsys.readouterr().out
    assert stats.files_processed == len(DOCS) and stats.skipped_files == 0
    assert "old-format resume file" in out and "restarts from scratch" in out and ".bak" in out
    assert not legacy.exists() and (logs / ".progress_bow.json.bak").exists()


def test_no_resume_still_ignores_the_progress_file_and_writes_none(tmp_path, data):
    _ingest(tmp_path, data)
    stats, _ = _ingest(tmp_path, data, resume=False)
    assert stats.files_processed == len(DOCS)


# ── fingerprint properties ────────────────────────────────────────────────────

class _Store:
    def __init__(self, path, collection):
        self.path, self.collection_name = path, collection


def _fp(path="db", collection="c", **chunk_kwargs):
    settings = pipeline._ingest_settings(_Store(path, collection), BagOfWordsEmbedder(), 10, chunk_kwargs)
    return pipeline._fingerprint(settings), settings


def test_equivalent_spellings_of_the_same_database_path_share_a_fingerprint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    absolute = str(tmp_path / "db")
    assert _fp("db")[0] == _fp(absolute)[0] == _fp("./db")[0] == _fp(absolute.replace("\\", "/"))[0]
    assert _fp("db")[0] == _fp("sub/../db")[0]
    if os.name == "nt":
        assert _fp(absolute.upper())[0] == _fp(absolute)[0]
    assert _fp("db")[0] != _fp("db2")[0] and _fp(collection="c")[0] != _fp(collection="d")[0]


def test_kwarg_order_enums_and_batching_do_not_change_the_fingerprint():
    a = _fp(strategy="text_baseline", chunk_size=300, overlap=0, batch_size=8)[0]
    b = _fp(overlap=0, chunk_size=300, strategy=ChunkingStrategy.TEXT_BASELINE, batch_size=64,
            write_batch_size=512)[0]
    assert a == b
    assert _fp(strategy="text_baseline", chunk_size=301)[0] != a


def test_settings_that_change_the_stored_chunks_change_the_fingerprint():
    base = _fp(strategy="text_baseline", chunk_size=300)[0]
    for change in (dict(target_tokens=50), dict(max_tokens=60), dict(enrichment="metadata_tagging"),
                   dict(enrichment_model="m"), dict(overlap_tokens=5), dict(chunk_size=300, overlap=10)):
        assert _fp(**{"strategy": "text_baseline", "chunk_size": 300, **change})[0] != base, change


def test_secrets_and_objects_are_never_recorded(tmp_path, data):
    secret = "sk-or-SECRET-VALUE"
    fp, settings = _fp(strategy="text_baseline", api_key=secret, llm_api_key=secret, enrichment_llm_client=object(),
                       llm_client=object(), model_name="chunker-model")
    blob = json.dumps(settings)
    assert secret not in blob and "api_key" not in blob and "client" not in blob
    assert settings["chunking"]["model_name"] == "chunker-model"
    assert fp == _fp(strategy="text_baseline", model_name="chunker-model")[0]

    _ingest(tmp_path, data, api_key=secret)
    (progress,) = list((tmp_path / "logs").glob(".progress*.json"))
    assert secret not in progress.read_text(encoding="utf8")
