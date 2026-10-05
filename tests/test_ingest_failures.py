"""What ingest() and the ingest command report when something fails: counts that match what was stored, a failed flag,
an opt-in exit code, and a check of the embedder's output before anything is written. Fake embedders only."""
from __future__ import annotations

import argparse
import sqlite3
from types import SimpleNamespace

import pytest

import cli.main as cli_main
import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.stores.chroma import ChromaStore
from tests.test_ingest_checkpoint import (  # noqa: F401  (fixtures)
    N_FILES,
    SlowEmbedder,
    _per_file,
    _run,
    _store,
    data,
    expected,
)


def _fts(tmp_path) -> int:
    conn = sqlite3.connect(str(tmp_path / "db" / "coll_fts.db"))
    try:
        return conn.execute("SELECT COUNT(*) FROM bm25_index").fetchone()[0]
    finally:
        conn.close()


class FlakyStore(ChromaStore):
    """the Chroma write of one flush fails (the n-th add call)"""
    fail_on = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adds = 0

    def add(self, *args, **kwargs):
        self.adds += 1
        if self.adds == self.fail_on:
            raise RuntimeError("disk full")
        return super().add(*args, **kwargs)


# ── commit 1: honest counts, the failed flag, --fail-on-error ────────────────

def test_chunks_stored_is_what_chroma_confirmed_after_a_failed_write(tmp_path, data):  # noqa: F811
    store = _store(tmp_path, cls=FlakyStore)
    stats = _run(tmp_path, data, store)
    assert store.count() > 0
    assert stats.chunks_stored == store.count()


def test_error_files_counts_only_the_files_that_did_not_get_all_their_chunks(tmp_path, data, expected):  # noqa: F811
    store = _store(tmp_path, cls=FlakyStore)
    stats = _run(tmp_path, data, store)
    stored = _per_file(store)
    incomplete = [name for name, n in expected.items() if stored.get(name, 0) != n]
    assert 0 < len(incomplete) < N_FILES, "some files failed and some did not"
    assert stats.error_files == len(incomplete)


def test_a_failed_run_is_flagged_and_a_clean_run_is_not(tmp_path, data):  # noqa: F811
    assert _run(tmp_path, data, _store(tmp_path, cls=FlakyStore)).failed is True
    clean = tmp_path / "clean"
    clean.mkdir()
    assert _run(clean, data, _store(clean)).failed is False


def test_the_existing_error_message_is_kept(tmp_path, data, capsys):  # noqa: F811
    _run(tmp_path, data, _store(tmp_path, cls=FlakyStore))
    assert "[ERROR] Bulk Indexing failed: Database write failed during ingest: disk full" in capsys.readouterr().out


def _ingest_args(**extra):
    base = {"source_type": "oran", "use_marker": False, "model": "minilm", "strategy": "text_baseline", "enrichment": "none",
            "data": None, "db_path": "db", "no_resume": True, "collection": None, "chunk_size": None, "overlap": None,
            "target_tokens": None, "max_tokens": None, "overlap_tokens": None}
    return argparse.Namespace(**{**base, **extra})


@pytest.fixture
def stub_ingest(monkeypatch):
    def install(**stats):
        monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: object())
        monkeypatch.setattr(cli_main, "_get_source", lambda *a, **k: object())
        monkeypatch.setattr(cli_main, "_get_store", lambda *a, **k: SimpleNamespace(collection_name="c"))
        monkeypatch.setattr(pipeline_module, "ingest", lambda **kw: SimpleNamespace(
            **{"files_processed": 0, "chunks_stored": 0, "skipped_files": 0, "error_files": 0, **stats}))
    return install


def test_the_exit_code_is_unchanged_without_the_flag(stub_ingest):
    stub_ingest(error_files=3, failed=True)
    cli_main._run_ingest(_ingest_args())                      # returns: the process exits 0, as before


@pytest.mark.parametrize("stats", [{"error_files": 3}, {"failed": True}, {"error_files": 1, "failed": True}])
def test_fail_on_error_exits_with_code_4_when_files_failed_or_the_run_failed(stub_ingest, stats):
    stub_ingest(**stats)
    with pytest.raises(SystemExit) as exc:
        cli_main._run_ingest(_ingest_args(fail_on_error=True))
    assert exc.value.code == cli_main.EXIT_INGEST_FAILED == 4


def test_fail_on_error_does_nothing_on_a_clean_run(stub_ingest):
    stub_ingest(failed=False)
    cli_main._run_ingest(_ingest_args(fail_on_error=True))


def test_fail_on_error_is_an_ingest_option_and_off_by_default():
    assert "--fail-on-error" in _help_of("ingest")
    assert "--fail-on-error" not in _help_of("search")


def _help_of(command: str) -> str:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        cli_main.main([command, "--help"])
    return buf.getvalue()


# ── commit 2: check the embedder's output before anything is written ─────────

class BadVectorEmbedder(SlowEmbedder):
    """returns a bad vector for every text containing `marker`; `kind` picks what is wrong"""

    def __init__(self, marker="function 3 ", kind="nan"):
        super().__init__()
        self.marker, self.kind = marker, kind

    def embed(self, texts, is_query=False):
        out = super().embed(texts, is_query)
        bad = []
        for text, vec in zip(texts, out):
            if self.marker in text and self.kind != "missing":
                vec = {"nan": [float("nan")] + vec[1:], "inf": [float("inf")] + vec[1:], "short": vec[:-1]}[self.kind]
            bad.append(vec)
        return bad[:-1] if self.kind == "missing" else bad


@pytest.mark.parametrize("kind", ["nan", "inf"])
def test_a_non_finite_vector_stops_the_run_before_its_batch_is_written(tmp_path, data, kind, capsys):  # noqa: F811
    store = _store(tmp_path)
    stats = _run(tmp_path, data, store, embedder=BadVectorEmbedder(kind=kind))
    out = capsys.readouterr().out
    assert stats.failed is True and stats.error_files > 0
    assert "NaN or infinite" in out and "23003-j00.md::bow::" in out, "the message names the chunk ids"
    ids = store._collection.get(include=[])["ids"]
    assert not any(i.startswith("23003-j00.md::") for i in ids), "nothing of the bad batch reached Chroma"
    assert _fts(tmp_path) == store.count() == stats.chunks_stored, "the two stores stay in step"


def test_a_vector_of_the_wrong_length_or_a_missing_vector_is_refused(tmp_path, data, capsys):  # noqa: F811
    for number, kind in enumerate(("short", "missing")):
        work = tmp_path / kind
        work.mkdir()
        stats = _run(work, data, _store(work), embedder=BadVectorEmbedder(kind=kind))
        out = capsys.readouterr().out
        assert stats.failed is True, kind
        assert "expected" in out and "vector" in out, out


def test_good_vectors_pass_through_unchanged(tmp_path, data):  # noqa: F811
    store = _store(tmp_path)
    stats = _run(tmp_path, data, store)
    assert stats.failed is False and stats.error_files == 0 and stats.chunks_stored == store.count() == _fts(tmp_path)


# ── commit 3: Chroma first, then BM25, one task per sub-batch; stop after the first failure ──

def test_the_bm25_index_never_runs_ahead_of_chroma_after_a_failed_write(tmp_path, data):  # noqa: F811
    store = _store(tmp_path, cls=FlakyStore)
    stats = _run(tmp_path, data, store)
    assert stats.failed is True
    assert _fts(tmp_path) == store.count() == stats.chunks_stored, "every BM25 row has its Chroma row"


def test_bm25_rows_are_written_only_after_the_chroma_write_of_the_same_rows(tmp_path, data, monkeypatch):  # noqa: F811
    store = _store(tmp_path)
    seen = []

    class CheckedFTS(pipeline_module.FTSStore):
        def sync(self, ids, documents, state):
            seen.append(len(ids))
            assert set(ids) <= set(store._collection.get(ids=list(ids), include=[])["ids"]), "BM25 write before its Chroma write"
            return super().sync(ids, documents, state)

    monkeypatch.setattr(pipeline_module, "FTSStore", CheckedFTS)
    stats = _run(tmp_path, data, store)
    assert seen and stats.failed is False and _fts(tmp_path) == store.count()


def test_no_further_writes_or_embedding_calls_are_started_after_the_first_failure(tmp_path, data):  # noqa: F811
    clean = tmp_path / "clean"
    clean.mkdir()
    clean_store, clean_embedder = _store(clean, cls=FlakyStore), SlowEmbedder()
    FlakyStore.fail_on = 10**6
    try:
        _run(clean, data, clean_store, embedder=clean_embedder)
    finally:
        FlakyStore.fail_on = 3
    failed_store, failed_embedder = _store(tmp_path, cls=FlakyStore), SlowEmbedder()
    stats = _run(tmp_path, data, failed_store, embedder=failed_embedder)
    assert stats.failed is True
    assert failed_store.adds < clean_store.adds, "writes stopped being dispatched"
    assert failed_embedder.calls < clean_embedder.calls, "embedding stopped too"


def test_a_failed_run_can_be_resumed_to_a_complete_consistent_index(tmp_path, data, expected):  # noqa: F811
    _run(tmp_path, data, _store(tmp_path, cls=FlakyStore))
    store = _store(tmp_path)
    stats = _run(tmp_path, data, store)
    assert stats.failed is False and _per_file(store) == expected
    assert _fts(tmp_path) == store.count()
