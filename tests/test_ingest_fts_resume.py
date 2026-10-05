"""After an interrupted and resumed ingest the BM25 (FTS5) index holds each chunk once."""
from __future__ import annotations

import pytest

from open_telco_rag.stores.chroma import ChromaStore
from open_telco_rag.stores.fts import FTSStore
from tests.test_ingest_checkpoint import (  # noqa: F401  (data is a fixture)
    Killed,
    SlowEmbedder,
    _run,
    _store,
    data,
)


def _fts_rows(tmp_path):
    return FTSStore(path=tmp_path / "db", collection_name="coll")._get_conn().execute(
        "SELECT COUNT(*), COUNT(DISTINCT doc_id) FROM bm25_index").fetchone()


def test_fts_rows_equal_chroma_chunks_after_a_failed_write_and_after_the_resume(tmp_path, data):  # noqa: F811
    class FlakyStore(ChromaStore):
        """the Chroma write of one flush fails: the BM25 write of the same rows must not happen either"""
        adds = 0

        def add(self, *args, **kwargs):
            FlakyStore.adds += 1
            if FlakyStore.adds == 3:
                raise RuntimeError("disk full")
            return super().add(*args, **kwargs)

    _run(tmp_path, data, _store(tmp_path, cls=FlakyStore))
    rows, distinct = _fts_rows(tmp_path)
    assert rows == distinct == _store(tmp_path).count(), "the failed run leaves FTS and Chroma in step"

    store = _store(tmp_path)
    _run(tmp_path, data, store)
    rows, distinct = _fts_rows(tmp_path)
    assert rows == distinct == store.count()


def test_a_kill_and_resume_leaves_one_fts_row_per_chunk(tmp_path, data):  # noqa: F811
    store = _store(tmp_path)
    with pytest.raises(Killed):
        _run(tmp_path, data, store, embedder=SlowEmbedder(die_on_call=8))
    _run(tmp_path, data, store)
    rows, distinct = _fts_rows(tmp_path)
    assert rows == distinct == store.count() > 0


def test_repeated_rows_from_an_older_interrupted_run_are_removed(tmp_path, data, capsys):  # noqa: F811
    store = _store(tmp_path)
    _run(tmp_path, data, store)
    fts = FTSStore(path=tmp_path / "db", collection_name="coll")
    rows = fts._get_conn().execute("SELECT doc_id, text FROM bm25_index LIMIT 4").fetchall()
    fts.add([r[0] for r in rows], [r[1] for r in rows])          # what the old code did on resume
    assert _fts_rows(tmp_path)[0] == store.count() + 4
    capsys.readouterr()

    _run(tmp_path, data, store)
    assert _fts_rows(tmp_path)[0] == store.count()
    assert "Removed 4 repeated BM25 rows" in capsys.readouterr().out


def test_an_uninterrupted_run_and_a_no_resume_rebuild_are_unchanged(tmp_path, data):  # noqa: F811
    store = _store(tmp_path)
    _run(tmp_path, data, store)
    assert _fts_rows(tmp_path) == (store.count(),) * 2
    _run(tmp_path, data, store, resume=False)
    assert _fts_rows(tmp_path) == (store.count(),) * 2
