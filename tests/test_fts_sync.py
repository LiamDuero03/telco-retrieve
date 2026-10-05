"""The BM25 (FTS5) index mirrors Chroma: when a chunk id is written again with different text, its old rows are
deleted by rowid and the new text is inserted. Fake embedders, no GPU, no network."""
from __future__ import annotations

import json
import re
import sqlite3

import pytest

from open_telco_rag.stores.fts import FTSStore
from tests.test_ingest_checkpoint import Killed, SlowEmbedder, _run, _store, data  # noqa: F401  (data is a fixture)

BOUNDED = {"bound_tokens": 30}          # splits the test chunks: same ids, different texts and more of them


def _fts_pairs(tmp_path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(str(tmp_path / "db" / "coll_fts.db"))
    try:
        return conn.execute("SELECT doc_id, text FROM bm25_index").fetchall()
    finally:
        conn.close()


def _assert_mirror(tmp_path, store) -> None:
    """every Chroma chunk has exactly one BM25 row with its text, and the index holds nothing else"""
    got = store._collection.get(include=["documents"])
    chroma = sorted(zip(got["ids"], got["documents"]))
    assert sorted(_fts_pairs(tmp_path)) == chroma


def _drop_progress(tmp_path, keep_bounded: bool) -> None:
    for path in (tmp_path / "logs").glob(".progress_*.json"):
        bounded = "bound_tokens" in json.dumps(json.loads(path.read_text(encoding="utf8"))["settings"])
        if bounded != keep_bounded:
            path.unlink()


# ── the changed-settings rerun: plain, bounded, forget the plain record, plain again ──

def test_the_index_mirrors_chroma_right_after_the_first_resume_under_changed_settings(tmp_path, data):  # noqa: F811
    store = _store(tmp_path)
    _run(tmp_path, data, store)
    _assert_mirror(tmp_path, store)
    _run(tmp_path, data, store, **BOUNDED)
    _assert_mirror(tmp_path, store)
    _drop_progress(tmp_path, keep_bounded=True)                  # the plain record is gone: plain runs again
    _run(tmp_path, data, store)
    _assert_mirror(tmp_path, store)                              # the FIRST resume, not the next one
    _run(tmp_path, data, store)
    _assert_mirror(tmp_path, store)


# ── unchanged settings: kill at several points, resume once ──────────────────

@pytest.mark.parametrize("die_on_call", [3, 6, 9, 12, 15])
@pytest.mark.parametrize("over", [{}, BOUNDED], ids=["plain", "bounded"])
def test_the_index_mirrors_chroma_after_the_first_resume_of_a_killed_run(tmp_path, data, die_on_call, over):  # noqa: F811
    store = _store(tmp_path)
    with pytest.raises(Killed):
        _run(tmp_path, data, store, embedder=SlowEmbedder(die_on_call=die_on_call), **over)
    _run(tmp_path, data, store, **over)
    _assert_mirror(tmp_path, store)
    assert len(_fts_pairs(tmp_path)) == store.count() > 0


# ── what happens to rows whose text differs ──────────────────────────────────

def test_a_row_with_different_text_is_replaced_when_its_chunk_is_written_again(tmp_path, data, capsys):  # noqa: F811
    store = _store(tmp_path)
    _run(tmp_path, data, store)
    fts = FTSStore(path=tmp_path / "db", collection_name="coll")
    doc_id = fts._get_conn().execute("SELECT doc_id FROM bm25_index LIMIT 1").fetchone()[0]
    fts.add([doc_id], ["text from some other chunking configuration"])
    assert len(_fts_pairs(tmp_path)) == store.count() + 1
    _drop_progress(tmp_path, keep_bounded=True)                  # every file is ingested again
    capsys.readouterr()

    _run(tmp_path, data, store)
    _assert_mirror(tmp_path, store)
    assert "text from some other chunking configuration" not in {t for _, t in _fts_pairs(tmp_path)}
    assert "Replaced 1 BM25 rows whose text had changed" in capsys.readouterr().out


def test_a_row_with_different_text_for_a_file_that_is_not_written_again_is_kept_and_reported(tmp_path, data, capsys):  # noqa: F811
    store = _store(tmp_path)
    _run(tmp_path, data, store)
    fts = FTSStore(path=tmp_path / "db", collection_name="coll")
    doc_id = fts._get_conn().execute("SELECT doc_id FROM bm25_index LIMIT 1").fetchone()[0]
    fts.add([doc_id], ["text from some other chunking configuration"])
    capsys.readouterr()

    _run(tmp_path, data, store)                                  # every file is recorded as done: nothing is written
    assert len(_fts_pairs(tmp_path)) == store.count() + 1, "nothing was dropped"
    assert "1 BM25 rows repeat a chunk id with different text" in capsys.readouterr().out


# ── FTSStore: rowids and hashes loaded once, no per-row table scan ───────────

def _store_with(tmp_path, rows):
    fts = FTSStore(path=tmp_path / "fts", collection_name="u")
    fts.add([i for i, _ in rows], [t for _, t in rows])
    return fts


def _pairs(fts):
    return sorted(fts._get_conn().execute("SELECT doc_id, text FROM bm25_index").fetchall())


def test_load_rows_gives_the_rowid_and_hash_of_every_row_per_id(tmp_path):
    fts = _store_with(tmp_path, [("a", "one"), ("b", "two"), ("b", "three")])
    state = fts.load_rows()
    assert set(state) == {"a", "b"} and len(state["a"]) == 1 and len(state["b"]) == 2
    assert state["a"][0][1] == hash("one") and {h for _, h in state["b"]} == {hash("two"), hash("three")}
    assert all(isinstance(rowid, int) for rows in state.values() for rowid, _ in rows)


def test_sync_inserts_new_ids_keeps_unchanged_rows_and_replaces_changed_ones(tmp_path):
    fts = _store_with(tmp_path, [("a", "one"), ("b", "two")])
    state = fts.load_rows()
    replaced = fts.sync(["a", "b", "c"], ["one", "two changed", "three"], state)
    assert replaced == 1
    assert _pairs(fts) == [("a", "one"), ("b", "two changed"), ("c", "three")]
    assert fts.sync(["a", "b", "c"], ["one", "two changed", "three"], state) == 0, "the state follows the writes"
    assert _pairs(fts) == [("a", "one"), ("b", "two changed"), ("c", "three")]


def test_sync_leaves_exactly_one_row_per_id_even_when_the_index_had_several(tmp_path):
    fts = _store_with(tmp_path, [("a", "one"), ("a", "other"), ("a", "one"), ("a", "third")])
    replaced = fts.sync(["a"], ["one"], fts.load_rows())
    assert _pairs(fts) == [("a", "one")] and replaced == 3


def test_sync_never_scans_the_table_per_row(tmp_path):
    n = 1500
    fts = _store_with(tmp_path, [(f"id{i}", f"text {i}") for i in range(n)])
    state = fts.load_rows()
    statements: list[str] = []
    fts._get_conn().set_trace_callback(statements.append)
    fts.sync([f"id{i}" for i in range(n)], [f"changed {i}" if i % 2 else f"text {i}" for i in range(n)], state)
    fts._get_conn().set_trace_callback(None)
    assert len(_pairs(fts)) == n
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert not selects, "the rows were looked up in memory, not in the table"
    deletes = [s for s in statements if s.lstrip().upper().startswith("DELETE")]
    assert deletes and all(re.search(r"WHERE\s+rowid\s+IN", s, re.IGNORECASE) for s in deletes), "deletes go by rowid"
    assert len(deletes) <= n // 100, "deletes are batched"
