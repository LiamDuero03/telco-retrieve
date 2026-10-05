"""FTSStore.load_rows() and sync(): the BM25 (FTS5) index can mirror Chroma. Rows are found by rowid from a state loaded
once, never by a lookup in the table per row."""
from __future__ import annotations

import re

from open_telco_rag.stores.fts import FTSStore

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
