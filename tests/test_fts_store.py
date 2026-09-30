"""Small-sample validation of FTSStore (open_telco_rag/stores/fts.py) --
the SQLite FTS5 BM25 sparse index. This already exists and is built
automatically by core/pipeline.py's ingest() alongside every Chroma
collection (same collection_name, {collection}_fts.db); these tests just
confirm add()/search() behave sanely, since no test previously covered it.
"""
from __future__ import annotations

from open_telco_rag.stores.fts import FTSStore

_DOCS = {
    "doc::0": "The AMF handles registration, connection and mobility management.",
    "doc::1": "Beamforming improves signal quality by directing radio energy spatially.",
    "doc::2": "The gNB is the 5G radio access network base station.",
    "doc::3": "RAN sharing allows multiple operators to share radio access infrastructure.",
    "doc::4": "3GPP TS 23.501 defines the 5G System architecture.",
}


def _wire_store(tmp_path) -> FTSStore:
    store = FTSStore(path=str(tmp_path), collection_name="test_collection")
    store.add(list(_DOCS.keys()), list(_DOCS.values()))
    return store


def test_search_returns_exact_multi_term_match_top_ranked(tmp_path):
    store = _wire_store(tmp_path)
    results = store.search("AMF registration mobility", top_k=3)
    assert results
    assert results[0][0] == "doc::0"


def test_search_ranks_more_relevant_doc_above_partial_overlap(tmp_path):
    store = _wire_store(tmp_path)
    results = store.search("beamforming radio energy", top_k=3)
    doc_ids = [doc_id for doc_id, _ in results]
    assert doc_ids[0] == "doc::1"
    # doc::2/doc::3 only share "radio" -- should score lower, not absent.
    assert results[0][1] > results[-1][1]


def test_search_returns_empty_for_no_matching_terms(tmp_path):
    store = _wire_store(tmp_path)
    assert store.search("nonexistent gibberish xyzzy", top_k=5) == []


def test_search_on_missing_db_file_returns_empty_not_an_error(tmp_path):
    store = FTSStore(path=str(tmp_path), collection_name="never_populated")
    store.delete()  # closes the connection and unlinks the file
    assert store.search("anything", top_k=5) == []


def test_db_file_named_after_collection_matches_ingest_convention(tmp_path):
    """core/pipeline.py's ingest() and core/retrieval.py's search() both
    derive FTSStore(path=store.path, collection_name=store.collection_name)
    from the paired Chroma store -- confirms the {collection}_fts.db naming
    this depends on."""
    _wire_store(tmp_path)
    assert (tmp_path / "test_collection_fts.db").exists()


def test_clear_removes_previously_indexed_documents(tmp_path):
    store = _wire_store(tmp_path)
    assert store.search("AMF registration mobility", top_k=1)
    store.clear()
    assert store.search("AMF registration mobility", top_k=1) == []
