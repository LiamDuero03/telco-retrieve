"""Unit tests for open_telco_rag.stores.chroma.ChromaStore."""
from __future__ import annotations

import pytest

from open_telco_rag.stores.chroma import ChromaStore


def _store(tmp_path) -> ChromaStore:
    return ChromaStore(path=str(tmp_path / "db"), collection_name="test")


def test_update_metadata_merges_not_replaces(tmp_path):
    """Regression guard for the merge-vs-replace assumption update_metadata()'s
    docstring depends on -- a metadata-only backfill must not silently drop
    fields it wasn't asked to touch."""
    store = _store(tmp_path)
    store.add(
        ids=["a"], embeddings=[[0.1, 0.2]], documents=["hello"],
        metadatas=[{"x": 1, "y": 2}],
    )

    store.update_metadata(ids=["a"], metadatas=[{"y": 99, "z": 3}])

    result = store.search([0.1, 0.2], n_results=1)
    assert result[0].metadata == {"x": 1, "y": 99, "z": 3}


def test_update_metadata_leaves_embeddings_and_documents_untouched(tmp_path):
    store = _store(tmp_path)
    store.add(
        ids=["a"], embeddings=[[0.1, 0.2]], documents=["original text"],
        metadatas=[{"x": 1}],
    )

    store.update_metadata(ids=["a"], metadatas=[{"x": 2}])

    result = store.search([0.1, 0.2], n_results=1)
    assert result[0].text == "original text"
    assert result[0].metadata == {"x": 2}


def test_update_metadata_handles_multiple_ids_independently(tmp_path):
    store = _store(tmp_path)
    store.add(
        ids=["a", "b"], embeddings=[[0.1, 0.2], [0.3, 0.4]],
        documents=["doc a", "doc b"], metadatas=[{"spec": "a"}, {"spec": "b"}],
    )

    store.update_metadata(ids=["a", "b"], metadatas=[{"version": "V19.1.0"}, {"version": "V19.2.0"}])

    got = {r.doc_id: r.metadata for r in store.search([0.1, 0.2], n_results=2)}
    assert got["a"]["version"] == "V19.1.0"
    assert got["b"]["version"] == "V19.2.0"


def test_get_metadata_preserves_requested_order(tmp_path):
    store = _store(tmp_path)
    store.add(
        ids=["a", "b", "c"], embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"spec": "a"}, {"spec": "b"}, {"spec": "c"}],
    )

    # Requested in a different order than inserted -- chromadb's own .get()
    # doesn't guarantee return order, this must correct for that.
    got = store.get_metadata(["c", "a", "b"])

    assert got == [{"spec": "c"}, {"spec": "a"}, {"spec": "b"}]


def test_get_metadata_returns_empty_dict_for_unknown_id(tmp_path):
    store = _store(tmp_path)
    store.add(ids=["a"], embeddings=[[0.1, 0.2]], documents=["doc a"], metadatas=[{"spec": "a"}])

    got = store.get_metadata(["a", "does-not-exist"])

    assert got == [{"spec": "a"}, {}]


def test_close_actually_releases_the_underlying_connection(tmp_path):
    """Regression guard: close() must be a real teardown, not a no-op --
    otherwise a caller relying on it before deleting the store's directory
    (to avoid SQLite file-locking keeping disk blocks allocated after
    shutil.rmtree()) gets no actual guarantee."""
    store = _store(tmp_path)
    store.add(ids=["a"], embeddings=[[0.1, 0.2]], documents=["hi"], metadatas=[{"x": 1}])

    store.close()

    with pytest.raises(Exception):
        store.all_doc_ids()


def test_context_manager_closes_on_exit(tmp_path):
    with _store(tmp_path) as store:
        store.add(ids=["a"], embeddings=[[0.1, 0.2]], documents=["hi"], metadatas=[{"x": 1}])

    with pytest.raises(Exception):
        store.all_doc_ids()


def test_get_metadata_reflects_a_prior_update(tmp_path):
    store = _store(tmp_path)
    store.add(ids=["a"], embeddings=[[0.1, 0.2]], documents=["doc a"], metadatas=[{"x": 1}])
    store.update_metadata(ids=["a"], metadatas=[{"spec_version": "V19.7.0"}])

    got = store.get_metadata(["a"])

    assert got == [{"x": 1, "spec_version": "V19.7.0"}]
