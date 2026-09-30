"""Ingest resume checkpoints: a file is recorded as done as soon as every one of its chunks is
confirmed stored, so a run that dies midway resumes from the last checkpoint instead of from zero."""
from __future__ import annotations

import json
import time
from collections import Counter

import pytest

from open_telco_rag.core.pipeline import ingest
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore
from tests.test_retrieval_methods import BagOfWordsEmbedder

N_FILES = 14
CHUNK = {"strategy": "text_baseline", "chunk_size": 200, "overlap": 0, "min_length": 10}
SMALL_BATCHES = {"batch_size": 3, "write_batch_size": 5}    # many flushes that do not line up with files


class Killed(BaseException):
    """not an Exception, so ingest() cannot catch it: stands in for a killed process"""


class SlowEmbedder(BagOfWordsEmbedder):
    """gives the background writers time to finish; dies on the given call"""

    def __init__(self, die_on_call=None):
        self.calls = 0
        self.die_on_call = die_on_call

    def embed(self, texts, is_query=False):
        self.calls += 1
        if self.die_on_call is not None and self.calls == self.die_on_call:
            raise Killed()
        time.sleep(0.03)
        return super().embed(texts, is_query)


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for i in range(N_FILES):
        body = "\n\n".join(f"Clause {i}.{j} says that function {i} shall support procedure {j} on the interface."
                           for j in range(3 + i % 4 * 3))
        (d / f"{23000 + i}-j00.md").write_text(f"# Spec {i}\n\n{body}\n", encoding="utf-8")
    return d


def _store(tmp_path, name="db", cls=ChromaStore):
    return cls(path=tmp_path / name, collection_name="coll", metadata={"hnsw:space": "cosine"})


def _run(tmp_path, data, store, embedder=None, resume=True, **over):
    return ingest(source=TPPSource(), store=store, embedder=embedder or SlowEmbedder(), data_dir=str(data),
                  resume=resume, log_dir=str(tmp_path / "logs"), **{**CHUNK, **SMALL_BATCHES, **over})


def _per_file(store):
    return Counter(i.split("::")[0] for i in store._collection.get(include=[])["ids"])


def _progress(tmp_path):
    files = [p for p in (tmp_path / "logs").glob(".progress_*.json") if not p.name.endswith(".bak")]
    assert len(files) <= 1
    return json.loads(files[0].read_text(encoding="utf8")) if files else None


@pytest.fixture
def expected(tmp_path, data):
    """chunks per file from an uninterrupted run in a separate database"""
    store = _store(tmp_path, "reference_db")
    (tmp_path / "reference").mkdir()
    _run(tmp_path / "reference", data, store, resume=False)
    return _per_file(store)


def test_a_killed_run_resumes_with_exact_counts_and_no_duplicates(tmp_path, data, expected):
    store = _store(tmp_path)
    with pytest.raises(Killed):
        _run(tmp_path, data, store, embedder=SlowEmbedder(die_on_call=8))
    done = set(_progress(tmp_path)["completed"])
    assert 0 < len(done) < N_FILES, "the checkpoints were written before the run died"

    stats = _run(tmp_path, data, store)
    assert stats.skipped_files == len(done) and stats.files_processed == N_FILES - len(done)
    ids = store._collection.get(include=[])["ids"]
    assert len(ids) == len(set(ids)) == sum(expected.values())
    assert _per_file(store) == expected, "no file lost or duplicated"
    assert sorted(_progress(tmp_path)["completed"]) == sorted(f.name for f in data.iterdir())


def test_a_file_is_recorded_only_when_all_its_chunks_are_stored(tmp_path, data, expected):
    store = _store(tmp_path)
    with pytest.raises(Killed):
        _run(tmp_path, data, store, embedder=SlowEmbedder(die_on_call=9))
    stored = _per_file(store)
    for name in _progress(tmp_path)["completed"]:
        assert stored[name] == expected[name], f"{name} is recorded as done but only partly stored"


def test_a_failing_store_write_does_not_record_files_that_were_not_stored(tmp_path, data, expected):
    class FlakyStore(ChromaStore):
        adds = 0

        def add(self, *args, **kwargs):
            FlakyStore.adds += 1
            if FlakyStore.adds == 4:
                raise RuntimeError("disk full")
            return super().add(*args, **kwargs)

    store = _store(tmp_path, cls=FlakyStore)
    stats = _run(tmp_path, data, store)
    assert stats.error_files > 0
    stored = _per_file(store)
    for name in (_progress(tmp_path) or {"completed": []})["completed"]:
        assert stored[name] == expected[name]

    resumed = _run(tmp_path, data, _store(tmp_path))
    assert resumed.error_files == 0 and _per_file(_store(tmp_path)) == expected


def test_an_uninterrupted_run_writes_the_same_progress_file_as_before(tmp_path, data):
    store = _store(tmp_path)
    stats = _run(tmp_path, data, store)
    progress = _progress(tmp_path)
    assert stats.files_processed == N_FILES and stats.chunks_stored == store.count()
    assert progress["version"] == 2 and progress["completed"] == sorted(f.name for f in data.iterdir())
    assert set(progress) == {"version", "fingerprint", "settings", "completed"}


def test_no_progress_is_written_without_resume(tmp_path, data):
    _run(tmp_path, data, _store(tmp_path), resume=False)
    assert _progress(tmp_path) is None


def test_a_progress_file_from_before_checkpointing_still_resumes(tmp_path, data):
    store = _store(tmp_path)
    _run(tmp_path, data, store)
    before = _progress(tmp_path)
    again = _run(tmp_path, data, store)
    assert again.files_processed == 0 and again.skipped_files == N_FILES
    assert _progress(tmp_path) == before
