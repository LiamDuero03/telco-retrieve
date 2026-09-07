"""Tests for ingest_multi_enrichment() — the chunk-once/fan-out-to-N-variants
orchestrator used by master_grid.py, and its two safety fixes:

  1. Completion marker (_ingest_complete.json): written into each variant's
     store directory only once a run finishes without a fatal exception, so
     a caller can tell a genuinely finished run apart from one interrupted
     mid-ingest (OOM/node-preemption/power loss) — a bare "does the
     directory have files in it" check can't, since VectorStore construction
     alone creates bootstrap files before any chunk is ever stored.
  2. enrichment_kwargs threading: lets a caller override e.g.
     acronym_expansion's default bundled glossary, which previously had no
     path to reach enrich_chunks() at all.

Uses real minilm on CPU and real multiprocessing (same as
test_ingest_integration.py) — this function's whole point is the
chunk-once/embed-per-variant fan-out, which isn't meaningfully testable
with fakes standing in for the process pool.
"""
from __future__ import annotations

import json

import pytest

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.core.pipeline import ingest_multi_enrichment
from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "23501-j00.md").write_text(
        "# Scope\n\nThe AMF exposes the N1 interface. RIC handles E2AP messages.\n",
        encoding="utf-8",
    )
    (d / "23502-j00.md").write_text(
        "# Scope\n\nThe SMF handles PDU session establishment via N11.\n",
        encoding="utf-8",
    )
    return d


def _make_stores(tmp_path, enrichment_strategies):
    return {
        ev: ChromaStore(
            path=tmp_path / f"chroma_{ev}", collection_name=f"test_{ev}_minilm",
            metadata={"hnsw:space": "cosine"},
        )
        for ev in enrichment_strategies
    }


def test_variant_isolation_and_completion_markers(data_dir, tmp_path):
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    enrichment_strategies = ["none", "acronym_expansion"]
    stores = _make_stores(tmp_path, enrichment_strategies)

    stats = ingest_multi_enrichment(
        source=TPPSource(), stores=stores, embedder=embedder, data_dir=str(data_dir),
        enrichment_strategies=enrichment_strategies,
        strategy="text_baseline", chunk_size=200, overlap=20, min_length=10,
        log_dir=str(tmp_path / "logs"),
    )

    # Chunking happened once, but both variants still got fully embedded/stored.
    for ev in enrichment_strategies:
        assert stats[ev].files_processed == 2
        assert stats[ev].error_files == 0
        assert stats[ev].chunks_stored > 0
        assert stores[ev].count() == stats[ev].chunks_stored

    # Real content check - the deep-copy-per-variant isolation actually holds,
    # not just that both stores independently have "some" chunks.
    none_docs = stores["none"]._collection.get(include=["documents"])["documents"]
    acro_docs = stores["acronym_expansion"]._collection.get(include=["documents"])["documents"]
    assert not any("[Acronym Expansions]" in t for t in none_docs)
    assert any("[Acronym Expansions]" in t for t in acro_docs)

    # Completion marker written for every variant, with sane content.
    for ev in enrichment_strategies:
        marker = tmp_path / f"chroma_{ev}" / "_ingest_complete.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["complete"] is True
        assert data["files_total"] == 2
        assert data["files_processed"] == 2
        assert data["error_files"] == 0
        assert data["chunks_stored"] == stats[ev].chunks_stored


def test_enrichment_kwargs_glossary_override_reaches_enrich_chunks(data_dir, tmp_path):
    """Regression test for a fixed gap: enrichment_kwargs (e.g. glossary_dict)
    used to have no path from ingest_multi_enrichment() to enrich_chunks() —
    anything passed via **chunk_kwargs was silently absorbed by
    chunk_document() instead and never reached enrichment, so a custom
    glossary was quietly ignored in favor of the bundled default."""
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    stores = _make_stores(tmp_path, ["acronym_expansion"])
    custom_glossary = {
        "RIC": "RAN Intelligent Controller (custom glossary)",
        "E2AP": "E2 Application Protocol (custom glossary)",
    }

    ingest_multi_enrichment(
        source=TPPSource(), stores=stores, embedder=embedder, data_dir=str(data_dir),
        enrichment_strategies=["acronym_expansion"],
        enrichment_kwargs={"glossary_dict": custom_glossary},
        strategy="text_baseline", chunk_size=200, overlap=20, min_length=10,
        log_dir=str(tmp_path / "logs"),
    )

    docs = stores["acronym_expansion"]._collection.get(include=["documents"])["documents"]
    assert any("custom glossary" in t for t in docs)


def test_completion_marker_written_when_data_dir_has_no_matching_files(tmp_path):
    """Edge case: the early-return path (no files found) must still mark
    completion — otherwise a corpus directory that's legitimately empty
    would look "interrupted" forever to a caller checking for the marker."""
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    stores = _make_stores(tmp_path, ["none"])

    stats = ingest_multi_enrichment(
        source=TPPSource(), stores=stores, embedder=embedder, data_dir=str(empty_dir),
        enrichment_strategies=["none"], strategy="text_baseline",
        chunk_size=200, overlap=20, log_dir=str(tmp_path / "logs"),
    )

    assert stats["none"].chunks_stored == 0
    marker = tmp_path / "chroma_none" / "_ingest_complete.json"
    assert marker.exists()
    assert json.loads(marker.read_text())["files_total"] == 0


def test_a_failed_background_write_prevents_a_false_completion_marker(data_dir, tmp_path, monkeypatch):
    """Regression: background store.add() failures in the writer thread
    pool were silent (wait() never checked .exception()), so ingest wrote
    a completion marker as if every chunk persisted -- caught via
    audit_ingestion_grid.py showing claimed counts far above real DB
    counts. Patches ChromaStore.add to raise; confirms no marker is
    written and stats reflect the failure."""
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    stores = _make_stores(tmp_path, ["none"])

    def _boom(self, ids, embeddings, documents, metadatas):
        raise RuntimeError("simulated concurrent-write failure")

    monkeypatch.setattr(ChromaStore, "add", _boom)

    stats = ingest_multi_enrichment(
        source=TPPSource(), stores=stores, embedder=embedder, data_dir=str(data_dir),
        enrichment_strategies=["none"], strategy="text_baseline",
        chunk_size=200, overlap=20, min_length=10, log_dir=str(tmp_path / "logs"),
    )

    marker = tmp_path / "chroma_none" / "_ingest_complete.json"
    assert not marker.exists(), (
        "a failed write must NOT produce a completion marker claiming success"
    )
    assert stats["none"].error_files > 0


def test_a_failed_completion_marker_write_does_not_misreport_a_successful_ingest(data_dir, tmp_path, monkeypatch):
    """Regression: post-write bookkeeping (marker write) used to share the
    embed-and-write loop's try/except, so one variant's marker-write
    failure got misreported as total data loss for every variant. Patches
    _write_completion_marker to fail for one variant only; confirms the
    other variant's stats/marker are unaffected."""
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    enrichment_strategies = ["none", "acronym_expansion"]
    stores = _make_stores(tmp_path, enrichment_strategies)

    real_write_marker = pipeline_module._write_completion_marker

    def _flaky_write_marker(store, files_total, stats):
        if getattr(store, "collection_name", "") == "test_none_minilm":
            raise OSError("simulated disk-full while writing the completion marker")
        return real_write_marker(store, files_total, stats)

    monkeypatch.setattr(pipeline_module, "_write_completion_marker", _flaky_write_marker)

    stats = ingest_multi_enrichment(
        source=TPPSource(), stores=stores, embedder=embedder, data_dir=str(data_dir),
        enrichment_strategies=enrichment_strategies,
        strategy="text_baseline", chunk_size=200, overlap=20, min_length=10,
        log_dir=str(tmp_path / "logs"),
    )

    # Every chunk was genuinely stored for BOTH variants -- confirmed
    # against the real Chroma collection, not just the in-memory stats.
    for ev in enrichment_strategies:
        assert stats[ev].chunks_stored > 0
        assert stores[ev].count() == stats[ev].chunks_stored
        assert stats[ev].error_files == 0, (
            f"'{ev}' was fully and correctly stored -- a completion-marker "
            f"write failure for a DIFFERENT variant must not inflate its error_files"
        )

    # The variant whose marker write failed correctly has no marker (so a
    # future run knows to re-verify/re-sync it) -- but the unaffected
    # variant's marker was written normally.
    assert not (tmp_path / "chroma_none" / "_ingest_complete.json").exists()
    assert (tmp_path / "chroma_acronym_expansion" / "_ingest_complete.json").exists()


def test_writes_are_split_into_sub_batches_never_exceeding_store_max_batch_size(tmp_path, monkeypatch):
    """Regression: writing 615k chunks accumulated into a write buffer
    larger than Chroma's real max_batch_size, crashing only after the
    progress bar reported 100% done. Spies on ChromaStore.add to confirm
    writes actually split at max_batch_size, not just that batch_size
    happens to stay under it."""
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    stores = _make_stores(tmp_path, ["none"])
    stores["none"].max_batch_size = 2  # artificially tiny, well below batch_size below

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    paragraphs = "\n\n".join(f"Paragraph {p} about AMF registration case {p}." for p in range(1, 9))
    (data_dir / "23501-j00.md").write_text(f"# Scope\n\n{paragraphs}\n", encoding="utf-8")

    observed_sizes = []
    real_add = ChromaStore.add

    def _spy_add(self, ids, embeddings, documents, metadatas):
        observed_sizes.append(len(ids))
        return real_add(self, ids, embeddings, documents, metadatas)

    monkeypatch.setattr(ChromaStore, "add", _spy_add)

    stats = ingest_multi_enrichment(
        source=TPPSource(), stores=stores, embedder=embedder, data_dir=str(data_dir),
        enrichment_strategies=["none"], strategy="text_baseline",
        chunk_size=60, overlap=0, min_length=10,
        batch_size=5,   # embed batch bigger than the tiny max_batch_size=2 above
        log_dir=str(tmp_path / "logs"),
    )

    assert stats["none"].chunks_stored > 2, "test didn't exercise multiple sub-batches"
    assert observed_sizes, "store.add() was never called"
    assert all(n <= 2 for n in observed_sizes), f"a write exceeded max_batch_size: {observed_sizes}"
    assert stores["none"].count() == stats["none"].chunks_stored


def test_chunks_stored_reflects_only_confirmed_writes_on_a_mid_variant_failure(tmp_path, monkeypatch):
    """Regression: chunks_stored used to increment at embed time, not
    write time, so a mid-buffer write failure over-reported it. Forces
    the second add() call to fail and confirms the count reflects only
    sub-batches that actually succeeded."""
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    stores = _make_stores(tmp_path, ["none"])
    stores["none"].max_batch_size = 2

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    paragraphs = "\n\n".join(f"Paragraph {p} about AMF registration case {p}." for p in range(1, 9))
    (data_dir / "23501-j00.md").write_text(f"# Scope\n\n{paragraphs}\n", encoding="utf-8")

    call_count = 0
    real_add = ChromaStore.add

    def _fail_second_call(self, ids, embeddings, documents, metadatas):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated write failure on the second sub-batch")
        return real_add(self, ids, embeddings, documents, metadatas)

    monkeypatch.setattr(ChromaStore, "add", _fail_second_call)

    stats = ingest_multi_enrichment(
        source=TPPSource(), stores=stores, embedder=embedder, data_dir=str(data_dir),
        enrichment_strategies=["none"], strategy="text_baseline",
        chunk_size=60, overlap=0, min_length=10,
        batch_size=5, log_dir=str(tmp_path / "logs"),
    )

    # Only the first sub-batch (max_batch_size=2) succeeded before the
    # second one raised -- chunks_stored must reflect exactly that count,
    # not the larger embed batch_size=5 or the full corpus.
    assert stats["none"].chunks_stored == 2
    assert stores["none"].count() == 2
    marker = tmp_path / "chroma_none" / "_ingest_complete.json"
    assert not marker.exists()


def test_missing_store_for_enrichment_strategy_raises_value_error(data_dir, tmp_path):
    """Fails fast with a clear error rather than a confusing KeyError deep
    inside the pipelined loop."""
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    stores = _make_stores(tmp_path, ["none"])  # missing "acronym_expansion"

    with pytest.raises(ValueError, match="acronym_expansion"):
        ingest_multi_enrichment(
            source=TPPSource(), stores=stores, embedder=embedder, data_dir=str(data_dir),
            enrichment_strategies=["none", "acronym_expansion"],
            strategy="text_baseline", chunk_size=200, overlap=20,
            log_dir=str(tmp_path / "logs"),
        )
