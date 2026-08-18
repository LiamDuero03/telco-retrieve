"""Lightweight end-to-end integration test: a real ingest() call against 2 tiny
local .md files, using the real all-MiniLM-L6-v2 embedding model on CPU.

The one test in this suite that loads a real model — everything else uses
fakes/fixtures. minilm is small (~80MB), CPU-only, no GPU/API key needed.
"""
from __future__ import annotations

import pytest

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.core.pipeline import ingest, search
from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore

# Filenames follow TSpec-LLM's pattern (\d{5}-\w+.md) so TPPSource.metadata()
# parses them realistically.
DOC_1 = """# Scope

This document specifies the Access and Mobility Management Function (AMF)
procedures for 5G System registration and connection management.

## Registration procedure

The UE initiates registration by sending a Registration Request message
to the AMF over the N1 interface.
"""

DOC_2 = """# Scope

This document specifies the Session Management Function (SMF) procedures
for 5G System PDU session establishment.

## PDU session establishment

The UE requests PDU session establishment via the AMF, which forwards
the request to the SMF over the N11 interface.
"""


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "23501-j00.md").write_text(DOC_1, encoding="utf-8")
    (d / "23502-j00.md").write_text(DOC_2, encoding="utf-8")
    return d


def test_ingest_two_files_end_to_end_with_minilm_on_cpu(data_dir, tmp_path):
    source   = TPPSource()
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    store    = ChromaStore(
        path=tmp_path / "chroma_db", collection_name="test_3gpp_minilm",
        metadata={"hnsw:space": "cosine"},
    )

    stats = ingest(
        source=source, store=store, embedder=embedder,
        data_dir=str(data_dir),
        strategy="text_baseline", chunk_size=300, overlap=30,
        resume=False, log_dir=str(tmp_path / "logs"),
    )

    assert stats.files_processed == 2
    assert stats.error_files == 0
    assert stats.skipped_files == 0
    assert stats.chunks_stored > 0
    assert store.count() == stats.chunks_stored

    # Proves the dense-vector round trip (embed -> store -> query) actually works.
    results = search(
        "How does the UE establish a PDU session?",
        store=store, embedder=embedder, top_k=2,
    )
    assert len(results) > 0
    assert any("SMF" in r.text or "PDU session" in r.text for r in results)


def test_ingest_pipelines_multiple_flush_batches_correctly(tmp_path):
    """Regression: forces multiple files to each exceed batch_size, exercising the
    `while len(buf_texts) >= batch_size: _flush(batch_size)` loop (not just the
    tail flush). Confirms every chunk lands exactly once, no duplication or loss.
    """
    source   = TPPSource()
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    store    = ChromaStore(
        path=tmp_path / "chroma_db", collection_name="test_multibatch_minilm",
        metadata={"hnsw:space": "cosine"},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Each file alone produces multiple chunks, well above batch_size=2.
    for i, topic in enumerate(["AMF registration", "SMF session", "UPF forwarding"], start=1):
        paragraphs = "\n\n".join(
            f"Paragraph {p} about {topic} handling case {p}." for p in range(1, 5)
        )
        (data_dir / f"2350{i}-j00.md").write_text(f"# Scope\n\n{paragraphs}\n", encoding="utf-8")

    stats = ingest(
        source=source, store=store, embedder=embedder,
        data_dir=str(data_dir),
        strategy="text_baseline", chunk_size=60, overlap=0,
        min_length=10,   # these deliberately-short paragraphs are below the default (100)
        batch_size=2,    # deliberately tiny -- forces multiple _flush() calls per file
        resume=False, log_dir=str(tmp_path / "logs"),
    )

    assert stats.files_processed == 3
    assert stats.error_files == 0
    assert stats.chunks_stored > 3   # multiple chunks per file at this chunk_size
    assert store.count() == stats.chunks_stored

    from open_telco_rag.stores.fts import FTSStore
    fts_store = FTSStore(path=str(tmp_path / "chroma_db"), collection_name="test_multibatch_minilm")
    hits = fts_store.search("UPF forwarding", top_k=5)
    assert len(hits) > 0


def test_ingest_write_batch_size_decoupled_from_embed_batch_size(tmp_path):
    """Regression: writes are consolidated into a separate, larger buffer than the
    embedding batch. Confirms every chunk lands exactly once even when
    write_batch_size and embed batch_size differ."""
    source   = TPPSource()
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    store    = ChromaStore(
        path=tmp_path / "chroma_db", collection_name="test_writebatch_minilm",
        metadata={"hnsw:space": "cosine"},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, topic in enumerate(["AMF registration", "SMF session", "UPF forwarding"], start=1):
        paragraphs = "\n\n".join(
            f"Paragraph {p} about {topic} handling case {p}." for p in range(1, 4)
        )
        (data_dir / f"2350{i}-j00.md").write_text(f"# Scope\n\n{paragraphs}\n", encoding="utf-8")

    stats = ingest(
        source=source, store=store, embedder=embedder,
        data_dir=str(data_dir),
        strategy="text_baseline", chunk_size=60, overlap=0,
        min_length=10,
        batch_size=1,          # embed one chunk at a time
        write_batch_size=3,    # but only write every 3 embedded chunks
        resume=False, log_dir=str(tmp_path / "logs"),
    )

    assert stats.files_processed == 3
    assert stats.error_files == 0
    assert stats.chunks_stored > 3
    assert store.count() == stats.chunks_stored


def test_a_failed_resume_progress_write_does_not_misreport_a_successful_ingest(tmp_path, monkeypatch):
    """Regression: the resume progress-file write shared the same try/except as the
    write-critical embed/store loop, so a failure writing the small JSON tracking
    file was misreported as a data-loss failure (error_files inflated). Patches
    Path.write_text to fail only for the progress-file path, confirms
    chunks_stored/error_files still reflect the real successful outcome.
    """
    from pathlib import Path as _Path

    source   = TPPSource()
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    store    = ChromaStore(
        path=tmp_path / "chroma_db", collection_name="test_resumefail_minilm",
        metadata={"hnsw:space": "cosine"},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "23501-j00.md").write_text(
        "# Scope\n\nThe AMF exposes the N1 interface for registration.\n", encoding="utf-8",
    )

    real_write_text = _Path.write_text

    def _flaky_write_text(self, *args, **kwargs):
        if self.name.startswith(".progress_"):
            raise OSError("simulated disk-full while writing the resume progress file")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "write_text", _flaky_write_text)

    stats = ingest(
        source=source, store=store, embedder=embedder,
        data_dir=str(data_dir),
        strategy="text_baseline", chunk_size=200, overlap=20, min_length=10,
        resume=True, log_dir=str(tmp_path / "logs"),
    )

    # Confirmed against the real Chroma collection, not just in-memory stats.
    assert stats.chunks_stored > 0
    assert store.count() == stats.chunks_stored
    assert stats.error_files == 0, (
        "a resume-progress-file write failure must not be misreported as "
        "a data-loss (write-critical) failure"
    )
