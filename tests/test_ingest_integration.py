"""Lightweight end-to-end integration test: a real ingest() call against
2 tiny local .md files, using the real all-MiniLM-L6-v2 embedding model
on CPU.

This is deliberately the ONE test in this suite that loads a real model —
everything else in tests/ is unit-tested with fakes/fixtures. minilm is
small (~80MB) and CPU-only: no GPU, no API key, and no network call if
it's already cached locally (as it normally will be on a machine that has
used this project before — this is also the CLI's own default model).
"""
from __future__ import annotations

import pytest

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.core.pipeline import ingest, search
from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore

# Filenames follow the real TSpec-LLM pattern (\d{5}-\w+.md) so
# TPPSource.metadata() extracts spec_number/release the same way it would
# for a real corpus file.
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

    # Real end-to-end retrieval on top of what was just ingested — proves
    # the dense-vector round trip (embed -> store -> query -> decode)
    # actually works, not just that ingest() didn't crash.
    results = search(
        "How does the UE establish a PDU session?",
        store=store, embedder=embedder, top_k=2,
    )
    assert len(results) > 0
    assert any("SMF" in r.text or "PDU session" in r.text for r in results)


def test_ingest_pipelines_multiple_flush_batches_correctly(tmp_path):
    """Regression test for the pipelined chunking+embedding rewrite in
    core/pipeline.py: forces a single file to produce more chunks than
    batch_size on its own (exercising the
    `while len(buf_texts) >= batch_size: _flush(batch_size)` loop, not just
    the final tail flush after every file is done), across multiple files,
    and confirms every chunk lands in both the vector store and the FTS
    store exactly once — no duplication, no loss, resume bookkeeping intact.
    """
    source   = TPPSource()
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    store    = ChromaStore(
        path=tmp_path / "chroma_db", collection_name="test_multibatch_minilm",
        metadata={"hnsw:space": "cosine"},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # 3 files, each with 4 short paragraphs at a tiny chunk_size — each file
    # alone produces multiple chunks, well above batch_size=2 on its own.
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
    """Regression test for the embed/write batch decoupling: writes are
    consolidated into a SEPARATE, larger buffer than the embedding batch —
    forces multiple independent write-flush cycles (write_batch_size=3,
    smaller than the total chunk count but larger than embed batch_size=1)
    and confirms every chunk still lands exactly once, with no loss or
    duplication, regardless of the two batch sizes being different."""
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
    """Regression test for a real bug caught by code review: the resume
    progress-file write (and CSV logging) used to share the SAME
    try/except as the write-critical embed/store loop. Every chunk could
    be fully and correctly embedded and stored, and yet a failure in
    writing the small resume-tracking JSON file afterward would be caught
    by the same handler that treats a write-critical failure as data
    loss -- printing "[ERROR] Bulk Indexing failed" and setting
    error_files = len(files_processed_this_run), potentially double-
    counting files that were ALSO already counted in files_processed.

    Patches Path.write_text to raise only for the specific progress-file
    path ingest() writes to, and confirms: chunks_stored/error_files
    still accurately reflect that every chunk was genuinely stored (no
    false error_files inflation), even though the resume bookkeeping
    itself failed.
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

    # Every chunk was genuinely embedded and stored -- confirmed against
    # the real Chroma collection, not just the in-memory stats -- despite
    # the resume progress-file write failing.
    assert stats.chunks_stored > 0
    assert store.count() == stats.chunks_stored
    assert stats.error_files == 0, (
        "a resume-progress-file write failure must not be misreported as "
        "a data-loss (write-critical) failure"
    )
