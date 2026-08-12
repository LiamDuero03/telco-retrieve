"""Core pipeline — parallelized ingestion, search, generate."""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
import threading
import time
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from dotenv import load_dotenv

# OPTIMIZATION: Import tqdm for high-performance CLI telemetry
from tqdm import tqdm

from open_telco_rag.types import Chunk, IngestStats, SearchResult
from open_telco_rag.stores.fts import FTSStore

if TYPE_CHECKING:
    from open_telco_rag.types import Embedder, Source, VectorStore

load_dotenv()

# GUARD: Ensure the project root is on sys.path for Windows "spawn"-mode
# multiprocessing workers, which start a fresh interpreter and don't
# necessarily inherit the parent's sys.path. Must run BEFORE importing
# `cli` below — `cli` isn't part of the open_telco_rag package (no
# editable-install redirect guarantees it), so resolving it can depend on
# the CWD being on sys.path. This now runs once per process (module import
# time — main process included, where it's a harmless no-op) instead of
# once per _process_file_worker(...)/_process_file_worker_rawchunk(...)
# call, which is the whole point of moving these imports up here: Python
# already caches successful imports in sys.modules regardless of where the
# `import` statement lives, so this was never about "warming a cache" —
# it's about not paying the (tiny but real) per-call lookup/rebind cost on
# every single file processed, and about failing fast with one clear
# traceback if the environment is broken, instead of the same failure
# repeated — confusingly, once per file — as a per-file "error" status.
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

from open_telco_rag.core.chunking import chunk_document, ChunkingStrategy
from open_telco_rag.core.enrichment import enrich_chunks
from cli.main import _get_source


# ── Top-Level Worker for Multi-Core Processing ────────────────────────────────
def _process_file_worker(args: tuple) -> dict:
    """Worker function executed across multiple CPU process workers.

    All modules this needs (chunk_document, ChunkingStrategy, enrich_chunks,
    _get_source, time, Path, sys/os) are imported at module scope above —
    including, critically, the sys.path guard that runs before `cli` is
    imported, so a freshly spawned Windows worker process can resolve it.
    """
    (
        file_path, data_dir, source_type_str, use_marker,
        chunk_kwargs, min_length, model_key
    ) = args

    file_path = Path(file_path)
    data_dir = Path(data_dir)
    rel = str(file_path.relative_to(data_dir))
    file_size_kb = round(file_path.stat().st_size / 1024, 1)

    result = {
        "rel": rel,
        "file_path": file_path,
        "file_size_kb": file_size_kb,
        "status": "error",
        "chunks": [],
        "metadatas": [],
        "ids": [],
        "error": None,
        "chunking_time_ms": 0.0,
        "expansion_ratio": 1.0,
        "avg_len": 0,
    }

    try:
        # Use local core factory function instead of reaching into the CLI
        source = _get_source(source_type_str, use_marker=use_marker)
        # Load & Clean
        text = source.load(file_path)
        if not text or not text.strip():
            result["status"] = "skipped"
            result["error"] = "empty after load"
            return result

        text = source.clean(text)

        # Chunking
        start_chunk_time = time.perf_counter()
        raw_char_count = len(text)

        is_native_markdown = file_path.suffix.lower() in [".md", ".yaml", ".yml"]
        is_marker_output = use_marker and file_path.suffix.lower() in [".pdf", ".docx"]
        is_structured = is_native_markdown or is_marker_output

        active_kwargs = chunk_kwargs.copy()
        active_kwargs["source_type"] = source_type_str

        user_strategy_str = active_kwargs.get("strategy", "text_baseline")
        try:
            user_strategy = ChunkingStrategy(user_strategy_str)
        except ValueError:
            user_strategy = ChunkingStrategy.TEXT_BASELINE

        structure_required = {
            ChunkingStrategy.HIERARCHICAL_MARKDOWN,
            ChunkingStrategy.PARENT_CHILD
        }

        if not is_structured and user_strategy in structure_required:
            active_kwargs["strategy"] = ChunkingStrategy.TEXT_BASELINE
        else:
            active_kwargs["strategy"] = user_strategy

        active_kwargs["is_structured"] = is_structured
        enrichment_strategy = active_kwargs.pop("enrichment", "none")
        enrichment_model = active_kwargs.pop("enrichment_model", "google/qwen-2.5-coder-32b")
        # Popped out here, before chunk_document() ever sees active_kwargs —
        # this is enrichment's LLM client (e.g. for llm_metadata), a
        # deliberately different kwarg key from active_kwargs["llm_client"]
        # (chunking's own, e.g. for lumber_chunker). Some chunking strategies
        # (chunk_sliding_window_tokens in particular) have no **kwargs
        # catch-all, so an unrelated key like this reaching chunk_document()
        # would TypeError instead of being silently ignored.
        enrichment_llm_client = active_kwargs.pop("enrichment_llm_client", None)

        chunks = chunk_document(text=text, **active_kwargs)
        chunks = [c for c in chunks if len(c.text.strip()) >= min_length]

        if enrichment_strategy and enrichment_strategy != "none" and chunks:
            meta = source.metadata(file_path)
            chunks = enrich_chunks(
                chunks=chunks,
                strategy=enrichment_strategy,
                model_name=enrichment_model,
                llm_client=enrichment_llm_client,
                source_document=meta.get("source", file_path.name),
                **meta
            )

        chunk_duration_sec = time.perf_counter() - start_chunk_time
        fused_char_count = sum(len(c.text) for c in chunks) if chunks else 0

        if not chunks:
            result["status"] = "skipped"
            result["error"] = "no valid chunks"
            return result

        meta = source.metadata(file_path)
        extracted_texts = []
        extracted_ids = []
        extracted_metas = []

        for j, c in enumerate(chunks):
            chunk_id = f"{rel}::{model_key}::{j}"
            base_meta = {
                **meta,
                "heading": c.heading or "",
                "chunk_index": j,
                "source_type": c.source_type
            }
            if hasattr(c, 'metadata') and c.metadata:
                base_meta.update(c.metadata)

            extracted_texts.append(c.text)
            extracted_ids.append(chunk_id)
            extracted_metas.append(base_meta)

        chunk_lens = [len(c.text) for c in chunks]
        
        result["status"] = "success"
        result["chunks"] = extracted_texts
        result["ids"] = extracted_ids
        result["metadatas"] = extracted_metas
        result["chunking_time_ms"] = chunk_duration_sec * 1000
        result["expansion_ratio"] = fused_char_count / max(1, raw_char_count)
        result["avg_len"] = sum(chunk_lens) // len(chunk_lens)
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


def _locked_add(lock: threading.Lock, fn, *args: Any) -> None:
    """Runs a store.add()/fts_store.add() call under a lock scoped to that
    specific store object.

    Both ingest() and ingest_multi_enrichment() dispatch writes to a
    background ThreadPoolExecutor(max_workers=8) so multiple stores (or,
    within ingest_multi_enrichment, multiple enrichment variants' separate
    stores) can be written to concurrently. But repeated flushes for the
    SAME store aren't guaranteed to be serialized by the pool itself —
    submit() doesn't block, so if chunks accumulate faster than one flush
    completes, two writes to the SAME ChromaDB collection (or the SAME
    FTS SQLite file) can end up in flight at once. Neither ChromaStore's
    local SQLite-backed collection nor FTSStore's own SQLite connection is
    guaranteed safe against that — concurrent writers on the same
    SQLite-backed store are a well-known way to get an internal error
    ("database is locked" or similar). This lock only serializes writes
    to the SAME store; writes to genuinely different stores (different
    enrichment variants, or a store vs. its own fts_store) still overlap
    freely, so this doesn't collapse Phase 2 back down to fully sequential.
    """
    with lock:
        fn(*args)


def _raise_first_write_failure(db_futures: list) -> None:
    """wait(db_futures) only waits for every future to reach a done state —
    it does NOT check whether any of them raised. A store.add() exception
    inside a background writer thread was previously discarded completely
    silently: the run would proceed straight to declaring success and
    writing _ingest_complete.json's completion marker as if every chunk
    had actually been persisted, even if most of the writes had silently
    failed. Confirmed in production: entire ingestion runs where
    _ingest_complete.json claimed hundreds of thousands of chunks stored,
    while the real ChromaDB collection held a few hundred. Call this right
    after wait() so a real write failure aborts the run loudly (caught by
    the existing try/except around Phase 2, which reports the error and
    skips the completion-marker write) instead of vanishing.
    """
    for future in db_futures:
        exc = future.exception()
        if exc is not None:
            raise RuntimeError(f"Database write failed during ingest: {exc}") from exc


# ── Ingest Orchestrator ───────────────────────────────────────────────────────
def ingest(
    source:      Source,
    store:      VectorStore,
    embedder:   Embedder,
    data_dir:   str | Path,
    min_length: int  = 100,
    batch_size: int  = 64,
    write_batch_size: int | None = None,
    resume:     bool = True,
    log_dir:    str | Path = "logs",
    **chunk_kwargs: Any,
) -> IngestStats:
    """Scan files, clean, chunk in parallel across CPU cores, embed, and store.

    batch_size controls how many texts get embedded together in one
    embedder.embed() call (the GPU-utilization-relevant batch). write_batch_size
    is separate and controls how many already-embedded rows accumulate before
    a store.add()/fts_store.add() call actually fires — defaults to
    8 * batch_size, so DB writes happen in fewer, larger transactions than
    embedding batches, without buffering the *entire* run's vectors in RAM
    (which doesn't scale to a multi-million-chunk corpus — the earlier
    all-at-once design this pipeline moved away from). Pass write_batch_size
    explicitly to tune the memory/transaction-count tradeoff — note the
    actual store.add()/fts_store.add() calls dispatched from a flush are
    still capped at `store`'s real max batch size regardless of what's
    passed here (see max_write_batch below), so an oversized write_batch_size
    only affects buffering, never risks a Chroma batch-size crash.
    """
    write_batch_size = write_batch_size or (batch_size * 8)

    data_dir = Path(data_dir)
    log_dir  = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_class_name = source.__class__.__name__.lower().replace("source", "")
    source_name = "3gpp" if raw_class_name == "tpp" else raw_class_name
    log_file    = log_dir / f"ingest_{source_name}_{embedder.model_key}_{timestamp}.csv"
    stats       = IngestStats()

    use_marker = getattr(source, '_use_marker', False)

    # Resume tracking
    progress_file = log_dir / f".progress_{embedder.model_key}.json"
    completed: set[str] = set()
    if resume and progress_file.exists():
        completed = set(json.loads(progress_file.read_text()))

    extensions = source.supported_extensions()
    files = [
        f for ext in extensions
        for f in sorted(data_dir.rglob(f"*{ext}"))
    ]

    _init_log(log_file)
    print(f"\nFound {len(files)} files | Model: {embedder.model_key}")

    store_path = getattr(store, "path", "chroma_db")
    store_col  = getattr(store, "collection_name", "documents")
    fts_store  = FTSStore(path=store_path, collection_name=store_col)

    if not resume:
        fts_store.clear()

    max_workers = max(1, (os.cpu_count() or 2) - 1)
    
    tasks = []
    for file_path in files:
        rel = str(file_path.relative_to(data_dir))
        if rel in completed:
            stats.skipped_files += 1
            continue

        tasks.append((
            file_path, data_dir, source_name, use_marker,
            chunk_kwargs, min_length, embedder.model_key
        ))

    if not tasks:
        return stats

    # ── PHASE 1: PARSING ──────────────────────────────────────────────────────
    # Every file is chunked before anything is embedded — one clean progress
    # bar for the whole parsing pass. This trades away overlapping CPU
    # chunking with GPU embedding (a prior version of this pipeline did that)
    # in favor of a simpler, easier-to-read two-phase run: parsing is CPU-only
    # and comparatively fast, and the actual bottlenecks are Phase 2 —
    # embedding, and writing to ChromaDB/the BM25 index — so that's where the
    # optimization effort below is spent. Trade-off worth knowing: every
    # chunk's text+metadata for the whole corpus is held in RAM for the
    # duration of Phase 1 (bounded by corpus size, not batch size) — fine for
    # a single source/model run, but worth watching on a very large corpus.
    print(f"\nPhase 1: Parsing {len(tasks)} files across {max_workers} CPU workers...")

    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict] = []
    files_to_log = []
    files_processed_this_run = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor, \
         tqdm(total=len(tasks), desc="Parsing Files", unit="file", colour="CYAN") as file_bar:

        futures = {executor.submit(_process_file_worker, task): task[0] for task in tasks}

        for future in as_completed(futures):
            res = future.result()
            f_path = res["file_path"]
            file_bar.update(1)

            if res["status"] == "success":
                all_texts.extend(res["chunks"])
                all_ids.extend(res["ids"])
                all_metas.extend(res["metadatas"])

                stats.total_chunking_time_sec += (res["chunking_time_ms"] / 1000)
                files_to_log.append((
                    f_path, res["file_size_kb"], len(res["chunks"]),
                    res["avg_len"], res["chunking_time_ms"], res["expansion_ratio"]
                ))
                files_processed_this_run.append(res["rel"])

            elif res["status"] == "skipped":
                stats.skipped_files += 1
                tqdm.write(f"  ⚠️ [SKIPPED] {f_path.name}: {res['error']}")
                _log_row(log_file, f_path, res["file_size_kb"], "skipped",
                         error=res["error"], model=embedder.model_key)
            else:
                stats.error_files += 1
                tqdm.write(f"  ❌ [ERROR] {f_path.name}: {res['error']}")
                _log_row(log_file, f_path, res["file_size_kb"], "error",
                         error=res["error"], model=embedder.model_key)

    total_chunks = len(all_texts)
    if total_chunks == 0:
        return stats

    # ── PHASE 2: EMBEDDING + STORAGE ──────────────────────────────────────────
    # This is the actual bottleneck, so: (1) embedding runs in GPU-sized
    # batches (batch_size); (2) DB writes are consolidated into far fewer,
    # larger transactions (write_batch_size) instead of one write per embed
    # batch — less per-transaction overhead in both ChromaDB and SQLite;
    # (3) the DB-writer thread pool is bigger here than the old combined-phase
    # version could safely use, since Phase 1's CPU workers have already
    # released every core by the time this runs — no contention to worry
    # about; (4) `store` is expected to carry chromadb's own bulk-load HNSW
    # tuning (hnsw:batch_size / hnsw:sync_threshold) in its collection
    # metadata — see _get_store() in cli/main.py — which reduces how often
    # Chroma syncs its HNSW graph to disk during the upsert calls below.
    print(f"\nPhase 2: Embedding & storing {total_chunks:,} chunks "
          f"(embed batch={batch_size}, write batch={write_batch_size})...")

    write_ids: list[str] = []
    write_embeds: list[list[float]] = []
    write_texts: list[str] = []
    write_metas: list[dict] = []
    db_futures = []
    # See _locked_add()'s docstring — repeated flushes can otherwise submit
    # two concurrent writes to the SAME store, which isn't safe.
    store_lock = threading.Lock()
    fts_lock = threading.Lock()

    # write_batch_size defaults to 8x batch_size and can exceed Chroma's
    # actual hard limit on a single add()/upsert() call (queried live via
    # ChromaStore.max_batch_size, since it derives from sqlite3's
    # SQLITE_MAX_VARIABLE_NUMBER and isn't fixed across environments).
    # Capping sub-batches dispatched to store.add() below at this value
    # avoids the crash without changing write_batch_size's role of
    # controlling how much gets buffered before a flush fires.
    max_write_batch = min(write_batch_size, getattr(store, "max_batch_size", write_batch_size) or write_batch_size)

    try:
        with ThreadPoolExecutor(max_workers=8) as db_writer, \
             tqdm(total=total_chunks, desc="Embedding & Storing", unit="chunk", colour="GREEN") as chunk_bar:

            def _flush_writes(n: int) -> None:
                """Dispatches exactly the first n already-embedded rows to
                background threads for the actual store.add()/
                fts_store.add() calls, split into sub-batches no larger
                than max_write_batch so a single call never exceeds
                Chroma's real hard limit — see max_write_batch above."""
                ids    = write_ids[:n]
                embeds = write_embeds[:n]
                texts  = write_texts[:n]
                metas  = write_metas[:n]
                del write_ids[:n]
                del write_embeds[:n]
                del write_texts[:n]
                del write_metas[:n]

                for start in range(0, n, max_write_batch):
                    end = min(start + max_write_batch, n)
                    db_futures.append(db_writer.submit(
                        _locked_add, store_lock, store.add,
                        ids[start:end], embeds[start:end], texts[start:end], metas[start:end],
                    ))
                    db_futures.append(db_writer.submit(
                        _locked_add, fts_lock, fts_store.add, ids[start:end], texts[start:end],
                    ))

            for start in range(0, total_chunks, batch_size):
                end = min(start + batch_size, total_chunks)
                batch_texts = all_texts[start:end]
                batch_ids   = all_ids[start:end]
                batch_metas = all_metas[start:end]

                embeds = embedder.embed(batch_texts)

                write_ids.extend(batch_ids)
                write_embeds.extend(embeds)
                write_texts.extend(batch_texts)
                write_metas.extend(batch_metas)

                stats.chunks_stored += len(batch_texts)
                chunk_bar.update(len(batch_texts))

                while len(write_ids) >= write_batch_size:
                    _flush_writes(write_batch_size)

            # Final partial write batch
            if write_ids:
                _flush_writes(len(write_ids))

            # Wait for all background database tasks to complete before exiting
            tqdm.write("  Flushing database writes to disk...")
            wait(db_futures)
            _raise_first_write_failure(db_futures)

    except Exception as e:
        print(f"\n[ERROR] Bulk Indexing failed: {e}")
        stats.error_files += len(files_processed_this_run)
        return stats

    # Only reached if every chunk above is confirmed actually stored.
    # Deliberately a SEPARATE try/except from the write-critical section
    # above: a failure in purely cosmetic bookkeeping (a CSV log row, the
    # resume progress-file write) after every chunk already landed safely
    # in the store must never be conflated with a data-loss failure, or
    # misreport a fully and correctly ingested file as "[ERROR] Bulk
    # Indexing failed".
    try:
        for log_item in files_to_log:
            f_path, f_size, f_chunks, f_avg, f_time, f_ratio = log_item
            _log_row(log_file, f_path, f_size, "success", chunks=f_chunks, model=embedder.model_key,
                     avg_len=f_avg, chunking_time_ms=f_time, expansion_ratio=f_ratio)
            stats.files_processed += 1

        if resume:
            completed.update(files_processed_this_run)
            progress_file.write_text(json.dumps(sorted(completed)))
    except Exception as e:
        print(f"\n[WARN] Post-write bookkeeping (CSV log / resume progress file) failed: "
              f"{e} — every chunk was still successfully stored; only the log/resume "
              f"file may be incomplete.")

    return stats


# ── Top-Level Worker: chunk-only, no enrichment ───────────────────────────────
def _process_file_worker_rawchunk(args: tuple) -> dict:
    """Chunk a single file WITHOUT applying enrichment.

    Used by ingest_multi_enrichment() so that a file's chunk boundaries —
    driven only by (strategy, chunk_size, overlap) — are computed exactly
    once and then reused for however many enrichment variants share that
    chunking config, instead of re-chunking identical text once per
    enrichment value.

    Modules this needs (chunk_document, ChunkingStrategy, _get_source, time,
    Path, sys/os) are imported at module scope — see _process_file_worker's
    docstring for why, including the sys.path guard ordering.
    """
    (file_path, data_dir, source_type_str, use_marker, chunk_kwargs, min_length) = args

    file_path = Path(file_path)
    data_dir = Path(data_dir)
    rel = str(file_path.relative_to(data_dir))
    file_size_kb = round(file_path.stat().st_size / 1024, 1)

    result = {
        "rel": rel,
        "file_path": file_path,
        "file_size_kb": file_size_kb,
        "status": "error",
        "chunks": [],
        "meta": {},
        "error": None,
        "chunking_time_ms": 0.0,
        "expansion_ratio": 1.0,
    }

    try:
        source = _get_source(source_type_str, use_marker=use_marker)
        text = source.load(file_path)
        if not text or not text.strip():
            result["status"] = "skipped"
            result["error"] = "empty after load"
            return result

        text = source.clean(text)

        start_chunk_time = time.perf_counter()
        raw_char_count = len(text)

        is_native_markdown = file_path.suffix.lower() in [".md", ".yaml", ".yml"]
        is_marker_output = use_marker and file_path.suffix.lower() in [".pdf", ".docx"]
        is_structured = is_native_markdown or is_marker_output

        active_kwargs = chunk_kwargs.copy()
        active_kwargs["source_type"] = source_type_str

        user_strategy_str = active_kwargs.get("strategy", "text_baseline")
        try:
            user_strategy = ChunkingStrategy(user_strategy_str)
        except ValueError:
            user_strategy = ChunkingStrategy.TEXT_BASELINE

        structure_required = {
            ChunkingStrategy.HIERARCHICAL_MARKDOWN,
            ChunkingStrategy.PARENT_CHILD,
        }

        if not is_structured and user_strategy in structure_required:
            active_kwargs["strategy"] = ChunkingStrategy.TEXT_BASELINE
        else:
            active_kwargs["strategy"] = user_strategy

        active_kwargs["is_structured"] = is_structured

        # Enrichment kwargs (if present) are irrelevant to chunking itself.
        active_kwargs.pop("enrichment", None)
        active_kwargs.pop("enrichment_model", None)

        chunks = chunk_document(text=text, **active_kwargs)
        chunks = [c for c in chunks if len(c.text.strip()) >= min_length]

        chunk_duration_sec = time.perf_counter() - start_chunk_time
        fused_char_count = sum(len(c.text) for c in chunks) if chunks else 0

        if not chunks:
            result["status"] = "skipped"
            result["error"] = "no valid chunks"
            return result

        result["status"] = "success"
        result["chunks"] = chunks
        result["meta"] = source.metadata(file_path)
        result["chunking_time_ms"] = chunk_duration_sec * 1000
        result["expansion_ratio"] = fused_char_count / max(1, raw_char_count)
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


# ── Completion marker — lets callers tell a finished run apart from one ───────
# ── that was interrupted mid-ingest (a directory existing with files in it
# ── proves nothing, since VectorStore construction alone creates bootstrap
# ── files before any chunk is stored). ────────────────────────────────────────
_COMPLETION_MARKER_NAME = "_ingest_complete.json"


def _write_completion_marker(store: VectorStore, files_total: int, stats: IngestStats) -> None:
    """Writes a small marker file into a store's own directory, recording
    that ingest_multi_enrichment() reached the end of its run for this
    variant without a fatal exception. Only ever called from a code path
    that's guaranteed to have completed successfully — never on the
    exception path, and never partially.

    Callers (e.g. master_grid.py) should check for this marker before
    treating an existing, non-empty run directory as "already done" — a
    process-level crash (OOM, node preemption, power loss) leaves a
    non-empty but incomplete directory behind too, which "does the
    directory have any files in it" can't tell apart from a genuinely
    finished run.
    """
    store_path = getattr(store, "path", None)
    if not store_path:
        return
    marker_path = Path(store_path) / _COMPLETION_MARKER_NAME
    marker_path.write_text(json.dumps({
        "complete":         True,
        "files_total":      files_total,
        "files_processed":  stats.files_processed,
        "skipped_files":    stats.skipped_files,
        "error_files":      stats.error_files,
        "chunks_stored":    stats.chunks_stored,
        "timestamp":        datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")


# ── Ingest Orchestrator: chunk once, fan out across enrichment variants ───────
def ingest_multi_enrichment(
    source: Source,
    stores: dict[str, VectorStore],
    embedder: Embedder,
    data_dir: str | Path,
    enrichment_strategies: list[str],
    enrichment_model: str = "google/qwen-2.5-coder-32b",
    enrichment_kwargs: dict[str, Any] | None = None,
    min_length: int = 100,
    batch_size: int = 64,
    write_batch_size: int | None = None,
    log_dir: str | Path = "logs",
    on_progress: Callable[[], None] | None = None,
    **chunk_kwargs: Any,
) -> dict[str, IngestStats]:
    """Chunk every file in data_dir exactly ONCE, then fan the resulting
    chunks out into one independently-enriched / embedded / stored variant
    per entry in `enrichment_strategies`.

    `stores` must contain exactly one VectorStore per enrichment strategy
    being processed (keyed by enrichment strategy name).

    Rationale: chunk_document() has no awareness of enrichment — enrichment
    is a post-chunking transform applied to the already-produced Chunk list.
    So calling plain ingest() once per enrichment value (as a naive grid
    sweep would) re-chunks byte-identical text N times for no reason. This
    collapses that into a single CPU-bound chunking pass per (strategy,
    chunk_size, overlap) combo, while still computing separate embeddings
    per enrichment variant (since enrichment can change chunk text, those
    genuinely can't be shared).

    Each variant's chunk list is deep-copied before enrichment is applied,
    so this is safe regardless of whether enrich_chunks() mutates its input
    Chunk objects in place or returns new ones.

    enrichment_kwargs: extra keyword arguments forwarded to enrich_chunks()
        for every variant — e.g. {"glossary_dict": {...}} or
        {"glossary_path": "..."} to override acronym_expansion's default
        bundled glossary. Without this, such overrides had no way to reach
        enrich_chunks() at all: anything passed via **chunk_kwargs only
        ever reaches chunk_document() (which silently ignores kwargs it
        doesn't recognize via its own **kwargs catch-all), never enrichment.

    on_progress: optional zero-argument callback invoked once per embed
        batch (i.e. after every chunk_bar.update() during Phase 2 — see
        _embed_and_write_batch below), purely as a liveness signal. Not used
        by this function itself; exists so an external caller (e.g.
        master_grid.py's per-model watchdog process) can detect "no
        progress for N minutes" and distinguish a genuinely hung run from
        one that's just slow — without this, a caller waiting on this
        function's return value has no visibility into whether it's still
        working until it either finishes or crashes. Exceptions raised by
        the callback itself are not caught — keep it cheap and
        non-blocking (e.g. a non-blocking queue.put()).

    On successful completion (no fatal exception), each variant's store
    directory gets a `_ingest_complete.json` marker — see
    _write_completion_marker() for why this matters for safe resuming.

    batch_size controls both embedding AND write granularity: each
    embed_batch of `batch_size` chunks is written to its variant's store +
    FTS index immediately afterward, inside the same loop — matching
    master_grid_llm_completion.py's proven pattern (store.add() called
    once per embed batch, no separate larger accumulation buffer). Each
    write is additionally split into sub-batches no larger than that
    variant's store's real max batch size (queried live from the Chroma
    client — see ChromaStore.max_batch_size — not hardcoded), so a large
    batch_size can never exceed Chroma's actual hard limit on a single
    add()/upsert() call. chunks_stored is only incremented after a
    sub-batch's store.add() call actually returns without raising, so a
    partial failure produces an accurate partial count instead of the
    full pre-failure total, and propagates as a failure immediately
    rather than being silently swallowed.

    write_batch_size is accepted for backward-compatible call sites but no
    longer has any effect — see the loud warning printed if it's passed.
    """
    from open_telco_rag.core.enrichment import enrich_chunks

    enrichment_kwargs = enrichment_kwargs or {}
    if write_batch_size is not None:
        # write_batch_size used to control a separate, larger buffer
        # (defaulting to 8x batch_size) flushed to store.add()/
        # fts_store.add() in one big call, which could exceed Chroma's
        # real hard limit and crash after the chunk_bar had already
        # reported the run 100% done. Writes are now sized to batch_size
        # (capped at the store's real max batch size) inside the same
        # loop as embedding instead, so write_batch_size no longer has a
        # role here — a caller that still passes it gets a loud warning.
        print(f"  ⚠️ ingest_multi_enrichment(): write_batch_size={write_batch_size} was "
              f"passed but is no longer used — writes are now sized to batch_size "
              f"(capped at each store's real max batch size). Remove this argument.")

    data_dir = Path(data_dir)
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    missing = set(enrichment_strategies) - set(stores.keys())
    if missing:
        raise ValueError(f"No store provided for enrichment strategies: {missing}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_class_name = source.__class__.__name__.lower().replace("source", "")
    source_name = "3gpp" if raw_class_name == "tpp" else raw_class_name
    strategy_label = str(chunk_kwargs.get("strategy", "unknown"))

    use_marker = getattr(source, "_use_marker", False)
    extensions = source.supported_extensions()
    files = [f for ext in extensions for f in sorted(data_dir.rglob(f"*{ext}"))]

    if not files:
        empty_stats = {ev: IngestStats() for ev in enrichment_strategies}
        for ev in enrichment_strategies:
            _write_completion_marker(stores[ev], 0, empty_stats[ev])
        return empty_stats

    max_workers = max(1, (os.cpu_count() or 2) - 1)
    tasks = [
        (file_path, data_dir, source_name, use_marker, chunk_kwargs, min_length)
        for file_path in files
    ]

    # Per-variant state: separate store, FTS index, buffers, stats, log file.
    variants: dict[str, dict] = {}
    for ev in enrichment_strategies:
        st = stores[ev]
        store_path = getattr(st, "path", "chroma_db")
        store_col = getattr(st, "collection_name", "documents")
        variants[ev] = {
            "store": st,
            "fts_store": FTSStore(path=store_path, collection_name=store_col),
            "buf_texts": [],
            "buf_ids": [],
            "buf_metas": [],
            "stats": IngestStats(),
            "files_to_log": [],
            "log_file": log_dir / (
                f"ingest_{source_name}_{embedder.model_key}_"
                f"{strategy_label}_{ev}_{timestamp}.csv"
            ),
            # Queried live from the store rather than hardcoded — see
            # ChromaStore.max_batch_size. Falls back to a conservative
            # 5000 for any VectorStore implementation that doesn't expose
            # it, safely under Chroma's own observed default (5461).
            "max_batch_size": getattr(st, "max_batch_size", None) or 5000,
        }
        _init_log(variants[ev]["log_file"])

    # ── PHASE 1: PARSING (once) ────────────────────────────────────────────────
    # One clean progress bar for the whole parsing pass, same as ingest() —
    # see that function's docstring for the two-phase-vs-pipelined trade-off.
    print(f"\nPhase 1: Parsing {len(tasks)} files once across {max_workers} CPU "
          f"workers (fanning out to {len(enrichment_strategies)} enrichment "
          f"variant(s) in Phase 2: {', '.join(enrichment_strategies)})...")

    parsed_results = []
    total_raw_chunks = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor, \
         tqdm(total=len(tasks), desc="Parsing Files", unit="file", colour="CYAN") as file_bar:

        futures = {
            executor.submit(_process_file_worker_rawchunk, task): task[0]
            for task in tasks
        }

        for future in as_completed(futures):
            res = future.result()
            f_path = res["file_path"]
            file_bar.update(1)

            if res["status"] == "success":
                parsed_results.append(res)
                total_raw_chunks += len(res["chunks"])
            elif res["status"] == "skipped":
                tqdm.write(f"  ⚠️ [SKIPPED] {f_path.name}: {res['error']}")
                for ev in enrichment_strategies:
                    variants[ev]["stats"].skipped_files += 1
                    _log_row(variants[ev]["log_file"], f_path, res["file_size_kb"],
                              "skipped", error=res["error"], model=embedder.model_key)
            else:
                tqdm.write(f"  ❌ [ERROR] {f_path.name}: {res['error']}")
                for ev in enrichment_strategies:
                    variants[ev]["stats"].error_files += 1
                    _log_row(variants[ev]["log_file"], f_path, res["file_size_kb"],
                              "error", error=res["error"], model=embedder.model_key)

    if total_raw_chunks == 0:
        for ev in enrichment_strategies:
            _write_completion_marker(variants[ev]["store"], len(files), variants[ev]["stats"])
        return {ev: variants[ev]["stats"] for ev in enrichment_strategies}

    # ── PHASE 2: ENRICH + EMBED + STORE, per variant ──────────────────────────
    # Enrichment strategies only ever transform a chunk's text/metadata, never
    # add or remove chunks — so every variant ends up with exactly
    # total_raw_chunks rows, and the combined total across all variants is
    # known upfront (unlike the old growing-total bar).
    #
    # Writes happen synchronously, directly inside this loop, one
    # store.add()/fts_store.add() call per embed batch — matching
    # master_grid_llm_completion.py's proven working pattern — instead of
    # the old design's separate, larger write_batch_size accumulation
    # buffer flushed via a background ThreadPoolExecutor. That older
    # design let a single flush exceed Chroma's real hard limit on one
    # add()/upsert() call (confirmed in production: write_batch_size=32768
    # at embed batch=4096, crashing with "Batch size of 32768 is greater
    # than max batch size of 5461" only AFTER the progress bar had already
    # reported the run 100% done), and incremented chunks_stored at embed
    # time rather than confirmed-write time, so a failed flush left
    # chunks_stored reporting the full pre-failure total instead of an
    # accurate partial count. See _embed_and_write_batch below.
    total_variant_chunks = total_raw_chunks * len(enrichment_strategies)
    print(f"\nPhase 2: Enriching, embedding & storing {total_raw_chunks:,} chunks "
          f"x {len(enrichment_strategies)} variant(s) = {total_variant_chunks:,} "
          f"total (embed batch={batch_size}, capped per-store at Chroma's real "
          f"max batch size)...")

    def _embed_and_write_batch(ev: str, n: int) -> None:
        """Embeds exactly the first n buffered chunks for this variant and
        writes them to its store + FTS index immediately, split into
        sub-batches no larger than this variant's store's real max batch
        size (variants[ev]["max_batch_size"], queried live from the store
        — see ChromaStore.max_batch_size). chunks_stored is only
        incremented once a sub-batch's store.add() call actually returns
        without raising — a write failure partway through this variant's
        buffer propagates immediately (caught by the try/except around
        Phase 2) with an accurate partial count already recorded, instead
        of the whole run continuing to embed and count chunks that were
        never actually stored.
        """
        v = variants[ev]
        texts = v["buf_texts"][:n]
        ids = v["buf_ids"][:n]
        metas = v["buf_metas"][:n]
        del v["buf_texts"][:n]
        del v["buf_ids"][:n]
        del v["buf_metas"][:n]

        embeds = embedder.embed(texts)

        max_write = v["max_batch_size"]
        for start in range(0, len(ids), max_write):
            end = min(start + max_write, len(ids))
            v["store"].add(ids[start:end], embeds[start:end], texts[start:end], metas[start:end])
            v["fts_store"].add(ids[start:end], texts[start:end])
            v["stats"].chunks_stored += (end - start)

        chunk_bar.update(len(texts))
        if on_progress is not None:
            on_progress()

    try:
        with tqdm(total=total_variant_chunks, desc="Embedding & Storing",
                  unit="chunk", colour="GREEN") as chunk_bar:

            for res in parsed_results:
                f_path = res["file_path"]

                for ev in enrichment_strategies:
                    v = variants[ev]

                    if ev != "none":
                        # enrichment_kwargs merged in last so it can
                        # override a res["meta"] key of the same name if
                        # a caller ever deliberately does that, rather
                        # than the two dicts colliding into a TypeError.
                        enrich_call_kwargs = {**res["meta"], **enrichment_kwargs}
                        variant_chunks = enrich_chunks(
                            chunks=copy.deepcopy(res["chunks"]),
                            strategy=ev,
                            model_name=enrichment_model,
                            source_document=res["meta"].get("source", f_path.name),
                            **enrich_call_kwargs,
                        )
                    else:
                        # Deep-copied even for "none" so nothing shared
                        # with other variants can ever be mutated across
                        # them, regardless of enrich_chunks' internals.
                        variant_chunks = copy.deepcopy(res["chunks"])

                    texts, ids, metas = [], [], []
                    for j, c in enumerate(variant_chunks):
                        chunk_id = f"{res['rel']}::{embedder.model_key}::{j}"
                        base_meta = {
                            **res["meta"],
                            "heading": c.heading or "",
                            "chunk_index": j,
                            "source_type": c.source_type,
                        }
                        if hasattr(c, "metadata") and c.metadata:
                            base_meta.update(c.metadata)
                        texts.append(c.text)
                        ids.append(chunk_id)
                        metas.append(base_meta)

                    v["buf_texts"].extend(texts)
                    v["buf_ids"].extend(ids)
                    v["buf_metas"].extend(metas)

                    v["stats"].total_chunking_time_sec += (res["chunking_time_ms"] / 1000)
                    chunk_lens = [len(t) for t in texts]
                    avg_len = sum(chunk_lens) // len(chunk_lens) if chunk_lens else 0
                    v["files_to_log"].append((
                        f_path, res["file_size_kb"], len(texts), avg_len,
                        res["chunking_time_ms"], res["expansion_ratio"]
                    ))

                    while len(v["buf_texts"]) >= batch_size:
                        _embed_and_write_batch(ev, batch_size)

            # Final partial batches — whatever's left once every file is enriched.
            for ev in enrichment_strategies:
                v = variants[ev]
                if v["buf_texts"]:
                    _embed_and_write_batch(ev, len(v["buf_texts"]))

    except Exception as e:
        print(f"\n[ERROR] Bulk multi-enrichment indexing failed: {e}")
        for ev in enrichment_strategies:
            variants[ev]["stats"].error_files += len(files)
        return {ev: variants[ev]["stats"] for ev in enrichment_strategies}

    # Only reached if every chunk, for every variant, is confirmed
    # actually stored. Deliberately a SEPARATE try/except from the
    # embed-and-write loop above: a bookkeeping failure for just one
    # variant (a CSV log row, or _write_completion_marker()) after every
    # chunk for every variant already landed safely must never set
    # error_files for every variant, or misreport a fully successful
    # multi-variant ingest as a total failure.
    for ev in enrichment_strategies:
        v = variants[ev]
        try:
            for f_path, f_size, f_chunks, f_avg, f_time, f_ratio in v["files_to_log"]:
                _log_row(v["log_file"], f_path, f_size, "success", chunks=f_chunks,
                          model=embedder.model_key, avg_len=f_avg,
                          chunking_time_ms=f_time, expansion_ratio=f_ratio)
                v["stats"].files_processed += 1

            # See _write_completion_marker()'s docstring for why this
            # matters for callers deciding whether it's safe to skip a
            # re-run.
            _write_completion_marker(v["store"], len(files), v["stats"])
        except Exception as e:
            print(f"\n[WARN] Post-write bookkeeping (CSV log / completion marker) failed "
                  f"for variant '{ev}': {e} — every chunk for this variant was still "
                  f"successfully stored; only its log file/completion marker may be "
                  f"incomplete, so a future run may needlessly re-ingest it.")

    return {ev: variants[ev]["stats"] for ev in enrichment_strategies}


# ── Search ────────────────────────────────────────────────────────────────────
def search(
    query:           str,
    store:           VectorStore,
    embedder:        Embedder,
    top_k:           int   = 8,
    threshold:       float = 1.0,
    reranker               = None,
    rerank_top_k:    int   = 5,
    search_method:   str   = "dense",
    resolve_parents: bool  = False,  
) -> list[SearchResult]:
    """Embed a query and retrieve nearest neighbours."""
    from open_telco_rag.core.retrieval import execute_retrieval
    
    return execute_retrieval(
        query=query,
        store=store,
        embedder=embedder,
        top_k=top_k,
        threshold=threshold,
        reranker=reranker,
        rerank_top_k=rerank_top_k,
        search_method=search_method, 
        resolve_parents=resolve_parents, 
    )

# ── Generate ──────────────────────────────────────────────────────────────────

def generate(
    question:    str,
    results:     list[SearchResult],
    generator    = None,
    source_type: str   = "telecom",
    # Legacy kwargs — ignored if generator is provided
    claude_model: str   = "claude-sonnet-4-6",
    max_tokens:   int   = 1000,
    temperature:  float = 0.0,
    max_retries:  int   = 3,
    response_schema: dict | None = None,
) -> dict:
    """Generate a grounded answer from retrieved chunks.

    response_schema, when given, is a JSON Schema object passed straight
    through to generator.generate() — see evaluation/evaluate.py's
    _mc_response_schema()/_tf_response_schema() for what builds these
    (structured, schema-constrained mc/tf verdicts instead of free-text
    parsing) and generators/local.py's generate() for how it's actually
    sent to the server (response_format: json_schema). None (the default)
    means "no structural constraint" — every existing caller (open-ended
    generation, and any generator that doesn't support/need this) is
    unaffected.
    """
    if generator is None:
        from open_telco_rag.generators.claude import ClaudeGenerator
        generator = ClaudeGenerator(
            model       = claude_model,
            max_tokens  = max_tokens,
            temperature = temperature,
            max_retries = max_retries,
        )
    return generator.generate(question, results, source_type=source_type, response_schema=response_schema)


# ── CSV logging ───────────────────────────────────────────────────────────────

_CSV_HEADERS = [
    "timestamp", "file_name", "file_size_kb", "status",
    "chunks", "avg_len", "embedding_model", "error_message", 
    "chunking_time_ms", "expansion_ratio"
]

def _init_log(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_CSV_HEADERS).writeheader()

def _log_row(
    path: Path, file_path: Path, size_kb: float, status: str,
    chunks: int = 0, avg_len: int = 0, model: str = "", error: str = "",
    chunking_time_ms: float = 0.0, expansion_ratio: float = 1.0
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_CSV_HEADERS).writerow({
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_name":       file_path.name,
            "file_size_kb":    size_kb,
            "status":          status,
            "chunks":          chunks,
            "avg_len":         avg_len,
            "embedding_model": model,
            "error_message":   error,
            "chunking_time_ms": round(chunking_time_ms, 2), 
            "expansion_ratio":  round(expansion_ratio, 3),
        })