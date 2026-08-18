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

from tqdm import tqdm

from open_telco_rag.types import Chunk, IngestStats, SearchResult
from open_telco_rag.stores.fts import FTSStore

if TYPE_CHECKING:
    from open_telco_rag.types import Embedder, Source, VectorStore

load_dotenv()

# Windows spawn workers don't inherit sys.path; must run before importing `cli`.
# Runs once at import so a broken env fails fast, not per-file.
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

from open_telco_rag.core.chunking import chunk_document, ChunkingStrategy
from open_telco_rag.core.enrichment import enrich_chunks
from cli.main import _get_source


# ── Top-Level Worker for Multi-Core Processing ────────────────────────────────
def _process_file_worker(args: tuple) -> dict:
    """Runs in a CPU process pool; needed modules are imported at module
    scope (incl. the sys.path guard) so spawned Windows workers can
    resolve `cli`."""
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
        # Reuses CLI's source factory instead of duplicating it
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
            ChunkingStrategy.PARENT_CHILD
        }

        if not is_structured and user_strategy in structure_required:
            active_kwargs["strategy"] = ChunkingStrategy.TEXT_BASELINE
        else:
            active_kwargs["strategy"] = user_strategy

        active_kwargs["is_structured"] = is_structured
        enrichment_strategy = active_kwargs.pop("enrichment", "none")
        enrichment_model = active_kwargs.pop("enrichment_model", "google/qwen-2.5-coder-32b")
        # Distinct from chunking's llm_client key; must pop before chunk_document()
        # (chunk_sliding_window_tokens has no **kwargs to absorb a stray key).
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
    """Runs a store.add()/fts_store.add() call under a lock scoped to that store.

    Concurrent writer threads to the same SQLite-backed store can hit
    "database is locked"; this serializes only same-store writes."""
    with lock:
        fn(*args)


def _raise_first_write_failure(db_futures: list) -> None:
    """wait() doesn't check for exceptions — a writer-thread failure was
    previously silently discarded (confirmed in production: the completion
    marker claimed far more chunks stored than the DB actually held). Call
    right after wait() so a write failure aborts the run loudly."""
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

    batch_size sizes embedder.embed() calls; write_batch_size (default 8x)
    controls how many embedded rows buffer before a store/FTS write — actual
    store.add() calls stay capped at the store's real max batch size (see
    max_write_batch), so an oversized value only affects buffering, never
    risks a crash."""
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

    # ── PHASE 1: PARSING ──
    # Two-phase (not pipelined) for simplicity: Phase 1 is CPU parsing, Phase
    # 2 is the real bottleneck (embed+write). Trade-off: all chunks stay in
    # RAM through Phase 1 — worth watching on a very large corpus.
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

    # ── PHASE 2: EMBEDDING + STORAGE ──
    # The real bottleneck: embeds in GPU-sized batches, batches DB writes into
    # fewer larger transactions, and uses a bigger writer pool now that Phase
    # 1's CPU workers have released their cores. `store` should carry Chroma's
    # HNSW bulk-load tuning (see _get_store() in cli/main.py).
    print(f"\nPhase 2: Embedding & storing {total_chunks:,} chunks "
          f"(embed batch={batch_size}, write batch={write_batch_size})...")

    write_ids: list[str] = []
    write_embeds: list[list[float]] = []
    write_texts: list[str] = []
    write_metas: list[dict] = []
    db_futures = []
    # See _locked_add(): repeated flushes could otherwise race two writes to the same store.
    store_lock = threading.Lock()
    fts_lock = threading.Lock()

    # write_batch_size (default 8x batch_size) can exceed Chroma's real
    # per-call limit (queried live via max_batch_size, since it derives from
    # SQLite's variable-count cap and isn't fixed). Sub-batches below are
    # capped at that value, keeping buffering size and write-call size separate.
    max_write_batch = min(write_batch_size, getattr(store, "max_batch_size", write_batch_size) or write_batch_size)

    try:
        with ThreadPoolExecutor(max_workers=8) as db_writer, \
             tqdm(total=total_chunks, desc="Embedding & Storing", unit="chunk", colour="GREEN") as chunk_bar:

            def _flush_writes(n: int) -> None:
                """Dispatches the first n embedded rows to background
                store.add()/fts_store.add() calls, sub-batched at
                max_write_batch to stay under Chroma's real limit."""
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

            if write_ids:
                _flush_writes(len(write_ids))

            tqdm.write("  Flushing database writes to disk...")
            wait(db_futures)
            _raise_first_write_failure(db_futures)

    except Exception as e:
        print(f"\n[ERROR] Bulk Indexing failed: {e}")
        stats.error_files += len(files_processed_this_run)
        return stats

    # Separate try/except from the write-critical section above: a bookkeeping
    # failure here (CSV log, resume file) must never be conflated with an
    # actual data-loss failure, or misreport a fully-ingested run as failed.
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
    """Chunks a file without enrichment, for ingest_multi_enrichment(): chunk
    boundaries are computed once and reused across enrichment variants
    instead of re-chunking per variant. Needed modules are imported at
    module scope — see _process_file_worker's docstring."""
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


# ── Completion marker: distinguishes a finished run from one interrupted
# mid-ingest (a non-empty store dir alone proves nothing — VectorStore
# construction creates bootstrap files before any chunk is stored). ──
_COMPLETION_MARKER_NAME = "_ingest_complete.json"


def _write_completion_marker(store: VectorStore, files_total: int, stats: IngestStats) -> None:
    """Writes a marker into a store's directory recording that this variant's
    run completed without a fatal exception; only ever called on the success
    path. Callers (e.g. master_grid.py) should check for this marker before
    treating a non-empty run directory as "already done" — a crash (OOM,
    preemption) can leave a non-empty but incomplete directory too."""
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
    """Chunks every file in data_dir once, then fans chunks out to one
    independently-enriched/embedded/stored variant per entry in
    `enrichment_strategies`. `stores` must map each strategy to its own
    VectorStore.

    Avoids re-chunking identical text once per enrichment value (as a naive
    per-variant ingest() call would): chunking is enrichment-agnostic, so it
    only needs to run once per (strategy, chunk_size, overlap) combo, while
    embeddings are still computed per variant (enrichment can change chunk
    text). Each variant's chunks are deep-copied before enrichment so it's
    safe whether enrich_chunks() mutates in place or not.

    enrichment_kwargs: forwarded to enrich_chunks() for every variant (e.g.
        glossary overrides) — **chunk_kwargs never reaches enrichment.
    on_progress: optional no-arg callback fired once per embed batch, purely
        as a liveness signal for callers (e.g. master_grid.py's watchdog) to
        detect a hung vs. slow run. Exceptions from it are not caught — keep
        it cheap and non-blocking.
    On success, each variant's store gets a `_ingest_complete.json` marker
        — see _write_completion_marker().
    batch_size sizes both the embed batch and the store/FTS write, written
        immediately per batch (no separate accumulation buffer), sub-batched
        at each store's real max batch size (see ChromaStore.max_batch_size)
        so it can't exceed Chroma's hard limit. chunks_stored only increments
        after a sub-batch write actually succeeds.
    write_batch_size: accepted for backward compatibility but unused — a
        warning prints if it's passed.
    """
    from open_telco_rag.core.enrichment import enrich_chunks

    enrichment_kwargs = enrichment_kwargs or {}
    if write_batch_size is not None:
        # This used to flush a separate larger buffer, which could exceed
        # Chroma's hard limit after the progress bar reported 100% done.
        # Writes are now sized to batch_size instead; this arg is a no-op.
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
            # Queried live from the store (see ChromaStore.max_batch_size);
            # falls back to a conservative 5000, safely under Chroma's observed 5461 default.
            "max_batch_size": getattr(st, "max_batch_size", None) or 5000,
        }
        _init_log(variants[ev]["log_file"])

    # ── PHASE 1: PARSING (once) — same two-phase design as ingest(), see its docstring.
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

    # ── PHASE 2: ENRICH + EMBED + STORE, per variant ──
    # Enrichment only transforms chunk text/metadata, never adds/removes
    # chunks, so the combined total across variants is known upfront.
    # Writes happen synchronously per embed batch (matching
    # master_grid_llm_completion.py's pattern), replacing an old buffered
    # design that could exceed Chroma's hard limit after the bar hit 100%
    # (confirmed in production) and over-counted chunks_stored on failure.
    # See _embed_and_write_batch below.
    total_variant_chunks = total_raw_chunks * len(enrichment_strategies)
    print(f"\nPhase 2: Enriching, embedding & storing {total_raw_chunks:,} chunks "
          f"x {len(enrichment_strategies)} variant(s) = {total_variant_chunks:,} "
          f"total (embed batch={batch_size}, capped per-store at Chroma's real "
          f"max batch size)...")

    def _embed_and_write_batch(ev: str, n: int) -> None:
        """Embeds the first n buffered chunks for this variant and writes them
        immediately, sub-batched at the store's real max batch size.
        chunks_stored increments only after each sub-batch write succeeds, so
        a failure partway through leaves an accurate partial count."""
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
                        # enrichment_kwargs merged last so it can override a
                        # colliding res["meta"] key instead of raising a TypeError.
                        enrich_call_kwargs = {**res["meta"], **enrichment_kwargs}
                        variant_chunks = enrich_chunks(
                            chunks=copy.deepcopy(res["chunks"]),
                            strategy=ev,
                            model_name=enrichment_model,
                            source_document=res["meta"].get("source", f_path.name),
                            **enrich_call_kwargs,
                        )
                    else:
                        # Deep-copied even for "none" so nothing is ever shared/mutated across variants.
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

    # Separate try/except from the embed-and-write loop: a bookkeeping
    # failure for one variant (CSV log, completion marker) must never mark
    # every variant as failed when the actual data landed safely.
    for ev in enrichment_strategies:
        v = variants[ev]
        try:
            for f_path, f_size, f_chunks, f_avg, f_time, f_ratio in v["files_to_log"]:
                _log_row(v["log_file"], f_path, f_size, "success", chunks=f_chunks,
                          model=embedder.model_key, avg_len=f_avg,
                          chunking_time_ms=f_time, expansion_ratio=f_ratio)
                v["stats"].files_processed += 1

            # See _write_completion_marker() — lets callers safely skip a finished re-run.
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

    response_schema, if given, is a JSON Schema forwarded to
    generator.generate() (see evaluate.py's _mc/_tf_response_schema() and
    generators/local.py's json_schema handling); None means unconstrained,
    so existing callers are unaffected."""
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