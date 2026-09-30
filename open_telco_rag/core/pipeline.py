"""Core pipeline - parallelized ingestion, search, generate."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from dotenv import load_dotenv

from tqdm import tqdm

from open_telco_rag.types import Chunk, IngestStats, SearchResult
from open_telco_rag.stores.fts import FTSStore

if TYPE_CHECKING:
    from open_telco_rag.types import Embedder, Source, VectorStore

load_dotenv()

# Windows spawn workers don't inherit sys.path; needed before importing `cli`.
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

from open_telco_rag.core.chunking import chunk_document, ChunkingStrategy
from open_telco_rag.core.enrichment import enrich_chunks
from cli.main import _get_source


# ── Multi-core worker ────────────────────────────────────────────────────────
def _span_metadata(chunk) -> dict:
    """char offsets + token count stored for token-sized chunks (span metric)"""
    return {"char_start": chunk.char_start, "char_end": chunk.char_end, "n_tokens": chunk.n_tokens}


# ── ingest resume bookkeeping ─────────────────────────────────────────────────
# the progress file lists fully ingested files and is tied to a fingerprint of the
# non-secret settings that decide what gets stored

_SECRET_KEY = re.compile(r"key|secret|token(?!s)|password|authorization|credential", re.IGNORECASE)
_NON_CONTENT_KWARGS = {"batch_size", "write_batch_size", "resume", "log_dir", "on_progress"}


def _norm_path(path: Any) -> str:
    """absolute, separator- and (on Windows) case-normalized path string"""
    return os.path.normcase(os.path.abspath(str(path))).replace("\\", "/")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None  # clients / callables / objects: not settings, not recorded


def _ingest_settings(store: Any, embedder: Any, min_length: int, chunk_kwargs: dict) -> dict:
    """non-secret settings that determine which chunks a run stores"""
    chunking = {}
    for key, value in sorted(chunk_kwargs.items()):
        if key in _NON_CONTENT_KWARGS or _SECRET_KEY.search(key):
            continue
        safe = _json_safe(value)
        if safe is not None:
            chunking[key] = safe
    return {
        "db_path": _norm_path(getattr(store, "path", "chroma_db")),
        "collection": getattr(store, "collection_name", "documents"),
        "model_key": embedder.model_key,
        "min_length": min_length,
        "chunking": chunking,
    }


def _fingerprint(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf8")).hexdigest()


def _set_aside(path: Path) -> Path:
    bak = path.with_name(path.name + ".bak")
    path.replace(bak)  # renamed, never deleted
    return bak


def _count_chunks(store: Any) -> int:
    try:
        return int(store.count())
    except Exception:  # noqa: BLE001
        return 0


def _load_progress(progress_file: Path, legacy_file: Path, fingerprint: str, store: Any) -> set[str]:
    """files already ingested under exactly these settings; empty (with a warning
    and the old file renamed to .bak) when the file is unreadable or foreign"""
    if legacy_file.exists():
        bak = _set_aside(legacy_file)
        print(f"\n[WARN] Found an old-format resume file ({legacy_file.name}). It does not record which "
              f"collection or settings it belongs to, so it cannot be trusted: this run restarts from "
              f"scratch. The old file was renamed to {bak.name}.")
    if not progress_file.exists():
        n_existing = _count_chunks(store)
        if n_existing:
            print(f"\n[WARN] No resume record matches these settings, but the target collection already "
                  f"holds {n_existing} chunks (the settings, database or log directory changed since it "
                  f"was ingested). Every file is ingested again over them, so chunks from different "
                  f"settings can end up mixed; use --no-resume to rebuild the collection or a new "
                  f"--collection.")
        return set()

    reason = None
    completed: set[str] = set()
    try:
        data = json.loads(progress_file.read_text(encoding="utf8"))
        if not isinstance(data, dict) or "completed" not in data:
            reason = "it is in an old or unrecognized format"
        elif data.get("fingerprint") != fingerprint:
            reason = "it was written for a different collection, database or chunking/enrichment configuration"
        else:
            completed = set(data["completed"])
    except (ValueError, OSError) as e:
        reason = f"it could not be read ({type(e).__name__})"

    if reason is None and completed and _count_chunks(store) == 0:
        reason = (f"it lists {len(completed)} finished file(s) but the collection is empty "
                  f"(it was deleted or rebuilt)")
    if reason is None:
        return completed

    bak = _set_aside(progress_file)
    msg = (f"\n[WARN] Ignoring resume file {progress_file.name}: {reason}. Starting fresh; "
           f"the old file was renamed to {bak.name}.")
    n_existing = _count_chunks(store)
    if n_existing:
        msg += (f" Note: the target collection already holds {n_existing} chunks, and files are "
                f"re-ingested over them, so chunks from different settings can end up mixed; use "
                f"--no-resume to rebuild the collection or a new --collection.")
    print(msg)
    return set()


def _record_index_settings(store: Any, fingerprint: str, settings: dict) -> None:
    """remember which settings a collection was ingested with, in a sidecar next
    to the database (travels with it, e.g. through push-hf), so evaluation results
    can state the chunking they ran against. keeps a history: a collection that
    was ingested under more than one configuration says so. never fails an ingest"""
    from open_telco_rag.evaluation.provenance import code_version, index_settings_file

    try:
        path = index_settings_file(store.path, store.collection_name)
        try:
            data = json.loads(path.read_text(encoding="utf8"))
            history = list(data.get("history", []))
        except (OSError, ValueError):
            history = []
        # the fingerprint covers the (local) db path; the recorded settings do not
        entry = {"fingerprint": fingerprint, "settings": {k: v for k, v in settings.items() if k != "db_path"},
                 "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                 "code": code_version()}
        if history and history[-1].get("fingerprint") == fingerprint:
            history[-1] = entry
        else:
            history.append(entry)
        path.write_text(json.dumps({"version": 1, "latest": entry, "history": history}), encoding="utf8")
    except Exception as e:  # noqa: BLE001 - bookkeeping only
        print(f"\n[WARN] Could not record the index settings ({e}); the ingest itself succeeded.")


def _unseen(ids: list[str], texts: list[str], seen: dict[str, int]) -> tuple[list[str], list[str]]:
    """the (id, text) pairs not already in the BM25 index with the same text"""
    new = [(i, t) for i, t in zip(ids, texts) if seen.get(i) != hash(t)]
    return [i for i, _ in new], [t for _, t in new]


def _write_progress(progress_file: Path, fingerprint: str, settings: dict, completed: set[str]) -> None:
    progress_file.write_text(json.dumps(
        {"version": 2, "fingerprint": fingerprint, "settings": settings, "completed": sorted(completed)}
    ), encoding="utf8")


def _process_file_worker(args: tuple) -> dict:
    """Runs in a CPU process pool; imports are module-scoped so spawned
    Windows workers can resolve `cli`."""
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
        # Reuse CLI's source factory.
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
        # Pop before chunk_document(); distinct from chunking's llm_client key.
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
            # offsets only exist for token-sized chunks (target_tokens/
            # max_tokens); default ingests store exactly what they always did
            if c.char_start >= 0:
                base_meta.update(_span_metadata(c))
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
    """Run a store.add() call under a per-store lock (serialises same-store
    SQLite writes)."""
    with lock:
        fn(*args)


def _raise_first_write_failure(db_futures: list) -> None:
    """Re-raise the first writer-thread exception (wait() doesn't check).
    Call right after wait()."""
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
    sizes the buffer before a store/FTS write. store.add() calls stay capped
    at the store's real max batch size."""
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

    settings    = _ingest_settings(store, embedder, min_length, chunk_kwargs)
    fingerprint = _fingerprint(settings)
    progress_file = log_dir / f".progress_{embedder.model_key}_{fingerprint[:12]}.json"
    legacy_file   = log_dir / f".progress_{embedder.model_key}.json"
    completed: set[str] = set()
    if resume:
        completed = _load_progress(progress_file, legacy_file, fingerprint, store)

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

    fts_seen: dict[str, int] = {}
    if not resume:
        fts_store.clear()
    else:
        # a run that was killed after some FTS writes but before the matching Chroma writes would
        # otherwise insert those rows a second time
        removed, repeated = fts_store.remove_exact_duplicates()
        if removed:
            print(f"\n[INFO] Removed {removed} repeated BM25 rows (same chunk id and text) left by an earlier interrupted run.")
        if repeated:
            print(f"\n[WARN] {repeated} BM25 rows repeat a chunk id with different text (chunks from different "
                  f"settings); they were left in place. Rebuild the collection with --no-resume to remove them.")
        fts_seen = fts_store.existing_hashes()

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

    # ── PHASE 1: PARSING ── (two-phase, not pipelined: CPU parse, then embed+write)
    print(f"\nPhase 1: Parsing {len(tasks)} files across {max_workers} CPU workers...")

    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict] = []
    files_to_log = []
    files_processed_this_run = []
    file_ends: list[tuple[str, int]] = []       # (file, index in all_* after its last chunk)

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
                file_ends.append((res["rel"], len(all_texts)))

                stats.total_chunking_time_sec += (res["chunking_time_ms"] / 1000)
                files_to_log.append((
                    f_path, res["file_size_kb"], len(res["chunks"]),
                    res["avg_len"], res["chunking_time_ms"], res["expansion_ratio"]
                ))
                files_processed_this_run.append(res["rel"])

            elif res["status"] == "skipped":
                stats.skipped_files += 1
                tqdm.write(f"  [SKIPPED] {f_path.name}: {res['error']}")
                _log_row(log_file, f_path, res["file_size_kb"], "skipped",
                         error=res["error"], model=embedder.model_key)
            else:
                stats.error_files += 1
                tqdm.write(f"  [ERROR] {f_path.name}: {res['error']}")
                _log_row(log_file, f_path, res["file_size_kb"], "error",
                         error=res["error"], model=embedder.model_key)

    total_chunks = len(all_texts)
    if total_chunks == 0:
        return stats

    # ── PHASE 2: EMBEDDING + STORAGE ── (the bottleneck)
    print(f"\nPhase 2: Embedding & storing {total_chunks:,} chunks "
          f"(embed batch={batch_size}, write batch={write_batch_size})...")

    write_ids: list[str] = []
    write_embeds: list[list[float]] = []
    write_texts: list[str] = []
    write_metas: list[dict] = []
    db_futures = []
    # resume checkpoints: a file counts as done once every chunk of it is confirmed stored
    dispatched = 0                              # rows handed to background writes so far
    flush_marks: list[tuple[int, list]] = []    # (rows dispatched, the futures writing them)
    confirmed = 0                               # rows whose writes all succeeded
    next_file = 0
    # Serialise same-store writes; see _locked_add().
    store_lock = threading.Lock()
    fts_lock = threading.Lock()

    # Cap write-call size at the store's real limit; buffering size stays separate.
    max_write_batch = min(write_batch_size, getattr(store, "max_batch_size", write_batch_size) or write_batch_size)

    try:
        with ThreadPoolExecutor(max_workers=8) as db_writer, \
             tqdm(total=total_chunks, desc="Embedding & Storing", unit="chunk", colour="GREEN") as chunk_bar:

            def _checkpoint() -> None:
                """write the files whose chunks are all confirmed stored to the progress
                file, so a killed run resumes from here; bookkeeping, never fails an ingest"""
                nonlocal confirmed, next_file
                while flush_marks and all(f.done() and not f.cancelled() and f.exception() is None
                                          for f in flush_marks[0][1]):
                    confirmed = flush_marks.pop(0)[0]
                newly = []
                while next_file < len(file_ends) and file_ends[next_file][1] <= confirmed:
                    newly.append(file_ends[next_file][0])
                    next_file += 1
                if newly and resume:
                    try:
                        completed.update(newly)
                        _write_progress(progress_file, fingerprint, settings, completed)
                    except Exception:  # noqa: BLE001
                        return                  # the end-of-run write reports a failing progress file

            def _flush_writes(n: int) -> None:
                """Dispatch the first n embedded rows to background writes,
                sub-batched at max_write_batch."""
                nonlocal dispatched
                first_future = len(db_futures)
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
                        _locked_add, fts_lock, fts_store.add, *_unseen(ids[start:end], texts[start:end], fts_seen),
                    ))
                dispatched += n
                flush_marks.append((dispatched, db_futures[first_future:]))

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
                _checkpoint()

            if write_ids:
                _flush_writes(len(write_ids))

            tqdm.write("  Flushing database writes to disk...")
            wait(db_futures)
            _raise_first_write_failure(db_futures)

    except Exception as e:
        print(f"\n[ERROR] Bulk Indexing failed: {e}")
        stats.error_files += len(files_processed_this_run)
        return stats

    # Separate from the write-critical section: a bookkeeping failure isn't data loss.
    try:
        for log_item in files_to_log:
            f_path, f_size, f_chunks, f_avg, f_time, f_ratio = log_item
            _log_row(log_file, f_path, f_size, "success", chunks=f_chunks, model=embedder.model_key,
                     avg_len=f_avg, chunking_time_ms=f_time, expansion_ratio=f_ratio)
            stats.files_processed += 1

        if resume:
            completed.update(files_processed_this_run)
            _write_progress(progress_file, fingerprint, settings, completed)
    except Exception as e:
        print(f"\n[WARN] Post-write bookkeeping (CSV log / resume progress file) failed: "
              f"{e} — every chunk was still successfully stored; only the log/resume "
              f"file may be incomplete.")

    if stats.files_processed:
        _record_index_settings(store, fingerprint, settings)

    return stats


# ── Worker: chunk-only, no enrichment ────────────────────────────────────────
def _process_file_worker_rawchunk(args: tuple) -> dict:
    """Chunks a file without enrichment for ingest_multi_enrichment()
    (boundaries computed once, reused per variant)."""
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


# Completion marker: distinguishes a finished run from one interrupted mid-ingest.
_COMPLETION_MARKER_NAME = "_ingest_complete.json"


def _write_completion_marker(store: VectorStore, files_total: int, stats: IngestStats) -> None:
    """Write a marker into the store dir on the success path. Callers check
    it before treating a non-empty run directory as already done."""
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


# ── Ingest: chunk once, fan out across enrichment variants ───────────────────
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
    """Chunk every file in data_dir once, then fan chunks out to one
    independently enriched/embedded/stored variant per entry in
    `enrichment_strategies`. `stores` maps each strategy to its own
    VectorStore. Chunks are deep-copied per variant before enrichment.

    enrichment_kwargs: forwarded to enrich_chunks() for every variant.
    on_progress: no-arg callback fired once per embed batch (liveness signal;
        exceptions not caught).
    batch_size: sizes both the embed batch and the store/FTS write,
        sub-batched at each store's real max batch size.
    write_batch_size: accepted for back-compat but unused (warns if passed).
    On success each variant's store gets a `_ingest_complete.json` marker.
    """
    from open_telco_rag.core.enrichment import enrich_chunks

    enrichment_kwargs = enrichment_kwargs or {}
    if write_batch_size is not None:
        # No longer used; writes are sized to batch_size.
        print(f"  ingest_multi_enrichment(): write_batch_size={write_batch_size} was "
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
            # From the store (see ChromaStore.max_batch_size); fallback 5000.
            "max_batch_size": getattr(st, "max_batch_size", None) or 5000,
        }
        _init_log(variants[ev]["log_file"])

    # ── PHASE 1: PARSING (once) ──
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
                tqdm.write(f"  [SKIPPED] {f_path.name}: {res['error']}")
                for ev in enrichment_strategies:
                    variants[ev]["stats"].skipped_files += 1
                    _log_row(variants[ev]["log_file"], f_path, res["file_size_kb"],
                              "skipped", error=res["error"], model=embedder.model_key)
            else:
                tqdm.write(f"  [ERROR] {f_path.name}: {res['error']}")
                for ev in enrichment_strategies:
                    variants[ev]["stats"].error_files += 1
                    _log_row(variants[ev]["log_file"], f_path, res["file_size_kb"],
                              "error", error=res["error"], model=embedder.model_key)

    if total_raw_chunks == 0:
        for ev in enrichment_strategies:
            _write_completion_marker(variants[ev]["store"], len(files), variants[ev]["stats"])
        return {ev: variants[ev]["stats"] for ev in enrichment_strategies}

    # ── PHASE 2: ENRICH + EMBED + STORE, per variant ──
    # Enrichment never adds/removes chunks, so the combined total is known
    # upfront. Writes happen synchronously per embed batch.
    total_variant_chunks = total_raw_chunks * len(enrichment_strategies)
    print(f"\nPhase 2: Enriching, embedding & storing {total_raw_chunks:,} chunks "
          f"x {len(enrichment_strategies)} variant(s) = {total_variant_chunks:,} "
          f"total (embed batch={batch_size}, capped per-store at Chroma's real "
          f"max batch size)...")

    def _embed_and_write_batch(ev: str, n: int) -> None:
        """Embed the first n buffered chunks for this variant and write them,
        sub-batched at the store's max. chunks_stored increments only after
        each sub-batch write succeeds."""
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
                        # enrichment_kwargs merged last so it can override a res["meta"] key.
                        enrich_call_kwargs = {**res["meta"], **enrichment_kwargs}
                        variant_chunks = enrich_chunks(
                            chunks=copy.deepcopy(res["chunks"]),
                            strategy=ev,
                            model_name=enrichment_model,
                            source_document=res["meta"].get("source", f_path.name),
                            **enrich_call_kwargs,
                        )
                    else:
                        # Deep-copied even for "none" so variants never share state.
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
                        if c.char_start >= 0:
                            base_meta.update(_span_metadata(c))
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

            # Final partial batches - whatever's left once every file is enriched.
            for ev in enrichment_strategies:
                v = variants[ev]
                if v["buf_texts"]:
                    _embed_and_write_batch(ev, len(v["buf_texts"]))

    except Exception as e:
        print(f"\n[ERROR] Bulk multi-enrichment indexing failed: {e}")
        for ev in enrichment_strategies:
            variants[ev]["stats"].error_files += len(files)
        return {ev: variants[ev]["stats"] for ev in enrichment_strategies}

    # Separate from the embed-and-write loop: a bookkeeping failure isn't data loss.
    for ev in enrichment_strategies:
        v = variants[ev]
        try:
            for f_path, f_size, f_chunks, f_avg, f_time, f_ratio in v["files_to_log"]:
                _log_row(v["log_file"], f_path, f_size, "success", chunks=f_chunks,
                          model=embedder.model_key, avg_len=f_avg,
                          chunking_time_ms=f_time, expansion_ratio=f_ratio)
                v["stats"].files_processed += 1

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
    context_token_budget: int | None = None,
    candidate_pool:  int | None = None,
    budget_overflow: str | None = None,
) -> list[SearchResult]:
    """Embed a query and retrieve nearest neighbours; the budget arguments are
    described on core.retrieval.execute_retrieval"""
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
        context_token_budget=context_token_budget,
        candidate_pool=candidate_pool,
        budget_overflow=budget_overflow,
    )

# ── Generate ──────────────────────────────────────────────────────────────────

def generate(
    question:    str,
    results:     list[SearchResult],
    generator    = None,
    source_type: str   = "telecom",
    # Legacy kwargs - ignored if generator is provided
    claude_model: str   = "claude-sonnet-4-6",
    max_tokens:   int   = 1000,
    temperature:  float = 0.0,
    max_retries:  int   = 3,
    response_schema: dict | None = None,
) -> dict:
    """Generate a grounded answer from retrieved chunks.

    response_schema, if given, is a JSON Schema forwarded to
    generator.generate(); None means unconstrained."""
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