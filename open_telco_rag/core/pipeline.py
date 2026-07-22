"""Core pipeline — ingest, search, generate."""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any


from dotenv import load_dotenv

from open_telco_rag.types import Chunk, IngestStats, SearchResult

if TYPE_CHECKING:
    from open_telco_rag.types import Embedder, Source, VectorStore

load_dotenv()

# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest(
    source:      Source,
    store:      VectorStore,
    embedder:   Embedder,
    data_dir:   str | Path,
    min_length: int  = 100,
    batch_size: int  = 64, # Default used for local, can be overridden to 1000 for OpenRouter
    resume:     bool = True,
    log_dir:    str | Path = "logs",
    **chunk_kwargs: Any,
) -> IngestStats:
    """Scan files, clean, chunk, globally batch embed and store."""

    data_dir = Path(data_dir)
    log_dir  = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = source.__class__.__name__.lower().replace("source", "")
    log_file    = log_dir / f"ingest_{source_name}_{embedder.model_key}_{timestamp}.csv"
    stats       = IngestStats()

    use_marker = getattr(source, '_use_marker', False)

    # Resume tracking
    progress_file     = log_dir / f".progress_{embedder.model_key}.json"
    completed: set[str] = set()
    if resume and progress_file.exists():
        completed = set(json.loads(progress_file.read_text()))

    # Find all supported files
    extensions = source.supported_extensions()
    files      = [
        f for ext in extensions
        for f in sorted(data_dir.rglob(f"*{ext}"))
    ]

    _init_log(log_file)
    print(f"\nFound {len(files)} files | model: {embedder.model_key}")
    print(f"Logging to: {log_file}\n")

    # ── PHASE 1: CHUNK ACCUMULATION ──
    global_texts = []
    global_ids = []
    global_metadatas = []
    files_to_log = [] # Holds stats to write to CSV logs after successful ingestion
    files_processed_this_run = []

    for file_path in files:
        rel = str(file_path.relative_to(data_dir))

        if rel in completed:
            print(f"  SKIP (already ingested): {file_path.name}")
            stats.skipped_files += 1
            continue

        file_size_kb = round(file_path.stat().st_size / 1024, 1)
        file_num = files.index(file_path) + 1
        print(f"[{file_num}/{len(files)}] Processing: {file_path.name} ({file_size_kb} KB)")

        # Load
        try:
            text = source.load(file_path)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            stats.error_files += 1
            stats.errors.append({"file": rel, "stage": "load", "error": str(e)})
            _log_row(log_file, file_path, file_size_kb, "error", error=str(e),
                     model=embedder.model_key)
            continue

        if not text or not text.strip():
            stats.skipped_files += 1
            _log_row(log_file, file_path, file_size_kb, "skipped",
                     error="empty after load", model=embedder.model_key)
            continue

        # Clean
        try:
            text = source.clean(text)
        except Exception as e:
            print(f"  ERROR cleaning: {e}")
            stats.error_files += 1
            _log_row(log_file, file_path, file_size_kb, "error",
                     error=f"clean: {e}", model=embedder.model_key)
            continue

        # Chunk
        try:
            from open_telco_rag.core.chunking import chunk_document, ChunkingStrategy
            from open_telco_rag.core.enrichment import enrich_chunks

            start_chunk_time = time.perf_counter()
            raw_char_count   = len(text) if text else 0

            source_type_str = source.__class__.__name__.lower().replace("source", "")

            is_native_markdown = file_path.suffix.lower() in [".md", ".yaml", ".yml"]
            is_marker_output   = use_marker and file_path.suffix.lower() in [".pdf", ".docx"]
            is_structured      = is_native_markdown or is_marker_output

            active_kwargs = chunk_kwargs.copy()
            active_kwargs["source_type"] = source_type_str

            user_strategy_str = active_kwargs.get("strategy", "text_baseline")
            
            try:
                user_strategy = ChunkingStrategy(user_strategy_str)
            except ValueError:
                print(f"  WARNING: Strategy '{user_strategy_str}' not recognized. Falling back to TEXT_BASELINE.")
                user_strategy = ChunkingStrategy.TEXT_BASELINE

            structure_required_strategies = {
                ChunkingStrategy.HIERARCHICAL_MARKDOWN,
                ChunkingStrategy.PARENT_CHILD
            }

            if not is_structured and user_strategy in structure_required_strategies:
                fallback_strategy = ChunkingStrategy.TEXT_BASELINE
                print(f"  ⚠️ WARNING: '{user_strategy.value}' requires structured headers, "
                      f"but '{file_path.name}' is unstructured. Falling back to '{fallback_strategy.value}'.")
                active_kwargs["strategy"] = fallback_strategy
            else:
                active_kwargs["strategy"] = user_strategy

            active_kwargs["is_structured"] = is_structured

            enrichment_strategy = active_kwargs.pop("enrichment", "none")
            enrichment_model    = active_kwargs.pop("enrichment_model", "openai/gpt-4o-mini")

            chunks = chunk_document(text=text, **active_kwargs)
            chunks = [c for c in chunks if len(c.text.strip()) >= min_length]

            if enrichment_strategy and enrichment_strategy != "none" and chunks:
                meta = source.metadata(file_path)
                llm_client = None
                if enrichment_strategy == "llm_metadata":
                    from open_telco_rag.generators.openrouter import OpenRouterGenerator
                    llm_client = OpenRouterGenerator(model=enrichment_model) 

                chunks = enrich_chunks(
                    chunks=chunks,
                    strategy=enrichment_strategy,
                    llm_client=llm_client,
                    model_name=enrichment_model,
                    source_document=meta.get("source", file_path.name),
                    **meta 
                )

            chunk_duration_sec = time.perf_counter() - start_chunk_time
            fused_char_count   = sum(len(c.text) for c in chunks) if chunks else 0

            stats.total_chunking_time_sec += chunk_duration_sec
            stats.total_raw_chars          += raw_char_count
            stats.total_fused_chars        += fused_char_count

            chunking_time_ms = chunk_duration_sec * 1000
            expansion_ratio  = fused_char_count / max(1, raw_char_count)

        except Exception as e:
            print(f"  ERROR chunking: {e}")
            stats.error_files += 1
            _log_row(log_file, file_path, file_size_kb, "error",
                     error=f"chunk: {e}", model=embedder.model_key)
            continue

        if not chunks:
            stats.skipped_files += 1
            _log_row(log_file, file_path, file_size_kb, "skipped",
                     error="no valid chunks", model=embedder.model_key)
            continue

        # Accumulate chunks globally instead of embedding immediately
        meta = source.metadata(file_path)
        for j, c in enumerate(chunks):
            chunk_id = f"{rel}::{embedder.model_key}::{j}"
            base_meta = {
                **meta, 
                "heading": c.heading or "",
                "chunk_index": j,
                "source_type": c.source_type
            }
            if hasattr(c, 'metadata') and c.metadata:
                base_meta.update(c.metadata)

            global_texts.append(c.text)
            global_ids.append(chunk_id)
            global_metadatas.append(base_meta)

        chunk_lens = [len(c.text) for c in chunks]
        print(f"  Buffered: {len(chunks)} chunks | avg: {sum(chunk_lens)//len(chunk_lens)}")
        
        # Buffer stats for logging after successful batch write
        files_to_log.append((file_path, file_size_kb, len(chunks), sum(chunk_lens)//len(chunk_lens), chunking_time_ms, expansion_ratio))
        files_processed_this_run.append(rel)

    # ── PHASE 2: GLOBAL BULK EMBEDDING ──
    total_chunks = len(global_texts)
    if total_chunks > 0:
        print(f"\n{'-'*60}")
        print(f"Phase 2: Bulk embedding {total_chunks} total chunks...")
        print(f"Batch size: {batch_size} chunks per API call")
        print(f"{'-'*60}")

        start_embed_time = time.perf_counter()
        
        try:
            for batch_start in range(0, total_chunks, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks)
                print(f"  Embedding batch [{batch_start}/{total_chunks}] (Size: {batch_end - batch_start})...")
                
                batch_texts = global_texts[batch_start:batch_end]
                batch_ids = global_ids[batch_start:batch_end]
                batch_metas = global_metadatas[batch_start:batch_end]

                # Single bulk call to OpenRouter / Local model!
                batch_embeds = embedder.embed(batch_texts)
                
                # Write to vector database
                store.add(ids=batch_ids, embeddings=batch_embeds, documents=batch_texts, metadatas=batch_metas)
                stats.chunks_stored += len(batch_texts)

            # Execution was successful! Log files and mark progress
            for log_item in files_to_log:
                f_path, f_size, f_chunks, f_avg, f_time, f_ratio = log_item
                _log_row(log_file, f_path, f_size, "success", chunks=f_chunks, model=embedder.model_key,
                         avg_len=f_avg, chunking_time_ms=f_time, expansion_ratio=f_ratio)
                stats.files_processed += 1

            if resume:
                completed.update(files_processed_this_run)
                progress_file.write_text(json.dumps(sorted(completed)))

        except Exception as e:
            print(f"\n[FATAL ERROR] Bulk Embedding/Database storage failed: {e}")
            stats.error_files += len(files_processed_this_run)
            for f_path, f_size, _, _, _, _ in files_to_log:
                _log_row(log_file, f_path, f_size, "error", error=f"bulk embed fail: {e}", model=embedder.model_key)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done — {stats.files_processed} files | "
          f"{stats.chunks_stored} chunks | "
          f"{stats.skipped_files} skipped | "
          f"{stats.error_files} errors")
    print(f"{'='*60}")

    return stats


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
) -> dict:
    """Generate a grounded answer from retrieved chunks."""
    if generator is None:
        from open_telco_rag.generators.claude import ClaudeGenerator
        generator = ClaudeGenerator(
            model       = claude_model,
            max_tokens  = max_tokens,
            temperature = temperature,
            max_retries = max_retries,
        )
    return generator.generate(question, results, source_type=source_type)


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
        