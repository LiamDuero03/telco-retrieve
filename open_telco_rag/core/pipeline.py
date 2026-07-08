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
    batch_size: int  = 64,
    resume:     bool = True,
    log_dir:    str | Path = "logs",
    **chunk_kwargs: Any,
) -> IngestStats:
    """Scan files, clean, chunk, embed and store."""

    data_dir = Path(data_dir)
    log_dir  = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = source.__class__.__name__.lower().replace("source", "")
    log_file    = log_dir / f"ingest_{source_name}_{embedder.model_key}_{timestamp}.csv"
    stats       = IngestStats()

    # ── NEW: detect whether Marker was used — affects chunk routing ───────────
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

        # ── Chunk ─────────────────────────────────────────────────────────────
        try:
            from open_telco_rag.core.chunking import chunk_document, ChunkingStrategy
            from open_telco_rag.core.enrichment import enrich_chunks
            source_type_str = source.__class__.__name__.lower().replace("source", "")

            is_native_markdown = file_path.suffix.lower() in [".md", ".yaml", ".yml"]
            is_marker_output   = use_marker and file_path.suffix.lower() in [".pdf", ".docx"]

            # Local runtime copy of runtime kwargs to prevent mutating shared state
            active_kwargs = chunk_kwargs.copy()
            active_kwargs["source_type"] = source_type_str

            # Fall back to structural text/markdown overrides ONLY if the user selected the baseline path.
            # Otherwise, respect the exact strategy chosen from the Streamlit UI.
            user_strategy = active_kwargs.get("strategy", "text")

            if user_strategy in ["text", "markdown"]:
                if is_native_markdown or is_marker_output:
                    active_kwargs["strategy"] = ChunkingStrategy.MARKDOWN
                else:
                    active_kwargs["strategy"] = ChunkingStrategy.TEXT

            # Extract our new enrichment parameter if it was passed through **chunk_kwargs
            enrichment_strategy = active_kwargs.pop("enrichment", "none")
            enrichment_model    = active_kwargs.pop("enrichment_model", "google/qwen-2.5-coder-32b") # <-- Capture it here

            # Execute the orchestrated master strategy chunker
            chunks = chunk_document(text=text, **active_kwargs)
            chunks = [c for c in chunks if len(c.text.strip()) >= min_length]

            if enrichment_strategy and enrichment_strategy != "none" and chunks:
                meta = source.metadata(file_path)
                
                # Setup LLM Client dynamically if the LLM metadata strategy is requested
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

        # Embed and store in batches
        try:
            meta = source.metadata(file_path)
            for batch_start in range(0, len(chunks), batch_size):
                batch  = chunks[batch_start:batch_start+batch_size]
                texts  = [c.text for c in batch]
                ids    = [f"{rel}::{embedder.model_key}::{batch_start + j}"
                        for j, c in enumerate(batch)]
                
                # Merge custom metadata (like hierarchical parent links) alongside standard parameters
                metas  = []
                for j, c in enumerate(batch):
                    base_meta = {
                        **meta, 
                        "heading": c.heading or "",
                        "chunk_index": batch_start + j,
                        "source_type": c.source_type
                    }
                    if hasattr(c, 'metadata') and c.metadata:
                        base_meta.update(c.metadata)
                    metas.append(base_meta)

                embeds = embedder.embed(texts)
                store.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
                stats.chunks_stored += len(batch)

            chunk_lens = [len(c.text) for c in chunks]
            print(f"  Chunks: {len(chunks)} | "
                  f"avg: {sum(chunk_lens)//len(chunk_lens)} | "
                  f"min: {min(chunk_lens)} | max: {max(chunk_lens)}")
            _log_row(log_file, file_path, file_size_kb, "success",
                     chunks=len(chunks), model=embedder.model_key,
                     avg_len=sum(chunk_lens)//len(chunk_lens))
            stats.files_processed += 1

            if resume:
                completed.add(rel)
                progress_file.write_text(json.dumps(sorted(completed)))

        except Exception as e:
            print(f"  ERROR embedding/storing: {e}")
            stats.error_files += 1
            _log_row(log_file, file_path, file_size_kb, "error",
                     error=f"embed/store: {e}", model=embedder.model_key)

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
    query:        str,
    store:        VectorStore,
    embedder:     Embedder,
    top_k:        int   = 8,
    threshold:    float = 1.0,
    reranker      = None,
    rerank_top_k: int   = 5,
) -> list[SearchResult]:
    """Embed a query and retrieve nearest neighbours.

    If reranker is provided, retrieves top_k * 3 candidates first,
    then reranks down to rerank_top_k for higher precision.
    """
    candidate_k = top_k * 3 if reranker is not None else top_k

    vector  = embedder.embed([query])[0]
    results = store.search(vector, n_results=candidate_k)
    results = [r for r in results if r.distance < threshold]

    if reranker is not None and results:
        results = reranker.rerank(query, results, top_k=rerank_top_k)

    return results


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
    "chunks", "avg_len", "embedding_model", "error_message"
]

def _init_log(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_CSV_HEADERS).writeheader()

def _log_row(
    path: Path, file_path: Path, size_kb: float, status: str,
    chunks: int = 0, avg_len: int = 0, model: str = "", error: str = ""
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
        })