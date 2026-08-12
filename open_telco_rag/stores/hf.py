"""Hugging Face Hub Database Exporter & Downloader for RAG Artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def push_collection_to_hf(
    db_path: str | Path,
    repo_id: str,
    path_in_repo: str | None = None,
    private: bool = True,
) -> None:
    """Uploads the raw ChromaDB vector directory and SQLite FTS5 database directly to HF Hub."""
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"❌ [Error] Local database directory '{db_path}' does not exist.")
        return

    folder_name = path_in_repo or db_path.name
    print(f"\n📦 Uploading database directory '{db_path}' to HF Hub -> '{repo_id}/{folder_name}'...")

    # HF_HUB_ENABLE_HF_TRANSFER is deprecated in current huggingface_hub
    # (hf_transfer isn't used anymore) -- setting it now just prints a
    # FutureWarning on every call instead of doing anything. The
    # high-performance-transfer successor is HF_XET_HIGH_PERFORMANCE (same
    # fix already applied at master_grid.py's module top level; setting it
    # again here too since this function is also called directly by
    # scripts that don't import master_grid.py).
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    api = HfApi()

    # Create dataset repo if it doesn't already exist
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    # Upload the entire folder (Chroma SQLite, HNSW index files, and SQLite FTS5 .db)
    api.upload_folder(
        folder_path=str(db_path),
        path_in_repo=folder_name,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"✅ Successfully uploaded database artifacts to: https://huggingface.co/datasets/{repo_id}/tree/main/{folder_name}")


def pull_collection_from_hf(
    repo_id: str,
    path_in_repo: str,
    local_dir: str | Path = "./runs",
) -> Path:
    """Downloads a pre-indexed RAG database folder from HF Hub directly for local evaluation."""
    # snapshot_download() always replicates a matched file's FULL
    # repo-relative path under whatever local_dir it's given — it does NOT
    # strip path_in_repo's own prefix off first. Passing an already-nested
    # local_dir would make it re-append the full path_in_repo on top,
    # landing the real files one directory level deeper than expected;
    # ChromaStore would then silently open an empty directory and create a
    # fresh empty collection — no exception, just store.count() == 0
    # downstream even though the download succeeded. base_dir is kept as
    # the ORIGINAL, un-nested local_dir and passed to snapshot_download
    # directly — correct regardless of how many segments path_in_repo has.
    base_dir = Path(local_dir)
    target_dir = base_dir / path_in_repo
    print(f"\n📥 Pulling database artifacts from HF Hub '{repo_id}/{path_in_repo}' -> '{target_dir}'...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"{path_in_repo}/*",
        local_dir=str(base_dir),
    )

    print(f"✅ Database ready at '{target_dir}'. You can now run evaluation!")
    return target_dir