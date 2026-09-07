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
        print(f"[Error] Local database directory '{db_path}' does not exist.")
        return

    folder_name = path_in_repo or db_path.name
    print(f"\nUploading database directory '{db_path}' to HF Hub -> '{repo_id}/{folder_name}'...")

    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    api = HfApi()

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    # Chroma SQLite + HNSW index + FTS5 .db together.
    api.upload_folder(
        folder_path=str(db_path),
        path_in_repo=folder_name,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Uploaded to: https://huggingface.co/datasets/{repo_id}/tree/main/{folder_name}")


def pull_collection_from_hf(
    repo_id: str,
    path_in_repo: str,
    local_dir: str | Path = "./runs",
) -> Path:
    """Downloads a pre-indexed RAG database folder from HF Hub for local evaluation."""
    # snapshot_download keeps each file's full repo path; keep local_dir un-nested.
    base_dir = Path(local_dir)
    target_dir = base_dir / path_in_repo
    print(f"\nPulling database artifacts from HF Hub '{repo_id}/{path_in_repo}' -> '{target_dir}'...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"{path_in_repo}/*",
        local_dir=str(base_dir),
    )

    print(f"Database ready at '{target_dir}'")
    return target_dir