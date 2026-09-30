"""Reproducibility record for an evaluation run.

collect_run_record() gathers what is needed to tell later exactly what was run:
code version, environment, resolved settings, model identities, the dataset and
the index it was run against. Nothing secret is recorded (no keys, base URLs or
local paths); anything that cannot be determined is None.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TOKENIZER = "cl100k_base"          # used for token budgets and token-sized chunks
_DEPENDENCIES = ("chromadb", "sentence-transformers", "transformers", "torch", "tiktoken", "numpy",
                 "requests", "anthropic", "accelerate")
_PACKAGE_DIR = Path(__file__).resolve().parents[2]


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    if not (_PACKAGE_DIR / ".git").exists():
        return None   # not a checkout: never pick up an unrelated enclosing repository
    try:
        out = subprocess.run(["git", *args], cwd=_PACKAGE_DIR, capture_output=True, text=True, timeout=10,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_version() -> dict:
    """git_commit / git_dirty are the string "unknown" when they cannot be determined
    (no .git directory, e.g. installed from a wheel; git not installed; git failing)"""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") if commit else None
    return {"name": "open-telco-rag", "version": _version("open-telco-rag"),
            "git_commit": commit or "unknown",
            "git_dirty": (bool(status) if status is not None else "unknown")}


def environment() -> dict:
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": {name: _version(name) for name in _DEPENDENCIES}}


def hash_questions(questions: list[Any]) -> str:
    """sha256 of the questions as evaluated (order-sensitive, canonical JSON)"""
    rows = [{f.name: getattr(q, f.name) for f in dataclasses.fields(q)} if dataclasses.is_dataclass(q) else q
            for q in questions]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode("utf8")).hexdigest()


def describe_embedder(embedder: Any) -> dict:
    keys = ("model_key", "_model_name", "_dimensions", "_query_prefix", "_document_prefix",
            "_query_prompt_name", "_document_prompt_name", "track_truncation", "batch_size")
    out = {"class": type(embedder).__name__}
    for key in keys:
        value = getattr(embedder, key, None)
        if value is not None and isinstance(value, (str, int, float, bool)):
            out[key.lstrip("_")] = value
    max_len = getattr(embedder, "max_seq_length", None)
    if isinstance(max_len, int):
        out["max_seq_length"] = max_len
    return out


def describe_reranker(reranker: Any) -> dict | None:
    if reranker is None:
        return None
    out = {"class": type(reranker).__name__}
    for key in ("model_name", "query_max_chars", "doc_max_chars"):
        value = getattr(reranker, key, None)
        if isinstance(value, (str, int, float)):
            out[key] = value
    return out


def describe_index(store: Any, db_path_hint: str | None = None) -> dict:
    info: dict = {"collection": getattr(store, "collection_name", None)}
    try:
        info["chunks"] = int(store.count())
    except Exception:  # noqa: BLE001 - provenance is best effort, never breaks a run
        info["chunks"] = None
    collection = getattr(store, "_collection", None)
    space = (getattr(collection, "metadata", None) or {}).get("hnsw:space")
    if space:
        info["distance_space"] = space
    settings = read_index_settings(getattr(store, "path", None), info["collection"])
    if settings:
        info["ingest_settings"] = settings
    return info


def index_settings_file(db_path: str | Path, collection: str) -> Path:
    return Path(db_path) / f".ingest_settings_{collection}.json"


def read_index_settings(db_path: str | Path | None, collection: str | None) -> dict | None:
    """the settings the collection was ingested with, from the sidecar ingest() writes; None if absent"""
    if not db_path or not collection:
        return None
    try:
        data = json.loads(index_settings_file(db_path, collection).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "latest" not in data:
        return None
    return {"fingerprint": data["latest"].get("fingerprint"), "settings": data["latest"].get("settings"),
            "configurations_seen": len(data.get("history", []))}


def collect_run_record(
    *, params: dict, embedder: Any, reranker: Any, store: Any, questions: list[Any],
    questions_name: str, started_at: datetime, extra: dict | None = None,
) -> dict:
    """everything needed to reproduce or audit this run; `params` are the resolved
    evaluate settings (no secrets), `extra` caller-supplied identity (e.g. judge model)"""
    now = datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "finished_at": now.isoformat(timespec="seconds"),
        "code": code_version(),
        "environment": environment(),
        "settings": params,
        "tokenizer": TOKENIZER,
        "embedder": describe_embedder(embedder),
        "reranker": describe_reranker(reranker),
        "index": describe_index(store),
        "dataset": {"questions_file": questions_name, "n_questions": len(questions),
                    "questions_sha256": hash_questions(questions)},
        "randomness": {"seed": None, "note": "the pipeline has no seeded randomness (only retry jitter); "
                                             "generation is deterministic only at temperature 0 and only as far "
                                             "as the provider is"},
        **({"extra": extra} if extra else {}),
    }
