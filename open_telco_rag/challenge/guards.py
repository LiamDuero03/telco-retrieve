from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "embedding_models.json"
PARENT_SAMPLE = 200


def _normal(name: str) -> str:
    candidate = os.path.abspath(name)
    return candidate if os.path.exists(candidate) else name


def _registry_name(key: str) -> str | None:
    entry = json.loads(REGISTRY.read_text(encoding="utf-8")).get(key)
    return _normal(entry["model_name"]) if entry else None


def index_embedder_key(db_path: str, collection: str, store) -> str | None:
    """Embedder key recorded for the index: the ingest sidecar first, else the chunk ids."""
    from open_telco_rag.evaluation.provenance import read_index_settings

    info = read_index_settings(db_path, collection)
    key = ((info or {}).get("settings") or {}).get("model_key")
    if key:
        return key
    ids = store._collection.peek(1).get("ids") or []
    parts = ids[0].split("::") if ids else []
    return parts[1] if len(parts) == 3 else None


def _same_embedder(settings: dict, index_key: str) -> bool:
    custom = settings["embedder_path"] or settings["embedder_id"]
    chosen_name = _normal(custom) if custom else _registry_name(settings["model"])
    if not custom and settings["model"] == index_key:
        return True
    return chosen_name is not None and chosen_name == _registry_name(index_key)


def check_embedder_matches_index(args: argparse.Namespace, settings: dict, store) -> None:
    index_key = index_embedder_key(args.db_path, args.collection, store)
    if index_key is None:
        print(
            "[WARN] the index records no embedder id; only the vector size is checked",
            file=sys.stderr,
        )
        return
    if _same_embedder(settings, index_key):
        return
    chosen = settings["embedder_path"] or settings["embedder_id"] or settings["model"]
    message = (
        f"the index was built with '{index_key}' but the embedder is '{chosen}'; "
        f"pass --model {index_key}"
    )
    if getattr(args, "allow_embedder_mismatch", False):
        print(f"[WARN] {message} (continuing: --allow-embedder-mismatch)", file=sys.stderr)
        return
    sys.exit(f"error: {message}, or --allow-embedder-mismatch to continue anyway")


def _has_parents(store) -> bool:
    rows = store._collection.get(limit=PARENT_SAMPLE, include=["metadatas"])["metadatas"]
    return any("parent_id" in (meta or {}) for meta in rows)


def check_flags_have_an_effect(args: argparse.Namespace, settings: dict, store) -> None:
    ignored = [
        flag
        for flag, dest in (("--rerank-top-k", "rerank_top_k"), ("--reranker-model", "reranker"))
        if getattr(args, dest, None) is not None
    ]
    if ignored and not settings["rerank"]:
        verb = "has" if len(ignored) == 1 else "have"
        sys.exit(
            f"error: {' and '.join(ignored)} {verb} no effect because reranking is off; "
            "add --rerank"
        )
    if settings["resolve_parents"] and not _has_parents(store):
        sys.exit(
            "error: --resolve-parents needs an index built with the parent_child strategy; "
            "this index has no parent chunks"
        )
