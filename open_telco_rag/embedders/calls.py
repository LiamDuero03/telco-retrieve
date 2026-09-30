"""Helpers for calling embedders that may or may not take the optional
is_query / apply_prefix arguments (custom embedders with embed(texts) work too)."""
from __future__ import annotations

import inspect
from typing import Any


def _accepts(embedder: Any, name: str) -> bool:
    try:
        params = inspect.signature(embedder.embed).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def embed_query(embedder: Any, query: str) -> list[float]:
    """embed a search query (applies the model's query prefix, if configured)"""
    if _accepts(embedder, "is_query"):
        return embedder.embed([query], is_query=True)[0]
    return embedder.embed([query])[0]


def embed_plain(embedder: Any, texts: list[str]) -> list[list[float]]:
    """embed texts for text-vs-text similarity (metrics): no query/document
    prefix, whichever model key is in use"""
    if _accepts(embedder, "apply_prefix"):
        return embedder.embed(texts, apply_prefix=False)
    return embedder.embed(texts)
