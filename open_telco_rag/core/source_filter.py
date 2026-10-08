"""Restrict retrieval to chunks of some source types (the `source_type` chunk metadata).

Dense search hands a metadata `where` to Chroma. The BM25 index only holds (doc_id, text), and
its schema and ids stay as they are, so BM25 hits are fetched with some headroom, their metadata
is read from Chroma by id and the hits are post-filtered in Python (a Chroma `get(ids=..., where=...)`
was 40x slower for a common type on 20,000 ids). The headroom is
BM25_OVERFETCH times the number of candidates wanted: a source that is rare in the collection
can still leave the list short, which is reported.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

BM25_OVERFETCH = 10


class SourceTypeError(ValueError):
    """a requested source type is not in the collections"""


def chroma_where(source_types: list[str] | None) -> dict | None:
    if not source_types:
        return None
    if len(source_types) == 1:
        return {"source_type": source_types[0]}
    return {"source_type": {"$in": list(source_types)}}


def check_source_types(stores: list[Any], source_types: list[str]) -> None:
    """raise SourceTypeError, listing what exists, unless every type occurs in some store"""
    def present(store, source_type):
        return bool(store._collection.get(where={"source_type": source_type}, limit=1, include=[])["ids"])

    unknown = [t for t in source_types if not any(present(s, t) for s in stores)]
    if not unknown:
        return
    available = sorted({m["source_type"] for s in stores
                        for m in s._collection.get(include=["metadatas"])["metadatas"]
                        if m and m.get("source_type")})
    raise SourceTypeError(
        f"unknown source type {', '.join(repr(t) for t in unknown)}; "
        f"available: {', '.join(available) or 'none'}")


def warn_if_short(found: int, wanted: int, source_types: list[str] | None) -> None:
    if source_types and found < wanted:
        logger.warning(
            "source type filter (%s): only %d of %d results match. Dense search returns every match "
            "it has; BM25 looks at the best %dx candidates before filtering, so a rare source can "
            "come up short.", ", ".join(source_types), found, wanted, BM25_OVERFETCH)
