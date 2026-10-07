"""Search several collections of one database with one query embedding.

Per-collection ranked lists are fused by reciprocal rank fusion over ranks only. Scores are
never mixed: the same chunk scored 1.31 vs 0.66 by distance space alone (l2 vs cosine), and a
BM25 score depends on what else is in the collection, so neither is comparable across
collections.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_telco_rag.core.retrieval import execute_retrieval
from open_telco_rag.embedders.calls import embed_query
from open_telco_rag.evaluation.provenance import read_index_settings
from open_telco_rag.types import SearchResult

RRF_K = 60

_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "embedding_models.json"
_PROFILE_KEYS = ("model_name", "dimensions", "query_prefix", "document_prefix",
                 "query_prompt_name", "document_prompt_name")


class CollectionSetupError(ValueError):
    """the requested collections cannot be searched together"""


@dataclass
class OpenCollection:
    name: str
    store: Any
    space: str
    count: int
    dimension: int | None


def validate_collection_names(values: list[str] | None) -> list[str]:
    names = list(values or [])
    for name in names:
        if "," in name:
            raise CollectionSetupError(
                f"--collection takes one name, got {name!r}; repeat --collection for several "
                f"collections (--collection a --collection b)")
    for name in names:
        if names.count(name) > 1:
            raise CollectionSetupError(f"collection {name!r} was given more than once")
    return names


def embedding_profile(model_key: str) -> dict | None:
    """what the registry says a model key embeds with; None for an unknown key"""
    try:
        entry = json.loads(_REGISTRY.read_text(encoding="utf8")).get(model_key)
    except (OSError, ValueError):
        return None
    return {k: entry.get(k) for k in _PROFILE_KEYS} if entry else None


def _model_problem(name: str, recorded: str, wanted: str) -> str | None:
    if recorded == wanted:
        return None
    a, b = embedding_profile(recorded), embedding_profile(wanted)
    if a and b:
        if a == b:
            return None
        if a["model_name"] == b["model_name"]:
            return (f"collection {name!r} was ingested with {recorded!r}, which uses different "
                    f"prefixes than {wanted!r} (same underlying model {a['model_name']})")
    return f"collection {name!r} was ingested with model {recorded!r} but this search uses {wanted!r}"


def _stored_dimension(store: Any) -> int | None:
    got = store._collection.get(limit=1, include=["embeddings"])
    emb = got["embeddings"]
    return len(emb[0]) if emb is not None and len(emb) else None


def open_collections(db_path: str | Path, names: list[str], embedder: Any,
                     say=print) -> list[OpenCollection]:
    """open the named collections after checking they can share one query embedding.
    nothing is created: a missing database or collection is an error"""
    import chromadb

    from open_telco_rag.stores.chroma import ChromaStore

    if not Path(db_path).is_dir():
        raise CollectionSetupError(f"database {str(db_path)!r} does not exist")
    existing = [c.name for c in chromadb.PersistentClient(path=str(db_path)).list_collections()]
    for name in names:
        if name not in existing:
            raise CollectionSetupError(
                f"collection {name!r} does not exist in {str(db_path)!r} "
                f"(available: {', '.join(existing) or 'none'})")

    wanted = embedder.model_key
    opened: list[OpenCollection] = []
    for name in names:
        store = ChromaStore(path=db_path, collection_name=name)
        space = ((store._collection.metadata or {}).get("hnsw:space") or "l2").lower()
        opened.append(OpenCollection(name, store, space, store.count(), None))

    for c in opened:
        settings = read_index_settings(db_path, c.name)
        recorded = ((settings or {}).get("settings") or {}).get("model_key")
        if recorded:
            problem = _model_problem(c.name, recorded, wanted)
            if problem:
                raise CollectionSetupError(problem)
        else:
            say(f"[WARN] collection {c.name!r} has no ingest settings file, so its model and "
                f"prefixes cannot be checked; only its stored vector dimension is.")

    first = opened[0]
    for c in opened[1:]:
        if c.space != first.space:
            raise CollectionSetupError(
                f"collection {c.name!r} uses distance space {c.space!r} but {first.name!r} uses "
                f"{first.space!r}; collections can only be searched together when they agree")

    expected = getattr(embedder, "dimensions", None)
    for c in opened:
        if c.count == 0:
            say(f"[NOTE] collection {c.name!r} is empty and contributes nothing.")
            continue
        c.dimension = _stored_dimension(c.store)
    sized = [c for c in opened if c.dimension is not None]
    for c in sized:
        if expected is not None and c.dimension != expected:
            other = next((o for o in sized if o.dimension == expected), None)
            raise CollectionSetupError(
                f"embedder {wanted!r} produces {expected}-dimensional vectors but collection "
                f"{c.name!r} stores {c.dimension}-dimensional ones"
                + (f" (collection {other.name!r} stores {other.dimension})" if other else ""))
        if c.dimension != sized[0].dimension:
            raise CollectionSetupError(
                f"collections {sized[0].name!r} ({sized[0].dimension}-dimensional) and {c.name!r} "
                f"({c.dimension}-dimensional) cannot share embedder {wanted!r}")
    return opened


class _FixedQuery:
    """hands the one query vector to every per-collection retrieval"""

    def __init__(self, vector):
        self._vector = vector

    def embed(self, texts, is_query=False):
        if self._vector is None:
            raise RuntimeError("no query vector was computed")
        return [self._vector]


def _tag(result: SearchResult, collection: str) -> SearchResult:
    return SearchResult(text=result.text, distance=result.distance, doc_id=result.doc_id,
                        metadata={**result.metadata, "collection": collection})


def fuse_collections(ranked: list[list[SearchResult]], k: int = RRF_K) -> list[SearchResult]:
    """reciprocal rank fusion keyed by (collection, doc_id); ties keep the order the
    collections were given in. every result must carry metadata['collection']"""
    scores: dict[tuple[str, str], float] = {}
    items: dict[tuple[str, str], SearchResult] = {}
    for results in ranked:
        for rank, r in enumerate(results):
            key = (r.metadata["collection"], r.doc_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            items.setdefault(key, r)
    return [items[key] for key in sorted(scores, key=lambda key: -scores[key])]


def multi_search(
    query: str,
    stores: list[Any],
    embedder: Any,
    top_k: int = 8,
    reranker=None,
    rerank_top_k: int = 5,
    search_method: str = "dense",
    resolve_parents: bool = False,
) -> list[SearchResult]:
    """one query, one embedding, every store searched with search_method, lists fused by rank,
    then rerank / top_k / parent resolution as in a single-collection search"""
    from open_telco_rag.core.retrieval import _resolve_parent_chunks

    method = search_method.lower()
    vector = embed_query(embedder, query) if method != "bm25" else None
    shared = _FixedQuery(vector)
    depth = top_k * 3 if reranker is not None else top_k

    ranked = []
    for store in stores:
        hits = execute_retrieval(query, store, shared, top_k=depth, search_method=method)
        ranked.append([_tag(r, store.collection_name) for r in hits])
    fused = fuse_collections(ranked)

    if reranker is not None and fused:
        fused = reranker.rerank(query, fused[:depth], top_k=rerank_top_k)
    else:
        fused = fused[:top_k]
    if resolve_parents:
        fused = _resolve_parent_chunks(fused)
    return fused
