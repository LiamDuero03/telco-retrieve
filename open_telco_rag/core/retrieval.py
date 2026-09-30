"""Search implementation: dense/bm25/hybrid retrieval, reranking, and
parent-chunk resolution for hierarchical parent-child chunking."""
from __future__ import annotations
import logging
import math
from typing import Any
from open_telco_rag.core.chunking import _count_tokens, _get_encoding
from open_telco_rag.embedders.calls import embed_query
from open_telco_rag.types import SearchResult, Embedder
from open_telco_rag.stores.fts import FTSStore

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_POOL = 30
DEFAULT_BUDGET_OVERFLOW = "stop"
_OVERFLOW_POLICIES = ("stop", "skip", "truncate")


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    enc = _get_encoding()
    return enc.decode(enc.encode(text)[:max_tokens])


def resolve_budget_args(
    context_token_budget: int | None,
    candidate_pool: int | None,
    budget_overflow: str | None,
) -> tuple[int | None, int | None, str | None]:
    """Validate the optional budget settings and fill in defaults.

    Returns (budget, pool, overflow); all None when no budget is set.
    candidate_pool and budget_overflow only mean something with a budget, so
    setting either one alone raises ValueError.
    """
    if context_token_budget is None:
        if candidate_pool is not None or budget_overflow is not None:
            raise ValueError(
                "candidate_pool and budget_overflow require context_token_budget; "
                "set a budget or leave both unset."
            )
        return None, None, None

    for name, value in (("context_token_budget", context_token_budget),
                        ("candidate_pool", candidate_pool)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if budget_overflow is not None and budget_overflow not in _OVERFLOW_POLICIES:
        raise ValueError(
            f"Unknown budget_overflow policy: {budget_overflow!r} (choose from {_OVERFLOW_POLICIES})"
        )
    return (
        context_token_budget,
        DEFAULT_CANDIDATE_POOL if candidate_pool is None else candidate_pool,
        DEFAULT_BUDGET_OVERFLOW if budget_overflow is None else budget_overflow,
    )


def fill_context_budget(
    candidates: list[SearchResult],
    budget: int,
    overflow: str = "stop",
    stats: dict | None = None,
) -> list[SearchResult]:
    """select a prefix of `candidates` (in rank order) that fits within `budget` tokens

    overflow, applied at the first candidate that does not fit:
      "stop"     halt, strict rank order (default)
      "skip"     leave it out and keep considering smaller ones
      "truncate" cut its text to fill the remaining budget, then stop

    stats, if given, is filled with candidates, kept, skipped, truncated,
    stopped_by_policy, pool_exhausted and empty. warns when every candidate is
    consumed without reaching the budget (candidate_pool too small)
    """
    if overflow not in _OVERFLOW_POLICIES:
        raise ValueError(f"Unknown budget_overflow policy: {overflow!r}")

    selected: list[SearchResult] = []
    used = 0
    pool_exhausted = True
    n_skipped = 0
    truncated = False

    for cand in candidates:
        cand_tokens = _count_tokens(cand.text)
        if used + cand_tokens <= budget:
            selected.append(cand)
            used += cand_tokens
            continue

        if overflow == "skip":
            n_skipped += 1
            continue

        if overflow == "truncate":
            remaining = budget - used
            truncated_text = _truncate_to_tokens(cand.text, remaining)
            if truncated_text:
                truncated = True
                selected.append(SearchResult(
                    text=truncated_text, distance=cand.distance,
                    metadata=cand.metadata, doc_id=cand.doc_id,
                ))
                used += _count_tokens(truncated_text)

        pool_exhausted = False  # stopped by policy, not by running out of candidates
        break

    if candidates and not selected:
        # nothing fit: the generator would silently get no context at all
        logger.warning(
            "fill_context_budget: the %d-token budget is smaller than the first candidate "
            "(%d tokens) under budget_overflow=%r, so the context is EMPTY. Raise the budget "
            "or use budget_overflow='truncate'.",
            budget, _count_tokens(candidates[0].text), overflow,
        )
    elif pool_exhausted and used < budget and candidates:
        logger.warning(
            "fill_context_budget: exhausted all %d candidate(s) at %d/%d "
            "tokens -- candidate_pool may be too small to fill this budget.",
            len(candidates), used, budget,
        )

    if stats is not None:
        stats.update(candidates=len(candidates), kept=len(selected), skipped=n_skipped, truncated=truncated,
                     stopped_by_policy=not pool_exhausted, pool_exhausted=pool_exhausted and used < budget,
                     empty=bool(candidates) and not selected)
    return selected


def _resolve_parent_chunks(results: list[SearchResult]) -> list[SearchResult]:
    """If results contain child chunks from hierarchical parent-child chunking,
    substitute child text with full parent text and deduplicate by parent_id.
    """
    resolved: list[SearchResult] = []
    seen_parents: set[str] = set()

    for r in results:
        meta = r.metadata or {}
        parent_id = meta.get("parent_id")
        full_parent_text = meta.get("full_parent_text")

        if parent_id and full_parent_text:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            if "parent_char_start" in meta:
                # token-sized parent_child: the text is now the parent's, so
                # its offsets are too (the span metrics read these)
                meta = {**meta, "char_start": meta["parent_char_start"],
                        "char_end": meta["parent_char_end"], "n_tokens": meta["parent_n_tokens"]}
            resolved.append(SearchResult(
                text=full_parent_text,
                distance=r.distance,
                metadata=meta,
                doc_id=r.doc_id,
            ))
        else:
            resolved.append(r)

    return resolved


def reciprocal_rank_fusion(dense_results: list[SearchResult], bm25_results: list[SearchResult], k: int = 60) -> list[SearchResult]:
    """Combines dense vector and sparse keyword scores using Reciprocal Rank Fusion."""
    fused_scores: dict[str, float] = {}
    
    for rank, item in enumerate(dense_results):
        fused_scores[item.doc_id] = fused_scores.get(item.doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    for rank, item in enumerate(bm25_results):
        fused_scores[item.doc_id] = fused_scores.get(item.doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    all_items = {item.doc_id: item for item in dense_results + bm25_results}
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [all_items[idx] for idx, _ in sorted_ids]


def execute_retrieval(
    query:           str,
    store:           Any,
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
    """Run dense / bm25 / hybrid retrieval, then optional reranking and parent resolution.

    with context_token_budget set, candidate_pool (default 30) replaces top_k and
    rerank_top_k as the pool size, and the budget is filled from it in rank order.
    candidate_pool / budget_overflow without a budget raise ValueError.
    """
    context_token_budget, candidate_pool, budget_overflow = resolve_budget_args(
        context_token_budget, candidate_pool, budget_overflow)
    search_method = search_method.lower()
    budgeted = context_token_budget is not None
    if budgeted:
        candidate_k = candidate_pool
    else:
        candidate_k = top_k * 3 if reranker is not None else top_k

    dense_results: list[SearchResult] = []
    bm25_results: list[SearchResult] = []

    # Dense vector pass
    if search_method in ["dense", "hybrid"]:
        vector = embed_query(embedder, query)
        dense_results = store.search(vector, n_results=candidate_k)
        
        if 0.0 < threshold < 1.0:
            space = "l2"
            if hasattr(store, "metadata") and store.metadata:
                space = store.metadata.get("hnsw:space", "l2").lower()
            elif hasattr(store, "_collection") and hasattr(store._collection, "metadata") and store._collection.metadata:
                space = store._collection.metadata.get("hnsw:space", "l2").lower()

            if space == "cosine":
                distance_limit = 1.0 - threshold
            else:
                distance_limit = (1.0 / threshold) - 1.0

            dense_results = [r for r in dense_results if r.distance < distance_limit]

    # BM25 keyword pass
    if search_method in ["bm25", "hybrid"]:
        store_path = getattr(store, "path", "chroma_db")
        store_col = getattr(store, "collection_name", "documents")
        
        fts_store = FTSStore(path=store_path, collection_name=store_col)
        bm25_hits = fts_store.search(query, top_k=candidate_k)

        if bm25_hits:
            hit_ids = [doc_id for doc_id, _ in bm25_hits]
            chroma_records = store._collection.get(ids=hit_ids, include=["documents", "metadatas"])
            
            id_to_doc = dict(zip(chroma_records["ids"], chroma_records["documents"]))
            id_to_meta = dict(zip(chroma_records["ids"], chroma_records["metadatas"]))
            
            for doc_id, score in bm25_hits:
                if doc_id in id_to_doc:
                    pseudo_distance = 1.0 / (1.0 + score)
                    bm25_results.append(SearchResult(
                        doc_id=doc_id,
                        text=id_to_doc[doc_id],
                        metadata=id_to_meta.get(doc_id, {}),
                        distance=pseudo_distance
                    ))


    # Fusion / selection
    if search_method == "dense":
        final_candidates = dense_results
    elif search_method == "bm25":
        final_candidates = bm25_results
    elif search_method == "hybrid":
        final_candidates = reciprocal_rank_fusion(dense_results, bm25_results)
    else:
        final_candidates = dense_results

    # Rerank
    if reranker is not None and final_candidates:
        rerank_n = candidate_pool if budgeted else rerank_top_k
        final_candidates = reranker.rerank(query, final_candidates, top_k=rerank_n)
    else:
        final_candidates = final_candidates[:candidate_pool] if budgeted else final_candidates[:top_k]

    # expand to parent contexts, deduplicated by parent_id (best-ranked child wins)
    if resolve_parents:
        final_candidates = _resolve_parent_chunks(final_candidates)

    if budgeted:
        final_candidates = fill_context_budget(final_candidates, context_token_budget, budget_overflow)

    return final_candidates