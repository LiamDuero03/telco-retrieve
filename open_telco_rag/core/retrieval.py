"""Search implementation: dense/bm25/hybrid retrieval, reranking, and
parent-chunk resolution for hierarchical parent-child chunking."""
from __future__ import annotations
import math
from typing import Any
from open_telco_rag.types import SearchResult, Embedder
from open_telco_rag.stores.fts import FTSStore


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
) -> list[SearchResult]:
    """Run dense / bm25 / hybrid retrieval, then optional reranking and parent resolution."""
    search_method = search_method.lower()
    candidate_k = top_k * 3 if reranker is not None else top_k

    dense_results: list[SearchResult] = []
    bm25_results: list[SearchResult] = []

    # Dense vector pass
    if search_method in ["dense", "hybrid"]:
        vector = embedder.embed([query])[0]
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
        final_candidates = reranker.rerank(query, final_candidates, top_k=rerank_top_k)
    else:
        final_candidates = final_candidates[:top_k]

    # Expand to parent contexts
    if resolve_parents:
        final_candidates = _resolve_parent_chunks(final_candidates)

    return final_candidates