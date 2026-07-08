"""Algorithmic retrieval strategies — Vector, BM25, and Rank Fusion."""

from __future__ import annotations
import math
from typing import Any
from open_telco_rag.types import SearchResult, Embedder

def reciprocal_rank_fusion(dense_results: list[SearchResult], bm25_results: list[SearchResult], k: int = 60) -> list[SearchResult]:
    """Combines dense vector and sparse keyword scores using Reciprocal Rank Fusion."""
    fused_scores: dict[str, float] = {}
    
    for rank, item in enumerate(dense_results):
        fused_scores[item.id] = fused_scores.get(item.id, 0.0) + 1.0 / (k + rank + 1)
        
    for rank, item in enumerate(bm25_results):
        fused_scores[item.id] = fused_scores.get(item.id, 0.0) + 1.0 / (k + rank + 1)
        
    all_items = {item.id: item for item in dense_results + bm25_results}
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [all_items[idx] for idx, _ in sorted_ids]


def execute_retrieval(
    query:         str,
    store:         Any,
    embedder:      Embedder,
    top_k:         int   = 8,
    threshold:     float = 1.0,
    reranker             = None,
    rerank_top_k:  int   = 5,
    search_method: str   = "dense",
) -> list[SearchResult]:
    """Orchestrates search pipeline routing handling Dense, BM25 keyword, and Hybrid models."""
    search_method = search_method.lower()
    candidate_k = top_k * 3 if reranker is not None else top_k

    dense_results: list[SearchResult] = []
    bm25_results: list[SearchResult] = []

    # 1. Execute Dense Vector Match Pass
    if search_method in ["dense", "hybrid"]:
        vector = embedder.embed([query])[0]
        dense_results = store.search(vector, n_results=candidate_k)
        dense_results = [r for r in dense_results if r.distance < threshold]

    # 2. Execute BM25 Keyword Filter Pass
    if search_method in ["bm25", "hybrid"]:
        all_records = store._collection.get(include=["documents", "metadatas"])
        documents = all_records.get("documents", [])
        ids = all_records.get("ids", [])
        metadatas = all_records.get("metadatas", [])

        if documents:
            query_terms = [w.lower() for w in query.split() if len(w) > 1]
            doc_tokens = [[w.lower() for w in doc.split()] for doc in documents]
            avg_doc_len = sum(len(d) for d in doc_tokens) / len(doc_tokens) if doc_tokens else 1
            
            k1, b = 1.5, 0.75
            scores = []

            for idx, tokens in enumerate(doc_tokens):
                score = 0.0
                doc_len = len(tokens)
                for term in query_terms:
                    tf = tokens.count(term)
                    if tf > 0:
                        df = sum(1 for d in doc_tokens if term in d)
                        idf = math.log((len(doc_tokens) - df + 0.5) / (df + 0.5) + 1.0)
                        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                if score > 0:
                    scores.append((idx, score))

            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:candidate_k]
            
            for idx, score in sorted_scores:
                pseudo_distance = 1.0 / (1.0 + score)
                bm25_results.append(SearchResult(
                    id=ids[idx],
                    text=documents[idx],
                    metadata=metadatas[idx] if metadatas else {},
                    distance=pseudo_distance
                ))

    # 3. Apply Multi-Method Fusion Selection
    if search_method == "dense":
        final_candidates = dense_results
    elif search_method == "bm25":
        final_candidates = bm25_results
    elif search_method == "hybrid":
        final_candidates = reciprocal_rank_fusion(dense_results, bm25_results)
    else:
        final_candidates = dense_results

    # 4. Handle Cross-Encoder Reranker Passes
    if reranker is not None and final_candidates:
        final_candidates = reranker.rerank(query, final_candidates, top_k=rerank_top_k)
    else:
        final_candidates = final_candidates[:top_k]

    return final_candidates