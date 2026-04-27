"""Cross-encoder reranker for post-retrieval scoring."""

from __future__ import annotations

from open_telco_rag.types import SearchResult


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model.

    Unlike bi-encoder embedding models which encode query and chunk
    independently, a cross-encoder takes both together and scores
    their relevance jointly — significantly more accurate but slower.
    Use after initial retrieval to rerank a larger candidate set down
    to a smaller, higher-quality set.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model      = None  # lazy load

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install with: pip install open-telco-rag[local-embed]"
                )
            print(f"Loading reranker model: {self._model_name}")
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query:   str,
        results: list[SearchResult],
        top_k:   int = 5,
    ) -> list[SearchResult]:
        """Rerank results by cross-encoder score and return top_k."""
        if not results:
            return results

        self._load()

        # Score each (query, chunk) pair jointly
        pairs  = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)

        # Attach rerank score to metadata and sort
        for result, score in zip(results, scores):
            result.metadata["rerank_score"] = round(float(score), 4)

        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [r for r, _ in reranked[:top_k]]