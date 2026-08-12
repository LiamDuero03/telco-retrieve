"""ChromaDB vector store backend."""

from __future__ import annotations

from pathlib import Path

from open_telco_rag.types import SearchResult


class ChromaStore:
    """Persistent ChromaDB vector store."""

    def __init__(
        self,
        path:            str | Path = "chroma_db",
        collection_name: str        = "documents",
        metadata:        dict | None = None,  
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

        self.path            = str(path)
        self.collection_name = collection_name
        self._client         = chromadb.PersistentClient(path=self.path)


        self._collection     = self._client.get_or_create_collection(
            name=collection_name,
            metadata=metadata
        )

        # Queried once at construction rather than hardcoded: Chroma raises
        # ValueError("Batch size of N is greater than max batch size of M")
        # if a single add()/upsert() call exceeds this, and M isn't a fixed
        # constant across chromadb/sqlite versions (it's derived from
        # sqlite3's SQLITE_MAX_VARIABLE_NUMBER, which changed in newer
        # sqlite builds) -- callers that need to split large writes into
        # sub-batches (see ingest()/ingest_multi_enrichment() in
        # core/pipeline.py) must size against the real value, not a guess.
        self.max_batch_size  = self._client.get_max_batch_size()

    def add(
        self,
        ids:        list[str],
        embeddings: list[list[float]],
        documents:  list[str],
        metadatas:  list[dict],
    ) -> None:
        """Upserts (not insert-only) — re-adding an existing id overwrites it."""
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        embedding: list[float],
        n_results: int = 5,
    ) -> list[SearchResult]:
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(
                text=doc,
                distance=dist,
                metadata=meta,
                doc_id=doc_id,
            )
            for doc, dist, meta, doc_id in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
                results["ids"][0],
            )
        ]

    def count(self) -> int:
        return self._collection.count()

    def exists(self, id: str) -> bool:
        results = self._collection.get(ids=[id])
        return len(results["ids"]) > 0