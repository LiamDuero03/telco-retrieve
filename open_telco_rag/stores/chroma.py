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

        # Real value from the client; varies by sqlite build.
        self.max_batch_size  = self._client.get_max_batch_size()

    def add(
        self,
        ids:        list[str],
        embeddings: list[list[float]],
        documents:  list[str],
        metadatas:  list[dict],
    ) -> None:
        """Upserts (not insert-only) - re-adding an existing id overwrites it."""
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def update_metadata(self, ids: list[str], metadatas: list[dict]) -> None:
        """Merge metadata into already-stored ids in place (embeddings/
        documents untouched). Matches chromadb .update() semantics: pass only
        the keys to add/overwrite."""
        self._collection.update(ids=ids, metadatas=metadatas)

    def get_metadata(self, ids: list[str]) -> list[dict]:
        """Stored metadata for `ids`, re-mapped to the requested order. An
        unknown id comes back as {}."""
        result = self._collection.get(ids=ids, include=["metadatas"])
        by_id = dict(zip(result["ids"], result["metadatas"]))
        return [by_id.get(i, {}) for i in ids]

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

    def all_doc_ids(self) -> list[str]:
        """Every chunk id in this collection (cheap ids-only query)."""
        return self._collection.get(include=[])["ids"]

    def close(self) -> None:
        """Release the SQLite connection(s). Call before deleting a store's
        directory."""
        self._client.close()

    def __enter__(self) -> "ChromaStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()