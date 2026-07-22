"""SQLite FTS5 inverted index sparse vector/keyword store."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class FTSStore:
    """Persistent SQLite FTS5 Full-Text Search Store for BM25 retrieval."""

    def __init__(
        self,
        path: str | Path = "chroma_db",
        collection_name: str = "documents",
    ):
        self.path = Path(path)
        self.collection_name = collection_name
        self.db_path = self.path / f"{self.collection_name}_fts.db"
        
        # Ensure parent directory exists
        self.path.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create FTS5 virtual table if it doesn't exist."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS bm25_index 
            USING fts5(doc_id UNINDEXED, text);
        """)
        conn.commit()
        conn.close()

    def add(self, ids: list[str], documents: list[str]) -> None:
        """Bulk insert documents into the FTS5 inverted index."""
        if not ids or not documents:
            return
            
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO bm25_index(doc_id, text) VALUES (?, ?)",
            zip(ids, documents)
        )
        conn.commit()
        conn.close()

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Instant O(terms) BM25 lookup returning a list of (doc_id, score) tuples."""
        if not self.db_path.exists():
            return []

        import re
        # Sanitize query for FTS5 syntax (extract alphanumeric terms)
        clean_words = re.findall(r'\w+', query)
        if not clean_words:
            return []

        fts_query = " OR ".join([f'"{w}"' for w in clean_words])

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # SQLite bm25() returns negative values where lower/more negative is better.
            # We negate it (-bm25) so higher is a better score.
            cursor.execute("""
                SELECT doc_id, -bm25(bm25_index) AS score 
                FROM bm25_index 
                WHERE bm25_index MATCH ? 
                ORDER BY score DESC 
                LIMIT ?
            """, (fts_query, top_k))
            results = cursor.fetchall()
        except sqlite3.OperationalError:
            results = []
        finally:
            conn.close()

        return results

    def clear(self) -> None:
        """Drop all indexed records."""
        if self.db_path.exists():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bm25_index;")
            conn.commit()
            conn.close()

    def delete(self) -> None:
        """Permanently delete the database file from disk."""
        if self.db_path.exists():
            self.db_path.unlink()