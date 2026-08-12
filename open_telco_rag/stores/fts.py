"""SQLite FTS5 inverted index sparse store (Optimized for High-Speed Disk I/O)."""

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
        
        self.path.mkdir(parents=True, exist_ok=True)
        self._conn = None  # Persistent connection handle
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Returns or creates a persistent SQLite connection handle."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self._conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = OFF;")
            
            # OPTIMIZATION: Allocate 128MB CPU RAM cache for B-tree index construction
            cursor.execute("PRAGMA cache_size = -128000;")
            cursor.execute("PRAGMA temp_store = MEMORY;")
            self._conn.commit()
        return self._conn

    def _init_db(self) -> None:
        """Create FTS5 table with optimized WAL mode and memory settings."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS bm25_index 
            USING fts5(doc_id UNINDEXED, text);
        """)
        conn.commit()

    def add(self, ids: list[str], documents: list[str]) -> None:
        """Bulk insert documents using the persistent connection handle."""
        if not ids or not documents:
            return
            
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO bm25_index(doc_id, text) VALUES (?, ?)",
            zip(ids, documents)
        )
        conn.commit()

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Instant O(terms) BM25 lookup returning a list of (doc_id, score) tuples."""
        if not self.db_path.exists():
            return []

        import re
        clean_words = re.findall(r'\w+', query)
        if not clean_words:
            return []

        fts_query = " OR ".join([f'"{w}"' for w in clean_words])

        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
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

        return results

    def clear(self) -> None:
        """Drop all indexed records instantly by dropping and recreating the virtual table."""
        if self.db_path.exists():
            conn = self._get_conn()
            cursor = conn.cursor()
            # OPTIMIZATION: Instant O(1) table reset instead of slow O(N) row deletions
            cursor.execute("DROP TABLE IF EXISTS bm25_index;")
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS bm25_index 
                USING fts5(doc_id UNINDEXED, text);
            """)
            conn.commit()


    def delete(self) -> None:
        """Permanently delete the database file from disk."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self.db_path.exists():
            self.db_path.unlink()