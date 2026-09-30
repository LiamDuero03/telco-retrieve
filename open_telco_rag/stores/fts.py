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
        self._conn = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Returns or creates a persistent SQLite connection handle."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self._conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = OFF;")
            
            # 128MB cache speeds up B-tree index construction
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

    def existing_hashes(self) -> dict[str, int]:
        """doc_id -> hash of its text for every indexed row; hashes are only comparable within one process"""
        return {doc_id: hash(text) for doc_id, text in self._get_conn().execute("SELECT doc_id, text FROM bm25_index")}

    def remove_exact_duplicates(self) -> tuple[int, int]:
        """delete rows that repeat both the doc_id and the text of an earlier row (left by an ingest that was
        killed and resumed). returns (rows removed, doc_ids still repeated with different text, left alone)"""
        conn = self._get_conn()
        total, distinct = conn.execute("SELECT COUNT(*), COUNT(DISTINCT doc_id) FROM bm25_index").fetchone()
        if total == distinct:
            return 0, 0
        conn.execute("DELETE FROM bm25_index WHERE rowid NOT IN (SELECT MIN(rowid) FROM bm25_index GROUP BY doc_id, text)")
        conn.commit()
        left, distinct = conn.execute("SELECT COUNT(*), COUNT(DISTINCT doc_id) FROM bm25_index").fetchone()
        return total - left, left - distinct

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
            # drop+recreate is O(1) vs. O(N) row-by-row deletion
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