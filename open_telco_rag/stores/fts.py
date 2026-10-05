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

    def load_rows(self) -> dict[str, list[tuple[int, int]]]:
        """doc_id -> [(rowid, hash of its text)] for every indexed row, read once; sync() works from this in memory.
        hashes are only comparable within one process"""
        state: dict[str, list[tuple[int, int]]] = {}
        for rowid, doc_id, text in self._get_conn().execute("SELECT rowid, doc_id, text FROM bm25_index"):
            state.setdefault(doc_id, []).append((rowid, hash(text)))
        return state

    def sync(self, ids: list[str], documents: list[str], state: dict[str, list[tuple[int, int]]]) -> int:
        """make the index mirror these (id, text) pairs: a row whose text is unchanged is kept, rows of the id with
        any other text (and repeats of the same text) are deleted by rowid, and a missing text is inserted.
        `state` is what load_rows() returned; it is updated, so calls can follow each other. one transaction, no
        lookup in the table. returns how many rows were deleted"""
        stale: list[int] = []
        insert: list[tuple[str, str]] = []
        for doc_id, text in zip(ids, documents):
            digest = hash(text)
            rows = state.get(doc_id, [])
            keep = next((row for row in rows if row[1] == digest), None)
            stale.extend(rowid for rowid, h in rows if (rowid, h) != keep and rowid is not None)
            if keep is None:
                insert.append((doc_id, text))
                state[doc_id] = [(None, digest)]          # inserted below; a later write of this id replaces it in memory only
            else:
                state[doc_id] = [keep]
        if not stale and not insert:
            return 0
        conn = self._get_conn()
        with conn:
            for start in range(0, len(stale), 500):
                chunk = stale[start:start + 500]
                conn.execute(f"DELETE FROM bm25_index WHERE rowid IN ({','.join('?' * len(chunk))})", chunk)
            if insert:
                conn.executemany("INSERT INTO bm25_index(doc_id, text) VALUES (?, ?)", insert)
        return len(stale)

    def repeated_row_count(self) -> int:
        """rows beyond the first of each doc_id (rows that repeat an id)"""
        total, distinct = self._get_conn().execute("SELECT COUNT(*), COUNT(DISTINCT doc_id) FROM bm25_index").fetchone()
        return total - distinct

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