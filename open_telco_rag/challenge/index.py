from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from open_telco_rag.stores.chroma import ChromaStore


class IndexProblem(ValueError):
    pass


def _collection_names(db: Path) -> list[str]:
    import chromadb

    listed = chromadb.PersistentClient(path=str(db)).list_collections()
    return sorted(c if isinstance(c, str) else c.name for c in listed)


def _fts_rows(fts: Path) -> int:
    connection = sqlite3.connect(f"file:{fts}?mode=ro", uri=True)
    try:
        return connection.execute("SELECT count(*) FROM bm25_index").fetchone()[0]
    except sqlite3.DatabaseError as error:
        raise IndexProblem(f"{fts} has no usable bm25_index table ({error})") from error
    finally:
        connection.close()


def _check_bm25_side(db: Path, collection: str, search_method: str, chunks: int) -> None:
    fts = db / f"{collection}_fts.db"
    if not fts.is_file():
        raise IndexProblem(
            f"search method '{search_method}' needs the BM25 database {fts}, which is missing. "
            f"The index has no "
            f"keyword side; use --search-method dense, or point --db-path at a database that ships "
            f"{collection}_fts.db"
        )
    rows = _fts_rows(fts)
    if rows == 0:
        raise IndexProblem(
            f"{fts} holds no rows, so '{search_method}' search would silently be dense only"
        )
    if rows != chunks:
        print(
            f"[WARN] {fts.name} has {rows} rows but the collection has {chunks} chunks",
            file=sys.stderr,
        )


def open_index(db_path: str | Path, collection: str, search_method: str) -> ChromaStore:
    """Open an existing collection without creating anything; fail if a part is missing."""
    db = Path(db_path)
    if not (db / "chroma.sqlite3").is_file():
        raise IndexProblem(f"no Chroma database at {db} (chroma.sqlite3 not found)")
    names = _collection_names(db)
    if collection not in names:
        raise IndexProblem(
            f"collection '{collection}' not found in {db}; available: {names or 'none'}"
        )
    store = ChromaStore(path=db, collection_name=collection)
    chunks = store.count()
    if chunks == 0:
        raise IndexProblem(f"collection '{collection}' in {db} is empty")
    if search_method != "dense":
        _check_bm25_side(db, collection, search_method, chunks)
    return store
