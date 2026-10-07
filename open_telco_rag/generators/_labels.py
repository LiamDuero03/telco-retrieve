"""Opt-in source labels for ask: one-line labels in front of each passage, a system prompt label
built from what was retrieved, and a per-passage Sources list."""
from __future__ import annotations

from open_telco_rag.types import SearchResult


def _rel_path(result: SearchResult) -> str:
    """the file path inside the data directory, from the chunk id (path::model::index)"""
    meta = result.metadata or {}
    parts = (result.doc_id or "").rsplit("::", 2)
    path = parts[0] if len(parts) == 3 else meta.get("source", "")
    return path.replace("\\", "/")


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def passage_label(result: SearchResult, default_collection: str | None = None) -> str:
    meta = result.metadata or {}
    fields = [
        ("collection", meta.get("collection", default_collection)),
        ("source_type", meta.get("source_type")),
        ("file", _rel_path(result)),
        ("release", meta.get("release")),
        ("version", None if meta.get("release") else meta.get("version") or meta.get("api_version")),
        ("heading", meta.get("heading")),
    ]
    return " | ".join(f"{key}: {value}" for key, value in fields if value)


def label_results(results: list[SearchResult], default_collection: str | None = None) -> list[SearchResult]:
    return [SearchResult(text=f"[{passage_label(r, default_collection)}]\n{r.text}", distance=r.distance,
                         metadata=r.metadata, doc_id=r.doc_id) for r in results]


def system_label(results: list[SearchResult], default_collection: str | None = None) -> str | None:
    """'3gpp and camara': the source types in context order, else the collection names, else None"""
    for key in ("source_type", "collection"):
        names = list(dict.fromkeys(
            (r.metadata or {}).get(key, default_collection if key == "collection" else None) for r in results))
        names = [n for n in names if n]
        if names:
            return _join(names)
    return None


def source_entries(results: list[SearchResult], default_collection: str | None = None) -> list[str]:
    """one entry per passage: 'collection: file'; the file's path instead of its name when
    the retrieved passages hold two different files of the same name in one collection"""
    def key(r):
        return (r.metadata or {}).get("collection", default_collection), (r.metadata or {}).get("source", "")

    paths: dict[tuple, set[str]] = {}
    for r in results:
        paths.setdefault(key(r), set()).add(_rel_path(r))
    entries = []
    for r in results:
        collection, name = key(r)
        file = _rel_path(r) if len(paths[key(r)]) > 1 else name
        entries.append(f"{collection}: {file}" if collection else file)
    return entries
