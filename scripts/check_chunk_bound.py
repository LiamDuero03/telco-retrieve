"""Report how chunks measure against a token bound, per tokenizer. Read-only: nothing is written or changed.

Two inputs:
  --collection NAME [--db-path chroma_db]   the stored chunk texts of a Chroma collection, read straight from
                                            its sqlite file in immutable read-only mode (the collection is not opened
                                            through chromadb, so no file is touched)
  --data-dir DIR --source-type 3gpp         a staged directory, chunked in memory with the same load / clean /
                                            chunk_document calls as ingest (--strategy, --chunk-size, --overlap,
                                            --target-tokens, --max-tokens, --bound-tokens as for `ingest`)

For every tokenizer in --tokenizers (cl100k, a tiktoken:<encoding>, an embedding model key such as bge-m3, a
Hugging Face model name or a tokenizer.json path; only tokenizer files are downloaded, never weights) it prints
chunk count, mean / p95 / max tokens and the number of chunks over --bound minus --margin. "Text retained" is
the share of non-whitespace characters of the cleaned documents that appears in the chunks (above 100% means
overlap or repeated table headers); for a collection it needs --data-dir with the source files too.

Usage:
    python scripts/check_chunk_bound.py --collection 3gpp_bge --db-path chroma_db --tokenizers cl100k bge-m3 --bound 512
    python scripts/check_chunk_bound.py --data-dir data/3gpp/3GPP-clean/Rel-18 --source-type 3gpp \
        --strategy hierarchical_markdown --target-tokens 400 --bound-tokens 512 --tokenizers cl100k minilm qwen3-embedding-0.6b
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from pathlib import Path

from open_telco_rag.core.chunk_bound import get_counter

_QUERY = """select e.embedding_id, m.string_value from embeddings e
            join segments s on s.id = e.segment_id and s.scope = 'METADATA'
            join collections c on c.id = s.collection
            join embedding_metadata m on m.id = e.id and m.key = 'chroma:document'
            where c.name = ?"""


def squash_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def read_collection(db_path: Path, name: str) -> list[tuple[str, str]]:
    """(chunk id, text) for a collection; the file is opened read-only and immutable, so it is never modified"""
    sqlite_file = (db_path / "chroma.sqlite3").resolve()
    if not sqlite_file.exists():
        raise SystemExit(f"error: {sqlite_file} not found")
    con = sqlite3.connect(f"file:{sqlite_file.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        rows = con.execute(_QUERY, (name,)).fetchall()
    finally:
        con.close()
    if not rows:
        raise SystemExit(f"error: collection {name!r} not found or empty in {sqlite_file}")
    return rows


def staged_documents(args) -> dict[str, str]:
    from cli.main import _get_source
    source = _get_source(args.source_type, use_marker=False)
    data_dir = Path(args.data_dir)
    docs = {}
    for ext in source.supported_extensions():
        for path in sorted(data_dir.rglob(f"*{ext}")):
            text = source.load(path)
            if text and text.strip():
                docs[str(path.relative_to(data_dir))] = source.clean(text)
    if not docs:
        raise SystemExit(f"error: no {source.supported_extensions()} files under {data_dir}")
    return docs


def chunk_staged(docs: dict[str, str], args) -> list[tuple[str, str]]:
    from open_telco_rag.core.chunking import chunk_document
    kwargs = {"strategy": args.strategy, "source_type": args.source_type}
    for key in ("chunk_size", "overlap", "target_tokens", "max_tokens", "overlap_tokens", "bound_tokens", "bound_tokenizer", "bound_margin"):
        if getattr(args, key) is not None:
            kwargs[key] = getattr(args, key)
    if args.target_tokens is None and args.max_tokens is None:
        kwargs.setdefault("chunk_size", 1024)
        kwargs.setdefault("overlap", 200)
    out = []
    for rel, text in docs.items():
        for i, chunk in enumerate(chunk_document(text, **kwargs)):
            if len(chunk.text.strip()) >= args.min_length:
                out.append((f"{rel}::{i}", chunk.text))
    return out


def retention(chunks: list[tuple[str, str]], docs: dict[str, str]) -> float | None:
    """non-whitespace characters of the chunks of the files present in docs over those of the documents"""
    norm = {re.sub(r"[\\/]+", "/", rel): text for rel, text in docs.items()}
    by_file: dict[str, int] = {}
    for chunk_id, text in chunks:
        rel = re.sub(r"[\\/]+", "/", chunk_id.split("::")[0])
        if rel in norm:
            by_file[rel] = by_file.get(rel, 0) + squash_len(text)
    total = sum(squash_len(norm[rel]) for rel in by_file)
    return sum(by_file.values()) / total if total else None


def measure(chunks: list[tuple[str, str]], tokenizer: str, limit: int) -> dict:
    count = get_counter(None if tokenizer in ("cl100k", "cl100k_base") else tokenizer)
    sizes = sorted(count(text) for _, text in chunks)
    p95 = sizes[min(len(sizes) - 1, int(0.95 * len(sizes)))]
    return {"tokenizer": tokenizer, "chunks": len(sizes), "mean": statistics.fmean(sizes), "p95": p95, "max": sizes[-1],
            "over_bound": sum(1 for n in sizes if n > limit)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection")
    p.add_argument("--db-path", default="chroma_db")
    p.add_argument("--data-dir", help="staged source directory (also the source files when --collection is used)")
    p.add_argument("--source-type", default="3gpp")
    p.add_argument("--tokenizers", nargs="+", default=["cl100k"])
    p.add_argument("--bound", type=int, default=512, help="the model's token limit (default 512)")
    p.add_argument("--margin", type=int, default=2, help="tokens reserved for special tokens (default 2)")
    p.add_argument("--min-length", type=int, default=100, help="staged mode: drop chunks shorter than this many characters, as ingest does")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    for name in ("strategy",):
        p.add_argument(f"--{name}", default="text_baseline")
    for name in ("chunk-size", "overlap", "target-tokens", "max-tokens", "overlap-tokens", "bound-tokens", "bound-margin"):
        p.add_argument(f"--{name}", type=int, default=None)
    p.add_argument("--bound-tokenizer", default=None)
    args = p.parse_args()
    if not args.collection and not args.data_dir:
        p.error("give --collection or --data-dir")

    docs = staged_documents(args) if args.data_dir else None
    chunks = read_collection(Path(args.db_path), args.collection) if args.collection else chunk_staged(docs, args)
    limit = args.bound - args.margin
    results = []
    for t in args.tokenizers:
        try:
            results.append(measure(chunks, t, limit))
        except Exception as e:  # noqa: BLE001 - one unavailable tokenizer must not hide the others
            results.append({"tokenizer": t, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"})
    kept = retention(chunks, docs) if docs else None

    if args.json:
        print(json.dumps({"chunks": len(chunks), "limit": limit, "text_retained": kept, "tokenizers": results}, indent=2))
        return
    source = f"collection {args.collection!r}" if args.collection else f"staged {args.data_dir}"
    print(f"{source}: {len(chunks):,} chunks, bound {args.bound} minus margin {args.margin} = {limit} tokens")
    print(f"{'tokenizer':32} {'chunks':>8} {'mean':>8} {'p95':>6} {'max':>6} {'over bound':>12}")
    for r in results:
        if "error" in r:
            print(f"{r['tokenizer'][:32]:32} unavailable: {r['error']}")
            continue
        print(f"{r['tokenizer'][:32]:32} {r['chunks']:>8,} {r['mean']:>8.1f} {r['p95']:>6} {r['max']:>6} {r['over_bound']:>6} ({r['over_bound'] / r['chunks']:.1%})")
    print("text retained: " + (f"{kept:.1%} of the non-whitespace characters of the cleaned documents" if kept is not None
                               else "n/a (give --data-dir with the source files to compare)"))


if __name__ == "__main__":
    main()
