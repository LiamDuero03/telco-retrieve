"""Estimate the $ cost of (re-)embedding an already-ingested ChromaDB
collection via OpenRouter's embeddings API, for every model in
configs/embedding_models.json.

Real chunk text -> real tiktoken counts -> per-token OpenRouter pricing.
Models with no price in the table are listed as "no price" (local-only models
such as the otel-* fine-tunes, or OpenRouter models missing from the snapshot
-- use --refresh-pricing to fetch live prices).

Usage:
    python scripts/estimate_embedding_cost.py --db-path chroma_db
    python scripts/estimate_embedding_cost.py --db-path chroma_db --collection <collection-name>
    python scripts/estimate_embedding_cost.py --db-path chroma_db --refresh-pricing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken

EMBEDDING_MODELS_FILE = Path(__file__).parent.parent / "configs" / "embedding_models.json"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/models?output_modalities=embeddings"

# snapshot of OpenRouter's per-token USD input price (pulled 2026-09-29 from
# OPENROUTER_EMBEDDINGS_URL), keyed by lowercased model_name. it goes stale and
# is incomplete (e.g. no qwen/qwen3-embedding-8b): use --refresh-pricing
PRICING_SNAPSHOT_2026_09_29: dict[str, float] = {
    "sentence-transformers/all-minilm-l6-v2":  0.000000005,
    "sentence-transformers/all-mpnet-base-v2": 0.000000005,
    "intfloat/e5-large-v2":                    0.00000001,
    "baai/bge-large-en-v1.5":                  0.00000001,
    "baai/bge-m3":                             0.00000001,
    "openai/text-embedding-3-small":           0.00000002,
    "openai/text-embedding-3-large":           0.00000013,
}


def _count_tokens(text: str) -> int:
    """cl100k_base token count; same convention as evaluate.py's
    _local_token_count() so this stays comparable to the rest of the
    pipeline's own token accounting."""
    if not text:
        return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def _fetch_live_pricing() -> dict[str, float]:
    import requests
    resp = requests.get(OPENROUTER_EMBEDDINGS_URL, timeout=30)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    return {
        m["id"].lower(): float(m["pricing"]["prompt"])
        for m in models
        if m.get("pricing", {}).get("prompt") is not None
    }


def _collection_stats(db_path: str, collection_name: str, batch_size: int) -> tuple[int, int]:
    """(n_chunks, total_tokens) for one collection, paginated so a
    598k-chunk collection doesn't need to sit in memory all at once."""
    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)

    n_chunks = collection.count()
    total_tokens = 0
    offset = 0
    while offset < n_chunks:
        batch = collection.get(include=["documents"], limit=batch_size, offset=offset)
        for doc in batch["documents"]:
            total_tokens += _count_tokens(doc)
        offset += batch_size
        print(f"    scanned {min(offset, n_chunks)}/{n_chunks} chunks...", end="\r")
    print()
    return n_chunks, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="chroma_db")
    parser.add_argument("--collection", default=None,
                         help="One collection name. Omit to scan every collection in --db-path.")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--refresh-pricing", action="store_true",
                         help="Re-fetch live OpenRouter pricing instead of using the baked-in snapshot.")
    args = parser.parse_args()

    if args.refresh_pricing:
        try:
            pricing = _fetch_live_pricing()
            print(f"  Using LIVE OpenRouter pricing ({len(pricing)} embedding models found).")
        except Exception as e:
            print(f"  [warn] live pricing fetch failed ({e}); falling back to the 2026-09-29 snapshot.")
            pricing = PRICING_SNAPSHOT_2026_09_29
    else:
        pricing = PRICING_SNAPSHOT_2026_09_29
        print("  Using the 2026-09-29 pricing snapshot (pass --refresh-pricing for live numbers).")

    embedding_models = json.loads(EMBEDDING_MODELS_FILE.read_text(encoding="utf-8"))

    import chromadb
    client = chromadb.PersistentClient(path=args.db_path)
    if args.collection:
        collection_names = [args.collection]
    else:
        collection_names = [c.name for c in client.list_collections()]
        if not collection_names:
            print(f"No collections found in '{args.db_path}'.")
            return

    for collection_name in collection_names:
        print(f"\n{'='*70}\nCollection: {collection_name}")
        n_chunks, total_tokens = _collection_stats(args.db_path, collection_name, args.batch_size)
        print(f"  chunks={n_chunks:,}  tokens={total_tokens:,}  "
              f"(mean {total_tokens / n_chunks:.0f} tokens/chunk)" if n_chunks else "  empty collection")
        if not n_chunks:
            continue

        print(f"\n  {'model_key':<22} {'model_name':<38} {'$/1M tok':>10} {'est. cost':>12}")
        print(f"  {'-'*84}")
        for model_key, cfg in embedding_models.items():
            model_name = cfg["model_name"]
            price = pricing.get(model_name.lower())
            if price is None:
                print(f"  {model_key:<22} {model_name:<38} {'no price':>10} {'--':>12}")
                continue
            price_per_million = price * 1_000_000
            cost = total_tokens * price
            print(f"  {model_key:<22} {model_name:<38} {price_per_million:>9.3f}$ {cost:>11.4f}$")


if __name__ == "__main__":
    main()
