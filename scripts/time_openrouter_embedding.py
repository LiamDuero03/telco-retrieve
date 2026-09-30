"""Measure OpenRouter embedding throughput on real chunk texts.

Same idea as time_embedding_throughput.py (local), but for OpenRouterEmbedder:
network-bound (HTTP batches of up to batch_size texts, concurrent workers),
including real API latency and rate limiting.

Makes REAL, PAID API calls (one per batch of --sample-size texts): keep the
sample small. Needs OPENROUTER_API_KEY in the environment or a .env file.

Usage:
    python scripts/time_openrouter_embedding.py --db-path chroma_db \
        --collection <collection-name> --model-key or-bge-m3 --sample-size 500
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--model-key", default="or-bge-m3",
                         help="An OpenRouter-served key in configs/embedding_models.json "
                              "(e.g. or-bge-m3, or-3-small, qwen3-embedding-8b).")
    parser.add_argument("--sample-size", type=int, default=3000,
                         help="Needs to be >> batch_size * max_workers to actually saturate "
                              "concurrency (e.g. 200*10=2000) -- a smaller sample under-measures "
                              "real throughput since most workers sit idle.")
    parser.add_argument("--batch-size", type=int, default=200,
                         help="Texts per HTTP request. Passed straight to OpenRouterEmbedder.")
    parser.add_argument("--max-workers", type=int, default=10,
                         help="Concurrent in-flight batch requests. Passed straight to "
                              "OpenRouterEmbedder. Higher isn't free -- OpenRouter's own rate "
                              "limiting (429 + backoff, already handled internally) can cap or "
                              "even hurt real gains past some point; this is for finding that "
                              "point empirically, not assuming higher is always better.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    import chromadb
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(args.collection)
    total_chunks = collection.count()
    if not total_chunks:
        raise SystemExit(f"collection '{args.collection}' is empty")

    sample_size = min(args.sample_size, total_chunks)
    print(f"Pulling {sample_size} real chunk texts from '{args.collection}' "
          f"({total_chunks} total)...")
    batch = collection.get(include=["documents"], limit=sample_size)
    texts = batch["documents"]

    from open_telco_rag.embedders.openrouter_embedder import OpenRouterEmbedder
    embedder = OpenRouterEmbedder(model_key=args.model_key, batch_size=args.batch_size,
                                   max_workers=args.max_workers)

    n_batches = -(-len(texts) // args.batch_size)  # ceil div
    if n_batches < args.max_workers:
        print(f"  [warn] {len(texts)} texts / batch_size {args.batch_size} = only {n_batches} "
              f"batch(es) -- fewer than max_workers={args.max_workers}, so concurrency won't be "
              f"fully exercised. Raise --sample-size for a trustworthy reading.")

    print(f"\nWarming up '{args.model_key}' (first HTTP connection/TLS handshake, "
          f"excluded from the timed measurement below)...")
    embedder.embed(texts[:1])

    print(f"\nTiming steady-state embed of {len(texts)} real chunks via OpenRouter "
          f"('{args.model_key}', batch_size={embedder.batch_size}, "
          f"max_workers={embedder.max_workers}, {n_batches} batches)...")
    embed_start = time.perf_counter()
    embedder.embed(texts)
    embed_seconds = time.perf_counter() - embed_start

    texts_per_sec = len(texts) / embed_seconds
    extrapolated_seconds = total_chunks / texts_per_sec

    print(f"\n{'='*70}")
    print(f"  Sample size:            {len(texts):,} chunks")
    print(f"  Embed time (sample):    {embed_seconds:.1f}s  ({texts_per_sec:.1f} chunks/s)")
    print(f"  Real total in collection: {total_chunks:,} chunks")
    print(f"  Extrapolated OpenRouter embed time for the FULL collection: "
          f"{extrapolated_seconds:.0f}s  (~{extrapolated_seconds / 60:.1f} min)")
    print(f"  NOTE: network-bound, real API call -- includes real latency/rate-limit "
          f"behaviour, not a local compute measurement. Excludes file parsing/chunking/DB writes.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
