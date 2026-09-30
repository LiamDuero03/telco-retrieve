"""Measure local embedding throughput on real chunk texts.

Runs stored chunk texts from an existing ChromaDB collection through the same
SentenceTransformerEmbedder.embed() call ingestion uses, then extrapolates texts/sec
to the collection's full chunk count. Only the embedding stage is timed (no file
parsing, chunking or ChromaDB writes).

Usage:
    python scripts/time_embedding_throughput.py --db-path chroma_db --collection <collection-name> --model-key bge --sample-size 5000
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--model-key", default="bge")
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

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

    from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder(model_key=args.model_key)

    print(f"\nWarming up '{args.model_key}' (model load + first CUDA forward pass, excluded from the timed measurement below)...")
    embedder.embed(texts[:1])

    print(f"\nTiming steady-state embed of {len(texts)} real chunks with '{args.model_key}'...")
    embed_start = time.perf_counter()
    embedder.embed(texts)
    embed_seconds = time.perf_counter() - embed_start

    texts_per_sec = len(texts) / embed_seconds
    extrapolated_seconds = total_chunks / texts_per_sec

    print(f"\n{'='*70}")
    print(f"  Sample size:            {len(texts):,} chunks")
    print(f"  Embed time (sample):    {embed_seconds:.1f}s  ({texts_per_sec:.1f} chunks/s)")
    print(f"  Real total in collection: {total_chunks:,} chunks")
    print(f"  Extrapolated embed time for the FULL collection: "
          f"{extrapolated_seconds:.0f}s  (~{extrapolated_seconds / 60:.1f} min)")
    print("  NOTE: embedding-stage only -- excludes file parsing/chunking/DB writes.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
