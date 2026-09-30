"""Smoke test for every LOCALLY-hosted embedder in configs/embedding_models.json
(OpenRouter-served keys are skipped: "or-*" keys and entries with
"backend": "openrouter" -- see open_telco_rag/embedders/openrouter_embedder.py).

For each model: embeds --n-chunks sample document chunks (is_query=False)
and --n-queries sample queries (is_query=True -- exercises query_prefix/
query_prompt_name where configured), then reports dimension, a unit-norm
check (descriptive only -- most sentence-transformers models are NOT
unit-norm without normalize_embeddings=True, so False is expected, not a
failure), throughput, peak GPU memory (None if no GPU -- skipped
gracefully, not an error), and an extrapolated time for --extrapolate-n
chunks.

Usage (tiny inputs -- this is a smoke test, not a benchmark):
    python scripts/embedder_smoke_test.py --n-chunks 20 --n-queries 5
    python scripts/embedder_smoke_test.py --models minilm bge-m3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder

_EMBEDDING_MODELS_FILE = Path(__file__).resolve().parents[1] / "configs" / "embedding_models.json"


def local_model_keys() -> list[str]:
    models = json.loads(_EMBEDDING_MODELS_FILE.read_text(encoding="utf-8"))
    return [k for k, cfg in models.items()
            if not k.startswith("or-") and cfg.get("backend") != "openrouter"]


def _sample_texts(n: int, label: str) -> list[str]:
    return [
        f"{label} {i}: the AMF handles registration and mobility management for 5G user equipment."
        for i in range(n)
    ]


def unit_norm_check(vectors: list[list[float]]) -> tuple[bool, float]:
    """Descriptive only: True if every vector's L2 norm is within 5% of 1.0.
    Most sentence-transformers models do NOT normalize by default
    (normalize_embeddings=True isn't passed here), so False is expected for
    most models, not a bug."""
    if not vectors:
        return True, 0.0
    norms = [math.sqrt(sum(x * x for x in v)) for v in vectors]
    max_deviation = max(abs(n - 1.0) for n in norms)
    return max_deviation < 0.05, max_deviation


def run_one(model_key: str, n_chunks: int, n_queries: int, extrapolate_n: int) -> dict:
    import torch

    report: dict = {"model_key": model_key, "error": None}
    try:
        # tracking on so truncation_rate below is a real number (off by default)
        embedder = SentenceTransformerEmbedder(model_key=model_key, track_truncation=True)
        chunks = _sample_texts(n_chunks, "Document chunk")
        queries = _sample_texts(n_queries, "Query")

        # Warm-up: triggers model load + first-forward-pass overhead outside
        # the timed section, so throughput reflects steady-state encoding,
        # not load time (load alone can dominate a tiny smoke-test sample).
        embedder.embed(["warm-up"], is_query=False)

        has_cuda = torch.cuda.is_available()
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        chunk_vectors = embedder.embed(chunks, is_query=False)
        elapsed = time.perf_counter() - start

        query_vectors = embedder.embed(queries, is_query=True)

        report["dimension"] = len(chunk_vectors[0]) if chunk_vectors else 0
        report["expected_dimension"] = embedder.dimensions
        is_unit_norm, max_dev = unit_norm_check(chunk_vectors)
        report["unit_norm"] = is_unit_norm
        report["unit_norm_max_deviation"] = round(max_dev, 4)
        throughput = n_chunks / max(elapsed, 1e-9)
        report["throughput_chunks_per_sec"] = throughput
        report["extrapolated_seconds_for_n"] = extrapolate_n / max(throughput, 1e-9)
        report["max_seq_length"] = embedder.max_seq_length
        report["truncation_rate"] = embedder.truncation_rate
        report["n_query_vectors"] = len(query_vectors)
        report["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if has_cuda else None
        )
    except Exception as e:  # noqa: BLE001
        report["error"] = str(e)

    return report


def print_report(report: dict, extrapolate_n: int) -> None:
    print(f"== {report['model_key']} ==")
    if report["error"]:
        print(f"  [ERROR] {report['error']}\n")
        return
    print(f"  dimension:       {report['dimension']} (expected {report['expected_dimension']})")
    print(f"  unit norm:       {report['unit_norm']} (max deviation {report['unit_norm_max_deviation']})")
    print(f"  throughput:      {report['throughput_chunks_per_sec']:.1f} chunks/s")
    print(f"  extrapolated:    {extrapolate_n:,} chunks -> {report['extrapolated_seconds_for_n'] / 60:.1f} min")
    print(f"  max_seq_length:  {report['max_seq_length']}  truncation_rate: {report['truncation_rate']:.1%}")
    if report["peak_gpu_memory_mb"] is not None:
        print(f"  peak GPU memory: {report['peak_gpu_memory_mb']:.1f} MB")
    else:
        print("  peak GPU memory: N/A (no GPU available -- skipped gracefully)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None, help="default: every local model in the registry")
    parser.add_argument("--n-chunks", type=int, default=200)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--extrapolate-n", type=int, default=100_000)
    args = parser.parse_args()

    models = args.models or local_model_keys()
    print(f"Smoke-testing {len(models)} local embedder(s): {models}\n")

    for model_key in models:
        report = run_one(model_key, args.n_chunks, args.n_queries, args.extrapolate_n)
        print_report(report, args.extrapolate_n)


if __name__ == "__main__":
    main()
