"""
Tier 1 retrieval metrics — no ground truth required.

  cosine_sim_stats  — mean / min / max similarity of retrieved chunks
  source_coverage   — number of distinct source documents retrieved
"""

from __future__ import annotations

import math
import statistics
from typing import Any


def cosine_sim_stats(results: list[Any]) -> tuple[float, float, float]:
    """
    Return (mean, min, max) similarity across retrieved chunks.

    ChromaDB with BGE/MiniLM uses L2 distance.
    We convert via exp(-d) which maps [0, ∞) → (0, 1] monotonically,
    matching the similarity display used in the rest of the pipeline.
    """
    if not results:
        return 0.0, 0.0, 0.0
    sims = [round(math.exp(-r.distance), 4) for r in results]
    return statistics.mean(sims), min(sims), max(sims)


def coverage_ratio(results: list[Any], threshold: float) -> float:
    """
    Fraction of retrieved chunks whose similarity to the query meets or
    exceeds `threshold` (converted from Chroma's raw distance the same way
    cosine_sim_stats does, so the two stay on the same scale).

    1.0 = every retrieved chunk is a strong match at this threshold; a low
    value suggests retrieval settled for weak/tangential chunks just to
    fill top_k, rather than genuinely relevant ones.

    Previously advertised by the CLI's --threshold flag ("Similarity
    threshold for coverage_ratio metric") but never actually computed
    anywhere in the pipeline — --threshold was silently a no-op.
    """
    if not results:
        return 0.0
    sims = [math.exp(-r.distance) for r in results]
    return sum(1 for s in sims if s >= threshold) / len(sims)


def _document_id(result: Any) -> str:
    """Best-effort unique document identifier for a retrieved chunk.

    Uses doc_id (the ChromaDB item id), not metadata["source"] — every
    sources/*.py loader's metadata() sets "source" to just the bare
    filename (path.name), which collapses to the same value for every
    chunk when a corpus stores documents under uniquely-named
    subdirectories with a generic converted-output filename (e.g. this
    3GPP corpus's "raw.md" for every spec). doc_id doesn't have this
    problem: it's built at ingestion time from the file's path relative to
    the ingestion data_dir (core/pipeline.py), so it includes every parent
    directory and stays unique per document even when the filename itself
    is generic. No re-ingestion needed — every already-pushed collection's
    doc_id already carries this.
    The real fix for FUTURE ingestion runs is still in sources/*.py's
    metadata() (use something path-relative there too, not path.name
    alone), but that only helps data ingested after the fix — this
    function fixes the metric for data that's already out there.
    """
    doc_id = getattr(result, "doc_id", None)
    if doc_id:
        # doc_id format: f"{rel}::{model_key}::{chunk_index}" -- the part
        # before the first "::" is the source file's relative path,
        # unique per document even when its own filename is generic.
        return doc_id.split("::", 1)[0]
    metadata = getattr(result, "metadata", None) or {}
    return metadata.get("source", "")


def source_coverage(results: list[Any]) -> int:
    """
    Number of distinct source documents in the retrieved chunks.

    Value of 1 → all chunks from one document (low diversity).
    Higher values → multi-document synthesis.
    """
    return len({_document_id(r) for r in results})


def canonical_source_hit(results: list[Any], expected: str | None) -> int:
    """
    Binary check: does the expected source identifier appear in retrieved results?

    Returns:
       1  — expected source found in at least one chunk's metadata or doc_id
       0  — expected source not found
      -1  — no canonical source defined for this question (not applicable)

    expected is a spec number like "23231" — checked case-insensitively
    against all metadata string values and the doc_id.

    No code change needed here for the source_coverage() "raw.md" bug
    (see _document_id()'s docstring) — this function already checks
    doc_id, which carries the real per-document uniqueness that
    metadata["source"] doesn't. What WAS also confirmed broken is the
    question data itself: a real draft QnA set's own canonical_source
    field was "raw.md" for effectively every question (the qna_generation
    flow that produced it extracted canonical_source from the same
    broken metadata["source"]/["filename"] fields, not from a chunk's
    doc_id) — checking doc_id here can't help until `expected` itself is
    a real spec identifier again; that's a telco-qna-generation-side
    fix (wherever its canonical-source extraction reads chunk metadata),
    outside this repo.
    """
    if expected is None:
        return -1
    exp_lower = expected.lower()
    for r in results:
        for v in r.metadata.values():
            if isinstance(v, str) and exp_lower in v.lower():
                return 1
        if hasattr(r, "doc_id") and exp_lower in r.doc_id.lower():
            return 1
    return 0