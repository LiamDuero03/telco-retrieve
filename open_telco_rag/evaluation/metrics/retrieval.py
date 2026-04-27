"""
Tier 1 retrieval metrics — no ground truth required.

  cosine_sim_stats      — mean / min / max similarity of retrieved chunks
  source_coverage       — number of distinct source documents retrieved
  canonical_source_hit  — expected source identifier present in results?
  coverage_ratio        — fraction of chunks passing the similarity threshold
"""

from __future__ import annotations

import statistics
from typing import Any


def cosine_sim_stats(results: list[Any]) -> tuple[float, float, float]:
    """
    Return (mean, min, max) cosine similarity across retrieved chunks.

    ChromaDB returns cosine distance d; similarity = 1 - d.
    High mean → query is well-represented in the corpus.
    Low mean  → domain gap or missing content.
    """
    if not results:
        return 0.0, 0.0, 0.0
    sims = [1.0 - r.distance for r in results]
    return statistics.mean(sims), min(sims), max(sims)


def source_coverage(results: list[Any]) -> int:
    """
    Number of distinct source documents in the retrieved chunks.

    Uses the 'source' metadata field (filename). Falls back to doc_id.
    Value of 1 → all chunks from one document.
    Higher values → multi-document synthesis.
    """
    return len({r.metadata.get("source", r.doc_id) for r in results})


def canonical_source_hit(results: list[Any], expected: str | None) -> int:
    """
    Binary check: does the expected source identifier appear in retrieved results?

    Returns:
       1  — expected source found in at least one chunk's metadata or doc_id
       0  — expected source not found
      -1  — no canonical source defined for this question (not applicable)

    expected can be a work group ("WG3"), series ("28"), repo name
    ("QualityOnDemand"), ISG group ("ENI"), etc. Checked case-insensitively
    against all metadata string values and the doc_id.
    """
    if expected is None:
        return -1
    exp_lower = expected.lower()
    for r in results:
        for v in r.metadata.values():
            if isinstance(v, str) and exp_lower in v.lower():
                return 1
        if exp_lower in r.doc_id.lower():
            return 1
    return 0


def coverage_ratio(results: list[Any], threshold: float = 0.5) -> float:
    """
    Fraction of retrieved chunks with cosine similarity >= threshold.

    Score = |{rᵢ | sᵢ ≥ τ}| / k

    A low value means retrieval is filling the top-k with low-confidence
    chunks, which motivates tighter thresholds or smaller top-k.
    """
    if not results:
        return 0.0
    passing = sum(1 for r in results if (1.0 - r.distance) >= threshold)
    return passing / len(results)