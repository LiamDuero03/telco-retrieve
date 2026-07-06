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


def source_coverage(results: list[Any]) -> int:
    """
    Number of distinct source documents in the retrieved chunks.

    Value of 1 → all chunks from one document (low diversity).
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

    expected is a spec number like "23231" — checked case-insensitively
    against all metadata string values and the doc_id.
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