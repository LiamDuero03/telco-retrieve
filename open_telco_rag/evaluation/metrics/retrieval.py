"""
Tier 1 retrieval metrics - no ground truth required.

  cosine_sim_stats           - mean / min / max similarity of retrieved chunks
  distinct_sources_retrieved - number of distinct source documents retrieved
  canonical_source_hit       - did any chunk from the expected DOCUMENT come back
  exact_chunk_hit            - did the exact source CHUNK a question was written
                                from come back (stronger than canonical_source_hit)
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def _similarity(result: Any) -> float:
    """Chroma L2 distance -> similarity via exp(-d), rounded to 4dp.
    Shared by cosine_sim_stats() and coverage_ratio()."""
    return round(math.exp(-result.distance), 4)


def cosine_sim_stats(results: list[Any]) -> tuple[float, float, float]:
    """Return (mean, min, max) similarity across retrieved chunks."""
    if not results:
        return 0.0, 0.0, 0.0
    sims = [_similarity(r) for r in results]
    return statistics.mean(sims), min(sims), max(sims)


def coverage_ratio(results: list[Any], threshold: float) -> float:
    """
    Fraction of retrieved chunks whose similarity to the query meets or
    exceeds `threshold` (same _similarity() conversion as cosine_sim_stats,
    so the two stay on the same scale).

    1.0 = every chunk is a strong match; low value suggests retrieval
    settled for weak/tangential chunks to fill top_k.
    """
    if not results:
        return 0.0
    sims = [_similarity(r) for r in results]
    return sum(1 for s in sims if s >= threshold) / len(sims)


def _document_id(result: Any) -> str:
    """Best-effort document identifier. Uses doc_id (path before "::"), not
    metadata["source"], which collapses to a generic filename across a corpus.
    """
    doc_id = getattr(result, "doc_id", None)
    if doc_id:
        # doc_id is "{rel}::{model_key}::{chunk_index}".
        return doc_id.split("::", 1)[0]
    metadata = getattr(result, "metadata", None) or {}
    return metadata.get("source", "")


def distinct_sources_retrieved(results: list[Any]) -> int:
    """Number of distinct source documents in the retrieved chunks (a raw
    count, not a ratio)."""
    return len({_document_id(r) for r in results})


# Max fraction of doc_ids `expected` may match before it's rejected as
# non-discriminating.
_DISCRIMINATING_FRACTION_THRESHOLD = 0.5

# (id(store), expected) -> bool. Cached: all_doc_ids() is expensive and
# stable within a run.
_discriminating_cache: dict[tuple[int, str], bool] = {}


def is_discriminating_source(expected: str, store: Any) -> bool:
    """True if `expected` (substring-matched, case-insensitive) appears in
    fewer than _DISCRIMINATING_FRACTION_THRESHOLD of store's doc_ids - i.e.
    specific enough to be a meaningful ground-truth check. Guards against a
    generic converter filename ("raw.md") matching every doc_id.

    False when the store doesn't support all_doc_ids() (can't verify).
    """
    all_doc_ids_fn = getattr(store, "all_doc_ids", None)
    if all_doc_ids_fn is None:
        return False

    cache_key = (id(store), expected)
    if cache_key in _discriminating_cache:
        return _discriminating_cache[cache_key]

    ids = all_doc_ids_fn()
    exp_lower = expected.lower()
    if not ids:
        result = False
    else:
        n_matching = sum(1 for doc_id in ids if exp_lower in doc_id.lower())
        result = (n_matching / len(ids)) < _DISCRIMINATING_FRACTION_THRESHOLD

    _discriminating_cache[cache_key] = result
    return result


# enrich_metadata_tagging() prepends "[Document Context -> ...]\n\n"; strip
# it before the text-containment checks below. Non-greedy DOTALL since a
# header value could contain a newline.
_METADATA_TAGGING_HEADER_RE = re.compile(r"^\[Document Context -> .*?\]\n\n", re.DOTALL)


def _strip_enrichment_header(text: str) -> str:
    return _METADATA_TAGGING_HEADER_RE.sub("", text, count=1)


def exact_chunk_hit(
    results: list[Any],
    expected_chunk_id: str | None,
    expected_chunk_text: str | None = None,
) -> int:
    """
    Binary check: was the EXACT chunk this question was generated from
    retrieved (not just any chunk from the same document)?

    Returns:
       1  -- the exact source chunk was retrieved
       0  -- neither doc_id nor text matched
      -1  -- not applicable: no recorded chunk provenance

    Two strategies, in order: (1) doc_id equality - only meaningful on the
    same collection (doc_id bakes in the model_key); (2) text containment
    (either direction) against expected_chunk_text, working across
    models/chunking configs, with any metadata_tagging header stripped first.
    """
    if not expected_chunk_id and not expected_chunk_text:
        return -1

    if expected_chunk_id:
        for r in results:
            if getattr(r, "doc_id", None) == expected_chunk_id:
                return 1

    if expected_chunk_text:
        needle = expected_chunk_text.strip()
        if needle:
            for r in results:
                raw_text = (getattr(r, "text", "") or "").strip()
                text = _strip_enrichment_header(raw_text).strip() if raw_text else raw_text
                if text and (needle in text or text in needle):
                    return 1

    return 0


def _any_metadata_or_doc_id_contains(result: Any, needle_lower: str) -> bool:
    for v in result.metadata.values():
        if isinstance(v, str) and needle_lower in v.lower():
            return True
    return hasattr(result, "doc_id") and needle_lower in result.doc_id.lower()


def canonical_source_hit(
    results: list[Any], expected: str | None, store: Any = None,
    expected_release: str | None = None,
) -> int:
    """
    Binary check: does the expected source identifier appear in retrieved results?

    Returns:
       1  - expected source (and release, if given) found in a chunk's
            metadata or doc_id
       0  - not found
      -1  - not applicable: no canonical source, or `expected` failed
            is_discriminating_source()'s guard

    expected is checked case-insensitively against all metadata string
    values and doc_id. expected_release, if given, additionally requires a
    release match. store=None skips the guard (for hand-built results/tests).
    """
    if expected is None:
        return -1
    if store is not None and not is_discriminating_source(expected, store):
        logger.warning(
            "canonical_source_hit: expected=%r matches too large a "
            "fraction of documents in this corpus to be a meaningful "
            "ground-truth check -- treating as not applicable rather "
            "than trusting a trivial match.", expected,
        )
        return -1
    exp_lower = expected.lower()
    rel_lower = expected_release.lower() if expected_release else None
    for r in results:
        if not _any_metadata_or_doc_id_contains(r, exp_lower):
            continue
        if rel_lower is None or _any_metadata_or_doc_id_contains(r, rel_lower):
            return 1
    return 0