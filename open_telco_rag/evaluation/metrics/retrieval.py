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
import weakref
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

# store -> {expected: bool}. cached because all_doc_ids() is expensive and
# stable within a run. keyed by the store object (weakly), not id(store):
# CPython reuses ids after garbage collection, so an id-keyed cache handed a
# dead store's answer to a different store
_discriminating_cache: "weakref.WeakKeyDictionary[Any, dict[str, bool]]" = weakref.WeakKeyDictionary()


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

    try:
        per_store = _discriminating_cache.setdefault(store, {})
    except TypeError:  # not weak-referenceable or not hashable: just don't cache
        per_store = None
    if per_store is not None and expected in per_store:
        return per_store[expected]

    ids = all_doc_ids_fn()
    exp_lower = expected.lower()
    if not ids:
        result = False
    else:
        n_matching = sum(1 for doc_id in ids if exp_lower in doc_id.lower())
        result = (n_matching / len(ids)) < _DISCRIMINATING_FRACTION_THRESHOLD

    if per_store is not None:
        per_store[expected] = result
    return result


# enrich_metadata_tagging() prepends "[Document Context -> ...]\n\n"; strip
# it before the text-containment checks below. Non-greedy DOTALL since a
# header value could contain a newline.
_METADATA_TAGGING_HEADER_RE = re.compile(r"^\[Document Context -> .*?\]\n\n", re.DOTALL)


def _strip_enrichment_header(text: str) -> str:
    return _METADATA_TAGGING_HEADER_RE.sub("", text, count=1)


def chunk_rank(
    results: list[Any],
    expected_chunk_id: str | None,
    expected_chunk_text: str | None = None,
) -> int:
    """
    1-indexed rank of the exact source chunk within `results` (rank order,
    index 0 = closest/first) -- the MRR-style companion to exact_chunk_hit().
    Same matching logic exact_chunk_hit() uses, factored out so both can
    share it without a second pass over `results`.

    Returns:
       -1  -- not applicable: no recorded chunk provenance (same condition
              exact_chunk_hit() returns -1 for)
        0  -- the exact chunk was not found anywhere in `results`
       1+  -- the 1-indexed position it was found at

    Two strategies, in order: (1) doc_id equality - only meaningful on the
    same collection (doc_id bakes in the model_key); (2) text containment
    (either direction) against expected_chunk_text, working across
    models/chunking configs, with any metadata_tagging header stripped first.
    """
    if not expected_chunk_id and not expected_chunk_text:
        return -1

    if expected_chunk_id:
        for i, r in enumerate(results, start=1):
            if getattr(r, "doc_id", None) == expected_chunk_id:
                return i

    if expected_chunk_text:
        needle = expected_chunk_text.strip()
        if needle:
            for i, r in enumerate(results, start=1):
                raw_text = (getattr(r, "text", "") or "").strip()
                text = _strip_enrichment_header(raw_text).strip() if raw_text else raw_text
                if text and (needle in text or text in needle):
                    return i

    return 0


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

    See chunk_rank() for where in `results` it was found, not just whether.
    """
    rank = chunk_rank(results, expected_chunk_id, expected_chunk_text)
    if rank < 0:
        return -1
    return 1 if rank > 0 else 0


def _has_valid_offsets(result: Any) -> bool:
    metadata = getattr(result, "metadata", None) or {}
    char_start = metadata.get("char_start", -1)
    char_end = metadata.get("char_end", -1)
    return char_start is not None and char_end is not None and char_start >= 0 and char_end >= 0


def _span_overlap_hit(
    results: list[Any],
    evidence_char_start: int | None,
    evidence_char_end: int | None,
    evidence_doc_id: str | None,
) -> int:
    """Shared logic for span_hit_at_k/span_hit_in_context: does any result
    from evidence_doc_id whose stored char_start/char_end are valid overlap
    [evidence_char_start, evidence_char_end) by any amount?

    Returns:
       1  - some chunk from evidence_doc_id overlaps the evidence span
       0  - no chunk overlaps it (offsets ARE available in this retrieval,
            so the check was meaningful and came back negative)
      -1  - not applicable: no evidence span on the question, OR none of
            `results` carries valid char_start/char_end at all (a legacy
            collection ingested before core/chunking.py's token-sizing
            path -- see Chunk.char_start/char_end)
    """
    if evidence_char_start is None or evidence_char_end is None or not evidence_doc_id:
        return -1

    any_offsets_available = False
    for r in results:
        if not _has_valid_offsets(r):
            continue
        any_offsets_available = True
        metadata = r.metadata or {}
        char_start = metadata["char_start"]
        char_end = metadata["char_end"]
        if _document_id(r) == evidence_doc_id and char_start < evidence_char_end and evidence_char_start < char_end:
            return 1

    return 0 if any_offsets_available else -1


def span_hit_at_k(
    results: list[Any],
    evidence_char_start: int | None,
    evidence_char_end: int | None,
    evidence_doc_id: str | None,
) -> int:
    """
    Binary check, chunker-agnostic alternative to exact_chunk_hit: does any
    RETRIEVED chunk's [char_start, char_end) span overlap the question's
    evidence span (see EvalQuestion.evidence_char_start/end/doc_id) by any
    amount -- not necessarily an exact chunk-boundary match, so it's
    meaningful across chunkers/chunk sizes that don't share the ground-truth
    chunk boundaries text_baseline/chunk_id-based matching assumes.

    See _span_overlap_hit() for the -1/0/1 sentinel convention.
    """
    return _span_overlap_hit(results, evidence_char_start, evidence_char_end, evidence_doc_id)


def span_hit_in_context(
    context: list[Any],
    evidence_char_start: int | None,
    evidence_char_end: int | None,
    evidence_doc_id: str | None,
) -> int:
    """
    Same overlap check as span_hit_at_k, but over `context` -- the chunks
    actually PLACED in the generator's context after budgeted assembly (see
    core.retrieval.fill_context_budget), not the full retrieved candidate
    set. A question can be a span_hit_at_k hit while missing span_hit_in_context
    (the evidence chunk was retrieved but didn't make it into the budget).

    See _span_overlap_hit() for the -1/0/1 sentinel convention.
    """
    return _span_overlap_hit(context, evidence_char_start, evidence_char_end, evidence_doc_id)


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