"""
Tier 1 retrieval metrics — no ground truth required.

  cosine_sim_stats           — mean / min / max similarity of retrieved chunks
  distinct_sources_retrieved — number of distinct source documents retrieved
  canonical_source_hit       — did any chunk from the expected DOCUMENT come back
  exact_chunk_hit            — did the exact source CHUNK a question was written
                                from come back (stronger than canonical_source_hit)
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def _similarity(result: Any) -> float:
    """Chroma L2 distance -> similarity. exp(-d) maps [0, ∞) -> (0, 1]
    monotonically, matching the similarity display used in the rest of the
    pipeline. Rounded to 4dp -- the single source of truth cosine_sim_stats()
    and coverage_ratio() both route through, so the "same" underlying value
    can't silently disagree in precision between the two metrics."""
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
    """Best-effort unique document identifier for a retrieved chunk.

    Uses doc_id, not metadata["source"] — sources/*.py's metadata() sets
    "source" to the bare filename (path.name), which collapses to the same
    value across a corpus with a generic converted-output filename (e.g.
    3GPP's "raw.md" for every spec). doc_id is built at ingestion time from
    the file's path relative to data_dir, so it stays unique even when the
    filename itself is generic — no re-ingestion needed, already-pushed
    collections carry this. A future fix in sources/*.py's metadata() would
    only help newly-ingested data; this function fixes the metric for
    existing data.
    """
    doc_id = getattr(result, "doc_id", None)
    if doc_id:
        # Format: f"{rel}::{model_key}::{chunk_index}" — part before first
        # "::" is the source file's relative path.
        return doc_id.split("::", 1)[0]
    metadata = getattr(result, "metadata", None) or {}
    return metadata.get("source", "")


def distinct_sources_retrieved(results: list[Any]) -> int:
    """
    Number of distinct source documents in the retrieved chunks.

    A raw count in [0, len(results)], not a 0-1 ratio (renamed from
    source_coverage, which implied one): 1 = low diversity, up to
    len(results) = every chunk from a different document.
    """
    return len({_document_id(r) for r in results})


# Fraction of doc_ids `expected` may substring-match before
# canonical_source_hit()'s guard rejects it as non-discriminating (see
# is_discriminating_source()). 0.5 is deliberately generous — a real
# per-document identifier should match one document almost always; the
# actual broken value ("raw.md") matched ~100% of doc_ids.
_DISCRIMINATING_FRACTION_THRESHOLD = 0.5

# (id(store), expected) -> bool. all_doc_ids() queries the full collection
# (can be huge); cached since the same canonical_source values recur across
# every question in one evaluate_corpus() run and the store doesn't change mid-run.
_discriminating_cache: dict[tuple[int, str], bool] = {}


def is_discriminating_source(expected: str, store: Any) -> bool:
    """True if `expected` (substring-matched against doc_ids, same
    case-insensitive semantics as canonical_source_hit()) appears in fewer
    than _DISCRIMINATING_FRACTION_THRESHOLD of all doc_ids in store's
    collection — i.e. specific enough to be a meaningful ground-truth check.

    Motivating bug: 3GPP's document converter names every output file
    "raw.md", so it's a substring of every chunk's doc_id regardless of
    which spec — and canonical_source was "raw.md" for ~100% of a real
    draft QnA set. Checking doc_id for "raw.md" was trivially true for any
    retrieval, correct or not (canonical_hit_rate = 1.000 on every grid
    combination).

    False when store doesn't support all_doc_ids() (can't verify, so don't
    silently assume fine — that would reintroduce the bug for such stores).

    Checks only doc_id, not the full metadata dict canonical_source_hit()
    also searches — all_doc_ids() is a cheap ids-only query, and the
    generic-filename failure mode shows up in doc_id regardless.
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


def exact_chunk_hit(
    results: list[Any],
    expected_chunk_id: str | None,
    expected_chunk_text: str | None = None,
) -> int:
    """
    Binary check: was the EXACT chunk this question was generated from
    actually retrieved -- not just any chunk from the same document.

    canonical_source_hit() above only proves "a chunk from the right
    document came back"; it says nothing about whether it was the same
    passage the question was written from, and needs is_discriminating_
    source()'s guard to avoid a trivial match on a generic doc-converter
    filename. This checks chunk identity directly, so it doesn't need
    that guard -- a chunk's own id or its full text is discriminating by
    construction.

    Returns:
       1  -- the exact source chunk was retrieved
       0  -- neither doc_id nor text matched
      -1  -- not applicable: this question has no recorded chunk
             provenance (e.g. predates chunk_id/source_text tracking, or
             was hand-authored)

    Two strategies, tried in order:
      1. doc_id equality. Exact, unambiguous when it matches -- but only
         meaningful when evaluating the SAME collection (same embedding
         model_key) the question's chunk was sampled from, since doc_id
         bakes in the model_key (f"{rel}::{model_key}::{chunk_index}" --
         see pipeline.py). A different model's collection never matches
         here even for the literal same source passage.
      2. Text containment against expected_chunk_text (the question's
         stored source_text -- see qna_generation's generator.
         _build_record()). Works across different embedding models/
         chunking configs, since it compares content, not an id that
         encodes the ingestion config. Containment either way (not just
         equality) tolerates minor boundary differences between the
         chunking strategy used at generation time and the one being
         evaluated here.
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
                text = (getattr(r, "text", "") or "").strip()
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
       1  — expected source (and release, if expected_release is given)
            found in at least one chunk's metadata or doc_id
       0  — expected source not found
      -1  — not applicable: no canonical source for this question, OR (when
            store is given) `expected` failed is_discriminating_source()'s
            guard (see that function's docstring). The two cases aren't
            distinguishable from this return value alone — callers needing
            to tell them apart should call is_discriminating_source()
            themselves (see evaluate_question()).

    expected is a spec number like "23231" — checked case-insensitively
    against all metadata string values and doc_id.

    expected_release, if given (see EvalQuestion.release / tgpp.py's
    canonical_id), additionally requires a candidate chunk to also match
    the release string -- otherwise two releases of the same spec (same
    spec number, different content) would count as a hit against each
    other. None (the default, and what questions predating release
    tracking use) skips this check entirely, unchanged from before.

    store defaults to None for backward compatibility with callers/tests
    that construct results by hand — passing None skips the guard entirely.
    Real evaluation runs always pass the real store.
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