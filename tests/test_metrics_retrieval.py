"""Tests for open_telco_rag.evaluation.metrics.retrieval — specifically
source_coverage()'s real, confirmed-in-production bug: metadata["source"]
is just path.name (the bare filename, no parent directory) for every
sources/*.py loader. For a corpus that stores each document under its own
uniquely-named subdirectory with a generic converted-output filename (this
TSpec-LLM-derived 3GPP corpus does exactly this — every spec's markdown
file is literally named "raw.md"), metadata["source"] collapses to the
SAME value for every chunk in the entire corpus, regardless of which
document it actually came from.

Confirmed directly against a real 24-combination grid's aggregated
all_results.json: metadata["source"] == "raw.md" for all 360/360
question_results, making source_coverage() return 1 ("one distinct
source") and canonical_source_hit() trivially "hit" for every single
question — both flat at their maximum value across every combination,
unable to discriminate good retrieval from bad at all.

doc_id (the ChromaDB item id, built at ingestion time as
f"{rel}::{model_key}::{chunk_index}" where rel is the file's path RELATIVE
TO the ingestion data_dir — see core/pipeline.py) does NOT have this
problem — it's unique per document even when the filename itself is
generic. source_coverage() now derives document identity from doc_id
instead of the broken metadata["source"] — critically, this requires NO
re-ingestion, since doc_id was already correct in every already-pushed
collection.
"""
from __future__ import annotations

from open_telco_rag.evaluation.metrics.retrieval import (
    _document_id, canonical_source_hit, coverage_ratio, source_coverage,
)
from open_telco_rag.types import SearchResult


def _chunk(doc_id: str, source: str = "raw.md", distance: float = 0.3, **extra_meta) -> SearchResult:
    return SearchResult(
        text="some chunk text", distance=distance,
        metadata={"source": source, **extra_meta}, doc_id=doc_id,
    )


# ── _document_id / source_coverage ──────────────────────────────────────────

def test_document_id_prefers_doc_id_over_the_broken_metadata_source_field():
    """The exact real-world case: metadata["source"] is the SAME generic
    "raw.md" for two chunks from genuinely DIFFERENT documents, but their
    doc_ids (built from the full relative ingestion path) differ."""
    chunk_a = _chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4", source="raw.md")
    chunk_b = _chunk(doc_id="29_series/29222-j20/raw.md::minilm::7", source="raw.md")

    assert _document_id(chunk_a) != _document_id(chunk_b)
    assert _document_id(chunk_a) == "38_series/38101-1-j10/raw.md"
    assert _document_id(chunk_b) == "29_series/29222-j20/raw.md"


def test_source_coverage_discriminates_when_metadata_source_is_uniformly_wrong():
    """Regression test for the exact confirmed production bug: with the
    OLD `r.metadata.get("source", r.doc_id)` logic, this would have
    returned 1 (both chunks report "source": "raw.md") even though they
    are from two completely different specs. Must now return 2."""
    results = [
        _chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4"),
        _chunk(doc_id="29_series/29222-j20/raw.md::minilm::7"),
    ]
    assert source_coverage(results) == 2


def test_source_coverage_still_returns_1_for_genuinely_single_document_retrieval():
    results = [
        _chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4"),
        _chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::9"),
    ]
    assert source_coverage(results) == 1


def test_document_id_falls_back_to_metadata_source_when_doc_id_is_missing():
    """Defensive fallback for hand-built SearchResults (e.g. other tests'
    fixtures) that don't set doc_id at all."""
    chunk = SearchResult(text="x", distance=0.1, metadata={"source": "some-file.md"}, doc_id="")
    assert _document_id(chunk) == "some-file.md"


# ── canonical_source_hit ─────────────────────────────────────────────────

def test_canonical_source_hit_returns_not_applicable_when_no_canonical_source():
    assert canonical_source_hit([_chunk(doc_id="a::minilm::0")], None) == -1


def test_canonical_source_hit_matches_via_doc_id_even_when_metadata_source_is_generic():
    """canonical_source_hit() already checked doc_id before this fix --
    confirms that path still works once `expected` is a real identifier,
    independent of source_coverage()'s metadata["source"] bug."""
    results = [_chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4", source="raw.md")]
    assert canonical_source_hit(results, "38101-1-j10") == 1


def test_canonical_source_hit_returns_zero_when_genuinely_not_found():
    results = [_chunk(doc_id="29_series/29222-j20/raw.md::minilm::7", source="raw.md")]
    assert canonical_source_hit(results, "38101-1-j10") == 0


# ── coverage_ratio (separate issue: default threshold is uncalibrated) ────

def test_coverage_ratio_is_not_discriminating_at_the_default_0_5_threshold():
    """Documents the SEPARATE (non-source-identity) finding: exp(-distance)
    similarity for this corpus/embedding combination routinely clears 0.5
    even for middling retrievals -- confirmed via a real grid's observed
    min_similarity values of 0.64-0.78+. Not a bug in coverage_ratio()'s
    own logic (it does exactly what it says), just a threshold poorly
    calibrated for this corpus -- see master_grid_eval.py's SIM_THRESHOLD."""
    import math
    # distance such that exp(-distance) ~= 0.65 -- a real observed "weak
    # end" similarity from the confirmed grid run.
    weak_distance = -math.log(0.65)
    results = [SearchResult(text="x", distance=weak_distance, metadata={}, doc_id=f"d{i}") for i in range(5)]
    assert coverage_ratio(results, threshold=0.5) == 1.0


def test_coverage_ratio_does_discriminate_at_a_higher_threshold():
    import math
    weak_distance = -math.log(0.65)
    results = [SearchResult(text="x", distance=weak_distance, metadata={}, doc_id=f"d{i}") for i in range(5)]
    assert coverage_ratio(results, threshold=0.75) == 0.0
