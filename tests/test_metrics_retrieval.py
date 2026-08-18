"""Tests for open_telco_rag.evaluation.metrics.retrieval.

distinct_sources_retrieved() (renamed from source_coverage()): real
production bug — metadata["source"] is just the bare filename for every
sources/*.py loader, and this TSpec-LLM-derived 3GPP corpus names every
converted file "raw.md", so metadata["source"] collapsed to the same value
for every chunk regardless of source document. Renamed to stop implying a
0-1 scale — it's a raw distinct-document count in [0, len(results)].

canonical_source_hit()'s is_discriminating_source() guard: same root cause,
worse — canonical_source_hit() trivially "hit" for every question since
every doc_id also contains "raw.md", and a real 24-combination grid showed
canonical_hit_rate = coverage_ratio = 1.000 on every combination (unable to
discriminate good retrieval from bad at all). The guard treats `expected`
as not applicable (-1) whenever it matches too large a fraction of all
doc_ids in the corpus to be a meaningful ground-truth check.

doc_id (the ChromaDB item id, "{rel}::{model_key}::{chunk_index}" where rel
is the file's path relative to the ingestion data_dir) doesn't have the
metadata["source"] problem — unique per document even with a generic filename.
"""
from __future__ import annotations

from open_telco_rag.evaluation.metrics.retrieval import (
    _document_id, canonical_source_hit, coverage_ratio, cosine_sim_stats,
    distinct_sources_retrieved, exact_chunk_hit, is_discriminating_source,
)
from open_telco_rag.types import SearchResult


def _chunk(doc_id: str, source: str = "raw.md", distance: float = 0.3, **extra_meta) -> SearchResult:
    return SearchResult(
        text="some chunk text", distance=distance,
        metadata={"source": source, **extra_meta}, doc_id=doc_id,
    )


# ── _document_id / distinct_sources_retrieved ───────────────────────────────

def test_document_id_prefers_doc_id_over_the_broken_metadata_source_field():
    """Real case: metadata["source"] is the same generic "raw.md" for two
    chunks from different documents, but their doc_ids differ."""
    chunk_a = _chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4", source="raw.md")
    chunk_b = _chunk(doc_id="29_series/29222-j20/raw.md::minilm::7", source="raw.md")

    assert _document_id(chunk_a) != _document_id(chunk_b)
    assert _document_id(chunk_a) == "38_series/38101-1-j10/raw.md"
    assert _document_id(chunk_b) == "29_series/29222-j20/raw.md"


def test_distinct_sources_retrieved_discriminates_when_metadata_source_is_uniformly_wrong():
    """Regression: old `metadata.get("source", doc_id)` logic returned 1
    here despite two different specs; must now return 2."""
    results = [
        _chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4"),
        _chunk(doc_id="29_series/29222-j20/raw.md::minilm::7"),
    ]
    assert distinct_sources_retrieved(results) == 2


def test_distinct_sources_retrieved_still_returns_1_for_genuinely_single_document_retrieval():
    results = [
        _chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4"),
        _chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::9"),
    ]
    assert distinct_sources_retrieved(results) == 1


def test_document_id_falls_back_to_metadata_source_when_doc_id_is_missing():
    """Fallback for hand-built SearchResults that don't set doc_id."""
    chunk = SearchResult(text="x", distance=0.1, metadata={"source": "some-file.md"}, doc_id="")
    assert _document_id(chunk) == "some-file.md"


# ── canonical_source_hit — unguarded (store=None), backward compatible ──────

def test_canonical_source_hit_returns_not_applicable_when_no_canonical_source():
    assert canonical_source_hit([_chunk(doc_id="a::minilm::0")], None) == -1


def test_canonical_source_hit_matches_via_doc_id_even_when_metadata_source_is_generic():
    """Confirms the pre-existing doc_id check still works for a real
    identifier; no store passed since the guard is opt-in."""
    results = [_chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4", source="raw.md")]
    assert canonical_source_hit(results, "38101-1-j10") == 1


def test_canonical_source_hit_returns_zero_when_genuinely_not_found():
    results = [_chunk(doc_id="29_series/29222-j20/raw.md::minilm::7", source="raw.md")]
    assert canonical_source_hit(results, "38101-1-j10") == 0


# ── is_discriminating_source() / canonical_source_hit(store=...) guard ──────
#
# _FakeStore simulates the real corpus: every doc converted to "raw.md" in a
# spec-specific subfolder. all_doc_ids() returns the whole corpus so
# is_discriminating_source() can see "raw.md" match ~100% vs a spec number matching one.

class _FakeStore:
    def __init__(self, doc_ids: list[str]):
        self._doc_ids = doc_ids
        self.all_doc_ids_call_count = 0

    def all_doc_ids(self) -> list[str]:
        self.all_doc_ids_call_count += 1
        return self._doc_ids


def _real_corpus_shaped_store(n_docs: int = 50) -> _FakeStore:
    """Every doc_id contains "raw.md" (real pattern); a real spec number
    matches only its own document."""
    doc_ids = [f"{30000 + i}_series/{30000 + i}-j10/raw.md::minilm::{c}"
               for i in range(n_docs) for c in range(3)]
    return _FakeStore(doc_ids)


def test_is_discriminating_source_rejects_a_generic_filename_matching_the_whole_corpus():
    store = _real_corpus_shaped_store()
    assert is_discriminating_source("raw.md", store) is False


def test_is_discriminating_source_accepts_a_real_spec_number_matching_one_document():
    store = _real_corpus_shaped_store()
    assert is_discriminating_source("30007", store) is True


def test_is_discriminating_source_caches_and_does_not_re_query_all_doc_ids():
    store = _real_corpus_shaped_store()
    is_discriminating_source("raw.md", store)
    is_discriminating_source("raw.md", store)
    is_discriminating_source("raw.md", store)
    assert store.all_doc_ids_call_count == 1


def test_is_discriminating_source_false_when_store_has_no_all_doc_ids():
    """Can't verify -> must not silently assume fine; the guard exists to
    stop blind trust in `expected`."""
    class _StoreWithoutAllDocIds:
        pass
    assert is_discriminating_source("30007", _StoreWithoutAllDocIds()) is False


def test_is_discriminating_source_false_for_empty_corpus_no_crash():
    store = _FakeStore([])
    assert is_discriminating_source("30007", store) is False


def test_canonical_source_hit_with_store_rejects_raw_md_even_though_naive_match_would_hit():
    """The regression test: a chunk that would naively "hit" on
    expected="raw.md" must be -1 (not applicable) once a store is
    provided, regardless of whether it's genuinely correct."""
    store = _real_corpus_shaped_store()
    genuinely_relevant_chunk = _chunk(doc_id="30002_series/30002-j10/raw.md::minilm::1")
    genuinely_unrelated_chunk = _chunk(doc_id="30049_series/30049-j10/raw.md::minilm::2")

    assert canonical_source_hit([genuinely_relevant_chunk], "raw.md", store=store) == -1
    assert canonical_source_hit([genuinely_unrelated_chunk], "raw.md", store=store) == -1


def test_canonical_source_hit_with_store_still_works_for_a_real_discriminating_identifier():
    """Confirms the guard doesn't break the good path: a real specific
    identifier still discriminates hit/miss with a store."""
    store = _real_corpus_shaped_store()
    chunk_from_expected_doc = _chunk(doc_id="30002_series/30002-j10/raw.md::minilm::1")
    chunk_from_other_doc = _chunk(doc_id="30049_series/30049-j10/raw.md::minilm::2")

    assert canonical_source_hit([chunk_from_expected_doc], "30002-j10", store=store) == 1
    assert canonical_source_hit([chunk_from_other_doc], "30002-j10", store=store) == 0


def test_canonical_source_hit_with_store_still_returns_not_applicable_for_none_expected():
    store = _real_corpus_shaped_store()
    assert canonical_source_hit([_chunk(doc_id="a::minilm::0")], None, store=store) == -1


# ── canonical_source_hit(expected_release=...) — same spec, different release ─
#
# 3GPP's canonical_source is a spec number alone (e.g. "25401"); it doesn't
# distinguish releases, and a spec's own metadata never carries version/
# release info when its filename is TSpec-LLM's generic "raw.md" (see
# tgpp.py's metadata()). A chunk from the WRONG release of the right spec
# must not count as a hit.

def test_canonical_source_hit_without_expected_release_ignores_release_entirely():
    """Backward compatible: old questions with no recorded release behave
    exactly as before -- a spec-number match alone is a hit."""
    results = [_chunk(doc_id="25_series/25401/raw.md::bge::4", release="Rel-17")]
    assert canonical_source_hit(results, "25401") == 1


def test_canonical_source_hit_rejects_a_match_from_the_wrong_release():
    """The regression: same spec number, but the retrieved chunk is from a
    different release than the question was generated from."""
    wrong_release_chunk = _chunk(doc_id="25_series/25401/raw.md::bge::4", release="Rel-17")
    assert canonical_source_hit([wrong_release_chunk], "25401", expected_release="Rel-19") == 0


def test_canonical_source_hit_accepts_a_match_from_the_matching_release():
    right_release_chunk = _chunk(doc_id="25_series/25401/raw.md::bge::4", release="Rel-19")
    assert canonical_source_hit([right_release_chunk], "25401", expected_release="Rel-19") == 1


def test_canonical_source_hit_expected_release_also_matches_via_canonical_id_style_doc_id():
    """Release doesn't have to be its own metadata field -- a doc_id or any
    metadata value containing it (e.g. tgpp.py's canonical_id combining
    spec+release) is enough, same substring convention as `expected`."""
    chunk = _chunk(doc_id="d::bge::0", canonical_id="25.401 (Rel-19)")
    assert canonical_source_hit([chunk], "25.401", expected_release="Rel-19") == 1
    assert canonical_source_hit([chunk], "25.401", expected_release="Rel-17") == 0


def test_canonical_source_hit_picks_a_later_result_matching_both_source_and_release():
    """A wrong-release chunk from the right spec must not short-circuit the
    scan -- a later, correctly-released chunk should still register a hit."""
    results = [
        _chunk(doc_id="25_series/25401/raw.md::bge::1", release="Rel-17"),
        _chunk(doc_id="25_series/25401/raw.md::bge::2", release="Rel-19"),
    ]
    assert canonical_source_hit(results, "25401", expected_release="Rel-19") == 1


# ── exact_chunk_hit — chunk-level, not document-level ───────────────────────
#
# canonical_source_hit() only proves "a chunk from the right document came
# back". exact_chunk_hit() proves the literal chunk the question was
# generated from came back, via doc_id equality or source_text containment.

def test_exact_chunk_hit_not_applicable_with_no_provenance():
    results = [_chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4")]
    assert exact_chunk_hit(results, None, None) == -1


def test_exact_chunk_hit_matches_on_exact_doc_id():
    results = [
        _chunk(doc_id="29_series/29222-j20/raw.md::minilm::7"),
        _chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::4"),
    ]
    assert exact_chunk_hit(results, "38_series/38101-1-j10/raw.md::minilm::4") == 1


def test_exact_chunk_hit_does_not_match_a_different_chunk_from_the_same_document():
    """The core distinction from canonical_source_hit: same document,
    different chunk index -- must be a miss, not a hit."""
    results = [_chunk(doc_id="38_series/38101-1-j10/raw.md::minilm::9")]
    assert exact_chunk_hit(results, "38_series/38101-1-j10/raw.md::minilm::4") == 0


def test_exact_chunk_hit_falls_back_to_text_match_when_doc_id_differs():
    """Simulates evaluating a DIFFERENT embedding model's collection than
    the one the question's chunk_id was recorded against -- doc_id embeds
    the model_key, so it never matches across models, but the underlying
    passage is still findable by its text."""
    original_text = "The AMF handles registration, connection and mobility management."
    results = [
        SearchResult(
            text=original_text, distance=0.1,
            metadata={"source": "23501.md"}, doc_id="23501.md::bge::12",
        ),
    ]
    assert exact_chunk_hit(
        results,
        expected_chunk_id="23501.md::minilm::4",   # different model_key, never matches
        expected_chunk_text=original_text,
    ) == 1


def test_exact_chunk_hit_text_match_tolerates_minor_boundary_differences():
    """Containment either way, not strict equality -- a slightly different
    chunking pass may retrieve a superset/subset of the exact original text."""
    results = [SearchResult(
        text="Preamble. The AMF handles registration. Trailing sentence.",
        distance=0.1, metadata={}, doc_id="d::minilm::0",
    )]
    assert exact_chunk_hit(
        results, expected_chunk_id=None,
        expected_chunk_text="The AMF handles registration.",
    ) == 1


def test_exact_chunk_hit_returns_zero_when_neither_strategy_matches():
    results = [SearchResult(
        text="Completely unrelated content about SMF selection.",
        distance=0.1, metadata={}, doc_id="other-doc::minilm::0",
    )]
    assert exact_chunk_hit(
        results, expected_chunk_id="23501.md::minilm::4",
        expected_chunk_text="The AMF handles registration.",
    ) == 0


# ── coverage_ratio (separate issue: default threshold is uncalibrated) ────
#
# The metric's own mechanism correctly discriminates low vs high similarity;
# the default threshold (0.5) is just too permissive for this corpus's
# real similarity distribution.

def test_coverage_ratio_is_not_discriminating_at_the_default_0_5_threshold():
    """Separate finding: exp(-distance) similarity routinely clears 0.5
    even for middling retrievals (real grid observed 0.64-0.78+). Not a
    logic bug — threshold is poorly calibrated; see master_grid_eval.py's
    SIM_THRESHOLD."""
    import math
    # exp(-distance) ~= 0.65: a real observed "weak end" similarity from the grid run.
    weak_distance = -math.log(0.65)
    results = [SearchResult(text="x", distance=weak_distance, metadata={}, doc_id=f"d{i}") for i in range(5)]
    assert coverage_ratio(results, threshold=0.5) == 1.0


def test_coverage_ratio_does_discriminate_at_a_higher_threshold():
    import math
    weak_distance = -math.log(0.65)
    results = [SearchResult(text="x", distance=weak_distance, metadata={}, doc_id=f"d{i}") for i in range(5)]
    assert coverage_ratio(results, threshold=0.75) == 0.0


def test_coverage_ratio_mechanism_correctly_discriminates_genuinely_bad_retrieval():
    """Other half of the finding: coverage_ratio isn't structurally broken
    like canonical_source_hit was — a genuinely low similarity (exp(-d)~=0.23,
    real minilm distance for an unrelated chunk) correctly reports 0.0."""
    import math
    bad_distance = -math.log(0.23)
    results = [SearchResult(text="x", distance=bad_distance, metadata={}, doc_id="d0")]
    assert coverage_ratio(results, threshold=0.5) == 0.0


def test_coverage_ratio_uses_the_same_rounded_precision_as_cosine_sim_stats():
    """Regression: coverage_ratio() used to re-derive the distance ->
    similarity conversion itself, skipping cosine_sim_stats()'s 4dp
    rounding -- so the SAME underlying similarity could report, and be
    thresholded, at different precision from the two metrics."""
    import math
    # exp(-d) ~= 0.99996, which rounds to exactly 1.0 at 4dp -- a threshold
    # of 1.0 only passes if coverage_ratio rounds the same way
    # cosine_sim_stats does; the old unrounded 0.99996 would fail it.
    distance = -math.log(0.99996)
    results = [SearchResult(text="x", distance=distance, metadata={}, doc_id="d0")]

    mean_sim, _, max_sim = cosine_sim_stats(results)
    assert mean_sim == max_sim == 1.0

    assert coverage_ratio(results, threshold=1.0) == 1.0
