"""Chunk ids embed the relative file path with the ingesting OS's separator
(backslashes on Windows) while question files usually use forward slashes.
Retrieval metrics must not depend on that."""
from __future__ import annotations

import pytest

from open_telco_rag.evaluation.metrics.retrieval import (
    chunk_rank, distinct_sources_retrieved, exact_chunk_hit, span_hit_at_k, span_hit_in_context,
)
from open_telco_rag.types import SearchResult

WIN_ID = "Rel-18\\38300-j10.md::minilm::0"
POSIX_ID = "Rel-18/38300-j10.md::minilm::0"


def _chunk(doc_id, start=62, end=236):
    return SearchResult(text="handover", distance=0.1, metadata={"char_start": start, "char_end": end},
                        doc_id=doc_id)


@pytest.mark.parametrize("stored", [WIN_ID, POSIX_ID])
@pytest.mark.parametrize("evidence_doc", ["Rel-18/38300-j10.md", "Rel-18\\38300-j10.md"])
def test_span_hit_ignores_the_path_separator(stored, evidence_doc):
    assert span_hit_at_k([_chunk(stored)], 100, 200, evidence_doc) == 1
    assert span_hit_in_context([_chunk(stored)], 100, 200, evidence_doc) == 1


def test_span_hit_still_misses_a_different_document_or_range():
    assert span_hit_at_k([_chunk(WIN_ID)], 100, 200, "Rel-18/23501-j10.md") == 0
    assert span_hit_at_k([_chunk(WIN_ID)], 500, 600, "Rel-18/38300-j10.md") == 0


@pytest.mark.parametrize("stored,expected", [(WIN_ID, POSIX_ID), (POSIX_ID, WIN_ID), (WIN_ID, WIN_ID)])
def test_chunk_rank_and_exact_hit_match_ids_across_separators(stored, expected):
    other = SearchResult(text="x", distance=0.2, metadata={}, doc_id="Rel-18/23501-j10.md::minilm::0")
    assert chunk_rank([other, _chunk(stored)], expected, None) == 2
    assert exact_chunk_hit([_chunk(stored)], expected, None) == 1


def test_a_different_chunk_index_still_does_not_match():
    assert exact_chunk_hit([_chunk("Rel-18\\38300-j10.md::minilm::1")], POSIX_ID, None) == 0


def test_distinct_sources_counts_one_document_once_whatever_the_separator():
    results = [_chunk(WIN_ID), _chunk("Rel-18/38300-j10.md::minilm::5"), _chunk("Rel-18/23501-j10.md::minilm::0")]
    assert distinct_sources_retrieved(results) == 2
