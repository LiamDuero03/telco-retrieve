"""Unit tests for open_telco_rag.core.enrichment's three deterministic
strategies (none, acronym_expansion, metadata_tagging) and enrich_chunks()
dispatch. enrich_llm_metadata (the LLM-based strategy) is covered
separately in test_enrich_llm_metadata.py, with httpx.AsyncClient mocked —
no real LLM/network calls anywhere in this test suite.
"""
from __future__ import annotations

import pytest

from open_telco_rag.core.enrichment import enrich_acronym_expansion, enrich_chunks, enrich_metadata_tagging
from open_telco_rag.types import Chunk


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, heading=None, chunk_index=idx, source_type="oran")


# ── enrich_acronym_expansion ───────────────────────────────────────────────────

GLOSSARY = {
    "RIC": "RAN Intelligent Controller",
    "E2AP": "E2 Application Protocol",
    "RAN": "Radio Access Network",
}


def test_appends_expansion_suffix_for_a_matched_acronym():
    chunks = [_chunk("The Near-RT RIC exposes the E2 interface.")]
    result = enrich_acronym_expansion(chunks, glossary_dict=GLOSSARY)

    assert "[Acronym Expansions]:" in result[0].text
    assert "RIC: RAN Intelligent Controller" in result[0].text
    assert result[0].metadata["acronyms_expanded"] == ["RIC"]


def test_word_boundary_prevents_matching_acronym_embedded_in_a_hyphenated_word():
    """'RAN' must not match inside 'O-RAN' — only a standalone 'RAN' counts."""
    chunks = [_chunk("This document is published by the O-RAN Alliance.")]
    result = enrich_acronym_expansion(chunks, glossary_dict=GLOSSARY)

    assert "[Acronym Expansions]" not in result[0].text
    assert not hasattr(result[0], "metadata")


def test_standalone_acronym_still_matches_word_boundary_check():
    chunks = [_chunk("The RAN architecture is defined here.")]
    result = enrich_acronym_expansion(chunks, glossary_dict=GLOSSARY)

    assert result[0].metadata["acronyms_expanded"] == ["RAN"]


def test_multiple_distinct_acronyms_are_all_expanded_and_sorted():
    chunks = [_chunk("E2AP is used between the RIC and E2 nodes over the RAN.")]
    result = enrich_acronym_expansion(chunks, glossary_dict=GLOSSARY)

    found = set(result[0].metadata["acronyms_expanded"])
    assert found == {"E2AP", "RIC", "RAN"}
    # suffix lines should be alphabetically sorted
    suffix = result[0].text.split("[Acronym Expansions]:")[1]
    assert suffix.index("E2AP:") < suffix.index("RAN:") < suffix.index("RIC:")


def test_chunk_with_no_matching_acronyms_is_left_completely_untouched():
    original_text = "Nothing here matches any glossary term."
    chunks = [_chunk(original_text)]
    result = enrich_acronym_expansion(chunks, glossary_dict=GLOSSARY)

    assert result[0].text == original_text
    assert not hasattr(result[0], "metadata")  # never even initialized


def test_empty_glossary_returns_chunks_unchanged():
    chunks = [_chunk("The RIC exposes E2AP.")]
    result = enrich_acronym_expansion(chunks, glossary_dict={})
    assert result[0].text == "The RIC exposes E2AP."


# ── enrich_metadata_tagging ────────────────────────────────────────────────────

def test_prepends_document_context_header_and_writes_metadata():
    chunks = [_chunk("Body content.")]
    result = enrich_metadata_tagging(chunks, source="doc1.md", release="Rel-19")

    assert result[0].metadata["source"] == "doc1.md"
    assert result[0].metadata["release"] == "Rel-19"
    assert result[0].metadata["enriched"] is True
    assert result[0].text == (
        "[Document Context -> Source: doc1.md | Release: Rel-19]\n\nBody content."
    )


def test_llm_client_and_model_name_kwargs_are_excluded_from_header_and_metadata():
    chunks = [_chunk("Body content.")]
    result = enrich_metadata_tagging(
        chunks, llm_client=object(), model_name="some-model", release="Rel-19",
    )

    assert "llm_client" not in result[0].metadata
    assert "model_name" not in result[0].metadata
    assert "release" in result[0].metadata
    assert "llm_client" not in result[0].text
    assert "model_name" not in result[0].text


def test_no_extra_kwargs_still_sets_enriched_flag_but_does_not_prepend_header():
    original_text = "Body content."
    chunks = [_chunk(original_text)]
    result = enrich_metadata_tagging(chunks)

    assert result[0].metadata == {"enriched": True}
    assert result[0].text == original_text  # nothing prepended — no header_parts


# ── enrich_chunks dispatch ─────────────────────────────────────────────────────

def test_dispatches_none_strategy_as_a_pure_passthrough():
    chunks = [_chunk("Untouched.")]
    result = enrich_chunks(chunks, "none")
    assert result[0].text == "Untouched."
    assert not hasattr(result[0], "metadata")


def test_dispatches_acronym_expansion_by_string_name():
    chunks = [_chunk("The RIC is here.")]
    result = enrich_chunks(chunks, "acronym_expansion", glossary_dict=GLOSSARY)
    assert result[0].metadata["acronyms_expanded"] == ["RIC"]


def test_dispatches_metadata_tagging_by_string_name():
    chunks = [_chunk("Body content.")]
    result = enrich_chunks(chunks, "metadata_tagging", source="doc1.md")
    assert result[0].metadata["source"] == "doc1.md"


def test_strategy_name_is_case_insensitive():
    chunks = [_chunk("Untouched.")]
    result = enrich_chunks(chunks, "NONE")
    assert result[0].text == "Untouched."


def test_unknown_strategy_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown enrichment strategy"):
        enrich_chunks([_chunk("x")], "not_a_real_strategy")
