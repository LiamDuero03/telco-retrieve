"""Unit tests for open_telco_rag.core.chunking's deterministic strategies.

chunk_lumber_chunker (the one LLM-aided strategy) is out of scope for this
CPU-only phase — see fake_llm_client in conftest.py if that's added later.
"""
from __future__ import annotations

import math

import pytest
import tiktoken

from open_telco_rag.core.chunking import (
    calculate_token_budget_safe_kwargs,
    chunk_document,
    chunk_hierarchical_markdown,
    chunk_parent_child,
    chunk_sliding_window_tokens,
    chunk_text_baseline,
)


# ── chunk_text_baseline ────────────────────────────────────────────────────────

def test_text_baseline_empty_text_returns_no_chunks():
    assert chunk_text_baseline("") == []
    assert chunk_text_baseline("   \n\n  ") == []


def test_text_baseline_single_short_paragraph_is_one_chunk():
    chunks = chunk_text_baseline("A short paragraph.", chunk_size=1000, source_type="oran")

    assert len(chunks) == 1
    assert chunks[0].text == "A short paragraph."
    assert chunks[0].chunk_index == 0
    assert chunks[0].heading is None
    assert chunks[0].source_type == "oran"


def test_text_baseline_splits_and_carries_overlap_into_the_next_chunk():
    para_a = "First paragraph sentence. " * 10
    para_b = "MARKER_TAIL sentence for overlap check."
    para_c = "Second block paragraph sentence. " * 10
    text = f"{para_a}\n\n{para_b}\n\n{para_c}"

    chunks = chunk_text_baseline(text, chunk_size=120, overlap=40)

    assert len(chunks) >= 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # the overlap window should carry MARKER_TAIL across the chunk boundary
    assert any("MARKER_TAIL" in c.text for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert "MARKER_TAIL" in joined


# ── chunk_hierarchical_markdown ────────────────────────────────────────────────

def test_hierarchical_markdown_empty_text_returns_no_chunks():
    assert chunk_hierarchical_markdown("") == []


def test_hierarchical_markdown_heading_matches_its_own_content_not_the_next_sections():
    """Regression: heading label used to leak from the next section; now flushed
    with the old heading before it's reassigned."""
    text = (
        "# Top\n\nIntro paragraph under Top.\n\n"
        "## Sub A\n\nContent under Sub A.\n\n"
        "### Sub A Detail\n\nDeep content.\n\n"
        "## Sub B\n\nContent under Sub B (Sub A Detail should no longer apply).\n\n"
    )
    chunks = chunk_hierarchical_markdown(text, chunk_size=4000)

    assert [c.heading for c in chunks] == [
        "Top",
        "Top > Sub A",
        "Top > Sub A > Sub A Detail",
        "Top > Sub B",                  # Sub A Detail correctly cleared
    ]
    assert "Intro paragraph under Top." in chunks[0].text
    assert "Content under Sub A." in chunks[1].text
    assert "Deep content." in chunks[2].text
    assert "Content under Sub B" in chunks[3].text


def test_hierarchical_markdown_flushes_on_size_overflow_within_a_section():
    text = "# Top\n\n" + "Paragraph text block. " * 50
    chunks = chunk_hierarchical_markdown(text, chunk_size=200)

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ── chunk_sliding_window_tokens ────────────────────────────────────────────────

def test_sliding_window_tokens_empty_text_returns_no_chunks():
    assert chunk_sliding_window_tokens("") == []


def test_sliding_window_tokens_splits_into_expected_number_of_chunks_with_no_overlap():
    tokenizer = tiktoken.get_encoding("cl100k_base")
    text = "token " * 250
    total_tokens = len(tokenizer.encode(text))

    chunks = chunk_sliding_window_tokens(text, chunk_tokens=100, overlap_tokens=0)

    assert len(chunks) == math.ceil(total_tokens / 100)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_sliding_window_tokens_single_chunk_when_overlap_at_least_chunk_size():
    """Guards against infinite loop when overlap_tokens >= chunk_tokens (explicit guard in source)."""
    text = "word " * 500
    chunks = chunk_sliding_window_tokens(text, chunk_tokens=10, overlap_tokens=10)
    assert len(chunks) == 1


# ── chunk_parent_child ─────────────────────────────────────────────────────────

def test_parent_child_children_are_linked_to_their_parent_via_metadata():
    text = (
        "# Top\n\nShort intro.\n\n"
        "## Section\n\n" + ("Paragraph text block. " * 30) + "\n\n"
    )
    chunks = chunk_parent_child(text, parent_size=2000, child_size=100, source_type="oran")

    assert len(chunks) >= 1
    for c in chunks:
        assert hasattr(c, "metadata")
        assert {"parent_id", "parent_context", "full_parent_text"} <= c.metadata.keys()
        assert c.metadata["parent_id"].startswith("oran_P_")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ── calculate_token_budget_safe_kwargs ─────────────────────────────────────────

def test_budget_unaffected_for_non_sliding_window_or_non_metadata_tagging():
    assert calculate_token_budget_safe_kwargs("text_baseline", 512, 50, "metadata_tagging") == (512, 50)
    assert calculate_token_budget_safe_kwargs("sliding_window_tokens", 512, 50, "none") == (512, 50)


def test_budget_reserves_space_for_metadata_tagging_on_sliding_window_tokens():
    size, overlap = calculate_token_budget_safe_kwargs("sliding_window_tokens", 512, 50, "metadata_tagging")
    assert (size, overlap) == (448, 38)  # 512-64, 50-int(64*0.2)=50-12


def test_budget_floors_chunk_size_at_128():
    size, overlap = calculate_token_budget_safe_kwargs("sliding_window_tokens", 150, 50, "metadata_tagging")
    assert size == 128
    assert overlap == 38


# ── chunk_document orchestrator ────────────────────────────────────────────────

def test_chunk_document_routes_text_baseline_by_string_name():
    text = "Paragraph one.\n\nParagraph two."
    via_document = chunk_document(text, strategy="text_baseline", chunk_size=1000, overlap=100)
    direct = chunk_text_baseline(text, chunk_size=1000, overlap=100)

    assert [c.text for c in via_document] == [c.text for c in direct]


def test_chunk_document_sliding_window_tokens_works_when_no_enrichment_kwarg_is_passed():
    """Mirrors real callers: pipeline.py pops enrichment kwarg, sweep_ingestions.py omits it."""
    text = "word " * 200
    via_document = chunk_document(text, strategy="sliding_window_tokens", chunk_size=100, overlap=10)
    direct = chunk_sliding_window_tokens(text, chunk_tokens=100, overlap_tokens=10)

    assert [c.text for c in via_document] == [c.text for c in direct]


def test_chunk_document_applies_token_budget_adjustment_when_enrichment_kwarg_is_passed():
    """Regression: chunk_document used kwargs.get() not pop(), leaking enrichment=
    into chunk_sliding_window_tokens (no **kwargs catch-all) and crashing."""
    text = "word " * 500
    via_document = chunk_document(
        text, strategy="sliding_window_tokens",
        chunk_size=512, overlap=50, enrichment="metadata_tagging",
    )
    direct = chunk_sliding_window_tokens(text, chunk_tokens=448, overlap_tokens=38)

    assert [c.text for c in via_document] == [c.text for c in direct]

    # enrichment="none" (the default) must also pass cleanly - old bug crashed regardless of value.
    chunk_document(text, strategy="sliding_window_tokens", chunk_size=100, overlap=10, enrichment="none")


def test_chunk_document_unknown_strategy_raises_value_error():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_document("some text", strategy="not_a_real_strategy")
