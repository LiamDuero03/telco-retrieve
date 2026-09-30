"""Unit tests for the target_tokens/max_tokens token-sizing path added to
every chunking strategy (see open_telco_rag/core/chunking.py).

Each strategy is checked against the two invariants the fork is designed
to guarantee on the new path:
  1. no chunk exceeds max_tokens
  2. doc_text[chunk.char_start:chunk.char_end] == chunk.text (exact slice)

...and that the legacy (character-sized) path is untouched: char_start/
char_end stay at their -1 default and chunk text is unaffected.
"""
from __future__ import annotations

import tiktoken

from open_telco_rag.core.chunking import (
    chunk_document,
    chunk_hierarchical_markdown,
    chunk_lumber_chunker,
    chunk_parent_child,
    chunk_sliding_window_tokens,
    chunk_text_baseline,
)

_ENC = tiktoken.get_encoding("cl100k_base")


def _n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


SAMPLE_DOC = (
    "# Introduction\n\n"
    "This document describes the E2 interface. " * 8 + "\n\n"
    "The E2 interface connects the Near-RT RIC to E2 nodes such as the CU and DU. " * 10 + "\n\n"
    "## Architecture\n\n"
    "The architecture separates control and user plane functions. " * 12 + "\n\n"
    "E2AP is the application protocol used over the E2 interface. " * 6 + "\n\n"
    "### Deep Section\n\n"
    "This nested subsection covers RIC subscription procedures in detail. " * 15 + "\n\n"
    "A final short paragraph closes out the document."
)


def _assert_exact_slices(doc_text: str, chunks) -> None:
    for c in chunks:
        assert c.char_start >= 0 and c.char_end > c.char_start, (c.char_start, c.char_end)
        assert doc_text[c.char_start:c.char_end] == c.text
        assert c.n_tokens == _n_tokens(c.text)


def _assert_max_tokens(chunks, max_tokens: int) -> None:
    for c in chunks:
        assert c.n_tokens <= max_tokens, f"chunk exceeds max_tokens: {c.n_tokens} > {max_tokens} ({c.text[:60]!r})"


# ── text_baseline ────────────────────────────────────────────────────────────

def test_text_baseline_token_path_respects_max_tokens_and_exact_slices():
    chunks = chunk_text_baseline(SAMPLE_DOC, target_tokens=60, max_tokens=80, overlap_tokens=10)
    assert len(chunks) >= 2
    _assert_max_tokens(chunks, 80)
    _assert_exact_slices(SAMPLE_DOC, chunks)


def test_text_baseline_token_path_overlap_repeats_a_paragraph():
    # Each paragraph is ~15 tokens; target_tokens=35 fits 2 per group,
    # overlap_tokens=15 pulls the last paragraph of a group into the next one.
    doc = "\n\n".join(f"Paragraph number {i} with some filler words to pad it out a little." for i in range(6))
    chunks = chunk_text_baseline(doc, target_tokens=35, max_tokens=40, overlap_tokens=15)
    assert len(chunks) >= 2
    texts = [c.text for c in chunks]
    # some paragraph content should appear in two consecutive chunks (the overlap)
    overlap_found = any(
        any(line in texts[i + 1] for line in texts[i].split("\n\n"))
        for i in range(len(texts) - 1)
    )
    assert overlap_found


def test_text_baseline_legacy_path_unaffected_by_new_fields():
    chunks = chunk_text_baseline("Paragraph one.\n\nParagraph two.", chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].char_start == -1
    assert chunks[0].char_end == -1
    assert chunks[0].n_tokens == 0
    assert chunks[0].text == "Paragraph one.\n\nParagraph two."


# ── hierarchical_markdown ────────────────────────────────────────────────────

def test_hierarchical_markdown_token_path_respects_max_tokens_and_exact_slices():
    chunks = chunk_hierarchical_markdown(SAMPLE_DOC, target_tokens=80, max_tokens=100)
    assert len(chunks) >= 2
    _assert_max_tokens(chunks, 100)
    _assert_exact_slices(SAMPLE_DOC, chunks)


def test_hierarchical_markdown_legacy_path_unaffected_by_new_fields():
    chunks = chunk_hierarchical_markdown("# Heading\n\nBody text.", chunk_size=4000)
    assert chunks[0].char_start == -1
    assert chunks[0].char_end == -1
    assert chunks[0].n_tokens == 0


# ── sliding_window_tokens ────────────────────────────────────────────────────

def test_sliding_window_tokens_token_path_offsets_and_cap():
    chunks = chunk_sliding_window_tokens(SAMPLE_DOC, target_tokens=40, max_tokens=40, overlap_tokens=5)
    assert len(chunks) >= 2
    _assert_max_tokens(chunks, 40)
    _assert_exact_slices(SAMPLE_DOC, chunks)


def test_sliding_window_tokens_via_chunk_document_does_not_inherit_legacy_overlap_default():
    # Regression: chunk_document's sliding_window_tokens special-casing used
    # to unconditionally inject legacy chunk_tokens=512/overlap_tokens=50
    # defaults into kwargs even when the caller passed target_tokens/
    # max_tokens, and the stray overlap_tokens=50 then got treated as an
    # explicit value on the token path (which distinguishes "not given" from
    # "given" via a sentinel, not via `is not None`) -- forcing near-total
    # overlap (a 20-token target with a 20-token document produced 62
    # chunks advancing by ~1 token each, instead of ~4).
    chunks = chunk_document(SAMPLE_DOC, strategy="sliding_window_tokens", target_tokens=20, max_tokens=25)
    _assert_max_tokens(chunks, 25)
    _assert_exact_slices(SAMPLE_DOC, chunks)
    # No overlap was requested -> consecutive chunks must not re-cover text.
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start >= a.char_end


def test_sliding_window_tokens_legacy_path_unchanged():
    # Same call shape existing callers (chunk_document's routing) use today.
    legacy = chunk_sliding_window_tokens(SAMPLE_DOC, chunk_tokens=50, overlap_tokens=5)
    assert all(c.char_start == -1 and c.char_end == -1 and c.n_tokens == 0 for c in legacy)
    # decoded+stripped text, not necessarily an exact slice -- that's the point of the fork
    assert all(c.text == c.text.strip() for c in legacy)


# ── parent_child ─────────────────────────────────────────────────────────────

def test_parent_child_token_path_respects_max_tokens_and_exact_slices():
    chunks = chunk_parent_child(SAMPLE_DOC, target_tokens=30, parent_target_tokens=120, max_tokens=40)
    assert len(chunks) >= 2
    _assert_max_tokens(chunks, 40)
    _assert_exact_slices(SAMPLE_DOC, chunks)
    assert all(c.metadata.get("parent_id") for c in chunks)


def test_parent_child_legacy_path_unaffected_by_new_fields():
    chunks = chunk_parent_child(SAMPLE_DOC, parent_size=2048, child_size=512, source_type="oran")
    assert len(chunks) >= 1
    assert all(c.char_start == -1 and c.char_end == -1 and c.n_tokens == 0 for c in chunks)


# ── lumber_chunker ───────────────────────────────────────────────────────────

def test_lumber_chunker_token_path_respects_max_tokens_and_exact_slices(fake_llm_client):
    # fake_llm_client always replies "OK" (no "ID <n>" match) -> every group
    # falls back to the fixed p_idx+3 path, which can still exceed max_tokens
    # and so exercises the _bounded_split cap.
    chunks = chunk_lumber_chunker(SAMPLE_DOC, llm_client=fake_llm_client, theta=40, max_tokens=50)
    assert len(chunks) >= 2
    _assert_max_tokens(chunks, 50)
    _assert_exact_slices(SAMPLE_DOC, chunks)


def test_lumber_chunker_legacy_path_unaffected_by_new_fields(fake_llm_client):
    chunks = chunk_lumber_chunker(SAMPLE_DOC, llm_client=fake_llm_client, theta=550)
    assert len(chunks) >= 1
    assert all(c.char_start == -1 and c.char_end == -1 and c.n_tokens == 0 for c in chunks)
