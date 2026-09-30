"""Token chunking fallbacks: a paragraph larger than max_tokens is split at
sentence boundaries, and as a last resort at a hard token cut. Every chunk stays
within max_tokens and is an exact slice of the source."""
from __future__ import annotations

import pytest
import tiktoken

from open_telco_rag.core.chunking import chunk_document

_ENC = tiktoken.get_encoding("cl100k_base")
STRATEGIES = ["text_baseline", "hierarchical_markdown", "parent_child"]


def _check(text, chunks, max_tokens):
    assert chunks
    for c in chunks:
        assert len(_ENC.encode(c.text)) <= max_tokens
        assert c.n_tokens == len(_ENC.encode(c.text))
        assert text[c.char_start:c.char_end] == c.text


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_one_huge_paragraph_of_sentences_is_split_at_sentences(strategy):
    text = " ".join(f"Sentence number {i} describes procedure {i} of the AMF." for i in range(200))
    chunks = chunk_document(text, strategy, target_tokens=40, max_tokens=50)
    _check(text, chunks, 50)
    assert len(chunks) > 5
    # sentence-level splitting: chunks end on sentence punctuation
    assert sum(c.text.rstrip().endswith(".") for c in chunks) >= len(chunks) - 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_one_huge_paragraph_without_punctuation_uses_the_hard_token_cut(strategy):
    text = " ".join(f"word{i}" for i in range(3000))          # no sentence boundaries at all
    chunks = chunk_document(text, strategy, target_tokens=60, max_tokens=64)
    _check(text, chunks, 64)
    assert "".join(c.text for c in chunks).replace(" ", "") == text.replace(" ", "")  # nothing lost


def test_hard_cut_on_multibyte_text_keeps_exact_slices():
    text = ("网络功能 " * 400) + ("🙂 5G réseau d'accès " * 200)
    chunks = chunk_document(text, "text_baseline", target_tokens=50, max_tokens=60)
    _check(text, chunks, 60)
    assert "".join(c.text for c in chunks).replace(" ", "") == text.replace(" ", "")


def test_a_single_token_budget_still_terminates():
    text = "alpha beta gamma delta " * 30
    chunks = chunk_document(text, "text_baseline", target_tokens=1, max_tokens=1)
    _check(text, chunks, 1)


@pytest.mark.parametrize("bad_text", ["", "   ", "\n\n\n"])
@pytest.mark.parametrize("strategy", STRATEGIES + ["sliding_window_tokens"])
def test_empty_or_blank_input_gives_no_chunks(strategy, bad_text):
    assert chunk_document(bad_text, strategy, target_tokens=50) == []


def test_text_shorter_than_one_chunk_gives_a_single_chunk():
    text = "The AMF handles registration."
    for strategy in STRATEGIES + ["sliding_window_tokens"]:
        chunks = chunk_document(text, strategy, target_tokens=200)
        assert len(chunks) == 1 and chunks[0].text.strip() == text


# ── legacy sliding window with overlap >= size drops text: unchanged, but now loud ──

def test_legacy_sliding_window_with_overlap_at_least_size_warns_and_keeps_its_old_output(caplog):
    import logging

    text = " ".join(f"Sentence {i} about procedure {i} of the AMF." for i in range(300))
    with caplog.at_level(logging.WARNING, logger="open_telco_rag.core.chunking"):
        chunks = chunk_document(text, "sliding_window_tokens", chunk_tokens=50, overlap_tokens=80)
    assert len(chunks) == 1                       # same output as before the warning existed
    assert any("dropped" in r.getMessage() and "overlap" in r.getMessage() for r in caplog.records)


def test_a_valid_legacy_sliding_window_does_not_warn(caplog):
    import logging

    text = " ".join(f"Sentence {i} about procedure {i} of the AMF." for i in range(300))
    with caplog.at_level(logging.WARNING, logger="open_telco_rag.core.chunking"):
        chunks = chunk_document(text, "sliding_window_tokens", chunk_tokens=50, overlap_tokens=10)
    assert len(chunks) > 5 and not caplog.records
