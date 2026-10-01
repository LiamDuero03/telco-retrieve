"""Token-sized chunking keeps every character of the input (synthetic text only)."""
from __future__ import annotations

import re

import pytest

from open_telco_rag.core.chunking import _sentence_spans, chunk_document

STRATEGIES = ["hierarchical_markdown", "text_baseline", "parent_child", "sliding_window_tokens"]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _table(n: int) -> str:
    return "\n".join(f"| field_{i} | type_{i % 4} | value {i * 7} |" for i in range(n))


# one blank-line-delimited block that mixes prose with lines that end without punctuation
BLOCK = "\n".join([
    " ".join(f"The function {i} shall be configured before the procedure starts." for i in range(12)),
    _table(40),
    "- a list item without a full stop",
    "    code_line(x, y)",
    " ".join(f"Closing sentence {i} ends here." for i in range(12)),
])
DOC = f"# Spec\n\n## 1 Scope\n\n{BLOCK}\n\n## 2 Other\n\nShort closing paragraph."


def test_every_line_is_covered_by_a_sentence_span():
    text = "first line without a stop\n| a | b |\n| c | d |\nlast line."
    covered = "".join(text[s:e] for s, e in _sentence_spans(text))
    assert _squash(covered) == _squash(text)


def test_a_punctuated_sentence_after_an_unpunctuated_line_is_still_split():
    text = "| a | b |\nOne sentence. Another sentence.\n"
    pieces = [text[s:e] for s, e in _sentence_spans(text)]
    assert pieces == ["| a | b |", "One sentence.", "Another sentence."]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_concatenated_chunks_equal_the_input_without_overlap(strategy):
    chunks = chunk_document(DOC, strategy, source_type="x", target_tokens=60, max_tokens=80)
    assert len(chunks) > 3
    assert _squash("".join(c.text for c in chunks)) == _squash(DOC)


@pytest.mark.parametrize("strategy", ["text_baseline", "sliding_window_tokens"])
def test_with_overlap_every_character_is_covered_by_some_chunk(strategy):
    chunks = chunk_document(DOC, strategy, source_type="x", target_tokens=60, max_tokens=80, overlap_tokens=10)
    covered = [False] * len(DOC)
    for c in chunks:
        assert DOC[c.char_start:c.char_end] == c.text
        for i in range(c.char_start, c.char_end):
            covered[i] = True
    missing = [i for i, ch in enumerate(DOC) if not ch.isspace() and not covered[i]]
    assert not missing
