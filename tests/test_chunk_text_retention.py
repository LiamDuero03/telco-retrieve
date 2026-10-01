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
    "see clause 4.4.2.0 for the details of the procedure",          # internal dots, no closing punctuation
    "| 4.4.2.0 | General | 91 |",
    "The value is 3.5 percent of the total and 7.25 of the rest",
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


@pytest.mark.parametrize("line", [
    "see clause 4.4.2.0 for the details",
    "| 4.4.2.0 | General | 91 |",
    "The value is 3.5 percent of the total",
    "version 1.2.3 released on 2026-01-02 (see 5.1)",
    "ends with a number 12.5",
])
def test_a_line_with_internal_punctuation_but_no_closing_punctuation_is_one_span(line):
    text = "first line.\n" + line + "\nlast line."
    pieces = [text[s:e] for s, e in _sentence_spans(text)]
    assert pieces == ["first line.", line, "last line."]


def test_spans_are_ordered_disjoint_and_cover_every_non_blank_character():
    text = ("Intro text. More 3.5 text! Really? Yes...\n\n   indented.line.with.dots\n| a | b |\n"
            "e.g. this one, and 4.2.1 that one\n...\n?!\nend")
    spans = _sentence_spans(text)
    assert all(a < b for a, b in spans) and all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    covered = "".join(text[s:e] for s, e in spans)
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
