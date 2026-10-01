"""Opt-in hard token bound (bound_tokens): every strategy, synthetic text, a stub tokenizer (no network)."""
from __future__ import annotations

import os
import re

import pytest

from open_telco_rag.core import pipeline
from open_telco_rag.core.chunk_bound import (
    enforce_bound,
    get_counter,
    split_to_bound,
    validate_bound,
)
from open_telco_rag.core.chunking import chunk_document
from open_telco_rag.types import Chunk

STRATEGIES = ["hierarchical_markdown", "text_baseline", "parent_child", "sliding_window_tokens"]
LIMIT = 50


def count(text: str) -> int:
    """stub tokenizer: a word of up to 4 characters, or one punctuation mark, is one token"""
    return len(re.findall(r"\w{1,4}|[^\w\s]", text))


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


HEADER = "| name | kind | value |\n|---|---|---|"
TABLE = HEADER + "\n" + "\n".join(f"| field_{i} | type_{i % 3} | v{i * 7} |" for i in range(70))

CASES = {
    "long_section": "# Spec\n\n## 1 Scope\n\n" + " ".join(f"Sentence number {i} says the AMF shall act." for i in range(60)),
    "giant_table": "# Spec\n\n## 2 Tables\n\nIntro line.\n\n" + TABLE + "\n\nAfter the table.",
    "unbroken_line": "# Spec\n\n## 3 Data\n\n" + "A1B2C3D4" * 90 + "\n\nShort tail paragraph.",
    "code_block": "# Spec\n\n## 4 Code\n\n```\n" + "\n".join(f"code_line_{i}(arg_a, arg_b) = {i};" for i in range(45)) + "\n```\n",
}


def chunks_for(doc: str, strategy: str, **extra):
    return chunk_document(doc, strategy, source_type="x", bound_tokens=LIMIT + 2, bound_tokenizer=count, bound_margin=2,
                          target_tokens=300, max_tokens=400, **extra)


def without_repeated_headers(chunks) -> str:
    joined = "\n".join(c.text for c in chunks)
    first = joined.find(HEADER)
    return joined[:first + len(HEADER)] + joined[first + len(HEADER):].replace(HEADER + "\n", "") if first >= 0 else joined


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_chunk_exceeds_the_bound_and_no_text_is_lost(strategy, case):
    doc = CASES[case]
    chunks = chunks_for(doc, strategy)
    assert chunks
    assert max(count(c.text) for c in chunks) <= LIMIT
    assert squash(without_repeated_headers(chunks)) == squash(doc)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_heading_breadcrumb_is_kept_on_every_piece(strategy):
    chunks = chunks_for(CASES["long_section"], strategy)
    if strategy in ("hierarchical_markdown", "parent_child"):
        assert all(c.heading for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_a_table_is_split_by_rows_with_the_header_repeated():
    pieces = [p for p, _, _ in split_to_bound(TABLE, LIMIT, count)]
    assert len(pieces) > 3
    assert all(p.startswith(HEADER) for p in pieces)
    rows = [line for p in pieces for line in p.split("\n")[2:]]
    assert rows == TABLE.split("\n")[2:], "every row exactly once, in order"
    assert all(count(p) <= LIMIT for p in pieces)


def test_a_table_inside_prose_does_not_lose_the_prose():
    text = "Lead sentence here. " * 10 + "\n" + TABLE + "\n" + "Trailing sentence there. " * 10
    pieces = [p for p, _, _ in split_to_bound(text, LIMIT, count)]
    assert all(count(p) <= LIMIT for p in pieces)
    assert squash(re.sub(re.escape(HEADER) + "\n", "", "\n".join(pieces), count=0)).count("Leadsentencehere.") == 10


def test_an_unbroken_line_is_cut_by_token_count_and_nothing_is_lost():
    text = "A1B2" * 500
    pieces = split_to_bound(text, LIMIT, count)
    assert "".join(p for p, _, _ in pieces) == text
    assert all(count(p) <= LIMIT for p, _, _ in pieces)
    assert len(pieces) == len(text) // (LIMIT * 4) + (len(text) % (LIMIT * 4) > 0)


def test_chunks_within_the_bound_are_returned_untouched():
    chunks = [Chunk(text="short text", heading="H", chunk_index=0, source_type="x")]
    assert enforce_bound(chunks, LIMIT, count)[0] is chunks[0]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_ids_are_unique_and_stable_through_the_ingest_worker(strategy, tmp_path):
    doc = tmp_path / "Rel-18" / "28312-i11.md"
    doc.parent.mkdir()
    doc.write_text(CASES["giant_table"] + "\n\n" + CASES["long_section"], encoding="utf8")
    kwargs = {"strategy": strategy, "bound_tokens": LIMIT + 2, "bound_tokenizer": count, "chunk_size": 400, "overlap": 0}
    if strategy == "sliding_window_tokens":
        kwargs = {"strategy": strategy, "bound_tokens": LIMIT + 2, "bound_tokenizer": count, "chunk_tokens": 300, "overlap_tokens": 0}

    def run():
        return pipeline._process_file_worker((doc, tmp_path, "3gpp", False, kwargs, 10, "minilm"))

    first, second = run(), run()
    assert first["status"] == "success"
    assert len(first["ids"]) == len(set(first["ids"])) and first["ids"] == second["ids"]
    assert first["chunks"] == second["chunks"]
    assert all(count(t) <= LIMIT for t in first["chunks"])


def test_token_sized_pieces_keep_exact_offsets():
    doc = CASES["long_section"]
    chunks = chunk_document(doc, "text_baseline", source_type="x", target_tokens=300, max_tokens=400,
                            bound_tokens=LIMIT + 2, bound_tokenizer=count)
    assert len(chunks) > 3
    for c in chunks:
        assert doc[c.char_start:c.char_end] == c.text


def test_default_output_is_identical_with_and_without_the_option_unset():
    for strategy in STRATEGIES:
        plain = chunk_document(CASES["long_section"], strategy, source_type="x", target_tokens=120, max_tokens=150)
        none = chunk_document(CASES["long_section"], strategy, source_type="x", target_tokens=120, max_tokens=150,
                              bound_tokens=None, bound_tokenizer=None, bound_margin=None)
        assert [(c.text, c.heading, c.chunk_index) for c in plain] == [(c.text, c.heading, c.chunk_index) for c in none]


def test_a_bound_larger_than_every_chunk_changes_nothing():
    plain = chunk_document(CASES["long_section"], "text_baseline", source_type="x", chunk_size=300, overlap=0)
    bounded = chunk_document(CASES["long_section"], "text_baseline", source_type="x", chunk_size=300, overlap=0,
                             bound_tokens=10_000, bound_tokenizer=count)
    assert [c.text for c in plain] == [c.text for c in bounded]


def test_the_margin_is_subtracted_from_the_bound():
    assert validate_bound(512, None, None) == 510
    assert validate_bound(512, None, 0) == 512
    assert validate_bound(512, "any", 12) == 500


@pytest.mark.parametrize("bad", [{"bound_tokens": 0}, {"bound_tokens": -1}, {"bound_tokens": 1.5}, {"bound_tokens": True},
                                 {"bound_tokens": 10, "bound_margin": 10}, {"bound_tokens": 10, "bound_margin": -1},
                                 {"bound_tokenizer": "x"}, {"bound_margin": 2}])
def test_invalid_settings_raise_value_error(bad):
    with pytest.raises(ValueError):
        chunk_document("some text", "text_baseline", source_type="x", **bad)


def test_the_bound_is_part_of_the_resume_fingerprint_including_the_tokenizer_name():
    class Store:
        path, collection_name = "db", "c"

    class Emb:
        model_key = "m"

    base = {"strategy": "text_baseline", "chunk_size": 300, "overlap": 0}
    plain = pipeline._fingerprint(pipeline._ingest_settings(Store, Emb, 100, base))
    bounded = pipeline._fingerprint(pipeline._ingest_settings(Store, Emb, 100, {**base, "bound_tokens": 512}))
    other_tok = pipeline._fingerprint(pipeline._ingest_settings(Store, Emb, 100, {**base, "bound_tokens": 512, "bound_tokenizer": "bge-m3"}))
    other_margin = pipeline._fingerprint(pipeline._ingest_settings(Store, Emb, 100, {**base, "bound_tokens": 512, "bound_margin": 12}))
    assert len({plain, bounded, other_tok, other_margin}) == 4
    settings = pipeline._ingest_settings(Store, Emb, 100, {**base, "bound_tokens": 512, "bound_tokenizer": "bge-m3"})
    assert settings["chunking"]["bound_tokenizer"] == "bge-m3"


def test_without_the_option_the_fingerprint_is_the_one_from_before():
    class Store:
        path, collection_name = "db", "c"

    class Emb:
        model_key = "m"

    settings = pipeline._ingest_settings(Store, Emb, 100, {"strategy": "text_baseline", "chunk_size": 300, "overlap": 0})
    assert set(settings["chunking"]) == {"strategy", "chunk_size", "overlap"}


@pytest.mark.skipif(not os.environ.get("RUN_REAL_TOKENIZER_TESTS"),
                    reason="needs network to fetch a tokenizer.json (set RUN_REAL_TOKENIZER_TESTS=1)")
def test_a_real_tokenizer_bounds_the_chunks_in_its_own_tokens():
    counter = get_counter("sentence-transformers/all-MiniLM-L6-v2")
    chunks = chunk_document(CASES["giant_table"] + CASES["long_section"], "text_baseline", source_type="x", chunk_size=2000, overlap=0,
                            bound_tokens=128, bound_tokenizer="sentence-transformers/all-MiniLM-L6-v2")
    assert max(counter(c.text) for c in chunks) <= 126
