"""bound_verify / bound_outliers: chunks that fit the main bound but exceed another embedder's limit."""
from __future__ import annotations

import logging

import pytest

from open_telco_rag.core import pipeline
from open_telco_rag.core.chunk_bound import enforce_bound, parse_verify, validate_bound
from open_telco_rag.core.chunking import chunk_document
from open_telco_rag.types import Chunk

TEXT = " ".join(f"Sentence {i} says the AMF shall act on value {i * 13}." for i in range(30))


def words(text: str) -> int:
    return len(text.split())


def double(text: str) -> int:
    """a verification tokenizer that counts twice as many tokens as the word counter"""
    return 2 * words(text)


def test_parse_verify():
    assert parse_verify("bge-m3=512, minilm=256") == [("bge-m3", 512), ("minilm", 256)]
    assert parse_verify(None) == [] and parse_verify("") == []
    for bad in ("bge-m3", "bge-m3=", "=512", "bge-m3=abc", "x=0"):
        with pytest.raises(ValueError):
            parse_verify(bad)


def test_default_is_warn_and_leaves_the_chunks_alone(caplog):
    chunks = [Chunk(text=TEXT, heading="H", chunk_index=0, source_type="x")]
    with caplog.at_level(logging.WARNING, logger="open_telco_rag.core.chunk_bound"):
        out = enforce_bound(chunks, 400, words, verify=[(double, 100)], margin=2)
    assert out[0] is chunks[0] and out[0].text == TEXT
    assert "exceed the" in caplog.text and "limit of 98 tokens" in caplog.text


def test_split_cuts_the_chunk_to_the_verification_limit_and_keeps_the_text():
    chunks = [Chunk(text=TEXT, heading="H", chunk_index=0, source_type="x")]
    out = enforce_bound(chunks, 400, words, verify=[(double, 100)], outliers="split", margin=2)
    assert len(out) > 1 and all(double(c.text) <= 98 for c in out)
    assert "".join(c.text for c in out).replace(" ", "") == TEXT.replace(" ", "")
    assert [c.heading for c in out] == ["H"] * len(out) and [c.chunk_index for c in out] == list(range(len(out)))


def test_a_chunk_that_fits_every_verifier_is_untouched(caplog):
    chunks = [Chunk(text="short chunk of text", heading="H", chunk_index=0, source_type="x")]
    with caplog.at_level(logging.WARNING):
        out = enforce_bound(chunks, 400, words, verify=[(double, 100)], outliers="split", margin=2)
    assert out[0] is chunks[0] and not caplog.records


def test_without_verifiers_nothing_is_checked_or_logged(caplog):
    chunks = [Chunk(text=TEXT, heading="H", chunk_index=0, source_type="x")]
    with caplog.at_level(logging.WARNING):
        assert enforce_bound(chunks, 400, words)[0] is chunks[0]
    assert not caplog.records


def test_through_chunk_document_with_real_tiktoken_encodings():
    kwargs = {"source_type": "x", "chunk_size": 4000, "overlap": 0, "bound_tokens": 80, "bound_margin": 0,
              "bound_verify": "tiktoken:cl100k_base=20"}
    warned = chunk_document(TEXT, "text_baseline", **kwargs)
    split = chunk_document(TEXT, "text_baseline", bound_outliers="split", **kwargs)
    assert len(split) > len(warned)
    from open_telco_rag.core.chunk_bound import get_counter
    assert max(get_counter("tiktoken:cl100k_base")(c.text) for c in split) <= 20


@pytest.mark.parametrize("bad", [{"bound_outliers": "split"}, {"bound_verify": "x=5"}])
def test_the_options_require_a_bound(bad):
    with pytest.raises(ValueError):
        chunk_document(TEXT, "text_baseline", source_type="x", **bad)


@pytest.mark.parametrize("bad", [{"bound_outliers": "split"}, {"bound_outliers": "nope", "bound_verify": "x=5"}, {"bound_verify": "x"}, {"bound_verify": 5}])
def test_invalid_verify_settings_raise(bad):
    with pytest.raises(ValueError):
        validate_bound(100, None, None, **bad)


def test_the_verification_settings_are_in_the_fingerprint():
    class Store:
        path, collection_name = "db", "c"

    class Emb:
        model_key = "m"

    base = {"strategy": "text_baseline", "chunk_size": 300, "overlap": 0, "bound_tokens": 400}
    fps = {pipeline._fingerprint(pipeline._ingest_settings(Store, Emb, 100, kw)) for kw in (
        base, {**base, "bound_verify": "bge-m3=512"}, {**base, "bound_verify": "bge-m3=256"},
        {**base, "bound_verify": "bge-m3=512", "bound_outliers": "split"})}
    assert len(fps) == 4
    settings = pipeline._ingest_settings(Store, Emb, 100, {**base, "bound_verify": "bge-m3=512"})
    assert settings["chunking"]["bound_outliers"] == "warn"
