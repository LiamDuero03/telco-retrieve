"""Tests for SentenceTransformerEmbedder's max_seq_length/truncation_rate
tracking (Step 5/6: master_grid_v2's dry-run needs to know, ahead of a real
ingestion run, which (embedder, target token size) pairs would silently
truncate chunks).

Uses the real all-MiniLM-L6-v2 model (small, CPU-only, no GPU/API key --
same exception documented in conftest.py for the ingest integration test),
since a meaningful truncation check needs a real tokenizer and a real
max_seq_length, not something worth mocking away.
"""
from __future__ import annotations

from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder


def test_max_seq_length_is_none_before_first_embed():
    embedder = SentenceTransformerEmbedder(model_key="minilm", track_truncation=True)
    assert embedder.max_seq_length is None
    assert embedder.truncation_rate == 0.0


def test_max_seq_length_known_after_first_embed_and_truncation_tracked():
    embedder = SentenceTransformerEmbedder(model_key="minilm", track_truncation=True)
    short_text = "The AMF handles registration and mobility management."
    # all-MiniLM-L6-v2's max_seq_length is 256 tokens; this comfortably exceeds it.
    long_text = "network function " * 400

    embedder.embed([short_text, long_text])

    assert embedder.max_seq_length is not None
    assert embedder.scored_count == 2
    assert embedder.truncated_count == 1
    assert embedder.truncation_rate == 0.5


def test_no_truncation_when_every_text_fits(caplog):
    import logging
    embedder = SentenceTransformerEmbedder(model_key="minilm", track_truncation=True)
    with caplog.at_level(logging.WARNING, logger="open_telco_rag.embedders.sentense_transformers"):
        embedder.embed(["short text one", "short text two"])

    assert embedder.truncated_count == 0
    assert embedder.truncation_rate == 0.0
    assert not any("exceed max_seq_length" in rec.message for rec in caplog.records)


def test_truncation_warns_once_per_batch_with_any_overlong_text(caplog):
    import logging
    embedder = SentenceTransformerEmbedder(model_key="minilm", track_truncation=True)
    long_text = "network function " * 400
    with caplog.at_level(logging.WARNING, logger="open_telco_rag.embedders.sentense_transformers"):
        embedder.embed(["short", long_text])

    warnings = [rec for rec in caplog.records if "exceed max_seq_length" in rec.message]
    assert len(warnings) == 1


def test_scored_count_accumulates_across_multiple_embed_calls():
    embedder = SentenceTransformerEmbedder(model_key="minilm", track_truncation=True)
    embedder.embed(["short one"])
    embedder.embed(["short two", "short three"])
    assert embedder.scored_count == 3
    assert embedder.truncated_count == 0


def test_tracking_is_off_by_default_and_does_no_extra_tokenization():
    embedder = SentenceTransformerEmbedder(model_key="minilm")
    assert embedder.track_truncation is False
    calls = []
    embedder._load()
    real_encode = embedder._model.tokenizer.encode
    embedder._model.tokenizer.encode = lambda *a, **k: calls.append(1) or real_encode(*a, **k)
    embedder.embed(["short text", "network function " * 400])
    assert calls == []
    assert embedder.scored_count == 0 and embedder.truncation_rate == 0.0


def test_tracking_does_not_change_the_embeddings():
    texts = ["short text", "network function " * 400]
    off = SentenceTransformerEmbedder(model_key="minilm").embed(texts)
    on = SentenceTransformerEmbedder(model_key="minilm", track_truncation=True).embed(texts)
    assert off == on
