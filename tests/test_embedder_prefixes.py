"""Tests for Step 6: per-model query_prefix/document_prefix and
query_prompt_name/document_prompt_name (configs/embedding_models.json),
applied in SentenceTransformerEmbedder.embed() based on the new is_query
param. _load() is bypassed by setting _model directly (a fake recording
what encode() was called with) -- no real model download needed.

the plain e5 and bge keys deliberately carry NO prefixes (that would change
their embeddings and mismatch existing indexes); the documented prefixes live
on the opt-in e5-prefixed / bge-prefixed keys.
"""
from __future__ import annotations

import json

import numpy as np

from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder


class _FakeModel:
    """Records encode() calls; returns one all-zero vector per text."""

    def __init__(self, dimensions: int, max_seq_length: int = 99999):
        self.dimensions = dimensions
        self.max_seq_length = max_seq_length
        self.device = "cpu"
        self.tokenizer = None  # length-check no-ops without a real tokenizer
        self.last_texts: list[str] = []
        self.last_kwargs: dict = {}

    def encode(self, texts, **kwargs):
        self.last_texts = list(texts)
        self.last_kwargs = kwargs
        return np.zeros((len(texts), self.dimensions))


def _wire_fake_model(embedder: SentenceTransformerEmbedder, dimensions: int = 8) -> _FakeModel:
    fake = _FakeModel(dimensions)
    embedder._model = fake
    return fake


def test_e5_prefixed_applies_query_and_document_prefixes_on_both_sides():
    embedder = SentenceTransformerEmbedder(model_key="e5-prefixed")
    fake = _wire_fake_model(embedder)

    embedder.embed(["what does the AMF do"], is_query=True)
    assert fake.last_texts == ["query: what does the AMF do"]

    embedder.embed(["the AMF handles registration"], is_query=False)
    assert fake.last_texts == ["passage: the AMF handles registration"]


def test_bge_prefixed_applies_query_prefix_only_not_document_prefix():
    embedder = SentenceTransformerEmbedder(model_key="bge-prefixed")
    fake = _wire_fake_model(embedder)

    embedder.embed(["a query"], is_query=True)
    assert fake.last_texts[0].startswith("Represent this sentence for searching relevant passages: ")

    embedder.embed(["a document chunk"], is_query=False)
    assert fake.last_texts == ["a document chunk"]  # no document_prefix configured


def test_qwen3_embedding_applies_instruction_style_query_prefix():
    embedder = SentenceTransformerEmbedder(model_key="qwen3-embedding-0.6b")
    fake = _wire_fake_model(embedder, dimensions=1024)

    embedder.embed(["what does the AMF do"], is_query=True)
    assert fake.last_texts[0].startswith("Instruct:")
    assert fake.last_texts[0].endswith("Query: what does the AMF do")

    embedder.embed(["a document chunk"], is_query=False)
    assert fake.last_texts == ["a document chunk"]  # no document prefix needed


def test_embeddinggemma_uses_prompt_name_not_a_literal_prefix():
    embedder = SentenceTransformerEmbedder(model_key="embeddinggemma-300m")
    fake = _wire_fake_model(embedder, dimensions=768)

    embedder.embed(["a query"], is_query=True)
    assert fake.last_texts == ["a query"]  # text itself is untouched
    assert fake.last_kwargs.get("prompt_name") == "Retrieval-query"

    embedder.embed(["a document"], is_query=False)
    assert fake.last_kwargs.get("prompt_name") == "Retrieval-document"


def test_bge_m3_local_needs_no_prefix_at_all():
    embedder = SentenceTransformerEmbedder(model_key="bge-m3")
    fake = _wire_fake_model(embedder, dimensions=1024)

    embedder.embed(["a query"], is_query=True)
    assert fake.last_texts == ["a query"]
    assert "prompt_name" not in fake.last_kwargs


def test_models_without_any_prefix_config_are_unaffected_by_is_query():
    """Backward compatibility: minilm/mpnet/otel-* have no prefix keys, so
    is_query must be a complete no-op for them."""
    for model_key in ("minilm", "mpnet", "otel-109m"):
        embedder = SentenceTransformerEmbedder(model_key=model_key)
        fake = _wire_fake_model(embedder)

        embedder.embed(["some text"], is_query=True)
        as_query = list(fake.last_texts)
        embedder.embed(["some text"], is_query=False)
        as_document = list(fake.last_texts)

        assert as_query == as_document == ["some text"], model_key


def test_default_is_query_false_matches_ingest_document_path():
    """embed() called with no is_query arg at all (every ingest call site)
    must behave exactly like is_query=False."""
    embedder = SentenceTransformerEmbedder(model_key="e5-prefixed")
    fake = _wire_fake_model(embedder)

    embedder.embed(["a document chunk"])
    assert fake.last_texts == ["passage: a document chunk"]


# ── Registry sanity: new entries resolve without downloading anything ──────

_EMBEDDING_MODELS_FILE = SentenceTransformerEmbedder.__init__.__defaults__[1]


def test_new_registry_entries_are_present_with_expected_ids_and_dimensions():
    available = json.loads(_EMBEDDING_MODELS_FILE.read_text(encoding="utf-8"))

    assert available["qwen3-embedding-0.6b"]["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert available["qwen3-embedding-0.6b"]["dimensions"] == 1024

    assert available["embeddinggemma-300m"]["model_name"] == "google/embeddinggemma-300m"
    assert available["embeddinggemma-300m"]["dimensions"] == 768

    assert available["bge-m3"]["model_name"] == "BAAI/bge-m3"
    assert available["bge-m3"]["dimensions"] == 1024


def test_new_entries_construct_without_loading_the_real_model():
    # Constructor only reads the JSON registry -- _load() (the real
    # download) is never triggered by __init__.
    for key in ("qwen3-embedding-0.6b", "embeddinggemma-300m", "bge-m3"):
        embedder = SentenceTransformerEmbedder(model_key=key)
        assert embedder._model is None


# ── backward compatibility: existing keys and embedders are unchanged ─────────

def test_plain_e5_and_bge_keys_still_embed_without_any_prefix():
    for key in ("e5", "bge"):
        embedder = SentenceTransformerEmbedder(model_key=key)
        fake = _wire_fake_model(embedder)
        embedder.embed(["some text"], is_query=True)
        assert fake.last_texts == ["some text"], key
        embedder.embed(["some text"])
        assert fake.last_texts == ["some text"], key


def test_only_the_intended_registry_keys_carry_prefix_settings():
    available = json.loads(_EMBEDDING_MODELS_FILE.read_text(encoding="utf-8"))
    prefix_keys = {"query_prefix", "document_prefix", "query_prompt_name", "document_prompt_name"}
    with_prefixes = {k for k, v in available.items() if prefix_keys & set(v)}
    assert with_prefixes == {"e5-prefixed", "bge-prefixed", "qwen3-embedding-0.6b",
                             "qwen3-embedding-8b", "embeddinggemma-300m",
                             "embeddinggemma-300m-fp32"}


def test_retrieval_works_with_an_embedder_that_has_no_is_query_parameter():
    from open_telco_rag.embedders.calls import embed_query as _embed_query

    class Legacy:
        def embed(self, texts):
            return [[1.0, 2.0] for _ in texts]

    class Modern:
        def embed(self, texts, is_query=False):
            return [[9.0 if is_query else 0.0] for _ in texts]

    assert _embed_query(Legacy(), "q") == [1.0, 2.0]
    assert _embed_query(Modern(), "q") == [9.0]


# ── text-vs-text metric embeddings are prefix-free for every key ──────────────

def _recording_embedders():
    """(plain, prefixed, fakes): e5 vs e5-prefixed with recording fake models."""
    plain = SentenceTransformerEmbedder(model_key="e5")
    prefixed = SentenceTransformerEmbedder(model_key="e5-prefixed")
    return plain, prefixed, _wire_fake_model(plain), _wire_fake_model(prefixed)


def test_embed_plain_sends_identical_text_for_plain_and_prefixed_keys():
    from open_telco_rag.embedders.calls import embed_plain

    plain, prefixed, fake_plain, fake_prefixed = _recording_embedders()
    embed_plain(plain, ["an answer", "a reference"])
    embed_plain(prefixed, ["an answer", "a reference"])
    assert fake_plain.last_texts == fake_prefixed.last_texts == ["an answer", "a reference"]
    assert fake_prefixed.last_kwargs == fake_plain.last_kwargs


def test_embed_plain_skips_prompt_names_too():
    from open_telco_rag.embedders.calls import embed_plain

    gemma = SentenceTransformerEmbedder(model_key="embeddinggemma-300m")
    fake = _wire_fake_model(gemma)
    embed_plain(gemma, ["text"])
    assert "prompt_name" not in fake.last_kwargs and fake.last_texts == ["text"]


def test_answer_correctness_and_relevancy_embeddings_ignore_prefixed_keys():
    from open_telco_rag.evaluation.metrics.generation import answer_relevancy
    from open_telco_rag.evaluation.metrics.ground_truth import answer_correctness

    def judge(prompt, max_tokens):
        return '{"questions": ["what does the AMF do?"], "claims": []}'

    seen = {}
    for name, key in (("plain", "e5"), ("prefixed", "e5-prefixed")):
        embedder = SentenceTransformerEmbedder(model_key=key)
        fake = _wire_fake_model(embedder)
        answer_correctness("the AMF registers UEs", "the AMF handles registration", embedder,
                           llm_call=judge)
        correctness_texts = list(fake.last_texts)
        answer_relevancy("what does the AMF do?", "it registers UEs", embedder, llm_call=judge)
        seen[name] = (correctness_texts, list(fake.last_texts))
    assert seen["plain"] == seen["prefixed"]
    assert all(not t.startswith(("query:", "passage:")) for group in seen["prefixed"] for t in group)


def test_retrieval_still_applies_the_query_prefix_for_prefixed_keys():
    from open_telco_rag.embedders.calls import embed_query

    embedder = SentenceTransformerEmbedder(model_key="e5-prefixed")
    fake = _wire_fake_model(embedder)
    embed_query(embedder, "what does the AMF do")
    assert fake.last_texts == ["query: what does the AMF do"]
