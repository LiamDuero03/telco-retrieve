"""Tests for Step 4: OTelReranker's configurable query_max_chars/
doc_max_chars (previously hardcoded 512/1024), truncated_count/scored_count/
truncation_rate, and load_reranker() reading the two keys optionally from
reranker_models.json.

_load() is bypassed by setting _model/_tokenizer directly (fake, no HF
download) -- see _FakeTokenizer/_FakeModel.
"""
from __future__ import annotations

import json
import logging

import torch

from open_telco_rag.embedders import reranker as reranker_module
from open_telco_rag.embedders.reranker import CrossEncoderReranker, OTelReranker, load_reranker
from open_telco_rag.types import SearchResult


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.last_pairs: list = []

    def __call__(self, pairs, padding=True, truncation=True, max_length=2048, return_tensors="pt"):
        self.last_pairs = pairs
        return _FakeBatchEncoding()


class _FakeModel:
    def __init__(self, tokenizer: _FakeTokenizer):
        self._tokenizer = tokenizer
        self.device = "cpu"

    def __call__(self, **kwargs):
        class _Out:
            pass
        out = _Out()
        out.logits = torch.zeros(len(self._tokenizer.last_pairs))
        return out


def _wired_reranker(**kwargs) -> tuple[OTelReranker, _FakeTokenizer]:
    r = OTelReranker(**kwargs)
    tok = _FakeTokenizer()
    r._tokenizer = tok
    r._model = _FakeModel(tok)
    return r, tok


def _result(text: str, doc_id: str) -> SearchResult:
    return SearchResult(text=text, distance=0.1, metadata={}, doc_id=doc_id)


# ── configurable limits, defaults preserved ─────────────────────────────────

def test_defaults_match_the_previously_hardcoded_values():
    r = OTelReranker()
    assert r.query_max_chars == 512
    assert r.doc_max_chars == 1024


def test_custom_limits_are_applied_in_score_batch():
    r, tok = _wired_reranker(query_max_chars=10, doc_max_chars=20)
    r._score_batch("x" * 50, ["y" * 50])
    assert tok.last_pairs == [["x" * 10, "y" * 20]]


# ── truncation counters ──────────────────────────────────────────────────────

def test_truncated_count_and_scored_count_track_truncation(caplog):
    r, _ = _wired_reranker(query_max_chars=100, doc_max_chars=10)
    with caplog.at_level(logging.DEBUG, logger="open_telco_rag.embedders.reranker"):
        r._score_batch("short query", ["a document that is definitely longer than ten chars", "short"])

    assert r.scored_count == 2
    assert r.truncated_count == 1
    assert r.truncation_rate == 0.5
    assert any("document truncated" in rec.message for rec in caplog.records)


def test_no_warning_and_zero_truncation_rate_when_nothing_is_truncated():
    r, _ = _wired_reranker(query_max_chars=1000, doc_max_chars=1000)
    r._score_batch("short query", ["short doc one", "short doc two"])
    assert r.truncated_count == 0
    assert r.truncation_rate == 0.0


def test_truncation_rate_is_zero_before_anything_is_scored():
    r, _ = _wired_reranker()
    assert r.truncation_rate == 0.0


def test_query_truncation_warns_once_and_counts_every_document(caplog):
    r, _ = _wired_reranker(query_max_chars=5, doc_max_chars=1000)
    with caplog.at_level(logging.DEBUG, logger="open_telco_rag.embedders.reranker"):
        r._score_batch("a query longer than five chars", ["doc a", "doc b", "doc c"])

    assert r.scored_count == 3
    assert r.truncated_count == 3  # query truncation affects every pair in the batch
    query_warnings = [rec for rec in caplog.records if "query truncated" in rec.message]
    assert len(query_warnings) == 1  # logged once per call, not once per document


# ── rerank() end-to-end still works with the fake model ────────────────────

def test_rerank_end_to_end_respects_top_k_and_sets_rerank_score():
    r, _ = _wired_reranker(query_max_chars=1000, doc_max_chars=1000)
    results = [_result(f"doc {i}", f"d{i}") for i in range(5)]

    out = r.rerank("some query", results, top_k=2)

    assert len(out) == 2
    assert all("rerank_score" in c.metadata for c in out)


# ── CrossEncoderReranker: explicitly unchanged (no truncation config) ──────

def test_cross_encoder_reranker_has_no_truncation_attributes():
    r = CrossEncoderReranker()
    assert not hasattr(r, "query_max_chars")
    assert not hasattr(r, "doc_max_chars")
    assert not hasattr(r, "truncation_rate")


# ── load_reranker(): optional query_max_chars/doc_max_chars from config ────

def test_load_reranker_reads_optional_truncation_keys(tmp_path, monkeypatch):
    config = {
        "custom-otel": {
            "model_name": "farbodtavakkoli/OTel-Reranker-0.6B",
            "backend": "otel",
            "query_max_chars": 256,
            "doc_max_chars": 2048,
        },
        "custom-otel-default-limits": {
            "model_name": "farbodtavakkoli/OTel-Reranker-0.6B",
            "backend": "otel",
        },
    }
    config_path = tmp_path / "reranker_models.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(reranker_module, "_RERANKER_MODELS_FILE", config_path)

    custom = load_reranker("custom-otel")
    assert isinstance(custom, OTelReranker)
    assert custom.query_max_chars == 256
    assert custom.doc_max_chars == 2048

    default_limits = load_reranker("custom-otel-default-limits")
    assert default_limits.query_max_chars == 512
    assert default_limits.doc_max_chars == 1024


# ── defaults equal the previously hardcoded behavior ──────────────────────────

def test_defaults_are_the_previous_hardcoded_limits():
    r = OTelReranker()
    assert (r.query_max_chars, r.doc_max_chars) == (512, 1024)


def test_default_scoring_truncates_exactly_like_the_old_hardcoded_slices():
    r, tok = _wired_reranker()
    query, doc = "q" * 700, "d" * 1500
    r._score_batch(query, [doc])
    assert tok.last_pairs == [[query[:512], doc[:1024]]]


def test_truncation_logs_stay_below_warning_by_default(caplog):
    r, _ = _wired_reranker()
    with caplog.at_level(logging.INFO, logger="open_telco_rag.embedders.reranker"):
        r._score_batch("q" * 700, ["d" * 1500])
    assert not caplog.records
    assert r.truncated_count == 1


def test_load_reranker_without_limit_keys_uses_defaults(monkeypatch, tmp_path):
    cfg = tmp_path / "rr.json"
    cfg.write_text(json.dumps({"k": {"model_name": "x/y", "backend": "otel"}}), encoding="utf8")
    monkeypatch.setattr(reranker_module, "_RERANKER_MODELS_FILE", cfg)
    r = load_reranker("k")
    assert (r.query_max_chars, r.doc_max_chars) == (512, 1024)
