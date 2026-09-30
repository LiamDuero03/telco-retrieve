"""Reranker truncation and embedder truncation/usage counters are recorded in
saved results when they exist, and omitted (no new keys) otherwise."""
from __future__ import annotations

import json
from pathlib import Path

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.evaluation.evaluate import _embedder_stats, _reranker_stats, evaluate_corpus
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.types import SearchResult

QUESTION = EvalQuestion(question="Which interface?", keywords=[], corpus="3gpp", answer_type="mc",
                        options={"A": "N1", "B": "E2"}, reference_answer="B")


def _run(monkeypatch, tmp_path, embedder=None, reranker=None):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: [
        SearchResult(text="alpha", distance=0.1, metadata={}, doc_id="c1")])
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "B", "sources": [], "chunks_used": 1})
    evaluate_corpus("3gpp", store=object(), embedder=embedder or object(), questions=[QUESTION],
                    model="m", output_dir=tmp_path, reranker=reranker, rag_generator_name="local",
                    rag_generator_model="local-model", progress_bar=False)
    return json.loads(next(Path(tmp_path).glob("eval_*.json")).read_text(encoding="utf8"))


def test_default_run_saves_no_component_stats(monkeypatch, tmp_path):
    data = _run(monkeypatch, tmp_path)
    assert "reranker_stats" not in data and "embedder_stats" not in data


def test_otel_reranker_truncation_is_saved(monkeypatch, tmp_path):
    class StubOTel:  # exposes the OTelReranker counters without loading a model
        scored_count, truncated_count, truncation_rate = 4, 1, 0.25
        query_max_chars, doc_max_chars = 5, 1000

        def rerank(self, q, c, top_k=5):
            return c[:top_k]

    data = _run(monkeypatch, tmp_path, reranker=StubOTel())
    assert data["reranker_stats"] == {"truncation_rate": 0.25, "truncated": 1, "scored": 4,
                                      "query_max_chars": 5, "doc_max_chars": 1000}


def test_reranker_without_counters_saves_nothing(monkeypatch, tmp_path):
    class Plain:
        def rerank(self, q, c, top_k=5):
            return c[:top_k]

    assert "reranker_stats" not in _run(monkeypatch, tmp_path, reranker=Plain())


def test_embedder_truncation_and_openrouter_usage_are_saved_when_present(monkeypatch, tmp_path):
    class Emb:
        scored_count, truncated_count, truncation_rate = 10, 2, 0.2
        total_prompt_tokens, total_cost_usd = 1234, 0.0005

    data = _run(monkeypatch, tmp_path, embedder=Emb())
    assert data["embedder_stats"] == {"truncation_rate": 0.2, "truncated": 2, "scored": 10,
                                      "prompt_tokens": 1234, "cost_usd": 0.0005}


def test_embedder_without_openrouter_usage_or_tracking_saves_nothing():
    class LocalEmb:  # e.g. SentenceTransformerEmbedder with tracking off
        scored_count = 0
        truncated_count = 0

    assert _embedder_stats(LocalEmb()) is None
    assert _embedder_stats(object()) is None
    assert _reranker_stats(object()) is None


def test_stats_read_the_real_otel_reranker_attributes():
    from open_telco_rag.embedders.reranker import OTelReranker

    rr = OTelReranker(query_max_chars=5, doc_max_chars=1000)  # constructor only, no model load
    rr.scored_count, rr.truncated_count = 4, 1
    assert _reranker_stats(rr) == {"truncation_rate": 0.25, "truncated": 1, "scored": 4,
                                   "query_max_chars": 5, "doc_max_chars": 1000}
