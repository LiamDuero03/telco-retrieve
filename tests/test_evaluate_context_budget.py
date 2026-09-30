"""Tests for evaluate_question()'s context_token_budget wiring -- confirms
result.context_tokens/n_chunks_used/budget/fill_ratio are populated when a
budget is set, and stay at their "not applicable" defaults when it isn't.

search() is patched on pipeline_module (same pattern as
test_evaluate_efficiency_tracking.py) so this exercises evaluate_question()'s
own budget-fill wiring without a real store/embedder.
"""
from __future__ import annotations

import tiktoken

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.evaluation.evaluate import evaluate_question
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.types import SearchResult

_ENC = tiktoken.get_encoding("cl100k_base")


def _n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _fake_chunks_multi() -> list[SearchResult]:
    return [
        SearchResult(text="alpha " * 20, distance=0.1, metadata={"source": "a.md"}, doc_id="c1"),
        SearchResult(text="bravo " * 20, distance=0.2, metadata={"source": "b.md"}, doc_id="c2"),
        SearchResult(text="charlie " * 20, distance=0.3, metadata={"source": "c.md"}, doc_id="c3"),
    ]


def _open_question() -> EvalQuestion:
    return EvalQuestion(
        question="What does the AMF handle?", keywords=[], corpus="3gpp",
        answer_type="open", reference_answer="Registration and mobility management.",
    )


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _judge_call_returning_empty_json():
    def _call(prompt, max_tokens=1024):
        return "[]"
    return _call


def test_budget_set_populates_context_fields(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks_multi())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "An answer.", "sources": [], "chunks_used": 1,
    })

    budget = _n_tokens("alpha " * 20) + 2  # fits exactly one of the three chunks

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json(),
        rag_generator_name="local", rag_model_name="local-model",
        context_token_budget=budget, candidate_pool=10, budget_overflow="stop",
    )

    assert result.error == ""
    assert result.budget == budget
    assert result.n_chunks_used == 1
    assert result.context_tokens <= budget
    assert result.fill_ratio == result.context_tokens / budget
    assert result.used_chunk_ids == ["c1"]


def test_budget_unset_leaves_context_fields_at_defaults(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks_multi())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "An answer.", "sources": [], "chunks_used": 1,
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json(),
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.error == ""
    assert result.budget is None
    assert result.n_chunks_used is None
    assert result.context_tokens is None
    assert result.fill_ratio is None
