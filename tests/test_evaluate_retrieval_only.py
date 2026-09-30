"""Tests for evaluate_question()'s retrieval_only mode (Task 2, Part A):
runs retrieval + retrieval-tier metrics, never calls a generator.

search()/generate() are patched on pipeline_module (same pattern as
test_evaluate_efficiency_tracking.py). generate() is patched to raise, so
any accidental generation call in retrieval_only mode fails the test loudly
rather than silently succeeding.
"""
from __future__ import annotations

import pytest

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.evaluation.evaluate import evaluate_question
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.types import SearchResult


def _fake_chunks():
    return [
        SearchResult(text="The AMF handles registration.", distance=0.2,
                     metadata={"source": "23501-j00.md"}, doc_id="23501-j00.md::minilm::0"),
        SearchResult(text="Something else entirely.", distance=0.5,
                     metadata={"source": "other.md"}, doc_id="other.md::minilm::0"),
    ]


def _never_call_generate(*a, **kw):
    raise AssertionError("generate() must never be called in retrieval_only mode")


def _open_question(**kwargs) -> EvalQuestion:
    return EvalQuestion(
        question="What does the AMF handle?", keywords=[], corpus="3gpp",
        answer_type="open", reference_answer="Registration and mobility management.",
        **kwargs,
    )


class _FakeEmbedder:
    def embed(self, texts, is_query=False):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_retrieval_only_never_calls_the_generator(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", _never_call_generate)

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        retrieval_only=True,
    )

    assert result.error == ""
    assert result.answer == ""


def test_retrieval_only_populates_retrieval_tier_metrics(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", _never_call_generate)

    q = _open_question(
        chunk_id="23501-j00.md::minilm::0",
        evidence_char_start=0, evidence_char_end=10, evidence_doc_id="23501-j00.md",
    )
    result = evaluate_question(q, store=object(), embedder=_FakeEmbedder(), retrieval_only=True)

    assert result.error == ""
    assert result.chunks_used == 2
    assert result.mean_similarity > 0
    assert result.exact_chunk_hit == 1
    assert result.distinct_sources_retrieved == 2


def test_retrieval_only_leaves_generation_fields_at_defaults(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", _never_call_generate)

    result = evaluate_question(_open_question(), store=object(), embedder=_FakeEmbedder(), retrieval_only=True)

    assert result.accuracy == -1.0
    assert result.mc_tf_outcome is None
    assert result.faithfulness == -1.0
    assert result.answer_relevancy == -1.0
    assert result.context_recall == -1.0
    assert result.answer_correctness == -1.0
    assert result.total_tokens == 0
    assert result.cost is None
    assert result.token_source == ""


def test_retrieval_only_respects_budget(monkeypatch):
    import tiktoken

    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", _never_call_generate)

    enc = tiktoken.get_encoding("cl100k_base")
    budget = len(enc.encode(_fake_chunks()[0].text)) + 1

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        retrieval_only=True, context_token_budget=budget, candidate_pool=10,
    )

    assert result.error == ""
    assert result.budget == budget
    assert result.context_tokens <= budget
    assert result.n_chunks_used == 1


def test_difficulty_is_copied_from_the_question_onto_the_result(monkeypatch):
    """Task 3 Part D: EvalQuestion.difficulty must survive into
    QuestionResult.difficulty (independent of retrieval_only -- this mode
    just gives the cheapest test path)."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", _never_call_generate)

    result = evaluate_question(
        _open_question(difficulty="intermediate"), store=object(), embedder=_FakeEmbedder(),
        retrieval_only=True,
    )
    assert result.difficulty == "intermediate"


def test_difficulty_defaults_to_none_when_the_question_has_none(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", _never_call_generate)

    result = evaluate_question(_open_question(), store=object(), embedder=_FakeEmbedder(), retrieval_only=True)
    assert result.difficulty is None


def test_baseline_and_retrieval_only_together_raises():
    with pytest.raises(ValueError):
        evaluate_question(_open_question(), store=object(), embedder=_FakeEmbedder(),
                           baseline=True, retrieval_only=True)


def test_evaluate_corpus_retrieval_only_never_calls_generator(monkeypatch):
    from open_telco_rag.evaluation.evaluate import evaluate_corpus

    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", _never_call_generate)

    qs = [_open_question(), _open_question()]
    rag_result, baseline_result = evaluate_corpus(
        "3gpp", store=object(), embedder=_FakeEmbedder(), questions=qs,
        retrieval_only=True, delay=0.0,
    )

    assert baseline_result is None
    assert rag_result.n_questions == 2
    assert all(r.error == "" for r in rag_result.question_results)
    assert all(r.answer == "" for r in rag_result.question_results)
