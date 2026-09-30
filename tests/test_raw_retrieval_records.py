"""Retrieval distances (opt-in) and budget overflow events are recorded; defaults are unchanged."""
from __future__ import annotations

import pytest

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.core.retrieval import fill_context_budget
from open_telco_rag.evaluation.evaluate import evaluate_question
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.evaluation.results import question_to_dict
from open_telco_rag.types import SearchResult


def _cands(sizes):
    return [SearchResult(text="word " * n, distance=0.1 * (i + 1), metadata={}, doc_id=f"d{i}")
            for i, n in enumerate(sizes)]


def test_fill_stats_describe_each_policy():
    stats = {}
    fill_context_budget(_cands([30, 30, 30]), 50, "stop", stats=stats)
    assert stats == dict(candidates=3, kept=1, skipped=0, truncated=False, stopped_by_policy=True,
                         pool_exhausted=False, empty=False)
    fill_context_budget(_cands([30, 30, 10]), 50, "skip", stats=stats)
    assert stats["kept"] == 2 and stats["skipped"] == 1 and stats["stopped_by_policy"] is False
    fill_context_budget(_cands([30, 30]), 50, "truncate", stats=stats)
    assert stats["kept"] == 2 and stats["truncated"] is True and stats["stopped_by_policy"] is True
    fill_context_budget(_cands([80]), 50, "stop", stats=stats)
    assert stats["empty"] is True and stats["kept"] == 0
    fill_context_budget(_cands([5, 5]), 500, "stop", stats=stats)
    assert stats["pool_exhausted"] is True and stats["skipped"] == 0


def test_stats_are_optional_and_do_not_change_the_selection():
    a = fill_context_budget(_cands([30, 30, 10]), 50, "skip")
    b = fill_context_budget(_cands([30, 30, 10]), 50, "skip", stats={})
    assert [x.doc_id for x in a] == [x.doc_id for x in b]


def _question():
    return EvalQuestion(question="Which interface?", keywords=[], corpus="3gpp", answer_type="mc",
                        options={"A": "N1", "B": "Xn"}, reference_answer="B")


def _run(monkeypatch, **kw):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **k: _cands([30, 30, 10]))
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **k: {"answer": "B", "sources": [], "chunks_used": 1})
    return question_to_dict(evaluate_question(_question(), store=object(), embedder=object(),
                                              rag_generator_name="local", rag_model_name="m", **kw))


def test_default_results_have_neither_distances_nor_budget_stats(monkeypatch):
    d = _run(monkeypatch)
    assert not {"candidate_distances", "used_distances", "budget_stats"} & set(d)


def test_distances_are_recorded_only_on_request_and_align_with_the_ids(monkeypatch):
    d = _run(monkeypatch, record_retrieval_scores=True)
    assert d["candidate_distances"] == pytest.approx([0.1, 0.2, 0.3])
    assert len(d["candidate_distances"]) == len(d["candidate_chunk_ids"])
    assert len(d["used_distances"]) == len(d["used_chunk_ids"])


def test_a_budgeted_question_records_the_overflow_events(monkeypatch):
    d = _run(monkeypatch, context_token_budget=50, candidate_pool=3, budget_overflow="skip")
    # skip left one chunk out and still had room (42/50 tokens) once the pool ran out
    assert d["budget_stats"] == dict(candidates=3, kept=2, skipped=1, truncated=False,
                                     stopped_by_policy=False, pool_exhausted=True, empty=False)
    assert "candidate_distances" not in d
