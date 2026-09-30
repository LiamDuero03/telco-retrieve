"""Budgeted context assembly (context_token_budget/candidate_pool/budget_overflow) is OPT-IN.

  (c) default retrieval is the plain top_k path
  (d) budgeted retrieval works
  error cases: candidate_pool/budget_overflow without a budget, bad values
  serialization: unset fields are omitted, old result files still load
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import tiktoken

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.core import chunking
from open_telco_rag.core.retrieval import execute_retrieval, resolve_budget_args
from open_telco_rag.evaluation.evaluate import evaluate_corpus, evaluate_question
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.evaluation.results import (
    CorpusResult,
    QuestionResult,
    aggregate,
    question_to_dict,
    result_to_dict,
    save_results,
)
from open_telco_rag.types import SearchResult

BUDGET_KEYS = {"context_tokens", "n_chunks_used", "budget", "fill_ratio"}
CORPUS_BUDGET_KEYS = {"mean_context_tokens", "mean_fill_ratio"}


class _Embedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _Store:
    def __init__(self, n=12):
        self.rows = [SearchResult(text=("word " * (10 + i)).strip(), distance=0.01 * i,
                                  metadata={}, doc_id=f"c{i}") for i in range(n)]

    def search(self, vector, n_results=5):
        return self.rows[:n_results]


def _ntok(text):
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


# ── default = plain top_k path ────────────────────────────────────────────────

def test_default_retrieval_equals_explicit_none_and_plain_top_k():
    store, emb = _Store(), _Embedder()
    plain = execute_retrieval("q", store, emb, top_k=4)
    explicit = execute_retrieval("q", store, emb, top_k=4, context_token_budget=None,
                                 candidate_pool=None, budget_overflow=None)
    assert [r.doc_id for r in plain] == [r.doc_id for r in explicit] == ["c0", "c1", "c2", "c3"]


def test_default_retrieval_never_initializes_the_tokenizer(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("tiktoken.get_encoding called without a budget")

    monkeypatch.setattr(tiktoken, "get_encoding", boom)
    monkeypatch.setattr(chunking, "_ENCODING_CACHE", {})
    assert execute_retrieval("q", _Store(), _Embedder(), top_k=3)


def test_importing_retrieval_does_not_initialize_the_tokenizer():
    code = "import tiktoken;tiktoken.get_encoding = lambda *a, **k: 1 / 0;import open_telco_rag.core.retrieval"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


# ── budgeted retrieval ────────────────────────────────────────────────────────

@pytest.mark.parametrize("overflow", ["stop", "skip", "truncate"])
def test_budgeted_retrieval_respects_budget(overflow):
    budget = 60
    out = execute_retrieval("q", _Store(), _Embedder(), context_token_budget=budget,
                            candidate_pool=10, budget_overflow=overflow)
    assert out and sum(_ntok(r.text) for r in out) <= budget


def test_budget_alone_uses_default_pool_and_policy():
    assert resolve_budget_args(100, None, None) == (100, 30, "stop")
    assert resolve_budget_args(None, None, None) == (None, None, None)


# ── error cases ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs", [{"candidate_pool": 10}, {"budget_overflow": "skip"},
                                    {"candidate_pool": 10, "budget_overflow": "stop"}])
def test_pool_or_overflow_without_budget_raises(kwargs):
    with pytest.raises(ValueError, match="require context_token_budget"):
        execute_retrieval("q", _Store(), _Embedder(), **kwargs)
    with pytest.raises(ValueError, match="require context_token_budget"):
        pipeline_module.search("q", _Store(), _Embedder(), **kwargs)


@pytest.mark.parametrize("budget", [0, -5, 1.5, "100", True])
def test_invalid_budget_raises(budget):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_budget_args(budget, None, None)


@pytest.mark.parametrize("pool", [0, -1, 2.5, True])
def test_invalid_pool_raises(pool):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_budget_args(100, pool, None)


def test_unknown_overflow_policy_raises():
    with pytest.raises(ValueError, match="Unknown budget_overflow"):
        resolve_budget_args(100, 10, "bogus")


def _question():
    return EvalQuestion(question="Which interface?", keywords=[], corpus="3gpp", answer_type="mc",
                        options={"A": "N1", "B": "E2"}, reference_answer="B")


def test_evaluate_question_rejects_pool_without_budget():
    with pytest.raises(ValueError, match="require context_token_budget"):
        evaluate_question(_question(), store=object(), embedder=object(), candidate_pool=10)


def test_evaluate_corpus_rejects_overflow_without_budget():
    with pytest.raises(ValueError, match="require context_token_budget"):
        evaluate_corpus("3gpp", store=object(), embedder=object(), questions=[_question()],
                        model="m", budget_overflow="skip")


# ── serialization ─────────────────────────────────────────────────────────────

def test_default_question_serializes_without_budget_keys():
    assert BUDGET_KEYS.isdisjoint(question_to_dict(_qr(question="q", corpus="c")))


def test_budgeted_question_serializes_budget_keys():
    d = question_to_dict(_qr(question="q", corpus="c", context_tokens=50,
                                        n_chunks_used=2, budget=100, fill_ratio=0.5))
    assert (d["context_tokens"], d["n_chunks_used"], d["budget"], d["fill_ratio"]) == (50, 2, 100, 0.5)


def test_aggregate_omits_budget_keys_without_budgeted_questions():
    agg = aggregate([_qr(question="q", corpus="c")])
    assert CORPUS_BUDGET_KEYS.isdisjoint(agg)


def test_aggregate_adds_budget_keys_with_budgeted_questions():
    agg = aggregate([_qr(question="q", corpus="c", context_tokens=50, n_chunks_used=2,
                                    budget=100, fill_ratio=0.5)])
    assert agg["mean_context_tokens"] == 50.0 and agg["mean_fill_ratio"] == 0.5


def _qr(**kw):
    kw.setdefault("answer", "a")
    kw.setdefault("sources", [])
    kw.setdefault("chunks_used", 0)
    return QuestionResult(**kw)


def _corpus_result(**over):
    base = {"corpus": "3gpp", "model": "m", "top_k": 5, "n_questions": 1,
                "question_results": [_qr(question="q", corpus="3gpp")]}
    base.update(over)
    return CorpusResult(**base)


def test_saved_default_results_have_no_budget_keys(tmp_path):
    path = save_results(_corpus_result(), tmp_path)
    data = json.loads(Path(path).read_text(encoding="utf8"))
    assert CORPUS_BUDGET_KEYS.isdisjoint(data)
    assert all(BUDGET_KEYS.isdisjoint(q) for q in data["question_results"])


def test_old_result_files_still_load():
    old = result_to_dict(_corpus_result())  # exactly what an old (pre-budget) file contains
    qs = old.pop("question_results")
    restored = CorpusResult(**old, question_results=[QuestionResult(**q) for q in qs])
    assert restored.mean_context_tokens is None
    assert restored.question_results[0].budget is None
    assert asdict(restored)["mean_fill_ratio"] is None  # raw asdict still shows the None default


# ── CLI ───────────────────────────────────────────────────────────────────────

def _eval_args(**over):
    base = {"model": "minilm", "generator": "claude", "generator_model": None, "no_save": True,
                "output": "results", "context_token_budget": None, "candidate_pool": None,
                "budget_overflow": None}
    base.update(over)
    return argparse.Namespace(**base)


def test_cli_rejects_pool_without_budget_before_any_work():
    import cli.main as cli_main

    with pytest.raises(SystemExit, match="require context_token_budget"):
        cli_main._run_evaluate(_eval_args(candidate_pool=10))


def test_cli_evaluate_help_lists_the_optional_flags():
    result = subprocess.run([sys.executable, "-m", "cli.main", "evaluate", "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    for flag in ("--context-token-budget", "--candidate-pool", "--budget-overflow"):
        assert flag in result.stdout


# ── a budget smaller than the first chunk ────────────────────────────────────

def _big_candidates(n=3, words=200):
    return [SearchResult(text="word " * words, distance=0.1, metadata={}, doc_id=f"d{i}") for i in range(n)]


@pytest.mark.parametrize("overflow", ["stop", "skip"])
def test_empty_context_from_a_too_small_budget_is_warned_about(overflow, caplog):
    import logging

    from open_telco_rag.core.retrieval import fill_context_budget

    with caplog.at_level(logging.WARNING, logger="open_telco_rag.core.retrieval"):
        assert fill_context_budget(_big_candidates(), 50, overflow) == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("context is EMPTY" in m and "truncate" in m for m in messages)
    assert not any("candidate_pool may be too small" in m for m in messages)  # not the misleading one


def test_truncate_and_normal_fills_do_not_warn(caplog):
    import logging

    from open_telco_rag.core.retrieval import fill_context_budget

    with caplog.at_level(logging.WARNING, logger="open_telco_rag.core.retrieval"):
        out = fill_context_budget(_big_candidates(), 50, "truncate")
        assert len(out) == 1
        assert fill_context_budget(_big_candidates(2, 10), 1000, "stop")
    assert [r for r in caplog.records if "EMPTY" in r.getMessage()] == []


def test_a_pool_that_runs_out_still_gets_the_pool_warning(caplog):
    import logging

    from open_telco_rag.core.retrieval import fill_context_budget

    with caplog.at_level(logging.WARNING, logger="open_telco_rag.core.retrieval"):
        out = fill_context_budget(_big_candidates(2, 10), 5000, "stop")
    assert len(out) == 2
    assert any("candidate_pool may be too small" in r.getMessage() for r in caplog.records)
