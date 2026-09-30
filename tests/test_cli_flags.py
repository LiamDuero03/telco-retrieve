"""Smoke tests for CLI flags that had no test: --no-leaderboard, --inject-metadata,
--include-keywords (evaluate) and --distance-metric (ingest)."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import cli.main as cli_main
import open_telco_rag.core.pipeline as pipeline_module
import open_telco_rag.evaluation.evaluate as evaluate_module
from open_telco_rag.evaluation.evaluate import evaluate_question
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.types import SearchResult


@pytest.fixture
def evaluate_cli(monkeypatch, tmp_path):
    seen = {"kwargs": None, "leaderboard": 0}

    def fake_corpus(**kw):
        seen["kwargs"] = kw
        return SimpleNamespace(top_k=5, question_results=[]), None

    import open_telco_rag.evaluation._claude as claude_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(evaluate_module, "evaluate_corpus", fake_corpus)
    monkeypatch.setattr(claude_module, "make_llm_call", lambda *a, **k: (lambda prompt, n: ""))
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli_main, "_get_store", lambda *a, **k: SimpleNamespace(count=lambda: 3))
    monkeypatch.setattr(cli_main, "_update_leaderboard",
                        lambda *a, **k: seen.__setitem__("leaderboard", seen["leaderboard"] + 1)
                        or seen.__setitem__("lb_kwargs", k))

    def run(*args):
        cli_main.main(["evaluate", "3gpp", "--no-save", "--leaderboard", str(tmp_path / "lb.json"), *args])

    return seen, run


def test_no_leaderboard_skips_the_leaderboard_update(evaluate_cli):
    seen, run = evaluate_cli
    run("--generator", "local")
    assert seen["leaderboard"] == 1
    run("--generator", "local", "--no-leaderboard")
    assert seen["leaderboard"] == 1                       # unchanged: not updated the second time


def test_inject_metadata_reaches_evaluation_and_defaults_off(evaluate_cli):
    seen, run = evaluate_cli
    run("--generator", "local")
    assert seen["kwargs"]["inject_metadata"] is False
    run("--generator", "local", "--inject-metadata")
    assert seen["kwargs"]["inject_metadata"] is True
    assert seen["lb_kwargs"]["inject_metadata"] is True


def test_include_keywords_reaches_evaluation_not_just_the_leaderboard(evaluate_cli):
    seen, run = evaluate_cli
    run("--generator", "local")
    assert seen["kwargs"].get("include_keywords", False) is False
    assert seen["lb_kwargs"]["include_keywords"] is False
    run("--generator", "local", "--include-keywords")
    assert seen["kwargs"]["include_keywords"] is True          # used to be recorded but never applied
    assert seen["lb_kwargs"]["include_keywords"] is True


def _search_query(monkeypatch, question, **kw):
    seen = {}

    def fake_search(query, *a, **k):
        seen["query"] = query
        return [SearchResult(text="alpha", distance=0.1, metadata={}, doc_id="c1")]

    monkeypatch.setattr(pipeline_module, "search", fake_search)
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **k: {"answer": "B", "sources": [], "chunks_used": 1})
    evaluate_question(question, store=object(), embedder=object(), rag_generator_name="local",
                      rag_model_name="m", **kw)
    return seen["query"]


def _question():
    return EvalQuestion(question="Which interface?", keywords=["Xn", "handover"], corpus="3gpp", answer_type="mc",
                        options={"A": "N1", "B": "Xn"}, reference_answer="B", canonical_source="38300")


def test_the_query_changes_only_when_the_flags_are_set(monkeypatch):
    plain = _search_query(monkeypatch, _question())
    assert plain == "Which interface?"
    assert "Keywords: Xn handover" in _search_query(monkeypatch, _question(), include_keywords=True)
    injected = _search_query(monkeypatch, _question(), inject_metadata=True)
    assert "38300" in injected and injected != plain


# ── --distance-metric ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("metric,space", [("l2", "l2"), ("cosine", "cosine"), ("ip", "ip")])
def test_distance_metric_sets_the_index_space(tmp_path, metric, space):
    store = cli_main._get_store("3gpp", "minilm", str(tmp_path / "db"), collection="dm_test",
                                distance_metric=metric)
    assert store._collection.metadata["hnsw:space"] == space


def test_distance_metric_defaults_to_l2_and_reaches_the_store_from_the_ingest_command(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli_main, "_get_source", lambda *a, **k: object())
    monkeypatch.setattr(cli_main, "_get_store", lambda *a, **k: captured.update(k) or SimpleNamespace(collection_name="c"))
    monkeypatch.setattr(pipeline_module, "ingest", lambda **kw: SimpleNamespace(
        files_processed=0, chunks_stored=0, skipped_files=0, error_files=0))
    base = {"source_type": "oran", "use_marker": False, "model": "minilm", "strategy": "text_baseline", "enrichment": "none",
                "data": None, "db_path": "db", "no_resume": True, "collection": None, "chunk_size": None, "overlap": None,
                "target_tokens": None, "max_tokens": None, "overlap_tokens": None}
    cli_main._run_ingest(argparse.Namespace(**base))
    assert captured["distance_metric"] == "l2"
    cli_main._run_ingest(argparse.Namespace(**base, distance_metric="cosine"))
    assert captured["distance_metric"] == "cosine"


# ── --log-level ───────────────────────────────────────────────────────────────

def test_log_level_configures_logging_only_when_given(monkeypatch):
    import logging

    monkeypatch.setattr(cli_main, "_run_info", lambda args: None)
    root = logging.getLogger()
    before = (root.level, list(root.handlers))
    try:
        cli_main.main(["info"])
        assert (root.level, list(root.handlers)) == before          # untouched by default
        cli_main.main(["--log-level", "DEBUG", "info"])
        assert root.level == logging.DEBUG and root.handlers
        assert "%(levelname)s" in root.handlers[0].formatter._fmt
    finally:
        root.handlers[:] = before[1]
        root.setLevel(before[0])


def test_log_level_rejects_unknown_levels(capsys):
    with pytest.raises(SystemExit):
        cli_main.main(["--log-level", "LOUD", "info"])
