"""Failed questions are counted, excluded from retrieval means, shown in the summary,
and a run where every question fails exits with a distinct status."""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

import cli.main as cli_main
import open_telco_rag.evaluation.evaluate as evaluate_module
from open_telco_rag.evaluation.results import (
    CorpusResult,
    QuestionResult,
    aggregate,
    print_summary,
    result_to_dict,
)


def _q(**kw):
    base = {"answer": "a", "sources": [], "chunks_used": 1, "question": "q", "corpus": "3gpp",
            "answer_type": "mc", "accuracy": 1.0, "mean_similarity": 0.8, "coverage_ratio": 1.0,
            "distinct_sources_retrieved": 2}
    base.update(kw)
    return QuestionResult(**base)


def _failed(error_type="ConnectionError"):
    return QuestionResult(question="q", corpus="3gpp", answer="", sources=[], chunks_used=0, answer_type="mc",
                          error="boom", error_type=error_type)


KEYS = {"n_errors", "error_rate", "errors_by_type"}


def test_a_healthy_run_has_no_failure_keys_and_unchanged_means():
    agg = aggregate([_q(), _q(mean_similarity=0.6)])
    assert KEYS.isdisjoint(agg) and KEYS.isdisjoint(agg["by_format"]["mc"])
    assert agg["mean_similarity"] == pytest.approx(0.7)


def test_failures_are_counted_by_type_and_leave_the_retrieval_means_alone():
    agg = aggregate([_q(), _q(mean_similarity=0.6), _failed(), _failed("Timeout")])
    assert agg["n_errors"] == 2 and agg["error_rate"] == 0.5
    assert agg["errors_by_type"] == {"ConnectionError": 1, "Timeout": 1}
    assert agg["mean_similarity"] == pytest.approx(0.7)             # was dragged down to 0.35 before
    assert agg["mean_coverage_ratio"] == 1.0 and agg["mean_distinct_sources_retrieved"] == 2.0
    assert agg["n_questions"] == 4                                  # failures stay in the denominator of the rate
    fmt = agg["by_format"]["mc"]
    assert fmt["n_errors"] == 2 and fmt["mean_similarity"] == pytest.approx(0.7)


def test_when_every_question_fails_the_retrieval_means_are_not_applicable_not_zero():
    agg = aggregate([_failed(), _failed()])
    assert agg["n_errors"] == 2 and agg["error_rate"] == 1.0
    assert agg["mean_similarity"] == -1.0 and agg["mean_coverage_ratio"] == -1.0


def test_failure_keys_are_saved_only_when_something_failed():
    def corpus(results):
        return CorpusResult(corpus="3gpp", model="m", top_k=5, question_results=results, **aggregate(results))

    assert KEYS.isdisjoint(result_to_dict(corpus([_q()])))
    saved = result_to_dict(corpus([_q(), _failed()]))
    assert saved["n_errors"] == 1 and saved["errors_by_type"] == {"ConnectionError": 1}


def test_the_summary_shows_failures_only_when_there_are_some():
    def summary(results):
        c = CorpusResult(corpus="3gpp", model="m", top_k=5, question_results=results, **aggregate(results))
        out = io.StringIO()
        with redirect_stdout(out):
            print_summary(c)
        return out.getvalue()

    assert "FAILED" not in summary([_q()])
    text = summary([_q(), _failed("Timeout")])
    assert "FAILED questions: 1/2 (50%)" in text and "Timeout x1" in text


# ── exit status ───────────────────────────────────────────────────────────────

@pytest.fixture
def evaluate_cli(monkeypatch, tmp_path):
    outcome = {"results": []}
    board = {"n": 0}

    def fake_corpus(**kw):
        agg = aggregate(outcome["results"])
        return CorpusResult(corpus="3gpp", model="m", top_k=5, question_results=outcome["results"], **agg), None

    import open_telco_rag.evaluation._claude as claude_module

    monkeypatch.setattr(evaluate_module, "evaluate_corpus", fake_corpus)
    monkeypatch.setattr(claude_module, "make_llm_call", lambda *a, **k: (lambda p, n: ""))
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli_main, "_get_store", lambda *a, **k: SimpleNamespace(count=lambda: 3))
    monkeypatch.setattr(cli_main, "_update_leaderboard", lambda *a, **k: board.__setitem__("n", board["n"] + 1))

    def run():
        cli_main.main(["evaluate", "3gpp", "--no-save", "--generator", "local", "--leaderboard",
                       str(tmp_path / "lb.json")])

    return outcome, board, run


def test_all_questions_failing_exits_with_a_distinct_status_and_message(evaluate_cli, capsys):
    outcome, board, run = evaluate_cli
    outcome["results"] = [_failed(), _failed()]
    with pytest.raises(SystemExit) as exc:
        run()
    assert exc.value.code == cli_main.EXIT_ALL_QUESTIONS_FAILED == 3
    assert "Every one of the 2 questions failed" in capsys.readouterr().err
    assert board["n"] == 0                                # no leaderboard row for an evaluation that never ran


def test_partial_failures_exit_normally_and_still_update_the_leaderboard(evaluate_cli):
    outcome, board, run = evaluate_cli
    outcome["results"] = [_q(), _failed()]
    run()                                                  # returns normally (exit status 0)
    assert board["n"] == 1


def test_a_healthy_run_is_unchanged(evaluate_cli):
    outcome, board, run = evaluate_cli
    outcome["results"] = [_q(), _q()]
    run()
    assert board["n"] == 1
