"""Regression: --search-method existed on `search`/`ask` but was never
registered as an argparse argument on `evaluate`, even though
_run_evaluate()/evaluate_corpus()/the leaderboard already thread it through
via getattr(args, "search_method", "dense") -- so it silently stayed at
"dense" regardless of intent. Now registered, matching search/ask exactly.
"""
from __future__ import annotations

import cli.main as cli_main


def test_evaluate_accepts_search_method_hybrid(monkeypatch):
    captured = {}

    def _fake_run_evaluate(args):
        captured["args"] = args

    monkeypatch.setattr(cli_main, "_run_evaluate", _fake_run_evaluate)

    cli_main.main(["evaluate", "3gpp", "--search-method", "hybrid"])

    assert captured["args"].search_method == "hybrid"


def test_evaluate_search_method_defaults_to_dense(monkeypatch):
    captured = {}

    def _fake_run_evaluate(args):
        captured["args"] = args

    monkeypatch.setattr(cli_main, "_run_evaluate", _fake_run_evaluate)

    cli_main.main(["evaluate", "3gpp"])

    assert captured["args"].search_method == "dense"


def test_evaluate_rejects_an_invalid_search_method(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_run_evaluate", lambda args: None)

    try:
        cli_main.main(["evaluate", "3gpp", "--search-method", "nonsense"])
        raised = False
    except SystemExit:
        raised = True

    assert raised, "argparse should reject an unlisted --search-method choice"
