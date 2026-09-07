"""Tests for cli/main.py's _update_leaderboard() corrupted-file handling.

Fix: a leaderboard.json that failed to parse used to be silently reset and
overwritten, destroying prior entries. Now backed up to a .corrupted-<timestamp>.json
sibling with a loud warning instead.
"""
from __future__ import annotations

import json

from cli.main import _update_leaderboard
from open_telco_rag.evaluation.results import CorpusResult


def _minimal_result(**overrides) -> CorpusResult:
    """Minimal CorpusResult — every field _update_leaderboard() reads has a usable default."""
    return CorpusResult(corpus="3gpp", model="minilm", top_k=5, n_questions=0, **overrides)


def test_appends_a_row_to_an_existing_valid_leaderboard(tmp_path, capsys):
    leaderboard_path = tmp_path / "leaderboard.json"
    leaderboard_path.write_text(json.dumps([{"run_name": "old_run"}]), encoding="utf-8")

    _update_leaderboard(
        leaderboard_path, _minimal_result(), rag_model="claude", evaluator_model="claude",
        corpus="3gpp", collection="3gpp_minilm",
    )

    rows = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["run_name"] == "old_run"
    assert rows[1]["corpus"] == "3gpp"
    # No corrupted-file warning for a valid file.
    assert "Could not parse" not in capsys.readouterr().out


def test_creates_a_fresh_leaderboard_when_none_exists(tmp_path):
    leaderboard_path = tmp_path / "leaderboard.json"

    _update_leaderboard(
        leaderboard_path, _minimal_result(), rag_model="claude", evaluator_model="claude",
        corpus="3gpp", collection="3gpp_minilm",
    )

    rows = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    assert len(rows) == 1


def test_corrupted_leaderboard_is_backed_up_not_silently_discarded(tmp_path, capsys):
    leaderboard_path = tmp_path / "leaderboard.json"
    leaderboard_path.write_text('{"not": "a list", truncated garbage', encoding="utf-8")

    _update_leaderboard(
        leaderboard_path, _minimal_result(), rag_model="claude", evaluator_model="claude",
        corpus="3gpp", collection="3gpp_minilm",
    )

    # Starts fresh with just the new row - old content unreadable, not merged, but not lost.
    rows = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["corpus"] == "3gpp"

    # Unreadable original preserved under a .corrupted-*.json sibling, not deleted.
    backups = list(tmp_path.glob("leaderboard.corrupted-*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == '{"not": "a list", truncated garbage'

    out = capsys.readouterr().out
    assert "Could not parse" in out
    assert "Backed up" in out


def test_all_not_applicable_metric_fields_are_clipped_consistently(tmp_path):
    """Regression: faithfulness/answer_relevancy weren't clipped the same
    way accuracy/coverage_ratio/context_recall/answer_correctness are, so
    -1.0 (CorpusResult's "not applicable" sentinel, e.g. for an mc/tf-only
    run where these open-question-only metrics never ran) was written
    straight into leaderboard.json for those two fields specifically."""
    leaderboard_path = tmp_path / "leaderboard.json"

    # _minimal_result() sets only corpus/model/top_k/n_questions -- every
    # mean_* metric field stays at its -1.0 default, i.e. exactly the
    # "nothing applicable computed" case this test targets.
    _update_leaderboard(
        leaderboard_path, _minimal_result(), rag_model="claude", evaluator_model="claude",
        corpus="3gpp", collection="3gpp_minilm",
    )

    row = json.loads(leaderboard_path.read_text(encoding="utf-8"))[0]
    for field in ("accuracy", "faithfulness", "answer_relevancy",
                  "coverage_ratio", "context_recall", "answer_correctness"):
        assert row[field] == 0.0, f"{field} should be clipped to 0.0 like its siblings, not left as -1.0"
