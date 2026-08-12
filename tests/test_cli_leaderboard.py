"""Tests for cli/main.py's _update_leaderboard() -- specifically the
corrupted-leaderboard-file handling added during cleanup step 4 (Production
Auditor). Real risk that fix closes: a leaderboard.json that fails to parse
(e.g. truncated by an interrupted write) used to be silently reset to []
and then unconditionally overwritten with just the one new row, destroying
every prior leaderboard entry with no warning. It now backs the unreadable
file up to a .corrupted-<timestamp>.json sibling and prints a loud warning
instead.
"""
from __future__ import annotations

import json

from cli.main import _update_leaderboard
from open_telco_rag.evaluation.results import CorpusResult


def _minimal_result(**overrides) -> CorpusResult:
    """A CorpusResult with just enough set for _update_leaderboard() to run
    end to end -- every field it reads has a usable default already, so
    only corpus is required."""
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

    # The new leaderboard starts fresh with just the new row (old content
    # was unreadable, not silently mergeable) -- but is not destroyed.
    rows = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["corpus"] == "3gpp"

    # The unreadable original is preserved under a .corrupted-*.json sibling,
    # not deleted.
    backups = list(tmp_path.glob("leaderboard.corrupted-*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == '{"not": "a list", truncated garbage'

    out = capsys.readouterr().out
    assert "Could not parse" in out
    assert "Backed up" in out
