"""Guard: everything that writes results goes through the serializers.

Raw dataclasses.asdict() would leak unset opt-in fields (None) into saved
output, so only evaluation/results.py may use it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cli.main as cli_main
from open_telco_rag.evaluation.results import CorpusResult, QuestionResult, save_results

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {ROOT / "open_telco_rag" / "evaluation" / "results.py"}
RAW = re.compile(r"\basdict\s*\(|\.__dict__\b|\bvars\s*\(")
OPT_IN_KEYS = {"context_tokens", "n_chunks_used", "budget", "fill_ratio",
               "mean_context_tokens", "mean_fill_ratio", "context_token_budget"}


def _sources():
    for base in ("open_telco_rag", "cli"):
        yield from (ROOT / base).rglob("*.py")


def test_only_results_module_uses_raw_asdict():
    offenders = []
    for path in _sources():
        if path in ALLOWED:
            continue
        for n, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
            if RAW.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, "use question_to_dict/result_to_dict instead:\n" + "\n".join(offenders)


def _default_result():
    q = QuestionResult(question="q", corpus="3gpp", answer="a", sources=[], chunks_used=0)
    return CorpusResult(corpus="3gpp", model="m", top_k=5, n_questions=1, question_results=[q])


def test_default_saved_results_carry_no_opt_in_keys(tmp_path):
    data = json.loads(Path(save_results(_default_result(), tmp_path)).read_text(encoding="utf8"))
    assert OPT_IN_KEYS.isdisjoint(data)
    assert all(OPT_IN_KEYS.isdisjoint(q) for q in data["question_results"])


def test_default_leaderboard_row_carries_no_budget_keys(tmp_path):
    board = tmp_path / "leaderboard.json"
    cli_main._update_leaderboard(board, _default_result(), rag_model="m", evaluator_model="e",
                                 corpus="3gpp", collection="c")
    row = json.loads(board.read_text(encoding="utf8"))[0]
    assert OPT_IN_KEYS.isdisjoint(row)


def test_budgeted_leaderboard_row_records_the_budget(tmp_path):
    board = tmp_path / "leaderboard.json"
    cli_main._update_leaderboard(board, _default_result(), rag_model="m", evaluator_model="e",
                                 corpus="3gpp", collection="c", context_token_budget=500,
                                 candidate_pool=30, budget_overflow="stop")
    row = json.loads(board.read_text(encoding="utf8"))[0]
    assert (row["context_token_budget"], row["candidate_pool"], row["budget_overflow"]) == (500, 30, "stop")
