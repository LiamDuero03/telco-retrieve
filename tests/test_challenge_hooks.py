import csv

import pytest

import cli.main as cli_main
from open_telco_rag.challenge import cli as challenge_cli
from open_telco_rag.challenge.prompt import Prompt
from open_telco_rag.challenge.tokens import count_tokens
from tests.fake_collections import FakeEmbedder
from tests.test_challenge_export import BIG_ID, FakeClient, build_index

HOOKS = "tests.test_challenge_hooks"


def custom_builder(question, passages, max_prompt_tokens):
    text = f"CUSTOM|{question.question}|{len(passages)}"
    messages = [{"role": "user", "content": text}]
    return Prompt(messages, text, count_tokens(text), len(passages))


def oversized_builder(question, passages, max_prompt_tokens):
    text = "word " * 5000
    return Prompt([{"role": "user", "content": text}], text, 5000, 0)


def last_letter_parser(text, options):
    return text.strip()[-1:].upper()


def run_export(tmp_path, monkeypatch, *extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    questions = tmp_path / "q.csv"
    questions.write_text(
        f'ID,question,options\n{BIG_ID},Which NF?,"{{""A"": ""AMF"", ""B"": ""SMF""}}"\n',
        encoding="utf-8",
    )
    client = FakeClient(["The answer is B"])
    monkeypatch.setattr(challenge_cli, "_build_embedder", lambda settings: FakeEmbedder())
    monkeypatch.setattr(challenge_cli, "_build_client", lambda *a, **k: client)
    if not (tmp_path / "db").exists():
        build_index(tmp_path / "db")
    out = tmp_path / "o.csv"
    cli_main.main(
        ["challenge", "export", "--questions", str(questions), "--out", str(out)]
        + [
            "--db-path",
            str(tmp_path / "db"),
            "--collection",
            "col",
            "--no-rerank",
            "--allow-embedder-mismatch",
            "--top-k",
            "5",
            *extra,
        ]
    )
    with open(out, encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))[1], client


def test_custom_prompt_builder_string_is_exported_and_sent(tmp_path, monkeypatch):
    row, client = run_export(tmp_path, monkeypatch, "--prompt-builder", f"{HOOKS}:custom_builder")
    assert row[1] == "CUSTOM|Which NF?|5" and row[2] == "B"
    assert client.seen[0][0] == [{"role": "user", "content": row[1]}]


def test_custom_answer_parser_is_used(tmp_path, monkeypatch):
    row, _ = run_export(tmp_path, monkeypatch, "--answer-parser", f"{HOOKS}:last_letter_parser")
    assert row[2] == "B" and row[1].startswith("You answer 3GPP questions")


def test_default_hooks_match_explicit_defaults(tmp_path, monkeypatch):
    plain, _ = run_export(tmp_path, monkeypatch)
    explicit, _ = run_export(
        tmp_path,
        monkeypatch,
        "--prompt-builder",
        "open_telco_rag.challenge.prompt:build_prompt",
        "--answer-parser",
        "open_telco_rag.challenge.normalise:normalise_answer",
    )
    assert plain == explicit


def test_hook_errors_are_clear_and_the_cap_still_applies(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="package.module:function"):
        run_export(tmp_path / "a", monkeypatch, "--prompt-builder", "nocolon")
    with pytest.raises(SystemExit, match="cannot import 'no_such_module'"):
        run_export(tmp_path / "b", monkeypatch, "--answer-parser", "no_such_module:f")
    with pytest.raises(SystemExit, match=f"question {BIG_ID}: prompt has \\d+ tokens, cap is 4096"):
        run_export(tmp_path / "c", monkeypatch, "--prompt-builder", f"{HOOKS}:oversized_builder")


def test_resolve_parents_reaches_retrieval(tmp_path, monkeypatch):
    seen = []
    real = __import__("open_telco_rag.core.retrieval", fromlist=["x"]).execute_retrieval
    monkeypatch.setattr(
        "open_telco_rag.core.retrieval.execute_retrieval",
        lambda *a, **kw: seen.append(kw["resolve_parents"]) or real(*a, **kw),
    )
    build_index(tmp_path / "a" / "db", parents=True)
    build_index(tmp_path / "b" / "db", parents=True)
    run_export(tmp_path / "a", monkeypatch)
    run_export(tmp_path / "b", monkeypatch, "--resolve-parents")
    assert seen == [False, True]
