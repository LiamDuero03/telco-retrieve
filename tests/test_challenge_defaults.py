import argparse

import pytest

import cli.main as cli_main
from open_telco_rag.challenge import cli as challenge_cli
from open_telco_rag.challenge.settings import describe, resolve_settings
from tests.fake_collections import FakeEmbedder
from tests.test_challenge_export import BIG_ID, FakeClient, build_index

PROFILE_DEFAULTS = {
    "profile": "challenge",
    "model": "otel-0.6b",
    "reranker": "otel-reranker-0.6b",
    "rerank": True,
    "rerank_top_k": 5,
    "search_method": "hybrid",
    "top_k": 10,
    "generator_model": "google/gemma-3n-E4B-it",
    "temperature": 0.0,
    "max_tokens": 8,
    "seed": 1234,
    "resolve_parents": False,
    "max_prompt_tokens": 4096,
    "timeout": 120.0,
}
QUESTION = f'ID,question,options\n{BIG_ID},Which NF?,"{{""A"": ""AMF"", ""B"": ""SMF""}}"\n'


def namespace(**given):
    return argparse.Namespace(**given)


def cli_export(tmp_path, monkeypatch, *argv, index=None, base=True, allow=True, questions=QUESTION):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "q.csv").write_text(questions, encoding="utf-8")
    client = FakeClient(["B"] * 100)
    monkeypatch.setattr(challenge_cli, "_build_embedder", lambda settings: FakeEmbedder())
    monkeypatch.setattr(challenge_cli, "_build_client", lambda *a, **k: client)
    monkeypatch.setattr(
        "open_telco_rag.embedders.reranker.load_reranker",
        lambda key: type("R", (), {"rerank": lambda self, q, r, top_k=5: r[:top_k]})(),
    )
    if not (tmp_path / "db").exists():
        (index or build_index)(tmp_path / "db")
    flags = ["--allow-embedder-mismatch"] if allow else []
    cli_main.main(
        [
            "challenge",
            "export",
            "--questions",
            str(tmp_path / "q.csv"),
            "--out",
            str(tmp_path / "o.csv"),
        ]
        + ["--db-path", str(tmp_path / "db"), "--collection", "col"]
        + (["--top-k", "5"] if base else [])
        + flags
        + list(argv)
    )
    return client


@pytest.mark.parametrize(
    "argv,key,value",
    [
        (["--top-k", "3"], "top_k", 3),
        (["--search-method", "dense"], "search_method", "dense"),
        (["--no-rerank"], "rerank", False),
        (["--rerank-top-k", "2"], "rerank_top_k", 2),
        (["--reranker-model", "r"], "reranker", "r"),
        (["--model", "minilm"], "model", "minilm"),
        (["--seed", "5"], "seed", 5),
        (["--temperature", "0.5"], "temperature", 0.5),
        (["--max-answer-tokens", "4"], "max_tokens", 4),
        (["--max-tokens", "4"], "max_tokens", 4),
        (["--max-prompt-tokens", "1000"], "max_prompt_tokens", 1000),
        (["--generator-model", "mine"], "generator_model", "mine"),
        (["--timeout", "9"], "timeout", 9.0),
        (["--resolve-parents"], "resolve_parents", True),
    ],
)
def test_every_default_is_overridable_by_its_flag(argv, key, value):
    parser = argparse.ArgumentParser()
    challenge_cli.add_challenge_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "challenge",
            "export",
            "--questions",
            "q",
            "--out",
            "o",
            "--db-path",
            "d",
            "--collection",
            "c",
            *argv,
        ]
    )
    assert resolve_settings(args)[key] == value


def test_effective_settings_are_one_stderr_line_and_stdout_stays_parseable(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    cli_export(tmp_path, monkeypatch)
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("[settings]")]
    assert len(lines) == 1 and lines[0] == describe(resolve_settings(namespace(top_k=5)))
    for needle in (
        "embedder=otel-0.6b",
        "reranker=otel-reranker-0.6b",
        "search=hybrid",
        "seed=1234",
        "generator_model=google/gemma-3n-E4B-it",
        "max_prompt_tokens=4096",
    ):
        assert needle in lines[0]
    assert len(captured.out.strip().splitlines()) == 1 and captured.out.startswith("Wrote 1 rows")


def test_max_answer_tokens_is_the_documented_name_and_max_tokens_stays_hidden(capsys):
    with pytest.raises(SystemExit):
        cli_main.main(["challenge", "export", "--help"])
    help_text = capsys.readouterr().out
    assert "--max-answer-tokens" in help_text and "--max-tokens" not in help_text


@pytest.mark.parametrize("flag", ["--max-answer-tokens", "--max-tokens"])
def test_both_names_reach_the_request(tmp_path, monkeypatch, flag):
    client = cli_export(tmp_path, monkeypatch, flag, "4")
    assert client.seen[0][1]["max_tokens"] == 4
