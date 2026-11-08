import argparse

import pytest

import cli.main as cli_main
from open_telco_rag.challenge import cli as challenge_cli
from tests.fake_collections import FakeEmbedder
from tests.test_challenge_export import BIG_ID, FakeClient, build_index

FLAGS = (
    "profile",
    "model",
    "reranker",
    "rerank",
    "rerank_top_k",
    "search_method",
    "top_k",
    "temperature",
    "max_tokens",
    "seed",
    "max_prompt_tokens",
)


def namespace(**given):
    return argparse.Namespace(**{key: given.get(key) for key in FLAGS})


def test_precedence_flag_then_profile_then_fallback_and_existing_defaults_kept():
    plain = challenge_cli.resolve_settings(namespace())
    assert (plain["model"], plain["reranker"], plain["rerank"], plain["search_method"]) == (
        "otel-0.6b",
        "otel-reranker-0.6b",
        True,
        "hybrid",
    )
    chosen = challenge_cli.resolve_settings(namespace(profile="challenge", top_k=3, rerank=False))
    assert (
        chosen["model"],
        chosen["search_method"],
        chosen["top_k"],
        chosen["rerank"],
        chosen["seed"],
    ) == ("otel-0.6b", "hybrid", 3, False, 1234)


def run_with_profile(tmp_path, monkeypatch, *extra):
    seen = {}
    client = FakeClient(["B"])
    questions = tmp_path / "q.csv"
    questions.write_text(
        f'ID,question,options\n{BIG_ID},Which NF?,"{{""A"": ""AMF"", ""B"": ""SMF""}}"\n',
        encoding="utf-8",
    )

    class Reranker:
        def rerank(self, query, results, top_k=5):
            seen["rerank_top_k"] = top_k
            return results[:top_k]

    monkeypatch.setattr(
        challenge_cli,
        "_build_embedder",
        lambda settings: seen.update(settings=settings) or FakeEmbedder(),
    )
    monkeypatch.setattr(challenge_cli, "_build_client", lambda *a, **k: client)
    monkeypatch.setattr(
        "open_telco_rag.embedders.reranker.load_reranker",
        lambda key: seen.update(reranker=key) or Reranker(),
    )
    build_index(tmp_path / "db")
    cli_main.main(
        [
            "challenge",
            "export",
            "--questions",
            str(questions),
            "--out",
            str(tmp_path / "o.csv"),
            "--db-path",
            str(tmp_path / "db"),
            "--collection",
            "col",
            "--allow-embedder-mismatch",
            *extra,
        ]
    )
    return seen, client


def test_cli_profile_applies_hybrid_rerank_and_seed(tmp_path, monkeypatch):
    seen, client = run_with_profile(tmp_path, monkeypatch, "--profile", "challenge")
    assert seen["settings"]["model"] == "otel-0.6b" and seen["reranker"] == "otel-reranker-0.6b"
    assert seen["rerank_top_k"] == 5
    assert client.seen[0][1] == {"temperature": 0.0, "max_tokens": 8, "seed": 1234}


def test_cli_flags_override_profile(tmp_path, monkeypatch):
    seen, client = run_with_profile(
        tmp_path,
        monkeypatch,
        "--profile",
        "challenge",
        "--no-rerank",
        "--search-method",
        "dense",
        "--seed",
        "5",
    )
    assert "reranker" not in seen and client.seen[0][1]["seed"] == 5


def test_prefix_flags_without_custom_embedder_are_rejected():
    with pytest.raises(SystemExit, match="need --embedder-path or --embedder-id"):
        challenge_cli._build_embedder(
            {
                "embedder_path": None,
                "embedder_id": None,
                "query_prefix": "q",
                "document_prefix": None,
                "model": "minilm",
            }
        )
