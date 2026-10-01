"""CLI: --bound-tokens / --bound-tokenizer / --bound-margin on ingest are opt-in and validated."""
from __future__ import annotations

import pytest

from tests.test_chunking_opt_in import (  # noqa: F401  (cli_ingest is a fixture)
    _cli_args,
    cli_ingest,
)


def test_without_the_flags_nothing_bound_related_reaches_ingest(cli_ingest):  # noqa: F811
    cli_main, captured = cli_ingest
    cli_main._run_ingest(_cli_args())
    assert not [k for k in captured if k.startswith("bound")]
    assert captured["chunk_size"] == 1024 and captured["overlap"] == 200


def test_the_flags_reach_ingest_and_work_with_character_or_token_sizing(cli_ingest):  # noqa: F811
    cli_main, captured = cli_ingest
    cli_main._run_ingest(_cli_args(bound_tokens=512, bound_tokenizer="bge-m3", bound_margin=12))
    assert (captured["bound_tokens"], captured["bound_tokenizer"], captured["bound_margin"]) == (512, "bge-m3", 12)
    assert captured["chunk_size"] == 1024
    captured.clear()
    cli_main._run_ingest(_cli_args(target_tokens=256, max_tokens=300, bound_tokens=512))
    assert captured["bound_tokens"] == 512 and captured["target_tokens"] == 256
    assert "bound_tokenizer" not in captured and "bound_margin" not in captured


def test_the_parser_accepts_the_three_flags(cli_ingest, capsys):  # noqa: F811
    cli_main, captured = cli_ingest
    cli_main.main(["ingest", "oran", "--bound-tokens", "512", "--bound-tokenizer", "minilm", "--bound-margin", "2"])
    assert captured["bound_tokens"] == 512 and captured["bound_tokenizer"] == "minilm"
    assert "Token bound: 510 tokens per chunk" in capsys.readouterr().out


@pytest.mark.parametrize("kwargs, message", [
    ({"bound_tokenizer": "bge-m3"}, "require --bound-tokens"),
    ({"bound_margin": 2}, "require --bound-tokens"),
    ({"bound_tokens": 0}, "bound_tokens must be a positive integer"),
    ({"bound_tokens": 10, "bound_margin": 10}, "bound_margin"),
])
def test_invalid_flag_combinations_exit_with_an_error(cli_ingest, kwargs, message):  # noqa: F811
    cli_main, _ = cli_ingest
    with pytest.raises(SystemExit, match=message):
        cli_main._run_ingest(_cli_args(**kwargs))
