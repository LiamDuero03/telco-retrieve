"""Regression: cli/main.py's config helpers (_load_chunking_strategies,
_load_enrichment_strategies, _strategy_requires_llm, _enrichment_requires_llm)
resolved configs/<file>.json relative to CWD only, so running from outside the
repo root silently fell back to degraded defaults (["text_baseline"], ["none"]),
making argparse reject valid --strategy/--enrichment choices. Fixed via
_resolve_config_path(): CWD-relative checked first, falls back to package-relative
(same pattern SentenceTransformerEmbedder/OpenRouterEmbedder already use).
"""
from __future__ import annotations

import cli.main as cli_main


def test_load_chunking_strategies_works_from_a_different_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a directory with no local configs/ at all

    strategies = cli_main._load_chunking_strategies()

    assert strategies != ["text_baseline"], (
        "fell back to the degraded default -- config resolution is still CWD-bound"
    )
    assert "sliding_window_tokens" in strategies
    assert "hierarchical_markdown" in strategies
    assert "parent_child" in strategies
    assert "lumber_chunker" in strategies


def test_load_enrichment_strategies_works_from_a_different_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    strategies = cli_main._load_enrichment_strategies()

    assert strategies != ["none"], (
        "fell back to the degraded default -- config resolution is still CWD-bound"
    )
    assert "metadata_tagging" in strategies
    assert "acronym_expansion" in strategies
    assert "llm_metadata" in strategies


def test_strategy_requires_llm_works_from_a_different_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Previously always False outside repo root, skipping required --llm-backend prompt.
    assert cli_main._strategy_requires_llm("lumber_chunker") is True
    assert cli_main._strategy_requires_llm("text_baseline") is False


def test_enrichment_requires_llm_works_from_a_different_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert cli_main._enrichment_requires_llm("llm_metadata") is True
    assert cli_main._enrichment_requires_llm("acronym_expansion") is False


def test_cwd_relative_configs_dir_still_takes_priority_when_present(tmp_path, monkeypatch):
    """Confirms CWD-relative configs/ still wins over the package-relative fallback."""
    local_configs = tmp_path / "configs"
    local_configs.mkdir()
    (local_configs / "chunking_strategies.json").write_text('{"only_this_one": {}}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert cli_main._load_chunking_strategies() == ["only_this_one"]
