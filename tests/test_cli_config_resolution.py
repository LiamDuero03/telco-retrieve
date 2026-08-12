"""Regression tests for a real, confirmed-by-reproduction bug: cli/main.py's
config-driven helpers (_load_chunking_strategies, _load_enrichment_strategies,
_strategy_requires_llm, _enrichment_requires_llm) used to resolve
"configs/<file>.json" relative to the process's CURRENT WORKING DIRECTORY
only. Running the CLI from anywhere other than the exact repo root (the
normal case for an installed console-script, or simply invoking the CLI
from a different shell CWD) made every one of these silently fall back to
a degraded default:

  _load_chunking_strategies()   -> ["text_baseline"] only
  _load_enrichment_strategies() -> ["none"] only

Since these two feed argparse's `choices=` for `--strategy`/`--enrichment`,
this made argparse reject `--strategy sliding_window_tokens` (or
hierarchical_markdown/parent_child/lumber_chunker) and `--enrichment
acronym_expansion` (or metadata_tagging/llm_metadata) as "invalid choice"
even though those strategies are fully implemented -- purely a function of
which directory the command happened to be run from.

Fixed via _resolve_config_path(): CWD-relative is still checked FIRST (so
existing behavior for anyone running from the repo root is unchanged), but
falls back to a package-relative location (cli/main.py's own directory ->
repo root -> configs/) that works regardless of CWD -- the same pattern
SentenceTransformerEmbedder/OpenRouterEmbedder already use correctly for
embedding_models.json.
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

    # Previously always False from a non-repo-root CWD, regardless of the
    # strategy -- would silently skip prompting for --llm-backend setup for
    # a strategy that genuinely needs it.
    assert cli_main._strategy_requires_llm("lumber_chunker") is True
    assert cli_main._strategy_requires_llm("text_baseline") is False


def test_enrichment_requires_llm_works_from_a_different_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert cli_main._enrichment_requires_llm("llm_metadata") is True
    assert cli_main._enrichment_requires_llm("acronym_expansion") is False


def test_cwd_relative_configs_dir_still_takes_priority_when_present(tmp_path, monkeypatch):
    """A deliberate local configs/ override (CWD-relative) must still win
    over the package-relative fallback -- confirms the fix didn't flip the
    two locations' precedence, only added the fallback."""
    local_configs = tmp_path / "configs"
    local_configs.mkdir()
    (local_configs / "chunking_strategies.json").write_text('{"only_this_one": {}}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert cli_main._load_chunking_strategies() == ["only_this_one"]
