"""Regression: --use-marker previously validated DATALAB_API_KEY lazily, inside
each Phase-1 worker process (core/marker_loader.py's _marker_load), so a missing
key surfaced as N independent per-file errors instead of one clear message before
ingestion started -- the same failure mode the pre-existing LLM connectivity
check (cli/main.py's "Fail fast once here") was written to avoid, just not
applied to the Marker path. _run_ingest now checks DATALAB_API_KEY upfront when
use_marker is set, mirroring that pattern.
"""
from __future__ import annotations

import argparse

import pytest

import cli.main as cli_main


def _args(**overrides) -> argparse.Namespace:
    base = dict(source_type="oran", use_marker=True, model="minilm")
    base.update(overrides)
    return argparse.Namespace(**base)


def test_use_marker_without_datalab_key_exits_before_any_ingestion_work(monkeypatch, capsys):
    # core.pipeline calls load_dotenv() on import (triggered by _run_ingest's
    # own `from open_telco_rag.core.pipeline import ingest`), which repopulates
    # an absent var from a real .env -- setenv("") survives that (load_dotenv
    # doesn't override already-present keys) while still reading as falsy.
    monkeypatch.setenv("DATALAB_API_KEY", "")

    with pytest.raises(SystemExit) as exc_info:
        cli_main._run_ingest(_args())

    assert exc_info.value.code == 1
    err = capsys.readouterr().out
    assert "DATALAB_API_KEY" in err


def test_use_marker_with_datalab_key_set_does_not_hit_the_preflight_exit(monkeypatch):
    monkeypatch.setenv("DATALAB_API_KEY", "test-key")
    # Stop execution right after the check we're testing, before real ingestion
    # work (embedder/store construction, file I/O) would run.
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached embedder construction")))

    with pytest.raises(RuntimeError, match="reached embedder construction"):
        cli_main._run_ingest(_args())


def test_marker_not_requested_skips_the_datalab_key_check_entirely(monkeypatch):
    monkeypatch.setenv("DATALAB_API_KEY", "")
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached embedder construction")))

    with pytest.raises(RuntimeError, match="reached embedder construction"):
        cli_main._run_ingest(_args(use_marker=False))
