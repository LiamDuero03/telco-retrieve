"""Tests for the promoted helper scripts in scripts/."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
ALL = ["embedder_smoke_test", "estimate_embedding_cost", "time_embedding_throughput",
       "time_openrouter_embedding"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ALL)
def test_help_runs_without_network_or_a_database(name):
    result = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py"), "--help"],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("name", ALL)
def test_scripts_have_no_hardcoded_secrets_paths_or_path_hacks(name):
    text = (SCRIPTS / f"{name}.py").read_text(encoding="utf8")
    assert "liam/" not in text and "dbs/3gpp" not in text
    assert "sys.path" not in text
    assert "sk-or-" not in text and "api_key=\"" not in text


# ── embedder_smoke_test ───────────────────────────────────────────────────────

def test_local_model_keys_skip_openrouter_served_entries():
    smoke = _load("embedder_smoke_test")
    keys = smoke.local_model_keys()
    for expected in ("minilm", "e5", "e5-prefixed", "qwen3-embedding-0.6b", "embeddinggemma-300m",
                     "bge-m3"):
        assert expected in keys
    for skipped in ("or-3-small", "or-bge-m3", "qwen3-embedding-8b"):
        assert skipped not in keys


def test_unit_norm_check():
    smoke = _load("embedder_smoke_test")
    assert smoke.unit_norm_check([[1.0, 0.0], [0.0, 1.0]]) == (True, 0.0)
    is_unit, deviation = smoke.unit_norm_check([[3.0, 4.0]])
    assert is_unit is False and deviation > 3.0
    assert smoke.unit_norm_check([]) == (True, 0.0)


def test_run_one_end_to_end_with_real_minilm_tiny_inputs():
    smoke = _load("embedder_smoke_test")
    report = smoke.run_one("minilm", n_chunks=3, n_queries=2, extrapolate_n=100)
    assert report["error"] is None
    assert report["dimension"] == report["expected_dimension"] == 384
    assert report["throughput_chunks_per_sec"] > 0 and report["n_query_vectors"] == 2
    assert report["max_seq_length"] == 256
    assert report["truncation_rate"] == 0.0


# ── estimate_embedding_cost ───────────────────────────────────────────────────

class _FakeCollection:
    name = "c1"

    def count(self):
        return 2

    def get(self, include=None, limit=None, offset=0):
        return {"documents": ["alpha beta gamma", "delta epsilon"][offset:offset + limit]}


def _fake_chromadb():
    module = types.ModuleType("chromadb")

    class Client:
        def __init__(self, path=None):
            pass

        def list_collections(self):
            return [_FakeCollection()]

        def get_collection(self, name):
            return _FakeCollection()

    module.PersistentClient = Client
    return module


def test_estimator_labels_models_without_a_price_and_prices_the_rest(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "chromadb", _fake_chromadb())
    monkeypatch.setattr(sys, "argv", ["estimate_embedding_cost.py", "--db-path", "x"])
    _load("estimate_embedding_cost").main()
    out = capsys.readouterr().out
    line = {ln.split()[0]: ln for ln in out.splitlines() if ln.startswith("  ") and " " in ln.strip()}
    assert "no price" in line["qwen3-embedding-8b"]      # not in the snapshot
    assert "no price" in line["otel-109m"]               # local-only
    assert "$" in line["or-3-small"] and "no price" not in line["or-3-small"]
    assert "local only" not in out


def test_estimator_exits_cleanly_on_an_empty_db(monkeypatch, capsys):
    module = _fake_chromadb()
    module.PersistentClient.list_collections = lambda self: []
    monkeypatch.setitem(sys.modules, "chromadb", module)
    monkeypatch.setattr(sys, "argv", ["estimate_embedding_cost.py", "--db-path", "x"])
    _load("estimate_embedding_cost").main()
    assert "No collections found" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["time_embedding_throughput", "time_openrouter_embedding"])
def test_timing_scripts_reject_an_empty_collection(name, monkeypatch):
    module = _fake_chromadb()
    module.PersistentClient.get_collection = lambda self, n: type(
        "Empty", (), {"count": lambda s: 0})()
    monkeypatch.setitem(sys.modules, "chromadb", module)
    monkeypatch.setattr(sys, "argv", [f"{name}.py", "--db-path", "x", "--collection", "c"])
    with pytest.raises(SystemExit, match="empty"):
        _load(name).main()
