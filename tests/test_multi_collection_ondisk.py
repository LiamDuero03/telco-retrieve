"""Multi-collection search on a real on-disk Chroma database, which the other tests (in-memory
Chroma) do not cover. No fixture from fake_collections is imported here on purpose."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from open_telco_rag.core import multi_search as ms
from tests.fake_collections import FakeEmbedder, make_collection

ROOT = Path(__file__).resolve().parents[1]


def _cli(cwd, *args):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-m", "tests.cli_fake_embedder", *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def test_real_cli_ingests_two_sources_and_searches_one_and_two_collections(tmp_path):
    """ingest 3gpp and camara into two on-disk collections, then search with the real CLI, each command in
    its own process as a user runs them"""
    gpp = tmp_path / "data" / "tgpp" / "Rel-18" / "28_series"
    gpp.mkdir(parents=True)
    (gpp / "28312-i11.md").write_text(
        "# Scope\n\n" + "The amf handles registration and mobility management for the ue. " * 12, encoding="utf8")
    cam = tmp_path / "data" / "camara" / "QualityOnDemand"
    cam.mkdir(parents=True)
    (cam / "README.md").write_text(
        "# QualityOnDemand\n\n" + "The device api offers registration of a boost session for a device. " * 12,
        encoding="utf8")
    common = ["--model", "fake64", "--db-path", "db", "--chunk-size", "300", "--overlap", "30"]
    for source, folder in (("3gpp", "tgpp"), ("camara", "camara")):
        done = _cli(tmp_path, "ingest", source, "--data", f"data/{folder}", *common)
        assert done.returncode == 0, done.stderr[-500:]

    one = _cli(tmp_path, "search", "registration", "--model", "fake64", "--db-path", "db", "--collection",
               "3gpp_fake64", "--top-k", "4")
    assert one.returncode == 0 and "28312-i11.md" in one.stdout and "README.md" not in one.stdout, one.stderr[-500:]

    two = _cli(tmp_path, "search", "registration", "--model", "fake64", "--db-path", "db", "--collection",
               "3gpp_fake64", "--collection", "camara_fake64", "--top-k", "4")
    assert two.returncode == 0, two.stderr[-500:]
    assert "| 3gpp_fake64 | 28312-i11.md" in two.stdout and "| camara_fake64 | README.md" in two.stdout

    comma = _cli(tmp_path, "search", "registration", "--model", "fake64", "--db-path", "db", "--collection",
                 "3gpp_fake64,camara_fake64")
    assert comma.returncode != 0 and "repeat --collection" in comma.stderr


@pytest.mark.skipif(not os.environ.get("RUN_ONDISK_STRESS"),
                    reason="about a minute: set RUN_ONDISK_STRESS=1 to repeat write-then-search on disk")
def test_repeated_search_right_after_an_in_process_write_never_hits_nothing_found_on_disk(tmp_path):
    """Chroma 1.5.9 can fail a query with 'Nothing found on disk' when a collection is read moments after
    it was written by the same process. About 1 iteration in 60 hit it before open_collections stopped
    calling count() on a fresh handle."""
    rows = [{"id": f"d{i}::fake64::0", "text": f"registration mobility number {i}", "source": f"d{i}.md"}
            for i in range(4)]
    embedder = FakeEmbedder()
    failures = []
    for i in range(150):
        db = str(tmp_path / f"db{i}")
        make_collection(db, "col_a", rows, embedder)
        make_collection(db, "col_b", rows[:3], embedder)
        try:
            opened = ms.open_collections(db, ["col_a", "col_b"], embedder, say=lambda *_: None)
            ms.multi_search("registration mobility", [c.store for c in opened], embedder, top_k=6)
        except Exception as e:  # noqa: BLE001
            failures.append(f"iteration {i}: {str(e)[:100]}")
    assert not failures, f"{len(failures)}/150 failed, first: {failures[0]}"
