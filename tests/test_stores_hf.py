"""Tests for pull_collection_from_hf() — confirmed-in-production bug:
snapshot_download() replicates a matched file's FULL repo-relative path
under local_dir, but the old code computed local_dir as
(Path(local_dir) / path_in_repo).parent, which only un-nests a
SINGLE-segment path_in_repo. Every grid script actually passes a
two-segment path ("final-ingestion-grid/{run_name}"), so files landed one
directory deeper than the function claimed — ChromaStore silently opened
an empty dir and created a fresh empty collection (store.count() == 0,
no exception) even though the download succeeded.

snapshot_download is mocked to replicate its REAL placement behavior
(not a no-op stub), otherwise this test wouldn't catch the bug.
"""
from __future__ import annotations

from pathlib import Path

from open_telco_rag.stores.hf import pull_collection_from_hf


def _fake_snapshot_download_matching_real_behavior(*, repo_id, repo_type, allow_patterns, local_dir):
    """Mimics real snapshot_download placement: writes the dummy file at
    {local_dir}/{path_in_repo}/... after stripping "/*" from allow_patterns."""
    repo_relative_dir = allow_patterns.rsplit("/*", 1)[0]
    target = Path(local_dir) / repo_relative_dir / "chroma.sqlite3"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("fake chroma db", encoding="utf-8")


def test_pull_with_a_two_segment_path_in_repo_finds_the_real_downloaded_files(tmp_path, monkeypatch):
    """Regression case: master_grid_eval.py/pilot_eval.py call this with a
    two-segment path_in_repo="final-ingestion-grid/{run_name}"."""
    import open_telco_rag.stores.hf as hf_module
    monkeypatch.setattr(hf_module, "snapshot_download", _fake_snapshot_download_matching_real_behavior)

    returned_path = pull_collection_from_hf(
        repo_id="fake/repo",
        path_in_repo="final-ingestion-grid/3gpp-r19_minilm_text_baseline_none_c1024_o200",
        local_dir=str(tmp_path),
    )

    assert (returned_path / "chroma.sqlite3").exists(), (
        f"snapshot_download's real placement and the function's returned path disagree — "
        f"returned {returned_path}, but files actually downloaded elsewhere"
    )


def test_pull_with_a_single_segment_path_in_repo_still_works(tmp_path, monkeypatch):
    """Single-segment path_in_repo was already correct before the fix;
    confirm it still works."""
    import open_telco_rag.stores.hf as hf_module
    monkeypatch.setattr(hf_module, "snapshot_download", _fake_snapshot_download_matching_real_behavior)

    returned_path = pull_collection_from_hf(
        repo_id="fake/repo",
        path_in_repo="some-run-name",
        local_dir=str(tmp_path),
    )

    assert (returned_path / "chroma.sqlite3").exists()


def test_pull_returns_the_expected_nested_path(tmp_path, monkeypatch):
    import open_telco_rag.stores.hf as hf_module
    monkeypatch.setattr(hf_module, "snapshot_download", _fake_snapshot_download_matching_real_behavior)

    returned_path = pull_collection_from_hf(
        repo_id="fake/repo", path_in_repo="final-ingestion-grid/my-run", local_dir=str(tmp_path),
    )

    assert returned_path == tmp_path / "final-ingestion-grid" / "my-run"
