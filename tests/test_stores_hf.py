"""Tests for open_telco_rag.stores.hf.pull_collection_from_hf() — a real,
confirmed-in-production bug: snapshot_download() always replicates a
matched file's FULL repo-relative path under whatever local_dir it's
given ("the file structure from the repo will be replicated in this
location" — its own docstring), it does NOT strip path_in_repo's own
prefix off first. The previous implementation computed snapshot_download's
own local_dir as (Path(local_dir) / path_in_repo).parent, which only
correctly un-nests path_in_repo when it's a SINGLE path segment. For a
two-segment path_in_repo like "final-ingestion-grid/{run_name}" (what
every grid script in this repo actually passes), that under-strips by one
segment — the real files land ONE directory level deeper than the
function claims to return, so ChromaStore silently opens an empty
directory and creates a fresh empty collection (no exception, just
store.count() == 0 downstream) even though the download itself genuinely
succeeded.

snapshot_download is mocked here to replicate its REAL documented
behavior (write matched files at {local_dir}/{full_repo_relative_path}),
not just stubbed to no-op — otherwise this test wouldn't actually catch
the path-nesting bug.
"""
from __future__ import annotations

from pathlib import Path

from open_telco_rag.stores.hf import pull_collection_from_hf


def _fake_snapshot_download_matching_real_behavior(*, repo_id, repo_type, allow_patterns, local_dir):
    """Mimics snapshot_download's real, documented placement behavior:
    a matched file's FULL repo-relative path is replicated under
    local_dir. allow_patterns is "{path_in_repo}/*" — strip the "/*" to
    recover path_in_repo, then write a dummy file at exactly the path
    snapshot_download would actually place it at.
    """
    repo_relative_dir = allow_patterns.rsplit("/*", 1)[0]
    target = Path(local_dir) / repo_relative_dir / "chroma.sqlite3"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("fake chroma db", encoding="utf-8")


def test_pull_with_a_two_segment_path_in_repo_finds_the_real_downloaded_files(tmp_path, monkeypatch):
    """The exact regression case: master_grid_eval.py/pilot_eval.py both
    call this with path_in_repo="final-ingestion-grid/{run_name}" (two
    segments)."""
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
    """The CLI's own pull-hf --help text documents path_in_repo as just
    "run_name" (one segment) — this case was already correct before the
    fix; confirm the fix didn't break it."""
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
