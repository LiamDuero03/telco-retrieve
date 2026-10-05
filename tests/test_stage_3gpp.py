"""scripts/stage_3gpp.py on a synthetic conversion log and a synthetic source tree: no network, no spec text."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_telco_rag.sources.tgpp import TPPSource

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "stage_3gpp.py"
REVISION = "a" * 40
_spec = importlib.util.spec_from_file_location("scripts_stage_3gpp", SCRIPT)
stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage)


def row(spec_id, filename, status="OK"):
    return {"spec_id": spec_id, "series": spec_id[:2] + "_series", "filename": filename, "status": status,
            "chars": 1, "docx_bytes": 1, "splits": 0, "elapsed_s": 1.0}


def body(label: str, header_version: str | None = None, pad: int = 6000) -> str:
    head = f"synthetic document {label}\n" + (f"Version {header_version}\n" if header_version else "")
    return head + "x" * pad + "\n"


LOG_ROWS = [
    row("38300", "38300-j20.docx"),                        # normal
    row("32111-1", "32111-1-j00.docx"),                    # multi-part
    row("36101", "36101-j10_cover.docx"),                  # _cover
    row("29522", "29522-j20_0_cover.docx"),                # _<n>_cover
    row("25214", "25214-j00.doc", "ERROR"),                # ERROR then OK, a .doc
    row("25214", "25214-j00.doc"),
    row("25331", "25331-j01.doc", "ERROR"),                # ERROR only
    row("25331", "25331-j01.doc", "ERROR"),
    row("23401", "23401-j10.docx"),                        # several OK rows, different versions
    row("23401", "23401-j20.docx"),
    row("28000", "28000-j00.docx"),                        # OK but no raw.md
    row("23700-75", "23700-75-j00.docx"),                  # a study-item number, not a part
    row("22242", "22242-j00.docx"),                        # a stub
    row("38101-3", "38101-3-j10.docx"),                    # header version disagrees with the log
    row("../../evil", "38300-j20.docx"),                   # hostile values
    row("/etc/passwd", "x.docx"),
    row("38\n300", "38300-j20.docx"),
    row("26101", "../38300-j20.docx"),
    row("26102", "..\\26102-j20.docx"),
    row("26103", "26103-j20.docx", "PENDING"),
    row("27001", "27001-k00.docx"),                        # a version letter the loader would read as a part
    row("21111", "21111-j00.docx"),                        # series disagrees with the spec number
]
LOG_ROWS[-1]["series"] = "99_series"
RAW = {                                                      # folder -> text
    "38300": body("a", "V19.2.0"), "32111-1": body("b", "V19.0.0"), "36101": body("c", "V19.1.0"),
    "29522": body("d"), "25214": body("e", "V19.0.0"), "23401": body("f", "V19.2.0"), "23700-75": body("g", "V19.0.0"),
    "22242": body("h", "V19.0.0", pad=100), "38101-3": body("i", "V19.9.9"), "29999": body("j"),
    "26101": body("k"), "27001": body("l"), "21111": body("m"),
}


def make_tree(root: Path, rows=LOG_ROWS, raw=RAW, release="Rel-19") -> Path:
    base = root / "marked" / release
    base.mkdir(parents=True)
    (base / "conversion_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for folder, text in raw.items():
        target = base / f"{folder[:2]}_series" / folder / "raw.md"
        target.parent.mkdir(parents=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    (base / "38_series" / "38300" / "image1.jpg").write_bytes(b"not an image")        # never downloaded
    return root


@pytest.fixture()
def tree(tmp_path):
    return make_tree(tmp_path / "src")


def run(tree, out, *extra):
    return stage.main(["--revision", REVISION, "--source", str(tree), "--out", str(out), *extra])


def manifest_of(out):
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_the_staged_names_are_what_the_loader_expects(tree, tmp_path):
    out = tmp_path / "out"
    assert run(tree, out) == 0
    staged = sorted(p.relative_to(out).as_posix() for p in (out / "Rel-19").rglob("*.md"))
    assert staged == ["Rel-19/22_series/22242-j00.md", "Rel-19/23_series/23401-j20.md", "Rel-19/23_series/23700-75.md",
                      "Rel-19/25_series/25214-j00.md", "Rel-19/29_series/29522-j20.md", "Rel-19/32_series/32111-1.md",
                      "Rel-19/36_series/36101-j10.md", "Rel-19/38_series/38101-3.md", "Rel-19/38_series/38300-j20.md"]
    meta = TPPSource().metadata
    m = meta(out / "Rel-19/38_series/38300-j20.md")
    assert (m["release"], m["series"], m["spec_number"], m["version"], m["canonical_id"]) == ("Rel-19", "38", "38.300", "j20", "38.300 (Rel-19)")
    assert "doc_part" not in m
    m = meta(out / "Rel-19/32_series/32111-1.md")
    assert (m["release"], m["series"], m["spec_number"], m["doc_part"], m["canonical_id"]) == ("Rel-19", "32", "32.111", "1", "32.111-1 (Rel-19)")
    assert "version" not in m
    m = meta(out / "Rel-19/36_series/36101-j10.md")                    # the _cover suffix must not leak into the name
    assert (m["spec_number"], m["version"]) == ("36.101", "j10") and "doc_part" not in m
    assert meta(out / "Rel-19/29_series/29522-j20.md")["version"] == "j20"
    assert meta(out / "Rel-19/23_series/23700-75.md")["doc_part"] == "75"


def test_manifest_content(tree, tmp_path):
    out = tmp_path / "out"
    run(tree, out)
    m = manifest_of(out)
    assert (m["manifest_schema_version"], m["dataset_repo"], m["revision"], m["release"]) == (1, "GSMA/3GPP", REVISION, "Rel-19")
    files = {f["staged_name"]: f for f in m["files"]}
    n = files["Rel-19/38_series/38300-j20.md"]
    assert n["version_decoded"] == "V19.2.0" and n["version_in_filename"] is True and n["h1_version_check"] == "match"
    assert n["sha256"] == hashlib.sha256((out / n["staged_name"]).read_bytes()).hexdigest()
    assert n["size"] == (out / n["staged_name"]).stat().st_size and n["version_source"] == "conversion_log"
    p = files["Rel-19/32_series/32111-1.md"]
    assert p["version_in_filename"] is False and p["doc_part"] == "1" and p["version_code"] == "j00" and p["doc_part_kind"] == "part"
    assert files["Rel-19/23_series/23700-75.md"]["doc_part_kind"] == "study_item_number"
    assert files["Rel-19/38_series/38101-3.md"]["h1_version_check"] == "mismatch"
    assert files["Rel-19/29_series/29522-j20.md"]["h1_version_check"] == "absent"
    assert files["Rel-19/22_series/22242-j00.md"]["is_stub"] is True and n["is_stub"] is False
    assert files["Rel-19/23_series/23401-j20.md"]["version_code"] == "j20"                # the last OK row wins
    flags = m["flags"]
    assert flags["error_only"] == ["25331"] and flags["error_then_ok"] == ["25214"] and flags["multi_ok_versions"] == ["23401"]
    assert flags["no_raw_md"] == ["28000"] and flags["raw_md_without_ok_row"] == ["26101", "29999"]
    reasons = {(e["spec_id"], e["reason"]) for e in m["excluded_specs"]}
    assert {("25331", "error_only"), ("28000", "no_raw_md"), ("29999", "no_ok_row"), ("22242", "stub_staged"),
            ("27001", "version_code_not_loadable"), ("21111", "invalid_log_row")} <= reasons
    assert m["counts"]["h1_version_check"] == {"match": 7, "mismatch": 1, "absent": 1}
    assert "other_files_in_listing_not_downloaded" not in m["counts"] and m["counts"]["doc_rows"] == 4
    assert json.loads((out / "staging_run.json").read_text())["listing_files_not_downloaded"] == 1


def test_hostile_log_values_never_reach_the_disk_or_the_names(tree, tmp_path):
    out = tmp_path / "out"
    run(tree, out)
    m = manifest_of(out)
    assert all(stage.STAGED_NAME.match(f["staged_name"]) for f in m["files"])
    kinds = [p["kind"] for p in m["problems"]]
    assert kinds.count("invalid_spec_id") == 3 and kinds.count("invalid_filename") == 2 and kinds.count("unknown_status") == 1
    dumped = json.dumps(m)
    assert "'../../evil'" in dumped          # shown only as quoted text
    for path in tmp_path.rglob("*"):
        assert path.resolve().is_relative_to(tmp_path.resolve())
    assert not (tmp_path / "evil").exists() and not list(tmp_path.glob("*.docx"))


def test_the_manifest_is_deterministic_and_has_no_local_paths(tree, tmp_path):
    run(tree, tmp_path / "one")
    run(tree, tmp_path / "two", "--raw-dir", str(tmp_path / "elsewhere"))
    first, second = (tmp_path / "one" / "manifest.json").read_bytes(), (tmp_path / "two" / "manifest.json").read_bytes()
    assert first == second
    text = first.decode("ascii")
    assert str(tmp_path) not in text and "Users" not in text and "Temp" not in text
    assert "timestamp" not in text and "finished" not in text
    run_info = json.loads((tmp_path / "one" / "staging_run.json").read_text())
    assert run_info["manifest_sha256"] == hashlib.sha256(first).hexdigest() and "finished_utc" in run_info


def test_a_rerun_is_idempotent_and_never_overwrites_different_content(tree, tmp_path):
    out = tmp_path / "out"
    run(tree, out)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "staging_run.json"}
    assert run(tree, out) == 0
    assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "staging_run.json"}
    target = out / "Rel-19/38_series/38300-j20.md"
    target.write_text("edited", encoding="utf-8")
    assert run(tree, out) == 1
    assert target.read_text(encoding="utf-8") == "edited"


def test_verify_detects_a_tampered_missing_or_extra_file(tree, tmp_path, capsys):
    out = tmp_path / "out"
    run(tree, out)
    assert stage.main(["--revision", REVISION, "--out", str(out), "--verify"]) == 0
    (out / "Rel-19/38_series/38300-j20.md").write_text("tampered", encoding="utf-8")
    (out / "Rel-19/32_series/32111-1.md").unlink()
    (out / "Rel-19/38_series/99999-j00.md").write_text("extra", encoding="utf-8")
    assert stage.main(["--revision", REVISION, "--out", str(out), "--verify"]) == 1
    printed = capsys.readouterr().out
    assert "changed: Rel-19/38_series/38300-j20.md" in printed and "missing: Rel-19/32_series/32111-1.md" in printed
    assert "not in manifest: Rel-19/38_series/99999-j00.md" in printed


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "a" * 41, "v1.0", ""])
def test_a_branch_name_or_a_short_hash_is_refused(tree, tmp_path, revision, capsys):
    assert stage.main(["--revision", revision, "--source", str(tree), "--out", str(tmp_path / "out")]) == 1
    assert "40-character" in capsys.readouterr().err and not (tmp_path / "out").exists()


@pytest.mark.parametrize("code,decoded", [("j20", "V19.2.0"), ("i11", "V18.1.1"), ("j00", "V19.0.0"), ("h3a", "V17.3.10")])
def test_version_decoding(code, decoded):
    assert stage.decode_version(code) == decoded


@pytest.mark.parametrize("bad", ["j2", "J20", "j2000", "../", ""])
def test_a_version_code_must_be_three_lowercase_alphanumerics(bad):
    with pytest.raises(ValueError):
        stage.decode_version(bad)


@pytest.mark.parametrize("name,ok", [
    ("38300-j20.docx", True), ("25214-j00.doc", True), ("36101-j10_cover.docx", True), ("29522-j20_0_cover.docx", True),
    ("38101-3-j10.docx", True), ("38300-j20.md", False), ("38300.docx", False), ("38300-j20.docx/..", False),
    ("../38300-j20.docx", False), ("38300-j20.docx\n", False), ("3830-j20.docx", False)])
def test_the_filename_pattern_is_strict(name, ok):
    assert bool(stage.FILENAME.match(name)) is ok


def test_stub_threshold_is_a_flag_only(tree, tmp_path):
    out = tmp_path / "out"
    run(tree, out, "--stub-bytes", "9000")
    m = manifest_of(out)
    assert sum(f["is_stub"] for f in m["files"]) == len(m["files"]) and len(m["files"]) == 9      # nothing is dropped


def test_dry_run_writes_nothing(tree, tmp_path, capsys):
    out = tmp_path / "out"
    assert run(tree, out, "--dry-run") == 0
    assert not out.exists() and not (tmp_path / "out_raw").exists()
    printed = capsys.readouterr().out
    assert "specs_staged" in printed and "no spec file was downloaded" in printed


class FakeHub:
    """a source that, like the Hub, gives LFS sha256 for some files and only a git blob id for the rest"""
    def __init__(self, root: Path, corrupt: str | None = None):
        self.inner, self.corrupt, self.fetched = stage.DirSource(root), corrupt, []

    def listing(self, release):
        listing = self.inner.listing(release)
        out = {}
        for path, entry in listing.items():
            data = (self.inner.root / path).read_bytes()
            lfs = hashlib.sha256(data).hexdigest() if path.endswith("38300/raw.md") else None
            out[path] = stage.Entry(entry.size, lfs, entry.blob_id)
        return out

    def fetch(self, path, raw_dir):
        self.fetched.append(path)
        target = Path(raw_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (self.inner.root / path).read_bytes()
        target.write_bytes(data + b"x" if path == self.corrupt else data)
        return target

    def fetch_bytes(self, path):
        return (self.inner.root / path).read_bytes()


def test_only_raw_md_and_the_log_are_downloaded_and_the_hashes_are_checked(tree, tmp_path):
    hub = FakeHub(tree)
    out = tmp_path / "out"
    assert stage.main(["--revision", REVISION, "--out", str(out)], source=hub) == 0
    assert all(p.endswith(("/raw.md", "conversion_log.jsonl")) for p in hub.fetched)
    assert not any(p.endswith(".jpg") for p in hub.fetched)
    assert not any(p.endswith(("29999/raw.md", "28000/raw.md")) for p in hub.fetched)           # nothing for excluded specs
    assert (tmp_path / "out_raw" / "marked" / "Rel-19" / "conversion_log.jsonl").is_file()     # the default raw dir sits next to --out


@pytest.mark.parametrize("corrupt", ["marked/Rel-19/38_series/38300/raw.md", "marked/Rel-19/32_series/32111-1/raw.md"])
def test_a_size_or_hash_mismatch_fails_the_run(tree, tmp_path, corrupt, capsys):
    out = tmp_path / "out"
    assert stage.main(["--revision", REVISION, "--out", str(out)], source=FakeHub(tree, corrupt=corrupt)) == 1
    assert "mismatch" in capsys.readouterr().err
    assert not (out / "manifest.json").exists()


def test_a_log_that_does_not_match_its_listing_entry_fails(tree, tmp_path):
    hub = FakeHub(tree)
    original = hub.listing
    hub.listing = lambda release: {p: (stage.Entry(e.size + 1, e.sha256, e.blob_id) if p.endswith("conversion_log.jsonl") else e)
                                   for p, e in original(release).items()}
    assert stage.main(["--revision", REVISION, "--out", str(tmp_path / "out")], source=hub) == 1


def test_retries_back_off_on_429_and_stop_on_other_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr(stage, "_sleep", sleeps.append)
    calls = {"n": 0}

    class Throttled(Exception):
        response = SimpleNamespace(status_code=429)

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Throttled()
        return "ok"
    assert stage.with_retries(flaky) == "ok" and sleeps == [1, 2]

    class NotFound(Exception):
        response = SimpleNamespace(status_code=404)

    def missing():
        raise NotFound()
    with pytest.raises(NotFound):
        stage.with_retries(missing)


def test_a_staged_tree_with_a_release_folder_above_it_is_warned_about(tree, tmp_path, capsys):
    out = tmp_path / "Rel-18" / "out"
    run(tree, out)
    assert "Rel-NN" in capsys.readouterr().err


def test_an_error_only_spec_is_a_warning_not_a_failure(tree, tmp_path, capsys):
    assert run(tree, tmp_path / "out") == 0
    assert "25331" in capsys.readouterr().out


def test_the_next_step_names_the_ingest_command(tree, tmp_path, capsys):
    out = tmp_path / "out"
    run(tree, out)
    assert f"open-telco-rag ingest 3gpp --data {out}" in capsys.readouterr().out


# ── the retry count and the "same manifest from a download and from --source" statement ──

class RetryingHub(FakeHub):
    """like FakeHub, but a fetch fails with HTTP 429 `throttled` times first (through the script's own retry wrapper),
    and the Hub library's own retries are simulated with its log message `hub_retries` times"""
    def __init__(self, root, throttled=0, hub_retries=0, extra_listing=()):
        super().__init__(root)
        self.throttled, self.hub_retries, self.extra_listing = throttled, hub_retries, extra_listing

    def listing(self, release):
        out = super().listing(release)
        out.update({path: stage.Entry(10, None, None) for path in self.extra_listing})     # more files in the Hub listing
        return out

    def fetch(self, path, raw_dir):
        import logging

        class Throttled(Exception):
            response = SimpleNamespace(status_code=429)

        def attempt():
            if path.endswith("38300/raw.md") and self.throttled > 0:
                self.throttled -= 1
                raise Throttled()
            return super(RetryingHub, self).fetch(path, raw_dir)
        if path.endswith("32111-1/raw.md"):
            for n in range(self.hub_retries):
                logging.getLogger("huggingface_hub.utils._http").warning(
                    "'429 Too Many Requests' thrown while requesting GET https://example.invalid/x. Retrying in 1s [Retry %d/5].", n + 1)
        return stage.with_retries(attempt)


def test_staging_run_records_the_real_retry_count(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "_sleep", lambda s: None)
    out = tmp_path / "out"
    hub = RetryingHub(tree, throttled=2, hub_retries=3)
    assert stage.main(["--revision", REVISION, "--out", str(out)], source=hub) == 0
    run = json.loads((out / "staging_run.json").read_text())
    assert run["retries_script"] == 2 and run["retries_hub"] == 3 and run["retries"] == 5


def test_the_retry_count_is_per_run_not_per_process(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "_sleep", lambda s: None)
    stage.main(["--revision", REVISION, "--out", str(tmp_path / "one")], source=RetryingHub(tree, throttled=1, hub_retries=1))
    stage.main(["--revision", REVISION, "--out", str(tmp_path / "two")], source=RetryingHub(tree))
    assert json.loads((tmp_path / "two" / "staging_run.json").read_text())["retries"] == 0


def test_a_run_without_retries_records_zero(tree, tmp_path):
    run_ = tmp_path / "out"
    run(tree, run_)
    info = json.loads((run_ / "staging_run.json").read_text())
    assert (info["retries"], info["retries_script"], info["retries_hub"]) == (0, 0, 0)


def test_the_manifest_is_identical_from_a_download_and_from_a_source_tree(tree, tmp_path):
    """the listing of a real download holds many figure files that a staged source tree does not: they must not
    reach the manifest. They are counted in staging_run.json."""
    run(tree, tmp_path / "from_tree")
    hub = RetryingHub(tree, extra_listing=[f"marked/Rel-19/38_series/38300/image{n}.jpg" for n in range(2, 50)])
    assert stage.main(["--revision", REVISION, "--out", str(tmp_path / "from_hub")], source=hub) == 0
    assert (tmp_path / "from_tree" / "manifest.json").read_bytes() == (tmp_path / "from_hub" / "manifest.json").read_bytes()
    assert json.loads((tmp_path / "from_hub" / "staging_run.json").read_text())["listing_files_not_downloaded"] == 49
    assert json.loads((tmp_path / "from_tree" / "staging_run.json").read_text())["listing_files_not_downloaded"] == 1
