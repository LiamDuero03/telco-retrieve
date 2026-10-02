"""Download the GSMA/3GPP Markdown conversions of one release and stage them under names the 3GPP loader reads.

Only marked/<release>/**/raw.md and marked/<release>/conversion_log.jsonl are downloaded (never the figure
images). Every file is verified against the listing of the pinned dataset revision, then copied to
<out>/<release>/<NN>_series/<name>.md. A manifest records what was staged and what was left out. Nothing is
committed or redistributed: the files go from the original source onto your machine. Check the dataset's
licence and terms before you use it.

The conversion log has one row per conversion attempt, with the spec id, the series, the name of the 3GPP
document that was converted (its name carries the version code, e.g. 38300-j20.docx) and a status. For each
spec the last row with status OK is used. The loader reads the version from the file name, so:

  spec without a part   ->  <5 digits>-<version code>.md      e.g. 38300-j20.md
  spec with a part      ->  <5 digits>-<part>.md              e.g. 32111-1.md (the loader allows one suffix,
                                                               so the version is only in the manifest)

Usage:
    python scripts/stage_3gpp.py --revision <40-hex commit> --out data/3gpp/staged
    python scripts/stage_3gpp.py --revision <40-hex commit> --out data/3gpp/staged --dry-run
    python scripts/stage_3gpp.py --revision <40-hex commit> --out data/3gpp/staged --verify
    open-telco-rag ingest 3gpp --data data/3gpp/staged ...

HF_TOKEN is read from the environment by huggingface_hub when it is set; it is never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from open_telco_rag.sources.tgpp import TPPSource

SCHEMA_VERSION = 1
DATASET_REPO = "GSMA/3GPP"
STAGED_NAME = re.compile(r"^(?P<release>Rel-\d{1,2})/(?P<series>\d{2}_series)/(?P<name>\d{5}(?:-[0-9a-z]{1,3})?\.md)\Z")
REVISION = re.compile(r"^[0-9a-f]{40}\Z")
RELEASE = re.compile(r"^Rel-\d{1,2}\Z")
SPEC_ID = re.compile(r"^\d{5}(?:-\d{1,2})?\Z")
SERIES = re.compile(r"^\d{2}_series\Z")
# <spec>[-<part>]-<version code>[_<n>][_cover].doc|docx ; the version code is three characters, 0-9 then a-z
FILENAME = re.compile(r"^(?P<spec>\d{5})(?:-(?P<part>\d{1,2}))?-(?P<ver>[0-9a-z]{3})(?:_\d+)?(?:_cover)?\.(?:docx|doc)\Z")
LOADER_VERSION = re.compile(r"^[a-j]\d{2}$")      # what TPPSource.metadata() reads as a version
H1_VERSION = re.compile(r"\bV(\d+)\.(\d+)\.(\d+)\b")
H1_LINES = 40
STATUSES = ("OK", "ERROR")


class StageError(Exception):
    """a size or hash mismatch, a name collision or a verify failure: the run must exit non-zero"""


class Entry(NamedTuple):
    size: int
    sha256: str | None      # set only where the Hub stores the file in LFS
    blob_id: str | None     # git blob id (sha1 of "blob <size>\0" + content)


# ── small helpers ─────────────────────────────────────────────────────────────

_retry_lock = threading.Lock()
_retries = 0
_sleep = time.sleep


def with_retries(fn, tries: int = 6, what: str = "request"):
    """call fn(); retry on network errors, HTTP 429 and 5xx with exponential backoff"""
    global _retries
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:                                
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status in (429, 500, 502, 503, 504) or (
                status is None and type(exc).__name__ in ("ConnectError", "ReadError", "ReadTimeout", "ConnectTimeout",
                                                         "RemoteProtocolError", "ConnectionError", "Timeout", "OSError"))
            if not retryable or attempt == tries - 1:
                raise
            with _retry_lock:
                _retries += 1
            _sleep(min(60, 2 ** attempt))
    raise AssertionError("unreachable")


def hash_file(path: Path) -> tuple[int, str, str]:
    """(size, sha256, git blob id) of a file, read once"""
    size = path.stat().st_size
    sha256, sha1 = hashlib.sha256(), hashlib.sha1(b"blob %d\0" % size)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha256.update(block)
            sha1.update(block)
    return size, sha256.hexdigest(), sha1.hexdigest()


def check_entry(path: Path, entry: Entry) -> str:
    """verify a downloaded file against its listing entry; returns its sha256"""
    size, sha256, blob = hash_file(path)
    if size != entry.size:
        raise StageError(f"size mismatch: listing says {entry.size}, file has {size}")
    if entry.sha256 is not None and sha256 != entry.sha256:
        raise StageError("sha256 mismatch against the listing")
    if entry.sha256 is None and entry.blob_id is not None and blob != entry.blob_id:
        raise StageError("git blob id mismatch against the listing")
    return sha256


def write_atomic(dest: Path, data: bytes | None = None, source: Path | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            if source is not None:
                with source.open("rb") as src:
                    shutil.copyfileobj(src, handle, 1 << 20)
            else:
                handle.write(data or b"")
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def inside(base: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(base.resolve())


def safe(value: object, limit: int = 60) -> str:
    """printable ascii form of untrusted text for messages and the manifest"""
    return ascii(str(value))[:limit]


# ── versions ──────────────────────────────────────────────────────────────────

def decode_version(code: str) -> str:
    """3GPP version code to V<major>.<minor>.<patch>: each character is 0-9 then a=10 ... z=35 (j20 = V19.2.0)"""
    if not re.fullmatch(r"[0-9a-z]{3}", code):
        raise ValueError(f"not a version code: {code!r}")
    major, minor, patch = (int(c, 36) for c in code)
    return f"V{major}.{minor}.{patch}"


def header_version_check(text: str, decoded: str) -> str:
    """'match', 'mismatch' or 'absent': does a V<maj>.<min>.<pat> string in the first lines agree with the log"""
    for line in text.splitlines()[:H1_LINES]:
        found = H1_VERSION.search(line)
        if found:
            return "match" if "V" + ".".join(found.groups()) == decoded else "mismatch"
    return "absent"


def doc_part_kind(spec_number: str) -> str:
    """study reports are numbered x.700 to x.899; there the part after the dash is a study-item number, not a part"""
    return "study_item_number" if 700 <= int(spec_number[3:]) <= 899 else "part"


# ── sources: where the listing and the files come from ───────────────────────

class HubSource:
    def __init__(self, revision: str):
        from huggingface_hub import HfApi
        self.revision, self.api = revision, HfApi()

    def listing(self, release: str) -> dict[str, Entry]:
        def run():
            tree = self.api.list_repo_tree(DATASET_REPO, repo_type="dataset", revision=self.revision,
                                           path_in_repo=f"marked/{release}", recursive=True, expand=True)
            return {item.path: Entry(item.size, item.lfs.sha256 if item.lfs else None, item.blob_id)
                    for item in tree if type(item).__name__ == "RepoFile"}
        return with_retries(run, what="listing")

    def fetch(self, path: str, raw_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(with_retries(lambda: hf_hub_download(DATASET_REPO, path, repo_type="dataset",
                                                         revision=self.revision, local_dir=str(raw_dir))))

    def fetch_bytes(self, path: str) -> bytes:
        from huggingface_hub import hf_hub_download
        with tempfile.TemporaryDirectory() as tmp:
            return Path(with_retries(lambda: hf_hub_download(DATASET_REPO, path, repo_type="dataset",
                                                             revision=self.revision, local_dir=tmp))).read_bytes()


class DirSource:
    """an already downloaded tree with the dataset's layout (<dir>/marked/<release>/...); offline"""
    def __init__(self, root: Path):
        self.root = Path(root)

    def listing(self, release: str) -> dict[str, Entry]:
        out = {}
        base = self.root / "marked" / release
        for file in sorted(p for p in base.rglob("*") if p.is_file()):
            size, _, blob = hash_file(file)
            out[file.relative_to(self.root).as_posix()] = Entry(size, None, blob)
        return out

    def fetch(self, path: str, raw_dir: Path) -> Path:
        return self.root / path

    def fetch_bytes(self, path: str) -> bytes:
        return (self.root / path).read_bytes()


# ── planning ──────────────────────────────────────────────────────────────────

def parse_log(data: bytes) -> tuple[list[dict], list[dict]]:
    """(rows, problems): the log is data only; rows that are not an object are reported and skipped"""
    rows, problems = [], []
    for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            problems.append({"kind": "unreadable_log_line", "detail": f"line {number}"})
            continue
        if not isinstance(row, dict):
            problems.append({"kind": "unreadable_log_line", "detail": f"line {number}"})
            continue
        rows.append(row)
    return rows, problems


def build_plan(rows: list[dict], listing: dict[str, Entry], release: str, stub_bytes: int) -> dict:
    """decide, per spec, what is staged under which name; nothing here touches the disk"""
    problems: list[dict] = []
    by_spec: dict[str, list[dict]] = {}
    for row in rows:
        spec_id, filename, status = row.get("spec_id"), row.get("filename"), row.get("status")
        if not isinstance(spec_id, str) or not SPEC_ID.match(spec_id):
            problems.append({"kind": "invalid_spec_id", "detail": safe(spec_id)})
            continue
        if status not in STATUSES:
            problems.append({"kind": "unknown_status", "spec_id": spec_id, "detail": safe(status)})
            continue
        if not isinstance(filename, str) or not FILENAME.match(filename):
            problems.append({"kind": "invalid_filename", "spec_id": spec_id, "detail": safe(filename)})
            continue
        by_spec.setdefault(spec_id, []).append(row)

    raw_paths = {}
    for path in listing:
        found = re.fullmatch(rf"marked/{re.escape(release)}/(\d{{2}}_series)/(\d{{5}}(?:-\d{{1,2}})?)/raw\.md", path)
        if found:
            raw_paths[found.group(2)] = (path, found.group(1))

    staged, excluded, flags = [], [], {"error_then_ok": [], "multi_ok_versions": [], "error_only": [], "no_raw_md": [],
                                       "raw_md_without_ok_row": []}
    names: dict[str, str] = {}
    for spec_id in sorted(by_spec):
        spec_rows = by_spec[spec_id]
        ok = [r for r in spec_rows if r["status"] == "OK"]
        if not ok:
            flags["error_only"].append(spec_id)
            excluded.append({"spec_id": spec_id, "reason": "error_only"})
            continue
        if any(r["status"] == "ERROR" for r in spec_rows[:spec_rows.index(ok[-1])]):
            flags["error_then_ok"].append(spec_id)
        last = ok[-1]
        versions = {FILENAME.match(r["filename"]).group("ver") for r in ok}
        if len(versions) > 1:
            flags["multi_ok_versions"].append(spec_id)
        match = FILENAME.match(last["filename"])
        spec, part, code = match.group("spec"), match.group("part"), match.group("ver")
        expected_id = spec + (f"-{part}" if part else "")
        series = f"{spec[:2]}_series"
        if expected_id != spec_id or last.get("series") != series:
            problems.append({"kind": "log_row_inconsistent", "spec_id": spec_id, "detail": safe(last["filename"])})
            excluded.append({"spec_id": spec_id, "reason": "invalid_log_row"})
            continue
        if not part and not LOADER_VERSION.match(code):
            problems.append({"kind": "version_code_not_loadable", "spec_id": spec_id, "detail": code})
            excluded.append({"spec_id": spec_id, "reason": "version_code_not_loadable"})
            continue
        if spec_id not in raw_paths or raw_paths[spec_id][1] != series:
            flags["no_raw_md"].append(spec_id)
            excluded.append({"spec_id": spec_id, "reason": "no_raw_md"})
            continue
        path = raw_paths[spec_id][0]
        name = f"{spec}-{part}.md" if part else f"{spec}-{code}.md"
        staged_name = f"{release}/{series}/{name}"
        if staged_name in names:
            raise StageError(f"name collision: {spec_id} and {names[staged_name]} both stage as {staged_name}")
        names[staged_name] = spec_id
        spec_number = f"{spec[:2]}.{spec[2:]}"
        entry = listing[path]
        item = {"source_path": path, "staged_name": staged_name, "size": entry.size, "spec_id": spec_id,
                "spec_number": spec_number, "series": series, "version_code": code,
                "version_decoded": decode_version(code), "version_source": "conversion_log",
                "version_in_filename": not part, "doc_part": part, "log_filename": last["filename"],
                "log_status": "OK", "is_stub": entry.size < stub_bytes}
        if part:
            item["doc_part_kind"] = doc_part_kind(spec_number)
        staged.append(item)
        if item["is_stub"]:
            excluded.append({"spec_id": spec_id, "reason": "stub_staged", "size": entry.size})
    for spec_id in sorted(set(raw_paths) - {s for s, rs in by_spec.items() if any(r["status"] == "OK" for r in rs)}):
        flags["raw_md_without_ok_row"].append(spec_id)
        excluded.append({"spec_id": spec_id, "reason": "no_ok_row"})
    return {"staged": staged, "excluded": sorted(excluded, key=lambda e: (e["spec_id"], e["reason"])),
            "flags": {k: sorted(v) for k, v in flags.items()}, "problems": problems}


def count_summary(rows: list[dict], plan: dict, listing: dict[str, Entry]) -> dict:
    staged = plan["staged"]
    return {
        "log_rows": len(rows), "ok_rows": sum(r.get("status") == "OK" for r in rows),
        "error_rows": sum(r.get("status") == "ERROR" for r in rows),
        "doc_rows": sum(isinstance(r.get("filename"), str) and r["filename"].lower().endswith(".doc") for r in rows),
        "cover_specs": len({r.get("spec_id") for r in rows if isinstance(r.get("filename"), str) and "_cover." in r["filename"]}),
        "raw_md_in_listing": sum(p.endswith("/raw.md") for p in listing),
        "other_files_in_listing_not_downloaded": sum(not p.endswith(("/raw.md", "conversion_log.jsonl")) for p in listing),
        "specs_staged": len(staged), "multi_part_specs_staged": sum(bool(s["doc_part"]) for s in staged),
        "stubs": sum(s["is_stub"] for s in staged), "problems": len(plan["problems"]),
    }


# ── staging ───────────────────────────────────────────────────────────────────

def download_all(source, plan: dict, listing: dict[str, Entry], raw_dir: Path, workers: int) -> dict[str, Path]:
    """fetch and verify every planned raw.md (files that already verify are kept); returns spec_id -> local path"""
    def one(item: dict) -> tuple[str, Path, str]:
        entry = listing[item["source_path"]]
        local = Path(source.fetch(item["source_path"], raw_dir))
        try:
            return item["spec_id"], local, check_entry(local, entry)
        except (StageError, OSError):
            if isinstance(source, DirSource):
                raise
            local.unlink(missing_ok=True)
            local = Path(source.fetch(item["source_path"], raw_dir))
            return item["spec_id"], local, check_entry(local, entry)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, plan["staged"]))
    for item, (_, _, sha) in zip(plan["staged"], results):
        item["sha256"] = sha
    return {spec: path for spec, path, _ in results}


def stage_files(plan: dict, locals_: dict[str, Path], out: Path) -> None:
    for item in plan["staged"]:
        dest = out / item["staged_name"]
        if not STAGED_NAME.match(item["staged_name"]) or not inside(out, dest):
            raise StageError(f"refusing to write {safe(item['staged_name'])}")
        if dest.exists():
            if hash_file(dest)[1] != item["sha256"]:
                raise StageError(f"{item['staged_name']} exists with different content; not overwriting")
            continue
        write_atomic(dest, source=locals_[item["spec_id"]])


def check_with_loader(plan: dict, out: Path) -> None:
    """the staged names must be what the 3GPP loader parses: compare its metadata with what the plan says"""
    source = TPPSource()
    for item in plan["staged"]:
        meta = source.metadata(out / item["staged_name"])
        release = item["staged_name"].split("/")[0]
        expect = {"spec_number": item["spec_number"], "series": item["spec_number"][:2], "release": release}
        if item["doc_part"]:
            expect["doc_part"] = item["doc_part"]
            expect["canonical_id"] = f"{item['spec_number']}-{item['doc_part']} ({release})"
        else:
            expect["version"] = item["version_code"]
            expect["canonical_id"] = f"{item['spec_number']} ({release})"
        bad = {k: (meta.get(k), v) for k, v in expect.items() if meta.get(k) != v}
        if bad or ("doc_part" in meta) != bool(item["doc_part"]):
            raise StageError(f"the loader reads {item['staged_name']} differently from the plan: {bad}")


def add_header_checks(plan: dict, locals_: dict[str, Path]) -> None:
    for item in plan["staged"]:
        head = "".join(line for _, line in zip(range(H1_LINES), locals_[item["spec_id"]].open(encoding="utf-8", errors="ignore")))
        item["h1_version_check"] = header_version_check(head, item["version_decoded"])


def build_manifest(args, plan: dict, rows: list[dict], listing: dict[str, Entry], log_sha256: str, log_size: int) -> dict:
    files = sorted(plan["staged"], key=lambda f: f["staged_name"])
    keys = ("source_path", "staged_name", "size", "sha256", "spec_number", "series", "version_code", "version_decoded",
            "version_source", "version_in_filename", "h1_version_check", "doc_part", "doc_part_kind", "is_stub",
            "log_status", "log_filename")
    counts = count_summary(rows, plan, listing)
    counts["h1_version_check"] = {k: sum(f["h1_version_check"] == k for f in files) for k in ("match", "mismatch", "absent")}
    return {
        "manifest_schema_version": SCHEMA_VERSION, "dataset_repo": DATASET_REPO, "revision": args.revision,
        "release": args.release, "stub_bytes": args.stub_bytes,
        "conversion_log": {"path": f"marked/{args.release}/conversion_log.jsonl", "sha256": log_sha256, "size": log_size},
        "counts": counts, "files": [{k: f[k] for k in keys if k in f} for f in files],
        "excluded_specs": plan["excluded"], "flags": plan["flags"], "problems": plan["problems"],
    }


def dump_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode("ascii")


def verify_staged(out: Path) -> list[str]:
    """re-hash every staged file against manifest.json; returns the problems found"""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    found, listed = [], set()
    for item in manifest["files"]:
        name = item["staged_name"]
        listed.add(name)
        path = out / name
        if not STAGED_NAME.match(name) or not inside(out, path):
            found.append(f"unsafe name in manifest: {safe(name)}")
        elif not path.is_file():
            found.append(f"missing: {name}")
        else:
            size, sha, _ = hash_file(path)
            if size != item["size"] or sha != item["sha256"]:
                found.append(f"changed: {name}")
    release = manifest["release"]
    for path in sorted((out / release).rglob("*.md")) if (out / release).is_dir() else []:
        name = path.relative_to(out).as_posix()
        if name not in listed:
            found.append(f"not in manifest: {name}")
    return found


# ── command line ──────────────────────────────────────────────────────────────

def print_plan(plan: dict, counts: dict, dry: bool) -> None:
    print(("Plan (dry run, nothing written)" if dry else "Plan") + ":")
    for key, value in counts.items():
        print(f"  {key:42} {value}")
    for key, specs in plan["flags"].items():
        print(f"  {key:42} {len(specs)}" + (f"  {', '.join(specs[:12])}{' ...' if len(specs) > 12 else ''}" if specs else ""))
    stubs = [s for s in plan["staged"] if s["is_stub"]]
    if stubs:
        print("  stubs (still staged): " + ", ".join(f"{s['spec_number']} ({s['size']} B)" for s in sorted(stubs, key=lambda s: s["size"])))
    for problem in plan["problems"][:20]:
        print(f"  PROBLEM {problem['kind']}: {problem.get('spec_id', '')} {problem.get('detail', '')}")


def main(argv: list[str] | None = None, source=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", required=True, help="40-hex commit of the dataset (a branch name is refused)")
    parser.add_argument("--release", default="Rel-19", help="release to stage (default %(default)s; only Rel-19 is tested)")
    parser.add_argument("--out", type=Path, required=True, help="staging directory, the one to give to `ingest --data`")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="download area (default: <out>_raw, next to --out, so that ingest on --out never sees it)")
    parser.add_argument("--dry-run", action="store_true", help="fetch only the listing and the log; write nothing")
    parser.add_argument("--stub-bytes", type=int, default=5120, help="files smaller than this are flagged is_stub (default %(default)s)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source", type=Path, default=None, help="stage from an already downloaded tree (offline)")
    parser.add_argument("--verify", action="store_true", help="re-hash the staged files against <out>/manifest.json and exit")
    args = parser.parse_args(argv)

    if not REVISION.match(args.revision):
        print("error: --revision must be a 40-character lowercase hex commit hash, not a branch or tag name", file=sys.stderr)
        return 1
    if not RELEASE.match(args.release):
        print("error: --release must look like Rel-19", file=sys.stderr)
        return 1
    out = args.out
    raw_dir = args.raw_dir or out.with_name(out.name + "_raw")
    if any(RELEASE.match(p) or SERIES.match(p) for p in out.resolve().parts):
        print("warning: --out has a Rel-NN or NN_series folder in its path; the loader would read it as the release/series", file=sys.stderr)

    if args.verify:
        problems = verify_staged(out)
        for line in problems:
            print("VERIFY", line)
        print(f"verify: {'FAILED, ' + str(len(problems)) + ' problem(s)' if problems else 'all staged files match the manifest'}")
        return 1 if problems else 0

    source = source or (DirSource(args.source) if args.source else HubSource(args.revision))
    log_path = f"marked/{args.release}/conversion_log.jsonl"
    try:
        listing = source.listing(args.release)
        if log_path not in listing:
            raise StageError(f"{log_path} is not in the listing of revision {args.revision}")
        if args.dry_run:
            log_bytes = source.fetch_bytes(log_path)
        else:
            log_local = Path(source.fetch(log_path, raw_dir))
            log_bytes = log_local.read_bytes()
        check_entry_bytes = hashlib.sha1(b"blob %d\0" % len(log_bytes) + log_bytes).hexdigest()
        entry = listing[log_path]
        if len(log_bytes) != entry.size or (entry.sha256 and hashlib.sha256(log_bytes).hexdigest() != entry.sha256) or (
                not entry.sha256 and entry.blob_id and check_entry_bytes != entry.blob_id):
            raise StageError("conversion_log.jsonl does not match its listing entry")
        rows, log_problems = parse_log(log_bytes)
        plan = build_plan(rows, listing, args.release, args.stub_bytes)
        plan["problems"] = log_problems + plan["problems"]
        counts = count_summary(rows, plan, listing)
        print_plan(plan, counts, args.dry_run)
        if args.dry_run:
            print("dry run: no spec file was downloaded and nothing was written")
            return 0

        locals_ = download_all(source, plan, listing, raw_dir, args.workers)
        add_header_checks(plan, locals_)
        stage_files(plan, locals_, out)
        check_with_loader(plan, out)
        log_sha = hashlib.sha256(log_bytes).hexdigest()
        manifest_bytes = dump_manifest(build_manifest(args, plan, rows, listing, log_sha, len(log_bytes)))
        write_atomic(out / "manifest.json", data=manifest_bytes)
        run = {"tool": "scripts/stage_3gpp.py", "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "python": sys.version.split()[0], "workers": args.workers, "retries": _retries,
               "files_staged": len(plan["staged"]), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
        write_atomic(out / "staging_run.json", data=(json.dumps(run, sort_keys=True, indent=1) + "\n").encode("ascii"))
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nstaged {len(plan['staged'])} files under {out / args.release}")
    print(f"manifest.json sha256: {run['manifest_sha256']}")
    if plan["flags"]["error_only"]:
        print("warning (expected): specs whose every conversion failed are not staged: " + ", ".join(plan["flags"]["error_only"]))
    print(f"\nnext: open-telco-rag ingest 3gpp --data {out} --strategy hierarchical_markdown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
