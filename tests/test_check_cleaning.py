"""scripts/check_cleaning.py: per-file retention, the largest deletion and the Scope heading, read-only (synthetic files)."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from open_telco_rag.sources.tgpp import TPPSource

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_cleaning.py"
COVER = "3rd Generation Partnership Project; Technical Specification Group Example"
FILLER = "\n\n".join(f"Paragraph {i} of the synthetic document says the function shall act." for i in range(40))

FILES = {
    # legacy cover at the top: removed up to "The present document"; Scope heading kept
    "21001/raw.md": f"# Title\n\n{COVER}\n\n{FILLER}\n\nThe present document has been developed within the project.\n\n"
                    "# --- 1 Scope\n\nBody of the scope.\n",
    # a citation far from the start: nothing may be removed; Scope heading kept
    "21002/raw.md": f"# Title\n\n## 1 Scope\n\nThe present document specifies a thing.\n\n{FILLER}\n\n{FILLER}\n\n"
                    f"## 9 References\n\n[1] {COVER}.\n\nThe present document ends here.\n",
    # a grid table of two row groups: the table step removes one of them
    "21003/raw.md": "# Title\n\n## 1 Scope\n\nBody.\n\n+------+------+\n" + "\n".join(f"| cell {i:03d} | value {i:03d} |" for i in range(30))
                    + "\n+------+------+\n| tail row | value |\n+------+------+\n\nAfter the table.\n",
}


def stage(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    for rel, text in FILES.items():
        path = data / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf8")
    return data


def snapshot(folder: Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def run(data: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--data-dir", str(data), "--source-type", "3gpp", *extra],
                          capture_output=True, text=True, timeout=300, check=False)


def test_the_report_has_retention_the_largest_deletion_and_the_scope_state(tmp_path):
    data = stage(tmp_path)
    out = tmp_path / "report.csv"
    result = run(data, "--output", str(out))
    assert result.returncode == 0, result.stderr
    rows = {r["file"].replace("\\", "/"): r for r in csv.DictReader(out.open(encoding="utf8"))}
    assert set(rows) == set(FILES)

    legacy = rows["21001/raw.md"]
    raw = FILES["21001/raw.md"]
    assert legacy["largest_step"] == "cover boilerplate" and int(legacy["largest_length"]) > 100
    assert legacy["scope_before"] == "True" and legacy["scope_after"] == "True"
    assert int(legacy["cleaned_chars"]) == len(TPPSource().clean(raw)) and int(legacy["raw_chars"]) == len(raw)

    citation = rows["21002/raw.md"]
    assert float(citation["retention_pct"]) >= 99.0 and citation["scope_after"] == "True"

    grid = rows["21003/raw.md"]
    assert grid["largest_step"] == "table header blocks" and int(grid["largest_length"]) > 400
    assert float(grid["retention_pct"]) < 100.0


def test_the_summary_and_worst_files_are_printed(tmp_path):
    result = run(stage(tmp_path), "--worst", "2")
    assert "3 file(s)" in result.stdout and "Scope heading lost in 0 of 3" in result.stdout
    assert result.stdout.count("raw.md") == 2


def test_nothing_is_written_without_output_and_only_the_output_with_it(tmp_path):
    data = stage(tmp_path)
    before = snapshot(tmp_path)
    run(data)
    assert snapshot(tmp_path) == before
    run(data, "--output", str(tmp_path / "out.csv"))
    after = snapshot(tmp_path)
    assert set(after) - set(before) == {"out.csv"} and {k: v for k, v in after.items() if k != "out.csv"} == before


def test_an_empty_directory_is_a_clean_error(tmp_path):
    (tmp_path / "empty").mkdir()
    result = run(tmp_path / "empty")
    assert result.returncode != 0 and "no" in (result.stderr + result.stdout)
