"""Report what the cleaning step removes from every staged file. Read-only: it writes only to --output.

For each file under --data-dir (the extensions of --source-type) it runs the same loader and clean pipeline as
ingest, step by step, and reports: raw characters, cleaned characters, retention, the single largest deletion
(which step removed it, where in that step's input it starts, how many characters), and whether a Scope heading
existed before and after cleaning. A summary and the files that lost the most come last.

The largest deletion is found by comparing a step's input and output line by line, so its length is the number
of characters removed in the largest changed run of lines, and its start is a character offset into the text
that step received. Sources without a clean_steps() method are reported as one step, "clean".

Usage:
    python scripts/check_cleaning.py --data-dir data/3gpp/GSMA-marker/Rel-19 --source-type 3gpp --output cleaning_report.csv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import re
from pathlib import Path

from open_telco_rag.core import cleaning
from open_telco_rag.sources.tgpp import _SCOPE_HEADING

SCOPE = re.compile(_SCOPE_HEADING, re.MULTILINE)
BASE_STEPS = (("page-number lines", cleaning.remove_standalone_page_numbers), ("copyright lines", cleaning.remove_copyright_lines),
              ("TOC-style lines", cleaning.remove_toc_entries), ("whitespace collapse", cleaning.collapse_whitespace))


def largest_deletion(before: str, after: str) -> tuple[int, int]:
    """(start offset in `before`, characters removed) of the largest run of lines that the step shortened"""
    if before == after:
        return 0, 0
    a, b = before.split("\n"), after.split("\n")
    offsets = [0]
    for line in a:
        offsets.append(offsets[-1] + len(line) + 1)
    best = (0, 0)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("delete", "replace"):
            removed = (offsets[i2] - offsets[i1]) - sum(len(line) + 1 for line in b[j1:j2])
            if removed > best[1]:
                best = (offsets[i1], removed)
    return best


def pipeline(source) -> list[tuple[str, object]]:
    """(name, function) steps that together equal source.clean() up to the final strip"""
    if hasattr(source, "clean_steps"):
        return [*BASE_STEPS, *source.clean_steps()]
    return [("clean", source.clean)]


def analyse(source, path: Path, rel: str) -> dict:
    raw = source.load(path)
    text, worst = raw, ("", 0, 0)
    for name, step in pipeline(source):
        out = step(text)
        start, length = largest_deletion(text, out)
        if length > worst[2]:
            worst = (name, start, length)
        text = out
    cleaned = text.strip()
    if cleaned != source.clean(raw):
        raise RuntimeError(f"{rel}: the step-by-step pipeline does not reproduce clean(); the script is out of date")
    return {"file": rel, "raw_chars": len(raw), "cleaned_chars": len(cleaned),
            "retention_pct": round(100 * len(cleaned) / len(raw), 2) if raw else 100.0,
            "largest_step": worst[0], "largest_start": worst[1], "largest_length": worst[2],
            "scope_before": bool(SCOPE.search(raw)), "scope_after": bool(SCOPE.search(cleaned))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--source-type", default="3gpp")
    p.add_argument("--output", type=Path, default=None, help="write the per-file table here as CSV (the only file written)")
    p.add_argument("--worst", type=int, default=10, help="how many of the lowest-retention files to list (default 10)")
    args = p.parse_args()

    from cli.main import _get_source
    source = _get_source(args.source_type, use_marker=False)
    files = sorted(f for ext in source.supported_extensions() for f in args.data_dir.rglob(f"*{ext}"))
    if not files:
        raise SystemExit(f"error: no {source.supported_extensions()} files under {args.data_dir}")
    rows = [analyse(source, f, str(f.relative_to(args.data_dir))) for f in files]

    if args.output:
        with open(args.output, "w", newline="", encoding="utf8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    kept = sorted(r["retention_pct"] for r in rows)
    lost_scope = [r for r in rows if r["scope_before"] and not r["scope_after"]]
    raw_total, cleaned_total = sum(r["raw_chars"] for r in rows), sum(r["cleaned_chars"] for r in rows)
    print(f"{len(rows)} file(s): {raw_total:,} raw characters, {cleaned_total:,} after cleaning ({100 * cleaned_total / raw_total:.1f}%)")
    print(f"retention per file: min {kept[0]:.1f}%, median {kept[len(kept) // 2]:.1f}%, max {kept[-1]:.1f}%; "
          f"files under 80%: {sum(k < 80 for k in kept)}; Scope heading lost in {len(lost_scope)} of "
          f"{sum(r['scope_before'] for r in rows)} that had one")
    print(f"\n{'file':40} {'raw':>10} {'cleaned':>10} {'kept':>7}  {'largest deletion (step, start, characters)':50} scope")
    for r in sorted(rows, key=lambda r: r["retention_pct"])[:args.worst]:
        scope = ("kept" if r["scope_after"] else "LOST") if r["scope_before"] else "-"
        print(f"{r['file'][-40:]:40} {r['raw_chars']:>10,} {r['cleaned_chars']:>10,} {r['retention_pct']:>6.1f}%  "
              f"{r['largest_step']} @ {r['largest_start']:,}, {r['largest_length']:,}".ljust(50) + f"  {scope}")


if __name__ == "__main__":
    main()
