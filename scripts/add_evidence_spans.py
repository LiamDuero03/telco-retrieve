"""Optional: add evidence spans to a question file, enabling the span metrics
(span_hit_at_k / span_hit_in_context) in `evaluate`.

Locates each question's ground-truth chunk text inside its cleaned source
document and writes evidence_char_start/evidence_char_end/evidence_doc_id
onto a NEW question file (refuses to overwrite the input, never drops a
question that couldn't be matched).

Usage:
    python scripts/add_evidence_spans.py \
        --questions-file configs/eval_questions/3gpp.json \
        --data-dir data/3gpp \
        --source-type 3gpp \
        --output configs/eval_questions/3gpp_with_spans.json

Input: a question file of the form {"questions": [...]} whose questions carry
`chunk_id` and `source_text`.

A question is matched by:
  1. reading chunk_id's "{rel}::{model_key}::{chunk_index}" prefix to find
     its source file under --data-dir
  2. loading + cleaning that file exactly as ingestion does
     (cli.main._get_source(source_type).load()/.clean())
  3. locating source_text in the cleaned text, exact match first, then a
     whitespace-normalised regex fallback (handles re-wrapped/re-flowed
     whitespace between when the question was generated and now)

Questions with no chunk_id/source_text, an unresolvable source file, or no
match under either strategy are left in the output UNCHANGED (evidence_*
fields stay None) and counted as unmatched/not-attempted in the report --
never silently dropped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _whitespace_pattern(source_text: str) -> re.Pattern:
    parts = [re.escape(p) for p in re.split(r"\s+", source_text.strip()) if p]
    return re.compile(r"\s+".join(parts), re.DOTALL)


def _locate_span(cleaned_text: str, source_text: str) -> tuple[int, int] | None:
    if not source_text or not source_text.strip():
        return None

    idx = cleaned_text.find(source_text)
    if idx >= 0:
        return idx, idx + len(source_text)

    stripped = source_text.strip()
    idx = cleaned_text.find(stripped)
    if idx >= 0:
        return idx, idx + len(stripped)

    match = _whitespace_pattern(source_text).search(cleaned_text)
    if match:
        return match.span()

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-type", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.resolve() == args.questions_file.resolve():
        raise SystemExit("--output must differ from --questions-file (the input is never overwritten)")

    from cli.main import _get_source

    data = json.loads(args.questions_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{args.questions_file}: expected an object with a \"questions\" list")
    questions = data.get("questions", [])
    if not questions:
        print(f"No questions found in {args.questions_file}")
        return

    source = _get_source(args.source_type, use_marker=False)
    doc_cache: dict[str, str | None] = {}
    doc_status: dict[str, str] = {}  # rel -> "missing" | "load_error" | "ok"

    n_not_attempted = 0  # no chunk_id/source_text to work from
    n_file_missing = 0
    n_load_error = 0
    n_unmatched = 0
    n_matched = 0

    for q in questions:
        chunk_id = q.get("chunk_id")
        source_text = q.get("source_text")
        if not chunk_id or not source_text:
            n_not_attempted += 1
            continue

        rel = chunk_id.split("::", 1)[0]
        if rel not in doc_cache:
            file_path = args.data_dir / rel
            if not file_path.exists():
                doc_cache[rel] = None
                doc_status[rel] = "missing"
            else:
                try:
                    raw = source.load(file_path)
                    doc_cache[rel] = source.clean(raw) if raw else ""
                    doc_status[rel] = "ok"
                except Exception as e:  # noqa: BLE001
                    print(f"  [ERROR] loading {file_path}: {e}")
                    doc_cache[rel] = None
                    doc_status[rel] = "load_error"

        if doc_status[rel] == "missing":
            n_file_missing += 1
            continue
        if doc_status[rel] == "load_error":
            n_load_error += 1
            continue

        cleaned_text = doc_cache[rel]

        span = _locate_span(cleaned_text, source_text)
        if span is None:
            n_unmatched += 1
            continue

        q["evidence_char_start"] = span[0]
        q["evidence_char_end"] = span[1]
        q["evidence_doc_id"] = rel
        n_matched += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {args.output} ({len(questions)} questions)")
    print(f"  matched:              {n_matched}")
    print(f"  unmatched (searched, not found): {n_unmatched}")
    print(f"  source file missing:  {n_file_missing}")
    print(f"  source file failed to load: {n_load_error}")
    print(f"  not attempted (no chunk_id/source_text): {n_not_attempted}")
    if n_unmatched or n_file_missing or n_load_error:
        print(f"  -> {n_unmatched + n_file_missing + n_load_error} question(s) kept unchanged in "
              f"the output, not dropped -- see above for why each group couldn't be matched.")


if __name__ == "__main__":
    main()
