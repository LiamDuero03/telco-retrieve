"""Golden check: stored chunk metadata for DEFAULT ingest settings.

Runs the real per-file ingest worker on a tiny markdown file and records the
metadata that would be written to the vector store for every chunking strategy
with default (character/legacy) sizing. Generated from pristine upstream/main;
opt-in features (token sizing, spans) must not add keys to this output.

Regenerate ONLY when a default change is deliberate and approved:
    python tests/test_ingest_metadata_golden.py --write
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

GOLDEN = Path(__file__).parent / "golden" / "ingest_metadata.json"

DOC = "\n\n".join(
    f"## Section {i}\n\n"
    + "\n\n".join(
        f"Paragraph {i}.{j}: the Near-RT RIC exposes E2 and A1 interfaces to xApps for procedure "
        f"{i * 10 + j}, and the SMO configures policies over O1."
        for j in range(1, 5)
    )
    for i in range(1, 5)
)


def _run(strategy: str, **chunk_kwargs) -> list[dict]:
    from open_telco_rag.core.pipeline import _process_file_worker

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        doc = data_dir / "Rel-18" / "28312-i11.md"
        doc.parent.mkdir()
        doc.write_text(DOC, encoding="utf8")
        kwargs = {"strategy": strategy, **chunk_kwargs}
        result = _process_file_worker((doc, data_dir, "3gpp", False, kwargs, 10, "minilm"))
    assert result["status"] == "success", result["error"]
    return [
        {"id": i, "text_len": len(t), "metadata": m}
        for i, t, m in zip(result["ids"], result["chunks"], result["metadatas"])
    ]


def compute() -> dict:
    out = {}
    for strategy in ("text_baseline", "hierarchical_markdown", "parent_child"):
        out[f"{strategy}/defaults"] = _run(strategy, chunk_size=1024, overlap=200)
    out["sliding_window_tokens/defaults"] = _run(
        "sliding_window_tokens", chunk_tokens=128, overlap_tokens=16)
    return json.loads(json.dumps(out, sort_keys=True, default=str))


def _posix_ids(rows: list[dict]) -> list[dict]:
    """chunk ids start with the path relative to the data directory, written with the operating system's
    separator; the golden was generated on Windows, so compare with forward slashes on both sides"""
    return [{**row, "id": row["id"].replace("\\", "/")} for row in rows]


def test_default_ingest_metadata_matches_golden():
    expected = json.loads(GOLDEN.read_text(encoding="utf8"))
    actual = compute()
    assert actual.keys() == expected.keys()
    for key in expected:
        assert _posix_ids(actual[key]) == _posix_ids(expected[key]), f"stored metadata changed for {key}"


def test_the_comparison_does_not_depend_on_the_path_separator():
    expected = json.loads(GOLDEN.read_text(encoding="utf8"))
    as_on_linux = {key: [{**row, "id": row["id"].replace("\\", "/")} for row in rows] for key, rows in expected.items()}
    assert any("\\" in row["id"] for rows in expected.values() for row in rows), "the golden holds Windows separators"
    for key in expected:
        assert _posix_ids(as_on_linux[key]) == _posix_ids(expected[key])


if __name__ == "__main__":
    if "--write" in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(compute(), indent=1, sort_keys=True) + "\n", encoding="utf8")
        print(f"wrote {GOLDEN}")
