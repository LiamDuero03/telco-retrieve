"""scripts/check_chunk_bound.py reports chunk sizes and retention without changing anything."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from open_telco_rag.core.pipeline import ingest
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore
from tests.test_retrieval_methods import BagOfWordsEmbedder

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_chunk_bound.py"
TABLE = "| a | b |\n|---|---|\n" + "\n".join(f"| row_{i} | value_{i} |" for i in range(80))
DOC = ("# Spec\n\n## 1 Scope\n\n" + " ".join(f"Sentence {i} says the AMF shall act." for i in range(60))
       + "\n\n" + TABLE + "\n\n" + "A1B2C3D4" * 120 + "\n")


def run(*args: str) -> dict:
    result = subprocess.run([sys.executable, str(SCRIPT), "--json", *args], capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def snapshot(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


def stage(tmp_path: Path) -> Path:
    data = tmp_path / "data" / "Rel-18"
    data.mkdir(parents=True, exist_ok=True)
    (data / "28312-i11.md").write_text(DOC, encoding="utf8")
    return tmp_path / "data"


def test_staged_mode_reports_sizes_and_full_retention_with_the_bound(tmp_path):
    data = stage(tmp_path)
    before = snapshot(data)
    bounded = run("--data-dir", str(data), "--source-type", "3gpp", "--strategy", "text_baseline", "--target-tokens", "400",
                  "--bound-tokens", "128", "--bound", "128", "--margin", "2", "--tokenizers", "cl100k", "--min-length", "1")
    assert bounded["limit"] == 126
    result = bounded["tokenizers"][0]
    assert result["max"] <= 126 and result["over_bound"] == 0 and result["chunks"] > 3
    assert bounded["text_retained"] >= 0.99
    assert snapshot(data) == before


def test_staged_mode_without_the_bound_counts_chunks_over_it(tmp_path):
    data = stage(tmp_path)
    unbounded = run("--data-dir", str(data), "--source-type", "3gpp", "--strategy", "text_baseline", "--target-tokens", "400",
                    "--bound", "128", "--margin", "2", "--tokenizers", "cl100k", "tiktoken:cl100k_base", "--min-length", "1")
    assert unbounded["tokenizers"][0]["over_bound"] > 0 and unbounded["tokenizers"][0]["max"] > 126
    assert [r["tokenizer"] for r in unbounded["tokenizers"]] == ["cl100k", "tiktoken:cl100k_base"]


def test_collection_mode_reads_the_database_without_changing_it(tmp_path):
    data = stage(tmp_path)
    store = ChromaStore(path=tmp_path / "db", collection_name="coll", metadata={"hnsw:space": "cosine"})
    ingest(source=TPPSource(), store=store, embedder=BagOfWordsEmbedder(), data_dir=str(data), resume=False,
           log_dir=str(tmp_path / "logs"), strategy="text_baseline", chunk_size=300, overlap=0, min_length=10)
    stored = store.count()
    del store
    before = snapshot(tmp_path / "db")

    report = run("--collection", "coll", "--db-path", str(tmp_path / "db"), "--tokenizers", "cl100k",
                 "--bound", "64", "--margin", "2", "--data-dir", str(data), "--source-type", "3gpp")
    assert report["chunks"] == stored
    assert report["text_retained"] is not None and 0.95 <= report["text_retained"] <= 1.02
    assert snapshot(tmp_path / "db") == before, "the database folder is byte-for-byte unchanged"


def test_an_unknown_collection_is_a_clean_error(tmp_path):
    data = stage(tmp_path)
    store = ChromaStore(path=tmp_path / "db", collection_name="coll", metadata={"hnsw:space": "cosine"})
    ingest(source=TPPSource(), store=store, embedder=BagOfWordsEmbedder(), data_dir=str(data), resume=False,
           log_dir=str(tmp_path / "logs"), strategy="text_baseline", chunk_size=300, overlap=0, min_length=10)
    del store
    result = subprocess.run([sys.executable, str(SCRIPT), "--collection", "nope", "--db-path", str(tmp_path / "db")],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode != 0 and "not found" in result.stderr + result.stdout


def test_an_unavailable_tokenizer_is_reported_and_does_not_hide_the_others(tmp_path):
    data = stage(tmp_path)
    report = run("--data-dir", str(data), "--source-type", "3gpp", "--tokenizers", "tiktoken:no_such_encoding", "--min-length", "1")
    assert report["tokenizers"][0]["chunks"] > 0 and "error" in report["tokenizers"][1]


STAGED = ["--data-dir", "{data}", "--source-type", "3gpp", "--strategy", "text_baseline", "--target-tokens", "400", "--min-length", "1"]


def staged(tmp_path, *extra):
    data = stage(tmp_path)
    return run(*[a.replace("{data}", str(data)) for a in STAGED], *extra)


def test_the_ratio_percentiles_and_outliers_are_reported_per_tokenizer(tmp_path):
    report = staged(tmp_path, "--tokenizers", "tiktoken:cl100k_base", "--outliers", "5", "--bound", "128", "--margin", "2")
    other = report["tokenizers"][1]
    ratio = other["ratio"]
    assert ratio["p50"] <= ratio["p99"] <= ratio["p99.9"] <= ratio["max"] and ratio["max"] == 1.0
    assert 0 < len(other["outliers"]) <= 5
    assert [o["tokens"] for o in other["outliers"]] == sorted((o["tokens"] for o in other["outliers"]), reverse=True)
    assert {"spec", "cl100k", "tokens", "kind", "text"} <= set(other["outliers"][0])
    assert all(len(o["text"]) <= 60 for o in other["outliers"])


def test_the_limit_override_and_its_source_are_reported(tmp_path):
    report = staged(tmp_path, "--tokenizers", "tiktoken:cl100k_base", "--limit", "tiktoken:cl100k_base=40", "--margin", "2")
    other = report["tokenizers"][1]
    assert other["limit"] == 40 and other["limit_source"] == "--limit" and other["over_limit"] > 0


def test_text_retained_is_reported_for_the_bound_pass_and_the_min_length_filter_separately(tmp_path):
    report = staged(tmp_path, "--bound-tokens", "64", "--bound", "64")
    stages = report["stages"]
    assert stages["strategy"] <= stages["after_bound_pass"] + 1e-9 and stages["after_bound_pass"] >= 0.99
    assert stages["after_min_length"] <= stages["after_bound_pass"] + 1e-9
    assert stages["dropped_by_min_length_chars"] >= 0


def test_calibration_finds_the_ceiling_that_leaves_nothing_over_the_limit(tmp_path):
    report = staged(tmp_path, "--tokenizers", "tiktoken:cl100k_base", "--calibrate", "--calibrate-limits", "64", "--margin", "2")
    per = report["calibration"]["64"]
    assert per["tiktoken:cl100k_base"]["zero"] == 62 and per["all"]["zero"] == 62
    assert per["tiktoken:cl100k_base"]["tolerance"] == 62 and not report["calibration_is_estimate"]


def test_a_collection_gets_an_estimate_not_a_calibration_run(tmp_path):
    data = stage(tmp_path)
    store = ChromaStore(path=tmp_path / "db", collection_name="coll", metadata={"hnsw:space": "cosine"})
    ingest(source=TPPSource(), store=store, embedder=BagOfWordsEmbedder(), data_dir=str(data), resume=False,
           log_dir=str(tmp_path / "logs"), strategy="text_baseline", chunk_size=300, overlap=0, min_length=10)
    del store
    report = run("--collection", "coll", "--db-path", str(tmp_path / "db"), "--tokenizers", "tiktoken:cl100k_base",
                 "--calibrate", "--calibrate-limits", "64")
    assert report["calibration_is_estimate"] and report["calibration"]["64"]["tiktoken:cl100k_base"]["zero"] == 62
    assert report["stages"] is None


@pytest.mark.skipif(not os.environ.get("RUN_REAL_TOKENIZER_TESTS"),
                    reason="needs network to fetch a tokenizer.json (set RUN_REAL_TOKENIZER_TESTS=1)")
def test_the_calibrated_ceiling_really_leaves_no_chunk_over_the_limit_for_a_real_tokenizer(tmp_path):
    common = ["--tokenizers", "minilm", "--margin", "2"]
    ceiling = staged(tmp_path, *common, "--calibrate", "--calibrate-limits", "96")["calibration"]["96"]["minilm"]["zero"]
    assert ceiling and ceiling <= 94
    check = staged(tmp_path, *common, "--limit", "minilm=96", "--bound-tokens", str(ceiling + 2))
    assert check["tokenizers"][1]["over_limit"] == 0
