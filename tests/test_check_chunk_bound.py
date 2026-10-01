"""scripts/check_chunk_bound.py reports chunk sizes and retention without changing anything."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

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
    data.mkdir(parents=True)
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
                    "--bound", "128", "--margin", "2", "--tokenizers", "cl100k", "tiktoken:o200k_base", "--min-length", "1")
    assert unbounded["tokenizers"][0]["over_bound"] > 0 and unbounded["tokenizers"][0]["max"] > 126
    assert [r["tokenizer"] for r in unbounded["tokenizers"]] == ["cl100k", "tiktoken:o200k_base"]


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
    report = run("--data-dir", str(data), "--source-type", "3gpp", "--tokenizers", "tiktoken:no_such_encoding", "cl100k", "--min-length", "1")
    assert "error" in report["tokenizers"][0] and report["tokenizers"][1]["chunks"] > 0
