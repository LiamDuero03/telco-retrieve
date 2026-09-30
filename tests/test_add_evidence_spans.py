"""Tests for scripts/add_evidence_spans.py: runs entirely from local files (no
HF or network use, checked structurally and by an end-to-end run), never
overwrites its input and never drops a question."""
from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "add_evidence_spans.py"
_spec = importlib.util.spec_from_file_location("scripts_add_evidence_spans", SCRIPT)
aes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aes)


def test_module_has_no_huggingface_hf_import_or_call():
    """structural guarantee: no huggingface_hub import, no push_/pull_ call,
    no HF_REPO reference anywhere in the source"""
    source = inspect.getsource(aes)
    forbidden = ["huggingface_hub", "push_collection_to_hf", "pull_collection_from_hf", "HF_REPO"]
    for token in forbidden:
        assert token not in source, f"add_evidence_spans.py must stay HF-free (found {token!r})"


# ── _locate_span / _whitespace_pattern ──────────────────────────────────────

def test_locate_span_exact_match():
    text = "Some preamble. The E2 interface connects the RIC. Some trailer."
    span = aes._locate_span(text, "The E2 interface connects the RIC.")
    assert span is not None
    start, end = span
    assert text[start:end] == "The E2 interface connects the RIC."


def test_locate_span_whitespace_normalized_fallback():
    cleaned = "The E2 interface\nconnects   the RIC in real time."
    needle = "The E2 interface connects the RIC"
    span = aes._locate_span(cleaned, needle)
    assert span is not None


def test_locate_span_no_match_returns_none():
    assert aes._locate_span("Completely unrelated text.", "Something that never appears.") is None


def test_locate_span_empty_needle_returns_none():
    assert aes._locate_span("Some text.", "") is None
    assert aes._locate_span("Some text.", "   ") is None


# ── main() end-to-end (local files only, no network) ────────────────────────

def test_main_end_to_end_matches_and_writes_never_dropping_unmatched(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "38_series" / "38401"
    data_dir.mkdir(parents=True)
    (data_dir / "raw.md").write_text(
        "# Scope\n\nThis document describes the E2 interface used by the Near-RT RIC.",
        encoding="utf-8",
    )

    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps({"questions": [
        {
            "question": "What does the E2 interface do?", "corpus": "3gpp",
            "chunk_id": "38_series/38401/raw.md::minilm::0",
            "source_text": "This document describes the E2 interface used by the Near-RT RIC.",
        },
        {"question": "No chunk_id/source_text at all", "corpus": "3gpp"},
        {
            "question": "Source text that will never match", "corpus": "3gpp",
            "chunk_id": "38_series/38401/raw.md::minilm::1",
            "source_text": "This text does not appear anywhere in the document.",
        },
    ]}), encoding="utf-8")

    output_file = tmp_path / "questions_with_spans.json"
    monkeypatch.setattr(sys, "argv", [
        "add_evidence_spans.py",
        "--questions-file", str(questions_file),
        "--data-dir", str(tmp_path / "data"),
        "--source-type", "3gpp",
        "--output", str(output_file),
    ])

    aes.main()

    assert output_file.exists()
    result = json.loads(output_file.read_text(encoding="utf-8"))
    questions = result["questions"]
    assert len(questions) == 3  # nothing dropped, including the unmatched/not-attempted ones

    matched = questions[0]
    assert matched["evidence_char_start"] is not None
    assert matched["evidence_char_end"] is not None
    assert matched["evidence_doc_id"] == "38_series\\38401\\raw.md" or matched["evidence_doc_id"] == "38_series/38401/raw.md"

    not_attempted = questions[1]
    assert "evidence_char_start" not in not_attempted

    unmatched = questions[2]
    assert "evidence_char_start" not in unmatched


def test_main_reports_counts_without_crashing_on_a_missing_source_file(tmp_path, monkeypatch, capsys):
    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps({"questions": [
        {
            "question": "Q", "corpus": "3gpp",
            "chunk_id": "38_series/missing/raw.md::minilm::0",
            "source_text": "irrelevant",
        },
    ]}), encoding="utf-8")

    output_file = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "add_evidence_spans.py",
        "--questions-file", str(questions_file),
        "--data-dir", str(tmp_path / "data"),  # never created -- file genuinely missing
        "--source-type", "3gpp",
        "--output", str(output_file),
    ])

    aes.main()

    out = capsys.readouterr().out
    assert "source file missing:  1" in out
    assert output_file.exists()


# ── promoted-script checks ────────────────────────────────────────────────────

def test_help_runs_without_network():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout and "--questions-file" in result.stdout


def test_no_hardcoded_paths_keys_or_path_hacks():
    text = SCRIPT.read_text(encoding="utf8")
    assert "liam/" not in text and "sys.path" not in text and "Rel-19" not in text


def test_refuses_to_overwrite_its_input(tmp_path, monkeypatch):
    questions_file = tmp_path / "q.json"
    questions_file.write_text(json.dumps({"questions": [{"question": "Q", "corpus": "3gpp"}]}),
                              encoding="utf-8")
    before = questions_file.read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["add_evidence_spans.py", "--questions-file", str(questions_file),
                                      "--data-dir", str(tmp_path), "--source-type", "3gpp",
                                      "--output", str(questions_file)])
    with pytest.raises(SystemExit, match="must differ"):
        aes.main()
    assert questions_file.read_text(encoding="utf-8") == before


def test_rejects_a_question_file_that_is_not_an_object(tmp_path, monkeypatch):
    questions_file = tmp_path / "q.json"
    questions_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["add_evidence_spans.py", "--questions-file", str(questions_file),
                                      "--data-dir", str(tmp_path), "--source-type", "3gpp",
                                      "--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit, match="questions"):
        aes.main()


def test_output_keeps_non_ascii_text_readable(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "38_series" / "38401"
    data_dir.mkdir(parents=True)
    (data_dir / "raw.md").write_text("Le r\u00e9seau d'acc\u00e8s radio.", encoding="utf-8")
    questions_file = tmp_path / "q.json"
    questions_file.write_text(json.dumps({"questions": [
        {"question": "Qu'est-ce ?", "corpus": "3gpp", "chunk_id": "38_series/38401/raw.md::m::0",
         "source_text": "Le r\u00e9seau d'acc\u00e8s radio."}]}), encoding="utf-8")
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["add_evidence_spans.py", "--questions-file", str(questions_file),
                                      "--data-dir", str(tmp_path / "data"), "--source-type", "3gpp",
                                      "--output", str(out)])
    aes.main()
    raw = out.read_text(encoding="utf-8")
    assert "r\u00e9seau" in raw and "\\u00e9" not in raw
