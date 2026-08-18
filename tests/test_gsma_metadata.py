"""Unit tests for GSMASource.metadata()'s filename parsing.

metadata() is pure path parsing -- no filesystem access.
"""
from __future__ import annotations

from pathlib import Path

from open_telco_rag.sources.gsma import GSMASource


def test_ig_document_from_docstring():
    meta = GSMASource().metadata(Path("data/gsma/IG.01 v1.0.docx"))
    assert meta["group"] == "IG"
    assert meta["doc_num"] == "01"
    assert meta["version"] == "v1.0"
    assert meta["topic"] == "Intelligence Group"


def test_opg_document_with_title_and_dash_separator():
    meta = GSMASource().metadata(Path("data/gsma/OPG.10 v2.0 Open Gateway.docx"))
    assert meta["group"] == "OPG"
    assert meta["doc_num"] == "10"
    assert meta["version"] == "v2.0"
    assert meta["title"] == "Open Gateway"
    assert meta["topic"] == "Operator Platform Group"


def test_sgp_document_three_part_version():
    meta = GSMASource().metadata(Path("data/gsma/SGP.21 v2.5.1.pdf"))
    assert meta["group"] == "SGP"
    assert meta["doc_num"] == "21"
    assert meta["version"] == "v2.5.1"
    assert meta["topic"] == "eSIM / SIM specifications"


def test_unmapped_group_gets_generic_topic():
    meta = GSMASource().metadata(Path("data/gsma/XYZ.05 v1.0.pdf"))
    assert meta["group"] == "XYZ"
    assert meta["topic"] == "GSMA Working Group"


def test_folder_is_recorded_from_parent_directory():
    meta = GSMASource().metadata(Path("data/gsma/SGP/SGP.21 v2.5.pdf"))
    assert meta["folder"] == "SGP"


def test_filename_without_group_number_pattern_has_no_group_or_topic():
    meta = GSMASource().metadata(Path("data/gsma/README.pdf"))
    assert "group" not in meta
    assert "doc_num" not in meta
    assert "topic" not in meta
    assert meta["source"] == "README.pdf"


def test_version_without_trailing_title_has_no_title_field():
    meta = GSMASource().metadata(Path("data/gsma/IG.01 v1.0.docx"))
    assert "title" not in meta
