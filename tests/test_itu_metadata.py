"""Unit tests for ITUSource.metadata()'s filename parsing.

metadata() is pure path parsing -- no filesystem access.
"""
from __future__ import annotations

from pathlib import Path

from open_telco_rag.sources.itu import ITUSource


def test_standard_recommendation_from_docstring():
    meta = ITUSource().metadata(Path("data/itu/Y/T-REC-Y.3172-201906-I!!PDF-E.pdf"))
    assert meta["series"] == "Y"
    assert meta["rec_number"] == "Y.3172"
    assert meta["doc_type"] == "Recommendation"
    assert meta["year"] == "2019"
    assert meta["month"] == "06"
    assert meta["status"] == "In force"
    assert meta["subfolder"] == "Y"


def test_supplement_document_type():
    meta = ITUSource().metadata(Path("data/itu/Y/T-REC-Y.Sup96-202511-I!!PDF-E.pdf"))
    assert meta["series"] == "Y"
    assert meta["rec_number"] == "Y.Sup96"
    assert meta["doc_type"] == "Supplement"


def test_g_series_recommendation():
    meta = ITUSource().metadata(Path("data/itu/G/T-REC-G.709-202006-I!!PDF-E.pdf"))
    assert meta["series"] == "G"
    assert meta["rec_number"] == "G.709"
    assert meta["series_title"] == "Transmission systems and media, digital systems and networks"


def test_all_status_codes_map_correctly():
    for code, expected in [("I", "In force"), ("S", "Superseded"), ("P", "Provisional"), ("W", "Withdrawn")]:
        meta = ITUSource().metadata(Path(f"data/itu/Q/T-REC-Q.931-199805-{code}!!PDF-E.pdf"))
        assert meta["status"] == expected, f"status code {code}"


def test_subfolder_not_set_when_parent_dir_is_not_a_known_series():
    meta = ITUSource().metadata(Path("data/itu/misc/T-REC-Y.3172-201906-I!!PDF-E.pdf"))
    assert "subfolder" not in meta
    # series/rec_number still come from the filename itself, independent of folder
    assert meta["series"] == "Y"


def test_unrecognized_filename_falls_back_without_crashing():
    meta = ITUSource().metadata(Path("data/itu/Y/README.pdf"))
    assert "series" not in meta
    assert "rec_number" not in meta
    assert "year" not in meta
    assert "status" not in meta
    assert meta["source"] == "README.pdf"
