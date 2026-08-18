"""Unit tests for ORANSource.metadata()'s filename parsing.

Only test coverage for this source before now was indirect (via ingest
end-to-end tests, if any). metadata() is pure path parsing -- no filesystem
access -- so these instantiate directly against synthetic paths.
"""
from __future__ import annotations

from pathlib import Path

from open_telco_rag.sources.oran import ORANSource


def test_full_example_from_docstring_extracts_every_field():
    meta = ORANSource().metadata(
        Path("data/oran/O-RAN.WG1.TR.Use-Cases-Analysis-Report-R005-v19.00.pdf")
    )
    assert meta["work_group"] == "WG1"
    assert meta["doc_type"] == "TR"
    assert meta["release"] == "R005"
    assert meta["version"] == "v19.00"
    assert meta["source"] == "O-RAN.WG1.TR.Use-Cases-Analysis-Report-R005-v19.00.pdf"
    assert meta["source_type"] == "oran"


def test_ts_document_type_and_higher_wg_number():
    meta = ORANSource().metadata(Path("data/oran/O-RAN.WG11.TS.Security-Protocol-R004-v05.00.pdf"))
    assert meta["work_group"] == "WG11"
    assert meta["doc_type"] == "TS"
    assert meta["release"] == "R004"
    assert meta["version"] == "v05.00"


def test_non_numbered_work_group_tifg():
    meta = ORANSource().metadata(Path("data/oran/O-RAN.TIFG.TR.Overview-R003-v02.00.pdf"))
    assert meta["work_group"] == "TIFG"
    assert meta["doc_type"] == "TR"


def test_lowercase_filename_is_still_parsed_case_insensitively():
    meta = ORANSource().metadata(Path("data/oran/o-ran.wg1.tr.use-cases-r005-v19.00.pdf"))
    assert meta["work_group"] == "WG1"
    assert meta["doc_type"] == "TR"
    assert meta["release"] == "R005"
    assert meta["version"] == "v19.00"


def test_missing_release_and_version_leave_fields_absent_not_crashing():
    meta = ORANSource().metadata(Path("data/oran/O-RAN.WG1.TR.Overview.pdf"))
    assert meta["work_group"] == "WG1"
    assert meta["doc_type"] == "TR"
    assert "release" not in meta
    assert "version" not in meta


def test_docx_extension_handled_same_as_pdf():
    meta = ORANSource().metadata(Path("data/oran/O-RAN.WG2.TS.Non-RT-RIC-R004-v03.00.docx"))
    assert meta["work_group"] == "WG2"
    assert meta["version"] == "v03.00"
    assert meta["source"] == "O-RAN.WG2.TS.Non-RT-RIC-R004-v03.00.docx"


def test_completely_unrecognized_filename_falls_back_gracefully():
    meta = ORANSource().metadata(Path("data/oran/README.pdf"))
    assert "work_group" not in meta
    assert "doc_type" not in meta
    assert "release" not in meta
    assert "version" not in meta
    assert meta["source"] == "README.pdf"
