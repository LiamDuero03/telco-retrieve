"""Unit tests for ETSISource.metadata().

Covers the folder->(doc_type, group) lookup table, its three fallback
heuristics (prefix/suffix pattern, keyword match, generic "etsi" folder
stem parsing), and doc_num/version extraction from PDF filenames. The
.yaml/.yml branch reads file content, so those cases use tmp_path.
"""
from __future__ import annotations

from pathlib import Path

from open_telco_rag.sources.etsi import ETSISource


def test_known_folder_maps_directly_to_doc_type_and_group():
    meta = ETSISource().metadata(Path("data/etsi/gs_NFV/nfv006v040101p.pdf"))
    assert meta["doc_type"] == "GS"
    assert meta["group"] == "NFV"
    assert meta["group_title"] == "Network Functions Virtualisation"
    assert meta["folder"] == "gs_NFV"


def test_hyphenated_group_folder_from_map():
    meta = ETSISource().metadata(Path("data/etsi/gs_NFV_SOL/sol006v040101p.pdf"))
    assert meta["doc_type"] == "GS"
    assert meta["group"] == "NFV-SOL"
    assert meta["group_title"] == "NFV Solutions"


def test_unmapped_folder_falls_back_to_gs_prefix_heuristic():
    meta = ETSISource().metadata(Path("data/etsi/gs_SAI2/sai002v010101p.pdf"))
    assert meta["doc_type"] == "GS"
    assert meta["group"] == "SAI2"


def test_unmapped_folder_falls_back_to_gr_suffix_heuristic():
    meta = ETSISource().metadata(Path("data/etsi/experiential_gr/eni-999v010101p.pdf"))
    assert meta["doc_type"] == "GR"
    assert meta["group"] == "experiential"


def test_unmapped_folder_falls_back_to_mec_keyword_heuristic():
    meta = ETSISource().metadata(Path("data/etsi/misc_mec_stuff/spec.pdf"))
    assert meta["doc_type"] == "GS"
    assert meta["group"] == "MEC"


def test_generic_etsi_folder_parses_group_from_filename_stem():
    meta = ETSISource().metadata(Path("data/etsi/etsi/gs_zsm002v010101p.pdf"))
    assert meta["doc_type"] == "GS"
    assert meta["group"] == "ZSM"


def test_doc_num_and_version_extracted_from_pdf_filename():
    meta = ETSISource().metadata(Path("data/etsi/gs_NFV/nfv006v040101p.pdf"))
    assert meta["doc_num"] == "006"
    assert meta["version"] == "4.1.1"


def test_ts_folder_short_key_from_map():
    meta = ETSISource().metadata(Path("data/etsi/ts/123456v010203p.pdf"))
    assert meta["doc_type"] == "TS"
    assert meta["group"] == "TC"
    assert meta["group_title"] == "ETSI Technical Committee"


def test_completely_unrecognized_folder_and_filename_does_not_crash():
    meta = ETSISource().metadata(Path("data/etsi/random_stuff/whatever.pdf"))
    assert "doc_type" not in meta
    assert "group" not in meta
    assert "doc_num" not in meta
    assert "version" not in meta
    assert meta["source"] == "whatever.pdf"


def test_yaml_branch_sets_openapi_doc_type_and_reads_info_block(tmp_path):
    spec_dir = tmp_path / "gs_MEC"
    spec_dir.mkdir()
    spec_path = spec_dir / "mec-location-api.yaml"
    spec_path.write_text(
        "openapi: 3.0.0\n"
        "info:\n"
        "  title: Location API\n"
        "  version: 2.1.1\n",
        encoding="utf-8",
    )

    meta = ETSISource().metadata(spec_path)

    assert meta["doc_type"] == "OpenAPI Specification"
    assert meta["api_title"] == "Location API"
    assert meta["api_version"] == "2.1.1"
    # yaml branch returns early -- PDF-only doc_num/version regexes never run
    assert "doc_num" not in meta


def test_yaml_branch_with_missing_info_block_does_not_crash(tmp_path):
    spec_path = tmp_path / "no_info.yaml"
    spec_path.write_text("paths: {}\n", encoding="utf-8")

    meta = ETSISource().metadata(spec_path)

    assert meta["doc_type"] == "OpenAPI Specification"
    assert "api_title" not in meta
