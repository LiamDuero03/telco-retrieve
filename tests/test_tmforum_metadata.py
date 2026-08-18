"""Unit tests for TMForumSource.metadata() and its _classify_repo() helper.

_classify_repo walks path components from the leaf upward looking for a
repo-type prefix (CTK_/RI_/Postman_/TMF.../TMFC.../IG...), falling back to a
handful of known non-API repo markers. metadata() is otherwise pure path
parsing -- no filesystem access.
"""
from __future__ import annotations

from pathlib import Path

from open_telco_rag.sources.tmforum import TMForumSource, _classify_repo


def test_ctk_repo_classified_with_tmf_number_and_api_name():
    result = _classify_repo(Path("data/tmforum/CTK_TMF622_ProductOrdering/tests/run.robot"))
    assert result["repo_type"] == "ctk"
    assert result["tmf_number"] == "TMF622"
    assert result["api_name"] == "ProductOrdering"


def test_ri_repo_classified_with_tmf_number_and_api_name():
    result = _classify_repo(Path("data/tmforum/RI-TMF620-Product_Catalog/src/main.js"))
    assert result["repo_type"] == "ri"
    assert result["tmf_number"] == "TMF620"
    assert result["api_name"] == "Product Catalog"


def test_postman_repo_classified_with_tmf_number_but_no_api_name():
    result = _classify_repo(Path("data/tmforum/Postman_TMF622_collection/spec.json"))
    assert result["repo_type"] == "postman"
    assert result["tmf_number"] == "TMF622"
    assert "api_name" not in result


def test_bare_tmf_repo_classified_as_api():
    result = _classify_repo(Path("data/tmforum/TMF622-Product_Ordering/openapi.yaml"))
    assert result["repo_type"] == "api"
    assert result["tmf_number"] == "TMF622"
    assert result["api_name"] == "Product Ordering"


def test_oda_component_repo_classified_with_tmfc_number():
    result = _classify_repo(Path("data/tmforum/TMFC001_PartyManagement/component.json"))
    assert result["repo_type"] == "oda_component"
    assert result["tmfc_number"] == "TMFC001"
    assert result["component"] == "PartyManagement"


def test_ig_repo_classified_with_ig_number():
    result = _classify_repo(Path("data/tmforum/IG1170_Guideline/doc.pdf"))
    assert result["repo_type"] == "ig"
    assert result["ig_number"] == "IG1170"


def test_known_non_api_marker_falls_back_to_oda():
    result = _classify_repo(Path("data/tmforum/ODA-Ready-for-publication/canvas.json"))
    assert result["repo_type"] == "oda"


def test_schemas_folder_classified_as_schema_repo_type():
    result = _classify_repo(Path("data/tmforum/some_repo/schemas/Entity.schema.json"))
    assert result["repo_type"] == "schema"


def test_completely_unrecognized_path_returns_empty_dict():
    result = _classify_repo(Path("data/tmforum/random_folder/file.txt"))
    assert result == {}


def test_metadata_merges_classify_repo_fields_and_adds_doc_type_for_docx():
    meta = TMForumSource().metadata(Path("data/tmforum/TMF622-Product_Ordering/spec_v4.0.0.docx"))
    assert meta["repo_type"] == "api"
    assert meta["tmf_number"] == "TMF622"
    assert meta["version"] == "v4.0.0"
    assert meta["doc_type"] == "Document"
    assert meta["folder"] == "TMF622-Product_Ordering"


def test_metadata_marks_json_schema_files_as_data_schema():
    meta = TMForumSource().metadata(Path("data/tmforum/some_repo/schemas/Entity.schema.json"))
    assert meta["doc_type"] == "Data Schema"
    assert meta["entity"] == "Entity"


def test_metadata_marks_plain_json_as_openapi_specification():
    meta = TMForumSource().metadata(Path("data/tmforum/TMF622-Product_Ordering/openapi.json"))
    assert meta["doc_type"] == "OpenAPI Specification"


def test_metadata_md_file_gets_readme_doc_type():
    meta = TMForumSource().metadata(Path("data/tmforum/TMF622-Product_Ordering/README.md"))
    assert meta["doc_type"] == "README"


def test_metadata_tmf_number_recovered_from_filename_when_not_in_path():
    """_classify_repo finds nothing walking the path, but the filename itself
    carries a TMF number -- the standalone regex fallback should still catch it."""
    meta = TMForumSource().metadata(Path("data/tmforum/misc/notes_TMF622.md"))
    assert meta["tmf_number"] == "TMF622"


def test_ig_variant_letter_is_captured_separately():
    result = _classify_repo(Path("data/tmforum/IG1170A_Guideline/doc.pdf"))
    assert result["repo_type"] == "ig"
    assert result["ig_number"] == "IG1170"
