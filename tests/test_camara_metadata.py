"""Unit tests for CAMARASource.metadata().

Unlike the other sources, the .yaml/.yml branch reads the file's content
(for api_version/api_title) -- those cases need a real file via tmp_path.
The .md/repo-name branches are pure path parsing.
"""
from __future__ import annotations

from pathlib import Path

from open_telco_rag.sources.camara import CAMARASource


def test_repo_name_is_the_first_camelcase_path_component():
    meta = CAMARASource().metadata(Path("data/camara/QualityOnDemand/documentation/api.md"))
    assert meta["repo_name"] == "QualityOnDemand"
    assert meta["api_name"] == "QualityOnDemand"


def test_known_infra_folders_are_skipped_when_hunting_for_repo_name():
    meta = CAMARASource().metadata(
        Path("data/camara/code/api_definitions/DeviceLocation/swaggers/spec.yaml")
    )
    assert meta.get("repo_name") == "DeviceLocation"


def test_readme_md_gets_api_overview_role():
    meta = CAMARASource().metadata(Path("data/camara/QualityOnDemand/README.md"))
    assert meta["file_type"] == "readme"
    assert meta["doc_role"] == "api_overview"


def test_user_story_md_gets_use_case_role():
    meta = CAMARASource().metadata(Path("data/camara/QualityOnDemand/documentation/UserStory_qod.md"))
    assert meta["file_type"] == "user_story"
    assert meta["doc_role"] == "use_case"


def test_meeting_minutes_md_gets_design_history_role():
    meta = CAMARASource().metadata(Path("data/camara/QualityOnDemand/MoM_2024_03_01.md"))
    assert meta["file_type"] == "meeting_minutes"
    assert meta["doc_role"] == "design_history"


def test_generic_md_falls_back_to_documentation_reference():
    meta = CAMARASource().metadata(Path("data/camara/QualityOnDemand/API_Design_Guidelines.md"))
    assert meta["file_type"] == "documentation"
    assert meta["doc_role"] == "reference"


def test_yaml_extracts_api_title_and_version_from_real_file(tmp_path):
    spec_path = tmp_path / "quality-on-demand.yaml"
    spec_path.write_text(
        "openapi: 3.0.3\n"
        "info:\n"
        "  title: Quality On Demand\n"
        "  version: 1.2.0\n",
        encoding="utf-8",
    )

    meta = CAMARASource().metadata(spec_path)

    assert meta["file_type"] == "openapi"
    assert meta["api_title"] == "Quality On Demand"
    assert meta["api_version"] == "1.2.0"
    assert meta["spec_name"] == "quality-on-demand"


def test_yaml_with_no_info_block_does_not_crash_and_omits_fields(tmp_path):
    spec_path = tmp_path / "empty.yaml"
    spec_path.write_text("paths: {}\n", encoding="utf-8")

    meta = CAMARASource().metadata(spec_path)

    assert meta["file_type"] == "openapi"
    assert "api_title" not in meta
    assert "api_version" not in meta


def test_malformed_yaml_does_not_crash_metadata_extraction(tmp_path):
    spec_path = tmp_path / "broken.yaml"
    spec_path.write_text("{ this: is not: valid: yaml: [", encoding="utf-8")

    meta = CAMARASource().metadata(spec_path)

    assert meta["file_type"] == "openapi"
    assert "api_title" not in meta
