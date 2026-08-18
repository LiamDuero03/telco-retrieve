"""Unit tests for TPPSource.metadata()'s handling of TSpec-LLM's file
layouts, and of release-qualified identity (canonical_id).

Regression coverage, two bugs:
1. TSpec-LLM's converter names every spec's markdown "raw.md", one per
   spec-named folder (e.g. 28312-i11/raw.md) -- metadata() used to read the
   identifier from the filename alone, so "source" and spec_number
   collapsed to a useless "raw.md" for every 3GPP document.
2. Even once fixed, the real per-document folder name is often bare
   (25401/raw.md) or part-numbered (38508-2/raw.md) -- no version at all --
   and spec_number itself never carries version/release either way. Two
   releases of the same spec (different content) were indistinguishable by
   anything downstream reads. canonical_id fixes this by folding in the
   release extracted from the "Rel-NN" ancestor directory, which is always
   available regardless of filename/folder shape.
"""
from __future__ import annotations

from pathlib import Path

from open_telco_rag.sources.tgpp import TPPSource


def test_directly_named_file_extracts_spec_number_as_before():
    meta = TPPSource().metadata(Path("data/3gpp/Rel-19/23_series/23003-j20.md"))
    assert meta["spec_number"] == "23.003"
    assert meta["version"] == "j20"
    assert meta["series"] == "23"
    assert meta["source"] == "23003-j20.md"
    assert meta["release"] == "Rel-19"
    assert meta["canonical_id"] == "23.003 (Rel-19)"


def test_generic_filename_reads_identifier_from_parent_folder():
    """Real TSpec-LLM layout: the spec-named folder holds a generic "raw.md"."""
    meta = TPPSource().metadata(
        Path("data/3gpp/Rel-18/28_series/28312-i80/raw.md")
    )
    assert meta["spec_number"] == "28.312"
    assert meta["version"] == "i80"
    assert meta["series"] == "28"
    assert meta["source"] == "28312-i80.md"
    assert meta["filename"] == "28312-i80.md"
    assert meta["canonical_id"] == "28.312 (Rel-18)"


def test_generic_filename_with_bare_spec_folder_no_version_at_all():
    """Real observed layout: the spec folder carries no version/part suffix
    at all -- e.g. 25_series/25401/raw.md. Must still extract spec_number
    and fold release into canonical_id, since the folder name alone can't."""
    meta = TPPSource().metadata(Path("data/3gpp/Rel-19/25_series/25401/raw.md"))
    assert meta["spec_number"] == "25.401"
    assert "version" not in meta
    assert "doc_part" not in meta
    assert meta["release"] == "Rel-19"
    assert meta["canonical_id"] == "25.401 (Rel-19)"


def test_generic_filename_with_multi_part_spec_folder():
    """Real observed layout: multi-part specs use a dash-digit suffix that
    is a PART number, not a version letter -- e.g. TS 38.508-2."""
    meta = TPPSource().metadata(Path("data/3gpp/Rel-19/38_series/38508-2/raw.md"))
    assert meta["spec_number"] == "38.508"
    assert meta["doc_part"] == "2"
    assert "version" not in meta
    assert meta["canonical_id"] == "38.508-2 (Rel-19)"


def test_same_spec_different_release_yields_different_canonical_id():
    """The core regression: two releases of the same spec, both with a
    bare (no-version) folder name, must not collapse to the same identity."""
    rel_17 = TPPSource().metadata(Path("data/3gpp/Rel-17/25_series/25401/raw.md"))
    rel_19 = TPPSource().metadata(Path("data/3gpp/Rel-19/25_series/25401/raw.md"))
    assert rel_17["spec_number"] == rel_19["spec_number"] == "25.401"
    assert rel_17["canonical_id"] != rel_19["canonical_id"]
    assert rel_17["canonical_id"] == "25.401 (Rel-17)"
    assert rel_19["canonical_id"] == "25.401 (Rel-19)"


def test_directory_scanned_release_wins_over_version_letter_inference():
    """A file whose version letter would infer one release, but that sits
    under a differently-numbered Rel-NN folder on disk (e.g. a draft/errata
    copy) -- the real directory is ground truth, not the inferred one."""
    meta = TPPSource().metadata(Path("data/3gpp/Rel-17/28_series/28312-i80/raw.md"))
    assert meta["release"] == "Rel-17"  # not "Rel-18", despite version letter "i"
    assert meta["canonical_id"] == "28.312 (Rel-17)"


def test_generic_filename_release_inferred_from_version_letter_when_no_rel_dir():
    meta = TPPSource().metadata(Path("data/3gpp/38_series/38300-i80/raw.md"))
    assert meta["release"] == "Rel-18"  # version letter "i" -> Rel-18


def test_generic_filename_case_insensitive():
    meta = TPPSource().metadata(Path("data/3gpp/28312-i80/RAW.MD"))
    assert meta["spec_number"] == "28.312"
    assert meta["source"] == "28312-i80.MD"


def test_generic_docx_filename_also_handled():
    meta = TPPSource().metadata(Path("data/3gpp/28312-i80/converted.docx"))
    assert meta["spec_number"] == "28.312"
    assert meta["source"] == "28312-i80.docx"


def test_non_generic_but_unparseable_filename_falls_back_gracefully():
    """A file that's neither the digit-prefixed pattern nor a known generic
    name -- no spec_number, but doesn't crash and keeps the real filename."""
    meta = TPPSource().metadata(Path("data/3gpp/28_series/README.md"))
    assert "spec_number" not in meta
    assert meta["source"] == "README.md"
    assert meta["canonical_id"] == "README"  # no spec_number/release -- falls back to the bare stem
