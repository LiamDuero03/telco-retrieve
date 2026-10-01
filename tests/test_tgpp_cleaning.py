"""TPPSource.clean(): cleanup must not delete text between unrelated markers (synthetic text only)."""
from __future__ import annotations

import re

import pytest

from open_telco_rag.sources.tgpp import TPPSource

clean = TPPSource().clean


def test_pandoc_superscript_is_still_removed():
    assert clean("the area is 5 m^2^ in total") == "the area is 5 m in total"


def test_latex_exponents_on_one_line_are_untouched():
    text = "a target error rate of $10^{-5}$ and $10^{-6}$ is used"
    assert clean(text) == text


def test_stray_carets_on_distant_lines_delete_nothing():
    filler = "\n\n".join(f"### 5.{i} Clause {i}\n\nBody text of clause {i}." for i in range(50))
    text = f"first $10^{{-5}}$ rate\n\n{filler}\n\nlast $2^{{3}}$ value"
    assert clean(text) == text.strip()


def test_a_caret_pair_with_spaces_between_is_not_a_superscript():
    text = "x^ and so on until y^"
    assert clean(text) == text


def test_the_legacy_cover_is_still_removed():
    text = ("3rd Generation Partnership Project; Technical Specification Group Example\n\nKeywords: x\n\n"
            "The present document has been developed within the project.\n\n# 1 Scope\n\nBody of the scope.")
    out = clean(text)
    assert out.startswith("The present document has been developed")
    assert "Keywords" not in out and "# 1 Scope" in out and "Body of the scope." in out


def test_the_scope_heading_survives_a_cover_without_the_present_document():
    text = ("3rd Generation Partnership Project; Technical Specification Group Example\n\n## Contents\n\nForeword text.\n\n"
            "---\n\n## 1 Scope\n\nThe present document provides an overview.\n\n## 2 References\n\nRefs.")
    out = clean(text)
    assert out == text.strip(), "nothing is removed when the Scope heading comes before 'The present document'"


def test_a_cover_with_no_present_document_at_all_removes_nothing():
    text = "3rd Generation Partnership Project; cover only\n\nMore cover lines.\n\n# 1 Scope\n\nScope body."
    assert clean(text) == text


def test_the_scope_heading_without_hashes_also_bounds_the_cover():
    text = "3rd Generation Partnership Project cover\n\n1 Scope\n\nThe present document covers x."
    assert clean(text) == text


def test_width_attribute_removal_stays_on_one_line():
    text = "an opening brace { in a sentence\n\nand later an image{width=5in} attribute"
    out = clean(text)
    assert "an opening brace { in a sentence" in out and "{width" not in out


def test_a_generated_document_keeps_almost_all_of_its_text():
    parts = ["3rd Generation Partnership Project; Technical Specification Group Synthetic", "", "Synthetic cover text.", "", "---", "", "# 1 Scope", "",
             "The present document specifies a synthetic interface.", ""]
    for i in range(1, 120):
        parts += [f"## {i + 1} Clause {i}", "", f"Text of clause {i} uses a rate of $10^{{-{i % 9 + 1}}}$ and a window of $2^{{{i % 5}}}$.", "",
                  "Every statement in this generated clause is ordinary prose that must survive cleaning.", ""]
    raw = "\n".join(parts)
    out = clean(raw)
    assert len(out) >= 0.95 * len(raw)
    assert len(re.findall(r"^## ", out, re.MULTILINE)) == 119 and "# 1 Scope" in out


def test_adjacent_latex_maths_without_a_space_is_untouched():
    text = "a $x^{2}$$y^{3}$ b"
    assert clean(text) == text


# ── the cover match belongs to the start of the document and stops at every form of the Scope heading ──────────

COVER = "3rd Generation Partnership Project; Technical Specification Group Example"


def _body(n: int, label: str = "Body") -> str:
    return "\n\n".join(f"{label} paragraph {i} of the synthetic document says the function shall act." for i in range(n))


def test_a_citation_deep_in_the_document_does_not_start_a_cover_match():
    text = (f"# Title\n\n## 1 Scope\n\nThe present document specifies things.\n\n{_body(120)}\n\n"
            f"## 2 References\n\n[5] {COVER}, Release 99.\n\n{_body(150, 'Later')}\n\n"
            "## 9 End\n\nThe present document ends here.")
    assert clean(text) == text


def test_a_cover_phrase_far_from_the_start_is_not_a_cover_even_without_a_scope_heading():
    text = f"{_body(100)}\n\nSee {COVER} for details.\n\n{_body(100, 'More')}\n\nThe present document is informative."
    assert clean(text) == text


def test_a_cover_whose_present_document_is_too_far_away_removes_nothing():
    text = f"{COVER}\n\n{_body(400, 'Contents')}\n\nThe present document follows."
    assert clean(text) == text


@pytest.mark.parametrize("heading", ["# --- 1 Scope", "## **1 Scope**", "1 Scope", "# 1 Scope", "## ---- **1 Scope**",
                                     "### ***1 Scope***", "# Scope", "# --- Scope"])
def test_every_form_of_the_scope_heading_stops_the_cover(heading):
    text = f"{COVER}\n\nForeword text of the cover.\n\n{heading}\n\nThe present document covers the synthetic subject."
    out = clean(text)
    assert heading in out and "The present document covers the synthetic subject." in out and "Foreword text" in out


def test_the_legacy_cover_with_a_scope_heading_after_it_is_still_removed():
    text = (f"# Title\n\n{COVER}\n\nKeywords: x\n\nThe present document has been developed within the project.\n\n"
            "# --- 1 Scope\n\nBody of the scope.")
    out = clean(text)
    assert "Keywords" not in out and out.startswith("# Title") and "# --- 1 Scope" in out and "Body of the scope." in out


def test_a_generated_long_document_with_a_late_citation_keeps_its_text_and_scope_heading():
    parts = ["# Title", "", COVER, "", "Cover text.", "", "The present document has been developed within the project.", "",
             "# --- 1 Scope", "", "The present document specifies a synthetic interface.", ""]
    for i in range(1, 300):
        parts += [f"## {i + 1} Clause {i}", "", f"Text of clause {i}, ordinary prose that must survive cleaning.", ""]
    parts += ["## 301 References", "", f"[1] {COVER}: a citation.", "", "The present document ends here."]
    raw = "\n".join(parts)
    out = clean(raw)
    assert len(out) >= 0.95 * len(raw) and "# --- 1 Scope" in out and "[1] 3rd Generation Partnership Project" in out
