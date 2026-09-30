"""TPPSource.clean(): cleanup must not delete text between unrelated markers (synthetic text only)."""
from __future__ import annotations

import re

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
