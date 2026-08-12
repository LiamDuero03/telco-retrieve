"""Unit tests for open_telco_rag.core.cleaning's regex-based text cleaners."""
from __future__ import annotations

from open_telco_rag.core.cleaning import (
    base_clean,
    collapse_whitespace,
    remove_copyright_lines,
    remove_standalone_page_numbers,
    remove_toc_entries,
)


# ── remove_standalone_page_numbers ────────────────────────────────────────────

def test_removes_line_containing_only_a_page_number():
    text = "Some heading\n42\nMore body text."
    result = remove_standalone_page_numbers(text)
    assert "42" not in result.split("\n")
    assert "Some heading" in result
    assert "More body text." in result


def test_leaves_digits_embedded_in_normal_text_alone():
    text = "TS 23.501 defines 5G System architecture."
    assert remove_standalone_page_numbers(text) == text


def test_leaves_a_line_with_surrounding_words_around_a_number_alone():
    text = "Page 42 of 100"
    assert remove_standalone_page_numbers(text) == text


# ── collapse_whitespace ────────────────────────────────────────────────────────

def test_collapses_three_or_more_newlines_to_two():
    assert collapse_whitespace("a\n\n\n\nb") == "a\n\nb"


def test_collapses_multiple_spaces_to_one():
    assert collapse_whitespace("a    b") == "a b"


def test_strips_leading_and_trailing_whitespace():
    assert collapse_whitespace("  \n  hello  \n  ") == "hello"


def test_two_newlines_are_left_untouched():
    assert collapse_whitespace("a\n\nb") == "a\n\nb"


# ── remove_toc_entries ─────────────────────────────────────────────────────────

def test_removes_a_dotted_toc_line():
    text = "4.2.1 Some Section Title ..................... 23"
    result = remove_toc_entries(text)
    assert "Some Section Title" not in result


def test_leaves_normal_prose_with_no_dot_leader_alone():
    text = "This sentence... has an ellipsis but is not a TOC line."
    assert remove_toc_entries(text) == text


def test_removes_toc_line_but_keeps_surrounding_body_text():
    text = "Intro paragraph.\n4.2.1 Section Title ..... 23\nBody continues here."
    result = remove_toc_entries(text)
    assert "Intro paragraph." in result
    assert "Body continues here." in result
    assert "Section Title" not in result


# ── remove_copyright_lines ─────────────────────────────────────────────────────

def test_removes_a_copyright_line():
    text = "Body text.\n© 2024 3GPP Organizational Partners.\nMore body text."
    result = remove_copyright_lines(text)
    assert "3GPP Organizational Partners" not in result
    assert "Body text." in result
    assert "More body text." in result


def test_leaves_text_with_no_copyright_symbol_alone():
    text = "No copyright symbol anywhere in this line."
    assert remove_copyright_lines(text) == text


# ── base_clean (all steps together) ───────────────────────────────────────────

def test_base_clean_strips_page_numbers_toc_and_copyright_and_collapses_whitespace():
    text = (
        "1 Scope\n\n\n\n"
        "This clause defines the scope.\n"
        "42\n"
        "4.2.1 Some Heading ..................... 23\n"
        "© 2024 3GPP Organizational Partners.\n"
        "This clause    has  extra   spaces.\n"
    )
    result = base_clean(text)

    assert "42" not in result.split("\n")
    assert "Some Heading" not in result
    assert "3GPP Organizational Partners" not in result
    assert "extra   spaces" not in result  # multi-space collapsed to one
    assert "This clause defines the scope." in result
    assert "\n\n\n" not in result  # no 3+ run of newlines survives
