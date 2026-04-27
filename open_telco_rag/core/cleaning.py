"""Base text cleaning utilities shared across all sources."""

from __future__ import annotations

import re


def remove_standalone_page_numbers(text: str) -> str:
    return re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def remove_toc_entries(text: str) -> str:
    """Remove table of contents lines like '4.2.1 ..... 23'."""
    return re.sub(r"^.+\.{3,}\s*\d+\s*$", "", text, flags=re.MULTILINE)


def remove_copyright_lines(text: str) -> str:
    return re.sub(r"©\s*\d{4}.*?\n", "", text)


def base_clean(text: str) -> str:
    """Apply all base cleaning steps in order."""
    text = remove_standalone_page_numbers(text)
    text = remove_copyright_lines(text)
    text = remove_toc_entries(text)
    text = collapse_whitespace(text)
    return text