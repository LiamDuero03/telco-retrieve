from __future__ import annotations

import re

_BARE_LETTER = re.compile(r"^\(?([A-Z])\)?[.:)]?$")
_LABELLED_LETTER = re.compile(
    r"^(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS)?\s*[:\-]?\s*"
    r"\(?([A-Z])\)?(?:[.:,\s]|$)"
)
_LEADING_LETTER = re.compile(r"^\(?([A-Z])[.):]\s")
_BOOLEAN = re.compile(r"^(?:ANSWER\s*(?:IS)?\s*[:\-]?\s*)?(TRUE|FALSE)\b")


def _option_letter(text: str, letters: set[str]) -> str:
    for pattern in (_BARE_LETTER, _LABELLED_LETTER, _LEADING_LETTER):
        match = pattern.match(text)
        if match:
            return match.group(1) if match.group(1) in letters else ""
    return ""


def normalise_answer(text: str, options: dict[str, str]) -> str:
    """A-E for multiple choice, TRUE or FALSE for an empty options object, else an empty string."""
    cleaned = (text or "").strip().upper()
    if not options:
        match = _BOOLEAN.match(cleaned)
        return match.group(1) if match else ""
    return _option_letter(cleaned, {key.strip().upper() for key in options})
