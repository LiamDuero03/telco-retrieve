from __future__ import annotations

import re
from pathlib import Path

_VENDORED = Path(__file__).parent / "metrics" / "token_efficiency.py"


def official_cap() -> int:
    """MAX_PROMPT_TOKENS read from the vendored official script without importing it."""
    match = re.search(
        r"^MAX_PROMPT_TOKENS\s*=\s*(\d+)", _VENDORED.read_text(encoding="utf-8"), re.MULTILINE
    )
    if not match:
        raise RuntimeError(f"MAX_PROMPT_TOKENS not found in {_VENDORED}")
    return int(match.group(1))
