"""The official metric scripts, vendored unmodified, loaded under their bare names."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_NAMES = ("accuracy", "token_efficiency", "overall_score")
_FOLDER = str(Path(__file__).parent)


def _load() -> list:
    for name in _NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, _FOLDER)
    try:
        return [importlib.import_module(name) for name in _NAMES]
    finally:
        sys.path.remove(_FOLDER)
        for name in _NAMES:
            sys.modules.pop(name, None)


accuracy, token_efficiency, overall_score = _load()
