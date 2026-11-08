from __future__ import annotations

import json
from pathlib import Path

PROFILE_DIRS = (
    Path("configs") / "profiles",
    Path(__file__).resolve().parents[2] / "configs" / "profiles",
)


def available_profiles() -> list[str]:
    return sorted(
        {
            path.stem
            for directory in PROFILE_DIRS
            if directory.is_dir()
            for path in directory.glob("*.json")
        }
    )


def load_profile(name: str, allowed_keys: set[str]) -> dict:
    """Settings of a named profile; a profile never names a collection or path."""
    for directory in PROFILE_DIRS:
        path = directory / f"{name}.json"
        if path.is_file():
            profile = json.loads(path.read_text(encoding="utf-8"))
            unknown = set(profile) - allowed_keys
            if unknown:
                raise ValueError(f"profile '{name}' has unknown keys {sorted(unknown)}")
            return profile
    raise ValueError(f"unknown profile '{name}'; available: {available_profiles() or 'none'}")
