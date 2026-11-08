from __future__ import annotations

import argparse
import os

from open_telco_rag.challenge.profile import load_profile

DEFAULT_PROFILE = "challenge"

GENERATOR_MODEL = "google/gemma-3n-E4B-it"

FALLBACKS = {
    "model": "otel-0.6b",
    "reranker": "otel-reranker-0.6b",
    "rerank": True,
    "rerank_top_k": 5,
    "search_method": "hybrid",
    "top_k": 10,
    "generator_model": GENERATOR_MODEL,
    "temperature": 0.0,
    "max_tokens": 8,
    "seed": 1234,
    "resolve_parents": False,
    "max_prompt_tokens": None,
    "timeout": 120.0,
}

EMBEDDER_KEYS = (
    "embedder_path",
    "embedder_id",
    "query_prefix",
    "document_prefix",
    "embedder_dimensions",
)


def _profile(name: str) -> dict:
    """The default profile may be absent: the fallbacks are the competition components too."""
    try:
        return load_profile(name, set(FALLBACKS))
    except ValueError:
        if name == DEFAULT_PROFILE:
            return {}
        raise


def profile_defaults(name: str = DEFAULT_PROFILE) -> dict:
    return {**FALLBACKS, **_profile(name)}


def resolve_settings(args: argparse.Namespace) -> dict:
    """Explicit flag, then LOCAL_LLM_MODEL (generator only), then the profile, then the fallback."""
    name = getattr(args, "profile", None) or DEFAULT_PROFILE
    profile = _profile(name)
    settings = {"profile": name}
    for key, fallback in FALLBACKS.items():
        flag = getattr(args, key, None)
        settings[key] = flag if flag is not None else profile.get(key, fallback)
    if getattr(args, "generator_model", None) is None and os.environ.get("LOCAL_LLM_MODEL"):
        settings["generator_model"] = os.environ["LOCAL_LLM_MODEL"]
    for key in EMBEDDER_KEYS:
        settings[key] = getattr(args, key, None)
    if settings["max_prompt_tokens"] is None:
        from open_telco_rag.challenge.caps import official_cap

        settings["max_prompt_tokens"] = official_cap()
    return settings


def describe(settings: dict) -> str:
    """One line with the effective settings, printed at the start of an export."""
    embedder = settings["embedder_path"] or settings["embedder_id"] or settings["model"]
    reranker = settings["reranker"] if settings["rerank"] else "off"
    parts = {
        "profile": settings["profile"],
        "embedder": embedder,
        "reranker": reranker,
        "rerank_top_k": settings["rerank_top_k"],
        "search": settings["search_method"],
        "top_k": settings["top_k"],
        "resolve_parents": "on" if settings["resolve_parents"] else "off",
        "generator_model": settings["generator_model"],
        "temperature": settings["temperature"],
        "seed": settings["seed"],
        "max_prompt_tokens": settings["max_prompt_tokens"],
        "max_answer_tokens": settings["max_tokens"],
        "timeout_s": settings["timeout"],
    }
    return "[settings] " + " ".join(f"{key}={value}" for key, value in parts.items())
