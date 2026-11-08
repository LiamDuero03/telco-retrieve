import argparse
import random

import pytest

from open_telco_rag.challenge import settings as challenge_settings
from open_telco_rag.challenge.embedder import build_custom_embedder
from open_telco_rag.challenge.metrics import token_efficiency
from open_telco_rag.challenge.profile import load_profile
from open_telco_rag.challenge.seed import set_all_seeds
from open_telco_rag.challenge.settings import resolve_settings

FALLBACKS = challenge_settings.FALLBACKS
PROFILE_DEFAULTS = {
    "profile": "challenge",
    "model": "otel-0.6b",
    "reranker": "otel-reranker-0.6b",
    "rerank": True,
    "rerank_top_k": 5,
    "search_method": "hybrid",
    "top_k": 10,
    "generator_model": "google/gemma-3n-E4B-it",
    "temperature": 0.0,
    "max_tokens": 8,
    "seed": 1234,
    "resolve_parents": False,
    "max_prompt_tokens": 4096,
    "timeout": 120.0,
}


def namespace(**given):
    return argparse.Namespace(**given)


def test_profile_values_and_no_collection_named():
    profile = load_profile("challenge", set(FALLBACKS))
    assert profile == {
        "model": "otel-0.6b",
        "reranker": "otel-reranker-0.6b",
        "rerank": True,
        "rerank_top_k": 5,
        "search_method": "hybrid",
        "top_k": 10,
        "generator_model": "google/gemma-3n-E4B-it",
        "temperature": 0.0,
        "max_tokens": 8,
        "seed": 1234,
        "max_prompt_tokens": 4096,
    }
    with pytest.raises(ValueError, match="available: \\['challenge'\\]"):
        load_profile("nope", set(FALLBACKS))


def test_profile_cap_equals_the_official_cap():
    profile = load_profile("challenge", set(FALLBACKS))
    assert profile["max_prompt_tokens"] == token_efficiency.MAX_PROMPT_TOKENS
    assert resolve_settings(namespace())["max_prompt_tokens"] == token_efficiency.MAX_PROMPT_TOKENS


def test_custom_embedder_uses_registry_fields_without_loading_a_model(tmp_path):
    embedder = build_custom_embedder(str(tmp_path), query_prefix="Q: ", document_prefix="D: ")
    assert embedder._model_name == str(tmp_path) and embedder._model is None
    assert embedder._apply_prefix(["x"], True) == ["Q: x"] and embedder._apply_prefix(
        ["x"], False
    ) == ["D: x"]


def test_seed_makes_python_draws_repeatable():
    set_all_seeds(7)
    first = random.random()
    set_all_seeds(7)
    assert random.random() == first


def test_no_flags_resolve_to_the_challenge_profile(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    settings = resolve_settings(namespace())
    assert {k: v for k, v in settings.items() if k in PROFILE_DEFAULTS} == PROFILE_DEFAULTS


def test_local_llm_model_env_beats_the_profile_and_the_flag_beats_the_env(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL", "from-env")
    assert resolve_settings(namespace())["generator_model"] == "from-env"
    assert resolve_settings(namespace(generator_model="flag"))["generator_model"] == "flag"


COMPETITION = {
    "model": "otel-0.6b",
    "reranker": "otel-reranker-0.6b",
    "rerank": True,
    "rerank_top_k": 5,
    "generator_model": "google/gemma-3n-E4B-it",
}
OTHER_MODELS = ("minilm", "ms-marco", "mpnet", "bge", "e5", "llama", "qwen")


def competition_only(settings):
    assert {key: settings[key] for key in COMPETITION} == COMPETITION
    flat = " ".join(str(value).lower() for value in settings.values())
    assert not any(name in flat for name in OTHER_MODELS)


def test_the_default_components_are_the_competition_ones_and_load_from_these_ids():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    embedders = json.loads((root / "embedding_models.json").read_text(encoding="utf-8"))
    rerankers = json.loads((root / "reranker_models.json").read_text(encoding="utf-8"))
    settings = resolve_settings(argparse.Namespace())
    competition_only(settings)
    assert embedders[settings["model"]]["model_name"] == "farbodtavakkoli/OTel-Embedding-0.6B"
    assert rerankers[settings["reranker"]]["model_name"] == "farbodtavakkoli/OTel-Reranker-0.6B"
    assert rerankers[settings["reranker"]]["backend"] == "otel"


def test_no_default_falls_back_to_another_model(monkeypatch, tmp_path):
    from open_telco_rag.challenge import profile

    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    competition_only(resolve_settings(argparse.Namespace()))
    competition_only(resolve_settings(argparse.Namespace(top_k=3, seed=7, temperature=0.2)))
    monkeypatch.setattr(profile, "PROFILE_DIRS", (tmp_path / "nowhere",))
    competition_only(resolve_settings(argparse.Namespace()))
    competition_only(resolve_settings(argparse.Namespace(profile="challenge", top_k=3)))
    (tmp_path / "challenge.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(profile, "PROFILE_DIRS", (tmp_path,))
    competition_only(resolve_settings(argparse.Namespace(search_method="dense")))
    with pytest.raises(ValueError, match="unknown profile 'other'"):
        resolve_settings(argparse.Namespace(profile="other"))


def test_no_other_model_is_named_anywhere_in_the_challenge_sources():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sources = list((root / "open_telco_rag" / "challenge").glob("*.py"))
    sources += list((root / "configs" / "profiles").glob("*.json"))
    found = [
        f"{path.name}: {name}"
        for path in sources
        for name in ("minilm", "ms-marco", "mpnet")
        if name in path.read_text(encoding="utf-8").lower()
    ]
    assert found == []
