"""The optional per-key `dtype` of an embedder (and a reranker) in the registry. Default unchanged: float16 on a GPU,
nothing on a CPU. Recorders stand in for the model and for torch.autocast; torch.cuda.is_available is monkeypatched,
so no GPU, no download and no network is needed."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from open_telco_rag.embedders import reranker as reranker_module
from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder

REGISTRY = json.loads(SentenceTransformerEmbedder.__init__.__defaults__[1].read_text(encoding="utf-8"))


class Recorder:
    """what the fake SentenceTransformer and the fake torch.autocast saw"""

    def __init__(self):
        self.model_kwargs = "not constructed"
        self.device = None
        self.autocasts: list[tuple[str, object]] = []
        self.encoded = 0


@pytest.fixture
def gpu(monkeypatch):
    import sentence_transformers

    seen = Recorder()

    class FakeModel:
        max_seq_length = 512

        def __init__(self, name, device=None, model_kwargs=None, **kwargs):
            seen.model_kwargs, seen.device, self.device = model_kwargs, device, device

        def encode(self, texts, **kwargs):
            seen.encoded += 1
            return np.zeros((len(texts), 4))

    class FakeAutocast:
        def __init__(self, device_type, dtype=None, **kwargs):
            seen.autocasts.append((device_type, dtype))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(torch, "autocast", FakeAutocast)
    return seen


def _embedder_for(tmp_path, **config):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"k": {"model_name": "some/model", "dimensions": 4, **config}}), encoding="utf-8")
    return SentenceTransformerEmbedder(model_key="k", models_file=path)


# ── the default is unchanged ─────────────────────────────────────────────────

LOCAL_KEYS = [k for k, v in REGISTRY.items() if v.get("backend") != "openrouter" and not k.startswith("or-")]


@pytest.mark.parametrize("key", [k for k in LOCAL_KEYS if "dtype" not in REGISTRY[k]])
def test_every_existing_key_still_loads_in_float16_and_autocasts_in_float16(gpu, key):
    SentenceTransformerEmbedder(model_key=key).embed(["text"])
    assert gpu.model_kwargs == {"torch_dtype": torch.float16} and gpu.device == "cuda"
    assert gpu.autocasts == [("cuda", torch.float16)]


def test_no_existing_key_has_a_dtype_and_only_the_new_gemma_key_has_one():
    assert {k for k, v in REGISTRY.items() if "dtype" in v} == {"embeddinggemma-300m-fp32"}


def test_a_cpu_run_is_unchanged_whatever_the_key_says(monkeypatch, tmp_path):
    import sentence_transformers
    calls = []

    class FakeModel:
        device, max_seq_length = "cpu", 512

        def __init__(self, name, device=None, **kwargs):
            calls.append((device, kwargs))

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 4))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: pytest.fail("no autocast on a CPU"))
    _embedder_for(tmp_path, dtype="bfloat16").embed(["text"])
    assert calls == [("cpu", {})]


# ── the key's dtype reaches both places ──────────────────────────────────────

@pytest.mark.parametrize("name", ["float16", "bfloat16"])
def test_the_autocast_dtype_follows_the_key(gpu, tmp_path, name):
    _embedder_for(tmp_path, dtype=name).embed(["text"])
    assert gpu.model_kwargs == {"torch_dtype": getattr(torch, name)}
    assert gpu.autocasts == [("cuda", getattr(torch, name))]


def test_float32_means_no_autocast(gpu, tmp_path):
    _embedder_for(tmp_path, dtype="float32").embed(["text"])
    assert gpu.model_kwargs == {"torch_dtype": torch.float32}
    assert gpu.autocasts == [] and gpu.encoded == 1, "encoded, but not under autocast"


def test_an_unknown_dtype_is_refused_when_the_embedder_is_built(tmp_path):
    with pytest.raises(ValueError, match="dtype.*float16.*bfloat16.*float32"):
        _embedder_for(tmp_path, dtype="int8")


# ── the new gemma key ────────────────────────────────────────────────────────

def test_the_fp32_gemma_key_is_the_same_model_in_float32():
    old, new = REGISTRY["embeddinggemma-300m"], REGISTRY["embeddinggemma-300m-fp32"]
    assert new["model_name"] == old["model_name"] == "google/embeddinggemma-300m"
    assert new["dimensions"] == old["dimensions"] == 768
    assert {k: v for k, v in new.items() if k not in ("dtype", "description")} == {k: v for k, v in old.items() if k != "description"}
    assert new["dtype"] == "float32" and "dtype" not in old


def test_the_fp32_gemma_key_loads_in_float32_without_autocast(gpu):
    SentenceTransformerEmbedder(model_key="embeddinggemma-300m-fp32").embed(["text"])
    assert gpu.model_kwargs == {"torch_dtype": torch.float32} and gpu.autocasts == []


def test_the_smoke_test_script_lists_the_new_key_and_keeps_the_old_ones():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "embedder_smoke_test.py"
    spec = importlib.util.spec_from_file_location("scripts_embedder_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = module.local_model_keys()
    assert "embeddinggemma-300m-fp32" in keys and "embeddinggemma-300m" in keys and "minilm" in keys
    for skipped in ("or-3-small", "qwen3-embedding-8b"):
        assert skipped not in keys


# ── the reranker ─────────────────────────────────────────────────────────────

@pytest.fixture
def rerank_load(monkeypatch):
    import transformers

    seen = {}

    class FakeModel:
        def eval(self):
            return self

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained",
                        lambda name, **kwargs: seen.update(kwargs) or FakeModel())
    return seen


@pytest.mark.parametrize("cuda,expected", [(True, torch.float16), (False, torch.float32)])
def test_the_reranker_default_is_float16_on_a_gpu_and_float32_on_a_cpu(monkeypatch, rerank_load, cuda, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    reranker_module.OTelReranker()._load()
    assert rerank_load["dtype"] == expected


@pytest.mark.parametrize("name", ["float16", "bfloat16", "float32"])
def test_the_reranker_takes_an_explicit_dtype(monkeypatch, rerank_load, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    reranker_module.OTelReranker(dtype=name)._load()
    assert rerank_load["dtype"] == getattr(torch, name)


def test_an_unknown_reranker_dtype_is_refused():
    with pytest.raises(ValueError, match="dtype"):
        reranker_module.OTelReranker(dtype="int8")


def test_the_reranker_registry_dtype_reaches_the_reranker(monkeypatch, tmp_path):
    path = tmp_path / "rerankers.json"
    path.write_text(json.dumps({"r": {"model_name": "x/y", "backend": "otel", "dtype": "bfloat16"},
                                "d": {"model_name": "x/y", "backend": "otel"}}), encoding="utf-8")
    monkeypatch.setattr(reranker_module, "_RERANKER_MODELS_FILE", path)
    assert reranker_module.load_reranker("r")._dtype == "bfloat16"
    assert reranker_module.load_reranker("d")._dtype is None
