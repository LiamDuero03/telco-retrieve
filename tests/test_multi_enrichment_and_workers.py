"""ingest_multi_enrichment stores span metadata only for token-sized chunks, and
the ingest CLI passes --embed-max-workers only when it is given."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import cli.main as cli_main
import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.core.pipeline import ingest_multi_enrichment
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore
from tests.test_retrieval_methods import DOCS, BagOfWordsEmbedder

SPAN_KEYS = {"char_start", "char_end", "n_tokens"}


def _run(tmp_path, **chunk_kwargs):
    data = tmp_path / "data"
    data.mkdir()
    for name, text in DOCS.items():
        (data / name).write_text(text * 3, encoding="utf-8")
    store = ChromaStore(path=tmp_path / "db", collection_name="t_none", metadata={"hnsw:space": "cosine"})
    stats = ingest_multi_enrichment(
        source=TPPSource(), stores={"none": store}, embedder=BagOfWordsEmbedder(), data_dir=str(data),
        enrichment_strategies=["none"], min_length=10, log_dir=str(tmp_path / "logs"),
        strategy="text_baseline", **chunk_kwargs)
    assert stats["none"].error_files == 0 and stats["none"].chunks_stored > 0
    return store._collection.get(include=["metadatas"])["metadatas"]


def test_token_sized_multi_enrichment_stores_offsets(tmp_path):
    metas = _run(tmp_path, target_tokens=40, max_tokens=50)
    assert metas and all(SPAN_KEYS <= set(m) and m["char_start"] >= 0 and m["n_tokens"] > 0 for m in metas)


def test_default_multi_enrichment_stores_no_offsets(tmp_path):
    metas = _run(tmp_path, chunk_size=300, overlap=0)
    assert metas and all(not (SPAN_KEYS & set(m)) for m in metas)


# ── --embed-max-workers ───────────────────────────────────────────────────────

def _ingest_args(**over):
    base = {"source_type": "oran", "use_marker": False, "model": "minilm", "strategy": "text_baseline",
                "enrichment": "none", "data": None, "db_path": "db", "no_resume": True, "collection": None,
                "chunk_size": None, "overlap": None, "target_tokens": None, "max_tokens": None,
                "overlap_tokens": None, "embed_backend": "openrouter"}
    base.update(over)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("workers,expected", [(None, {}), (4, {"max_workers": 4})])
def test_embed_max_workers_is_passed_only_when_given(monkeypatch, workers, expected):
    seen = {}

    def fake_embedder(model_key, backend="local", max_workers=None):
        seen.update(model=model_key, backend=backend, max_workers=max_workers)
        return object()

    monkeypatch.setattr(cli_main, "_get_embedder", fake_embedder)
    monkeypatch.setattr(cli_main, "_get_source", lambda *a, **k: object())
    monkeypatch.setattr(cli_main, "_get_store", lambda *a, **k: SimpleNamespace(collection_name="c"))
    monkeypatch.setattr(pipeline_module, "ingest", lambda **kw: SimpleNamespace(
        files_processed=0, chunks_stored=0, skipped_files=0, error_files=0))

    cli_main._run_ingest(_ingest_args(embed_max_workers=workers))
    assert seen["max_workers"] == expected.get("max_workers")
    assert seen["backend"] == "openrouter"
