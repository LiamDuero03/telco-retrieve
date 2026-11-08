import argparse

import pytest

from open_telco_rag.challenge.guards import (
    check_embedder_matches_index,
    check_flags_have_an_effect,
    index_embedder_key,
)
from open_telco_rag.challenge.hooks import load_callable


class FakeCollection:
    def __init__(self, ids, metadatas):
        self.ids, self.metadatas = ids, metadatas

    def peek(self, n):
        return {"ids": self.ids[:n]}

    def get(self, limit, include):
        return {"metadatas": self.metadatas[:limit]}


class FakeStore:
    def __init__(self, ids=("a.md::otel-0.6b::0",), metadatas=({},)):
        self._collection = FakeCollection(list(ids), list(metadatas))


def settings(**over):
    base = {
        "model": "otel-0.6b",
        "embedder_path": None,
        "embedder_id": None,
        "rerank": True,
        "resolve_parents": False,
    }
    return base | over


def args(**over):
    base = {
        "db_path": "/nonexistent",
        "collection": "c",
        "rerank_top_k": None,
        "reranker": None,
        "allow_embedder_mismatch": False,
    }
    return argparse.Namespace(**(base | over))


def test_index_embedder_key_comes_from_the_chunk_ids_without_a_sidecar():
    assert index_embedder_key("/nonexistent", "c", FakeStore()) == "otel-0.6b"
    assert index_embedder_key("/nonexistent", "c", FakeStore(ids=["plain"])) is None


def test_embedder_guard_matches_by_key_or_by_model_name_and_aborts_otherwise():
    check_embedder_matches_index(args(), settings(), FakeStore())
    check_embedder_matches_index(
        args(), settings(embedder_id="farbodtavakkoli/OTel-Embedding-0.6B"), FakeStore()
    )
    with pytest.raises(SystemExit, match="pass --model otel-0.6b"):
        check_embedder_matches_index(args(), settings(model="minilm"), FakeStore())
    check_embedder_matches_index(
        args(allow_embedder_mismatch=True), settings(model="minilm"), FakeStore()
    )


def test_flag_guards():
    check_flags_have_an_effect(args(rerank_top_k=3), settings(), FakeStore())
    with pytest.raises(SystemExit, match="--rerank-top-k has no effect"):
        check_flags_have_an_effect(args(rerank_top_k=3), settings(rerank=False), FakeStore())
    with pytest.raises(SystemExit, match="--resolve-parents needs"):
        check_flags_have_an_effect(args(), settings(resolve_parents=True), FakeStore())
    check_flags_have_an_effect(
        args(), settings(resolve_parents=True), FakeStore(metadatas=[{}, {"parent_id": "P1"}])
    )


def test_load_callable_resolves_module_function_and_rejects_the_rest():
    assert load_callable("os.path:join").__name__ == "join"
    for spec, text in (
        ("nocolon", "package.module:function"),
        ("no_such_mod:f", "cannot import"),
        ("os.path:nofunc", "no callable"),
        ("os:name", "no callable"),
    ):
        with pytest.raises(ValueError, match=text):
            load_callable(spec)


def test_same_local_folder_matches_however_the_path_is_written(tmp_path, monkeypatch):
    import json

    from open_telco_rag.challenge import guards

    model = tmp_path / "ft_model"
    model.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"custom-ft": {"model_name": str(model), "dimensions": 384}}), encoding="utf-8"
    )
    monkeypatch.setattr(guards, "REGISTRY", registry)
    monkeypatch.chdir(tmp_path)
    store = FakeStore(ids=["a.md::custom-ft::0"])
    for written in (str(model), "ft_model", "./ft_model", str(tmp_path / "x" / ".." / "ft_model")):
        check_embedder_matches_index(args(), settings(embedder_path=written), store)
    (tmp_path / "other").mkdir()
    with pytest.raises(SystemExit, match="pass --model custom-ft"):
        check_embedder_matches_index(args(), settings(embedder_path="other"), store)
