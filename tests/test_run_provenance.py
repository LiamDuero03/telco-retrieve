"""Reproducibility record, ingest settings sidecar, determinism and no-secrets checks."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.core.pipeline import ingest
from open_telco_rag.evaluation.evaluate import evaluate_corpus
from open_telco_rag.evaluation.provenance import (
    SCHEMA_VERSION,
    collect_run_record,
    hash_questions,
    read_index_settings,
)
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.evaluation.results import CorpusResult, QuestionResult
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore
from open_telco_rag.types import SearchResult
from tests.test_retrieval_methods import DOCS, BagOfWordsEmbedder

SECRET = "sk-or-THIS-MUST-NEVER-APPEAR"


def _questions(n=4):
    return [EvalQuestion(question=f"Which interface {i}?", keywords=[], corpus="3gpp", answer_type="mc",
                         options={"A": "N1", "B": "Xn"}, reference_answer="B") for i in range(n)]


class _Store:
    collection_name = "c1"
    path = "somewhere"

    def count(self):
        return 7


def _record(**over):
    kw = {"params": {"top_k": 5}, "embedder": SimpleNamespace(model_key="minilm", _model_name="x/y", _dimensions=3),
              "reranker": None, "store": _Store(), "questions": _questions(), "questions_name": "q.json",
              "started_at": datetime.now(timezone.utc)}
    kw.update(over)
    return collect_run_record(**kw)


def test_the_record_has_the_expected_sections():
    r = _record()
    assert r["schema_version"] == SCHEMA_VERSION == 1
    assert r["code"]["name"] == "open-telco-rag" and "git_commit" in r["code"] and "git_dirty" in r["code"]
    assert r["environment"]["python"] and "torch" in r["environment"]["dependencies"]
    assert r["settings"] == {"top_k": 5} and r["tokenizer"] == "cl100k_base"
    assert r["embedder"]["model_key"] == "minilm" and r["embedder"]["model_name"] == "x/y"
    assert r["index"] == {"collection": "c1", "chunks": 7}
    assert r["dataset"]["n_questions"] == 4 and len(r["dataset"]["questions_sha256"]) == 64
    assert r["randomness"]["seed"] is None and "retry jitter" in r["randomness"]["note"]
    assert r["started_at"] <= r["finished_at"]


def test_the_dataset_hash_tracks_content_and_order():
    qs = _questions()
    assert hash_questions(qs) == hash_questions(_questions())
    assert hash_questions(qs) != hash_questions(list(reversed(qs)))
    changed = _questions()
    changed[0].question = "different"
    assert hash_questions(qs) != hash_questions(changed)


def test_the_record_holds_no_secrets_and_no_local_paths(tmp_path):
    embedder = SimpleNamespace(model_key="or-3-small", _model_name="m", api_key=SECRET, _api_key=SECRET,
                               session=object(), url="http://10.1.2.3:8000/v1", batch_size=200)
    store = SimpleNamespace(collection_name="c1", path=str(tmp_path / "db"), count=lambda: 1)
    record = _record(embedder=embedder, store=store, extra={"judge": {"generator": "local", "model": "m"}})
    blob = json.dumps(record)
    assert SECRET not in blob and "10.1.2.3" not in blob and str(tmp_path) not in blob
    assert record["extra"]["judge"]["model"] == "m"


# ── evaluation output ─────────────────────────────────────────────────────────

def _evaluate(monkeypatch, tmp_path, name="out", concurrency=1, n=6, **kw):
    monkeypatch.setattr(pipeline_module, "search", lambda q, *a, **k: [
        SearchResult(text=f"chunk for {q}", distance=0.1, metadata={"source": "a.md"}, doc_id="a.md::m::0")])
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **k: {"answer": "B", "sources": [], "chunks_used": 1})
    result, _ = evaluate_corpus("3gpp", store=_Store(), embedder=SimpleNamespace(model_key="minilm"),
                                questions=_questions(n), model="minilm", output_dir=tmp_path / name,
                                rag_generator_name="local", rag_generator_model="m", delay=0,
                                max_concurrency=concurrency, **kw)
    saved = json.loads(next((tmp_path / name).glob("eval_*.json")).read_text(encoding="utf8"))
    return result, saved


def test_every_saved_result_carries_a_run_record(monkeypatch, tmp_path):
    _, saved = _evaluate(monkeypatch, tmp_path, run_info={"judge": {"generator": "claude", "model": "j"}},
                         top_k=3, search_method="hybrid")
    rec = saved["run_record"]
    assert rec["settings"]["top_k"] == 3 and rec["settings"]["search_method"] == "hybrid"
    assert rec["settings"]["rag_generator_model"] == "m" and rec["settings"]["retrieval_only"] is False
    assert rec["dataset"]["questions_file"] == "custom_list" and rec["dataset"]["n_questions"] == 6
    assert rec["extra"]["judge"] == {"generator": "claude", "model": "j"}


def test_old_result_files_without_a_record_still_load(monkeypatch, tmp_path):
    _, saved = _evaluate(monkeypatch, tmp_path)
    saved.pop("run_record")
    questions = saved.pop("question_results")
    restored = CorpusResult(**saved, question_results=[QuestionResult(**q) for q in questions])
    assert restored.run_record is None


def test_an_api_key_passed_to_the_evaluation_never_reaches_the_output(monkeypatch, tmp_path, capsys):
    _result, saved = _evaluate(monkeypatch, tmp_path, rag_api_key=SECRET)
    out = capsys.readouterr()
    assert SECRET not in json.dumps(saved) and SECRET not in out.out + out.err


# ── determinism ───────────────────────────────────────────────────────────────

_VOLATILE = ("latency", "started_at", "finished_at", "timestamp", "wall_clock", "git_dirty", "_ms")


def _stable(obj):
    if isinstance(obj, dict):
        return {k: _stable(v) for k, v in obj.items() if not any(m in k for m in _VOLATILE)}
    if isinstance(obj, list):
        return [_stable(v) for v in obj]
    return obj


def test_the_same_inputs_give_identical_results(monkeypatch, tmp_path):
    _, first = _evaluate(monkeypatch, tmp_path, name="a")
    _, second = _evaluate(monkeypatch, tmp_path, name="b")
    assert _stable(first) == _stable(second)


@pytest.mark.parametrize("concurrency", [1, 4])
def test_results_keep_the_question_order_under_concurrency(monkeypatch, tmp_path, concurrency):
    _result, saved = _evaluate(monkeypatch, tmp_path, concurrency=concurrency, n=12)
    assert [q["question"] for q in saved["question_results"]] == [f"Which interface {i}?" for i in range(12)]
    _, serial = _evaluate(monkeypatch, tmp_path, name="serial", concurrency=1, n=12)
    assert _stable(saved["question_results"]) == _stable(serial["question_results"])


# ── ingest settings sidecar ───────────────────────────────────────────────────

def _ingest(tmp_path, collection="coll_a", **over):
    data = tmp_path / "data"
    if not data.exists():
        data.mkdir()
        for name, text in DOCS.items():
            (data / name).write_text(text, encoding="utf-8")
    store = ChromaStore(path=tmp_path / "db", collection_name=collection, metadata={"hnsw:space": "cosine"})
    kw = {"strategy": "text_baseline", "chunk_size": 300, "overlap": 0, "min_length": 10, **over}
    ingest(source=TPPSource(), store=store, embedder=BagOfWordsEmbedder(), data_dir=str(data), resume=False,
           log_dir=str(tmp_path / "logs"), **kw)
    return store


def test_ingest_records_its_settings_without_local_paths(tmp_path):
    store = _ingest(tmp_path)
    info = read_index_settings(store.path, "coll_a")
    assert info["settings"]["chunking"] == {"strategy": "text_baseline", "chunk_size": 300, "overlap": 0}
    assert info["settings"]["model_key"] == "bow" and info["configurations_seen"] == 1
    assert "db_path" not in info["settings"]
    sidecar = (Path(store.path) / ".ingest_settings_coll_a.json").read_text(encoding="utf8")
    assert str(tmp_path) not in sidecar


def test_a_collection_ingested_under_two_configurations_says_so(tmp_path):
    _ingest(tmp_path)
    store = _ingest(tmp_path, chunk_size=500)
    info = read_index_settings(store.path, "coll_a")
    assert info["settings"]["chunking"]["chunk_size"] == 500 and info["configurations_seen"] == 2


def test_the_run_record_names_the_index_settings(tmp_path):
    store = _ingest(tmp_path)
    record = collect_run_record(params={}, embedder=BagOfWordsEmbedder(), reranker=None, store=store,
                                questions=_questions(1), questions_name="q.json", started_at=datetime.now(timezone.utc))
    assert record["index"]["ingest_settings"]["settings"]["chunking"]["strategy"] == "text_baseline"
    assert record["index"]["distance_space"] == "cosine" and record["index"]["chunks"] > 0
    assert str(tmp_path) not in json.dumps(record)


def test_a_missing_or_corrupt_sidecar_is_just_absent(tmp_path):
    assert read_index_settings(tmp_path, "nope") is None
    (tmp_path / ".ingest_settings_bad.json").write_text("{not json", encoding="utf8")
    assert read_index_settings(tmp_path, "bad") is None
    assert read_index_settings(None, None) is None


def test_the_questions_file_name_from_the_caller_is_recorded_not_a_path(monkeypatch, tmp_path):
    _, saved = _evaluate(monkeypatch, tmp_path, run_info={"questions_file": "my_questions.json"})
    assert saved["run_record"]["dataset"]["questions_file"] == "my_questions.json"


# ── git information degrades to "unknown" ─────────────────────────────────────

def test_code_version_is_unknown_without_a_git_directory(tmp_path, monkeypatch):
    from open_telco_rag.evaluation import provenance

    monkeypatch.setattr(provenance, "_PACKAGE_DIR", tmp_path)            # e.g. installed from a wheel
    code = provenance.code_version()
    assert code["git_commit"] == "unknown" and code["git_dirty"] == "unknown"
    assert code["name"] == "open-telco-rag" and code["version"]
    assert provenance.collect_run_record(
        params={}, embedder=SimpleNamespace(model_key="m"), reranker=None, store=_Store(),
        questions=_questions(1), questions_name="q", started_at=datetime.now(timezone.utc),
    )["code"]["git_commit"] == "unknown"


@pytest.mark.parametrize("failure", [FileNotFoundError("git"), OSError("boom")])
def test_code_version_is_unknown_when_git_cannot_run(tmp_path, monkeypatch, failure):
    import subprocess

    from open_telco_rag.evaluation import provenance

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(provenance, "_PACKAGE_DIR", tmp_path)

    def broken(*a, **k):
        raise failure

    monkeypatch.setattr(subprocess, "run", broken)
    assert provenance.code_version()["git_commit"] == "unknown"


def test_code_version_reports_a_real_checkout(monkeypatch):
    from open_telco_rag.evaluation import provenance

    code = provenance.code_version()
    assert code["git_commit"] != "unknown" and len(code["git_commit"]) == 40
    assert isinstance(code["git_dirty"], bool)
