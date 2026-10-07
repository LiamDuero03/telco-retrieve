"""ask --label-sources: labelled passages, a system prompt label from the results, a Sources: list
that names collection and file per passage. Fakes only: no network, no model."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli.main as cli_main
from open_telco_rag.generators import _labels
from open_telco_rag.types import SearchResult
from tests.fake_collections import FakeEmbedder, make_collection


def _res(doc_id, text, collection=None, **meta):
    md = dict(meta)
    if collection:
        md["collection"] = collection
    return SearchResult(text=text, distance=0.1, doc_id=doc_id, metadata=md)


R1 = _res("Rel-18/28_series/28312-i11.md::bow::0", "AMF text", "col_a", source="28312-i11.md",
          source_type="3gpp", release="Rel-18", heading="AMF")
R2 = _res("QualityOnDemand/README.md::bow::2", "QoD text", "col_b", source="README.md",
          source_type="camara", heading="Usage")
R3 = _res("DeviceLocation/README.md::bow::0", "Loc text", "col_b", source="README.md", source_type="camara")


# ── labels ───────────────────────────────────────────────────────────────────

def test_label_is_one_line_with_collection_and_available_metadata():
    label = _labels.passage_label(R1)
    assert label == ("collection: col_a | source_type: 3gpp | file: Rel-18/28_series/28312-i11.md | "
                     "release: Rel-18 | heading: AMF")
    assert "\n" not in label


def test_label_skips_what_is_missing_and_normalises_windows_paths():
    r = _res("DeviceLocation\\README.md::bow::0", "x", source="README.md")
    assert _labels.passage_label(r) == "file: DeviceLocation/README.md"
    assert _labels.passage_label(r, "col_z") == "collection: col_z | file: DeviceLocation/README.md"


def test_version_is_shown_when_there_is_no_release():
    r = _res("x.md::bow::0", "t", source="x.md", version="i11")
    assert _labels.passage_label(r).endswith("version: i11")


def test_labelled_results_prefix_each_passage_and_leave_the_originals_alone():
    out = _labels.label_results([R1, R2])
    assert out[0].text.startswith("[collection: col_a | source_type: 3gpp") and out[0].text.endswith("\nAMF text")
    assert R1.text == "AMF text" and out[1].doc_id == R2.doc_id


def test_system_label_from_source_types_then_collections():
    assert _labels.system_label([R1, R2, R3]) == "3gpp and camara"
    assert _labels.system_label([R2, R3]) == "camara"
    assert _labels.system_label([R1, R2, _res("z::b::0", "t", "col_c", source_type="etsi")]) == "3gpp, camara and etsi"
    assert _labels.system_label([_res("z::b::0", "t", "col_c")]) == "col_c"
    assert _labels.system_label([_res("z::b::0", "t")]) is None
    assert _labels.system_label([]) is None


def test_sources_are_per_passage_in_context_order_and_tell_equal_file_names_apart():
    entries = _labels.source_entries([R2, R1, R3])
    assert entries == ["col_b: QualityOnDemand/README.md", "col_a: 28312-i11.md", "col_b: DeviceLocation/README.md"]


def test_a_unique_file_name_stays_short_and_the_default_collection_fills_in():
    only = _res("a/b.md::bow::0", "t", source="b.md")
    assert _labels.source_entries([only], "single_col") == ["single_col: b.md"]
    assert _labels.source_entries([only]) == ["b.md"]


# ── prompt shape for the three generators ────────────────────────────────────

def _sent_openrouter(monkeypatch, results, source_type):
    import open_telco_rag.evaluation._claude as claude_module
    from open_telco_rag.generators.openrouter import OpenRouterGenerator
    seen = {}

    def post(requests_mod, url, headers=None, json_payload=None, timeout=None, **kw):
        seen.update(json_payload)
        return SimpleNamespace(json=lambda: {"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr(claude_module, "_post_with_retry", post)
    OpenRouterGenerator(api_key="test").generate("Q?", results, source_type=source_type)
    return seen["messages"][0]["content"], seen["messages"][1]["content"]


def _sent_local(monkeypatch, results, source_type):
    import requests
    from open_telco_rag.generators.local import LocalGenerator
    seen = {}

    def post(url, headers=None, json=None, timeout=None):
        seen.update(json)
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr(requests, "post", post)
    LocalGenerator(model="m", base_url="http://x/v1").generate("Q?", results, source_type=source_type)
    return seen["messages"][0]["content"], seen["messages"][1]["content"]


def _sent_claude(monkeypatch, results, source_type):
    import open_telco_rag.generators.claude as claude_module
    seen = {}

    class Client:
        def __init__(self, api_key=None):
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(text="ok")], usage=None)

    monkeypatch.setattr(claude_module.anthropic, "Anthropic", Client)
    claude_module.ClaudeGenerator(api_key="test").generate("Q?", results, source_type=source_type)
    prompt = seen["messages"][0]["content"]
    return prompt, prompt


SENDERS = {"openrouter": _sent_openrouter, "local": _sent_local, "claude": _sent_claude}


@pytest.mark.parametrize("name", SENDERS)
def test_labelled_prompt_shape(monkeypatch, name):
    labelled = _labels.label_results([R1, R2])
    system, user = SENDERS[name](monkeypatch, labelled, _labels.system_label([R1, R2]))
    assert "You are a 3gpp and camara technical expert." in system
    first, second = (user.index("[collection: col_a"), user.index("[collection: col_b"))
    assert first < second < user.index("Question: Q?")
    assert "heading: AMF]\nAMF text\n\n---\n\n[collection: col_b" in user


@pytest.mark.parametrize("name", SENDERS)
def test_unlabelled_prompt_is_the_old_one(monkeypatch, name):
    system, user = SENDERS[name](monkeypatch, [R1, R2], "oran")
    assert "You are a oran technical expert." in system
    assert "[collection:" not in user and "AMF text\n\n---\n\nQoD text" in user


# ── CLI ──────────────────────────────────────────────────────────────────────

ROWS = [
    {"id": "d1/README.md::fake64::0", "source": "README.md", "source_type": "camara",
     "text": "the device api handles registration of a session"},
    {"id": "d2/README.md::fake64::0", "source": "README.md", "source_type": "camara",
     "text": "the device api handles registration of a boost"},
]


@pytest.fixture
def ask(tmp_path, monkeypatch, capsys):
    embedder = FakeEmbedder()
    make_collection(str(tmp_path / "db"), "camara_fake64", ROWS, embedder)
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: embedder)
    seen = {}

    class Generator:
        def generate(self, question, results, source_type="telecom", **kw):
            seen["results"], seen["source_type"] = results, source_type
            return {"answer": "A", "sources": [r.metadata.get("source") for r in results],
                    "chunks_used": len(results)}

    monkeypatch.setattr(cli_main, "_get_generator", lambda *a, **k: Generator())

    def run(*flags):
        capsys.readouterr()
        cli_main.main(["ask", "registration session", "--db-path", str(tmp_path / "db"), "--model", "fake64",
                       "--collection", "camara_fake64", "--top-k", "2", *flags])
        return capsys.readouterr().out

    return run, seen


def test_default_ask_is_unchanged(ask):
    run, seen = ask
    out = run()
    assert out == "\nAnswer:\nA\n\nSources:     ['README.md', 'README.md']\nChunks used: 2\n"
    assert seen["source_type"] == "oran" and all("[collection" not in r.text for r in seen["results"])


def test_label_sources_labels_passages_and_builds_the_system_label(ask):
    run, seen = ask
    run("--label-sources")
    assert seen["source_type"] == "camara"
    assert all(r.text.startswith("[collection: camara_fake64 | source_type: camara | file: d") for r in seen["results"])


def test_label_sources_sources_line_tells_equal_file_names_apart(ask):
    run, _ = ask
    out = run("--label-sources")
    assert "'camara_fake64: d1/README.md'" in out and "'camara_fake64: d2/README.md'" in out
    assert "Chunks used: 2" in out


def test_explicit_source_type_wins_over_the_built_label(ask):
    run, seen = ask
    run("--label-sources", "--source-type", "oran")
    assert seen["source_type"] == "oran"
    assert seen["results"][0].text.startswith("[collection:")
