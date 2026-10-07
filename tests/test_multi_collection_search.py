"""search / ask over one or several collections of one database (fake embedders, real Chroma + FTS)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import cli.main as cli_main
from open_telco_rag.core import multi_search as ms
from open_telco_rag.types import SearchResult
from tests.fake_collections import (  # noqa: F401  (the fixture applies to every test of this file)
    FakeEmbedder, make_collection, in_memory_chroma, registry)

QUERY = "registration mobility"

ROWS_A = [
    {"id": "a.md::fake64::0", "text": "the amf handles registration and mobility management",
     "source": "a.md", "heading": "AMF", "meta": {"release": "Rel-18", "work_group": "WG1"}},
    {"id": "a.md::fake64::1", "text": "the smf handles session management and ip allocation",
     "source": "a.md", "heading": "SMF"},
    {"id": "b.md::fake64::0", "text": "the upf forwards user plane traffic for sessions",
     "source": "b.md", "heading": "UPF"},
    {"id": "c.md::fake64::0", "text": "mobility management registration update procedure",
     "source": "c.md", "heading": "MM"},
]

ROWS_B = [
    {"id": "README.md::fake64::0", "text": "the quality api registration of a device for a boost session",
     "source": "README.md", "source_type": "camara", "heading": "Quality"},
    {"id": "a.md::fake64::0", "text": "the location api reports mobility of a device inside a geofence",
     "source": "a.md", "source_type": "camara", "heading": "Location"},
    {"id": "spec.yaml::fake64::0", "text": "post sessions creates a boost session for the device",
     "source": "spec.yaml", "source_type": "camara", "heading": "POST /sessions"},
]


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "db")


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def three(db, embedder):
    """two filled collections that share the id a.md::fake64::0, plus an empty one"""
    make_collection(db, "col_a", ROWS_A, embedder)
    make_collection(db, "col_b", ROWS_B, embedder)
    make_collection(db, "col_empty", [], embedder)
    return db


@pytest.fixture
def cli(monkeypatch, embedder, capsys):
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: embedder)

    def run(cmd, query, db, *collections, **opts):
        argv = [cmd, query, "--db-path", db, "--model", embedder.model_key]
        for name in collections:
            argv += ["--collection", name]
        for flag, value in opts.items():
            argv += [f"--{flag.replace('_', '-')}", str(value)]
        capsys.readouterr()
        cli_main.main(argv)
        return capsys.readouterr().out

    return run


def _open(db, names, embedder, notes=None):
    return ms.open_collections(db, names, embedder, say=(notes.append if notes is not None else print))


# ── fusion ───────────────────────────────────────────────────────────────────

def _r(collection, doc_id, distance):
    return SearchResult(text=doc_id, distance=distance, doc_id=doc_id, metadata={"collection": collection})


def test_fusion_uses_ranks_only_and_ties_keep_collection_order():
    # the second list has distances 100x larger: only ranks may matter
    near = [_r("a", "a1", 0.01), _r("a", "a2", 0.02), _r("a", "a3", 0.03)]
    far = [_r("b", "b1", 1.0), _r("b", "b2", 2.0), _r("b", "b3", 3.0)]
    fused = ms.fuse_collections([near, far])
    assert [r.doc_id for r in fused] == ["a1", "b1", "a2", "b2", "a3", "b3"]
    assert [r.doc_id for r in ms.fuse_collections([far, near])] == ["b1", "a1", "b2", "a2", "b3", "a3"]


def test_fusion_never_merges_equal_ids_of_different_collections():
    fused = ms.fuse_collections([[_r("a", "same", 0.1)], [_r("b", "same", 0.9)]])
    assert [(r.metadata["collection"], r.doc_id) for r in fused] == [("a", "same"), ("b", "same")]


def test_rrf_constant_is_named_and_is_sixty():
    assert ms.RRF_K == 60
    score_gap = 1 / (60 + 1) - 1 / (60 + 2)
    a, b = ms.fuse_collections([[_r("a", "x", 0), _r("a", "y", 0)], [_r("b", "z", 0)]])[:2]
    assert (a.doc_id, b.doc_id) == ("x", "z") and score_gap > 0


# ── multi search over real collections ───────────────────────────────────────

@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_results_come_from_every_collection_and_carry_the_collection(three, embedder, method):
    opened = _open(three, ["col_a", "col_b"], embedder)
    results = ms.multi_search(QUERY, [c.store for c in opened], embedder, top_k=6, search_method=method)
    assert {r.metadata["collection"] for r in results} == {"col_a", "col_b"}
    assert all(r.metadata["source_type"] in ("3gpp", "camara") for r in results)


def test_same_id_in_two_collections_gives_two_results(three, embedder):
    opened = _open(three, ["col_a", "col_b"], embedder)
    results = ms.multi_search("amf registration mobility location geofence", [c.store for c in opened],
                              embedder, top_k=8)
    same = [r for r in results if r.doc_id == "a.md::fake64::0"]
    assert sorted(r.metadata["collection"] for r in same) == ["col_a", "col_b"]
    assert same[0].text != same[1].text


def test_dense_ranking_interleaves_by_rank(three, embedder):
    opened = _open(three, ["col_a", "col_b"], embedder)
    results = ms.multi_search(QUERY, [c.store for c in opened], embedder, top_k=6)
    assert [r.metadata["collection"] for r in results] == ["col_a", "col_b"] * 3


def test_query_is_embedded_once_for_all_collections(three, embedder):
    opened = _open(three, ["col_a", "col_b", "col_empty"], embedder)
    embedder.calls.clear()
    ms.multi_search(QUERY, [c.store for c in opened if c.count], embedder, top_k=4, search_method="hybrid")
    assert embedder.calls == [([QUERY], True)]
    embedder.calls.clear()
    ms.multi_search(QUERY, [c.store for c in opened if c.count], embedder, top_k=4, search_method="bm25")
    assert embedder.calls == []


def test_top_k_is_applied_after_fusion(three, embedder):
    opened = _open(three, ["col_a", "col_b"], embedder)
    results = ms.multi_search(QUERY, [c.store for c in opened], embedder, top_k=3)
    assert len(results) == 3


def test_rerank_sees_three_times_top_k_fused_candidates_and_keeps_rerank_top_k(three, embedder):
    seen = {}

    class Reranker:
        def rerank(self, query, results, top_k=5):
            seen["n"] = len(results)
            return list(reversed(results))[:top_k]

    opened = _open(three, ["col_a", "col_b"], embedder)
    results = ms.multi_search(QUERY, [c.store for c in opened], embedder, top_k=2, reranker=Reranker(),
                              rerank_top_k=1)
    assert seen["n"] == 6 and len(results) == 1


def test_parent_ids_are_not_merged_across_collections(db, embedder):
    parent = {"parent_id": "3gpp_P_0", "full_parent_text": "PARENT TEXT"}
    make_collection(db, "col_a", [{"id": "x::0", "text": "registration mobility one", "meta": parent}], embedder)
    make_collection(db, "col_b", [{"id": "y::0", "text": "registration mobility two", "meta": parent}], embedder)
    opened = _open(db, ["col_a", "col_b"], embedder)
    results = ms.multi_search(QUERY, [c.store for c in opened], embedder, top_k=4, resolve_parents=True)
    assert [r.metadata["collection"] for r in results] == ["col_a", "col_b"]
    assert all(r.text == "PARENT TEXT" for r in results)


# ── checks before searching ──────────────────────────────────────────────────

def test_empty_collection_contributes_nothing_and_prints_a_note(three, embedder):
    notes = []
    opened = _open(three, ["col_a", "col_empty", "col_b"], embedder, notes)
    assert [c.count for c in opened][1] == 0
    assert any("col_empty" in n and "empty" in n for n in notes)


def test_nonexistent_collection_is_an_error_and_is_not_created(three, embedder):
    with pytest.raises(ms.CollectionSetupError, match="'col_missing' does not exist"):
        _open(three, ["col_a", "col_missing"], embedder)
    import chromadb
    assert "col_missing" not in [c.name for c in chromadb.PersistentClient(path=three).list_collections()]


def test_missing_database_is_an_error_and_is_not_created(tmp_path, embedder):
    with pytest.raises(ms.CollectionSetupError, match="does not exist"):
        _open(str(tmp_path / "nope"), ["a", "b"], embedder)
    assert not (tmp_path / "nope").exists()


def test_different_distance_space_is_refused_naming_the_collection(db, embedder):
    make_collection(db, "col_a", ROWS_A, embedder, space="l2")
    make_collection(db, "col_cos", ROWS_B, embedder, space="cosine")
    with pytest.raises(ms.CollectionSetupError, match="'col_cos' uses distance space 'cosine'"):
        _open(db, ["col_a", "col_cos"], embedder)


def test_different_model_is_refused_naming_the_collection(db, embedder):
    make_collection(db, "col_a", ROWS_A, embedder)
    make_collection(db, "col_other", ROWS_B, embedder, sidecar_model="bge")
    with pytest.raises(ms.CollectionSetupError, match="'col_other' was ingested with model 'bge'"):
        _open(db, ["col_a", "col_other"], embedder)


def test_same_model_but_different_prefix_is_refused(db, monkeypatch, tmp_path):
    same = {"model_name": "org/model", "dimensions": 64}
    monkeypatch.setattr(ms, "_REGISTRY", registry(tmp_path / "models.json", {
        "plain": {**same}, "prefixed": {**same, "query_prefix": "query: "}}))
    embedder = FakeEmbedder("plain")
    make_collection(db, "col_a", ROWS_A, embedder)
    make_collection(db, "col_p", ROWS_B, embedder, sidecar_model="prefixed")
    with pytest.raises(ms.CollectionSetupError, match="'col_p'.*different prefixes"):
        _open(db, ["col_a", "col_p"], embedder)


def test_two_keys_with_identical_registry_entries_are_accepted(db, monkeypatch, tmp_path):
    entry = {"model_name": "org/model", "dimensions": 64}
    monkeypatch.setattr(ms, "_REGISTRY", registry(tmp_path / "models.json", {"k1": entry, "k2": dict(entry)}))
    embedder = FakeEmbedder("k1")
    make_collection(db, "col_a", ROWS_A, embedder)
    make_collection(db, "col_b", ROWS_B, embedder, sidecar_model="k2")
    assert len(_open(db, ["col_a", "col_b"], embedder)) == 2


def test_dimension_mismatch_names_the_embedder_and_both_collections(db):
    small, wide = FakeEmbedder("fake32", 32), FakeEmbedder("fake64", 64)
    make_collection(db, "col_32", ROWS_A, small, sidecar_model=None)
    make_collection(db, "col_64", ROWS_B, wide, sidecar_model=None)
    with pytest.raises(ms.CollectionSetupError) as err:
        _open(db, ["col_64", "col_32"], wide, [])
    message = str(err.value)
    assert "fake64" in message and "'col_32'" in message and "'col_64'" in message
    assert "32" in message and "64" in message


def test_collections_that_disagree_with_each_other_are_refused_even_without_the_embedder_size(db):
    one, two = FakeEmbedder("fake32", 32), FakeEmbedder("fake16", 16)
    make_collection(db, "col_32", ROWS_A, one, sidecar_model=None)
    make_collection(db, "col_16", ROWS_B, two, sidecar_model=None)
    bare = SimpleNamespace(model_key="x")
    with pytest.raises(ms.CollectionSetupError, match="'col_32'.*'col_16'"):
        _open(db, ["col_32", "col_16"], bare, [])


def test_missing_sidecar_falls_back_to_dimension_and_warns(db, embedder):
    make_collection(db, "col_a", ROWS_A, embedder, sidecar_model=None)
    make_collection(db, "col_b", ROWS_B, embedder)
    notes = []
    opened = _open(db, ["col_a", "col_b"], embedder, notes)
    assert len(opened) == 2 and opened[0].dimension == 64
    warnings = [n for n in notes if "[WARN]" in n]
    assert len(warnings) == 1 and "'col_a'" in warnings[0] and "cannot be checked" in warnings[0]


def test_comma_separated_names_are_rejected_with_a_clear_message():
    with pytest.raises(ms.CollectionSetupError, match="repeat --collection"):
        ms.validate_collection_names(["a,b"])
    assert ms.validate_collection_names(None) == [] and ms.validate_collection_names(["a", "b"]) == ["a", "b"]
    with pytest.raises(ms.CollectionSetupError, match="more than once"):
        ms.validate_collection_names(["a", "a"])


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_rejects_a_comma_list_for_search_and_ask_without_touching_chroma(three, cli):
    for cmd in ("search", "ask"):
        with pytest.raises(SystemExit) as exc:
            cli(cmd, QUERY, three, "col_a,col_b")
        assert "repeat --collection" in str(exc.value)


def test_cli_search_over_two_collections_prints_the_collection_and_keeps_equal_files_apart(three, cli):
    out = cli("search", "amf registration mobility location geofence", three, "col_a", "col_b", top_k=8)
    assert "| col_a | a.md |" in out and "| col_b | a.md |" in out
    assert "results returned for" in out


def test_cli_nonexistent_collection_among_several_exits_naming_it(three, cli):
    with pytest.raises(SystemExit) as exc:
        cli("search", QUERY, three, "col_a", "col_nope")
    assert "'col_nope' does not exist" in str(exc.value)


def test_cli_label_is_the_real_distance_space_per_collection(db, embedder, cli):
    make_collection(db, "col_l2", ROWS_A, embedder, space="l2")
    make_collection(db, "col_cos", ROWS_A, embedder, space="cosine")
    make_collection(db, "col_ip", ROWS_A, embedder, space="ip")
    assert " | L2 " in cli("search", QUERY, db, "col_l2", top_k=2)
    assert " | COSINE " in cli("search", QUERY, db, "col_cos", top_k=2)
    assert " | IP " in cli("search", QUERY, db, "col_ip", top_k=2)


def test_cli_label_for_several_collections_uses_their_shared_space(db, embedder, cli):
    make_collection(db, "col_c1", ROWS_A, embedder, space="cosine")
    make_collection(db, "col_c2", ROWS_B, embedder, space="cosine")
    out = cli("search", QUERY, db, "col_c1", "col_c2", top_k=4)
    assert " | COSINE " in out and " | L2 " not in out


def test_cli_all_collections_empty_prints_no_results(db, embedder, cli):
    make_collection(db, "col_e1", [], embedder)
    make_collection(db, "col_e2", [], embedder)
    out = cli("search", QUERY, db, "col_e1", "col_e2")
    assert "No results found." in out and "[NOTE]" in out


def test_cli_ask_over_several_collections_reaches_the_generator_with_each(three, embedder, monkeypatch, cli):
    seen = {}

    class Generator:
        def generate(self, question, results, source_type="telecom", **kw):
            seen["results"], seen["source_type"] = results, source_type
            return {"answer": "A", "sources": [r.metadata["source"] for r in results], "chunks_used": len(results)}

    monkeypatch.setattr(cli_main, "_get_generator", lambda *a, **k: Generator())
    out = cli("ask", QUERY, three, "col_a", "col_b", top_k=4)
    assert {r.metadata["collection"] for r in seen["results"]} == {"col_a", "col_b"}
    assert "Chunks used: 4" in out and seen["source_type"] == "oran"


# ── a single --collection (or none) is what it was ───────────────────────────

SINGLE_ROWS = [
    {"id": "a.md::fake64::0", "text": "the amf handles registration and mobility management", "source": "a.md",
     "meta": {"release": "Rel-18", "work_group": "WG1"}, "heading": "AMF"},
    {"id": "a.md::fake64::1", "text": "the smf handles session management and ip allocation", "source": "a.md",
     "heading": "SMF"},
    {"id": "b.md::fake64::0", "text": "the upf forwards user plane traffic for sessions", "source": "b.md",
     "heading": "UPF"},
    {"id": "c.md::fake64::0", "text": "mobility management registration update procedure", "source": "c.md",
     "heading": "MM"},
]

# printed by main (fdda583) for the collection built from SINGLE_ROWS
SINGLE_EXPECTED = {
    "dense": """
--- Result 1 | c.md | MM | similarity 0.479 | L2 0.735 ---
mobility management registration update procedure

--- Result 2 | a.md | WG1 | Rel-18 | AMF | similarity 0.394 | L2 0.931 ---
the amf handles registration and mobility management

--- Result 3 | a.md | SMF | similarity 0.212 | L2 1.553 ---
the smf handles session management and ip allocation

3 results returned for: 'registration mobility'
""",
    "bm25": """
--- Result 1 | c.md | MM | similarity 0.368 | L2 1.000 ---
mobility management registration update procedure

--- Result 2 | a.md | WG1 | Rel-18 | AMF | similarity 0.368 | L2 1.000 ---
the amf handles registration and mobility management

2 results returned for: 'registration mobility'
""",
    "hybrid": """
--- Result 1 | c.md | MM | similarity 0.368 | L2 1.000 ---
mobility management registration update procedure

--- Result 2 | a.md | WG1 | Rel-18 | AMF | similarity 0.368 | L2 1.000 ---
the amf handles registration and mobility management

--- Result 3 | a.md | SMF | similarity 0.212 | L2 1.553 ---
the smf handles session management and ip allocation

3 results returned for: 'registration mobility'
""",
}


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_single_collection_search_output_is_what_main_printed(db, embedder, cli, method):
    make_collection(db, "only_one", SINGLE_ROWS, embedder)
    assert cli("search", QUERY, db, "only_one", top_k=3, search_method=method) == SINGLE_EXPECTED[method]


def test_single_collection_ask_output_is_what_main_printed(db, embedder, monkeypatch, cli):
    make_collection(db, "only_one", SINGLE_ROWS, embedder)
    seen = {}

    class Generator:
        def generate(self, question, results, source_type="telecom", **kw):
            seen["results"] = results
            return {"answer": "A", "sources": [r.metadata.get("source") for r in results], "chunks_used": len(results)}

    monkeypatch.setattr(cli_main, "_get_generator", lambda *a, **k: Generator())
    out = cli("ask", QUERY, db, "only_one", top_k=3)
    assert out == "\nAnswer:\nA\n\nSources:     ['c.md', 'a.md', 'a.md']\nChunks used: 3\n"
    assert all("collection" not in r.metadata for r in seen["results"])


def test_no_collection_flag_still_uses_the_default_name(db, embedder, cli):
    make_collection(db, "oran_fake64", SINGLE_ROWS, embedder)
    assert cli("search", QUERY, db, top_k=3).startswith("\n--- Result 1 | c.md | MM")


def test_single_value_does_not_run_the_multi_collection_checks(db, embedder, cli):
    make_collection(db, "legacy_no_sidecar", SINGLE_ROWS, embedder, sidecar_model=None)
    out = cli("search", QUERY, db, "legacy_no_sidecar", top_k=3)
    assert "[WARN]" not in out and "[NOTE]" not in out


def test_sidecar_helper_writes_the_format_the_checks_read(db, embedder):
    make_collection(db, "col_a", ROWS_A, embedder)
    data = json.loads(next(__import__("pathlib").Path(db).glob(".ingest_settings_col_a.json")).read_text("utf8"))
    assert data["latest"]["settings"]["model_key"] == "fake64"


def test_extra_retrieval_kwargs_are_forwarded_to_every_collection(three, embedder, monkeypatch):
    calls = []
    real = ms.execute_retrieval

    def spy(*args, **kwargs):
        calls.append(kwargs)
        kwargs.pop("marker")
        return real(*args, **kwargs)

    monkeypatch.setattr(ms, "execute_retrieval", spy)
    opened = _open(three, ["col_a", "col_b"], embedder)
    ms.multi_search(QUERY, [c.store for c in opened], embedder, top_k=2, marker=True)
    assert len(calls) == 2
