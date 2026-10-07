"""--where-source-type: restrict search / ask to chunks of some source types (fake embedders)."""
from __future__ import annotations

import logging

import pytest

import cli.main as cli_main
from open_telco_rag.core import source_filter
from open_telco_rag.core.pipeline import search
from open_telco_rag.core.retrieval import execute_retrieval
from tests.fake_collections import FakeEmbedder, make_collection

QUERY = "registration session device"

GPP = [{"id": f"28{i}.md::fake64::0", "source": f"28{i}.md", "source_type": "3gpp",
        "text": f"the amf handles registration and session management for the ue number {i}"} for i in range(4)]
CAMARA = [{"id": f"api{i}/README.md::fake64::0", "source": "README.md", "source_type": "camara",
           "text": f"the device api {i} offers registration of a session for a device"} for i in range(4)]
ETSI = [{"id": "zsm.md::fake64::0", "source": "zsm.md", "source_type": "etsi",
         "text": "zero touch session automation for a closed loop"}]


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "db")


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def mixed(db, embedder):
    """one collection holding three sources"""
    return make_collection(db, "mixed", GPP + CAMARA + ETSI, embedder)


def _types(results):
    return {r.metadata["source_type"] for r in results}


# ── one collection, several sources ──────────────────────────────────────────

def test_unfiltered_search_sees_every_source(mixed, embedder):
    assert _types(search(QUERY, store=mixed, embedder=embedder, top_k=9)) == {"3gpp", "camara", "etsi"}


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_filter_keeps_only_the_wanted_source(mixed, embedder, method):
    results = search(QUERY, store=mixed, embedder=embedder, top_k=4, search_method=method,
                     source_types=["camara"])
    assert len(results) == 4 and _types(results) == {"camara"}


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_repeated_filter_allows_several_sources(mixed, embedder, method):
    results = search(QUERY, store=mixed, embedder=embedder, top_k=9, search_method=method,
                     source_types=["camara", "etsi"])
    assert _types(results) == {"camara", "etsi"}


def test_hybrid_filters_before_fusion(mixed, embedder):
    """the dense list and the BM25 list are both filtered, so fusion never sees another source"""
    seen = {}
    real = source_filter.chroma_where

    def spy(types):
        seen["where"] = real(types)
        return seen["where"]

    import open_telco_rag.core.retrieval as retrieval
    retrieval.chroma_where = spy
    try:
        results = execute_retrieval(QUERY, mixed, embedder, top_k=3, search_method="hybrid",
                                    source_types=["3gpp"])
    finally:
        retrieval.chroma_where = real
    assert _types(results) == {"3gpp"} and seen["where"] == {"source_type": "3gpp"}


def test_where_for_several_types_uses_in():
    assert source_filter.chroma_where(["a"]) == {"source_type": "a"}
    assert source_filter.chroma_where(["a", "b"]) == {"source_type": {"$in": ["a", "b"]}}
    assert source_filter.chroma_where(None) is None and source_filter.chroma_where([]) is None


def test_no_filter_never_passes_a_where_to_the_store(mixed, embedder):
    seen = []
    real = mixed.search
    mixed.search = lambda *a, **k: (seen.append(k), real(*a, **k))[1]
    search(QUERY, store=mixed, embedder=embedder, top_k=3)
    assert seen == [{"n_results": 3}]


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_no_filter_is_byte_identical(mixed, embedder, method):
    plain = search(QUERY, store=mixed, embedder=embedder, top_k=5, search_method=method)
    none = search(QUERY, store=mixed, embedder=embedder, top_k=5, search_method=method, source_types=None)
    assert [(r.doc_id, r.distance, r.text, r.metadata) for r in plain] == \
           [(r.doc_id, r.distance, r.text, r.metadata) for r in none]


# ── unknown source type ──────────────────────────────────────────────────────

def test_unknown_source_type_lists_the_available_values(mixed):
    with pytest.raises(source_filter.SourceTypeError) as err:
        source_filter.check_source_types([mixed], ["camarra"])
    message = str(err.value)
    assert "'camarra'" in message and "3gpp, camara, etsi" in message


def test_a_known_type_among_unknown_ones_still_fails(mixed):
    with pytest.raises(source_filter.SourceTypeError, match="'nope'"):
        source_filter.check_source_types([mixed], ["camara", "nope"])


def test_known_types_pass(mixed):
    source_filter.check_source_types([mixed], ["camara", "3gpp"])


# ── over-fetch and short lists ───────────────────────────────────────────────

@pytest.fixture
def rare(db, embedder):
    """60 common-source chunks that match the query better than the 5 rare-source ones"""
    common = [{"id": f"c{i}.md::fake64::0", "source": f"c{i}.md", "source_type": "3gpp",
               "text": "orchid orchid orchid quartz " + f"filler{i}"} for i in range(60)]
    few = [{"id": f"r{i}.md::fake64::0", "source": f"r{i}.md", "source_type": "etsi",
            "text": "orchid quartz " + " ".join(f"pad{j}" for j in range(12))} for i in range(5)]
    return make_collection(db, "rare", common + few, embedder)


def test_overfetch_factor_is_documented_and_ten():
    assert source_filter.BM25_OVERFETCH == 10


def test_bm25_with_a_rare_source_returns_a_short_list_and_warns(rare, embedder, caplog):
    with caplog.at_level(logging.WARNING):
        results = search("orchid quartz", store=rare, embedder=embedder, top_k=3, search_method="bm25",
                         source_types=["etsi"])
    assert len(results) < 3
    assert any("etsi" in m and "3" in m and "match" in m for m in caplog.messages)


def test_a_larger_overfetch_recovers_the_rare_source(rare, embedder, monkeypatch, caplog):
    monkeypatch.setattr(source_filter, "BM25_OVERFETCH", 30)
    import open_telco_rag.core.retrieval as retrieval
    monkeypatch.setattr(retrieval, "BM25_OVERFETCH", 30)
    with caplog.at_level(logging.WARNING):
        results = search("orchid quartz", store=rare, embedder=embedder, top_k=3, search_method="bm25",
                         source_types=["etsi"])
    assert len(results) == 3 and _types(results) == {"etsi"} and not caplog.messages


def test_dense_with_a_rare_source_is_never_short_while_enough_chunks_exist(rare, embedder, caplog):
    with caplog.at_level(logging.WARNING):
        results = search("orchid quartz", store=rare, embedder=embedder, top_k=3, source_types=["etsi"])
    assert len(results) == 3 and not caplog.messages


def test_short_dense_list_warns(mixed, embedder, caplog):
    with caplog.at_level(logging.WARNING):
        results = search(QUERY, store=mixed, embedder=embedder, top_k=5, source_types=["etsi"])
    assert len(results) == 1
    assert any("only 1 of 5" in m for m in caplog.messages)


# ── CLI ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def cli(monkeypatch, embedder, capsys):
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: embedder)

    def run(cmd, db, *flags, query=QUERY):
        capsys.readouterr()
        cli_main.main([cmd, query, "--db-path", db, "--model", "fake64", "--collection", "mixed", *flags])
        return capsys.readouterr().out

    return run


def test_cli_search_filter_flag(mixed, cli, db):
    out = cli("search", db, "--where-source-type", "3gpp", "--top-k", "4")
    assert "28" in out and "README.md" not in out


def test_cli_search_filter_flag_is_repeatable(mixed, cli, db):
    out = cli("search", db, "--where-source-type", "3gpp", "--where-source-type", "etsi", "--top-k", "9")
    assert "zsm.md" in out and "README.md" not in out


def test_cli_unknown_source_type_exits_with_the_available_values(mixed, cli, db):
    with pytest.raises(SystemExit) as exc:
        cli("search", db, "--where-source-type", "camarra")
    assert "'camarra'" in str(exc.value) and "3gpp, camara, etsi" in str(exc.value)


def test_cli_ask_gives_the_generator_only_the_wanted_source(mixed, cli, db, monkeypatch):
    seen = {}

    class Generator:
        def generate(self, question, results, source_type="telecom", **kw):
            seen["types"] = {r.metadata["source_type"] for r in results}
            return {"answer": "A", "sources": [], "chunks_used": len(results)}

    monkeypatch.setattr(cli_main, "_get_generator", lambda *a, **k: Generator())
    cli("ask", db, "--where-source-type", "camara", "--top-k", "4")
    assert seen["types"] == {"camara"}


def test_cli_without_the_flag_prints_what_it_printed_before(mixed, cli, db):
    assert "README.md" in cli("search", db, "--top-k", "9")


# ── across several collections (needs the multi-collection PR) ───────────────

def test_filter_across_two_collections(db, embedder, monkeypatch, capsys):
    pytest.importorskip("open_telco_rag.core.multi_search")
    make_collection(db, "col_a", GPP + CAMARA, embedder)
    make_collection(db, "col_b", CAMARA[:2] + ETSI, embedder)
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: embedder)
    cli_main.main(["search", QUERY, "--db-path", db, "--model", "fake64", "--collection", "col_a",
                   "--collection", "col_b", "--where-source-type", "camara", "--top-k", "9"])
    out = capsys.readouterr().out
    assert "README.md" in out and "zsm.md" not in out and "28" not in out.replace("fake64", "")
    assert "| col_a |" in out and "| col_b |" in out
