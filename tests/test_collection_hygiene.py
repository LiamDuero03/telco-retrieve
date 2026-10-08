"""search/ask never create a collection, info --collection, delete of a missing collection, --no-resume
only touches the BM25 rows of the run, and the 'same relative path, second source skipped' warning."""
from __future__ import annotations

import chromadb
import pytest

import cli.main as cli_main
from open_telco_rag.core.pipeline import ingest
from open_telco_rag.sources.camara import CAMARASource
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore
from open_telco_rag.stores.fts import FTSStore
from tests.fake_collections import FakeEmbedder, in_memory_chroma, make_collection  # noqa: F401
from tests.test_retrieval_methods import BagOfWordsEmbedder

ROWS = [{"id": "a.md::fake64::0", "text": "the amf handles registration", "source": "a.md"}]


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "db")


@pytest.fixture
def cli(monkeypatch, capsys):
    embedder = FakeEmbedder()
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: embedder)

    def run(*argv):
        capsys.readouterr()
        cli_main.main(list(argv))
        return capsys.readouterr().out

    return run


def _names(db):
    return sorted(c.name for c in chromadb.PersistentClient(path=db).list_collections())


# ── (i) search / ask do not create collections ───────────────────────────────

@pytest.mark.parametrize("cmd", ["search", "ask"])
def test_missing_named_collection_is_an_error_and_is_not_created(db, cli, cmd):
    make_collection(db, "real_fake64", ROWS, FakeEmbedder())
    with pytest.raises(SystemExit) as exc:
        cli(cmd, "registration", "--db-path", db, "--model", "fake64", "--collection", "nope_fake64")
    assert "'nope_fake64' does not exist" in str(exc.value) and "real_fake64" in str(exc.value)
    assert _names(db) == ["real_fake64"]


@pytest.mark.parametrize("cmd", ["search", "ask"])
def test_missing_default_collection_is_an_error_not_an_empty_oran_collection(db, cli, cmd):
    make_collection(db, "camara_fake64", ROWS, FakeEmbedder())
    with pytest.raises(SystemExit) as exc:
        cli(cmd, "registration", "--db-path", db, "--model", "fake64")
    assert "'oran_fake64' does not exist" in str(exc.value)
    assert _names(db) == ["camara_fake64"]


def test_missing_database_is_an_error_and_is_not_created(tmp_path, cli):
    with pytest.raises(SystemExit) as exc:
        cli("search", "registration", "--db-path", str(tmp_path / "nope"), "--model", "fake64")
    assert "does not exist" in str(exc.value) and not (tmp_path / "nope").exists()


def test_existing_collection_still_searches(db, cli):
    make_collection(db, "real_fake64", ROWS, FakeEmbedder())
    out = cli("search", "registration", "--db-path", db, "--model", "fake64", "--collection", "real_fake64")
    assert "a.md" in out


# ── (ii) info --collection ───────────────────────────────────────────────────

def test_info_collection_shows_only_that_collection(db, cli):
    make_collection(db, "one_fake64", ROWS, FakeEmbedder())
    make_collection(db, "two_fake64", ROWS * 1 + [{"id": "b.md::fake64::0", "text": "x y z", "source": "b"}],
                    FakeEmbedder())
    out = cli("info", "--db-path", db, "--collection", "two_fake64")
    assert "two_fake64: 2 chunks" in out and "one_fake64" not in out
    both = cli("info", "--db-path", db)
    assert "one_fake64: 1 chunks" in both and "two_fake64: 2 chunks" in both


def test_info_missing_collection_exits_non_zero_naming_it(db, cli):
    make_collection(db, "one_fake64", ROWS, FakeEmbedder())
    with pytest.raises(SystemExit) as exc:
        cli("info", "--db-path", db, "--collection", "nope_fake64")
    assert exc.value.code not in (0, None) and "nope_fake64" in str(exc.value)


# ── (iv) delete of a missing collection ──────────────────────────────────────

def test_delete_missing_collection_exits_non_zero(db, cli):
    make_collection(db, "one_fake64", ROWS, FakeEmbedder())
    with pytest.raises(SystemExit) as exc:
        cli("delete", "nope_fake64", "--db-path", db)
    assert exc.value.code not in (0, None)
    assert _names(db) == ["one_fake64"]


def test_delete_existing_collection_still_exits_zero_and_removes_everything(db, cli):
    make_collection(db, "one_fake64", ROWS, FakeEmbedder())
    assert "Deleted collection" in cli("delete", "one_fake64", "--db-path", db)
    assert _names(db) == []


# ── (iii) --no-resume and the BM25 index ─────────────────────────────────────

CHUNK = {"strategy": "text_baseline", "chunk_size": 200, "overlap": 0, "min_length": 10}


@pytest.fixture
def two_sources(tmp_path):
    """a 3GPP-layout corpus and a CAMARA-layout corpus with different relative paths"""
    gpp = tmp_path / "tgpp" / "Rel-18" / "28_series"
    gpp.mkdir(parents=True)
    for i in range(2):
        (gpp / f"2831{i}-i11.md").write_text(
            "# Spec\n\n" + "\n\n".join(f"Clause {i}.{j} says the amf shall support registration procedure {j}."
                                       for j in range(6)), encoding="utf8")
    cam = tmp_path / "camara"
    for repo in ("QualityOnDemand", "DeviceLocation"):
        (cam / repo).mkdir(parents=True)
        (cam / repo / "README.md").write_text(
            "# " + repo + "\n\n" + "\n\n".join(f"The {repo} api offers boost procedure {j} for a device." for j in range(6)),
            encoding="utf8")
    return tmp_path / "tgpp", cam


def _ingest(tmp_path, source, data, resume=True):
    store = ChromaStore(path=tmp_path / "db", collection_name="shared", metadata={"hnsw:space": "l2"})
    stats = ingest(source=source, store=store, embedder=BagOfWordsEmbedder(), data_dir=str(data), resume=resume,
                   log_dir=str(tmp_path / "logs"), batch_size=8, **CHUNK)
    return store, stats


def _fts(tmp_path):
    fts = FTSStore(path=tmp_path / "db", collection_name="shared")
    return fts, fts._get_conn().execute("SELECT COUNT(*), COUNT(DISTINCT doc_id) FROM bm25_index").fetchone()


def test_no_resume_of_a_second_source_keeps_the_first_sources_bm25_rows(tmp_path, two_sources, capsys):
    gpp, cam = two_sources
    store, _ = _ingest(tmp_path, TPPSource(), gpp)
    first = store.count()
    store, _ = _ingest(tmp_path, CAMARASource(), cam, resume=False)
    assert store.count() > first
    fts, (rows, distinct) = _fts(tmp_path)
    assert rows == distinct == store.count(), "BM25 and Chroma hold the same chunks"
    assert fts.search("amf registration", top_k=3), "the 3GPP chunks are still found by BM25"


def test_no_resume_reports_how_many_bm25_rows_it_removed(tmp_path, two_sources, capsys):
    gpp, _ = two_sources
    store, _ = _ingest(tmp_path, TPPSource(), gpp)
    capsys.readouterr()
    _, _ = _ingest(tmp_path, TPPSource(), gpp, resume=False)
    out = capsys.readouterr().out
    assert f"removed {store.count()} BM25 rows" in out
    _, (rows, distinct) = _fts(tmp_path)
    assert rows == distinct == store.count()


def test_no_resume_on_a_fresh_collection_removes_nothing(tmp_path, two_sources, capsys):
    gpp, _ = two_sources
    _ingest(tmp_path, TPPSource(), gpp, resume=False)
    assert "removed" not in capsys.readouterr().out


# ── (v) same relative path under another source type ─────────────────────────

def test_second_source_with_the_same_relative_path_is_skipped_with_a_warning(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, word in ((a, "alpha"), (b, "beta")):
        d.mkdir()
        (d / "README.md").write_text(f"# {word}\n\n" + f"The {word} routine records orchid events. " * 8, encoding="utf8")
    store, _ = _ingest(tmp_path, TPPSource(), a)
    capsys.readouterr()
    store, second = _ingest(tmp_path, CAMARASource(), b)
    out = capsys.readouterr().out
    assert second.skipped_files == 1 and second.files_processed == 0
    assert "README.md" in out and "'3gpp'" in out and "'camara'" in out and "skipped" in out
    stored = store._collection.get(include=["metadatas", "documents"])
    assert {m["source_type"] for m in stored["metadatas"]} == {"3gpp"}, "what is written is unchanged"
    assert all("alpha" in d for d in stored["documents"])


def test_a_plain_resume_of_the_same_source_does_not_warn(tmp_path, two_sources, capsys):
    gpp, _ = two_sources
    _ingest(tmp_path, TPPSource(), gpp)
    capsys.readouterr()
    _, stats = _ingest(tmp_path, TPPSource(), gpp)
    assert stats.skipped_files == 2 and "[WARN]" not in capsys.readouterr().out
