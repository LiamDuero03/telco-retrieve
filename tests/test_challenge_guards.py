import pytest

from tests.test_challenge_defaults import cli_export
from tests.test_challenge_export import build_index


def test_embedder_other_than_the_index_one_aborts_with_what_to_pass(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as info:
        cli_export(tmp_path, monkeypatch, allow=False)
    assert str(info.value) == (
        "error: the index was built with 'fake64' but the embedder is 'otel-0.6b'; "
        "pass --model fake64, or --allow-embedder-mismatch to continue anyway"
    )


def test_matching_model_key_passes(tmp_path, monkeypatch):
    cli_export(tmp_path, monkeypatch, "--model", "fake64", allow=False)
    assert (tmp_path / "o.csv").exists()


def test_allow_flag_turns_the_abort_into_a_warning(tmp_path, monkeypatch, capsys):
    cli_export(tmp_path, monkeypatch)
    assert "[WARN] the index was built with 'fake64'" in capsys.readouterr().err
    assert (tmp_path / "o.csv").exists()


def test_same_model_passed_by_its_hugging_face_id_matches_the_registry_key(tmp_path, monkeypatch):
    def otel_index(path):
        build_index(path, sidecar="otel-0.6b")

    cli_export(
        tmp_path,
        monkeypatch,
        "--embedder-id",
        "farbodtavakkoli/OTel-Embedding-0.6B",
        index=otel_index,
        allow=False,
    )
    assert (tmp_path / "o.csv").exists()
    with pytest.raises(
        SystemExit, match="the index was built with 'otel-0.6b' but the embedder is"
    ):
        cli_export(
            tmp_path / "other",
            monkeypatch,
            "--embedder-path",
            "/my/model",
            index=otel_index,
            allow=False,
        )


def test_index_without_a_sidecar_falls_back_to_the_chunk_ids(tmp_path, monkeypatch):
    def no_sidecar(path):
        build_index(path, sidecar=None)

    with pytest.raises(SystemExit, match="built with 'fake64'"):
        cli_export(tmp_path, monkeypatch, index=no_sidecar, allow=False)


def test_index_without_any_recorded_id_keeps_the_dimension_check_and_warns(
    tmp_path, monkeypatch, capsys
):
    def anonymous(path):
        build_index(path, sidecar=None, plain_ids=True)

    cli_export(tmp_path, monkeypatch, index=anonymous, allow=False)
    assert "records no embedder id; only the vector size is checked" in capsys.readouterr().err
    assert (tmp_path / "o.csv").exists()


@pytest.mark.parametrize(
    "argv,message",
    [
        (
            ["--no-rerank", "--rerank-top-k", "2"],
            "--rerank-top-k has no effect because reranking is off; add --rerank",
        ),
        (
            ["--no-rerank", "--reranker-model", "x"],
            "--reranker-model has no effect because reranking is off; add --rerank",
        ),
        (
            ["--no-rerank", "--rerank-top-k", "2", "--reranker-model", "x"],
            "--rerank-top-k and --reranker-model have no effect because reranking is off",
        ),
    ],
)
def test_rerank_options_without_rerank_are_an_error(tmp_path, monkeypatch, argv, message):
    with pytest.raises(SystemExit) as info:
        cli_export(tmp_path, monkeypatch, *argv)
    assert str(info.value).startswith(f"error: {message}")
    assert not (tmp_path / "o.csv").exists()


def test_rerank_options_with_rerank_on_are_fine(tmp_path, monkeypatch):
    cli_export(tmp_path, monkeypatch, "--rerank-top-k", "2", "--reranker-model", "x")
    assert (tmp_path / "o.csv").exists()


def test_resolve_parents_needs_an_index_with_parents(tmp_path, monkeypatch):
    with pytest.raises(
        SystemExit, match="--resolve-parents needs an index built with the parent_child"
    ):
        cli_export(tmp_path, monkeypatch, "--resolve-parents")
    cli_export(
        tmp_path / "p",
        monkeypatch,
        "--resolve-parents",
        index=lambda path: build_index(path, parents=True),
    )
    assert (tmp_path / "p" / "o.csv").exists()
