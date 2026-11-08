import pytest

import cli.main as cli_main
from open_telco_rag.challenge import cli as challenge_cli
from tests.fake_collections import FakeEmbedder
from tests.test_challenge_export import BIG_ID, FakeClient, build_index

KEEP = object()


def export_cli(tmp_path, monkeypatch, *extra, embedder=FakeEmbedder, questions=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "q.csv"
    path.write_text(
        questions
        or f'ID,question,options\n{BIG_ID},Which NF?,"{{""A"": ""AMF"", ""B"": ""SMF""}}"\n',
        encoding="utf-8",
    )
    if embedder is not KEEP:
        monkeypatch.setattr(challenge_cli, "_build_embedder", lambda settings: embedder())
    monkeypatch.setattr(challenge_cli, "_build_client", lambda *a, **k: FakeClient(["B"]))
    if not (tmp_path / "db").exists():
        build_index(tmp_path / "db")
    out = tmp_path / "o.csv"
    cli_main.main(
        ["challenge", "export", "--questions", str(path), "--out", str(out)]
        + [
            "--db-path",
            str(tmp_path / "db"),
            "--collection",
            "col",
            "--no-rerank",
            "--allow-embedder-mismatch",
            "--top-k",
            "5",
            *extra,
        ]
    )
    return out


def test_reranker_model_flag_selects_the_reranker(tmp_path, monkeypatch):
    seen = []

    class Reranker:
        def rerank(self, query, results, top_k=5):
            return results[:top_k]

    monkeypatch.setattr(
        "open_telco_rag.embedders.reranker.load_reranker",
        lambda key: seen.append(key) or Reranker(),
    )
    export_cli(tmp_path, monkeypatch, "--rerank", "--reranker-model", "my-reranker")
    assert seen == ["my-reranker"]


def test_unwritable_output_path_is_a_one_line_error(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match=r"^error: cannot write .*partial"):
        export_cli(tmp_path, monkeypatch, "--out", str(tmp_path))


def test_unknown_embedder_key_is_a_one_line_error(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match=r"^error: cannot load the embedder: .*nope"):
        export_cli(tmp_path, monkeypatch, "--model", "nope", embedder=KEEP)


def test_unknown_reranker_key_is_a_one_line_error(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match=r"^error: cannot load the reranker: .*nope"):
        export_cli(tmp_path, monkeypatch, "--rerank", "--reranker-model", "nope")


def test_embedder_failure_is_not_blamed_on_a_question(tmp_path, monkeypatch):
    class Broken:
        def embed(self, texts, is_query=False, **kw):
            raise RuntimeError("weights missing")

    with pytest.raises(SystemExit) as info:
        export_cli(tmp_path, monkeypatch, embedder=Broken)
    assert str(info.value).startswith("error: cannot run the embedder: weights missing")
    assert "question" not in str(info.value)


@pytest.mark.parametrize("seed", ["-5", str(2**32), str(2**40)])
def test_out_of_range_seed_is_rejected_by_the_parser(tmp_path, monkeypatch, seed, capsys):
    with pytest.raises(SystemExit) as info:
        export_cli(tmp_path, monkeypatch, "--seed", seed)
    assert info.value.code == 2 and "--seed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--temperature", "-1"),
        ("--max-tokens", "0"),
        ("--max-tokens", "-1"),
        ("--top-k", "0"),
        ("--rerank-top-k", "0"),
        ("--max-prompt-tokens", "0"),
        ("--max-prompt-tokens", "-5"),
        ("--embedder-dimensions", "-1"),
    ],
)
def test_nonsensical_numbers_are_rejected_by_the_parser(tmp_path, monkeypatch, flag, value, capsys):
    with pytest.raises(SystemExit) as info:
        export_cli(tmp_path, monkeypatch, flag, value)
    assert info.value.code == 2 and flag in capsys.readouterr().err


def test_a_raising_plugin_aborts_with_the_question_id_not_a_traceback(tmp_path):
    from open_telco_rag.challenge.export import ExportAborted, ExportOptions, export_submission
    from open_telco_rag.challenge.questions import ChallengeQuestion

    def raising_builder(question, passages, cap):
        raise RuntimeError("boom in plugin")

    def raising_parser(text, options):
        raise ValueError("parser boom")

    questions = [ChallengeQuestion(BIG_ID, "Q?", {"A": "x"})]
    for field, hook, text in (
        ("prompt_builder", raising_builder, "prompt builder raised RuntimeError: boom in plugin"),
        ("answer_parser", raising_parser, "answer parser raised ValueError: parser boom"),
    ):
        options = ExportOptions(max_prompt_tokens=4096, **{field: hook})
        with pytest.raises(ExportAborted, match=f"question {BIG_ID}: {text}"):
            export_submission(
                questions, lambda q: [], FakeClient(["B"]), tmp_path / "o.csv", options
            )
        assert (tmp_path / "o.csv.partial").exists()


def test_score_with_a_missing_file_is_a_one_line_error(tmp_path):
    for submission, reference in (("nope.csv", "ref.csv"), ("sub.csv", "nope.csv")):
        (tmp_path / "sub.csv").write_text('"ID","input_prompt","answer"\n', encoding="utf-8")
        (tmp_path / "ref.csv").write_text("ID,answer\n1,A\n", encoding="utf-8")
        with pytest.raises(SystemExit, match=r"^error: .*(nope\.csv)"):
            cli_main.main(
                [
                    "challenge",
                    "score",
                    "--submission",
                    str(tmp_path / submission),
                    "--reference",
                    str(tmp_path / reference),
                ]
            )


def test_test_csv_without_an_options_column_is_an_error(tmp_path):
    from open_telco_rag.challenge.questions import load_questions

    path = tmp_path / "t.csv"
    path.write_text("ID,question\n1,Which NF?\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing options column"):
        load_questions(path)


def test_test_csv_without_rows_is_an_error(tmp_path):
    from open_telco_rag.challenge.questions import load_questions

    path = tmp_path / "t.csv"
    path.write_text("ID,question,options\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no questions"):
        load_questions(path)


def test_check_names_the_bom_instead_of_blaming_the_header(tmp_path):
    from open_telco_rag.challenge.check import check_submission

    questions = tmp_path / "t.csv"
    questions.write_text("ID,question,options\n1,Q?,{}\n", encoding="utf-8")
    sub = tmp_path / "s.csv"
    sub.write_bytes(b'\xef\xbb\xbf"ID","input_prompt","answer"\n"1","a prompt","TRUE"\n')
    issues = check_submission(sub, questions).issues
    assert issues == ["file starts with a UTF-8 BOM"]


def test_export_command_never_imports_scoring_modules(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "run_export.py"
    script.write_text(
        "import sys\n"
        "import cli.main as m\n"
        "from open_telco_rag.challenge import cli as c\n"
        "from tests.fake_collections import FakeEmbedder\n"
        "from tests.test_challenge_export import FakeClient, build_index\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1]); build_index(root / 'db')\n"
        "(root / 'q.csv').write_text('ID,question,options\\n1,Q?,{}\\n', encoding='utf-8')\n"
        "c._build_embedder = lambda settings: FakeEmbedder()\n"
        "c._build_client = lambda *a, **k: FakeClient(['TRUE'])\n"
        "m.main(['challenge', 'export', '--questions', str(root / 'q.csv'), '--out', str(root / 'o.csv'),\n"
        "        '--db-path', str(root / 'db'), '--collection', 'col',\n"
        "        '--no-rerank', '--allow-embedder-mismatch'])\n"
        "bad = [n for n in sys.modules if n in ('accuracy', 'token_efficiency', 'overall_score')\n"
        "       or n.startswith('open_telco_rag.challenge.metrics') or n.endswith(('challenge.score', 'challenge.check'))]\n"
        "print('IMPORTED', bad)\n",
        encoding="utf-8",
    )
    repo = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
        env={**__import__("os").environ, "PYTHONPATH": repo},
    )
    assert done.returncode == 0, done.stderr[-500:]
    assert "IMPORTED []" in done.stdout


def test_default_cap_equals_the_official_cap_without_importing_it():
    from open_telco_rag.challenge.caps import official_cap
    from open_telco_rag.challenge.metrics import token_efficiency

    assert official_cap() == token_efficiency.MAX_PROMPT_TOKENS


def test_any_method_other_than_dense_needs_the_bm25_side(tmp_path):
    from open_telco_rag.challenge.index import IndexProblem, open_index
    from tests.test_challenge_export import build_index

    db = build_index(tmp_path, with_fts=False)
    with pytest.raises(IndexProblem, match="col_fts.db"):
        open_index(db, "col", "my-new-fusion")
    assert not (db / "col_fts.db").exists()


def test_cap_above_the_official_cap_warns(tmp_path, monkeypatch, capsys):
    export_cli(tmp_path, monkeypatch, "--max-prompt-tokens", "100000")
    assert "above the official cap of 4096" in capsys.readouterr().err


def test_failed_run_says_that_an_older_output_file_is_stale(tmp_path, monkeypatch):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "o.csv").write_text("old complete file", encoding="utf-8")
    monkeypatch.setattr(challenge_cli, "_build_client", lambda *a, **k: None)
    with pytest.raises(SystemExit) as info:
        export_cli(tmp_path, monkeypatch, "--max-prompt-tokens", "10")
    assert f"{tmp_path / 'o.csv'} is from an earlier run and was not updated" in str(info.value)


def test_bad_hook_spec_is_reported_before_any_model_is_loaded(tmp_path, monkeypatch):
    class MustNotLoad:
        def __init__(self):
            raise AssertionError("embedder loaded before the hook spec was checked")

    with pytest.raises(SystemExit, match="package.module:function"):
        export_cli(tmp_path, monkeypatch, "--prompt-builder", "nocolon", embedder=MustNotLoad)


@pytest.mark.parametrize(
    "flag",
    ["--seed", "--top-k", "--rerank-top-k", "--max-tokens", "--max-prompt-tokens", "--temperature"],
)
def test_non_numeric_value_message_is_readable(tmp_path, monkeypatch, flag, capsys):
    with pytest.raises(SystemExit):
        export_cli(tmp_path, monkeypatch, flag, "abc")
    err = capsys.readouterr().err
    assert (
        f"argument {flag}: invalid" in err
        and "'abc'" in err
        and "_" not in err.split("invalid")[1].split("value")[0]
    )


def test_wrong_embedder_for_the_index_says_what_to_pass(tmp_path, monkeypatch):
    class Wide(FakeEmbedder):
        dimensions = 1024

        def embed(self, texts, is_query=False, **kw):
            return [[0.1] * 1024 for _ in texts]

    with pytest.raises(SystemExit) as info:
        export_cli(tmp_path, monkeypatch, embedder=Wide)
    message = str(info.value)
    assert message.startswith(
        "error: the embedder produces 1024-dimensional vectors but the index holds 64-dimensional ones"
    )
    assert "pass the model the index was built with" in message and "question" not in message
