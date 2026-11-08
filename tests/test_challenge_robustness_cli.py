import csv
import io

import pytest

from open_telco_rag.challenge.client import Completion, GeneratorError
from open_telco_rag.challenge.export import (
    ExportAborted,
    ExportOptions,
    export_submission,
)
from open_telco_rag.challenge.questions import ChallengeQuestion
from open_telco_rag.types import SearchResult
from tests.test_challenge_defaults import cli_export

QUESTIONS = [
    ChallengeQuestion(f"{i:03d}", f"Question {i}?", {"A": "x", "B": "y"}) for i in range(5)
]


def retrieve(query):
    return [SearchResult(text=f"passage for {query}", distance=0.1, metadata={}, doc_id="d")]


class Scripted:
    """Answers B; raises the given error on the listed 1-based call numbers."""

    def __init__(self, fail_on=(), error=None):
        self.fail_on, self.error, self.calls = set(fail_on), error, 0

    def complete(self, messages, **settings):
        self.calls += 1
        if self.calls in self.fail_on:
            raise self.error
        return Completion("B", messages, False)


def rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


class Response:
    def __init__(self, status):
        self.status_code = status

    def json(self):
        return {"choices": [{"message": {"content": "B"}}]}


def test_progress_goes_to_stderr_and_stdout_stays_one_line(tmp_path, monkeypatch, capsys):
    body = "ID,question,options\n" + "".join(
        f'{i:03d},Q{i}?,"{{""A"": ""x""}}"\n' for i in range(26)
    )
    cli_export(tmp_path, monkeypatch, questions=body)
    captured = capsys.readouterr()
    assert "[progress] 25/26 questions" in captured.err
    assert captured.out.strip().splitlines() == [captured.out.strip()] and captured.out.startswith(
        "Wrote 26 rows"
    )


def interrupted(tmp_path, stop_after=2):
    client = Scripted(fail_on={stop_after + 1}, error=GeneratorError("down", retryable=False))
    with pytest.raises(ExportAborted):
        export_submission(QUESTIONS, retrieve, client, tmp_path / "o.csv", ExportOptions(4096))
    return tmp_path / "o.csv.partial"


def reference(tmp_path):
    export_submission(QUESTIONS, retrieve, Scripted(), tmp_path / "ref.csv", ExportOptions(4096))
    return (tmp_path / "ref.csv").read_bytes()


def corrupt(partial, how):
    text = partial.read_text(encoding="utf-8")
    if how == "header":
        partial.write_text(text.replace('"ID"', '"id"', 1), encoding="utf-8")
    elif how == "ids":
        partial.write_text(text.replace('"001"', '"999"', 1), encoding="utf-8")
    elif how == "too many":
        buffer = io.StringIO(newline="")
        parsed = list(csv.reader(io.StringIO(text, newline="")))
        csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(
            parsed + [[f"{i:03d}", "p", "A"] for i in range(2, 9)]
        )
        partial.write_text(buffer.getvalue(), encoding="utf-8")
    elif how == "binary":
        partial.write_bytes(b"\xff\xfe\x00 not utf8")
    elif how == "garbage tail":
        partial.write_text(text + "not a row at all", encoding="utf-8")
    elif how == "empty":
        partial.write_text("", encoding="utf-8")


def test_cli_resume_continues_and_says_so(tmp_path, monkeypatch, capsys):
    import cli.main as cli_main
    from open_telco_rag.challenge import cli as challenge_cli
    from tests.fake_collections import FakeEmbedder
    from tests.test_challenge_export import build_index

    body = "ID,question,options\n" + "".join(
        f'{i:03d},Q{i}?,"{{""A"": ""x""}}"\n' for i in range(4)
    )
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "q.csv").write_text(body, encoding="utf-8")
    build_index(tmp_path / "db")
    monkeypatch.setattr(challenge_cli, "_build_embedder", lambda settings: FakeEmbedder())
    monkeypatch.setattr(
        "open_telco_rag.embedders.reranker.load_reranker",
        lambda key: type("R", (), {"rerank": lambda s, q, r, top_k=5: r[:top_k]})(),
    )
    argv = [
        "challenge",
        "export",
        "--questions",
        str(tmp_path / "q.csv"),
        "--out",
        str(tmp_path / "o.csv"),
        "--db-path",
        str(tmp_path / "db"),
        "--collection",
        "col",
        "--allow-embedder-mismatch",
    ]
    client = Scripted(fail_on={3}, error=GeneratorError("down", retryable=False))
    monkeypatch.setattr(challenge_cli, "_build_client", lambda *a, **k: client)
    with pytest.raises(SystemExit):
        cli_main.main(argv)
    capsys.readouterr()
    client2 = Scripted()
    monkeypatch.setattr(challenge_cli, "_build_client", lambda *a, **k: client2)
    cli_main.main(argv + ["--resume"])
    err = capsys.readouterr().err
    assert "[resume] kept 2 finished rows from the partial file" in err
    assert client2.calls == 2 and [r[0] for r in rows(tmp_path / "o.csv")[1:]] == [
        "000",
        "001",
        "002",
        "003",
    ]
