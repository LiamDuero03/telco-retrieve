import csv
import io

import pytest
import requests

from open_telco_rag.challenge.client import ChatClient, Completion, GeneratorError
from open_telco_rag.challenge.export import (
    ExportAborted,
    ExportOptions,
    export_submission,
)
from open_telco_rag.challenge.questions import ChallengeQuestion
from open_telco_rag.types import SearchResult

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


@pytest.mark.parametrize(
    "outcome,retryable",
    [
        (Response(429), True),
        (Response(500), True),
        (Response(503), True),
        (Response(400), False),
        (Response(401), False),
        (Response(404), False),
        (requests.exceptions.Timeout("slow"), True),
        (requests.exceptions.ConnectionError("down"), True),
        (requests.exceptions.InvalidURL("bad"), False),
        (requests.exceptions.MissingSchema("bad"), False),
    ],
)
def test_only_transient_failures_are_marked_retryable(monkeypatch, outcome, retryable):
    def post(*args, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("open_telco_rag.challenge.client.requests.post", post)
    with pytest.raises(GeneratorError) as info:
        ChatClient("m", base_url="http://x/v1").complete([{"role": "user", "content": "q"}])
    assert info.value.retryable is retryable


def test_timeout_reaches_the_request(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "open_telco_rag.challenge.client.requests.post",
        lambda *a, **kw: seen.append(kw["timeout"]) or Response(200),
    )
    ChatClient("m", base_url="http://x/v1", timeout=7).complete([{"role": "user", "content": "q"}])
    ChatClient("m", base_url="http://x/v1").complete([{"role": "user", "content": "q"}])
    assert seen == [7, 120]


def test_three_retries_with_backoff_then_success(tmp_path):
    sleeps, client = [], Scripted(fail_on={1, 2, 3}, error=GeneratorError("HTTP 503"))
    export_submission(
        QUESTIONS[:1],
        retrieve,
        client,
        tmp_path / "o.csv",
        ExportOptions(4096, sleep=sleeps.append),
    )
    assert client.calls == 4 and sleeps == [1.0, 2.0, 4.0]
    assert rows(tmp_path / "o.csv")[1][2] == "B"


def test_after_three_retries_the_run_aborts_naming_the_question_and_writes_no_blank_row(tmp_path):
    client = Scripted(fail_on=set(range(2, 100)), error=GeneratorError("HTTP 429"))
    with pytest.raises(ExportAborted, match="question 001: generator failed after 4 attempt"):
        export_submission(
            QUESTIONS,
            retrieve,
            client,
            tmp_path / "o.csv",
            ExportOptions(4096, sleep=lambda s: None),
        )
    partial = rows(tmp_path / "o.csv.partial")
    assert [r[0] for r in partial[1:]] == ["000"] and all(r[2] for r in partial[1:])
    assert not (tmp_path / "o.csv").exists()


def test_a_permanent_error_is_not_retried(tmp_path):
    sleeps, client = [], Scripted(fail_on={1}, error=GeneratorError("HTTP 400", retryable=False))
    with pytest.raises(ExportAborted, match="question 000: generator failed after 1 attempt"):
        export_submission(
            QUESTIONS,
            retrieve,
            client,
            tmp_path / "o.csv",
            ExportOptions(4096, sleep=sleeps.append),
        )
    assert client.calls == 1 and sleeps == []


def test_progress_is_reported_every_25_questions(tmp_path):
    many = [ChallengeQuestion(f"{i:03d}", f"Q{i}?", {"A": "x"}) for i in range(60)]
    seen = []
    export_submission(
        many,
        retrieve,
        Scripted(),
        tmp_path / "o.csv",
        ExportOptions(4096, progress=lambda d, t: seen.append((d, t))),
    )
    assert seen == [(25, 60), (50, 60)]


def interrupted(tmp_path, stop_after=2):
    client = Scripted(fail_on={stop_after + 1}, error=GeneratorError("down", retryable=False))
    with pytest.raises(ExportAborted):
        export_submission(QUESTIONS, retrieve, client, tmp_path / "o.csv", ExportOptions(4096))
    return tmp_path / "o.csv.partial"


def reference(tmp_path):
    export_submission(QUESTIONS, retrieve, Scripted(), tmp_path / "ref.csv", ExportOptions(4096))
    return (tmp_path / "ref.csv").read_bytes()


def test_resume_completes_without_duplicates_or_skips_and_equals_an_uninterrupted_run(tmp_path):
    interrupted(tmp_path)
    client = Scripted()
    result = export_submission(
        QUESTIONS, retrieve, client, tmp_path / "o.csv", ExportOptions(4096, resume=True)
    )
    assert client.calls == 3 and result.resumed_rows == 2 and result.n_rows == 5
    assert (tmp_path / "o.csv").read_bytes() == reference(tmp_path)
    assert not (tmp_path / "o.csv.partial").exists()


def test_resume_drops_a_row_cut_in_half_and_redoes_it(tmp_path):
    partial = interrupted(tmp_path)
    raw = partial.read_bytes()
    partial.write_bytes(raw[: len(raw) - 25])
    client = Scripted()
    result = export_submission(
        QUESTIONS, retrieve, client, tmp_path / "o.csv", ExportOptions(4096, resume=True)
    )
    assert result.resumed_rows == 1 and client.calls == 4
    assert (tmp_path / "o.csv").read_bytes() == reference(tmp_path)


def test_resume_without_a_partial_starts_from_the_beginning(tmp_path):
    client = Scripted()
    result = export_submission(
        QUESTIONS, retrieve, client, tmp_path / "o.csv", ExportOptions(4096, resume=True)
    )
    assert result.resumed_rows == 0 and client.calls == 5


def test_resume_with_a_complete_partial_only_renames_it(tmp_path):
    export_submission(QUESTIONS, retrieve, Scripted(), tmp_path / "ref.csv", ExportOptions(4096))
    (tmp_path / "ref.csv").rename(tmp_path / "o.csv.partial")
    client = Scripted()
    result = export_submission(
        QUESTIONS, retrieve, client, tmp_path / "o.csv", ExportOptions(4096, resume=True)
    )
    assert client.calls == 0 and result.resumed_rows == 5 and (tmp_path / "o.csv").exists()


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


@pytest.mark.parametrize("how", ["header", "ids", "too many", "binary", "garbage tail", "empty"])
def test_resume_refuses_a_partial_that_does_not_match_and_leaves_it_alone(tmp_path, how):
    partial = interrupted(tmp_path)
    corrupt(partial, how)
    before = partial.read_bytes()
    client = Scripted()
    with pytest.raises(ExportAborted, match="cannot resume from"):
        export_submission(
            QUESTIONS, retrieve, client, tmp_path / "o.csv", ExportOptions(4096, resume=True)
        )
    assert client.calls == 0 and partial.read_bytes() == before


def test_without_resume_an_existing_partial_is_replaced(tmp_path):
    interrupted(tmp_path)
    client = Scripted()
    export_submission(QUESTIONS, retrieve, client, tmp_path / "o.csv", ExportOptions(4096))
    assert client.calls == 5
