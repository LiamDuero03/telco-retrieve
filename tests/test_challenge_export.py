import csv
import io
import subprocess
import sys

import pytest

import cli.main as cli_main
from open_telco_rag.challenge import cli as challenge_cli
from open_telco_rag.challenge.client import ChatClient, Completion, GeneratorError
from open_telco_rag.challenge.export import ExportAborted, ExportOptions, export_submission
from open_telco_rag.challenge.index import IndexProblem, open_index
from open_telco_rag.challenge.questions import ChallengeQuestion
from open_telco_rag.types import SearchResult
from tests.fake_collections import FakeEmbedder, make_collection

BIG_ID = "781260748538"
QUESTIONS = [
    ChallengeQuestion(BIG_ID, 'Which NF, "really", selects\nthe UPF?', {"A": "AMF", "B": "SMF"}),
    ChallengeQuestion("0042", "The UE registers.", {}),
]


def retrieve(query):
    return [
        SearchResult(
            text="Ünïcode <|endoftext|>, passage",
            distance=0.1,
            metadata={"spec_number": "23.501", "release": "Rel-19"},
            doc_id="d::m::0",
        )
    ]


class FakeClient:
    def __init__(self, replies, fail_on=None):
        self.replies, self.fail_on, self.seen = list(replies), fail_on, []

    def complete(self, messages, **settings):
        self.seen.append((messages, settings))
        if self.fail_on and len(self.seen) >= self.fail_on:
            raise GeneratorError("down")
        return Completion(self.replies.pop(0), messages, True)


def test_submission_bytes_quoting_and_atomic_write(tmp_path):
    out = tmp_path / "sub.csv"
    client = FakeClient(["B", "garbage"])
    result = export_submission(
        QUESTIONS, retrieve, client, out, ExportOptions(max_prompt_tokens=4096, seed=1234)
    )
    raw = out.read_bytes()
    assert (
        raw.startswith(b'"ID","input_prompt","answer"\n')
        and b"\r" not in raw
        and not raw.startswith(b"\xef\xbb\xbf")
    )
    assert not (tmp_path / "sub.csv.partial").exists()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
    assert rows[1:] == [
        [BIG_ID, client.seen[0][0][0]["content"] + "\n\n" + client.seen[0][0][1]["content"], "B"],
        ["0042", rows[2][1], ""],
    ]
    assert client.seen[0][1] == {"temperature": 0.0, "max_tokens": 8, "seed": 1234}
    assert (result.n_rows, result.empty_answers) == (2, 1) and result.total_prompt_tokens > 0


def test_abort_names_question_keeps_partial_and_backs_off(tmp_path):
    out, delays = tmp_path / "sub.csv", []
    client = FakeClient(["B", "TRUE"], fail_on=2)
    with pytest.raises(ExportAborted, match="question 0042"):
        export_submission(
            QUESTIONS,
            retrieve,
            client,
            out,
            ExportOptions(max_prompt_tokens=4096, sleep=delays.append),
        )
    assert (
        not out.exists()
        and (tmp_path / "sub.csv.partial").read_text(encoding="utf-8").count(BIG_ID) == 1
    )
    assert delays == [1.0, 2.0, 4.0]


def test_export_never_imports_scoring_modules():
    code = (
        "import sys, open_telco_rag.challenge.export; "
        "names = ('open_telco_rag.challenge.metrics', 'open_telco_rag.challenge.score', "
        "'open_telco_rag.challenge.check', 'accuracy', 'token_efficiency', 'overall_score'); "
        "bad = [m for m in sys.modules if m in names or m.startswith('open_telco_rag.challenge.metrics.')]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def build_index(tmp_path, with_fts=True, parents=False, sidecar="default", plain_ids=False):
    rows = [
        {
            "id": f"chunk{i}" if plain_ids else f"a.md::fake64::{i}",
            "text": f"UE registration request {i}",
            "source": "a.md",
            "meta": {"spec_number": "23.501", "release": "Rel-19"}
            | ({"parent_id": f"P{i // 2}"} if parents else {}),
        }
        for i in range(6)
    ]
    make_collection(
        tmp_path, "col", rows, FakeEmbedder(), space="cosine", sidecar_model=sidecar
    ).close()
    if not with_fts:
        (tmp_path / "col_fts.db").unlink()
    return tmp_path


def test_open_index_fails_clearly_without_bm25_side_and_creates_nothing(tmp_path):
    db = build_index(tmp_path, with_fts=False)
    before = sorted(p.name for p in db.iterdir())
    assert open_index(db, "col", "dense").count() == 6
    with pytest.raises(IndexProblem, match="col_fts.db"):
        open_index(db, "col", "hybrid")
    with pytest.raises(IndexProblem, match="not found"):
        open_index(db, "other", "dense")
    assert sorted(p.name for p in db.iterdir()) == before


def run_cli(tmp_path, monkeypatch, *extra):
    questions = tmp_path / "q.csv"
    questions.write_text(
        f'ID,question,options\n{BIG_ID},Which NF?,"{{""A"": ""AMF"", ""B"": ""SMF""}}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(challenge_cli, "_build_embedder", lambda settings: FakeEmbedder())
    monkeypatch.setattr(challenge_cli, "_build_client", lambda *a, **k: FakeClient(["B"]))
    out = tmp_path / "out.csv"
    cli_main.main(
        [
            "challenge",
            "export",
            "--questions",
            str(questions),
            "--out",
            str(out),
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


def test_cli_export_hybrid_end_to_end(tmp_path, monkeypatch):
    build_index(tmp_path / "db")
    out = run_cli(tmp_path, monkeypatch, "--search-method", "hybrid", "--top-k", "3")
    with open(out, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert (
        rows[0] == ["ID", "input_prompt", "answer"] and rows[1][0] == BIG_ID and rows[1][2] == "B"
    )
    assert "Spec: 23.501 | Release: Rel-19" in rows[1][1]


def test_cli_export_hybrid_without_fts_exits_with_message(tmp_path, monkeypatch):
    build_index(tmp_path / "db", with_fts=False)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(tmp_path, monkeypatch, "--search-method", "hybrid")
    assert "col_fts.db" in str(exit_info.value) and not (tmp_path / "out.csv").exists()


class FakeResponse:
    def __init__(self, status, content="B"):
        self.status_code, self._content = status, content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_chat_client_sets_options_and_drops_a_rejected_seed(monkeypatch):
    sent = []

    def post(url, json, headers, timeout):
        sent.append(dict(json))
        return FakeResponse(422 if "seed" in json else 200, " B ")

    monkeypatch.setattr("open_telco_rag.challenge.client.requests.post", post)
    client = ChatClient("m", base_url="http://x/v1")
    messages = [{"role": "user", "content": "hi"}]
    done = client.complete(messages, seed=7)
    assert (done.text, done.messages, done.seed_sent) == ("B", messages, False)
    assert (
        sent[0]["seed"] == 7
        and "seed" not in sent[1]
        and sent[1]["temperature"] == 0.0
        and sent[1]["max_tokens"] == 8
    )
    client.complete(messages, seed=7)
    assert "seed" not in sent[2]
