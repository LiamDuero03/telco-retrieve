import hashlib
from pathlib import Path

import pytest

from open_telco_rag.challenge import metrics
from open_telco_rag.challenge.check import check_submission
from open_telco_rag.challenge.client import Completion
from open_telco_rag.challenge.export import HEADER, ExportOptions, csv_writer, export_submission
from open_telco_rag.challenge.metrics import accuracy, token_efficiency
from open_telco_rag.challenge.questions import load_questions
from open_telco_rag.challenge.score import score_submission
from open_telco_rag.challenge.tokens import count_tokens

BIG_ID = "781260748538"
OFFICIAL_SHA256 = {
    "accuracy.py": "f3ca4ce766d5cb07711bf32c584fdc92434068cdd97946ccb3f3904f081d1abe",
    "overall_score.py": "6c97f7d2f9ac518bdc5494debcaf3ecda2d8fe2749399eafe99b591b201ea160",
    "token_efficiency.py": "269143e357c2f5ca1093d094278951ca434cc3bdafd99ae743b67bc1ac1558ec",
}
OFFICIAL_SCRIPT = Path(metrics.__file__).parent / "overall_score.py"


class Reply:
    def __init__(self, answers):
        self.answers = list(answers)

    def complete(self, messages, **settings):
        return Completion(self.answers.pop(0), messages, False)


@pytest.fixture
def files(tmp_path):
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(
        f'ID,question,options\n{BIG_ID},Which?,"{{""A"": ""x"", ""B"": ""y""}}"\n2,True?,{{}}\n',
        encoding="utf-8",
    )
    sub = tmp_path / "sub.csv"
    export_submission(
        load_questions(test_csv),
        lambda q: [],
        Reply(["B", "TRUE"]),
        sub,
        ExportOptions(max_prompt_tokens=4096),
    )
    ref = tmp_path / "ref.csv"
    ref.write_text(f"ID,answer\n{BIG_ID},B\n2,FALSE\n", encoding="utf-8")
    return test_csv, sub, ref


def test_vendored_scripts_are_unmodified():
    for name, digest in OFFICIAL_SHA256.items():
        assert (
            hashlib.sha256((Path(metrics.__file__).parent / name).read_bytes()).hexdigest()
            == digest
        )


def test_export_constants_match_the_official_ones():
    assert HEADER == accuracy.EXPECTED_COLUMNS == token_efficiency.EXPECTED_COLUMNS
    for text in ("plain", "a <|endoftext|> b", "Ünïcode ✓ 5G-GUTI", "x" * 5000):
        assert count_tokens(text) == token_efficiency.count_tokens(text)


def test_check_accepts_exported_file_and_flags_each_defect(files):
    test_csv, sub, _ = files
    assert check_submission(sub, test_csv).ok
    good = sub.read_bytes()
    cases = {
        "BOM": b"\xef\xbb\xbf" + good,
        "CR": good.replace(b"\n", b"\r\n"),
        "unquoted": good.replace(b'"B"', b"B"),
        "header": good.replace(b'"ID"', b'"id"', 1),
        "answer": good.replace(b',"B"\n', b',"Z"\n'),
    }
    for name, data in cases.items():
        sub.write_bytes(data)
        assert not check_submission(sub, test_csv).ok, name


def test_check_uses_the_official_cap(files):
    test_csv, sub, _ = files
    long_prompt = "word " * (token_efficiency.MAX_PROMPT_TOKENS + 10)
    with sub.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_writer(handle)
        writer.writerows([HEADER, [BIG_ID, long_prompt, "B"], ["2", "short", "TRUE"]])
    issues = check_submission(sub, test_csv).issues
    expected = (
        f"question {BIG_ID}: input_prompt has {count_tokens(long_prompt)} tokens, "
        f"cap is {token_efficiency.MAX_PROMPT_TOKENS}"
    )
    assert issues == [expected]


def test_score_submission_values(files):
    _, sub, ref = files
    result = score_submission(sub, ref)
    assert result.accuracy == 50.0 and not result.minimum_met
    assert result.overall == pytest.approx(0.7 * result.accuracy + 0.3 * result.token_efficiency)
