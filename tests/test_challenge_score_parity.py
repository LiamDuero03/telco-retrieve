import subprocess
import sys
from pathlib import Path

import pytest

import cli.main as cli_main
from open_telco_rag.challenge import metrics
from open_telco_rag.challenge.client import Completion
from open_telco_rag.challenge.export import HEADER, ExportOptions, csv_writer, export_submission
from open_telco_rag.challenge.metrics import token_efficiency
from open_telco_rag.challenge.questions import load_questions

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


def write_submission(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv_writer(handle).writerows([HEADER, *rows])


def official_output(sub, ref):
    done = subprocess.run(
        [sys.executable, str(OFFICIAL_SCRIPT), str(ref), str(sub)],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout, done.returncode


def challenge_output(sub, ref, capsys):
    capsys.readouterr()
    code = 0
    try:
        cli_main.main(["challenge", "score", "--submission", str(sub), "--reference", str(ref)])
    except SystemExit as exit_info:
        code = exit_info.code
    return capsys.readouterr().out, code


def test_score_matches_the_official_scripts_run_directly(files, tmp_path, capsys):
    _, _, ref = files
    long_prompt = "word " * (token_efficiency.MAX_PROMPT_TOKENS + 10)
    submissions = {
        "valid": [[BIG_ID, "short prompt", "B"], ["2", "other prompt", "TRUE"]],
        "over_cap": [[BIG_ID, long_prompt, "B"], ["2", "other prompt", "FALSE"]],
        "missing_row": [[BIG_ID, "short prompt", "B"]],
        "duplicate_id": [
            [BIG_ID, "short prompt", "B"],
            [BIG_ID, "short prompt", "B"],
            ["2", "p", "FALSE"],
        ],
        "empty_prompt": [[BIG_ID, " ", "B"], ["2", "p", "FALSE"]],
    }
    for name, rows in submissions.items():
        sub = tmp_path / f"{name}.csv"
        write_submission(sub, rows)
        expected = official_output(sub, ref)
        assert challenge_output(sub, ref, capsys) == expected, name
    assert official_output(tmp_path / "valid.csv", ref)[0].startswith("Accuracy:")
    assert "error: Duplicate IDs" in official_output(tmp_path / "duplicate_id.csv", ref)[0]
