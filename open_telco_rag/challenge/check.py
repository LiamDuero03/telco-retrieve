from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

from open_telco_rag.challenge.export import HEADER, canonical_text
from open_telco_rag.challenge.metrics import accuracy, token_efficiency
from open_telco_rag.challenge.questions import ChallengeQuestion, load_questions


@dataclass
class CheckReport:
    issues: list[str] = field(default_factory=list)
    n_rows: int = 0
    total_prompt_tokens: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues


def _format_issues(raw: bytes, rows: list[list[str]]) -> list[str]:
    issues = []
    if raw.startswith(b"\xef\xbb\xbf"):
        issues.append("file starts with a UTF-8 BOM")
    if b"\r" in raw:
        issues.append("file contains CR characters; LF line endings are required")
    if canonical_text(rows) != raw.decode("utf-8").lstrip("\ufeff").replace("\r\n", "\n"):
        issues.append(
            "not in canonical form: every field must be quoted, no blank lines, no stray whitespace"
        )
    return issues


def _official_reader_issue(path: str | Path) -> str:
    try:
        for _ in token_efficiency.read_csv(str(path)):
            pass
    except token_efficiency.MetricError as error:
        return f"the official reader rejects the file: {error}"
    return ""


def _answer_issue(question: ChallengeQuestion, answer: str) -> str:
    allowed = (
        {"TRUE", "FALSE"}
        if question.is_true_false
        else {key.strip().upper() for key in question.options}
    )
    if answer and (accuracy.normalise_answer(answer) != answer or answer not in allowed):
        return f"question {question.id}: answer '{answer}' is not one of {sorted(allowed)} or empty"
    return ""


def _prompt_issue(question: ChallengeQuestion, prompt: str) -> tuple[str, int]:
    if not prompt.strip():
        return f"question {question.id}: empty input_prompt", 0
    tokens = token_efficiency.count_tokens(prompt)
    if tokens > token_efficiency.MAX_PROMPT_TOKENS:
        return (
            f"question {question.id}: input_prompt has {tokens} tokens, "
            f"cap is {token_efficiency.MAX_PROMPT_TOKENS}"
        ), tokens
    return "", tokens


def _row_issues(
    questions: list[ChallengeQuestion], body: list[list[str]], report: CheckReport
) -> None:
    for question, (_, prompt, answer) in zip(questions, body):
        issue, tokens = _prompt_issue(question, prompt)
        report.total_prompt_tokens += tokens
        report.issues += [text for text in (issue, _answer_issue(question, answer)) if text]


def check_submission(submission_path: str | Path, questions_path: str | Path) -> CheckReport:
    report = CheckReport()
    raw = Path(submission_path).read_bytes()
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        report.issues.append(f"not readable as UTF-8 CSV: {error}")
        return report
    if not rows or rows[0] != HEADER:
        report.issues.append(f"header must be exactly {','.join(HEADER)}")
        return report
    body = rows[1:]
    report.n_rows = len(body)
    report.issues += _format_issues(raw, rows)
    report.issues += [issue for issue in (_official_reader_issue(submission_path),) if issue]
    if any(len(row) != len(HEADER) for row in body):
        report.issues.append(f"every row needs exactly {len(HEADER)} fields")
        return report
    questions = load_questions(questions_path)
    if [row[0] for row in body] != [q.id for q in questions]:
        report.issues.append(
            f"IDs differ from {questions_path}: {len(body)} rows against "
            f"{len(questions)} questions, "
            "same IDs in the same order required"
        )
        return report
    _row_issues(questions, body, report)
    return report
