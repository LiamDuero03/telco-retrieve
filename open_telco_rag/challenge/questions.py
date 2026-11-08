from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

ID_COLUMNS = ("id", "question_id")


@dataclass
class ChallengeQuestion:
    id: str
    question: str
    options: dict[str, str]

    @property
    def is_true_false(self) -> bool:
        return not self.options


def _id_column(header: list[str], path: Path) -> str:
    found = [name for name in header if name.strip().lower() in ID_COLUMNS]
    if len(found) != 1:
        raise ValueError(
            f"{path}: expected exactly one ID column (ID or question_id), found {found or header}"
        )
    return found[0]


def _parse_options(raw: str, path: Path, question_id: str) -> dict[str, str]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        options = json.loads(text)
    except ValueError as error:
        raise ValueError(
            f"{path}: question {question_id}: options is not valid JSON ({error})"
        ) from error
    if not isinstance(options, dict) or not all(isinstance(v, str) for v in options.values()):
        raise ValueError(
            f"{path}: question {question_id}: options must be a JSON object of strings"
        )
    return options


def load_questions(path: str | Path) -> list[ChallengeQuestion]:
    """Read a Test CSV; IDs stay strings exactly as written."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [name.strip() for name in reader.fieldnames or []]
        reader.fieldnames = header
        id_column = _id_column(header, path)
        for column in ("question", "options"):
            if column not in header:
                hint = " (use {} for true/false questions)" if column == "options" else ""
                raise ValueError(f"{path}: missing {column} column{hint}")
        questions: list[ChallengeQuestion] = []
        seen: set[str] = set()
        for row in reader:
            question_id = row[id_column]
            if question_id in seen:
                raise ValueError(f"{path}: duplicate question ID {question_id}")
            seen.add(question_id)
            options = _parse_options(row["options"], path, question_id)
            questions.append(ChallengeQuestion(question_id, row["question"], options))
    if not questions:
        raise ValueError(f"{path}: no questions")
    return questions
