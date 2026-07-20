"""
Evaluation question dataclass and loader.

Questions live in configs/eval_questions/{corpus}.json — one file per corpus.
This module handles loading and validation only. No hardcoded questions here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Resolves to open_telco_rag/../../configs/eval_questions/
_QUESTIONS_DIR = Path(__file__).parent.parent.parent / "configs" / "eval_questions"

SUPPORTED_CORPORA = ["oran", "3gpp", "camara", "tmforum", "gsma", "itu", "etsi"]


@dataclass
class EvalQuestion:
    """One evaluation question with optional ground truth."""
    question:         str
    keywords:         list[str]
    corpus:           str
    answer_type:      str = "open"          
    options:          dict | None = None
    canonical_source: str | None = None
    reference_answer: str | None = None   # None until validated by domain expert

    @classmethod
    def from_dict(cls, d: dict) -> "EvalQuestion":
        return cls(
            question         = d["question"],
            keywords         = d.get("keywords", []),
            corpus           = d.get("corpus", ""),
            answer_type      = d.get("answer_type", "open"),
            options          = d.get("options", None),
            canonical_source = d.get("canonical_source"),
            reference_answer = d.get("reference_answer"),
        )


def load_questions(
    corpus: str,
    questions_dir: Path | None = None,
) -> list[EvalQuestion]:
    """
    Load evaluation questions for one corpus from its JSON file.

    Args:
        corpus:        one of SUPPORTED_CORPORA
        questions_dir: override default path (useful in tests)

    Returns:
        list of EvalQuestion

    Raises:
        FileNotFoundError: if the JSON file does not exist
        ValueError:        if corpus is not in SUPPORTED_CORPORA
    """
    if corpus not in SUPPORTED_CORPORA:
        raise ValueError(
            f"Unknown corpus '{corpus}'. "
            f"Supported: {SUPPORTED_CORPORA}"
        )

    qdir = questions_dir or _QUESTIONS_DIR
    path = qdir / f"{corpus}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {path}\n"
            f"Expected at: configs/eval_questions/{corpus}.json"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("questions", [])

    if not items:
        raise ValueError(f"No questions found in {path}")

    return [EvalQuestion.from_dict(item) for item in items]


def load_all_questions(
    questions_dir: Path | None = None,
) -> dict[str, list[EvalQuestion]]:
    """
    Load questions for all supported corpora.

    Returns:
        dict mapping corpus name → list of EvalQuestion
    """
    return {
        corpus: load_questions(corpus, questions_dir)
        for corpus in SUPPORTED_CORPORA
    }