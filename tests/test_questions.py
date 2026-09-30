"""Unit tests for EvalQuestion's release field -- threads a question's
source release (e.g. "Rel-18") through from configs/eval_questions/*.json
to metrics.retrieval.canonical_source_hit()'s release-aware check.
"""
from __future__ import annotations

from open_telco_rag.evaluation.questions import EvalQuestion


def test_from_dict_picks_up_release_when_present():
    q = EvalQuestion.from_dict({
        "question": "What does 25.401 define?", "corpus": "3gpp",
        "canonical_source": "25401", "release": "Rel-19",
    })
    assert q.release == "Rel-19"


def test_from_dict_defaults_release_to_none_for_older_records():
    q = EvalQuestion.from_dict({
        "question": "What does 25.401 define?", "corpus": "3gpp",
        "canonical_source": "25401",
    })
    assert q.release is None


def test_from_dict_picks_up_evidence_span_when_present():
    q = EvalQuestion.from_dict({
        "question": "What does 25.401 define?", "corpus": "3gpp",
        "evidence_char_start": 120, "evidence_char_end": 340,
        "evidence_doc_id": "38_series/38401-j00/raw.md",
    })
    assert q.evidence_char_start == 120
    assert q.evidence_char_end == 340
    assert q.evidence_doc_id == "38_series/38401-j00/raw.md"


def test_from_dict_defaults_evidence_span_to_none_for_older_records():
    q = EvalQuestion.from_dict({"question": "What does 25.401 define?", "corpus": "3gpp"})
    assert q.evidence_char_start is None
    assert q.evidence_char_end is None
    assert q.evidence_doc_id is None


def test_from_dict_picks_up_difficulty_when_present():
    # Matches draft_questions/*.json's own qna_generation output shape --
    # see Task 3 Part D in CHANGES.md.
    q = EvalQuestion.from_dict({
        "question": "What does 25.401 define?", "corpus": "3gpp", "difficulty": "intermediate",
    })
    assert q.difficulty == "intermediate"


def test_from_dict_defaults_difficulty_to_none_for_files_without_it():
    # True today for every file under configs/eval_questions/.
    q = EvalQuestion.from_dict({"question": "What does 25.401 define?", "corpus": "3gpp"})
    assert q.difficulty is None
