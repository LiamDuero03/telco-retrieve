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
