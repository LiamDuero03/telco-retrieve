"""Shared zero-retrieval answer for the RAG generators, used when `results`
is empty. Worded so metrics scoring treats it as an abstain, not a wrong claim.
"""
from __future__ import annotations

import json

# Contains "does not contain" so scorers see a not-found answer; deliberately
# doesn't lead with No/Yes/True/False (evaluate.py reads a bare leading "No"
# as a committed FALSE).
ZERO_RETRIEVAL_TEXT = (
    "The retrieval step returned zero chunks; the provided context does "
    "not contain enough information to answer this question."
)


def zero_retrieval_answer(response_schema: dict | None) -> str:
    """The "answer" string for empty results: a schema-conformant
    {"verdict": "abstained", ...} JSON object when response_schema requests
    one (mc/tf), else ZERO_RETRIEVAL_TEXT. Pass None for a generator that
    never emits structured output (e.g. ClaudeGenerator).
    """
    if response_schema is None:
        return ZERO_RETRIEVAL_TEXT
    verdict_key = "option" if "option" in response_schema.get("properties", {}) else "boolean"
    return json.dumps({
        "verdict": "abstained",
        verdict_key: None,
        "justification": ZERO_RETRIEVAL_TEXT,
    })
