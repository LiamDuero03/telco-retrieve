"""Shared zero-retrieval answer for the three RAG-answer generators
(claude/local/openrouter), used when `results` is empty.

Regression fix: each generator used to return the plain text "No relevant
chunks found." on zero retrieval. For mc/tf questions, that text isn't
valid structured-verdict JSON (see evaluate.py's _parse_structured_
verdict()), so it fell through to the prose fallback -- which doesn't
recognize it as a not-found answer either (it isn't in
metrics.generation._NOT_FOUND_PHRASES), so it was scored "incorrect"
instead of "abstained". That silently punished good retrieval-failure
detection exactly where it matters most: a genuinely empty retrieval.
"""
from __future__ import annotations

import json

# Deliberately contains "does not contain" -- matches
# metrics.generation._NOT_FOUND_PHRASES, so both faithfulness()/
# answer_relevancy() and the mc/tf prose fallback recognize this as a
# not-found answer rather than a genuine (and wrong) claim. Also
# deliberately does NOT start with "No"/"Yes"/"True"/"False" as one of its
# first three words -- evaluate.py's _tf_committed_bool() reads a bare
# leading "No" as a committed FALSE answer, which would short-circuit
# straight to "incorrect" against a "True" reference before the
# not-found phrase check is even reached.
ZERO_RETRIEVAL_TEXT = (
    "The retrieval step returned zero chunks; the provided context does "
    "not contain enough information to answer this question."
)


def zero_retrieval_answer(response_schema: dict | None) -> str:
    """The "answer" string a generator should return when results is
    empty: a schema-conformant {"verdict": "abstained", ...} JSON object
    when response_schema requests one (mc/tf), else ZERO_RETRIEVAL_TEXT.

    Pass None explicitly (regardless of the caller's own response_schema)
    for a generator that never emits structured output at all (e.g.
    ClaudeGenerator) -- suddenly returning JSON only on this one code path
    would be a more surprising inconsistency than just staying prose,
    which the shared not-found phrase list already handles correctly.
    """
    if response_schema is None:
        return ZERO_RETRIEVAL_TEXT
    verdict_key = "option" if "option" in response_schema.get("properties", {}) else "boolean"
    return json.dumps({
        "verdict": "abstained",
        verdict_key: None,
        "justification": ZERO_RETRIEVAL_TEXT,
    })
