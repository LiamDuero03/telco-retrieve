"""
Tier 1 generation metrics — no ground truth required.

  faithfulness       — every claim grounded in retrieved chunks
  answer_relevancy   — answer addresses the intent of the question
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import numpy as np

from open_telco_rag.evaluation._claude import claude_call, parse_json

logger = logging.getLogger(__name__)

# Phrases that indicate the RAG failed to retrieve relevant content.
# These answers should not score highly on relevancy or faithfulness --
# and, for mc/tf questions, are the SAME signal used by evaluate.py's
# _match_deterministic_answer() to classify an answer as "abstained"
# rather than "incorrect" (see that function's own docstring). Shared
# here, not duplicated, so both call sites recognize exactly the same set
# of phrasings — extended with a few more "I can't determine this"-style
# phrasings _build_generation_question()'s own mc/tf instruction
# ("...say so explicitly instead of guessing") is likely to elicit,
# beyond the original retrieval-failure-specific wording.
_NOT_FOUND_PHRASES = [
    "does not contain",
    "not provided in",
    "not available in",
    "cannot be found",
    "no information",
    "not mentioned",
    "not in the context",
    "context does not",
    "provided context does not",
    "cannot determine",
    "unable to determine",
    "insufficient information",
    "not enough information",
    "no way to determine",
]

def _is_not_found_answer(answer: str) -> bool:
    a = answer.lower()
    return any(phrase in a for phrase in _NOT_FOUND_PHRASES)


def _debug_log(metric: str, step: str, raw: str) -> None:
    print(f"\n  [DEBUG] {metric} / {step}")
    print(f"  {'─'*60}")
    print(f"  {raw[:800]}")
    print(f"  {'─'*60}")


def faithfulness(
    answer: str,
    chunks: list[Any],
    llm_call: Callable[[str, int], str] = claude_call,
    debug: bool = False,
) -> float:
    """
    Fraction of atomic claims in the answer supported by the retrieved chunks.

    Returns 0.0 immediately if the answer indicates retrieval failure
    (e.g. 'the context does not contain...'), since there are no factual
    claims to verify.

    Score = supported_claims / total_claims
    1.0 = fully grounded, < 1.0 = hallucination present.
    """
    if not answer or not chunks:
        return 0.0

    # Don't reward "not found" answers with high faithfulness
    if _is_not_found_answer(answer):
        if debug:
            _debug_log("faithfulness", "skipped — not-found answer", answer[:200])
        return 0.0

    context = "\n\n---\n\n".join(c.text for c in chunks)

    decompose_prompt = f"""Break this answer into individual atomic factual claims.
Each claim should be a single verifiable statement.
Return ONLY a JSON array of strings, one string per claim.
No preamble, no markdown fences.

Answer:
{answer}"""

    try:
        raw_claims = llm_call(decompose_prompt, 512)
        if debug:
            _debug_log("faithfulness", "decompose_claims → raw response", raw_claims)
        claims = parse_json(raw_claims)
        if not isinstance(claims, list) or not claims:
            return 0.0
    except Exception as e:
        logger.warning("faithfulness: decompose_claims raised %s: %s — scoring 0.0",
                        type(e).__name__, e)
        if debug:
            _debug_log("faithfulness", "decompose_claims → exception", str(e))
        return 0.0

    verify_prompt = f"""For each claim below, answer YES if it can be directly
inferred from the context provided, NO if it cannot.
Return ONLY a JSON array of booleans (true/false), one per claim, same order.
No preamble, no markdown fences.

Context:
{context[:6000]}

Claims:
{json.dumps(claims)}"""

    try:
        raw_verdicts = llm_call(verify_prompt, 512)
        if debug:
            _debug_log("faithfulness", "verify_claims → raw response", raw_verdicts)
        verdicts = parse_json(raw_verdicts)
        if debug:
            _debug_log("faithfulness", "verify_claims → parsed",
                       f"verdicts={verdicts}  len={len(verdicts)}  expected={len(claims)}")
        if not isinstance(verdicts, list) or len(verdicts) == 0:
            return 0.0
        # Pad with False if fewer verdicts than claims (conservative)
        while len(verdicts) < len(claims):
            verdicts.append(False)
        verdicts = verdicts[:len(claims)]
        return sum(1 for v in verdicts if v is True) / len(verdicts)
    except Exception as e:
        logger.warning("faithfulness: verify_claims raised %s: %s — scoring 0.0",
                        type(e).__name__, e)
        if debug:
            _debug_log("faithfulness", "verify_claims → exception", str(e))
        return 0.0


def answer_relevancy(
    question: str,
    answer: str,
    embedder: Any,
    n_reverse: int = 3,
    llm_call: Callable[[str, int], str] = claude_call,
    debug: bool = False,
) -> float:
    """
    Mean cosine similarity between the original question and N questions
    reverse-engineered from the answer.

    Returns 0.0 if the answer indicates retrieval failure — a 'not found'
    answer is not relevant regardless of how well it echoes the question.

    Score = (1/N) Σ cos(embed(qᵢ), embed(q₀))
    High → answer is on-topic. Low → answer drifts from question intent.
    """
    if not answer:
        return 0.0

    # Don't reward "not found" answers with high relevancy
    if _is_not_found_answer(answer):
        if debug:
            _debug_log("answer_relevancy", "skipped — not-found answer", answer[:200])
        return 0.0

    prompt = f"""Generate {n_reverse} different questions that this answer
is responding to. Focus on the factual content of the answer.
Return ONLY a JSON array of {n_reverse} strings.
No preamble, no markdown fences.

Answer:
{answer}"""

    try:
        raw = llm_call(prompt, 256)
        if debug:
            _debug_log("answer_relevancy", "reverse_questions → raw response", raw)
        reverse_qs = parse_json(raw)
        # An EMPTY (but validly-parsed) list would otherwise reach
        # np.mean([]) below, which returns NaN rather than raising —
        # silently poisoning answer_relevancy instead of the clean 0.0
        # faithfulness() already returns for the analogous empty-claims
        # case above.
        if not isinstance(reverse_qs, list) or not reverse_qs:
            return 0.0
        reverse_qs = [str(q) for q in reverse_qs[:n_reverse]]
        if debug:
            _debug_log("answer_relevancy", "reverse_questions → parsed",
                       "\n  ".join(reverse_qs))
    except Exception as e:
        logger.warning("answer_relevancy: reverse_questions raised %s: %s — scoring 0.0",
                        type(e).__name__, e)
        if debug:
            _debug_log("answer_relevancy", "reverse_questions → exception", str(e))
        return 0.0

    try:
        embeddings = embedder.embed([question] + reverse_qs)
        q0   = np.array(embeddings[0])
        sims = []
        for emb in embeddings[1:]:
            qi  = np.array(emb)
            cos = float(np.dot(q0, qi) /
                        (np.linalg.norm(q0) * np.linalg.norm(qi) + 1e-10))
            sims.append(cos)
        return float(np.mean(sims))
    except Exception as e:
        logger.warning("answer_relevancy: embedding raised %s: %s — scoring 0.0",
                        type(e).__name__, e)
        if debug:
            _debug_log("answer_relevancy", "embedding → exception", str(e))
        return 0.0