"""
Tier 1 generation metrics — no ground truth required.

  keyword_score      — expected terms present in the answer
  faithfulness       — every claim grounded in retrieved chunks
  answer_relevancy   — answer addresses the intent of the question

All LLM-calling functions accept optional llm_call and debug parameters.
Default is claude_call so existing call sites need no changes.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import numpy as np

from open_telco_rag.evaluation._claude import claude_call, parse_json

logger = logging.getLogger(__name__)


def _debug_log(metric: str, step: str, raw: str) -> None:
    """Print a formatted debug block to stdout."""
    print(f"\n  [DEBUG] {metric} / {step}")
    print(f"  {'─'*60}")
    print(f"  {raw[:800]}")
    print(f"  {'─'*60}")


def keyword_score(answer: str, keywords: list[str]) -> float:
    """
    Fraction of expected keywords present in the answer (case-insensitive).

    Score = |{k ∈ keywords | k ∈ answer}| / |keywords|
    """
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in answer_lower)
    return hits / len(keywords)


def faithfulness(
    answer: str,
    chunks: list[Any],
    llm_call: Callable[[str, int], str] = claude_call,
    debug: bool = False,
) -> float:
    """
    Fraction of atomic claims in the answer supported by the retrieved chunks.

    Step 1 (llm_call): decompose answer into atomic claims.
    Step 2 (llm_call): verify each claim against the chunk context.

    Score = supported_claims / total_claims
    A score of 1.0 means every claim is grounded; < 1.0 indicates hallucination.
    """
    if not answer or not chunks:
        return 0.0

    context = "\n\n---\n\n".join(c.text for c in chunks)

    decompose_prompt = f"""Break this answer into individual atomic factual claims.
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
            if debug:
                _debug_log("faithfulness", "decompose_claims → parse result",
                           f"FAILED — not a list or empty: {type(claims)} = {claims}")
            return 0.0
    except Exception as e:
        if debug:
            _debug_log("faithfulness", "decompose_claims → exception", str(e))
        return 0.0

    verify_prompt = f"""For each claim, answer YES if it can be inferred from
the context, NO if it cannot.
Return ONLY a JSON array of booleans (true/false), one per claim, same order.
No preamble, no markdown fences.

Context:
{context[:6000]}

Claims:
{json.dumps(claims)}"""

    try:
        raw_verdicts = llm_call(verify_prompt, 512)  # 512 — enough for 20+ booleans
        if debug:
            _debug_log("faithfulness", "verify_claims → raw response", raw_verdicts)
        verdicts = parse_json(raw_verdicts)
        if debug:
            _debug_log("faithfulness", "verify_claims → parsed",
                       f"verdicts={verdicts}  len={len(verdicts)}  expected={len(claims)}")
        if not isinstance(verdicts, list) or len(verdicts) == 0:
            if debug:
                _debug_log("faithfulness", "verify_claims → empty or not a list", str(verdicts))
            return 0.0
        # Graceful mismatch: if Llama returns fewer verdicts than claims, score
        # on what we have rather than returning 0.0. Extra claims beyond the
        # returned verdicts are treated as unsupported (conservative).
        if len(verdicts) != len(claims):
            if debug:
                _debug_log("faithfulness", "verify_claims → shape mismatch (using partial)",
                           f"got {len(verdicts)} verdicts for {len(claims)} claims — "
                           f"scoring {len(verdicts)} + {len(claims)-len(verdicts)} unsupported")
            # Pad with False for any claims without a verdict
            while len(verdicts) < len(claims):
                verdicts.append(False)
            verdicts = verdicts[:len(claims)]
        return sum(1 for v in verdicts if v is True) / len(verdicts)
    except Exception as e:
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

    Score = (1/N) Σ cos(embed(qᵢ), embed(q₀))

    High score → answer is on-topic.
    Low score  → answer drifts from the question intent.
    """
    if not answer:
        return 0.0

    prompt = f"""Generate {n_reverse} questions that this answer appears to be
responding to. Return ONLY a JSON array of {n_reverse} strings.
No preamble, no markdown fences.

Answer:
{answer}"""

    try:
        raw = llm_call(prompt, 256)
        if debug:
            _debug_log("answer_relevancy", "reverse_questions → raw response", raw)
        reverse_qs = parse_json(raw)
        if not isinstance(reverse_qs, list):
            if debug:
                _debug_log("answer_relevancy", "reverse_questions → not a list",
                           f"{type(reverse_qs)}: {reverse_qs}")
            return 0.0
        reverse_qs = [str(q) for q in reverse_qs[:n_reverse]]
        if debug:
            _debug_log("answer_relevancy", "reverse_questions → parsed",
                       "\n  ".join(reverse_qs))
    except Exception as e:
        if debug:
            _debug_log("answer_relevancy", "reverse_questions → exception", str(e))
        return 0.0

    try:
        embeddings = embedder.embed([question] + reverse_qs)
        q0         = np.array(embeddings[0])
        sims       = []
        for emb in embeddings[1:]:
            qi  = np.array(emb)
            cos = float(np.dot(q0, qi) /
                        (np.linalg.norm(q0) * np.linalg.norm(qi) + 1e-10))
            sims.append(cos)
        return float(np.mean(sims))
    except Exception as e:
        if debug:
            _debug_log("answer_relevancy", "embedding → exception", str(e))
        return 0.0