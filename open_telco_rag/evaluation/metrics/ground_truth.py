"""
Tier 2 metrics — ground truth required.

  context_recall      — reference answer statements found in retrieved chunks
  answer_correctness  — factual F1 + semantic similarity vs reference answer

These are only computed when reference_answer is populated in the question JSON.

All LLM-calling functions accept optional llm_call and debug parameters.
Default is claude_call so existing call sites need no changes.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from open_telco_rag.evaluation._claude import claude_call, parse_json
from open_telco_rag.evaluation.metrics.generation import _debug_log


def context_recall(
    reference_answer: str,
    chunks: list[Any],
    llm_call: Callable[[str, int], str] = claude_call,
    debug: bool = False,
) -> float:
    """
    Fraction of statements in the reference answer that can be attributed
    to the retrieved chunks.

    Score = supported_statements / total_statements_in_reference
    """
    if not reference_answer or not chunks:
        return 0.0

    context = "\n\n---\n\n".join(c.text for c in chunks)

    prompt = f"""You are given a reference answer and retrieved context chunks.
For each statement in the reference answer, determine if it can be attributed
to the context.

Return ONLY a JSON object with two integer keys:
  "total": number of statements in the reference answer,
  "supported": number of those statements attributable to the context.

No preamble, no markdown fences.

Reference answer:
{reference_answer}

Context:
{context[:6000]}"""

    try:
        raw = llm_call(prompt, 256)
        if debug:
            _debug_log("context_recall", "raw response", raw)
        obj       = parse_json(raw)
        total     = int(obj.get("total", 0))
        supported = int(obj.get("supported", 0))
        if debug:
            _debug_log("context_recall", "parsed", f"total={total} supported={supported}")
        return min(1.0, supported / total) if total > 0 else 0.0
    except Exception as e:
        if debug:
            _debug_log("context_recall", "exception", str(e))
        return 0.0


def answer_correctness(
    answer: str,
    reference: str,
    embedder: Any,
    weight: float = 0.5,
    llm_call: Callable[[str, int], str] = claude_call,
    debug: bool = False,
) -> float:
    """
    Weighted combination of factual F1 and semantic similarity.

    Score = w * F1(answer, reference) + (1-w) * cos(embed(answer), embed(reference))
    """
    if not answer or not reference:
        return 0.0

    f1_prompt = f"""Compare the generated answer to the reference answer.
Return ONLY a JSON object with three integer keys:
  "correct":   facts in generated answer that are correct vs reference,
  "incorrect": facts in generated answer that are wrong or hallucinated,
  "missing":   facts in reference answer missing from generated answer.

No preamble, no markdown fences.

Generated answer:
{answer}

Reference answer:
{reference}"""

    try:
        raw = llm_call(f1_prompt, 256)
        if debug:
            _debug_log("answer_correctness", "f1 → raw response", raw)
        obj       = parse_json(raw)
        correct   = max(0, int(obj.get("correct", 0)))
        incorrect = max(0, int(obj.get("incorrect", 0)))
        missing   = max(0, int(obj.get("missing", 0)))
        if debug:
            _debug_log("answer_correctness", "f1 → parsed",
                       f"correct={correct} incorrect={incorrect} missing={missing}")
        precision = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
        recall    = correct / (correct + missing)   if (correct + missing)   > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
    except Exception as e:
        if debug:
            _debug_log("answer_correctness", "f1 → exception", str(e))
        f1 = 0.0

    try:
        embeddings = embedder.embed([answer, reference])
        a_emb      = np.array(embeddings[0])
        r_emb      = np.array(embeddings[1])
        sem        = float(np.dot(a_emb, r_emb) /
                          (np.linalg.norm(a_emb) * np.linalg.norm(r_emb) + 1e-10))
        sem        = max(0.0, sem)
    except Exception as e:
        if debug:
            _debug_log("answer_correctness", "embedding → exception", str(e))
        sem = 0.0

    return weight * f1 + (1.0 - weight) * sem