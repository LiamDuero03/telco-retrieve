from __future__ import annotations

from dataclasses import dataclass

from open_telco_rag.challenge.questions import ChallengeQuestion
from open_telco_rag.challenge.tokens import count_tokens
from open_telco_rag.types import SearchResult

SYSTEM_MESSAGE = "You answer 3GPP questions from the provided context. Reply with only the answer."
MULTIPLE_CHOICE_INSTRUCTION = "Answer with only the letter of the correct option."
TRUE_FALSE_INSTRUCTION = "Answer with only TRUE or FALSE."


class PromptTooLong(ValueError):
    pass


@dataclass
class Prompt:
    messages: list[dict]
    input_prompt: str
    n_tokens: int
    n_passages: int


def _passage(number: int, result: SearchResult) -> str:
    meta = result.metadata or {}
    spec = meta.get("spec_number") or meta.get("source")
    label = " | ".join(
        part
        for part in (
            f"Spec: {spec}" if spec else "",
            f"Release: {meta['release']}" if meta.get("release") else "",
        )
        if part
    )
    return f"[{number}] {label}\n{result.text}" if label else f"[{number}]\n{result.text}"


def _user_message(question: ChallengeQuestion, passages: list[SearchResult]) -> str:
    context = "\n\n".join(_passage(i, r) for i, r in enumerate(passages, start=1)) or "None"
    parts = [f"Passages:\n{context}", f"Question: {question.question}"]
    if question.is_true_false:
        parts.append(TRUE_FALSE_INSTRUCTION)
    else:
        parts.append(
            "Options:\n"
            + "\n".join(f"{letter}. {text}" for letter, text in question.options.items())
        )
        parts.append(MULTIPLE_CHOICE_INSTRUCTION)
    return "\n\n".join(parts)


def _assemble(question: ChallengeQuestion, passages: list[SearchResult]) -> Prompt:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": _user_message(question, passages)},
    ]
    input_prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    return Prompt(messages, input_prompt, count_tokens(input_prompt), len(passages))


def build_prompt(
    question: ChallengeQuestion, passages: list[SearchResult], max_prompt_tokens: int | None = None
) -> Prompt:
    """Messages and input_prompt from one place; drop lowest-ranked passages to fit the cap."""
    for kept in range(len(passages), -1, -1):
        prompt = _assemble(question, passages[:kept])
        if max_prompt_tokens is None or prompt.n_tokens <= max_prompt_tokens:
            return prompt
    raise PromptTooLong(
        f"question {question.id}: {prompt.n_tokens} tokens without any passage, "
        f"cap is {max_prompt_tokens}"
    )
