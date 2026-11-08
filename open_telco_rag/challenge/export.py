from __future__ import annotations

import csv
import io
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from open_telco_rag.challenge.client import Completion, GeneratorError
from open_telco_rag.challenge.normalise import normalise_answer
from open_telco_rag.challenge.prompt import Prompt, PromptTooLong, build_prompt
from open_telco_rag.challenge.questions import ChallengeQuestion
from open_telco_rag.challenge.tokens import count_tokens
from open_telco_rag.types import SearchResult

HEADER = ["ID", "input_prompt", "answer"]
PromptBuilder = Callable[[ChallengeQuestion, list[SearchResult], int | None], Prompt]
AnswerParser = Callable[[str, dict[str, str]], str]


def csv_writer(handle: TextIO) -> csv._writer:
    return csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")


class ExportAborted(RuntimeError):
    pass


class ChatBackend(Protocol):
    def complete(self, messages: list[dict], **settings) -> Completion: ...


def canonical_text(rows: list[list[str]]) -> str:
    buffer = io.StringIO(newline="")
    csv_writer(buffer).writerows(rows)
    return buffer.getvalue()


@dataclass
class ExportResult:
    path: Path
    n_rows: int
    total_prompt_tokens: int
    empty_answers: int
    resumed_rows: int = 0


def _retrieve_or_abort(
    retrieve: Callable[[str], list[SearchResult]], question: ChallengeQuestion
) -> list[SearchResult]:
    try:
        return retrieve(question.question)
    except Exception as error:
        raise ExportAborted(f"question {question.id}: retrieval failed ({error})") from error


@dataclass
class ExportOptions:
    max_prompt_tokens: int | None
    temperature: float = 0.0
    max_tokens: int = 8
    seed: int | None = None
    attempts: int = 4
    sleep: Callable[[float], None] = time.sleep
    resume: bool = False
    progress_every: int = 25
    progress: Callable[[int, int], None] | None = None
    prompt_builder: PromptBuilder = build_prompt
    answer_parser: AnswerParser = normalise_answer


def _call_plugin(name: str, question: ChallengeQuestion, function, *args):
    try:
        return function(*args)
    except Exception as error:
        raise ExportAborted(
            f"question {question.id}: {name} raised {type(error).__name__}: {error}"
        ) from error


def _validated_tokens(question: ChallengeQuestion, prompt: Prompt, cap: int | None) -> int:
    if not isinstance(prompt, Prompt) or not prompt.input_prompt.strip():
        raise ExportAborted(f"question {question.id}: the prompt builder returned no usable Prompt")
    n_tokens = count_tokens(prompt.input_prompt)
    if cap is not None and n_tokens > cap:
        raise PromptTooLong(f"question {question.id}: prompt has {n_tokens} tokens, cap is {cap}")
    return n_tokens


def _complete_with_retries(
    client: ChatBackend, question: ChallengeQuestion, messages: list[dict], options: ExportOptions
) -> str:
    delay = 1.0
    for attempt in range(options.attempts):
        made = attempt + 1
        try:
            return client.complete(
                messages,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                seed=options.seed,
            ).text
        except GeneratorError as error:
            last_error = error
            if not error.retryable:
                break
            if attempt < options.attempts - 1:
                options.sleep(delay)
                delay *= 2
    raise ExportAborted(
        f"question {question.id}: generator failed after {made} attempt(s) ({last_error})"
    ) from last_error


def _answer_row(
    question: ChallengeQuestion,
    retrieve: Callable[[str], list[SearchResult]],
    client: ChatBackend,
    options: ExportOptions,
) -> tuple[list[str], int]:
    passages = _retrieve_or_abort(retrieve, question)
    prompt = _call_plugin(
        "prompt builder",
        question,
        options.prompt_builder,
        question,
        passages,
        options.max_prompt_tokens,
    )
    n_tokens = _validated_tokens(question, prompt, options.max_prompt_tokens)
    raw = _complete_with_retries(client, question, prompt.messages, options)
    answer = _call_plugin("answer parser", question, options.answer_parser, raw, question.options)
    if not isinstance(answer, str):
        raise ExportAborted(
            f"question {question.id}: answer parser returned {type(answer).__name__}"
        )
    return [question.id, prompt.input_prompt, answer], n_tokens


def _refuse(partial: Path, reason: str) -> ExportAborted:
    return ExportAborted(f"cannot resume from {partial}: {reason}")


def _read_partial(partial: Path, questions: list[ChallengeQuestion]) -> tuple[int, int, int]:
    """Validate a partial file, drop an unfinished last row, return (rows, tokens, empty)."""
    try:
        with partial.open(encoding="utf-8", newline="") as handle:
            text = handle.read()
    except UnicodeDecodeError as error:
        raise _refuse(partial, "it is not UTF-8") from error
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or rows[0] != HEADER:
        raise _refuse(partial, f"its header is not {','.join(HEADER)}")
    body = rows[1:] if canonical_text(rows) == text else rows[1:-1]
    prefix = canonical_text([HEADER, *body])
    tail = text[len(prefix) :]
    if not text.startswith(prefix) or (tail and not tail.startswith('"')):
        raise _refuse(partial, "it is not an output of this command")
    if [row[0] for row in body] != [q.id for q in questions[: len(body)]]:
        raise _refuse(partial, f"its {len(body)} IDs differ from the first IDs of the Test file")
    with partial.open("r+b") as handle:
        handle.truncate(len(prefix.encode("utf-8")))
    tokens = sum(count_tokens(row[1]) for row in body)
    return len(body), tokens, sum(not row[2] for row in body)


def _report_progress(options: ExportOptions, done: int, total: int) -> None:
    if done % options.progress_every == 0:
        (options.progress or (lambda d, t: None))(done, total)


def export_submission(
    questions: list[ChallengeQuestion],
    retrieve: Callable[[str], list[SearchResult]],
    client: ChatBackend,
    out_path: str | Path,
    options: ExportOptions,
) -> ExportResult:
    """Write rows to <out>.partial and rename on success; a failure keeps the partial."""
    out = Path(out_path)
    partial = out.with_name(out.name + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    resumed = (0, 0, 0)
    if options.resume and partial.exists():
        resumed = _read_partial(partial, questions)
    done, tokens, empty = resumed
    with partial.open("a" if done else "w", encoding="utf-8", newline="") as handle:
        writer = csv_writer(handle)
        if not done:
            writer.writerow(HEADER)
        for question in questions[done:]:
            row, n_tokens = _answer_row(question, retrieve, client, options)
            writer.writerow(row)
            handle.flush()
            done, tokens, empty = done + 1, tokens + n_tokens, empty + (not row[2])
            _report_progress(options, done, len(questions))
        os.fsync(handle.fileno())
    os.replace(partial, out)
    return ExportResult(out, len(questions), tokens, empty, resumed[0])


def stderr_progress(done: int, total: int) -> None:
    print(f"[progress] {done}/{total} questions", file=sys.stderr, flush=True)
