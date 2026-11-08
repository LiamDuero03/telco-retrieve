import csv

import pytest

from open_telco_rag.challenge import prompt as prompt_module
from open_telco_rag.challenge.export import ExportOptions, export_submission
from open_telco_rag.challenge.metrics import token_efficiency
from open_telco_rag.challenge.prompt import PromptTooLong, build_prompt
from open_telco_rag.challenge.questions import ChallengeQuestion
from open_telco_rag.types import SearchResult
from tests.test_challenge_check import Reply

CAP = token_efficiency.MAX_PROMPT_TOKENS


def passage(text="plain passage text"):
    return SearchResult(
        text=text, distance=0.1, metadata={"spec_number": "23.501", "release": "Rel-19"}, doc_id="d"
    )


def question(text="Which NF selects the UPF?", options=None):
    return ChallengeQuestion("1", text, {"A": "AMF", "B": "SMF"} if options is None else options)


def official(prompt) -> int:
    return token_efficiency.count_tokens(prompt.input_prompt)


def test_tokens_are_exactly_the_official_count_of_input_prompt():
    built = build_prompt(question(), [passage(), passage("second")])
    assert built.n_tokens == official(built)
    assert (
        built.input_prompt == built.messages[0]["content"] + "\n\n" + built.messages[1]["content"]
    )


def test_system_message_passages_question_and_options_all_count():
    base = official(build_prompt(question(), [passage()]))
    assert official(build_prompt(question(), [passage("extra words " * 40)])) > base
    assert official(build_prompt(question("A much longer question " * 20), [passage()])) > base
    assert (
        official(build_prompt(question(options={"A": "AMF " * 50, "B": "SMF"}), [passage()])) > base
    )


def test_a_long_system_message_raises_t_and_trips_the_cap(monkeypatch):
    short = build_prompt(question(), [passage()], CAP)
    monkeypatch.setattr(prompt_module, "SYSTEM_MESSAGE", "You answer questions. " * 1500)
    with pytest.raises(PromptTooLong, match="question 1"):
        build_prompt(question(), [passage()], CAP)
    assert official(build_prompt(question(), [passage()])) > short.n_tokens + 1500


def test_special_token_text_is_counted_not_an_error():
    plain = build_prompt(question(), [passage("alpha beta")])
    special = build_prompt(question(), [passage("alpha <|endoftext|> beta")])
    assert special.n_tokens == official(special) and special.n_tokens >= plain.n_tokens + 3


def test_export_cap_and_total_use_the_metric_string_and_tokenizer(tmp_path):
    questions = [question(), ChallengeQuestion("2", "The UE registers.", {})]
    out = tmp_path / "sub.csv"
    result = export_submission(
        questions,
        lambda q: [passage("a <|endoftext|> b")],
        Reply(["B", "TRUE"]),
        out,
        ExportOptions(max_prompt_tokens=CAP),
    )
    with open(out, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))[1:]
    counts = [token_efficiency.count_tokens(row[1]) for row in rows]
    assert sum(counts) == result.total_prompt_tokens and max(counts) <= CAP
    reference = tmp_path / "ref.csv"
    reference.write_text("ID,answer\n1,B\n2,TRUE\n", encoding="utf-8")
    baseline = token_efficiency.BASELINE_TOKENS_PER_QUESTION * 2
    assert token_efficiency.token_efficiency(str(reference), str(out)) == pytest.approx(
        100 * baseline / (baseline + sum(counts))
    )
