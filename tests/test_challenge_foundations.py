import pytest

from open_telco_rag.challenge.normalise import normalise_answer
from open_telco_rag.challenge.questions import load_questions
from open_telco_rag.challenge.tokens import count_tokens

BIG_ID = "781260748538"


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_ids_stay_strings_for_both_header_names(tmp_path):
    body = f'{BIG_ID},Which NF?,"{{""A"": ""AMF"", ""B"": ""SMF""}}"\n007,Is it?,{{}}\n'
    for header in ("ID", "question_id"):
        qs = load_questions(write(tmp_path / "t.csv", f"{header},question,options\n{body}"))
        assert [q.id for q in qs] == [BIG_ID, "007"]
        assert qs[0].options == {"A": "AMF", "B": "SMF"} and qs[1].is_true_false


def test_loader_rejects_duplicates_and_missing_id(tmp_path):
    with pytest.raises(ValueError, match="duplicate question ID 1"):
        load_questions(write(tmp_path / "a.csv", "ID,question,options\n1,q,{}\n1,q,{}\n"))
    with pytest.raises(ValueError, match="ID column"):
        load_questions(write(tmp_path / "b.csv", "num,question,options\n1,q,{}\n"))


@pytest.mark.parametrize(
    "text,options,expected",
    [
        ("B", {"A": "x", "B": "y"}, "B"),
        (" b. ", {"A": "x", "B": "y"}, "B"),
        ("(C)", {"A": "x", "C": "z"}, "C"),
        ("Answer: B", {"A": "x", "B": "y"}, "B"),
        ("The answer is a", {"A": "x", "B": "y"}, "A"),
        ("B) SMF", {"A": "x", "B": "y"}, "B"),
        ("E", {"A": "x", "B": "y"}, ""),
        ("A UE sends", {"A": "x"}, ""),
        ("", {"A": "x"}, ""),
        ("true", {}, "TRUE"),
        ("False.", {}, "FALSE"),
        ("Answer is TRUE", {}, "TRUE"),
        ("maybe", {}, ""),
    ],
)
def test_normalise_answer(text, options, expected):
    assert normalise_answer(text, options) == expected


def test_count_tokens_accepts_special_token_text():
    assert count_tokens("a <|endoftext|> b") > 3
