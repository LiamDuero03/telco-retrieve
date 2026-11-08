import pytest

from open_telco_rag.challenge.prompt import PromptTooLong, build_prompt
from open_telco_rag.challenge.questions import ChallengeQuestion
from open_telco_rag.challenge.tokens import count_tokens
from open_telco_rag.types import SearchResult

BIG_ID = "781260748538"


def passages(n):
    return [
        SearchResult(
            text=f"passage text number {i} " * 5,
            distance=0.1 * i,
            metadata={"spec_number": f"23.50{i}", "release": "Rel-19"},
            doc_id=f"d{i}",
        )
        for i in range(n)
    ]


def test_prompt_layout_and_input_prompt_matches_messages():
    q = ChallengeQuestion(BIG_ID, "Which NF?", {"A": "AMF", "B": "SMF"})
    p = build_prompt(q, passages(2))
    system, user = (m["content"] for m in p.messages)
    assert p.input_prompt == f"{system}\n\n{user}"
    assert (
        system == "You answer 3GPP questions from the provided context. Reply with only the answer."
    )
    assert "[1] Spec: 23.500 | Release: Rel-19" in user and "A. AMF\nB. SMF" in user
    assert user.endswith("Answer with only the letter of the correct option.")
    assert p.n_tokens == count_tokens(p.input_prompt) and p.n_passages == 2


def test_true_false_prompt_has_no_options_block():
    p = build_prompt(ChallengeQuestion("1", "The UE registers.", {}), passages(1))
    user = p.messages[1]["content"]
    assert "Options:" not in user and user.endswith("Answer with only TRUE or FALSE.")


def test_cap_boundary_drops_lowest_ranked_and_errors_with_id():
    q = ChallengeQuestion(BIG_ID, "Which NF?", {"A": "AMF"})
    ps = passages(4)
    sizes = [build_prompt(q, ps[:k]).n_tokens for k in range(5)]
    assert build_prompt(q, ps, sizes[4]).n_passages == 4
    assert build_prompt(q, ps, sizes[4] - 1).n_passages == 3
    assert build_prompt(q, ps, sizes[0]).n_passages == 0
    with pytest.raises(PromptTooLong, match=BIG_ID):
        build_prompt(q, ps, sizes[0] - 1)
