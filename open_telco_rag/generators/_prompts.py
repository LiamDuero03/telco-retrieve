"""Shared RAG-generation system prompts, selected via each Generator's
`prompt_mode` param (also `--prompt-mode` on the targeted-eval scripts).

"strict"  -- answer only from the context, abstain otherwise. Isolates
             retrieval faithfulness.
"hybrid"  -- prioritise the context but fall back to the model's own
             knowledge when it's incomplete; abstain only if neither answers.

Results from different prompt_mode values are not directly comparable - don't
mix them in one table without labelling which is which.
"""
from __future__ import annotations

STRICT = (
    "You are a {source_type} technical expert. "
    "Answer questions using only the context provided. "
    "If the answer is not in the context, say so clearly. "
    "Be precise and technical."
)

HYBRID = (
    "You are a {source_type} technical expert. "
    "Prioritize the provided context — it is the specific, authoritative "
    "source for this question. If the context does not fully answer it, "
    "supplement it with your own reliable knowledge of {source_type} "
    "standards and specifications. Only say you don't know if you cannot "
    "answer confidently from the context OR your own knowledge combined. "
    "When you rely on knowledge beyond the provided context, note that "
    "briefly. Be precise and technical."
)

PROMPT_VARIANTS = {"strict": STRICT, "hybrid": HYBRID}
DEFAULT_PROMPT_MODE = "strict"


def resolve_system_prompt(source_type: str, prompt_mode: str = DEFAULT_PROMPT_MODE) -> str:
    try:
        template = PROMPT_VARIANTS[prompt_mode]
    except KeyError:
        raise ValueError(
            f"Unknown prompt_mode {prompt_mode!r} -- choose from {sorted(PROMPT_VARIANTS)}"
        )
    return template.format(source_type=source_type)
