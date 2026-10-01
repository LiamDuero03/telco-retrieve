"""Opt-in hard token bound for chunks, enforced by a final splitting pass over any strategy's output.

A chunk over the bound is split at the coarsest boundary that works: paragraphs, then table/text
blocks, then sentences, then lines, and last a split by token count. A markdown table is split by
rows with its header rows repeated. Chunks within the bound are returned untouched.
"""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from open_telco_rag.types import Chunk

Counter = Callable[[str], int]
_COUNTERS: dict[str, Counter] = {}
_SEPARATOR_ROW = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


def _registry_model_name(name: str) -> str:
    """map an embedding registry key such as "bge-m3" to its model name; anything else is returned as given"""
    path = Path(__file__).resolve().parents[2] / "configs" / "embedding_models.json"
    try:
        return json.loads(path.read_text(encoding="utf8")).get(name, {}).get("model_name", name)
    except (OSError, ValueError):
        return name


def get_counter(tokenizer: str | Counter | None = None) -> Counter:
    """a function counting the tokens of a text.

    None is cl100k_base, the encoding of the other token paths. "tiktoken:<encoding>" is any tiktoken
    encoding. Anything else is an embedding registry key, a Hugging Face model name (only its
    tokenizer.json is fetched, never the weights) or the path of a tokenizer.json. A callable is used as is.
    Special tokens are not counted: the bound's margin reserves room for them."""
    if callable(tokenizer):
        return tokenizer
    if tokenizer is None:
        from open_telco_rag.core.chunking import _count_tokens
        return _count_tokens
    if tokenizer in _COUNTERS:
        return _COUNTERS[tokenizer]
    if tokenizer.startswith("tiktoken:"):
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer.split(":", 1)[1])
        counter: Counter = lambda text: len(enc.encode(text)) if text else 0
    else:
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError("a Hugging Face tokenizer needs the `tokenizers` package; "
                              "install with: pip install open-telco-rag[local-embed]") from e
        model = _registry_model_name(tokenizer)
        tok = Tokenizer.from_file(model) if model.endswith(".json") and Path(model).exists() else Tokenizer.from_pretrained(model)
        tok.no_truncation()      # some tokenizer.json files cap encode() at the model's max length
        tok.no_padding()
        counter = lambda text: len(tok.encode(text, add_special_tokens=False).ids) if text else 0
    _COUNTERS[tokenizer] = counter
    return counter


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _runs(text: str, a: int, b: int, pattern: str) -> list[tuple[int, int]]:
    """non-blank spans of text[a:b] separated by `pattern`"""
    out, pos = [], a
    for m in list(re.finditer(pattern, text[a:b])) + [None]:
        end = b if m is None else a + m.start()
        s, e = _trim(text, pos, end)
        if s < e:
            out.append((s, e))
        if m is not None:
            pos = a + m.end()
    return out


def _paragraphs(text: str, a: int, b: int) -> list[tuple[int, int]]:
    return _runs(text, a, b, r"\n[ \t]*\n")


def _lines(text: str, a: int, b: int) -> list[tuple[int, int]]:
    return _runs(text, a, b, r"\n")


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _blocks(text: str, a: int, b: int) -> list[tuple[int, int]]:
    """consecutive table lines and consecutive other lines are one block each"""
    out: list[tuple[int, int]] = []
    kind = None
    for s, e in _lines(text, a, b):
        k = _is_table_line(text[s:e])
        if out and k == kind:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
        kind = k
    return out


def _sentences(text: str, a: int, b: int) -> list[tuple[int, int]]:
    from open_telco_rag.core.chunking import _sentence_spans
    return [(a + s, a + e) for s, e in _sentence_spans(text[a:b])]


_UNITS = (_paragraphs, _blocks, _sentences, _lines)
Piece = tuple[str, int, int]       # (text, start, end); text is text[start:end] except for repeated-header table parts


def _fit(text: str, a: int, b: int, limit: int, count: Counter) -> int:
    """the largest n with count(text[a:a+n]) <= limit (at least 1), by exponential then binary search"""
    rest = b - a
    lo, n = 0, min(rest, max(1, limit))
    while True:
        if count(text[a:a + n]) > limit:
            hi = n
            break
        lo = n
        if n == rest:
            return n
        n = min(rest, n * 2)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(text[a:a + mid]) <= limit:
            lo = mid
        else:
            hi = mid
    return max(lo, 1)       # one character that alone exceeds the bound is emitted as it is


def _hard(text: str, a: int, b: int, limit: int, count: Counter) -> list[Piece]:
    out: list[Piece] = []
    while a < b:
        n = _fit(text, a, b, limit, count)
        out.append((text[a:a + n], a, a + n))
        a += n
    return out


def _table_header(text: str, a: int, b: int) -> str | None:
    """the header rows (a row and its |---| separator row) of the last table in text[a:b] that has one"""
    lines = _lines(text, a, b)
    found = None
    for k in range(len(lines) - 1):
        if _is_table_line(text[lines[k][0]:lines[k][1]]) and _SEPARATOR_ROW.match(text[lines[k + 1][0]:lines[k + 1][1]]):
            found = text[lines[k][0]:lines[k + 1][1]]
    return found


def _split_table(text: str, a: int, b: int, limit: int, count: Counter, hint: str | None = None) -> list[Piece] | None:
    """split a markdown table by rows; the header rows are repeated at the top of every part after the first.
    A table without header rows (the rest of a table that an earlier chunk started) uses `hint`, the header of
    that earlier table, on every part."""
    lines = _lines(text, a, b)
    if len(lines) < 3 or not all(_is_table_line(text[s:e]) for s, e in lines):
        return None
    own = _SEPARATOR_ROW.match(text[lines[1][0]:lines[1][1]]) is not None
    header = text[lines[0][0]:lines[1][1]] if own else hint
    if header is None:                      # no header to repeat: plain row packing
        return _pack(text, lines, limit, count, 3)
    if count(header) * 2 >= limit:          # no room left for rows
        return None
    rows = lines[2:] if own else lines
    out: list[Piece] = []
    i, first = 0, own
    while i < len(rows):
        prefix = "" if first else header + "\n"
        start = lines[0][0] if first else rows[i][0]
        j = i
        while j < len(rows) and count(prefix + text[start:rows[j][1]]) <= limit:
            j += 1
        if j == i and first:                # the header and one row do not fit together: the header goes alone
            out.append((header, lines[0][0], lines[1][1]))
            first = False
            continue
        if j == i:                          # one row does not fit under the repeated header: split the row itself
            out.extend(_split(text, rows[i][0], rows[i][1], limit, count, 2))
            i += 1
            continue
        out.append((prefix + text[start:rows[j - 1][1]], start, rows[j - 1][1]))
        i, first = j, False
    return out


def _split(text: str, a: int, b: int, limit: int, count: Counter, level: int = 0, hint: str | None = None) -> list[Piece]:
    a, b = _trim(text, a, b)
    if a >= b:
        return []
    if count(text[a:b]) <= limit:
        return [(text[a:b], a, b)]
    table = _split_table(text, a, b, limit, count, hint)
    if table is not None:
        return table
    for lv in range(level, len(_UNITS)):
        units = _UNITS[lv](text, a, b)
        if len(units) > 1:
            return _pack(text, units, limit, count, lv + 1, hint)
    return _hard(text, a, b, limit, count)


def _pack(text: str, units: list[tuple[int, int]], limit: int, count: Counter, next_level: int,
          hint: str | None = None) -> list[Piece]:
    """join consecutive units while they fit; a unit that does not fit alone is split at the next level"""
    out: list[Piece] = []
    group: tuple[int, int] | None = None
    for k, (s, e) in enumerate(units):
        if group is not None and count(text[group[0]:e]) <= limit:
            group = (group[0], e)
            continue
        if group is not None:
            out.append((text[group[0]:group[1]], *group))
            group = None
        if count(text[s:e]) <= limit:
            group = (s, e)
        else:
            out.extend(_split(text, s, e, limit, count, next_level, hint if k == 0 else None))
    if group is not None:
        out.append((text[group[0]:group[1]], *group))
    return out


def split_to_bound(text: str, limit: int, count: Counter, hint: str | None = None) -> list[Piece]:
    """pieces of `text`, each at most `limit` tokens, in order; `hint` is the header of a table that the text continues"""
    return _split(text, 0, len(text), limit, count, 0, hint)


def enforce_bound(chunks: list[Chunk], limit: int, tokenizer: str | Counter | None = None) -> list[Chunk]:
    """split every chunk of more than `limit` tokens; chunks within the bound are the same objects.
    Heading, source type and any metadata are kept; token offsets are kept exact for token-sized chunks.
    chunk_index is renumbered when anything was split."""
    from open_telco_rag.core.chunking import _count_tokens

    count = get_counter(tokenizer)
    out: list[Chunk] = []
    changed = False
    carry: str | None = None                # header of a table that the previous chunk ended inside
    for chunk in chunks:
        text = chunk.text
        previous, carry = carry, (_table_header(text, 0, len(text)) or carry)
        if not _is_table_line(text.rstrip().rsplit("\n", 1)[-1]):
            carry = None
        if count(text) <= limit:
            out.append(chunk)
            continue
        changed = True
        for piece, start, end in split_to_bound(text, limit, count, previous):
            new = Chunk(text=piece, heading=chunk.heading, chunk_index=0, source_type=chunk.source_type)
            if getattr(chunk, "metadata", None):
                new.metadata = copy.deepcopy(chunk.metadata)
            if chunk.char_start >= 0:
                new.char_start = chunk.char_start + start
                new.char_end = chunk.char_start + end
                new.n_tokens = _count_tokens(piece)
            out.append(new)
    if changed:
        for i, chunk in enumerate(out):
            chunk.chunk_index = i
    return out


def validate_bound(bound_tokens: Any, bound_tokenizer: Any, bound_margin: Any) -> int:
    """check the three settings and return the effective limit (bound minus margin)"""
    if isinstance(bound_tokens, bool) or not isinstance(bound_tokens, int) or bound_tokens <= 0:
        raise ValueError(f"bound_tokens must be a positive integer, got {bound_tokens!r}")
    margin = 2 if bound_margin is None else bound_margin
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0 or margin >= bound_tokens:
        raise ValueError(f"bound_margin must be an integer from 0 to bound_tokens - 1, got {bound_margin!r}")
    if bound_tokenizer is not None and not (isinstance(bound_tokenizer, str) and bound_tokenizer) and not callable(bound_tokenizer):
        raise ValueError(f"bound_tokenizer must be a name, a tokenizer.json path or a callable, got {bound_tokenizer!r}")
    return bound_tokens - margin
