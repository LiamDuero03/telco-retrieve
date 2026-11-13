"""TelcoRetrieve token-efficiency metric (Zindi custom metric, Python 3.13).

    TokenEfficiency = 100 x BaselineToken / (BaselineToken + TotalTokenSize)  MAXIMISE, 0 < value <= 100 (percent)

    BaselineToken  = BASELINE_TOKENS_PER_QUESTION * number of questions in the reference file
    TotalTokenSize = tokens of the `input_prompt` column, summed over all questions

Accuracy is a separate metric (Zindi's accuracy metric on the `answer` column); this script
looks only at `input_prompt`.

Submission: CSV, UTF-8, header row exactly   ID,input_prompt,answer   one row per question.
Reference:  CSV with a header row; the first column is the ID (the other columns are not read).

Usage:   python token_efficiency.py Reference.csv submission.csv
Output:  one number (0 to 100) on stdout, or one line "error: ..." and exit code 1.

Tokens are counted with tiktoken, vocabulary cl100k_base, over the whole input_prompt string.
There is no fallback counter: if tiktoken or its vocabulary is not available the run stops with
an error. For an offline machine, put the vocabulary file in a folder and set TIKTOKEN_CACHE_DIR.
"""
import csv
import sys

# ----------------------------- challenge settings ----------------------------
BASELINE_TOKENS_PER_QUESTION = 850   # not confirmed
MAX_PROMPT_TOKENS = 4096              # a longer prompt makes the submission invalid
SCORE_SCALE = 100                     # the score is a percentage
ID_COL, PROMPT_COL, ANSWER_COL = "ID", "input_prompt", "answer"
# -----------------------------------------------------------------------------

EXPECTED_COLUMNS = [ID_COL, PROMPT_COL, ANSWER_COL]
csv.field_size_limit(1 << 30)


class MetricError(Exception):
    """Any problem with the submission; the message is shown to the participant."""


# ------------------------------ token counting -------------------------------
_encoder = None


def count_tokens(text):
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            raise MetricError("the token counter (tiktoken, cl100k_base) is not available: %s" % type(exc).__name__)
    # disallowed_special=(): text such as "<|endoftext|>" inside a prompt is counted, not an error
    return len(_encoder.encode(text, disallowed_special=()))


# ------------------------------- data processing -----------------------------
def read_reference_ids(path):
    """Reference: header row, first column is the ID."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            next(reader, None)
            return [row[0].strip() for row in reader if row]
    except UnicodeDecodeError:
        raise MetricError("unknown file type, the file must be UTF-8")


def read_csv(path):
    """Submission, read one row at a time. Yields [id, {input_prompt, answer}]."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise MetricError("the file is empty")
            if reader.fieldnames != EXPECTED_COLUMNS:
                found = ",".join(reader.fieldnames)
                raise MetricError("unexpected columns '%s', the header must be exactly %s" % (
                    found if len(found) <= 60 else found[:60] + "...", ",".join(EXPECTED_COLUMNS)))
            for n, row in enumerate(reader, start=2):
                if None in row or any(v is None for v in row.values()):
                    raise MetricError("row %d does not have exactly %d columns" % (n, len(EXPECTED_COLUMNS)))
                id_value = row.pop(ID_COL).strip()
                yield [id_value, row]
    except UnicodeDecodeError:
        raise MetricError("unknown file type, the file must be UTF-8")


def _ids_error(prefix, ids):
    error = "%s IDs %s" % (prefix, ", ".join(ids[:5]))
    return error + " and more" if len(ids) > 5 else error


# --------------------------------- metric ------------------------------------
def token_efficiency(reference_path, submission_path):
    reference = read_reference_ids(reference_path)
    if not reference:
        raise MetricError("the reference file has no rows")
    if len(set(reference)) != len(reference):
        raise MetricError("the reference file has duplicate IDs")
    reference = set(reference)

    tokens, duplicates, empty, too_long = {}, [], [], []
    for id_, row in read_csv(submission_path):
        if id_ in tokens or id_ in duplicates:
            duplicates.append(id_)
            continue
        if id_ not in reference:          # IDs that are not in the reference are ignored
            continue
        prompt = row[PROMPT_COL]
        if not prompt.strip():
            empty.append(id_)
            tokens[id_] = 0
            continue
        n_tokens = count_tokens(prompt)
        if n_tokens > MAX_PROMPT_TOKENS:
            too_long.append(id_)
        tokens[id_] = n_tokens

    missing = sorted(reference - tokens.keys())
    if duplicates:
        raise MetricError(_ids_error("Duplicate", duplicates))
    if missing:
        raise MetricError(_ids_error("Missing entries for", missing))
    if empty:
        raise MetricError(_ids_error("Empty input_prompt for", empty))
    if too_long:
        raise MetricError(_ids_error("input_prompt over %d tokens for" % MAX_PROMPT_TOKENS, too_long))

    baseline = BASELINE_TOKENS_PER_QUESTION * len(reference)
    return SCORE_SCALE * baseline / (baseline + sum(tokens.values()))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python token_efficiency.py Reference.csv submission.csv")
    try:
        print(token_efficiency(sys.argv[1], sys.argv[2]))
    except MetricError as e:
        print("error: %s" % e)
        sys.exit(1)
