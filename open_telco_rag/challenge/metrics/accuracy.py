"""TelcoRetrieve accuracy metric (Zindi custom metric, Python 3.13).

    Accuracy = 100 x correct answers / number of questions in the reference file   MAXIMISE, 0 to 100 (percent)

Questions are multiple-choice (answer = one option letter A to E) or true/false (answer = TRUE or
FALSE, any case). Exact match with the reference. An answer in any other form ("(B)", "Answer: B",
"T", empty) counts as wrong. The `input_prompt` column is not read here.

Submission: CSV, UTF-8, header row exactly   ID,input_prompt,answer   one row per question.
Reference:  CSV with a header row   ID,answer

Usage:   python accuracy.py Reference.csv submission.csv
Output:  one number (0 to 100) on stdout, or one line "error: ..." and exit code 1.
"""
import csv
import sys

SCORE_SCALE = 100                     # the score is a percentage
ID_COL, PROMPT_COL, ANSWER_COL = "ID", "input_prompt", "answer"
EXPECTED_COLUMNS = [ID_COL, PROMPT_COL, ANSWER_COL]
MC_LETTERS = "ABCDE"
csv.field_size_limit(1 << 30)


class MetricError(Exception):
    """Any problem with the submission; the message is shown to the participant."""


def normalise_answer(raw):
    """One letter A-E, or TRUE / FALSE (any case). Anything else -> None."""
    s = (raw or "").strip()
    if s.upper() in ("TRUE", "FALSE"):
        return s.upper()
    if len(s) == 1 and s.upper() in MC_LETTERS:
        return s.upper()
    return None


def read_reference(path):
    """Reference: header row, first column ID, second column the answer. Returns {id: answer}."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            next(reader, None)
            result = {}
            for row in reader:
                if not row:
                    continue
                if len(row) < 2:
                    raise MetricError("the reference file needs two columns: ID,answer")
                id_, gold = row[0].strip(), normalise_answer(row[1])
                if gold is None:
                    raise MetricError("the reference answer for %s is not a letter or TRUE/FALSE" % id_)
                if id_ in result:
                    raise MetricError("the reference file has duplicate IDs")
                result[id_] = gold
            return result
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


def accuracy(reference_path, submission_path):
    reference = read_reference(reference_path)
    if not reference:
        raise MetricError("the reference file has no rows")

    result, duplicates = {}, []
    for id_, row in read_csv(submission_path):
        if id_ in result or id_ in duplicates:
            duplicates.append(id_)
            continue
        if id_ in reference:              # IDs that are not in the reference are ignored
            result[id_] = normalise_answer(row[ANSWER_COL]) == reference[id_]

    if duplicates:
        raise MetricError(_ids_error("Duplicate", duplicates))
    missing = sorted(reference.keys() - result.keys())
    if missing:
        raise MetricError(_ids_error("Missing entries for", missing))
    return SCORE_SCALE * sum(result.values()) / len(reference)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python accuracy.py Reference.csv submission.csv")
    try:
        print(accuracy(sys.argv[1], sys.argv[2]))
    except MetricError as e:
        print("error: %s" % e)
        sys.exit(1)
