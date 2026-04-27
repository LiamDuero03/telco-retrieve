"""
Result dataclasses, serialisation, and display for evaluation runs.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class QuestionResult:
    """All metric scores for one question."""
    question:     str
    corpus:       str
    answer:       str
    sources:      list[str]
    chunks_used:  int

    # ── Tier 1: no ground truth ───────────────────────────────────────────────
    keyword_score:    float = 0.0
    faithfulness:     float = 0.0
    answer_relevancy: float = 0.0
    mean_similarity:  float = 0.0
    min_similarity:   float = 0.0
    max_similarity:   float = 0.0
    source_coverage:  int   = 0
    canonical_hit:    int   = -1   # -1 = not applicable (no canonical defined)
    coverage_ratio:   float = 0.0

    # ── Tier 2: ground truth required (-1.0 = not computed) ──────────────────
    context_recall:      float = -1.0
    answer_correctness:  float = -1.0

    error: str = ""


@dataclass
class CorpusResult:
    """Aggregate results across all questions for one corpus."""
    corpus:    str
    model:     str = ""
    top_k:     int = 5
    timestamp: str = ""

    n_questions: int = 0

    # ── Tier 1 aggregates ─────────────────────────────────────────────────────
    mean_keyword_score:    float = 0.0
    mean_faithfulness:     float = 0.0
    mean_answer_relevancy: float = 0.0
    mean_similarity:       float = 0.0
    mean_source_coverage:  float = 0.0
    canonical_hit_rate:    float = -1.0
    mean_coverage_ratio:   float = 0.0

    # ── Tier 2 aggregates (-1.0 = not computed) ───────────────────────────────
    mean_context_recall:      float = -1.0
    mean_answer_correctness:  float = -1.0

    question_results: list[QuestionResult] = field(default_factory=list)


def aggregate(question_results: list[QuestionResult]) -> dict:
    """Compute aggregate metrics from a list of QuestionResults."""

    def _mean(values: list[float]) -> float:
        valid = [v for v in values if v >= 0]
        return statistics.mean(valid) if valid else -1.0

    canonical_qs = [r for r in question_results if r.canonical_hit >= 0]

    return dict(
        n_questions=             len(question_results),
        mean_keyword_score=      _mean([r.keyword_score     for r in question_results]),
        mean_faithfulness=       _mean([r.faithfulness      for r in question_results]),
        mean_answer_relevancy=   _mean([r.answer_relevancy  for r in question_results]),
        mean_similarity=         _mean([r.mean_similarity   for r in question_results]),
        mean_source_coverage=    _mean([float(r.source_coverage) for r in question_results]),
        canonical_hit_rate=     (sum(r.canonical_hit for r in canonical_qs) /
                                  len(canonical_qs) if canonical_qs else -1.0),
        mean_coverage_ratio=     _mean([r.coverage_ratio    for r in question_results]),
        mean_context_recall=     _mean([r.context_recall    for r in question_results]),
        mean_answer_correctness= _mean([r.answer_correctness for r in question_results]),
    )


def save_results(result: CorpusResult, output_dir: Path) -> Path:
    """Serialise CorpusResult to a timestamped JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts       = result.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{result.corpus}_{result.model}_{ts}.json"
    path     = output_dir / filename
    path.write_text(json.dumps(asdict(result), indent=2, default=str),
                    encoding="utf-8")
    print(f"  Saved → {path}")
    return path


def print_summary(result: CorpusResult) -> None:
    """Print a formatted summary table to stdout."""
    w = 62
    print(f"\n{'='*w}")
    print(f"  {result.corpus.upper()}  |  model={result.model}  |  "
          f"top-k={result.top_k}  |  n={result.n_questions}")
    print(f"{'='*w}")
    print(f"  {'Metric':<30} {'Score':>8}")
    print(f"  {'-'*38}")

    rows = [
        ("Keyword score",         result.mean_keyword_score),
        ("Faithfulness",          result.mean_faithfulness),
        ("Answer relevancy",      result.mean_answer_relevancy),
        ("Mean similarity",       result.mean_similarity),
        ("Source coverage (mean)",result.mean_source_coverage),
        ("Canonical hit rate",    result.canonical_hit_rate),
        ("Coverage ratio",        result.mean_coverage_ratio),
        ("Context recall",        result.mean_context_recall),
        ("Answer correctness",    result.mean_answer_correctness),
    ]
    for label, value in rows:
        if value < 0:
            print(f"  {label:<30} {'N/A':>8}")
        else:
            fmt = ".1f" if label == "Source coverage (mean)" else ".3f"
            print(f"  {label:<30} {value:>8{fmt}}")
    print(f"{'='*w}\n")