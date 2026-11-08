from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from open_telco_rag.challenge.metrics import accuracy, overall_score, token_efficiency

MetricError = (accuracy.MetricError, token_efficiency.MetricError)


@dataclass
class ScoreResult:
    accuracy: float
    token_efficiency: float
    overall: float
    minimum_met: bool


def score_submission(submission_path: str | Path, reference_path: str | Path) -> ScoreResult:
    """The official metrics on a submission; errors are the official MetricError."""
    accuracy_score = accuracy.accuracy(str(reference_path), str(submission_path))
    efficiency_score = token_efficiency.token_efficiency(str(reference_path), str(submission_path))
    overall = (
        overall_score.WEIGHT_ACCURACY * accuracy_score
        + overall_score.WEIGHT_TOKEN_EFFICIENCY * efficiency_score
    )
    return ScoreResult(
        accuracy_score, efficiency_score, overall, accuracy_score >= overall_score.MIN_ACCURACY
    )
