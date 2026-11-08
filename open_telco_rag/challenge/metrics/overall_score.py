"""Scores a submission locally with both metrics, the way the leaderboard shows them.

    python overall_score.py Reference.csv submission.csv

Prints accuracy, token efficiency, the overall score and whether the 80% minimum accuracy is met.
Overall = 0.7 x Accuracy + 0.3 x TokenEfficiency, all three out of 100. On Zindi the weights are configured on the
platform (provisionally 70% / 30%); this file is for participants to check a CSV before submitting.
"""
import sys

import accuracy as A
import token_efficiency as T

WEIGHT_ACCURACY, WEIGHT_TOKEN_EFFICIENCY = 0.7, 0.3   # provisional
MIN_ACCURACY = 80                                      # percent, minimum to advance to code evaluation

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python overall_score.py Reference.csv submission.csv")
    try:
        acc = A.accuracy(sys.argv[1], sys.argv[2])
        eff = T.token_efficiency(sys.argv[1], sys.argv[2])
    except (A.MetricError, T.MetricError) as e:
        print("error: %s" % e)
        sys.exit(1)
    print("Accuracy:          %.2f" % acc)
    print("Token efficiency:  %.2f" % eff)
    print("Overall:           %.2f" % (WEIGHT_ACCURACY * acc + WEIGHT_TOKEN_EFFICIENCY * eff))
    print("Minimum accuracy (%d%%): %s" % (MIN_ACCURACY, "met" if acc >= MIN_ACCURACY else "NOT met, not eligible for code evaluation"))
