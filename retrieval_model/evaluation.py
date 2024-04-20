import numpy as np


def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score)


def calculate_n_dcg(actual_scores, target_scores):
    actual_scores = normalize_scores(actual_scores)
    paired_scores = [((actual * 0.5) if target == 0.0 else actual, target) for actual, target in
                     zip(actual_scores, target_scores)]
    paired_scores.sort(key=lambda x: x[1], reverse=True)
    dcg = sum((2 ** true - 1) / np.log2(i + 2) for i, (true, pred) in enumerate(paired_scores))
    true_scores_sorted = sorted(actual_scores, reverse=True)
    i_dcg = sum((2 ** true - 1) / np.log2(i + 2) for i, true in enumerate(true_scores_sorted))
    n_dcg = dcg / i_dcg
    return n_dcg
