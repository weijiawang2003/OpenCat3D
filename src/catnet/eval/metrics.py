"""Re-ID, open-set, temporal, and graph metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from math import sqrt


def rank_at_k(cases: Sequence[dict[str, object]], *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    known = [case for case in cases if case.get("is_known", True)]
    if not known:
        return 0.0
    hits = 0
    for case in known:
        true_cat_id = case["true_cat_id"]
        candidates = list(case.get("candidates", []))[:k]
        if any(candidate[0] == true_cat_id for candidate in candidates):
            hits += 1
    return hits / len(known)


def mean_average_precision(cases: Sequence[dict[str, object]]) -> float:
    known = [case for case in cases if case.get("is_known", True)]
    if not known:
        return 0.0
    ap_values = []
    for case in known:
        true_cat_id = case["true_cat_id"]
        hits = 0
        precision_sum = 0.0
        for rank, candidate in enumerate(case.get("candidates", []), start=1):
            if candidate[0] == true_cat_id:
                hits += 1
                precision_sum += hits / rank
        ap_values.append(precision_sum / hits if hits else 0.0)
    return sum(ap_values) / len(ap_values)


def baks_baus(cases: Sequence[dict[str, object]], *, threshold: float) -> dict[str, float]:
    known_cases = [case for case in cases if case.get("is_known", True)]
    unknown_cases = [case for case in cases if not case.get("is_known", True)]

    known_correct = 0
    for case in known_cases:
        top = _top_candidate(case)
        if top is not None and top[0] == case["true_cat_id"] and float(top[1]) >= threshold:
            known_correct += 1
    unknown_rejected = 0
    for case in unknown_cases:
        top = _top_candidate(case)
        top_score = float(top[1]) if top is not None else -1.0
        if top_score < threshold:
            unknown_rejected += 1

    baks = known_correct / len(known_cases) if known_cases else 0.0
    baus = unknown_rejected / len(unknown_cases) if unknown_cases else 0.0
    balanced = sqrt(baks * baus) if baks > 0.0 and baus > 0.0 else 0.0
    total = len(known_cases) + len(unknown_cases)
    rejection_accuracy = (
        (known_correct + unknown_rejected) / total if total else 0.0
    )
    return {
        "BAKS": baks,
        "BAUS": baus,
        "BAKS_BAUS": balanced,
        "rejection_accuracy": rejection_accuracy,
    }


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = [
        (score, idx)
        for idx, (label, score) in enumerate(zip(labels, scores, strict=True))
        if label == 1
    ]
    negatives = [
        (score, idx)
        for idx, (label, score) in enumerate(zip(labels, scores, strict=True))
        if label == 0
    ]
    if not positives or not negatives:
        return 0.0
    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    rank_sum = 0.0
    for rank, (idx, _score) in enumerate(ranked, start=1):
        if labels[idx] == 1:
            rank_sum += rank
    n_pos = len(positives)
    n_neg = len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def tnr_at_tpr(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    target_tpr: float = 0.95,
) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sorted(
        [score for label, score in zip(labels, scores, strict=True) if label == 1],
        reverse=True,
    )
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.0
    cutoff_index = min(len(positives) - 1, max(0, int(target_tpr * len(positives)) - 1))
    threshold = positives[cutoff_index]
    true_negatives = sum(1 for score in negatives if score < threshold)
    return true_negatives / len(negatives)


def cluster_purity(assignments: Iterable[tuple[str, str]]) -> float:
    clusters: dict[str, Counter[str]] = defaultdict(Counter)
    total = 0
    for true_cat_id, predicted_cluster_id in assignments:
        clusters[predicted_cluster_id][true_cat_id] += 1
        total += 1
    if total == 0:
        return 0.0
    pure = sum(counter.most_common(1)[0][1] for counter in clusters.values() if counter)
    return pure / total


def merge_error_rate(proposals: Iterable[tuple[str, str, bool]]) -> float:
    proposals_list = list(proposals)
    if not proposals_list:
        return 0.0
    errors = sum(1 for _left, _right, same_identity in proposals_list if not same_identity)
    return errors / len(proposals_list)


def split_error_rate(assignments: Iterable[tuple[str, str]]) -> float:
    true_to_predicted: dict[str, set[str]] = defaultdict(set)
    for true_cat_id, predicted_cluster_id in assignments:
        true_to_predicted[true_cat_id].add(predicted_cluster_id)
    if not true_to_predicted:
        return 0.0
    split_count = sum(1 for predicted in true_to_predicted.values() if len(predicted) > 1)
    return split_count / len(true_to_predicted)


def _top_candidate(case: dict[str, object]) -> tuple[str, float] | None:
    candidates = list(case.get("candidates", []))
    if not candidates:
        return None
    top = candidates[0]
    return str(top[0]), float(top[1])
