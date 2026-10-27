"""Growth-aware Re-ID evaluator with time-gap and metadata buckets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from catnet.eval.metrics import auroc, baks_baus, mean_average_precision, rank_at_k, tnr_at_tpr


@dataclass(frozen=True)
class GrowthEvalCase:
    query_obs_id: str
    true_cat_id: str | None
    candidates: list[tuple[str, float]]
    is_known: bool
    time_gap_days: float | None = None
    age_bin: str | None = None
    camera_id: str | None = None
    site_id: str | None = None
    view: str | None = None

    def to_metric_case(self) -> dict[str, object]:
        return {
            "query_obs_id": self.query_obs_id,
            "true_cat_id": self.true_cat_id,
            "candidates": list(self.candidates),
            "is_known": self.is_known,
        }


class GrowthAwareEvaluator:
    def __init__(
        self,
        *,
        threshold: float = 0.86,
        time_gap_bins: tuple[float, ...] = (0, 7, 30, 90, 180, 365),
    ) -> None:
        self.threshold = threshold
        self.time_gap_bins = time_gap_bins

    def evaluate(self, cases: list[GrowthEvalCase]) -> dict[str, Any]:
        metric_cases = [case.to_metric_case() for case in cases]
        labels = [1 if case.is_known else 0 for case in cases]
        scores = [_top_score(case) for case in cases]
        open_set = baks_baus(metric_cases, threshold=self.threshold)
        overall = {
            "rank1": rank_at_k(metric_cases, k=1),
            "rank5": rank_at_k(metric_cases, k=5),
            "mAP": mean_average_precision(metric_cases),
            "AUROC": auroc(labels, scores),
            "TNR@95TPR": tnr_at_tpr(labels, scores, target_tpr=0.95),
            **open_set,
        }
        time_gap_curve = self._time_gap_curve(cases)
        return {
            "schema_version": "1.0",
            "threshold": self.threshold,
            "overall": overall,
            "time_gap_curve": time_gap_curve,
            "age_bin": self._bucket_metrics(cases, key="age_bin"),
            "camera": self._bucket_metrics(cases, key="camera_id"),
            "site": self._bucket_metrics(cases, key="site_id"),
            "view": self._bucket_metrics(cases, key="view"),
            "identity_persistence_score": self._identity_persistence(cases),
            "long_term_drift_metric": self._long_term_drift(time_gap_curve),
            "case_count": len(cases),
        }

    def _time_gap_curve(self, cases: list[GrowthEvalCase]) -> list[dict[str, Any]]:
        curve = []
        edges = list(self.time_gap_bins)
        for lower, upper in zip(edges, [*edges[1:], None], strict=False):
            bucket = [
                case
                for case in cases
                if case.time_gap_days is not None
                and case.time_gap_days >= lower
                and (upper is None or case.time_gap_days < upper)
            ]
            curve.append(
                {
                    "bin": f"{lower:g}-{upper:g}d" if upper is not None else f"{lower:g}d+",
                    "lower_days": lower,
                    "upper_days": upper,
                    "case_count": len(bucket),
                    "metrics": self._basic_bucket_metrics(bucket),
                }
            )
        missing = [case for case in cases if case.time_gap_days is None]
        if missing:
            curve.append(
                {
                    "bin": "unknown",
                    "lower_days": None,
                    "upper_days": None,
                    "case_count": len(missing),
                    "metrics": self._basic_bucket_metrics(missing),
                }
            )
        return curve

    def _bucket_metrics(
        self,
        cases: list[GrowthEvalCase],
        *,
        key: str,
    ) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[GrowthEvalCase]] = {}
        for case in cases:
            value = getattr(case, key) or "unknown"
            buckets.setdefault(value, []).append(case)
        return {
            name: self._basic_bucket_metrics(bucket)
            for name, bucket in sorted(buckets.items())
        }

    def _basic_bucket_metrics(self, cases: list[GrowthEvalCase]) -> dict[str, float]:
        metric_cases = [case.to_metric_case() for case in cases]
        return {
            "rank1": rank_at_k(metric_cases, k=1),
            "mAP": mean_average_precision(metric_cases),
            **baks_baus(metric_cases, threshold=self.threshold),
        }

    def _identity_persistence(self, cases: list[GrowthEvalCase]) -> float:
        known = [case for case in cases if case.is_known]
        if not known:
            return 0.0
        persistent = 0
        for case in known:
            top = case.candidates[0] if case.candidates else None
            if top and top[0] == case.true_cat_id and top[1] >= self.threshold:
                persistent += 1
        return persistent / len(known)

    def _long_term_drift(self, time_gap_curve: list[dict[str, Any]]) -> float:
        populated = [
            item for item in time_gap_curve if item["case_count"] > 0 and item["bin"] != "unknown"
        ]
        if len(populated) < 2:
            return 0.0
        first = populated[0]["metrics"]["rank1"]
        last = populated[-1]["metrics"]["rank1"]
        return max(0.0, first - last)


def _top_score(case: GrowthEvalCase) -> float:
    return float(case.candidates[0][1]) if case.candidates else -1.0


def report_as_dict(report: dict[str, Any]) -> dict[str, Any]:
    return asdict(report) if hasattr(report, "__dataclass_fields__") else dict(report)
