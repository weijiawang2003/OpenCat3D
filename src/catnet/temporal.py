"""Temporal and growth-aware compatibility scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

from catnet.schema import Observation


@dataclass(frozen=True)
class TemporalScore:
    score: float
    gap_days: float | None
    evidence: dict[str, object] = field(default_factory=dict)


class TemporalGrowthScorer:
    """Score whether two observations are plausible for one growing cat."""

    def score(self, query: Observation | None, candidate: Observation | None) -> TemporalScore:
        if query is None or candidate is None:
            return TemporalScore(
                score=0.72,
                gap_days=None,
                evidence={"reason": "missing_observation"},
            )
        if query.captured_at is None or candidate.captured_at is None:
            return TemporalScore(
                score=0.74,
                gap_days=None,
                evidence={"reason": "missing_timestamp"},
            )

        gap_days = abs((query.captured_at - candidate.captured_at).total_seconds()) / 86400.0
        if gap_days <= 7:
            base = 0.96
        elif gap_days <= 30:
            base = 0.92
        elif gap_days <= 90:
            base = 0.86
        elif gap_days <= 180:
            base = 0.80
        elif gap_days <= 365:
            base = 0.73
        else:
            base = 0.66

        evidence: dict[str, object] = {"gap_days": round(gap_days, 3)}
        if query.site_id and candidate.site_id:
            if query.site_id == candidate.site_id:
                base += 0.04
                evidence["site"] = "same"
            else:
                base -= 0.06
                evidence["site"] = "different"
        if query.camera_id and candidate.camera_id and query.camera_id == candidate.camera_id:
            base += 0.02
            evidence["camera"] = "same"
        if query.age_bin and candidate.age_bin:
            evidence["age_transition"] = f"{candidate.age_bin}->{query.age_bin}"
            if query.age_bin == candidate.age_bin:
                base += 0.02
            elif {query.age_bin, candidate.age_bin}.issubset({"kitten", "juvenile", "adult"}):
                base += 0.01
            else:
                base -= 0.03
        return TemporalScore(score=max(0.0, min(1.0, base)), gap_days=gap_days, evidence=evidence)
