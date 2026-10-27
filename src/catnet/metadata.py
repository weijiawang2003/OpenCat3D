"""Metadata posterior scoring for CatNet open-set decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import fabs
from typing import Any

from catnet.schema import Observation


@dataclass(frozen=True)
class MetadataScore:
    score: float
    evidence: dict[str, object] = field(default_factory=dict)


class MetadataPosteriorScorer:
    """Score non-visual context compatibility for two cat observations.

    The scorer is deliberately conservative. Metadata can support or weaken a
    visual match, but should not decide identity alone because crowd-sourced
    locations, camera labels, and uploader ids can be noisy.
    """

    def score(self, query: Observation | None, candidate: Observation | None) -> MetadataScore:
        if query is None or candidate is None:
            return MetadataScore(score=0.72, evidence={"reason": "missing_observation"})

        score = 0.72
        evidence: dict[str, object] = {}

        score += _species_adjustment(query, candidate, evidence)
        score += _site_adjustment(query, candidate, evidence)
        score += _camera_adjustment(query, candidate, evidence)
        score += _geohash_adjustment(query, candidate, evidence)
        score += _uploader_adjustment(query, candidate, evidence)
        score += _encounter_adjustment(query, candidate, evidence)
        score += _source_adjustment(query, candidate, evidence)
        score += _time_of_day_adjustment(query, candidate, evidence)
        score += _quality_adjustment(query, candidate, evidence)

        return MetadataScore(score=max(0.0, min(1.0, score)), evidence=evidence)


def _species_adjustment(
    query: Observation,
    candidate: Observation,
    evidence: dict[str, object],
) -> float:
    if not query.species or not candidate.species:
        return 0.0
    if query.species == candidate.species:
        evidence["species"] = "same"
        return 0.04
    evidence["species"] = "different"
    return -0.30


def _site_adjustment(
    query: Observation,
    candidate: Observation,
    evidence: dict[str, object],
) -> float:
    if not query.site_id or not candidate.site_id:
        evidence["site"] = "missing"
        return 0.0
    if query.site_id == candidate.site_id:
        evidence["site"] = "same"
        return 0.08
    evidence["site"] = "different"
    return -0.08


def _camera_adjustment(
    query: Observation,
    candidate: Observation,
    evidence: dict[str, object],
) -> float:
    if not query.camera_id or not candidate.camera_id:
        evidence["camera"] = "missing"
        return 0.0
    if query.camera_id == candidate.camera_id:
        evidence["camera"] = "same"
        return 0.06
    evidence["camera"] = "different"
    return -0.02 if query.site_id == candidate.site_id and query.site_id else -0.04


def _geohash_adjustment(
    query: Observation,
    candidate: Observation,
    evidence: dict[str, object],
) -> float:
    if not query.geohash_public or not candidate.geohash_public:
        return 0.0
    if query.geohash_public == candidate.geohash_public:
        evidence["geohash_public"] = "same"
        return 0.08
    prefix = _common_prefix_len(query.geohash_public, candidate.geohash_public)
    evidence["geohash_public"] = f"prefix:{prefix}"
    if prefix >= 5:
        return 0.04
    if prefix >= 3:
        return 0.01
    return -0.04


def _uploader_adjustment(
    query: Observation,
    candidate: Observation,
    evidence: dict[str, object],
) -> float:
    query_uploader = _text_metadata(query, "uploader")
    candidate_uploader = _text_metadata(candidate, "uploader")
    if not query_uploader or not candidate_uploader:
        return 0.0
    if query_uploader == candidate_uploader:
        evidence["uploader"] = "same"
        return 0.02
    evidence["uploader"] = "different"
    return -0.01


def _encounter_adjustment(
    query: Observation,
    candidate: Observation,
    evidence: dict[str, object],
) -> float:
    if not query.encounter_id or not candidate.encounter_id:
        return 0.0
    if query.encounter_id == candidate.encounter_id:
        evidence["encounter"] = "same"
        return 0.08
    evidence["encounter"] = "different"
    return 0.0


def _source_adjustment(
    query: Observation,
    candidate: Observation,
    evidence: dict[str, object],
) -> float:
    if query.source_type == candidate.source_type:
        evidence["source_type"] = "same"
        return 0.01
    evidence["source_type"] = "different"
    return 0.0


def _time_of_day_adjustment(
    query: Observation,
    candidate: Observation,
    evidence: dict[str, object],
) -> float:
    if query.captured_at is None or candidate.captured_at is None:
        return 0.0
    query_hour = query.captured_at.hour + query.captured_at.minute / 60.0
    candidate_hour = candidate.captured_at.hour + candidate.captured_at.minute / 60.0
    diff = min(fabs(query_hour - candidate_hour), 24.0 - fabs(query_hour - candidate_hour))
    evidence["time_of_day_gap_hours"] = round(diff, 3)
    if diff <= 2.0:
        evidence["time_of_day"] = "similar"
        return 0.04
    if diff <= 6.0:
        evidence["time_of_day"] = "nearby"
        return 0.02
    evidence["time_of_day"] = "different"
    return -0.02


def _quality_adjustment(
    query: Observation,
    candidate: Observation,
    evidence: dict[str, object],
) -> float:
    combined = min(query.quality_score, candidate.quality_score)
    evidence["min_quality_score"] = round(float(combined), 4)
    if combined >= 0.80:
        return 0.02
    if combined < 0.35:
        return -0.04
    if combined < 0.55:
        return -0.02
    return 0.0


def _text_metadata(observation: Observation, key: str) -> str | None:
    value: Any = observation.metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _common_prefix_len(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count
