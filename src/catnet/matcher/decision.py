"""Open-set match decisions with temporal evidence and version metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from catnet.matcher.faiss_index import SearchCandidate
from catnet.metadata import MetadataPosteriorScorer
from catnet.schema import DecisionType, MatchDecision, ModelVersion, Observation
from catnet.temporal import TemporalGrowthScorer


@dataclass(frozen=True)
class OpenSetDecisionConfig:
    match_threshold: float = 0.86
    review_threshold: float = 0.76
    min_margin: float = 0.035
    visual_weight: float = 0.78
    temporal_weight: float = 0.14
    metadata_weight: float = 0.08
    decision_rule_version: str = "catnet-open-set-rule-v1"


class OpenSetDecider:
    def __init__(
        self,
        *,
        model_version: ModelVersion,
        index_version: str,
        config: OpenSetDecisionConfig | None = None,
        temporal_scorer: TemporalGrowthScorer | None = None,
        metadata_scorer: MetadataPosteriorScorer | None = None,
    ) -> None:
        self.model_version = model_version
        self.index_version = index_version
        self.config = config or OpenSetDecisionConfig()
        self.temporal_scorer = temporal_scorer or TemporalGrowthScorer()
        self.metadata_scorer = metadata_scorer or MetadataPosteriorScorer()

    def decide(
        self,
        candidates: list[SearchCandidate],
        *,
        query_observation: Observation,
        observations_by_id: Mapping[str, Observation],
    ) -> MatchDecision:
        if not candidates:
            return self._decision(
                decision=DecisionType.CREATE_NEW_IDENTITY,
                cat_id=None,
                confidence=1.0,
                fused_score=0.0,
                visual_score=0.0,
                local_score=None,
                temporal_score=0.5,
                metadata_score=None,
                top_candidate=None,
                explanation={"reason": "no_candidates", "open_set": "create_new_identity"},
            )

        top = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else -1.0
        margin = top.score - second_score
        top_global_score = float(top.metadata.get("global_score", top.score))
        top_local_score = _optional_float(top.metadata.get("local_score"))
        second_global_score = (
            float(candidates[1].metadata.get("global_score", second_score))
            if len(candidates) > 1
            else -1.0
        )
        candidate_observation = observations_by_id.get(top.obs_id)
        temporal = self.temporal_scorer.score(query_observation, candidate_observation)
        metadata = self.metadata_scorer.score(query_observation, candidate_observation)
        fused = _weighted_fused_score(
            visual_score=top.score,
            temporal_score=temporal.score,
            metadata_score=metadata.score,
            config=self.config,
        )
        explanation = {
            "top_visual_score": round(top.score, 6),
            "second_visual_score": round(second_score, 6),
            "top_global_score": round(top_global_score, 6),
            "second_global_score": round(second_global_score, 6),
            "top_local_score": round(top_local_score, 6)
            if top_local_score is not None
            else None,
            "metadata_score": round(metadata.score, 6),
            "margin": round(margin, 6),
            "min_margin": self.config.min_margin,
            "match_threshold": self.config.match_threshold,
            "review_threshold": self.config.review_threshold,
            "temporal": temporal.evidence,
            "metadata": metadata.evidence,
        }
        if "local_rerank" in top.metadata:
            explanation["local_rerank"] = top.metadata["local_rerank"]

        if top.cat_id and fused >= self.config.match_threshold and margin >= self.config.min_margin:
            confidence = _scaled_confidence(
                fused,
                low=self.config.match_threshold,
                high=1.0,
                floor=0.62,
            )
            explanation["open_set"] = "accepted_existing_identity"
            return self._decision(
                decision=DecisionType.MATCH_EXISTING,
                cat_id=top.cat_id,
                confidence=confidence,
                fused_score=fused,
                visual_score=top.score,
                local_score=top_local_score,
                temporal_score=temporal.score,
                metadata_score=metadata.score,
                top_candidate=top,
                explanation=explanation,
            )

        if fused < self.config.review_threshold:
            confidence = _scaled_confidence(
                self.config.review_threshold - fused,
                low=0.0,
                high=self.config.review_threshold + 1.0,
                floor=0.55,
            )
            explanation["open_set"] = "below_rejection_threshold"
            return self._decision(
                decision=DecisionType.CREATE_NEW_IDENTITY,
                cat_id=None,
                confidence=confidence,
                fused_score=fused,
                visual_score=top.score,
                local_score=top_local_score,
                temporal_score=temporal.score,
                metadata_score=metadata.score,
                top_candidate=top,
                explanation=explanation,
            )

        explanation["open_set"] = "ambiguous_needs_review"
        return self._decision(
            decision=DecisionType.NEEDS_REVIEW,
            cat_id=None,
            confidence=max(0.0, min(1.0, 1.0 - abs(fused - self.config.match_threshold))),
            fused_score=fused,
            visual_score=top.score,
            local_score=top_local_score,
            temporal_score=temporal.score,
            metadata_score=metadata.score,
            top_candidate=top,
            explanation=explanation,
        )

    def _decision(
        self,
        *,
        decision: DecisionType,
        cat_id: str | None,
        confidence: float,
        fused_score: float,
        visual_score: float,
        local_score: float | None,
        temporal_score: float,
        metadata_score: float | None,
        top_candidate: SearchCandidate | None,
        explanation: dict[str, object],
    ) -> MatchDecision:
        return MatchDecision(
            decision=decision,
            cat_id=cat_id,
            confidence=max(0.0, min(1.0, confidence)),
            fused_score=max(-1.0, min(1.0, fused_score)),
            visual_score=max(-1.0, min(1.0, visual_score)),
            local_score=None
            if local_score is None
            else max(-1.0, min(1.0, local_score)),
            temporal_score=max(0.0, min(1.0, temporal_score)),
            metadata_score=None
            if metadata_score is None
            else max(0.0, min(1.0, metadata_score)),
            top_candidate_obs_id=top_candidate.obs_id if top_candidate else None,
            top_candidate_cat_id=top_candidate.cat_id if top_candidate else None,
            model_version=self.model_version.model_version,
            index_version=self.index_version,
            calibration_version=self.model_version.calibration_version,
            decision_rule_version=self.config.decision_rule_version,
            explanation=explanation,
        )


def _scaled_confidence(value: float, *, low: float, high: float, floor: float) -> float:
    if high <= low:
        return floor
    scaled = (value - low) / (high - low)
    return max(floor, min(1.0, floor + (1.0 - floor) * scaled))


def _weighted_fused_score(
    *,
    visual_score: float,
    temporal_score: float,
    metadata_score: float,
    config: OpenSetDecisionConfig,
) -> float:
    total = config.visual_weight + config.temporal_weight + config.metadata_weight
    if total <= 1e-12:
        return visual_score
    return (
        config.visual_weight * visual_score
        + config.temporal_weight * temporal_score
        + config.metadata_weight * metadata_score
    ) / total


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
