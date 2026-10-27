"""Ambiguity-aware active-learning queue for CatNet identity review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from catnet.feedback import closed_review_task_ids
from catnet.schema import DecisionType, MatchEdge, StrictModel, stable_id, utc_now
from catnet.storage import CatNetStore


class AmbiguityQueueConfig(StrictModel):
    schema_version: str = "1.0"
    review_decision_bonus: float = Field(default=0.45, ge=0.0)
    low_margin_bonus: float = Field(default=0.25, ge=0.0)
    disagreement_bonus: float = Field(default=0.20, ge=0.0)
    temporal_conflict_bonus: float = Field(default=0.10, ge=0.0)
    min_priority: float = Field(default=0.15, ge=0.0, le=1.0)
    version: str = "catnet-ambiguity-queue-v1"


class AmbiguityTask(StrictModel):
    schema_version: str = "1.0"
    task_id: str
    edge_id: str
    src_obs_id: str
    dst_obs_id: str | None = None
    src_cat_id: str | None = None
    dst_cat_id: str | None = None
    decision: DecisionType
    priority_score: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str]
    global_score: float
    local_score: float | None = None
    temporal_score: float
    metadata_score: float | None = None
    fused_score: float
    confidence: float
    model_version: str
    decision_rule_version: str
    created_at: str
    evidence: dict[str, Any] = Field(default_factory=dict)


def build_ambiguity_queue(
    edges: list[MatchEdge],
    *,
    config: AmbiguityQueueConfig | None = None,
    max_tasks: int | None = None,
) -> list[AmbiguityTask]:
    cfg = config or AmbiguityQueueConfig()
    tasks = [_task_from_edge(edge, cfg) for edge in edges]
    filtered = [task for task in tasks if task.priority_score >= cfg.min_priority]
    ordered = sorted(
        filtered,
        key=lambda task: (task.priority_score, task.created_at, task.edge_id),
        reverse=True,
    )
    return ordered[:max_tasks] if max_tasks is not None else ordered


def write_ambiguity_queue(
    *,
    store: CatNetStore,
    output_path: Path,
    config: AmbiguityQueueConfig | None = None,
    max_tasks: int = 100,
) -> Path:
    tasks = build_ambiguity_queue(
        list(store.catalog.match_edges.values()),
        config=config,
        max_tasks=max_tasks,
    )
    closed_task_ids = closed_review_task_ids(store)
    tasks = [task for task in tasks if task.task_id not in closed_task_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "queue_version": (config or AmbiguityQueueConfig()).version,
        "task_count": len(tasks),
        "tasks": [task.model_dump(mode="json") for task in tasks],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _task_from_edge(edge: MatchEdge, config: AmbiguityQueueConfig) -> AmbiguityTask:
    priority = 0.0
    reasons: list[str] = []

    if edge.decision == DecisionType.NEEDS_REVIEW:
        priority += config.review_decision_bonus
        reasons.append("needs_review_decision")

    margin = _optional_float(edge.explanation.get("margin"))
    min_margin = _optional_float(edge.explanation.get("min_margin"))
    if min_margin is None:
        min_margin = 0.035
    if margin is not None and margin < min_margin:
        priority += config.low_margin_bonus * (1.0 - max(0.0, margin / max(min_margin, 1e-6)))
        reasons.append("low_candidate_margin")

    local_score = edge.local_score
    if local_score is not None:
        disagreement = abs(edge.global_score - local_score)
        if disagreement >= 0.10:
            priority += config.disagreement_bonus * min(1.0, disagreement)
            reasons.append("global_local_disagreement")

    if edge.global_score >= 0.82 and edge.temporal_score <= 0.72:
        priority += config.temporal_conflict_bonus
        reasons.append("visual_temporal_conflict")

    if (
        edge.metadata_score is not None
        and edge.global_score >= 0.82
        and edge.metadata_score <= 0.62
    ):
        priority += config.temporal_conflict_bonus
        reasons.append("visual_metadata_conflict")

    if edge.decision == DecisionType.CREATE_NEW_IDENTITY and edge.global_score >= 0.76:
        priority += 0.15
        reasons.append("high_score_new_identity")

    if edge.confidence < 0.70:
        priority += 0.10 * (1.0 - edge.confidence)
        reasons.append("low_decision_confidence")

    priority = max(0.0, min(1.0, priority))
    if not reasons:
        reasons.append("low_priority_trace")
    return AmbiguityTask(
        task_id=stable_id("alq", edge.edge_id, config.version),
        edge_id=edge.edge_id,
        src_obs_id=edge.src_obs_id,
        dst_obs_id=edge.dst_obs_id,
        src_cat_id=edge.src_cat_id,
        dst_cat_id=edge.dst_cat_id,
        decision=edge.decision,
        priority_score=priority,
        reason_codes=reasons,
        global_score=edge.global_score,
        local_score=edge.local_score,
        temporal_score=edge.temporal_score,
        metadata_score=edge.metadata_score,
        fused_score=edge.fused_score,
        confidence=edge.confidence,
        model_version=edge.model_version,
        decision_rule_version=edge.decision_rule_version,
        created_at=utc_now().isoformat(),
        evidence={
            "calibration_version": edge.calibration_version,
            "index_version": edge.index_version,
            "explanation": edge.explanation,
        },
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
