"""Durable reviewer feedback constraints for CatNet identity decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from catnet.schema import (
    FeedbackConstraint,
    FeedbackConstraintType,
    IdentityEventType,
    Observation,
    StrictModel,
    stable_id,
    utc_now,
)
from catnet.storage import CatNetStore

FeedbackActionType = Literal[
    "merge_proposal",
    "assignment_proposal",
    "split_proposal",
    "blocked_merge",
    "closed_review_task",
    "invalid_constraint",
    "no_op",
]


class FeedbackApplyAction(StrictModel):
    schema_version: str = "1.0"
    action_type: FeedbackActionType
    constraint_id: str
    obs_ids: list[str] = Field(default_factory=list)
    cat_ids: list[str] = Field(default_factory=list)
    review_task_id: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackApplyReport(StrictModel):
    schema_version: str = "1.0"
    dry_run: bool
    constraint_count: int
    action_count: int
    actions: list[FeedbackApplyAction]


def add_feedback_constraint(
    store: CatNetStore,
    *,
    constraint_type: FeedbackConstraintType | str,
    left_obs_id: str | None = None,
    right_obs_id: str | None = None,
    left_cat_id: str | None = None,
    right_cat_id: str | None = None,
    review_task_id: str | None = None,
    reviewer: str | None = None,
    actor: str = "system",
    source: str = "manual",
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> FeedbackConstraint:
    resolved_type = FeedbackConstraintType(constraint_type)
    constraint = FeedbackConstraint(
        constraint_id=stable_id(
            "fbc",
            resolved_type.value,
            left_obs_id or "",
            right_obs_id or "",
            left_cat_id or "",
            right_cat_id or "",
            review_task_id or "",
            utc_now().isoformat(),
        ),
        constraint_type=resolved_type,
        left_obs_id=left_obs_id,
        right_obs_id=right_obs_id,
        left_cat_id=left_cat_id,
        right_cat_id=right_cat_id,
        review_task_id=review_task_id,
        reviewer=reviewer,
        actor=actor,
        source=source,
        reason=reason,
        metadata=metadata or {},
    )
    store.add_feedback_constraint(constraint)
    return constraint


def import_feedback_constraints(store: CatNetStore, labels_path: Path) -> list[FeedbackConstraint]:
    records = _read_feedback_records(labels_path)
    constraints: list[FeedbackConstraint] = []
    for record in records:
        payload = dict(record)
        if "constraint_type" not in payload and "label" in payload:
            payload["constraint_type"] = _label_to_constraint_type(str(payload["label"]))
        payload.pop("label", None)
        if "constraint_id" not in payload:
            payload["constraint_id"] = stable_id(
                "fbc",
                payload.get("constraint_type", ""),
                payload.get("left_obs_id", ""),
                payload.get("right_obs_id", ""),
                payload.get("left_cat_id", ""),
                payload.get("right_cat_id", ""),
                payload.get("review_task_id", ""),
                utc_now().isoformat(),
            )
        constraint = FeedbackConstraint.model_validate(payload)
        store.add_feedback_constraint(constraint)
        constraints.append(constraint)
    return constraints


def apply_feedback_constraints(store: CatNetStore, *, dry_run: bool = True) -> FeedbackApplyReport:
    constraints = sorted(
        [
            constraint
            for constraint in store.catalog.feedback_constraints.values()
            if constraint.status == "open"
        ],
        key=lambda item: (item.created_at, item.constraint_id),
    )
    blockers = _blocking_constraints(store, constraints)
    actions: list[FeedbackApplyAction] = []
    for constraint in constraints:
        actions.extend(_actions_for_constraint(store, constraint, blockers))

    if not dry_run:
        _record_feedback_apply_events(store, actions)
        for constraint in constraints:
            store.catalog.feedback_constraints[constraint.constraint_id] = constraint.model_copy(
                update={"status": "applied", "applied_at": utc_now()}
            )
    return FeedbackApplyReport(
        dry_run=dry_run,
        constraint_count=len(constraints),
        action_count=len(actions),
        actions=actions,
    )


def closed_review_task_ids(store: CatNetStore) -> set[str]:
    return {
        constraint.review_task_id
        for constraint in store.catalog.feedback_constraints.values()
        if constraint.review_task_id
        and not _missing_references(store, constraint)
        and constraint.constraint_type
        in {
            FeedbackConstraintType.MUST_LINK,
            FeedbackConstraintType.CANNOT_LINK,
            FeedbackConstraintType.SAME_IDENTITY,
            FeedbackConstraintType.DIFFERENT_IDENTITY,
            FeedbackConstraintType.UNSURE,
        }
    }


def write_feedback_apply_report(report: FeedbackApplyReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _actions_for_constraint(
    store: CatNetStore,
    constraint: FeedbackConstraint,
    blockers: set[tuple[str, str]],
) -> list[FeedbackApplyAction]:
    actions: list[FeedbackApplyAction] = []
    missing = _missing_references(store, constraint)
    if missing:
        return [
            FeedbackApplyAction(
                action_type="invalid_constraint",
                constraint_id=constraint.constraint_id,
                review_task_id=constraint.review_task_id,
                reason_codes=["missing_reference"],
                metadata={"missing_references": missing},
            )
        ]

    if constraint.review_task_id:
        actions.append(
            FeedbackApplyAction(
                action_type="closed_review_task",
                constraint_id=constraint.constraint_id,
                review_task_id=constraint.review_task_id,
                reason_codes=["reviewer_feedback_received"],
            )
        )

    if constraint.constraint_type == FeedbackConstraintType.UNSURE:
        actions.append(
            FeedbackApplyAction(
                action_type="no_op",
                constraint_id=constraint.constraint_id,
                review_task_id=constraint.review_task_id,
                reason_codes=["reviewer_unsure"],
            )
        )
        return actions

    left_obs, right_obs = _constraint_observations(store, constraint)
    left_cat_id = constraint.left_cat_id or (left_obs.cat_id if left_obs else None)
    right_cat_id = constraint.right_cat_id or (right_obs.cat_id if right_obs else None)
    obs_ids = [
        value
        for value in [constraint.left_obs_id, constraint.right_obs_id]
        if value is not None
    ]
    cat_ids = [value for value in [left_cat_id, right_cat_id] if value is not None]

    if constraint.constraint_type in {
        FeedbackConstraintType.MUST_LINK,
        FeedbackConstraintType.SAME_IDENTITY,
    }:
        pair = _pair_key(left_cat_id, right_cat_id)
        if pair is not None and pair in blockers:
            actions.append(
                FeedbackApplyAction(
                    action_type="blocked_merge",
                    constraint_id=constraint.constraint_id,
                    obs_ids=obs_ids,
                    cat_ids=cat_ids,
                    reason_codes=["cannot_link_blocks_must_link"],
                )
            )
            return actions
        if left_cat_id and right_cat_id and left_cat_id != right_cat_id:
            actions.append(
                FeedbackApplyAction(
                    action_type="merge_proposal",
                    constraint_id=constraint.constraint_id,
                    obs_ids=obs_ids,
                    cat_ids=cat_ids,
                    reason_codes=["reviewer_same_identity"],
                    metadata={"source_cat_id": right_cat_id, "target_cat_id": left_cat_id},
                )
            )
            return actions
        if (left_cat_id and right_obs and not right_obs.cat_id) or (
            right_cat_id and left_obs and not left_obs.cat_id
        ):
            target_cat_id = left_cat_id or right_cat_id
            unassigned_obs_id = (
                right_obs.obs_id if right_obs and not right_obs.cat_id else left_obs.obs_id
            )
            actions.append(
                FeedbackApplyAction(
                    action_type="assignment_proposal",
                    constraint_id=constraint.constraint_id,
                    obs_ids=[unassigned_obs_id],
                    cat_ids=[target_cat_id] if target_cat_id else [],
                    reason_codes=["reviewer_same_identity_unassigned_observation"],
                )
            )
            return actions
        actions.append(
            FeedbackApplyAction(
                action_type="no_op",
                constraint_id=constraint.constraint_id,
                obs_ids=obs_ids,
                cat_ids=cat_ids,
                reason_codes=["same_identity_already_satisfied"],
            )
        )
        return actions

    if constraint.constraint_type in {
        FeedbackConstraintType.CANNOT_LINK,
        FeedbackConstraintType.DIFFERENT_IDENTITY,
    }:
        if left_cat_id and right_cat_id and left_cat_id == right_cat_id:
            actions.append(
                FeedbackApplyAction(
                    action_type="split_proposal",
                    constraint_id=constraint.constraint_id,
                    obs_ids=obs_ids,
                    cat_ids=[left_cat_id],
                    reason_codes=["reviewer_different_identity_within_cat"],
                )
            )
            return actions
        actions.append(
            FeedbackApplyAction(
                action_type="blocked_merge",
                constraint_id=constraint.constraint_id,
                obs_ids=obs_ids,
                cat_ids=cat_ids,
                reason_codes=["reviewer_cannot_link"],
            )
        )
    return actions


def _constraint_observations(
    store: CatNetStore,
    constraint: FeedbackConstraint,
) -> tuple[Observation | None, Observation | None]:
    left = (
        store.catalog.observations.get(constraint.left_obs_id)
        if constraint.left_obs_id is not None
        else None
    )
    right = (
        store.catalog.observations.get(constraint.right_obs_id)
        if constraint.right_obs_id is not None
        else None
    )
    return left, right


def _missing_references(
    store: CatNetStore,
    constraint: FeedbackConstraint,
) -> list[str]:
    missing: list[str] = []
    if (
        constraint.left_obs_id is not None
        and constraint.left_obs_id not in store.catalog.observations
    ):
        missing.append(f"left_obs_id:{constraint.left_obs_id}")
    if (
        constraint.right_obs_id is not None
        and constraint.right_obs_id not in store.catalog.observations
    ):
        missing.append(f"right_obs_id:{constraint.right_obs_id}")
    if (
        constraint.left_cat_id is not None
        and constraint.left_cat_id not in store.catalog.identities
    ):
        missing.append(f"left_cat_id:{constraint.left_cat_id}")
    if (
        constraint.right_cat_id is not None
        and constraint.right_cat_id not in store.catalog.identities
    ):
        missing.append(f"right_cat_id:{constraint.right_cat_id}")
    return missing


def _blocking_constraints(
    store: CatNetStore,
    constraints: list[FeedbackConstraint],
) -> set[tuple[str, str]]:
    blockers: set[tuple[str, str]] = set()
    for constraint in constraints:
        if constraint.constraint_type not in {
            FeedbackConstraintType.CANNOT_LINK,
            FeedbackConstraintType.DIFFERENT_IDENTITY,
        }:
            continue
        left_obs, right_obs = _constraint_observations(store, constraint)
        left_cat_id = constraint.left_cat_id or (left_obs.cat_id if left_obs else None)
        right_cat_id = constraint.right_cat_id or (right_obs.cat_id if right_obs else None)
        pair = _pair_key(left_cat_id, right_cat_id)
        if pair is not None:
            blockers.add(pair)
    return blockers


def _pair_key(left: str | None, right: str | None) -> tuple[str, str] | None:
    if left is None or right is None:
        return None
    return tuple(sorted((left, right)))


def _record_feedback_apply_events(store: CatNetStore, actions: list[FeedbackApplyAction]) -> None:
    for action in actions:
        if action.action_type not in {"merge_proposal", "assignment_proposal", "split_proposal"}:
            continue
        for cat_id in action.cat_ids[:1]:
            identity = store.catalog.identities.get(cat_id)
            store.record_identity_event(
                cat_id=cat_id,
                event_type=IdentityEventType.UPDATED,
                reason=f"feedback {action.action_type}",
                previous_identity_version=identity.identity_version if identity else None,
                payload=action.model_dump(mode="json"),
            )


def _read_feedback_records(labels_path: Path) -> list[dict[str, Any]]:
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    if labels_path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "constraints" in payload:
        payload = payload["constraints"]
    if not isinstance(payload, list):
        raise ValueError("feedback import must be JSONL, a JSON list, or contain constraints")
    return [dict(record) for record in payload]


def _label_to_constraint_type(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in {"same", "same_cat", "same_identity", "must_link", "match"}:
        return FeedbackConstraintType.MUST_LINK.value
    if normalized in {"different", "different_cat", "different_identity", "cannot_link"}:
        return FeedbackConstraintType.CANNOT_LINK.value
    if normalized in {"unsure", "skip"}:
        return FeedbackConstraintType.UNSURE.value
    raise ValueError(f"unknown feedback label: {label}")
