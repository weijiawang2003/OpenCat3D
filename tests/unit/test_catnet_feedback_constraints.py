import json
from datetime import UTC, datetime

from catnet.active_learning import build_ambiguity_queue, write_ambiguity_queue
from catnet.feedback import (
    add_feedback_constraint,
    apply_feedback_constraints,
    import_feedback_constraints,
)
from catnet.schema import DecisionType, FeedbackConstraintType, MatchEdge, Observation
from catnet.storage import CatNetStore


def test_must_link_creates_merge_proposal_and_cannot_link_blocks(tmp_path) -> None:
    store = _store_with_two_identities(tmp_path)
    constraints = [
        add_feedback_constraint(
            store,
            constraint_type=FeedbackConstraintType.CANNOT_LINK,
            left_obs_id="obs_a",
            right_obs_id="obs_b",
            left_cat_id=store.catalog.observations["obs_a"].cat_id,
            right_cat_id=store.catalog.observations["obs_b"].cat_id,
            reviewer="qa",
        ),
        add_feedback_constraint(
            store,
            constraint_type=FeedbackConstraintType.MUST_LINK,
            left_obs_id="obs_a",
            right_obs_id="obs_b",
            reviewer="qa",
        ),
    ]
    store.save()

    report = apply_feedback_constraints(CatNetStore(tmp_path), dry_run=True)

    assert [constraint.constraint_type for constraint in constraints]
    assert any(action.action_type == "blocked_merge" for action in report.actions)
    assert any("cannot_link_blocks_must_link" in action.reason_codes for action in report.actions)


def test_obs_only_cannot_link_blocks_obs_only_must_link(tmp_path) -> None:
    store = _store_with_two_identities(tmp_path)
    add_feedback_constraint(
        store,
        constraint_type=FeedbackConstraintType.CANNOT_LINK,
        left_obs_id="obs_a",
        right_obs_id="obs_b",
        reviewer="qa",
    )
    add_feedback_constraint(
        store,
        constraint_type=FeedbackConstraintType.MUST_LINK,
        left_obs_id="obs_a",
        right_obs_id="obs_b",
        reviewer="qa",
    )

    report = apply_feedback_constraints(store, dry_run=True)

    assert any(
        action.action_type == "blocked_merge"
        and "cannot_link_blocks_must_link" in action.reason_codes
        for action in report.actions
    )


def test_feedback_import_and_same_identity_assignment_proposal(tmp_path) -> None:
    store = _store_with_one_identity_and_unassigned_observation(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "label": "same",
                "left_obs_id": "obs_a",
                "right_obs_id": "obs_unassigned",
                "review_task_id": "task_1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    imported = import_feedback_constraints(store, labels_path)
    report = apply_feedback_constraints(store, dry_run=True)

    assert imported[0].constraint_type == FeedbackConstraintType.MUST_LINK
    assert any(action.action_type == "assignment_proposal" for action in report.actions)
    assert any(action.action_type == "closed_review_task" for action in report.actions)


def test_feedback_closes_active_learning_task(tmp_path) -> None:
    store = _store_with_two_identities(tmp_path)
    edge = _edge("edge_review", "obs_a", "obs_b", decision=DecisionType.NEEDS_REVIEW)
    store.add_match_edge(edge)
    task = build_ambiguity_queue([edge])[0]
    add_feedback_constraint(
        store,
        constraint_type=FeedbackConstraintType.CANNOT_LINK,
        left_obs_id="obs_a",
        right_obs_id="obs_b",
        review_task_id=task.task_id,
        reviewer="qa",
    )
    store.save()

    output_path = write_ambiguity_queue(
        store=CatNetStore(tmp_path),
        output_path=tmp_path / "queue.json",
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["task_count"] == 0


def test_missing_feedback_reference_does_not_close_review_task(tmp_path) -> None:
    store = _store_with_two_identities(tmp_path)
    edge = _edge("edge_review", "obs_a", "obs_b", decision=DecisionType.NEEDS_REVIEW)
    store.add_match_edge(edge)
    task = build_ambiguity_queue([edge])[0]
    add_feedback_constraint(
        store,
        constraint_type=FeedbackConstraintType.MUST_LINK,
        left_obs_id="obs_missing",
        right_obs_id="obs_b",
        review_task_id=task.task_id,
        reviewer="qa",
    )

    report = apply_feedback_constraints(store, dry_run=True)
    output_path = write_ambiguity_queue(store=store, output_path=tmp_path / "queue.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert report.actions[0].action_type == "invalid_constraint"
    assert "missing_reference" in report.actions[0].reason_codes
    assert payload["task_count"] == 1


def test_feedback_apply_is_idempotent(tmp_path) -> None:
    store = _store_with_two_identities(tmp_path)
    add_feedback_constraint(
        store,
        constraint_type=FeedbackConstraintType.MUST_LINK,
        left_obs_id="obs_a",
        right_obs_id="obs_b",
        reviewer="qa",
    )

    first = apply_feedback_constraints(store, dry_run=False)
    first_version_count = len(store.catalog.identity_versions)
    applied_at = next(iter(store.catalog.feedback_constraints.values())).applied_at
    second = apply_feedback_constraints(store, dry_run=False)

    assert first.action_count > 0
    assert second.action_count == 0
    assert len(store.catalog.identity_versions) == first_version_count
    assert next(iter(store.catalog.feedback_constraints.values())).applied_at == applied_at


def _store_with_two_identities(root) -> CatNetStore:
    store = CatNetStore(root)
    first = store.create_identity(seed="cat-a", first_seen_at=_captured_at(0))
    second = store.create_identity(seed="cat-b", first_seen_at=_captured_at(1))
    store.upsert_observation(_obs("obs_a", first.cat_id))
    store.upsert_observation(_obs("obs_b", second.cat_id))
    return store


def _store_with_one_identity_and_unassigned_observation(root) -> CatNetStore:
    store = CatNetStore(root)
    first = store.create_identity(seed="cat-a", first_seen_at=_captured_at(0))
    store.upsert_observation(_obs("obs_a", first.cat_id))
    store.upsert_observation(_obs("obs_unassigned", None))
    return store


def _obs(obs_id: str, cat_id: str | None) -> Observation:
    return Observation(
        obs_id=obs_id,
        observation_version=f"{obs_id}_v1",
        media_id=f"media_{obs_id}",
        cat_id=cat_id,
        captured_at=_captured_at(0),
    )


def _edge(
    edge_id: str,
    src_obs_id: str,
    dst_obs_id: str,
    *,
    decision: DecisionType,
) -> MatchEdge:
    return MatchEdge(
        edge_id=edge_id,
        src_obs_id=src_obs_id,
        dst_obs_id=dst_obs_id,
        src_cat_id="cat_a",
        dst_cat_id="cat_b",
        global_score=0.80,
        local_score=0.50,
        temporal_score=0.50,
        metadata_score=0.50,
        fused_score=0.70,
        decision=decision,
        confidence=0.60,
        model_version="model_v1",
        index_version="idx_v1",
        calibration_version="cal_v1",
        decision_rule_version="rule_v1",
    )


def _captured_at(day: int) -> datetime:
    return datetime(2026, 1, 1 + day, 10, tzinfo=UTC)
