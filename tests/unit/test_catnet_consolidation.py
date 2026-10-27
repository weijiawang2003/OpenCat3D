from datetime import UTC, datetime

from catnet.consolidation import run_consolidation, write_consolidation_report
from catnet.feedback import add_feedback_constraint
from catnet.schema import DecisionType, FeedbackConstraintType, MatchEdge, Observation
from catnet.storage import CatNetStore


def test_high_confidence_provisional_identities_propose_merge(tmp_path) -> None:
    store, cat_a, cat_b = _store_with_two_cats(tmp_path)
    store.add_match_edge(_edge("edge_merge", "obs_a", "obs_b", cat_a, cat_b))

    report = run_consolidation(store)

    assert len(report.merge_proposals) == 1
    assert report.merge_proposals[0].source_edge_ids == ["edge_merge"]
    assert report.merge_proposals[0].source_cat_id == cat_a


def test_cannot_link_blocks_consolidation_merge(tmp_path) -> None:
    store, cat_a, cat_b = _store_with_two_cats(tmp_path)
    store.add_match_edge(_edge("edge_merge", "obs_a", "obs_b", cat_a, cat_b))
    add_feedback_constraint(
        store,
        constraint_type=FeedbackConstraintType.CANNOT_LINK,
        left_obs_id="obs_a",
        right_obs_id="obs_b",
        left_cat_id=cat_a,
        right_cat_id=cat_b,
    )

    report = run_consolidation(store)

    assert not report.merge_proposals
    assert len(report.blocked_proposals) == 1
    assert report.blocked_proposals[0].reason_codes == ["cannot_link_blocks_merge"]


def test_conflicting_evidence_produces_split_proposal(tmp_path) -> None:
    store, cat_a, _cat_b = _store_with_two_cats(tmp_path)
    store.catalog.observations["obs_b"] = store.catalog.observations["obs_b"].model_copy(
        update={"cat_id": cat_a}
    )
    store.add_match_edge(
        _edge(
            "edge_conflict",
            "obs_a",
            "obs_b",
            cat_a,
            cat_a,
            local_score=0.20,
            metadata_score=0.20,
        )
    )
    add_feedback_constraint(
        store,
        constraint_type=FeedbackConstraintType.CANNOT_LINK,
        left_obs_id="obs_a",
        right_obs_id="obs_b",
    )

    report = run_consolidation(store)
    output = write_consolidation_report(report, tmp_path / "consolidation.json")

    assert len(report.split_proposals) >= 1
    assert output.exists()
    assert any(
        "global_local_conflict" in proposal.reason_codes
        for proposal in report.split_proposals
    )


def test_consolidation_apply_is_idempotent(tmp_path) -> None:
    store, cat_a, cat_b = _store_with_two_cats(tmp_path)
    store.add_match_edge(_edge("edge_merge", "obs_a", "obs_b", cat_a, cat_b))

    first = run_consolidation(store, apply=True)
    first_version_count = len(store.catalog.identity_versions)
    second = run_consolidation(store, apply=True)

    assert first.merge_proposals
    assert second.merge_proposals
    assert len(store.catalog.identity_versions) == first_version_count


def _store_with_two_cats(root) -> tuple[CatNetStore, str, str]:
    store = CatNetStore(root)
    first = store.create_identity(seed="cat-a", first_seen_at=_captured_at(0))
    second = store.create_identity(seed="cat-b", first_seen_at=_captured_at(1))
    store.upsert_observation(_obs("obs_a", first.cat_id))
    store.upsert_observation(_obs("obs_b", second.cat_id))
    return store, first.cat_id, second.cat_id


def _obs(obs_id: str, cat_id: str) -> Observation:
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
    src_cat_id: str,
    dst_cat_id: str,
    *,
    local_score: float = 0.92,
    metadata_score: float = 0.90,
) -> MatchEdge:
    return MatchEdge(
        edge_id=edge_id,
        src_obs_id=src_obs_id,
        dst_obs_id=dst_obs_id,
        src_cat_id=src_cat_id,
        dst_cat_id=dst_cat_id,
        global_score=0.94,
        local_score=local_score,
        temporal_score=0.90,
        metadata_score=metadata_score,
        fused_score=0.93,
        decision=DecisionType.MATCH_EXISTING,
        confidence=0.91,
        model_version="model_v1",
        index_version="idx_v1",
        calibration_version="cal_v1",
        decision_rule_version="rule_v1",
    )


def _captured_at(day: int) -> datetime:
    return datetime(2026, 1, 1 + day, 10, tzinfo=UTC)
