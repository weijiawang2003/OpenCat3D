import json

from catnet.active_learning import build_ambiguity_queue, write_ambiguity_queue
from catnet.schema import DecisionType, MatchEdge
from catnet.storage import CatNetStore


def test_ambiguity_queue_prioritizes_review_and_score_disagreement() -> None:
    review_edge = _edge(
        "edge_review",
        decision=DecisionType.NEEDS_REVIEW,
        global_score=0.86,
        local_score=0.54,
        temporal_score=0.68,
        metadata_score=0.55,
        fused_score=0.78,
        confidence=0.62,
        explanation={"margin": 0.01, "min_margin": 0.035},
    )
    accepted_edge = _edge(
        "edge_accept",
        decision=DecisionType.MATCH_EXISTING,
        global_score=0.95,
        local_score=0.93,
        temporal_score=0.95,
        metadata_score=0.94,
        fused_score=0.94,
        confidence=0.94,
        explanation={"margin": 0.12, "min_margin": 0.035},
    )

    tasks = build_ambiguity_queue([accepted_edge, review_edge])

    assert [task.edge_id for task in tasks] == ["edge_review"]
    assert "needs_review_decision" in tasks[0].reason_codes
    assert "low_candidate_margin" in tasks[0].reason_codes
    assert "global_local_disagreement" in tasks[0].reason_codes
    assert "visual_temporal_conflict" in tasks[0].reason_codes
    assert "visual_metadata_conflict" in tasks[0].reason_codes


def test_write_ambiguity_queue_from_store(tmp_path) -> None:
    store = CatNetStore(tmp_path / "catnet")
    store.add_match_edge(
        _edge(
            "edge_new",
            decision=DecisionType.CREATE_NEW_IDENTITY,
            global_score=0.80,
            local_score=0.77,
            temporal_score=0.90,
            metadata_score=0.88,
            fused_score=0.76,
            confidence=0.66,
            explanation={"margin": 0.02, "min_margin": 0.035},
        )
    )
    store.save()

    path = write_ambiguity_queue(
        store=CatNetStore(tmp_path / "catnet"),
        output_path=tmp_path / "queue" / "ambiguity.json",
        max_tasks=10,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["task_count"] == 1
    assert payload["tasks"][0]["edge_id"] == "edge_new"
    assert "high_score_new_identity" in payload["tasks"][0]["reason_codes"]


def _edge(
    edge_id: str,
    *,
    decision: DecisionType,
    global_score: float,
    local_score: float | None,
    temporal_score: float,
    metadata_score: float | None,
    fused_score: float,
    confidence: float,
    explanation: dict[str, object],
) -> MatchEdge:
    return MatchEdge(
        edge_id=edge_id,
        src_obs_id=f"src_{edge_id}",
        dst_obs_id=f"dst_{edge_id}",
        src_cat_id="cat_a",
        dst_cat_id="cat_b",
        global_score=global_score,
        local_score=local_score,
        temporal_score=temporal_score,
        metadata_score=metadata_score,
        fused_score=fused_score,
        decision=decision,
        confidence=confidence,
        model_version="model_v1",
        index_version="idx_v1",
        calibration_version="cal_v1",
        decision_rule_version="rule_v1",
        explanation=explanation,
    )
