"""Proposal-first CatNet identity graph consolidation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from catnet.schema import (
    DecisionType,
    FeedbackConstraint,
    FeedbackConstraintType,
    IdentityEventType,
    MatchEdge,
    StrictModel,
    stable_id,
    utc_now,
)
from catnet.storage import CatNetStore

ConsolidationProposalType = Literal["merge", "split", "blocked"]


class ConsolidationProposal(StrictModel):
    schema_version: str = "1.0"
    proposal_id: str
    proposal_type: ConsolidationProposalType
    source_cat_id: str | None = None
    target_cat_id: str | None = None
    cat_id: str | None = None
    obs_ids: list[str] = Field(default_factory=list)
    source_edge_ids: list[str] = Field(default_factory=list)
    blocking_constraint_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsolidationReport(StrictModel):
    schema_version: str = "1.0"
    consolidation_version: str = "catnet-consolidation-v1"
    created_at: str
    apply: bool
    merge_proposals: list[ConsolidationProposal] = Field(default_factory=list)
    split_proposals: list[ConsolidationProposal] = Field(default_factory=list)
    blocked_proposals: list[ConsolidationProposal] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def run_consolidation(
    store: CatNetStore,
    *,
    apply: bool = False,
) -> ConsolidationReport:
    constraints = list(store.catalog.feedback_constraints.values())
    merge_proposals: list[ConsolidationProposal] = []
    split_proposals: list[ConsolidationProposal] = []
    blocked_proposals: list[ConsolidationProposal] = []

    for edge in sorted(store.catalog.match_edges.values(), key=lambda item: item.edge_id):
        if _is_high_confidence_merge_edge(edge):
            blockers = _blocking_constraints_for_edge(edge, constraints, store)
            if blockers:
                blocked_proposals.append(_blocked_proposal(edge, blockers))
            elif edge.src_cat_id != edge.dst_cat_id:
                merge_proposals.append(_merge_proposal(edge))
        if _is_same_cat_conflict(edge):
            split_proposals.append(_split_proposal_from_edge(edge))

    for constraint in constraints:
        proposal = _split_proposal_from_constraint(store, constraint)
        if proposal is not None:
            split_proposals.append(proposal)

    report = ConsolidationReport(
        created_at=utc_now().isoformat(),
        apply=apply,
        merge_proposals=_dedupe_proposals(merge_proposals),
        split_proposals=_dedupe_proposals(split_proposals),
        blocked_proposals=_dedupe_proposals(blocked_proposals),
        metadata={
            "edge_count": len(store.catalog.match_edges),
            "constraint_count": len(constraints),
        },
    )
    if apply:
        _record_consolidation_events(store, report)
    return report


def write_consolidation_report(report: ConsolidationReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _is_high_confidence_merge_edge(edge: MatchEdge) -> bool:
    if edge.src_cat_id is None or edge.dst_cat_id is None:
        return False
    if edge.src_cat_id == edge.dst_cat_id:
        return False
    return (
        edge.decision == DecisionType.MATCH_EXISTING
        and edge.fused_score >= 0.86
        and edge.confidence >= 0.80
    )


def _is_same_cat_conflict(edge: MatchEdge) -> bool:
    if edge.src_cat_id is None or edge.dst_cat_id is None or edge.src_cat_id != edge.dst_cat_id:
        return False
    visual_local_conflict = edge.global_score >= 0.84 and (
        edge.local_score is not None and edge.local_score <= 0.45
    )
    metadata_conflict = edge.global_score >= 0.84 and (
        edge.metadata_score is not None and edge.metadata_score <= 0.30
    )
    temporal_conflict = edge.global_score >= 0.84 and edge.temporal_score <= 0.20
    return visual_local_conflict or metadata_conflict or temporal_conflict


def _merge_proposal(edge: MatchEdge) -> ConsolidationProposal:
    return ConsolidationProposal(
        proposal_id=stable_id("cmerge", edge.src_cat_id, edge.dst_cat_id, edge.edge_id),
        proposal_type="merge",
        source_cat_id=edge.src_cat_id,
        target_cat_id=edge.dst_cat_id,
        obs_ids=[value for value in [edge.src_obs_id, edge.dst_obs_id] if value is not None],
        source_edge_ids=[edge.edge_id],
        confidence=edge.confidence,
        reason_codes=["high_confidence_match_edge", "proposal_only"],
        metadata={
            "fused_score": edge.fused_score,
            "global_score": edge.global_score,
            "local_score": edge.local_score,
            "temporal_score": edge.temporal_score,
            "metadata_score": edge.metadata_score,
        },
    )


def _blocked_proposal(
    edge: MatchEdge,
    blockers: list[FeedbackConstraint],
) -> ConsolidationProposal:
    return ConsolidationProposal(
        proposal_id=stable_id(
            "cblock",
            edge.src_cat_id,
            edge.dst_cat_id,
            edge.edge_id,
            ",".join(sorted(blocker.constraint_id for blocker in blockers)),
        ),
        proposal_type="blocked",
        source_cat_id=edge.src_cat_id,
        target_cat_id=edge.dst_cat_id,
        obs_ids=[value for value in [edge.src_obs_id, edge.dst_obs_id] if value is not None],
        source_edge_ids=[edge.edge_id],
        blocking_constraint_ids=[blocker.constraint_id for blocker in blockers],
        confidence=edge.confidence,
        reason_codes=["cannot_link_blocks_merge"],
        metadata={"fused_score": edge.fused_score},
    )


def _split_proposal_from_edge(edge: MatchEdge) -> ConsolidationProposal:
    reason_codes = ["conflicting_evidence_within_identity", "proposal_only"]
    if edge.local_score is not None and edge.local_score <= 0.45:
        reason_codes.append("global_local_conflict")
    if edge.metadata_score is not None and edge.metadata_score <= 0.30:
        reason_codes.append("metadata_conflict")
    if edge.temporal_score <= 0.20:
        reason_codes.append("temporal_conflict")
    return ConsolidationProposal(
        proposal_id=stable_id("csplit", edge.src_cat_id, edge.edge_id),
        proposal_type="split",
        cat_id=edge.src_cat_id,
        obs_ids=[value for value in [edge.src_obs_id, edge.dst_obs_id] if value is not None],
        source_edge_ids=[edge.edge_id],
        confidence=max(0.0, min(1.0, edge.confidence)),
        reason_codes=reason_codes,
        metadata={
            "global_score": edge.global_score,
            "local_score": edge.local_score,
            "temporal_score": edge.temporal_score,
            "metadata_score": edge.metadata_score,
        },
    )


def _split_proposal_from_constraint(
    store: CatNetStore,
    constraint: FeedbackConstraint,
) -> ConsolidationProposal | None:
    if constraint.constraint_type not in {
        FeedbackConstraintType.CANNOT_LINK,
        FeedbackConstraintType.DIFFERENT_IDENTITY,
    }:
        return None
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
    left_cat_id = constraint.left_cat_id or (left.cat_id if left else None)
    right_cat_id = constraint.right_cat_id or (right.cat_id if right else None)
    if left_cat_id is None or right_cat_id is None or left_cat_id != right_cat_id:
        return None
    obs_ids = [
        value
        for value in [constraint.left_obs_id, constraint.right_obs_id]
        if value is not None
    ]
    return ConsolidationProposal(
        proposal_id=stable_id("csplit", constraint.constraint_id, left_cat_id),
        proposal_type="split",
        cat_id=left_cat_id,
        obs_ids=obs_ids,
        blocking_constraint_ids=[constraint.constraint_id],
        confidence=1.0,
        reason_codes=["reviewer_cannot_link_within_identity", "proposal_only"],
    )


def _blocking_constraints_for_edge(
    edge: MatchEdge,
    constraints: list[FeedbackConstraint],
    store: CatNetStore,
) -> list[FeedbackConstraint]:
    blockers: list[FeedbackConstraint] = []
    edge_cat_pair = _pair_key(edge.src_cat_id, edge.dst_cat_id)
    edge_obs_pair = _pair_key(edge.src_obs_id, edge.dst_obs_id)
    for constraint in constraints:
        if constraint.constraint_type not in {
            FeedbackConstraintType.CANNOT_LINK,
            FeedbackConstraintType.DIFFERENT_IDENTITY,
        }:
            continue
        constraint_cat_pair = _constraint_cat_pair(constraint, store)
        constraint_obs_pair = _pair_key(constraint.left_obs_id, constraint.right_obs_id)
        if edge_cat_pair is not None and constraint_cat_pair == edge_cat_pair:
            blockers.append(constraint)
            continue
        if edge_obs_pair is not None and constraint_obs_pair == edge_obs_pair:
            blockers.append(constraint)
    return blockers


def _constraint_cat_pair(
    constraint: FeedbackConstraint,
    store: CatNetStore,
) -> tuple[str, str] | None:
    left_cat_id = constraint.left_cat_id
    right_cat_id = constraint.right_cat_id
    if left_cat_id is None and constraint.left_obs_id:
        left = store.catalog.observations.get(constraint.left_obs_id)
        left_cat_id = left.cat_id if left else None
    if right_cat_id is None and constraint.right_obs_id:
        right = store.catalog.observations.get(constraint.right_obs_id)
        right_cat_id = right.cat_id if right else None
    return _pair_key(left_cat_id, right_cat_id)


def _pair_key(left: str | None, right: str | None) -> tuple[str, str] | None:
    if left is None or right is None:
        return None
    return tuple(sorted((left, right)))


def _dedupe_proposals(proposals: list[ConsolidationProposal]) -> list[ConsolidationProposal]:
    seen: set[str] = set()
    deduped: list[ConsolidationProposal] = []
    for proposal in proposals:
        if proposal.proposal_id in seen:
            continue
        seen.add(proposal.proposal_id)
        deduped.append(proposal)
    return deduped


def _record_consolidation_events(store: CatNetStore, report: ConsolidationReport) -> None:
    for proposal in [*report.merge_proposals, *report.split_proposals]:
        if _proposal_event_exists(store, proposal.proposal_id):
            continue
        cat_id = proposal.target_cat_id or proposal.cat_id or proposal.source_cat_id
        if cat_id is None:
            continue
        identity = store.catalog.identities.get(cat_id)
        store.record_identity_event(
            cat_id=cat_id,
            event_type=IdentityEventType.UPDATED,
            reason=f"consolidation {proposal.proposal_type} proposal",
            previous_identity_version=identity.identity_version if identity else None,
            payload=proposal.model_dump(mode="json"),
        )


def _proposal_event_exists(store: CatNetStore, proposal_id: str) -> bool:
    for version in store.catalog.identity_versions.values():
        if version.payload.get("proposal_id") == proposal_id:
            return True
    return False
