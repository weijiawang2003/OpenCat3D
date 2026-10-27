"""Small JSON storage layer for local CatNet development and tests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from catnet.schema import (
    BranchType,
    CatIdentity,
    EmbeddingVector,
    FeedbackConstraint,
    Identity3DAsset,
    IdentityEventType,
    IdentityStatus,
    IdentityVersion,
    MatchEdge,
    MediaAsset,
    Observation,
    StrictModel,
    stable_id,
    utc_now,
)


class CatNetCatalog(StrictModel):
    schema_version: str = "1.0"
    media_assets: dict[str, MediaAsset] = Field(default_factory=dict)
    identities: dict[str, CatIdentity] = Field(default_factory=dict)
    observations: dict[str, Observation] = Field(default_factory=dict)
    embeddings: dict[str, EmbeddingVector] = Field(default_factory=dict)
    match_edges: dict[str, MatchEdge] = Field(default_factory=dict)
    identity_versions: dict[str, IdentityVersion] = Field(default_factory=dict)
    feedback_constraints: dict[str, FeedbackConstraint] = Field(default_factory=dict)
    identity_3d_assets: dict[str, Identity3DAsset] = Field(default_factory=dict)


class CatNetStore:
    """Persist CatNet records as a single auditable JSON catalog."""

    def __init__(self, root_dir: Path, *, filename: str = "catalog.json") -> None:
        self.root_dir = Path(root_dir)
        self.path = self.root_dir / filename
        self.catalog = self._load()

    def save(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        payload = self.catalog.model_dump(mode="json")
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def upsert_media(self, media: MediaAsset) -> MediaAsset:
        self.catalog.media_assets[media.media_id] = media
        return media

    def upsert_identity(self, identity: CatIdentity) -> CatIdentity:
        self.catalog.identities[identity.cat_id] = identity
        return identity

    def upsert_observation(self, observation: Observation) -> Observation:
        self.catalog.observations[observation.obs_id] = observation
        if observation.cat_id:
            self._extend_identity_window(observation.cat_id, observation.captured_at)
        return observation

    def upsert_embedding(self, embedding: EmbeddingVector) -> EmbeddingVector:
        self.catalog.embeddings[embedding.emb_id] = embedding
        return embedding

    def add_match_edge(self, edge: MatchEdge) -> MatchEdge:
        self.catalog.match_edges[edge.edge_id] = edge
        return edge

    def add_feedback_constraint(self, constraint: FeedbackConstraint) -> FeedbackConstraint:
        self.catalog.feedback_constraints[constraint.constraint_id] = constraint
        return constraint

    def upsert_3d_asset(self, asset: Identity3DAsset) -> Identity3DAsset:
        self.catalog.identity_3d_assets[asset.asset_id] = asset
        return asset

    def record_identity_event(
        self,
        *,
        cat_id: str,
        event_type: IdentityEventType,
        reason: str,
        previous_identity_version: str | None = None,
        actor: str = "system",
        payload: dict[str, Any] | None = None,
    ) -> IdentityVersion:
        version = IdentityVersion(
            identity_version=stable_id("idv", cat_id, event_type, reason, utc_now().isoformat()),
            cat_id=cat_id,
            event_type=event_type,
            reason=reason,
            previous_identity_version=previous_identity_version,
            actor=actor,
            payload=payload or {},
        )
        self.catalog.identity_versions[version.identity_version] = version
        return version

    def create_identity(
        self,
        *,
        seed: str,
        first_seen_at: datetime | None = None,
        status: IdentityStatus = IdentityStatus.PROVISIONAL,
        reason: str = "open-set new identity",
    ) -> CatIdentity:
        cat_id = stable_id("cat", seed, first_seen_at.isoformat() if first_seen_at else "unknown")
        existing = self.catalog.identities.get(cat_id)
        if existing is not None:
            return existing
        version = self.record_identity_event(
            cat_id=cat_id,
            event_type=IdentityEventType.CREATED,
            reason=reason,
            payload={"seed": seed},
        )
        identity = CatIdentity(
            cat_id=cat_id,
            identity_version=version.identity_version,
            status=status,
            first_seen_at=first_seen_at,
            last_seen_at=first_seen_at,
        )
        self.upsert_identity(identity)
        return identity

    def assign_observation(self, obs_id: str, cat_id: str, *, reason: str) -> Observation:
        observation = self.catalog.observations[obs_id]
        updated = observation.model_copy(update={"cat_id": cat_id})
        self.catalog.observations[obs_id] = updated
        identity = self.catalog.identities[cat_id]
        version = self.record_identity_event(
            cat_id=cat_id,
            event_type=IdentityEventType.OBSERVATION_ASSIGNED,
            reason=reason,
            previous_identity_version=identity.identity_version,
            payload={"obs_id": obs_id},
        )
        self.catalog.identities[cat_id] = identity.model_copy(
            update={
                "identity_version": version.identity_version,
                "updated_at": utc_now(),
            }
        )
        self._extend_identity_window(cat_id, updated.captured_at)
        return updated

    def list_embeddings(
        self,
        *,
        branch_type: BranchType | None = None,
        model_version: str | None = None,
    ) -> list[EmbeddingVector]:
        embeddings = list(self.catalog.embeddings.values())
        if branch_type is not None:
            embeddings = [
                embedding for embedding in embeddings if embedding.branch_type == branch_type
            ]
        if model_version is not None:
            embeddings = [
                embedding for embedding in embeddings if embedding.model_version == model_version
            ]
        return sorted(embeddings, key=lambda item: item.emb_id)

    def observations_for_embeddings(
        self,
        embeddings: Iterable[EmbeddingVector],
    ) -> dict[str, Observation]:
        obs_ids = {embedding.obs_id for embedding in embeddings}
        return {
            obs_id: observation
            for obs_id, observation in self.catalog.observations.items()
            if obs_id in obs_ids
        }

    def _load(self) -> CatNetCatalog:
        if not self.path.exists():
            return CatNetCatalog()
        return CatNetCatalog.model_validate_json(self.path.read_text(encoding="utf-8"))

    def _extend_identity_window(self, cat_id: str, captured_at: datetime | None) -> None:
        if captured_at is None or cat_id not in self.catalog.identities:
            return
        identity = self.catalog.identities[cat_id]
        first_seen_at = identity.first_seen_at
        last_seen_at = identity.last_seen_at
        if first_seen_at is None or captured_at < first_seen_at:
            first_seen_at = captured_at
        if last_seen_at is None or captured_at > last_seen_at:
            last_seen_at = captured_at
        self.catalog.identities[cat_id] = identity.model_copy(
            update={
                "first_seen_at": first_seen_at,
                "last_seen_at": last_seen_at,
                "updated_at": utc_now(),
            }
        )
