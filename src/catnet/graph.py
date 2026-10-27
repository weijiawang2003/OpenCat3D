"""Identity graph operations for merge, split, and rollback decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from catnet.schema import (
    CatIdentity,
    IdentityEventType,
    IdentityStatus,
    IdentityVersion,
    MatchEdge,
    Observation,
    stable_id,
    utc_now,
)


@dataclass
class IdentityGraph:
    identities: dict[str, CatIdentity] = field(default_factory=dict)
    observations: dict[str, Observation] = field(default_factory=dict)
    edges: dict[str, MatchEdge] = field(default_factory=dict)
    versions: dict[str, IdentityVersion] = field(default_factory=dict)

    def add_identity(self, identity: CatIdentity) -> None:
        self.identities[identity.cat_id] = identity

    def add_observation(self, observation: Observation) -> None:
        self.observations[observation.obs_id] = observation

    def add_edge(self, edge: MatchEdge) -> None:
        self.edges[edge.edge_id] = edge

    def merge_identities(
        self,
        *,
        source_cat_id: str,
        target_cat_id: str,
        reason: str,
    ) -> IdentityVersion:
        if source_cat_id == target_cat_id:
            raise ValueError("cannot merge an identity into itself")
        source = self.identities[source_cat_id]
        target = self.identities[target_cat_id]
        version = self._record_version(
            cat_id=source_cat_id,
            event_type=IdentityEventType.MERGED,
            reason=reason,
            previous_identity_version=source.identity_version,
            payload={"target_cat_id": target_cat_id},
        )
        self.identities[source_cat_id] = source.model_copy(
            update={
                "status": IdentityStatus.MERGED,
                "merged_into_cat_id": target_cat_id,
                "identity_version": version.identity_version,
                "updated_at": utc_now(),
            }
        )
        target_window = _merged_seen_window(source, target)
        self.identities[target_cat_id] = target.model_copy(
            update={
                "first_seen_at": target_window[0],
                "last_seen_at": target_window[1],
                "updated_at": utc_now(),
            }
        )
        for obs_id, observation in list(self.observations.items()):
            if observation.cat_id == source_cat_id:
                self.observations[obs_id] = observation.model_copy(update={"cat_id": target_cat_id})
        return version

    def split_identity(
        self,
        *,
        cat_id: str,
        obs_ids: list[str],
        reason: str,
        new_cat_id: str | None = None,
    ) -> CatIdentity:
        if not obs_ids:
            raise ValueError("split requires at least one observation")
        source = self.identities[cat_id]
        resolved_new_cat_id = new_cat_id or stable_id(
            "cat",
            cat_id,
            "split",
            ",".join(sorted(obs_ids)),
        )
        version = self._record_version(
            cat_id=resolved_new_cat_id,
            event_type=IdentityEventType.SPLIT,
            reason=reason,
            previous_identity_version=source.identity_version,
            payload={"source_cat_id": cat_id, "obs_ids": list(obs_ids)},
        )
        split_times = [
            self.observations[obs_id].captured_at
            for obs_id in obs_ids
            if obs_id in self.observations and self.observations[obs_id].captured_at is not None
        ]
        identity = CatIdentity(
            cat_id=resolved_new_cat_id,
            identity_version=version.identity_version,
            status=IdentityStatus.PROVISIONAL,
            first_seen_at=min(split_times) if split_times else None,
            last_seen_at=max(split_times) if split_times else None,
        )
        self.identities[resolved_new_cat_id] = identity
        for obs_id in obs_ids:
            observation = self.observations[obs_id]
            if observation.cat_id != cat_id:
                raise ValueError(f"observation {obs_id} is not assigned to {cat_id}")
            self.observations[obs_id] = observation.model_copy(
                update={"cat_id": resolved_new_cat_id}
            )
        self.identities[cat_id] = source.model_copy(update={"status": IdentityStatus.SPLIT})
        return identity

    def rollback(self, *, identity_version: str, reason: str) -> IdentityVersion:
        version = self.versions[identity_version]
        return self._record_version(
            cat_id=version.cat_id,
            event_type=IdentityEventType.ROLLBACK,
            reason=reason,
            previous_identity_version=identity_version,
            payload={"rolled_back_event": version.model_dump(mode="json")},
        )

    def _record_version(
        self,
        *,
        cat_id: str,
        event_type: IdentityEventType,
        reason: str,
        previous_identity_version: str | None,
        payload: dict[str, Any],
    ) -> IdentityVersion:
        version = IdentityVersion(
            identity_version=stable_id("idv", cat_id, event_type, reason, utc_now().isoformat()),
            cat_id=cat_id,
            event_type=event_type,
            reason=reason,
            previous_identity_version=previous_identity_version,
            payload=payload,
        )
        self.versions[version.identity_version] = version
        return version


def _merged_seen_window(
    source: CatIdentity,
    target: CatIdentity,
) -> tuple[object | None, object | None]:
    starts = [value for value in [source.first_seen_at, target.first_seen_at] if value is not None]
    ends = [value for value in [source.last_seen_at, target.last_seen_at] if value is not None]
    return (min(starts) if starts else None, max(ends) if ends else None)
