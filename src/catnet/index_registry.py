"""Versioned CatNet index registry with hot/cold alias rollback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from catnet.matcher.faiss_index import FaissIndex
from catnet.schema import BranchType, IndexVersion, StrictModel, stable_id, utc_now
from catnet.storage import CatNetStore

IndexAlias = Literal["hot", "cold", "main", "candidate"]


class IndexManifest(StrictModel):
    schema_version: str = "1.0"
    index_version: IndexVersion
    relative_path: str
    sha256: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexAliasRecord(StrictModel):
    schema_version: str = "1.0"
    alias: str
    index_version: str
    previous_index_version: str | None = None
    updated_at: str
    actor: str = "system"

    @field_validator("alias", "index_version")
    @classmethod
    def required_strings_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("index alias records require non-empty strings")
        return value


class IndexPublishEvent(StrictModel):
    schema_version: str = "1.0"
    event_id: str
    event_type: Literal["built", "published", "rolled_back"]
    alias: str
    index_version: str
    previous_index_version: str | None = None
    actor: str = "system"
    reason: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRegistryState(StrictModel):
    schema_version: str = "1.0"
    manifests: dict[str, IndexManifest] = Field(default_factory=dict)
    aliases: dict[str, IndexAliasRecord] = Field(default_factory=dict)
    events: list[IndexPublishEvent] = Field(default_factory=list)


class IndexBuildResult(StrictModel):
    schema_version: str = "1.0"
    index_version: str
    alias: str
    vector_count: int
    registry_path: str
    index_path: str


class IndexRegistry:
    """Durable alias registry for immutable CatNet vector index files."""

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.path = self.registry_dir / "registry.json"
        self.index_dir = self.registry_dir / "indexes"
        self.state = self._load()

    def build_from_store(
        self,
        store: CatNetStore,
        *,
        alias: str = "candidate",
        model_version: str | None = None,
        shard_key: str = "species:cat",
        actor: str = "system",
    ) -> IndexBuildResult:
        embeddings = store.list_embeddings(branch_type=BranchType.FUSED)
        selected_model_version = model_version or _latest_model_version(embeddings)
        if selected_model_version is not None:
            embeddings = [
                embedding
                for embedding in embeddings
                if embedding.model_version == selected_model_version
            ]
        if not embeddings:
            detail = (
                f" for model_version={selected_model_version}"
                if selected_model_version
                else ""
            )
            raise ValueError(f"no fused embeddings available for index build{detail}")
        observations = store.observations_for_embeddings(embeddings)
        index = FaissIndex.from_embeddings(
            embeddings,
            observations,
            alias=alias,
            shard_key=shard_key,
        )
        now = utc_now()
        index_id = stable_id(
            "idx",
            index.index_version.model_version,
            shard_key,
            len(index.records),
            now.isoformat(),
        )
        index.index_version = index.index_version.model_copy(
            update={
                "index_version": index_id,
                "alias": alias,
                "vector_count": len(index.records),
                "created_at": now,
            }
        )
        index_path = self.index_dir / f"{index_id}.json"
        index.save(index_path)
        manifest = self._manifest_for(index_path, index.index_version)
        self.state.manifests[index_id] = manifest
        self._append_event(
            event_type="built",
            alias=alias,
            index_version=index_id,
            previous_index_version=self.state.aliases.get(alias).index_version
            if alias in self.state.aliases
            else None,
            actor=actor,
            reason="index built from CatNet store",
            metadata={"vector_count": len(index.records), "model_version": selected_model_version},
        )
        self.publish(alias=alias, index_version=index_id, actor=actor, reason="index build publish")
        return IndexBuildResult(
            index_version=index_id,
            alias=alias,
            vector_count=len(index.records),
            registry_path=str(self.path),
            index_path=str(index_path),
        )

    def publish(
        self,
        *,
        alias: str,
        index_version: str,
        actor: str = "system",
        reason: str = "manual publish",
    ) -> IndexAliasRecord:
        if index_version not in self.state.manifests:
            raise ValueError(f"index version is not registered: {index_version}")
        manifest = self.state.manifests[index_version]
        if not self._manifest_is_loadable(manifest):
            raise ValueError(f"index version is not loadable: {index_version}")
        previous = self.state.aliases.get(alias)
        now = utc_now().isoformat()
        record = IndexAliasRecord(
            alias=alias,
            index_version=index_version,
            previous_index_version=previous.index_version if previous else None,
            updated_at=now,
            actor=actor,
        )
        self.state.aliases[alias] = record
        self._append_event(
            event_type="published",
            alias=alias,
            index_version=index_version,
            previous_index_version=record.previous_index_version,
            actor=actor,
            reason=reason,
        )
        self.save()
        return record

    def rollback(
        self,
        *,
        alias: str,
        actor: str = "system",
        reason: str = "rollback alias",
    ) -> IndexAliasRecord:
        current = self.state.aliases.get(alias)
        if current is None:
            raise ValueError(f"alias is not published: {alias}")
        if current.previous_index_version is None:
            raise ValueError(f"alias has no rollback target: {alias}")
        if current.previous_index_version not in self.state.manifests:
            raise ValueError(f"rollback target is not registered: {current.previous_index_version}")
        record = IndexAliasRecord(
            alias=alias,
            index_version=current.previous_index_version,
            previous_index_version=current.index_version,
            updated_at=utc_now().isoformat(),
            actor=actor,
        )
        self.state.aliases[alias] = record
        self._append_event(
            event_type="rolled_back",
            alias=alias,
            index_version=record.index_version,
            previous_index_version=record.previous_index_version,
            actor=actor,
            reason=reason,
        )
        self.save()
        return record

    def status(self) -> dict[str, Any]:
        return self.state.model_dump(mode="json")

    def resolve_alias(self, alias: str) -> IndexManifest | None:
        record = self.state.aliases.get(alias)
        if record is None:
            return None
        return self.state.manifests.get(record.index_version)

    def load_current_index(self, alias: str = "hot") -> FaissIndex | None:
        manifest = self.resolve_alias(alias)
        if manifest is None:
            return None
        path = self._manifest_path(manifest)
        if path is None:
            return None
        try:
            return FaissIndex.load(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def save(self) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        payload = self.state.model_dump(mode="json")
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def _load(self) -> IndexRegistryState:
        if not self.path.exists():
            return IndexRegistryState()
        return IndexRegistryState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def _manifest_for(self, index_path: Path, index_version: IndexVersion) -> IndexManifest:
        return IndexManifest(
            index_version=index_version,
            relative_path=index_path.relative_to(self.registry_dir).as_posix(),
            sha256=_sha256(index_path),
            created_at=utc_now().isoformat(),
        )

    def _manifest_path(self, manifest: IndexManifest) -> Path | None:
        registry_root = self.registry_dir.resolve()
        path = (self.registry_dir / manifest.relative_path).resolve()
        if not path.is_relative_to(registry_root):
            return None
        if not path.is_file():
            return None
        if _sha256(path) != manifest.sha256:
            return None
        return path

    def _manifest_is_loadable(self, manifest: IndexManifest) -> bool:
        path = self._manifest_path(manifest)
        if path is None:
            return False
        try:
            FaissIndex.load(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
        return True

    def _append_event(
        self,
        *,
        event_type: Literal["built", "published", "rolled_back"],
        alias: str,
        index_version: str,
        previous_index_version: str | None,
        actor: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.state.events.append(
            IndexPublishEvent(
                event_id=stable_id(
                    "idxevt",
                    event_type,
                    alias,
                    index_version,
                    utc_now().isoformat(),
                ),
                event_type=event_type,
                alias=alias,
                index_version=index_version,
                previous_index_version=previous_index_version,
                actor=actor,
                reason=reason,
                created_at=utc_now().isoformat(),
                metadata=metadata or {},
            )
        )


def _latest_model_version(embeddings: list[object]) -> str | None:
    if not embeddings:
        return None
    latest = max(
        embeddings,
        key=lambda embedding: (
            getattr(embedding, "created_at", ""),
            getattr(embedding, "model_version", ""),
        ),
    )
    return str(latest.model_version)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
