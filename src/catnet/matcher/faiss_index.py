"""FAISS-compatible vector index wrapper with a deterministic NumPy fallback."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from catnet.schema import (
    BranchType,
    EmbeddingVector,
    IndexBackend,
    IndexVersion,
    Observation,
    stable_id,
)


@dataclass(frozen=True)
class IndexRecord:
    record_id: str
    obs_id: str
    embedding_id: str
    cat_id: str | None
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchCandidate:
    record_id: str
    obs_id: str
    embedding_id: str
    cat_id: str | None
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class FaissIndex:
    """A flat inner-product index that uses FAISS when available.

    The NumPy backend is exact and is kept as the default fallback so local
    tests do not require the heavy FAISS wheel.
    """

    def __init__(
        self,
        *,
        dim: int,
        model_version: str,
        alias: str = "catnet-hot",
        shard_key: str = "species:cat",
        index_version: IndexVersion | None = None,
    ) -> None:
        backend = (
            IndexBackend.FAISS_FLAT_IP
            if _faiss_available()
            else IndexBackend.NUMPY_FLAT_IP
        )
        self.index_version = index_version or IndexVersion(
            index_version=stable_id("idx", model_version, alias, shard_key, dim),
            backend=backend,
            shard_key=shard_key,
            alias=alias,
            model_version=model_version,
            dim=dim,
        )
        self.dim = dim
        self.records: list[IndexRecord] = []

    @classmethod
    def from_embeddings(
        cls,
        embeddings: list[EmbeddingVector],
        observations: Mapping[str, Observation],
        *,
        alias: str = "catnet-hot",
        shard_key: str = "species:cat",
    ) -> FaissIndex:
        fused = [embedding for embedding in embeddings if embedding.branch_type == BranchType.FUSED]
        if not fused:
            model_version = embeddings[0].model_version if embeddings else "unknown"
            dim = embeddings[0].dim if embeddings else 1
            return cls(dim=dim, model_version=model_version, alias=alias, shard_key=shard_key)
        index = cls(
            dim=fused[0].dim,
            model_version=fused[0].model_version,
            alias=alias,
            shard_key=shard_key,
        )
        for embedding in fused:
            observation = observations.get(embedding.obs_id)
            metadata = _metadata_from_observation(observation)
            index.add_embedding(
                embedding,
                cat_id=observation.cat_id if observation else None,
                metadata=metadata,
            )
        return index

    def add_embedding(
        self,
        embedding: EmbeddingVector,
        *,
        cat_id: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> IndexRecord:
        if embedding.vector is None:
            raise ValueError("FaissIndex requires in-record vectors for local indexing")
        if embedding.dim != self.dim:
            raise ValueError(f"embedding dim {embedding.dim} does not match index dim {self.dim}")
        record = IndexRecord(
            record_id=stable_id("rec", embedding.emb_id, cat_id or "", len(self.records)),
            obs_id=embedding.obs_id,
            embedding_id=embedding.emb_id,
            cat_id=cat_id,
            vector=_normalize(embedding.vector).tolist(),
            metadata=dict(metadata or {}),
        )
        self.records.append(record)
        self.index_version = self.index_version.model_copy(
            update={"vector_count": len(self.records)}
        )
        return record

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        exclude_obs_ids: set[str] | None = None,
    ) -> list[SearchCandidate]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.records:
            return []
        query = _normalize(query_vector)
        if query.shape[0] != self.dim:
            raise ValueError(f"query dim {query.shape[0]} does not match index dim {self.dim}")

        candidates: list[SearchCandidate] = []
        excluded = exclude_obs_ids or set()
        for record in self.records:
            if record.obs_id in excluded:
                continue
            if not _passes_filters(record.metadata, filters or {}):
                continue
            score = float(np.dot(query, np.asarray(record.vector, dtype=np.float64)))
            candidates.append(
                SearchCandidate(
                    record_id=record.record_id,
                    obs_id=record.obs_id,
                    embedding_id=record.embedding_id,
                    cat_id=record.cat_id,
                    score=score,
                    metadata=dict(record.metadata),
                )
            )
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_version": self.index_version.model_dump(mode="json"),
            "records": [record.__dict__ for record in self.records],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FaissIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        index_version = IndexVersion.model_validate(payload["index_version"])
        index = cls(
            dim=index_version.dim,
            model_version=index_version.model_version,
            alias=index_version.alias,
            shard_key=index_version.shard_key,
            index_version=index_version,
        )
        adapter = TypeAdapter(list[IndexRecord])
        index.records = adapter.validate_python(payload.get("records", []))
        return index


def _faiss_available() -> bool:
    return (
        importlib.util.find_spec("faiss") is not None
        or importlib.util.find_spec("faiss_cpu") is not None
    )


def _normalize(vector: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return np.zeros_like(array)
    return array / norm


def _metadata_from_observation(observation: Observation | None) -> dict[str, Any]:
    if observation is None:
        return {}
    return {
        "species": observation.species,
        "site_id": observation.site_id,
        "camera_id": observation.camera_id,
        "captured_at": observation.captured_at.isoformat() if observation.captured_at else None,
        "age_bin": observation.age_bin,
        "face_view": observation.face_view,
        "body_view": observation.body_view,
        "quality_score": observation.quality_score,
    }


def _passes_filters(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        if key == "captured_after":
            captured_at = _parse_dt(metadata.get("captured_at"))
            if captured_at is None or captured_at < _parse_dt(expected):
                return False
            continue
        if key == "captured_before":
            captured_at = _parse_dt(metadata.get("captured_at"))
            if captured_at is None or captured_at > _parse_dt(expected):
                return False
            continue
        actual = metadata.get(key)
        if isinstance(expected, set | list | tuple):
            if actual not in expected:
                return False
        elif expected is not None and actual != expected:
            return False
    return True


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"unsupported datetime filter value: {value!r}")
