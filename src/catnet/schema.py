"""Versioned CatNet data contracts.

The schema keeps identity, observations, embeddings, decisions, and indices
separate so every match can be replayed against the exact model/index versions
that produced it.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class IdentityStatus(StrEnum):
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    MERGED = "merged"
    SPLIT = "split"
    ARCHIVED = "archived"


class ObservationSourceType(StrEnum):
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"
    SCAN = "scan"


class BranchType(StrEnum):
    FACE = "face"
    BODY = "body"
    METADATA = "metadata"
    FUSED = "fused"


class DecisionType(StrEnum):
    MATCH_EXISTING = "match_existing"
    CREATE_NEW_IDENTITY = "create_new_identity"
    NEEDS_REVIEW = "needs_review"


class IdentityEventType(StrEnum):
    CREATED = "created"
    OBSERVATION_ASSIGNED = "observation_assigned"
    MERGED = "merged"
    SPLIT = "split"
    ROLLBACK = "rollback"
    UPDATED = "updated"


class IndexBackend(StrEnum):
    FAISS_FLAT_IP = "faiss_flat_ip"
    NUMPY_FLAT_IP = "numpy_flat_ip"
    IVF_FLAT = "ivf_flat"
    HNSW = "hnsw"
    IVF_PQ = "ivf_pq"


class FeedbackConstraintType(StrEnum):
    MUST_LINK = "must_link"
    CANNOT_LINK = "cannot_link"
    SAME_IDENTITY = "same_identity"
    DIFFERENT_IDENTITY = "different_identity"
    UNSURE = "unsure"


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:length]}"


def utc_now() -> datetime:
    return datetime.now(UTC)


def _ensure_utc(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value[:-1] + "+00:00" if value.endswith("Z") else value
        value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _validate_id(value: str) -> str:
    if not value or not ID_RE.fullmatch(value):
        raise ValueError("id must be non-empty and contain only safe identifier characters")
    return value


class ModelVersion(StrictModel):
    schema_version: str = "1.0"
    model_version: str
    backbone: str
    train_dataset_snapshot: str
    calibration_version: str
    embedding_dim: int = Field(gt=0)
    branch_types: list[BranchType] = Field(default_factory=lambda: list(BranchType))
    created_at: datetime = Field(default_factory=utc_now)
    notes: str | None = None

    @field_validator("model_version", "backbone", "train_dataset_snapshot", "calibration_version")
    @classmethod
    def strings_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("version fields must not be empty")
        return value


class IndexVersion(StrictModel):
    schema_version: str = "1.0"
    index_version: str
    backend: IndexBackend
    shard_key: str
    alias: str
    model_version: str
    metric: Literal["ip", "cosine"] = "ip"
    dim: int = Field(gt=0)
    vector_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("index_version", "shard_key", "alias", "model_version")
    @classmethod
    def ids_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("index fields must not be empty")
        return value


class MediaAsset(StrictModel):
    schema_version: str = "1.0"
    media_id: str
    storage_uri: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str
    uploader: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("media_id")
    @classmethod
    def media_id_must_be_safe(cls, value: str) -> str:
        return _validate_id(value)

    @field_validator("storage_uri", "mime_type")
    @classmethod
    def strings_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("string fields must not be empty")
        return value

    @field_validator("sha256")
    @classmethod
    def sha256_must_be_hex(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value


class CatIdentity(StrictModel):
    schema_version: str = "1.0"
    cat_id: str
    identity_version: str
    status: IdentityStatus = IdentityStatus.PROVISIONAL
    display_name: str | None = None
    sex: str | None = None
    breed_guess: str | None = None
    coat_pattern_summary: str | None = None
    birth_estimate_range: str | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    canonical_face_media_id: str | None = None
    canonical_body_media_id: str | None = None
    merged_into_cat_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cat_id", "identity_version")
    @classmethod
    def ids_must_be_safe(cls, value: str) -> str:
        return _validate_id(value)

    @field_validator("first_seen_at", "last_seen_at", mode="before")
    @classmethod
    def datetimes_are_utc(cls, value: datetime | None) -> datetime | None:
        return _ensure_utc(value)

    @model_validator(mode="after")
    def seen_window_must_be_ordered(self) -> CatIdentity:
        if self.first_seen_at and self.last_seen_at and self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at must be at or after first_seen_at")
        return self


class Observation(StrictModel):
    schema_version: str = "1.0"
    obs_id: str
    observation_version: str
    media_id: str
    cat_id: str | None = None
    species: str = "cat"
    captured_at: datetime | None = None
    encounter_id: str | None = None
    source_type: ObservationSourceType = ObservationSourceType.IMAGE
    camera_id: str | None = None
    site_id: str | None = None
    geohash_public: str | None = None
    lat_private: float | None = Field(default=None, ge=-90.0, le=90.0)
    lon_private: float | None = Field(default=None, ge=-180.0, le=180.0)
    age_bin: str | None = None
    size_bin: str | None = None
    face_view: str | None = None
    body_view: str | None = None
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reviewer_state: str = "unreviewed"
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("obs_id", "observation_version", "media_id")
    @classmethod
    def ids_must_be_safe(cls, value: str) -> str:
        return _validate_id(value)

    @field_validator("captured_at", mode="before")
    @classmethod
    def captured_at_is_utc(cls, value: datetime | None) -> datetime | None:
        return _ensure_utc(value)


class EmbeddingVector(StrictModel):
    schema_version: str = "1.0"
    emb_id: str
    obs_id: str
    model_version: str
    branch_type: BranchType
    dim: int = Field(gt=0)
    vector: list[float] | None = None
    vector_uri: str | None = None
    norm: float = Field(ge=0.0)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("emb_id", "obs_id")
    @classmethod
    def ids_must_be_safe(cls, value: str) -> str:
        return _validate_id(value)

    @field_validator("model_version")
    @classmethod
    def model_version_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_version must not be empty")
        return value

    @model_validator(mode="after")
    def vector_contract_is_consistent(self) -> EmbeddingVector:
        if self.vector is None and self.vector_uri is None:
            raise ValueError("embedding must include vector or vector_uri")
        if self.vector is not None and len(self.vector) != self.dim:
            raise ValueError("dim must match vector length")
        return self


class MatchEdge(StrictModel):
    schema_version: str = "1.0"
    edge_id: str
    src_obs_id: str
    dst_obs_id: str | None = None
    src_cat_id: str | None = None
    dst_cat_id: str | None = None
    global_score: float = Field(ge=-1.0, le=1.0)
    local_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    temporal_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata_score: float | None = Field(default=None, ge=0.0, le=1.0)
    fused_score: float = Field(ge=-1.0, le=1.0)
    decision: DecisionType
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
    index_version: str
    calibration_version: str
    decision_rule_version: str
    explanation: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("edge_id", "src_obs_id")
    @classmethod
    def ids_must_be_safe(cls, value: str) -> str:
        return _validate_id(value)


class IdentityVersion(StrictModel):
    schema_version: str = "1.0"
    identity_version: str
    cat_id: str
    event_type: IdentityEventType
    reason: str
    created_at: datetime = Field(default_factory=utc_now)
    previous_identity_version: str | None = None
    actor: str = "system"
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("identity_version", "cat_id")
    @classmethod
    def ids_must_be_safe(cls, value: str) -> str:
        return _validate_id(value)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be empty")
        return value


class FeedbackConstraint(StrictModel):
    schema_version: str = "1.0"
    constraint_id: str
    constraint_type: FeedbackConstraintType
    left_obs_id: str | None = None
    right_obs_id: str | None = None
    left_cat_id: str | None = None
    right_cat_id: str | None = None
    reviewer: str | None = None
    actor: str = "system"
    source: str = "manual"
    review_task_id: str | None = None
    reason: str | None = None
    status: Literal["open", "applied", "superseded"] = "open"
    created_at: datetime = Field(default_factory=utc_now)
    applied_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "constraint_id",
        "left_obs_id",
        "right_obs_id",
        "left_cat_id",
        "right_cat_id",
        "review_task_id",
    )
    @classmethod
    def optional_ids_must_be_safe(cls, value: str | None) -> str | None:
        return _validate_id(value) if value is not None else value

    @field_validator("actor", "source")
    @classmethod
    def source_fields_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("feedback actor/source must not be empty")
        return value

    @field_validator("applied_at", mode="before")
    @classmethod
    def applied_at_is_utc(cls, value: datetime | str | None) -> datetime | None:
        return _ensure_utc(value)

    @model_validator(mode="after")
    def link_contract_is_consistent(self) -> FeedbackConstraint:
        has_obs_pair = self.left_obs_id is not None or self.right_obs_id is not None
        has_cat_pair = self.left_cat_id is not None or self.right_cat_id is not None
        if has_obs_pair and (self.left_obs_id is None or self.right_obs_id is None):
            raise ValueError("feedback observation constraints require both observation ids")
        if has_cat_pair and (self.left_cat_id is None or self.right_cat_id is None):
            raise ValueError("feedback identity constraints require both cat ids")
        if (
            not has_obs_pair
            and not has_cat_pair
            and self.constraint_type != FeedbackConstraintType.UNSURE
        ):
            raise ValueError("feedback constraint requires an observation or identity pair")
        return self


class Identity3DAsset(StrictModel):
    schema_version: str = "1.0"
    asset_id: str
    cat_id: str
    source_obs_ids: list[str] = Field(default_factory=list)
    project_dir: str
    reconstruction_version: str
    mesh_path: str | None = None
    glb_path: str | None = None
    texture_path: str | None = None
    quality_report_path: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("asset_id", "cat_id")
    @classmethod
    def ids_must_be_safe(cls, value: str) -> str:
        return _validate_id(value)

    @field_validator("project_dir", "reconstruction_version")
    @classmethod
    def required_strings_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("3D asset project/version fields must not be empty")
        return value

    @field_validator("source_obs_ids")
    @classmethod
    def source_obs_ids_must_be_safe(cls, value: list[str]) -> list[str]:
        for obs_id in value:
            _validate_id(obs_id)
        return value


class MatchDecision(StrictModel):
    schema_version: str = "1.0"
    decision: DecisionType
    cat_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    fused_score: float = Field(ge=-1.0, le=1.0)
    visual_score: float = Field(ge=-1.0, le=1.0)
    local_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    temporal_score: float = Field(ge=0.0, le=1.0)
    metadata_score: float | None = Field(default=None, ge=0.0, le=1.0)
    top_candidate_obs_id: str | None = None
    top_candidate_cat_id: str | None = None
    model_version: str
    index_version: str
    calibration_version: str
    decision_rule_version: str
    explanation: dict[str, Any] = Field(default_factory=dict)
