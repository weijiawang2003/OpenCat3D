"""Persistent schemas and lightweight runtime records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ModelT = TypeVar("ModelT", bound=BaseModel)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CaptureMode(StrEnum):
    AUTO = "auto"
    ALBUM = "album"
    SESSION = "session"
    STATIC_SCAN = "static_scan"


class SourceType(StrEnum):
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"


class TensorStorageFormat(StrEnum):
    SAFETENSORS = "safetensors"
    NPZ = "npz"
    NPY = "npy"
    IMAGE = "image"
    JSON = "json"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class DenseEvidenceKind(StrEnum):
    DEPTH = "depth"
    NORMAL = "normal"
    TRACKS = "tracks"
    CORRESPONDENCE = "correspondence"
    CAMERA_PRIOR = "camera_prior"
    RENDER_PROBE = "render_probe"


class EvidenceProvenance(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    GENERATED = "generated"
    UNKNOWN = "unknown"


class TensorRef(StrictModel):
    """JSON-safe reference to a tensor-like payload stored outside the schema."""

    path: str
    shape: list[int] = Field(default_factory=list)
    dtype: str
    sha256: str
    storage: TensorStorageFormat
    semantic: str | None = None

    @field_validator("path")
    @classmethod
    def path_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("path must not be empty")
        return value

    @field_validator("shape")
    @classmethod
    def dimensions_must_be_positive(cls, value: list[int]) -> list[int]:
        if any(dim <= 0 for dim in value):
            raise ValueError("shape dimensions must be positive")
        return value

    @field_validator("sha256")
    @classmethod
    def sha256_must_be_hex(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value


class CaptureManifest(StrictModel):
    schema_version: str = "1.0"
    project_id: str
    subject_id: str | None = None
    mode: CaptureMode = CaptureMode.AUTO
    source_paths: list[str] = Field(default_factory=list)
    known_body_length_m: float | None = None
    camera_intrinsics_path: str | None = None
    user_notes: str | None = None
    consent_record: str | None = None
    random_seed: int = 20260625

    @field_validator("project_id")
    @classmethod
    def project_id_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("project_id must not be empty")
        return value

    @field_validator("known_body_length_m")
    @classmethod
    def known_length_must_be_positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("known_body_length_m must be positive")
        return value


class FrameRecord(StrictModel):
    frame_id: str
    source_path: str
    source_type: SourceType
    source_media_id: str | None = None
    source_frame_index: int | None = Field(default=None, ge=0)
    timestamp_s: float | None = None
    exif_orientation: int | None = None
    exif_focal_mm: float | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str
    duplicate_group: str | None = None

    @field_validator("sha256")
    @classmethod
    def sha256_must_be_hex(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("exif_orientation")
    @classmethod
    def orientation_must_be_exif_value(cls, value: int | None) -> int | None:
        if value is not None and value not in range(1, 9):
            raise ValueError("exif_orientation must be in 1..8")
        return value


class CameraPrior(StrictModel):
    convention: Literal["opencv"] = "opencv"
    model: Literal["perspective", "weak_perspective"] = "perspective"
    fov_deg: float | None = Field(default=None, gt=0, lt=180)
    intrinsics_path: str | None = None
    extrinsics_path: str | None = None


def _identity_3x3() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class PerspectiveCameraState(StrictModel):
    convention: Literal["opencv"] = "opencv"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    rotation_world_to_camera: list[list[float]] = Field(default_factory=_identity_3x3)
    translation_world_to_camera: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @model_validator(mode="after")
    def matrices_have_expected_shapes(self) -> PerspectiveCameraState:
        if len(self.rotation_world_to_camera) != 3 or any(
            len(row) != 3 for row in self.rotation_world_to_camera
        ):
            raise ValueError("rotation_world_to_camera must be 3x3")
        if len(self.translation_world_to_camera) != 3:
            raise ValueError("translation_world_to_camera must have length 3")
        return self


class FrameObservationMeta(StrictModel):
    frame_id: str
    image_srgb_path: str
    image_linear_path: str | None = None
    hard_mask: TensorRef
    soft_mask: TensorRef
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    keypoints_xyc: TensorRef
    dense_tracks: TensorRef | None = None
    track_visibility: TensorRef | None = None
    view_hint: TensorRef | None = None
    quality: dict[str, float] = Field(default_factory=dict)
    camera_prior: CameraPrior = Field(default_factory=CameraPrior)

    @model_validator(mode="after")
    def bbox_is_ordered(self) -> FrameObservationMeta:
        x0, y0, x1, y1 = self.bbox_xyxy
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox_xyxy must satisfy x1 > x0 and y1 > y0")
        return self


class InitPredictionMeta(StrictModel):
    frame_id: str
    shape_code: TensorRef
    pose_6d: TensorRef
    global_orient_6d: TensorRef
    camera: PerspectiveCameraState
    vertices: TensorRef
    joints: TensorRef
    confidence: float = Field(ge=0.0, le=1.0)
    per_joint_confidence: TensorRef
    source_backend: str
    backend_version: str | None = None
    license_id: str | None = None


class FitStateMeta(StrictModel):
    beta: TensorRef
    canonical_displacement: TensorRef | None = None
    poses_6d: TensorRef
    cameras_path: str
    light_codes: TensorRef | None = None
    exposure_scale: TensorRef | None = None
    exposure_bias: TensorRef | None = None
    vertex_confidence: TensorRef | None = None
    loss_history_path: str


class FitReport(StrictModel):
    schema_version: str = "1.0"
    project_id: str
    success: bool
    num_frames: int = Field(ge=0)
    started_at: datetime
    completed_at: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    loss_history: dict[str, list[float]] = Field(default_factory=dict)
    output_paths: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    deterministic: bool = True


class StageManifest(StrictModel):
    schema_version: str = "1.0"
    stage_name: str
    status: StageStatus
    input_hash: str
    config_hash: str
    code_version: str
    random_seed: int
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    cache_key: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None

    @field_validator("input_hash", "config_hash")
    @classmethod
    def hashes_must_be_hex(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("hashes must be 64 lowercase hex characters")
        return value


class DenseEvidenceFrame(StrictModel):
    frame_id: str
    provenance: EvidenceProvenance = EvidenceProvenance.INFERRED
    camera: PerspectiveCameraState | None = None
    depth: TensorRef | None = None
    normal: TensorRef | None = None
    confidence: TensorRef | None = None
    tracks_xy: TensorRef | None = None
    track_visibility: TensorRef | None = None
    correspondences: TensorRef | None = None
    render_rgb: TensorRef | None = None
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("frame_id")
    @classmethod
    def frame_id_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("frame_id must not be empty")
        return value

    @model_validator(mode="after")
    def must_have_payload(self) -> DenseEvidenceFrame:
        payloads = (
            self.camera,
            self.depth,
            self.normal,
            self.confidence,
            self.tracks_xy,
            self.track_visibility,
            self.correspondences,
            self.render_rgb,
        )
        if all(payload is None for payload in payloads):
            raise ValueError("dense evidence frame must contain at least one payload")
        return self


class DenseEvidenceManifest(StrictModel):
    schema_version: str = "1.0"
    project_id: str
    source_backend: str
    backend_version: str | None = None
    license_id: str | None = None
    deterministic: bool = True
    evidence_kinds: list[DenseEvidenceKind]
    frame_ids: list[str]
    frames: list[DenseEvidenceFrame]
    input_hash: str | None = None
    config_hash: str | None = None
    generated_evidence_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: list[str] = Field(default_factory=list)

    @field_validator("project_id", "source_backend")
    @classmethod
    def required_strings_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required string must not be empty")
        return value

    @field_validator("evidence_kinds")
    @classmethod
    def evidence_kinds_must_not_be_empty(
        cls,
        value: list[DenseEvidenceKind],
    ) -> list[DenseEvidenceKind]:
        if not value:
            raise ValueError("evidence_kinds must not be empty")
        return value

    @field_validator("input_hash", "config_hash")
    @classmethod
    def optional_hashes_must_be_hex(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("hashes must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def frame_ids_match_frames(self) -> DenseEvidenceManifest:
        if not self.frame_ids:
            raise ValueError("frame_ids must not be empty")
        actual = [frame.frame_id for frame in self.frames]
        if actual != self.frame_ids:
            raise ValueError("frame_ids must match frames in order")
        return self


@dataclass
class FrameObservation:
    frame_id: str
    image_srgb: Any
    image_linear: Any
    hard_mask: Any
    soft_mask: Any
    bbox_xyxy: Any
    keypoints_xyc: Any
    dense_tracks: Any | None
    track_visibility: Any | None
    view_hint: Any
    quality: dict[str, float]
    camera_prior: CameraPrior


@dataclass
class InitPrediction:
    shape_code: Any
    pose_6d: Any
    global_orient_6d: Any
    camera: PerspectiveCameraState
    vertices: Any
    joints: Any
    confidence: float
    per_joint_confidence: Any
    source_backend: str


@dataclass
class FitState:
    beta: Any
    canonical_displacement: Any | None
    poses_6d: Any
    cameras: Any
    light_codes: Any | None
    exposure_scale: Any | None
    exposure_bias: Any | None
    vertex_confidence: Any | None
    loss_history: dict[str, list[float]] = field(default_factory=dict)


def write_json_model(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_json_model(model_type: type[ModelT], path: Path) -> ModelT:
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def model_to_canonical_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
