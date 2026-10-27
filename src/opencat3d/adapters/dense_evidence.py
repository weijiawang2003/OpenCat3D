"""Local dense evidence adapter contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from opencat3d.adapters.base import ProviderInfo
from opencat3d.errors import AdapterUnavailableError
from opencat3d.io.cache import sha256_file
from opencat3d.registry import ModelRegistryEntry, assert_profile_allowed
from opencat3d.schemas import (
    DenseEvidenceFrame,
    DenseEvidenceKind,
    DenseEvidenceManifest,
    EvidenceProvenance,
    PerspectiveCameraState,
    TensorRef,
    TensorStorageFormat,
    write_json_model,
)

LOCAL_FILE_DENSE_EVIDENCE_BACKEND = "opencat3d/local_file_dense_evidence"
LOCAL_FILE_DENSE_EVIDENCE_VERSION = "1.0"
DENSE_EVIDENCE_MANIFEST_FILENAME = "dense_evidence_manifest.json"


@dataclass(frozen=True)
class DenseEvidenceRequest:
    project_id: str
    project_dir: Path
    observation_manifest_path: Path
    output_dir: Path
    frame_ids: tuple[str, ...]
    evidence_kinds: tuple[DenseEvidenceKind, ...]
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.project_id.strip():
            raise ValueError("project_id must not be empty")
        if not self.frame_ids:
            raise ValueError("frame_ids must not be empty")
        if not self.evidence_kinds:
            raise ValueError("evidence_kinds must not be empty")


class DenseEvidenceProvider(Protocol):
    def info(self) -> ProviderInfo: ...

    def estimate(self, request: DenseEvidenceRequest) -> DenseEvidenceManifest: ...


@dataclass(frozen=True)
class DenseEvidencePayload:
    frame_id: str
    kind: DenseEvidenceKind
    path: Path | None = None
    provenance: EvidenceProvenance = EvidenceProvenance.INFERRED
    semantic: str | None = None
    camera: PerspectiveCameraState | None = None

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame_id must not be empty")
        if self.kind == DenseEvidenceKind.CAMERA_PRIOR:
            if self.camera is None:
                raise ValueError("camera_prior payloads require camera")
            if self.path is not None:
                raise ValueError("camera_prior payloads must not include path")
            return
        if self.path is None:
            raise ValueError(f"{self.kind.value} payloads require path")


@dataclass(frozen=True)
class LocalFileDenseEvidenceProvider:
    payloads: tuple[DenseEvidencePayload, ...]
    name: str = LOCAL_FILE_DENSE_EVIDENCE_BACKEND
    version: str = LOCAL_FILE_DENSE_EVIDENCE_VERSION
    license_id: str = "project"

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            version=self.version,
            license_id=self.license_id,
        )

    def estimate(self, request: DenseEvidenceRequest) -> DenseEvidenceManifest:
        if not request.observation_manifest_path.exists():
            raise FileNotFoundError(
                f"observation manifest does not exist: {request.observation_manifest_path}"
            )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        payloads_by_key = {
            (payload.frame_id, payload.kind): payload for payload in self.payloads
        }
        used_payloads: list[DenseEvidencePayload] = []
        frames: list[DenseEvidenceFrame] = []
        payload_records: list[dict[str, object]] = []

        for frame_id in request.frame_ids:
            frame_payloads: dict[DenseEvidenceKind, DenseEvidencePayload] = {}
            for kind in request.evidence_kinds:
                payload = payloads_by_key.get((frame_id, kind))
                if payload is None:
                    raise ValueError(
                        "missing dense evidence payload for "
                        f"frame_id={frame_id!r}, kind={kind.value!r}"
                    )
                frame_payloads[kind] = payload
                used_payloads.append(payload)
            frame, records = _frame_from_dense_evidence_payloads(
                frame_id,
                frame_payloads,
                output_dir=request.output_dir,
            )
            frames.append(frame)
            payload_records.extend(records)

        manifest = DenseEvidenceManifest(
            project_id=request.project_id,
            source_backend=self.name,
            backend_version=self.version,
            license_id=self.license_id,
            deterministic=True,
            evidence_kinds=list(request.evidence_kinds),
            frame_ids=list(request.frame_ids),
            frames=frames,
            input_hash=_dense_evidence_input_hash(
                request=request,
                payload_records=payload_records,
            ),
            config_hash=_dense_evidence_config_hash(
                request=request,
                provider_info=self.info(),
            ),
            generated_evidence_fraction=_generated_payload_fraction(used_payloads),
            created_at=datetime(1970, 1, 1, tzinfo=UTC),
            notes=[
                "Local file import only; payloads supervise CatTemplate "
                "and do not replace fixed topology."
            ],
        )
        write_json_model(
            manifest,
            request.output_dir / DENSE_EVIDENCE_MANIFEST_FILENAME,
        )
        return manifest


@dataclass(frozen=True)
class UnavailableDenseEvidenceProvider:
    model_name: str
    profile: str = "research"
    registry: Mapping[str, ModelRegistryEntry] | None = None

    def info(self) -> ProviderInfo:
        registry = None if self.registry is None else dict(self.registry)
        entry = assert_profile_allowed(self.model_name, self.profile, registry)
        return ProviderInfo(
            name=self.model_name,
            version="unavailable",
            license_id=entry.weight_license,
        )

    def estimate(self, request: DenseEvidenceRequest) -> DenseEvidenceManifest:
        info = self.info()
        requested = ", ".join(kind.value for kind in request.evidence_kinds)
        raise AdapterUnavailableError(
            f"dense evidence backend '{info.name}' is optional and not installed; "
            f"requested evidence: {requested}; output_dir={request.output_dir}"
        )


def unavailable_dense_evidence_provider(
    model_name: str,
    *,
    profile: str = "research",
    registry: Mapping[str, ModelRegistryEntry] | None = None,
) -> UnavailableDenseEvidenceProvider:
    return UnavailableDenseEvidenceProvider(
        model_name=model_name,
        profile=profile,
        registry=registry,
    )


def _frame_from_dense_evidence_payloads(
    frame_id: str,
    payloads: Mapping[DenseEvidenceKind, DenseEvidencePayload],
    *,
    output_dir: Path,
) -> tuple[DenseEvidenceFrame, list[dict[str, object]]]:
    kwargs: dict[str, Any] = {"frame_id": frame_id}
    provenances = {payload.provenance for payload in payloads.values()}
    kwargs["provenance"] = (
        provenances.pop() if len(provenances) == 1 else EvidenceProvenance.UNKNOWN
    )
    records: list[dict[str, object]] = []
    for kind, payload in sorted(payloads.items(), key=lambda item: item[0].value):
        if kind == DenseEvidenceKind.CAMERA_PRIOR:
            kwargs["camera"] = payload.camera
            records.append(
                {
                    "frame_id": frame_id,
                    "kind": kind.value,
                    "provenance": payload.provenance.value,
                    "sha256": "camera_prior",
                }
            )
            continue
        ref = _import_npy_payload(payload, output_dir=output_dir)
        _assign_tensor_ref(kwargs, kind, ref)
        records.append(
            {
                "frame_id": frame_id,
                "kind": kind.value,
                "path": ref.path,
                "sha256": ref.sha256,
                "shape": ref.shape,
                "dtype": ref.dtype,
                "provenance": payload.provenance.value,
            }
        )
    frame = DenseEvidenceFrame(**kwargs)
    return frame, records


def _import_npy_payload(
    payload: DenseEvidencePayload,
    *,
    output_dir: Path,
) -> TensorRef:
    if payload.path is None:
        raise ValueError(f"{payload.kind.value} payload requires path")
    source_path = payload.path.expanduser().resolve()
    if source_path.suffix.lower() != ".npy":
        raise ValueError(f"dense evidence payload must be .npy: {source_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"dense evidence payload does not exist: {source_path}")
    array = np.load(source_path, mmap_mode="r")
    _validate_dense_evidence_array(payload.kind, array)
    imported_path = (
        output_dir
        / "payloads"
        / payload.frame_id
        / f"{payload.kind.value}.npy"
    )
    imported_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path != imported_path.resolve():
        shutil.copyfile(source_path, imported_path)
    semantic = payload.semantic or _default_dense_evidence_semantic(payload.kind)
    return TensorRef(
        path=str(imported_path.resolve()),
        shape=[int(dim) for dim in array.shape],
        dtype=str(array.dtype),
        sha256=sha256_file(imported_path),
        storage=TensorStorageFormat.NPY,
        semantic=semantic,
    )


def _assign_tensor_ref(
    frame_kwargs: dict[str, Any],
    kind: DenseEvidenceKind,
    ref: TensorRef,
) -> None:
    if kind == DenseEvidenceKind.DEPTH:
        frame_kwargs["depth"] = ref
    elif kind == DenseEvidenceKind.NORMAL:
        frame_kwargs["normal"] = ref
    elif kind == DenseEvidenceKind.TRACKS:
        frame_kwargs["tracks_xy"] = ref
    elif kind == DenseEvidenceKind.CORRESPONDENCE:
        frame_kwargs["correspondences"] = ref
    elif kind == DenseEvidenceKind.RENDER_PROBE:
        frame_kwargs["render_rgb"] = ref
    else:
        raise ValueError(f"unsupported tensor dense evidence kind: {kind.value}")


def _validate_dense_evidence_array(
    kind: DenseEvidenceKind,
    array: np.ndarray,
) -> None:
    if kind == DenseEvidenceKind.DEPTH and array.ndim != 2:
        raise ValueError("depth evidence must have shape [height,width]")
    if kind == DenseEvidenceKind.NORMAL and (
        array.ndim != 3 or int(array.shape[-1]) != 3
    ):
        raise ValueError("normal evidence must have shape [height,width,3]")
    if kind == DenseEvidenceKind.TRACKS and (
        array.ndim != 3 or int(array.shape[-1]) != 2
    ):
        raise ValueError("tracks evidence must have shape [frame,track,xy]")
    if kind == DenseEvidenceKind.CORRESPONDENCE and (
        array.ndim != 2 or int(array.shape[-1]) < 4
    ):
        raise ValueError("correspondence evidence must have shape [match,>=4]")
    if kind == DenseEvidenceKind.RENDER_PROBE and (
        array.ndim != 3 or int(array.shape[-1]) not in {3, 4}
    ):
        raise ValueError("render_probe evidence must have shape [height,width,3|4]")


def _default_dense_evidence_semantic(kind: DenseEvidenceKind) -> str:
    return {
        DenseEvidenceKind.DEPTH: "metric_depth_z",
        DenseEvidenceKind.NORMAL: "camera_normal_xyz",
        DenseEvidenceKind.TRACKS: "tracks_xy",
        DenseEvidenceKind.CORRESPONDENCE: "correspondence_xyxy",
        DenseEvidenceKind.RENDER_PROBE: "render_probe_rgb",
    }.get(kind, kind.value)


def _dense_evidence_input_hash(
    *,
    request: DenseEvidenceRequest,
    payload_records: list[dict[str, object]],
) -> str:
    observation_sha = sha256_file(request.observation_manifest_path)
    payload = {
        "observation_manifest_sha256": observation_sha,
        "payloads": payload_records,
        "frame_ids": list(request.frame_ids),
        "evidence_kinds": [kind.value for kind in request.evidence_kinds],
    }
    return _sha256_canonical_json(payload)


def _dense_evidence_config_hash(
    *,
    request: DenseEvidenceRequest,
    provider_info: ProviderInfo,
) -> str:
    payload = {
        "provider": provider_info.__dict__,
        "config": _json_safe(request.config),
    }
    return _sha256_canonical_json(payload)


def _generated_payload_fraction(payloads: list[DenseEvidencePayload]) -> float:
    if not payloads:
        return 0.0
    generated = sum(
        1 for payload in payloads if payload.provenance == EvidenceProvenance.GENERATED
    )
    return generated / len(payloads)


def _sha256_canonical_json(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
