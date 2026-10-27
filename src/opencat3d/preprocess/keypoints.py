"""Deterministic keypoint validation and pose seed bootstrap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import radians, tan
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opencat3d.io.cache import sha256_file
from opencat3d.preprocess.face_anchors import (
    RAW_FACE_ANCHOR_SOURCE,
    extract_observed_face_anchor_targets,
)
from opencat3d.schemas import (
    FrameObservationMeta,
    InitPredictionMeta,
    PerspectiveCameraState,
    TensorRef,
    TensorStorageFormat,
    write_json_model,
)
from opencat3d.template.cat_template import (
    create_placeholder_cat_template,
    load_cat_template_npz,
)


@dataclass(frozen=True)
class KeypointValidation:
    visible_count: int
    visible_fraction: float
    mean_confidence: float
    in_bounds_fraction: float
    in_bbox_fraction: float


@dataclass(frozen=True)
class KeypointBootstrapFrameResult:
    frame_id: str
    init_meta_path: Path
    visible_count: int
    mean_confidence: float
    in_bbox_fraction: float


@dataclass(frozen=True)
class RawFrameViewQuality:
    label: str
    frontal_face_confidence: float
    keypoint_weight_scale: float


@dataclass(frozen=True)
class SyntheticKeypointBootstrapResult:
    project_dir: Path
    manifest_path: Path
    frame_count: int
    mean_keypoint_confidence: float
    mean_in_bbox_fraction: float
    frames: tuple[KeypointBootstrapFrameResult, ...]


@dataclass(frozen=True)
class RawKeypointBootstrapResult:
    project_dir: Path
    manifest_path: Path
    frame_count: int
    mean_keypoint_confidence: float
    mean_in_bbox_fraction: float
    photo_frame_count: int
    video_frame_count: int
    enriched_keypoint_frame_count: int
    enriched_video_keypoint_frame_count: int
    observed_face_anchor_frame_count: int
    observed_face_anchor_count: int
    mean_face_anchor_confidence: float
    mean_frontal_face_confidence: float
    mean_keypoint_weight_scale: float
    frames: tuple[KeypointBootstrapFrameResult, ...]


def validate_keypoints_xyc(
    keypoints_xyc: NDArray[np.floating],
    *,
    image_width: int,
    image_height: int,
    bbox_xyxy: NDArray[np.floating] | list[float] | tuple[float, float, float, float],
    min_in_bbox_fraction: float = 0.75,
) -> KeypointValidation:
    if not 0.0 <= min_in_bbox_fraction <= 1.0:
        raise ValueError("min_in_bbox_fraction must be in [0,1]")
    keypoints = np.asarray(keypoints_xyc, dtype=np.float64)
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError("keypoints_xyc must have shape (N,3)")
    if not np.all(np.isfinite(keypoints)):
        raise ValueError("keypoints_xyc must be finite")
    if np.any((keypoints[:, 2] < 0.0) | (keypoints[:, 2] > 1.0)):
        raise ValueError("keypoint confidence must be in [0,1]")

    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,):
        raise ValueError("bbox_xyxy must have shape (4,)")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox_xyxy must satisfy x1 > x0 and y1 > y0")

    visible = keypoints[:, 2] > 0.0
    visible_count = int(visible.sum())
    if visible_count == 0:
        raise ValueError("at least one keypoint must be visible")

    visible_xy = keypoints[visible, :2]
    in_bounds = (
        (visible_xy[:, 0] >= 0.0)
        & (visible_xy[:, 0] < float(image_width))
        & (visible_xy[:, 1] >= 0.0)
        & (visible_xy[:, 1] < float(image_height))
    )
    if not np.all(in_bounds):
        raise ValueError("visible keypoints must lie inside image bounds")

    in_bbox = (
        (visible_xy[:, 0] >= x0)
        & (visible_xy[:, 0] < x1)
        & (visible_xy[:, 1] >= y0)
        & (visible_xy[:, 1] < y1)
    )
    in_bbox_fraction = float(np.mean(in_bbox))
    if in_bbox_fraction < min_in_bbox_fraction:
        raise ValueError("visible keypoints must mostly lie inside bbox_xyxy")

    return KeypointValidation(
        visible_count=visible_count,
        visible_fraction=float(visible_count / keypoints.shape[0]),
        mean_confidence=float(np.mean(keypoints[:, 2])),
        in_bounds_fraction=float(np.mean(in_bounds)),
        in_bbox_fraction=in_bbox_fraction,
    )


def bootstrap_raw_keypoints_and_pose(
    project_dir: Path,
    *,
    keypoint_confidence: float = 0.5,
    force: bool = False,
) -> RawKeypointBootstrapResult:
    if not 0.0 < keypoint_confidence <= 1.0:
        raise ValueError("keypoint_confidence must be in (0,1]")

    observations_dir = project_dir / "observations" / "raw"
    manifest_path = observations_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} is required")

    manifest = _read_json(manifest_path)
    template_path = _ensure_project_template(project_dir)
    template = load_cat_template_npz(template_path)
    init_dir = project_dir / "init" / "raw"
    init_dir.mkdir(parents=True, exist_ok=True)
    shape_code = np.zeros(template.shape_basis.shape[0], dtype=np.float32)
    pose_6d = np.tile(_identity_rotation_6d(), (template.joints_rest.shape[0], 1)).astype(
        np.float32
    )
    global_orient_6d = _identity_rotation_6d().astype(np.float32)
    vertices = template.shaped_vertices(shape_code).astype(np.float32)
    joints = template.joints(shape_code).astype(np.float32)

    frame_results: list[KeypointBootstrapFrameResult] = []
    confidences: list[float] = []
    bbox_fractions: list[float] = []
    updated_manifest_frames: list[dict[str, Any]] = []
    media_counts = {"photo": 0, "video": 0}
    enriched_counts = {"photo": 0, "video": 0}
    face_anchor_frame_count = 0
    face_anchor_counts: list[int] = []
    face_anchor_confidences: list[float] = []
    frontal_confidences: list[float] = []
    keypoint_weight_scales: list[float] = []
    view_label_counts: dict[str, int] = {}

    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("raw observation manifest contains no frames")

    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("raw observation manifest frame must be an object")
        frame_id = str(frame["frame_id"])
        media_type = _raw_frame_media_type(frame)
        media_counts[media_type] = int(media_counts.get(media_type, 0)) + 1
        observation_path = Path(
            str(frame.get("observation_path", observations_dir / frame_id / "observation.json"))
        )
        meta = FrameObservationMeta.model_validate_json(
            observation_path.read_text(encoding="utf-8")
        )
        image_srgb = _load_npz_array(Path(meta.image_srgb_path), "image_srgb")
        hard_mask = _load_npz_array(Path(meta.hard_mask.path), "hard_mask")
        image_height, image_width = _image_height_width(image_srgb)
        view_quality = _raw_frame_view_quality(
            frame,
            meta,
            image_width=image_width,
            image_height=image_height,
        )
        effective_keypoint_confidence = float(
            keypoint_confidence * view_quality.keypoint_weight_scale
        )
        keypoints = _bbox_keypoints_xyc(
            meta.bbox_xyxy,
            joint_count=template.joints_rest.shape[0],
            confidence=effective_keypoint_confidence,
        )
        validation = validate_keypoints_xyc(
            keypoints,
            image_width=image_width,
            image_height=image_height,
            bbox_xyxy=meta.bbox_xyxy,
            min_in_bbox_fraction=1.0,
        )
        face_anchor_extraction = extract_observed_face_anchor_targets(
            image_srgb,
            hard_mask,
            meta.bbox_xyxy,
            frontal_face_confidence=view_quality.frontal_face_confidence,
        )
        face_anchor_path = observation_path.parent / "face_anchor_targets.json"
        face_anchor_path.write_text(
            json.dumps(
                face_anchor_extraction.as_manifest(frame_id=frame_id),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        face_anchor_count = len(face_anchor_extraction.targets)
        face_anchor_counts.append(face_anchor_count)
        if face_anchor_count > 0:
            face_anchor_frame_count += 1
            face_anchor_confidences.extend(
                target.confidence for target in face_anchor_extraction.targets.values()
            )

        output_dir = init_dir / frame_id
        output_dir.mkdir(parents=True, exist_ok=True)
        init_meta_path = output_dir / "init_prediction.json"
        if not force and init_meta_path.exists():
            raise FileExistsError(f"{init_meta_path} already exists; pass force=True to overwrite")

        keypoints_ref = _save_array_ref(
            observation_path.parent / "keypoints_bbox.npz",
            "keypoints_xyc",
            keypoints,
        )
        quality = dict(meta.quality)
        quality.update(
            {
                "raw_keypoint_bootstrap": 1.0,
                "raw_keypoint_base_confidence": float(keypoint_confidence),
                "raw_frontal_face_confidence": view_quality.frontal_face_confidence,
                "raw_keypoint_view_weight": view_quality.keypoint_weight_scale,
                "raw_face_anchor_extraction": 1.0,
                "raw_face_anchor_target_count": float(face_anchor_count),
                "raw_face_anchor_mean_confidence": float(
                    face_anchor_extraction.metrics["mean_confidence"]
                ),
                "keypoint_confidence": validation.mean_confidence,
                "keypoint_in_bbox_fraction": validation.in_bbox_fraction,
            }
        )
        updated_meta = FrameObservationMeta(
            frame_id=meta.frame_id,
            image_srgb_path=meta.image_srgb_path,
            image_linear_path=meta.image_linear_path,
            hard_mask=meta.hard_mask,
            soft_mask=meta.soft_mask,
            bbox_xyxy=meta.bbox_xyxy,
            keypoints_xyc=keypoints_ref,
            dense_tracks=meta.dense_tracks,
            track_visibility=meta.track_visibility,
            view_hint=meta.view_hint,
            quality=quality,
            camera_prior=meta.camera_prior,
        )
        write_json_model(updated_meta, observation_path)
        updated_frame = dict(frame)
        updated_frame["keypoints_xyc"] = keypoints_ref.model_dump(mode="json")
        updated_frame["feature_anchors_xyc_path"] = str(face_anchor_path)
        updated_frame["feature_anchor_source"] = RAW_FACE_ANCHOR_SOURCE
        updated_frame["observed_face_anchor_count"] = int(face_anchor_count)
        updated_frame["observed_face_anchor_mean_confidence"] = float(
            face_anchor_extraction.metrics["mean_confidence"]
        )
        updated_frame["raw_view_label"] = view_quality.label
        updated_frame["raw_frontal_face_confidence"] = view_quality.frontal_face_confidence
        updated_frame["raw_keypoint_view_weight"] = view_quality.keypoint_weight_scale
        updated_quality = dict(updated_frame.get("quality", {}))
        updated_quality.update(quality)
        updated_frame["quality"] = updated_quality
        updated_manifest_frames.append(updated_frame)
        enriched_counts[media_type] = int(enriched_counts.get(media_type, 0)) + 1
        view_label_counts[view_quality.label] = (
            int(view_label_counts.get(view_quality.label, 0)) + 1
        )
        frontal_confidences.append(view_quality.frontal_face_confidence)
        keypoint_weight_scales.append(view_quality.keypoint_weight_scale)

        camera = _default_raw_camera(image_width=image_width, image_height=image_height)
        per_joint_confidence = keypoints[:, 2].astype(np.float32)
        init_meta = InitPredictionMeta(
            frame_id=frame_id,
            shape_code=_save_array_ref(output_dir / "shape_code.npz", "shape_code", shape_code),
            pose_6d=_save_array_ref(output_dir / "pose_6d.npz", "pose_6d", pose_6d),
            global_orient_6d=_save_array_ref(
                output_dir / "global_orient_6d.npz",
                "global_orient_6d",
                global_orient_6d,
            ),
            camera=camera,
            vertices=_save_array_ref(output_dir / "vertices.npz", "vertices", vertices),
            joints=_save_array_ref(output_dir / "joints.npz", "joints", joints),
            confidence=validation.mean_confidence,
            per_joint_confidence=_save_array_ref(
                output_dir / "per_joint_confidence.npz",
                "per_joint_confidence",
                per_joint_confidence,
            ),
            source_backend="opencat3d.raw_bbox_keypoint_bootstrap",
            backend_version="1.0",
            license_id=None,
        )
        write_json_model(init_meta, init_meta_path)
        confidences.append(validation.mean_confidence)
        bbox_fractions.append(validation.in_bbox_fraction)
        frame_results.append(
            KeypointBootstrapFrameResult(
                frame_id=frame_id,
                init_meta_path=init_meta_path,
                visible_count=validation.visible_count,
                mean_confidence=validation.mean_confidence,
                in_bbox_fraction=validation.in_bbox_fraction,
            )
        )

    bootstrap_manifest_path = init_dir / "manifest.json"
    bootstrap_manifest = {
        "schema_version": "1.0",
        "source": "opencat3d.raw_bbox_keypoint_bootstrap",
        "template_path": str(template_path),
        "frame_count": len(frame_results),
        "photo_frame_count": int(media_counts.get("photo", 0)),
        "video_frame_count": int(media_counts.get("video", 0)),
        "enriched_keypoint_frame_count": len(frame_results),
        "enriched_video_keypoint_frame_count": int(enriched_counts.get("video", 0)),
        "feature_anchor_source": RAW_FACE_ANCHOR_SOURCE,
        "observed_face_anchor_frame_count": int(face_anchor_frame_count),
        "observed_face_anchor_count": int(sum(face_anchor_counts)),
        "mean_face_anchor_confidence": (
            float(np.mean(face_anchor_confidences)) if face_anchor_confidences else 0.0
        ),
        "mean_keypoint_confidence": float(np.mean(confidences)),
        "mean_in_bbox_fraction": float(np.mean(bbox_fractions)),
        "frames": [
            {
                "frame_id": frame.frame_id,
                "init_prediction": str(frame.init_meta_path),
                "visible_count": frame.visible_count,
                "mean_confidence": frame.mean_confidence,
                "in_bbox_fraction": frame.in_bbox_fraction,
                "raw_view_label": next(
                    item["raw_view_label"]
                    for item in updated_manifest_frames
                    if item["frame_id"] == frame.frame_id
                ),
                "raw_frontal_face_confidence": next(
                    item["raw_frontal_face_confidence"]
                    for item in updated_manifest_frames
                    if item["frame_id"] == frame.frame_id
                ),
                "raw_keypoint_view_weight": next(
                    item["raw_keypoint_view_weight"]
                    for item in updated_manifest_frames
                    if item["frame_id"] == frame.frame_id
                ),
                "feature_anchors_xyc_path": next(
                    item["feature_anchors_xyc_path"]
                    for item in updated_manifest_frames
                    if item["frame_id"] == frame.frame_id
                ),
                "observed_face_anchor_count": next(
                    item["observed_face_anchor_count"]
                    for item in updated_manifest_frames
                    if item["frame_id"] == frame.frame_id
                ),
                "observed_face_anchor_mean_confidence": next(
                    item["observed_face_anchor_mean_confidence"]
                    for item in updated_manifest_frames
                    if item["frame_id"] == frame.frame_id
                ),
            }
            for frame in frame_results
        ],
        "view_label_counts": view_label_counts,
        "mean_frontal_face_confidence": float(np.mean(frontal_confidences)),
        "mean_keypoint_weight_scale": float(np.mean(keypoint_weight_scales)),
    }
    bootstrap_manifest_path.write_text(
        json.dumps(bootstrap_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    updated_manifest = dict(manifest)
    updated_manifest["frames"] = updated_manifest_frames
    updated_manifest["frame_count"] = len(updated_manifest_frames)
    updated_manifest["keypoint_bootstrap"] = {
        "source": "opencat3d.raw_bbox_keypoint_bootstrap",
        "manifest_path": str(bootstrap_manifest_path),
        "frame_count": len(frame_results),
        "photo_frame_count": int(media_counts.get("photo", 0)),
        "video_frame_count": int(media_counts.get("video", 0)),
        "enriched_keypoint_frame_count": len(frame_results),
        "enriched_video_keypoint_frame_count": int(enriched_counts.get("video", 0)),
        "feature_anchor_source": RAW_FACE_ANCHOR_SOURCE,
        "observed_face_anchor_frame_count": int(face_anchor_frame_count),
        "observed_face_anchor_count": int(sum(face_anchor_counts)),
        "mean_face_anchor_confidence": (
            float(np.mean(face_anchor_confidences)) if face_anchor_confidences else 0.0
        ),
        "view_label_counts": view_label_counts,
        "mean_frontal_face_confidence": float(np.mean(frontal_confidences)),
        "mean_keypoint_weight_scale": float(np.mean(keypoint_weight_scales)),
        "mean_keypoint_confidence": float(np.mean(confidences)),
        "mean_in_bbox_fraction": float(np.mean(bbox_fractions)),
    }
    manifest_path.write_text(
        json.dumps(updated_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RawKeypointBootstrapResult(
        project_dir=project_dir,
        manifest_path=bootstrap_manifest_path,
        frame_count=len(frame_results),
        mean_keypoint_confidence=float(np.mean(confidences)),
        mean_in_bbox_fraction=float(np.mean(bbox_fractions)),
        photo_frame_count=int(media_counts.get("photo", 0)),
        video_frame_count=int(media_counts.get("video", 0)),
        enriched_keypoint_frame_count=len(frame_results),
        enriched_video_keypoint_frame_count=int(enriched_counts.get("video", 0)),
        observed_face_anchor_frame_count=int(face_anchor_frame_count),
        observed_face_anchor_count=int(sum(face_anchor_counts)),
        mean_face_anchor_confidence=(
            float(np.mean(face_anchor_confidences)) if face_anchor_confidences else 0.0
        ),
        mean_frontal_face_confidence=float(np.mean(frontal_confidences)),
        mean_keypoint_weight_scale=float(np.mean(keypoint_weight_scales)),
        frames=tuple(frame_results),
    )


def bootstrap_synthetic_keypoints_and_pose(
    project_dir: Path,
    *,
    force: bool = False,
) -> SyntheticKeypointBootstrapResult:
    observations_dir = project_dir / "observations" / "synthetic"
    manifest_path = observations_dir / "manifest.json"
    cameras_path = observations_dir / "cameras.json"
    if not manifest_path.exists() or not cameras_path.exists():
        raise FileNotFoundError("synthetic manifest and cameras.json are required")

    manifest = _read_json(manifest_path)
    cameras_raw = _read_json(cameras_path)
    template = load_cat_template_npz(Path(str(manifest["template_path"])))
    init_dir = project_dir / "init" / "synthetic"
    init_dir.mkdir(parents=True, exist_ok=True)
    shape_code = np.zeros(template.shape_basis.shape[0], dtype=np.float32)
    pose_6d = np.tile(_identity_rotation_6d(), (template.joints_rest.shape[0], 1)).astype(
        np.float32
    )
    global_orient_6d = _identity_rotation_6d().astype(np.float32)
    vertices = template.shaped_vertices(shape_code).astype(np.float32)
    joints = template.joints(shape_code).astype(np.float32)

    frame_results: list[KeypointBootstrapFrameResult] = []
    confidences: list[float] = []
    bbox_fractions: list[float] = []

    for frame in manifest.get("frames", []):
        frame_id = str(frame["frame_id"])
        frame_dir = observations_dir / frame_id
        output_dir = init_dir / frame_id
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = frame_dir / "observation.json"
        meta = FrameObservationMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        camera = PerspectiveCameraState.model_validate(cameras_raw[frame_id])
        keypoints = _load_npz_array(Path(meta.keypoints_xyc.path), "keypoints_xyc")
        validation = validate_keypoints_xyc(
            keypoints,
            image_width=camera.image_width,
            image_height=camera.image_height,
            bbox_xyxy=meta.bbox_xyxy,
        )
        per_joint_confidence = keypoints[:, 2].astype(np.float32)
        init_meta_path = output_dir / "init_prediction.json"
        if not force and init_meta_path.exists():
            raise FileExistsError(f"{init_meta_path} already exists; pass force=True to overwrite")

        init_meta = InitPredictionMeta(
            frame_id=frame_id,
            shape_code=_save_array_ref(output_dir / "shape_code.npz", "shape_code", shape_code),
            pose_6d=_save_array_ref(output_dir / "pose_6d.npz", "pose_6d", pose_6d),
            global_orient_6d=_save_array_ref(
                output_dir / "global_orient_6d.npz",
                "global_orient_6d",
                global_orient_6d,
            ),
            camera=camera,
            vertices=_save_array_ref(output_dir / "vertices.npz", "vertices", vertices),
            joints=_save_array_ref(output_dir / "joints.npz", "joints", joints),
            confidence=validation.mean_confidence,
            per_joint_confidence=_save_array_ref(
                output_dir / "per_joint_confidence.npz",
                "per_joint_confidence",
                per_joint_confidence,
            ),
            source_backend="opencat3d.synthetic_keypoint_bootstrap",
            backend_version="1.0",
            license_id=None,
        )
        write_json_model(init_meta, init_meta_path)
        confidences.append(validation.mean_confidence)
        bbox_fractions.append(validation.in_bbox_fraction)
        frame_results.append(
            KeypointBootstrapFrameResult(
                frame_id=frame_id,
                init_meta_path=init_meta_path,
                visible_count=validation.visible_count,
                mean_confidence=validation.mean_confidence,
                in_bbox_fraction=validation.in_bbox_fraction,
            )
        )

    if not frame_results:
        raise ValueError("synthetic manifest contains no frames")

    bootstrap_manifest_path = init_dir / "manifest.json"
    bootstrap_manifest = {
        "schema_version": "1.0",
        "source": "opencat3d.synthetic_keypoint_bootstrap",
        "template_path": str(manifest["template_path"]),
        "frame_count": len(frame_results),
        "mean_keypoint_confidence": float(np.mean(confidences)),
        "mean_in_bbox_fraction": float(np.mean(bbox_fractions)),
        "frames": [
            {
                "frame_id": frame.frame_id,
                "init_prediction": str(frame.init_meta_path),
                "visible_count": frame.visible_count,
                "mean_confidence": frame.mean_confidence,
                "in_bbox_fraction": frame.in_bbox_fraction,
            }
            for frame in frame_results
        ],
    }
    bootstrap_manifest_path.write_text(
        json.dumps(bootstrap_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SyntheticKeypointBootstrapResult(
        project_dir=project_dir,
        manifest_path=bootstrap_manifest_path,
        frame_count=len(frame_results),
        mean_keypoint_confidence=float(np.mean(confidences)),
        mean_in_bbox_fraction=float(np.mean(bbox_fractions)),
        frames=tuple(frame_results),
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _raw_frame_media_type(frame: dict[str, Any]) -> str:
    if frame.get("media_type") == "video" or frame.get("source_type") == "video_frame":
        return "video"
    return "photo"


def _raw_frame_view_quality(
    frame: dict[str, Any],
    meta: FrameObservationMeta,
    *,
    image_width: int,
    image_height: int,
) -> RawFrameViewQuality:
    bbox = np.asarray(meta.bbox_xyxy, dtype=np.float64)
    x0, y0, x1, y1 = bbox
    width_fraction = max(0.0, float(x1 - x0)) / max(float(image_width), 1.0)
    height_fraction = max(0.0, float(y1 - y0)) / max(float(image_height), 1.0)
    aspect = width_fraction / max(height_fraction, 1e-6)
    bbox_area_ratio = float(
        meta.quality.get("bbox_area_ratio", width_fraction * height_fraction)
    )
    mask_area_ratio = float(meta.quality.get("mask_area_ratio", 0.0))
    fill_score = _clamp01(mask_area_ratio / max(bbox_area_ratio, 1e-6))
    center_x = float(x0 + x1) / max(2.0 * float(image_width), 1.0)
    center_y = float(y0 + y1) / max(2.0 * float(image_height), 1.0)
    center_score = _clamp01(1.0 - float(np.hypot(center_x - 0.5, center_y - 0.5)) / 0.55)
    square_score = _clamp01(1.0 - abs(float(np.log(max(aspect, 1e-6)))) / float(np.log(2.5)))
    area_score = _clamp01(
        1.0 - abs(float(np.log(max(bbox_area_ratio, 1e-6) / 0.24))) / float(np.log(4.0))
    )
    touches_border = float(meta.quality.get("touches_image_border", 0.0))
    border_score = 1.0 - _clamp01(touches_border)
    frontal_face_confidence = _clamp01(
        (0.42 * square_score + 0.24 * center_score + 0.20 * fill_score + 0.14 * area_score)
        * (0.75 + 0.25 * border_score)
    )
    keypoint_weight_scale = _clamp01(0.35 + 0.65 * frontal_face_confidence)
    if frontal_face_confidence >= 0.72:
        label = "frontal_face_candidate"
    elif aspect >= 1.55:
        label = "side_body_candidate"
    elif aspect <= 0.75:
        label = "vertical_body_candidate"
    else:
        label = "oblique_body_candidate"
    if frame.get("media_type") == "video" or frame.get("source_type") == "video_frame":
        label = f"video_{label}"
    return RawFrameViewQuality(
        label=label,
        frontal_face_confidence=round(float(frontal_face_confidence), 9),
        keypoint_weight_scale=round(float(keypoint_weight_scale), 9),
    )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _load_npz_array(path: Path, key: str) -> NDArray[np.float32]:
    with np.load(path) as data:
        if key not in data:
            raise ValueError(f"{path} is missing array {key!r}")
        return np.asarray(data[key], dtype=np.float32)


def _save_array_ref(path: Path, key: str, array: NDArray[np.floating]) -> TensorRef:
    np.savez_compressed(path, **{key: array})
    return TensorRef(
        path=str(path),
        shape=list(array.shape),
        dtype=str(array.dtype),
        sha256=sha256_file(path),
        storage=TensorStorageFormat.NPZ,
        semantic=key,
    )


def _identity_rotation_6d() -> NDArray[np.float64]:
    return np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def _ensure_project_template(project_dir: Path) -> Path:
    template_path = project_dir / "assets" / "cat_template" / "placeholder_cat_template.npz"
    if not template_path.exists():
        template_path.parent.mkdir(parents=True, exist_ok=True)
        create_placeholder_cat_template().save_npz(template_path)
    return template_path


def _image_height_width(image_srgb: NDArray[np.floating]) -> tuple[int, int]:
    image = np.asarray(image_srgb)
    if image.ndim != 3:
        raise ValueError("image_srgb must have shape (3,H,W) or (H,W,3)")
    if image.shape[0] == 3:
        return int(image.shape[1]), int(image.shape[2])
    if image.shape[2] == 3:
        return int(image.shape[0]), int(image.shape[1])
    raise ValueError("image_srgb must have an RGB channel dimension of length 3")


def _bbox_keypoints_xyc(
    bbox_xyxy: list[float],
    *,
    joint_count: int,
    confidence: float,
) -> NDArray[np.float32]:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,):
        raise ValueError("bbox_xyxy must have shape (4,)")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox_xyxy must satisfy x1 > x0 and y1 > y0")
    fractions = np.asarray(
        [
            [0.50, 0.50],
            [0.50, 0.18],
            [0.50, 0.82],
            [0.25, 0.50],
            [0.75, 0.50],
            [0.30, 0.28],
            [0.70, 0.28],
            [0.30, 0.72],
            [0.70, 0.72],
            [0.18, 0.62],
            [0.82, 0.62],
        ],
        dtype=np.float64,
    )
    if joint_count > fractions.shape[0]:
        repeats = int(np.ceil(joint_count / fractions.shape[0]))
        fractions = np.tile(fractions, (repeats, 1))
    fractions = fractions[:joint_count]
    xy = np.column_stack(
        [
            x0 + fractions[:, 0] * (x1 - x0),
            y0 + fractions[:, 1] * (y1 - y0),
        ]
    )
    conf = np.full((joint_count, 1), confidence, dtype=np.float64)
    return np.concatenate([xy, conf], axis=1).astype(np.float32)


def _default_raw_camera(*, image_width: int, image_height: int) -> PerspectiveCameraState:
    focal = 0.5 * float(max(image_width, image_height)) / tan(radians(45.0) * 0.5)
    return PerspectiveCameraState(
        image_width=image_width,
        image_height=image_height,
        fx=focal,
        fy=focal,
        cx=float(image_width) * 0.5,
        cy=float(image_height) * 0.5,
        translation_world_to_camera=[0.0, 0.0, 4.0],
    )
