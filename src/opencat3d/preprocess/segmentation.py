"""Deterministic segmentation bootstrap for synthetic RGB observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from opencat3d.io.cache import sha256_file
from opencat3d.schemas import (
    FrameObservationMeta,
    FrameRecord,
    SourceType,
    TensorRef,
    TensorStorageFormat,
    write_json_model,
)

RawSegmentationBackend = Literal["alpha_or_border_color_v1", "white_subject_v1", "auto_subject_v1"]


@dataclass(frozen=True)
class SegmentationFrameResult:
    frame_id: str
    hard_mask_path: Path
    soft_mask_path: Path
    bbox_xyxy: tuple[float, float, float, float]
    mask_area_ratio: float
    iou_with_previous_mask: float
    bbox_area_ratio: float = 0.0
    touches_image_border: bool = False
    full_frame_bbox: bool = False
    backend: str = ""


@dataclass(frozen=True)
class _MaskCandidate:
    backend: str
    mask: NDArray[np.float32]
    score: float


@dataclass(frozen=True)
class SyntheticSegmentationResult:
    project_dir: Path
    manifest_path: Path
    frame_count: int
    mean_iou_with_previous_masks: float
    frames: tuple[SegmentationFrameResult, ...]


@dataclass(frozen=True)
class RawSegmentationResult:
    project_dir: Path
    manifest_path: Path
    frame_count: int
    mean_mask_area_ratio: float
    frames: tuple[SegmentationFrameResult, ...]
    mean_bbox_area_ratio: float = 0.0
    full_frame_bbox_fraction: float = 0.0


def segment_synthetic_rgb(
    image_srgb: NDArray[np.floating],
    *,
    threshold: float = 0.05,
) -> NDArray[np.float32]:
    """Segment synthetic cat pixels from CHW or HWC RGB tensors."""

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    image = np.asarray(image_srgb, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError("image_srgb must have shape (3,H,W), (1,H,W), (H,W,3), or (H,W,1)")

    if image.shape[0] in {1, 3}:
        intensity = np.max(image, axis=0)
    elif image.shape[2] in {1, 3}:
        intensity = np.max(image, axis=2)
    else:
        raise ValueError("image_srgb must have an RGB/channel dimension of length 1 or 3")

    mask = (intensity > threshold).astype(np.float32)
    if mask.sum() <= 0.0:
        raise ValueError("segmentation produced an empty mask")
    return mask


def load_image_srgb(
    image_path: Path,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
        array = np.asarray(rgba, dtype=np.float32) / 255.0
    image_srgb = np.moveaxis(array[:, :, :3], 2, 0).astype(np.float32)
    alpha = array[:, :, 3].astype(np.float32)
    return image_srgb, alpha


def segment_raw_foreground(
    image_srgb: NDArray[np.floating],
    *,
    alpha: NDArray[np.floating] | None = None,
    alpha_threshold: float = 0.05,
    color_delta_threshold: float = 0.08,
    segmentation_backend: RawSegmentationBackend = "auto_subject_v1",
) -> NDArray[np.float32]:
    mask, _ = _segment_raw_foreground_with_backend(
        image_srgb,
        alpha=alpha,
        alpha_threshold=alpha_threshold,
        color_delta_threshold=color_delta_threshold,
        segmentation_backend=segmentation_backend,
    )
    return mask


def _segment_raw_foreground_with_backend(
    image_srgb: NDArray[np.floating],
    *,
    alpha: NDArray[np.floating] | None = None,
    alpha_threshold: float = 0.05,
    color_delta_threshold: float = 0.08,
    segmentation_backend: RawSegmentationBackend = "auto_subject_v1",
) -> tuple[NDArray[np.float32], str]:
    if not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("alpha_threshold must be in [0,1]")
    if color_delta_threshold < 0.0:
        raise ValueError("color_delta_threshold must be non-negative")
    if segmentation_backend not in {
        "alpha_or_border_color_v1",
        "white_subject_v1",
        "auto_subject_v1",
    }:
        raise ValueError(f"unsupported raw segmentation backend: {segmentation_backend}")

    rgb = _as_hwc_rgb(image_srgb)
    alpha_array = None if alpha is None else np.asarray(alpha, dtype=np.float32)
    if alpha_array is not None:
        if alpha_array.shape != rgb.shape[:2]:
            raise ValueError("alpha must match image height and width")
        if np.any(alpha_array < 0.999):
            mask = alpha_array > alpha_threshold
            if mask.sum() <= 0:
                raise ValueError("raw segmentation produced an empty mask")
            return mask.astype(np.float32), "alpha_v1"

    border_mask = _foreground_from_border_color(rgb, color_delta_threshold).astype(np.float32)
    if segmentation_backend == "alpha_or_border_color_v1":
        if border_mask.sum() <= 0:
            raise ValueError("raw segmentation produced an empty mask")
        return border_mask, "alpha_or_border_color_v1"

    white_mask = _foreground_from_white_subject(rgb)
    if segmentation_backend == "white_subject_v1":
        if white_mask.sum() <= 0:
            raise ValueError("raw segmentation produced an empty mask")
        return white_mask, "white_subject_v1"

    candidates = [
        _MaskCandidate(
            backend="alpha_or_border_color_v1",
            mask=border_mask,
            score=_foreground_candidate_score(border_mask),
        ),
        _MaskCandidate(
            backend="white_subject_v1",
            mask=white_mask,
            score=_foreground_candidate_score(white_mask),
        ),
    ]
    best = max(candidates, key=lambda item: item.score)
    mask = best.mask

    if mask.sum() <= 0:
        raise ValueError("raw segmentation produced an empty mask")
    return mask.astype(np.float32), best.backend


def bbox_from_mask(mask: NDArray[np.floating]) -> tuple[float, float, float, float]:
    mask_2d = _as_2d_mask(mask)
    ys, xs = np.nonzero(mask_2d > 0.0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("cannot compute bbox for an empty mask")
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def segment_raw_image_observations(
    project_dir: Path,
    *,
    alpha_threshold: float = 0.05,
    color_delta_threshold: float = 0.08,
    segmentation_backend: RawSegmentationBackend = "auto_subject_v1",
    force: bool = False,
) -> RawSegmentationResult:
    raw_manifest_path = project_dir / "raw" / "frames.json"
    observations_dir = project_dir / "observations" / "raw"
    manifest_path = observations_dir / "manifest.json"
    if not raw_manifest_path.exists():
        raise FileNotFoundError(f"{raw_manifest_path} is required")
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} already exists; pass force=True to overwrite")

    raw_manifest = _read_json(raw_manifest_path)
    frames_raw = raw_manifest.get("frames", [])
    if not isinstance(frames_raw, list) or not frames_raw:
        raise ValueError("raw frame manifest contains no frames")

    frame_results: list[SegmentationFrameResult] = []
    frame_metas: list[dict[str, Any]] = []
    mask_area_ratios: list[float] = []
    bbox_area_ratios: list[float] = []
    full_frame_bboxes: list[float] = []
    observations_dir.mkdir(parents=True, exist_ok=True)
    for frame_raw in frames_raw:
        frame = FrameRecord.model_validate(frame_raw)
        frame_dir = observations_dir / frame.frame_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        image_srgb, alpha = load_image_srgb(Path(frame.source_path))
        image_height, image_width = image_srgb.shape[1:]
        hard_mask, selected_backend = _segment_raw_foreground_with_backend(
            image_srgb,
            alpha=alpha,
            alpha_threshold=alpha_threshold,
            color_delta_threshold=color_delta_threshold,
            segmentation_backend=segmentation_backend,
        )
        soft_mask = hard_mask.copy()
        bbox = bbox_from_mask(hard_mask)
        keypoints_xyc = np.zeros((1, 3), dtype=np.float32)

        image_ref = _save_array_ref(
            frame_dir / "image_srgb.npz",
            "image_srgb",
            image_srgb.astype(np.float32),
            "image_srgb",
        )
        hard_ref = _save_array_ref(
            frame_dir / "hard_mask.npz",
            "hard_mask",
            hard_mask[None].astype(np.float32),
            "hard_mask",
        )
        soft_ref = _save_array_ref(
            frame_dir / "soft_mask.npz",
            "soft_mask",
            soft_mask[None].astype(np.float32),
            "soft_mask",
        )
        keypoint_ref = _save_array_ref(
            frame_dir / "keypoints_xyc.npz",
            "keypoints_xyc",
            keypoints_xyc,
            "keypoints",
        )
        mask_area_ratio = float(hard_mask.mean())
        bbox_width = max(0.0, float(bbox[2] - bbox[0]))
        bbox_height = max(0.0, float(bbox[3] - bbox[1]))
        bbox_area_ratio = float(
            (bbox_width * bbox_height) / max(float(image_width * image_height), 1.0)
        )
        touches_image_border = bool(
            bbox[0] <= 0.0
            or bbox[1] <= 0.0
            or bbox[2] >= float(image_width)
            or bbox[3] >= float(image_height)
        )
        full_frame_bbox = bool(touches_image_border and bbox_area_ratio >= 0.95)
        meta = FrameObservationMeta(
            frame_id=frame.frame_id,
            image_srgb_path=image_ref.path,
            image_linear_path=None,
            hard_mask=hard_ref,
            soft_mask=soft_ref,
            bbox_xyxy=list(bbox),
            keypoints_xyc=keypoint_ref,
            quality={
                "raw_foreground_segmentation": 1.0,
                "alpha_threshold": float(alpha_threshold),
                "color_delta_threshold": float(color_delta_threshold),
                "mask_area_ratio": mask_area_ratio,
                "bbox_area_ratio": bbox_area_ratio,
                "touches_image_border": float(touches_image_border),
                "full_frame_bbox": float(full_frame_bbox),
                "keypoint_confidence": 0.0,
            },
        )
        observation_path = frame_dir / "observation.json"
        write_json_model(meta, observation_path)
        meta_json = meta.model_dump(mode="json")
        meta_json["raw_source_path"] = frame.source_path
        meta_json["source_type"] = frame.source_type.value
        meta_json["media_type"] = (
            "video" if frame.source_type == SourceType.VIDEO_FRAME else "photo"
        )
        if frame.source_media_id is not None:
            meta_json["source_media_id"] = frame.source_media_id
        if frame.source_frame_index is not None:
            meta_json["source_frame_index"] = frame.source_frame_index
        meta_json["observation_path"] = str(observation_path)
        meta_json["image_width"] = image_width
        meta_json["image_height"] = image_height
        meta_json["segmentation_backend"] = selected_backend
        frame_metas.append(meta_json)
        mask_area_ratios.append(mask_area_ratio)
        bbox_area_ratios.append(bbox_area_ratio)
        full_frame_bboxes.append(float(full_frame_bbox))
        frame_results.append(
            SegmentationFrameResult(
                frame_id=frame.frame_id,
                hard_mask_path=Path(hard_ref.path),
                soft_mask_path=Path(soft_ref.path),
                bbox_xyxy=bbox,
                mask_area_ratio=mask_area_ratio,
                iou_with_previous_mask=1.0,
                bbox_area_ratio=bbox_area_ratio,
                touches_image_border=touches_image_border,
                full_frame_bbox=full_frame_bbox,
                backend=selected_backend,
            )
        )

    manifest = {
        "schema_version": "1.0",
        "segmentation_backend": segmentation_backend,
        "source": "opencat3d.raw_foreground_segmentation",
        "project_id": raw_manifest.get("project_id", project_dir.name),
        "frame_count": len(frame_results),
        "alpha_threshold": float(alpha_threshold),
        "color_delta_threshold": float(color_delta_threshold),
        "mean_mask_area_ratio": float(np.mean(mask_area_ratios)),
        "mean_bbox_area_ratio": float(np.mean(bbox_area_ratios)),
        "full_frame_bbox_fraction": float(np.mean(full_frame_bboxes)),
        "frames": frame_metas,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RawSegmentationResult(
        project_dir=project_dir,
        manifest_path=manifest_path,
        frame_count=len(frame_results),
        mean_mask_area_ratio=float(np.mean(mask_area_ratios)),
        frames=tuple(frame_results),
        mean_bbox_area_ratio=float(np.mean(bbox_area_ratios)),
        full_frame_bbox_fraction=float(np.mean(full_frame_bboxes)),
    )


def segment_synthetic_rgb_observations(
    project_dir: Path,
    *,
    threshold: float = 0.05,
    force: bool = False,
) -> SyntheticSegmentationResult:
    observations_dir = project_dir / "observations" / "synthetic"
    manifest_path = observations_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} is required")

    manifest = _read_json(manifest_path)
    updated_frames: list[dict[str, Any]] = []
    frame_results: list[SegmentationFrameResult] = []
    previous_ious: list[float] = []

    for frame in manifest.get("frames", []):
        frame_id = str(frame["frame_id"])
        frame_dir = observations_dir / frame_id
        meta_path = frame_dir / "observation.json"
        meta = FrameObservationMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        image = _load_npz_array(Path(meta.image_srgb_path), "image_srgb")
        previous_mask = _load_npz_array(Path(meta.hard_mask.path), "hard_mask")

        hard_mask = segment_synthetic_rgb(image, threshold=threshold)
        soft_mask = hard_mask.copy()
        bbox = bbox_from_mask(hard_mask)
        hard_mask_chw = hard_mask[None].astype(np.float32)
        soft_mask_chw = soft_mask[None].astype(np.float32)
        hard_path = frame_dir / "hard_mask_rgb.npz"
        soft_path = frame_dir / "soft_mask_rgb.npz"
        if not force and (hard_path.exists() or soft_path.exists()):
            raise FileExistsError(f"{frame_dir} already contains RGB segmentation outputs")

        hard_ref = _save_array_ref(hard_path, "hard_mask", hard_mask_chw, "hard_mask")
        soft_ref = _save_array_ref(soft_path, "soft_mask", soft_mask_chw, "soft_mask")
        iou = _mask_iou(hard_mask, previous_mask)
        previous_ious.append(iou)
        quality = dict(meta.quality)
        quality.update(
            {
                "rgb_segmentation": 1.0,
                "segmentation_threshold": float(threshold),
                "mask_area_ratio": float(hard_mask.mean()),
                "mask_iou_with_previous": iou,
            }
        )
        updated_meta = FrameObservationMeta(
            frame_id=meta.frame_id,
            image_srgb_path=meta.image_srgb_path,
            image_linear_path=meta.image_linear_path,
            hard_mask=hard_ref,
            soft_mask=soft_ref,
            bbox_xyxy=list(bbox),
            keypoints_xyc=meta.keypoints_xyc,
            dense_tracks=meta.dense_tracks,
            track_visibility=meta.track_visibility,
            view_hint=meta.view_hint,
            quality=quality,
            camera_prior=meta.camera_prior,
        )
        write_json_model(updated_meta, meta_path)
        updated_frames.append(updated_meta.model_dump(mode="json"))
        frame_results.append(
            SegmentationFrameResult(
                frame_id=frame_id,
                hard_mask_path=hard_path,
                soft_mask_path=soft_path,
                bbox_xyxy=bbox,
                mask_area_ratio=float(hard_mask.mean()),
                iou_with_previous_mask=iou,
            )
        )

    if not frame_results:
        raise ValueError("synthetic manifest contains no frames")
    manifest["frames"] = updated_frames
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SyntheticSegmentationResult(
        project_dir=project_dir,
        manifest_path=manifest_path,
        frame_count=len(frame_results),
        mean_iou_with_previous_masks=float(np.mean(previous_ious)),
        frames=tuple(frame_results),
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_npz_array(path: Path, key: str) -> NDArray[np.float32]:
    with np.load(path) as data:
        if key not in data:
            raise ValueError(f"{path} is missing array {key!r}")
        return np.asarray(data[key], dtype=np.float32)


def _save_array_ref(path: Path, key: str, array: NDArray[np.floating], semantic: str) -> TensorRef:
    np.savez_compressed(path, **{key: array})
    return TensorRef(
        path=str(path),
        shape=list(array.shape),
        dtype=str(array.dtype),
        sha256=sha256_file(path),
        storage=TensorStorageFormat.NPZ,
        semantic=semantic,
    )


def _mask_iou(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    mask_a = _as_2d_mask(a) > 0.0
    mask_b = _as_2d_mask(b) > 0.0
    intersection = np.logical_and(mask_a, mask_b).sum(dtype=np.float64)
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.float64)
    if union <= 0.0:
        return 1.0
    return float(intersection / union)


def _as_2d_mask(mask: NDArray[np.floating]) -> NDArray[np.float32]:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    raise ValueError("mask must have shape (H,W) or (1,H,W)")


def _as_hwc_rgb(image_srgb: NDArray[np.floating]) -> NDArray[np.float32]:
    image = np.asarray(image_srgb, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError("image_srgb must have shape (3,H,W) or (H,W,3)")
    if image.shape[0] == 3:
        return np.moveaxis(image, 0, 2)
    if image.shape[2] == 3:
        return image
    raise ValueError("image_srgb must have an RGB channel dimension of length 3")


def _foreground_from_border_color(
    image_hwc: NDArray[np.float32],
    color_delta_threshold: float,
) -> NDArray[np.bool_]:
    height, width, _ = image_hwc.shape
    border_width = max(1, min(height, width) // 16)
    border_pixels = np.concatenate(
        [
            image_hwc[:border_width, :, :].reshape(-1, 3),
            image_hwc[-border_width:, :, :].reshape(-1, 3),
            image_hwc[:, :border_width, :].reshape(-1, 3),
            image_hwc[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    color_delta = image_hwc - background_color[None, None, :]
    distance = np.sqrt(np.mean(color_delta * color_delta, axis=2))
    return distance > color_delta_threshold


def _foreground_from_white_subject(
    image_hwc: NDArray[np.float32],
    *,
    max_side: int = 384,
) -> NDArray[np.float32]:
    height, width, _ = image_hwc.shape
    small, scale = _resize_hwc_rgb(image_hwc, max_side=max_side)
    small_height, small_width, _ = small.shape
    max_channel = np.max(small, axis=2)
    min_channel = np.min(small, axis=2)
    chroma = max_channel - min_channel
    luma = (
        0.299 * small[:, :, 0]
        + 0.587 * small[:, :, 1]
        + 0.114 * small[:, :, 2]
    )
    yy, xx = np.mgrid[0:small_height, 0:small_width]
    center_prior = np.exp(
        -(
            ((xx - small_width * 0.50) / max(small_width * 0.50, 1.0)) ** 2
            + ((yy - small_height * 0.58) / max(small_height * 0.55, 1.0)) ** 2
        )
    )

    luma_cutoff = max(float(np.quantile(luma, 0.72)), 0.48)
    mask = (
        (luma > luma_cutoff)
        & (chroma < 0.25)
        & (min_channel > 0.38)
        & (center_prior > 0.02)
    )
    if float(np.mean(mask)) < 0.01:
        relaxed_cutoff = max(float(np.quantile(luma, 0.62)), 0.38)
        mask = (
            (luma > relaxed_cutoff)
            & (chroma < 0.32)
            & (min_channel > 0.30)
            & (center_prior > 0.02)
        )

    mask = _morphological_dilate(_morphological_erode(mask, iterations=1), iterations=2)
    component = _best_centered_component(mask)
    if component is None:
        return np.zeros((height, width), dtype=np.float32)

    component_mask = np.zeros_like(mask, dtype=bool)
    component_mask[component[:, 0], component[:, 1]] = True
    component_mask = _morphological_dilate(component_mask, iterations=3)
    if scale < 1.0:
        image = Image.fromarray(component_mask.astype(np.uint8) * 255)
        image = image.resize((width, height), Image.Resampling.NEAREST)
        component_mask = np.asarray(image, dtype=np.uint8) > 0
    return component_mask.astype(np.float32)


def _best_centered_component(mask: NDArray[np.bool_]) -> NDArray[np.int64] | None:
    height, width = mask.shape
    best_score = float("-inf")
    best_component: NDArray[np.int64] | None = None
    for component in _connected_components(mask):
        area_ratio = float(component.shape[0] / max(height * width, 1))
        if area_ratio < 0.001 or area_ratio > 0.45:
            continue
        y0, x0 = np.min(component, axis=0)
        y1, x1 = np.max(component, axis=0) + 1
        bbox_area_ratio = float(((x1 - x0) * (y1 - y0)) / max(height * width, 1))
        if bbox_area_ratio > 0.75:
            continue
        center_x = (x0 + x1) / max(2.0 * width, 1.0)
        center_y = (y0 + y1) / max(2.0 * height, 1.0)
        center_score = float(
            np.exp(-(((center_x - 0.50) / 0.42) ** 2 + ((center_y - 0.58) / 0.50) ** 2))
        )
        fill_ratio = area_ratio / max(bbox_area_ratio, 1e-6)
        touches_border = x0 <= 1 or y0 <= 1 or x1 >= width - 1 or y1 >= height - 1
        score = (
            area_ratio * 2.0
            + bbox_area_ratio * 0.8
            + center_score * 0.5
            + min(fill_ratio, 1.0) * 0.2
            - float(touches_border) * 0.6
        )
        if score > best_score:
            best_score = score
            best_component = component
    return best_component


def _foreground_candidate_score(mask: NDArray[np.floating]) -> float:
    mask_2d = _as_2d_mask(mask) > 0.0
    area_ratio = float(np.mean(mask_2d))
    if area_ratio <= 0.0:
        return float("-inf")
    height, width = mask_2d.shape
    y0, x0, y1, x1 = _mask_bounds_yxyx(mask_2d)
    bbox_area_ratio = float(((x1 - x0) * (y1 - y0)) / max(height * width, 1))
    fill_ratio = area_ratio / max(bbox_area_ratio, 1e-6)
    center_x = (x0 + x1) / max(2.0 * width, 1.0)
    center_y = (y0 + y1) / max(2.0 * height, 1.0)
    center_score = float(
        np.exp(-(((center_x - 0.50) / 0.45) ** 2 + ((center_y - 0.58) / 0.50) ** 2))
    )
    touches_border = x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height
    full_frame_bbox = touches_border and bbox_area_ratio >= 0.95
    return (
        min(area_ratio, 0.45) * 1.0
        + min(bbox_area_ratio, 0.55) * 0.7
        + min(fill_ratio, 1.0) * 0.2
        + center_score * 0.5
        - max(0.0, bbox_area_ratio - 0.65) * 2.0
        - float(touches_border) * 0.35
        - float(full_frame_bbox) * 1.5
    )


def _connected_components(mask: NDArray[np.bool_]) -> list[NDArray[np.int64]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[NDArray[np.int64]] = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys, xs, strict=False):
        if seen[start_y, start_x]:
            continue
        seen[start_y, start_x] = True
        stack = [(int(start_y), int(start_x))]
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        components.append(np.asarray(points, dtype=np.int64))
    return components


def _mask_bounds_yxyx(mask: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("cannot compute bounds for an empty mask")
    return int(ys.min()), int(xs.min()), int(ys.max() + 1), int(xs.max() + 1)


def _morphological_dilate(mask: NDArray[np.bool_], *, iterations: int) -> NDArray[np.bool_]:
    output = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(output, 1, mode="constant", constant_values=False)
        output = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return output


def _morphological_erode(mask: NDArray[np.bool_], *, iterations: int) -> NDArray[np.bool_]:
    output = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(output, 1, mode="constant", constant_values=True)
        output = (
            padded[:-2, :-2]
            & padded[:-2, 1:-1]
            & padded[:-2, 2:]
            & padded[1:-1, :-2]
            & padded[1:-1, 1:-1]
            & padded[1:-1, 2:]
            & padded[2:, :-2]
            & padded[2:, 1:-1]
            & padded[2:, 2:]
        )
    return output


def _resize_hwc_rgb(
    image_hwc: NDArray[np.float32],
    *,
    max_side: int,
) -> tuple[NDArray[np.float32], float]:
    height, width, _ = image_hwc.shape
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 1.0:
        return image_hwc, 1.0
    target_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    image = Image.fromarray(np.clip(image_hwc * 255.0, 0.0, 255.0).astype(np.uint8))
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0, scale
