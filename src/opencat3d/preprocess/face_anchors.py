"""Observed raw face-anchor targets from local image evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

RAW_FACE_ANCHOR_SOURCE = "opencat3d.raw_observed_face_anchor_targets_v1"
_MAX_EXACT_COMPONENT_PIXELS = 20_000
_MAX_LOCAL_COMPONENT_SIDE = 192

_FACE_ANCHOR_NAMES = (
    "left_eye_center",
    "right_eye_center",
    "nose_tip",
    "muzzle_center",
    "left_ear_inner",
    "right_ear_inner",
)


@dataclass(frozen=True)
class FaceAnchorTarget:
    name: str
    x: float
    y: float
    confidence: float
    pixel_count: int

    def as_json(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "x": round(float(self.x), 6),
            "y": round(float(self.y), 6),
            "confidence": round(float(self.confidence), 9),
            "pixel_count": int(self.pixel_count),
        }


@dataclass(frozen=True)
class FaceAnchorExtraction:
    targets: dict[str, FaceAnchorTarget]
    checks: dict[str, bool]
    metrics: dict[str, float]

    def as_manifest(self, *, frame_id: str) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "source": RAW_FACE_ANCHOR_SOURCE,
            "provenance": "observed_pixels",
            "frame_id": frame_id,
            "target_count": len(self.targets),
            "mean_confidence": float(self.metrics["mean_confidence"]),
            "targets": {
                name: target.as_json() for name, target in sorted(self.targets.items())
            },
            "checks": self.checks,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class _ComponentSummary:
    count: int
    sum_x: float
    sum_y: float
    pixels_y: tuple[int, ...]
    pixels_x: tuple[int, ...]

    @property
    def center_xy(self) -> tuple[float, float]:
        if self.count <= 0:
            return 0.0, 0.0
        return self.sum_x / float(self.count), self.sum_y / float(self.count)


def extract_observed_face_anchor_targets(
    image_srgb: NDArray[np.floating],
    hard_mask: NDArray[np.floating],
    bbox_xyxy: NDArray[np.floating] | list[float] | tuple[float, float, float, float],
    *,
    frontal_face_confidence: float,
) -> FaceAnchorExtraction:
    """Estimate face-anchor 2D targets from observed pixels only.

    The extractor intentionally returns no target when local evidence is missing.
    BBox geometry is used only to define search regions; accepted targets come
    from foreground pixels selected by color/luma masks.
    """

    rgb = _as_hwc_rgb(image_srgb)
    mask = _as_mask2d(hard_mask, rgb.shape[:2])
    bbox = _clipped_bbox(bbox_xyxy, width=rgb.shape[1], height=rgb.shape[0])
    x0, y0, x1, y1 = bbox
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    face_confidence = _clamp01(frontal_face_confidence)

    foreground_count = int(np.count_nonzero(mask))
    checks = {
        "nonempty_foreground": foreground_count > 0,
        "bbox_valid": bbox_width > 1.0 and bbox_height > 1.0,
    }
    if foreground_count <= 0:
        return _empty_extraction(checks)

    luma = _rgb_luma(rgb)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    red_excess = rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])
    foreground_luma = luma[mask]
    dark_cutoff = min(float(np.quantile(foreground_luma, 0.14)), 0.45)
    bright_cutoff = max(float(np.quantile(foreground_luma, 0.62)), 0.50)

    upper = _bbox_region_mask(rgb.shape[:2], bbox, x_range=(0.05, 0.95), y_range=(0.00, 0.66))
    left_eye_region = _bbox_region_mask(
        rgb.shape[:2],
        bbox,
        x_range=(0.12, 0.50),
        y_range=(0.05, 0.52),
    )
    right_eye_region = _bbox_region_mask(
        rgb.shape[:2],
        bbox,
        x_range=(0.50, 0.88),
        y_range=(0.05, 0.52),
    )
    center_face = _bbox_region_mask(
        rgb.shape[:2],
        bbox,
        x_range=(0.23, 0.77),
        y_range=(0.22, 0.82),
    )
    upper_sides = _bbox_region_mask(
        rgb.shape[:2],
        bbox,
        x_range=(0.04, 0.96),
        y_range=(0.00, 0.42),
    )

    dark_candidate = mask & upper & (luma <= dark_cutoff) & (chroma >= 0.015)
    left_eye = _best_component(
        dark_candidate & left_eye_region,
        expected_center_xy=(x0 + 0.34 * bbox_width, y0 + 0.28 * bbox_height),
        min_pixels=3,
    )
    right_eye = _best_component(
        dark_candidate & right_eye_region,
        expected_center_xy=(x0 + 0.66 * bbox_width, y0 + 0.28 * bbox_height),
        min_pixels=3,
    )

    targets: dict[str, FaceAnchorTarget] = {}
    if left_eye is not None:
        targets["left_eye_center"] = _target_from_component(
            "left_eye_center",
            left_eye,
            base_confidence=face_confidence,
            min_pixels=3,
        )
    if right_eye is not None:
        targets["right_eye_center"] = _target_from_component(
            "right_eye_center",
            right_eye,
            base_confidence=face_confidence,
            min_pixels=3,
        )

    eye_pair = left_eye is not None and right_eye is not None
    checks["paired_eye_evidence"] = bool(eye_pair)
    if eye_pair:
        left_center = _mask_center_xy(left_eye)
        right_center = _mask_center_xy(right_eye)
        eye_mid_x = 0.5 * (left_center[0] + right_center[0])
        eye_mid_y = 0.5 * (left_center[1] + right_center[1])
        eye_span = max(abs(right_center[0] - left_center[0]), bbox_width * 0.12)
        nose_region = _ellipse_mask(
            rgb.shape[:2],
            center_xy=(eye_mid_x, eye_mid_y + max(0.55 * eye_span, 0.075 * bbox_height)),
            radius_xy=(
                max(0.40 * eye_span, 0.050 * bbox_width),
                max(0.32 * eye_span, 0.040 * bbox_height),
            ),
        )
        muzzle_region = _ellipse_mask(
            rgb.shape[:2],
            center_xy=(eye_mid_x, eye_mid_y + max(0.82 * eye_span, 0.120 * bbox_height)),
            radius_xy=(
                max(0.90 * eye_span, 0.100 * bbox_width),
                max(0.55 * eye_span, 0.070 * bbox_height),
            ),
        )
    else:
        nose_region = center_face & _bbox_region_mask(
            rgb.shape[:2],
            bbox,
            x_range=(0.32, 0.68),
            y_range=(0.40, 0.72),
        )
        muzzle_region = center_face & _bbox_region_mask(
            rgb.shape[:2],
            bbox,
            x_range=(0.24, 0.76),
            y_range=(0.50, 0.84),
        )

    warm = (
        mask
        & (red_excess > 0.035)
        & (chroma > 0.045)
        & (rgb[:, :, 0] > 0.34)
        & (luma > 0.20)
        & (luma < 0.92)
    )
    pink = warm & (red_excess > 0.060) & (rgb[:, :, 0] > rgb[:, :, 1] + 0.020)
    nose_component = _best_component(
        (pink & nose_region) | (warm & nose_region),
        expected_center_xy=(x0 + 0.50 * bbox_width, y0 + 0.56 * bbox_height),
        min_pixels=4,
    )
    if nose_component is not None:
        targets["nose_tip"] = _target_from_component(
            "nose_tip",
            nose_component,
            base_confidence=max(face_confidence, 0.35 if eye_pair else 0.0),
            min_pixels=4,
        )

    muzzle_component = _best_component(
        mask & muzzle_region & (luma >= bright_cutoff) & (chroma < 0.24),
        expected_center_xy=(x0 + 0.50 * bbox_width, y0 + 0.70 * bbox_height),
        min_pixels=10,
    )
    if muzzle_component is not None:
        targets["muzzle_center"] = _target_from_component(
            "muzzle_center",
            muzzle_component,
            base_confidence=max(face_confidence * 0.85, 0.30 if eye_pair else 0.0),
            min_pixels=10,
        )

    ear_candidate = warm & upper_sides
    x_coords = np.arange(rgb.shape[1], dtype=np.float64)[None, :]
    mid_x = x0 + 0.50 * bbox_width
    left_ear = _best_component(
        ear_candidate & (x_coords <= mid_x),
        expected_center_xy=(x0 + 0.25 * bbox_width, y0 + 0.14 * bbox_height),
        min_pixels=4,
    )
    right_ear = _best_component(
        ear_candidate & (x_coords >= mid_x),
        expected_center_xy=(x0 + 0.75 * bbox_width, y0 + 0.14 * bbox_height),
        min_pixels=4,
    )
    if left_ear is not None:
        targets["left_ear_inner"] = _target_from_component(
            "left_ear_inner",
            left_ear,
            base_confidence=max(face_confidence * 0.70, 0.25 if eye_pair else 0.0),
            min_pixels=4,
        )
    if right_ear is not None:
        targets["right_ear_inner"] = _target_from_component(
            "right_ear_inner",
            right_ear,
            base_confidence=max(face_confidence * 0.70, 0.25 if eye_pair else 0.0),
            min_pixels=4,
        )

    targets = {
        name: target
        for name, target in targets.items()
        if target.confidence > 0.0 and name in _FACE_ANCHOR_NAMES
    }
    checks.update(
        {
            "left_eye_observed": "left_eye_center" in targets,
            "right_eye_observed": "right_eye_center" in targets,
            "nose_observed": "nose_tip" in targets,
            "muzzle_observed": "muzzle_center" in targets,
            "ear_inner_observed": (
                "left_ear_inner" in targets or "right_ear_inner" in targets
            ),
        }
    )
    confidences = [target.confidence for target in targets.values()]
    metrics = {
        "foreground_pixel_count": float(foreground_count),
        "face_anchor_target_count": float(len(targets)),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "frontal_face_confidence": float(face_confidence),
        "dark_cutoff": float(dark_cutoff),
        "bright_cutoff": float(bright_cutoff),
    }
    return FaceAnchorExtraction(targets=targets, checks=checks, metrics=metrics)


def _empty_extraction(checks: dict[str, bool]) -> FaceAnchorExtraction:
    return FaceAnchorExtraction(
        targets={},
        checks=dict(checks),
        metrics={
            "foreground_pixel_count": 0.0,
            "face_anchor_target_count": 0.0,
            "mean_confidence": 0.0,
            "frontal_face_confidence": 0.0,
            "dark_cutoff": 0.0,
            "bright_cutoff": 0.0,
        },
    )


def _as_hwc_rgb(image_srgb: NDArray[np.floating]) -> NDArray[np.float32]:
    image = np.asarray(image_srgb, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError("image_srgb must have shape (3,H,W) or (H,W,3)")
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, 2)
    if image.shape[2] != 3:
        raise ValueError("image_srgb must have three RGB channels")
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _as_mask2d(mask: NDArray[np.floating], shape: tuple[int, int]) -> NDArray[np.bool_]:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != shape:
        raise ValueError("hard_mask must match image height and width")
    return np.asarray(array > 0.5, dtype=np.bool_)


def _clipped_bbox(
    bbox_xyxy: NDArray[np.floating] | list[float] | tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,):
        raise ValueError("bbox_xyxy must have shape (4,)")
    x0, y0, x1, y1 = bbox
    x0 = float(np.clip(x0, 0.0, float(width - 1)))
    y0 = float(np.clip(y0, 0.0, float(height - 1)))
    x1 = float(np.clip(x1, x0 + 1.0, float(width)))
    y1 = float(np.clip(y1, y0 + 1.0, float(height)))
    return x0, y0, x1, y1


def _rgb_luma(image: NDArray[np.float32]) -> NDArray[np.float32]:
    return (
        0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    ).astype(np.float32)


def _bbox_region_mask(
    shape: tuple[int, int],
    bbox_xyxy: tuple[float, float, float, float],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> NDArray[np.bool_]:
    x0, y0, x1, y1 = bbox_xyxy
    height, width = shape
    xs = np.arange(width, dtype=np.float64)[None, :]
    ys = np.arange(height, dtype=np.float64)[:, None]
    left = x0 + x_range[0] * (x1 - x0)
    right = x0 + x_range[1] * (x1 - x0)
    top = y0 + y_range[0] * (y1 - y0)
    bottom = y0 + y_range[1] * (y1 - y0)
    return (xs >= left) & (xs < right) & (ys >= top) & (ys < bottom)


def _ellipse_mask(
    shape: tuple[int, int],
    *,
    center_xy: tuple[float, float],
    radius_xy: tuple[float, float],
) -> NDArray[np.bool_]:
    height, width = shape
    xs = np.arange(width, dtype=np.float64)[None, :]
    ys = np.arange(height, dtype=np.float64)[:, None]
    cx, cy = center_xy
    rx, ry = max(radius_xy[0], 1.0), max(radius_xy[1], 1.0)
    return (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0


def _best_component(
    mask: NDArray[np.bool_],
    *,
    expected_center_xy: tuple[float, float],
    min_pixels: int,
) -> NDArray[np.bool_] | None:
    source = np.asarray(mask, dtype=np.bool_)
    active_count = int(np.count_nonzero(source))
    if active_count < min_pixels:
        return None
    expected = np.asarray(expected_center_xy, dtype=np.float64)
    if active_count > _MAX_EXACT_COMPONENT_PIXELS:
        return _bounded_component_near_expected(
            source,
            expected_center_xy=(float(expected[0]), float(expected[1])),
            min_pixels=min_pixels,
        )

    components = _connected_component_summaries(source)
    if not components:
        return None

    def rank(component: _ComponentSummary) -> tuple[float, float]:
        count = float(component.count)
        center = np.asarray(component.center_xy, dtype=np.float64)
        distance = float(np.linalg.norm(center - expected))
        return (min(count / max(float(min_pixels), 1.0), 12.0), -distance)

    best = max(components, key=rank)
    if best.count < min_pixels:
        return None
    return _component_summary_mask(source.shape, best)


def _bounded_component_near_expected(
    source: NDArray[np.bool_],
    *,
    expected_center_xy: tuple[float, float],
    min_pixels: int,
) -> NDArray[np.bool_] | None:
    seed = _nearest_active_pixel(source, expected_center_xy=expected_center_xy)
    if seed is None:
        return None
    crop = _component_crop_bounds(
        source.shape,
        center_yx=seed,
        side=_MAX_LOCAL_COMPONENT_SIDE,
    )
    component = _flood_component_in_crop(source, seed_yx=seed, crop_yx=crop)
    if component is None or int(np.count_nonzero(component)) < min_pixels:
        return None
    return component


def _nearest_active_pixel(
    source: NDArray[np.bool_],
    *,
    expected_center_xy: tuple[float, float],
) -> tuple[int, int] | None:
    ys, xs = np.nonzero(source)
    if xs.size == 0:
        return None
    expected_x, expected_y = expected_center_xy
    distances = (xs.astype(np.float64) - expected_x) ** 2 + (
        ys.astype(np.float64) - expected_y
    ) ** 2
    index = int(np.argmin(distances))
    return int(ys[index]), int(xs[index])


def _component_crop_bounds(
    shape: tuple[int, int],
    *,
    center_yx: tuple[int, int],
    side: int,
) -> tuple[int, int, int, int]:
    height, width = shape
    center_y, center_x = center_yx
    half = max(1, side // 2)
    y0 = max(0, int(center_y) - half)
    x0 = max(0, int(center_x) - half)
    y1 = min(height, y0 + side)
    x1 = min(width, x0 + side)
    y0 = max(0, y1 - side)
    x0 = max(0, x1 - side)
    return y0, y1, x0, x1


def _flood_component_in_crop(
    source: NDArray[np.bool_],
    *,
    seed_yx: tuple[int, int],
    crop_yx: tuple[int, int, int, int],
) -> NDArray[np.bool_] | None:
    y0, y1, x0, x1 = crop_yx
    seed_y, seed_x = seed_yx
    if not (y0 <= seed_y < y1 and x0 <= seed_x < x1):
        return None
    crop = source[y0:y1, x0:x1]
    local_seed_y = seed_y - y0
    local_seed_x = seed_x - x0
    if not bool(crop[local_seed_y, local_seed_x]):
        return None
    visited = np.zeros(crop.shape, dtype=np.bool_)
    stack = [(int(local_seed_y), int(local_seed_x))]
    visited[local_seed_y, local_seed_x] = True
    pixels_y: list[int] = []
    pixels_x: list[int] = []
    height, width = crop.shape
    while stack:
        y, x = stack.pop()
        pixels_y.append(y + y0)
        pixels_x.append(x + x0)
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if bool(crop[ny, nx]) and not bool(visited[ny, nx]):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    if not pixels_y:
        return None
    component = np.zeros(source.shape, dtype=np.bool_)
    component[np.asarray(pixels_y, dtype=np.int64), np.asarray(pixels_x, dtype=np.int64)] = True
    return component


def _connected_component_summaries(mask: NDArray[np.bool_]) -> list[_ComponentSummary]:
    source = np.asarray(mask, dtype=np.bool_)
    visited = np.zeros(source.shape, dtype=np.bool_)
    height, width = source.shape
    components: list[_ComponentSummary] = []
    ys, xs = np.nonzero(source)
    for seed_y, seed_x in zip(ys, xs, strict=True):
        if visited[seed_y, seed_x]:
            continue
        stack = [(int(seed_y), int(seed_x))]
        visited[seed_y, seed_x] = True
        pixels_y: list[int] = []
        pixels_x: list[int] = []
        sum_y = 0.0
        sum_x = 0.0
        while stack:
            y, x = stack.pop()
            pixels_y.append(y)
            pixels_x.append(x)
            sum_y += float(y)
            sum_x += float(x)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if source[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append(
            _ComponentSummary(
                count=len(pixels_y),
                sum_x=sum_x,
                sum_y=sum_y,
                pixels_y=tuple(pixels_y),
                pixels_x=tuple(pixels_x),
            )
        )
    return components


def _component_summary_mask(
    shape: tuple[int, int],
    component: _ComponentSummary,
) -> NDArray[np.bool_]:
    mask = np.zeros(shape, dtype=np.bool_)
    mask[
        np.asarray(component.pixels_y, dtype=np.int64),
        np.asarray(component.pixels_x, dtype=np.int64),
    ] = True
    return mask


def _mask_center_xy(mask: NDArray[np.bool_]) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def _target_from_component(
    name: str,
    component: NDArray[np.bool_],
    *,
    base_confidence: float,
    min_pixels: int,
) -> FaceAnchorTarget:
    x, y = _mask_center_xy(component)
    count = int(np.count_nonzero(component))
    evidence = min(1.0, count / max(float(min_pixels * 2), 1.0))
    confidence = _clamp01(base_confidence * (0.40 + 0.60 * evidence))
    return FaceAnchorTarget(
        name=name,
        x=float(x),
        y=float(y),
        confidence=confidence,
        pixel_count=count,
    )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
