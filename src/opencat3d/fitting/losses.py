"""Deterministic fitting losses used by the optimizer skeleton."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from opencat3d.camera.conventions import project_world_points
from opencat3d.schemas import PerspectiveCameraState
from opencat3d.template.cat_template import CatTemplate

FACE_DETAIL_REGION_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "muzzle",
    "left_ear_inner",
    "right_ear_inner",
)
FEATURE_ANCHOR_TARGET_NAMES = (
    "left_eye_center",
    "right_eye_center",
    "nose_tip",
    "muzzle_center",
    "mouth_line",
    "left_ear_inner",
    "right_ear_inner",
    "left_front_paw",
    "right_front_paw",
    "left_hind_paw",
    "right_hind_paw",
    "tail_base",
    "tail_tip",
)


def bbox_diagonal(bbox_xyxy: NDArray[np.floating] | list[float], eps: float = 1e-9) -> float:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,):
        raise ValueError("bbox_xyxy must have shape (4,)")
    width = max(float(bbox[2] - bbox[0]), 0.0)
    height = max(float(bbox[3] - bbox[1]), 0.0)
    return max(float(np.hypot(width, height)), eps)


def keypoint_reprojection_loss(
    projected_xy_depth: NDArray[np.floating],
    observed_xyc: NDArray[np.floating],
    bbox_xyxy: NDArray[np.floating] | list[float],
    *,
    sigma: float = 0.08,
    robust: bool = True,
) -> float:
    """Confidence-weighted normalized 2D keypoint reprojection loss."""

    projected = np.asarray(projected_xy_depth, dtype=np.float64)
    observed = np.asarray(observed_xyc, dtype=np.float64)
    if projected.shape[0] != observed.shape[0] or projected.shape[1] < 2 or observed.shape[1] != 3:
        raise ValueError("projected must be [J,>=2] and observed must be [J,3]")

    confidence = np.clip(observed[:, 2], 0.0, 1.0)
    visible = confidence > 0.0
    if not np.any(visible):
        return 0.0

    residual = np.linalg.norm(projected[:, :2] - observed[:, :2], axis=1)
    normalized = residual / bbox_diagonal(bbox_xyxy)
    values = geman_mcclure(normalized, sigma=sigma) if robust else normalized * normalized
    weight_sum = float(confidence[visible].sum())
    return float(np.sum(values[visible] * confidence[visible]) / max(weight_sum, 1e-12))


def soft_iou_loss(
    predicted_mask: NDArray[np.floating],
    observed_mask: NDArray[np.floating],
    *,
    eps: float = 1e-6,
) -> float:
    """Soft silhouette IoU loss, `1 - intersection / union`."""

    predicted = np.clip(np.asarray(predicted_mask, dtype=np.float64), 0.0, 1.0)
    observed = np.clip(np.asarray(observed_mask, dtype=np.float64), 0.0, 1.0)
    if predicted.shape != observed.shape:
        raise ValueError("predicted_mask and observed_mask must have matching shapes")
    intersection = float(np.sum(predicted * observed))
    union = float(np.sum(predicted + observed - predicted * observed))
    if union <= eps:
        return 0.0
    return float(1.0 - (intersection + eps) / (union + eps))


def geman_mcclure(values: NDArray[np.floating], *, sigma: float = 1.0) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    squared = arr * arr
    sigma_squared = sigma * sigma
    return squared / (squared + sigma_squared)


def region_detail_vertex_weights(
    template: CatTemplate,
    *,
    region_names: tuple[str, ...] = FACE_DETAIL_REGION_NAMES,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Return per-vertex weights for region-targeted geometric detail fitting."""

    if not region_names:
        raise ValueError("region_names must not be empty")
    weights = np.zeros(template.vertices_canonical.shape[0], dtype=np.float64)
    for name in region_names:
        region = template.region(name)
        region_weight = float(region.detail_weight) * (1.0 + region.texture_priority / 100.0)
        weights[region.vertex_indices] = np.maximum(weights[region.vertex_indices], region_weight)
    if normalize:
        max_weight = float(np.max(weights))
        if max_weight > 0.0:
            weights = weights / max_weight
    return weights


def face_region_detail_loss(
    vertices: NDArray[np.floating],
    target_vertices: NDArray[np.floating],
    template: CatTemplate,
    *,
    region_names: tuple[str, ...] = FACE_DETAIL_REGION_NAMES,
    vertex_confidence: NDArray[np.floating] | None = None,
    robust_sigma: float = 0.03,
) -> float:
    """Weighted canonical geometry loss for face, muzzle, nose, and ear detail."""

    current = np.asarray(vertices, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    expected_shape = template.vertices_canonical.shape
    if current.shape != expected_shape or target.shape != expected_shape:
        raise ValueError(f"vertices and target_vertices must have shape {expected_shape}")
    if robust_sigma <= 0.0:
        raise ValueError("robust_sigma must be positive")

    weights = region_detail_vertex_weights(
        template,
        region_names=region_names,
        normalize=False,
    )
    if vertex_confidence is not None:
        confidence = np.asarray(vertex_confidence, dtype=np.float64)
        if confidence.shape != (expected_shape[0],):
            raise ValueError(f"vertex_confidence must have shape ({expected_shape[0]},)")
        weights = weights * np.clip(confidence, 0.0, 1.0)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return 0.0

    residual = np.linalg.norm(current - target, axis=1)
    robust_values = geman_mcclure(residual, sigma=robust_sigma)
    return float(np.sum(weights * robust_values) / weight_sum)


def feature_anchor_observation_weights(
    template: CatTemplate,
    *,
    anchor_names: tuple[str, ...] = FEATURE_ANCHOR_TARGET_NAMES,
    normalize: bool = True,
) -> dict[str, float]:
    """Return deterministic region-priority weights for feature-anchor observations."""

    anchors = _select_feature_anchors(template, anchor_names)
    weights: dict[str, float] = {}
    for anchor in anchors:
        region = template.region(anchor.region_name)
        weights[anchor.name] = float(
            region.detail_weight * (1.0 + region.texture_priority / 100.0)
        )
    if normalize and weights:
        max_weight = max(weights.values())
        if max_weight > 0.0:
            weights = {name: float(weight / max_weight) for name, weight in weights.items()}
    return weights


def feature_anchor_world_points(
    template: CatTemplate,
    vertices: NDArray[np.floating],
    *,
    anchor_names: tuple[str, ...] = FEATURE_ANCHOR_TARGET_NAMES,
    root_translation: NDArray[np.floating] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Return current fixed-topology support-centroid points for template anchors."""

    current = np.asarray(vertices, dtype=np.float64)
    expected_shape = template.vertices_canonical.shape
    if current.shape != expected_shape:
        raise ValueError(f"vertices must have shape {expected_shape}")
    translation = (
        np.zeros(3, dtype=np.float64)
        if root_translation is None
        else np.asarray(root_translation, dtype=np.float64)
    )
    if translation.shape != (3,):
        raise ValueError("root_translation must have shape (3,)")

    points: dict[str, NDArray[np.float64]] = {}
    for anchor in _select_feature_anchors(template, anchor_names):
        support = current[anchor.support_vertex_indices]
        points[anchor.name] = np.mean(support, axis=0) + translation
    return points


def feature_anchor_projection_loss(
    template: CatTemplate,
    vertices: NDArray[np.floating],
    camera: PerspectiveCameraState,
    observed_xyc_by_name: Mapping[str, Sequence[float]],
    bbox_xyxy: NDArray[np.floating] | list[float],
    *,
    root_translation: NDArray[np.floating] | None = None,
    anchor_names: tuple[str, ...] = FEATURE_ANCHOR_TARGET_NAMES,
    sigma: float = 0.08,
    robust: bool = True,
) -> float:
    """Region-weighted normalized 2D loss for observed template feature anchors."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if not observed_xyc_by_name:
        return 0.0

    known_names = {anchor.name for anchor in template.feature_anchors}
    unknown_observed = sorted(set(observed_xyc_by_name).difference(known_names))
    if unknown_observed:
        raise ValueError(f"unknown feature anchor observation(s): {unknown_observed}")

    points_by_name = feature_anchor_world_points(
        template,
        vertices,
        anchor_names=anchor_names,
        root_translation=root_translation,
    )
    weights_by_name = feature_anchor_observation_weights(
        template,
        anchor_names=anchor_names,
        normalize=False,
    )
    selected_names: list[str] = []
    observed_rows: list[NDArray[np.float64]] = []
    predicted_points: list[NDArray[np.float64]] = []
    weights: list[float] = []
    for name in anchor_names:
        observed_raw = observed_xyc_by_name.get(name)
        if observed_raw is None:
            continue
        observed = np.asarray(observed_raw, dtype=np.float64)
        if observed.shape != (3,):
            raise ValueError(f"observed feature anchor {name!r} must have shape (3,)")
        confidence = float(np.clip(observed[2], 0.0, 1.0))
        if confidence <= 0.0:
            continue
        selected_names.append(name)
        observed_rows.append(observed)
        predicted_points.append(points_by_name[name])
        weights.append(confidence * float(weights_by_name[name]))

    if not selected_names:
        return 0.0

    projected = project_world_points(np.asarray(predicted_points, dtype=np.float64), camera)
    observed = np.asarray(observed_rows, dtype=np.float64)
    anchor_weights = np.asarray(weights, dtype=np.float64)
    residual = np.linalg.norm(projected[:, :2] - observed[:, :2], axis=1)
    normalized = residual / bbox_diagonal(bbox_xyxy)
    values = geman_mcclure(normalized, sigma=sigma) if robust else normalized * normalized
    weight_sum = float(np.sum(anchor_weights))
    if weight_sum <= 0.0:
        return 0.0
    return float(np.sum(values * anchor_weights) / weight_sum)


def _select_feature_anchors(template: CatTemplate, anchor_names: tuple[str, ...]):
    if not anchor_names:
        raise ValueError("anchor_names must not be empty")
    anchors_by_name = {anchor.name: anchor for anchor in template.feature_anchors}
    missing = [name for name in anchor_names if name not in anchors_by_name]
    if missing:
        raise ValueError(f"unknown CatTemplate feature anchor(s): {missing}")
    return tuple(anchors_by_name[name] for name in anchor_names)
