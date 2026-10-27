"""Bounded fixed-topology geometry detail priors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opencat3d.template.cat_template import CatTemplate

FACE_DETAIL_DEFORMATION_VERSION = "face_detail_region_deformation_v1"
FACE_SURFACE_ANCHOR_VERSION = "face_surface_anchor_deformation_v1"


@dataclass(frozen=True)
class FaceDetailDeformationResult:
    vertices: NDArray[np.float64]
    manifest: dict[str, object]


@dataclass(frozen=True)
class FaceSurfaceAnchorGroup:
    name: str
    region_name: str
    role: str
    vertex_indices: NDArray[np.int64]


_REGION_DISPLACEMENTS: tuple[tuple[str, tuple[float, float, float], float], ...] = (
    ("left_eye", (0.0, 0.0, 1.0), 0.018),
    ("right_eye", (0.0, 0.0, 1.0), 0.018),
    ("nose", (0.0, -0.10, 1.0), 0.032),
    ("muzzle", (0.0, -0.12, 1.0), 0.016),
    ("left_ear_inner", (0.16, -0.18, 0.42), 0.014),
    ("right_ear_inner", (-0.16, -0.18, 0.42), 0.014),
)


@dataclass(frozen=True)
class _SurfaceAnchorSpec:
    name: str
    region_name: str
    role: str
    normalized_center: tuple[float, float, float]
    radii_fraction: tuple[float, float, float]
    axis: tuple[float, float, float]
    magnitude: float


_SURFACE_ANCHORS: tuple[_SurfaceAnchorSpec, ...] = (
    _SurfaceAnchorSpec(
        "left_eye_core",
        "left_eye",
        "eye_cornea_bulge",
        (0.50, 0.52, 0.82),
        (0.62, 0.62, 0.72),
        (0.0, 0.02, 1.0),
        0.020,
    ),
    _SurfaceAnchorSpec(
        "right_eye_core",
        "right_eye",
        "eye_cornea_bulge",
        (0.50, 0.52, 0.82),
        (0.62, 0.62, 0.72),
        (0.0, 0.02, 1.0),
        0.020,
    ),
    _SurfaceAnchorSpec(
        "left_eye_socket",
        "left_eye",
        "eye_socket_relief",
        (0.50, 0.48, 0.32),
        (0.70, 0.70, 0.48),
        (0.0, -0.04, -1.0),
        0.010,
    ),
    _SurfaceAnchorSpec(
        "right_eye_socket",
        "right_eye",
        "eye_socket_relief",
        (0.50, 0.48, 0.32),
        (0.70, 0.70, 0.48),
        (0.0, -0.04, -1.0),
        0.010,
    ),
    _SurfaceAnchorSpec(
        "nose_tip",
        "nose",
        "nose_leather_tip",
        (0.50, 0.48, 0.86),
        (0.70, 0.70, 0.64),
        (0.0, -0.08, 1.0),
        0.022,
    ),
    _SurfaceAnchorSpec(
        "left_muzzle_pad",
        "muzzle",
        "muzzle_pad_volume",
        (0.34, 0.40, 0.76),
        (0.34, 0.44, 0.42),
        (-0.18, -0.05, 1.0),
        0.012,
    ),
    _SurfaceAnchorSpec(
        "right_muzzle_pad",
        "muzzle",
        "muzzle_pad_volume",
        (0.66, 0.40, 0.76),
        (0.34, 0.44, 0.42),
        (0.18, -0.05, 1.0),
        0.012,
    ),
    _SurfaceAnchorSpec(
        "mouth_groove",
        "muzzle",
        "mouth_line_indent",
        (0.50, 0.28, 0.78),
        (0.26, 0.34, 0.36),
        (0.0, -0.28, -1.0),
        0.032,
    ),
    _SurfaceAnchorSpec(
        "left_ear_inner_bowl",
        "left_ear_inner",
        "ear_inner_bowl",
        (0.50, 0.50, 0.42),
        (0.70, 0.70, 0.60),
        (0.08, -0.16, 0.58),
        0.010,
    ),
    _SurfaceAnchorSpec(
        "right_ear_inner_bowl",
        "right_ear_inner",
        "ear_inner_bowl",
        (0.50, 0.50, 0.42),
        (0.70, 0.70, 0.60),
        (-0.08, -0.16, 0.58),
        0.010,
    ),
)


def apply_bounded_face_detail_deformation(
    template: CatTemplate,
    vertices: NDArray[np.floating],
    *,
    max_total_displacement: float = 0.040,
) -> FaceDetailDeformationResult:
    """Apply small deterministic face-region deformation without changing topology."""

    if max_total_displacement <= 0.0:
        raise ValueError("max_total_displacement must be positive")
    base = np.asarray(vertices, dtype=np.float64)
    if base.shape != template.vertices_canonical.shape:
        raise ValueError(f"vertices must have shape {template.vertices_canonical.shape}")

    detailed = base.copy()
    displacement = np.zeros_like(base, dtype=np.float64)
    anchor_displacement = np.zeros_like(base, dtype=np.float64)
    region_entries: list[dict[str, object]] = []
    for region_name, axis, magnitude in _REGION_DISPLACEMENTS:
        region = template.region(region_name)
        indices = region.vertex_indices
        weights = _region_falloff(base[indices])
        direction = _normalized_axis(axis)
        region_delta = weights[:, None] * direction[None, :] * float(magnitude)
        displacement[indices] += region_delta
        region_entries.append(
            {
                "name": region_name,
                "vertex_count": int(indices.size),
                "magnitude": float(magnitude),
                "axis": [float(value) for value in direction],
                "max_weight": float(np.max(weights)) if weights.size else 0.0,
            }
        )

    anchor_entries: list[dict[str, object]] = []
    for spec in _SURFACE_ANCHORS:
        region = template.region(spec.region_name)
        indices = region.vertex_indices
        weights = _surface_anchor_weights(
            base[indices],
            normalized_center=spec.normalized_center,
            radii_fraction=spec.radii_fraction,
        )
        direction = _normalized_axis(spec.axis)
        region_delta = weights[:, None] * direction[None, :] * float(spec.magnitude)
        displacement[indices] += region_delta
        anchor_displacement[indices] += region_delta
        active_weights = weights > 0.0
        anchor_entries.append(
            {
                "name": spec.name,
                "region": spec.region_name,
                "role": spec.role,
                "vertex_count": int(np.count_nonzero(active_weights)),
                "magnitude": float(spec.magnitude),
                "axis": [float(value) for value in direction],
                "max_weight": float(np.max(weights)) if weights.size else 0.0,
            }
        )

    norms = np.linalg.norm(displacement, axis=1)
    too_large = norms > max_total_displacement
    if np.any(too_large):
        capped_displacement = max_total_displacement * (1.0 - 1e-12)
        displacement[too_large] *= (capped_displacement / norms[too_large])[:, None]
        norms = np.linalg.norm(displacement, axis=1)
    anchor_norms = np.linalg.norm(anchor_displacement, axis=1)
    anchor_active = anchor_norms > 0.0
    detailed += displacement
    active = norms > 0.0
    manifest = {
        "applied": bool(np.any(active)),
        "source": FACE_DETAIL_DEFORMATION_VERSION,
        "surface_anchor_source": FACE_SURFACE_ANCHOR_VERSION,
        "surface_anchor_count": len(anchor_entries),
        "surface_anchor_vertex_count": int(np.count_nonzero(anchor_active)),
        "surface_anchor_max_displacement": (
            float(np.max(anchor_norms)) if anchor_norms.size else 0.0
        ),
        "vertex_count": int(base.shape[0]),
        "deformed_vertex_count": int(np.count_nonzero(active)),
        "max_displacement": float(np.max(norms)) if norms.size else 0.0,
        "mean_displacement": float(np.mean(norms[active])) if np.any(active) else 0.0,
        "max_total_displacement": float(max_total_displacement),
        "regions": region_entries,
        "surface_anchors": anchor_entries,
    }
    return FaceDetailDeformationResult(vertices=detailed, manifest=manifest)


def face_surface_anchor_vertex_groups(
    template: CatTemplate,
    vertices: NDArray[np.floating],
) -> tuple[FaceSurfaceAnchorGroup, ...]:
    """Return deterministic vertex groups for named face surface anchors."""

    base = np.asarray(vertices, dtype=np.float64)
    if base.shape != template.vertices_canonical.shape:
        raise ValueError(f"vertices must have shape {template.vertices_canonical.shape}")
    groups: list[FaceSurfaceAnchorGroup] = []
    for spec in _SURFACE_ANCHORS:
        region = template.region(spec.region_name)
        weights = _surface_anchor_weights(
            base[region.vertex_indices],
            normalized_center=spec.normalized_center,
            radii_fraction=spec.radii_fraction,
        )
        groups.append(
            FaceSurfaceAnchorGroup(
                name=spec.name,
                region_name=spec.region_name,
                role=spec.role,
                vertex_indices=region.vertex_indices[weights > 0.0],
            )
        )
    return tuple(groups)


def _region_falloff(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    if vertices.size == 0:
        return np.zeros(0, dtype=np.float64)
    center = np.mean(vertices, axis=0)
    distances = np.linalg.norm(vertices - center[None, :], axis=1)
    max_distance = max(float(np.max(distances)), 1e-9)
    normalized = distances / max_distance
    return np.clip(1.0 - 0.35 * normalized * normalized, 0.55, 1.0)


def _surface_anchor_weights(
    vertices: NDArray[np.float64],
    *,
    normalized_center: tuple[float, float, float],
    radii_fraction: tuple[float, float, float],
) -> NDArray[np.float64]:
    if vertices.size == 0:
        return np.zeros(0, dtype=np.float64)
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    extents = np.maximum(maxs - mins, 1e-9)
    center = mins + np.asarray(normalized_center, dtype=np.float64) * extents
    radii = np.maximum(np.asarray(radii_fraction, dtype=np.float64) * extents, 1e-9)
    normalized = (vertices - center[None, :]) / radii[None, :]
    distances = np.sum(normalized * normalized, axis=1)
    return np.clip(1.0 - distances, 0.0, 1.0)


def _normalized_axis(axis: tuple[float, float, float]) -> NDArray[np.float64]:
    arr = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("axis must be non-zero")
    return arr / norm
