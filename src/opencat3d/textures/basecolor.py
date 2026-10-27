"""Observed-only canonical basecolor texture projection."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from math import log1p, radians, tan
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from opencat3d.fitting.optimizer import (
    FitObservation,
    FitProblem,
    OptimizerConfig,
    _optimize_frame_translation,
)
from opencat3d.io.cache import sha256_file
from opencat3d.schemas import FrameObservationMeta, PerspectiveCameraState
from opencat3d.synthetic.renderer import render_buffers

DEFAULT_TEXTURE_SIZE = 256
RAW_FACE_PRIORITY_TEXTURE_SIZE = 512
RAW_FACE_TARGET_TEXTURE_SIZE = 1024
RAW_TEMPLATE_REGION_V2_TEXTURE_TARGET_SIZE = 2048
_DIRECT_PHOTO_BROAD_FACE_DETAIL_THRESHOLD = 0.45
_DIRECT_PHOTO_BROAD_FACE_RELAXED_DETAIL_THRESHOLD = 0.30
_DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_SUPPORT = 0.90
_DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_DENSITY = 0.75
_DIRECT_PHOTO_BROAD_FACE_MIN_QUALITY = 0.45
_STRICT_TEXTURE_ONLY_FACE_ROLLBACK_GROUPS = {
    "iris_pupil": (
        "left_eye_shadow",
        "right_eye_shadow",
        "left_eye_core",
        "right_eye_core",
        "left_eye_iris",
        "right_eye_iris",
        "left_eye_pupil",
        "right_eye_pupil",
    ),
    "nose_muzzle": (
        "left_muzzle",
        "right_muzzle",
        "left_whisker_pad",
        "right_whisker_pad",
        "left_whisker_root",
        "right_whisker_root",
        "nose",
        "mouth_line",
    ),
}
_STRICT_TEXTURE_ONLY_FACE_ROLLBACK_METRICS = {
    "iris_pupil": ("iris_to_pupil_rgb_distance",),
    "nose_muzzle": (
        "nose_to_muzzle_rgb_distance",
        "mouth_to_muzzle_luma_delta",
        "min_required_contrast",
    ),
}
_STRICT_TEXTURE_ONLY_FACE_ROLLBACK_EPSILON = 1e-9


@dataclass(frozen=True)
class BasecolorTextureResult:
    texture_dir: Path
    manifest_path: Path
    basecolor_path: Path
    confidence_path: Path
    source_view_id_path: Path
    observed_face_count: int
    unknown_face_count: int
    observed_texel_fraction: float
    width: int
    height: int


@dataclass(frozen=True)
class _TextureProjectionInputs:
    observations: tuple[FitObservation, ...]
    root_translations: NDArray[np.float64]
    images_by_frame: dict[str, NDArray[np.float32]]
    frame_metadata: tuple[dict[str, Any], ...]
    source_manifest_path: Path
    fit_frame_count: int
    supplemental_frame_count: int


@dataclass(frozen=True)
class _FaceFeaturePalette:
    source_frame_id: str
    source_score: float
    eye_rgb: NDArray[np.uint8]
    iris_rgb: NDArray[np.uint8]
    nose_rgb: NDArray[np.uint8]
    muzzle_rgb: NDArray[np.uint8]
    ear_inner_rgb: NDArray[np.uint8]
    mouth_rgb: NDArray[np.uint8]
    sample_counts: dict[str, int]
    source_patches: dict[str, NDArray[np.uint8]]
    source_patch_masks: dict[str, NDArray[np.uint8]]
    source_patch_mask_observed_counts: dict[str, int]
    source_patch_mask_densified_counts: dict[str, int]
    source_patch_mask_densification_gains: dict[str, float]
    source_patch_localization_candidate_counts: dict[str, int]
    source_patch_localization_selected_counts: dict[str, int]
    source_patch_localization_fractions: dict[str, float]
    source_patch_landmark_confidences: dict[str, float]
    source_patch_scores: dict[str, float]
    source_patch_frame_ids: dict[str, str]
    source_patch_frontal_face_confidences: dict[str, float]
    source_patch_view_weights: dict[str, float]
    source_patch_view_labels: dict[str, str]
    source_patch_view_weighted_ranks: dict[str, float]
    source_patch_feedback_components: dict[str, str]
    source_patch_feedback_scores: dict[str, float]
    source_patch_frontal_photo_priorities: dict[str, float] = field(default_factory=dict)
    source_patch_frontal_photo_priority_diagnostics: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    source_patch_direct_photo_candidates: dict[str, bool] = field(default_factory=dict)
    source_patch_observed_anchor_guided: dict[str, bool] = field(default_factory=dict)
    allowed_source_patch_sample_keys: tuple[str, ...] = ()
    source_patch_candidate_diagnostics: dict[str, list[dict[str, object]]] = field(
        default_factory=dict
    )
    pupil_rgb: NDArray[np.uint8] | None = None


_SOURCE_PATCH_SAMPLE_KEYS = (
    "left_eye",
    "right_eye",
    "eye",
    "iris",
    "left_iris",
    "right_iris",
    "left_pupil",
    "right_pupil",
    "nose",
    "muzzle",
    "ear_inner",
    "mouth",
)
_FACE_PRIORITY_SOURCE_PATCH_KEYS = frozenset(_SOURCE_PATCH_SAMPLE_KEYS)
_BROAD_FACE_SOURCE_PATCH_KEYS = frozenset({"muzzle", "nose", "mouth"})
_BACKGROUND_GREEN_GATED_SOURCE_PATCH_KEYS = frozenset(
    {
        "muzzle",
        "nose",
        "ear_inner",
        "mouth",
    }
)
_DIRECT_PHOTO_NOSE_MIN_COLOR_SCORE = 0.65
_DIRECT_PHOTO_EAR_INNER_MIN_QUALITY = 0.45
_DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_QUALITY = 0.35
_DIRECT_PHOTO_EAR_INNER_MIN_DETAIL = 0.70
_DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_DETAIL = 0.85
_DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_SUPPORT = 0.95
_DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_DENSITY = 0.85
_DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_VIEW_SCORE = 0.75
_DIRECT_PHOTO_IRIS_MIN_QUALITY = 0.45
_DIRECT_PHOTO_IRIS_RELAXED_MIN_QUALITY = 0.18
_DIRECT_PHOTO_IRIS_MIN_DETAIL = 0.25
_DIRECT_PHOTO_IRIS_RELAXED_MIN_DETAIL = 0.75
_DIRECT_PHOTO_IRIS_RELAXED_MIN_SUPPORT = 0.80
_DIRECT_PHOTO_IRIS_RELAXED_MIN_DENSITY = 0.60
_DIRECT_PHOTO_IRIS_RELAXED_MIN_VIEW_SCORE = 0.55
_DIRECT_PHOTO_PUPIL_MIN_QUALITY = 0.10
_DIRECT_PHOTO_PUPIL_MIN_DARKNESS = 0.55
_DIRECT_PHOTO_PUPIL_MIN_SUPPORT = 0.45
_DIRECT_PHOTO_PUPIL_MIN_DENSITY = 0.10
_DIRECT_PHOTO_PUPIL_MIN_VIEW_SCORE = 0.55
_SOURCE_PATCH_FEEDBACK_COMPONENT_PRIORITY = {
    "source_evidence_score": 5,
    "source_luma_score": 4,
    "confident_texel_score": 3,
    "texture_luma_score": 2,
    "unique_rgb_score": 1,
    "combined_score": 0,
}
_LANDMARK_ALIGNED_SOURCE_PATCH_KEYS = frozenset(
    {
        "left_eye",
        "right_eye",
        "eye",
        "iris",
        "left_iris",
        "right_iris",
        "left_pupil",
        "right_pupil",
        "nose",
        "muzzle",
        "mouth",
    }
)
_OBSERVED_FACE_ANCHOR_SOURCE_PATCH_VERSION = "observed_face_anchor_source_patch_v1"
_OBSERVED_FACE_ANCHOR_ALIASES: dict[str, tuple[str, ...]] = {
    "left_eye_center": ("left_eye_center", "left_eye"),
    "right_eye_center": ("right_eye_center", "right_eye"),
    "nose_tip": ("nose_tip", "nose"),
    "muzzle_center": ("muzzle_center", "muzzle"),
}
_BROAD_EYE_SOURCE_SHAPE_KEYS = frozenset({"left_eye", "right_eye", "eye"})
_SOURCE_PATCH_FEEDBACK_ALIASES = {
    "left_iris": "left_eye",
    "right_iris": "right_eye",
    "left_pupil": "left_eye",
    "right_pupil": "right_eye",
}
_COMPACT_EYE_SOURCE_SHAPE_KEYS = frozenset(
    {
        "left_iris",
        "right_iris",
        "left_pupil",
        "right_pupil",
    }
)
_IRIS_SOURCE_LUMA_CORE_KEYS = frozenset({"left_iris", "right_iris"})
_PUPIL_SOURCE_DARK_CORE_KEYS = frozenset({"left_pupil", "right_pupil"})
_IRIS_MAX_FEATHERED_EDGE_TO_CORE_RATIO = 1.25
_IRIS_COMPACT_CORE_TEXEL_COUNT = 32
_IRIS_COMPACT_MAX_FEATHERED_EDGE_TO_CORE_RATIO = 0.75
_IRIS_EDGE_CORE_RATIO_ENV = "OPENCAT3D_IRIS_EDGE_CORE_RATIO"
_FACE_TEXTURE_BAKE_REGION_FEATURES = {
    "left_eye": ("left_eye_shadow", "left_eye_core", "left_eye_iris", "left_eye_pupil"),
    "right_eye": ("right_eye_shadow", "right_eye_core", "right_eye_iris", "right_eye_pupil"),
    "nose": ("nose",),
    "muzzle": (
        "left_muzzle",
        "right_muzzle",
        "left_whisker_pad",
        "right_whisker_pad",
        "left_whisker_root",
        "right_whisker_root",
        "mouth_line",
    ),
    "left_ear_inner": ("left_ear_inner",),
    "right_ear_inner": ("right_ear_inner",),
}
_FACE_REGION_PROJECTION_PROTECTION_VERSION = "supplemental_face_region_protection_v1"
_TEXTURE_ONLY_SOURCE_PATCH_PRIORITY_VERSION = "texture_only_source_patch_priority_v1"
_TEXTURE_ONLY_SOURCE_PATCH_BASE_VIEW_SCORE = 0.55
_TEXTURE_ONLY_SOURCE_PATCH_MAX_VIEW_SCORE = 0.72
_TEXTURE_ONLY_SOURCE_PATCH_PREFLIGHT_GAIN_SPAN = 0.50
_TEXTURE_ONLY_SOURCE_PATCH_PREFLIGHT_GAIN = 0.22
_TEXTURE_ONLY_SOURCE_PATCH_BROAD_FACE_RELAXED_MIN_DENSITY = 0.70
_TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_KEYS = frozenset(
    {
        "left_eye",
        "right_eye",
        "eye",
        "left_iris",
        "right_iris",
        "left_pupil",
        "right_pupil",
    }
)
_TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_BASE = 0.68
_TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_PRIORITY_GAIN = 0.30
_TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_DETAIL_GAIN = 0.06
_TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_QUALITY_GAIN = 0.03
_TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_MAX = 0.94
_TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_MIN_SUPPORT = 0.90
_TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_MIN_DENSITY = 0.75
_DEFAULT_PROTECTED_FACE_REGION_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "muzzle",
    "left_ear_inner",
    "right_ear_inner",
)
_FACE_TEXTURE_BAKE_MIN_SCORE = 0.70
_FACE_TEXTURE_BAKE_MIN_CONFIDENT_TEXEL_FRACTION = 0.05
_FACE_TEXTURE_BAKE_MIN_UNIQUE_RGB_COUNT = 8
_FACE_TEXTURE_BAKE_MIN_TEXTURE_LUMA_STD = 0.010
_FACE_TEXTURE_BAKE_MIN_SOURCE_LUMA_STD = 0.006
_FACE_TEXTURE_BAKE_TARGET_CONFIDENT_TEXEL_FRACTION = 0.12
_FACE_TEXTURE_BAKE_TARGET_UNIQUE_RGB_COUNT = 64
_FACE_TEXTURE_BAKE_TARGET_TEXTURE_LUMA_STD = 0.04
_FACE_TEXTURE_BAKE_TARGET_SOURCE_LUMA_STD = 0.02
_MIN_SOURCE_PATCH_QUALITY_SCORE = 0.04
_MIN_SOURCE_PATCH_FUSION_WEIGHT = 0.35
_MAX_SOURCE_PATCH_FUSION_WEIGHT = 0.96
_ZERO_PRIORITY_DIRECT_PHOTO_RANK_PENALTY = 0.96
_EYE_SOURCE_DETAIL_REINFORCEMENT_FEATURES = frozenset(
    {
        "left_eye_iris",
        "right_eye_iris",
        "left_eye_pupil",
        "right_eye_pupil",
    }
)
_EYE_SOURCE_DETAIL_REINFORCEMENT_MAX_ALPHA = {
    "left_eye_iris": 0.60,
    "right_eye_iris": 0.60,
    "left_eye_pupil": 0.78,
    "right_eye_pupil": 0.78,
}
_EYE_SOURCE_DETAIL_REINFORCEMENT_MIN_ALPHA = {
    "left_eye_iris": 0.18,
    "right_eye_iris": 0.18,
}
_SOURCE_PATCH_FEATHER_PROFILES = {
    "left_eye": {"edge_width": 0.18, "x_scale": 1.08, "y_scale": 0.86},
    "right_eye": {"edge_width": 0.18, "x_scale": 1.08, "y_scale": 0.86},
    "eye": {"edge_width": 0.18, "x_scale": 1.08, "y_scale": 0.86},
    "iris": {"edge_width": 0.14, "x_scale": 1.04, "y_scale": 0.92},
    "left_iris": {"edge_width": 0.12, "x_scale": 1.02, "y_scale": 0.90},
    "right_iris": {"edge_width": 0.12, "x_scale": 1.02, "y_scale": 0.90},
    "left_pupil": {"edge_width": 0.10, "x_scale": 0.96, "y_scale": 0.92},
    "right_pupil": {"edge_width": 0.10, "x_scale": 0.96, "y_scale": 0.92},
    "nose": {"edge_width": 0.18, "x_scale": 1.02, "y_scale": 0.88},
    "muzzle": {"edge_width": 0.22, "x_scale": 1.10, "y_scale": 0.88},
    "ear_inner": {"edge_width": 0.20, "x_scale": 1.00, "y_scale": 0.90},
    "mouth": {"edge_width": 0.12, "x_scale": 1.12, "y_scale": 0.82},
}
_SOURCE_PATCH_FOOTPRINT_PROFILES = {
    "left_eye": {"target_density": 0.055, "min_scale": 0.50, "max_scale": 0.82},
    "right_eye": {"target_density": 0.055, "min_scale": 0.50, "max_scale": 0.82},
    "eye": {"target_density": 0.055, "min_scale": 0.50, "max_scale": 0.82},
    "iris": {"target_density": 0.070, "min_scale": 0.48, "max_scale": 0.80},
    "left_iris": {"target_density": 0.075, "min_scale": 0.46, "max_scale": 0.76},
    "right_iris": {"target_density": 0.075, "min_scale": 0.46, "max_scale": 0.76},
    "left_pupil": {"target_density": 0.090, "min_scale": 0.42, "max_scale": 0.70},
    "right_pupil": {"target_density": 0.090, "min_scale": 0.42, "max_scale": 0.70},
    "nose": {"target_density": 0.120, "min_scale": 0.68, "max_scale": 0.94},
    "muzzle": {"target_density": 0.120, "min_scale": 0.70, "max_scale": 0.92},
    "ear_inner": {"target_density": 0.035, "min_scale": 0.48, "max_scale": 0.80},
    "mouth": {"target_density": 0.050, "min_scale": 0.58, "max_scale": 0.90},
}
_SOURCE_PATCH_LANDMARK_DESTINATION_SCALES = {
    "left_eye": 0.58,
    "right_eye": 0.58,
    "eye": 0.58,
    "iris": 0.62,
    "left_iris": 0.50,
    "right_iris": 0.50,
    "left_pupil": 0.54,
    "right_pupil": 0.54,
    "nose": 0.72,
    "mouth": 0.68,
    "muzzle": 0.88,
}
_SOURCE_PATCH_SHAPE_PROFILES = {
    "left_eye": {"dilation_fraction": 0.20, "min_radius": 3, "max_radius": 18},
    "right_eye": {"dilation_fraction": 0.20, "min_radius": 3, "max_radius": 18},
    "eye": {"dilation_fraction": 0.20, "min_radius": 3, "max_radius": 18},
    "iris": {"dilation_fraction": 0.16, "min_radius": 2, "max_radius": 14},
    "left_iris": {"dilation_fraction": 0.12, "min_radius": 2, "max_radius": 10},
    "right_iris": {"dilation_fraction": 0.12, "min_radius": 2, "max_radius": 10},
    "left_pupil": {"dilation_fraction": 0.10, "min_radius": 1, "max_radius": 8},
    "right_pupil": {"dilation_fraction": 0.10, "min_radius": 1, "max_radius": 8},
    "nose": {"dilation_fraction": 0.16, "min_radius": 3, "max_radius": 18},
    "muzzle": {"dilation_fraction": 0.14, "min_radius": 4, "max_radius": 20},
    "ear_inner": {"dilation_fraction": 0.18, "min_radius": 3, "max_radius": 16},
    "mouth": {"dilation_fraction": 0.12, "min_radius": 2, "max_radius": 10},
}
_SOURCE_SHAPE_MASK_EXACT_MAX_PIXELS = 65_536
_SOURCE_PATCH_CROP_PROFILES = {
    "left_eye": {"padding_fraction": 0.28, "min_padding": 3},
    "right_eye": {"padding_fraction": 0.28, "min_padding": 3},
    "eye": {"padding_fraction": 0.28, "min_padding": 3},
    "iris": {"padding_fraction": 0.22, "min_padding": 2},
    "left_iris": {"padding_fraction": 0.20, "min_padding": 2},
    "right_iris": {"padding_fraction": 0.20, "min_padding": 2},
    "left_pupil": {"padding_fraction": 0.18, "min_padding": 1},
    "right_pupil": {"padding_fraction": 0.18, "min_padding": 1},
    "nose": {"padding_fraction": 0.30, "min_padding": 3},
    "muzzle": {"padding_fraction": 0.32, "min_padding": 4},
    "ear_inner": {"padding_fraction": 0.30, "min_padding": 3},
    "mouth": {"padding_fraction": 0.35, "min_padding": 2},
}
_FACE_ANATOMICAL_UV_ANCHOR_VERSION = "face_anatomical_uv_anchor_v1"
_FACE_FEATURE_ANATOMICAL_UV_ANCHORS = {
    "left_ear_inner": (0.292, 0.955),
    "right_ear_inner": (0.708, 0.955),
    "left_muzzle": (0.430, 0.855),
    "right_muzzle": (0.570, 0.855),
    "left_whisker_pad": (0.382, 0.866),
    "right_whisker_pad": (0.618, 0.866),
    "left_whisker_root": (0.410, 0.842),
    "right_whisker_root": (0.590, 0.842),
    "left_eye_shadow": (0.375, 0.927),
    "right_eye_shadow": (0.625, 0.927),
    "left_eye_core": (0.375, 0.930),
    "right_eye_core": (0.625, 0.930),
    "left_eye_iris": (0.375, 0.932),
    "right_eye_iris": (0.625, 0.932),
    "left_eye_pupil": (0.375, 0.934),
    "right_eye_pupil": (0.625, 0.934),
    "nose": (0.500, 0.885),
    "mouth_line": (0.500, 0.845),
}
_WHISKER_PAD_FEATURE_NAMES = frozenset(
    {
        "left_whisker_pad",
        "right_whisker_pad",
        "left_whisker_root",
        "right_whisker_root",
    }
)


@dataclass(frozen=True)
class _PatchPaintStats:
    painted_texel_count: int
    weighted_texel_count: float
    core_texel_count: int
    feathered_edge_texel_count: int
    mean_alpha: float
    shape_mask_texel_count: int
    shape_alpha_mean: float
    applied_mask: NDArray[np.bool_]
    edge_clamp_removed_texel_count: int = 0
    edge_clamp_max_ratio: float = 0.0


@dataclass(frozen=True)
class _RawTextureViewQuality:
    label: str
    frontal_face_confidence: float
    view_weight: float


@dataclass(frozen=True)
class _PatchFootprint:
    radius_uv: tuple[float, float]
    density: float
    radius_scale: float
    area_scale: float
    x_scale: float
    y_scale: float
    landmark_destination_scale: float


@dataclass(frozen=True)
class _SourcePatchCrop:
    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    observed_mask_texel_count: int
    densified_mask_texel_count: int
    densification_gain: float


def write_observed_basecolor_texture(
    project_dir: Path,
    problem: FitProblem,
    *,
    mode: str,
    beta: NDArray[np.floating],
    root_translations: NDArray[np.floating],
    texture_size: int = DEFAULT_TEXTURE_SIZE,
    seed_texture_dir: Path | None = None,
    force: bool = False,
) -> BasecolorTextureResult:
    """Project observed image colors onto the fixed-topology UV texture."""

    project_dir = Path(project_dir).resolve()
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported texture projection mode: {mode}")
    if texture_size <= 0:
        raise ValueError("texture_size must be positive")
    texture_target_report = _valid_face_texture_target_report(
        project_dir,
        mode=mode,
    )
    translations = np.asarray(root_translations, dtype=np.float64)
    if translations.shape != (len(problem.observations), 3):
        raise ValueError("root_translations must have shape [num_frames,3]")

    texture_dir = project_dir / "outputs" / "textures"
    manifest_path = texture_dir / "texture_manifest.json"
    basecolor_path = texture_dir / "basecolor.png"
    confidence_path = texture_dir / "basecolor_confidence.png"
    source_view_id_path = texture_dir / "source_view_id.png"
    existing_paths = (basecolor_path, confidence_path, source_view_id_path)
    if not force and all(path.exists() for path in existing_paths) and manifest_path.exists():
        return _read_existing_basecolor_result(
            texture_dir=texture_dir,
            manifest_path=manifest_path,
            basecolor_path=basecolor_path,
            confidence_path=confidence_path,
            source_view_id_path=source_view_id_path,
        )
    for path in (basecolor_path, confidence_path, source_view_id_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass force=True to overwrite")

    vertices = problem.template.shaped_vertices(np.asarray(beta, dtype=np.float64))
    projection_inputs = _texture_projection_inputs(
        project_dir,
        problem,
        mode=mode,
        beta=beta,
        vertices=vertices,
        root_translations=translations,
    )
    images_by_frame = projection_inputs.images_by_frame
    projection_observations = projection_inputs.observations
    projection_translations = projection_inputs.root_translations
    foreground_fallback_color = _foreground_fallback_color(
        projection_observations,
        images_by_frame,
    )
    face_count = int(problem.template.faces.shape[0])
    rgb_sum = np.zeros((face_count, 3), dtype=np.float64)
    pixel_count = np.zeros(face_count, dtype=np.int64)
    source_pixel_counts = np.zeros((face_count, len(projection_observations)), dtype=np.int64)
    face_projection_protection = _face_region_projection_protection_report(
        problem,
        frame_metadata=projection_inputs.frame_metadata,
    )
    protected_face_mask = np.asarray(
        face_projection_protection["protected_face_mask"],
        dtype=np.bool_,
    )
    protected_projection_frame_count = 0
    protected_projection_excluded_sample_count = 0

    for view_index, (observation, translation) in enumerate(
        zip(projection_observations, projection_translations, strict=True)
    ):
        image = images_by_frame[observation.frame_id]
        if image.shape[:2] != observation.hard_mask.shape:
            raise ValueError("observation image and mask dimensions must match")
        buffers = render_buffers(
            vertices + translation[None, :],
            problem.template.faces,
            observation.camera,
        )
        observed = np.asarray(observation.hard_mask, dtype=np.float32) >= 0.5
        valid = (buffers.face_index >= 0) & observed
        metadata = (
            projection_inputs.frame_metadata[view_index]
            if view_index < len(projection_inputs.frame_metadata)
            else {"frame_id": observation.frame_id}
        )
        if _exclude_protected_face_projection(metadata) and np.any(protected_face_mask):
            protected_projection_frame_count += 1
            protected_pixels = _protected_projection_pixel_mask(
                buffers.face_index,
                valid_mask=valid,
                protected_face_mask=protected_face_mask,
            )
            protected_projection_excluded_sample_count += int(np.count_nonzero(protected_pixels))
            valid &= ~protected_pixels
        _accumulate_face_color_samples(
            rgb_sum,
            pixel_count,
            source_pixel_counts,
            face_index_buffer=buffers.face_index,
            image=image,
            valid_mask=valid,
            view_index=view_index,
        )

    observed_faces = pixel_count > 0
    face_colors = np.zeros((face_count, 3), dtype=np.float64)
    face_colors[observed_faces] = rgb_sum[observed_faces] / pixel_count[observed_faces, None]
    repaired_face_count = 0
    if mode == "raw":
        face_colors, repaired_face_count = _repair_dark_raw_face_colors(
            face_colors,
            observed_faces,
            foreground_fallback_color,
        )
    best_source_view = np.argmax(source_pixel_counts, axis=1).astype(np.int64)

    basecolor, confidence, source_view = _rasterize_face_colors_to_uv(
        problem,
        face_colors,
        observed_faces,
        best_source_view,
        foreground_fallback_color=foreground_fallback_color,
        texture_size=texture_size,
    )
    facial_features = _empty_facial_feature_manifest()
    if mode == "raw":
        source_patch_dir = texture_dir / "face_source_patches"
        _prepare_face_source_patch_dir(source_patch_dir, force=force)
        source_patch_feedback = _source_patch_feedback_from_previous_diagnostics(
            project_dir,
            mode=mode,
        )
        facial_features = _paint_best_raw_facial_feature_anchors(
            basecolor,
            confidence,
            source_view,
            problem=problem,
            observations=projection_observations,
            images_by_frame=images_by_frame,
            frame_metadata=projection_inputs.frame_metadata,
            foreground_fallback_color=foreground_fallback_color,
            source_patch_dir=source_patch_dir,
            source_patch_feedback=source_patch_feedback,
        )
    basecolor_seed = _apply_basecolor_seed(
        basecolor,
        confidence,
        source_view,
        seed_texture_dir=seed_texture_dir,
    )
    face_feature_rollback = _apply_texture_only_face_feature_rollback(
        basecolor,
        confidence,
        source_view,
        facial_features,
        seed_texture_dir=seed_texture_dir,
    )
    basecolor_seed["strict_face_feature_rollback"] = face_feature_rollback
    texture_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(basecolor).save(basecolor_path)
    Image.fromarray(confidence).save(confidence_path)
    Image.fromarray(source_view).save(source_view_id_path)

    observed_texel_fraction = float(np.count_nonzero(confidence > 0) / confidence.size)
    observed_face_count = int(np.count_nonzero(observed_faces))
    unknown_face_count = int(face_count - observed_face_count)
    _update_texture_manifest(
        manifest_path,
        mode=mode,
        basecolor_path=basecolor_path,
        confidence_path=confidence_path,
        source_view_id_path=source_view_id_path,
        texture_size=texture_size,
        observed_face_count=observed_face_count,
        unknown_face_count=unknown_face_count,
        observed_texel_fraction=observed_texel_fraction,
        foreground_fallback_color=foreground_fallback_color,
        repaired_face_count=repaired_face_count,
        facial_features=facial_features,
        face_uv_regions=problem.template.face_uv_region_manifest(),
        texture_target_report=texture_target_report,
        projection_inputs=projection_inputs,
        basecolor_seed=basecolor_seed,
        face_region_projection_protection={
            key: value
            for key, value in face_projection_protection.items()
            if key != "protected_face_mask"
        }
        | {
            "protected_projection_frame_count": int(protected_projection_frame_count),
            "protected_projection_excluded_sample_count": int(
                protected_projection_excluded_sample_count
            ),
        },
    )
    return BasecolorTextureResult(
        texture_dir=texture_dir,
        manifest_path=manifest_path,
        basecolor_path=basecolor_path,
        confidence_path=confidence_path,
        source_view_id_path=source_view_id_path,
        observed_face_count=observed_face_count,
        unknown_face_count=unknown_face_count,
        observed_texel_fraction=observed_texel_fraction,
        width=texture_size,
        height=texture_size,
    )


def _apply_basecolor_seed(
    basecolor: NDArray[np.uint8],
    confidence: NDArray[np.uint8],
    source_view: NDArray[np.uint8],
    *,
    seed_texture_dir: Path | None,
) -> dict[str, object]:
    """Fill still-unknown texels from a previous observed baked texture."""

    if seed_texture_dir is None:
        return {"applied": False, "reason": "not_requested"}
    seed_texture_dir = Path(seed_texture_dir)
    seed_basecolor_path = seed_texture_dir / "basecolor.png"
    seed_confidence_path = seed_texture_dir / "basecolor_confidence.png"
    seed_source_view_path = seed_texture_dir / "source_view_id.png"
    required_paths = (seed_basecolor_path, seed_confidence_path, seed_source_view_path)
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        return {
            "applied": False,
            "reason": "missing_seed_texture",
            "seed_texture_dir": str(seed_texture_dir),
            "missing_paths": missing_paths,
        }

    seed_basecolor = np.asarray(Image.open(seed_basecolor_path).convert("RGB"), dtype=np.uint8)
    seed_confidence = np.asarray(Image.open(seed_confidence_path).convert("L"), dtype=np.uint8)
    seed_source_view = np.asarray(Image.open(seed_source_view_path).convert("L"), dtype=np.uint8)
    expected_shape = basecolor.shape[:2]
    if seed_basecolor.shape[:2] != expected_shape or seed_confidence.shape != expected_shape:
        return {
            "applied": False,
            "reason": "seed_texture_size_mismatch",
            "seed_texture_dir": str(seed_texture_dir),
            "seed_basecolor_size": [
                int(seed_basecolor.shape[1]),
                int(seed_basecolor.shape[0]),
            ],
            "seed_confidence_size": [
                int(seed_confidence.shape[1]),
                int(seed_confidence.shape[0]),
            ],
            "target_size": [int(expected_shape[1]), int(expected_shape[0])],
        }
    if seed_source_view.shape != expected_shape:
        return {
            "applied": False,
            "reason": "seed_source_view_size_mismatch",
            "seed_texture_dir": str(seed_texture_dir),
            "seed_source_view_size": [
                int(seed_source_view.shape[1]),
                int(seed_source_view.shape[0]),
            ],
            "target_size": [int(expected_shape[1]), int(expected_shape[0])],
        }

    unknown_mask = confidence == 0
    seed_observed_mask = seed_confidence > 0
    fill_mask = unknown_mask & seed_observed_mask
    filled_texel_count = int(np.count_nonzero(fill_mask))
    if filled_texel_count > 0:
        basecolor[fill_mask] = seed_basecolor[fill_mask]
        confidence[fill_mask] = seed_confidence[fill_mask]
        source_view[fill_mask] = seed_source_view[fill_mask]
    return {
        "applied": True,
        "version": "baked_basecolor_seed_v1",
        "seed_texture_dir": str(seed_texture_dir),
        "basecolor_path": str(seed_basecolor_path),
        "confidence_path": str(seed_confidence_path),
        "source_view_id_path": str(seed_source_view_path),
        "seed_observed_texel_count": int(np.count_nonzero(seed_observed_mask)),
        "candidate_unknown_texel_count_before_seed": int(np.count_nonzero(unknown_mask)),
        "filled_texel_count": filled_texel_count,
        "target_texel_count": int(confidence.size),
    }


def _apply_texture_only_face_feature_rollback(
    basecolor: NDArray[np.uint8],
    confidence: NDArray[np.uint8],
    source_view: NDArray[np.uint8],
    facial_features: dict[str, object],
    *,
    seed_texture_dir: Path | None,
) -> dict[str, object]:
    """Restore protected face texels from the seed when a texture-only bake regresses."""

    if seed_texture_dir is None:
        return {"applied": False, "reason": "not_requested"}
    if not facial_features.get("applied", False):
        return {"applied": False, "reason": "candidate_has_no_facial_features"}

    seed_texture_dir = Path(seed_texture_dir)
    seed_basecolor_path = seed_texture_dir / "basecolor.png"
    seed_confidence_path = seed_texture_dir / "basecolor_confidence.png"
    seed_source_view_path = seed_texture_dir / "source_view_id.png"
    seed_manifest_path = seed_texture_dir / "texture_manifest.json"
    required_paths = (
        seed_basecolor_path,
        seed_confidence_path,
        seed_source_view_path,
        seed_manifest_path,
    )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        return {
            "applied": False,
            "reason": "missing_seed_face_feature_artifacts",
            "seed_texture_dir": str(seed_texture_dir),
            "missing_paths": missing_paths,
        }

    seed_basecolor = np.asarray(Image.open(seed_basecolor_path).convert("RGB"), dtype=np.uint8)
    seed_confidence = np.asarray(Image.open(seed_confidence_path).convert("L"), dtype=np.uint8)
    seed_source_view = np.asarray(Image.open(seed_source_view_path).convert("L"), dtype=np.uint8)
    expected_shape = basecolor.shape[:2]
    if (
        seed_basecolor.shape[:2] != expected_shape
        or seed_confidence.shape != expected_shape
        or seed_source_view.shape != expected_shape
    ):
        return {
            "applied": False,
            "reason": "seed_texture_size_mismatch",
            "seed_texture_dir": str(seed_texture_dir),
            "target_size": [int(expected_shape[1]), int(expected_shape[0])],
            "seed_basecolor_size": [
                int(seed_basecolor.shape[1]),
                int(seed_basecolor.shape[0]),
            ],
            "seed_confidence_size": [
                int(seed_confidence.shape[1]),
                int(seed_confidence.shape[0]),
            ],
            "seed_source_view_size": [
                int(seed_source_view.shape[1]),
                int(seed_source_view.shape[0]),
            ],
        }

    try:
        seed_manifest_raw = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "applied": False,
            "reason": "invalid_seed_texture_manifest",
            "seed_texture_dir": str(seed_texture_dir),
            "error": str(exc),
        }
    if not isinstance(seed_manifest_raw, dict):
        return {
            "applied": False,
            "reason": "invalid_seed_texture_manifest",
            "seed_texture_dir": str(seed_texture_dir),
        }
    seed_facial_features = seed_manifest_raw.get("facial_features", {})
    if not isinstance(seed_facial_features, dict):
        return {
            "applied": False,
            "reason": "seed_has_no_facial_features",
            "seed_texture_dir": str(seed_texture_dir),
        }

    candidate_features = _facial_feature_list(facial_features)
    seed_features = _facial_feature_list(seed_facial_features)
    if not candidate_features or not seed_features:
        return {
            "applied": False,
            "reason": "missing_feature_entries",
            "seed_texture_dir": str(seed_texture_dir),
            "candidate_feature_count": int(len(candidate_features)),
            "seed_feature_count": int(len(seed_features)),
        }

    candidate_contrast = _facial_feature_contrast(candidate_features)
    seed_contrast = _facial_feature_contrast(seed_features)
    rollback_groups = _texture_only_face_feature_rollback_groups(
        candidate_contrast,
        seed_contrast,
    )
    if not rollback_groups:
        facial_features["strict_texture_only_face_feature_rollback"] = {
            "applied": False,
            "reason": "face_feature_contrast_not_regressed",
            "candidate_contrast": candidate_contrast,
            "seed_contrast": seed_contrast,
        }
        return facial_features["strict_texture_only_face_feature_rollback"]

    features_by_name = {str(feature.get("name", "")): feature for feature in candidate_features}
    seed_features_by_name = {str(feature.get("name", "")): feature for feature in seed_features}
    rolled_back_features: list[str] = []
    skipped_features: list[str] = []
    rolled_back_texels = np.zeros(expected_shape, dtype=np.bool_)
    texture_size = int(expected_shape[0])

    for group_name in rollback_groups:
        for feature_name in _STRICT_TEXTURE_ONLY_FACE_ROLLBACK_GROUPS[group_name]:
            seed_feature = seed_features_by_name.get(feature_name)
            candidate_feature = features_by_name.get(feature_name)
            if seed_feature is None:
                skipped_features.append(feature_name)
                continue
            support_feature = candidate_feature if candidate_feature is not None else seed_feature
            mask = _feature_texture_mask(support_feature, texture_size=texture_size)
            copy_mask = mask & (seed_confidence > 0)
            if not np.any(copy_mask):
                skipped_features.append(feature_name)
                continue
            basecolor[copy_mask] = seed_basecolor[copy_mask]
            confidence[copy_mask] = seed_confidence[copy_mask]
            source_view[copy_mask] = seed_source_view[copy_mask]
            rolled_back_texels |= copy_mask
            restored_feature = dict(seed_feature)
            restored_feature["texture_only_face_feature_rollback"] = True
            restored_feature["texture_only_face_feature_rollback_source"] = (
                "baseline_seed_texture_v1"
            )
            if candidate_feature is not None:
                restored_feature["pre_rollback_rgb"] = candidate_feature.get("rgb", [])
                restored_feature["pre_rollback_source_frame_id"] = str(
                    candidate_feature.get("source_frame_id", "")
                )
                candidate_feature.clear()
                candidate_feature.update(restored_feature)
            else:
                candidate_features.append(restored_feature)
                features_by_name[feature_name] = restored_feature
            rolled_back_features.append(feature_name)

    updated_contrast = _facial_feature_contrast(candidate_features)
    rollback_report = {
        "applied": bool(rolled_back_features),
        "version": "strict_texture_only_face_feature_rollback_v1",
        "reason": "candidate_contrast_regressed_against_seed",
        "seed_texture_dir": str(seed_texture_dir),
        "seed_manifest_path": str(seed_manifest_path),
        "rollback_groups": list(rollback_groups),
        "rolled_back_feature_names": sorted(set(rolled_back_features)),
        "skipped_feature_names": sorted(set(skipped_features)),
        "rolled_back_texel_count": int(np.count_nonzero(rolled_back_texels)),
        "candidate_contrast_before": candidate_contrast,
        "seed_contrast": seed_contrast,
        "candidate_contrast_after": updated_contrast,
    }
    facial_features["features"] = candidate_features
    facial_features["contrast"] = updated_contrast
    facial_features["strict_texture_only_face_feature_rollback"] = rollback_report
    return rollback_report


def _texture_only_face_feature_rollback_groups(
    candidate_contrast: dict[str, object],
    seed_contrast: dict[str, object],
) -> tuple[str, ...]:
    rollback_groups: list[str] = []
    for group_name, metric_names in _STRICT_TEXTURE_ONLY_FACE_ROLLBACK_METRICS.items():
        for metric_name in metric_names:
            candidate_value = _contrast_metric(candidate_contrast, metric_name)
            seed_value = _contrast_metric(seed_contrast, metric_name)
            if candidate_value + _STRICT_TEXTURE_ONLY_FACE_ROLLBACK_EPSILON < seed_value:
                rollback_groups.append(group_name)
                break
    return tuple(rollback_groups)


def _contrast_metric(contrast: dict[str, object], metric_name: str) -> float:
    try:
        value = float(contrast.get(metric_name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value


def _facial_feature_list(facial_features: dict[str, object]) -> list[dict[str, object]]:
    raw_features = facial_features.get("features", [])
    if not isinstance(raw_features, list):
        return []
    return [feature for feature in raw_features if isinstance(feature, dict)]


def _feature_texture_mask(
    feature: dict[str, object],
    *,
    texture_size: int,
) -> NDArray[np.bool_]:
    uv = feature.get("uv", [])
    radius_uv = feature.get("radius_uv", [])
    if not isinstance(uv, list) or not isinstance(radius_uv, list):
        return np.zeros((texture_size, texture_size), dtype=np.bool_)
    if len(uv) != 2 or len(radius_uv) != 2:
        return np.zeros((texture_size, texture_size), dtype=np.bool_)
    return _ellipse_mask(
        uv=np.asarray([float(uv[0]), float(uv[1])], dtype=np.float64),
        radius_uv=(float(radius_uv[0]), float(radius_uv[1])),
        texture_size=texture_size,
    )


def texture_size_for_mode(mode: str, *, project_dir: Path | None = None) -> int:
    if mode == "raw":
        if project_dir is not None:
            target_report = _valid_face_texture_target_report(project_dir, mode=mode)
            if target_report is not None:
                return int(target_report["target_texture_size"])
        return RAW_FACE_PRIORITY_TEXTURE_SIZE
    if mode == "synthetic":
        return DEFAULT_TEXTURE_SIZE
    raise ValueError(f"unsupported texture projection mode: {mode}")


def _valid_face_texture_target_report(
    project_dir: Path,
    *,
    mode: str,
) -> dict[str, object] | None:
    if mode != "raw":
        return None
    report_path = project_dir / "outputs" / "quality" / "raw_face_texture_targets.json"
    if not report_path.exists():
        return None
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not bool(raw.get("success", False)):
        return None
    if raw.get("source") != "opencat3d.face_texture_upsampling_targets_v1":
        return None
    output_contract = raw.get("output_contract", {})
    if not isinstance(output_contract, dict):
        return None
    if bool(output_contract.get("topology_changed", True)):
        return None
    if bool(output_contract.get("texture_resampled", True)):
        return None
    metrics = raw.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    target_size = int(metrics.get("target_texture_size", 0))
    target_scale = float(metrics.get("target_texture_scale", 0.0))
    requested_texels = int(metrics.get("face_texture_target_requested_texel_count", 0))
    allocated_texels = int(metrics.get("face_texture_target_allocated_texel_count", 0))
    target_source = "opencat3d.face_texture_upsampling_targets_v1"
    template_region_v2_contract = raw.get("template_region_v2_target_contract", {})
    if isinstance(template_region_v2_contract, dict):
        v2_size = int(metrics.get("template_region_v2_texture_target_size", 0))
        v2_requested_texels = int(
            metrics.get("template_region_v2_texture_target_requested_texel_count", 0)
        )
        v2_allocated_texels = int(
            metrics.get("template_region_v2_texture_target_allocated_texel_count", 0)
        )
        v2_valid = (
            template_region_v2_contract.get("source")
            == "opencat3d.cat_template_region_v2_texture_targets_v1"
            and template_region_v2_contract.get("region_scope") == "facial_detail_uv_islands"
            and bool(template_region_v2_contract.get("observed_data_only", False))
            and not bool(template_region_v2_contract.get("topology_changed", True))
            and not bool(template_region_v2_contract.get("texture_resampled", True))
            and v2_size >= RAW_TEMPLATE_REGION_V2_TEXTURE_TARGET_SIZE
            and v2_requested_texels > 0
            and v2_allocated_texels >= v2_requested_texels
        )
        if v2_valid and v2_size >= target_size:
            target_size = v2_size
            target_scale = float(metrics.get("template_region_v2_texture_target_scale", 0.0))
            requested_texels = v2_requested_texels
            allocated_texels = v2_allocated_texels
            target_source = str(template_region_v2_contract["source"])
    if target_size < RAW_FACE_TARGET_TEXTURE_SIZE:
        return None
    if target_size & (target_size - 1):
        return None
    return {
        "path": str(report_path),
        "target_source": target_source,
        "target_texture_size": target_size,
        "target_texture_scale": target_scale,
        "target_requested_texel_count": requested_texels,
        "target_allocated_texel_count": allocated_texels,
        "template_region_v2_texture_target_used": float(
            target_source == "opencat3d.cat_template_region_v2_texture_targets_v1"
        ),
        "template_region_v2_texture_target_size": int(
            metrics.get("template_region_v2_texture_target_size", 0)
        ),
        "template_region_v2_texture_target_scale": float(
            metrics.get("template_region_v2_texture_target_scale", 0.0)
        ),
        "template_region_v2_texture_target_island_count": int(
            metrics.get("template_region_v2_texture_target_island_count", 0)
        ),
        "template_region_v2_texture_target_facial_island_count": int(
            metrics.get("template_region_v2_texture_target_facial_island_count", 0)
        ),
        "template_region_v2_texture_target_requested_texel_count": int(
            metrics.get("template_region_v2_texture_target_requested_texel_count", 0)
        ),
        "template_region_v2_texture_target_allocated_texel_count": int(
            metrics.get("template_region_v2_texture_target_allocated_texel_count", 0)
        ),
        "template_region_v2_texture_target_eye_min_texels": int(
            metrics.get("template_region_v2_texture_target_eye_min_texels", 0)
        ),
        "template_region_v2_texture_target_nose_texels": int(
            metrics.get("template_region_v2_texture_target_nose_texels", 0)
        ),
        "template_region_v2_texture_target_muzzle_texels": int(
            metrics.get("template_region_v2_texture_target_muzzle_texels", 0)
        ),
        "template_region_v2_texture_target_ear_inner_min_texels": int(
            metrics.get("template_region_v2_texture_target_ear_inner_min_texels", 0)
        ),
    }


def _texture_projection_inputs(
    project_dir: Path,
    problem: FitProblem,
    *,
    mode: str,
    beta: NDArray[np.floating],
    vertices: NDArray[np.float64],
    root_translations: NDArray[np.float64],
) -> _TextureProjectionInputs:
    source_manifest_path = _observation_manifest_path(project_dir, mode=mode)
    images_by_frame = _load_observation_images(
        project_dir,
        mode=mode,
        manifest_path=source_manifest_path,
    )
    if mode != "raw":
        return _TextureProjectionInputs(
            observations=problem.observations,
            root_translations=root_translations,
            images_by_frame=images_by_frame,
            frame_metadata=tuple(
                {"frame_id": observation.frame_id} for observation in problem.observations
            ),
            source_manifest_path=source_manifest_path,
            fit_frame_count=len(problem.observations),
            supplemental_frame_count=0,
        )

    selected: dict[str, tuple[FitObservation, NDArray[np.float64]]] = {
        observation.frame_id: (observation, root_translations[index])
        for index, observation in enumerate(problem.observations)
    }
    observations_dir = project_dir / "observations" / mode
    frames = _load_observation_frames(source_manifest_path)
    cache_key = _texture_projection_cache_key(
        source_manifest_path=source_manifest_path,
        frame_ids=[str(frame["frame_id"]) for frame in frames],
        beta=np.asarray(beta, dtype=np.float64),
        root_translations=root_translations,
        template_vertex_count=int(problem.template.vertices_canonical.shape[0]),
        template_face_count=int(problem.template.faces.shape[0]),
    )
    cached_translations = _load_texture_projection_translation_cache(
        project_dir,
        mode=mode,
        cache_key=cache_key,
        expected_frame_count=len(frames),
    )
    if cached_translations is not None:
        projection_observations = []
        supplemental_count = 0
        for frame in frames:
            frame_id = str(frame["frame_id"])
            selected_item = selected.get(frame_id)
            if selected_item is not None:
                observation, _translation = selected_item
            else:
                observation = _load_raw_texture_observation(
                    frame,
                    observations_dir=observations_dir,
                )
                supplemental_count += 1
            projection_observations.append(observation)
        return _TextureProjectionInputs(
            observations=tuple(projection_observations),
            root_translations=cached_translations,
            images_by_frame=images_by_frame,
            frame_metadata=tuple(frames),
            source_manifest_path=source_manifest_path,
            fit_frame_count=len(problem.observations),
            supplemental_frame_count=supplemental_count,
        )

    projection_observations: list[FitObservation] = []
    projection_translations: list[NDArray[np.float64]] = []
    supplemental_count = 0
    joints = problem.template.joints(np.asarray(beta, dtype=np.float64))
    texture_config = OptimizerConfig(
        camera_root_steps=4,
        shape_steps=0,
        pose_steps=0,
        silhouette_weight=1.0,
    )
    for frame in frames:
        frame_id = str(frame["frame_id"])
        selected_item = selected.get(frame_id)
        if selected_item is not None:
            observation, translation = selected_item
            projection_observations.append(observation)
            projection_translations.append(np.asarray(translation, dtype=np.float64))
            continue

        observation = _load_raw_texture_observation(frame, observations_dir=observations_dir)
        history: list[float] = []
        translation = _optimize_frame_translation(
            vertices,
            joints,
            problem.template.faces,
            problem.template,
            observation,
            np.zeros(3, dtype=np.float64),
            texture_config,
            history,
        )
        projection_observations.append(observation)
        projection_translations.append(translation)
        supplemental_count += 1

    if not projection_observations:
        raise ValueError("texture projection manifest contains no frames")
    stacked_translations = np.stack(projection_translations, axis=0)
    _write_texture_projection_translation_cache(
        project_dir,
        mode=mode,
        cache_key=cache_key,
        frame_ids=[str(frame["frame_id"]) for frame in frames],
        root_translations=stacked_translations,
        supplemental_frame_count=supplemental_count,
    )
    return _TextureProjectionInputs(
        observations=tuple(projection_observations),
        root_translations=stacked_translations,
        images_by_frame=images_by_frame,
        frame_metadata=tuple(frames),
        source_manifest_path=source_manifest_path,
        fit_frame_count=len(problem.observations),
        supplemental_frame_count=supplemental_count,
    )


def _texture_projection_cache_key(
    *,
    source_manifest_path: Path,
    frame_ids: list[str],
    beta: NDArray[np.float64],
    root_translations: NDArray[np.float64],
    template_vertex_count: int,
    template_face_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"opencat3d.texture_projection_translation_cache.v1")
    digest.update(sha256_file(source_manifest_path).encode("ascii"))
    digest.update(json.dumps(frame_ids, separators=(",", ":")).encode("utf-8"))
    digest.update(str(int(template_vertex_count)).encode("ascii"))
    digest.update(str(int(template_face_count)).encode("ascii"))
    digest.update(np.ascontiguousarray(beta, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(root_translations, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _texture_projection_translation_cache_path(project_dir: Path, *, mode: str) -> Path:
    return project_dir / "outputs" / "textures" / f"{mode}_projection_translations.npz"


def _load_texture_projection_translation_cache(
    project_dir: Path,
    *,
    mode: str,
    cache_key: str,
    expected_frame_count: int,
) -> NDArray[np.float64] | None:
    path = _texture_projection_translation_cache_path(project_dir, mode=mode)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            raw_key = str(np.asarray(data["cache_key"]).item())
            translations = np.asarray(data["root_translations"], dtype=np.float64)
    except (KeyError, OSError, ValueError):
        return None
    if raw_key != cache_key:
        return None
    if translations.shape != (expected_frame_count, 3):
        return None
    if not np.all(np.isfinite(translations)):
        return None
    return translations


def _write_texture_projection_translation_cache(
    project_dir: Path,
    *,
    mode: str,
    cache_key: str,
    frame_ids: list[str],
    root_translations: NDArray[np.float64],
    supplemental_frame_count: int,
) -> None:
    path = _texture_projection_translation_cache_path(project_dir, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_key=np.asarray(cache_key),
            frame_ids=np.asarray(frame_ids),
            root_translations=np.asarray(root_translations, dtype=np.float64),
            supplemental_frame_count=np.asarray(int(supplemental_frame_count), dtype=np.int64),
        )
    tmp_path.replace(path)


def _observation_manifest_path(project_dir: Path, *, mode: str) -> Path:
    observations_dir = project_dir / "observations" / mode
    if mode == "raw":
        texture_manifest_path = observations_dir / "texture_manifest.json"
        if texture_manifest_path.exists():
            return texture_manifest_path
    return observations_dir / "manifest.json"


def _load_observation_images(
    project_dir: Path,
    *,
    mode: str,
    manifest_path: Path | None = None,
) -> dict[str, NDArray[np.float32]]:
    observations_dir = project_dir / "observations" / mode
    manifest_path = manifest_path or observations_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} is required")
    frames = _load_observation_frames(manifest_path)

    images: dict[str, NDArray[np.float32]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("observation manifest frame must be an object")
        frame_id = str(frame["frame_id"])
        observation_path = Path(
            str(frame.get("observation_path", observations_dir / frame_id / "observation.json"))
        )
        meta = FrameObservationMeta.model_validate_json(
            observation_path.read_text(encoding="utf-8")
        )
        with np.load(meta.image_srgb_path) as data:
            image = np.asarray(data["image_srgb"], dtype=np.float32)
        images[frame_id] = _image_to_hwc_rgb(image)
    return images


def _load_observation_frames(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("observation manifest contains no frames")
    frame_objects: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("observation manifest frame must be an object")
        frame_objects.append(dict(frame))
    return frame_objects


def _load_raw_texture_observation(
    frame: dict[str, Any],
    *,
    observations_dir: Path,
) -> FitObservation:
    frame_id = str(frame["frame_id"])
    observation_path = Path(
        str(frame.get("observation_path", observations_dir / frame_id / "observation.json"))
    )
    meta = FrameObservationMeta.model_validate_json(observation_path.read_text(encoding="utf-8"))
    with np.load(meta.hard_mask.path) as data:
        hard_mask = np.asarray(data["hard_mask"][0], dtype=np.float32)
    image_height, image_width = hard_mask.shape
    return FitObservation(
        frame_id=frame_id,
        camera=_default_raw_camera(image_width=image_width, image_height=image_height),
        keypoints_xyc=_bbox_keypoints_xyc(meta.bbox_xyxy, joint_count=11, confidence=0.5),
        hard_mask=hard_mask,
        bbox_xyxy=np.asarray(meta.bbox_xyxy, dtype=np.float64),
        keypoint_weight_scale=float(meta.quality.get("raw_keypoint_view_weight", 1.0)),
        feature_anchors_xyc=_feature_anchor_targets_from_texture_frame(
            frame,
            observations_dir=observations_dir,
        ),
    )


def _feature_anchor_targets_from_texture_frame(
    frame: dict[str, Any],
    *,
    observations_dir: Path,
) -> dict[str, tuple[float, float, float]]:
    target_path_raw = frame.get("feature_anchors_xyc_path")
    if target_path_raw is None:
        return {}
    target_path = Path(str(target_path_raw))
    if not target_path.is_absolute():
        target_path = observations_dir / target_path
    if not target_path.exists():
        raise FileNotFoundError(f"{target_path} is required by raw texture feature anchor manifest")
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{target_path} must contain a JSON object")
    targets_raw = payload.get("targets", {})
    if not isinstance(targets_raw, dict):
        raise ValueError(f"{target_path} targets must be a JSON object")
    targets: dict[str, tuple[float, float, float]] = {}
    for name, value in targets_raw.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ValueError(f"{target_path} target entries must be JSON objects")
        x = float(value["x"])
        y = float(value["y"])
        confidence = float(np.clip(float(value["confidence"]), 0.0, 1.0))
        if confidence <= 0.0:
            continue
        targets[name] = (x, y, confidence)
    return targets


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


def _bbox_keypoints_xyc(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    *,
    joint_count: int,
    confidence: float,
) -> NDArray[np.float64]:
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    base = np.asarray(
        [
            [0.50, 0.58],
            [0.50, 0.72],
            [0.34, 0.78],
            [0.24, 0.90],
            [0.78, 0.90],
            [0.66, 0.78],
            [0.50, 0.32],
            [0.38, 0.26],
            [0.62, 0.26],
            [0.50, 0.15],
            [0.50, 0.04],
        ],
        dtype=np.float64,
    )[:joint_count]
    xy = np.column_stack(
        [
            x0 + base[:, 0] * (x1 - x0),
            y0 + base[:, 1] * (y1 - y0),
        ]
    )
    conf = np.full((xy.shape[0], 1), confidence, dtype=np.float64)
    return np.concatenate([xy, conf], axis=1)


def _image_to_hwc_rgb(image: NDArray[np.floating]) -> NDArray[np.float32]:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("image_srgb must have shape [3,H,W] or [H,W,3]")
    if arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    elif arr.shape[-1] != 3:
        raise ValueError("image_srgb must have an RGB channel dimension of length 3")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _foreground_fallback_color(
    observations: tuple[FitObservation, ...],
    images_by_frame: dict[str, NDArray[np.float32]],
) -> NDArray[np.float64]:
    foreground_pixels: list[NDArray[np.float32]] = []
    for observation in observations:
        image = images_by_frame[observation.frame_id]
        mask = np.asarray(observation.hard_mask, dtype=np.float32) >= 0.5
        if image.shape[:2] != mask.shape:
            raise ValueError("observation image and mask dimensions must match")
        pixels = image[mask]
        if pixels.size:
            foreground_pixels.append(pixels.astype(np.float32))

    if not foreground_pixels:
        return np.asarray([0.5, 0.5, 0.5], dtype=np.float64)

    pixels = np.concatenate(foreground_pixels, axis=0)
    luma = _rgb_luma(pixels)
    chroma = np.max(pixels, axis=1) - np.min(pixels, axis=1)
    bright_cutoff = float(np.quantile(luma, 0.70))
    bright_neutral = pixels[(luma >= bright_cutoff) & (chroma < 0.24)]
    if bright_neutral.shape[0] >= max(16, pixels.shape[0] // 100):
        fallback = np.median(bright_neutral, axis=0)
    else:
        fallback = np.median(pixels, axis=0)
    return np.clip(np.asarray(fallback, dtype=np.float64), 0.0, 1.0)


def _repair_dark_raw_face_colors(
    face_colors: NDArray[np.float64],
    observed_faces: NDArray[np.bool_],
    foreground_fallback_color: NDArray[np.float64],
) -> tuple[NDArray[np.float64], int]:
    fallback_luma = float(_rgb_luma(foreground_fallback_color[None, :])[0])
    fallback_chroma = float(np.max(foreground_fallback_color) - np.min(foreground_fallback_color))
    if fallback_luma < 0.55:
        return face_colors, 0

    repaired = np.asarray(face_colors, dtype=np.float64).copy()
    observed_indices = np.nonzero(observed_faces)[0]
    if observed_indices.size == 0:
        return repaired, 0

    observed = repaired[observed_indices]
    observed_luma = _rgb_luma(observed)
    observed_chroma = np.max(observed, axis=1) - np.min(observed, axis=1)
    dark_threshold = max(0.28, fallback_luma - 0.32)
    dark = observed_luma < dark_threshold
    repaired[observed_indices[dark]] = foreground_fallback_color[None, :]
    repair_count = int(np.count_nonzero(dark))

    if fallback_luma >= 0.62 and fallback_chroma < 0.24:
        gray_shadow = (~dark) & (observed_luma < fallback_luma - 0.12) & (observed_chroma < 0.28)
        gray_shadow_indices = observed_indices[gray_shadow]
        if gray_shadow_indices.size:
            repaired[gray_shadow_indices] = (
                0.88 * foreground_fallback_color[None, :] + 0.12 * repaired[gray_shadow_indices]
            )
            repair_count += int(gray_shadow_indices.size)

    return repaired, repair_count


def _rgb_luma(rgb: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(rgb, dtype=np.float64)
    return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]


def _empty_facial_feature_manifest() -> dict[str, object]:
    return {
        "applied": False,
        "source": "none",
        "source_patch_feedback_source": "none",
        "source_patch_feedback_sample_count": 0,
        "source_patch_feedback_used_count": 0,
        "source_patch_feedback_components": [],
        "source_patch_bake_selection_source": "none",
        "source_patch_bake_candidate_count": 0,
        "source_patch_bake_selected_candidate": "none",
        "source_patch_bake_selected_feedback": False,
        "source_patch_bake_visual_quality_passed_count": 0,
        "source_patch_bake_visual_quality_min_score": 0.0,
        "source_patch_bake_visual_quality_mean_score": 0.0,
        "source_patch_bake_candidates": [],
        "feature_count": 0,
        "painted_texel_count": 0,
        "features": [],
        "contrast": _empty_face_feature_contrast(),
    }


def _source_patch_feedback_from_previous_diagnostics(
    project_dir: Path,
    *,
    mode: str,
) -> dict[str, str]:
    if mode != "raw":
        return {}
    diagnostics_path = project_dir / "outputs" / "quality" / "raw_face_texture_diagnostics.json"
    try:
        payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    regions = payload.get("regions", [])
    if not isinstance(regions, list):
        return {}
    feedback: dict[str, str] = {}
    for item in regions:
        if not isinstance(item, dict):
            continue
        sample_key = _source_patch_feedback_sample_key(str(item.get("name", "")))
        if not sample_key:
            continue
        component = _source_patch_feedback_component(item)
        if not component:
            continue
        existing = feedback.get(sample_key, "")
        if _source_patch_feedback_component_priority(
            component
        ) > _source_patch_feedback_component_priority(existing):
            feedback[sample_key] = component
    return feedback


def _source_patch_feedback_sample_key(region_name: str) -> str:
    return {
        "left_eye": "left_eye",
        "right_eye": "right_eye",
        "nose": "nose",
        "muzzle": "muzzle",
        "left_muzzle": "muzzle",
        "right_muzzle": "muzzle",
        "left_ear_inner": "ear_inner",
        "right_ear_inner": "ear_inner",
    }.get(region_name, "")


def _source_patch_feedback_component(region: dict[str, Any]) -> str:
    reasons = region.get("texture_visual_quality_failure_reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    reason_components = [
        _source_patch_feedback_component_from_reason(str(reason)) for reason in reasons
    ]
    for component in reason_components:
        if component:
            return component
    return _normalize_source_patch_feedback_component(
        str(region.get("texture_visual_quality_weakest_component", ""))
    )


def _source_patch_feedback_component_from_reason(reason: str) -> str:
    return {
        "coverage": "confident_texel_score",
        "flat_texture": "texture_luma_score",
        "source_detail": "source_luma_score",
        "source_evidence": "source_evidence_score",
        "combined_score": "combined_score",
    }.get(reason, "")


def _normalize_source_patch_feedback_component(component: str) -> str:
    if component in _SOURCE_PATCH_FEEDBACK_COMPONENT_PRIORITY:
        return component
    return ""


def _source_patch_feedback_component_priority(component: str) -> int:
    return _SOURCE_PATCH_FEEDBACK_COMPONENT_PRIORITY.get(component, -1)


def _source_patch_feedback_alias(sample_key: str) -> str:
    return _SOURCE_PATCH_FEEDBACK_ALIASES.get(sample_key, "")


def _paint_best_raw_facial_feature_anchors(
    basecolor: NDArray[np.uint8],
    confidence: NDArray[np.uint8],
    source_view: NDArray[np.uint8],
    *,
    problem: FitProblem,
    observations: tuple[FitObservation, ...],
    images_by_frame: dict[str, NDArray[np.float32]],
    frame_metadata: tuple[dict[str, Any], ...],
    foreground_fallback_color: NDArray[np.float64],
    source_patch_dir: Path,
    source_patch_feedback: dict[str, str],
) -> dict[str, object]:
    candidate_inputs: list[tuple[str, dict[str, str]]] = [("generic", {})]
    if source_patch_feedback:
        candidate_inputs.append(("feedback", dict(source_patch_feedback)))

    candidate_reports: list[dict[str, object]] = []
    selected_label = "generic"
    selected_feedback: dict[str, str] = {}
    selected_rank: tuple[float, ...] | None = None
    for label, feedback in candidate_inputs:
        candidate_basecolor = basecolor.copy()
        candidate_confidence = confidence.copy()
        candidate_source_view = source_view.copy()
        candidate_manifest = _paint_raw_facial_feature_anchors(
            candidate_basecolor,
            candidate_confidence,
            candidate_source_view,
            problem=problem,
            observations=observations,
            images_by_frame=images_by_frame,
            frame_metadata=frame_metadata,
            foreground_fallback_color=foreground_fallback_color,
            source_patch_dir=source_patch_dir,
            source_patch_feedback=feedback,
            write_source_patches=False,
        )
        report = _raw_face_texture_bake_candidate_report(
            label,
            candidate_basecolor,
            candidate_confidence,
            candidate_manifest,
        )
        candidate_reports.append(report)
        rank = _raw_face_texture_bake_candidate_rank(report)
        if selected_rank is None or rank > selected_rank:
            selected_rank = rank
            selected_label = label
            selected_feedback = feedback

    final_manifest = _paint_raw_facial_feature_anchors(
        basecolor,
        confidence,
        source_view,
        problem=problem,
        observations=observations,
        images_by_frame=images_by_frame,
        frame_metadata=frame_metadata,
        foreground_fallback_color=foreground_fallback_color,
        source_patch_dir=source_patch_dir,
        source_patch_feedback=selected_feedback,
        write_source_patches=True,
    )
    selected_report = _raw_face_texture_bake_candidate_report(
        selected_label,
        basecolor,
        confidence,
        final_manifest,
    )
    selected_metrics = _raw_face_texture_bake_selection_metrics(
        candidate_reports,
        selected_label=selected_label,
        selected_report=selected_report,
    )
    final_manifest.update(selected_metrics)
    return final_manifest


def _paint_raw_facial_feature_anchors(
    basecolor: NDArray[np.uint8],
    confidence: NDArray[np.uint8],
    source_view: NDArray[np.uint8],
    *,
    problem: FitProblem,
    observations: tuple[FitObservation, ...],
    images_by_frame: dict[str, NDArray[np.float32]],
    frame_metadata: tuple[dict[str, Any], ...],
    foreground_fallback_color: NDArray[np.float64],
    source_patch_dir: Path,
    source_patch_feedback: dict[str, str],
    write_source_patches: bool = True,
) -> dict[str, object]:
    texture_size = int(basecolor.shape[0])
    palette = _raw_facial_feature_palette(
        observations,
        images_by_frame,
        frame_metadata,
        foreground_fallback_color,
        source_patch_feedback=source_patch_feedback,
    )
    pupil_rgb = (
        np.asarray(palette.pupil_rgb, dtype=np.uint8)
        if palette.pupil_rgb is not None
        else palette.eye_rgb
    )
    feature_specs = (
        {
            "name": "left_ear_inner",
            "xyz": (-0.20, 1.20, 0.98),
            "radius_uv": (0.070, 0.055),
            "rgb": palette.ear_inner_rgb,
            "sample_key": "ear_inner",
        },
        {
            "name": "right_ear_inner",
            "xyz": (0.20, 1.20, 0.98),
            "radius_uv": (0.070, 0.055),
            "rgb": palette.ear_inner_rgb,
            "sample_key": "ear_inner",
        },
        {
            "name": "left_muzzle",
            "xyz": (-0.070, 0.88, 1.285),
            "radius_uv": (0.070, 0.040),
            "rgb": palette.muzzle_rgb,
            "sample_key": "muzzle",
        },
        {
            "name": "right_muzzle",
            "xyz": (0.070, 0.88, 1.285),
            "radius_uv": (0.070, 0.040),
            "rgb": palette.muzzle_rgb,
            "sample_key": "muzzle",
        },
        {
            "name": "left_whisker_pad",
            "xyz": (-0.105, 0.875, 1.300),
            "radius_uv": (0.035, 0.018),
            "rgb": palette.muzzle_rgb,
            "sample_key": "muzzle",
        },
        {
            "name": "right_whisker_pad",
            "xyz": (0.105, 0.875, 1.300),
            "radius_uv": (0.035, 0.018),
            "rgb": palette.muzzle_rgb,
            "sample_key": "muzzle",
        },
        {
            "name": "left_whisker_root",
            "xyz": (-0.065, 0.845, 1.310),
            "radius_uv": (0.026, 0.012),
            "rgb": palette.muzzle_rgb,
            "sample_key": "muzzle",
        },
        {
            "name": "right_whisker_root",
            "xyz": (0.065, 0.845, 1.310),
            "radius_uv": (0.026, 0.012),
            "rgb": palette.muzzle_rgb,
            "sample_key": "muzzle",
        },
        {
            "name": "left_eye_shadow",
            "xyz": (-0.12, 1.02, 1.18),
            "radius_uv": (0.078, 0.048),
            "rgb": palette.eye_rgb,
            "sample_key": "left_eye",
        },
        {
            "name": "right_eye_shadow",
            "xyz": (0.12, 1.02, 1.18),
            "radius_uv": (0.078, 0.048),
            "rgb": palette.eye_rgb,
            "sample_key": "right_eye",
        },
        {
            "name": "left_eye_core",
            "xyz": (-0.12, 1.02, 1.185),
            "radius_uv": (0.046, 0.030),
            "rgb": pupil_rgb,
            "sample_key": "left_eye",
        },
        {
            "name": "right_eye_core",
            "xyz": (0.12, 1.02, 1.185),
            "radius_uv": (0.046, 0.030),
            "rgb": pupil_rgb,
            "sample_key": "right_eye",
        },
        {
            "name": "left_eye_iris",
            "xyz": (-0.12, 1.02, 1.188),
            "radius_uv": (0.032, 0.022),
            "rgb": palette.iris_rgb,
            "sample_key": "left_iris",
        },
        {
            "name": "right_eye_iris",
            "xyz": (0.12, 1.02, 1.188),
            "radius_uv": (0.032, 0.022),
            "rgb": palette.iris_rgb,
            "sample_key": "right_iris",
        },
        {
            "name": "left_eye_pupil",
            "xyz": (-0.12, 1.02, 1.191),
            "radius_uv": (0.016, 0.012),
            "rgb": pupil_rgb,
            "sample_key": "left_pupil",
        },
        {
            "name": "right_eye_pupil",
            "xyz": (0.12, 1.02, 1.191),
            "radius_uv": (0.016, 0.012),
            "rgb": pupil_rgb,
            "sample_key": "right_pupil",
        },
        {
            "name": "nose",
            "xyz": (0.0, 0.90, 1.36),
            "radius_uv": (0.056, 0.034),
            "rgb": palette.nose_rgb,
            "sample_key": "nose",
        },
        {
            "name": "mouth_line",
            "xyz": (0.0, 0.85, 1.315),
            "radius_uv": (0.060, 0.018),
            "rgb": palette.mouth_rgb,
            "sample_key": "mouth",
        },
    )

    features: list[dict[str, object]] = []
    painted_texels = np.zeros(basecolor.shape[:2], dtype=np.bool_)
    observed_feature_texels = np.zeros(basecolor.shape[:2], dtype=np.bool_)
    source_patch_texels = np.zeros(basecolor.shape[:2], dtype=np.bool_)
    source_patch_nominal_texels = np.zeros(basecolor.shape[:2], dtype=np.bool_)
    whisker_pad_texels = np.zeros(basecolor.shape[:2], dtype=np.bool_)
    feature_source_view_id = _feature_source_view_id(palette.source_frame_id, observations)
    vertices = np.asarray(problem.template.vertices_canonical, dtype=np.float64)
    source_patch_count = 0
    source_patch_quality_scores: list[float] = []
    source_patch_quality_passed_count = 0
    source_patch_fusion_weights: list[float] = []
    source_patch_footprint_area_scales: list[float] = []
    source_patch_fusion_weighted_texels = 0.0
    source_patch_core_texel_count = 0
    source_patch_feathered_edge_texel_count = 0
    source_patch_alpha_sum = 0.0
    source_patch_alpha_texel_count = 0
    source_patch_shape_mask_texel_count = 0
    source_patch_shape_alpha_sum = 0.0
    source_patch_shape_alpha_texel_count = 0
    source_patch_source_mask_observed_texel_count = 0
    source_patch_source_mask_densified_texel_count = 0
    source_patch_source_mask_densification_gains: list[float] = []
    source_patch_localization_candidate_texel_count = 0
    source_patch_localization_selected_texel_count = 0
    source_patch_localization_fractions: list[float] = []
    source_patch_landmark_aligned_count = 0
    source_patch_landmark_confidences: list[float] = []
    source_patch_observed_anchor_guided_count = 0
    source_patch_landmark_destination_scaled_count = 0
    source_patch_landmark_destination_scales: list[float] = []
    source_patch_frontal_face_confidences: list[float] = []
    source_patch_view_weights: list[float] = []
    source_patch_view_weighted_ranks: list[float] = []
    source_patch_frontal_photo_priorities: list[float] = []
    source_patch_direct_photo_candidate_count = 0
    source_patch_zero_priority_direct_photo_count = 0
    source_patch_view_labels: set[str] = set()
    source_patch_feedback_used_count = 0
    source_patch_feedback_components: set[str] = set()
    visibility_reinforcement_texel_count = 0
    visibility_reinforcement_feature_count = 0
    eye_source_detail_reinforcement_texel_count = 0
    eye_source_detail_reinforcement_feature_count = 0
    for spec in feature_specs:
        feature_name = str(spec["name"])
        uv = _face_feature_destination_uv(vertices, feature_name=feature_name, spec=spec)
        destination_anchor_source = _face_feature_destination_anchor_source(feature_name)
        color = np.asarray(spec["rgb"], dtype=np.uint8)
        sample_key = str(spec["sample_key"])
        observed_sample_count = int(palette.sample_counts.get(sample_key, 0))
        patch = palette.source_patches.get(sample_key)
        patch_mask = palette.source_patch_masks.get(sample_key)
        source_mask_observed_texel_count = int(
            palette.source_patch_mask_observed_counts.get(sample_key, 0)
        )
        source_mask_densified_texel_count = int(
            palette.source_patch_mask_densified_counts.get(sample_key, 0)
        )
        source_mask_densification_gain = float(
            palette.source_patch_mask_densification_gains.get(sample_key, 0.0)
        )
        localization_candidate_texel_count = int(
            palette.source_patch_localization_candidate_counts.get(
                sample_key,
                observed_sample_count,
            )
        )
        localization_selected_texel_count = int(
            palette.source_patch_localization_selected_counts.get(
                sample_key,
                observed_sample_count,
            )
        )
        localization_fraction = float(
            palette.source_patch_localization_fractions.get(
                sample_key,
                localization_selected_texel_count / max(localization_candidate_texel_count, 1),
            )
        )
        landmark_confidence = float(palette.source_patch_landmark_confidences.get(sample_key, 0.0))
        source_patch_quality_score = float(palette.source_patch_scores.get(sample_key, 0.0))
        source_patch_frontal_face_confidence = float(
            palette.source_patch_frontal_face_confidences.get(sample_key, 0.0)
        )
        source_patch_view_weight = float(palette.source_patch_view_weights.get(sample_key, 0.0))
        source_patch_view_label = str(palette.source_patch_view_labels.get(sample_key, "unknown"))
        source_patch_view_weighted_rank = float(
            palette.source_patch_view_weighted_ranks.get(sample_key, 0.0)
        )
        source_patch_frontal_photo_priority = float(
            palette.source_patch_frontal_photo_priorities.get(sample_key, 0.0)
        )
        source_patch_direct_photo_candidate = bool(
            palette.source_patch_direct_photo_candidates.get(sample_key, False)
        )
        source_patch_observed_anchor_guided = bool(
            palette.source_patch_observed_anchor_guided.get(sample_key, False)
        )
        source_patch_feedback_component = str(
            palette.source_patch_feedback_components.get(sample_key, "")
        )
        source_patch_feedback_score = float(
            palette.source_patch_feedback_scores.get(sample_key, 0.0)
        )
        sample_source_frame_id = str(
            palette.source_patch_frame_ids.get(sample_key, palette.source_frame_id)
        )
        sample_source_view_id = _feature_source_view_id(sample_source_frame_id, observations)
        used_source_patch = (
            patch is not None
            and patch.size > 0
            and patch_mask is not None
            and patch_mask.size > 0
            and observed_sample_count > 0
        )
        source_patch_luma_std = _source_patch_luma_std(patch) if used_source_patch else 0.0
        source_patch_unique_rgb_count = (
            _source_patch_unique_rgb_count(patch) if used_source_patch else 0
        )
        destination_sample_count = _source_patch_destination_sample_count(
            sample_key,
            sample_count=observed_sample_count,
            source_mask_observed_texel_count=source_mask_observed_texel_count,
        )
        nominal_radius_u, nominal_radius_v = spec["radius_uv"]
        footprint = _source_patch_footprint(
            sample_key,
            nominal_radius_uv=(float(nominal_radius_u), float(nominal_radius_v)),
            patch=patch,
            sample_count=destination_sample_count,
            used_source_patch=used_source_patch,
            landmark_confidence=landmark_confidence,
        )
        radius_u, radius_v = footprint.radius_uv
        nominal_mask = _ellipse_mask(
            uv=uv,
            radius_uv=(float(nominal_radius_u), float(nominal_radius_v)),
            texture_size=texture_size,
        )
        mask = _ellipse_mask(
            uv=uv,
            radius_uv=(radius_u, radius_v),
            texture_size=texture_size,
        )
        if not np.any(mask):
            continue
        source_patch_fusion_weight = _source_patch_fusion_weight(source_patch_quality_score)
        source_patch_feather_profile = _source_patch_feather_profile(sample_key)
        source_patch_feather_width = source_patch_feather_profile["edge_width"]
        nominal_texel_count = int(np.count_nonzero(nominal_mask))
        adjusted_texel_count = int(np.count_nonzero(mask))
        paint_stats = _PatchPaintStats(
            painted_texel_count=0,
            weighted_texel_count=0.0,
            core_texel_count=0,
            feathered_edge_texel_count=0,
            mean_alpha=0.0,
            shape_mask_texel_count=0,
            shape_alpha_mean=0.0,
            applied_mask=np.zeros_like(mask),
        )
        visibility_reinforcement_mask = np.zeros_like(mask)
        visibility_reinforcement_alpha = 0.0
        if used_source_patch:
            paint_stats = _paint_patch_into_mask(
                basecolor,
                mask,
                patch,
                sample_key=sample_key,
                source_shape_mask=patch_mask,
                fusion_weight=source_patch_fusion_weight,
                feather_width=source_patch_feather_width,
                feather_x_scale=source_patch_feather_profile["x_scale"],
                feather_y_scale=source_patch_feather_profile["y_scale"],
            )
            source_patch_count += 1
            source_patch_texels |= paint_stats.applied_mask
            source_patch_nominal_texels |= nominal_mask
            source_patch_quality_scores.append(source_patch_quality_score)
            source_patch_fusion_weights.append(source_patch_fusion_weight)
            source_patch_footprint_area_scales.append(footprint.area_scale)
            source_patch_fusion_weighted_texels += paint_stats.weighted_texel_count
            source_patch_core_texel_count += paint_stats.core_texel_count
            source_patch_feathered_edge_texel_count += paint_stats.feathered_edge_texel_count
            source_patch_alpha_sum += paint_stats.mean_alpha * paint_stats.painted_texel_count
            source_patch_alpha_texel_count += paint_stats.painted_texel_count
            source_patch_shape_mask_texel_count += paint_stats.shape_mask_texel_count
            source_patch_shape_alpha_sum += (
                paint_stats.shape_alpha_mean * paint_stats.painted_texel_count
            )
            source_patch_shape_alpha_texel_count += paint_stats.painted_texel_count
            source_patch_source_mask_observed_texel_count += source_mask_observed_texel_count
            source_patch_source_mask_densified_texel_count += source_mask_densified_texel_count
            source_patch_source_mask_densification_gains.append(source_mask_densification_gain)
            source_patch_localization_candidate_texel_count += localization_candidate_texel_count
            source_patch_localization_selected_texel_count += localization_selected_texel_count
            source_patch_localization_fractions.append(localization_fraction)
            source_patch_landmark_confidences.append(landmark_confidence)
            if landmark_confidence > 0.0:
                source_patch_landmark_aligned_count += 1
            if source_patch_observed_anchor_guided:
                source_patch_observed_anchor_guided_count += 1
            if footprint.landmark_destination_scale < 1.0:
                source_patch_landmark_destination_scaled_count += 1
                source_patch_landmark_destination_scales.append(
                    footprint.landmark_destination_scale
                )
            source_patch_frontal_face_confidences.append(source_patch_frontal_face_confidence)
            source_patch_view_weights.append(source_patch_view_weight)
            source_patch_view_weighted_ranks.append(source_patch_view_weighted_rank)
            source_patch_frontal_photo_priorities.append(source_patch_frontal_photo_priority)
            if source_patch_direct_photo_candidate:
                source_patch_direct_photo_candidate_count += 1
                if source_patch_frontal_photo_priority <= 0.0:
                    source_patch_zero_priority_direct_photo_count += 1
            source_patch_view_labels.add(source_patch_view_label)
            if source_patch_quality_score >= _MIN_SOURCE_PATCH_QUALITY_SCORE:
                source_patch_quality_passed_count += 1
            if source_patch_feedback_component:
                source_patch_feedback_used_count += 1
                source_patch_feedback_components.add(source_patch_feedback_component)
        else:
            basecolor[mask] = color[None, :]
        visibility_reinforcement_detail_preserved = bool(
            used_source_patch
            and str(spec["name"]) in _EYE_SOURCE_DETAIL_REINFORCEMENT_FEATURES
            and np.any(paint_stats.applied_mask)
        )
        visibility_reinforcement_support_mask = (
            paint_stats.applied_mask if visibility_reinforcement_detail_preserved else mask
        )
        visibility_reinforcement_alpha = _feature_visibility_reinforcement_alpha(
            feature_name=str(spec["name"]),
            basecolor=basecolor,
            mask=visibility_reinforcement_support_mask,
            color=color,
        )
        if visibility_reinforcement_detail_preserved:
            visibility_reinforcement_alpha = _eye_source_detail_reinforcement_alpha(
                str(spec["name"]),
                visibility_reinforcement_alpha,
                source_patch_luma_std=source_patch_luma_std,
                source_patch_unique_rgb_count=source_patch_unique_rgb_count,
            )
        if visibility_reinforcement_alpha > 0.0:
            visibility_reinforcement_mask = _paint_feature_visibility_reinforcement(
                basecolor,
                visibility_reinforcement_support_mask,
                color,
                alpha=visibility_reinforcement_alpha,
                preserve_detail=visibility_reinforcement_detail_preserved,
            )
            if used_source_patch:
                visibility_texel_count = int(np.count_nonzero(visibility_reinforcement_mask))
                source_patch_texels |= visibility_reinforcement_mask
                source_patch_nominal_texels |= nominal_mask
                source_patch_shape_mask_texel_count += visibility_texel_count
                source_patch_core_texel_count += visibility_texel_count
                source_patch_fusion_weighted_texels += (
                    visibility_reinforcement_alpha * visibility_texel_count
                )
                source_patch_alpha_sum += visibility_reinforcement_alpha * visibility_texel_count
                source_patch_alpha_texel_count += visibility_texel_count
                source_patch_shape_alpha_sum += visibility_texel_count
                source_patch_shape_alpha_texel_count += visibility_texel_count
        source_patch_path = ""
        source_patch_sha256 = ""
        source_patch_mask_path = ""
        source_patch_mask_sha256 = ""
        uv_support_mask_path = ""
        uv_support_mask_sha256 = ""
        source_patch_width = 0
        source_patch_height = 0
        source_patch_mask_width = 0
        source_patch_mask_height = 0
        if (
            write_source_patches
            and used_source_patch
            and patch is not None
            and patch_mask is not None
        ):
            path = _write_feature_source_patch(
                source_patch_dir,
                feature_name=str(spec["name"]),
                patch=patch,
            )
            source_patch_path = str(path)
            source_patch_sha256 = sha256_file(path)
            source_patch_height, source_patch_width = patch.shape[:2]
            mask_path = _write_feature_source_patch_mask(
                source_patch_dir,
                feature_name=str(spec["name"]),
                patch_mask=patch_mask,
            )
            source_patch_mask_path = str(mask_path)
            source_patch_mask_sha256 = sha256_file(mask_path)
            source_patch_mask_height, source_patch_mask_width = patch_mask.shape[:2]
        painted_mask = (
            paint_stats.applied_mask if used_source_patch else mask
        ) | visibility_reinforcement_mask
        if write_source_patches and used_source_patch:
            uv_mask_path = _write_feature_uv_support_mask(
                source_patch_dir,
                feature_name=str(spec["name"]),
                support_mask=painted_mask,
            )
            uv_support_mask_path = str(uv_mask_path)
            uv_support_mask_sha256 = sha256_file(uv_mask_path)
        feature_source_patch_texel_count = (
            int(np.count_nonzero(painted_mask)) if used_source_patch else 0
        )
        feature_visibility_reinforcement_texel_count = int(
            np.count_nonzero(visibility_reinforcement_mask)
        )
        if feature_visibility_reinforcement_texel_count > 0:
            visibility_reinforcement_texel_count += feature_visibility_reinforcement_texel_count
            visibility_reinforcement_feature_count += 1
            if visibility_reinforcement_detail_preserved:
                eye_source_detail_reinforcement_texel_count += (
                    feature_visibility_reinforcement_texel_count
                )
                eye_source_detail_reinforcement_feature_count += 1
        painted_texels |= painted_mask
        if feature_name in _WHISKER_PAD_FEATURE_NAMES:
            whisker_pad_texels |= painted_mask
        provenance = (
            "observed" if sample_source_view_id > 0 and observed_sample_count > 0 else "inferred"
        )
        observed_texel_count = 0
        if provenance == "observed":
            confidence[painted_mask] = 255
            source_view[painted_mask] = np.uint8(sample_source_view_id)
            observed_feature_texels |= painted_mask
            observed_texel_count = int(np.count_nonzero(painted_mask))
        features.append(
            {
                "name": str(spec["name"]),
                "uv": [round(float(uv[0]), 6), round(float(uv[1]), 6)],
                "destination_anchor_source": destination_anchor_source,
                "rgb": [int(channel) for channel in color],
                "radius_uv": [float(radius_u), float(radius_v)],
                "nominal_radius_uv": [float(nominal_radius_u), float(nominal_radius_v)],
                "sample_key": sample_key,
                "source_frame_id": sample_source_frame_id,
                "source_view_id": int(sample_source_view_id),
                "sample_count": observed_sample_count,
                "provenance": provenance,
                "observed_texel_count": observed_texel_count,
                "source_patch_used": bool(used_source_patch),
                "source_patch_quality_score": round(float(source_patch_quality_score), 6),
                "source_patch_view_weighted_rank": round(
                    float(source_patch_view_weighted_rank if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_frontal_photo_priority": round(
                    float(source_patch_frontal_photo_priority if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_direct_photo_candidate": bool(
                    used_source_patch and source_patch_direct_photo_candidate
                ),
                "source_patch_zero_priority_direct_photo": bool(
                    used_source_patch
                    and source_patch_direct_photo_candidate
                    and source_patch_frontal_photo_priority <= 0.0
                ),
                "source_patch_frontal_face_confidence": round(
                    float(source_patch_frontal_face_confidence if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_view_weight": round(
                    float(source_patch_view_weight if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_view_label": source_patch_view_label
                if used_source_patch
                else "unknown",
                "source_patch_view_weighting_used": bool(used_source_patch),
                "source_patch_quality_threshold": _MIN_SOURCE_PATCH_QUALITY_SCORE,
                "source_patch_quality_passed": bool(
                    used_source_patch
                    and source_patch_quality_score >= _MIN_SOURCE_PATCH_QUALITY_SCORE
                ),
                "source_patch_feedback_component": source_patch_feedback_component
                if used_source_patch
                else "",
                "source_patch_feedback_score": round(
                    float(source_patch_feedback_score if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_feedback_used": bool(
                    used_source_patch and source_patch_feedback_component
                ),
                "source_patch_luma_std": round(float(source_patch_luma_std), 6),
                "source_patch_unique_rgb_count": int(source_patch_unique_rgb_count),
                "source_patch_fusion_weight": round(
                    float(source_patch_fusion_weight if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_fusion_min_weight": _MIN_SOURCE_PATCH_FUSION_WEIGHT,
                "source_patch_fusion_max_weight": _MAX_SOURCE_PATCH_FUSION_WEIGHT,
                "source_patch_feather_width": round(
                    float(source_patch_feather_width if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_feather_x_scale": round(
                    float(source_patch_feather_profile["x_scale"] if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_feather_y_scale": round(
                    float(source_patch_feather_profile["y_scale"] if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_mean_blend_alpha": round(
                    float(paint_stats.mean_alpha),
                    6,
                ),
                "source_patch_edge_clamp_max_ratio": round(
                    float(paint_stats.edge_clamp_max_ratio),
                    6,
                ),
                "source_patch_edge_clamp_removed_texel_count": int(
                    paint_stats.edge_clamp_removed_texel_count
                ),
                "source_patch_shape_mask_used": bool(used_source_patch),
                "source_patch_shape_mask_texel_count": int(paint_stats.shape_mask_texel_count),
                "source_patch_shape_alpha_mean": round(
                    float(paint_stats.shape_alpha_mean),
                    6,
                ),
                "source_patch_source_mask_observed_texel_count": int(
                    source_mask_observed_texel_count if used_source_patch else 0
                ),
                "source_patch_source_mask_densified_texel_count": int(
                    source_mask_densified_texel_count if used_source_patch else 0
                ),
                "source_patch_source_mask_densification_gain": round(
                    float(source_mask_densification_gain if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_destination_sample_count": int(
                    destination_sample_count if used_source_patch else 0
                ),
                "source_patch_localization_candidate_texel_count": int(
                    localization_candidate_texel_count if used_source_patch else 0
                ),
                "source_patch_localization_selected_texel_count": int(
                    localization_selected_texel_count if used_source_patch else 0
                ),
                "source_patch_localization_fraction": round(
                    float(localization_fraction if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_landmark_confidence": round(
                    float(landmark_confidence if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_landmark_aligned": bool(
                    used_source_patch and landmark_confidence > 0.0
                ),
                "source_patch_observed_anchor_guided": bool(
                    used_source_patch and source_patch_observed_anchor_guided
                ),
                "source_patch_landmark_destination_scale": round(
                    float(footprint.landmark_destination_scale if used_source_patch else 1.0),
                    6,
                ),
                "source_patch_landmark_destination_scaled": bool(
                    used_source_patch and footprint.landmark_destination_scale < 1.0
                ),
                "source_patch_footprint_density": round(
                    float(footprint.density if used_source_patch else 0.0),
                    6,
                ),
                "source_patch_footprint_radius_scale": round(
                    float(footprint.radius_scale if used_source_patch else 1.0),
                    6,
                ),
                "source_patch_footprint_area_scale": round(
                    float(footprint.area_scale if used_source_patch else 1.0),
                    6,
                ),
                "source_patch_footprint_x_scale": round(
                    float(footprint.x_scale if used_source_patch else 1.0),
                    6,
                ),
                "source_patch_footprint_y_scale": round(
                    float(footprint.y_scale if used_source_patch else 1.0),
                    6,
                ),
                "source_patch_core_texel_count": int(paint_stats.core_texel_count),
                "source_patch_feathered_edge_texel_count": int(
                    paint_stats.feathered_edge_texel_count
                ),
                "source_patch_nominal_texel_count": nominal_texel_count if used_source_patch else 0,
                "source_patch_ellipse_texel_count": adjusted_texel_count
                if used_source_patch
                else 0,
                "source_patch_footprint_reduction_fraction": round(
                    float(1.0 - paint_stats.painted_texel_count / max(nominal_texel_count, 1))
                    if used_source_patch
                    else 0.0,
                    6,
                ),
                "source_patch_path": source_patch_path,
                "source_patch_sha256": source_patch_sha256,
                "source_patch_mask_path": source_patch_mask_path,
                "source_patch_mask_sha256": source_patch_mask_sha256,
                "uv_support_mask_path": uv_support_mask_path,
                "uv_support_mask_sha256": uv_support_mask_sha256,
                "uv_support_mask_width": int(texture_size if uv_support_mask_path else 0),
                "uv_support_mask_height": int(texture_size if uv_support_mask_path else 0),
                "uv_support_mask_texel_count": feature_source_patch_texel_count,
                "source_patch_width": int(source_patch_width),
                "source_patch_height": int(source_patch_height),
                "source_patch_mask_width": int(source_patch_mask_width),
                "source_patch_mask_height": int(source_patch_mask_height),
                "source_patch_texel_count": feature_source_patch_texel_count,
                "visibility_reinforcement_used": bool(np.any(visibility_reinforcement_mask)),
                "visibility_reinforcement_source": (
                    "source_patch_detail_preserving_eye_v1"
                    if (
                        visibility_reinforcement_detail_preserved
                        and np.any(visibility_reinforcement_mask)
                    )
                    else (
                        "observed_palette_anatomical_anchor_v1"
                        if np.any(visibility_reinforcement_mask)
                        else "none"
                    )
                ),
                "visibility_reinforcement_detail_preserved": bool(
                    visibility_reinforcement_detail_preserved
                    and np.any(visibility_reinforcement_mask)
                ),
                "visibility_reinforcement_alpha": round(
                    float(visibility_reinforcement_alpha),
                    6,
                ),
                "visibility_reinforcement_texel_count": int(
                    feature_visibility_reinforcement_texel_count
                ),
            }
        )

    destination_anchor_metrics = _facial_feature_destination_anchor_metrics(features)
    return {
        "applied": bool(features),
        "source": "observed_face_priority_texture_projection_v1",
        "destination_anchor_source": _FACE_ANATOMICAL_UV_ANCHOR_VERSION,
        "source_frame_id": palette.source_frame_id,
        "source_score": round(float(palette.source_score), 6),
        "source_view_id": int(feature_source_view_id),
        "source_patch_dir": str(source_patch_dir),
        "source_patch_feedback_source": "previous_face_texture_diagnostics_v1"
        if palette.source_patch_feedback_components
        else "none",
        "source_patch_feedback_sample_count": int(len(palette.source_patch_feedback_components)),
        "source_patch_feedback_used_count": int(source_patch_feedback_used_count),
        "source_patch_feedback_components": sorted(source_patch_feedback_components),
        "visibility_reinforcement_source": "observed_palette_anatomical_anchor_v1"
        if visibility_reinforcement_feature_count
        else "none",
        "visibility_reinforcement_feature_count": int(visibility_reinforcement_feature_count),
        "visibility_reinforcement_texel_count": int(visibility_reinforcement_texel_count),
        "eye_source_detail_reinforcement_feature_count": int(
            eye_source_detail_reinforcement_feature_count
        ),
        "eye_source_detail_reinforcement_texel_count": int(
            eye_source_detail_reinforcement_texel_count
        ),
        "eye_source_detail_reinforcement_texel_fraction": float(
            eye_source_detail_reinforcement_texel_count
            / max(visibility_reinforcement_texel_count, 1)
        ),
        "whisker_pad_source": (
            "observed_muzzle_source_patch_anatomical_anchor_v1"
            if np.any(whisker_pad_texels)
            else "none"
        ),
        "whisker_pad_feature_count": int(
            sum(
                1
                for feature in features
                if str(feature.get("name", "")) in _WHISKER_PAD_FEATURE_NAMES
            )
        ),
        "whisker_pad_texel_count": int(np.count_nonzero(whisker_pad_texels)),
        "whisker_pad_texel_fraction": float(
            np.count_nonzero(whisker_pad_texels) / max(np.count_nonzero(painted_texels), 1)
        ),
        "sample_counts": palette.sample_counts,
        "feature_count": len(features),
        "painted_texel_count": int(np.count_nonzero(painted_texels)),
        "observed_texel_count": int(np.count_nonzero(observed_feature_texels)),
        "observed_texel_fraction": float(
            np.count_nonzero(observed_feature_texels) / max(np.count_nonzero(painted_texels), 1)
        ),
        "source_patch_count": int(source_patch_count),
        "source_patch_artifact_count": int(
            sum(1 for feature in features if str(feature.get("source_patch_path", "")))
        ),
        "source_patch_quality_threshold": _MIN_SOURCE_PATCH_QUALITY_SCORE,
        "source_patch_quality_passed_count": int(source_patch_quality_passed_count),
        "source_patch_quality_total_count": int(len(source_patch_quality_scores)),
        "source_patch_quality_fraction": float(
            source_patch_quality_passed_count / max(len(source_patch_quality_scores), 1)
        ),
        "source_patch_min_quality_score": float(min(source_patch_quality_scores, default=0.0)),
        "source_patch_mean_quality_score": float(
            np.mean(source_patch_quality_scores) if source_patch_quality_scores else 0.0
        ),
        "source_patch_view_weighting_source": "raw_frontal_face_confidence_v1",
        "source_patch_view_weighting_used": bool(source_patch_view_weights),
        "source_patch_view_weighted_count": int(len(source_patch_view_weights)),
        "source_patch_frontal_photo_priority_source": ("direct_observed_photo_patch_v1"),
        "source_patch_frontal_photo_priority_used": bool(
            any(value > 0.0 for value in source_patch_frontal_photo_priorities)
        ),
        "source_patch_frontal_photo_priority_count": int(
            len(source_patch_frontal_photo_priorities)
        ),
        "source_patch_max_frontal_photo_priority": float(
            max(source_patch_frontal_photo_priorities, default=0.0)
        ),
        "source_patch_mean_frontal_photo_priority": float(
            np.mean(source_patch_frontal_photo_priorities)
            if source_patch_frontal_photo_priorities
            else 0.0
        ),
        "source_patch_direct_photo_candidate_count": int(source_patch_direct_photo_candidate_count),
        "source_patch_zero_priority_direct_photo_count": int(
            source_patch_zero_priority_direct_photo_count
        ),
        "source_patch_candidate_diagnostic_source": ("raw_face_source_patch_candidate_scores_v1"),
        "source_patch_candidate_diagnostic_feature_count": int(
            len(palette.source_patch_candidate_diagnostics)
        ),
        "source_patch_candidate_diagnostic_count": int(
            sum(
                len(candidates)
                for candidates in palette.source_patch_candidate_diagnostics.values()
            )
        ),
        "source_patch_candidate_diagnostics": (palette.source_patch_candidate_diagnostics),
        "source_patch_min_view_weight": float(min(source_patch_view_weights, default=0.0)),
        "source_patch_mean_view_weight": float(
            np.mean(source_patch_view_weights) if source_patch_view_weights else 0.0
        ),
        "source_patch_min_frontal_face_confidence": float(
            min(source_patch_frontal_face_confidences, default=0.0)
        ),
        "source_patch_mean_frontal_face_confidence": float(
            np.mean(source_patch_frontal_face_confidences)
            if source_patch_frontal_face_confidences
            else 0.0
        ),
        "source_patch_min_view_weighted_rank": float(
            min(source_patch_view_weighted_ranks, default=0.0)
        ),
        "source_patch_mean_view_weighted_rank": float(
            np.mean(source_patch_view_weighted_ranks) if source_patch_view_weighted_ranks else 0.0
        ),
        "source_patch_view_labels": sorted(source_patch_view_labels),
        "source_patch_fusion_min_weight": float(min(source_patch_fusion_weights, default=0.0)),
        "source_patch_fusion_mean_weight": float(
            np.mean(source_patch_fusion_weights) if source_patch_fusion_weights else 0.0
        ),
        "source_patch_fusion_weighted_texel_count": int(round(source_patch_fusion_weighted_texels)),
        "source_patch_nominal_texel_count": int(np.count_nonzero(source_patch_nominal_texels)),
        "source_patch_footprint_reduction_fraction": float(
            1.0
            - np.count_nonzero(source_patch_texels)
            / max(np.count_nonzero(source_patch_nominal_texels), 1)
        ),
        "source_patch_mean_footprint_area_scale": float(
            np.mean(source_patch_footprint_area_scales)
            if source_patch_footprint_area_scales
            else 1.0
        ),
        "source_patch_min_footprint_area_scale": float(
            min(source_patch_footprint_area_scales, default=1.0)
        ),
        "source_patch_shape_mask_texel_count": int(source_patch_shape_mask_texel_count),
        "source_patch_shape_mask_fraction": float(
            np.count_nonzero(source_patch_texels) / max(source_patch_shape_mask_texel_count, 1)
        ),
        "source_patch_shape_alpha_mean": float(
            source_patch_shape_alpha_sum / max(source_patch_shape_alpha_texel_count, 1)
        ),
        "source_patch_source_mask_observed_texel_count": int(
            source_patch_source_mask_observed_texel_count
        ),
        "source_patch_source_mask_densified_texel_count": int(
            source_patch_source_mask_densified_texel_count
        ),
        "source_patch_source_mask_densification_gain": float(
            source_patch_source_mask_densified_texel_count
            / max(source_patch_source_mask_observed_texel_count, 1)
        ),
        "source_patch_mean_source_mask_densification_gain": float(
            np.mean(source_patch_source_mask_densification_gains)
            if source_patch_source_mask_densification_gains
            else 0.0
        ),
        "source_patch_localization_candidate_texel_count": int(
            source_patch_localization_candidate_texel_count
        ),
        "source_patch_localization_selected_texel_count": int(
            source_patch_localization_selected_texel_count
        ),
        "source_patch_localization_fraction": float(
            source_patch_localization_selected_texel_count
            / max(source_patch_localization_candidate_texel_count, 1)
        ),
        "source_patch_mean_localization_fraction": float(
            np.mean(source_patch_localization_fractions)
            if source_patch_localization_fractions
            else 0.0
        ),
        "source_patch_landmark_aligned_count": int(source_patch_landmark_aligned_count),
        "source_patch_mean_landmark_confidence": float(
            np.mean(source_patch_landmark_confidences) if source_patch_landmark_confidences else 0.0
        ),
        "source_patch_observed_anchor_guided_source": (_OBSERVED_FACE_ANCHOR_SOURCE_PATCH_VERSION),
        "source_patch_observed_anchor_guided_count": int(source_patch_observed_anchor_guided_count),
        "source_patch_observed_anchor_guided_fraction": float(
            source_patch_observed_anchor_guided_count / max(source_patch_count, 1)
        ),
        "source_patch_landmark_destination_scaled_count": int(
            source_patch_landmark_destination_scaled_count
        ),
        "source_patch_mean_landmark_destination_scale": float(
            np.mean(source_patch_landmark_destination_scales)
            if source_patch_landmark_destination_scales
            else 1.0
        ),
        "source_patch_core_texel_count": int(source_patch_core_texel_count),
        "source_patch_core_fraction": float(
            source_patch_core_texel_count / max(np.count_nonzero(source_patch_texels), 1)
        ),
        "source_patch_feathered_edge_texel_count": int(source_patch_feathered_edge_texel_count),
        "source_patch_feathered_edge_fraction": float(
            source_patch_feathered_edge_texel_count / max(np.count_nonzero(source_patch_texels), 1)
        ),
        "source_patch_mean_blend_alpha": float(
            source_patch_alpha_sum / max(source_patch_alpha_texel_count, 1)
        ),
        "source_patch_texel_count": int(np.count_nonzero(source_patch_texels)),
        "source_patch_texel_fraction": float(
            np.count_nonzero(source_patch_texels) / max(np.count_nonzero(painted_texels), 1)
        ),
        **destination_anchor_metrics,
        "contrast": _facial_feature_contrast(features),
        "features": features,
    }


def _raw_face_texture_bake_selection_metrics(
    candidate_reports: list[dict[str, object]],
    *,
    selected_label: str,
    selected_report: dict[str, object],
) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    for report in candidate_reports:
        label = str(report.get("label", ""))
        chosen = label == selected_label
        reports.append({**report, "selected": chosen})
    return {
        "source_patch_bake_selection_source": (
            "generic_vs_feedback_candidate_diagnostic_v1"
            if len(candidate_reports) > 1
            else "single_candidate_diagnostic_v1"
        ),
        "source_patch_bake_candidate_count": int(len(candidate_reports)),
        "source_patch_bake_selected_candidate": selected_label,
        "source_patch_bake_selected_feedback": selected_label == "feedback",
        "source_patch_bake_visual_quality_passed_count": int(
            selected_report.get("visual_quality_passed_count", 0)
        ),
        "source_patch_bake_visual_quality_min_score": float(
            selected_report.get("visual_quality_min_score", 0.0)
        ),
        "source_patch_bake_visual_quality_mean_score": float(
            selected_report.get("visual_quality_mean_score", 0.0)
        ),
        "source_patch_bake_candidates": reports,
    }


def _raw_face_texture_bake_candidate_report(
    label: str,
    basecolor: NDArray[np.uint8],
    confidence: NDArray[np.uint8],
    facial_features: dict[str, object],
) -> dict[str, object]:
    features_raw = facial_features.get("features", [])
    features = [item for item in features_raw if isinstance(item, dict)]
    features_by_name = {str(item.get("name", "")): item for item in features}
    region_reports = []
    scores: list[float] = []
    confident_fractions: list[float] = []
    unique_counts: list[int] = []
    source_luma_values: list[float] = []
    texture_size = int(basecolor.shape[0])
    for region_name, feature_names in _FACE_TEXTURE_BAKE_REGION_FEATURES.items():
        region_features = [
            features_by_name[name] for name in feature_names if name in features_by_name
        ]
        mask = _raw_face_texture_bake_region_mask(
            region_features,
            texture_size=texture_size,
        )
        texture_metrics = _raw_face_texture_bake_texture_metrics(
            basecolor,
            confidence,
            mask,
        )
        source_metrics = _raw_face_texture_bake_source_metrics(region_features)
        quality = _raw_face_texture_bake_visual_quality(
            texture_metrics,
            source_metrics,
        )
        scores.append(float(quality["score"]))
        confident_fractions.append(float(texture_metrics["confident_texel_fraction"]))
        unique_counts.append(int(texture_metrics["unique_rgb_count"]))
        source_luma_values.append(float(source_metrics["source_luma_std"]))
        region_reports.append(
            {
                "name": region_name,
                "score": round(float(quality["score"]), 6),
                "passed": bool(quality["passed"]),
                "weakest_component": str(quality["weakest_component"]),
                "confident_texel_fraction": round(
                    float(texture_metrics["confident_texel_fraction"]),
                    6,
                ),
                "unique_rgb_count": int(texture_metrics["unique_rgb_count"]),
                "texture_luma_std": round(float(texture_metrics["luma_std"]), 6),
                "source_luma_std": round(float(source_metrics["source_luma_std"]), 6),
                "source_patch_used": bool(source_metrics["source_patch_used"]),
            }
        )

    patch_quality_scores = [
        float(item.get("source_patch_quality_score", 0.0))
        for item in features
        if bool(item.get("source_patch_used", False))
    ]
    feedback_used_count = sum(
        1 for item in features if bool(item.get("source_patch_feedback_used", False))
    )
    passed_count = sum(1 for item in region_reports if bool(item["passed"]))
    return {
        "label": label,
        "region_count": len(region_reports),
        "visual_quality_passed_count": int(passed_count),
        "visual_quality_min_score": round(float(min(scores, default=0.0)), 6),
        "visual_quality_mean_score": round(
            float(np.mean(scores)) if scores else 0.0,
            6,
        ),
        "min_confident_texel_fraction": round(
            float(min(confident_fractions, default=0.0)),
            6,
        ),
        "min_unique_rgb_count": int(min(unique_counts, default=0)),
        "mean_source_luma_std": round(
            float(np.mean(source_luma_values)) if source_luma_values else 0.0,
            6,
        ),
        "mean_source_patch_quality_score": round(
            float(np.mean(patch_quality_scores)) if patch_quality_scores else 0.0,
            6,
        ),
        "feedback_used_count": int(feedback_used_count),
        "regions": region_reports,
    }


def _raw_face_texture_bake_candidate_rank(
    report: dict[str, object],
) -> tuple[float, ...]:
    return (
        float(report.get("visual_quality_passed_count", 0.0)),
        float(report.get("visual_quality_min_score", 0.0)),
        float(report.get("visual_quality_mean_score", 0.0)),
        float(report.get("min_confident_texel_fraction", 0.0)),
        float(report.get("min_unique_rgb_count", 0.0)),
        float(report.get("mean_source_luma_std", 0.0)),
        float(report.get("mean_source_patch_quality_score", 0.0)),
    )


def _raw_face_texture_bake_region_mask(
    features: list[dict[str, object]],
    *,
    texture_size: int,
) -> NDArray[np.bool_]:
    mask = np.zeros((texture_size, texture_size), dtype=np.bool_)
    for feature in features:
        uv = feature.get("uv", [])
        radius_uv = feature.get("radius_uv", [])
        if not isinstance(uv, list) or not isinstance(radius_uv, list):
            continue
        if len(uv) != 2 or len(radius_uv) != 2:
            continue
        mask |= _ellipse_mask(
            uv=(float(uv[0]), float(uv[1])),
            radius_uv=(float(radius_uv[0]), float(radius_uv[1])),
            texture_size=texture_size,
        )
    return mask


def _raw_face_texture_bake_texture_metrics(
    basecolor: NDArray[np.uint8],
    confidence: NDArray[np.uint8],
    mask: NDArray[np.bool_],
) -> dict[str, float | int]:
    if not np.any(mask):
        return {
            "unique_rgb_count": 0,
            "luma_std": 0.0,
            "confident_texel_fraction": 0.0,
        }
    pixels = basecolor[mask]
    unique_count = int(np.unique(pixels.reshape(-1, 3), axis=0).shape[0])
    luma_std = _source_patch_luma_std(pixels.reshape(-1, 1, 3))
    confident_fraction = float(np.count_nonzero(confidence[mask]) / max(pixels.shape[0], 1))
    return {
        "unique_rgb_count": unique_count,
        "luma_std": luma_std,
        "confident_texel_fraction": confident_fraction,
    }


def _raw_face_texture_bake_source_metrics(
    features: list[dict[str, object]],
) -> dict[str, float | bool]:
    used = [item for item in features if bool(item.get("source_patch_used", False))]
    source_luma = [float(item.get("source_patch_luma_std", 0.0)) for item in used]
    return {
        "source_patch_used": bool(used) and len(used) == len(features),
        "source_luma_std": float(max(source_luma, default=0.0)),
    }


def _raw_face_texture_bake_visual_quality(
    texture_metrics: dict[str, float | int],
    source_metrics: dict[str, float | bool],
) -> dict[str, float | bool | str]:
    unique_count = int(texture_metrics["unique_rgb_count"])
    texture_luma_std = float(texture_metrics["luma_std"])
    confident_fraction = float(texture_metrics["confident_texel_fraction"])
    source_luma_std = float(source_metrics["source_luma_std"])
    source_patch_used = bool(source_metrics["source_patch_used"])
    components = {
        "confident_texel_score": _ratio_score(
            confident_fraction,
            _FACE_TEXTURE_BAKE_TARGET_CONFIDENT_TEXEL_FRACTION,
        ),
        "texture_luma_score": _ratio_score(
            texture_luma_std,
            _FACE_TEXTURE_BAKE_TARGET_TEXTURE_LUMA_STD,
        ),
        "unique_rgb_score": _unique_rgb_score(
            unique_count,
            target=_FACE_TEXTURE_BAKE_TARGET_UNIQUE_RGB_COUNT,
        ),
        "source_luma_score": _ratio_score(
            source_luma_std,
            _FACE_TEXTURE_BAKE_TARGET_SOURCE_LUMA_STD,
        ),
        "source_evidence_score": 1.0 if source_patch_used else 0.0,
    }
    score = (
        0.25 * components["confident_texel_score"]
        + 0.25 * components["texture_luma_score"]
        + 0.20 * components["unique_rgb_score"]
        + 0.20 * components["source_luma_score"]
        + 0.10 * components["source_evidence_score"]
    )
    passed = (
        score >= _FACE_TEXTURE_BAKE_MIN_SCORE
        and confident_fraction >= _FACE_TEXTURE_BAKE_MIN_CONFIDENT_TEXEL_FRACTION
        and unique_count >= _FACE_TEXTURE_BAKE_MIN_UNIQUE_RGB_COUNT
        and texture_luma_std >= _FACE_TEXTURE_BAKE_MIN_TEXTURE_LUMA_STD
        and source_luma_std >= _FACE_TEXTURE_BAKE_MIN_SOURCE_LUMA_STD
        and source_patch_used
    )
    return {
        "score": float(score),
        "passed": bool(passed),
        "weakest_component": min(
            components.items(),
            key=lambda item: (item[1], item[0]),
        )[0],
    }


def _ratio_score(value: float, target: float) -> float:
    return float(np.clip(value / max(target, 1e-12), 0.0, 1.0))


def _unique_rgb_score(value: int, *, target: int) -> float:
    if value <= 1:
        return 0.0
    return float(np.clip(np.log2(float(value)) / np.log2(float(target)), 0.0, 1.0))


def _prepare_face_source_patch_dir(path: Path, *, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not force:
        return
    for child in path.glob("*.png"):
        child.unlink()


def _write_feature_source_patch(
    source_patch_dir: Path,
    *,
    feature_name: str,
    patch: NDArray[np.uint8],
) -> Path:
    source_patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = source_patch_dir / f"{feature_name}.png"
    Image.fromarray(patch).save(patch_path)
    return patch_path


def _write_feature_source_patch_mask(
    source_patch_dir: Path,
    *,
    feature_name: str,
    patch_mask: NDArray[np.uint8],
) -> Path:
    source_patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = source_patch_dir / f"{feature_name}_mask.png"
    Image.fromarray(patch_mask).save(patch_path)
    return patch_path


def _write_feature_uv_support_mask(
    source_patch_dir: Path,
    *,
    feature_name: str,
    support_mask: NDArray[np.bool_],
) -> Path:
    source_patch_dir.mkdir(parents=True, exist_ok=True)
    mask_path = source_patch_dir / f"{feature_name}_uv_support.png"
    mask_image = np.asarray(support_mask, dtype=np.bool_).astype(np.uint8) * np.uint8(255)
    Image.fromarray(mask_image).save(mask_path)
    return mask_path


def _feature_source_view_id(
    source_frame_id: str,
    observations: tuple[FitObservation, ...],
) -> int:
    for index, observation in enumerate(observations):
        if observation.frame_id == source_frame_id:
            return min(index + 1, 255)
    return 0


def _source_patch_footprint(
    sample_key: str,
    *,
    nominal_radius_uv: tuple[float, float],
    patch: NDArray[np.uint8] | None,
    sample_count: int,
    used_source_patch: bool,
    landmark_confidence: float = 0.0,
) -> _PatchFootprint:
    radius_u, radius_v = (float(nominal_radius_uv[0]), float(nominal_radius_uv[1]))
    if not used_source_patch or patch is None or patch.size == 0 or sample_count <= 0:
        return _PatchFootprint(
            radius_uv=(radius_u, radius_v),
            density=0.0,
            radius_scale=1.0,
            area_scale=1.0,
            x_scale=1.0,
            y_scale=1.0,
            landmark_destination_scale=1.0,
        )

    height, width = patch.shape[:2]
    density = float(sample_count / max(float(width * height), 1.0))
    profile = _source_patch_footprint_profile(sample_key)
    radius_scale = float(
        np.clip(
            np.sqrt(density / max(profile["target_density"], 1e-12)),
            profile["min_scale"],
            profile["max_scale"],
        )
    )
    patch_aspect = max(float(width) / max(float(height), 1.0), 1e-6)
    nominal_aspect = max(radius_u / max(radius_v, 1e-12), 1e-6)
    aspect_scale = float(np.clip(np.sqrt(patch_aspect / nominal_aspect), 0.72, 1.28))
    landmark_destination_scale = _source_patch_landmark_destination_scale(
        sample_key,
        landmark_confidence,
    )
    x_scale = float(np.clip(radius_scale * aspect_scale * landmark_destination_scale, 0.22, 1.0))
    y_scale = float(np.clip(radius_scale / aspect_scale * landmark_destination_scale, 0.22, 1.0))
    return _PatchFootprint(
        radius_uv=(max(radius_u * x_scale, 1e-6), max(radius_v * y_scale, 1e-6)),
        density=density,
        radius_scale=radius_scale,
        area_scale=x_scale * y_scale,
        x_scale=x_scale,
        y_scale=y_scale,
        landmark_destination_scale=landmark_destination_scale,
    )


def _source_patch_destination_sample_count(
    sample_key: str,
    *,
    sample_count: int,
    source_mask_observed_texel_count: int,
) -> int:
    if sample_key in _IRIS_SOURCE_LUMA_CORE_KEYS and source_mask_observed_texel_count > 0:
        return int(source_mask_observed_texel_count)
    return int(max(sample_count, 0))


def _source_patch_footprint_profile(sample_key: str) -> dict[str, float]:
    raw = _SOURCE_PATCH_FOOTPRINT_PROFILES.get(
        sample_key,
        {"target_density": 0.070, "min_scale": 0.55, "max_scale": 0.90},
    )
    return {
        "target_density": float(raw["target_density"]),
        "min_scale": float(raw["min_scale"]),
        "max_scale": float(raw["max_scale"]),
    }


def _source_patch_landmark_destination_scale(
    sample_key: str,
    landmark_confidence: float,
) -> float:
    confidence = float(np.clip(landmark_confidence, 0.0, 1.0))
    if confidence <= 0.0:
        return 1.0
    target_scale = float(_SOURCE_PATCH_LANDMARK_DESTINATION_SCALES.get(sample_key, 1.0))
    return float(np.clip(1.0 - confidence * (1.0 - target_scale), 0.18, 1.0))


def _source_shape_mask_from_local_mask(
    local_mask: NDArray[np.bool_],
    *,
    sample_key: str,
) -> NDArray[np.bool_]:
    mask = np.asarray(local_mask, dtype=np.bool_)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros(mask.shape, dtype=np.bool_)
    if sample_key in _COMPACT_EYE_SOURCE_SHAPE_KEYS:
        return _compact_eye_source_shape_mask(mask, sample_key=sample_key)
    if sample_key in _BROAD_EYE_SOURCE_SHAPE_KEYS:
        compact_eye = _broad_eye_source_shape_mask(mask)
        if compact_eye is not None:
            return compact_eye
    feature_width = int(xs.max() - xs.min() + 1)
    feature_height = int(ys.max() - ys.min() + 1)
    profile = _source_patch_shape_profile(sample_key)
    dilation_radius = int(
        np.clip(
            round(min(feature_width, feature_height) * profile["dilation_fraction"]),
            int(profile["min_radius"]),
            int(profile["max_radius"]),
        )
    )
    filter_size = dilation_radius * 2 + 1
    if mask.size > _SOURCE_SHAPE_MASK_EXACT_MAX_PIXELS:
        return _bounded_source_shape_morphology(mask, dilation_radius=dilation_radius)
    return _source_shape_morphology(mask, filter_size=filter_size, closing_radius=dilation_radius)


def _source_shape_morphology(
    mask: NDArray[np.bool_],
    *,
    filter_size: int,
    closing_radius: int,
) -> NDArray[np.bool_]:
    mask_image = Image.fromarray(mask.astype(np.uint8) * np.uint8(255))
    dilated_image = mask_image.filter(ImageFilter.MaxFilter(filter_size))
    closed_image = dilated_image.filter(ImageFilter.MinFilter(max(3, closing_radius | 1)))
    dilated = (
        np.asarray(
            closed_image.filter(ImageFilter.MaxFilter(3)),
            dtype=np.uint8,
        )
        > 0
    )
    return dilated


def _bounded_source_shape_morphology(
    mask: NDArray[np.bool_],
    *,
    dilation_radius: int,
) -> NDArray[np.bool_]:
    stride = int(max(1, np.ceil(np.sqrt(mask.size / _SOURCE_SHAPE_MASK_EXACT_MAX_PIXELS))))
    sampled = np.asarray(mask[::stride, ::stride], dtype=np.bool_)
    sampled_radius = max(1, int(np.ceil(float(dilation_radius) / float(stride))))
    sampled_dilated = _source_shape_morphology(
        sampled,
        filter_size=sampled_radius * 2 + 1,
        closing_radius=sampled_radius,
    )
    upsampled = np.repeat(
        np.repeat(sampled_dilated, stride, axis=0),
        stride,
        axis=1,
    )[: mask.shape[0], : mask.shape[1]]
    return np.asarray(upsampled | mask, dtype=np.bool_)


def _source_observed_mask_from_local_mask(
    crop: NDArray[np.float32],
    local_mask: NDArray[np.bool_],
    *,
    sample_key: str,
) -> NDArray[np.bool_]:
    mask = np.asarray(local_mask, dtype=np.bool_)
    if sample_key not in _IRIS_SOURCE_LUMA_CORE_KEYS:
        return mask
    return _iris_source_luma_core_mask(crop, mask)


def _iris_source_luma_core_mask(
    crop: NDArray[np.float32],
    local_mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    mask = np.asarray(local_mask, dtype=np.bool_)
    observed_count = int(np.count_nonzero(mask))
    if observed_count < 8:
        return mask
    luma = _rgb_luma(np.asarray(crop, dtype=np.float32))
    values = luma[mask]
    if values.size == 0:
        return mask
    low_luma = float(np.quantile(values, 0.10))
    core_cutoff = min(float(np.quantile(values, 0.80)), low_luma + 0.18)
    core = mask & (luma <= core_cutoff)
    min_core_count = max(4, int(round(observed_count * 0.25)))
    if int(np.count_nonzero(core)) < min_core_count:
        return mask
    component = _largest_connected_component_mask(core)
    if int(np.count_nonzero(component)) < min_core_count:
        return core
    return component


def _compact_eye_source_shape_mask(
    mask: NDArray[np.bool_],
    *,
    sample_key: str,
) -> NDArray[np.bool_]:
    component = _largest_connected_component_mask(mask)
    compact = _elongated_eye_component_mask(
        component,
        radius_x_multiplier=0.92,
        radius_y_multiplier=0.62,
        max_width_fraction=0.30,
        max_height_fraction=0.20,
    )
    if compact is not None:
        return compact
    mask_image = Image.fromarray(component.astype(np.uint8) * np.uint8(255))
    return np.asarray(mask_image.filter(ImageFilter.MaxFilter(3)), dtype=np.uint8) > 0


def _broad_eye_source_shape_mask(
    mask: NDArray[np.bool_],
) -> NDArray[np.bool_] | None:
    component = _largest_connected_component_mask(mask)
    return _elongated_eye_component_mask(
        component,
        radius_x_multiplier=1.35,
        radius_y_multiplier=0.82,
        max_width_fraction=0.36,
        max_height_fraction=0.26,
    )


def _elongated_eye_component_mask(
    component: NDArray[np.bool_],
    *,
    radius_x_multiplier: float,
    radius_y_multiplier: float,
    max_width_fraction: float,
    max_height_fraction: float,
) -> NDArray[np.bool_] | None:
    ys, xs = np.nonzero(component)
    if xs.size < 8:
        return None
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    aspect = float(width) / max(float(height), 1.0)
    elongated_vertical = aspect < 0.72 and height >= 12
    elongated_horizontal = aspect > 3.20 and width >= 12
    if not elongated_vertical and not elongated_horizontal:
        return None

    area = float(xs.size)
    radius = max(float(np.sqrt(area / np.pi)), 1.0)
    center_x = float(np.median(xs))
    center_y = float(np.median(ys))
    radius_x = min(
        max(radius * float(radius_x_multiplier), 2.0),
        max(float(width) * float(max_width_fraction), 2.0),
    )
    radius_y = min(
        max(radius * float(radius_y_multiplier), 2.0),
        max(float(height) * float(max_height_fraction), 2.0),
    )
    grid_y, grid_x = np.indices(component.shape, dtype=np.float64)
    compact = (
        ((grid_x - center_x) / max(radius_x, 1e-6)) ** 2
        + ((grid_y - center_y) / max(radius_y, 1e-6)) ** 2
    ) <= 1.0
    mask_image = Image.fromarray(compact.astype(np.uint8) * np.uint8(255))
    return np.asarray(mask_image.filter(ImageFilter.MaxFilter(3)), dtype=np.uint8) > 0


def _largest_connected_component_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    source = np.asarray(mask, dtype=np.bool_)
    visited = np.zeros(source.shape, dtype=np.bool_)
    best_component: list[tuple[int, int]] = []
    height, width = source.shape
    for start_y, start_x in zip(*np.nonzero(source), strict=True):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        component: list[tuple[int, int]] = []
        while stack:
            current_y, current_x = stack.pop()
            component.append((current_y, current_x))
            for neighbor_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                for neighbor_x in range(
                    max(0, current_x - 1),
                    min(width, current_x + 2),
                ):
                    if source[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                        visited[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))
        if len(component) > len(best_component):
            best_component = component
    component_mask = np.zeros(source.shape, dtype=np.bool_)
    for y, x in best_component:
        component_mask[y, x] = True
    return component_mask


def _source_patch_shape_profile(sample_key: str) -> dict[str, float | int]:
    raw = _SOURCE_PATCH_SHAPE_PROFILES.get(
        sample_key,
        {"dilation_fraction": 0.14, "min_radius": 2, "max_radius": 14},
    )
    return {
        "dilation_fraction": float(raw["dilation_fraction"]),
        "min_radius": int(raw["min_radius"]),
        "max_radius": int(raw["max_radius"]),
    }


def _paint_patch_into_mask(
    basecolor: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    patch: NDArray[np.uint8],
    *,
    sample_key: str,
    source_shape_mask: NDArray[np.uint8] | None,
    fusion_weight: float,
    feather_width: float,
    feather_x_scale: float,
    feather_y_scale: float,
) -> _PatchPaintStats:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return _PatchPaintStats(
            painted_texel_count=0,
            weighted_texel_count=0.0,
            core_texel_count=0,
            feathered_edge_texel_count=0,
            mean_alpha=0.0,
            shape_mask_texel_count=0,
            shape_alpha_mean=0.0,
            applied_mask=np.zeros_like(mask),
        )
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    patch_image = Image.fromarray(patch).resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )
    patch_resized = np.asarray(patch_image, dtype=np.uint8)
    source_alpha = _resized_source_shape_alpha(source_shape_mask, width=width, height=height)
    local_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    region = basecolor[y0 : y1 + 1, x0 : x1 + 1]
    alpha = float(np.clip(fusion_weight, 0.0, 1.0))
    feather = _source_patch_feather_alpha(
        local_mask,
        feather_width=feather_width,
        x_scale=feather_x_scale,
        y_scale=feather_y_scale,
    )
    alpha_map = alpha * feather * source_alpha
    effective_local_mask = local_mask & (alpha_map > 1e-6)
    (
        effective_local_mask,
        edge_clamp_removed_texel_count,
        edge_clamp_max_ratio,
    ) = _clamp_iris_feathered_edge_support(
        effective_local_mask,
        sample_key=sample_key,
        feather=feather,
        source_alpha=source_alpha,
    )
    local_alpha = alpha_map[effective_local_mask]
    base_pixels = region[effective_local_mask].astype(np.float64)
    patch_pixels = patch_resized[effective_local_mask].astype(np.float64)
    fused = (1.0 - local_alpha[:, None]) * base_pixels + local_alpha[:, None] * patch_pixels
    region[effective_local_mask] = np.clip(np.rint(fused), 0, 255).astype(np.uint8)
    applied_mask = np.zeros_like(mask)
    applied_mask[y0 : y1 + 1, x0 : x1 + 1] = effective_local_mask
    shaped_source_mask = local_mask & (source_alpha > 1e-6)
    return _PatchPaintStats(
        painted_texel_count=int(np.count_nonzero(effective_local_mask)),
        weighted_texel_count=float(np.sum(local_alpha)),
        core_texel_count=int(
            np.count_nonzero(effective_local_mask & (feather >= 0.999) & (source_alpha >= 0.95))
        ),
        feathered_edge_texel_count=int(
            np.count_nonzero(effective_local_mask & ((feather < 0.999) | (source_alpha < 0.95)))
        ),
        mean_alpha=float(np.mean(local_alpha)) if local_alpha.size else 0.0,
        shape_mask_texel_count=int(np.count_nonzero(shaped_source_mask)),
        shape_alpha_mean=float(np.mean(source_alpha[effective_local_mask]))
        if np.any(effective_local_mask)
        else 0.0,
        applied_mask=applied_mask,
        edge_clamp_removed_texel_count=edge_clamp_removed_texel_count,
        edge_clamp_max_ratio=edge_clamp_max_ratio,
    )


def _clamp_iris_feathered_edge_support(
    effective_local_mask: NDArray[np.bool_],
    *,
    sample_key: str,
    feather: NDArray[np.float64],
    source_alpha: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], int, float]:
    if sample_key not in _IRIS_SOURCE_LUMA_CORE_KEYS:
        return effective_local_mask, 0, 0.0
    core_mask = effective_local_mask & (feather >= 0.999) & (source_alpha >= 0.95)
    edge_mask = effective_local_mask & ~core_mask
    core_count = int(np.count_nonzero(core_mask))
    edge_count = int(np.count_nonzero(edge_mask))
    max_edge_to_core_ratio = _iris_max_feathered_edge_to_core_ratio(
        core_count=core_count,
    )
    if core_count <= 0 or edge_count <= 1:
        return effective_local_mask, 0, max_edge_to_core_ratio
    max_edge_count = max(
        1,
        int(np.ceil(core_count * max_edge_to_core_ratio)),
    )
    if edge_count <= max_edge_count:
        return effective_local_mask, 0, max_edge_to_core_ratio

    edge_indices = np.flatnonzero(edge_mask.ravel())
    edge_scores = (feather * source_alpha).ravel()[edge_indices]
    ordered = np.lexsort((edge_indices, -edge_scores))
    keep_indices = edge_indices[ordered[:max_edge_count]]
    clamped = np.array(effective_local_mask, dtype=np.bool_, copy=True)
    clamped_flat = clamped.ravel()
    clamped_flat[edge_indices] = False
    clamped_flat[keep_indices] = True
    return clamped, edge_count - max_edge_count, max_edge_to_core_ratio


def _iris_max_feathered_edge_to_core_ratio(*, core_count: int | None = None) -> float:
    raw = os.environ.get(_IRIS_EDGE_CORE_RATIO_ENV)
    if raw is None:
        if core_count is not None and core_count <= _IRIS_COMPACT_CORE_TEXEL_COUNT:
            return min(
                _IRIS_MAX_FEATHERED_EDGE_TO_CORE_RATIO,
                _IRIS_COMPACT_MAX_FEATHERED_EDGE_TO_CORE_RATIO,
            )
        return _IRIS_MAX_FEATHERED_EDGE_TO_CORE_RATIO
    try:
        value = float(raw)
    except ValueError:
        return _IRIS_MAX_FEATHERED_EDGE_TO_CORE_RATIO
    return float(np.clip(value, 0.25, 4.0))


def _feature_visibility_reinforcement_alpha(
    *,
    feature_name: str,
    basecolor: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    color: NDArray[np.uint8],
) -> float:
    if not np.any(mask):
        return 0.0
    sampled = basecolor[mask].astype(np.float64)
    mean_rgb = np.mean(sampled, axis=0)
    target_rgb = np.asarray(color, dtype=np.float64)
    distance = _rgb_distance_255(mean_rgb, target_rgb)
    if feature_name in {"left_eye_iris", "right_eye_iris"}:
        if distance <= 0.18:
            return 0.0
        return 0.72
    if feature_name in {"left_eye_pupil", "right_eye_pupil"}:
        if distance <= 0.10:
            return 0.0
        return 0.82
    if feature_name == "mouth_line":
        if distance <= 0.25:
            return 0.0
        return 0.88
    return 0.0


def _eye_source_detail_reinforcement_alpha(
    feature_name: str,
    alpha: float,
    *,
    source_patch_luma_std: float = 0.0,
    source_patch_unique_rgb_count: int = 0,
) -> float:
    min_alpha = 0.0
    if (
        source_patch_luma_std >= _FACE_TEXTURE_BAKE_MIN_SOURCE_LUMA_STD
        and source_patch_unique_rgb_count >= _FACE_TEXTURE_BAKE_MIN_UNIQUE_RGB_COUNT
    ):
        min_alpha = _EYE_SOURCE_DETAIL_REINFORCEMENT_MIN_ALPHA.get(feature_name, 0.0)
    return float(
        min(
            max(float(alpha), float(min_alpha)),
            _EYE_SOURCE_DETAIL_REINFORCEMENT_MAX_ALPHA.get(feature_name, float(alpha)),
        )
    )


def _paint_feature_visibility_reinforcement(
    basecolor: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    color: NDArray[np.uint8],
    *,
    alpha: float,
    preserve_detail: bool = False,
) -> NDArray[np.bool_]:
    applied_mask = np.zeros_like(mask)
    if not np.any(mask):
        return applied_mask
    alpha = float(np.clip(alpha, 0.0, 1.0))
    pixels = basecolor[mask].astype(np.float64)
    target = np.asarray(color, dtype=np.float64)
    if preserve_detail and pixels.size:
        mean_rgb = np.mean(pixels, axis=0)
        target_mean = (1.0 - alpha) * mean_rgb + alpha * target
        detail = pixels - mean_rgb[None, :]
        fused = target_mean[None, :] + detail
    else:
        fused = (1.0 - alpha) * pixels + alpha * target[None, :]
    basecolor[mask] = np.clip(np.rint(fused), 0, 255).astype(np.uint8)
    applied_mask[mask] = True
    return applied_mask


def _resized_source_shape_alpha(
    source_shape_mask: NDArray[np.uint8] | None,
    *,
    width: int,
    height: int,
) -> NDArray[np.float64]:
    if source_shape_mask is None or source_shape_mask.size == 0:
        return np.ones((height, width), dtype=np.float64)
    mask_image = Image.fromarray(np.asarray(source_shape_mask, dtype=np.uint8)).resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )
    alpha = np.asarray(mask_image, dtype=np.float64) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def _source_patch_fusion_weight(quality_score: float) -> float:
    normalized = (float(quality_score) - _MIN_SOURCE_PATCH_QUALITY_SCORE) / max(
        1.0 - _MIN_SOURCE_PATCH_QUALITY_SCORE, 1e-12
    )
    normalized = float(np.clip(normalized, 0.0, 1.0))
    curved = normalized**0.75
    return float(
        _MIN_SOURCE_PATCH_FUSION_WEIGHT
        + (_MAX_SOURCE_PATCH_FUSION_WEIGHT - _MIN_SOURCE_PATCH_FUSION_WEIGHT) * curved
    )


def _source_patch_feather_width(sample_key: str) -> float:
    return float(_source_patch_feather_profile(sample_key)["edge_width"])


def _source_patch_feather_profile(sample_key: str) -> dict[str, float]:
    raw = _SOURCE_PATCH_FEATHER_PROFILES.get(
        sample_key,
        {"edge_width": 0.20, "x_scale": 1.0, "y_scale": 0.90},
    )
    return {
        "edge_width": float(raw["edge_width"]),
        "x_scale": float(raw["x_scale"]),
        "y_scale": float(raw["y_scale"]),
    }


def _source_patch_feather_alpha(
    local_mask: NDArray[np.bool_],
    *,
    feather_width: float,
    x_scale: float,
    y_scale: float,
) -> NDArray[np.float64]:
    height, width = local_mask.shape
    if height <= 0 or width <= 0:
        return np.zeros(local_mask.shape, dtype=np.float64)
    width_norm = float(np.clip(feather_width, 0.05, 0.95))
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)
    rx = max(0.5 * float(width) * float(np.clip(x_scale, 0.50, 1.50)), 1.0)
    ry = max(0.5 * float(height) * float(np.clip(y_scale, 0.50, 1.50)), 1.0)
    ys = np.arange(height, dtype=np.float64)
    xs = np.arange(width, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    radial = np.sqrt(((grid_x - cx) / rx) ** 2 + ((grid_y - cy) / ry) ** 2)
    alpha = np.clip((1.0 - radial) / width_norm, 0.0, 1.0)
    alpha[~local_mask] = 0.0
    return alpha


def _empty_face_feature_contrast() -> dict[str, object]:
    return {
        "source": "none",
        "eye_to_muzzle_luma_delta": 0.0,
        "mouth_to_muzzle_luma_delta": 0.0,
        "nose_to_muzzle_rgb_distance": 0.0,
        "iris_to_pupil_rgb_distance": 0.0,
        "min_required_contrast": 0.0,
        "required_contrast_passed": False,
    }


def _facial_feature_contrast(features: list[dict[str, object]]) -> dict[str, object]:
    colors = {
        str(feature.get("name")): np.asarray(feature.get("rgb", [0, 0, 0]), dtype=np.float64)
        for feature in features
        if isinstance(feature.get("rgb"), list)
    }
    muzzle = _mean_feature_color(colors, ("left_muzzle", "right_muzzle"))
    eye_core = _mean_feature_color(
        colors,
        ("left_eye_pupil", "right_eye_pupil", "left_eye_core", "right_eye_core"),
    )
    iris = _mean_feature_color(colors, ("left_eye_iris", "right_eye_iris"))
    nose = _mean_feature_color(colors, ("nose",))
    mouth = _mean_feature_color(colors, ("mouth_line",))

    eye_to_muzzle = max(0.0, _rgb_luma_255(muzzle) - _rgb_luma_255(eye_core))
    mouth_to_muzzle = max(0.0, _rgb_luma_255(muzzle) - _rgb_luma_255(mouth))
    nose_to_muzzle = _rgb_distance_255(nose, muzzle)
    iris_to_pupil = _rgb_distance_255(iris, eye_core)
    min_required = min(eye_to_muzzle, mouth_to_muzzle, nose_to_muzzle)
    return {
        "source": "observed_texture_face_feature_contrast_v1",
        "eye_to_muzzle_luma_delta": round(float(eye_to_muzzle), 6),
        "mouth_to_muzzle_luma_delta": round(float(mouth_to_muzzle), 6),
        "nose_to_muzzle_rgb_distance": round(float(nose_to_muzzle), 6),
        "iris_to_pupil_rgb_distance": round(float(iris_to_pupil), 6),
        "min_required_contrast": round(float(min_required), 6),
        "required_contrast_passed": bool(min_required >= 0.18),
    }


def _mean_feature_color(
    colors: dict[str, NDArray[np.float64]],
    names: tuple[str, ...],
) -> NDArray[np.float64]:
    selected = [colors[name] for name in names if name in colors]
    if not selected:
        return np.zeros(3, dtype=np.float64)
    return np.mean(np.stack(selected, axis=0), axis=0)


def _rgb_luma_255(rgb: NDArray[np.float64]) -> float:
    return float((0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0)


def _rgb_distance_255(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(np.linalg.norm((a - b) / 255.0))


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _metadata_float(
    metadata: dict[str, Any],
    quality: dict[str, Any],
    key: str,
) -> float | None:
    raw = metadata.get(key, quality.get(key))
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _raw_texture_view_quality(
    observation: FitObservation,
    metadata: dict[str, Any],
) -> _RawTextureViewQuality:
    quality_raw = metadata.get("quality", {})
    quality = quality_raw if isinstance(quality_raw, dict) else {}
    fallback = _estimated_raw_texture_view_quality(
        observation,
        metadata=metadata,
        quality=quality,
    )
    frontal = _metadata_float(metadata, quality, "raw_frontal_face_confidence")
    view_weight = _metadata_float(metadata, quality, "raw_keypoint_view_weight")
    if frontal is None and view_weight is not None:
        frontal = (float(view_weight) - 0.35) / 0.65
    if frontal is None:
        frontal = fallback.frontal_face_confidence
    frontal = _clamp01(frontal)
    if view_weight is None:
        observation_weight = float(getattr(observation, "keypoint_weight_scale", 1.0))
        if np.isfinite(observation_weight) and not np.isclose(observation_weight, 1.0):
            view_weight = observation_weight
        else:
            view_weight = 0.35 + 0.65 * frontal
    view_weight = _clamp01(view_weight)
    raw_label = metadata.get("raw_view_label")
    label = str(raw_label) if raw_label else fallback.label
    return _RawTextureViewQuality(
        label=label,
        frontal_face_confidence=round(float(frontal), 9),
        view_weight=round(float(view_weight), 9),
    )


def _estimated_raw_texture_view_quality(
    observation: FitObservation,
    *,
    metadata: dict[str, Any],
    quality: dict[str, Any],
) -> _RawTextureViewQuality:
    mask = np.asarray(observation.hard_mask, dtype=np.float32) >= 0.5
    image_height, image_width = mask.shape
    x0, y0, x1, y1 = (float(value) for value in observation.bbox_xyxy)
    width_fraction = max(0.0, x1 - x0) / max(float(image_width), 1.0)
    height_fraction = max(0.0, y1 - y0) / max(float(image_height), 1.0)
    aspect = width_fraction / max(height_fraction, 1e-6)
    bbox_area_ratio = float(quality.get("bbox_area_ratio", width_fraction * height_fraction))
    mask_area_ratio = float(
        quality.get(
            "mask_area_ratio",
            float(np.count_nonzero(mask)) / max(float(mask.size), 1.0),
        )
    )
    fill_score = _clamp01(mask_area_ratio / max(bbox_area_ratio, 1e-6))
    center_x = float(x0 + x1) / max(2.0 * float(image_width), 1.0)
    center_y = float(y0 + y1) / max(2.0 * float(image_height), 1.0)
    center_score = _clamp01(1.0 - float(np.hypot(center_x - 0.5, center_y - 0.5)) / 0.55)
    square_score = _clamp01(1.0 - abs(float(np.log(max(aspect, 1e-6)))) / float(np.log(2.5)))
    area_score = _clamp01(
        1.0 - abs(float(np.log(max(bbox_area_ratio, 1e-6) / 0.24))) / float(np.log(4.0))
    )
    touches_border = _clamp01(float(quality.get("touches_image_border", 0.0)))
    frontal = _clamp01(
        (0.42 * square_score + 0.24 * center_score + 0.20 * fill_score + 0.14 * area_score)
        * (0.75 + 0.25 * (1.0 - touches_border))
    )
    if frontal >= 0.72:
        label = "frontal_face_candidate"
    elif aspect >= 1.55:
        label = "side_body_candidate"
    elif aspect <= 0.75:
        label = "vertical_body_candidate"
    else:
        label = "oblique_body_candidate"
    if metadata.get("media_type") == "video" or metadata.get("source_type") == "video_frame":
        label = f"video_{label}"
    return _RawTextureViewQuality(
        label=label,
        frontal_face_confidence=round(float(frontal), 9),
        view_weight=round(float(0.35 + 0.65 * frontal), 9),
    )


def _source_patch_view_weighted_rank(
    sample_key: str,
    *,
    landmark_confidence: float,
    feedback_score: float,
    quality_score: float,
    view_weight: float,
    frontal_face_confidence: float,
    feature_contrast_score: float,
    localization_selected_count: int,
    localization_candidate_count: int,
    source_density_score: float,
    source_detail_score: float,
    frontal_photo_priority: float = 0.0,
) -> float:
    view_score = _clamp01(0.55 * view_weight + 0.45 * frontal_face_confidence)
    landmark_score = _clamp01(landmark_confidence)
    feedback = _clamp01(feedback_score)
    quality = _clamp01(quality_score)
    contrast = _clamp01(feature_contrast_score)
    detail = _clamp01(source_detail_score)
    photo_priority = _clamp01(frontal_photo_priority)
    support = _source_patch_localized_support_score(
        sample_key,
        selected_count=localization_selected_count,
        candidate_count=localization_candidate_count,
    )
    density = _clamp01(source_density_score)
    view_confidence_penalty = _source_patch_view_confidence_penalty(
        sample_key,
        frontal_face_confidence=frontal_face_confidence,
        view_weight=view_weight,
    )
    if photo_priority <= 0.0:
        if sample_key == "ear_inner":
            return (
                float(
                    0.24 * detail
                    + 0.20 * quality
                    + 0.18 * feedback
                    + 0.16 * view_score
                    + 0.10 * contrast
                    + 0.08 * support
                    + 0.04 * landmark_score
                )
                * view_confidence_penalty
            )
        if sample_key in _FACE_PRIORITY_SOURCE_PATCH_KEYS:
            if sample_key in _BROAD_FACE_SOURCE_PATCH_KEYS:
                return (
                    float(
                        0.20 * quality
                        + 0.18 * landmark_score
                        + 0.18 * support
                        + 0.14 * contrast
                        + 0.12 * view_score
                        + 0.10 * density
                        + 0.08 * feedback
                    )
                    * view_confidence_penalty
                )
            return (
                float(
                    0.26 * landmark_score
                    + 0.20 * view_score
                    + 0.20 * contrast
                    + 0.15 * support
                    + 0.12 * quality
                    + 0.07 * feedback
                )
                * view_confidence_penalty
            )
        return float(0.52 * quality + 0.24 * landmark_score + 0.24 * feedback)
    if sample_key == "ear_inner":
        return (
            float(
                0.24 * detail
                + 0.10 * quality
                + 0.06 * feedback
                + 0.10 * view_score
                + 0.08 * contrast
                + 0.06 * support
                + 0.32 * photo_priority
                + 0.04 * landmark_score
            )
            * view_confidence_penalty
        )
    if sample_key in _IRIS_SOURCE_LUMA_CORE_KEYS:
        rank = float(
            0.20 * detail
            + 0.08 * quality
            + 0.10 * view_score
            + 0.08 * contrast
            + 0.12 * support
            + 0.08 * density
            + 0.50 * photo_priority
            + 0.02 * landmark_score
        )
        return min(1.0, rank) * view_confidence_penalty
    if sample_key in _PUPIL_SOURCE_DARK_CORE_KEYS:
        rank = float(
            0.12 * detail
            + 0.10 * quality
            + 0.10 * view_score
            + 0.08 * contrast
            + 0.12 * support
            + 0.06 * density
            + 0.48 * photo_priority
            + 0.04 * landmark_score
        )
        rank = min(1.0, rank + 0.28 * photo_priority)
        return min(1.0, rank) * view_confidence_penalty
    if sample_key in _FACE_PRIORITY_SOURCE_PATCH_KEYS:
        if sample_key in _BROAD_FACE_SOURCE_PATCH_KEYS:
            rank = float(
                0.18 * quality
                + 0.14 * landmark_score
                + 0.17 * support
                + 0.13 * contrast
                + 0.11 * view_score
                + 0.10 * density
                + 0.10 * photo_priority
                + 0.07 * feedback
            )
            if sample_key == "nose":
                rank = min(1.0, rank + 0.38 * photo_priority)
            elif sample_key == "muzzle":
                rank = min(1.0, rank + 0.14 * photo_priority)
            return rank * view_confidence_penalty
        return (
            float(
                0.16 * landmark_score
                + 0.17 * view_score
                + 0.18 * contrast
                + 0.13 * support
                + 0.10 * quality
                + 0.20 * photo_priority
                + 0.06 * feedback
            )
            * view_confidence_penalty
        )
    return float(0.52 * quality + 0.24 * landmark_score + 0.24 * feedback)


def _source_patch_frontal_photo_priority(
    sample_key: str,
    *,
    metadata: dict[str, Any],
    view_quality: _RawTextureViewQuality,
    patch: NDArray[np.uint8],
    selected_count: int,
    candidate_count: int,
    sample_count: int,
    source_quality_score: float,
) -> float:
    """Rank direct still-photo patches when they contain strong observed face evidence."""

    diagnostic = _source_patch_frontal_photo_priority_diagnostic(
        sample_key,
        metadata=metadata,
        view_quality=view_quality,
        patch=patch,
        selected_count=selected_count,
        candidate_count=candidate_count,
        sample_count=sample_count,
        source_quality_score=source_quality_score,
    )
    return float(diagnostic["priority"])


def _texture_only_source_patch_priority_candidate(metadata: dict[str, Any]) -> bool:
    if not bool(metadata.get("texture_only_supplemental", False)):
        return False
    if not _source_patch_candidate_allowed(metadata):
        return False
    policy = metadata.get("texture_only_face_region_projection_protection")
    policy_allowed = bool(
        isinstance(policy, dict) and bool(policy.get("source_patch_allowed", False))
    )
    return bool(policy_allowed or metadata.get("texture_only_source_patch_allowed", False))


def _texture_only_source_patch_effective_view_score(
    metadata: dict[str, Any],
    view_score: float,
) -> float:
    raw_view_score = _clamp01(view_score)
    if not _texture_only_source_patch_priority_candidate(metadata):
        return float(raw_view_score)
    try:
        preflight_score = float(metadata.get("texture_only_source_preflight_score", 0.0))
    except (TypeError, ValueError):
        preflight_score = 0.0
    gain = _clamp01((preflight_score - 1.0) / _TEXTURE_ONLY_SOURCE_PATCH_PREFLIGHT_GAIN_SPAN)
    admission_floor = min(
        _TEXTURE_ONLY_SOURCE_PATCH_MAX_VIEW_SCORE,
        _TEXTURE_ONLY_SOURCE_PATCH_BASE_VIEW_SCORE
        + _TEXTURE_ONLY_SOURCE_PATCH_PREFLIGHT_GAIN * gain,
    )
    return float(max(raw_view_score, admission_floor))


def _source_patch_frontal_photo_priority_diagnostic(
    sample_key: str,
    *,
    metadata: dict[str, Any],
    view_quality: _RawTextureViewQuality,
    patch: NDArray[np.uint8],
    selected_count: int,
    candidate_count: int,
    sample_count: int,
    source_quality_score: float,
) -> dict[str, object]:
    """Explain the direct still-photo priority gate for source patch candidates."""

    direct_photo_candidate = _source_patch_direct_uploaded_photo_candidate(metadata)
    direct_upload_score = 1.0
    raw_view_score = _clamp01(
        0.50 * view_quality.view_weight + 0.50 * view_quality.frontal_face_confidence
    )
    texture_only_source_patch_candidate = _texture_only_source_patch_priority_candidate(metadata)
    view_score = _texture_only_source_patch_effective_view_score(
        metadata,
        raw_view_score,
    )
    support = _source_patch_localized_support_score(
        sample_key,
        selected_count=selected_count,
        candidate_count=candidate_count,
    )
    density = _source_patch_density_score(
        sample_key,
        patch,
        sample_count=sample_count,
    )
    detail = _source_patch_diagnostic_detail_score(patch)
    pupil_darkness = _source_patch_pupil_darkness_score(patch)
    quality = _clamp01(source_quality_score)
    quality_min = 0.0
    detail_min = 0.0
    evidence = 0.0
    rejection_reason = ""
    if not direct_photo_candidate:
        rejection_reason = "not_direct_uploaded_photo"
    elif view_score < 0.50:
        rejection_reason = "view_score_below_min"
        detail_min = 0.0
    if sample_key in _BROAD_FACE_SOURCE_PATCH_KEYS:
        quality_min = _DIRECT_PHOTO_BROAD_FACE_MIN_QUALITY
        detail_threshold = _direct_photo_broad_face_detail_threshold(
            support=support,
            density=density,
            quality=quality,
            texture_only_source_patch_candidate=texture_only_source_patch_candidate,
        )
        detail_min = detail_threshold
        if not rejection_reason and quality < quality_min:
            rejection_reason = "quality_below_min"
        elif not rejection_reason and detail < detail_threshold:
            rejection_reason = "detail_below_min"
        else:
            evidence = 0.34 * support + 0.28 * density + 0.22 * detail + 0.16 * quality
    elif sample_key == "ear_inner":
        quality_min = (
            _DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_QUALITY
            if (
                detail >= _DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_DETAIL
                and support >= _DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_SUPPORT
                and density >= _DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_DENSITY
                and view_score >= _DIRECT_PHOTO_EAR_INNER_RELAXED_MIN_VIEW_SCORE
            )
            else _DIRECT_PHOTO_EAR_INNER_MIN_QUALITY
        )
        detail_min = _DIRECT_PHOTO_EAR_INNER_MIN_DETAIL
        if not rejection_reason and detail < detail_min:
            rejection_reason = "detail_below_min"
        elif not rejection_reason and quality < quality_min:
            rejection_reason = "quality_below_min"
        else:
            evidence = 0.30 * detail + 0.26 * support + 0.24 * density + 0.20 * quality
    elif sample_key in _IRIS_SOURCE_LUMA_CORE_KEYS:
        quality_min = (
            _DIRECT_PHOTO_IRIS_RELAXED_MIN_QUALITY
            if (
                detail >= _DIRECT_PHOTO_IRIS_RELAXED_MIN_DETAIL
                and support >= _DIRECT_PHOTO_IRIS_RELAXED_MIN_SUPPORT
                and density >= _DIRECT_PHOTO_IRIS_RELAXED_MIN_DENSITY
                and view_score >= _DIRECT_PHOTO_IRIS_RELAXED_MIN_VIEW_SCORE
            )
            else _DIRECT_PHOTO_IRIS_MIN_QUALITY
        )
        detail_min = (
            _DIRECT_PHOTO_IRIS_RELAXED_MIN_DETAIL
            if quality_min == _DIRECT_PHOTO_IRIS_RELAXED_MIN_QUALITY
            else _DIRECT_PHOTO_IRIS_MIN_DETAIL
        )
        if not rejection_reason and detail < detail_min:
            rejection_reason = "detail_below_min"
        elif not rejection_reason and quality < quality_min:
            rejection_reason = "quality_below_min"
        else:
            evidence = 0.30 * detail + 0.25 * support + 0.25 * density + 0.20 * quality
    elif sample_key in _PUPIL_SOURCE_DARK_CORE_KEYS:
        quality_min = _DIRECT_PHOTO_PUPIL_MIN_QUALITY
        detail_min = 0.0
        if not rejection_reason and view_score < _DIRECT_PHOTO_PUPIL_MIN_VIEW_SCORE:
            rejection_reason = "view_score_below_min"
        elif not rejection_reason and support < _DIRECT_PHOTO_PUPIL_MIN_SUPPORT:
            rejection_reason = "support_below_min"
        elif not rejection_reason and density < _DIRECT_PHOTO_PUPIL_MIN_DENSITY:
            rejection_reason = "density_below_min"
        elif not rejection_reason and pupil_darkness < _DIRECT_PHOTO_PUPIL_MIN_DARKNESS:
            rejection_reason = "darkness_below_min"
        elif not rejection_reason and quality < quality_min:
            rejection_reason = "quality_below_min"
        else:
            evidence = (
                0.34 * pupil_darkness
                + 0.26 * support
                + 0.18 * density
                + 0.12 * view_score
                + 0.10 * quality
            )
    elif sample_key in _FACE_PRIORITY_SOURCE_PATCH_KEYS:
        quality_min = 0.45
        detail_min = 0.25 if quality < 0.55 else 0.0
        if not rejection_reason and quality < quality_min:
            rejection_reason = "quality_below_min"
        elif not rejection_reason and detail_min > 0.0 and detail < detail_min:
            rejection_reason = "detail_below_min"
        else:
            evidence = 0.32 * detail + 0.26 * support + 0.22 * density + 0.20 * quality
    else:
        if not rejection_reason:
            rejection_reason = "unsupported_sample_key"
    priority = (
        _clamp01(direct_upload_score * view_score * evidence) if not rejection_reason else 0.0
    )
    return {
        "status": "accepted" if priority > 0.0 else "rejected",
        "rejection_reason": rejection_reason,
        "priority": float(priority),
        "direct_photo_candidate": bool(direct_photo_candidate),
        "view_score": float(view_score),
        "raw_view_score": float(raw_view_score),
        "view_score_min": 0.50,
        "texture_only_source_patch_candidate": bool(texture_only_source_patch_candidate),
        "texture_only_source_patch_priority_version": (
            _TEXTURE_ONLY_SOURCE_PATCH_PRIORITY_VERSION
            if texture_only_source_patch_candidate
            else ""
        ),
        "texture_only_source_patch_effective_view_applied": bool(
            view_score > raw_view_score + 1e-12
        ),
        "support_score": float(support),
        "density_score": float(density),
        "pupil_darkness_score": float(pupil_darkness),
        "pupil_darkness_min": (
            float(_DIRECT_PHOTO_PUPIL_MIN_DARKNESS)
            if sample_key in _PUPIL_SOURCE_DARK_CORE_KEYS
            else 0.0
        ),
        "detail_score": float(detail),
        "detail_min": float(detail_min),
        "quality_score": float(quality),
        "quality_min": float(quality_min),
        "evidence_score": float(evidence),
    }


def _direct_photo_broad_face_detail_threshold(
    *,
    support: float,
    density: float,
    quality: float,
    texture_only_source_patch_candidate: bool = False,
) -> float:
    relaxed_density_min = (
        _TEXTURE_ONLY_SOURCE_PATCH_BROAD_FACE_RELAXED_MIN_DENSITY
        if texture_only_source_patch_candidate
        else _DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_DENSITY
    )
    if (
        support >= _DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_SUPPORT
        and density >= relaxed_density_min
        and quality >= _DIRECT_PHOTO_BROAD_FACE_MIN_QUALITY
    ):
        return _DIRECT_PHOTO_BROAD_FACE_RELAXED_DETAIL_THRESHOLD
    return _DIRECT_PHOTO_BROAD_FACE_DETAIL_THRESHOLD


def _texture_only_source_patch_rank_floor(
    sample_key: str,
    priority_diagnostic: dict[str, object],
) -> float:
    if sample_key not in _TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_KEYS:
        return 0.0
    if str(priority_diagnostic.get("status", "")) != "accepted":
        return 0.0
    if not bool(priority_diagnostic.get("texture_only_source_patch_candidate", False)):
        return 0.0
    priority = _clamp01(float(priority_diagnostic.get("priority", 0.0)))
    support = _clamp01(float(priority_diagnostic.get("support_score", 0.0)))
    density = _clamp01(float(priority_diagnostic.get("density_score", 0.0)))
    if (
        priority <= 0.0
        or support < _TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_MIN_SUPPORT
        or density < _TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_MIN_DENSITY
    ):
        return 0.0
    detail = _clamp01(float(priority_diagnostic.get("detail_score", 0.0)))
    quality = _clamp01(float(priority_diagnostic.get("quality_score", 0.0)))
    return float(
        min(
            _TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_MAX,
            _TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_BASE
            + _TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_PRIORITY_GAIN * priority
            + _TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_DETAIL_GAIN * detail
            + _TEXTURE_ONLY_SOURCE_PATCH_RANK_FLOOR_QUALITY_GAIN * quality,
        )
    )


def _source_patch_direct_uploaded_photo_candidate(metadata: dict[str, Any]) -> bool:
    media_type = str(metadata.get("media_type", "")).lower()
    source_type = str(metadata.get("source_type", "")).lower()
    if media_type != "photo" and source_type != "image":
        return False
    if _texture_only_source_patch_priority_candidate(metadata):
        return True
    source_media_id = str(metadata.get("source_media_id", "")).strip()
    source_name = Path(source_media_id).name if source_media_id else ""
    source_stem = Path(source_name).stem
    image_suffix = Path(source_name).suffix.lower()
    auto_staged = (
        source_stem.startswith("frame_")
        and len(source_stem) == len("frame_0000")
        and source_stem.removeprefix("frame_").isdigit()
    )
    return bool(source_name and image_suffix in {".jpg", ".jpeg", ".png"} and not auto_staged)


def _source_patch_view_confidence_penalty(
    sample_key: str,
    *,
    frontal_face_confidence: float,
    view_weight: float,
) -> float:
    floor = _source_patch_frontal_confidence_floor(sample_key)
    if floor <= 0.0:
        return 1.0
    frontal = _clamp01(frontal_face_confidence)
    if frontal >= floor:
        return 1.0
    expected_view_weight = 0.35 + 0.65 * floor
    frontal_gap = (floor - frontal) / max(floor, 1e-12)
    view_gap = max(0.0, expected_view_weight - _clamp01(view_weight)) / max(
        expected_view_weight,
        1e-12,
    )
    return float(max(0.50, 1.0 - 1.50 * frontal_gap - 0.50 * view_gap))


def _source_patch_frontal_confidence_floor(sample_key: str) -> float:
    if sample_key in _BROAD_FACE_SOURCE_PATCH_KEYS:
        return 0.60
    if sample_key == "ear_inner":
        return 0.40
    if sample_key in _FACE_PRIORITY_SOURCE_PATCH_KEYS:
        return 0.50
    return 0.0


def _source_patch_localized_support_score(
    sample_key: str,
    *,
    selected_count: int,
    candidate_count: int,
) -> float:
    selected = max(int(selected_count), 0)
    candidate = max(int(candidate_count), 0)
    if selected <= 0 or candidate <= 0:
        return 0.0
    expected = {
        "left_eye": 512.0,
        "right_eye": 512.0,
        "eye": 1024.0,
        "iris": 768.0,
        "left_iris": 512.0,
        "right_iris": 512.0,
        "left_pupil": 64.0,
        "right_pupil": 64.0,
        "nose": 192.0,
        "muzzle": 2048.0,
        "ear_inner": 1024.0,
        "mouth": 512.0,
    }.get(sample_key, 512.0)
    coverage = min(log1p(float(selected)) / log1p(expected), 1.0)
    locality = min(float(selected) / max(float(candidate), 1.0), 1.0)
    return float(np.sqrt(max(coverage * locality, 0.0)))


def _source_patch_density_score(
    sample_key: str,
    patch: NDArray[np.uint8],
    *,
    sample_count: int,
) -> float:
    if patch.size == 0 or sample_count <= 0:
        return 0.0
    height, width = patch.shape[:2]
    density = float(sample_count) / max(float(height * width), 1.0)
    target_density = float(_source_patch_footprint_profile(sample_key)["target_density"])
    return _clamp01(np.sqrt(density / max(target_density, 1e-12)))


def _source_patch_foreground_color_score(
    sample_key: str,
    patch: NDArray[np.uint8],
    *,
    mask: NDArray[np.uint8] | None = None,
) -> float:
    if sample_key not in _BACKGROUND_GREEN_GATED_SOURCE_PATCH_KEYS:
        return 1.0
    if patch.size == 0:
        return 0.0
    patch_rgb = np.asarray(patch[..., :3], dtype=np.float64)
    if mask is not None and mask.shape == patch_rgb.shape[:2] and np.any(np.asarray(mask) > 0):
        pixels = patch_rgb[np.asarray(mask) > 0]
    else:
        pixels = patch_rgb.reshape(-1, 3)
    if pixels.size == 0:
        return 0.0
    if pixels.shape[0] > 4096:
        stride = int(np.ceil(pixels.shape[0] / 4096.0))
        pixels = pixels[::stride][:4096]
    red = pixels[:, 0]
    green = pixels[:, 1]
    blue = pixels[:, 2]
    green_margin = green - np.maximum(red, blue)
    green_fraction = float(
        np.mean(
            (green > 42.0) & (green_margin > 10.0) & (green > red * 1.08) & (green > blue * 1.08)
        )
    )
    mean_green_margin = float(np.mean(np.clip(green_margin, 0.0, None)))
    if sample_key == "mouth":
        if green_fraction >= 0.18 or mean_green_margin >= 10.0:
            return 0.0
        if green_fraction <= 0.04 and mean_green_margin <= 4.0:
            return 1.0
        fraction_score = 1.0 - (green_fraction - 0.04) / 0.14
        margin_score = 1.0 - (mean_green_margin - 4.0) / 6.0
        return _clamp01(min(fraction_score, margin_score))
    if green_fraction >= 0.45 or mean_green_margin >= 18.0:
        return 0.15
    if green_fraction <= 0.12 and mean_green_margin <= 6.0:
        return 1.0
    fraction_score = 1.0 - (green_fraction - 0.12) / 0.33
    margin_score = 1.0 - (mean_green_margin - 6.0) / 12.0
    return max(0.15, _clamp01(min(fraction_score, margin_score)))


def _source_patch_feature_contrast_score(
    sample_key: str,
    candidate: _FaceFeaturePalette,
) -> float:
    return _source_patch_feature_contrast_score_from_colors(
        sample_key,
        eye_rgb=candidate.eye_rgb,
        iris_rgb=candidate.iris_rgb,
        pupil_rgb=_palette_pupil_rgb(candidate),
        nose_rgb=candidate.nose_rgb,
        muzzle_rgb=candidate.muzzle_rgb,
        ear_inner_rgb=candidate.ear_inner_rgb,
        mouth_rgb=candidate.mouth_rgb,
    )


def _source_patch_feature_contrast_score_from_colors(
    sample_key: str,
    *,
    eye_rgb: NDArray[np.uint8],
    iris_rgb: NDArray[np.uint8],
    nose_rgb: NDArray[np.uint8],
    muzzle_rgb: NDArray[np.uint8],
    ear_inner_rgb: NDArray[np.uint8],
    mouth_rgb: NDArray[np.uint8],
    pupil_rgb: NDArray[np.uint8] | None = None,
) -> float:
    eye_rgb_f = eye_rgb.astype(np.float64)
    iris_rgb_f = iris_rgb.astype(np.float64)
    pupil_rgb_f = eye_rgb_f if pupil_rgb is None else np.asarray(pupil_rgb, dtype=np.float64)
    nose_rgb_f = nose_rgb.astype(np.float64)
    muzzle_rgb_f = muzzle_rgb.astype(np.float64)
    ear_inner_rgb_f = ear_inner_rgb.astype(np.float64)
    mouth_rgb_f = mouth_rgb.astype(np.float64)
    eye_luma = _rgb_luma_255(eye_rgb_f)
    pupil_luma = _rgb_luma_255(pupil_rgb_f)
    iris_luma = _rgb_luma_255(iris_rgb_f)
    muzzle_luma = _rgb_luma_255(muzzle_rgb_f)
    mouth_luma = _rgb_luma_255(mouth_rgb_f)
    if sample_key in {"left_eye", "right_eye", "eye"}:
        return _clamp01(max(0.0, muzzle_luma - eye_luma) / 0.35)
    if sample_key in {"left_pupil", "right_pupil"}:
        return _clamp01(max(0.0, muzzle_luma - pupil_luma) / 0.35)
    if sample_key in {"iris", "left_iris", "right_iris"}:
        iris_eye_distance = _rgb_distance_255(
            iris_rgb_f,
            pupil_rgb_f,
        )
        return _clamp01(max(iris_eye_distance, abs(iris_luma - pupil_luma)) / 0.20)
    if sample_key == "nose":
        return _clamp01(
            _rgb_distance_255(
                nose_rgb_f,
                muzzle_rgb_f,
            )
            / 0.35
        )
    if sample_key == "mouth":
        return _clamp01(max(0.0, muzzle_luma - mouth_luma) / 0.35)
    if sample_key == "ear_inner":
        return _clamp01(
            _rgb_distance_255(
                ear_inner_rgb_f,
                muzzle_rgb_f,
            )
            / 0.25
        )
    if sample_key == "muzzle":
        return _clamp01(
            max(
                muzzle_luma - eye_luma,
                _rgb_distance_255(
                    muzzle_rgb_f,
                    nose_rgb_f,
                ),
            )
            / 0.35
        )
    return 0.0


def _mean_uint8_color(
    colors: list[NDArray[np.uint8]],
    fallback: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    if not colors:
        return fallback
    stacked = np.stack([color.astype(np.float64) for color in colors], axis=0)
    return np.clip(np.rint(np.mean(stacked, axis=0)), 0, 255).astype(np.uint8)


def _palette_pupil_rgb(palette: _FaceFeaturePalette) -> NDArray[np.uint8]:
    if palette.pupil_rgb is None:
        return palette.eye_rgb
    return np.asarray(palette.pupil_rgb, dtype=np.uint8)


def _raw_facial_feature_palette(
    observations: tuple[FitObservation, ...],
    images_by_frame: dict[str, NDArray[np.float32]],
    frame_metadata: tuple[dict[str, Any], ...],
    foreground_fallback_color: NDArray[np.float64],
    *,
    source_patch_feedback: dict[str, str] | None = None,
) -> _FaceFeaturePalette:
    if source_patch_feedback is None:
        source_patch_feedback = {}
    metadata_by_frame = {
        str(item.get("frame_id")): item for item in frame_metadata if item.get("frame_id")
    }
    fallback_eye = _rgb_float_to_uint8(np.asarray([0.06, 0.045, 0.035], dtype=np.float64))
    fallback_nose = _rgb_float_to_uint8(
        0.75 * np.asarray([0.88, 0.52, 0.56], dtype=np.float64)
        + 0.25 * np.asarray(foreground_fallback_color, dtype=np.float64)
    )
    fallback_muzzle = _rgb_float_to_uint8(foreground_fallback_color)
    best: _FaceFeaturePalette | None = None
    candidates: list[_FaceFeaturePalette] = []

    for observation in observations:
        image = images_by_frame[observation.frame_id]
        mask = np.asarray(observation.hard_mask, dtype=np.float32) >= 0.5
        if image.shape[:2] != mask.shape or not np.any(mask):
            continue
        metadata = metadata_by_frame.get(observation.frame_id, {})
        if not _source_patch_candidate_allowed(metadata):
            continue
        candidate = _face_feature_palette_from_observation(
            observation,
            image,
            mask,
            metadata=metadata,
            fallback_eye=fallback_eye,
            fallback_nose=fallback_nose,
            fallback_muzzle=fallback_muzzle,
        )
        candidates.append(candidate)
        if best is None or candidate.source_score > best.source_score:
            best = candidate

    if best is not None:
        return _merge_best_source_patches(
            best,
            candidates,
            source_patch_feedback=source_patch_feedback,
        )

    return _FaceFeaturePalette(
        source_frame_id="none",
        source_score=0.0,
        eye_rgb=fallback_eye,
        iris_rgb=fallback_eye,
        nose_rgb=fallback_nose,
        muzzle_rgb=fallback_muzzle,
        ear_inner_rgb=fallback_nose,
        mouth_rgb=fallback_eye,
        sample_counts={
            "eye": 0,
            "iris": 0,
            "left_eye": 0,
            "right_eye": 0,
            "left_iris": 0,
            "right_iris": 0,
            "left_pupil": 0,
            "right_pupil": 0,
            "nose": 0,
            "muzzle": 0,
            "ear_inner": 0,
            "mouth": 0,
        },
        source_patches={},
        source_patch_masks={},
        source_patch_mask_observed_counts={},
        source_patch_mask_densified_counts={},
        source_patch_mask_densification_gains={},
        source_patch_localization_candidate_counts={},
        source_patch_localization_selected_counts={},
        source_patch_localization_fractions={},
        source_patch_landmark_confidences={},
        source_patch_scores={},
        source_patch_frame_ids={},
        source_patch_frontal_face_confidences={},
        source_patch_view_weights={},
        source_patch_view_labels={},
        source_patch_view_weighted_ranks={},
        source_patch_feedback_components={},
        source_patch_feedback_scores={},
        source_patch_direct_photo_candidates={},
        source_patch_observed_anchor_guided={},
        source_patch_candidate_diagnostics={},
    )


def _merge_best_source_patches(
    base: _FaceFeaturePalette,
    candidates: list[_FaceFeaturePalette],
    *,
    source_patch_feedback: dict[str, str],
) -> _FaceFeaturePalette:
    def _copy_allowed_base_keys(mapping: dict[str, Any]) -> dict[str, Any]:
        if not base.allowed_source_patch_sample_keys:
            return dict(mapping)
        return {
            key: value
            for key, value in mapping.items()
            if _source_patch_sample_key_allowed(base, key)
        }

    sample_counts = _copy_allowed_base_keys(base.sample_counts)
    source_patches = _copy_allowed_base_keys(base.source_patches)
    source_patch_masks = _copy_allowed_base_keys(base.source_patch_masks)
    source_patch_mask_observed_counts = _copy_allowed_base_keys(
        base.source_patch_mask_observed_counts
    )
    source_patch_mask_densified_counts = _copy_allowed_base_keys(
        base.source_patch_mask_densified_counts
    )
    source_patch_mask_densification_gains = _copy_allowed_base_keys(
        base.source_patch_mask_densification_gains
    )
    source_patch_localization_candidate_counts = _copy_allowed_base_keys(
        base.source_patch_localization_candidate_counts
    )
    source_patch_localization_selected_counts = _copy_allowed_base_keys(
        base.source_patch_localization_selected_counts
    )
    source_patch_localization_fractions = _copy_allowed_base_keys(
        base.source_patch_localization_fractions
    )
    source_patch_landmark_confidences = _copy_allowed_base_keys(
        base.source_patch_landmark_confidences
    )
    source_patch_scores = _copy_allowed_base_keys(base.source_patch_scores)
    source_patch_frame_ids = _copy_allowed_base_keys(base.source_patch_frame_ids)
    source_patch_frontal_face_confidences = _copy_allowed_base_keys(
        base.source_patch_frontal_face_confidences
    )
    source_patch_view_weights = _copy_allowed_base_keys(base.source_patch_view_weights)
    source_patch_view_labels = _copy_allowed_base_keys(base.source_patch_view_labels)
    source_patch_view_weighted_ranks = _copy_allowed_base_keys(
        base.source_patch_view_weighted_ranks
    )
    source_patch_frontal_photo_priorities = _copy_allowed_base_keys(
        base.source_patch_frontal_photo_priorities
    )
    source_patch_frontal_photo_priority_diagnostics = _copy_allowed_base_keys(
        base.source_patch_frontal_photo_priority_diagnostics
    )
    source_patch_direct_photo_candidates = _copy_allowed_base_keys(
        base.source_patch_direct_photo_candidates
    )
    source_patch_observed_anchor_guided = _copy_allowed_base_keys(
        base.source_patch_observed_anchor_guided
    )
    source_patch_feedback_components = _copy_allowed_base_keys(
        base.source_patch_feedback_components
    )
    source_patch_feedback_scores = _copy_allowed_base_keys(base.source_patch_feedback_scores)
    source_patch_candidate_diagnostics: dict[str, list[dict[str, object]]] = {}
    selected_eye_colors: list[NDArray[np.uint8]] = []
    selected_pupil_colors: list[NDArray[np.uint8]] = []
    selected_iris_colors: list[NDArray[np.uint8]] = []
    selected_nose_colors: list[NDArray[np.uint8]] = []
    selected_muzzle_colors: list[NDArray[np.uint8]] = []
    selected_ear_inner_colors: list[NDArray[np.uint8]] = []
    selected_mouth_colors: list[NDArray[np.uint8]] = []
    for key in _SOURCE_PATCH_SAMPLE_KEYS:
        best_index = -1
        best_candidate: _FaceFeaturePalette | None = None
        feedback_component = _normalize_source_patch_feedback_component(
            source_patch_feedback.get(key, "")
            or source_patch_feedback.get(_source_patch_feedback_alias(key), "")
        )
        best_feedback_score = 0.0
        best_rank = (-1.0, -1.0, -1.0, -1.0, -1.0, -1, -1.0, 0)
        candidate_diagnostics: list[dict[str, object]] = []
        for index, candidate in enumerate(candidates):
            if not _source_patch_sample_key_allowed(candidate, key):
                continue
            patch = candidate.source_patches.get(key)
            if patch is None or patch.size == 0:
                continue
            landmark_confidence = (
                float(candidate.source_patch_landmark_confidences.get(key, 0.0))
                if key in _LANDMARK_ALIGNED_SOURCE_PATCH_KEYS
                else 0.0
            )
            quality_score = float(candidate.source_patch_scores.get(key, 0.0))
            view_weight = float(candidate.source_patch_view_weights.get(key, 0.0))
            frontal_face_confidence = float(
                candidate.source_patch_frontal_face_confidences.get(key, 0.0)
            )
            frontal_photo_priority = float(
                candidate.source_patch_frontal_photo_priorities.get(key, 0.0)
            )
            direct_photo_candidate = bool(
                candidate.source_patch_direct_photo_candidates.get(key, False)
            )
            if (
                key == "nose"
                and direct_photo_candidate
                and frontal_photo_priority > 0.0
                and _direct_photo_nose_color_score(candidate.nose_rgb)
                < _DIRECT_PHOTO_NOSE_MIN_COLOR_SCORE
            ):
                frontal_photo_priority = 0.0
            feature_contrast_score = _source_patch_feature_contrast_score(key, candidate)
            foreground_color_score = _source_patch_foreground_color_score(
                key,
                patch,
                mask=candidate.source_patch_masks.get(key),
            )
            feedback_score = _source_patch_feedback_score(
                feedback_component,
                patch=patch,
                sample_count=int(candidate.sample_counts.get(key, 0)),
                observed_count=int(candidate.source_patch_mask_observed_counts.get(key, 0)),
                densified_count=int(candidate.source_patch_mask_densified_counts.get(key, 0)),
            )
            view_weighted_rank = _source_patch_view_weighted_rank(
                key,
                landmark_confidence=landmark_confidence,
                feedback_score=feedback_score,
                quality_score=quality_score,
                view_weight=view_weight,
                frontal_face_confidence=frontal_face_confidence,
                feature_contrast_score=feature_contrast_score,
                localization_selected_count=int(
                    candidate.source_patch_localization_selected_counts.get(key, 0)
                ),
                localization_candidate_count=int(
                    candidate.source_patch_localization_candidate_counts.get(key, 0)
                ),
                source_density_score=_source_patch_density_score(
                    key,
                    patch,
                    sample_count=int(candidate.sample_counts.get(key, 0)),
                ),
                source_detail_score=_source_patch_diagnostic_detail_score(patch),
                frontal_photo_priority=frontal_photo_priority,
            )
            raw_view_weighted_rank = view_weighted_rank
            view_weighted_rank *= foreground_color_score
            view_weighted_rank = _source_patch_rank_with_direct_photo_penalty(
                view_weighted_rank,
                direct_photo_candidate=direct_photo_candidate,
                frontal_photo_priority=frontal_photo_priority,
            )
            priority_diagnostic = candidate.source_patch_frontal_photo_priority_diagnostics.get(
                key, {}
            )
            texture_only_rank_floor = _texture_only_source_patch_rank_floor(
                key,
                priority_diagnostic,
            )
            if texture_only_rank_floor > 0.0:
                view_weighted_rank = max(view_weighted_rank, texture_only_rank_floor)
            direct_photo_rank_penalty = (
                _ZERO_PRIORITY_DIRECT_PHOTO_RANK_PENALTY
                if direct_photo_candidate and frontal_photo_priority <= 0.0
                else 1.0
            )
            rank = (
                view_weighted_rank,
                frontal_photo_priority,
                landmark_confidence,
                feedback_score,
                quality_score,
                int(candidate.sample_counts.get(key, 0)),
                float(candidate.source_score),
                -index,
            )
            candidate_diagnostics.append(
                _source_patch_candidate_diagnostic(
                    key,
                    candidate,
                    candidate_index=index,
                    feedback_component=feedback_component,
                    feedback_score=feedback_score,
                    feature_contrast_score=feature_contrast_score,
                    foreground_color_score=foreground_color_score,
                    direct_photo_candidate=direct_photo_candidate,
                    direct_photo_rank_penalty=direct_photo_rank_penalty,
                    texture_only_rank_floor=texture_only_rank_floor,
                    raw_view_weighted_rank=raw_view_weighted_rank,
                    weighted_rank=view_weighted_rank,
                    rank=rank,
                )
            )
            if rank > best_rank:
                best_rank = rank
                best_candidate = candidate
                best_index = index
                best_feedback_score = feedback_score
        if best_candidate is None or best_index < 0:
            if candidate_diagnostics:
                source_patch_candidate_diagnostics[key] = candidate_diagnostics
            continue
        source_patch_candidate_diagnostics[key] = [
            {**item, "selected": int(item["candidate_index"]) == best_index}
            for item in candidate_diagnostics
        ]
        source_patches[key] = best_candidate.source_patches[key]
        source_patch_masks[key] = best_candidate.source_patch_masks[key]
        source_patch_mask_observed_counts[key] = int(
            best_candidate.source_patch_mask_observed_counts.get(key, 0)
        )
        source_patch_mask_densified_counts[key] = int(
            best_candidate.source_patch_mask_densified_counts.get(key, 0)
        )
        source_patch_mask_densification_gains[key] = float(
            best_candidate.source_patch_mask_densification_gains.get(key, 0.0)
        )
        source_patch_localization_candidate_counts[key] = int(
            best_candidate.source_patch_localization_candidate_counts.get(key, 0)
        )
        source_patch_localization_selected_counts[key] = int(
            best_candidate.source_patch_localization_selected_counts.get(key, 0)
        )
        source_patch_localization_fractions[key] = float(
            best_candidate.source_patch_localization_fractions.get(key, 1.0)
        )
        source_patch_landmark_confidences[key] = float(
            best_candidate.source_patch_landmark_confidences.get(key, 0.0)
        )
        source_patch_scores[key] = float(best_candidate.source_patch_scores.get(key, 0.0))
        source_patch_frame_ids[key] = str(best_candidate.source_frame_id)
        source_patch_frontal_face_confidences[key] = float(
            best_candidate.source_patch_frontal_face_confidences.get(key, 0.0)
        )
        source_patch_view_weights[key] = float(
            best_candidate.source_patch_view_weights.get(key, 0.0)
        )
        source_patch_view_labels[key] = str(
            best_candidate.source_patch_view_labels.get(key, "unknown")
        )
        source_patch_view_weighted_ranks[key] = float(best_rank[0])
        source_patch_frontal_photo_priorities[key] = float(
            best_candidate.source_patch_frontal_photo_priorities.get(key, 0.0)
        )
        source_patch_frontal_photo_priority_diagnostics[key] = dict(
            best_candidate.source_patch_frontal_photo_priority_diagnostics.get(key, {})
        )
        source_patch_direct_photo_candidates[key] = bool(
            best_candidate.source_patch_direct_photo_candidates.get(key, False)
        )
        source_patch_observed_anchor_guided[key] = bool(
            best_candidate.source_patch_observed_anchor_guided.get(key, False)
        )
        if key in {"left_eye", "right_eye", "eye", "left_pupil", "right_pupil"}:
            selected_eye_colors.append(best_candidate.eye_rgb)
        if key in {"left_pupil", "right_pupil"}:
            selected_pupil_colors.append(_palette_pupil_rgb(best_candidate))
        if key in {"iris", "left_iris", "right_iris"}:
            selected_iris_colors.append(best_candidate.iris_rgb)
        if key == "nose":
            selected_nose_colors.append(best_candidate.nose_rgb)
        if key == "muzzle":
            selected_muzzle_colors.append(best_candidate.muzzle_rgb)
        if key == "ear_inner":
            selected_ear_inner_colors.append(best_candidate.ear_inner_rgb)
        if key == "mouth":
            selected_mouth_colors.append(best_candidate.mouth_rgb)
        if feedback_component:
            source_patch_feedback_components[key] = feedback_component
            source_patch_feedback_scores[key] = float(best_feedback_score)
        sample_counts[key] = int(best_candidate.sample_counts.get(key, 0))
    return _FaceFeaturePalette(
        source_frame_id=base.source_frame_id,
        source_score=base.source_score,
        eye_rgb=_mean_uint8_color(selected_eye_colors, base.eye_rgb),
        iris_rgb=_mean_uint8_color(selected_iris_colors, base.iris_rgb),
        nose_rgb=_mean_uint8_color(selected_nose_colors, base.nose_rgb),
        muzzle_rgb=_mean_uint8_color(selected_muzzle_colors, base.muzzle_rgb),
        ear_inner_rgb=_mean_uint8_color(
            selected_ear_inner_colors,
            base.ear_inner_rgb,
        ),
        mouth_rgb=_mean_uint8_color(selected_mouth_colors, base.mouth_rgb),
        sample_counts=sample_counts,
        source_patches=source_patches,
        source_patch_masks=source_patch_masks,
        source_patch_mask_observed_counts=source_patch_mask_observed_counts,
        source_patch_mask_densified_counts=source_patch_mask_densified_counts,
        source_patch_mask_densification_gains=source_patch_mask_densification_gains,
        source_patch_localization_candidate_counts=(source_patch_localization_candidate_counts),
        source_patch_localization_selected_counts=source_patch_localization_selected_counts,
        source_patch_localization_fractions=source_patch_localization_fractions,
        source_patch_landmark_confidences=source_patch_landmark_confidences,
        source_patch_scores=source_patch_scores,
        source_patch_frame_ids=source_patch_frame_ids,
        source_patch_frontal_face_confidences=source_patch_frontal_face_confidences,
        source_patch_view_weights=source_patch_view_weights,
        source_patch_view_labels=source_patch_view_labels,
        source_patch_view_weighted_ranks=source_patch_view_weighted_ranks,
        source_patch_feedback_components=source_patch_feedback_components,
        source_patch_feedback_scores=source_patch_feedback_scores,
        source_patch_frontal_photo_priorities=source_patch_frontal_photo_priorities,
        source_patch_frontal_photo_priority_diagnostics=(
            source_patch_frontal_photo_priority_diagnostics
        ),
        source_patch_direct_photo_candidates=source_patch_direct_photo_candidates,
        source_patch_observed_anchor_guided=source_patch_observed_anchor_guided,
        source_patch_candidate_diagnostics=source_patch_candidate_diagnostics,
        pupil_rgb=_mean_uint8_color(
            selected_pupil_colors,
            _palette_pupil_rgb(base),
        ),
    )


def _source_patch_candidate_diagnostic(
    sample_key: str,
    candidate: _FaceFeaturePalette,
    *,
    candidate_index: int,
    feedback_component: str,
    feedback_score: float,
    feature_contrast_score: float,
    foreground_color_score: float,
    direct_photo_candidate: bool,
    direct_photo_rank_penalty: float,
    texture_only_rank_floor: float,
    raw_view_weighted_rank: float,
    weighted_rank: float,
    rank: tuple[float, ...],
) -> dict[str, object]:
    patch = candidate.source_patches[sample_key]
    selected_count = int(candidate.source_patch_localization_selected_counts.get(sample_key, 0))
    candidate_count = int(candidate.source_patch_localization_candidate_counts.get(sample_key, 0))
    sample_count = int(candidate.sample_counts.get(sample_key, 0))
    priority_diagnostic = candidate.source_patch_frontal_photo_priority_diagnostics.get(
        sample_key,
        {},
    )
    return {
        "sample_key": sample_key,
        "candidate_index": int(candidate_index),
        "source_frame_id": str(
            candidate.source_patch_frame_ids.get(
                sample_key,
                candidate.source_frame_id,
            )
        ),
        "source_score": round(float(candidate.source_score), 6),
        "sample_count": sample_count,
        "source_patch_width": int(patch.shape[1]),
        "source_patch_height": int(patch.shape[0]),
        "source_patch_luma_std": round(float(_source_patch_luma_std(patch)), 6),
        "source_patch_unique_rgb_count": int(_source_patch_unique_rgb_count(patch)),
        "source_patch_quality_score": round(
            float(candidate.source_patch_scores.get(sample_key, 0.0)),
            6,
        ),
        "source_patch_detail_score": round(float(_source_patch_diagnostic_detail_score(patch)), 6),
        "source_patch_density_score": round(
            float(_source_patch_density_score(sample_key, patch, sample_count=sample_count)),
            6,
        ),
        "source_patch_localized_support_score": round(
            float(
                _source_patch_localized_support_score(
                    sample_key,
                    selected_count=selected_count,
                    candidate_count=candidate_count,
                )
            ),
            6,
        ),
        "source_patch_localization_candidate_count": candidate_count,
        "source_patch_localization_selected_count": selected_count,
        "source_patch_localization_fraction": round(
            float(candidate.source_patch_localization_fractions.get(sample_key, 0.0)),
            6,
        ),
        "source_patch_landmark_confidence": round(
            float(candidate.source_patch_landmark_confidences.get(sample_key, 0.0)),
            6,
        ),
        "source_patch_observed_anchor_guided": bool(
            candidate.source_patch_observed_anchor_guided.get(sample_key, False)
        ),
        "source_patch_sample_key_policy_allowed": bool(
            _source_patch_sample_key_allowed(candidate, sample_key)
        ),
        "source_patch_allowed_sample_keys": list(candidate.allowed_source_patch_sample_keys),
        "source_patch_frontal_face_confidence": round(
            float(candidate.source_patch_frontal_face_confidences.get(sample_key, 0.0)),
            6,
        ),
        "source_patch_view_weight": round(
            float(candidate.source_patch_view_weights.get(sample_key, 0.0)),
            6,
        ),
        "source_patch_view_label": str(
            candidate.source_patch_view_labels.get(sample_key, "unknown")
        ),
        "source_patch_feature_contrast_score": round(float(feature_contrast_score), 6),
        "source_patch_foreground_color_score": round(float(foreground_color_score), 6),
        "source_patch_direct_photo_candidate": bool(direct_photo_candidate),
        "source_patch_zero_priority_direct_photo_rank_penalty": round(
            float(direct_photo_rank_penalty),
            6,
        ),
        "source_patch_texture_only_rank_floor": round(
            float(texture_only_rank_floor),
            6,
        ),
        "source_patch_feedback_component": feedback_component,
        "source_patch_feedback_score": round(float(feedback_score), 6),
        "source_patch_raw_view_weighted_rank": round(float(raw_view_weighted_rank), 6),
        "source_patch_view_weighted_rank": round(float(weighted_rank), 6),
        "source_patch_frontal_photo_priority": round(
            float(candidate.source_patch_frontal_photo_priorities.get(sample_key, 0.0)),
            6,
        ),
        "source_patch_frontal_photo_priority_status": str(
            priority_diagnostic.get("status", "unknown")
        ),
        "source_patch_frontal_photo_priority_rejection_reason": str(
            priority_diagnostic.get("rejection_reason", "unknown")
        ),
        "source_patch_frontal_photo_priority_view_score": round(
            float(priority_diagnostic.get("view_score", 0.0)),
            6,
        ),
        "source_patch_frontal_photo_priority_raw_view_score": round(
            float(priority_diagnostic.get("raw_view_score", 0.0)),
            6,
        ),
        "source_patch_texture_only_source_patch_candidate": bool(
            priority_diagnostic.get("texture_only_source_patch_candidate", False)
        ),
        "source_patch_texture_only_effective_view_applied": bool(
            priority_diagnostic.get(
                "texture_only_source_patch_effective_view_applied",
                False,
            )
        ),
        "source_patch_frontal_photo_priority_support_score": round(
            float(priority_diagnostic.get("support_score", 0.0)),
            6,
        ),
        "source_patch_frontal_photo_priority_density_score": round(
            float(priority_diagnostic.get("density_score", 0.0)),
            6,
        ),
        "source_patch_frontal_photo_priority_detail_min": round(
            float(priority_diagnostic.get("detail_min", 0.0)),
            6,
        ),
        "source_patch_frontal_photo_priority_quality_min": round(
            float(priority_diagnostic.get("quality_min", 0.0)),
            6,
        ),
        "source_patch_frontal_photo_priority_evidence_score": round(
            float(priority_diagnostic.get("evidence_score", 0.0)),
            6,
        ),
        "selection_rank": [round(float(value), 6) for value in rank],
        "selected": False,
    }


def _accumulate_face_color_samples(
    rgb_sum: NDArray[np.float64],
    pixel_count: NDArray[np.int64],
    source_pixel_counts: NDArray[np.int64],
    *,
    face_index_buffer: NDArray[np.integer],
    image: NDArray[np.floating],
    valid_mask: NDArray[np.bool_],
    view_index: int,
) -> None:
    face_indices = np.asarray(face_index_buffer[valid_mask], dtype=np.int64)
    if face_indices.size == 0:
        return
    face_count = int(rgb_sum.shape[0])
    samples = np.asarray(image[valid_mask], dtype=np.float64)
    counts = np.bincount(face_indices, minlength=face_count)[:face_count].astype(np.int64)
    pixel_count[:] += counts
    source_pixel_counts[:, view_index] += counts
    for channel in range(3):
        rgb_sum[:, channel] += np.bincount(
            face_indices,
            weights=samples[:, channel],
            minlength=face_count,
        )[:face_count]


def _source_patch_feedback_score(
    component: str,
    *,
    patch: NDArray[np.uint8],
    sample_count: int,
    observed_count: int,
    densified_count: int,
) -> float:
    component = _normalize_source_patch_feedback_component(component)
    if not component or patch.size == 0:
        return 0.0
    if component == "source_evidence_score":
        return 1.0
    if component in {"source_luma_score", "texture_luma_score"}:
        return _source_patch_luma_detail_score(patch)
    if component == "unique_rgb_score":
        unique_count = _bounded_unique_rgb_count(patch, max_samples=4096)
        return float(np.clip(unique_count / 32.0, 0.0, 1.0))
    if component == "confident_texel_score":
        return _source_patch_coverage_feedback_score(
            sample_count=sample_count,
            observed_count=observed_count,
            densified_count=densified_count,
        )
    if component == "combined_score":
        return float(
            0.5 * _source_patch_luma_detail_score(patch)
            + 0.5
            * _source_patch_coverage_feedback_score(
                sample_count=sample_count,
                observed_count=observed_count,
                densified_count=densified_count,
            )
        )
    return 0.0


def _source_patch_luma_detail_score(patch: NDArray[np.uint8]) -> float:
    return float(np.clip(_source_patch_luma_std(patch) / 0.075, 0.0, 1.0))


def _source_patch_diagnostic_detail_score(patch: NDArray[np.uint8]) -> float:
    return float(np.clip(_source_patch_luma_std(patch) / 0.020, 0.0, 1.0))


def _source_patch_luma_std(patch: NDArray[np.uint8]) -> float:
    if patch.size == 0:
        return 0.0
    rgb = patch.astype(np.float64) / 255.0
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return float(np.std(luma))


def _observed_face_anchor_prior(
    observation: FitObservation,
    *,
    shape: tuple[int, int],
) -> dict[str, object] | None:
    anchors = observation.feature_anchors_xyc
    if not anchors:
        return None
    left_eye = _observed_anchor_point(
        anchors,
        "left_eye_center",
        shape=shape,
        bbox_xyxy=observation.bbox_xyxy,
    )
    right_eye = _observed_anchor_point(
        anchors,
        "right_eye_center",
        shape=shape,
        bbox_xyxy=observation.bbox_xyxy,
    )
    if left_eye is None or right_eye is None:
        return None
    left_x, left_y, _left_confidence = left_eye
    right_x, right_y, _right_confidence = right_eye
    x0, y0, x1, y1 = (float(value) for value in observation.bbox_xyxy)
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    eye_span = float(right_x - left_x)
    eye_y_gap = abs(float(right_y - left_y))
    if eye_span <= max(3.0, bbox_width * 0.06):
        return None
    if eye_span >= bbox_width * 0.92:
        return None
    if eye_y_gap >= max(bbox_height * 0.28, eye_span * 0.70):
        return None

    targets: dict[str, tuple[float, float, float]] = {
        "left_eye_center": left_eye,
        "right_eye_center": right_eye,
    }
    nose_tip = _observed_anchor_point(
        anchors,
        "nose_tip",
        shape=shape,
        bbox_xyxy=observation.bbox_xyxy,
    )
    if nose_tip is not None and nose_tip[1] > min(left_y, right_y):
        targets["nose_tip"] = nose_tip
    muzzle_center = _observed_anchor_point(
        anchors,
        "muzzle_center",
        shape=shape,
        bbox_xyxy=observation.bbox_xyxy,
    )
    if muzzle_center is not None:
        min_muzzle_y = nose_tip[1] if nose_tip is not None else min(left_y, right_y)
        if muzzle_center[1] >= min_muzzle_y:
            targets["muzzle_center"] = muzzle_center

    confidences = [value[2] for value in targets.values()]
    return {
        "targets": targets,
        "eye_span": eye_span,
        "eye_mid": (0.5 * (left_x + right_x), 0.5 * (left_y + right_y)),
        "confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def _observed_anchor_point(
    anchors: dict[str, tuple[float, float, float]],
    canonical_name: str,
    *,
    shape: tuple[int, int],
    bbox_xyxy: NDArray[np.floating],
) -> tuple[float, float, float] | None:
    raw: tuple[float, float, float] | None = None
    for name in _OBSERVED_FACE_ANCHOR_ALIASES.get(canonical_name, (canonical_name,)):
        candidate = anchors.get(name)
        if candidate is not None:
            raw = candidate
            break
    if raw is None:
        return None
    x, y, confidence = (float(value) for value in raw)
    if not np.isfinite([x, y, confidence]).all() or confidence <= 0.0:
        return None
    height, width = shape
    if x < 0.0 or y < 0.0 or x >= float(width) or y >= float(height):
        return None
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    margin_x = max(2.0, bbox_width * 0.16)
    margin_y = max(2.0, bbox_height * 0.16)
    if x < x0 - margin_x or x > x1 + margin_x or y < y0 - margin_y or y > y1 + margin_y:
        return None
    return x, y, float(np.clip(confidence, 0.0, 1.0))


def _anchor_feature_region(
    shape: tuple[int, int],
    *,
    center_xy: tuple[float, float],
    radius_xy: tuple[float, float],
    foreground_mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    return _image_ellipse_mask(shape, center_xy=center_xy, radius_xy=radius_xy) & np.asarray(
        foreground_mask, dtype=np.bool_
    )


def _source_patch_pupil_darkness_score(patch: NDArray[np.uint8]) -> float:
    if patch.size == 0:
        return 0.0
    rgb = patch.astype(np.float64) / 255.0
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    low_luma = float(np.quantile(luma, 0.20))
    dark_fraction = float(np.mean(luma <= 0.32))
    dark_quantile_score = _clamp01((0.48 - low_luma) / 0.36)
    return float(_clamp01(0.65 * dark_quantile_score + 0.35 * dark_fraction))


def _source_patch_unique_rgb_count(patch: NDArray[np.uint8]) -> int:
    if patch.size == 0:
        return 0
    return _bounded_unique_rgb_count(patch, max_samples=4096)


def _source_patch_coverage_feedback_score(
    *,
    sample_count: int,
    observed_count: int,
    densified_count: int,
) -> float:
    support = max(sample_count, observed_count, 0)
    support_score = min(log1p(float(support)) / log1p(256.0), 1.0)
    if densified_count > 0:
        density_score = np.clip(observed_count / densified_count, 0.0, 1.0)
    else:
        density_score = 0.0
    return float(0.65 * support_score + 0.35 * density_score)


def _face_feature_palette_from_observation(
    observation: FitObservation,
    image: NDArray[np.float32],
    mask: NDArray[np.bool_],
    *,
    metadata: dict[str, Any],
    fallback_eye: NDArray[np.uint8],
    fallback_nose: NDArray[np.uint8],
    fallback_muzzle: NDArray[np.uint8],
) -> _FaceFeaturePalette:
    foreground = image[mask]
    luma_image = _rgb_luma(image)
    chroma_image = np.max(image, axis=2) - np.min(image, axis=2)
    red_excess = image[:, :, 0] - np.maximum(image[:, :, 1], image[:, :, 2])
    luma = luma_image[mask]
    dark_cutoff = min(float(np.quantile(luma, 0.10)), 0.38)
    bright_cutoff = max(float(np.quantile(luma, 0.68)), 0.50)

    upper = _bbox_region_mask(
        mask.shape,
        observation.bbox_xyxy,
        x_range=(0.05, 0.95),
        y_range=(0.00, 0.62),
    )
    left_eye_region = _bbox_region_mask(
        mask.shape,
        observation.bbox_xyxy,
        x_range=(0.14, 0.50),
        y_range=(0.05, 0.48),
    )
    right_eye_region = _bbox_region_mask(
        mask.shape,
        observation.bbox_xyxy,
        x_range=(0.50, 0.86),
        y_range=(0.05, 0.48),
    )
    face_center = _bbox_region_mask(
        mask.shape,
        observation.bbox_xyxy,
        x_range=(0.24, 0.76),
        y_range=(0.18, 0.78),
    )
    lower_center = _bbox_region_mask(
        mask.shape,
        observation.bbox_xyxy,
        x_range=(0.30, 0.70),
        y_range=(0.44, 0.84),
    )
    upper_sides = _bbox_region_mask(
        mask.shape,
        observation.bbox_xyxy,
        x_range=(0.04, 0.96),
        y_range=(0.00, 0.38),
    )

    x0, y0, x1, y1 = (float(value) for value in observation.bbox_xyxy)
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    left_eye_expected_xy = (x0 + 0.34 * bbox_width, y0 + 0.25 * bbox_height)
    right_eye_expected_xy = (x0 + 0.66 * bbox_width, y0 + 0.25 * bbox_height)
    iris_expected_xy = (x0 + 0.50 * bbox_width, y0 + 0.28 * bbox_height)
    nose_expected_xy = (x0 + 0.50 * bbox_width, y0 + 0.56 * bbox_height)
    muzzle_expected_xy = (x0 + 0.50 * bbox_width, y0 + 0.67 * bbox_height)
    observed_anchor_guided_by_key: dict[str, bool] = {}
    landmark_confidences_by_key: dict[str, float] = {}

    dark_all = mask & (luma_image <= dark_cutoff)
    anchor_prior = _observed_face_anchor_prior(observation, shape=mask.shape)
    anchor_targets: dict[str, tuple[float, float, float]] = {}
    if anchor_prior is not None:
        anchor_targets = dict(anchor_prior["targets"])
        left_anchor = anchor_targets["left_eye_center"]
        right_anchor = anchor_targets["right_eye_center"]
        left_eye_expected_xy = (left_anchor[0], left_anchor[1])
        right_eye_expected_xy = (right_anchor[0], right_anchor[1])
        iris_expected_xy = tuple(anchor_prior["eye_mid"])
        eye_span = float(anchor_prior["eye_span"])
        left_eye_anchor_region = _anchor_feature_region(
            mask.shape,
            center_xy=left_eye_expected_xy,
            radius_xy=(
                max(eye_span * 0.20, bbox_width * 0.030, 3.0),
                max(eye_span * 0.16, bbox_height * 0.026, 2.0),
            ),
            foreground_mask=mask,
        )
        right_eye_anchor_region = _anchor_feature_region(
            mask.shape,
            center_xy=right_eye_expected_xy,
            radius_xy=(
                max(eye_span * 0.20, bbox_width * 0.030, 3.0),
                max(eye_span * 0.16, bbox_height * 0.026, 2.0),
            ),
            foreground_mask=mask,
        )
        eye_candidate = dark_all | (
            mask & (chroma_image > 0.030) & (luma_image < min(bright_cutoff, 0.78))
        )
        dark = dark_all
        left_eye_candidate = eye_candidate & left_eye_anchor_region
        right_eye_candidate = eye_candidate & right_eye_anchor_region
        left_eye, left_eye_localization = _localized_feature_component_mask(
            left_eye_candidate,
            luma_image=luma_image,
            expected_center_xy=left_eye_expected_xy,
            min_pixels=2,
            max_components=2,
            fallback_mask=left_eye_candidate,
        )
        right_eye, right_eye_localization = _localized_feature_component_mask(
            right_eye_candidate,
            luma_image=luma_image,
            expected_center_xy=right_eye_expected_xy,
            min_pixels=2,
            max_components=2,
            fallback_mask=right_eye_candidate,
        )
        iris_search_region = mask & (left_eye_anchor_region | right_eye_anchor_region)
        left_confidence = float(left_anchor[2])
        right_confidence = float(right_anchor[2])
        landmark_confidence = float(anchor_prior["confidence"])
        observed_anchor_guided_by_key["left_eye"] = True
        landmark_confidences_by_key["left_eye"] = left_confidence
        observed_anchor_guided_by_key["right_eye"] = True
        landmark_confidences_by_key["right_eye"] = right_confidence
        for key in ("eye",):
            observed_anchor_guided_by_key[key] = True
            landmark_confidences_by_key[key] = 0.5 * (left_confidence + right_confidence)
    else:
        landmark_prior = _paired_eye_landmark_prior(
            mask,
            luma_image=luma_image,
            bbox_xyxy=observation.bbox_xyxy,
            dark_cutoff=dark_cutoff,
        )
        if landmark_prior is not None:
            dark = dark_all
            left_eye = np.asarray(landmark_prior["left_eye_mask"], dtype=np.bool_)
            right_eye = np.asarray(landmark_prior["right_eye_mask"], dtype=np.bool_)
            left_eye_localization = dict(landmark_prior["left_eye_localization"])
            right_eye_localization = dict(landmark_prior["right_eye_localization"])
            iris_search_region = mask & np.asarray(
                landmark_prior["eye_search_region"], dtype=np.bool_
            )
            landmark_confidence = float(landmark_prior["confidence"])
            left_center = _mask_center_xy(left_eye)
            right_center = _mask_center_xy(right_eye)
            if left_center is not None:
                left_eye_expected_xy = left_center
            if right_center is not None:
                right_eye_expected_xy = right_center
            if left_center is not None and right_center is not None:
                iris_expected_xy = (
                    0.5 * (left_center[0] + right_center[0]),
                    0.5 * (left_center[1] + right_center[1]),
                )
        else:
            dark = dark_all & upper
            left_eye_candidate = dark & left_eye_region
            right_eye_candidate = dark & right_eye_region
            left_eye, left_eye_localization = _localized_feature_component_mask(
                left_eye_candidate,
                luma_image=luma_image,
                expected_center_xy=left_eye_expected_xy,
                min_pixels=8,
                max_components=2,
                fallback_mask=left_eye_candidate,
            )
            right_eye, right_eye_localization = _localized_feature_component_mask(
                right_eye_candidate,
                luma_image=luma_image,
                expected_center_xy=right_eye_expected_xy,
                min_pixels=8,
                max_components=2,
                fallback_mask=right_eye_candidate,
            )
            iris_search_region = mask & upper & (left_eye_region | right_eye_region)
            landmark_confidence = 0.0
    eye_mask = left_eye | right_eye
    if np.count_nonzero(eye_mask) < 8:
        eye_mask = dark
    iris_mask = (
        iris_search_region
        & (luma_image > dark_cutoff)
        & (luma_image < min(bright_cutoff, 0.74))
        & (chroma_image > 0.035)
    )
    if np.count_nonzero(iris_mask) < 2:
        iris_mask = iris_search_region & (luma_image > dark_cutoff) & (luma_image < bright_cutoff)
    if np.count_nonzero(iris_mask) < 2:
        iris_mask = eye_mask
    iris_mask, iris_localization = _localized_feature_component_mask(
        iris_mask,
        luma_image=luma_image,
        expected_center_xy=iris_expected_xy,
        min_pixels=2,
        max_components=4,
        fallback_mask=iris_mask,
    )
    x_coords = np.arange(mask.shape[1], dtype=np.float64)[None, :]
    face_mid_x = float(iris_expected_xy[0])
    left_iris_region = iris_search_region & (x_coords <= face_mid_x)
    right_iris_region = iris_search_region & (x_coords >= face_mid_x)
    left_iris_candidate = iris_mask & left_iris_region
    right_iris_candidate = iris_mask & right_iris_region
    if np.count_nonzero(left_iris_candidate) < 2:
        left_iris_candidate = left_eye
    if np.count_nonzero(right_iris_candidate) < 2:
        right_iris_candidate = right_eye
    left_iris, left_iris_localization = _localized_feature_component_mask(
        left_iris_candidate,
        luma_image=luma_image,
        expected_center_xy=left_eye_expected_xy,
        min_pixels=2,
        max_components=2,
        fallback_mask=left_eye,
    )
    right_iris, right_iris_localization = _localized_feature_component_mask(
        right_iris_candidate,
        luma_image=luma_image,
        expected_center_xy=right_eye_expected_xy,
        min_pixels=2,
        max_components=2,
        fallback_mask=right_eye,
    )
    left_pupil_candidate = _dark_eye_core_mask(left_eye, luma_image)
    right_pupil_candidate = _dark_eye_core_mask(right_eye, luma_image)
    left_pupil, left_pupil_localization = _localized_feature_component_mask(
        left_pupil_candidate,
        luma_image=luma_image,
        expected_center_xy=left_eye_expected_xy,
        min_pixels=2,
        max_components=1,
        fallback_mask=left_eye,
    )
    right_pupil, right_pupil_localization = _localized_feature_component_mask(
        right_pupil_candidate,
        luma_image=luma_image,
        expected_center_xy=right_eye_expected_xy,
        min_pixels=2,
        max_components=1,
        fallback_mask=right_eye,
    )
    nose_region, muzzle_region, mouth_region = _face_regions_from_eye_pair(
        mask.shape,
        observation.bbox_xyxy,
        left_eye,
        right_eye,
        fallback_nose_region=face_center,
        fallback_muzzle_region=face_center,
        fallback_mouth_region=lower_center,
    )

    warm = (
        mask
        & (red_excess > 0.045)
        & (chroma_image > 0.060)
        & (image[:, :, 0] > 0.38)
        & (luma_image > 0.22)
        & (luma_image < 0.90)
    )
    pink = (
        warm
        & (red_excess > 0.075)
        & (image[:, :, 0] > 0.46)
        & (image[:, :, 0] > image[:, :, 1] + 0.030)
        & (image[:, :, 0] > image[:, :, 2] + 0.020)
    )
    nose_anchor_guided = False
    nose_anchor = anchor_targets.get("nose_tip")
    if nose_anchor is not None:
        nose_expected_xy = (nose_anchor[0], nose_anchor[1])
        nose_anchor_region = _anchor_feature_region(
            mask.shape,
            center_xy=nose_expected_xy,
            radius_xy=(
                max(float(anchor_prior["eye_span"]) * 0.34, bbox_width * 0.050, 3.0)
                if anchor_prior is not None
                else max(bbox_width * 0.050, 3.0),
                max(float(anchor_prior["eye_span"]) * 0.26, bbox_height * 0.040, 3.0)
                if anchor_prior is not None
                else max(bbox_height * 0.040, 3.0),
            ),
            foreground_mask=mask,
        )
        nose_mask = pink & nose_anchor_region
        if np.count_nonzero(nose_mask) < 6:
            nose_mask = warm & nose_anchor_region
        nose_anchor_guided = bool(np.count_nonzero(nose_mask) >= 6)
    else:
        nose_mask = pink & nose_region
    if np.count_nonzero(nose_mask) < 6:
        nose_mask = pink & face_center
    if np.count_nonzero(nose_mask) < 6:
        nose_mask = pink
    if np.count_nonzero(nose_mask) < 6:
        nose_mask = warm & face_center
    if np.count_nonzero(nose_mask) < 6:
        nose_mask = warm & lower_center
    if np.count_nonzero(nose_mask) < 6:
        nose_mask = warm
    nose_mask, nose_localization = _localized_feature_component_mask(
        nose_mask,
        luma_image=luma_image,
        expected_center_xy=nose_expected_xy,
        min_pixels=6,
        max_components=2,
        fallback_mask=nose_mask,
    )
    if nose_anchor_guided and np.count_nonzero(nose_mask) > 0:
        observed_anchor_guided_by_key["nose"] = True
        landmark_confidences_by_key["nose"] = float(nose_anchor[2])
    ear_mask = warm & upper_sides
    if np.count_nonzero(ear_mask) < 6:
        ear_mask = warm

    muzzle_anchor_guided = False
    muzzle_localization: dict[str, float | int]
    muzzle_anchor = anchor_targets.get("muzzle_center")
    if muzzle_anchor is not None:
        muzzle_expected_xy = (muzzle_anchor[0], muzzle_anchor[1])
        muzzle_anchor_region = _anchor_feature_region(
            mask.shape,
            center_xy=muzzle_expected_xy,
            radius_xy=(
                max(float(anchor_prior["eye_span"]) * 0.72, bbox_width * 0.100, 5.0)
                if anchor_prior is not None
                else max(bbox_width * 0.100, 5.0),
                max(float(anchor_prior["eye_span"]) * 0.44, bbox_height * 0.060, 4.0)
                if anchor_prior is not None
                else max(bbox_height * 0.060, 4.0),
            ),
            foreground_mask=mask,
        )
        muzzle_mask = (
            mask & muzzle_anchor_region & (luma_image >= bright_cutoff) & (chroma_image < 0.24)
        )
        if np.count_nonzero(muzzle_mask) < 12:
            muzzle_mask = mask & muzzle_anchor_region & (luma_image >= dark_cutoff)
        if np.count_nonzero(muzzle_mask) >= 12:
            muzzle_mask, muzzle_localization = _localized_feature_component_mask(
                muzzle_mask,
                luma_image=luma_image,
                expected_center_xy=muzzle_expected_xy,
                min_pixels=12,
                max_components=4,
                fallback_mask=mask & muzzle_anchor_region,
            )
            muzzle_anchor_guided = True
        else:
            muzzle_mask = (
                mask & muzzle_region & (luma_image >= bright_cutoff) & (chroma_image < 0.22)
            )
            muzzle_localization = _identity_localization_metrics(muzzle_mask)
    else:
        muzzle_mask = mask & muzzle_region & (luma_image >= bright_cutoff) & (chroma_image < 0.22)
        muzzle_localization = _identity_localization_metrics(muzzle_mask)
    if not muzzle_anchor_guided and np.count_nonzero(muzzle_mask) < 12:
        muzzle_mask = mask & face_center & (luma_image >= bright_cutoff) & (chroma_image < 0.22)
        muzzle_localization = _identity_localization_metrics(muzzle_mask)
    if not muzzle_anchor_guided and np.count_nonzero(muzzle_mask) < 12:
        muzzle_mask = mask & (luma_image >= bright_cutoff) & (chroma_image < 0.24)
        muzzle_localization = _identity_localization_metrics(muzzle_mask)
    if muzzle_anchor_guided and np.count_nonzero(muzzle_mask) > 0:
        observed_anchor_guided_by_key["muzzle"] = True
        landmark_confidences_by_key["muzzle"] = float(muzzle_anchor[2])
    mouth_mask = dark & mouth_region
    if np.count_nonzero(mouth_mask) < 4:
        mouth_mask = dark & lower_center
    if np.count_nonzero(mouth_mask) < 4:
        mouth_mask = eye_mask

    eye_count = int(np.count_nonzero(eye_mask))
    iris_count = int(np.count_nonzero(iris_mask))
    left_iris_count = int(np.count_nonzero(left_iris))
    right_iris_count = int(np.count_nonzero(right_iris))
    left_pupil_count = int(np.count_nonzero(left_pupil))
    right_pupil_count = int(np.count_nonzero(right_pupil))
    nose_count = int(np.count_nonzero(nose_mask))
    pink_nose_count = int(np.count_nonzero(pink & nose_region))
    muzzle_count = int(np.count_nonzero(muzzle_mask))
    ear_count = int(np.count_nonzero(ear_mask))
    mouth_count = int(np.count_nonzero(mouth_mask))
    left_count = int(np.count_nonzero(left_eye))
    right_count = int(np.count_nonzero(right_eye))
    foreground_count = max(int(foreground.shape[0]), 1)
    paired_eye_score = min(left_count, right_count) / foreground_count
    warm_score = nose_count / foreground_count
    muzzle_score = muzzle_count / foreground_count
    media_bonus = 1.0 if metadata.get("media_type") == "photo" else 0.0
    border_penalty = float(metadata.get("quality", {}).get("touches_image_border", 0.0)) * 0.10
    view_quality = _raw_texture_view_quality(observation, metadata)
    score = (
        media_bonus
        + 0.80 * view_quality.frontal_face_confidence
        + 0.45 * view_quality.view_weight
        + min(landmark_confidence * 2.2, 2.2)
        + min(paired_eye_score * 500.0, 1.8)
        + min(warm_score * 180.0, 1.2)
        + min((pink_nose_count / foreground_count) * 240.0, 1.0)
        + min(muzzle_score * 20.0, 0.5)
        - border_penalty
    )

    eye_rgb = _sample_low_luma_color(image[eye_mask], fallback_eye)
    iris_rgb = _sample_iris_color(image[iris_mask], eye_rgb)
    pupil_rgb = _sample_dark_pupil_color(image[left_pupil | right_pupil], eye_rgb)
    nose_rgb = _sample_warm_color(image[nose_mask], fallback_nose)
    muzzle_rgb = _sample_median_color(image[muzzle_mask], fallback_muzzle)
    ear_rgb = _sample_warm_color(image[ear_mask], nose_rgb)
    mouth_rgb = _sample_low_luma_color(image[mouth_mask], eye_rgb)
    sample_counts = {
        "eye": eye_count,
        "iris": iris_count,
        "left_eye": left_count,
        "right_eye": right_count,
        "left_iris": left_iris_count,
        "right_iris": right_iris_count,
        "left_pupil": left_pupil_count,
        "right_pupil": right_pupil_count,
        "nose": nose_count,
        "pink_nose": pink_nose_count,
        "muzzle": muzzle_count,
        "ear_inner": ear_count,
        "mouth": mouth_count,
    }
    source_patch_crops = {
        "left_eye": _patch_from_feature_mask(image, left_eye, sample_key="left_eye"),
        "right_eye": _patch_from_feature_mask(image, right_eye, sample_key="right_eye"),
        "eye": _patch_from_feature_mask(image, eye_mask, sample_key="eye"),
        "iris": _patch_from_feature_mask(image, iris_mask, sample_key="iris"),
        "left_iris": _patch_from_feature_mask(
            image,
            left_iris,
            sample_key="left_iris",
        ),
        "right_iris": _patch_from_feature_mask(
            image,
            right_iris,
            sample_key="right_iris",
        ),
        "left_pupil": _patch_from_feature_mask(
            image,
            left_pupil,
            sample_key="left_pupil",
        ),
        "right_pupil": _patch_from_feature_mask(
            image,
            right_pupil,
            sample_key="right_pupil",
        ),
        "nose": _patch_from_feature_mask(image, nose_mask, sample_key="nose"),
        "muzzle": _patch_from_feature_mask(image, muzzle_mask, sample_key="muzzle"),
        "ear_inner": _patch_from_feature_mask(image, ear_mask, sample_key="ear_inner"),
        "mouth": _patch_from_feature_mask(image, mouth_mask, sample_key="mouth"),
    }
    source_patches = {
        key: crop.image for key, crop in source_patch_crops.items() if crop is not None
    }
    source_patch_masks = {
        key: crop.mask for key, crop in source_patch_crops.items() if crop is not None
    }
    source_patch_mask_observed_counts = {
        key: crop.observed_mask_texel_count
        for key, crop in source_patch_crops.items()
        if crop is not None
    }
    source_patch_mask_densified_counts = {
        key: crop.densified_mask_texel_count
        for key, crop in source_patch_crops.items()
        if crop is not None
    }
    source_patch_mask_densification_gains = {
        key: crop.densification_gain for key, crop in source_patch_crops.items() if crop is not None
    }
    localization_counts = {
        "left_eye": left_eye_localization,
        "right_eye": right_eye_localization,
        "eye": _combined_localization_metrics(
            left_eye_localization,
            right_eye_localization,
        ),
        "iris": iris_localization,
        "left_iris": left_iris_localization,
        "right_iris": right_iris_localization,
        "left_pupil": left_pupil_localization,
        "right_pupil": right_pupil_localization,
        "nose": nose_localization,
        "muzzle": muzzle_localization,
        "ear_inner": _identity_localization_metrics(ear_mask),
        "mouth": _identity_localization_metrics(mouth_mask),
    }
    source_patch_localization_candidate_counts = {
        key: int(value["candidate_count"]) for key, value in localization_counts.items()
    }
    source_patch_localization_selected_counts = {
        key: int(value["selected_count"]) for key, value in localization_counts.items()
    }
    source_patch_localization_fractions = {
        key: float(value["selected_fraction"]) for key, value in localization_counts.items()
    }
    source_patch_landmark_confidences = {
        key: (
            float(landmark_confidences_by_key.get(key, landmark_confidence))
            if key in _LANDMARK_ALIGNED_SOURCE_PATCH_KEYS
            else 0.0
        )
        for key in localization_counts
    }
    source_patch_observed_anchor_guided = {
        key: bool(observed_anchor_guided_by_key.get(key, False)) for key in source_patches
    }
    source_patch_scores = {
        key: _source_patch_quality_score(
            key,
            patch=patch,
            sample_count=int(sample_counts.get(key, 0)),
            foreground_count=foreground_count,
        )
        for key, patch in source_patches.items()
    }
    source_patch_frontal_face_confidences = {
        key: view_quality.frontal_face_confidence for key in source_patches
    }
    source_patch_view_weights = {key: view_quality.view_weight for key in source_patches}
    source_patch_view_labels = {key: view_quality.label for key in source_patches}
    source_patch_frontal_photo_priority_diagnostics = {
        key: _source_patch_frontal_photo_priority_diagnostic(
            key,
            metadata=metadata,
            view_quality=view_quality,
            patch=source_patches[key],
            selected_count=int(source_patch_localization_selected_counts.get(key, 0)),
            candidate_count=int(source_patch_localization_candidate_counts.get(key, 0)),
            sample_count=int(sample_counts.get(key, 0)),
            source_quality_score=float(source_patch_scores.get(key, 0.0)),
        )
        for key in source_patches
    }
    source_patch_frontal_photo_priorities = {
        key: float(diagnostic["priority"])
        for key, diagnostic in source_patch_frontal_photo_priority_diagnostics.items()
    }
    source_patch_direct_photo_candidates = {
        key: _source_patch_direct_uploaded_photo_candidate(metadata) for key in source_patches
    }
    source_patch_view_weighted_ranks = {
        key: _source_patch_rank_with_direct_photo_penalty(
            _source_patch_view_weighted_rank(
                key,
                landmark_confidence=float(source_patch_landmark_confidences.get(key, 0.0)),
                feedback_score=0.0,
                quality_score=float(source_patch_scores.get(key, 0.0)),
                view_weight=view_quality.view_weight,
                frontal_face_confidence=view_quality.frontal_face_confidence,
                feature_contrast_score=_source_patch_feature_contrast_score_from_colors(
                    key,
                    eye_rgb=eye_rgb,
                    iris_rgb=iris_rgb,
                    pupil_rgb=pupil_rgb,
                    nose_rgb=nose_rgb,
                    muzzle_rgb=muzzle_rgb,
                    ear_inner_rgb=ear_rgb,
                    mouth_rgb=mouth_rgb,
                ),
                localization_selected_count=int(
                    source_patch_localization_selected_counts.get(key, 0)
                ),
                localization_candidate_count=int(
                    source_patch_localization_candidate_counts.get(key, 0)
                ),
                source_density_score=_source_patch_density_score(
                    key,
                    source_patches[key],
                    sample_count=int(sample_counts.get(key, 0)),
                ),
                source_detail_score=_source_patch_diagnostic_detail_score(
                    source_patches[key],
                ),
                frontal_photo_priority=float(source_patch_frontal_photo_priorities.get(key, 0.0)),
            )
            * _source_patch_foreground_color_score(
                key,
                source_patches[key],
                mask=source_patch_masks.get(key),
            ),
            direct_photo_candidate=bool(source_patch_direct_photo_candidates.get(key, False)),
            frontal_photo_priority=float(source_patch_frontal_photo_priorities.get(key, 0.0)),
        )
        for key in source_patches
    }
    allowed_source_patch_sample_keys = _source_patch_allowed_sample_keys(metadata)
    return _FaceFeaturePalette(
        source_frame_id=observation.frame_id,
        source_score=float(score),
        eye_rgb=eye_rgb,
        iris_rgb=iris_rgb,
        nose_rgb=nose_rgb,
        muzzle_rgb=muzzle_rgb,
        ear_inner_rgb=ear_rgb,
        mouth_rgb=mouth_rgb,
        sample_counts=sample_counts,
        source_patches=source_patches,
        source_patch_masks=source_patch_masks,
        source_patch_mask_observed_counts=source_patch_mask_observed_counts,
        source_patch_mask_densified_counts=source_patch_mask_densified_counts,
        source_patch_mask_densification_gains=source_patch_mask_densification_gains,
        source_patch_localization_candidate_counts=(source_patch_localization_candidate_counts),
        source_patch_localization_selected_counts=(source_patch_localization_selected_counts),
        source_patch_localization_fractions=source_patch_localization_fractions,
        source_patch_landmark_confidences=source_patch_landmark_confidences,
        source_patch_scores=source_patch_scores,
        source_patch_frame_ids={key: observation.frame_id for key in source_patches},
        source_patch_frontal_face_confidences=source_patch_frontal_face_confidences,
        source_patch_view_weights=source_patch_view_weights,
        source_patch_view_labels=source_patch_view_labels,
        source_patch_view_weighted_ranks=source_patch_view_weighted_ranks,
        source_patch_feedback_components={},
        source_patch_feedback_scores={},
        source_patch_observed_anchor_guided=source_patch_observed_anchor_guided,
        source_patch_frontal_photo_priorities=source_patch_frontal_photo_priorities,
        source_patch_frontal_photo_priority_diagnostics=(
            source_patch_frontal_photo_priority_diagnostics
        ),
        source_patch_direct_photo_candidates=source_patch_direct_photo_candidates,
        allowed_source_patch_sample_keys=allowed_source_patch_sample_keys,
        pupil_rgb=pupil_rgb,
    )


def _source_patch_rank_with_direct_photo_penalty(
    rank: float,
    *,
    direct_photo_candidate: bool,
    frontal_photo_priority: float,
) -> float:
    if direct_photo_candidate and frontal_photo_priority <= 0.0:
        return float(rank * _ZERO_PRIORITY_DIRECT_PHOTO_RANK_PENALTY)
    return float(rank)


def _direct_photo_nose_color_score(nose_rgb: NDArray[np.uint8]) -> float:
    rgb = np.asarray(nose_rgb, dtype=np.float64) / 255.0
    if rgb.shape != (3,) or not np.all(np.isfinite(rgb)):
        return 0.0
    red, green, blue = float(rgb[0]), float(rgb[1]), float(rgb[2])
    red_level = _clamp01((red - 0.62) / 0.18)
    warm_margin = _clamp01((red - max(green, blue)) / 0.16)
    pink_balance = 1.0 - _clamp01(abs(green - blue) / 0.16)
    return float(_clamp01(0.45 * red_level + 0.40 * warm_margin + 0.15 * pink_balance))


def _patch_from_feature_mask(
    image: NDArray[np.float32],
    mask: NDArray[np.bool_],
    *,
    sample_key: str,
) -> _SourcePatchCrop | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    height, width = image.shape[:2]
    feature_width = int(xs.max() - xs.min() + 1)
    feature_height = int(ys.max() - ys.min() + 1)
    crop_profile = _source_patch_crop_profile(sample_key)
    pad_x = max(
        int(crop_profile["min_padding"]),
        int(round(feature_width * float(crop_profile["padding_fraction"]))),
    )
    pad_y = max(
        int(crop_profile["min_padding"]),
        int(round(feature_height * float(crop_profile["padding_fraction"]))),
    )
    x0 = max(int(xs.min()) - pad_x, 0)
    x1 = min(int(xs.max()) + pad_x + 1, width)
    y0 = max(int(ys.min()) - pad_y, 0)
    y1 = min(int(ys.max()) + pad_y + 1, height)
    crop = image[y0:y1, x0:x1, :]
    local_mask = mask[y0:y1, x0:x1]
    if crop.size == 0 or not np.any(local_mask):
        return None
    observed_mask = _source_observed_mask_from_local_mask(
        crop,
        local_mask,
        sample_key=sample_key,
    )
    if not np.any(observed_mask):
        return None
    masked_pixels = crop[observed_mask]
    fill = np.median(masked_pixels, axis=0)
    patch = np.zeros_like(crop)
    patch[:, :] = fill[None, None, :]
    patch[observed_mask] = crop[observed_mask]
    source_shape_mask = _source_shape_mask_from_local_mask(
        observed_mask,
        sample_key=sample_key,
    )
    observed_mask_texel_count = int(np.count_nonzero(observed_mask))
    densified_mask_texel_count = int(np.count_nonzero(source_shape_mask))
    return _SourcePatchCrop(
        image=_rgb_float_to_uint8(patch),
        mask=(source_shape_mask.astype(np.uint8) * np.uint8(255)),
        observed_mask_texel_count=observed_mask_texel_count,
        densified_mask_texel_count=densified_mask_texel_count,
        densification_gain=float(densified_mask_texel_count / max(observed_mask_texel_count, 1)),
    )


def _dark_eye_core_mask(
    eye_mask: NDArray[np.bool_],
    luma_image: NDArray[np.floating],
) -> NDArray[np.bool_]:
    mask = np.asarray(eye_mask, dtype=np.bool_)
    if not np.any(mask):
        return mask
    luma = np.asarray(luma_image, dtype=np.float64)
    values = luma[mask]
    cutoff = float(np.quantile(values, 0.45))
    core = mask & (luma <= cutoff)
    if np.count_nonzero(core) < 2:
        return mask
    return core


def _source_patch_quality_score(
    sample_key: str,
    *,
    patch: NDArray[np.uint8],
    sample_count: int,
    foreground_count: int,
) -> float:
    if patch.size == 0 or sample_count <= 0:
        return 0.0
    height, width = patch.shape[:2]
    if height <= 0 or width <= 0:
        return 0.0
    rgb = patch.astype(np.float64) / 255.0
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    luma_std = float(np.std(luma))
    unique_count = _bounded_unique_rgb_count(patch, max_samples=4096)
    aspect_score = min(height, width) / max(max(height, width), 1)
    sample_fraction = sample_count / max(foreground_count, 1)
    count_score = min(log1p(float(sample_count)) / log1p(128.0), 1.0)
    density_score = min(sample_fraction * 300.0, 1.0)
    detail_score = min(luma_std / 0.075, 1.0)
    unique_score = min(unique_count / 16.0, 1.0)
    area_score = min((height * width) / 512.0, 1.0)
    feature_weight = {
        "left_eye": 1.15,
        "right_eye": 1.15,
        "eye": 1.10,
        "iris": 1.05,
        "left_iris": 1.20,
        "right_iris": 1.20,
        "left_pupil": 1.25,
        "right_pupil": 1.25,
        "nose": 1.00,
        "muzzle": 0.95,
        "ear_inner": 0.95,
        "mouth": 1.05,
    }.get(sample_key, 1.0)
    raw_score = (
        0.28 * count_score
        + 0.24 * density_score
        + 0.22 * detail_score
        + 0.16 * unique_score
        + 0.10 * area_score
    )
    compactness_multiplier = 0.25 + 0.75 * aspect_score
    footprint_multiplier = _source_patch_feature_footprint_multiplier(
        sample_key,
        width=width,
        height=height,
    )
    return float(
        max(
            0.0,
            min(
                raw_score * compactness_multiplier * feature_weight * footprint_multiplier,
                1.0,
            ),
        )
    )


def _bounded_unique_rgb_count(
    patch: NDArray[np.uint8],
    *,
    max_samples: int,
) -> int:
    """Estimate RGB variety with deterministic bounded work for large source patches."""

    if patch.size == 0 or max_samples <= 0:
        return 0
    pixels = np.asarray(patch, dtype=np.uint8).reshape(-1, 3)
    if pixels.shape[0] > max_samples:
        sample_indices = np.linspace(
            0,
            pixels.shape[0] - 1,
            num=max_samples,
            dtype=np.int64,
        )
        pixels = pixels[sample_indices]
    quantized = pixels // np.uint8(4)
    return int(np.unique(quantized, axis=0).shape[0])


def _source_patch_feature_footprint_multiplier(
    sample_key: str,
    *,
    width: int,
    height: int,
) -> float:
    max_area = {
        "left_eye": 18_000.0,
        "right_eye": 18_000.0,
        "eye": 34_000.0,
        "iris": 34_000.0,
        "left_iris": 10_000.0,
        "right_iris": 10_000.0,
        "left_pupil": 7_500.0,
        "right_pupil": 7_500.0,
        "nose": 24_000.0,
        "mouth": 18_000.0,
    }.get(sample_key)
    if max_area is None:
        return 1.0
    area = max(float(width * height), 1.0)
    if area <= max_area:
        return 1.0
    return float(np.clip(np.sqrt(max_area / area), 0.30, 1.0))


def _source_patch_crop_profile(sample_key: str) -> dict[str, float | int]:
    raw = _SOURCE_PATCH_CROP_PROFILES.get(
        sample_key,
        {"padding_fraction": 0.35, "min_padding": 2},
    )
    return {
        "padding_fraction": float(raw["padding_fraction"]),
        "min_padding": int(raw["min_padding"]),
    }


def _paired_eye_landmark_prior(
    mask: NDArray[np.bool_],
    *,
    luma_image: NDArray[np.floating],
    bbox_xyxy: NDArray[np.floating],
    dark_cutoff: float,
) -> dict[str, object] | None:
    foreground = np.asarray(mask, dtype=np.bool_)
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count <= 0:
        return None
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    dark_threshold = min(float(dark_cutoff), 0.34)
    dark = foreground & (np.asarray(luma_image, dtype=np.float64) <= dark_threshold)
    min_pixels = max(1, min(8, foreground_count // 64))
    components = _feature_components(dark, luma_image=luma_image)
    eye_candidates: list[dict[str, object]] = []
    max_count = max(64, int(round(foreground_count * 0.035)))
    max_width = max(8.0, bbox_width * 0.18)
    max_height = max(8.0, bbox_height * 0.22)
    for component in components:
        count = int(component["count"])
        box_width = float(component["box_width"])
        box_height = float(component["box_height"])
        compactness = float(component["compactness"])
        if count < min_pixels or count > max_count:
            continue
        if box_width > max_width or box_height > max_height:
            continue
        if compactness < 0.18:
            continue
        eye_candidates.append(component)

    if len(eye_candidates) < 2:
        return _tiny_paired_eye_landmark_prior(
            foreground,
            dark_mask=dark,
            bbox_xyxy=bbox_xyxy,
        )

    best_pair: tuple[float, dict[str, object], dict[str, object]] | None = None
    min_dx = max(2.0, bbox_width * 0.025)
    max_dx = max(min_dx + 1.0, bbox_width * 0.55)
    for first_index, first in enumerate(eye_candidates):
        for second in eye_candidates[first_index + 1 :]:
            cx1 = float(first["center_x"])
            cy1 = float(first["center_y"])
            cx2 = float(second["center_x"])
            cy2 = float(second["center_y"])
            dx = abs(cx2 - cx1)
            dy = abs(cy2 - cy1)
            if dx < min_dx or dx > max_dx:
                continue
            if dy > max(4.0, bbox_height * 0.22, dx * 0.55):
                continue
            count1 = int(first["count"])
            count2 = int(second["count"])
            balance = min(count1, count2) / max(max(count1, count2), 1)
            size_score = min(log1p(float(count1 + count2)) / log1p(512.0), 1.0)
            darkness_score = 1.0 - 0.5 * (float(first["mean_luma"]) + float(second["mean_luma"]))
            compactness_score = 0.5 * (float(first["compactness"]) + float(second["compactness"]))
            y_alignment = 1.0 - min(dy / max(bbox_height * 0.22, 1.0), 1.0)
            separation_score = min(dx / max(bbox_width * 0.10, 1.0), 1.0)
            score = (
                0.24 * balance
                + 0.22 * size_score
                + 0.22 * darkness_score
                + 0.16 * compactness_score
                + 0.10 * y_alignment
                + 0.06 * separation_score
            )
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, first, second)

    if best_pair is None:
        return _tiny_paired_eye_landmark_prior(
            foreground,
            dark_mask=dark,
            bbox_xyxy=bbox_xyxy,
        )

    score, first, second = best_pair
    left_component, right_component = (
        (first, second)
        if float(first["center_x"]) <= float(second["center_x"])
        else (second, first)
    )
    left_mask = _component_mask(foreground.shape, left_component)
    right_mask = _component_mask(foreground.shape, right_component)
    eye_search_region = _component_pair_search_region(
        foreground.shape,
        left_component,
        right_component,
        pad_x=max(4, int(round(bbox_width * 0.035))),
        pad_y=max(4, int(round(bbox_height * 0.050))),
    )
    left_count = int(left_component["count"])
    right_count = int(right_component["count"])
    return {
        "confidence": float(np.clip(score, 0.0, 1.0)),
        "left_eye_mask": left_mask,
        "right_eye_mask": right_mask,
        "eye_search_region": eye_search_region,
        "left_eye_localization": {
            "candidate_count": left_count,
            "selected_count": int(np.count_nonzero(left_mask)),
            "selected_fraction": 1.0,
        },
        "right_eye_localization": {
            "candidate_count": right_count,
            "selected_count": int(np.count_nonzero(right_mask)),
            "selected_fraction": 1.0,
        },
    }


def _tiny_paired_eye_landmark_prior(
    mask: NDArray[np.bool_],
    *,
    dark_mask: NDArray[np.bool_],
    bbox_xyxy: NDArray[np.floating],
) -> dict[str, object] | None:
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    if bbox_width > 24.0 or bbox_height > 24.0:
        return None
    upper = _bbox_region_mask(
        np.asarray(mask).shape,
        bbox_xyxy,
        x_range=(0.0, 1.0),
        y_range=(0.0, 0.70),
    )
    ys, xs = np.nonzero(np.asarray(dark_mask, dtype=np.bool_) & upper)
    if xs.size < 2:
        return None
    order = np.lexsort((ys, xs))
    left_x = int(xs[order[0]])
    left_y = int(ys[order[0]])
    right_x = int(xs[order[-1]])
    right_y = int(ys[order[-1]])
    if right_x <= left_x:
        return None
    left_mask = np.zeros(np.asarray(mask).shape, dtype=np.bool_)
    right_mask = np.zeros(np.asarray(mask).shape, dtype=np.bool_)
    left_mask[left_y, left_x] = True
    right_mask[right_y, right_x] = True
    eye_search_region = _component_pair_search_region(
        np.asarray(mask).shape,
        {
            "xs": np.asarray([left_x], dtype=np.int64),
            "ys": np.asarray([left_y], dtype=np.int64),
        },
        {
            "xs": np.asarray([right_x], dtype=np.int64),
            "ys": np.asarray([right_y], dtype=np.int64),
        },
        pad_x=1,
        pad_y=1,
    )
    return {
        "confidence": 0.5,
        "left_eye_mask": left_mask,
        "right_eye_mask": right_mask,
        "eye_search_region": eye_search_region,
        "left_eye_localization": {
            "candidate_count": 1,
            "selected_count": 1,
            "selected_fraction": 1.0,
        },
        "right_eye_localization": {
            "candidate_count": 1,
            "selected_count": 1,
            "selected_fraction": 1.0,
        },
    }


def _component_mask(
    shape: tuple[int, int],
    component: dict[str, object],
) -> NDArray[np.bool_]:
    selected = np.zeros(shape, dtype=np.bool_)
    ys = np.asarray(component["ys"], dtype=np.int64)
    xs = np.asarray(component["xs"], dtype=np.int64)
    selected[ys, xs] = True
    return selected


def _component_pair_search_region(
    shape: tuple[int, int],
    first: dict[str, object],
    second: dict[str, object],
    *,
    pad_x: int,
    pad_y: int,
) -> NDArray[np.bool_]:
    xs = np.concatenate(
        [
            np.asarray(first["xs"], dtype=np.int64),
            np.asarray(second["xs"], dtype=np.int64),
        ]
    )
    ys = np.concatenate(
        [
            np.asarray(first["ys"], dtype=np.int64),
            np.asarray(second["ys"], dtype=np.int64),
        ]
    )
    if xs.size == 0 or ys.size == 0:
        return np.zeros(shape, dtype=np.bool_)
    height, width = shape
    x0 = max(int(xs.min()) - int(pad_x), 0)
    x1 = min(int(xs.max()) + int(pad_x) + 1, width)
    y0 = max(int(ys.min()) - int(pad_y), 0)
    y1 = min(int(ys.max()) + int(pad_y) + 1, height)
    region = np.zeros(shape, dtype=np.bool_)
    region[y0:y1, x0:x1] = True
    return region


def _localized_feature_component_mask(
    candidate_mask: NDArray[np.bool_],
    *,
    luma_image: NDArray[np.floating],
    expected_center_xy: tuple[float, float],
    min_pixels: int,
    max_components: int,
    fallback_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], dict[str, float | int]]:
    candidate = np.asarray(candidate_mask, dtype=np.bool_)
    candidate_count = int(np.count_nonzero(candidate))
    fallback = np.asarray(fallback_mask, dtype=np.bool_)
    if candidate_count < max(int(min_pixels), 1):
        return _bounded_feature_fallback_mask(candidate, fallback)

    components = _feature_components(candidate, luma_image=luma_image)
    if not components:
        return _bounded_feature_fallback_mask(candidate, fallback)

    ex, ey = expected_center_xy
    height, width = candidate.shape
    norm = max(float(np.hypot(width, height)), 1.0)
    scored: list[tuple[float, int, dict[str, object]]] = []
    for index, component in enumerate(components):
        count = int(component["count"])
        if count < int(min_pixels):
            continue
        cx = float(component["center_x"])
        cy = float(component["center_y"])
        distance = float(np.hypot(cx - ex, cy - ey) / norm)
        size_score = min(log1p(float(count)) / log1p(float(candidate_count)), 1.0)
        darkness_score = 1.0 - float(component["mean_luma"])
        compactness = float(component["compactness"])
        score = 0.42 * size_score + 0.28 * darkness_score + 0.20 * compactness - 1.15 * distance
        scored.append((score, -index, component))

    if not scored:
        return _bounded_feature_fallback_mask(candidate, fallback)

    selected = np.zeros_like(candidate)
    for _score, _tie_breaker, component in sorted(scored, reverse=True)[:max_components]:
        _paint_component_selection(selected, candidate, component)
    selected_count = int(np.count_nonzero(selected))
    if selected_count < max(int(min_pixels), 1):
        selected, _metrics = _bounded_feature_fallback_mask(candidate, fallback)
        selected_count = int(np.count_nonzero(selected))
    return selected, {
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "selected_fraction": float(selected_count / max(candidate_count, 1)),
    }


def _bounded_feature_fallback_mask(
    candidate: NDArray[np.bool_],
    fallback: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], dict[str, float | int]]:
    candidate = np.asarray(candidate, dtype=np.bool_)
    fallback = np.asarray(fallback, dtype=np.bool_)
    candidate_count = int(np.count_nonzero(candidate))
    if candidate_count > 0:
        selected = candidate & fallback
        if not np.any(selected):
            selected = candidate.copy()
    else:
        selected = fallback.copy()
    selected_count = int(np.count_nonzero(selected))
    return selected, {
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "selected_fraction": float(selected_count / max(candidate_count, 1)),
    }


def _feature_components(
    mask: NDArray[np.bool_],
    *,
    luma_image: NDArray[np.floating],
) -> list[dict[str, object]]:
    ys_all, xs_all = np.nonzero(mask)
    if xs_all.size == 0:
        return []
    total_mask_count = int(xs_all.size)
    y0, y1 = int(ys_all.min()), int(ys_all.max())
    x0, x1 = int(xs_all.min()), int(xs_all.max())
    local_full = np.asarray(mask[y0 : y1 + 1, x0 : x1 + 1], dtype=np.bool_)
    stride = int(max(1, np.ceil(np.sqrt(local_full.size / 65_536.0))))
    local = local_full[::stride, ::stride]
    luma_local = np.asarray(luma_image, dtype=np.float64)[y0 : y1 + 1, x0 : x1 + 1][
        ::stride,
        ::stride,
    ]
    visited = np.zeros(local.shape, dtype=np.bool_)
    components: list[dict[str, object]] = []
    height, width = local.shape
    start_ys, start_xs = np.nonzero(local)
    for start_y, start_x in zip(start_ys, start_xs, strict=True):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        ys: list[int] = []
        xs: list[int] = []
        while stack:
            cy, cx = stack.pop()
            ys.append(cy * stride + y0)
            xs.append(cx * stride + x0)
            for ny in range(max(cy - 1, 0), min(cy + 2, height)):
                for nx in range(max(cx - 1, 0), min(cx + 2, width)):
                    if visited[ny, nx] or not local[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if not xs:
            continue
        xs_array = np.asarray(xs, dtype=np.int64)
        ys_array = np.asarray(ys, dtype=np.int64)
        box_width = int(xs_array.max() - xs_array.min() + 1)
        box_height = int(ys_array.max() - ys_array.min() + 1)
        approx_count = min(int(xs_array.size * stride * stride), total_mask_count)
        compactness = float(min(box_width, box_height) / max(max(box_width, box_height), 1))
        sample_y = np.asarray([(y - y0) // stride for y in ys], dtype=np.int64)
        sample_x = np.asarray([(x - x0) // stride for x in xs], dtype=np.int64)
        components.append(
            {
                "count": approx_count,
                "center_x": float(np.mean(xs_array)),
                "center_y": float(np.mean(ys_array)),
                "mean_luma": float(np.mean(luma_local[sample_y, sample_x])),
                "compactness": compactness,
                "box_width": box_width,
                "box_height": box_height,
                "xs": xs_array,
                "ys": ys_array,
                "stride": stride,
            }
        )
    return components


def _paint_component_selection(
    selected: NDArray[np.bool_],
    candidate: NDArray[np.bool_],
    component: dict[str, object],
) -> None:
    ys = np.asarray(component["ys"], dtype=np.int64)
    xs = np.asarray(component["xs"], dtype=np.int64)
    stride = int(component.get("stride", 1))
    if ys.size == 0 or xs.size == 0:
        return
    if stride <= 1:
        selected[ys, xs] = True
        return
    y0 = max(int(ys.min()) - stride + 1, 0)
    y1 = min(int(ys.max()) + stride, selected.shape[0])
    x0 = max(int(xs.min()) - stride + 1, 0)
    x1 = min(int(xs.max()) + stride, selected.shape[1])
    selected[y0:y1, x0:x1] |= candidate[y0:y1, x0:x1]


def _identity_localization_metrics(mask: NDArray[np.bool_]) -> dict[str, float | int]:
    count = int(np.count_nonzero(mask))
    return {
        "candidate_count": count,
        "selected_count": count,
        "selected_fraction": 1.0 if count > 0 else 0.0,
    }


def _combined_localization_metrics(
    first: dict[str, float | int],
    second: dict[str, float | int],
) -> dict[str, float | int]:
    candidate = int(first["candidate_count"]) + int(second["candidate_count"])
    selected = int(first["selected_count"]) + int(second["selected_count"])
    return {
        "candidate_count": candidate,
        "selected_count": selected,
        "selected_fraction": float(selected / max(candidate, 1)),
    }


def _face_regions_from_eye_pair(
    shape: tuple[int, int],
    bbox_xyxy: NDArray[np.floating],
    left_eye: NDArray[np.bool_],
    right_eye: NDArray[np.bool_],
    *,
    fallback_nose_region: NDArray[np.bool_],
    fallback_muzzle_region: NDArray[np.bool_],
    fallback_mouth_region: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    left_center = _mask_center_xy(left_eye)
    right_center = _mask_center_xy(right_eye)
    if left_center is None or right_center is None:
        return fallback_nose_region, fallback_muzzle_region, fallback_mouth_region

    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    eye_center_x = 0.5 * (left_center[0] + right_center[0])
    eye_center_y = 0.5 * (left_center[1] + right_center[1])
    eye_span = max(abs(right_center[0] - left_center[0]), bbox_width * 0.12)
    nose_center = (
        eye_center_x,
        eye_center_y + max(eye_span * 0.56, bbox_height * 0.070),
    )
    muzzle_center = (
        eye_center_x,
        eye_center_y + max(eye_span * 0.78, bbox_height * 0.115),
    )
    mouth_center = (
        eye_center_x,
        eye_center_y + max(eye_span * 1.02, bbox_height * 0.155),
    )
    nose_region = _image_ellipse_mask(
        shape,
        center_xy=nose_center,
        radius_xy=(
            max(eye_span * 0.48, bbox_width * 0.055),
            max(eye_span * 0.36, bbox_height * 0.045),
        ),
    )
    muzzle_region = _image_ellipse_mask(
        shape,
        center_xy=muzzle_center,
        radius_xy=(
            max(eye_span * 0.95, bbox_width * 0.105),
            max(eye_span * 0.56, bbox_height * 0.070),
        ),
    )
    mouth_region = _image_ellipse_mask(
        shape,
        center_xy=mouth_center,
        radius_xy=(
            max(eye_span * 0.50, bbox_width * 0.060),
            max(eye_span * 0.20, bbox_height * 0.025),
        ),
    )
    return nose_region, muzzle_region, mouth_region


def _mask_center_xy(mask: NDArray[np.bool_]) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(np.median(xs)), float(np.median(ys))


def _image_ellipse_mask(
    shape: tuple[int, int],
    *,
    center_xy: tuple[float, float],
    radius_xy: tuple[float, float],
) -> NDArray[np.bool_]:
    height, width = shape
    cx, cy = center_xy
    rx = max(float(radius_xy[0]), 1.0)
    ry = max(float(radius_xy[1]), 1.0)
    x0 = max(int(np.floor(cx - rx)), 0)
    x1 = min(int(np.ceil(cx + rx)), width - 1)
    y0 = max(int(np.floor(cy - ry)), 0)
    y1 = min(int(np.ceil(cy + ry)), height - 1)
    mask = np.zeros(shape, dtype=np.bool_)
    if x1 < x0 or y1 < y0:
        return mask
    xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = ((grid_x - cx) / rx) ** 2 + ((grid_y - cy) / ry) ** 2 <= 1.0
    mask[y0 : y1 + 1, x0 : x1 + 1] = inside
    return mask


def _bbox_region_mask(
    shape: tuple[int, int],
    bbox_xyxy: NDArray[np.floating],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> NDArray[np.bool_]:
    height, width = shape
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    left = int(np.clip(np.floor(x0 + x_range[0] * bbox_width), 0, width))
    right = int(np.clip(np.ceil(x0 + x_range[1] * bbox_width), 0, width))
    top = int(np.clip(np.floor(y0 + y_range[0] * bbox_height), 0, height))
    bottom = int(np.clip(np.ceil(y0 + y_range[1] * bbox_height), 0, height))
    region = np.zeros(shape, dtype=np.bool_)
    if right > left and bottom > top:
        region[top:bottom, left:right] = True
    return region


def _sample_median_color(
    pixels: NDArray[np.float32],
    fallback: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    if pixels.size == 0:
        return np.asarray(fallback, dtype=np.uint8)
    return _rgb_float_to_uint8(np.median(np.asarray(pixels, dtype=np.float32), axis=0))


def _sample_iris_color(
    pixels: NDArray[np.float32],
    fallback: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    if pixels.size == 0:
        return np.asarray(fallback, dtype=np.uint8)
    array = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
    array = array[np.all(np.isfinite(array), axis=1)]
    if array.size == 0:
        return np.asarray(fallback, dtype=np.uint8)
    if float(np.nanmax(array)) > 1.5:
        array = array / 255.0
    luma = _rgb_luma(array)
    chroma = np.max(array, axis=1) - np.min(array, axis=1)
    warm = 0.5 * (array[:, 0] + array[:, 1]) - array[:, 2]
    bright_cutoff = float(np.quantile(luma, 0.35))
    chroma_cutoff = float(np.quantile(chroma, 0.45))
    warm_cutoff = float(np.quantile(warm, 0.45))
    selected = array[(luma >= bright_cutoff) & (chroma >= chroma_cutoff) & (warm >= warm_cutoff)]
    if selected.shape[0] < max(2, int(round(array.shape[0] * 0.12))):
        selected = array[luma >= float(np.quantile(luma, 0.50))]
    if selected.size == 0:
        selected = array
    return _rgb_float_to_uint8(np.quantile(selected, 0.60, axis=0))


def _sample_low_luma_color(
    pixels: NDArray[np.float32],
    fallback: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    if pixels.size == 0:
        return np.asarray(fallback, dtype=np.uint8)
    array = np.asarray(pixels, dtype=np.float32)
    return _rgb_float_to_uint8(np.quantile(array, 0.10, axis=0))


def _sample_dark_pupil_color(
    pixels: NDArray[np.float32],
    fallback: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    if pixels.size == 0:
        return np.asarray(fallback, dtype=np.uint8)
    array = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
    array = array[np.all(np.isfinite(array), axis=1)]
    if array.size == 0:
        return np.asarray(fallback, dtype=np.uint8)
    if float(np.nanmax(array)) > 1.5:
        array = array / 255.0
    luma = _rgb_luma(array)
    order = np.argsort(luma, kind="stable")
    count = max(1, int(np.ceil(float(order.size) * 0.08)))
    selected = array[order[:count]]
    return _rgb_float_to_uint8(np.quantile(selected, 0.10, axis=0))


def _sample_warm_color(
    pixels: NDArray[np.float32],
    fallback: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    if pixels.size == 0:
        return np.asarray(fallback, dtype=np.uint8)
    array = np.asarray(pixels, dtype=np.float32)
    red_excess = array[:, 0] - np.maximum(array[:, 1], array[:, 2])
    threshold = float(np.quantile(red_excess, 0.60))
    selected = array[red_excess >= threshold]
    if selected.size == 0:
        selected = array
    return _rgb_float_to_uint8(np.median(selected, axis=0))


def _raw_facial_feature_colors(
    observations: tuple[FitObservation, ...],
    images_by_frame: dict[str, NDArray[np.float32]],
    foreground_fallback_color: NDArray[np.float64],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    foreground_pixels: list[NDArray[np.float32]] = []
    for observation in observations:
        image = images_by_frame[observation.frame_id]
        mask = np.asarray(observation.hard_mask, dtype=np.float32) >= 0.5
        if image.shape[:2] != mask.shape:
            raise ValueError("observation image and mask dimensions must match")
        pixels = image[mask]
        if pixels.size:
            foreground_pixels.append(pixels.astype(np.float32))

    fallback_luma = float(_rgb_luma(foreground_fallback_color[None, :])[0])
    if not foreground_pixels:
        return (
            _rgb_float_to_uint8(np.asarray([0.06, 0.045, 0.035], dtype=np.float64)),
            _rgb_float_to_uint8(np.asarray([0.88, 0.52, 0.56], dtype=np.float64)),
        )

    pixels = np.concatenate(foreground_pixels, axis=0)
    luma = _rgb_luma(pixels)
    dark_cutoff = min(float(np.quantile(luma, 0.08)), max(0.24, fallback_luma - 0.22))
    dark_pixels = pixels[luma <= dark_cutoff]
    if fallback_luma < 0.35:
        eye = np.asarray([0.55, 0.42, 0.18], dtype=np.float64)
    elif dark_pixels.shape[0] >= max(8, pixels.shape[0] // 500):
        eye = 0.72 * np.median(dark_pixels, axis=0)
    else:
        eye = np.asarray([0.06, 0.045, 0.035], dtype=np.float64)

    chroma = np.max(pixels, axis=1) - np.min(pixels, axis=1)
    red_excess = pixels[:, 0] - np.maximum(pixels[:, 1], pixels[:, 2])
    warm = (
        (red_excess > 0.070)
        & (chroma > 0.100)
        & (pixels[:, 0] > 0.45)
        & (luma > 0.28)
        & (luma < 0.88)
    )
    warm_pixels = pixels[warm]
    if warm_pixels.shape[0] >= max(8, pixels.shape[0] // 1000):
        nose = np.median(warm_pixels, axis=0)
    else:
        nose = 0.75 * np.asarray([0.88, 0.52, 0.56], dtype=np.float64) + 0.25 * np.asarray(
            foreground_fallback_color,
            dtype=np.float64,
        )

    return _rgb_float_to_uint8(eye), _rgb_float_to_uint8(nose)


def _rgb_float_to_uint8(rgb: NDArray[np.floating]) -> NDArray[np.uint8]:
    return np.clip(np.rint(np.asarray(rgb, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def _face_feature_destination_uv(
    vertices: NDArray[np.float64],
    *,
    feature_name: str,
    spec: dict[str, object],
) -> NDArray[np.float64]:
    anchor = _FACE_FEATURE_ANATOMICAL_UV_ANCHORS.get(feature_name)
    if anchor is not None:
        return np.asarray(anchor, dtype=np.float64)
    x, _y, z = spec["xyz"]
    return _canonical_xz_uv(vertices, x=float(x), z=float(z))


def _face_feature_destination_anchor_source(feature_name: str) -> str:
    if feature_name in _FACE_FEATURE_ANATOMICAL_UV_ANCHORS:
        return _FACE_ANATOMICAL_UV_ANCHOR_VERSION
    return "canonical_xz_uv"


def _facial_feature_destination_anchor_metrics(
    features: list[dict[str, object]],
) -> dict[str, object]:
    by_name = {str(feature.get("name", "")): feature for feature in features}
    anatomical_count = sum(
        1
        for feature in features
        if feature.get("destination_anchor_source") == _FACE_ANATOMICAL_UV_ANCHOR_VERSION
    )

    def uv(name: str) -> tuple[float, float]:
        raw = by_name.get(name, {}).get("uv", [0.0, 0.0])
        if not isinstance(raw, list | tuple) or len(raw) != 2:
            return (0.0, 0.0)
        return (float(raw[0]), float(raw[1]))

    left_eye = uv("left_eye_core")
    right_eye = uv("right_eye_core")
    nose = uv("nose")
    mouth = uv("mouth_line")
    left_muzzle = uv("left_muzzle")
    right_muzzle = uv("right_muzzle")
    left_ear = uv("left_ear_inner")
    right_ear = uv("right_ear_inner")
    eye_v = float((left_eye[1] + right_eye[1]) * 0.5)
    eye_mid_u = float((left_eye[0] + right_eye[0]) * 0.5)
    eye_half_spacing_left = float(eye_mid_u - left_eye[0])
    eye_half_spacing_right = float(right_eye[0] - eye_mid_u)
    muzzle_v = float((left_muzzle[1] + right_muzzle[1]) * 0.5)
    ear_v = float((left_ear[1] + right_ear[1]) * 0.5)
    eye_to_nose_v_gap = float(eye_v - nose[1])
    nose_to_mouth_v_gap = float(nose[1] - mouth[1])
    nose_to_muzzle_v_gap = float(nose[1] - muzzle_v)
    ear_to_eye_v_gap = float(ear_v - eye_v)
    eye_symmetry_u_error = float(
        abs(eye_mid_u - nose[0]) + abs(eye_half_spacing_left - eye_half_spacing_right)
    )
    order_valid = (
        anatomical_count >= 12
        and eye_to_nose_v_gap >= 0.025
        and nose_to_mouth_v_gap >= 0.035
        and nose_to_muzzle_v_gap >= 0.010
        and ear_to_eye_v_gap >= 0.010
        and eye_symmetry_u_error <= 0.035
    )
    return {
        "destination_anchor_anatomical_count": int(anatomical_count),
        "destination_anchor_order_valid": bool(order_valid),
        "destination_anchor_eye_to_nose_v_gap": eye_to_nose_v_gap,
        "destination_anchor_nose_to_mouth_v_gap": nose_to_mouth_v_gap,
        "destination_anchor_nose_to_muzzle_v_gap": nose_to_muzzle_v_gap,
        "destination_anchor_ear_to_eye_v_gap": ear_to_eye_v_gap,
        "destination_anchor_eye_symmetry_u_error": eye_symmetry_u_error,
    }


def _canonical_xz_uv(
    vertices: NDArray[np.float64],
    *,
    x: float,
    z: float,
) -> NDArray[np.float64]:
    mins = vertices.min(axis=0)
    spans = np.maximum(vertices.max(axis=0) - mins, 1e-9)
    return np.asarray(
        [
            (x - mins[0]) / spans[0],
            (z - mins[2]) / spans[2],
        ],
        dtype=np.float64,
    )


def _ellipse_mask(
    *,
    uv: NDArray[np.float64],
    radius_uv: tuple[float, float],
    texture_size: int,
) -> NDArray[np.bool_]:
    center = _uv_to_pixel_xy(np.asarray([uv], dtype=np.float64), texture_size)[0]
    radius_x = max(float(radius_uv[0]) * float(texture_size - 1), 1.0)
    radius_y = max(float(radius_uv[1]) * float(texture_size - 1), 1.0)
    x0 = max(int(np.floor(center[0] - radius_x)), 0)
    x1 = min(int(np.ceil(center[0] + radius_x)), texture_size - 1)
    y0 = max(int(np.floor(center[1] - radius_y)), 0)
    y1 = min(int(np.ceil(center[1] + radius_y)), texture_size - 1)
    mask = np.zeros((texture_size, texture_size), dtype=np.bool_)
    if x1 < x0 or y1 < y0:
        return mask
    xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = ((grid_x - center[0]) / radius_x) ** 2 + ((grid_y - center[1]) / radius_y) ** 2 <= 1.0
    mask[y0 : y1 + 1, x0 : x1 + 1] = inside
    return mask


def _rasterize_face_colors_to_uv(
    problem: FitProblem,
    face_colors: NDArray[np.float64],
    observed_faces: NDArray[np.bool_],
    best_source_view: NDArray[np.integer],
    *,
    foreground_fallback_color: NDArray[np.floating],
    texture_size: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    fallback_rgb = np.clip(
        np.rint(np.asarray(foreground_fallback_color, dtype=np.float64) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    basecolor = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    basecolor[:, :] = fallback_rgb[None, None, :]
    confidence = np.zeros((texture_size, texture_size), dtype=np.uint8)
    source_view = np.zeros((texture_size, texture_size), dtype=np.uint8)
    uv_vertices = np.asarray(problem.template.uv_vertices, dtype=np.float64)
    uv_faces = np.asarray(problem.template.uv_faces, dtype=np.int64)
    if uv_faces.shape != problem.template.faces.shape:
        raise ValueError("uv_faces must match mesh face topology")

    for face_index in range(uv_faces.shape[0]):
        if not bool(observed_faces[face_index]):
            continue
        uv = uv_vertices[uv_faces[face_index]]
        pixels = _uv_to_pixel_xy(uv, texture_size)
        mask = _triangle_mask(pixels, texture_size)
        if not np.any(mask):
            centroid = np.rint(np.mean(pixels, axis=0)).astype(np.int64)
            x = int(np.clip(centroid[0], 0, texture_size - 1))
            y = int(np.clip(centroid[1], 0, texture_size - 1))
            mask[y, x] = True
        color = np.clip(np.rint(face_colors[face_index] * 255.0), 0, 255).astype(np.uint8)
        basecolor[mask] = color
        confidence[mask] = 255
        source_view[mask] = np.uint8(min(int(best_source_view[face_index]) + 1, 255))
    return basecolor, confidence, source_view


def _face_region_projection_protection_report(
    problem: FitProblem,
    *,
    frame_metadata: tuple[dict[str, Any], ...],
) -> dict[str, object]:
    protected_names: set[str] = set()
    applied_frame_count = 0
    for metadata in frame_metadata:
        if not _exclude_protected_face_projection(metadata):
            continue
        applied_frame_count += 1
        protected_names.update(_protected_face_projection_region_names(metadata))
    if not protected_names:
        protected_face_mask = np.zeros(problem.template.faces.shape[0], dtype=np.bool_)
        return {
            "applied": False,
            "version": _FACE_REGION_PROJECTION_PROTECTION_VERSION,
            "protected_projection_policy_frame_count": 0,
            "protected_region_names": [],
            "protected_region_count": 0,
            "protected_face_count": 0,
            "protected_face_fraction": 0.0,
            "protected_face_mask": protected_face_mask,
        }
    protected_face_mask = _protected_face_region_mask(
        problem,
        protected_region_names=tuple(sorted(protected_names)),
    )
    protected_face_count = int(np.count_nonzero(protected_face_mask))
    return {
        "applied": bool(applied_frame_count > 0 and protected_face_count > 0),
        "version": _FACE_REGION_PROJECTION_PROTECTION_VERSION,
        "protected_projection_policy_frame_count": int(applied_frame_count),
        "protected_region_names": sorted(protected_names),
        "protected_region_count": int(len(protected_names)),
        "protected_face_count": protected_face_count,
        "protected_face_fraction": float(
            protected_face_count / max(int(problem.template.faces.shape[0]), 1)
        ),
        "protected_face_mask": protected_face_mask,
    }


def _exclude_protected_face_projection(metadata: dict[str, Any]) -> bool:
    policy = metadata.get("texture_only_face_region_projection_protection")
    if isinstance(policy, dict):
        if bool(policy.get("broad_projection_allowed", False)):
            return False
        return bool(policy.get("applied", False) or policy.get("broad_projection_excluded", False))
    return bool(metadata.get("texture_only_face_region_projection_protected", False))


def _source_patch_candidate_allowed(metadata: dict[str, Any]) -> bool:
    if not bool(metadata.get("texture_only_supplemental", False)):
        return True
    policy = metadata.get("texture_only_face_region_projection_protection")
    if isinstance(policy, dict) and "source_patch_allowed" in policy:
        return bool(policy["source_patch_allowed"])
    if "texture_only_source_patch_allowed" in metadata:
        return bool(metadata["texture_only_source_patch_allowed"])
    return True


def _source_patch_allowed_sample_keys(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw_keys = metadata.get("texture_only_source_patch_allowed_sample_keys")
    if raw_keys is None:
        return ()
    if isinstance(raw_keys, str):
        raw_items: tuple[object, ...] = (raw_keys,)
    elif isinstance(raw_keys, list | tuple | set):
        raw_items = tuple(raw_keys)
    else:
        return ()
    allowed: list[str] = []
    for raw_key in raw_items:
        key = str(raw_key)
        if key not in _FACE_PRIORITY_SOURCE_PATCH_KEYS or key in allowed:
            continue
        allowed.append(key)
    return tuple(allowed)


def _source_patch_sample_key_allowed(
    candidate: _FaceFeaturePalette,
    sample_key: str,
) -> bool:
    allowed = candidate.allowed_source_patch_sample_keys
    if not allowed:
        return True
    if sample_key in allowed:
        return True
    if sample_key.startswith("left_") or sample_key.startswith("right_"):
        paired_key = sample_key.split("_", 1)[1]
        return paired_key in allowed
    return False


def _protected_face_projection_region_names(
    metadata: dict[str, Any],
) -> tuple[str, ...]:
    policy = metadata.get("texture_only_face_region_projection_protection")
    raw_names: object = None
    if isinstance(policy, dict):
        raw_names = policy.get("protected_region_names")
    if raw_names is None:
        raw_names = metadata.get("texture_only_protected_face_region_names")
    if not isinstance(raw_names, list | tuple):
        return _DEFAULT_PROTECTED_FACE_REGION_NAMES
    names = tuple(str(name) for name in raw_names if str(name))
    return names or _DEFAULT_PROTECTED_FACE_REGION_NAMES


def _protected_face_region_mask(
    problem: FitProblem,
    *,
    protected_region_names: tuple[str, ...],
) -> NDArray[np.bool_]:
    protected_names = {str(name) for name in protected_region_names if str(name)}
    if not protected_names:
        return np.zeros(problem.template.faces.shape[0], dtype=np.bool_)
    uv_vertices = np.asarray(problem.template.uv_vertices, dtype=np.float64)
    uv_faces = np.asarray(problem.template.uv_faces, dtype=np.int64)
    if uv_faces.shape != problem.template.faces.shape:
        raise ValueError("uv_faces must match mesh face topology")
    centroids = np.mean(uv_vertices[uv_faces], axis=1)
    protected = np.zeros(uv_faces.shape[0], dtype=np.bool_)
    region_manifest = problem.template.face_uv_region_manifest()
    regions = region_manifest.get("regions", [])
    if not isinstance(regions, list):
        return protected
    for region in regions:
        if not isinstance(region, dict):
            continue
        if str(region.get("name", "")) not in protected_names:
            continue
        bounds = region.get("uv_bounds", [])
        if not isinstance(bounds, list | tuple) or len(bounds) != 4:
            continue
        u0, v0, u1, v1 = (float(value) for value in bounds)
        eps = 1e-9
        protected |= (
            (centroids[:, 0] >= u0 - eps)
            & (centroids[:, 0] <= u1 + eps)
            & (centroids[:, 1] >= v0 - eps)
            & (centroids[:, 1] <= v1 + eps)
        )
    return protected


def _protected_projection_pixel_mask(
    face_index_buffer: NDArray[np.integer],
    *,
    valid_mask: NDArray[np.bool_],
    protected_face_mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    protected_pixels = np.zeros(face_index_buffer.shape, dtype=np.bool_)
    if not np.any(valid_mask) or not np.any(protected_face_mask):
        return protected_pixels
    face_indices = np.asarray(face_index_buffer[valid_mask], dtype=np.int64)
    protected_values = protected_face_mask[face_indices]
    protected_pixels[valid_mask] = protected_values
    return protected_pixels


def _uv_to_pixel_xy(uv: NDArray[np.float64], texture_size: int) -> NDArray[np.float64]:
    clipped = np.clip(uv, 0.0, 1.0)
    x = clipped[:, 0] * float(texture_size - 1)
    y = (1.0 - clipped[:, 1]) * float(texture_size - 1)
    return np.stack([x, y], axis=-1)


def _triangle_mask(
    tri_xy: NDArray[np.float64],
    texture_size: int,
) -> NDArray[np.bool_]:
    mask = np.zeros((texture_size, texture_size), dtype=np.bool_)
    min_xy = np.floor(tri_xy.min(axis=0)).astype(np.int64)
    max_xy = np.ceil(tri_xy.max(axis=0)).astype(np.int64)
    x0 = max(int(min_xy[0]), 0)
    y0 = max(int(min_xy[1]), 0)
    x1 = min(int(max_xy[0]), texture_size - 1)
    y1 = min(int(max_xy[1]), texture_size - 1)
    if x1 < x0 or y1 < y0:
        return mask

    a, b, c = tri_xy
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(denom)) < 1e-12:
        return mask
    xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    w0 = ((b[1] - c[1]) * (grid_x - c[0]) + (c[0] - b[0]) * (grid_y - c[1])) / denom
    w1 = ((c[1] - a[1]) * (grid_x - c[0]) + (a[0] - c[0]) * (grid_y - c[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    mask[y0 : y1 + 1, x0 : x1 + 1] = inside
    return mask


def _update_texture_manifest(
    manifest_path: Path,
    *,
    mode: str,
    basecolor_path: Path,
    confidence_path: Path,
    source_view_id_path: Path,
    texture_size: int,
    observed_face_count: int,
    unknown_face_count: int,
    observed_texel_fraction: float,
    foreground_fallback_color: NDArray[np.floating],
    repaired_face_count: int,
    facial_features: dict[str, object],
    face_uv_regions: dict[str, object],
    texture_target_report: dict[str, object] | None,
    projection_inputs: _TextureProjectionInputs,
    basecolor_seed: dict[str, object],
    face_region_projection_protection: dict[str, object],
) -> None:
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            manifest = raw
    manifest.setdefault("schema_version", "1.0")
    manifest["basecolor_source"] = "opencat3d.observed_basecolor_projection"
    manifest["mode"] = mode
    artifacts = manifest.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        manifest["artifacts"] = artifacts
    artifacts.update(
        {
            "basecolor_png": _manifest_artifact_path(basecolor_path, manifest_path),
            "basecolor_confidence_png": _manifest_artifact_path(
                confidence_path,
                manifest_path,
            ),
            "source_view_id_png": _manifest_artifact_path(
                source_view_id_path,
                manifest_path,
            ),
        }
    )
    sha = manifest.setdefault("sha256", {})
    if not isinstance(sha, dict):
        sha = {}
        manifest["sha256"] = sha
    sha.update(
        {
            "basecolor_png": sha256_file(basecolor_path),
            "basecolor_confidence_png": sha256_file(confidence_path),
            "source_view_id_png": sha256_file(source_view_id_path),
        }
    )
    unknown_rgb = [
        int(channel)
        for channel in np.clip(
            np.rint(np.asarray(foreground_fallback_color, dtype=np.float64) * 255.0),
            0,
            255,
        ).astype(np.uint8)
    ]
    manifest["basecolor"] = {
        "width": texture_size,
        "height": texture_size,
        "texture_resolution_source": _texture_resolution_source(
            mode,
            texture_size=texture_size,
            texture_target_report=texture_target_report,
        ),
        "unknown_rgb": unknown_rgb,
        "unknown_rgb_source": "observed_foreground_high_luma_median",
        "foreground_fallback_rgb": unknown_rgb,
        "dark_observed_face_repair_count": int(repaired_face_count),
        "projection_manifest_path": str(projection_inputs.source_manifest_path),
        "projection_frame_count": int(len(projection_inputs.observations)),
        "fit_frame_count": int(projection_inputs.fit_frame_count),
        "supplemental_texture_frame_count": int(projection_inputs.supplemental_frame_count),
        "confidence_unknown": 0,
        "confidence_observed": 255,
        "source_view_id_unknown": 0,
        "observed_face_count": observed_face_count,
        "unknown_face_count": unknown_face_count,
        "observed_texel_fraction": observed_texel_fraction,
        "generated_texel_fraction": 0.0,
        "baked_basecolor_seed": basecolor_seed,
        "face_region_projection_protection": face_region_projection_protection,
    }
    if texture_target_report is not None:
        manifest["basecolor"]["face_texture_target_source"] = str(
            texture_target_report["target_source"]
        )
        manifest["basecolor"]["face_texture_target_report_path"] = str(
            texture_target_report["path"]
        )
        manifest["basecolor"]["face_texture_target_size"] = int(
            texture_target_report["target_texture_size"]
        )
        manifest["basecolor"]["face_texture_target_scale"] = float(
            texture_target_report["target_texture_scale"]
        )
        manifest["basecolor"]["face_texture_target_requested_texel_count"] = int(
            texture_target_report["target_requested_texel_count"]
        )
        manifest["basecolor"]["face_texture_target_allocated_texel_count"] = int(
            texture_target_report["target_allocated_texel_count"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_bake_used"] = float(
            texture_target_report["template_region_v2_texture_target_used"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_size"] = int(
            texture_target_report["template_region_v2_texture_target_size"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_scale"] = float(
            texture_target_report["template_region_v2_texture_target_scale"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_island_count"] = int(
            texture_target_report["template_region_v2_texture_target_island_count"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_facial_island_count"] = int(
            texture_target_report["template_region_v2_texture_target_facial_island_count"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_requested_texel_count"] = int(
            texture_target_report["template_region_v2_texture_target_requested_texel_count"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_allocated_texel_count"] = int(
            texture_target_report["template_region_v2_texture_target_allocated_texel_count"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_eye_min_texels"] = int(
            texture_target_report["template_region_v2_texture_target_eye_min_texels"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_nose_texels"] = int(
            texture_target_report["template_region_v2_texture_target_nose_texels"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_muzzle_texels"] = int(
            texture_target_report["template_region_v2_texture_target_muzzle_texels"]
        )
        manifest["basecolor"]["template_region_v2_texture_target_ear_inner_min_texels"] = int(
            texture_target_report["template_region_v2_texture_target_ear_inner_min_texels"]
        )
    manifest["facial_features"] = facial_features
    manifest["template_face_uv_regions"] = face_uv_regions
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest_artifact_path(path: Path, manifest_path: Path) -> str:
    project_root = manifest_path.parent.parent.parent
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _texture_resolution_source(
    mode: str,
    *,
    texture_size: int,
    texture_target_report: dict[str, object] | None,
) -> str:
    if mode == "raw" and texture_target_report is not None:
        target_size = int(texture_target_report["target_texture_size"])
        if (
            float(texture_target_report["template_region_v2_texture_target_used"]) == 1.0
            and texture_size >= target_size >= RAW_TEMPLATE_REGION_V2_TEXTURE_TARGET_SIZE
        ):
            return "raw_template_region_v2_texture_target_2048"
        if texture_size >= target_size >= RAW_FACE_TARGET_TEXTURE_SIZE:
            return "raw_face_texture_target_1024"
    if mode == "raw" and texture_size >= RAW_FACE_PRIORITY_TEXTURE_SIZE:
        return "raw_face_priority_512"
    return "default_256"


def _read_existing_basecolor_result(
    *,
    texture_dir: Path,
    manifest_path: Path,
    basecolor_path: Path,
    confidence_path: Path,
    source_view_id_path: Path,
) -> BasecolorTextureResult:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("texture_manifest.json must contain an object")
    basecolor = manifest.get("basecolor", {})
    if not isinstance(basecolor, dict):
        raise ValueError("texture_manifest.json is missing basecolor metadata")
    return BasecolorTextureResult(
        texture_dir=texture_dir,
        manifest_path=manifest_path,
        basecolor_path=basecolor_path,
        confidence_path=confidence_path,
        source_view_id_path=source_view_id_path,
        observed_face_count=int(basecolor.get("observed_face_count", 0)),
        unknown_face_count=int(basecolor.get("unknown_face_count", 0)),
        observed_texel_fraction=float(basecolor.get("observed_texel_fraction", 0.0)),
        width=int(basecolor.get("width", 0)),
        height=int(basecolor.get("height", 0)),
    )
