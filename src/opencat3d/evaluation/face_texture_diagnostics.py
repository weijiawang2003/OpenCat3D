"""Observed face-texture crop diagnostics for raw reconstruction quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from opencat3d.io.cache import sha256_file

REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS = (
    "left_eye",
    "right_eye",
    "nose",
    "muzzle",
    "left_ear_inner",
    "right_ear_inner",
)
DIRECT_PHOTO_FACE_CANDIDATE_SAMPLE_KEYS = (
    "left_eye",
    "right_eye",
    "left_iris",
    "right_iris",
    "left_pupil",
    "right_pupil",
    "nose",
    "muzzle",
    "ear_inner",
)
_DIRECT_PHOTO_BROAD_FACE_DETAIL_THRESHOLD = 0.45
_DIRECT_PHOTO_BROAD_FACE_RELAXED_DETAIL_THRESHOLD = 0.30
_DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_SUPPORT = 0.90
_DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_DENSITY = 0.75
_DIRECT_PHOTO_BROAD_FACE_MIN_QUALITY = 0.45
_DIRECT_PHOTO_PUPIL_MIN_QUALITY = 0.10
_DIRECT_PHOTO_PUPIL_MIN_SUPPORT = 0.45
_DIRECT_PHOTO_PUPIL_MIN_DENSITY = 0.10

_SOURCE_REGION_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "left_eye": (0.30, 0.12, 0.50, 0.36),
    "right_eye": (0.50, 0.12, 0.70, 0.36),
    "nose": (0.42, 0.32, 0.58, 0.50),
    "muzzle": (0.30, 0.34, 0.70, 0.64),
    "left_ear_inner": (0.16, 0.00, 0.42, 0.30),
    "right_ear_inner": (0.58, 0.00, 0.84, 0.30),
}

_REGION_SOURCE_PATCH_FEATURES: dict[str, tuple[str, ...]] = {
    "left_eye": ("left_eye_core", "left_eye_iris", "left_eye_pupil", "left_eye_shadow"),
    "right_eye": ("right_eye_core", "right_eye_iris", "right_eye_pupil", "right_eye_shadow"),
    "nose": ("nose",),
    "muzzle": ("left_muzzle", "right_muzzle", "mouth_line"),
    "left_ear_inner": ("left_ear_inner",),
    "right_ear_inner": ("right_ear_inner",),
}

_DETAIL_UNIQUE_RGB_THRESHOLD = 2
_DETAIL_LUMA_STD_THRESHOLD = 0.005
_VISUAL_QUALITY_MIN_SCORE = 0.70
_VISUAL_QUALITY_MIN_CONFIDENT_TEXEL_FRACTION = 0.05
_VISUAL_QUALITY_MIN_UNIQUE_RGB_COUNT = 8
_VISUAL_QUALITY_MIN_TEXTURE_LUMA_STD = 0.010
_VISUAL_QUALITY_MIN_SOURCE_LUMA_STD = 0.006
_VISUAL_QUALITY_TARGET_CONFIDENT_TEXEL_FRACTION = 0.12
_VISUAL_QUALITY_TARGET_UNIQUE_RGB_COUNT = 64
_VISUAL_QUALITY_TARGET_TEXTURE_LUMA_STD = 0.04
_VISUAL_QUALITY_TARGET_SOURCE_LUMA_STD = 0.02
_ISOLATED_TEXTURE_CROP_BOUNDS_SOURCE = "isolated_feature_footprint_uv_bounds_v1"
_TEXTURE_REGION_ISOLATION_LIMITS: dict[str, tuple[float, float, float, float]] = {
    "left_eye": (0.335, 0.900, 0.415, 0.955),
    "right_eye": (0.585, 0.900, 0.665, 0.955),
    "nose": (0.455, 0.866, 0.545, 0.918),
    "muzzle": (0.365, 0.812, 0.635, 0.858),
    "left_ear_inner": (0.240, 0.936, 0.332, 0.998),
    "right_ear_inner": (0.668, 0.936, 0.760, 0.998),
}


@dataclass(frozen=True)
class FaceTextureDiagnosticResult:
    report_path: Path
    contact_sheet_path: Path
    crop_dir: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class DirectPhotoFaceCandidateDiagnosticResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class _ExactSourcePatch:
    image: NDArray[np.uint8]
    source_frame_id: str
    feature_names: list[str]
    patch_paths: list[str]


def evaluate_direct_photo_face_candidate_report(
    project_dir: Path,
    *,
    mode: str = "raw",
    force: bool = False,
) -> DirectPhotoFaceCandidateDiagnosticResult:
    """Explain direct-photo face source-patch eligibility without mutating outputs."""

    if mode != "raw":
        raise ValueError(f"unsupported direct-photo face candidate mode: {mode}")

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / f"{mode}_direct_photo_face_candidates.json"
    texture_manifest_path = project_dir / "outputs" / "textures" / "texture_manifest.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    manifest = _read_json(texture_manifest_path)
    facial_features = _object(
        manifest.get("facial_features"),
        "texture_manifest.facial_features",
    )
    frames = _texture_projection_frames(project_dir, manifest)
    frame_by_id = {str(frame.get("frame_id", "")): frame for frame in frames}
    candidate_by_key = _source_patch_candidate_diagnostics(facial_features)
    selected_by_key = _selected_source_patch_features_by_sample_key(facial_features)

    sample_reports: list[dict[str, object]] = []
    all_candidate_reports: list[dict[str, object]] = []
    for sample_key in DIRECT_PHOTO_FACE_CANDIDATE_SAMPLE_KEYS:
        selected_feature = selected_by_key.get(sample_key, {})
        selected_frame_id = str(selected_feature.get("source_frame_id", ""))
        candidate_metrics_missing = False
        raw_candidates = candidate_by_key.get(sample_key, [])
        if not raw_candidates and selected_feature:
            candidate_metrics_missing = True
            raw_candidates = [
                _candidate_from_selected_feature(sample_key, selected_feature)
            ]
        candidate_reports = [
            _direct_photo_candidate_report(
                sample_key,
                candidate,
                frame_by_id.get(str(candidate.get("source_frame_id", "")), {}),
                selected_frame_id=selected_frame_id,
                candidate_metrics_missing=candidate_metrics_missing,
            )
            for candidate in raw_candidates
            if isinstance(candidate, dict)
        ]
        candidate_reports.sort(key=_direct_photo_candidate_sort_key)
        all_candidate_reports.extend(candidate_reports)
        direct_candidates = [
            candidate
            for candidate in candidate_reports
            if bool(candidate["direct_uploaded_photo"])
        ]
        eligible_direct_candidates = [
            candidate
            for candidate in direct_candidates
            if bool(candidate["direct_photo_eligible"])
        ]
        selected_direct = next(
            (
                candidate
                for candidate in candidate_reports
                if bool(candidate["selected"])
                and bool(candidate["direct_uploaded_photo"])
            ),
            None,
        )
        best_direct = direct_candidates[0] if direct_candidates else {}
        sample_reports.append(
            {
                "sample_key": sample_key,
                "selected_frame_id": selected_frame_id,
                "selected_source_media_id": str(
                    (frame_by_id.get(selected_frame_id) or {}).get(
                        "source_media_id",
                        "",
                    )
                ),
                "selected_direct_photo": bool(selected_direct),
                "selected_frontal_photo_priority": round(
                    float(
                        selected_direct.get("source_patch_frontal_photo_priority", 0.0)
                        if selected_direct
                        else 0.0
                    ),
                    6,
                ),
                "candidate_metrics_missing": bool(candidate_metrics_missing),
                "candidate_count": int(len(candidate_reports)),
                "direct_photo_candidate_count": int(len(direct_candidates)),
                "eligible_direct_photo_count": int(len(eligible_direct_candidates)),
                "best_direct_photo_frame_id": str(best_direct.get("source_frame_id", "")),
                "best_direct_photo_source_media_id": str(
                    best_direct.get("source_media_id", "")
                ),
                "best_direct_photo_priority": round(
                    float(best_direct.get("source_patch_frontal_photo_priority", 0.0)),
                    6,
                ),
                "best_direct_photo_rejection_reasons": list(
                    best_direct.get("direct_photo_rejection_reasons", [])
                ),
                "candidates": candidate_reports,
            }
        )

    direct_candidate_count = sum(
        1
        for candidate in all_candidate_reports
        if bool(candidate["direct_uploaded_photo"])
    )
    eligible_direct_count = sum(
        1
        for candidate in all_candidate_reports
        if bool(candidate["direct_photo_eligible"])
    )
    selected_direct_count = sum(
        1
        for sample in sample_reports
        if bool(sample["selected_direct_photo"])
    )
    max_priority = max(
        (
            float(candidate["source_patch_frontal_photo_priority"])
            for candidate in all_candidate_reports
        ),
        default=0.0,
    )
    checks = {
        "texture_manifest_exists": texture_manifest_path.exists(),
        "facial_features_present": bool(facial_features),
        "required_sample_keys_reported": len(sample_reports)
        == len(DIRECT_PHOTO_FACE_CANDIDATE_SAMPLE_KEYS),
        "candidate_decisions_present": all(
            "direct_photo_rejection_reasons" in candidate
            for candidate in all_candidate_reports
        ),
        "metadata_only": True,
    }
    metrics: dict[str, float | int] = {
        "direct_photo_face_candidate_sample_key_count": int(len(sample_reports)),
        "direct_photo_face_candidate_total_count": int(len(all_candidate_reports)),
        "direct_photo_face_candidate_direct_count": int(direct_candidate_count),
        "direct_photo_face_candidate_eligible_count": int(eligible_direct_count),
        "direct_photo_face_candidate_selected_count": int(selected_direct_count),
        "direct_photo_face_candidate_rejected_count": int(
            max(direct_candidate_count - eligible_direct_count, 0)
        ),
        "direct_photo_face_candidate_missing_metric_feature_count": int(
            sum(1 for sample in sample_reports if bool(sample["candidate_metrics_missing"]))
        ),
        "direct_photo_face_candidate_max_priority": float(max_priority),
    }
    success = all(checks.values())
    report = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "source": "opencat3d.direct_photo_face_candidate_diagnostics_v1",
        "success": success,
        "checks": checks,
        "metrics": metrics,
        "required_sample_keys": list(DIRECT_PHOTO_FACE_CANDIDATE_SAMPLE_KEYS),
        "samples": sample_reports,
        "output_contract": {
            "topology_changed": False,
            "texture_changed": False,
            "target_representation": "metadata_only_quality_artifact",
        },
        "artifacts": {
            "texture_manifest": str(texture_manifest_path),
            "projection_manifest": str(
                _object(manifest.get("basecolor"), "texture_manifest.basecolor").get(
                    "projection_manifest_path",
                    "",
                )
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DirectPhotoFaceCandidateDiagnosticResult(
        report_path=report_path,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def evaluate_face_texture_diagnostics(
    project_dir: Path,
    *,
    mode: str = "raw",
    force: bool = False,
) -> FaceTextureDiagnosticResult:
    """Write deterministic texture/source crop diagnostics for face regions.

    The diagnostic is intentionally evidence-oriented: it reports texture color detail,
    texture confidence/provenance, and a source-frame crop side by side. It does not
    infer missing facial structure or mutate the asset.
    """

    if mode != "raw":
        raise ValueError(f"unsupported face texture diagnostic mode: {mode}")

    output_dir = project_dir / "outputs" / "quality"
    crop_dir = output_dir / f"{mode}_face_texture_crops"
    report_path = output_dir / f"{mode}_face_texture_diagnostics.json"
    contact_sheet_path = output_dir / f"{mode}_face_texture_crops.png"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")
    if contact_sheet_path.exists() and not force:
        raise FileExistsError(
            f"{contact_sheet_path} already exists; pass force=True to overwrite"
        )

    texture_dir = project_dir / "outputs" / "textures"
    texture_manifest_path = texture_dir / "texture_manifest.json"
    basecolor_path = texture_dir / "basecolor.png"
    confidence_path = texture_dir / "basecolor_confidence.png"
    source_view_id_path = texture_dir / "source_view_id.png"

    manifest = _read_json(texture_manifest_path)
    basecolor = np.asarray(Image.open(basecolor_path).convert("RGB"), dtype=np.uint8)
    confidence = np.asarray(Image.open(confidence_path).convert("L"), dtype=np.uint8)
    source_view_id = np.asarray(Image.open(source_view_id_path).convert("L"), dtype=np.uint8)
    height, width = basecolor.shape[:2]
    if confidence.shape != (height, width):
        raise ValueError("basecolor_confidence.png must match basecolor dimensions")
    if source_view_id.shape != (height, width):
        raise ValueError("source_view_id.png must match basecolor dimensions")

    regions = _face_regions(manifest)
    feature_by_name = _facial_feature_entries(manifest)
    frames = _texture_projection_frames(project_dir, manifest)
    frame_by_id = {str(frame.get("frame_id", "")): frame for frame in frames}
    preferred_frame_id = _preferred_source_frame_id(manifest, frame_by_id)
    crop_dir.mkdir(parents=True, exist_ok=True)

    region_reports: list[dict[str, object]] = []
    contact_rows: list[tuple[str, Image.Image, Image.Image | None]] = []
    for name in REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS:
        region = regions.get(name)
        if region is None:
            continue
        uv_bounds, crop_bounds_source = _texture_region_uv_bounds(
            name,
            region,
            feature_by_name,
        )
        pixel_bounds = _uv_pixel_bounds(
            uv_bounds,
            width=width,
            height=height,
        )
        x0, y0, x1, y1 = pixel_bounds
        texture_crop = basecolor[y0 : y1 + 1, x0 : x1 + 1]
        confidence_crop = confidence[y0 : y1 + 1, x0 : x1 + 1]
        source_id_crop = source_view_id[y0 : y1 + 1, x0 : x1 + 1]
        texture_crop_path = crop_dir / f"{name}_texture.png"
        Image.fromarray(texture_crop).save(texture_crop_path)

        dominant_frame_id, dominant_fraction = _dominant_source_frame(
            source_id_crop,
            frames,
        )
        source_frame_id = preferred_frame_id or dominant_frame_id
        source_crop_path = crop_dir / f"{name}_source.png"
        source_crop = None
        source_metrics: dict[str, object] = {
            "source_frame_id": "",
            "source_crop_path": "",
            "source_crop_written": False,
            "source_mean_rgb": [0, 0, 0],
            "source_luma_std": 0.0,
            "texture_source_mean_rgb_distance": 0.0,
            "source_crop_exact_patch_used": False,
            "source_patch_feature_names": [],
            "source_patch_paths": [],
        }
        exact_source_patch = _exact_source_patch_for_region(name, feature_by_name)
        if exact_source_patch is not None:
            source_crop = exact_source_patch.image
            Image.fromarray(source_crop).save(source_crop_path)
            source_metrics = _source_comparison_metrics(
                texture_crop,
                source_crop,
                source_frame_id=exact_source_patch.source_frame_id or source_frame_id,
                source_crop_path=source_crop_path,
            )
            source_metrics.update(
                {
                    "source_crop_exact_patch_used": True,
                    "source_patch_feature_names": exact_source_patch.feature_names,
                    "source_patch_paths": exact_source_patch.patch_paths,
                }
            )
        elif source_frame_id and source_frame_id in frame_by_id:
            source_crop = _source_crop_for_region(frame_by_id[source_frame_id], name)
            if source_crop is not None:
                Image.fromarray(source_crop).save(source_crop_path)
                source_metrics = _source_comparison_metrics(
                    texture_crop,
                    source_crop,
                    source_frame_id=source_frame_id,
                    source_crop_path=source_crop_path,
                )

        texture_metrics = _texture_crop_metrics(texture_crop, confidence_crop)
        visual_quality_metrics = _visual_quality_metrics(texture_metrics, source_metrics)
        region_reports.append(
            {
                "name": name,
                "uv_bounds": [round(float(value), 9) for value in region["uv_bounds"]],
                "crop_uv_bounds": [round(float(value), 9) for value in uv_bounds],
                "crop_bounds_source": crop_bounds_source,
                "texture_crop_isolated": bool(
                    crop_bounds_source == _ISOLATED_TEXTURE_CROP_BOUNDS_SOURCE
                ),
                "pixel_bounds": pixel_bounds,
                "texture_crop_path": str(texture_crop_path),
                "texture_width": int(texture_crop.shape[1]),
                "texture_height": int(texture_crop.shape[0]),
                "dominant_source_frame_id": dominant_frame_id,
                "dominant_source_texel_fraction": round(float(dominant_fraction), 6),
                **texture_metrics,
                **source_metrics,
                **visual_quality_metrics,
            }
        )
        contact_rows.append(
            (
                name,
                Image.fromarray(texture_crop),
                Image.fromarray(source_crop) if source_crop is not None else None,
            )
        )

    compared_regions = sum(
        1 for region in region_reports if bool(region.get("source_crop_written", False))
    )
    exact_patch_regions = sum(
        1
        for region in region_reports
        if bool(region.get("source_crop_exact_patch_used", False))
    )
    detail_regions = sum(
        1 for region in region_reports if bool(region.get("texture_detail_present", False))
    )
    unique_counts = [
        int(region["texture_unique_rgb_count"])
        for region in region_reports
        if "texture_unique_rgb_count" in region
    ]
    confidence_fractions = [
        float(region["texture_confident_texel_fraction"])
        for region in region_reports
        if "texture_confident_texel_fraction" in region
    ]
    source_distances = [
        float(region["texture_source_mean_rgb_distance"])
        for region in region_reports
        if bool(region.get("source_crop_written", False))
    ]
    isolated_crop_regions = sum(
        1 for region in region_reports if bool(region.get("texture_crop_isolated", False))
    )
    overlap_pairs, overlap_fraction = _crop_overlap_metrics(region_reports)
    visual_quality_regions = sum(
        1
        for region in region_reports
        if "texture_visual_quality_score" in region
    )
    visual_quality_passed_regions = sum(
        1
        for region in region_reports
        if bool(region.get("texture_visual_quality_passed", False))
    )
    visual_quality_scores = [
        float(region["texture_visual_quality_score"])
        for region in region_reports
        if "texture_visual_quality_score" in region
    ]

    _write_contact_sheet(contact_sheet_path, contact_rows)
    checks = {
        "texture_manifest_exists": texture_manifest_path.exists(),
        "basecolor_exists": basecolor_path.exists(),
        "confidence_exists": confidence_path.exists(),
        "source_view_id_exists": source_view_id_path.exists(),
        "texture_size_valid": width == height and width > 0,
        "required_regions_present": len(region_reports)
        == len(REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS),
        "all_texture_crops_written": all(
            Path(str(region["texture_crop_path"])).exists() for region in region_reports
        ),
        "source_reference_frame_available": bool(
            preferred_frame_id or frames or exact_patch_regions > 0
        ),
        "all_source_crops_written": compared_regions == len(region_reports),
        "all_source_crops_exact_patches": exact_patch_regions == len(region_reports),
        "all_regions_have_texture_detail": detail_regions == len(region_reports),
        "all_texture_crops_visibly_useful": visual_quality_passed_regions
        == len(region_reports),
        "all_texture_crops_isolated": isolated_crop_regions == len(region_reports),
        "texture_crops_non_overlapping": overlap_pairs == 0,
        "contact_sheet_written": contact_sheet_path.exists(),
        "no_generated_texture_claimed": float(
            _object(manifest.get("basecolor"), "texture_manifest.basecolor").get(
                "generated_texel_fraction",
                1.0,
            )
        )
        == 0.0,
    }
    metrics: dict[str, float | int] = {
        "face_texture_crop_region_count": int(len(region_reports)),
        "face_texture_crop_texture_width": int(width),
        "face_texture_crop_texture_height": int(height),
        "face_texture_crop_compared_region_count": int(compared_regions),
        "face_texture_crop_exact_source_patch_region_count": int(exact_patch_regions),
        "face_texture_crop_exact_source_patch_fraction": float(
            exact_patch_regions / max(len(region_reports), 1)
        ),
        "face_texture_crop_detail_region_count": int(detail_regions),
        "face_texture_crop_min_unique_rgb_count": int(min(unique_counts, default=0)),
        "face_texture_crop_min_confident_texel_fraction": float(
            min(confidence_fractions, default=0.0)
        ),
        "face_texture_crop_mean_confident_texel_fraction": float(
            np.mean(confidence_fractions) if confidence_fractions else 0.0
        ),
        "face_texture_crop_mean_source_rgb_delta": float(
            np.mean(source_distances) if source_distances else 0.0
        ),
        "face_texture_crop_isolated_region_count": int(isolated_crop_regions),
        "face_texture_crop_overlap_pair_count": int(overlap_pairs),
        "face_texture_crop_overlap_fraction": float(overlap_fraction),
        "face_texture_crop_visual_quality_region_count": int(visual_quality_regions),
        "face_texture_crop_visual_quality_passed_count": int(
            visual_quality_passed_regions
        ),
        "face_texture_crop_visual_quality_fraction": float(
            visual_quality_passed_regions / max(len(region_reports), 1)
        ),
        "face_texture_crop_visual_quality_min_score": float(
            min(visual_quality_scores, default=0.0)
        ),
        "face_texture_crop_visual_quality_mean_score": float(
            np.mean(visual_quality_scores) if visual_quality_scores else 0.0
        ),
        "face_texture_crop_contact_sheet_width": int(
            Image.open(contact_sheet_path).size[0] if contact_sheet_path.exists() else 0
        ),
        "face_texture_crop_contact_sheet_height": int(
            Image.open(contact_sheet_path).size[1] if contact_sheet_path.exists() else 0
        ),
    }
    success = all(checks.values())
    report = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "source": "opencat3d.face_texture_crop_diagnostics_v1",
        "success": success,
        "checks": checks,
        "metrics": metrics,
        "required_regions": list(REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS),
        "regions": region_reports,
        "artifacts": {
            "texture_manifest": str(texture_manifest_path),
            "basecolor": str(basecolor_path),
            "confidence": str(confidence_path),
            "source_view_id": str(source_view_id_path),
            "crop_dir": str(crop_dir),
            "contact_sheet_png": str(contact_sheet_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return FaceTextureDiagnosticResult(
        report_path=report_path,
        contact_sheet_path=contact_sheet_path,
        crop_dir=crop_dir,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def _face_regions(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    face_uv_regions = _object(
        manifest.get("template_face_uv_regions"),
        "texture_manifest.template_face_uv_regions",
    )
    regions = face_uv_regions.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("template_face_uv_regions.regions must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, region_raw in enumerate(regions):
        region = _object(region_raw, f"template_face_uv_regions.regions[{index}]")
        name = str(region.get("name", ""))
        if name in REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS:
            bounds = region.get("uv_bounds", [])
            if not isinstance(bounds, list) or len(bounds) != 4:
                raise ValueError(f"face UV region {name} must have uv_bounds")
            result[name] = region
    return result


def _texture_projection_frames(
    project_dir: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    basecolor = _object(manifest.get("basecolor"), "texture_manifest.basecolor")
    manifest_path = Path(
        str(
            basecolor.get(
                "projection_manifest_path",
                project_dir / "observations" / "raw" / "texture_manifest.json",
            )
        )
    )
    if not manifest_path.exists():
        manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    payload = _read_json(manifest_path)
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        return []
    return [dict(frame) for frame in frames if isinstance(frame, dict)]


def _preferred_source_frame_id(
    manifest: dict[str, Any],
    frame_by_id: dict[str, dict[str, Any]],
) -> str:
    facial_features = manifest.get("facial_features", {})
    if isinstance(facial_features, dict):
        frame_id = str(facial_features.get("source_frame_id", ""))
        if frame_id in frame_by_id:
            return frame_id
    return next(iter(frame_by_id), "")


def _facial_feature_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    facial_features = manifest.get("facial_features", {})
    if not isinstance(facial_features, dict):
        return {}
    features = facial_features.get("features", [])
    if not isinstance(features, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for raw in features:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", ""))
        if name:
            result[name] = dict(raw)
    return result


def _source_patch_candidate_diagnostics(
    facial_features: dict[str, Any],
) -> dict[str, list[dict[str, object]]]:
    raw = facial_features.get("source_patch_candidate_diagnostics", {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[dict[str, object]]] = {}
    for key, candidates in raw.items():
        if not isinstance(candidates, list):
            continue
        result[str(key)] = [
            dict(candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
    return result


def _selected_source_patch_features_by_sample_key(
    facial_features: dict[str, Any],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    features = facial_features.get("features", [])
    if not isinstance(features, list):
        return result
    for feature in features:
        if not isinstance(feature, dict):
            continue
        if not bool(feature.get("source_patch_used", False)):
            continue
        sample_key = str(feature.get("sample_key", ""))
        if sample_key and sample_key not in result:
            result[sample_key] = dict(feature)
    return result


def _candidate_from_selected_feature(
    sample_key: str,
    feature: dict[str, object],
) -> dict[str, object]:
    return {
        "sample_key": sample_key,
        "candidate_index": 0,
        "source_frame_id": str(feature.get("source_frame_id", "")),
        "source_patch_quality_score": float(
            feature.get("source_patch_quality_score", 0.0)
        ),
        "source_patch_detail_score": min(
            float(feature.get("source_patch_luma_std", 0.0)) / 0.020,
            1.0,
        ),
        "source_patch_density_score": 0.0,
        "source_patch_localized_support_score": float(
            feature.get("source_patch_localization_fraction", 0.0)
        ),
        "source_patch_localization_candidate_count": 0,
        "source_patch_localization_selected_count": int(
            feature.get("sample_count", 0)
        ),
        "source_patch_localization_fraction": float(
            feature.get("source_patch_localization_fraction", 0.0)
        ),
        "source_patch_frontal_face_confidence": float(
            feature.get("source_patch_frontal_face_confidence", 0.0)
        ),
        "source_patch_view_weight": float(feature.get("source_patch_view_weight", 0.0)),
        "source_patch_view_label": str(feature.get("source_patch_view_label", "")),
        "source_patch_frontal_photo_priority": float(
            feature.get("source_patch_frontal_photo_priority", 0.0)
        ),
        "source_patch_view_weighted_rank": float(
            feature.get("source_patch_view_weighted_rank", 0.0)
        ),
        "selected": True,
    }


def _direct_photo_candidate_report(
    sample_key: str,
    candidate: dict[str, object],
    frame: dict[str, Any],
    *,
    selected_frame_id: str,
    candidate_metrics_missing: bool,
) -> dict[str, object]:
    decision = _direct_photo_candidate_decision(sample_key, candidate, frame)
    frame_id = str(candidate.get("source_frame_id", ""))
    return {
        "sample_key": sample_key,
        "candidate_index": int(candidate.get("candidate_index", 0)),
        "source_frame_id": frame_id,
        "source_media_id": str(frame.get("source_media_id", "")),
        "media_type": str(frame.get("media_type", "")),
        "source_type": str(frame.get("source_type", "")),
        "selected": bool(candidate.get("selected", False))
        or (bool(selected_frame_id) and frame_id == selected_frame_id),
        "candidate_metrics_missing": bool(candidate_metrics_missing),
        "source_patch_quality_score": round(
            _numeric(candidate.get("source_patch_quality_score")),
            6,
        ),
        "source_patch_detail_score": round(
            _numeric(candidate.get("source_patch_detail_score")),
            6,
        ),
        "source_patch_density_score": round(
            _numeric(candidate.get("source_patch_density_score")),
            6,
        ),
        "source_patch_localized_support_score": round(
            _numeric(candidate.get("source_patch_localized_support_score")),
            6,
        ),
        "source_patch_frontal_face_confidence": round(
            _numeric(candidate.get("source_patch_frontal_face_confidence")),
            6,
        ),
        "source_patch_view_weight": round(
            _numeric(candidate.get("source_patch_view_weight")),
            6,
        ),
        "source_patch_view_score": round(float(decision["view_score"]), 6),
        "source_patch_view_label": str(candidate.get("source_patch_view_label", "")),
        "source_patch_view_weighted_rank": round(
            _numeric(candidate.get("source_patch_view_weighted_rank")),
            6,
        ),
        "source_patch_frontal_photo_priority": round(
            _numeric(candidate.get("source_patch_frontal_photo_priority")),
            6,
        ),
        "direct_uploaded_photo": bool(decision["direct_uploaded_photo"]),
        "auto_staged_photo": bool(decision["auto_staged_photo"]),
        "direct_photo_eligible": bool(decision["eligible"]),
        "direct_photo_rejection_reasons": list(decision["rejection_reasons"]),
        "direct_photo_thresholds": decision["thresholds"],
    }


def _direct_photo_candidate_decision(
    sample_key: str,
    candidate: dict[str, object],
    frame: dict[str, Any],
) -> dict[str, object]:
    media_type = str(frame.get("media_type", "")).lower()
    source_type = str(frame.get("source_type", "")).lower()
    source_media_id = str(frame.get("source_media_id", "")).strip()
    source_name = Path(source_media_id).name if source_media_id else ""
    suffix = Path(source_name).suffix.lower()
    source_stem = Path(source_name).stem
    auto_staged = (
        source_stem.startswith("frame_")
        and len(source_stem) == len("frame_0000")
        and source_stem.removeprefix("frame_").isdigit()
    )
    photo_like = media_type == "photo" or source_type == "image"
    direct_uploaded_photo = (
        bool(source_name)
        and photo_like
        and suffix in {".jpg", ".jpeg", ".png"}
        and not auto_staged
    )
    view_weight = _numeric(candidate.get("source_patch_view_weight"))
    frontal_face_confidence = _numeric(
        candidate.get("source_patch_frontal_face_confidence")
    )
    view_score = max(
        0.0,
        min(1.0, 0.50 * view_weight + 0.50 * frontal_face_confidence),
    )
    quality = _numeric(candidate.get("source_patch_quality_score"))
    detail = _numeric(candidate.get("source_patch_detail_score"))
    support = _numeric(candidate.get("source_patch_localized_support_score"))
    density = _numeric(candidate.get("source_patch_density_score"))
    reasons: list[str] = []
    if not source_name:
        reasons.append("missing_source_media_id")
    if not photo_like:
        reasons.append("not_photo_media")
    if suffix not in {".jpg", ".jpeg", ".png"}:
        reasons.append("unsupported_photo_extension")
    if auto_staged:
        reasons.append("auto_staged_photo_name")
    if direct_uploaded_photo and view_score < 0.50:
        reasons.append("low_frontal_view_score")
    thresholds = {
        "min_view_score": 0.50,
        "min_quality_score": _DIRECT_PHOTO_BROAD_FACE_MIN_QUALITY,
        "min_broad_face_detail_score": _direct_photo_broad_face_detail_threshold(
            support=support,
            density=density,
            quality=quality,
        ),
        "min_ear_inner_detail_score": 0.70,
        "min_eye_detail_score": 0.25,
        "min_pupil_quality_score": _DIRECT_PHOTO_PUPIL_MIN_QUALITY,
        "min_pupil_support_score": _DIRECT_PHOTO_PUPIL_MIN_SUPPORT,
        "min_pupil_density_score": _DIRECT_PHOTO_PUPIL_MIN_DENSITY,
        "broad_face_relaxed_min_support": (
            _DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_SUPPORT
        ),
        "broad_face_relaxed_min_density": (
            _DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_DENSITY
        ),
    }
    if direct_uploaded_photo:
        if sample_key in {"muzzle", "nose", "mouth"}:
            if detail < thresholds["min_broad_face_detail_score"]:
                reasons.append("low_broad_face_patch_detail")
            if quality < _DIRECT_PHOTO_BROAD_FACE_MIN_QUALITY:
                reasons.append("low_patch_quality")
        elif sample_key == "ear_inner":
            if detail < 0.70:
                reasons.append("low_ear_inner_patch_detail")
            if quality < 0.45:
                reasons.append("low_patch_quality")
        elif sample_key in {"left_pupil", "right_pupil"}:
            if quality < _DIRECT_PHOTO_PUPIL_MIN_QUALITY:
                reasons.append("low_patch_quality")
            if support < _DIRECT_PHOTO_PUPIL_MIN_SUPPORT:
                reasons.append("low_pupil_support")
            if density < _DIRECT_PHOTO_PUPIL_MIN_DENSITY:
                reasons.append("low_pupil_density")
        elif sample_key in {
            "left_eye",
            "right_eye",
            "eye",
            "iris",
            "left_iris",
            "right_iris",
        }:
            if quality < 0.45:
                reasons.append("low_patch_quality")
            if detail < 0.25 and quality < 0.55:
                reasons.append("low_eye_detail_quality_pair")
        else:
            reasons.append("unsupported_sample_key")
    priority = _numeric(candidate.get("source_patch_frontal_photo_priority"))
    if direct_uploaded_photo and not reasons and priority <= 0.0:
        reasons.append("zero_frontal_photo_priority")
    return {
        "direct_uploaded_photo": direct_uploaded_photo,
        "auto_staged_photo": auto_staged,
        "eligible": direct_uploaded_photo and not reasons and priority > 0.0,
        "rejection_reasons": reasons,
        "view_score": view_score,
        "thresholds": thresholds,
    }


def _direct_photo_broad_face_detail_threshold(
    *,
    support: float,
    density: float,
    quality: float,
) -> float:
    if (
        support >= _DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_SUPPORT
        and density >= _DIRECT_PHOTO_BROAD_FACE_RELAXED_MIN_DENSITY
        and quality >= _DIRECT_PHOTO_BROAD_FACE_MIN_QUALITY
    ):
        return _DIRECT_PHOTO_BROAD_FACE_RELAXED_DETAIL_THRESHOLD
    return _DIRECT_PHOTO_BROAD_FACE_DETAIL_THRESHOLD


def _direct_photo_candidate_sort_key(candidate: dict[str, object]) -> tuple[object, ...]:
    return (
        not bool(candidate["selected"]),
        not bool(candidate["direct_uploaded_photo"]),
        -float(candidate["source_patch_frontal_photo_priority"]),
        -float(candidate["source_patch_view_weighted_rank"]),
        str(candidate["source_frame_id"]),
    )


def _numeric(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return result


def _exact_source_patch_for_region(
    region_name: str,
    feature_by_name: dict[str, dict[str, Any]],
) -> _ExactSourcePatch | None:
    images: list[Image.Image] = []
    feature_names: list[str] = []
    patch_paths: list[str] = []
    source_frame_id = ""
    for feature_name in _REGION_SOURCE_PATCH_FEATURES[region_name]:
        feature = feature_by_name.get(feature_name)
        if feature is None or not bool(feature.get("source_patch_used", False)):
            continue
        patch_path = Path(str(feature.get("source_patch_path", "")))
        if not patch_path.exists():
            continue
        expected_sha = str(feature.get("source_patch_sha256", ""))
        if expected_sha and sha256_file(patch_path) != expected_sha:
            continue
        image = Image.open(patch_path).convert("RGB")
        if image.size[0] <= 0 or image.size[1] <= 0:
            continue
        images.append(image)
        feature_names.append(feature_name)
        patch_paths.append(str(patch_path))
        if not source_frame_id:
            source_frame_id = str(feature.get("source_frame_id", ""))
    if not images:
        return None
    combined = images[0] if len(images) == 1 else _combine_source_patch_images(images)
    return _ExactSourcePatch(
        image=np.asarray(combined, dtype=np.uint8),
        source_frame_id=source_frame_id,
        feature_names=feature_names,
        patch_paths=patch_paths,
    )


def _texture_region_uv_bounds(
    region_name: str,
    region: dict[str, Any],
    feature_by_name: dict[str, dict[str, Any]],
) -> tuple[list[float], str]:
    feature_bounds: list[tuple[float, float, float, float]] = []
    for feature_name in _REGION_SOURCE_PATCH_FEATURES[region_name]:
        feature = feature_by_name.get(feature_name)
        if feature is None:
            continue
        uv_raw = feature.get("uv")
        radius_raw = feature.get("radius_uv")
        if not isinstance(uv_raw, list | tuple) or len(uv_raw) != 2:
            continue
        if not isinstance(radius_raw, list | tuple) or len(radius_raw) != 2:
            continue
        u, v = (float(value) for value in uv_raw)
        radius_u, radius_v = (max(float(value), 1e-6) for value in radius_raw)
        pad_u = max(radius_u * 0.18, 0.006)
        pad_v = max(radius_v * 0.18, 0.006)
        feature_bounds.append(
            (
                max(0.0, u - radius_u - pad_u),
                max(0.0, v - radius_v - pad_v),
                min(1.0, u + radius_u + pad_u),
                min(1.0, v + radius_v + pad_v),
            )
        )
    if feature_bounds:
        u0 = min(bound[0] for bound in feature_bounds)
        v0 = min(bound[1] for bound in feature_bounds)
        u1 = max(bound[2] for bound in feature_bounds)
        v1 = max(bound[3] for bound in feature_bounds)
        limits = _TEXTURE_REGION_ISOLATION_LIMITS.get(region_name)
        if limits is not None:
            u0 = max(u0, limits[0])
            v0 = max(v0, limits[1])
            u1 = min(u1, limits[2])
            v1 = min(v1, limits[3])
        if u0 < u1 and v0 < v1:
            return [u0, v0, u1, v1], _ISOLATED_TEXTURE_CROP_BOUNDS_SOURCE
    bounds = region.get("uv_bounds", [])
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValueError(f"face UV region {region_name} must have uv_bounds")
    return [float(value) for value in bounds], "template_face_uv_region_bounds"


def _crop_overlap_metrics(region_reports: list[dict[str, object]]) -> tuple[int, float]:
    pairs = 0
    overlap_area = 0.0
    total_area = 0.0
    bounds_by_name: list[tuple[str, tuple[int, int, int, int]]] = []
    for region in region_reports:
        raw = region.get("pixel_bounds")
        if not isinstance(raw, list) or len(raw) != 4:
            continue
        x0, y0, x1, y1 = (int(value) for value in raw)
        total_area += float(max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1))
        bounds_by_name.append((str(region.get("name", "")), (x0, y0, x1, y1)))
    for index, (_name_a, a) in enumerate(bounds_by_name):
        ax0, ay0, ax1, ay1 = a
        for _name_b, b in bounds_by_name[index + 1 :]:
            bx0, by0, bx1, by1 = b
            ix0 = max(ax0, bx0)
            iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1)
            iy1 = min(ay1, by1)
            if ix0 <= ix1 and iy0 <= iy1:
                pairs += 1
                overlap_area += float((ix1 - ix0 + 1) * (iy1 - iy0 + 1))
    return pairs, float(overlap_area / max(total_area, 1.0))


def _combine_source_patch_images(images: list[Image.Image]) -> Image.Image:
    target_h = max(max(image.size[1], 1) for image in images)
    resized: list[Image.Image] = []
    for image in images:
        scale = target_h / max(image.size[1], 1)
        resized.append(
            image.resize(
                (max(1, int(round(image.size[0] * scale))), target_h),
                Image.Resampling.NEAREST,
            )
        )
    gap = 4
    width = sum(image.size[0] for image in resized) + gap * max(len(resized) - 1, 0)
    canvas = Image.new("RGB", (width, target_h), (238, 238, 238))
    x = 0
    for image in resized:
        canvas.paste(image, (x, 0))
        x += image.size[0] + gap
    return canvas


def _uv_pixel_bounds(
    uv_bounds: object,
    *,
    width: int,
    height: int,
) -> list[int]:
    if not isinstance(uv_bounds, list) or len(uv_bounds) != 4:
        raise ValueError("uv_bounds must contain four values")
    u0, v0, u1, v1 = (float(value) for value in uv_bounds)
    if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
        raise ValueError("uv_bounds values must satisfy 0 <= u0 < u1 <= 1")
    x0 = int(max(0, min(width - 1, round(u0 * float(width - 1)))))
    x1 = int(max(0, min(width - 1, round(u1 * float(width - 1)))))
    y0 = int(max(0, min(height - 1, round((1.0 - v1) * float(height - 1)))))
    y1 = int(max(0, min(height - 1, round((1.0 - v0) * float(height - 1)))))
    return [x0, y0, max(x0, x1), max(y0, y1)]


def _dominant_source_frame(
    source_id_crop: NDArray[np.uint8],
    frames: list[dict[str, Any]],
) -> tuple[str, float]:
    nonzero = source_id_crop[source_id_crop > 0]
    if nonzero.size == 0:
        return "", 0.0
    values, counts = np.unique(nonzero, return_counts=True)
    order = np.lexsort((values, -counts))
    source_id = int(values[order[0]])
    frame_index = source_id - 1
    frame_id = ""
    if 0 <= frame_index < len(frames):
        frame_id = str(frames[frame_index].get("frame_id", ""))
    return frame_id, float(counts[order[0]] / max(source_id_crop.size, 1))


def _source_crop_for_region(
    frame: dict[str, Any],
    region_name: str,
) -> NDArray[np.uint8] | None:
    image = _load_frame_image(frame)
    if image is None:
        return None
    bbox = frame.get("bbox_xyxy", [0.0, 0.0, float(image.shape[1]), float(image.shape[0])])
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    rx0, ry0, rx1, ry1 = _SOURCE_REGION_BOUNDS[region_name]
    bx0, by0, bx1, by1 = (float(value) for value in bbox)
    x0 = int(round(bx0 + rx0 * (bx1 - bx0)))
    y0 = int(round(by0 + ry0 * (by1 - by0)))
    x1 = int(round(bx0 + rx1 * (bx1 - bx0)))
    y1 = int(round(by0 + ry1 * (by1 - by0)))
    x0 = max(0, min(image.shape[1] - 1, x0))
    y0 = max(0, min(image.shape[0] - 1, y0))
    x1 = max(x0, min(image.shape[1] - 1, x1))
    y1 = max(y0, min(image.shape[0] - 1, y1))
    return image[y0 : y1 + 1, x0 : x1 + 1].copy()


def _load_frame_image(frame: dict[str, Any]) -> NDArray[np.uint8] | None:
    image_path = Path(str(frame.get("image_srgb_path", "")))
    if image_path.exists():
        with np.load(image_path) as data:
            image = np.asarray(data["image_srgb"], dtype=np.float32)
        if image.ndim != 3:
            return None
        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] != 3:
            return None
        return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)

    raw_path = Path(str(frame.get("raw_source_path", "")))
    if raw_path.exists():
        return np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
    return None


def _texture_crop_metrics(
    texture_crop: NDArray[np.uint8],
    confidence_crop: NDArray[np.uint8],
) -> dict[str, object]:
    luma = _luma(texture_crop)
    unique_count = int(np.unique(texture_crop.reshape(-1, 3), axis=0).shape[0])
    luma_std = float(np.std(luma))
    return {
        "texture_mean_rgb": _mean_rgb(texture_crop),
        "texture_luma_std": luma_std,
        "texture_unique_rgb_count": unique_count,
        "texture_confident_texel_fraction": float(
            np.count_nonzero(confidence_crop) / max(confidence_crop.size, 1)
        ),
        "texture_detail_present": bool(
            unique_count >= _DETAIL_UNIQUE_RGB_THRESHOLD
            and luma_std >= _DETAIL_LUMA_STD_THRESHOLD
        ),
    }


def _source_comparison_metrics(
    texture_crop: NDArray[np.uint8],
    source_crop: NDArray[np.uint8],
    *,
    source_frame_id: str,
    source_crop_path: Path,
) -> dict[str, object]:
    texture_mean = np.asarray(_mean_rgb(texture_crop), dtype=np.float64)
    source_mean = np.asarray(_mean_rgb(source_crop), dtype=np.float64)
    rgb_distance = float(np.linalg.norm(texture_mean - source_mean) / (np.sqrt(3.0) * 255.0))
    return {
        "source_frame_id": source_frame_id,
        "source_crop_path": str(source_crop_path),
        "source_crop_written": True,
        "source_mean_rgb": [int(value) for value in source_mean],
        "source_luma_std": float(np.std(_luma(source_crop))),
        "texture_source_mean_rgb_distance": rgb_distance,
    }


def _visual_quality_metrics(
    texture_metrics: dict[str, object],
    source_metrics: dict[str, object],
) -> dict[str, object]:
    unique_count = int(texture_metrics.get("texture_unique_rgb_count", 0))
    texture_luma_std = float(texture_metrics.get("texture_luma_std", 0.0))
    confident_fraction = float(
        texture_metrics.get("texture_confident_texel_fraction", 0.0)
    )
    source_luma_std = float(source_metrics.get("source_luma_std", 0.0))
    source_crop_written = bool(source_metrics.get("source_crop_written", False))
    exact_source_patch = bool(source_metrics.get("source_crop_exact_patch_used", False))

    unique_score = _unique_rgb_score(unique_count)
    texture_luma_score = _ratio_score(
        texture_luma_std,
        _VISUAL_QUALITY_TARGET_TEXTURE_LUMA_STD,
    )
    confident_score = _ratio_score(
        confident_fraction,
        _VISUAL_QUALITY_TARGET_CONFIDENT_TEXEL_FRACTION,
    )
    source_luma_score = _ratio_score(
        source_luma_std,
        _VISUAL_QUALITY_TARGET_SOURCE_LUMA_STD,
    )
    source_evidence_score = (
        1.0 if exact_source_patch else 0.65 if source_crop_written else 0.0
    )
    components = {
        "confident_texel_score": confident_score,
        "texture_luma_score": texture_luma_score,
        "unique_rgb_score": unique_score,
        "source_luma_score": source_luma_score,
        "source_evidence_score": source_evidence_score,
    }
    quality_score = (
        0.25 * confident_score
        + 0.25 * texture_luma_score
        + 0.20 * unique_score
        + 0.20 * source_luma_score
        + 0.10 * source_evidence_score
    )
    passed = (
        quality_score >= _VISUAL_QUALITY_MIN_SCORE
        and confident_fraction >= _VISUAL_QUALITY_MIN_CONFIDENT_TEXEL_FRACTION
        and unique_count >= _VISUAL_QUALITY_MIN_UNIQUE_RGB_COUNT
        and texture_luma_std >= _VISUAL_QUALITY_MIN_TEXTURE_LUMA_STD
        and source_luma_std >= _VISUAL_QUALITY_MIN_SOURCE_LUMA_STD
        and source_crop_written
    )
    failure_reasons = _visual_quality_failure_reasons(
        quality_score=quality_score,
        confident_fraction=confident_fraction,
        unique_count=unique_count,
        texture_luma_std=texture_luma_std,
        source_luma_std=source_luma_std,
        source_crop_written=source_crop_written,
    )
    return {
        "texture_visual_quality_score": round(float(quality_score), 6),
        "texture_visual_quality_passed": bool(passed),
        "texture_visual_quality_components": {
            key: round(float(value), 6) for key, value in sorted(components.items())
        },
        "texture_visual_quality_failure_reasons": failure_reasons,
        "texture_visual_quality_weakest_component": min(
            components.items(),
            key=lambda item: (item[1], item[0]),
        )[0],
    }


def _visual_quality_failure_reasons(
    *,
    quality_score: float,
    confident_fraction: float,
    unique_count: int,
    texture_luma_std: float,
    source_luma_std: float,
    source_crop_written: bool,
) -> list[str]:
    reasons: list[str] = []
    if confident_fraction < _VISUAL_QUALITY_MIN_CONFIDENT_TEXEL_FRACTION:
        reasons.append("coverage")
    if (
        unique_count < _VISUAL_QUALITY_MIN_UNIQUE_RGB_COUNT
        or texture_luma_std < _VISUAL_QUALITY_MIN_TEXTURE_LUMA_STD
    ):
        reasons.append("flat_texture")
    if not source_crop_written:
        reasons.append("source_evidence")
    elif source_luma_std < _VISUAL_QUALITY_MIN_SOURCE_LUMA_STD:
        reasons.append("source_detail")
    if quality_score < _VISUAL_QUALITY_MIN_SCORE:
        reasons.append("combined_score")
    return reasons


def _ratio_score(value: float, target: float) -> float:
    return float(max(0.0, min(1.0, value / max(target, 1e-12))))


def _unique_rgb_score(value: int) -> float:
    if value <= 1:
        return 0.0
    return float(
        max(
            0.0,
            min(
                1.0,
                np.log2(float(value)) / np.log2(float(_VISUAL_QUALITY_TARGET_UNIQUE_RGB_COUNT)),
            ),
        )
    )


def _write_contact_sheet(
    path: Path,
    rows: list[tuple[str, Image.Image, Image.Image | None]],
) -> None:
    cell_w = 180
    cell_h = 132
    label_w = 150
    row_h = cell_h + 28
    sheet = Image.new(
        "RGB",
        (label_w + 2 * cell_w, max(row_h * max(len(rows), 1), row_h)),
        (250, 250, 250),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), "region", fill=(20, 20, 20))
    draw.text((label_w + 8, 6), "texture", fill=(20, 20, 20))
    draw.text((label_w + cell_w + 8, 6), "source", fill=(20, 20, 20))
    for row_index, (name, texture, source) in enumerate(rows):
        y = row_index * row_h + 24
        draw.text((8, y + 8), name, fill=(20, 20, 20))
        sheet.paste(_fit_image(texture.convert("RGB"), cell_w, cell_h), (label_w, y))
        source_image = (
            source.convert("RGB")
            if source is not None
            else _missing_source(cell_w, cell_h)
        )
        sheet.paste(_fit_image(source_image, cell_w, cell_h), (label_w + cell_w, y))
        draw.line([(0, y + cell_h + 4), (sheet.size[0], y + cell_h + 4)], fill=(218, 218, 218))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = Image.new("RGB", (width, height), (238, 238, 238))
    scale = min(width / max(image.size[0], 1), height / max(image.size[1], 1))
    resized = image.resize(
        (max(1, int(round(image.size[0] * scale))), max(1, int(round(image.size[1] * scale)))),
        Image.Resampling.NEAREST,
    )
    x = (width - resized.size[0]) // 2
    y = (height - resized.size[1]) // 2
    fitted.paste(resized, (x, y))
    return fitted


def _missing_source(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (230, 230, 230))
    draw = ImageDraw.Draw(image)
    draw.text((12, height // 2 - 8), "missing", fill=(90, 90, 90))
    return image


def _mean_rgb(image: NDArray[np.uint8]) -> list[int]:
    if image.size == 0:
        return [0, 0, 0]
    rgb = np.mean(image.reshape(-1, 3).astype(np.float64), axis=0)
    return [int(round(float(value))) for value in rgb]


def _luma(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    array = image.astype(np.float64) / 255.0
    return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value
