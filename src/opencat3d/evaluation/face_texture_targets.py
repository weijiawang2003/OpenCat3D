"""Deterministic face texture upsampling target reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Any

from opencat3d.evaluation.face_uv_density import REQUIRED_FACE_UV_REGIONS

FACE_TEXTURE_TARGET_SIZE = 1024
TEMPLATE_REGION_V2_TEXTURE_TARGET_MIN_SIZE = 2048


@dataclass(frozen=True)
class FaceTextureTargetEvaluationResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


def evaluate_face_texture_targets_report(
    project_dir: Path,
    *,
    mode: str = "raw",
    force: bool = False,
) -> FaceTextureTargetEvaluationResult:
    """Write deterministic future texture-size targets from face UV density."""

    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported face texture target mode: {mode}")

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / f"{mode}_face_texture_targets.json"
    density_report_path = output_dir / f"{mode}_face_uv_density.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    density_report = _read_json(density_report_path)
    density_metrics = _object(density_report.get("metrics"), "density_report.metrics")
    regions_raw = density_report.get("regions", [])
    regions = [region for region in regions_raw if isinstance(region, dict)]
    current_width = int(density_metrics.get("texture_width", 0))
    current_height = int(density_metrics.get("texture_height", 0))
    current_size = max(current_width, current_height)
    density_floor = min(
        (
            float(region.get("target_texel_density", 0.0))
            for region in regions
            if float(region.get("target_texel_density", 0.0)) > 0.0
        ),
        default=0.0,
    )
    region_targets = _region_targets(
        regions,
        density_floor=density_floor,
        current_size=current_size,
    )
    max_scale = max(
        (float(region["required_texture_scale"]) for region in region_targets),
        default=1.0,
    )
    target_texture_size = _next_power_of_two(ceil(float(current_size) * max_scale))
    target_texture_size = max(
        current_size,
        min(target_texture_size, FACE_TEXTURE_TARGET_SIZE),
    )
    texture_scale = target_texture_size / max(float(current_size), 1.0)
    region_targets = _apply_texture_target(region_targets, texture_scale=texture_scale)
    names = {str(region.get("name", "")) for region in region_targets}
    template_region_v2_targets = _template_region_v2_texture_targets(
        density_report,
        current_size=current_size,
    )
    template_region_v2_target_size = int(
        next(
            (
                int(region["target_texture_size"])
                for region in template_region_v2_targets
            ),
            max(TEMPLATE_REGION_V2_TEXTURE_TARGET_MIN_SIZE, current_size),
        )
    )
    template_region_v2_target_scale = template_region_v2_target_size / max(
        float(current_size),
        1.0,
    )

    requested_texels = int(
        sum(int(region["requested_texels"]) for region in region_targets)
    )
    target_texels = int(
        sum(int(region["target_allocated_texels"]) for region in region_targets)
    )
    eye_targets = [
        int(region["target_allocated_texels"])
        for region in region_targets
        if str(region["name"]) in {"left_eye", "right_eye"}
    ]
    ear_targets = [
        int(region["target_allocated_texels"])
        for region in region_targets
        if str(region["name"]) in {"left_ear_inner", "right_ear_inner"}
    ]
    template_region_v2_requested_texels = int(
        sum(int(region["requested_texels"]) for region in template_region_v2_targets)
    )
    template_region_v2_target_texels = int(
        sum(
            int(region["target_allocated_texels"])
            for region in template_region_v2_targets
        )
    )
    template_region_v2_eye_targets = [
        int(region["target_allocated_texels"])
        for region in template_region_v2_targets
        if str(region["name"]) in {"left_eye", "right_eye"}
    ]
    template_region_v2_ear_targets = [
        int(region["target_allocated_texels"])
        for region in template_region_v2_targets
        if str(region["name"]) in {"left_ear_inner", "right_ear_inner"}
    ]
    metrics: dict[str, float | int] = {
        "current_texture_width": current_width,
        "current_texture_height": current_height,
        "target_texture_size": int(target_texture_size),
        "target_texture_scale": float(texture_scale),
        "face_texture_target_region_count": int(len(region_targets)),
        "face_texture_target_requested_texel_count": requested_texels,
        "face_texture_target_allocated_texel_count": target_texels,
        "face_texture_target_max_region_density_scale": float(
            max(
                (float(region["density_scale"]) for region in region_targets),
                default=0.0,
            )
        ),
        "face_texture_target_eye_min_texels": int(min(eye_targets, default=0)),
        "face_texture_target_nose_texels": int(
            next(
                (
                    int(region["target_allocated_texels"])
                    for region in region_targets
                    if str(region["name"]) == "nose"
                ),
                0,
            )
        ),
        "face_texture_target_ear_inner_min_texels": int(min(ear_targets, default=0)),
        "template_region_v2_texture_target_size": int(
            template_region_v2_target_size
        ),
        "template_region_v2_texture_target_scale": float(
            template_region_v2_target_scale
        ),
        "template_region_v2_texture_target_island_count": int(
            len(template_region_v2_targets)
        ),
        "template_region_v2_texture_target_facial_island_count": int(
            sum(
                1
                for region in template_region_v2_targets
                if str(region["allocation_role"]) == "facial_detail"
            )
        ),
        "template_region_v2_texture_target_requested_texel_count": (
            template_region_v2_requested_texels
        ),
        "template_region_v2_texture_target_allocated_texel_count": (
            template_region_v2_target_texels
        ),
        "template_region_v2_texture_target_eye_min_texels": int(
            min(template_region_v2_eye_targets, default=0)
        ),
        "template_region_v2_texture_target_nose_texels": int(
            next(
                (
                    int(region["target_allocated_texels"])
                    for region in template_region_v2_targets
                    if str(region["name"]) == "nose"
                ),
                0,
            )
        ),
        "template_region_v2_texture_target_muzzle_texels": int(
            next(
                (
                    int(region["target_allocated_texels"])
                    for region in template_region_v2_targets
                    if str(region["name"]) == "muzzle"
                ),
                0,
            )
        ),
        "template_region_v2_texture_target_ear_inner_min_texels": int(
            min(template_region_v2_ear_targets, default=0)
        ),
        "template_region_v2_texture_target_max_density_scale": float(
            max(
                (
                    float(region["density_scale"])
                    for region in template_region_v2_targets
                ),
                default=0.0,
            )
        ),
    }
    checks = {
        "face_uv_density_report_exists": density_report_path.exists(),
        "face_uv_density_report_success": bool(density_report.get("success", False)),
        "texture_size_square": current_width == current_height and current_size > 0,
        "density_floor_positive": density_floor > 0.0,
        "required_regions_present": set(REQUIRED_FACE_UV_REGIONS).issubset(names),
        "target_texture_size_power_of_two": _is_power_of_two(target_texture_size),
        "target_texture_size_not_smaller": target_texture_size >= current_size,
        "target_texels_meet_requested": all(
            int(region["target_allocated_texels"]) >= int(region["requested_texels"])
            for region in region_targets
        ),
        "template_region_v2_texture_targets_present": bool(template_region_v2_targets),
        "template_region_v2_texture_target_size_high_resolution": (
            template_region_v2_target_size
            >= max(current_size, TEMPLATE_REGION_V2_TEXTURE_TARGET_MIN_SIZE)
        ),
        "template_region_v2_texture_target_size_power_of_two": _is_power_of_two(
            template_region_v2_target_size
        ),
        "template_region_v2_texture_target_texels_meet_requested": all(
            int(region["target_allocated_texels"]) >= int(region["requested_texels"])
            for region in template_region_v2_targets
        ),
        "template_region_v2_texture_target_required_features_present": (
            {"left_eye", "right_eye", "nose", "muzzle", "left_ear_inner", "right_ear_inner"}
        ).issubset({str(region["name"]) for region in template_region_v2_targets}),
        "template_region_v2_texture_target_facial_only": all(
            str(region["allocation_role"]) == "facial_detail"
            for region in template_region_v2_targets
        ),
        "metadata_only_no_texture_resample": True,
    }
    success = all(checks.values())
    report = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "source": "opencat3d.face_texture_upsampling_targets_v1",
        "success": success,
        "checks": checks,
        "metrics": metrics,
        "required_regions": list(REQUIRED_FACE_UV_REGIONS),
        "density_report": str(density_report_path),
        "output_contract": {
            "topology_changed": False,
            "texture_resampled": False,
            "target_representation": "metadata_only_future_texture_bake",
        },
        "template_region_v2_target_contract": {
            "source": "opencat3d.cat_template_region_v2_texture_targets_v1",
            "region_scope": "facial_detail_uv_islands",
            "target_texture_size": int(template_region_v2_target_size),
            "target_texture_scale": float(template_region_v2_target_scale),
            "observed_data_only": True,
            "topology_changed": False,
            "texture_resampled": False,
            "downstream_bake_required": True,
        },
        "regions": region_targets,
        "template_region_v2_texture_targets": template_region_v2_targets,
        "artifacts": {
            "density_report": str(density_report_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return FaceTextureTargetEvaluationResult(
        report_path=report_path,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def _region_targets(
    regions: list[dict[str, Any]],
    *,
    density_floor: float,
    current_size: int,
) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    for region in regions:
        allocated_texels = int(region.get("allocated_texels", 0))
        density = float(region.get("target_texel_density", 0.0))
        density_scale = density / max(density_floor, 1e-12)
        if current_size >= FACE_TEXTURE_TARGET_SIZE:
            requested_texels = allocated_texels
            required_texture_scale = 1.0
        else:
            requested_texels = int(ceil(float(allocated_texels) * max(density_scale, 1.0)))
            required_texture_scale = sqrt(
                requested_texels / max(float(allocated_texels), 1.0)
            )
        targets.append(
            {
                "name": str(region.get("name", "")),
                "semantic": str(region.get("semantic", "")),
                "current_allocated_texels": allocated_texels,
                "target_texel_density": density,
                "density_scale": float(density_scale),
                "requested_texels": requested_texels,
                "required_texture_scale": float(required_texture_scale),
                "current_texture_size": int(current_size),
                "target_texture_size": int(current_size),
                "target_allocated_texels": allocated_texels,
            }
        )
    return targets


def _template_region_v2_texture_targets(
    density_report: dict[str, Any],
    *,
    current_size: int,
) -> list[dict[str, object]]:
    islands_raw = density_report.get("unique_template_region_v2_uv_islands", [])
    islands = [island for island in islands_raw if isinstance(island, dict)]
    facial_islands = [
        island
        for island in islands
        if str(island.get("allocation_role", "")) == "facial_detail"
    ]
    density_floor = min(
        (
            float(island.get("target_texel_density", 0.0))
            for island in facial_islands
            if float(island.get("target_texel_density", 0.0)) > 0.0
        ),
        default=0.0,
    )
    required_texture_scale = max(
        (
            sqrt(
                max(
                    float(island.get("target_texel_density", 0.0))
                    / max(density_floor, 1e-12),
                    1.0,
                )
            )
            for island in facial_islands
        ),
        default=1.0,
    )
    if current_size >= TEMPLATE_REGION_V2_TEXTURE_TARGET_MIN_SIZE:
        target_texture_size = current_size
    else:
        target_texture_size = _next_power_of_two(
            ceil(float(max(current_size, 1)) * required_texture_scale)
        )
        target_texture_size = max(
            target_texture_size,
            TEMPLATE_REGION_V2_TEXTURE_TARGET_MIN_SIZE,
        )
    texture_scale = target_texture_size / max(float(current_size), 1.0)
    targets: list[dict[str, object]] = []
    for island in facial_islands:
        allocated_texels = int(island.get("allocated_texels", 0))
        density = float(island.get("target_texel_density", 0.0))
        density_scale = density / max(density_floor, 1e-12)
        density_requested_texels = int(
            ceil(float(allocated_texels) * max(density_scale, 1.0))
        )
        target_allocated_texels = int(
            ceil(float(allocated_texels) * texture_scale * texture_scale)
        )
        requested_texels = min(density_requested_texels, target_allocated_texels)
        feature_names = island.get("feature_names", [])
        targets.append(
            {
                "name": str(island.get("name", "")),
                "semantic": str(island.get("semantic", "")),
                "source": str(island.get("source", "")),
                "source_region": str(island.get("source_region", "")),
                "allocation_role": str(island.get("allocation_role", "")),
                "feature_names": [
                    str(name) for name in feature_names
                ]
                if isinstance(feature_names, list)
                else [],
                "uv_bounds": [
                    round(float(value), 9)
                    for value in island.get("uv_bounds", [])
                    if isinstance(value, int | float)
                ],
                "current_allocated_texels": allocated_texels,
                "density_weighted_texels": int(
                    island.get("density_weighted_texels", 0)
                ),
                "target_texel_density": density,
                "density_scale": float(density_scale),
                "requested_texels": requested_texels,
                "required_texture_scale": float(
                    sqrt(requested_texels / max(float(allocated_texels), 1.0))
                ),
                "current_texture_size": int(current_size),
                "target_texture_size": int(target_texture_size),
                "target_allocated_texels": target_allocated_texels,
                "target_surplus_texels": target_allocated_texels - requested_texels,
                "texture_priority": int(island.get("texture_priority", 0)),
            }
        )
    return targets


def _apply_texture_target(
    regions: list[dict[str, object]],
    *,
    texture_scale: float,
) -> list[dict[str, object]]:
    applied: list[dict[str, object]] = []
    for region in regions:
        current_texels = int(region["current_allocated_texels"])
        target_texels = int(ceil(float(current_texels) * texture_scale * texture_scale))
        updated = dict(region)
        updated["target_texture_size"] = int(
            round(float(region["current_texture_size"]) * texture_scale)
        )
        updated["target_allocated_texels"] = target_texels
        updated["target_surplus_texels"] = target_texels - int(region["requested_texels"])
        applied.append(updated)
    return applied


def _next_power_of_two(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value
