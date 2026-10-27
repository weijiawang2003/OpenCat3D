"""Face UV texture-density reports for fixed-topology texture contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REQUIRED_FACE_UV_REGIONS = (
    "face",
    "left_eye",
    "right_eye",
    "nose",
    "muzzle",
    "mouth_line",
    "left_ear_inner",
    "right_ear_inner",
)

REQUIRED_TEMPLATE_REGION_V2_UV_REGIONS = (
    "face",
    "muzzle",
    "eyes",
    "nose",
    "ear_interiors",
    "paws",
    "tail",
    "torso",
    "belly",
    "limbs",
)

_REGION_COLORS: dict[str, tuple[int, int, int, int]] = {
    "face": (70, 112, 196, 72),
    "left_eye": (42, 176, 112, 116),
    "right_eye": (42, 176, 112, 116),
    "nose": (224, 88, 96, 128),
    "muzzle": (236, 204, 84, 96),
    "mouth_line": (184, 70, 164, 144),
    "left_ear_inner": (232, 140, 80, 112),
    "right_ear_inner": (232, 140, 80, 112),
}


@dataclass(frozen=True)
class FaceUvDensityEvaluationResult:
    report_path: Path
    overlay_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


def evaluate_face_uv_density_report(
    project_dir: Path,
    *,
    mode: str = "raw",
    force: bool = False,
) -> FaceUvDensityEvaluationResult:
    """Write a deterministic texture-space overlay for face UV density."""

    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported face UV density mode: {mode}")

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / f"{mode}_face_uv_density.json"
    overlay_path = output_dir / f"{mode}_face_uv_density.png"
    texture_manifest_path = project_dir / "outputs" / "textures" / "texture_manifest.json"
    basecolor_path = project_dir / "outputs" / "textures" / "basecolor.png"
    mesh_validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")
    if overlay_path.exists() and not force:
        raise FileExistsError(f"{overlay_path} already exists; pass force=True to overwrite")

    manifest = _read_json(texture_manifest_path)
    face_uv_regions = manifest.get("template_face_uv_regions", {})
    if not isinstance(face_uv_regions, dict):
        face_uv_regions = {}
    regions_raw = face_uv_regions.get("regions", [])
    regions = [region for region in regions_raw if isinstance(region, dict)]
    basecolor = manifest.get("basecolor", {})
    if not isinstance(basecolor, dict):
        basecolor = {}
    width = int(basecolor.get("width", 0))
    height = int(basecolor.get("height", 0))
    region_reports = _region_reports(regions, width=width, height=height)
    template_region_reports = _template_region_v2_uv_island_reports(
        mesh_validation_path,
        width=width,
        height=height,
    )
    unique_template_islands = _unique_template_region_v2_uv_islands(
        template_region_reports
    )
    names = {str(region.get("name", "")) for region in regions}
    template_region_names = {
        str(region.get("name", "")) for region in template_region_reports
    }
    bounds_valid = all(bool(region["bounds_valid"]) for region in region_reports)
    positive_texels = all(int(region["allocated_texels"]) > 0 for region in region_reports)
    density_positive = all(
        float(region["target_texel_density"]) > 0.0 for region in region_reports
    )
    template_region_bounds_valid = all(
        bool(island["bounds_valid"])
        for region in template_region_reports
        for island in _islands(region)
    )
    template_region_positive_texels = all(
        int(island["allocated_texels"]) > 0
        for region in template_region_reports
        for island in _islands(region)
    )
    template_region_density_positive = all(
        float(island["target_texel_density"]) > 0.0
        for region in template_region_reports
        for island in _islands(region)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_overlay(
        basecolor_path,
        overlay_path,
        width=width,
        height=height,
        regions=region_reports,
    )

    eye_texels = [
        int(region["allocated_texels"])
        for region in region_reports
        if str(region["name"]) in {"left_eye", "right_eye"}
    ]
    ear_texels = [
        int(region["allocated_texels"])
        for region in region_reports
        if str(region["name"]) in {"left_ear_inner", "right_ear_inner"}
    ]
    template_eye_texels = [
        int(island["allocated_texels"])
        for island in unique_template_islands
        if str(island["name"]) in {"left_eye", "right_eye"}
    ]
    template_ear_texels = [
        int(island["allocated_texels"])
        for island in unique_template_islands
        if str(island["name"]) in {"left_ear_inner", "right_ear_inner"}
    ]
    template_region_by_name = {
        str(region.get("name", "")): region for region in template_region_reports
    }
    total_allocated = int(
        sum(int(region["allocated_texels"]) for region in region_reports)
    )
    template_unique_total = int(
        sum(int(island["allocated_texels"]) for island in unique_template_islands)
    )
    template_unique_weighted_total = int(
        sum(int(island["density_weighted_texels"]) for island in unique_template_islands)
    )
    facial_template_islands = [
        island
        for island in unique_template_islands
        if str(island["allocation_role"]) == "facial_detail"
    ]
    metrics: dict[str, float | int] = {
        "texture_width": width,
        "texture_height": height,
        "face_uv_region_count": int(len(regions)),
        "face_uv_region_total_area": float(
            sum(float(region["uv_area"]) for region in region_reports)
        ),
        "face_uv_region_allocated_texel_count": total_allocated,
        "face_uv_region_min_allocated_texels": int(
            min((int(region["allocated_texels"]) for region in region_reports), default=0)
        ),
        "face_uv_region_max_allocated_texels": int(
            max((int(region["allocated_texels"]) for region in region_reports), default=0)
        ),
        "face_uv_eye_min_allocated_texels": int(min(eye_texels, default=0)),
        "face_uv_nose_allocated_texels": int(
            next(
                (
                    int(region["allocated_texels"])
                    for region in region_reports
                    if str(region["name"]) == "nose"
                ),
                0,
            )
        ),
        "face_uv_ear_inner_min_allocated_texels": int(min(ear_texels, default=0)),
        "template_region_v2_uv_region_count": int(len(template_region_reports)),
        "template_region_v2_uv_required_region_count": int(
            len(REQUIRED_TEMPLATE_REGION_V2_UV_REGIONS)
        ),
        "template_region_v2_uv_required_present_count": int(
            len(set(REQUIRED_TEMPLATE_REGION_V2_UV_REGIONS).intersection(template_region_names))
        ),
        "template_region_v2_uv_island_count": int(len(unique_template_islands)),
        "template_region_v2_region_scoped_uv_island_count": int(
            sum(int(region["uv_island_count"]) for region in template_region_reports)
        ),
        "template_region_v2_uv_island_allocated_texel_count": template_unique_total,
        "template_region_v2_uv_island_density_weighted_texel_count": (
            template_unique_weighted_total
        ),
        "template_region_v2_uv_island_min_allocated_texels": int(
            min(
                (int(island["allocated_texels"]) for island in unique_template_islands),
                default=0,
            )
        ),
        "template_region_v2_face_uv_island_count": int(
            template_region_by_name.get("face", {}).get("uv_island_count", 0)
        ),
        "template_region_v2_facial_uv_island_count": int(len(facial_template_islands)),
        "template_region_v2_facial_uv_island_allocated_texel_count": int(
            sum(int(island["allocated_texels"]) for island in facial_template_islands)
        ),
        "template_region_v2_body_region_uv_island_count": int(
            sum(
                1
                for island in unique_template_islands
                if str(island["source"])
                == "opencat3d.cat_template_region_contract_v2_bounds"
            )
        ),
        "template_region_v2_eye_min_allocated_texels": int(
            min(template_eye_texels, default=0)
        ),
        "template_region_v2_nose_allocated_texels": int(
            next(
                (
                    int(island["allocated_texels"])
                    for island in unique_template_islands
                    if str(island["name"]) == "nose"
                ),
                0,
            )
        ),
        "template_region_v2_muzzle_allocated_texels": int(
            next(
                (
                    int(island["allocated_texels"])
                    for island in unique_template_islands
                    if str(island["name"]) == "muzzle"
                ),
                0,
            )
        ),
        "template_region_v2_ear_inner_min_allocated_texels": int(
            min(template_ear_texels, default=0)
        ),
        "template_region_v2_facial_density_weighted_texel_count": int(
            sum(int(island["density_weighted_texels"]) for island in facial_template_islands)
        ),
    }
    checks = {
        "texture_manifest_exists": texture_manifest_path.exists(),
        "basecolor_exists": basecolor_path.exists(),
        "mesh_validation_exists": mesh_validation_path.exists(),
        "face_uv_regions_present": bool(regions),
        "required_face_uv_regions_present": set(REQUIRED_FACE_UV_REGIONS).issubset(names),
        "texture_size_valid": width > 0 and height > 0,
        "face_uv_region_bounds_valid": bounds_valid,
        "face_uv_region_density_positive": density_positive,
        "face_uv_region_texels_positive": positive_texels,
        "template_region_v2_uv_islands_present": bool(template_region_reports),
        "required_template_region_v2_uv_regions_present": set(
            REQUIRED_TEMPLATE_REGION_V2_UV_REGIONS
        ).issubset(template_region_names),
        "template_region_v2_uv_island_bounds_valid": template_region_bounds_valid,
        "template_region_v2_uv_island_density_positive": (
            template_region_density_positive
        ),
        "template_region_v2_uv_island_texels_positive": template_region_positive_texels,
        "face_uv_density_overlay_written": overlay_path.exists(),
    }
    success = all(checks.values())
    report = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "source": "opencat3d.face_uv_texture_density_v1",
        "success": success,
        "checks": checks,
        "metrics": metrics,
        "required_regions": list(REQUIRED_FACE_UV_REGIONS),
        "required_template_region_v2_regions": list(REQUIRED_TEMPLATE_REGION_V2_UV_REGIONS),
        "regions": region_reports,
        "template_regions_v2": template_region_reports,
        "unique_template_region_v2_uv_islands": unique_template_islands,
        "artifacts": {
            "texture_manifest": str(texture_manifest_path),
            "mesh_validation": str(mesh_validation_path),
            "basecolor": str(basecolor_path),
            "overlay_png": str(overlay_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return FaceUvDensityEvaluationResult(
        report_path=report_path,
        overlay_path=overlay_path,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def _region_reports(
    regions: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for region in regions:
        bounds = region.get("uv_bounds", [])
        bounds_valid = isinstance(bounds, list) and len(bounds) == 4
        if bounds_valid:
            u0, v0, u1, v1 = (float(value) for value in bounds)
            bounds_valid = 0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0
        else:
            u0, v0, u1, v1 = 0.0, 0.0, 0.0, 0.0
        uv_area = max(0.0, u1 - u0) * max(0.0, v1 - v0)
        allocated_texels = int(round(uv_area * float(width) * float(height)))
        pixel_bounds = _pixel_bounds((u0, v0, u1, v1), width=width, height=height)
        reports.append(
            {
                "name": str(region.get("name", "")),
                "semantic": str(region.get("semantic", "")),
                "uv_bounds": [round(float(value), 9) for value in (u0, v0, u1, v1)],
                "pixel_bounds": pixel_bounds,
                "uv_area": float(uv_area),
                "allocated_texels": max(0, allocated_texels),
                "target_texel_density": float(region.get("target_texel_density", 0.0)),
                "texture_priority": int(region.get("texture_priority", 0)),
                "bounds_valid": bool(bounds_valid),
            }
        )
    return reports


def _template_region_v2_uv_island_reports(
    mesh_validation_path: Path,
    *,
    width: int,
    height: int,
) -> list[dict[str, object]]:
    if not mesh_validation_path.exists():
        return []
    mesh_validation = _read_json(mesh_validation_path)
    manifest = mesh_validation.get("template_regions_v2", {})
    if not isinstance(manifest, dict):
        return []
    regions_raw = manifest.get("regions", [])
    if not isinstance(regions_raw, list):
        return []
    reports: list[dict[str, object]] = []
    for raw_region in regions_raw:
        if not isinstance(raw_region, dict):
            continue
        islands_raw = raw_region.get("uv_islands", [])
        if not isinstance(islands_raw, list):
            islands_raw = []
        island_reports = [
            _template_uv_island_report(island, width=width, height=height)
            for island in islands_raw
            if isinstance(island, dict)
        ]
        uv_island_names = raw_region.get("uv_island_names", [])
        reports.append(
            {
                "name": str(raw_region.get("name", "")),
                "semantic": str(raw_region.get("semantic", "")),
                "reconstruction_role": str(raw_region.get("reconstruction_role", "")),
                "primary_uv_island": str(raw_region.get("primary_uv_island", "")),
                "uv_island_count": int(raw_region.get("uv_island_count", 0)),
                "uv_island_names": [
                    str(name) for name in uv_island_names
                ]
                if isinstance(uv_island_names, list)
                else [],
                "allocated_texels": int(
                    sum(int(island["allocated_texels"]) for island in island_reports)
                ),
                "density_weighted_texels": int(
                    sum(int(island["density_weighted_texels"]) for island in island_reports)
                ),
                "uv_islands": island_reports,
            }
        )
    return reports


def _template_uv_island_report(
    island: dict[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, object]:
    bounds = island.get("uv_bounds", [])
    bounds_valid = isinstance(bounds, list) and len(bounds) == 4
    if bounds_valid:
        u0, v0, u1, v1 = (float(value) for value in bounds)
        bounds_valid = 0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0
    else:
        u0, v0, u1, v1 = 0.0, 0.0, 0.0, 0.0
    uv_area = max(0.0, u1 - u0) * max(0.0, v1 - v0)
    allocated_texels = max(0, int(round(uv_area * float(width) * float(height))))
    target_density = float(island.get("target_texel_density", 0.0))
    weighted_texels = max(0, int(round(float(allocated_texels) * target_density)))
    feature_names = island.get("feature_names", [])
    return {
        "name": str(island.get("name", "")),
        "semantic": str(island.get("semantic", "")),
        "source": str(island.get("source", "")),
        "source_region": str(island.get("source_region", "")),
        "allocation_role": str(island.get("allocation_role", "")),
        "feature_names": [str(name) for name in feature_names]
        if isinstance(feature_names, list)
        else [],
        "uv_bounds": [round(float(value), 9) for value in (u0, v0, u1, v1)],
        "pixel_bounds": _pixel_bounds((u0, v0, u1, v1), width=width, height=height),
        "uv_area": float(uv_area),
        "allocated_texels": allocated_texels,
        "target_texel_density": target_density,
        "density_weighted_texels": weighted_texels,
        "texture_priority": int(island.get("texture_priority", 0)),
        "bounds_valid": bool(bounds_valid),
    }


def _unique_template_region_v2_uv_islands(
    template_region_reports: list[dict[str, object]],
) -> list[dict[str, object]]:
    unique: dict[tuple[str, str], dict[str, object]] = {}
    for region in template_region_reports:
        for island in _islands(region):
            key = (str(island["source"]), str(island["name"]))
            unique.setdefault(key, island)
    return list(unique.values())


def _islands(region: dict[str, object]) -> list[dict[str, object]]:
    islands = region.get("uv_islands", [])
    if not isinstance(islands, list):
        return []
    return [island for island in islands if isinstance(island, dict)]


def _pixel_bounds(
    uv_bounds: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> list[int]:
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    u0, v0, u1, v1 = uv_bounds
    x0 = int(max(0, min(width - 1, round(u0 * float(width - 1)))))
    x1 = int(max(0, min(width - 1, round(u1 * float(width - 1)))))
    y0 = int(max(0, min(height - 1, round((1.0 - v1) * float(height - 1)))))
    y1 = int(max(0, min(height - 1, round((1.0 - v0) * float(height - 1)))))
    return [x0, y0, max(x0, x1), max(y0, y1)]


def _write_overlay(
    basecolor_path: Path,
    overlay_path: Path,
    *,
    width: int,
    height: int,
    regions: list[dict[str, object]],
) -> None:
    if basecolor_path.exists():
        base = Image.open(basecolor_path).convert("RGBA")
    else:
        base = Image.new("RGBA", (max(width, 1), max(height, 1)), (255, 255, 255, 255))
    if base.size != (width, height) and width > 0 and height > 0:
        base = base.resize((width, height), Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for region in regions:
        color = _REGION_COLORS.get(str(region["name"]), (80, 80, 80, 96))
        x0, y0, x1, y1 = (int(value) for value in region["pixel_bounds"])
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=color[:3] + (255,), width=2)
    composite = Image.alpha_composite(base, overlay).convert("RGB")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(overlay_path)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data
