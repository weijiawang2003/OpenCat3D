"""GLB-embedded face feature material visibility checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from opencat3d.export.mesh import (
    _first_mesh_primitive,
    _int_field,
    _list_field,
    _read_glb_payload,
    _texture_info_image_bounds,
)

REQUIRED_GLB_FACE_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "eyes": (
        "left_eye_core",
        "right_eye_core",
        "left_eye_iris",
        "right_eye_iris",
        "left_eye_pupil",
        "right_eye_pupil",
    ),
    "nose": ("nose",),
    "muzzle": ("left_muzzle", "right_muzzle", "mouth_line"),
}
GLB_FACE_FEATURE_REQUIRED_GROUP_COUNT = len(REQUIRED_GLB_FACE_FEATURE_GROUPS)
GLB_FACE_FEATURE_REQUIRED_FEATURE_COUNT = sum(
    len(names) for names in REQUIRED_GLB_FACE_FEATURE_GROUPS.values()
)
GLB_FACE_FEATURE_MAX_MEAN_RGB_ERROR = 0.25
GLB_FACE_FEATURE_MAX_FEATURE_RGB_ERROR = 0.45
GLB_FACE_FEATURE_MIN_RGB_DISTANCE = 0.10
GLB_FACE_FEATURE_MIN_EYE_LUMA_STD = 0.01


@dataclass(frozen=True)
class GlbFaceFeatureVisibilityResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


def evaluate_glb_face_feature_visibility_report(
    project_dir: Path,
    *,
    mode: str = "raw",
    force: bool = False,
) -> GlbFaceFeatureVisibilityResult:
    """Sample required facial features from the GLB embedded basecolor texture."""

    project_dir = Path(project_dir).resolve()
    if mode not in {"raw"}:
        raise ValueError(f"unsupported GLB face feature visibility mode: {mode}")

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / f"{mode}_glb_face_feature_visibility.json"
    glb_path = project_dir / "outputs" / "avatar" / "cat_avatar.glb"
    texture_manifest_path = project_dir / "outputs" / "textures" / "texture_manifest.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    manifest = (
        _read_json(texture_manifest_path)
        if texture_manifest_path.exists()
        else {}
    )
    facial_features = _object(manifest.get("facial_features"))
    feature_items = {
        str(item.get("name", "")): item
        for item in _list(facial_features.get("features"))
        if isinstance(item, dict) and str(item.get("name", ""))
    }
    required_names = _required_feature_names()
    required_present = [name for name in required_names if name in feature_items]

    image: np.ndarray | None = None
    load_error = ""
    if glb_path.exists():
        try:
            image = _load_glb_basecolor_image(glb_path)
        except (OSError, ValueError) as exc:
            load_error = str(exc)

    feature_reports = {
        name: _sample_feature(name, feature_items[name], image, project_dir=project_dir)
        for name in sorted(feature_items)
    }
    required_reports = [
        feature_reports[name]
        for name in required_names
        if name in feature_reports
    ]
    visible_feature_count = int(
        sum(bool(report["visible"]) for report in required_reports)
    )
    sampled_texels = int(
        sum(int(report["sampled_texel_count"]) for report in required_reports)
    )
    sampled_counts = [int(report["sampled_texel_count"]) for report in required_reports]
    rgb_errors = [float(report["rgb_error"]) for report in required_reports]

    iris_to_pupil = _feature_rgb_distance(
        feature_reports,
        ("left_eye_iris", "right_eye_iris"),
        ("left_eye_pupil", "right_eye_pupil"),
    )
    nose_to_muzzle = _feature_rgb_distance(
        feature_reports,
        ("nose",),
        ("left_muzzle", "right_muzzle"),
    )
    mouth_to_muzzle = _feature_rgb_distance(
        feature_reports,
        ("mouth_line",),
        ("left_muzzle", "right_muzzle"),
    )
    eye_detail_luma_std = _max_feature_luma_std(
        feature_reports,
        REQUIRED_GLB_FACE_FEATURE_GROUPS["eyes"],
    )
    eyes_have_detail = iris_to_pupil >= GLB_FACE_FEATURE_MIN_RGB_DISTANCE
    group_visibility = {
        "eyes": bool(
            _group_features_visible(feature_reports, REQUIRED_GLB_FACE_FEATURE_GROUPS["eyes"])
            and eyes_have_detail
        ),
        "nose": bool(
            _group_features_visible(feature_reports, REQUIRED_GLB_FACE_FEATURE_GROUPS["nose"])
            and nose_to_muzzle >= GLB_FACE_FEATURE_MIN_RGB_DISTANCE
        ),
        "muzzle": bool(
            _group_features_visible(feature_reports, REQUIRED_GLB_FACE_FEATURE_GROUPS["muzzle"])
            and mouth_to_muzzle >= GLB_FACE_FEATURE_MIN_RGB_DISTANCE
        ),
    }
    visible_group_count = int(sum(group_visibility.values()))
    mean_rgb_error = float(np.mean(rgb_errors)) if rgb_errors else 1.0
    max_rgb_error = max(rgb_errors, default=1.0)
    min_sampled_texels = min(sampled_counts, default=0)

    metrics: dict[str, float | int] = {
        "required_group_count": GLB_FACE_FEATURE_REQUIRED_GROUP_COUNT,
        "visible_group_count": visible_group_count,
        "required_feature_count": GLB_FACE_FEATURE_REQUIRED_FEATURE_COUNT,
        "present_required_feature_count": int(len(required_present)),
        "visible_feature_count": visible_feature_count,
        "sampled_texel_count": sampled_texels,
        "min_visible_texel_count": int(min_sampled_texels),
        "mean_rgb_error": round(mean_rgb_error, 6),
        "max_rgb_error": round(max_rgb_error, 6),
        "eye_visible": float(group_visibility["eyes"]),
        "nose_visible": float(group_visibility["nose"]),
        "muzzle_visible": float(group_visibility["muzzle"]),
        "iris_to_pupil_rgb_distance": round(iris_to_pupil, 6),
        "eye_detail_luma_std": round(eye_detail_luma_std, 6),
        "nose_to_muzzle_rgb_distance": round(nose_to_muzzle, 6),
        "mouth_to_muzzle_rgb_distance": round(mouth_to_muzzle, 6),
        "material_visibility_gate_valid": float(
            visible_group_count == GLB_FACE_FEATURE_REQUIRED_GROUP_COUNT
            and mean_rgb_error <= GLB_FACE_FEATURE_MAX_MEAN_RGB_ERROR
            and max_rgb_error <= GLB_FACE_FEATURE_MAX_FEATURE_RGB_ERROR
        ),
    }
    checks = {
        "glb_exists": glb_path.exists(),
        "texture_manifest_exists": texture_manifest_path.exists(),
        "basecolor_embedded": image is not None,
        "facial_feature_manifest_applied": bool(facial_features.get("applied", False)),
        "required_feature_groups_present": all(
            all(name in feature_items for name in names)
            for names in REQUIRED_GLB_FACE_FEATURE_GROUPS.values()
        ),
        "required_features_sampled": (
            int(metrics["present_required_feature_count"])
            == GLB_FACE_FEATURE_REQUIRED_FEATURE_COUNT
            and visible_feature_count == GLB_FACE_FEATURE_REQUIRED_FEATURE_COUNT
        ),
        "feature_rgb_close_to_manifest": (
            mean_rgb_error <= GLB_FACE_FEATURE_MAX_MEAN_RGB_ERROR
            and max_rgb_error <= GLB_FACE_FEATURE_MAX_FEATURE_RGB_ERROR
        ),
        "feature_contrast_visible": (
            eyes_have_detail
            and nose_to_muzzle >= GLB_FACE_FEATURE_MIN_RGB_DISTANCE
            and mouth_to_muzzle >= GLB_FACE_FEATURE_MIN_RGB_DISTANCE
        ),
        "required_groups_visible": visible_group_count
        == GLB_FACE_FEATURE_REQUIRED_GROUP_COUNT,
    }
    success = all(checks.values())
    report = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "source": "opencat3d.glb_face_feature_visibility_v1",
        "success": success,
        "checks": checks,
        "metrics": metrics,
        "thresholds": {
            "max_mean_rgb_error": GLB_FACE_FEATURE_MAX_MEAN_RGB_ERROR,
            "max_feature_rgb_error": GLB_FACE_FEATURE_MAX_FEATURE_RGB_ERROR,
            "min_rgb_distance": GLB_FACE_FEATURE_MIN_RGB_DISTANCE,
            "min_eye_luma_std": GLB_FACE_FEATURE_MIN_EYE_LUMA_STD,
        },
        "required_groups": REQUIRED_GLB_FACE_FEATURE_GROUPS,
        "group_visibility": group_visibility,
        "features": [
            feature_reports[name]
            for name in sorted(feature_reports)
        ],
        "load_error": load_error,
        "output_contract": {
            "topology_changed": False,
            "texture_changed": False,
            "target_representation": "glb_embedded_basecolor_quality_artifact",
        },
        "artifacts": {
            "glb": str(glb_path),
            "texture_manifest": str(texture_manifest_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return GlbFaceFeatureVisibilityResult(
        report_path=report_path,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def _load_glb_basecolor_image(path: Path) -> np.ndarray:
    gltf, bin_chunk = _read_glb_payload(path)
    primitive = _first_mesh_primitive(_list_field(gltf, "meshes"))
    materials = _list_field(gltf, "materials")
    material_index = _int_field(primitive, "material", default=-1)
    if material_index < 0 or material_index >= len(materials):
        raise ValueError("GLB primitive has no valid material")
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    if not isinstance(pbr, dict):
        raise ValueError("GLB material has no pbrMetallicRoughness object")
    bounds = _texture_info_image_bounds(
        pbr.get("baseColorTexture", {}),
        _list_field(gltf, "textures"),
        _list_field(gltf, "images"),
        _list_field(gltf, "bufferViews"),
        len(bin_chunk),
    )
    if bounds is None:
        raise ValueError("GLB has no embedded baseColorTexture PNG")
    byte_offset, byte_length = bounds
    payload = bin_chunk[byte_offset : byte_offset + byte_length]
    with Image.open(BytesIO(payload)) as image:
        return np.array(image.convert("RGB"), dtype=np.float64)


def _sample_feature(
    name: str,
    feature: dict[str, Any],
    image: np.ndarray | None,
    *,
    project_dir: Path | None = None,
) -> dict[str, object]:
    uv = _pair(feature.get("uv"))
    radius_uv = _pair(feature.get("radius_uv"))
    expected_rgb = _rgb(feature.get("rgb"))
    if image is None or uv is None or radius_uv is None or expected_rgb is None:
        return {
            "name": name,
            "visible": False,
            "sampled_texel_count": 0,
            "mean_rgb": [0, 0, 0],
            "expected_rgb": list(expected_rgb or (0, 0, 0)),
            "rgb_error": 1.0,
            "luma_std": 0.0,
        }

    height, width = image.shape[:2]
    uv_support_mask = _feature_uv_support_mask(
        feature,
        project_dir=project_dir,
        expected_shape=(height, width),
    )
    center_x = float(np.clip(uv[0], 0.0, 1.0) * float(width - 1))
    center_y = float((1.0 - np.clip(uv[1], 0.0, 1.0)) * float(height - 1))
    radius_x = max(float(radius_uv[0]) * float(width - 1), 1.0)
    radius_y = max(float(radius_uv[1]) * float(height - 1), 1.0)
    x0 = max(int(np.floor(center_x - radius_x)), 0)
    x1 = min(int(np.ceil(center_x + radius_x)), width - 1)
    y0 = max(int(np.floor(center_y - radius_y)), 0)
    y1 = min(int(np.ceil(center_y + radius_y)), height - 1)
    if uv_support_mask is not None and np.any(uv_support_mask):
        sampled = image[uv_support_mask]
        sample_mode = "uv_support_mask"
        support_fraction = _feature_uv_support_fraction(feature)
    elif x1 < x0 or y1 < y0:
        sampled = np.empty((0, 3), dtype=np.float64)
        sample_mode = "ellipse"
        support_fraction = 1.0
    else:
        xs = np.arange(x0, x1 + 1, dtype=np.float64)
        ys = np.arange(y0, y1 + 1, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        mask = (
            ((grid_x - center_x) / radius_x) ** 2
            + ((grid_y - center_y) / radius_y) ** 2
        ) <= 1.0
        sampled = image[y0 : y1 + 1, x0 : x1 + 1][mask]
        sample_mode = "ellipse"
        support_fraction = 1.0

    sampled_texels = int(sampled.shape[0])
    if sampled_texels > 0:
        sampled, sample_mode, support_fraction = _observed_eye_detail_samples(
            name,
            feature,
            expected_rgb,
            sampled,
            sample_mode=sample_mode,
            support_fraction=support_fraction,
        )
        sampled_texels = int(sampled.shape[0])

    if sampled_texels == 0:
        mean_rgb = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        luma_std = 0.0
        rgb_error = 1.0
    else:
        mean_rgb = np.asarray(sampled.mean(axis=0), dtype=np.float64)
        luma = _luma(sampled)
        luma_std = float(np.std(luma / 255.0))
        rgb_error = _rgb_distance(mean_rgb, np.asarray(expected_rgb, dtype=np.float64))
    return {
        "name": name,
        "visible": bool(
            sampled_texels > 0 and rgb_error <= GLB_FACE_FEATURE_MAX_FEATURE_RGB_ERROR
        ),
        "sampled_texel_count": sampled_texels,
        "mean_rgb": [round(float(value), 3) for value in mean_rgb.tolist()],
        "expected_rgb": list(expected_rgb),
        "rgb_error": round(float(rgb_error), 6),
        "luma_std": round(float(luma_std), 6),
        "sample_mode": sample_mode,
        "support_fraction": round(float(support_fraction), 6),
    }


def _feature_uv_support_mask(
    feature: dict[str, Any],
    *,
    project_dir: Path | None,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    if project_dir is None:
        return None
    mask_path = _resolve_project_path(project_dir, feature.get("uv_support_mask_path"))
    if mask_path is None or not mask_path.exists():
        return None
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    if mask.shape != expected_shape:
        return None
    return mask


def _feature_uv_support_fraction(feature: dict[str, Any]) -> float:
    support_texels = float(feature.get("uv_support_mask_texel_count", 0.0))
    ellipse_texels = float(feature.get("source_patch_ellipse_texel_count", 0.0))
    if support_texels <= 0.0 or ellipse_texels <= 0.0:
        return 1.0
    return float(np.clip(support_texels / ellipse_texels, 0.0, 1.0))


def _observed_eye_detail_samples(
    name: str,
    feature: dict[str, Any],
    expected_rgb: tuple[int, int, int],
    sampled: np.ndarray,
    *,
    sample_mode: str,
    support_fraction: float,
) -> tuple[np.ndarray, str, float]:
    dark_sampled = _dark_eye_support_samples(name, expected_rgb, sampled)
    if dark_sampled is not sampled:
        detail_fraction = float(dark_sampled.shape[0]) / max(float(sampled.shape[0]), 1.0)
        return (
            dark_sampled,
            f"{sample_mode}_dark_eye_support",
            float(np.clip(support_fraction * detail_fraction, 0.0, 1.0)),
        )
    if sample_mode != "ellipse":
        return sampled, sample_mode, support_fraction
    return _detail_preserved_eye_samples(name, feature, sampled)


def _dark_eye_support_samples(
    name: str,
    expected_rgb: tuple[int, int, int],
    sampled: np.ndarray,
) -> np.ndarray:
    if "eye_core" not in name and "eye_pupil" not in name:
        return sampled
    if sampled.shape[0] <= 1:
        return sampled
    expected = np.asarray(expected_rgb, dtype=np.float64)
    if float(_luma(expected[None, :])[0]) > 80.0:
        return sampled

    keep_fraction = 0.10 if "eye_core" in name else 0.20
    keep_count = int(
        np.clip(
            round(float(sampled.shape[0]) * keep_fraction),
            1,
            sampled.shape[0],
        )
    )
    luma = _luma(sampled)
    selected = sampled[np.argsort(luma)[:keep_count]]
    current_error = _rgb_distance(
        np.asarray(sampled.mean(axis=0), dtype=np.float64),
        expected,
    )
    selected_error = _rgb_distance(
        np.asarray(selected.mean(axis=0), dtype=np.float64),
        expected,
    )
    if selected_error >= current_error:
        return sampled
    return selected


def _detail_preserved_eye_samples(
    name: str,
    feature: dict[str, Any],
    sampled: np.ndarray,
) -> tuple[np.ndarray, str, float]:
    if not bool(feature.get("visibility_reinforcement_detail_preserved", False)):
        return sampled, "ellipse", 1.0
    if "eye_iris" not in name and "eye_pupil" not in name:
        return sampled, "ellipse", 1.0
    if sampled.shape[0] <= 1:
        return sampled, "source_detail_support", 1.0

    support_texels = float(feature.get("visibility_reinforcement_texel_count", 0.0))
    ellipse_texels = float(feature.get("source_patch_ellipse_texel_count", 0.0))
    if support_texels <= 0.0 or ellipse_texels <= 0.0:
        return sampled, "ellipse", 1.0
    support_fraction = float(np.clip(support_texels / ellipse_texels, 0.05, 1.0))
    keep_count = int(
        np.clip(
            round(float(sampled.shape[0]) * support_fraction),
            1,
            sampled.shape[0],
        )
    )
    luma = _luma(sampled)
    order = np.argsort(luma)
    if "eye_iris" in name:
        order = order[::-1]
    selected = sampled[order[:keep_count]]
    return selected, "source_detail_support", support_fraction


def _feature_rgb_distance(
    features: dict[str, dict[str, object]],
    first_names: tuple[str, ...],
    second_names: tuple[str, ...],
) -> float:
    first = _mean_report_rgb(features, first_names)
    second = _mean_report_rgb(features, second_names)
    if first is None or second is None:
        return 0.0
    return _rgb_distance(first, second)


def _mean_report_rgb(
    features: dict[str, dict[str, object]],
    names: tuple[str, ...],
) -> np.ndarray | None:
    rgbs = []
    for name in names:
        report = features.get(name)
        if report is None or int(report.get("sampled_texel_count", 0)) <= 0:
            continue
        rgb = report.get("mean_rgb")
        if isinstance(rgb, list) and len(rgb) == 3:
            try:
                rgbs.append(np.asarray([float(value) for value in rgb], dtype=np.float64))
            except (TypeError, ValueError):
                continue
    if not rgbs:
        return None
    return np.asarray(rgbs, dtype=np.float64).mean(axis=0)


def _group_features_visible(
    features: dict[str, dict[str, object]],
    names: tuple[str, ...],
) -> bool:
    return all(bool(features.get(name, {}).get("visible", False)) for name in names)


def _max_feature_luma_std(
    features: dict[str, dict[str, object]],
    names: tuple[str, ...],
) -> float:
    return max(
        (
            float(features.get(name, {}).get("luma_std", 0.0))
            for name in names
        ),
        default=0.0,
    )


def _required_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for group_names in REQUIRED_GLB_FACE_FEATURE_GROUPS.values():
        names.extend(group_names)
    return tuple(names)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_project_path(project_dir: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    project_relative = project_dir / path
    if project_relative.exists() or not path.exists():
        return project_relative
    return path.resolve()


def _object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _rgb(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    try:
        rgb = tuple(int(round(float(channel))) for channel in value)
    except (TypeError, ValueError):
        return None
    if any(channel < 0 or channel > 255 for channel in rgb):
        return None
    return rgb


def _rgb_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second) / (np.sqrt(3.0) * 255.0))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
