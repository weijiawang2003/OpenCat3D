"""Deterministic neutral PBR companion texture outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from opencat3d.io.cache import sha256_file


@dataclass(frozen=True)
class PbrTextureResult:
    texture_dir: Path
    manifest_path: Path
    normal_path: Path
    roughness_path: Path
    ao_path: Path
    width: int
    height: int
    observed_texel_fraction: float
    unknown_texel_fraction: float
    generated_texel_fraction: float
    face_material_detail_applied: bool = False
    face_material_detail_feature_count: int = 0
    face_material_detail_texel_count: int = 0
    face_material_detail_normal_variance: float = 0.0


def write_neutral_pbr_textures(
    project_dir: Path,
    *,
    force: bool = False,
) -> PbrTextureResult:
    """Write conservative PBR maps without claiming unobserved material detail."""

    texture_dir = project_dir / "outputs" / "textures"
    manifest_path = texture_dir / "texture_manifest.json"
    confidence_path = texture_dir / "basecolor_confidence.png"
    normal_path = texture_dir / "normal.png"
    roughness_path = texture_dir / "roughness.png"
    ao_path = texture_dir / "ao.png"
    existing_paths = (normal_path, roughness_path, ao_path)
    if not force and all(path.exists() for path in existing_paths) and manifest_path.exists():
        return _read_existing_pbr_result(
            texture_dir=texture_dir,
            manifest_path=manifest_path,
            normal_path=normal_path,
            roughness_path=roughness_path,
            ao_path=ao_path,
        )
    for path in existing_paths:
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass force=True to overwrite")
    if not confidence_path.exists():
        raise FileNotFoundError(f"{confidence_path} is required before PBR texture export")

    confidence = np.asarray(Image.open(confidence_path).convert("L"), dtype=np.uint8)
    observed = confidence > 0
    height, width = confidence.shape
    normal = np.zeros((height, width, 3), dtype=np.uint8)
    normal[:, :, 0] = 128
    normal[:, :, 1] = 128
    normal[:, :, 2] = 255
    roughness = np.full((height, width), 255, dtype=np.uint8)
    roughness[observed] = 204
    ao = np.full((height, width), 255, dtype=np.uint8)
    detail_metrics = _apply_face_material_detail(
        normal,
        roughness,
        ao,
        manifest_path=manifest_path,
    )

    texture_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normal).save(normal_path)
    Image.fromarray(roughness).save(roughness_path)
    Image.fromarray(ao).save(ao_path)

    observed_texel_fraction = float(np.count_nonzero(observed) / observed.size)
    result = PbrTextureResult(
        texture_dir=texture_dir,
        manifest_path=manifest_path,
        normal_path=normal_path,
        roughness_path=roughness_path,
        ao_path=ao_path,
        width=width,
        height=height,
        observed_texel_fraction=observed_texel_fraction,
        unknown_texel_fraction=1.0 - observed_texel_fraction,
        generated_texel_fraction=0.0,
        face_material_detail_applied=bool(detail_metrics["applied"]),
        face_material_detail_feature_count=int(detail_metrics["feature_count"]),
        face_material_detail_texel_count=int(detail_metrics["texel_count"]),
        face_material_detail_normal_variance=float(detail_metrics["normal_variance"]),
    )
    _update_texture_manifest(result)
    return result


def _update_texture_manifest(result: PbrTextureResult) -> None:
    manifest: dict[str, Any] = {}
    if result.manifest_path.exists():
        raw = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            manifest = raw
    manifest.setdefault("schema_version", "1.0")
    manifest["pbr_source"] = "opencat3d.neutral_pbr_priors"
    artifacts = manifest.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        manifest["artifacts"] = artifacts
    artifacts.update(
        {
            "normal_png": _manifest_artifact_path(
                result.normal_path,
                result.manifest_path,
            ),
            "roughness_png": _manifest_artifact_path(
                result.roughness_path,
                result.manifest_path,
            ),
            "ao_png": _manifest_artifact_path(result.ao_path, result.manifest_path),
        }
    )
    sha = manifest.setdefault("sha256", {})
    if not isinstance(sha, dict):
        sha = {}
        manifest["sha256"] = sha
    sha.update(
        {
            "normal_png": sha256_file(result.normal_path),
            "roughness_png": sha256_file(result.roughness_path),
            "ao_png": sha256_file(result.ao_path),
        }
    )
    manifest["pbr"] = {
        "width": result.width,
        "height": result.height,
        "normal_space": "tangent",
        "normal_rgb_flat": [128, 128, 255],
        "roughness_observed": 204,
        "roughness_unknown": 255,
        "ao_unoccluded": 255,
        "metallic_factor": 0.0,
        "observed_texel_fraction": result.observed_texel_fraction,
        "unknown_texel_fraction": result.unknown_texel_fraction,
        "generated_texel_fraction": result.generated_texel_fraction,
        "face_material_detail_applied": result.face_material_detail_applied,
        "face_material_detail_source": (
            "inferred_from_observed_face_feature_anchors_v1"
            if result.face_material_detail_applied
            else "none"
        ),
        "face_material_detail_feature_count": result.face_material_detail_feature_count,
        "face_material_detail_texel_count": result.face_material_detail_texel_count,
        "face_material_detail_normal_variance": result.face_material_detail_normal_variance,
        "inferred_detail_texel_fraction": result.face_material_detail_texel_count
        / max(result.width * result.height, 1),
    }
    result.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest_artifact_path(path: Path, manifest_path: Path) -> str:
    project_root = manifest_path.parent.parent.parent
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_existing_pbr_result(
    *,
    texture_dir: Path,
    manifest_path: Path,
    normal_path: Path,
    roughness_path: Path,
    ao_path: Path,
) -> PbrTextureResult:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("texture_manifest.json must contain an object")
    pbr = manifest.get("pbr", {})
    if not isinstance(pbr, dict):
        raise ValueError("texture_manifest.json is missing pbr metadata")
    return PbrTextureResult(
        texture_dir=texture_dir,
        manifest_path=manifest_path,
        normal_path=normal_path,
        roughness_path=roughness_path,
        ao_path=ao_path,
        width=int(pbr.get("width", 0)),
        height=int(pbr.get("height", 0)),
        observed_texel_fraction=float(pbr.get("observed_texel_fraction", 0.0)),
        unknown_texel_fraction=float(pbr.get("unknown_texel_fraction", 0.0)),
        generated_texel_fraction=float(pbr.get("generated_texel_fraction", 0.0)),
        face_material_detail_applied=bool(pbr.get("face_material_detail_applied", False)),
        face_material_detail_feature_count=int(
            pbr.get("face_material_detail_feature_count", 0)
        ),
        face_material_detail_texel_count=int(pbr.get("face_material_detail_texel_count", 0)),
        face_material_detail_normal_variance=float(
            pbr.get("face_material_detail_normal_variance", 0.0)
        ),
    )


def _apply_face_material_detail(
    normal: np.ndarray,
    roughness: np.ndarray,
    ao: np.ndarray,
    *,
    manifest_path: Path,
) -> dict[str, float | int | bool]:
    manifest = _read_manifest(manifest_path)
    facial_features = manifest.get("facial_features", {})
    if not isinstance(facial_features, dict) or not bool(facial_features.get("applied", False)):
        return _empty_detail_metrics()
    features = facial_features.get("features", [])
    if not isinstance(features, list):
        return _empty_detail_metrics()

    detail_texels = np.zeros(roughness.shape, dtype=np.bool_)
    feature_count = 0
    for item in features:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        uv = _feature_uv(item)
        radius_uv = _feature_radius_uv(item)
        if uv is None or radius_uv is None:
            continue
        profile = _material_profile(name)
        mask, radial_x, radial_y, falloff = _ellipse_detail_fields(
            uv=uv,
            radius_uv=radius_uv,
            shape=roughness.shape,
        )
        if not np.any(mask):
            continue
        feature_count += 1
        detail_texels |= mask
        relief = float(profile["relief"])
        nx = np.zeros(roughness.shape, dtype=np.float64)
        ny = np.zeros(roughness.shape, dtype=np.float64)
        nx[mask] = radial_x[mask] * relief
        ny[mask] = radial_y[mask] * relief
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
        normal[mask, 0] = np.clip(np.rint(128.0 + nx[mask] * 127.0), 0, 255).astype(np.uint8)
        normal[mask, 1] = np.clip(np.rint(128.0 + ny[mask] * 127.0), 0, 255).astype(np.uint8)
        normal[mask, 2] = np.clip(np.rint(128.0 + nz[mask] * 127.0), 0, 255).astype(np.uint8)
        roughness[mask] = np.minimum(roughness[mask], int(profile["roughness"]))
        target_ao = np.clip(
            np.rint(255.0 - float(profile["ao_drop"]) * falloff[mask]),
            0,
            255,
        ).astype(np.uint8)
        ao[mask] = np.minimum(ao[mask], target_ao)

    detail_count = int(np.count_nonzero(detail_texels))
    if detail_count == 0:
        return _empty_detail_metrics()
    changed_normal = normal[detail_texels].astype(np.float64)
    flat = np.asarray([128.0, 128.0, 255.0], dtype=np.float64)
    normal_variance = float(np.mean(np.linalg.norm(changed_normal - flat[None, :], axis=1)))
    return {
        "applied": True,
        "feature_count": feature_count,
        "texel_count": detail_count,
        "normal_variance": round(normal_variance, 6),
    }


def _empty_detail_metrics() -> dict[str, float | int | bool]:
    return {
        "applied": False,
        "feature_count": 0,
        "texel_count": 0,
        "normal_variance": 0.0,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _feature_uv(item: dict[str, Any]) -> np.ndarray | None:
    value = item.get("uv")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return np.asarray([float(value[0]), float(value[1])], dtype=np.float64)


def _feature_radius_uv(item: dict[str, Any]) -> tuple[float, float] | None:
    value = item.get("radius_uv")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    radius_u = max(float(value[0]), 1e-6)
    radius_v = max(float(value[1]), 1e-6)
    return radius_u, radius_v


def _material_profile(name: str) -> dict[str, float | int]:
    if "pupil" in name or "iris" in name or "eye_core" in name:
        return {"relief": 0.10, "roughness": 88, "ao_drop": 10.0}
    if name == "nose":
        return {"relief": 0.14, "roughness": 132, "ao_drop": 18.0}
    if "mouth" in name:
        return {"relief": 0.18, "roughness": 156, "ao_drop": 42.0}
    if "ear_inner" in name:
        return {"relief": 0.11, "roughness": 180, "ao_drop": 28.0}
    if "muzzle" in name:
        return {"relief": 0.07, "roughness": 196, "ao_drop": 12.0}
    return {"relief": 0.06, "roughness": 190, "ao_drop": 8.0}


def _ellipse_detail_fields(
    *,
    uv: np.ndarray,
    radius_uv: tuple[float, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    center_x = float(np.clip(uv[0], 0.0, 1.0) * float(width - 1))
    center_y = float((1.0 - np.clip(uv[1], 0.0, 1.0)) * float(height - 1))
    radius_x = max(float(radius_uv[0]) * float(width - 1), 1.0)
    radius_y = max(float(radius_uv[1]) * float(height - 1), 1.0)
    x0 = max(int(np.floor(center_x - radius_x)), 0)
    x1 = min(int(np.ceil(center_x + radius_x)), width - 1)
    y0 = max(int(np.floor(center_y - radius_y)), 0)
    y1 = min(int(np.ceil(center_y + radius_y)), height - 1)
    mask = np.zeros(shape, dtype=np.bool_)
    radial_x = np.zeros(shape, dtype=np.float64)
    radial_y = np.zeros(shape, dtype=np.float64)
    falloff = np.zeros(shape, dtype=np.float64)
    if x1 < x0 or y1 < y0:
        return mask, radial_x, radial_y, falloff
    xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    local_x = (grid_x - center_x) / radius_x
    local_y = (grid_y - center_y) / radius_y
    radius_sq = local_x * local_x + local_y * local_y
    inside = radius_sq <= 1.0
    local_falloff = np.clip(1.0 - radius_sq, 0.0, 1.0)
    mask[y0 : y1 + 1, x0 : x1 + 1] = inside
    radial_x[y0 : y1 + 1, x0 : x1 + 1] = local_x * local_falloff
    radial_y[y0 : y1 + 1, x0 : x1 + 1] = local_y * local_falloff
    falloff[y0 : y1 + 1, x0 : x1 + 1] = local_falloff
    return mask, radial_x, radial_y, falloff
