"""Deterministic artifact validation audit report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opencat3d.export.mesh import validate_glb_structure
from opencat3d.export.skeleton import validate_skeleton_metadata_file
from opencat3d.fitting.optimizer import validate_fit_state_safetensors
from opencat3d.io.cache import sha256_file


@dataclass(frozen=True)
class ArtifactValidationResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]


def write_quality_validation_report(
    project_dir: Path,
    *,
    mode: str,
    force: bool = False,
) -> ArtifactValidationResult:
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported validation mode: {mode}")
    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / "validation.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    mesh_validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"
    mesh_validation = _read_json(mesh_validation_path)
    fit_report = _read_json(fit_report_path)
    vertex_count = int(mesh_validation["vertex_count"])
    face_count = int(mesh_validation["face_count"])
    joint_count = int(mesh_validation["joint_count"])

    required_paths = _required_artifacts(project_dir, mode=mode)
    glb_path = required_paths["glb"]
    skeleton_path = required_paths["skeleton"]
    texture_manifest_path = required_paths["texture_manifest"]
    texture_manifest = _read_json(texture_manifest_path)
    basecolor = texture_manifest.get("basecolor", {})
    expected_texture_size = 0
    if isinstance(basecolor, dict):
        expected_texture_size = int(basecolor.get("width", 0))
    glb_structure = validate_glb_structure(
        glb_path,
        expected_vertex_count=vertex_count,
        expected_face_count=face_count,
        expected_textured=True,
        expected_texture_size=expected_texture_size if expected_texture_size > 0 else None,
        expected_pbr_textures=True,
        expected_skinned=True,
        expected_animated=True,
    )
    skeleton_report = validate_skeleton_metadata_file(
        skeleton_path,
        expected_mode=mode,
        expected_vertex_count=vertex_count,
        expected_face_count=face_count,
        expected_joint_count=joint_count,
    )
    fit_state_report = validate_fit_state_safetensors(
        required_paths["fit_state_safetensors"],
        expected_mode=mode,
        expected_frame_count=int(fit_report["num_frames"]),
        expected_joint_count=joint_count,
    )
    texture_checks = _texture_manifest_checks(texture_manifest, project_dir)
    mesh_checks = {
        "mesh_validation_exists": mesh_validation_path.exists(),
        "finite_vertices": bool(mesh_validation.get("finite_vertices", False)),
        "valid_faces": bool(mesh_validation.get("valid_faces", False)),
        "skeleton_metadata_valid": bool(
            mesh_validation.get("skeleton_metadata_valid", False)
        ),
        "glb_structure_valid": bool(mesh_validation.get("glb_structure_valid", False)),
        "glb_basecolor_texture_valid": bool(
            mesh_validation.get("glb_basecolor_texture_valid", False)
        ),
        "glb_pbr_textures_valid": bool(mesh_validation.get("glb_pbr_textures_valid", False)),
        "glb_face_detail_geometry_valid": (
            mode != "raw"
            or bool(mesh_validation.get("glb_face_detail_geometry_valid", False))
        ),
        "glb_skinning_valid": bool(mesh_validation.get("glb_skinning_valid", False)),
        "glb_idle_animation_valid": bool(
            mesh_validation.get("glb_idle_animation_valid", False)
        ),
        "pbr_textures_valid": bool(mesh_validation.get("pbr_textures_valid", False)),
    }
    artifact_checks = {
        f"{name}_exists": path.exists() for name, path in required_paths.items()
    }
    checks = {
        **artifact_checks,
        **{f"mesh_{name}": value for name, value in mesh_checks.items()},
        "glb_structure_valid": glb_structure.success,
        "skeleton_metadata_valid": skeleton_report.success,
        "fit_state_safetensors_valid": fit_state_report.success,
        **{f"texture_{name}": value for name, value in texture_checks.items()},
    }
    success = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "success": success,
        "checks": checks,
        "artifacts": {name: str(path) for name, path in sorted(required_paths.items())},
        "sha256": {
            name: sha256_file(path)
            for name, path in sorted(required_paths.items())
            if path.exists() and path.is_file()
        },
        "glb_structure_checks": glb_structure.checks,
        "skeleton_checks": skeleton_report.checks,
        "fit_state_checks": fit_state_report.checks,
        "fit_state_metadata": fit_state_report.metadata,
        "mesh_validation_checks": mesh_checks,
        "texture_manifest_checks": texture_checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ArtifactValidationResult(report_path=report_path, success=success, checks=checks)


def _required_artifacts(project_dir: Path, *, mode: str) -> dict[str, Path]:
    quality_dir = project_dir / "outputs" / "quality"
    texture_dir = project_dir / "outputs" / "textures"
    avatar_dir = project_dir / "outputs" / "avatar"
    return {
        "ao": texture_dir / "ao.png",
        "basecolor": texture_dir / "basecolor.png",
        "basecolor_confidence": texture_dir / "basecolor_confidence.png",
        "coverage_provenance": quality_dir / "coverage_provenance.npz",
        "coverage_report": quality_dir / "coverage.json",
        "fit_report": project_dir / "outputs" / "fitting" / "fit_report.json",
        "fit_state_safetensors": (
            project_dir / "outputs" / "fitting" / "fit_state.safetensors"
        ),
        "glb": avatar_dir / "cat_avatar.glb",
        "idle_animation": avatar_dir / "animations" / "idle.glb",
        "joint_projection": quality_dir / f"{mode}_joint_projection.json",
        "mesh_validation": avatar_dir / "mesh_validation.json",
        "normal": texture_dir / "normal.png",
        "obj": avatar_dir / "cat_canonical.obj",
        "ply": avatar_dir / "cat_canonical.ply",
        "provenance": texture_dir / "provenance.png",
        "provenance_confidence": texture_dir / "confidence.png",
        "roughness": texture_dir / "roughness.png",
        "silhouette_residuals": quality_dir / f"{mode}_silhouette_residuals.json",
        "silhouette_residual_manifest": (
            quality_dir / f"{mode}_silhouette_residuals" / "manifest.json"
        ),
        "skeleton": avatar_dir / "skeleton.json",
        "source_view_id": texture_dir / "source_view_id.png",
        "texture_manifest": texture_dir / "texture_manifest.json",
    }


def _texture_manifest_checks(
    manifest: dict[str, Any],
    project_dir: Path,
) -> dict[str, bool]:
    artifacts = manifest.get("artifacts", {})
    sha = manifest.get("sha256", {})
    basecolor = manifest.get("basecolor", {})
    pbr = manifest.get("pbr", {})
    return {
        "basecolor_source_observed": (
            manifest.get("basecolor_source") == "opencat3d.observed_basecolor_projection"
        ),
        "pbr_source_neutral": manifest.get("pbr_source") == "opencat3d.neutral_pbr_priors",
        "basecolor_generated_fraction_zero": _float_field(
            basecolor,
            "generated_texel_fraction",
            default=1.0,
        )
        == 0.0,
        "pbr_generated_fraction_zero": _float_field(
            pbr,
            "generated_texel_fraction",
            default=1.0,
        )
        == 0.0,
        "pbr_metallic_zero": _float_field(pbr, "metallic_factor", default=1.0) == 0.0,
        "artifact_paths_valid": _manifest_artifact_paths_valid(artifacts, project_dir),
        "artifact_hashes_valid": _manifest_hashes_valid(artifacts, sha, project_dir),
    }


def _manifest_artifact_paths_valid(artifacts: object, project_dir: Path) -> bool:
    if not isinstance(artifacts, dict):
        return False
    required = {
        "ao_png",
        "basecolor_confidence_png",
        "basecolor_png",
        "confidence_png",
        "normal_png",
        "provenance_png",
        "roughness_png",
        "source_view_id_png",
    }
    if not required.issubset(set(artifacts)):
        return False
    for key in required:
        path = Path(str(artifacts[key]))
        if not path.is_absolute():
            path = project_dir / path
        if not path.exists():
            return False
    return True


def _manifest_hashes_valid(
    artifacts: object,
    sha: object,
    project_dir: Path,
) -> bool:
    if not isinstance(artifacts, dict) or not isinstance(sha, dict):
        return False
    for key, raw_path in artifacts.items():
        if key not in sha:
            return False
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = project_dir / path
        if not path.exists() or sha256_file(path) != str(sha[key]):
            return False
    return True


def _float_field(value: object, key: str, *, default: float) -> float:
    if not isinstance(value, dict):
        return default
    try:
        return float(value.get(key, default))
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data
