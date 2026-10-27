"""Synthetic reconstruction evaluation report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opencat3d.evaluation.coverage import evaluate_mesh_coverage
from opencat3d.evaluation.heldout import evaluate_synthetic_heldout_views
from opencat3d.evaluation.projection import evaluate_joint_projection_report
from opencat3d.evaluation.residuals import evaluate_silhouette_residual_maps
from opencat3d.evaluation.validation import write_quality_validation_report
from opencat3d.export.mesh import (
    read_glb_metadata,
    read_obj,
    read_ply,
    validate_glb_structure,
    validate_reloaded_mesh,
)
from opencat3d.export.skeleton import validate_skeleton_metadata_file
from opencat3d.template.cat_template import CAT_TEMPLATE_FACE_COUNT, CAT_TEMPLATE_VERTEX_COUNT

SYNTHETIC_METRIC_THRESHOLDS: dict[str, float | int] = {
    "min_loss_reduction_fraction": 0.80,
    "max_final_loss": 1.0,
    "max_final_beta_error": 0.12,
    "min_beta_error_reduction_fraction": 0.10,
    "expected_vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
    "expected_face_count": CAT_TEMPLATE_FACE_COUNT,
    "expected_joint_count": 11,
}


@dataclass(frozen=True)
class SyntheticEvaluationResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


def evaluate_synthetic_reconstruction(
    project_dir: Path,
    *,
    force: bool = False,
) -> SyntheticEvaluationResult:
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"
    mesh_validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    obj_path = project_dir / "outputs" / "avatar" / "cat_canonical.obj"
    ply_path = project_dir / "outputs" / "avatar" / "cat_canonical.ply"
    glb_path = project_dir / "outputs" / "avatar" / "cat_avatar.glb"
    skeleton_path = project_dir / "outputs" / "avatar" / "skeleton.json"
    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / "synthetic_evaluation.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    fit_report = _read_json(fit_report_path)
    mesh_validation = _read_json(mesh_validation_path)
    heldout_result = evaluate_synthetic_heldout_views(project_dir, force=force)
    coverage_result = evaluate_mesh_coverage(project_dir, mode="synthetic", force=force)
    projection_result = evaluate_joint_projection_report(
        project_dir,
        mode="synthetic",
        force=force,
    )
    residual_result = evaluate_silhouette_residual_maps(
        project_dir,
        mode="synthetic",
        force=force,
    )
    validation_result = write_quality_validation_report(
        project_dir,
        mode="synthetic",
        force=force,
    )
    output_paths = _as_object(fit_report.get("output_paths"), "fit_report.output_paths")
    overlay_manifest_path = Path(
        str(
            output_paths.get(
                "overlay_manifest",
                project_dir / "outputs" / "fitting" / "per_frame_overlay" / "manifest.json",
            )
        )
    )
    initial_loss = float(fit_report["metrics"]["initial_loss"])
    final_loss = float(fit_report["metrics"]["final_loss"])
    initial_keypoint_loss = float(fit_report["metrics"]["initial_keypoint_loss"])
    final_keypoint_loss = float(fit_report["metrics"]["final_keypoint_loss"])
    initial_silhouette_loss = float(fit_report["metrics"]["initial_silhouette_loss"])
    final_silhouette_loss = float(fit_report["metrics"]["final_silhouette_loss"])
    loss_reduction = initial_loss - final_loss
    loss_reduction_fraction = loss_reduction / max(initial_loss, 1e-12)
    initial_beta_error = float(fit_report["metrics"].get("initial_beta_error", 0.0))
    final_beta_error = float(fit_report["metrics"].get("final_beta_error", 0.0))
    beta_error_reduction = initial_beta_error - final_beta_error
    beta_error_reduction_fraction = beta_error_reduction / max(initial_beta_error, 1e-12)

    obj_mesh = read_obj(obj_path)
    ply_mesh = read_ply(ply_path)
    validate_reloaded_mesh(obj_mesh, ply_mesh)
    glb_metadata = read_glb_metadata(glb_path)
    vertex_count = int(mesh_validation["vertex_count"])
    face_count = int(mesh_validation["face_count"])
    thresholds = dict(SYNTHETIC_METRIC_THRESHOLDS)
    glb_structure = validate_glb_structure(
        glb_path,
        expected_vertex_count=vertex_count,
        expected_face_count=face_count,
        expected_textured=True,
        expected_pbr_textures=True,
        expected_skinned=True,
        expected_animated=True,
    )
    skeleton_report = validate_skeleton_metadata_file(
        skeleton_path,
        expected_mode="synthetic",
        expected_vertex_count=vertex_count,
        expected_face_count=face_count,
        expected_joint_count=int(thresholds["expected_joint_count"]),
    )
    overlay_valid, overlay_mean_silhouette_iou = _overlay_manifest_valid(
        overlay_manifest_path,
        expected_frame_count=int(fit_report["num_frames"]),
    )

    checks = {
        "fit_report_success": bool(fit_report["success"]),
        "loss_reduced": final_loss < initial_loss,
        "keypoint_loss_reduced": final_keypoint_loss < initial_keypoint_loss,
        "silhouette_loss_reduced": final_silhouette_loss < initial_silhouette_loss,
        "loss_finite": _is_finite(initial_loss) and _is_finite(final_loss),
        "loss_reduction_fraction_threshold": (
            loss_reduction_fraction >= float(thresholds["min_loss_reduction_fraction"])
        ),
        "final_loss_threshold": final_loss <= float(thresholds["max_final_loss"]),
        "beta_error_not_increased": final_beta_error <= initial_beta_error + 1e-12,
        "final_beta_error_threshold": (
            final_beta_error <= float(thresholds["max_final_beta_error"])
        ),
        "beta_error_reduction_fraction_threshold": (
            beta_error_reduction_fraction
            >= float(thresholds["min_beta_error_reduction_fraction"])
        ),
        "obj_exists": obj_path.exists(),
        "ply_exists": ply_path.exists(),
        "glb_exists": glb_path.exists(),
        "skeleton_exists": skeleton_path.exists(),
        "obj_ply_reload_match": True,
        "glb_metadata_valid": (
            glb_metadata["vertex_count"] == vertex_count
            and glb_metadata["face_count"] == face_count
        ),
        "glb_structure_valid": glb_structure.success
        and bool(mesh_validation.get("glb_structure_valid", False)),
        "glb_basecolor_texture_valid": bool(
            mesh_validation.get("glb_basecolor_texture_valid", False)
            and glb_structure.checks.get("expected_textured", False)
            and glb_structure.checks.get("texture_material_valid", False)
        ),
        "glb_pbr_textures_valid": bool(
            mesh_validation.get("glb_pbr_textures_valid", False)
            and glb_structure.checks.get("expected_pbr_textures", False)
            and glb_structure.checks.get("pbr_material_textures_valid", False)
            and glb_structure.checks.get("pbr_metallic_factor_zero", False)
        ),
        "glb_skinning_valid": bool(
            mesh_validation.get("glb_skinning_valid", False)
            and glb_structure.checks.get("expected_skinned", False)
            and glb_structure.checks.get("skin_valid", False)
            and glb_structure.checks.get("weights_normalized", False)
        ),
        "glb_idle_animation_valid": bool(
            mesh_validation.get("glb_idle_animation_valid", False)
            and glb_structure.checks.get("expected_animated", False)
            and glb_structure.checks.get("animation_valid", False)
            and glb_structure.checks.get("animation_rotations_normalized", False)
        ),
        "template_vertex_count": vertex_count == int(thresholds["expected_vertex_count"]),
        "template_face_count": face_count == int(thresholds["expected_face_count"]),
        "mesh_vertices_finite": bool(mesh_validation["finite_vertices"]),
        "mesh_faces_valid": bool(mesh_validation["valid_faces"]),
        "skeleton_metadata_valid": skeleton_report.success
        and bool(mesh_validation.get("skeleton_metadata_valid", False)),
        "skeleton_skin_weights_valid": bool(
            skeleton_report.checks.get("skin_weight_row_sums_valid", False)
            and skeleton_report.checks.get("skin_weight_indices_valid", False)
            and skeleton_report.checks.get("skin_weight_values_valid", False)
        ),
        "fit_overlays_present": overlay_valid,
        "coverage_report_valid": coverage_result.success,
        "texture_basecolor_valid": bool(
            coverage_result.checks.get("texture_basecolor_written", False)
            and float(coverage_result.metrics["basecolor_observed_texel_fraction"]) > 0.0
            and float(coverage_result.metrics["basecolor_generated_texel_fraction"]) == 0.0
        ),
        "pbr_texture_outputs_valid": bool(
            coverage_result.checks.get("pbr_textures_written", False)
            and float(coverage_result.metrics["pbr_observed_texel_fraction"]) > 0.0
            and float(coverage_result.metrics["pbr_generated_texel_fraction"]) == 0.0
        ),
        "heldout_view_consistency": heldout_result.success,
        "joint_projection_report_valid": projection_result.success,
        "silhouette_residual_maps_valid": residual_result.success,
        "artifact_validation_report_valid": validation_result.success,
    }
    metrics: dict[str, float | int] = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "loss_reduction_fraction": loss_reduction_fraction,
        "initial_keypoint_loss": initial_keypoint_loss,
        "final_keypoint_loss": final_keypoint_loss,
        "keypoint_loss_reduction": initial_keypoint_loss - final_keypoint_loss,
        "initial_silhouette_loss": initial_silhouette_loss,
        "final_silhouette_loss": final_silhouette_loss,
        "silhouette_loss_reduction": initial_silhouette_loss - final_silhouette_loss,
        "initial_beta_error": initial_beta_error,
        "final_beta_error": final_beta_error,
        "beta_error_reduction": beta_error_reduction,
        "beta_error_reduction_fraction": beta_error_reduction_fraction,
        "heldout_mean_silhouette_iou": float(
            heldout_result.metrics["heldout_mean_silhouette_iou"]
        ),
        "heldout_mean_keypoint_reprojection_loss": float(
            heldout_result.metrics["heldout_mean_keypoint_reprojection_loss"]
        ),
        "heldout_frame_count": int(heldout_result.metrics["heldout_frame_count"]),
        "overlay_mean_silhouette_iou": overlay_mean_silhouette_iou,
        "observed_face_fraction": float(coverage_result.metrics["observed_face_fraction"]),
        "observed_vertex_fraction": float(
            coverage_result.metrics["observed_vertex_fraction"]
        ),
        "generated_face_count": int(coverage_result.metrics["generated_face_count"]),
        "basecolor_observed_texel_fraction": float(
            coverage_result.metrics["basecolor_observed_texel_fraction"]
        ),
        "basecolor_generated_texel_fraction": float(
            coverage_result.metrics["basecolor_generated_texel_fraction"]
        ),
        "pbr_observed_texel_fraction": float(
            coverage_result.metrics["pbr_observed_texel_fraction"]
        ),
        "pbr_generated_texel_fraction": float(
            coverage_result.metrics["pbr_generated_texel_fraction"]
        ),
        "joint_projection_mean_keypoint_loss": float(
            projection_result.metrics["mean_keypoint_reprojection_loss"]
        ),
        "joint_projection_mean_pixel_error": float(
            projection_result.metrics["mean_pixel_error"]
        ),
        "silhouette_residual_mean_union_error": float(
            residual_result.metrics["mean_union_error_fraction"]
        ),
        "silhouette_residual_mean_false_negative": float(
            residual_result.metrics["mean_false_negative_fraction"]
        ),
        "silhouette_residual_mean_false_positive": float(
            residual_result.metrics["mean_false_positive_fraction"]
        ),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "joint_count": skeleton_report.joint_count,
    }
    success = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "success": success,
        "checks": checks,
        "thresholds": thresholds,
        "metrics": metrics,
        "artifacts": {
            "fit_report": str(fit_report_path),
            "fit_state_safetensors": str(
                project_dir / "outputs" / "fitting" / "fit_state.safetensors"
            ),
            "mesh_validation": str(mesh_validation_path),
            "obj": str(obj_path),
            "ply": str(ply_path),
            "glb": str(glb_path),
            "idle_animation": str(
                project_dir / "outputs" / "avatar" / "animations" / "idle.glb"
            ),
            "skeleton": str(skeleton_path),
            "overlay_manifest": str(overlay_manifest_path),
            "coverage_report": str(coverage_result.report_path),
            "coverage_provenance": str(coverage_result.provenance_path),
            "texture_manifest": str(coverage_result.texture_manifest_path),
            "basecolor": str(project_dir / "outputs" / "textures" / "basecolor.png"),
            "basecolor_confidence": str(
                project_dir / "outputs" / "textures" / "basecolor_confidence.png"
            ),
            "source_view_id": str(project_dir / "outputs" / "textures" / "source_view_id.png"),
            "normal": str(project_dir / "outputs" / "textures" / "normal.png"),
            "roughness": str(project_dir / "outputs" / "textures" / "roughness.png"),
            "ao": str(project_dir / "outputs" / "textures" / "ao.png"),
            "heldout_evaluation": str(heldout_result.report_path),
            "joint_projection": str(projection_result.report_path),
            "silhouette_residuals": str(residual_result.report_path),
            "silhouette_residual_manifest": str(residual_result.manifest_path),
            "validation_report": str(validation_result.report_path),
        },
        "glb_structure_checks": glb_structure.checks,
        "skeleton_checks": skeleton_report.checks,
        "joint_projection_checks": projection_result.checks,
        "silhouette_residual_checks": residual_result.checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SyntheticEvaluationResult(
        report_path=report_path,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _as_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _overlay_manifest_valid(path: Path, *, expected_frame_count: int) -> tuple[bool, float]:
    if not path.exists():
        return False, 0.0
    try:
        payload = _read_json(path)
        frames = payload.get("frames", [])
        if not isinstance(frames, list):
            return False, 0.0
        overlays_exist = all(
            isinstance(frame, dict) and Path(str(frame.get("overlay_path", ""))).exists()
            for frame in frames
        )
        frame_count = int(payload.get("frame_count", -1))
        mean_iou = float(payload.get("mean_silhouette_iou", 0.0))
    except (OSError, ValueError, TypeError):
        return False, 0.0
    valid = (
        frame_count == expected_frame_count
        and len(frames) == expected_frame_count
        and overlays_exist
    )
    return valid, mean_iou


def _is_finite(value: float) -> bool:
    return value not in {float("inf"), float("-inf")} and value == value
