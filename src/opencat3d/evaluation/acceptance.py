"""Final synthetic reconstruction acceptance audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opencat3d.evaluation.heldout import SYNTHETIC_HELDOUT_THRESHOLDS
from opencat3d.evaluation.synthetic import SYNTHETIC_METRIC_THRESHOLDS
from opencat3d.fitting.optimizer import load_fit_state

SYNTHETIC_ACCEPTANCE_THRESHOLDS: dict[str, float | int] = {
    "min_num_views": 3,
    "min_segmentation_iou_with_previous": 0.99,
    "min_keypoint_mean_confidence": 0.99,
    "min_keypoint_in_bbox_fraction": 0.75,
    "expected_vertex_count": SYNTHETIC_METRIC_THRESHOLDS["expected_vertex_count"],
    "expected_face_count": SYNTHETIC_METRIC_THRESHOLDS["expected_face_count"],
    "expected_joint_count": SYNTHETIC_METRIC_THRESHOLDS["expected_joint_count"],
    "min_loss_reduction_fraction": SYNTHETIC_METRIC_THRESHOLDS[
        "min_loss_reduction_fraction"
    ],
    "max_final_loss": SYNTHETIC_METRIC_THRESHOLDS["max_final_loss"],
    "max_final_beta_error": SYNTHETIC_METRIC_THRESHOLDS["max_final_beta_error"],
    "min_beta_error_reduction_fraction": SYNTHETIC_METRIC_THRESHOLDS[
        "min_beta_error_reduction_fraction"
    ],
    "min_heldout_silhouette_iou": SYNTHETIC_HELDOUT_THRESHOLDS[
        "min_heldout_silhouette_iou"
    ],
    "max_heldout_keypoint_reprojection_loss": SYNTHETIC_HELDOUT_THRESHOLDS[
        "max_heldout_keypoint_reprojection_loss"
    ],
}


@dataclass(frozen=True)
class SyntheticAcceptanceAuditResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


def evaluate_synthetic_acceptance(
    project_dir: Path,
    *,
    force: bool = False,
) -> SyntheticAcceptanceAuditResult:
    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / "synthetic_acceptance_audit.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    observation_manifest_path = project_dir / "observations" / "synthetic" / "manifest.json"
    init_manifest_path = project_dir / "init" / "synthetic" / "manifest.json"
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"
    mesh_validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    evaluation_path = output_dir / "synthetic_evaluation.json"
    heldout_path = output_dir / "synthetic_heldout_evaluation.json"
    validation_path = output_dir / "validation.json"
    obj_path = project_dir / "outputs" / "avatar" / "cat_canonical.obj"
    ply_path = project_dir / "outputs" / "avatar" / "cat_canonical.ply"
    glb_path = project_dir / "outputs" / "avatar" / "cat_avatar.glb"
    idle_animation_path = project_dir / "outputs" / "avatar" / "animations" / "idle.glb"
    skeleton_path = project_dir / "outputs" / "avatar" / "skeleton.json"
    joint_projection_path = output_dir / "synthetic_joint_projection.json"
    residual_path = output_dir / "synthetic_silhouette_residuals.json"
    basecolor_path = project_dir / "outputs" / "textures" / "basecolor.png"
    basecolor_confidence_path = project_dir / "outputs" / "textures" / "basecolor_confidence.png"
    normal_path = project_dir / "outputs" / "textures" / "normal.png"
    roughness_path = project_dir / "outputs" / "textures" / "roughness.png"
    ao_path = project_dir / "outputs" / "textures" / "ao.png"

    observation_manifest = _read_json(observation_manifest_path)
    init_manifest = _read_json(init_manifest_path)
    fit_report = _read_json(fit_report_path)
    mesh_validation = _read_json(mesh_validation_path)
    evaluation = _read_json(evaluation_path)
    heldout = _read_json(heldout_path)

    thresholds = dict(SYNTHETIC_ACCEPTANCE_THRESHOLDS)
    frame_count = _frame_count(observation_manifest)
    segmentation_ious = _segmentation_ious(observation_manifest)
    min_segmentation_iou = min(segmentation_ious) if segmentation_ious else 0.0
    mean_segmentation_iou = float(np.mean(segmentation_ious)) if segmentation_ious else 0.0
    keypoint_mean_confidence = float(init_manifest.get("mean_keypoint_confidence", 0.0))
    keypoint_mean_in_bbox_fraction = float(init_manifest.get("mean_in_bbox_fraction", 0.0))

    fit_metrics = _object(fit_report.get("metrics"), "fit_report.metrics")
    eval_checks = _bool_map(evaluation.get("checks"), "synthetic_evaluation.checks")
    heldout_metrics = _object(heldout.get("metrics"), "synthetic_heldout_evaluation.metrics")
    vertex_count = int(mesh_validation.get("vertex_count", -1))
    face_count = int(mesh_validation.get("face_count", -1))
    joint_count = int(mesh_validation.get("joint_count", -1))
    fit_state_finite, fit_state_path, fit_state_source_format = _fit_state_valid(
        project_dir,
        frame_count=frame_count,
        joint_count=joint_count,
    )
    loss_history_finite = _loss_history_finite(fit_report.get("loss_history", {}))

    initial_loss = float(fit_metrics["initial_loss"])
    final_loss = float(fit_metrics["final_loss"])
    initial_keypoint_loss = float(fit_metrics["initial_keypoint_loss"])
    final_keypoint_loss = float(fit_metrics["final_keypoint_loss"])
    initial_silhouette_loss = float(fit_metrics["initial_silhouette_loss"])
    final_silhouette_loss = float(fit_metrics["final_silhouette_loss"])
    initial_beta_error = float(fit_metrics["initial_beta_error"])
    final_beta_error = float(fit_metrics["final_beta_error"])
    heldout_silhouette_iou = float(heldout_metrics["heldout_mean_silhouette_iou"])
    heldout_keypoint_loss = float(heldout_metrics["heldout_mean_keypoint_reprojection_loss"])

    checks = {
        "synthetic_multi_view_reconstruction": (
            bool(evaluation.get("success"))
            and bool(heldout.get("success"))
            and frame_count >= int(thresholds["min_num_views"])
            and int(fit_report.get("num_frames", 0)) == frame_count
        ),
        "segmentation_outputs_present": (
            len(segmentation_ious) == frame_count
            and min_segmentation_iou
            >= float(thresholds["min_segmentation_iou_with_previous"])
        ),
        "keypoint_pose_initialization_present": (
            int(init_manifest.get("frame_count", 0)) == frame_count
            and keypoint_mean_confidence
            >= float(thresholds["min_keypoint_mean_confidence"])
            and keypoint_mean_in_bbox_fraction
            >= float(thresholds["min_keypoint_in_bbox_fraction"])
            and float(fit_metrics.get("init_prediction_used", 0.0)) == 1.0
        ),
        "fixed_topology_cat_template": (
            vertex_count == int(thresholds["expected_vertex_count"])
            and face_count == int(thresholds["expected_face_count"])
            and bool(mesh_validation.get("finite_vertices"))
            and bool(mesh_validation.get("valid_faces"))
            and eval_checks.get("template_vertex_count", False)
            and eval_checks.get("template_face_count", False)
        ),
        "skeleton_metadata_export": (
            skeleton_path.exists()
            and joint_count == int(thresholds["expected_joint_count"])
            and bool(mesh_validation.get("skeleton_metadata_valid"))
            and eval_checks.get("skeleton_exists", False)
            and eval_checks.get("skeleton_metadata_valid", False)
            and eval_checks.get("skeleton_skin_weights_valid", False)
        ),
        "shared_identity_across_views": (
            frame_count >= int(thresholds["min_num_views"])
            and final_beta_error <= initial_beta_error + 1e-12
            and final_beta_error <= float(thresholds["max_final_beta_error"])
            and float(fit_metrics["beta_error_reduction_fraction"])
            >= float(thresholds["min_beta_error_reduction_fraction"])
        ),
        "optimizer_converged_without_nan": (
            bool(fit_report.get("success"))
            and _is_finite(initial_loss)
            and _is_finite(final_loss)
            and final_loss < initial_loss
            and final_loss <= float(thresholds["max_final_loss"])
            and float(fit_metrics["loss_reduction_fraction"])
            >= float(thresholds["min_loss_reduction_fraction"])
            and fit_state_finite
            and loss_history_finite
        ),
        "component_losses_reduce": (
            final_keypoint_loss < initial_keypoint_loss
            and final_silhouette_loss < initial_silhouette_loss
            and eval_checks.get("keypoint_loss_reduced", False)
            and eval_checks.get("silhouette_loss_reduced", False)
        ),
        "canonical_mesh_exports_valid": (
            obj_path.exists()
            and ply_path.exists()
            and glb_path.exists()
            and eval_checks.get("obj_exists", False)
            and eval_checks.get("ply_exists", False)
            and eval_checks.get("glb_exists", False)
            and eval_checks.get("obj_ply_reload_match", False)
        ),
        "glb_export_valid": (
            glb_path.exists()
            and eval_checks.get("glb_metadata_valid", False)
            and eval_checks.get("glb_structure_valid", False)
            and eval_checks.get("glb_basecolor_texture_valid", False)
            and eval_checks.get("glb_pbr_textures_valid", False)
            and eval_checks.get("glb_skinning_valid", False)
            and eval_checks.get("glb_idle_animation_valid", False)
        ),
        "basic_animation_export_valid": (
            idle_animation_path.exists()
            and eval_checks.get("glb_idle_animation_valid", False)
        ),
        "per_frame_overlays_present": eval_checks.get("fit_overlays_present", False),
        "coverage_report_valid": eval_checks.get("coverage_report_valid", False),
        "observed_basecolor_texture_valid": (
            basecolor_path.exists()
            and basecolor_confidence_path.exists()
            and eval_checks.get("texture_basecolor_valid", False)
        ),
        "pbr_texture_outputs_valid": (
            normal_path.exists()
            and roughness_path.exists()
            and ao_path.exists()
            and eval_checks.get("pbr_texture_outputs_valid", False)
        ),
        "joint_projection_report_valid": (
            joint_projection_path.exists()
            and eval_checks.get("joint_projection_report_valid", False)
        ),
        "silhouette_residual_maps_valid": (
            residual_path.exists()
            and eval_checks.get("silhouette_residual_maps_valid", False)
        ),
        "artifact_validation_report_valid": (
            validation_path.exists()
            and eval_checks.get("artifact_validation_report_valid", False)
        ),
        "heldout_view_consistency": (
            bool(heldout.get("success"))
            and eval_checks.get("heldout_view_consistency", False)
            and heldout_silhouette_iou >= float(thresholds["min_heldout_silhouette_iou"])
            and heldout_keypoint_loss
            <= float(thresholds["max_heldout_keypoint_reprojection_loss"])
        ),
    }
    passed_check_count = sum(1 for passed in checks.values() if passed)
    metrics: dict[str, float | int] = {
        "frame_count": frame_count,
        "segmentation_mean_iou_with_previous_masks": mean_segmentation_iou,
        "segmentation_min_iou_with_previous_masks": min_segmentation_iou,
        "keypoint_mean_confidence": keypoint_mean_confidence,
        "keypoint_mean_in_bbox_fraction": keypoint_mean_in_bbox_fraction,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_fraction": float(fit_metrics["loss_reduction_fraction"]),
        "initial_keypoint_loss": initial_keypoint_loss,
        "final_keypoint_loss": final_keypoint_loss,
        "initial_silhouette_loss": initial_silhouette_loss,
        "final_silhouette_loss": final_silhouette_loss,
        "initial_beta_error": initial_beta_error,
        "final_beta_error": final_beta_error,
        "beta_error_reduction_fraction": float(
            fit_metrics["beta_error_reduction_fraction"]
        ),
        "heldout_mean_silhouette_iou": heldout_silhouette_iou,
        "heldout_mean_keypoint_reprojection_loss": heldout_keypoint_loss,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "joint_count": joint_count,
        "passed_check_count": passed_check_count,
        "total_check_count": len(checks),
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
            "observation_manifest": str(observation_manifest_path),
            "init_manifest": str(init_manifest_path),
            "fit_report": str(fit_report_path),
            "fit_state": str(fit_state_path),
            "fit_state_source_format": fit_state_source_format,
            "mesh_validation": str(mesh_validation_path),
            "synthetic_evaluation": str(evaluation_path),
            "heldout_evaluation": str(heldout_path),
            "validation_report": str(validation_path),
            "obj": str(obj_path),
            "ply": str(ply_path),
            "glb": str(glb_path),
            "idle_animation": str(idle_animation_path),
            "skeleton": str(skeleton_path),
            "basecolor": str(basecolor_path),
            "basecolor_confidence": str(basecolor_confidence_path),
            "normal": str(normal_path),
            "roughness": str(roughness_path),
            "ao": str(ao_path),
            "joint_projection": str(joint_projection_path),
            "silhouette_residuals": str(residual_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SyntheticAcceptanceAuditResult(
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


def _frame_count(observation_manifest: dict[str, Any]) -> int:
    frames = observation_manifest.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("synthetic observation manifest frames must be a list")
    return len(frames)


def _segmentation_ious(observation_manifest: dict[str, Any]) -> list[float]:
    ious: list[float] = []
    frames = observation_manifest.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("synthetic observation manifest frames must be a list")
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("synthetic observation manifest frame must be an object")
        quality = frame.get("quality", {})
        if not isinstance(quality, dict):
            raise ValueError("synthetic observation frame quality must be an object")
        if float(quality.get("rgb_segmentation", 0.0)) != 1.0:
            continue
        ious.append(float(quality.get("mask_iou_with_previous", 0.0)))
    return ious


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _bool_map(value: object, name: str) -> dict[str, bool]:
    raw = _object(value, name)
    return {str(key): bool(item) for key, item in raw.items()}


def _fit_state_valid(
    project_dir: Path,
    *,
    frame_count: int,
    joint_count: int,
) -> tuple[bool, Path, str]:
    default_path = project_dir / "outputs" / "fitting" / "fit_state.safetensors"
    try:
        fit_state = load_fit_state(
            project_dir,
            mode="synthetic",
            expected_frame_count=frame_count,
            expected_joint_count=joint_count,
        )
    except (FileNotFoundError, ValueError):
        return False, default_path, "missing_or_invalid"
    arrays = [
        fit_state.beta,
        fit_state.target_beta,
        fit_state.initial_beta,
        fit_state.root_translations,
        fit_state.initial_root_translations,
        fit_state.poses_6d,
        fit_state.initial_poses_6d,
    ]
    valid = all(np.asarray(array).size > 0 and np.all(np.isfinite(array)) for array in arrays)
    return valid, fit_state.path, fit_state.source_format


def _loss_history_finite(loss_history: object) -> bool:
    if not isinstance(loss_history, dict):
        return False
    for values in loss_history.values():
        if not isinstance(values, list):
            return False
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            return False
    return True


def _is_finite(value: float) -> bool:
    return value not in {float("inf"), float("-inf")} and value == value
