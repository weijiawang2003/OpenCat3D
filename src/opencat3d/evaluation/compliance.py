"""Objective-level OpenCat3D compliance audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opencat3d.template.cat_template import CAT_TEMPLATE_FACE_COUNT, CAT_TEMPLATE_VERTEX_COUNT


@dataclass(frozen=True)
class ObjectiveComplianceResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]


def write_objective_compliance_audit(
    project_dir: Path,
    *,
    force: bool = False,
) -> ObjectiveComplianceResult:
    """Write a compact requirement-to-evidence audit for a synthetic pipeline run."""

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / "opencat3d_compliance_audit.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    summary_path = output_dir / "synthetic_pipeline_summary.json"
    acceptance_path = output_dir / "synthetic_acceptance_audit.json"
    evaluation_path = output_dir / "synthetic_evaluation.json"
    validation_path = output_dir / "validation.json"
    mesh_validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"

    summary = _read_json(summary_path)
    acceptance = _read_json(acceptance_path)
    evaluation = _read_json(evaluation_path)
    validation = _read_json(validation_path)
    mesh_validation = _read_json(mesh_validation_path)
    fit_report = _read_json(fit_report_path)

    summary_steps = _object(summary.get("steps"), "synthetic_pipeline_summary.steps")
    summary_metrics = _object(
        summary.get("metrics"),
        "synthetic_pipeline_summary.metrics",
    )
    summary_artifacts = _object(
        summary.get("artifacts"),
        "synthetic_pipeline_summary.artifacts",
    )
    acceptance_checks = _bool_map(
        acceptance.get("checks"),
        "synthetic_acceptance_audit.checks",
    )
    acceptance_metrics = _object(
        acceptance.get("metrics"),
        "synthetic_acceptance_audit.metrics",
    )
    evaluation_checks = _bool_map(evaluation.get("checks"), "synthetic_evaluation.checks")
    validation_checks = _bool_map(validation.get("checks"), "validation.checks")
    fit_metrics = _object(fit_report.get("metrics"), "fit_report.metrics")

    generated_free_texture = (
        _float(summary_metrics, "basecolor_generated_texel_fraction") == 0.0
        and _float(summary_metrics, "pbr_generated_texel_fraction") == 0.0
        and int(_float(summary_metrics, "generated_face_count")) == 0
    )
    source_scan = _source_scan_checks()
    requirements = [
        _requirement(
            "multi_image_input",
            "Take multiple images of the same cat or deterministic synthetic views.",
            bool(summary_steps.get("synthetic_create"))
            and int(_float(acceptance_metrics, "frame_count")) >= 3,
            [summary_path, acceptance_path],
            {
                "frame_count": int(_float(acceptance_metrics, "frame_count")),
                "synthetic_create": bool(summary_steps.get("synthetic_create")),
            },
        ),
        _requirement(
            "cat_segmentation",
            "Detect and segment the cat before fitting.",
            bool(summary_steps.get("preprocess"))
            and acceptance_checks.get("segmentation_outputs_present", False),
            [summary_path, acceptance_path],
            {
                "segmentation_mean_iou_with_previous_masks": _float(
                    summary_metrics,
                    "segmentation_mean_iou_with_previous_masks",
                ),
            },
        ),
        _requirement(
            "keypoints_and_pose",
            "Estimate keypoints and pose initialization for each view.",
            acceptance_checks.get("keypoint_pose_initialization_present", False)
            and _float(fit_metrics, "init_prediction_used") == 1.0,
            [acceptance_path, fit_report_path],
            {
                "keypoint_mean_confidence": _float(
                    acceptance_metrics,
                    "keypoint_mean_confidence",
                ),
                "fit_init_prediction_used": _float(fit_metrics, "init_prediction_used"),
            },
        ),
        _requirement(
            "fixed_topology_cat_template",
            "Fit and export a fixed-topology CatTemplate mesh.",
            acceptance_checks.get("fixed_topology_cat_template", False)
            and int(mesh_validation.get("vertex_count", -1)) == CAT_TEMPLATE_VERTEX_COUNT
            and int(mesh_validation.get("face_count", -1)) == CAT_TEMPLATE_FACE_COUNT,
            [acceptance_path, mesh_validation_path],
            {
                "vertex_count": int(mesh_validation.get("vertex_count", -1)),
                "face_count": int(mesh_validation.get("face_count", -1)),
                "joint_count": int(mesh_validation.get("joint_count", -1)),
            },
        ),
        _requirement(
            "shared_identity_across_views",
            "Optimize one shared 3D identity across all fitted views.",
            acceptance_checks.get("shared_identity_across_views", False)
            and _float(fit_metrics, "final_beta_error")
            <= _float(fit_metrics, "initial_beta_error"),
            [acceptance_path, fit_report_path],
            {
                "initial_beta_error": _float(fit_metrics, "initial_beta_error"),
                "final_beta_error": _float(fit_metrics, "final_beta_error"),
                "beta_error_reduction_fraction": _float(
                    fit_metrics,
                    "beta_error_reduction_fraction",
                ),
            },
        ),
        _requirement(
            "canonical_mesh_outputs",
            "Output canonical OBJ, PLY, and GLB mesh artifacts.",
            acceptance_checks.get("canonical_mesh_exports_valid", False)
            and _artifact_exists(summary_artifacts, "obj")
            and _artifact_exists(summary_artifacts, "ply")
            and _artifact_exists(summary_artifacts, "glb"),
            [acceptance_path, Path(str(summary_artifacts.get("obj", "")))],
            {
                "obj": str(summary_artifacts.get("obj", "")),
                "ply": str(summary_artifacts.get("ply", "")),
                "glb": str(summary_artifacts.get("glb", "")),
            },
        ),
        _requirement(
            "valid_glb_avatar",
            "Export a valid GLB avatar with skeleton, PBR textures, and idle animation.",
            acceptance_checks.get("glb_export_valid", False)
            and acceptance_checks.get("skeleton_metadata_export", False)
            and acceptance_checks.get("basic_animation_export_valid", False)
            and validation_checks.get("glb_structure_valid", False),
            [acceptance_path, validation_path, mesh_validation_path],
            {
                "glb_structure_valid": validation_checks.get("glb_structure_valid", False),
                "glb_skinning_valid": validation_checks.get("glb_skinning_valid", False),
                "glb_idle_animation_valid": validation_checks.get(
                    "glb_idle_animation_valid",
                    False,
                ),
            },
        ),
        _requirement(
            "synthetic_reconstruction_tests",
            "Pass deterministic synthetic reconstruction tests and thresholds.",
            bool(summary.get("success"))
            and bool(acceptance.get("success"))
            and bool(evaluation.get("success"))
            and bool(validation.get("success")),
            [summary_path, acceptance_path, evaluation_path, validation_path],
            {
                "pipeline_success": bool(summary.get("success")),
                "acceptance_success": bool(acceptance.get("success")),
                "evaluation_success": bool(evaluation.get("success")),
                "validation_success": bool(validation.get("success")),
            },
        ),
        _requirement(
            "optimizer_no_nan",
            "Ensure optimizer convergence without NaN or invalid fit-state tensors.",
            acceptance_checks.get("optimizer_converged_without_nan", False)
            and validation_checks.get("fit_state_safetensors_valid", False),
            [acceptance_path, validation_path, fit_report_path],
            {
                "initial_loss": _float(fit_metrics, "initial_loss"),
                "final_loss": _float(fit_metrics, "final_loss"),
                "fit_state_source_format": str(
                    summary_artifacts.get("fit_state_source_format", "")
                ),
            },
        ),
        _requirement(
            "losses_reduce",
            "Reduce silhouette and keypoint losses over fitting iterations.",
            acceptance_checks.get("component_losses_reduce", False)
            and evaluation_checks.get("keypoint_loss_reduced", False)
            and evaluation_checks.get("silhouette_loss_reduced", False)
            and _float(fit_metrics, "final_keypoint_loss")
            < _float(fit_metrics, "initial_keypoint_loss")
            and _float(fit_metrics, "final_silhouette_loss")
            < _float(fit_metrics, "initial_silhouette_loss"),
            [acceptance_path, evaluation_path, fit_report_path],
            {
                "initial_keypoint_loss": _float(fit_metrics, "initial_keypoint_loss"),
                "final_keypoint_loss": _float(fit_metrics, "final_keypoint_loss"),
                "initial_silhouette_loss": _float(
                    fit_metrics,
                    "initial_silhouette_loss",
                ),
                "final_silhouette_loss": _float(fit_metrics, "final_silhouette_loss"),
            },
        ),
        _requirement(
            "geometry_truth_over_generated_imagination",
            "Keep observed pixels dominant over generated geometry or texture.",
            generated_free_texture
            and validation_checks.get("texture_basecolor_generated_fraction_zero", False)
            and validation_checks.get("texture_pbr_generated_fraction_zero", False),
            [summary_path, validation_path],
            {
                "generated_face_count": int(_float(summary_metrics, "generated_face_count")),
                "basecolor_generated_texel_fraction": _float(
                    summary_metrics,
                    "basecolor_generated_texel_fraction",
                ),
                "pbr_generated_texel_fraction": _float(
                    summary_metrics,
                    "pbr_generated_texel_fraction",
                ),
                "observed_face_fraction": _float(summary_metrics, "observed_face_fraction"),
            },
        ),
        _requirement(
            "safetensors_primary_fit_state",
            "Use deterministic safetensors fit state as the primary fit artifact.",
            str(summary_artifacts.get("fit_state_source_format")) == "safetensors"
            and str(summary_artifacts.get("fit_state", "")).endswith("fit_state.safetensors")
            and not (project_dir / "outputs" / "fitting" / "fit_state.npz").exists(),
            [summary_path, project_dir / "outputs" / "fitting" / "fit_state.safetensors"],
            {
                "fit_state": str(summary_artifacts.get("fit_state", "")),
                "fit_state_source_format": str(
                    summary_artifacts.get("fit_state_source_format", "")
                ),
                "legacy_npz_exists": (
                    project_dir / "outputs" / "fitting" / "fit_state.npz"
                ).exists(),
            },
        ),
        _requirement(
            "no_external_api_or_training_runtime",
            "Avoid external API clients and large-model training paths in core code.",
            source_scan["no_external_api_clients"]
            and source_scan["no_training_entrypoints"],
            [Path(__file__).parents[1]],
            source_scan,
        ),
    ]
    checks = {str(item["id"]): bool(item["passed"]) for item in requirements}
    success = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": "synthetic",
        "success": success,
        "checks": checks,
        "requirements": requirements,
        "artifacts": {
            "pipeline_summary": str(summary_path),
            "acceptance_audit": str(acceptance_path),
            "synthetic_evaluation": str(evaluation_path),
            "validation_report": str(validation_path),
            "mesh_validation": str(mesh_validation_path),
            "fit_report": str(fit_report_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ObjectiveComplianceResult(report_path=report_path, success=success, checks=checks)


def write_raw_objective_compliance_audit(
    project_dir: Path,
    *,
    force: bool = False,
) -> ObjectiveComplianceResult:
    """Write a compact requirement-to-evidence audit for a raw pipeline run."""

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / "raw_opencat3d_compliance_audit.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    summary_path = output_dir / "raw_pipeline_summary.json"
    evaluation_path = output_dir / "raw_evaluation.json"
    validation_path = output_dir / "validation.json"
    mesh_validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"

    summary = _read_json(summary_path)
    evaluation = _read_json(evaluation_path)
    validation = _read_json(validation_path)
    mesh_validation = _read_json(mesh_validation_path)
    fit_report = _read_json(fit_report_path)

    summary_steps = _object(summary.get("steps"), "raw_pipeline_summary.steps")
    summary_metrics = _object(summary.get("metrics"), "raw_pipeline_summary.metrics")
    summary_artifacts = _object(summary.get("artifacts"), "raw_pipeline_summary.artifacts")
    evaluation_checks = _bool_map(evaluation.get("checks"), "raw_evaluation.checks")
    validation_checks = _bool_map(validation.get("checks"), "validation.checks")
    fit_metrics = _object(fit_report.get("metrics"), "fit_report.metrics")

    generated_free_texture = (
        _float(summary_metrics, "basecolor_generated_texel_fraction") == 0.0
        and _float(summary_metrics, "pbr_generated_texel_fraction") == 0.0
        and int(_float(summary_metrics, "generated_face_count")) == 0
    )
    raw_frame_count = int(_float(summary_metrics, "raw_frame_count"))
    source_scan = _source_scan_checks()
    requirements = [
        _requirement(
            "multi_image_input",
            "Take multiple images of the same cat from raw input frames.",
            bool(summary_steps.get("raw_ingest")) and raw_frame_count >= 2,
            [summary_path, Path(str(summary_artifacts.get("raw_manifest", "")))],
            {
                "raw_frame_count": raw_frame_count,
                "raw_ingest": bool(summary_steps.get("raw_ingest")),
            },
        ),
        _requirement(
            "cat_segmentation",
            "Detect and segment the cat before raw fitting.",
            bool(summary_steps.get("raw_segmentation"))
            and _float(summary_metrics, "segmentation_mean_mask_area_ratio") > 0.0
            and _artifact_exists(summary_artifacts, "observation_manifest"),
            [summary_path, Path(str(summary_artifacts.get("observation_manifest", "")))],
            {
                "segmentation_mean_mask_area_ratio": _float(
                    summary_metrics,
                    "segmentation_mean_mask_area_ratio",
                ),
            },
        ),
        _requirement(
            "keypoints_and_pose",
            "Estimate keypoints and pose initialization for raw views.",
            bool(summary_steps.get("raw_keypoint_bootstrap"))
            and _float(summary_metrics, "keypoint_mean_confidence") > 0.0
            and _float(summary_metrics, "keypoint_mean_in_bbox_fraction") >= 1.0
            and _float(fit_metrics, "init_prediction_used") == 1.0,
            [summary_path, fit_report_path],
            {
                "keypoint_mean_confidence": _float(
                    summary_metrics,
                    "keypoint_mean_confidence",
                ),
                "keypoint_mean_in_bbox_fraction": _float(
                    summary_metrics,
                    "keypoint_mean_in_bbox_fraction",
                ),
                "fit_init_prediction_used": _float(fit_metrics, "init_prediction_used"),
            },
        ),
        _requirement(
            "fixed_topology_cat_template",
            "Fit and export a fixed-topology CatTemplate mesh.",
            evaluation_checks.get("template_vertex_count", False)
            and evaluation_checks.get("template_face_count", False)
            and evaluation_checks.get("mesh_vertices_finite", False)
            and evaluation_checks.get("mesh_faces_valid", False)
            and int(mesh_validation.get("vertex_count", -1)) == CAT_TEMPLATE_VERTEX_COUNT
            and int(mesh_validation.get("face_count", -1)) == CAT_TEMPLATE_FACE_COUNT,
            [evaluation_path, mesh_validation_path],
            {
                "vertex_count": int(mesh_validation.get("vertex_count", -1)),
                "face_count": int(mesh_validation.get("face_count", -1)),
                "joint_count": int(mesh_validation.get("joint_count", -1)),
            },
        ),
        _requirement(
            "shared_identity_across_views",
            "Keep one shared 3D identity state across all raw views.",
            evaluation_checks.get("multi_frame_input", False)
            and evaluation_checks.get("fit_state_finite", False)
            and int(fit_report.get("num_frames", 0)) == raw_frame_count,
            [evaluation_path, fit_report_path],
            {
                "num_frames": int(fit_report.get("num_frames", 0)),
                "raw_frame_count": raw_frame_count,
                "fit_state_finite": evaluation_checks.get("fit_state_finite", False),
            },
        ),
        _requirement(
            "canonical_mesh_outputs",
            "Output canonical OBJ, PLY, and GLB mesh artifacts.",
            evaluation_checks.get("obj_exists", False)
            and evaluation_checks.get("ply_exists", False)
            and evaluation_checks.get("glb_exists", False)
            and evaluation_checks.get("obj_ply_reload_match", False)
            and _artifact_exists(summary_artifacts, "obj")
            and _artifact_exists(summary_artifacts, "ply")
            and _artifact_exists(summary_artifacts, "glb"),
            [evaluation_path, Path(str(summary_artifacts.get("obj", "")))],
            {
                "obj": str(summary_artifacts.get("obj", "")),
                "ply": str(summary_artifacts.get("ply", "")),
                "glb": str(summary_artifacts.get("glb", "")),
            },
        ),
        _requirement(
            "valid_glb_avatar",
            "Export a valid GLB avatar with skeleton, PBR textures, and idle animation.",
            evaluation_checks.get("glb_structure_valid", False)
            and evaluation_checks.get("glb_skinning_valid", False)
            and evaluation_checks.get("glb_idle_animation_valid", False)
            and evaluation_checks.get("skeleton_metadata_valid", False)
            and validation_checks.get("glb_structure_valid", False),
            [evaluation_path, validation_path, mesh_validation_path],
            {
                "glb_structure_valid": validation_checks.get("glb_structure_valid", False),
                "glb_skinning_valid": evaluation_checks.get("glb_skinning_valid", False),
                "glb_idle_animation_valid": evaluation_checks.get(
                    "glb_idle_animation_valid",
                    False,
                ),
            },
        ),
        _requirement(
            "raw_reconstruction_quality_gates",
            "Pass deterministic raw reconstruction quality gates.",
            bool(summary.get("success"))
            and bool(evaluation.get("success"))
            and bool(validation.get("success"))
            and _float(summary_metrics, "raw_quality_passed_check_count")
            == _float(summary_metrics, "raw_quality_total_check_count"),
            [summary_path, evaluation_path, validation_path],
            {
                "pipeline_success": bool(summary.get("success")),
                "evaluation_success": bool(evaluation.get("success")),
                "validation_success": bool(validation.get("success")),
                "passed_check_count": int(
                    _float(summary_metrics, "raw_quality_passed_check_count")
                ),
                "total_check_count": int(
                    _float(summary_metrics, "raw_quality_total_check_count")
                ),
            },
        ),
        _requirement(
            "optimizer_no_nan",
            "Ensure raw optimizer convergence without NaN or invalid fit-state tensors.",
            evaluation_checks.get("fit_report_success", False)
            and evaluation_checks.get("loss_finite", False)
            and evaluation_checks.get("fit_state_finite", False)
            and validation_checks.get("fit_state_safetensors_valid", False),
            [evaluation_path, validation_path, fit_report_path],
            {
                "initial_loss": _float(fit_metrics, "initial_loss"),
                "final_loss": _float(fit_metrics, "final_loss"),
                "fit_state_source_format": str(
                    summary_artifacts.get("fit_state_source_format", "")
                ),
            },
        ),
        _requirement(
            "losses_reduce",
            "Reduce silhouette and keypoint losses over raw fitting iterations.",
            evaluation_checks.get("keypoint_loss_reduced", False)
            and evaluation_checks.get("silhouette_loss_reduced", False)
            and _float(fit_metrics, "final_keypoint_loss")
            < _float(fit_metrics, "initial_keypoint_loss")
            and _float(fit_metrics, "final_silhouette_loss")
            < _float(fit_metrics, "initial_silhouette_loss"),
            [evaluation_path, fit_report_path],
            {
                "initial_keypoint_loss": _float(fit_metrics, "initial_keypoint_loss"),
                "final_keypoint_loss": _float(fit_metrics, "final_keypoint_loss"),
                "initial_silhouette_loss": _float(
                    fit_metrics,
                    "initial_silhouette_loss",
                ),
                "final_silhouette_loss": _float(fit_metrics, "final_silhouette_loss"),
            },
        ),
        _requirement(
            "geometry_truth_over_generated_imagination",
            "Keep observed pixels dominant over generated geometry or texture.",
            generated_free_texture
            and validation_checks.get("texture_basecolor_generated_fraction_zero", False)
            and validation_checks.get("texture_pbr_generated_fraction_zero", False),
            [summary_path, validation_path],
            {
                "generated_face_count": int(_float(summary_metrics, "generated_face_count")),
                "basecolor_generated_texel_fraction": _float(
                    summary_metrics,
                    "basecolor_generated_texel_fraction",
                ),
                "pbr_generated_texel_fraction": _float(
                    summary_metrics,
                    "pbr_generated_texel_fraction",
                ),
                "observed_face_fraction": _float(summary_metrics, "observed_face_fraction"),
            },
        ),
        _requirement(
            "safetensors_primary_fit_state",
            "Use deterministic safetensors fit state as the primary fit artifact.",
            str(summary_artifacts.get("fit_state_source_format")) == "safetensors"
            and str(summary_artifacts.get("fit_state", "")).endswith("fit_state.safetensors")
            and not (project_dir / "outputs" / "fitting" / "fit_state.npz").exists(),
            [summary_path, project_dir / "outputs" / "fitting" / "fit_state.safetensors"],
            {
                "fit_state": str(summary_artifacts.get("fit_state", "")),
                "fit_state_source_format": str(
                    summary_artifacts.get("fit_state_source_format", "")
                ),
                "legacy_npz_exists": (
                    project_dir / "outputs" / "fitting" / "fit_state.npz"
                ).exists(),
            },
        ),
        _requirement(
            "no_external_api_or_training_runtime",
            "Avoid external API clients and large-model training paths in core code.",
            source_scan["no_external_api_clients"]
            and source_scan["no_training_entrypoints"],
            [Path(__file__).parents[1]],
            source_scan,
        ),
    ]
    checks = {str(item["id"]): bool(item["passed"]) for item in requirements}
    success = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": "raw",
        "success": success,
        "checks": checks,
        "requirements": requirements,
        "artifacts": {
            "pipeline_summary": str(summary_path),
            "raw_evaluation": str(evaluation_path),
            "validation_report": str(validation_path),
            "mesh_validation": str(mesh_validation_path),
            "fit_report": str(fit_report_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ObjectiveComplianceResult(report_path=report_path, success=success, checks=checks)


def _requirement(
    requirement_id: str,
    text: str,
    passed: bool,
    evidence_paths: list[Path],
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "id": requirement_id,
        "requirement": text,
        "passed": bool(passed),
        "evidence_paths": [str(path) for path in evidence_paths],
        "evidence": evidence,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _bool_map(value: object, name: str) -> dict[str, bool]:
    raw = _object(value, name)
    return {str(key): bool(item) for key, item in raw.items()}


def _float(values: dict[str, Any], key: str) -> float:
    try:
        return float(values[key])
    except KeyError as exc:
        raise ValueError(f"missing numeric field: {key}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field: {key}") from exc


def _artifact_exists(artifacts: dict[str, Any], key: str) -> bool:
    raw_path = artifacts.get(key)
    return isinstance(raw_path, str) and Path(raw_path).exists()


def _source_scan_checks() -> dict[str, object]:
    source_root = Path(__file__).parents[1]
    forbidden_external = (
        "import " + "op" + "enai",
        "from " + "op" + "enai",
        "import " + "req" + "uests",
        "from " + "req" + "uests",
        "import " + "ht" + "tpx",
        "from " + "ht" + "tpx",
        "urllib" + ".request",
        "bo" + "to3",
    )
    forbidden_training = (
        "Tra" + "iner(",
        ".fi" + "t(",
        "train" + "_model",
        "def " + "train(",
        "Lightning" + "Module",
    )
    external_hits = _scan_source_for(source_root, forbidden_external)
    training_hits = _scan_source_for(source_root, forbidden_training)
    return {
        "source_root": str(source_root),
        "no_external_api_clients": not external_hits,
        "no_training_entrypoints": not training_hits,
        "external_api_hits": external_hits,
        "training_hits": training_hits,
    }


def _scan_source_for(source_root: Path, needles: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        if path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(needle in line for needle in needles):
                hits.append(f"{path}:{line_number}")
    return hits
