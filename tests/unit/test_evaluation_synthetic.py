import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from opencat3d.cli import app
from opencat3d.evaluation.synthetic import (
    SYNTHETIC_METRIC_THRESHOLDS,
    evaluate_synthetic_reconstruction,
)
from opencat3d.export.mesh import export_fitted_canonical_mesh
from opencat3d.fitting.optimizer import OptimizerConfig, run_synthetic_fit
from opencat3d.synthetic.generator import create_synthetic_project


def _create_fitted_exported_project(project_dir) -> None:
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=20),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )
    export_fitted_canonical_mesh(project_dir)


def test_evaluate_synthetic_reconstruction_writes_success_report(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    _create_fitted_exported_project(project_dir)

    result = evaluate_synthetic_reconstruction(project_dir)

    assert result.success
    assert result.report_path.exists()
    assert result.checks["loss_reduced"]
    assert result.checks["keypoint_loss_reduced"]
    assert result.checks["silhouette_loss_reduced"]
    assert result.checks["loss_reduction_fraction_threshold"]
    assert result.checks["final_loss_threshold"]
    assert result.checks["beta_error_not_increased"]
    assert result.checks["final_beta_error_threshold"]
    assert result.checks["beta_error_reduction_fraction_threshold"]
    assert result.checks["glb_metadata_valid"]
    assert result.checks["glb_structure_valid"]
    assert result.checks["glb_basecolor_texture_valid"]
    assert result.checks["glb_pbr_textures_valid"]
    assert result.checks["glb_skinning_valid"]
    assert result.checks["glb_idle_animation_valid"]
    assert result.checks["skeleton_metadata_valid"]
    assert result.checks["skeleton_skin_weights_valid"]
    assert result.checks["template_vertex_count"]
    assert result.checks["template_face_count"]
    assert result.checks["fit_overlays_present"]
    assert result.checks["coverage_report_valid"]
    assert result.checks["texture_basecolor_valid"]
    assert result.checks["pbr_texture_outputs_valid"]
    assert result.checks["heldout_view_consistency"]
    assert result.checks["joint_projection_report_valid"]
    assert result.checks["silhouette_residual_maps_valid"]
    assert result.checks["artifact_validation_report_valid"]
    assert result.metrics["loss_reduction_fraction"] > 0.0
    assert result.metrics["final_keypoint_loss"] < result.metrics["initial_keypoint_loss"]
    assert result.metrics["final_silhouette_loss"] < result.metrics["initial_silhouette_loss"]
    assert result.metrics["heldout_mean_silhouette_iou"] >= 0.80
    assert result.metrics["pbr_observed_texel_fraction"] > 0.0
    assert result.metrics["pbr_generated_texel_fraction"] == 0.0
    assert (
        result.metrics["loss_reduction_fraction"]
        >= SYNTHETIC_METRIC_THRESHOLDS["min_loss_reduction_fraction"]
    )
    assert result.metrics["final_loss"] <= SYNTHETIC_METRIC_THRESHOLDS["max_final_loss"]
    assert (
        result.metrics["final_beta_error"]
        <= SYNTHETIC_METRIC_THRESHOLDS["max_final_beta_error"]
    )
    assert (
        result.metrics["beta_error_reduction_fraction"]
        >= SYNTHETIC_METRIC_THRESHOLDS["min_beta_error_reduction_fraction"]
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["thresholds"] == SYNTHETIC_METRIC_THRESHOLDS
    assert report["metrics"]["vertex_count"] == SYNTHETIC_METRIC_THRESHOLDS[
        "expected_vertex_count"
    ]
    assert report["metrics"]["face_count"] == SYNTHETIC_METRIC_THRESHOLDS[
        "expected_face_count"
    ]
    assert report["metrics"]["joint_count"] == 11
    assert report["metrics"]["final_beta_error"] <= report["metrics"]["initial_beta_error"]
    assert report["metrics"]["keypoint_loss_reduction"] > 0.0
    assert report["metrics"]["silhouette_loss_reduction"] > 0.0
    assert report["metrics"]["overlay_mean_silhouette_iou"] >= 0.0
    assert report["metrics"]["observed_face_fraction"] > 0.0
    assert report["metrics"]["observed_vertex_fraction"] > 0.0
    assert report["metrics"]["generated_face_count"] == 0
    assert report["metrics"]["basecolor_observed_texel_fraction"] > 0.0
    assert report["metrics"]["basecolor_generated_texel_fraction"] == 0.0
    assert report["metrics"]["joint_projection_mean_keypoint_loss"] <= 0.01
    assert report["metrics"]["joint_projection_mean_pixel_error"] >= 0.0
    assert report["metrics"]["silhouette_residual_mean_union_error"] <= 0.35
    assert report["metrics"]["silhouette_residual_mean_false_negative"] >= 0.0
    assert report["metrics"]["silhouette_residual_mean_false_positive"] >= 0.0
    assert all(report["glb_structure_checks"].values())
    assert report["glb_structure_checks"]["expected_textured"]
    assert report["glb_structure_checks"]["expected_pbr_textures"]
    assert report["glb_structure_checks"]["expected_animated"]
    assert report["glb_structure_checks"]["texture_material_valid"]
    assert report["glb_structure_checks"]["pbr_material_textures_valid"]
    assert report["glb_structure_checks"]["pbr_metallic_factor_zero"]
    assert report["glb_structure_checks"]["animation_valid"]
    assert report["artifacts"]["glb"].endswith("cat_avatar.glb")
    assert report["artifacts"]["fit_state_safetensors"].endswith(
        "outputs/fitting/fit_state.safetensors"
    )
    assert report["artifacts"]["idle_animation"].endswith("outputs/avatar/animations/idle.glb")
    assert report["artifacts"]["skeleton"].endswith("outputs/avatar/skeleton.json")
    assert report["artifacts"]["overlay_manifest"].endswith(
        "outputs/fitting/per_frame_overlay/manifest.json"
    )
    assert report["artifacts"]["coverage_report"].endswith("outputs/quality/coverage.json")
    assert report["artifacts"]["coverage_provenance"].endswith(
        "outputs/quality/coverage_provenance.npz"
    )
    assert report["artifacts"]["texture_manifest"].endswith(
        "outputs/textures/texture_manifest.json"
    )
    assert report["artifacts"]["basecolor"].endswith("outputs/textures/basecolor.png")
    assert report["artifacts"]["basecolor_confidence"].endswith(
        "outputs/textures/basecolor_confidence.png"
    )
    assert report["artifacts"]["source_view_id"].endswith(
        "outputs/textures/source_view_id.png"
    )
    assert report["artifacts"]["normal"].endswith("outputs/textures/normal.png")
    assert report["artifacts"]["roughness"].endswith("outputs/textures/roughness.png")
    assert report["artifacts"]["ao"].endswith("outputs/textures/ao.png")
    assert report["artifacts"]["heldout_evaluation"].endswith("synthetic_heldout_evaluation.json")
    assert all(report["skeleton_checks"].values())
    assert all(report["joint_projection_checks"].values())
    assert all(report["silhouette_residual_checks"].values())
    assert report["artifacts"]["joint_projection"].endswith(
        "outputs/quality/synthetic_joint_projection.json"
    )
    assert report["artifacts"]["silhouette_residuals"].endswith(
        "outputs/quality/synthetic_silhouette_residuals.json"
    )
    assert report["artifacts"]["silhouette_residual_manifest"].endswith(
        "outputs/quality/synthetic_silhouette_residuals/manifest.json"
    )
    assert report["artifacts"]["validation_report"].endswith("outputs/quality/validation.json")
    validation = json.loads(
        Path(report["artifacts"]["validation_report"]).read_text(encoding="utf-8")
    )
    assert validation["success"] is True
    assert validation["checks"]["glb_exists"]
    assert validation["checks"]["glb_structure_valid"]
    assert validation["checks"]["fit_state_safetensors_valid"]
    assert validation["checks"]["texture_artifact_hashes_valid"]
    assert validation["fit_state_checks"]["pose_shapes_valid"]
    assert validation["texture_manifest_checks"]["pbr_source_neutral"]
    assert "fit_state_safetensors" in validation["sha256"]
    assert "glb" in validation["sha256"]


def test_evaluate_synthetic_reconstruction_fails_metric_thresholds(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    _create_fitted_exported_project(project_dir)
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"
    fit_report = json.loads(fit_report_path.read_text(encoding="utf-8"))
    initial_loss = float(fit_report["metrics"]["initial_loss"])
    initial_beta_error = float(fit_report["metrics"]["initial_beta_error"])
    degraded_final_loss = initial_loss * 0.5
    degraded_final_beta_error = float(
        SYNTHETIC_METRIC_THRESHOLDS["max_final_beta_error"]
    ) + 0.005
    assert degraded_final_beta_error < initial_beta_error
    fit_report["metrics"]["final_loss"] = degraded_final_loss
    fit_report["metrics"]["final_beta_error"] = degraded_final_beta_error
    fit_report_path.write_text(json.dumps(fit_report, indent=2, sort_keys=True) + "\n")

    result = evaluate_synthetic_reconstruction(project_dir, force=True)

    assert not result.success
    assert result.checks["loss_reduced"]
    assert not result.checks["loss_reduction_fraction_threshold"]
    assert result.checks["beta_error_not_increased"]
    assert not result.checks["final_beta_error_threshold"]
    assert not result.checks["beta_error_reduction_fraction_threshold"]


def test_evaluate_cli_runs_full_synthetic_chain(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "demo"

    create_result = runner.invoke(
        app,
        ["synthetic", "create", str(project_dir), "--num-views", "3", "--image-size", "64"],
    )
    fit_result = runner.invoke(app, ["fit", str(project_dir), "--mode", "synthetic"])
    export_result = runner.invoke(app, ["export", str(project_dir), "--mode", "synthetic"])
    evaluate_result = runner.invoke(app, ["evaluate", str(project_dir), "--mode", "synthetic"])

    assert create_result.exit_code == 0, create_result.output
    assert fit_result.exit_code == 0, fit_result.output
    assert export_result.exit_code == 0, export_result.output
    assert evaluate_result.exit_code == 0, evaluate_result.output
    assert "success=True" in evaluate_result.output
    assert (project_dir / "outputs" / "quality" / "synthetic_evaluation.json").exists()
