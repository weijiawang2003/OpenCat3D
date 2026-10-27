import json

from opencat3d.evaluation.acceptance import (
    SYNTHETIC_ACCEPTANCE_THRESHOLDS,
    evaluate_synthetic_acceptance,
)
from opencat3d.pipeline import run_synthetic_pipeline
from opencat3d.template.cat_template import CAT_TEMPLATE_FACE_COUNT, CAT_TEMPLATE_VERTEX_COUNT


def test_evaluate_synthetic_acceptance_writes_success_audit(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    pipeline_result = run_synthetic_pipeline(project_dir, num_views=3, image_size=64, seed=999)

    result = evaluate_synthetic_acceptance(project_dir, force=True)

    assert pipeline_result.success
    assert result.success
    assert result.report_path.exists()
    assert all(result.checks.values())
    assert result.metrics["frame_count"] == 3
    assert result.metrics["segmentation_mean_iou_with_previous_masks"] == 1.0
    assert result.metrics["keypoint_mean_confidence"] == 1.0
    assert result.metrics["final_loss"] < result.metrics["initial_loss"]
    assert result.metrics["final_keypoint_loss"] < result.metrics["initial_keypoint_loss"]
    assert (
        result.metrics["final_silhouette_loss"]
        < result.metrics["initial_silhouette_loss"]
    )
    assert result.metrics["vertex_count"] == CAT_TEMPLATE_VERTEX_COUNT
    assert result.metrics["face_count"] == CAT_TEMPLATE_FACE_COUNT
    assert result.metrics["joint_count"] == 11
    assert result.metrics["passed_check_count"] == result.metrics["total_check_count"]
    assert result.checks["per_frame_overlays_present"]
    assert result.checks["coverage_report_valid"]
    assert result.checks["observed_basecolor_texture_valid"]
    assert result.checks["pbr_texture_outputs_valid"]
    assert result.checks["skeleton_metadata_export"]
    assert result.checks["basic_animation_export_valid"]
    assert result.checks["joint_projection_report_valid"]
    assert result.checks["silhouette_residual_maps_valid"]
    assert result.checks["artifact_validation_report_valid"]
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["thresholds"] == SYNTHETIC_ACCEPTANCE_THRESHOLDS
    assert report["artifacts"]["fit_state"].endswith("outputs/fitting/fit_state.safetensors")
    assert report["artifacts"]["fit_state_source_format"] == "safetensors"
    assert report["artifacts"]["glb"].endswith("cat_avatar.glb")
    assert report["artifacts"]["idle_animation"].endswith(
        "outputs/avatar/animations/idle.glb"
    )
    assert report["artifacts"]["skeleton"].endswith("outputs/avatar/skeleton.json")
    assert report["artifacts"]["basecolor"].endswith("outputs/textures/basecolor.png")
    assert report["artifacts"]["basecolor_confidence"].endswith(
        "outputs/textures/basecolor_confidence.png"
    )
    assert report["artifacts"]["normal"].endswith("outputs/textures/normal.png")
    assert report["artifacts"]["roughness"].endswith("outputs/textures/roughness.png")
    assert report["artifacts"]["ao"].endswith("outputs/textures/ao.png")
    assert report["artifacts"]["joint_projection"].endswith(
        "outputs/quality/synthetic_joint_projection.json"
    )
    assert report["artifacts"]["silhouette_residuals"].endswith(
        "outputs/quality/synthetic_silhouette_residuals.json"
    )
    assert report["artifacts"]["heldout_evaluation"].endswith(
        "synthetic_heldout_evaluation.json"
    )
    assert report["artifacts"]["validation_report"].endswith("outputs/quality/validation.json")


def test_evaluate_synthetic_acceptance_fails_when_component_loss_does_not_reduce(
    tmp_path,
) -> None:
    project_dir = tmp_path / "demo"
    run_synthetic_pipeline(project_dir, num_views=3, image_size=64, seed=999)
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"
    fit_report = json.loads(fit_report_path.read_text(encoding="utf-8"))
    fit_report["metrics"]["final_keypoint_loss"] = fit_report["metrics"][
        "initial_keypoint_loss"
    ]
    fit_report_path.write_text(
        json.dumps(fit_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = evaluate_synthetic_acceptance(project_dir, force=True)

    assert not result.success
    assert not result.checks["component_losses_reduce"]
    assert result.checks["synthetic_multi_view_reconstruction"]
