import json

import numpy as np
from PIL import Image

from opencat3d.evaluation.compliance import (
    write_objective_compliance_audit,
    write_raw_objective_compliance_audit,
)
from opencat3d.pipeline import run_raw_pipeline, run_synthetic_pipeline

EXPECTED_SYNTHETIC_REQUIREMENT_IDS = [
    "multi_image_input",
    "cat_segmentation",
    "keypoints_and_pose",
    "fixed_topology_cat_template",
    "shared_identity_across_views",
    "canonical_mesh_outputs",
    "valid_glb_avatar",
    "synthetic_reconstruction_tests",
    "optimizer_no_nan",
    "losses_reduce",
    "geometry_truth_over_generated_imagination",
    "safetensors_primary_fit_state",
    "no_external_api_or_training_runtime",
]

EXPECTED_RAW_REQUIREMENT_IDS = [
    "multi_image_input",
    "cat_segmentation",
    "keypoints_and_pose",
    "fixed_topology_cat_template",
    "shared_identity_across_views",
    "canonical_mesh_outputs",
    "valid_glb_avatar",
    "raw_reconstruction_quality_gates",
    "optimizer_no_nan",
    "losses_reduce",
    "geometry_truth_over_generated_imagination",
    "safetensors_primary_fit_state",
    "no_external_api_or_training_runtime",
]


def test_write_objective_compliance_audit_maps_requirements_to_evidence(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    pipeline_result = run_synthetic_pipeline(
        project_dir,
        num_views=3,
        image_size=64,
        seed=999,
    )

    report = json.loads(pipeline_result.compliance_audit_path.read_text(encoding="utf-8"))

    assert pipeline_result.success
    assert report["success"] is True
    assert set(report["checks"]) == set(EXPECTED_SYNTHETIC_REQUIREMENT_IDS)
    assert [item["id"] for item in report["requirements"]] == EXPECTED_SYNTHETIC_REQUIREMENT_IDS
    assert all(report["checks"].values())
    assert report["artifacts"]["pipeline_summary"].endswith(
        "outputs/quality/synthetic_pipeline_summary.json"
    )
    assert report["artifacts"]["acceptance_audit"].endswith(
        "outputs/quality/synthetic_acceptance_audit.json"
    )
    assert report["artifacts"]["validation_report"].endswith(
        "outputs/quality/validation.json"
    )
    source_requirement = _requirement(report, "no_external_api_or_training_runtime")
    assert source_requirement["evidence"]["external_api_hits"] == []
    assert source_requirement["evidence"]["training_hits"] == []
    fit_state_requirement = _requirement(report, "safetensors_primary_fit_state")
    assert fit_state_requirement["evidence"]["fit_state"].endswith(
        "outputs/fitting/fit_state.safetensors"
    )
    assert fit_state_requirement["evidence"]["legacy_npz_exists"] is False


def test_write_raw_objective_compliance_audit_maps_requirements_to_evidence(tmp_path) -> None:
    project_dir = tmp_path / "project"
    input_dir = tmp_path / "inputs"
    _write_raw_inputs(input_dir)

    pipeline_result = run_raw_pipeline(
        project_dir,
        input_paths=[input_dir],
        color_delta_threshold=0.10,
        camera_root_steps=2,
    )

    report = json.loads(pipeline_result.compliance_audit_path.read_text(encoding="utf-8"))

    assert pipeline_result.success
    assert report["mode"] == "raw"
    assert report["success"] is True
    assert set(report["checks"]) == set(EXPECTED_RAW_REQUIREMENT_IDS)
    assert [item["id"] for item in report["requirements"]] == EXPECTED_RAW_REQUIREMENT_IDS
    assert all(report["checks"].values())
    assert report["artifacts"]["pipeline_summary"].endswith(
        "outputs/quality/raw_pipeline_summary.json"
    )
    assert report["artifacts"]["raw_evaluation"].endswith(
        "outputs/quality/raw_evaluation.json"
    )
    source_requirement = _requirement(report, "no_external_api_or_training_runtime")
    assert source_requirement["evidence"]["external_api_hits"] == []
    assert source_requirement["evidence"]["training_hits"] == []
    input_requirement = _requirement(report, "multi_image_input")
    assert input_requirement["evidence"]["raw_frame_count"] == 2
    fit_state_requirement = _requirement(report, "safetensors_primary_fit_state")
    assert fit_state_requirement["evidence"]["fit_state"].endswith(
        "outputs/fitting/fit_state.safetensors"
    )
    assert fit_state_requirement["evidence"]["legacy_npz_exists"] is False


def test_write_objective_compliance_audit_fails_on_current_loss_evidence(tmp_path) -> None:
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

    result = write_objective_compliance_audit(project_dir, force=True)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert not result.success
    assert not result.checks["losses_reduce"]
    assert not report["success"]
    loss_requirement = _requirement(report, "losses_reduce")
    assert loss_requirement["passed"] is False
    assert (
        loss_requirement["evidence"]["final_keypoint_loss"]
        == loss_requirement["evidence"]["initial_keypoint_loss"]
    )


def test_write_raw_objective_compliance_audit_fails_on_current_loss_evidence(tmp_path) -> None:
    project_dir = tmp_path / "project"
    input_dir = tmp_path / "inputs"
    _write_raw_inputs(input_dir)
    run_raw_pipeline(
        project_dir,
        input_paths=[input_dir],
        color_delta_threshold=0.10,
        camera_root_steps=2,
    )
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"
    fit_report = json.loads(fit_report_path.read_text(encoding="utf-8"))
    fit_report["metrics"]["final_silhouette_loss"] = fit_report["metrics"][
        "initial_silhouette_loss"
    ]
    fit_report_path.write_text(
        json.dumps(fit_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = write_raw_objective_compliance_audit(project_dir, force=True)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert not result.success
    assert not result.checks["losses_reduce"]
    assert not report["success"]
    loss_requirement = _requirement(report, "losses_reduce")
    assert loss_requirement["passed"] is False
    assert (
        loss_requirement["evidence"]["final_silhouette_loss"]
        == loss_requirement["evidence"]["initial_silhouette_loss"]
    )


def _requirement(report: dict, requirement_id: str) -> dict:
    for item in report["requirements"]:
        if item["id"] == requirement_id:
            return item
    raise AssertionError(f"missing requirement {requirement_id}")


def _write_raw_inputs(input_dir) -> None:
    input_dir.mkdir()
    _write_rgb_png(input_dir / "b.png", width=8, height=8, bbox=(2, 2, 6, 6))
    _write_rgb_png(input_dir / "a.png", width=8, height=8, bbox=(1, 1, 5, 5))


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = np.asarray([222, 220, 208], dtype=np.uint8)
    if x1 - x0 >= 4 and y1 - y0 >= 4:
        image[y0, x0 + 1, :] = np.asarray([118, 104, 54], dtype=np.uint8)
        image[y0, x1 - 2, :] = np.asarray([118, 104, 54], dtype=np.uint8)
        image[y0 + 1, x0 + 1, :] = np.asarray([10, 9, 8], dtype=np.uint8)
        image[y0 + 1, x1 - 2, :] = np.asarray([10, 9, 8], dtype=np.uint8)
        image[y0 + 2, x0 + 2, :] = np.asarray([224, 136, 146], dtype=np.uint8)
        image[y0 + 3, x0 + 2, :] = np.asarray([52, 36, 38], dtype=np.uint8)
    Image.fromarray(image).save(path)
