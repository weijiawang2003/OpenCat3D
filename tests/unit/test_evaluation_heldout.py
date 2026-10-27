import json

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from opencat3d.evaluation.heldout import (
    SYNTHETIC_HELDOUT_THRESHOLDS,
    evaluate_synthetic_heldout_views,
)
from opencat3d.export.mesh import export_fitted_canonical_mesh
from opencat3d.fitting.optimizer import OptimizerConfig, run_synthetic_fit
from opencat3d.preprocess.keypoints import bootstrap_synthetic_keypoints_and_pose
from opencat3d.preprocess.segmentation import segment_synthetic_rgb_observations
from opencat3d.synthetic.generator import create_synthetic_project


def _create_fitted_project(project_dir) -> None:
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    segment_synthetic_rgb_observations(project_dir)
    bootstrap_synthetic_keypoints_and_pose(project_dir)
    run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=30, silhouette_weight=1.0),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )


def test_evaluate_synthetic_heldout_views_writes_success_report(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    _create_fitted_project(project_dir)

    result = evaluate_synthetic_heldout_views(project_dir)

    assert result.success
    assert result.report_path.exists()
    assert result.checks["heldout_silhouette_iou_threshold"]
    assert result.checks["heldout_keypoint_reprojection_threshold"]
    assert result.metrics["heldout_frame_count"] == 1
    assert (
        result.metrics["heldout_mean_silhouette_iou"]
        >= SYNTHETIC_HELDOUT_THRESHOLDS["min_heldout_silhouette_iou"]
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["heldout_frame_ids"] == ["synthetic_002"]
    assert report["artifacts"]["fit_state"].endswith("outputs/fitting/fit_state.safetensors")
    assert report["artifacts"]["fit_state_source_format"] == "safetensors"
    assert report["success"] is True


def test_evaluate_synthetic_heldout_views_fails_shifted_fit_state(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    _create_fitted_project(project_dir)
    fit_state_path = project_dir / "outputs" / "fitting" / "fit_state.safetensors"
    with safe_open(fit_state_path, framework="numpy") as handle:
        metadata = dict(handle.metadata() or {})
        payload = {key: np.asarray(handle.get_tensor(key)).copy() for key in handle.keys()}
    root_translations = np.asarray(payload["root_translations"], dtype=np.float64)
    root_translations[-1, 0] += 1.5
    payload["root_translations"] = root_translations
    save_file(payload, fit_state_path, metadata=metadata)

    result = evaluate_synthetic_heldout_views(project_dir, force=True)

    assert not result.success
    assert not result.checks["heldout_silhouette_iou_threshold"]


def test_evaluate_synthetic_reconstruction_includes_heldout_report(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    _create_fitted_project(project_dir)
    export_fitted_canonical_mesh(project_dir)

    from opencat3d.evaluation.synthetic import evaluate_synthetic_reconstruction

    result = evaluate_synthetic_reconstruction(project_dir)

    assert result.success
    assert result.checks["heldout_view_consistency"]
    assert result.metrics["heldout_mean_silhouette_iou"] >= 0.80
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["artifacts"]["heldout_evaluation"].endswith(
        "synthetic_heldout_evaluation.json"
    )
