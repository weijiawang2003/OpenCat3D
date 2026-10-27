import json

import numpy as np
from PIL import Image

from opencat3d.evaluation.residuals import (
    RESIDUAL_LEGEND,
    SILHOUETTE_RESIDUAL_THRESHOLDS,
    evaluate_silhouette_residual_maps,
)
from opencat3d.export.mesh import export_fitted_canonical_mesh
from opencat3d.fitting.optimizer import OptimizerConfig, run_raw_fit, run_synthetic_fit
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.preprocess.keypoints import bootstrap_raw_keypoints_and_pose
from opencat3d.preprocess.segmentation import segment_raw_image_observations
from opencat3d.synthetic.generator import create_synthetic_project


def test_evaluate_silhouette_residual_maps_writes_synthetic_artifacts(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=20),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )

    result = evaluate_silhouette_residual_maps(project_dir, mode="synthetic")

    assert result.success
    assert result.report_path.name == "synthetic_silhouette_residuals.json"
    assert result.manifest_path.exists()
    assert result.metrics["frame_count"] == 3
    assert (
        result.metrics["mean_union_error_fraction"]
        <= SILHOUETTE_RESIDUAL_THRESHOLDS["synthetic_max_mean_union_error_fraction"]
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert report["mode"] == "synthetic"
    assert report["artifacts"]["fit_state"].endswith("outputs/fitting/fit_state.safetensors")
    assert report["artifacts"]["fit_state_source_format"] == "safetensors"
    assert manifest["legend"] == RESIDUAL_LEGEND
    assert len(report["frames"]) == 3
    first_path = result.frames[0].residual_path
    assert first_path.exists()
    first_image = Image.open(first_path)
    assert first_image.mode == "RGB"
    assert first_image.size == (64, 64)


def test_evaluate_silhouette_residual_maps_writes_raw_artifacts(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_project(project_dir)

    result = evaluate_silhouette_residual_maps(project_dir, mode="raw")

    assert result.success
    assert result.report_path.name == "raw_silhouette_residuals.json"
    assert all(result.checks.values())
    assert result.metrics["frame_count"] == 2
    assert (
        result.metrics["mean_union_error_fraction"]
        <= SILHOUETTE_RESIDUAL_THRESHOLDS["raw_max_mean_union_error_fraction"]
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["artifacts"]["fit_state"].endswith("outputs/fitting/fit_state.safetensors")
    assert report["artifacts"]["fit_state_source_format"] == "safetensors"
    assert report["artifacts"]["manifest"].endswith(
        "outputs/quality/raw_silhouette_residuals/manifest.json"
    )
    assert all(str(frame["residual_path"]).endswith("_residual.png") for frame in report["frames"])


def _create_raw_fitted_project(project_dir) -> None:
    input_dir = project_dir.parent / "inputs"
    input_dir.mkdir()
    _write_rgb_png(input_dir / "b.png", width=8, height=8, bbox=(2, 2, 6, 6))
    _write_rgb_png(input_dir / "a.png", width=8, height=8, bbox=(1, 1, 5, 5))
    ingest_image_sources(project_dir, [input_dir], min_frames=2)
    segment_raw_image_observations(project_dir, color_delta_threshold=0.10)
    bootstrap_raw_keypoints_and_pose(project_dir)
    run_raw_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=2, shape_steps=0, silhouette_weight=1.0),
        initial_root_offset=np.asarray([0.0, 0.0, 0.0]),
    )
    export_fitted_canonical_mesh(project_dir, mode="raw")


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = 51
    Image.fromarray(image).save(path)
