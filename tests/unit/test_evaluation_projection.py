import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from opencat3d.evaluation.projection import (
    JOINT_PROJECTION_THRESHOLDS,
    evaluate_joint_projection_report,
)
from opencat3d.export.mesh import export_fitted_canonical_mesh
from opencat3d.fitting.optimizer import OptimizerConfig, run_raw_fit, run_synthetic_fit
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.preprocess.keypoints import bootstrap_raw_keypoints_and_pose
from opencat3d.preprocess.segmentation import segment_raw_image_observations
from opencat3d.synthetic.generator import create_synthetic_project


def test_evaluate_joint_projection_report_writes_synthetic_quality_report(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=20),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )
    export_fitted_canonical_mesh(project_dir)

    result = evaluate_joint_projection_report(project_dir, mode="synthetic")

    assert result.success
    assert result.report_path.name == "synthetic_joint_projection.json"
    assert all(result.checks.values())
    assert result.metrics["frame_count"] == 3
    assert result.metrics["joint_count"] == 11
    assert (
        result.metrics["mean_keypoint_reprojection_loss"]
        <= JOINT_PROJECTION_THRESHOLDS["synthetic_max_mean_keypoint_reprojection_loss"]
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "synthetic"
    assert report["artifacts"]["fit_state"].endswith("outputs/fitting/fit_state.safetensors")
    assert report["artifacts"]["fit_state_source_format"] == "safetensors"
    assert report["artifacts"]["skeleton"].endswith("outputs/avatar/skeleton.json")
    assert len(report["frames"]) == 3
    assert report["frames"][0]["positive_depth_fraction"] == 1.0


def test_evaluate_joint_projection_report_writes_raw_quality_report(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_exported_project(project_dir)

    result = evaluate_joint_projection_report(project_dir, mode="raw")

    assert result.success
    assert result.report_path.name == "raw_joint_projection.json"
    assert all(result.checks.values())
    assert result.metrics["frame_count"] == 2
    assert result.metrics["joint_count"] == 11
    assert (
        result.metrics["mean_keypoint_reprojection_loss"]
        <= JOINT_PROJECTION_THRESHOLDS["raw_max_mean_keypoint_reprojection_loss"]
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "raw"
    assert report["artifacts"]["fit_state"].endswith("outputs/fitting/fit_state.safetensors")
    assert report["artifacts"]["fit_state_source_format"] == "safetensors"
    assert len(report["frames"]) == 2
    assert min(frame["visible_joint_count"] for frame in report["frames"]) == 11


def test_evaluate_joint_projection_report_ignores_sparse_zero_confidence_raw_keypoints(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_exported_project(project_dir)
    _replace_first_raw_observation_with_sparse_placeholder(project_dir)

    result = evaluate_joint_projection_report(project_dir, mode="raw")

    assert result.success
    assert all(result.checks.values())
    assert result.metrics["frame_count"] == 2
    assert result.metrics["observed_keypoint_frame_count"] == 1
    assert result.metrics["sparse_keypoint_frame_count"] == 1
    assert result.metrics["missing_keypoint_frame_count"] == 1
    assert result.metrics["mean_keypoint_reprojection_loss"] > 0.0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["frames"][0]["source_keypoint_count"] == 1
    assert report["frames"][0]["evaluated_keypoint_count"] == 1
    assert report["frames"][0]["visible_joint_count"] == 0
    assert report["frames"][1]["visible_joint_count"] == 11


def test_evaluate_joint_projection_report_fails_when_skeleton_is_stale(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=20),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )
    export_fitted_canonical_mesh(project_dir)
    skeleton_path = project_dir / "outputs" / "avatar" / "skeleton.json"
    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
    skeleton["joints"][0]["position"][0] += 0.1
    skeleton_path.write_text(json.dumps(skeleton, indent=2, sort_keys=True) + "\n")

    result = evaluate_joint_projection_report(project_dir, mode="synthetic")

    assert not result.success
    assert not result.checks["skeleton_matches_fit_state"]
    assert result.checks["skeleton_metadata_valid"]


def _create_raw_fitted_exported_project(project_dir) -> None:
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


def _replace_first_raw_observation_with_sparse_placeholder(project_dir) -> None:
    manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_frame = manifest["frames"][0]
    observation_path = Path(str(first_frame["observation_path"]))
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    keypoint_path = Path(str(observation["keypoints_xyc"]["path"]))
    np.savez_compressed(
        keypoint_path,
        keypoints_xyc=np.zeros((1, 3), dtype=np.float32),
    )
    observation["keypoints_xyc"]["shape"] = [1, 3]
    observation["keypoints_xyc"]["sha256"] = _sha256_file(keypoint_path)
    observation["quality"]["keypoint_confidence"] = 0.0
    observation_path.write_text(
        json.dumps(observation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = 51
    Image.fromarray(image).save(path)
