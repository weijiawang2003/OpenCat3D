import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file
from typer.testing import CliRunner

from opencat3d.camera.conventions import project_world_points
from opencat3d.cli import app
from opencat3d.fitting.losses import keypoint_reprojection_loss, soft_iou_loss
from opencat3d.fitting.optimizer import (
    OptimizerConfig,
    fit_staged,
    fit_to_checkpoint,
    frame_loss,
    frame_loss_components,
    load_fit_checkpoint,
    load_raw_fit_initialization,
    load_raw_fit_problem,
    load_synthetic_fit_initialization,
    load_synthetic_fit_problem,
    resume_fit_from_checkpoint,
    run_raw_fit,
    run_synthetic_fit,
    save_fit_checkpoint,
    total_loss,
    total_loss_components,
    validate_fit_state_safetensors,
)
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.io.cache import sha256_file
from opencat3d.preprocess.keypoints import (
    bootstrap_raw_keypoints_and_pose,
    bootstrap_synthetic_keypoints_and_pose,
)
from opencat3d.preprocess.segmentation import (
    segment_raw_image_observations,
    segment_synthetic_rgb_observations,
)
from opencat3d.synthetic.generator import create_synthetic_project
from opencat3d.synthetic.renderer import render_silhouette
from opencat3d.template.cat_template import CAT_TEMPLATE_VERTEX_COUNT


def test_keypoint_reprojection_loss_prefers_true_projection(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=4, image_size=96, seed=123)
    problem = load_synthetic_fit_problem(project_dir)
    observation = problem.observations[0]
    joints = problem.template.joints(problem.beta)

    true_projection = project_world_points(joints, observation.camera)
    shifted_projection = project_world_points(
        joints + np.asarray([0.25, 0.0, 0.0]),
        observation.camera,
    )

    true_loss = keypoint_reprojection_loss(
        true_projection,
        observation.keypoints_xyc,
        observation.bbox_xyxy,
        robust=False,
    )
    shifted_loss = keypoint_reprojection_loss(
        shifted_projection,
        observation.keypoints_xyc,
        observation.bbox_xyxy,
        robust=False,
    )

    assert true_loss < 1e-10
    assert shifted_loss > true_loss


def test_silhouette_loss_prefers_true_render(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=4, image_size=96, seed=123)
    problem = load_synthetic_fit_problem(project_dir)
    observation = problem.observations[0]
    vertices = problem.template.shaped_vertices(problem.beta)

    true_mask = render_silhouette(vertices, problem.template.faces, observation.camera)
    shifted_mask = render_silhouette(
        vertices + np.asarray([0.35, 0.0, 0.0]),
        problem.template.faces,
        observation.camera,
    )

    true_loss = soft_iou_loss(true_mask, observation.hard_mask)
    shifted_loss = soft_iou_loss(shifted_mask, observation.hard_mask)

    assert true_loss < 1e-6
    assert shifted_loss > true_loss


def test_staged_optimizer_reduces_camera_root_loss(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    problem = load_synthetic_fit_problem(project_dir)
    initial_translation = np.tile(np.asarray([0.18, -0.06, 0.20]), (len(problem.observations), 1))
    config = OptimizerConfig(
        camera_root_steps=35,
        lr=0.15,
        finite_difference_eps=2e-3,
        keypoint_weight=25.0,
        silhouette_weight=1.0,
    )

    initial_loss = total_loss(problem, initial_translation, config)
    result = fit_staged(
        problem,
        config=config,
        initial_root_translations=initial_translation,
    )

    assert np.isfinite(result.final_loss)
    assert result.final_loss < initial_loss * 0.35
    assert np.linalg.norm(result.root_translations, axis=1).mean() < 0.08
    assert result.poses_6d.shape == (3, problem.template.joints_rest.shape[0], 6)
    assert all(np.isfinite(values).all() for values in result.loss_history.values())


def test_staged_optimizer_recovers_shared_beta(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=4, image_size=64, seed=999)
    problem = load_synthetic_fit_problem(project_dir)
    initial_beta = np.zeros_like(problem.beta)
    config = OptimizerConfig(
        camera_root_steps=20,
        shape_steps=35,
        shape_lr=0.20,
        silhouette_weight=1.0,
        shape_prior_weight=0.0,
    )

    result = fit_staged(problem, config=config, initial_beta=initial_beta)

    initial_error = np.linalg.norm(initial_beta - problem.beta)
    final_error = np.linalg.norm(result.beta - problem.beta)
    assert np.isfinite(final_error)
    assert final_error < initial_error * 0.65
    assert result.loss_history["shape"][0] >= result.loss_history["shape"][-1]


def test_staged_optimizer_is_repeatable(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=4, image_size=64, seed=999)
    problem = load_synthetic_fit_problem(project_dir)
    initial_translation = np.tile(np.asarray([0.18, -0.06, 0.20]), (len(problem.observations), 1))
    initial_beta = np.zeros_like(problem.beta)
    config = OptimizerConfig(
        camera_root_steps=20,
        shape_steps=30,
        shape_lr=0.20,
        silhouette_weight=1.0,
        shape_prior_weight=0.0,
    )

    first = fit_staged(
        problem,
        config=config,
        initial_root_translations=initial_translation,
        initial_beta=initial_beta,
    )
    second = fit_staged(
        problem,
        config=config,
        initial_root_translations=initial_translation,
        initial_beta=initial_beta,
    )

    np.testing.assert_allclose(first.beta, second.beta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        first.root_translations,
        second.root_translations,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(first.poses_6d, second.poses_6d, rtol=0.0, atol=0.0)
    assert first.initial_loss == second.initial_loss
    assert first.final_loss == second.final_loss
    assert first.loss_history == second.loss_history


def test_checkpoint_resume_matches_uninterrupted_fit(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=4, image_size=64, seed=999)
    problem = load_synthetic_fit_problem(project_dir)
    initial_translation = np.tile(np.asarray([0.18, -0.06, 0.20]), (len(problem.observations), 1))
    initial_beta = np.zeros_like(problem.beta)
    config = OptimizerConfig(
        camera_root_steps=20,
        shape_steps=30,
        shape_lr=0.20,
        silhouette_weight=1.0,
        shape_prior_weight=0.0,
    )

    uninterrupted = fit_staged(
        problem,
        config=config,
        initial_root_translations=initial_translation,
        initial_beta=initial_beta,
    )
    checkpoint = fit_to_checkpoint(
        problem,
        config=config,
        initial_root_translations=initial_translation,
        initial_beta=initial_beta,
    )
    checkpoint_path = tmp_path / "fit_checkpoint.npz"
    save_fit_checkpoint(checkpoint, checkpoint_path)

    loaded = load_fit_checkpoint(checkpoint_path)
    resumed = resume_fit_from_checkpoint(problem, loaded, config=config)

    assert checkpoint_path.exists()
    assert loaded.stage == "after_camera_root"
    np.testing.assert_allclose(resumed.beta, uninterrupted.beta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        resumed.root_translations,
        uninterrupted.root_translations,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(resumed.poses_6d, uninterrupted.poses_6d, rtol=0.0, atol=0.0)
    assert resumed.initial_loss == uninterrupted.initial_loss
    assert resumed.final_loss == uninterrupted.final_loss
    assert resumed.loss_history == uninterrupted.loss_history


def test_load_synthetic_fit_initialization_reads_pose_seed_manifest(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=4, image_size=64, seed=999)
    segment_synthetic_rgb_observations(project_dir)
    bootstrap_synthetic_keypoints_and_pose(project_dir)
    problem = load_synthetic_fit_problem(project_dir)

    initialization = load_synthetic_fit_initialization(project_dir, problem)

    assert initialization is not None
    assert initialization.beta.shape == problem.beta.shape
    assert initialization.poses_6d.shape == (4, problem.template.joints_rest.shape[0], 6)
    assert initialization.frame_ids == tuple(
        observation.frame_id for observation in problem.observations
    )
    assert initialization.mean_confidence == 1.0


def test_load_raw_fit_problem_and_initialization_read_raw_manifests(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_initialized_project(project_dir)

    problem = load_raw_fit_problem(project_dir)
    initialization = load_raw_fit_initialization(project_dir, problem)

    assert len(problem.observations) == 2
    assert problem.beta.shape == (3,)
    assert problem.template.shaped_vertices(problem.beta).shape == (
        CAT_TEMPLATE_VERTEX_COUNT,
        3,
    )
    assert problem.observations[0].frame_id == "frame_0000"
    assert problem.observations[0].keypoints_xyc.shape == (11, 3)
    assert problem.observations[0].hard_mask.shape == (8, 8)
    assert 0.35 <= problem.observations[0].keypoint_weight_scale <= 1.0
    assert initialization is not None
    assert initialization.beta.shape == problem.beta.shape
    assert initialization.poses_6d.shape == (2, problem.template.joints_rest.shape[0], 6)
    assert initialization.frame_ids == tuple(
        observation.frame_id for observation in problem.observations
    )
    assert 0.0 < initialization.mean_confidence <= 0.5

    vertices = problem.template.shaped_vertices(problem.beta)
    joints = problem.template.joints(problem.beta)
    root_translation = np.asarray([0.2, 0.0, 0.0], dtype=np.float64)
    unweighted_components = frame_loss_components(
        vertices,
        joints,
        problem.template.faces,
        replace(problem.observations[0], keypoint_weight_scale=1.0),
        root_translation,
        OptimizerConfig(keypoint_weight=1.0, silhouette_weight=0.0),
    )
    weighted_components = frame_loss_components(
        vertices,
        joints,
        problem.template.faces,
        replace(problem.observations[0], keypoint_weight_scale=0.25),
        root_translation,
        OptimizerConfig(keypoint_weight=1.0, silhouette_weight=0.0),
    )
    assert np.isclose(weighted_components.keypoint, unweighted_components.keypoint * 0.25)


def test_load_raw_fit_problem_rebases_copied_project_local_paths(tmp_path) -> None:
    original_dir = tmp_path / "original"
    copied_dir = tmp_path / "copied"
    _create_raw_initialized_project(original_dir)
    shutil.copytree(original_dir, copied_dir)
    shutil.rmtree(original_dir)

    problem = load_raw_fit_problem(copied_dir)
    initialization = load_raw_fit_initialization(copied_dir, problem)

    assert len(problem.observations) == 2
    assert problem.observations[0].keypoints_xyc.shape == (11, 3)
    assert problem.observations[0].hard_mask.shape == (8, 8)
    assert initialization is not None
    assert initialization.poses_6d.shape == (2, problem.template.joints_rest.shape[0], 6)


def test_fit_problem_derives_conservative_feature_anchor_targets(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)

    problem = load_synthetic_fit_problem(project_dir)
    observed_anchor_targets = problem.observations[0].feature_anchors_xyc

    assert set(observed_anchor_targets) == {
        "left_ear_inner",
        "right_ear_inner",
        "left_front_paw",
        "right_front_paw",
        "left_hind_paw",
        "right_hind_paw",
        "tail_base",
        "tail_tip",
    }
    assert "left_eye_center" not in observed_anchor_targets
    assert "nose_tip" not in observed_anchor_targets
    assert all(len(value) == 3 for value in observed_anchor_targets.values())
    assert all(0.0 < value[2] <= 1.0 for value in observed_anchor_targets.values())


def test_feature_anchor_component_is_gated_and_weighted(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    problem = load_synthetic_fit_problem(project_dir)
    root_translations = np.tile(
        np.asarray([0.12, -0.03, 0.18], dtype=np.float64),
        (len(problem.observations), 1),
    )
    base_config = OptimizerConfig(
        keypoint_weight=0.0,
        silhouette_weight=0.0,
        shape_prior_weight=0.0,
        target_beta_weight=0.0,
    )

    default_components = total_loss_components(
        problem,
        root_translations,
        replace(base_config, feature_anchor_weight=0.0),
    )
    enabled_components = total_loss_components(
        problem,
        root_translations,
        replace(base_config, feature_anchor_weight=7.0),
    )

    assert default_components.feature_anchor == 0.0
    assert default_components.total == 0.0
    assert enabled_components.feature_anchor > 0.0
    assert np.isclose(
        enabled_components.total,
        7.0 * enabled_components.feature_anchor,
    )


def test_run_raw_fit_writes_report_and_state(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_initialized_project(project_dir)

    report = run_raw_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=2, shape_steps=0, silhouette_weight=1.0),
        initial_root_offset=np.asarray([0.0, 0.0, 0.0]),
    )

    fit_state_safetensors_path = project_dir / "outputs" / "fitting" / "fit_state.safetensors"
    assert report.success
    assert report.num_frames == 2
    assert report.metrics["fit_mode_raw"] == 1.0
    assert report.metrics["init_prediction_used"] == 1.0
    assert report.output_paths["fit_state"] == str(fit_state_safetensors_path)
    assert report.output_paths["fit_state_safetensors"].endswith(
        "outputs/fitting/fit_state.safetensors"
    )
    assert report.output_paths["init_manifest"].endswith("init/raw/manifest.json")
    assert report.output_paths["overlay_manifest"].endswith(
        "outputs/fitting/per_frame_overlay/manifest.json"
    )
    assert report.metrics["overlay_frame_count"] == 2.0
    assert report.metrics["overlay_mean_silhouette_iou"] >= 0.0
    assert np.isfinite(report.metrics["final_loss"])
    overlay_manifest = json.loads(
        Path(report.output_paths["overlay_manifest"]).read_text(encoding="utf-8")
    )
    assert overlay_manifest["frame_count"] == 2
    assert len(overlay_manifest["frames"]) == 2
    assert overlay_manifest["mean_silhouette_iou"] >= 0.0
    first_overlay_path = Path(overlay_manifest["frames"][0]["overlay_path"])
    assert first_overlay_path.exists()
    assert Image.open(first_overlay_path).size == (8, 8)
    assert not (project_dir / "outputs" / "fitting" / "fit_state.npz").exists()
    safetensors_state = load_file(fit_state_safetensors_path)
    assert safetensors_state["beta"].shape == (3,)
    assert safetensors_state["target_beta"].shape == (3,)
    assert safetensors_state["root_translations"].shape == (2, 3)
    assert safetensors_state["poses_6d"].shape == (2, 11, 6)
    fit_state_validation = validate_fit_state_safetensors(
        fit_state_safetensors_path,
        expected_mode="raw",
        expected_frame_count=2,
        expected_joint_count=11,
    )
    assert fit_state_validation.success
    assert fit_state_validation.metadata["mode"] == "raw"


def test_fit_cli_runs_raw_project(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "project"
    _create_raw_initialized_project(project_dir)

    result = runner.invoke(
        app,
        [
            "fit",
            str(project_dir),
            "--mode",
            "raw",
            "--camera-root-steps",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "final_loss=" in result.output
    assert (project_dir / "outputs" / "fitting" / "fit_report.json").exists()
    assert (
        project_dir / "outputs" / "fitting" / "per_frame_overlay" / "manifest.json"
    ).exists()


def test_run_synthetic_fit_uses_init_predictions_when_available(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    segment_synthetic_rgb_observations(project_dir)
    bootstrap_synthetic_keypoints_and_pose(project_dir)
    seed_beta = np.asarray([0.03, -0.02, 0.01], dtype=np.float32)
    pose_marker = 0.5
    _rewrite_all_shape_codes(project_dir, seed_beta)
    _rewrite_first_pose_marker(project_dir, pose_marker)

    report = run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=2, shape_steps=0, silhouette_weight=1.0),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )

    fit_state_safetensors_path = project_dir / "outputs" / "fitting" / "fit_state.safetensors"
    assert report.metrics["init_prediction_used"] == 1.0
    assert report.metrics["init_prediction_mean_confidence"] == 1.0
    assert report.output_paths["init_manifest"].endswith("init/synthetic/manifest.json")
    safetensors_state = load_file(fit_state_safetensors_path)
    np.testing.assert_allclose(safetensors_state["initial_beta"], seed_beta)
    assert safetensors_state["poses_6d"][0, 0, 0] == pose_marker
    assert not (project_dir / "outputs" / "fitting" / "fit_state.npz").exists()


def test_run_synthetic_fit_records_loss_component_diagnostics(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)

    report = run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=30, silhouette_weight=1.0),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )

    assert report.metrics["final_keypoint_loss"] < report.metrics["initial_keypoint_loss"]
    assert report.metrics["final_silhouette_loss"] < report.metrics["initial_silhouette_loss"]
    assert report.metrics["keypoint_loss_reduction"] > 0.0
    assert report.metrics["silhouette_loss_reduction"] > 0.0
    assert report.metrics["feature_anchor_weight"] == 0.0
    assert report.metrics["mean_feature_anchor_target_count"] == 8.0
    assert report.metrics["initial_feature_anchor_loss"] == 0.0
    assert report.metrics["final_feature_anchor_loss"] == 0.0
    assert len(report.loss_history["keypoint_component"]) == 3
    assert len(report.loss_history["silhouette_component"]) == 3
    assert len(report.loss_history["feature_anchor_component"]) == 3
    assert report.loss_history["feature_anchor_component"] == [0.0, 0.0, 0.0]
    assert (
        report.loss_history["keypoint_component"][0]
        > report.loss_history["keypoint_component"][-1]
    )
    assert (
        report.loss_history["silhouette_component"][0]
        > report.loss_history["silhouette_component"][-1]
    )


def test_frame_loss_is_finite_for_empty_or_perturbed_translation(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    problem = load_synthetic_fit_problem(project_dir)
    observation = problem.observations[0]
    vertices = problem.template.shaped_vertices(problem.beta)
    joints = problem.template.joints(problem.beta)

    loss = frame_loss(
        vertices,
        joints,
        problem.template.faces,
        observation,
        np.asarray([0.0, 0.0, 0.0]),
    )

    assert np.isfinite(loss)
    assert loss >= 0.0


def test_run_synthetic_fit_writes_report_and_state(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)

    report = run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=25, silhouette_weight=1.0),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )

    fit_state_safetensors_path = project_dir / "outputs" / "fitting" / "fit_state.safetensors"
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"
    assert report.success
    assert not (project_dir / "outputs" / "fitting" / "fit_state.npz").exists()
    assert fit_state_safetensors_path.exists()
    assert fit_report_path.exists()
    assert report.output_paths["fit_state"] == str(fit_state_safetensors_path)
    assert report.output_paths["fit_state_safetensors"] == str(fit_state_safetensors_path)
    assert report.metrics["final_loss"] < report.metrics["initial_loss"]
    assert report.metrics["final_beta_error"] < report.metrics["initial_beta_error"]
    safetensors_state = load_file(fit_state_safetensors_path)
    assert set(safetensors_state) == {
        "beta",
        "target_beta",
        "initial_beta",
        "root_translations",
        "initial_root_translations",
        "poses_6d",
        "initial_poses_6d",
    }
    assert safetensors_state["beta"].shape == (3,)
    assert safetensors_state["root_translations"].shape == (3, 3)
    assert safetensors_state["poses_6d"].shape == (3, 11, 6)
    fit_state_validation = validate_fit_state_safetensors(
        fit_state_safetensors_path,
        expected_mode="synthetic",
        expected_frame_count=3,
        expected_joint_count=11,
    )
    assert fit_state_validation.success
    assert all(fit_state_validation.checks.values())


def _rewrite_all_shape_codes(project_dir, seed_beta: np.ndarray) -> None:
    manifest = _read_init_manifest(project_dir)
    for frame in manifest["frames"]:
        init_path = Path(frame["init_prediction"])
        init_meta = _read_json(init_path)
        shape_path = Path(init_meta["shape_code"]["path"])
        np.savez_compressed(shape_path, shape_code=seed_beta)
        init_meta["shape_code"]["sha256"] = sha256_file(shape_path)
        _write_json(init_path, init_meta)


def _rewrite_first_pose_marker(project_dir, pose_marker: float) -> None:
    manifest = _read_init_manifest(project_dir)
    init_path = Path(manifest["frames"][0]["init_prediction"])
    init_meta = _read_json(init_path)
    pose_path = Path(init_meta["pose_6d"]["path"])
    with np.load(pose_path) as data:
        pose = np.asarray(data["pose_6d"], dtype=np.float32)
    pose[0, 0] = pose_marker
    np.savez_compressed(pose_path, pose_6d=pose)
    init_meta["pose_6d"]["sha256"] = sha256_file(pose_path)
    _write_json(init_path, init_meta)


def _read_init_manifest(project_dir):
    return _read_json(project_dir / "init" / "synthetic" / "manifest.json")


def _read_json(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _create_raw_initialized_project(project_dir) -> None:
    input_dir = project_dir.parent / "inputs"
    input_dir.mkdir()
    _write_rgb_png(input_dir / "b.png", width=8, height=8, bbox=(2, 2, 6, 6))
    _write_rgb_png(input_dir / "a.png", width=8, height=8, bbox=(1, 1, 5, 5))
    ingest_image_sources(project_dir, [input_dir], min_frames=2)
    segment_raw_image_observations(project_dir, color_delta_threshold=0.10)
    bootstrap_raw_keypoints_and_pose(project_dir)


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = 51
    Image.fromarray(image).save(path)
