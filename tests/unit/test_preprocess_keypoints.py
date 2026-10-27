import json

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from opencat3d.cli import app
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.preprocess.keypoints import (
    bootstrap_raw_keypoints_and_pose,
    bootstrap_synthetic_keypoints_and_pose,
    validate_keypoints_xyc,
)
from opencat3d.preprocess.segmentation import (
    segment_raw_image_observations,
    segment_synthetic_rgb_observations,
)
from opencat3d.schemas import FrameObservationMeta, InitPredictionMeta
from opencat3d.synthetic.generator import create_synthetic_project
from opencat3d.template.cat_template import CAT_TEMPLATE_VERTEX_COUNT


def test_validate_keypoints_xyc_accepts_visible_points_in_bbox() -> None:
    keypoints = np.asarray(
        [
            [4.0, 5.0, 1.0],
            [8.0, 9.0, 0.5],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    validation = validate_keypoints_xyc(
        keypoints,
        image_width=16,
        image_height=16,
        bbox_xyxy=[2.0, 3.0, 10.0, 12.0],
    )

    assert validation.visible_count == 2
    assert validation.visible_fraction == 2 / 3
    assert validation.in_bounds_fraction == 1.0
    assert validation.in_bbox_fraction == 1.0
    assert validation.mean_confidence == 0.5


def test_validate_keypoints_xyc_rejects_visible_points_outside_bbox() -> None:
    keypoints = np.asarray([[12.0, 5.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="bbox_xyxy"):
        validate_keypoints_xyc(
            keypoints,
            image_width=16,
            image_height=16,
            bbox_xyxy=[2.0, 3.0, 10.0, 12.0],
        )


def test_bootstrap_synthetic_keypoints_and_pose_writes_init_predictions(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=4, image_size=64, seed=999)
    segment_synthetic_rgb_observations(project_dir)

    result = bootstrap_synthetic_keypoints_and_pose(project_dir)

    assert result.frame_count == 4
    assert result.mean_keypoint_confidence == 1.0
    assert result.mean_in_bbox_fraction == 1.0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 4
    assert len(manifest["frames"]) == 4
    first_meta = InitPredictionMeta.model_validate_json(
        result.frames[0].init_meta_path.read_text(encoding="utf-8")
    )
    assert first_meta.source_backend == "opencat3d.synthetic_keypoint_bootstrap"
    assert first_meta.confidence == 1.0
    assert first_meta.shape_code.shape == [3]
    assert first_meta.pose_6d.shape == [11, 6]
    assert first_meta.global_orient_6d.shape == [6]
    assert first_meta.vertices.shape == [CAT_TEMPLATE_VERTEX_COUNT, 3]
    assert first_meta.joints.shape == [11, 3]
    assert first_meta.per_joint_confidence.shape == [11]
    with np.load(first_meta.pose_6d.path) as data:
        pose = data["pose_6d"]
    np.testing.assert_array_equal(
        pose[0],
        np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )


def test_bootstrap_raw_keypoints_and_pose_writes_init_predictions(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_segmented_project(project_dir)

    result = bootstrap_raw_keypoints_and_pose(project_dir, keypoint_confidence=0.5)

    assert result.frame_count == 2
    assert 0.0 < result.mean_keypoint_confidence <= 0.5
    assert result.mean_in_bbox_fraction == 1.0
    assert result.photo_frame_count == 2
    assert result.video_frame_count == 0
    assert result.enriched_keypoint_frame_count == 2
    assert result.enriched_video_keypoint_frame_count == 0
    assert 0.0 < result.mean_frontal_face_confidence <= 1.0
    assert 0.35 <= result.mean_keypoint_weight_scale <= 1.0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "opencat3d.raw_bbox_keypoint_bootstrap"
    assert manifest["frame_count"] == 2
    assert manifest["photo_frame_count"] == 2
    assert manifest["video_frame_count"] == 0
    assert manifest["enriched_keypoint_frame_count"] == 2
    assert manifest["enriched_video_keypoint_frame_count"] == 0
    assert manifest["mean_frontal_face_confidence"] == pytest.approx(
        result.mean_frontal_face_confidence
    )
    assert manifest["mean_keypoint_weight_scale"] == pytest.approx(
        result.mean_keypoint_weight_scale
    )
    assert (project_dir / "assets" / "cat_template" / "placeholder_cat_template.npz").exists()
    first_meta = InitPredictionMeta.model_validate_json(
        result.frames[0].init_meta_path.read_text(encoding="utf-8")
    )
    assert first_meta.source_backend == "opencat3d.raw_bbox_keypoint_bootstrap"
    assert 0.0 < first_meta.confidence <= 0.5
    assert first_meta.camera.image_width == 12
    assert first_meta.camera.image_height == 10
    assert first_meta.shape_code.shape == [3]
    assert first_meta.pose_6d.shape == [11, 6]
    assert first_meta.vertices.shape == [CAT_TEMPLATE_VERTEX_COUNT, 3]
    assert first_meta.joints.shape == [11, 3]
    observation_meta = FrameObservationMeta.model_validate_json(
        (project_dir / "observations" / "raw" / "frame_0000" / "observation.json").read_text(
            encoding="utf-8"
        )
    )
    assert observation_meta.quality["raw_keypoint_bootstrap"] == 1.0
    assert observation_meta.quality["raw_keypoint_base_confidence"] == 0.5
    assert 0.0 < observation_meta.quality["raw_frontal_face_confidence"] <= 1.0
    assert 0.35 <= observation_meta.quality["raw_keypoint_view_weight"] <= 1.0
    assert observation_meta.quality["keypoint_confidence"] == pytest.approx(
        0.5 * observation_meta.quality["raw_keypoint_view_weight"]
    )
    assert first_meta.confidence == pytest.approx(
        observation_meta.quality["keypoint_confidence"]
    )
    with np.load(observation_meta.keypoints_xyc.path) as data:
        keypoints = data["keypoints_xyc"]
    assert keypoints.shape == (11, 3)
    np.testing.assert_allclose(keypoints[:, 2], observation_meta.quality["keypoint_confidence"])
    x0, y0, x1, y1 = observation_meta.bbox_xyxy
    assert np.all((keypoints[:, 0] >= x0) & (keypoints[:, 0] < x1))
    assert np.all((keypoints[:, 1] >= y0) & (keypoints[:, 1] < y1))
    raw_manifest = json.loads(
        (project_dir / "observations" / "raw" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw_manifest["keypoint_bootstrap"]["enriched_keypoint_frame_count"] == 2
    assert raw_manifest["keypoint_bootstrap"]["mean_keypoint_weight_scale"] == pytest.approx(
        result.mean_keypoint_weight_scale
    )
    assert raw_manifest["frames"][0]["keypoints_xyc"]["shape"] == [11, 3]
    assert raw_manifest["frames"][0]["quality"]["raw_keypoint_bootstrap"] == 1.0
    assert raw_manifest["frames"][0]["raw_view_label"] == "frontal_face_candidate"
    assert raw_manifest["frames"][0]["raw_keypoint_view_weight"] == pytest.approx(
        raw_manifest["frames"][0]["quality"]["raw_keypoint_view_weight"]
    )


def test_bootstrap_raw_keypoints_and_pose_enriches_video_frames_in_manifest(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    _create_raw_segmented_project(project_dir)
    _mark_first_raw_frame_as_video(project_dir)

    result = bootstrap_raw_keypoints_and_pose(project_dir, keypoint_confidence=0.5)

    assert result.photo_frame_count == 1
    assert result.video_frame_count == 1
    assert result.enriched_keypoint_frame_count == 2
    assert result.enriched_video_keypoint_frame_count == 1
    raw_manifest = json.loads(
        (project_dir / "observations" / "raw" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw_manifest["keypoint_bootstrap"]["video_frame_count"] == 1
    assert raw_manifest["keypoint_bootstrap"]["enriched_video_keypoint_frame_count"] == 1
    video_frame = raw_manifest["frames"][0]
    assert video_frame["media_type"] == "video"
    assert video_frame["raw_view_label"].startswith("video_")
    assert video_frame["keypoints_xyc"]["shape"] == [11, 3]
    assert video_frame["quality"]["raw_keypoint_bootstrap"] == 1.0
    assert video_frame["quality"]["raw_keypoint_base_confidence"] == 0.5
    assert 0.35 <= video_frame["raw_keypoint_view_weight"] <= 1.0
    with np.load(video_frame["keypoints_xyc"]["path"]) as data:
        keypoints = data["keypoints_xyc"]
    assert keypoints.shape == (11, 3)
    np.testing.assert_allclose(keypoints[:, 2], video_frame["quality"]["keypoint_confidence"])


def test_bootstrap_raw_keypoints_and_pose_is_repeatable_with_force(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_segmented_project(project_dir)

    first = bootstrap_raw_keypoints_and_pose(project_dir)
    second = bootstrap_raw_keypoints_and_pose(project_dir, force=True)

    assert first.frame_count == second.frame_count
    assert first.mean_keypoint_confidence == second.mean_keypoint_confidence
    assert [frame.visible_count for frame in first.frames] == [
        frame.visible_count for frame in second.frames
    ]


def test_preprocess_bootstrap_raw_cli_runs_after_segmentation(tmp_path) -> None:
    runner = CliRunner()
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    project_dir = tmp_path / "project"
    _write_rgb_png(image_a, width=8, height=8, bbox=(2, 2, 6, 6))
    _write_rgb_png(image_b, width=8, height=8, bbox=(1, 1, 5, 5))

    ingest_result = runner.invoke(
        app,
        [
            "ingest",
            str(project_dir),
            "--input",
            str(image_a),
            "--input",
            str(image_b),
        ],
    )
    segment_result = runner.invoke(
        app,
        ["preprocess", "segment-raw", str(project_dir), "--color-delta-threshold", "0.10"],
    )
    bootstrap_result = runner.invoke(app, ["preprocess", "bootstrap-raw", str(project_dir)])

    assert ingest_result.exit_code == 0, ingest_result.output
    assert segment_result.exit_code == 0, segment_result.output
    assert bootstrap_result.exit_code == 0, bootstrap_result.output
    assert "raw keypoint bootstrap complete frames=2" in bootstrap_result.output
    assert (project_dir / "init" / "raw" / "manifest.json").exists()


def test_bootstrap_synthetic_keypoints_and_pose_is_repeatable_with_force(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=123)
    segment_synthetic_rgb_observations(project_dir)

    first = bootstrap_synthetic_keypoints_and_pose(project_dir)
    second = bootstrap_synthetic_keypoints_and_pose(project_dir, force=True)

    assert first.frame_count == second.frame_count
    assert first.mean_keypoint_confidence == second.mean_keypoint_confidence
    assert [frame.visible_count for frame in first.frames] == [
        frame.visible_count for frame in second.frames
    ]


def _create_raw_segmented_project(project_dir) -> None:
    input_dir = project_dir.parent / "inputs"
    input_dir.mkdir()
    _write_rgb_png(input_dir / "b.png", width=12, height=10, bbox=(4, 3, 9, 8))
    _write_rgb_png(input_dir / "a.png", width=12, height=10, bbox=(3, 2, 8, 7))
    ingest_image_sources(project_dir, [input_dir], min_frames=2)
    segment_raw_image_observations(project_dir, color_delta_threshold=0.10)


def _mark_first_raw_frame_as_video(project_dir) -> None:
    manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["source_type"] = "video_frame"
    manifest["frames"][0]["media_type"] = "video"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = 51
    Image.fromarray(image).save(path)
