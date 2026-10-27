import json

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from opencat3d.cli import app
from opencat3d.fitting.optimizer import load_synthetic_fit_problem
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.preprocess.segmentation import (
    bbox_from_mask,
    segment_raw_foreground,
    segment_raw_image_observations,
    segment_synthetic_rgb,
    segment_synthetic_rgb_observations,
)
from opencat3d.schemas import FrameObservationMeta
from opencat3d.synthetic.generator import create_synthetic_project


def test_segment_synthetic_rgb_thresholds_foreground_and_bbox() -> None:
    image = np.zeros((3, 8, 10), dtype=np.float32)
    image[:, 2:6, 3:8] = 0.7

    mask = segment_synthetic_rgb(image)

    assert mask.shape == (8, 10)
    assert mask.dtype == np.float32
    assert mask.sum() == 20.0
    assert bbox_from_mask(mask) == (3.0, 2.0, 8.0, 6.0)


def test_segment_raw_foreground_uses_border_background_color() -> None:
    image = np.ones((3, 8, 10), dtype=np.float32)
    image[:, 2:6, 3:8] = 0.2

    mask = segment_raw_foreground(image, color_delta_threshold=0.10)

    assert mask.sum() == 20.0
    assert bbox_from_mask(mask) == (3.0, 2.0, 8.0, 6.0)


def test_segment_raw_image_observations_writes_raw_observation_contract(tmp_path) -> None:
    project_dir = tmp_path / "project"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_rgb_png(input_dir / "b.png", width=12, height=10, bbox=(4, 3, 9, 8))
    _write_rgb_png(input_dir / "a.png", width=12, height=10, bbox=(3, 2, 8, 7))
    ingest_image_sources(project_dir, [input_dir], min_frames=2)

    result = segment_raw_image_observations(project_dir, color_delta_threshold=0.10)

    assert result.frame_count == 2
    assert result.manifest_path == project_dir / "observations" / "raw" / "manifest.json"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "opencat3d.raw_foreground_segmentation"
    assert manifest["segmentation_backend"] == "auto_subject_v1"
    assert manifest["frame_count"] == 2
    assert np.isclose(manifest["mean_mask_area_ratio"], 25.0 / 120.0)
    assert np.isclose(manifest["mean_bbox_area_ratio"], 25.0 / 120.0)
    assert manifest["full_frame_bbox_fraction"] == 0.0
    assert np.isclose(result.mean_bbox_area_ratio, 25.0 / 120.0)
    assert result.full_frame_bbox_fraction == 0.0
    first = manifest["frames"][0]
    assert first["bbox_xyxy"] == [3.0, 2.0, 8.0, 7.0]
    assert first["image_width"] == 12
    assert first["image_height"] == 10
    assert first["segmentation_backend"] == "alpha_or_border_color_v1"
    meta = FrameObservationMeta.model_validate_json(
        (project_dir / "observations" / "raw" / "frame_0000" / "observation.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta.quality["raw_foreground_segmentation"] == 1.0
    assert np.isclose(meta.quality["bbox_area_ratio"], 25.0 / 120.0)
    assert meta.quality["touches_image_border"] == 0.0
    assert meta.quality["full_frame_bbox"] == 0.0
    with np.load(meta.image_srgb_path) as data:
        assert data["image_srgb"].shape == (3, 10, 12)
    with np.load(meta.hard_mask.path) as data:
        hard_mask = data["hard_mask"]
    with np.load(meta.keypoints_xyc.path) as data:
        keypoints = data["keypoints_xyc"]
    assert hard_mask.shape == (1, 10, 12)
    assert hard_mask.sum() == 25.0
    assert keypoints.shape == (1, 3)
    assert keypoints.sum() == 0.0


def test_segment_raw_image_observations_can_use_white_subject_backend(tmp_path) -> None:
    project_dir = tmp_path / "project"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_gray_room_white_cat(input_dir / "a.png")
    _write_gray_room_white_cat(input_dir / "b.png")
    ingest_image_sources(project_dir, [input_dir], min_frames=2)

    result = segment_raw_image_observations(
        project_dir,
        segmentation_backend="white_subject_v1",
    )

    assert result.frame_count == 2
    assert result.frames[0].backend == "white_subject_v1"
    assert result.frames[0].bbox_area_ratio < 0.40
    assert not result.frames[0].full_frame_bbox


def test_segment_raw_image_observations_uses_alpha_mask(tmp_path) -> None:
    project_dir = tmp_path / "project"
    image_path = tmp_path / "alpha.png"
    rgba = np.zeros((6, 8, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[1:5, 2:7, 3] = 255
    Image.fromarray(rgba).save(image_path)
    ingest_image_sources(project_dir, [image_path], min_frames=1)

    result = segment_raw_image_observations(project_dir)

    assert result.frame_count == 1
    assert result.frames[0].bbox_xyxy == (2.0, 1.0, 7.0, 5.0)
    assert np.isclose(result.frames[0].mask_area_ratio, 20.0 / 48.0)


def test_preprocess_segment_raw_cli_runs_after_ingest(tmp_path) -> None:
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

    assert ingest_result.exit_code == 0, ingest_result.output
    assert segment_result.exit_code == 0, segment_result.output
    assert "raw segmentation complete frames=2" in segment_result.output
    assert (project_dir / "observations" / "raw" / "manifest.json").exists()


def test_segment_synthetic_rgb_observations_updates_fitting_contract(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=4, image_size=64, seed=999)
    original_masks: dict[str, np.ndarray] = {}
    observations_dir = project_dir / "observations" / "synthetic"
    for frame_dir in sorted(observations_dir.glob("synthetic_*")):
        meta = FrameObservationMeta.model_validate_json(
            (frame_dir / "observation.json").read_text(encoding="utf-8")
        )
        with np.load(meta.hard_mask.path) as data:
            original_masks[meta.frame_id] = np.asarray(data["hard_mask"], dtype=np.float32)

    result = segment_synthetic_rgb_observations(project_dir)

    assert result.frame_count == 4
    assert result.mean_iou_with_previous_masks == 1.0
    manifest = json.loads((observations_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["frames"]) == 4
    problem = load_synthetic_fit_problem(project_dir)
    assert len(problem.observations) == 4
    for observation in problem.observations:
        frame_dir = observations_dir / observation.frame_id
        meta = FrameObservationMeta.model_validate_json(
            (frame_dir / "observation.json").read_text(encoding="utf-8")
        )
        assert meta.hard_mask.path.endswith("hard_mask_rgb.npz")
        assert meta.soft_mask.path.endswith("soft_mask_rgb.npz")
        assert meta.quality["rgb_segmentation"] == 1.0
        np.testing.assert_array_equal(
            observation.hard_mask[None],
            original_masks[observation.frame_id],
        )
        assert observation.bbox_xyxy.shape == (4,)


def test_segment_synthetic_rgb_observations_is_repeatable_with_force(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=123)

    first = segment_synthetic_rgb_observations(project_dir)
    second = segment_synthetic_rgb_observations(project_dir, force=True)

    assert first.frame_count == second.frame_count
    assert first.mean_iou_with_previous_masks == second.mean_iou_with_previous_masks
    assert [frame.bbox_xyxy for frame in first.frames] == [
        frame.bbox_xyxy for frame in second.frames
    ]


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = 51
    Image.fromarray(image).save(path)


def _write_gray_room_white_cat(path) -> None:
    image = np.full((64, 96, 3), 112, dtype=np.uint8)
    image[:18, :, :] = 135
    image[42:, :, :] = 90
    image[25:46, 34:70, :] = 230
    image[18:30, 40:50, :] = 235
    image[18:30, 55:65, :] = 235
    image[30:36, 48:56, :] = [205, 190, 185]
    Image.fromarray(image).save(path)
