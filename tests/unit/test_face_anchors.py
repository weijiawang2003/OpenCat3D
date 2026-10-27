import json

import numpy as np
from PIL import Image

from opencat3d.fitting.optimizer import load_raw_fit_problem
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.preprocess import face_anchors
from opencat3d.preprocess.face_anchors import extract_observed_face_anchor_targets
from opencat3d.preprocess.keypoints import bootstrap_raw_keypoints_and_pose
from opencat3d.preprocess.segmentation import segment_raw_image_observations


def test_observed_face_anchor_extractor_finds_local_pixel_features() -> None:
    image, mask, bbox = _face_fixture()

    extraction = extract_observed_face_anchor_targets(
        image,
        mask,
        bbox,
        frontal_face_confidence=0.9,
    )

    assert set(extraction.targets) == {
        "left_eye_center",
        "right_eye_center",
        "nose_tip",
        "muzzle_center",
        "left_ear_inner",
        "right_ear_inner",
    }
    assert extraction.checks["paired_eye_evidence"] is True
    assert extraction.metrics["mean_confidence"] > 0.0
    assert extraction.targets["left_eye_center"].x < extraction.targets["right_eye_center"].x
    assert extraction.targets["nose_tip"].y > extraction.targets["left_eye_center"].y
    assert extraction.targets["muzzle_center"].y > extraction.targets["nose_tip"].y


def test_best_component_bounds_large_dense_masks_near_expected_anchor() -> None:
    mask = np.zeros((720, 960), dtype=np.bool_)
    mask[80:650, 120:900] = True

    component = face_anchors._best_component(
        mask,
        expected_center_xy=(260.0, 280.0),
        min_pixels=10,
    )

    assert component is not None
    center_x, center_y = face_anchors._mask_center_xy(component)
    assert abs(center_x - 260.0) <= 1.0
    assert abs(center_y - 280.0) <= 1.0
    assert 10 <= int(np.count_nonzero(component)) <= 192 * 192


def test_raw_bootstrap_writes_and_loads_observed_face_anchor_targets(tmp_path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_face_png(input_dir / "a.png")
    _write_face_png(input_dir / "b.png")
    project_dir = tmp_path / "project"
    ingest_image_sources(project_dir, [input_dir], min_frames=2)
    segment_raw_image_observations(project_dir, color_delta_threshold=0.10)

    result = bootstrap_raw_keypoints_and_pose(project_dir)

    assert result.observed_face_anchor_frame_count == 2
    assert result.observed_face_anchor_count >= 12
    assert result.mean_face_anchor_confidence > 0.0
    manifest = json.loads((project_dir / "observations" / "raw" / "manifest.json").read_text())
    first_frame = manifest["frames"][0]
    target_path = first_frame["feature_anchors_xyc_path"]
    target_manifest = json.loads(open(target_path, encoding="utf-8").read())
    assert target_manifest["provenance"] == "observed_pixels"
    assert target_manifest["target_count"] >= 6

    problem = load_raw_fit_problem(project_dir)
    loaded_targets = problem.observations[0].feature_anchors_xyc
    assert "left_eye_center" in loaded_targets
    assert "right_eye_center" in loaded_targets
    assert "nose_tip" in loaded_targets
    assert "muzzle_center" in loaded_targets
    assert loaded_targets["left_eye_center"][2] > 0.0


def _face_fixture() -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    height = 96
    width = 96
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, :, :] = np.asarray([0.18, 0.18, 0.18], dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)
    bbox = (10.0, 10.0, 86.0, 86.0)
    mask[10:86, 10:86] = 1.0
    image[10:86, 10:86, :] = np.asarray([0.93, 0.92, 0.86], dtype=np.float32)
    image[18:28, 22:31, :] = np.asarray([0.95, 0.57, 0.60], dtype=np.float32)
    image[18:28, 65:74, :] = np.asarray([0.95, 0.57, 0.60], dtype=np.float32)
    image[31:39, 30:38, :] = np.asarray([0.13, 0.08, 0.025], dtype=np.float32)
    image[31:39, 58:66, :] = np.asarray([0.13, 0.08, 0.025], dtype=np.float32)
    image[50:58, 45:53, :] = np.asarray([0.98, 0.42, 0.48], dtype=np.float32)
    image[62:73, 35:62, :] = np.asarray([0.98, 0.97, 0.91], dtype=np.float32)
    return image, mask, bbox


def _write_face_png(path) -> None:
    image, mask, _bbox = _face_fixture()
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.asarray(mask * 255.0, dtype=np.uint8)
    Image.fromarray(rgba).save(path)
