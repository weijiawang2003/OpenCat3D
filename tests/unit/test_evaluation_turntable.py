import json

import numpy as np
from PIL import Image

from opencat3d.evaluation.turntable import (
    REQUIRED_FACE_TURNTABLE_ANCHORS,
    evaluate_face_turntable_report,
)
from opencat3d.export.mesh import export_fitted_canonical_mesh
from opencat3d.fitting.optimizer import OptimizerConfig, run_raw_fit
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.preprocess.keypoints import bootstrap_raw_keypoints_and_pose
from opencat3d.preprocess.segmentation import segment_raw_image_observations


def test_evaluate_face_turntable_report_writes_raw_anchor_overlays(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_exported_project(project_dir)

    result = evaluate_face_turntable_report(project_dir, mode="raw")

    assert result.success
    assert all(result.checks.values())
    assert result.report_path.name == "raw_face_turntable.json"
    assert result.manifest_path.exists()
    assert result.metrics["turntable_view_count"] == 3
    assert result.metrics["required_anchor_visible_fraction"] == 1.0
    assert result.metrics["front_anchor_pixel_count"] > 0
    assert result.metrics["textured_turntable_view_count"] == 3
    assert result.metrics["textured_turntable_written_count"] == 3
    assert result.metrics["textured_turntable_front_unique_rgb_count"] >= 2
    assert result.metrics["textured_turntable_front_luma_std"] > 0.0
    assert result.metrics["textured_turntable_front_mouth_line_pixel_count"] > 0
    assert result.metrics["textured_turntable_front_required_feature_fraction"] > 0.0

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["checks"]["front_textured_mouth_line_visible"]
    assert report["required_anchors"] == list(REQUIRED_FACE_TURNTABLE_ANCHORS)
    assert all(report["front_required_anchor_visible"].values())
    assert len(report["frames"]) == 3
    front = next(frame for frame in report["frames"] if frame["view"] == "front")
    assert set(REQUIRED_FACE_TURNTABLE_ANCHORS).issubset(front["visible_anchor_names"])
    assert front["textured_feature_pixel_counts"]["mouth_line"] > 0
    overlay = Image.open(front["overlay_path"])
    assert overlay.mode == "RGB"
    assert overlay.size == (256, 256)
    textured = Image.open(front["textured_path"])
    assert textured.mode == "RGB"
    assert textured.size == (256, 256)


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


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = 51
    Image.fromarray(image).save(path)
