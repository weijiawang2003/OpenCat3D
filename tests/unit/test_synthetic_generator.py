import json

import numpy as np
from typer.testing import CliRunner

from opencat3d.cli import app
from opencat3d.schemas import FrameObservationMeta, PerspectiveCameraState, read_json_model
from opencat3d.synthetic.generator import create_synthetic_project
from opencat3d.synthetic.renderer import render_buffers
from opencat3d.template.cat_template import create_placeholder_cat_template


def test_synthetic_project_writes_nonempty_observations(tmp_path) -> None:
    result = create_synthetic_project(tmp_path / "demo", num_views=6, image_size=96, seed=123)

    assert result.frame_count == 6
    assert result.template_path.exists()
    manifest = json.loads(result.observation_manifest_path.read_text(encoding="utf-8"))
    assert manifest["num_views"] == 6
    assert len(manifest["frames"]) == 6

    first_meta = read_json_model(
        FrameObservationMeta,
        tmp_path / "demo" / "observations" / "synthetic" / "synthetic_000" / "observation.json",
    )
    with np.load(first_meta.hard_mask.path) as data:
        hard_mask = data["hard_mask"]
    with np.load(first_meta.keypoints_xyc.path) as data:
        keypoints = data["keypoints_xyc"]

    assert hard_mask.shape == (1, 96, 96)
    assert hard_mask.sum() > 0.0
    assert keypoints.shape == (11, 3)
    assert np.isfinite(keypoints).all()
    assert keypoints[:, 2].mean() > 0.7


def test_synthetic_project_is_deterministic(tmp_path) -> None:
    a = create_synthetic_project(tmp_path / "a", num_views=4, image_size=64, seed=999)
    b = create_synthetic_project(tmp_path / "b", num_views=4, image_size=64, seed=999)

    manifest_a = json.loads(a.observation_manifest_path.read_text(encoding="utf-8"))
    manifest_b = json.loads(b.observation_manifest_path.read_text(encoding="utf-8"))
    hashes_a = [frame["hard_mask"]["sha256"] for frame in manifest_a["frames"]]
    hashes_b = [frame["hard_mask"]["sha256"] for frame in manifest_b["frames"]]

    assert hashes_a == hashes_b


def test_synthetic_cli_create(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "demo"

    result = runner.invoke(
        app,
        [
            "synthetic",
            "create",
            str(project_dir),
            "--num-views",
            "3",
            "--image-size",
            "64",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (project_dir / "observations" / "synthetic" / "manifest.json").exists()
    assert "frames=3" in result.output


def test_renderer_uses_nearest_depth_independent_of_face_order() -> None:
    camera = PerspectiveCameraState(
        image_width=32,
        image_height=32,
        fx=24.0,
        fy=24.0,
        cx=15.5,
        cy=15.5,
    )
    near = np.asarray([[-0.7, -0.7, 2.0], [0.7, -0.7, 2.0], [0.0, 0.7, 2.0]])
    far = np.asarray([[-0.7, -0.7, 4.0], [0.7, -0.7, 4.0], [0.0, 0.7, 4.0]])
    vertices = np.vstack([near, far])
    near_then_far = np.asarray([[0, 1, 2], [3, 4, 5]])
    far_then_near = np.asarray([[3, 4, 5], [0, 1, 2]])

    a = render_buffers(vertices, near_then_far, camera)
    b = render_buffers(vertices, far_then_near, camera)
    visible_depth = a.depth[np.isfinite(a.depth)]

    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_allclose(a.depth, b.depth, equal_nan=True)
    assert visible_depth.size > 0
    np.testing.assert_allclose(visible_depth, 2.0)
    assert a.face_index[16, 16] == 0
    assert b.face_index[16, 16] == 1


def test_renderer_skips_degenerate_and_behind_camera_triangles() -> None:
    camera = PerspectiveCameraState(
        image_width=32,
        image_height=32,
        fx=24.0,
        fy=24.0,
        cx=15.5,
        cy=15.5,
    )
    vertices = np.asarray(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [-0.5, -0.5, -1.0],
            [0.5, -0.5, -1.0],
            [0.0, 0.5, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]])

    buffers = render_buffers(vertices, faces, camera)

    assert buffers.mask.sum() == 0.0
    assert not np.isfinite(buffers.depth).any()
    assert np.all(buffers.face_index == -1)


def test_renderer_buffers_match_synthetic_cat_silhouette() -> None:
    template = create_placeholder_cat_template()
    camera = PerspectiveCameraState(
        image_width=96,
        image_height=96,
        fx=90.0,
        fy=90.0,
        cx=47.5,
        cy=47.5,
        translation_world_to_camera=[0.0, -0.55, 4.0],
    )

    buffers = render_buffers(template.vertices_canonical, template.faces, camera)

    inside = buffers.mask > 0.0
    outside = ~inside
    assert inside.sum() > 0
    assert np.isfinite(buffers.depth[inside]).all()
    assert not np.isfinite(buffers.depth[outside]).any()
    assert np.all(buffers.face_index[inside] >= 0)
    assert np.all(buffers.face_index[outside] == -1)
