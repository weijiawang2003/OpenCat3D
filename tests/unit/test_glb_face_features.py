import numpy as np
from PIL import Image

from opencat3d.evaluation.glb_face_features import _sample_feature


def test_uv_support_mask_sampler_uses_exact_manifest_mask(tmp_path) -> None:
    image = np.full((16, 16, 3), 48.0, dtype=np.float64)
    image[2:5, 10:14, :] = np.asarray([190.0, 172.0, 150.0], dtype=np.float64)
    support_mask = np.zeros((16, 16), dtype=np.uint8)
    support_mask[2:5, 10:14] = 255
    mask_path = tmp_path / "left_eye_iris_uv_support.png"
    Image.fromarray(support_mask).save(mask_path)
    feature = {
        "uv": [0.5, 0.5],
        "radius_uv": [0.35, 0.35],
        "rgb": [190, 172, 150],
        "uv_support_mask_path": str(mask_path),
        "uv_support_mask_texel_count": 12,
        "source_patch_ellipse_texel_count": 80,
    }

    report = _sample_feature(
        "left_eye_iris",
        feature,
        image,
        project_dir=tmp_path,
    )

    assert report["sample_mode"] == "uv_support_mask"
    assert report["sampled_texel_count"] == 12
    assert report["support_fraction"] == 0.15
    assert report["rgb_error"] == 0.0
    assert report["visible"] is True


def test_uv_support_mask_sampler_accepts_existing_relative_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "project"
    mask_path = project_dir / "outputs" / "textures" / "nose_uv_support.png"
    mask_path.parent.mkdir(parents=True)
    image = np.full((16, 16, 3), 220.0, dtype=np.float64)
    image[6:9, 6:9, :] = np.asarray([156.0, 126.0, 111.0], dtype=np.float64)
    support_mask = np.zeros((16, 16), dtype=np.uint8)
    support_mask[6:9, 6:9] = 255
    Image.fromarray(support_mask).save(mask_path)
    feature = {
        "uv": [0.5, 0.5],
        "radius_uv": [0.35, 0.35],
        "rgb": [156, 126, 111],
        "uv_support_mask_path": str(mask_path.relative_to(tmp_path)),
        "uv_support_mask_texel_count": 9,
        "source_patch_ellipse_texel_count": 80,
    }

    report = _sample_feature(
        "nose",
        feature,
        image,
        project_dir=project_dir,
    )

    assert report["sample_mode"] == "uv_support_mask"
    assert report["sampled_texel_count"] == 9
    assert report["rgb_error"] == 0.0


def test_dark_eye_core_sampler_uses_observed_dark_support_subset(tmp_path) -> None:
    image = np.full((20, 20, 3), 215.0, dtype=np.float64)
    image[5:13, 5:13, :] = np.asarray([160.0, 145.0, 132.0], dtype=np.float64)
    image[8:11, 8:11, :] = np.asarray([24.0, 20.0, 26.0], dtype=np.float64)
    support_mask = np.zeros((20, 20), dtype=np.uint8)
    support_mask[5:13, 5:13] = 255
    mask_path = tmp_path / "right_eye_core_uv_support.png"
    Image.fromarray(support_mask).save(mask_path)
    feature = {
        "uv": [0.5, 0.5],
        "radius_uv": [0.35, 0.35],
        "rgb": [13, 9, 15],
        "uv_support_mask_path": str(mask_path),
        "uv_support_mask_texel_count": 64,
        "source_patch_ellipse_texel_count": 80,
    }

    report = _sample_feature(
        "right_eye_core",
        feature,
        image,
        project_dir=tmp_path,
    )

    assert report["sample_mode"] == "uv_support_mask_dark_eye_support"
    assert report["sampled_texel_count"] < 64
    assert report["rgb_error"] < 0.10
    assert report["visible"] is True


def test_source_detail_eye_sampler_uses_bright_iris_support() -> None:
    image = np.full((16, 16, 3), 48.0, dtype=np.float64)
    image[6:10, 6:10, :] = np.asarray([180.0, 170.0, 160.0], dtype=np.float64)
    feature = {
        "uv": [0.5, 0.5],
        "radius_uv": [0.35, 0.35],
        "rgb": [180, 170, 160],
        "visibility_reinforcement_detail_preserved": True,
        "visibility_reinforcement_texel_count": 16,
        "source_patch_ellipse_texel_count": 80,
    }

    detail_report = _sample_feature("left_eye_iris", feature, image)
    ellipse_report = _sample_feature(
        "left_eye_iris",
        {**feature, "visibility_reinforcement_detail_preserved": False},
        image,
    )

    assert detail_report["sample_mode"] == "source_detail_support"
    assert detail_report["sampled_texel_count"] < ellipse_report["sampled_texel_count"]
    assert detail_report["rgb_error"] < ellipse_report["rgb_error"]
    assert detail_report["visible"] is True
