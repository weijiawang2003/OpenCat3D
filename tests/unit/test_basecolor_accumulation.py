import json

import numpy as np
from PIL import Image

from opencat3d.textures.basecolor import (
    _accumulate_face_color_samples,
    _apply_basecolor_seed,
    _apply_texture_only_face_feature_rollback,
    _facial_feature_contrast,
    _load_texture_projection_translation_cache,
    _texture_projection_cache_key,
    _write_texture_projection_translation_cache,
)


def test_accumulate_face_color_samples_matches_visible_face_loop() -> None:
    face_index = np.asarray(
        [
            [0, 0, 1, -1],
            [1, 2, 2, 2],
        ],
        dtype=np.int64,
    )
    image = np.asarray(
        [
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.7, 0.1, 0.0], [0.9, 0.9, 0.9]],
            [[0.0, 0.5, 0.4], [0.3, 0.2, 0.1], [0.4, 0.4, 0.2], [0.8, 0.0, 0.1]],
        ],
        dtype=np.float64,
    )
    valid = face_index >= 0
    rgb_sum = np.zeros((3, 3), dtype=np.float64)
    pixel_count = np.zeros(3, dtype=np.int64)
    source_pixel_counts = np.zeros((3, 2), dtype=np.int64)

    _accumulate_face_color_samples(
        rgb_sum,
        pixel_count,
        source_pixel_counts,
        face_index_buffer=face_index,
        image=image,
        valid_mask=valid,
        view_index=1,
    )

    expected_rgb_sum = np.asarray(
        [
            [0.3, 0.5, 0.7],
            [0.7, 0.6, 0.4],
            [1.5, 0.6, 0.4],
        ],
        dtype=np.float64,
    )
    expected_counts = np.asarray([2, 2, 3], dtype=np.int64)
    assert np.allclose(rgb_sum, expected_rgb_sum)
    assert np.array_equal(pixel_count, expected_counts)
    assert np.array_equal(source_pixel_counts[:, 1], expected_counts)
    assert np.array_equal(source_pixel_counts[:, 0], np.zeros(3, dtype=np.int64))


def test_texture_projection_translation_cache_roundtrip_and_invalidates(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "texture_manifest.json"
    manifest_path.write_text(
        '{"frames":[{"frame_id":"a"},{"frame_id":"b"}]}\n',
        encoding="utf-8",
    )
    frame_ids = ["a", "b"]
    beta = np.asarray([0.1, -0.2], dtype=np.float64)
    fitted_roots = np.asarray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float64)
    projection_roots = np.asarray([[0.0, 0.1, 0.2], [0.8, 0.9, 1.0]], dtype=np.float64)
    cache_key = _texture_projection_cache_key(
        source_manifest_path=manifest_path,
        frame_ids=frame_ids,
        beta=beta,
        root_translations=fitted_roots,
        template_vertex_count=12,
        template_face_count=20,
    )

    _write_texture_projection_translation_cache(
        tmp_path,
        mode="raw",
        cache_key=cache_key,
        frame_ids=frame_ids,
        root_translations=projection_roots,
        supplemental_frame_count=1,
    )

    loaded = _load_texture_projection_translation_cache(
        tmp_path,
        mode="raw",
        cache_key=cache_key,
        expected_frame_count=2,
    )
    assert loaded is not None
    assert np.allclose(loaded, projection_roots)
    assert (
        _load_texture_projection_translation_cache(
            tmp_path,
            mode="raw",
            cache_key="different",
            expected_frame_count=2,
        )
        is None
    )
    assert (
        _load_texture_projection_translation_cache(
            tmp_path,
            mode="raw",
            cache_key=cache_key,
            expected_frame_count=3,
        )
        is None
    )


def test_apply_basecolor_seed_fills_only_unknown_texels(tmp_path) -> None:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    seed_basecolor = np.asarray(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    seed_confidence = np.asarray([[255, 0], [255, 255]], dtype=np.uint8)
    seed_source = np.asarray([[1, 0], [2, 3]], dtype=np.uint8)
    Image.fromarray(seed_basecolor).save(seed_dir / "basecolor.png")
    Image.fromarray(seed_confidence).save(seed_dir / "basecolor_confidence.png")
    Image.fromarray(seed_source).save(seed_dir / "source_view_id.png")

    basecolor = np.asarray(
        [
            [[200, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 200, 0]],
        ],
        dtype=np.uint8,
    )
    confidence = np.asarray([[255, 0], [0, 255]], dtype=np.uint8)
    source_view = np.asarray([[9, 0], [0, 8]], dtype=np.uint8)

    report = _apply_basecolor_seed(
        basecolor,
        confidence,
        source_view,
        seed_texture_dir=seed_dir,
    )

    assert report["applied"] is True
    assert report["filled_texel_count"] == 1
    assert report["seed_observed_texel_count"] == 3
    assert report["candidate_unknown_texel_count_before_seed"] == 2
    assert basecolor[0, 0].tolist() == [200, 0, 0]
    assert basecolor[1, 1].tolist() == [0, 200, 0]
    assert basecolor[1, 0].tolist() == [70, 80, 90]
    assert confidence.tolist() == [[255, 0], [255, 255]]
    assert source_view.tolist() == [[9, 0], [2, 8]]


def test_texture_only_face_feature_rollback_restores_seed_contrast(tmp_path) -> None:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    texture_size = 32
    seed_basecolor = np.full((texture_size, texture_size, 3), 240, dtype=np.uint8)
    seed_confidence = np.full((texture_size, texture_size), 255, dtype=np.uint8)
    seed_source = np.full((texture_size, texture_size), 7, dtype=np.uint8)
    candidate_body_color = [20, 120, 40]

    seed_features = [
        _feature("left_muzzle", [0.22, 0.68], [0.035, 0.028], [235, 232, 225]),
        _feature("right_muzzle", [0.34, 0.68], [0.035, 0.028], [235, 232, 225]),
        _feature("mouth_line", [0.28, 0.61], [0.030, 0.012], [80, 55, 48]),
        _feature("nose", [0.28, 0.74], [0.026, 0.020], [225, 142, 150]),
        _feature("left_eye_shadow", [0.68, 0.72], [0.038, 0.026], [40, 34, 30]),
        _feature("right_eye_shadow", [0.82, 0.72], [0.038, 0.026], [40, 34, 30]),
        _feature("left_eye_core", [0.68, 0.72], [0.028, 0.020], [30, 23, 18]),
        _feature("right_eye_core", [0.82, 0.72], [0.028, 0.020], [30, 23, 18]),
        _feature("left_eye_iris", [0.68, 0.72], [0.022, 0.016], [185, 130, 55]),
        _feature("right_eye_iris", [0.82, 0.72], [0.022, 0.016], [185, 130, 55]),
        _feature("left_eye_pupil", [0.68, 0.72], [0.010, 0.008], [8, 7, 6]),
        _feature("right_eye_pupil", [0.82, 0.72], [0.010, 0.008], [8, 7, 6]),
    ]
    _paint_feature_centers(seed_basecolor, seed_features)
    Image.fromarray(seed_basecolor).save(seed_dir / "basecolor.png")
    Image.fromarray(seed_confidence).save(seed_dir / "basecolor_confidence.png")
    Image.fromarray(seed_source).save(seed_dir / "source_view_id.png")
    (seed_dir / "texture_manifest.json").write_text(
        json.dumps(
            {
                "facial_features": {
                    "applied": True,
                    "features": seed_features,
                    "contrast": _facial_feature_contrast(seed_features),
                }
            }
        ),
        encoding="utf-8",
    )

    candidate_features = [
        _feature(
            str(feature["name"]),
            feature["uv"],
            feature["radius_uv"],
            [170, 164, 150],
            source_frame_id="new_weak_face",
        )
        for feature in seed_features
    ]
    basecolor = np.full((texture_size, texture_size, 3), candidate_body_color, dtype=np.uint8)
    confidence = np.full((texture_size, texture_size), 255, dtype=np.uint8)
    source_view = np.full((texture_size, texture_size), 3, dtype=np.uint8)
    facial_features = {
        "applied": True,
        "features": candidate_features,
        "contrast": _facial_feature_contrast(candidate_features),
    }

    report = _apply_texture_only_face_feature_rollback(
        basecolor,
        confidence,
        source_view,
        facial_features,
        seed_texture_dir=seed_dir,
    )

    assert report["applied"] is True
    assert set(report["rollback_groups"]) == {"iris_pupil", "nose_muzzle"}
    assert "nose" in report["rolled_back_feature_names"]
    assert "left_eye_pupil" in report["rolled_back_feature_names"]
    assert facial_features["strict_texture_only_face_feature_rollback"] == report
    assert (
        facial_features["contrast"]["iris_to_pupil_rgb_distance"]
        == (_facial_feature_contrast(seed_features)["iris_to_pupil_rgb_distance"])
    )
    assert (
        facial_features["contrast"]["nose_to_muzzle_rgb_distance"]
        == (_facial_feature_contrast(seed_features)["nose_to_muzzle_rgb_distance"])
    )
    assert basecolor[0, 0].tolist() == candidate_body_color
    nose = next(feature for feature in facial_features["features"] if feature["name"] == "nose")
    assert nose["rgb"] == [225, 142, 150]
    assert nose["texture_only_face_feature_rollback"] is True
    assert nose["pre_rollback_rgb"] == [170, 164, 150]
    assert np.count_nonzero(source_view == 7) == report["rolled_back_texel_count"]


def _feature(
    name: str,
    uv: list[float],
    radius_uv: list[float],
    rgb: list[int],
    *,
    source_frame_id: str = "baseline_face",
) -> dict[str, object]:
    return {
        "name": name,
        "uv": uv,
        "radius_uv": radius_uv,
        "rgb": rgb,
        "source_frame_id": source_frame_id,
        "source_view_id": 1,
        "sample_count": 16,
        "provenance": "observed",
    }


def _paint_feature_centers(
    texture: np.ndarray,
    features: list[dict[str, object]],
) -> None:
    height, width = texture.shape[:2]
    for feature in features:
        uv = feature["uv"]
        rgb = feature["rgb"]
        assert isinstance(uv, list)
        assert isinstance(rgb, list)
        x = int(round(float(uv[0]) * float(width - 1)))
        y = int(round((1.0 - float(uv[1])) * float(height - 1)))
        texture[y, x] = np.asarray(rgb, dtype=np.uint8)
