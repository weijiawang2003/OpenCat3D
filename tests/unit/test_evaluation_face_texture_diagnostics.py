import json

import numpy as np
from PIL import Image

from opencat3d.evaluation.face_texture_diagnostics import (
    REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS,
    _visual_quality_metrics,
    evaluate_direct_photo_face_candidate_report,
    evaluate_face_texture_diagnostics,
)
from opencat3d.io.cache import sha256_file
from opencat3d.template.cat_template import create_placeholder_cat_template


def test_evaluate_face_texture_diagnostics_writes_crops_and_contact_sheet(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    texture_dir = project_dir / "outputs" / "textures"
    texture_dir.mkdir(parents=True)
    observation_dir = project_dir / "observations" / "raw" / "frame_0001"
    observation_dir.mkdir(parents=True)

    template = create_placeholder_cat_template()
    basecolor = _gradient_texture(1024)
    confidence = np.full((1024, 1024), 255, dtype=np.uint8)
    source_view = np.ones((1024, 1024), dtype=np.uint8)
    Image.fromarray(basecolor).save(texture_dir / "basecolor.png")
    Image.fromarray(confidence).save(texture_dir / "basecolor_confidence.png")
    Image.fromarray(source_view).save(texture_dir / "source_view_id.png")
    source_patch_dir = texture_dir / "face_source_patches"
    feature_entries = [
        _write_source_patch(source_patch_dir, "left_eye_core", (18, 16, 12)),
        _write_source_patch(source_patch_dir, "right_eye_core", (20, 17, 14)),
        _write_source_patch(source_patch_dir, "nose", (216, 128, 138)),
        _write_source_patch(source_patch_dir, "left_muzzle", (228, 224, 212)),
        _write_source_patch(source_patch_dir, "right_muzzle", (222, 220, 210)),
        _write_source_patch(source_patch_dir, "mouth_line", (80, 54, 50)),
        _write_source_patch(source_patch_dir, "left_ear_inner", (224, 158, 170)),
        _write_source_patch(source_patch_dir, "right_ear_inner", (220, 150, 164)),
    ]

    source_image = np.full((96, 128, 3), 255, dtype=np.float32)
    source_image[18:78, 28:100, :] = np.asarray([0.86, 0.84, 0.78], dtype=np.float32)
    source_image[28:38, 40:52, :] = np.asarray([0.08, 0.07, 0.05], dtype=np.float32)
    source_image[28:38, 76:88, :] = np.asarray([0.08, 0.07, 0.05], dtype=np.float32)
    source_image[46:56, 58:70, :] = np.asarray([0.88, 0.54, 0.58], dtype=np.float32)
    np.savez_compressed(observation_dir / "image_srgb.npz", image_srgb=source_image)

    texture_manifest_path = project_dir / "observations" / "raw" / "texture_manifest.json"
    texture_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "test",
                "frames": [
                    {
                        "frame_id": "frame_0001",
                        "image_srgb_path": str(observation_dir / "image_srgb.npz"),
                        "bbox_xyxy": [28.0, 18.0, 100.0, 78.0],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (texture_dir / "texture_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "basecolor": {
                    "width": 1024,
                    "height": 1024,
                    "generated_texel_fraction": 0.0,
                    "projection_manifest_path": str(texture_manifest_path),
                },
                "facial_features": {
                    "source_frame_id": "frame_0001",
                    "features": feature_entries,
                },
                "template_face_uv_regions": template.face_uv_region_manifest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_face_texture_diagnostics(project_dir, mode="raw")

    assert result.success
    assert all(result.checks.values())
    assert result.report_path.name == "raw_face_texture_diagnostics.json"
    assert result.contact_sheet_path.name == "raw_face_texture_crops.png"
    assert result.contact_sheet_path.exists()
    assert result.crop_dir.exists()
    assert result.metrics["face_texture_crop_region_count"] == len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    )
    assert result.metrics["face_texture_crop_texture_width"] == 1024
    assert result.metrics["face_texture_crop_texture_height"] == 1024
    assert result.metrics["face_texture_crop_compared_region_count"] == len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    )
    assert result.metrics["face_texture_crop_exact_source_patch_region_count"] == len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    )
    assert result.metrics["face_texture_crop_exact_source_patch_fraction"] == 1.0
    assert result.metrics["face_texture_crop_detail_region_count"] == len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    )
    assert result.metrics["face_texture_crop_min_unique_rgb_count"] >= 2
    assert result.metrics["face_texture_crop_min_confident_texel_fraction"] == 1.0
    assert result.metrics["face_texture_crop_isolated_region_count"] == len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    )
    assert result.metrics["face_texture_crop_overlap_pair_count"] == 0
    assert result.metrics["face_texture_crop_overlap_fraction"] == 0.0
    assert result.metrics["face_texture_crop_visual_quality_region_count"] == len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    )
    assert result.metrics["face_texture_crop_visual_quality_passed_count"] == len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    )
    assert result.metrics["face_texture_crop_visual_quality_fraction"] == 1.0
    assert result.metrics["face_texture_crop_visual_quality_min_score"] >= 0.70
    assert result.metrics["face_texture_crop_visual_quality_mean_score"] >= result.metrics[
        "face_texture_crop_visual_quality_min_score"
    ]

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["source"] == "opencat3d.face_texture_crop_diagnostics_v1"
    assert report["required_regions"] == list(REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS)
    assert all(region["texture_crop_isolated"] is True for region in report["regions"])
    assert all(
        region["crop_bounds_source"] == "isolated_feature_footprint_uv_bounds_v1"
        for region in report["regions"]
    )
    assert all(region["crop_uv_bounds"] != region["uv_bounds"] for region in report["regions"])
    assert {region["name"] for region in report["regions"]} == set(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    )
    for region in report["regions"]:
        assert region["source_crop_written"] is True
        assert region["source_crop_exact_patch_used"] is True
        assert region["source_patch_feature_names"]
        assert region["source_patch_paths"]
        assert region["texture_detail_present"] is True
        assert region["texture_visual_quality_passed"] is True
        assert region["texture_visual_quality_score"] >= 0.70
        assert region["texture_visual_quality_components"]["source_evidence_score"] == 1.0
        assert region["texture_visual_quality_failure_reasons"] == []
        assert region["texture_visual_quality_weakest_component"]
        assert Image.open(region["texture_crop_path"]).size[0] > 0
        assert Image.open(region["source_crop_path"]).size[0] > 0
    contact_sheet = Image.open(result.contact_sheet_path)
    assert contact_sheet.mode == "RGB"
    assert contact_sheet.size[0] > 0
    assert contact_sheet.size[1] > 0


def test_visual_quality_gate_rejects_flat_present_crops() -> None:
    metrics = _visual_quality_metrics(
        {
            "texture_unique_rgb_count": 1,
            "texture_luma_std": 0.0,
            "texture_confident_texel_fraction": 1.0,
        },
        {
            "source_luma_std": 0.0,
            "source_crop_written": True,
            "source_crop_exact_patch_used": True,
        },
    )

    assert metrics["texture_visual_quality_passed"] is False
    assert metrics["texture_visual_quality_score"] < 0.70
    assert metrics["texture_visual_quality_failure_reasons"] == [
        "flat_texture",
        "source_detail",
        "combined_score",
    ]
    assert metrics["texture_visual_quality_weakest_component"] == "source_luma_score"

    missing_source_metrics = _visual_quality_metrics(
        {
            "texture_unique_rgb_count": 64,
            "texture_luma_std": 0.04,
            "texture_confident_texel_fraction": 0.01,
        },
        {
            "source_luma_std": 0.0,
            "source_crop_written": False,
            "source_crop_exact_patch_used": False,
        },
    )

    assert missing_source_metrics["texture_visual_quality_passed"] is False
    assert missing_source_metrics["texture_visual_quality_failure_reasons"] == [
        "coverage",
        "source_evidence",
        "combined_score",
    ]


def test_direct_photo_face_candidate_report_explains_rejections(tmp_path) -> None:
    project_dir = tmp_path / "project"
    texture_dir = project_dir / "outputs" / "textures"
    texture_dir.mkdir(parents=True)
    projection_manifest_path = project_dir / "observations" / "raw" / "texture_manifest.json"
    projection_manifest_path.parent.mkdir(parents=True)
    projection_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frames": [
                    {
                        "frame_id": "direct_face",
                        "source_media_id": "14.jpg",
                        "media_type": "photo",
                        "source_type": "image",
                    },
                    {
                        "frame_id": "auto_staged",
                        "source_media_id": "frame_0004.jpg",
                        "media_type": "photo",
                        "source_type": "image",
                    },
                    {
                        "frame_id": "video_face",
                        "source_media_id": "v1.mp4",
                        "media_type": "video",
                        "source_type": "video_frame",
                    },
                    {
                        "frame_id": "weak_direct",
                        "source_media_id": "16.jpg",
                        "media_type": "photo",
                        "source_type": "image",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_diagnostics = {
        "left_eye": [
            _candidate("left_eye", "direct_face", selected=True, priority=0.61),
            _candidate("left_eye", "auto_staged", priority=0.0),
            _candidate("left_eye", "video_face", priority=0.0),
        ],
        "right_eye": [_candidate("right_eye", "direct_face", priority=0.58)],
        "left_pupil": [
            _candidate(
                "left_pupil",
                "direct_face",
                quality=0.12,
                detail=0.01,
                priority=0.42,
                selected=True,
            )
        ],
        "nose": [
            _candidate(
                "nose",
                "weak_direct",
                quality=0.20,
                detail=0.18,
                priority=0.0,
            )
        ],
        "muzzle": [_candidate("muzzle", "direct_face", priority=0.55)],
        "ear_inner": [
            _candidate(
                "ear_inner",
                "weak_direct",
                quality=0.60,
                detail=0.35,
                priority=0.0,
            )
        ],
    }
    (texture_dir / "texture_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "basecolor": {
                    "projection_manifest_path": str(projection_manifest_path),
                },
                "facial_features": {
                    "features": [
                        {
                            "name": "left_eye_core",
                            "sample_key": "left_eye",
                            "source_frame_id": "direct_face",
                            "source_patch_used": True,
                        }
                    ],
                    "source_patch_candidate_diagnostics": candidate_diagnostics,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_direct_photo_face_candidate_report(project_dir, mode="raw")

    assert result.success
    assert result.metrics["direct_photo_face_candidate_sample_key_count"] == 9
    assert result.metrics["direct_photo_face_candidate_direct_count"] == 6
    assert result.metrics["direct_photo_face_candidate_eligible_count"] == 4
    assert result.metrics["direct_photo_face_candidate_selected_count"] == 2
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    samples = {sample["sample_key"]: sample for sample in report["samples"]}
    assert set(samples) == {
        "left_eye",
        "right_eye",
        "left_iris",
        "right_iris",
        "left_pupil",
        "right_pupil",
        "nose",
        "muzzle",
        "ear_inner",
    }
    assert samples["left_eye"]["selected_direct_photo"] is True
    assert samples["left_pupil"]["selected_direct_photo"] is True
    assert samples["left_pupil"]["eligible_direct_photo_count"] == 1
    assert samples["left_pupil"]["best_direct_photo_rejection_reasons"] == []
    left_pupil = samples["left_pupil"]["candidates"][0]
    assert left_pupil["direct_photo_eligible"] is True
    assert "low_eye_detail_quality_pair" not in left_pupil[
        "direct_photo_rejection_reasons"
    ]
    assert left_pupil["direct_photo_thresholds"]["min_pupil_quality_score"] == 0.10
    auto_candidate = next(
        candidate
        for candidate in samples["left_eye"]["candidates"]
        if candidate["source_frame_id"] == "auto_staged"
    )
    assert auto_candidate["direct_uploaded_photo"] is False
    assert "auto_staged_photo_name" in auto_candidate["direct_photo_rejection_reasons"]
    assert "low_broad_face_patch_detail" in samples["nose"][
        "best_direct_photo_rejection_reasons"
    ]
    assert "low_patch_quality" in samples["nose"]["best_direct_photo_rejection_reasons"]
    assert "low_ear_inner_patch_detail" in samples["ear_inner"][
        "best_direct_photo_rejection_reasons"
    ]


def _gradient_texture(size: int) -> np.ndarray:
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(255, 0, size, dtype=np.uint8)
    grid_x, grid_y = np.meshgrid(x, y)
    return np.stack(
        [
            grid_x,
            grid_y,
            ((grid_x.astype(np.uint16) + grid_y.astype(np.uint16)) // 2).astype(np.uint8),
        ],
        axis=-1,
    )


def _write_source_patch(
    source_patch_dir,
    feature_name: str,
    rgb: tuple[int, int, int],
) -> dict[str, object]:
    source_patch_dir.mkdir(parents=True, exist_ok=True)
    patch = np.full((14, 18, 3), rgb, dtype=np.uint8)
    patch[:, ::3, :] = np.asarray(
        [max(rgb[0] - 10, 0), max(rgb[1] - 10, 0), max(rgb[2] - 10, 0)],
        dtype=np.uint8,
    )
    path = source_patch_dir / f"{feature_name}.png"
    Image.fromarray(patch).save(path)
    uv_by_name = {
        "left_eye_core": [0.375, 0.930],
        "right_eye_core": [0.625, 0.930],
        "nose": [0.500, 0.885],
        "left_muzzle": [0.430, 0.855],
        "right_muzzle": [0.570, 0.855],
        "mouth_line": [0.500, 0.825],
        "left_ear_inner": [0.292, 0.955],
        "right_ear_inner": [0.708, 0.955],
    }
    radius_by_name = {
        "left_eye_core": [0.020, 0.018],
        "right_eye_core": [0.020, 0.018],
        "nose": [0.032, 0.029],
        "left_muzzle": [0.052, 0.036],
        "right_muzzle": [0.052, 0.036],
        "mouth_line": [0.020, 0.008],
        "left_ear_inner": [0.038, 0.037],
        "right_ear_inner": [0.038, 0.037],
    }

    return {
        "name": feature_name,
        "uv": uv_by_name[feature_name],
        "radius_uv": radius_by_name[feature_name],
        "source_frame_id": "frame_0001",
        "source_patch_used": True,
        "source_patch_path": str(path),
        "source_patch_sha256": sha256_file(path),
        "source_patch_width": int(patch.shape[1]),
        "source_patch_height": int(patch.shape[0]),
    }


def _candidate(
    sample_key: str,
    frame_id: str,
    *,
    quality: float = 0.72,
    detail: float = 0.80,
    priority: float = 0.0,
    selected: bool = False,
) -> dict[str, object]:
    return {
        "sample_key": sample_key,
        "candidate_index": 0,
        "source_frame_id": frame_id,
        "source_patch_quality_score": quality,
        "source_patch_detail_score": detail,
        "source_patch_density_score": 0.8,
        "source_patch_localized_support_score": 0.9,
        "source_patch_frontal_face_confidence": 0.80,
        "source_patch_view_weight": 0.80,
        "source_patch_view_label": "frontal_face_candidate",
        "source_patch_view_weighted_rank": 0.82,
        "source_patch_frontal_photo_priority": priority,
        "selected": selected,
    }
