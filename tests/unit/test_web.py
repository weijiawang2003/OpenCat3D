import json
from datetime import UTC, datetime, timedelta
from http.client import HTTPConnection
from io import BytesIO
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import opencat3d.web as web
from catnet.storage import CatNetStore
from opencat3d.cli import app
from opencat3d.web import (
    UploadedFile,
    preflight_reconstruction_media_batch,
    render_identity_job,
    render_index,
    render_job,
    run_identity_image_job,
    run_uploaded_image_job,
    stage_reconstruction_uploads,
)


def test_uploaded_image_job_runs_raw_pipeline_and_links_outputs(tmp_path) -> None:
    uploads = [
        UploadedFile("b.png", _png_bytes(bbox=(2, 2, 6, 6))),
        UploadedFile("a.png", _png_bytes(bbox=(1, 1, 5, 5))),
    ]

    result = run_uploaded_image_job(
        uploads,
        work_dir=tmp_path,
        color_delta_threshold=0.10,
        camera_root_steps=2,
    )

    assert result.result.success
    assert result.job_id.startswith("job_")
    assert (result.upload_dir / "frame_0000.png").exists()
    assert (result.upload_dir / "frame_0001.png").exists()
    assert result.result.glb_path == result.project_dir / "outputs" / "avatar" / "cat_avatar.glb"
    assert result.result.glb_path.exists()
    assert result.result.obj_path.exists()
    assert result.result.ply_path.exists()
    assert result.result.compliance_audit_path.exists()
    media_manifest = result.upload_dir / "media_manifest.json"
    assert media_manifest.exists()
    media_frames = json.loads(media_manifest.read_text(encoding="utf-8"))["frames"]
    assert media_frames["frame_0000.png"]["source_type"] == "image"

    html = render_job(result.project_dir, job_id=result.job_id)
    assert f"/files/{result.job_id}/outputs/avatar/cat_avatar.glb" in html
    assert "vertices /" in html
    assert "raw_frame_count" in html
    assert "raw_input_frame_count" in html
    assert "raw_selected_frame_count" in html
    assert "raw_texture_frame_count" in html
    assert "Frame selection" in html
    assert "segmentation_full_frame_bbox_fraction" in html
    assert "overlay_mean_silhouette_iou" in html
    assert "raw_face_feature_min_required_contrast" in html
    assert "raw_glb_face_feature_material_visibility_gate_valid" in html
    assert "GLB face features" in html
    assert "Face close-up" in html
    assert "Face source comparison" in html
    assert "raw_face_texture_closeup_required_feature_fraction" in html
    assert "raw_face_texture_source_compare_required_feature_fraction" in html


def test_result_page_renders_face_texture_diagnostic_crops(tmp_path) -> None:
    project_dir = tmp_path / "project"
    quality_dir = project_dir / "outputs" / "quality"
    avatar_dir = project_dir / "outputs" / "avatar"
    texture_dir = project_dir / "outputs" / "textures"
    crop_dir = quality_dir / "raw_face_texture_crops"
    crop_dir.mkdir(parents=True)
    avatar_dir.mkdir(parents=True)
    texture_dir.mkdir(parents=True)

    glb_path = avatar_dir / "cat_avatar.glb"
    diagnostics_path = quality_dir / "raw_face_texture_diagnostics.json"
    bake_comparison_path = quality_dir / "raw_face_texture_bake_comparison.json"
    texture_manifest_path = texture_dir / "texture_manifest.json"
    texture_crop = crop_dir / "left_eye_texture.png"
    source_crop = crop_dir / "left_eye_source.png"
    glb_path.write_bytes(b"glb")
    texture_crop.write_bytes(_png_bytes(bbox=(1, 1, 6, 6)))
    source_crop.write_bytes(_png_bytes(bbox=(2, 2, 7, 7)))

    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "artifacts": {
                    "glb": str(glb_path),
                    "face_texture_diagnostics": str(diagnostics_path),
                    "face_texture_bake_comparison": str(bake_comparison_path),
                    "texture_manifest": str(texture_manifest_path),
                },
                "metrics": {
                    "vertex_count": 12,
                    "face_count": 10,
                    "raw_facial_feature_source_patch_bake_candidate_count": 2,
                    "raw_facial_feature_source_patch_bake_selected_feedback": 1.0,
                    "raw_facial_feature_source_patch_bake_visual_quality_min_score": (
                        0.950344
                    ),
                    "raw_facial_feature_source_patch_bake_visual_quality_mean_score": (
                        0.991724
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    diagnostics_path.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "name": "left_eye",
                        "texture_crop_path": str(texture_crop),
                        "source_crop_path": str(source_crop),
                        "texture_unique_rgb_count": 18,
                        "texture_confident_texel_fraction": 0.5,
                        "texture_visual_quality_score": 0.42,
                        "texture_visual_quality_passed": False,
                        "texture_visual_quality_failure_reasons": [
                            "flat_texture",
                            "source_detail",
                        ],
                        "texture_visual_quality_weakest_component": "texture_luma_score",
                        "source_crop_exact_patch_used": True,
                        "texture_crop_isolated": True,
                        "dominant_source_frame_id": "frame_0001",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    bake_comparison_path.write_text(
        json.dumps(
            {
                "success": True,
                "selected_candidate": "feedback",
                "baseline_candidate": "generic",
                "compared_region_count": 2,
            }
        ),
        encoding="utf-8",
    )
    texture_manifest_path.write_text(
        json.dumps(
            {
                "facial_features": {
                    "source_patch_bake_candidate_count": 2,
                    "source_patch_bake_selected_candidate": "feedback",
                    "source_patch_bake_selected_feedback": True,
                    "source_patch_bake_candidates": [
                        {
                            "label": "generic",
                            "selected": False,
                            "visual_quality_passed_count": 5,
                            "visual_quality_min_score": 0.920578,
                            "visual_quality_mean_score": 0.96525,
                            "min_confident_texel_fraction": 0.401,
                            "mean_source_luma_std": 0.016,
                            "feedback_used_count": 0,
                            "regions": [
                                {
                                    "name": "left_eye",
                                    "score": 0.900,
                                    "passed": True,
                                    "confident_texel_fraction": 0.401,
                                    "source_luma_std": 0.016,
                                    "weakest_component": "source_luma_score",
                                },
                                {
                                    "name": "left_ear_inner",
                                    "score": 0.920578,
                                    "passed": True,
                                    "confident_texel_fraction": 0.1197,
                                    "source_luma_std": 0.012,
                                    "weakest_component": "source_luma_score",
                                },
                            ],
                        },
                        {
                            "label": "feedback",
                            "selected": True,
                            "visual_quality_passed_count": 6,
                            "visual_quality_min_score": 0.950344,
                            "visual_quality_mean_score": 0.991724,
                            "min_confident_texel_fraction": 0.452,
                            "mean_source_luma_std": 0.021,
                            "feedback_used_count": 13,
                            "regions": [
                                {
                                    "name": "left_eye",
                                    "score": 0.900,
                                    "passed": True,
                                    "confident_texel_fraction": 0.401,
                                    "source_luma_std": 0.016,
                                    "weakest_component": "source_luma_score",
                                },
                                {
                                    "name": "left_ear_inner",
                                    "score": 1.000,
                                    "passed": True,
                                    "confident_texel_fraction": 0.452,
                                    "source_luma_std": 0.021,
                                    "weakest_component": "confident_texel_score",
                                },
                            ],
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    html = render_job(project_dir, job_id="job_face")

    assert "Face Diagnostics" in html
    assert "/files/job_face/outputs/quality/raw_face_texture_crops/left_eye_texture.png" in html
    assert "/files/job_face/outputs/quality/raw_face_texture_crops/left_eye_source.png" in html
    assert "/files/job_face/outputs/quality/raw_face_texture_diagnostics.json" in html
    assert "Unique RGB" in html
    assert "50.0%" in html
    assert "Visual Score" in html
    assert "0.420" in html
    assert "Visual Gate" in html
    assert "Failed" in html
    assert "Quality Issues" in html
    assert "Texture is too flat" in html
    assert "Source crop lacks detail" in html
    assert "Weakest signal: Texture contrast." in html
    assert "Exact Source" in html
    assert "Isolated" in html
    assert "frame_0001" in html
    assert "Face Bake Candidates" in html
    assert "Face bake comparison" in html
    assert "/files/job_face/outputs/quality/raw_face_texture_bake_comparison.json" in html
    assert "Texture Manifest JSON" in html
    assert "/files/job_face/outputs/textures/texture_manifest.json" in html
    assert "generic" in html
    assert "feedback / feedback" in html
    assert "Compared" in html
    assert "Selected" in html
    assert "0.921" in html
    assert "0.950" in html
    assert "45.2%" in html
    assert "Feedback Used" in html
    assert "Region Comparison" in html
    assert "feedback compared against generic" in html
    assert "Left Ear Inner" in html
    assert "+0.079" in html
    assert "+33.2 pp" in html
    assert "Score +0.079" in html
    assert "Coverage" in html
    assert "raw_facial_feature_source_patch_bake_candidate_count" in html
    assert "raw_facial_feature_source_patch_bake_visual_quality_mean_score" in html


def test_result_page_renders_texture_source_acquisition_hint(tmp_path) -> None:
    project_dir = tmp_path / "project"
    quality_dir = project_dir / "outputs" / "quality"
    avatar_dir = project_dir / "outputs" / "avatar"
    quality_dir.mkdir(parents=True)
    avatar_dir.mkdir(parents=True)

    glb_path = avatar_dir / "cat_avatar.glb"
    preflight_path = quality_dir / "texture_source_preflight.json"
    glb_path.write_bytes(b"glb")
    preflight_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.texture_source_preflight",
                "source_frame_count": 8,
                "accepted_frame_count": 0,
                "duplicate_frame_count": 3,
                "rejected_frame_count": 5,
                "ranked_frames": [
                    {
                        "source_frame_id": "frame_0006",
                        "duplicate": False,
                        "rank_score": 0.324017716,
                        "quality_gate": {
                            "accepted": False,
                            "reason": "low_face_detail_score",
                            "face_texture_detail_score": 0.324017716,
                            "face_texture_detail_score_floor": 0.48,
                        },
                    },
                    {
                        "source_frame_id": "frame_0002",
                        "duplicate": True,
                        "rank_score": -0.490923434,
                        "quality_gate": {
                            "accepted": True,
                            "reason": "face_detail_score_passed",
                            "face_texture_detail_score": 0.509076566,
                            "face_texture_detail_score_floor": 0.48,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "artifacts": {
                    "glb": str(glb_path),
                    "texture_source_preflight": str(preflight_path),
                },
                "metrics": {
                    "vertex_count": 12,
                    "face_count": 10,
                    "raw_texture_source_preflight_accepted_frame_count": 0,
                    "raw_texture_source_preflight_duplicate_frame_count": 3,
                    "raw_texture_source_preflight_rejected_frame_count": 5,
                    "raw_texture_source_preflight_top_score": 0.324017716,
                },
            }
        ),
        encoding="utf-8",
    )

    html = render_job(project_dir, job_id="job_need_media")

    assert "Texture source preflight" in html
    assert "/files/job_need_media/outputs/quality/texture_source_preflight.json" in html
    assert "Media Needed" in html
    assert "Needs media" in html
    assert "Accepted Unique" in html
    assert "Best Non-Duplicate" in html
    assert "0.324" in html
    assert "Required Score" in html
    assert "0.480" in html
    assert "frame_0006" in html
    assert "non-duplicate frontal close-up of eyes, nose, muzzle" in html
    assert "3 high-overlap frames were already covered by the baseline" in html
    assert "5 frames failed face-detail quality" in html
    assert "raw_texture_source_preflight_accepted_frame_count" in html
    assert "raw_texture_source_preflight_top_score" in html


def test_result_page_renders_source_frame_acquisition_gap(tmp_path) -> None:
    project_dir = tmp_path / "project"
    quality_dir = project_dir / "outputs" / "quality"
    avatar_dir = project_dir / "outputs" / "avatar"
    quality_dir.mkdir(parents=True)
    avatar_dir.mkdir(parents=True)

    glb_path = avatar_dir / "cat_avatar.glb"
    gap_path = quality_dir / "source_frame_acquisition_gap.json"
    glb_path.write_bytes(b"glb")
    _write_source_frame_acquisition_gap_report(gap_path)
    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "artifacts": {
                    "glb": str(glb_path),
                    "source_frame_acquisition_gap": str(gap_path),
                },
                "metrics": {
                    "vertex_count": 12,
                    "face_count": 10,
                    "observed_face_fraction": 0.32832618,
                    "observed_vertex_fraction": 0.4394572,
                },
            }
        ),
        encoding="utf-8",
    )

    html = render_job(project_dir, job_id="job_gap")

    assert "Source Frame Acquisition Gap" in html
    assert "Run Later" in html
    assert "/files/job_gap/outputs/quality/source_frame_acquisition_gap.json" in html
    assert "insufficient_new_unique_evidence" in html
    assert "basecolor_observed_texel_fraction" in html
    assert "observed_face_fraction" in html
    assert "32.8%" in html
    assert "frontal_face" in html
    assert "rear_body" in html
    assert "new_unique_evidence_required" in html
    assert "Run preflight before full reconstruction" in html


def test_result_page_renders_template_region_v2_contract_panel(tmp_path) -> None:
    project_dir = tmp_path / "project"
    quality_dir = project_dir / "outputs" / "quality"
    avatar_dir = project_dir / "outputs" / "avatar"
    quality_dir.mkdir(parents=True)
    avatar_dir.mkdir(parents=True)

    mesh_validation_path = avatar_dir / "mesh_validation.json"
    glb_face_feature_visibility_path = (
        quality_dir / "raw_glb_face_feature_visibility.json"
    )
    face_closeup_path = quality_dir / "raw_face_turntable" / "face_closeup.png"
    source_comparison_path = quality_dir / "raw_face_turntable" / "face_source_comparison.png"
    face_closeup_path.parent.mkdir(parents=True)
    region_names = [
        "face",
        "muzzle",
        "eyes",
        "nose",
        "ear_interiors",
        "paws",
        "tail",
        "torso",
        "belly",
        "limbs",
    ]
    regions = [
        {
            "name": name,
            "semantic": name,
            "source_regions": [name],
            "vertex_count": 113 if name == "belly" else 40 + index,
            "face_count": 211 if name == "belly" else 80 + index,
            "reconstruction_role": (
                "ventral_markings"
                if name == "belly"
                else "identity_shape_and_texture"
                if name == "face"
                else "pose_and_silhouette"
            ),
            "uv_bounds": [0.1, 0.2, 0.3, 0.4],
        }
        for index, name in enumerate(region_names)
    ]
    mesh_validation_path.write_text(
        json.dumps(
            {
                "template_region_v2_count": 10,
                "template_region_v2_contract_valid": True,
                "glb_template_regions_v2_valid": True,
                "template_regions_v2": {
                    "schema_version": "2.0",
                    "source": "opencat3d.cat_template_region_contract_v2",
                    "required_region_names": region_names,
                    "region_count": 10,
                    "validation": {
                        "valid": True,
                        "missing_required_regions": [],
                    },
                    "regions": regions,
                },
            }
        ),
        encoding="utf-8",
    )
    glb_face_feature_visibility_path.write_text("{}", encoding="utf-8")
    face_closeup_path.write_bytes(_png_bytes(bbox=(1, 1, 6, 6)))
    source_comparison_path.write_bytes(_png_bytes(bbox=(2, 2, 7, 7)))
    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "artifacts": {
                    "face_texture_closeup": str(face_closeup_path),
                    "face_texture_source_comparison": str(source_comparison_path),
                    "glb_face_feature_visibility": str(
                        glb_face_feature_visibility_path
                    ),
                },
                "metrics": {
                    "vertex_count": 958,
                    "face_count": 1864,
                    "raw_template_region_v2_count": 10,
                    "raw_template_region_v2_required_region_count": 10,
                    "raw_template_region_v2_required_present_count": 10,
                    "raw_template_region_v2_contract_valid": 1.0,
                    "raw_glb_template_region_v2_contract_valid": 1.0,
                    "raw_template_region_v2_belly_vertex_count": 113,
                    "raw_template_region_v2_texture_target_bake_used": 1.0,
                    "raw_template_region_v2_texture_target_reported_size": 2048,
                    "raw_template_region_v2_texture_target_reported_scale": 2.0,
                    "raw_template_region_v2_texture_target_reported_island_count": 8,
                    "raw_template_region_v2_texture_target_reported_facial_island_count": 8,
                    "raw_template_region_v2_texture_target_reported_requested_texel_count": 4096,
                    "raw_template_region_v2_texture_target_reported_allocated_texel_count": 8192,
                    "raw_template_region_v2_texture_target_size": 2048,
                    "raw_template_region_v2_texture_target_scale": 2.0,
                    "raw_template_region_v2_texture_target_island_count": 8,
                    "raw_template_region_v2_texture_target_facial_island_count": 8,
                    "raw_template_region_v2_texture_target_requested_texel_count": 4096,
                    "raw_template_region_v2_texture_target_allocated_texel_count": 8192,
                    "raw_template_region_v2_texture_target_eye_min_texels": 1024,
                    "raw_template_region_v2_texture_target_nose_texels": 512,
                    "raw_template_region_v2_texture_target_muzzle_texels": 2048,
                    "raw_template_region_v2_texture_target_ear_inner_min_texels": 1024,
                    "raw_template_region_v2_texture_target_max_density_scale": 3.0,
                    "raw_glb_texture_size": 2048,
                    "raw_glb_texture_dimensions_valid": 1.0,
                    "raw_glb_pbr_texture_dimensions_valid": 1.0,
                    "raw_glb_basecolor_texture_valid": 1.0,
                    "raw_glb_pbr_textures_valid": 1.0,
                    "raw_glb_embedded_texture_dimension_gate_valid": 1.0,
                    "raw_glb_face_feature_required_group_count": 3,
                    "raw_glb_face_feature_visible_group_count": 3,
                    "raw_glb_face_feature_required_feature_count": 10,
                    "raw_glb_face_feature_present_required_feature_count": 10,
                    "raw_glb_face_feature_visible_feature_count": 10,
                    "raw_glb_face_feature_sampled_texel_count": 4096,
                    "raw_glb_face_feature_min_visible_texel_count": 128,
                    "raw_glb_face_feature_mean_rgb_error": 0.0,
                    "raw_glb_face_feature_max_rgb_error": 0.0,
                    "raw_glb_face_feature_eye_visible": 1.0,
                    "raw_glb_face_feature_nose_visible": 1.0,
                    "raw_glb_face_feature_muzzle_visible": 1.0,
                    "raw_glb_face_feature_iris_to_pupil_rgb_distance": 0.24,
                    "raw_glb_face_feature_eye_detail_luma_std": 0.08,
                    "raw_glb_face_feature_nose_to_muzzle_rgb_distance": 0.32,
                    "raw_glb_face_feature_mouth_to_muzzle_rgb_distance": 0.32,
                    "raw_glb_face_feature_material_visibility_gate_valid": 1.0,
                    "raw_face_texture_closeup_width": 512,
                    "raw_face_texture_closeup_height": 512,
                    "raw_face_texture_closeup_feature_count": 14,
                    "raw_face_texture_closeup_required_feature_count": 10,
                    "raw_face_texture_closeup_required_present_count": 10,
                    "raw_face_texture_closeup_required_feature_fraction": 1.0,
                    "raw_face_texture_closeup_crop_texel_count": 12544,
                    "raw_face_texture_closeup_unique_rgb_count": 8,
                    "raw_face_texture_closeup_luma_std": 0.12,
                    "raw_face_texture_source_compare_source_feature_available_count": 14,
                    "raw_face_texture_source_compare_feature_count": 14,
                    "raw_face_texture_source_compare_required_feature_count": 10,
                    "raw_face_texture_source_compare_required_present_count": 10,
                    "raw_face_texture_source_compare_required_feature_fraction": 1.0,
                    "raw_face_texture_source_compare_mean_rgb_distance": 0.08,
                    "raw_face_texture_source_compare_max_rgb_distance": 0.16,
                    "raw_face_texture_source_compare_mean_luma_delta": 0.04,
                    "raw_face_texture_source_compare_max_luma_delta": 0.09,
                    "raw_face_texture_source_compare_source_mean_luma_std": 0.11,
                    "raw_face_texture_source_compare_baked_mean_luma_std": 0.10,
                },
            }
        ),
        encoding="utf-8",
    )

    html = render_job(project_dir, job_id="job_regions")

    assert "CatTemplate v2 Regions" in html
    assert "Fixed-topology anatomical contract" in html
    assert "/files/job_regions/outputs/avatar/mesh_validation.json" in html
    assert "Mesh Validation JSON" in html
    assert "Contract" in html
    assert "GLB Contract" in html
    assert "Required Present" in html
    assert "Face" in html
    assert "Belly" in html
    assert "identity_shape_and_texture" in html
    assert "ventral_markings" in html
    assert "113" in html
    assert "211" in html
    assert "raw_template_region_v2_contract_valid" in html
    assert "raw_glb_template_region_v2_contract_valid" in html
    assert "raw_template_region_v2_texture_target_size" in html
    assert "raw_template_region_v2_texture_target_facial_island_count" in html
    assert "raw_template_region_v2_texture_target_bake_used" in html
    assert "raw_template_region_v2_texture_target_reported_size" in html
    assert "raw_glb_texture_size" in html
    assert "raw_glb_texture_dimensions_valid" in html
    assert "raw_glb_pbr_texture_dimensions_valid" in html
    assert "raw_glb_embedded_texture_dimension_gate_valid" in html
    assert "raw_glb_face_feature_visible_group_count" in html
    assert "raw_glb_face_feature_material_visibility_gate_valid" in html
    assert "/files/job_regions/outputs/quality/raw_glb_face_feature_visibility.json" in html
    assert "raw_face_texture_closeup_required_feature_fraction" in html
    assert "/files/job_regions/outputs/quality/raw_face_turntable/face_closeup.png" in html
    assert "raw_face_texture_source_compare_required_feature_fraction" in html
    assert (
        "/files/job_regions/outputs/quality/raw_face_turntable/face_source_comparison.png"
        in html
    )


def test_uploaded_video_job_extracts_frames_before_raw_pipeline(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_extract_video_frames(video_path, output_dir, *, min_frames, max_frames, force):
        captured["video_path"] = video_path
        captured["output_dir"] = output_dir
        captured["min_frames"] = min_frames
        captured["max_frames"] = max_frames
        captured["force"] = force
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for index in range(3):
            path = output_dir / f"fake_{index}.jpg"
            path.write_bytes(b"fake-frame")
            frame_paths.append(path)
        return SimpleNamespace(frame_paths=tuple(frame_paths), selected_indices=(0, 7, 14))

    def fake_run_raw_pipeline(project_dir, *, input_paths, **kwargs):
        captured["project_dir"] = project_dir
        captured["input_paths"] = input_paths
        captured["pipeline_kwargs"] = kwargs
        avatar_dir = project_dir / "outputs" / "avatar"
        quality_dir = project_dir / "outputs" / "quality"
        avatar_dir.mkdir(parents=True, exist_ok=True)
        quality_dir.mkdir(parents=True, exist_ok=True)
        glb = avatar_dir / "cat_avatar.glb"
        obj = avatar_dir / "cat_canonical.obj"
        ply = avatar_dir / "cat_canonical.ply"
        skeleton = avatar_dir / "skeleton.json"
        compliance = quality_dir / "raw_opencat3d_compliance_audit.json"
        summary = quality_dir / "raw_evaluation.json"
        for path in (glb, obj, ply, skeleton, compliance, summary):
            path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            glb_path=glb,
            obj_path=obj,
            ply_path=ply,
            skeleton_path=skeleton,
            compliance_audit_path=compliance,
            summary_path=summary,
            evaluation_report_path=summary,
            fit_report_path=summary,
        )

    monkeypatch.setattr(web, "extract_video_frames", fake_extract_video_frames)
    monkeypatch.setattr(web, "run_raw_pipeline", fake_run_raw_pipeline)

    result = run_uploaded_image_job(
        [UploadedFile("cat.mp4", b"video-bytes")],
        work_dir=tmp_path,
        camera_root_steps=3,
    )

    assert result.result.success
    assert captured["video_path"].name == "source.mp4"
    assert captured["min_frames"] == 2
    assert captured["max_frames"] == 24
    assert captured["force"] is True
    staged = sorted(result.upload_dir.glob("frame_*.jpg"))
    assert [path.name for path in staged] == ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]
    assert captured["input_paths"] == [result.upload_dir]
    assert captured["pipeline_kwargs"]["camera_root_steps"] == 3
    assert captured["pipeline_kwargs"]["max_selected_frames"] == 12
    media_manifest = json.loads((result.upload_dir / "media_manifest.json").read_text())
    assert media_manifest["frames"]["frame_0001.jpg"]["source_type"] == "video_frame"
    assert media_manifest["frames"]["frame_0001.jpg"]["source_frame_index"] == 7


def test_stage_reconstruction_uploads_preserves_mixed_media_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_extract_video_frames(video_path, output_dir, *, min_frames, max_frames, force):
        assert video_path.name == "source.mp4"
        assert min_frames == 2
        assert max_frames == 24
        assert force is True
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for index in range(2):
            path = output_dir / f"video_{index}.jpg"
            path.write_bytes(f"video-frame-{index}".encode("ascii"))
            frame_paths.append(path)
        return SimpleNamespace(frame_paths=tuple(frame_paths), selected_indices=(3, 9))

    monkeypatch.setattr(web, "extract_video_frames", fake_extract_video_frames)
    upload_dir = tmp_path / "uploads" / "job_mixed"
    upload_dir.mkdir(parents=True)
    (upload_dir / "frame_stale.jpg").write_bytes(b"stale")

    result = stage_reconstruction_uploads(
        [
            UploadedFile("cat_b.png", _png_bytes(bbox=(2, 2, 6, 6))),
            UploadedFile("cat_video.mp4", b"video-bytes"),
            UploadedFile("cat_a.png", _png_bytes(bbox=(1, 1, 5, 5))),
        ],
        upload_dir=upload_dir,
        force=True,
    )

    assert result.frame_count == 4
    assert result.photo_count == 2
    assert result.video_count == 1
    assert result.has_video is True
    assert [path.name for path in result.frame_paths] == [
        "frame_0000.png",
        "frame_0001.png",
        "frame_0002.jpg",
        "frame_0003.jpg",
    ]
    assert not (upload_dir / "frame_stale.jpg").exists()
    manifest = json.loads(result.media_manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "opencat3d.web.upload_staging"
    assert manifest["frames"]["frame_0000.png"]["source_media_id"] == "cat_a.png"
    assert manifest["frames"]["frame_0001.png"]["source_media_id"] == "cat_b.png"
    assert manifest["frames"]["frame_0002.jpg"]["source_type"] == "video_frame"
    assert manifest["frames"]["frame_0002.jpg"]["source_frame_index"] == 3
    assert manifest["frames"]["frame_0003.jpg"]["source_frame_index"] == 9


def test_preflight_reconstruction_media_batch_deduplicates_existing_uploads(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_extract_video_frames(video_path, output_dir, *, min_frames, max_frames, force):
        assert video_path.name == "source.mp4"
        assert min_frames == 2
        assert max_frames == 24
        assert force is True
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for index in range(2):
            path = output_dir / f"video_{index}.jpg"
            path.write_bytes(f"video-frame-{index}".encode("ascii"))
            frame_paths.append(path)
        return SimpleNamespace(frame_paths=tuple(frame_paths), selected_indices=(4, 8))

    monkeypatch.setattr(web, "extract_video_frames", fake_extract_video_frames)
    duplicate_photo = _png_bytes(bbox=(1, 1, 5, 5))
    existing_dir = tmp_path / "existing_upload"
    existing_dir.mkdir()
    (existing_dir / "frame_0000.png").write_bytes(duplicate_photo)
    (existing_dir / "frame_0001.png").write_bytes(_png_bytes(bbox=(2, 2, 6, 6)))
    (existing_dir / "media_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "test",
                "frames": {
                    "frame_0000.png": {
                        "media_type": "photo",
                        "source_type": "image",
                        "source_media_id": "already_uploaded.png",
                    },
                    "frame_0001.png": {
                        "media_type": "photo",
                        "source_type": "image",
                        "source_media_id": "body.png",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = preflight_reconstruction_media_batch(
        [
            UploadedFile("new_duplicate.png", duplicate_photo),
            UploadedFile("new_unique.png", _png_bytes(bbox=(3, 3, 7, 7))),
            UploadedFile("clip.mp4", b"video-bytes"),
        ],
        upload_dir=tmp_path / "combined",
        existing_upload_dirs=(existing_dir,),
        force=True,
    )

    assert result.input_frame_count == 6
    assert result.existing_frame_count == 2
    assert result.staged_upload_frame_count == 4
    assert result.unique_frame_count == 5
    assert result.duplicate_frame_count == 1
    assert result.duplicate_group_count == 1
    assert result.existing_unique_frame_count == 2
    assert result.existing_duplicate_frame_count == 0
    assert result.new_unique_frame_count == 3
    assert result.new_duplicate_frame_count == 1
    assert result.photo_frame_count == 3
    assert result.video_frame_count == 2
    assert [path.name for path in result.frame_paths] == [
        "frame_0000.png",
        "frame_0001.png",
        "frame_0002.jpg",
        "frame_0003.jpg",
        "frame_0004.png",
    ]
    manifest = json.loads(result.media_manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "opencat3d.web.media_batch_preflight"
    assert manifest["frames"]["frame_0000.png"]["source_media_id"] == (
        "already_uploaded.png"
    )
    assert manifest["frames"]["frame_0000.png"]["origin"] == "existing_upload"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["new_unique_frame_count"] == 3
    assert report["new_duplicate_frame_count"] == 1
    duplicate_group = report["duplicate_groups"][0]
    assert duplicate_group["kept_source_media_id"] == "already_uploaded.png"
    duplicate_record = duplicate_group["duplicate_records"][0]
    assert duplicate_record["origin"] == "new_upload"
    assert duplicate_record["source_media_id"] == "new_duplicate.png"


def test_run_uploaded_media_preflight_job_invokes_preflight_only_command(
    monkeypatch,
    tmp_path,
) -> None:
    uploads = [
        UploadedFile("a.png", _png_bytes(bbox=(1, 1, 6, 6))),
        UploadedFile("b.png", _png_bytes(bbox=(2, 2, 7, 7))),
    ]
    captured: dict[str, object] = {}

    def fake_run_preflight_command(command, *, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        loop_dir = Path(command[command.index("--loop-dir") + 1])
        media_preflight_path = loop_dir / "raw_media_preflight_report.json"
        promotion_gate_path = loop_dir / "raw_media_preflight_promotion_gate.json"
        loop_dir.mkdir(parents=True, exist_ok=True)
        (loop_dir / "raw_improvement_loop_report.json").write_text(
            json.dumps(
                {
                    "mode": "preflight_only",
                    "media_preflight": {
                        "report_path": str(media_preflight_path),
                        "ready_for_reconstruction": True,
                        "ready_for_promotion": True,
                        "warnings": [],
                    },
                    "media_preflight_promotion_gate": {
                        "report_path": str(promotion_gate_path),
                        "passed": True,
                        "rejection_reasons": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        web,
        "_run_improve_raw_job_preflight_command",
        fake_run_preflight_command,
    )

    result = web.run_uploaded_media_preflight_job(
        uploads,
        work_dir=tmp_path,
        min_selected_frames=2,
        max_selected_frames=6,
    )

    command = captured["command"]
    assert "--preflight-only" in command
    assert "--upload-dir" in command
    assert "--loop-dir" in command
    assert captured["cwd"] == web.PROJECT_ROOT
    assert result.job_id.startswith("preflight_")
    assert result.ready_for_reconstruction is True
    assert result.ready_for_promotion is True
    assert result.warnings == ()
    assert result.report_path == result.loop_dir / "raw_improvement_loop_report.json"


def test_run_uploaded_media_preflight_job_reuses_existing_uploads_and_cache(
    monkeypatch,
    tmp_path,
) -> None:
    existing_dir = tmp_path / "existing_upload"
    existing_dir.mkdir()
    duplicate_photo = _png_bytes(bbox=(1, 1, 5, 5))
    (existing_dir / "frame_0000.png").write_bytes(duplicate_photo)
    (existing_dir / "media_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.web.upload_staging",
                "frames": {
                    "frame_0000.png": {
                        "media_type": "photo",
                        "source_type": "image",
                        "source_media_id": "already_uploaded.png",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cache_report = tmp_path / "cache_report.json"
    cache_report.write_text("{}", encoding="utf-8")
    uploads = [
        UploadedFile("new_duplicate.png", duplicate_photo),
        UploadedFile("new_unique.png", _png_bytes(bbox=(2, 2, 6, 6))),
    ]
    captured: dict[str, object] = {}

    def fake_run_preflight_command(command, *, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        loop_dir = Path(command[command.index("--loop-dir") + 1])
        media_preflight_path = loop_dir / "raw_media_preflight_report.json"
        promotion_gate_path = loop_dir / "raw_media_preflight_promotion_gate.json"
        loop_dir.mkdir(parents=True, exist_ok=True)
        (loop_dir / "raw_improvement_loop_report.json").write_text(
            json.dumps(
                {
                    "mode": "preflight_only",
                    "media_preflight": {
                        "report_path": str(media_preflight_path),
                        "ready_for_reconstruction": True,
                        "ready_for_promotion": True,
                        "warnings": [],
                    },
                    "media_preflight_promotion_gate": {
                        "report_path": str(promotion_gate_path),
                        "passed": True,
                        "rejection_reasons": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        web,
        "_run_improve_raw_job_preflight_command",
        fake_run_preflight_command,
    )

    result = web.run_uploaded_media_preflight_job(
        uploads,
        work_dir=tmp_path,
        min_selected_frames=2,
        max_selected_frames=6,
        existing_upload_dirs=(existing_dir,),
        segmentation_cache_report=cache_report,
    )

    command = captured["command"]
    assert "--segmentation-cache-report" in command
    assert command[command.index("--segmentation-cache-report") + 1] == str(
        cache_report.resolve()
    )
    media_report = json.loads(
        (result.upload_dir / "media_preflight_report.json").read_text(encoding="utf-8")
    )
    assert media_report["existing_frame_count"] == 1
    assert media_report["new_unique_frame_count"] == 1
    assert media_report["new_duplicate_frame_count"] == 1
    assert media_report["duplicate_groups"][0]["kept_source_media_id"] == (
        "already_uploaded.png"
    )


def test_run_uploaded_media_preflight_job_writes_source_frame_gap_with_baseline(
    monkeypatch,
    tmp_path,
) -> None:
    baseline_dir = tmp_path / "baseline_project"
    quality_dir = baseline_dir / "outputs" / "quality"
    quality_dir.mkdir(parents=True)
    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "basecolor_observed_texel_fraction": 0.41,
                    "joint_projection_mean_keypoint_loss": 0.091,
                    "observed_face_fraction": 0.33,
                    "observed_vertex_fraction": 0.44,
                    "overlay_mean_silhouette_iou": 0.35,
                    "raw_facial_feature_count": 14,
                    "raw_quality_passed_check_count": 72,
                    "raw_quality_total_check_count": 72,
                    "silhouette_residual_mean_union_error": 0.64,
                }
            }
        ),
        encoding="utf-8",
    )
    existing_dir = tmp_path / "existing_upload"
    existing_dir.mkdir()
    duplicate_photo = _png_bytes(bbox=(1, 1, 5, 5))
    (existing_dir / "frame_0000.png").write_bytes(duplicate_photo)
    (existing_dir / "media_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.web.upload_staging",
                "frames": {
                    "frame_0000.png": {
                        "media_type": "photo",
                        "source_type": "image",
                        "source_media_id": "already_uploaded.png",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    texture_probe_report = tmp_path / "texture_probe.json"
    texture_probe_report.write_text(
        json.dumps(
            {
                "best_is_baseline": True,
                "pipeline_reports": [
                    {
                        "texture_only_added_unique_frame_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_preflight_command(command, *, cwd):
        del cwd
        loop_dir = Path(command[command.index("--loop-dir") + 1])
        media_preflight_path = loop_dir / "raw_media_preflight_report.json"
        promotion_gate_path = loop_dir / "raw_media_preflight_promotion_gate.json"
        loop_dir.mkdir(parents=True, exist_ok=True)
        media_preflight_path.write_text(
            json.dumps(
                {
                    "face_texture_frame_count": 2,
                    "geometry_safe_frame_count": 2,
                    "ready_for_reconstruction": True,
                    "ready_for_promotion": True,
                    "texture_usable_frame_count": 3,
                    "view_group_counts": {
                        "frontal_face": 1,
                        "side_body": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        promotion_gate_path.write_text("{}", encoding="utf-8")
        (loop_dir / "raw_improvement_loop_report.json").write_text(
            json.dumps(
                {
                    "mode": "preflight_only",
                    "media_preflight": {
                        "report_path": str(media_preflight_path),
                        "ready_for_reconstruction": True,
                        "ready_for_promotion": True,
                        "warnings": [],
                    },
                    "media_preflight_promotion_gate": {
                        "report_path": str(promotion_gate_path),
                        "passed": True,
                        "rejection_reasons": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        web,
        "_run_improve_raw_job_preflight_command",
        fake_run_preflight_command,
    )

    result = web.run_uploaded_media_preflight_job(
        [
            UploadedFile("new_duplicate.png", duplicate_photo),
            UploadedFile("new_unique.png", _png_bytes(bbox=(2, 2, 6, 6))),
        ],
        work_dir=tmp_path,
        min_selected_frames=2,
        max_selected_frames=6,
        existing_upload_dirs=(existing_dir,),
        baseline_project_dir=baseline_dir,
        texture_probe_report=texture_probe_report,
    )

    assert result.source_frame_acquisition_gap_report_path == (
        result.loop_dir / "source_frame_acquisition_gap.json"
    )
    gap = json.loads(
        result.source_frame_acquisition_gap_report_path.read_text(encoding="utf-8")
    )
    assert gap["stop_condition"] == {
        "reason": "insufficient_new_unique_evidence",
        "run_reconstruction_now": False,
    }
    assert gap["metrics"]["new_unique_frame_count"] == 1
    assert gap["metrics"]["new_duplicate_frame_count"] == 1
    loop_report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert loop_report["source_frame_acquisition_gap"]["report_path"] == str(
        result.source_frame_acquisition_gap_report_path
    )
    assert loop_report["artifacts"]["source_frame_acquisition_gap"] == str(
        result.source_frame_acquisition_gap_report_path
    )

    html = web.render_reconstruction_preflight_job(tmp_path, job_id=result.job_id)

    assert "Source Frame Acquisition Gap" in html
    assert "Run Later" in html
    assert "new_unique_evidence_required" in html


def test_reconstruction_preflight_context_resolves_state_paths(tmp_path) -> None:
    root = tmp_path / "web_runs"
    baseline_dir = root / "projects" / "best_project"
    quality_dir = baseline_dir / "outputs" / "quality"
    preflight_job_dir = root / "preflight_jobs" / "preflight_best"
    preflight_upload_dir = root / "preflight_uploads" / "preflight_best"
    texture_probe_report = root / "loop" / "texture_probe" / "raw_improvement_loop_report.json"
    quality_dir.mkdir(parents=True)
    preflight_job_dir.mkdir(parents=True)
    preflight_upload_dir.mkdir(parents=True)
    texture_probe_report.parent.mkdir(parents=True)
    (quality_dir / "raw_pipeline_summary.json").write_text("{}", encoding="utf-8")
    raw_preflight_report = preflight_job_dir / "raw_improvement_loop_report.json"
    raw_preflight_report.write_text("{}", encoding="utf-8")
    (preflight_upload_dir / "media_manifest.json").write_text("{}", encoding="utf-8")
    texture_probe_report.write_text("{}", encoding="utf-8")
    outside_report = tmp_path / "outside.json"
    outside_report.write_text("{}", encoding="utf-8")
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "current_best_project": str(baseline_dir),
                "last_new_downloads_preflight_raw_report": str(raw_preflight_report),
                "last_texture_only_probe_project_raw_report": str(texture_probe_report),
                "last_real_media_preflight_loop_report": str(outside_report),
            }
        ),
        encoding="utf-8",
    )

    context = web._resolve_reconstruction_preflight_context(  # noqa: SLF001
        root,
        state_path=state_path,
    )

    assert context.baseline_project_dir == baseline_dir.resolve()
    assert context.segmentation_cache_report == raw_preflight_report.resolve()
    assert context.existing_upload_dirs == (preflight_upload_dir.resolve(),)
    assert context.texture_probe_report == texture_probe_report.resolve()


def test_render_reconstruction_preflight_job_summarizes_gate(tmp_path) -> None:
    job_id = "preflight_render"
    loop_dir = tmp_path / "preflight_jobs" / job_id
    upload_dir = tmp_path / "preflight_uploads" / job_id
    loop_dir.mkdir(parents=True)
    upload_dir.mkdir(parents=True)
    (loop_dir / "raw_improvement_loop_report.json").write_text(
        json.dumps(
            {
                "mode": "preflight_only",
                "media_preflight": {
                    "ready_for_reconstruction": True,
                    "ready_for_promotion": False,
                    "geometry_safe_frame_count": 2,
                    "face_texture_frame_count": 0,
                    "texture_usable_frame_count": 2,
                    "recommended_geometry_frame_ids": ["body_a", "body_b"],
                    "recommended_texture_frame_ids": ["body_a", "body_b"],
                    "warnings": ["missing_high_confidence_face_texture_frame"],
                },
                "media_preflight_promotion_gate": {
                    "passed": False,
                    "rejection_reasons": ["face_texture_frame_present"],
                },
            }
        ),
        encoding="utf-8",
    )
    (upload_dir / "media_preflight_report.json").write_text(
        json.dumps(
            {
                "input_frame_count": 5,
                "unique_frame_count": 4,
                "duplicate_frame_count": 1,
                "existing_frame_count": 2,
                "new_unique_frame_count": 2,
                "new_duplicate_frame_count": 1,
                "photo_frame_count": 3,
                "video_frame_count": 1,
                "duplicate_groups": [
                    {
                        "kept_source_media_id": "already_uploaded.png",
                        "duplicate_records": [{"source_media_id": "new_duplicate.png"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    html = web.render_reconstruction_preflight_job(tmp_path, job_id=job_id)

    assert "Needs more evidence" in html
    assert "body_a" in html
    assert "missing_high_confidence_face_texture_frame" in html
    assert "face_texture_frame_present" in html
    assert "Media Batch" in html
    assert "Duplicate Frames" in html
    assert "already_uploaded.png kept" in html


def test_render_reconstruction_preflight_job_renders_source_frame_gap(
    tmp_path,
) -> None:
    job_id = "preflight_gap"
    loop_dir = tmp_path / "preflight_jobs" / job_id
    upload_dir = tmp_path / "preflight_uploads" / job_id
    loop_dir.mkdir(parents=True)
    upload_dir.mkdir(parents=True)
    gap_path = loop_dir / "source_frame_acquisition_gap.json"
    _write_source_frame_acquisition_gap_report(gap_path)
    (loop_dir / "raw_improvement_loop_report.json").write_text(
        json.dumps(
            {
                "mode": "preflight_only",
                "media_preflight": {
                    "ready_for_reconstruction": False,
                    "ready_for_promotion": False,
                    "geometry_safe_frame_count": 1,
                    "face_texture_frame_count": 0,
                    "texture_usable_frame_count": 1,
                    "recommended_geometry_frame_ids": ["body_a"],
                    "recommended_texture_frame_ids": ["body_a"],
                    "warnings": ["missing_new_unique_frames"],
                },
                "media_preflight_promotion_gate": {
                    "passed": False,
                    "rejection_reasons": ["new_unique_frame_count"],
                },
                "source_frame_acquisition_gap": {
                    "report_path": str(gap_path),
                },
            }
        ),
        encoding="utf-8",
    )
    (upload_dir / "media_preflight_report.json").write_text(
        json.dumps(
            {
                "input_frame_count": 3,
                "unique_frame_count": 2,
                "duplicate_frame_count": 1,
                "existing_frame_count": 2,
                "new_unique_frame_count": 0,
                "new_duplicate_frame_count": 1,
                "photo_frame_count": 2,
                "video_frame_count": 1,
                "duplicate_groups": [],
            }
        ),
        encoding="utf-8",
    )

    html = web.render_reconstruction_preflight_job(tmp_path, job_id=job_id)

    assert "Source Frame Acquisition Gap" in html
    assert "Run Reconstruction" in html
    assert "no" in html
    assert "new_unique_evidence_required" in html
    assert "side_body" in html
    assert "Do not run full reconstruction yet" in html


def test_uploaded_scan_jobs_are_rejected_until_adapters_exist(tmp_path) -> None:
    with pytest.raises(ValueError, match="Scan upload is not implemented"):
        run_uploaded_image_job(
            [
                UploadedFile("cat.obj", b"not-a-scan-adapter"),
                UploadedFile("cat.png", _png_bytes(bbox=(1, 1, 5, 5))),
            ],
            work_dir=tmp_path,
        )


def test_job_id_includes_quality_parameters() -> None:
    uploads = [
        UploadedFile("a.png", _png_bytes(bbox=(1, 1, 5, 5))),
        UploadedFile("b.png", _png_bytes(bbox=(2, 2, 6, 6))),
    ]

    smoke_test_id = web._job_id(  # noqa: SLF001
        uploads,
        color_delta_threshold=0.10,
        camera_root_steps=2,
        segmentation_backend="auto_subject_v1",
    )
    quality_id = web._job_id(  # noqa: SLF001
        uploads,
        color_delta_threshold=0.10,
        camera_root_steps=30,
        segmentation_backend="auto_subject_v1",
    )
    backend_id = web._job_id(  # noqa: SLF001
        uploads,
        color_delta_threshold=0.10,
        camera_root_steps=30,
        segmentation_backend="alpha_or_border_color_v1",
    )

    assert smoke_test_id != quality_id
    assert backend_id != quality_id


def test_index_page_exposes_images_and_marks_future_adapters() -> None:
    html = render_index()

    assert "OpenCat3D" in html
    assert "Cat Identity" in html
    assert "Identify Cat" in html
    assert "Add cat photos or video" in html
    assert "Selected media" in html
    assert "No media selected yet." in html
    assert "data-file-input" in html
    assert "data-file-list" in html
    assert "DataTransfer" in html
    assert "stagedFiles" in html
    assert "Add Photos Or Video" in html
    assert "Check Media" in html
    assert 'formaction="/jobs/preflight"' in html
    assert 'name="camera_root_steps" type="number" value="30"' in html
    assert "2 is only a smoke test" in html
    assert "Videos" in html
    assert "Video upload uses deterministic frame extraction" in html
    assert "Scans" in html
    assert html.count("Adapter needed") == 1


def test_identity_image_job_creates_then_matches_identity(tmp_path) -> None:
    first_upload = [UploadedFile("cat.png", _png_bytes(bbox=(1, 1, 5, 5)))]
    captured_at = datetime(2026, 1, 1, 10, tzinfo=UTC)

    first = run_identity_image_job(
        first_upload,
        work_dir=tmp_path,
        captured_at=captured_at,
        site_id="yard",
        camera_id="front",
    )
    second = run_identity_image_job(
        first_upload,
        work_dir=tmp_path,
        captured_at=captured_at + timedelta(days=1),
        site_id="yard",
        camera_id="front",
    )

    assert first.results[0].decision == "create_new_identity"
    assert second.results[0].decision == "match_existing"
    assert second.results[0].cat_id == first.results[0].cat_id
    assert (tmp_path / "catnet" / "catalog.json").exists()
    store = CatNetStore(tmp_path / "catnet")
    matched_edges = [
        edge
        for edge in store.catalog.match_edges.values()
        if edge.decision.value == "match_existing"
    ]
    assert matched_edges
    assert matched_edges[-1].local_score is not None
    assert matched_edges[-1].metadata_score is not None
    assert matched_edges[-1].explanation["local_rerank"]["status"] == "ok"
    assert matched_edges[-1].explanation["metadata"]["site"] == "same"

    html = render_identity_job(tmp_path, job_id=second.job_id)
    assert "Identity Decisions" in html
    assert second.results[0].cat_id in html


def test_http_post_parses_multipart_uploads(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_uploaded_image_job(
        uploads,
        *,
        work_dir,
        color_delta_threshold,
        camera_root_steps,
        segmentation_backend,
    ):
        captured["uploads"] = uploads
        captured["work_dir"] = work_dir
        captured["color_delta_threshold"] = color_delta_threshold
        captured["camera_root_steps"] = camera_root_steps
        captured["segmentation_backend"] = segmentation_backend
        return SimpleNamespace(job_id="job_http")

    monkeypatch.setattr(web, "run_uploaded_image_job", fake_run_uploaded_image_job)
    server = web.make_web_server(host="127.0.0.1", port=0, work_dir=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "OpenCat3DTestBoundary"
        body = _multipart_body(
            boundary=boundary,
            files=[
                ("b.png", _png_bytes(bbox=(2, 2, 6, 6))),
                ("a.png", _png_bytes(bbox=(1, 1, 5, 5))),
            ],
            fields={
                "color_delta_threshold": "0.25",
                "camera_root_steps": "7",
            },
        )
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        connection.request(
            "POST",
            "/jobs",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 303
    assert response.headers["Location"] == "/jobs/job_http"
    assert [upload.filename for upload in captured["uploads"]] == ["b.png", "a.png"]
    assert captured["work_dir"] == tmp_path.resolve()
    assert captured["color_delta_threshold"] == 0.25
    assert captured["camera_root_steps"] == 7
    assert captured["segmentation_backend"] == "auto_subject_v1"


def test_http_post_parses_preflight_uploads(monkeypatch, tmp_path) -> None:
    captured = {}
    fake_context = web.ReconstructionPreflightContext(
        existing_upload_dirs=(tmp_path / "existing",),
        segmentation_cache_report=tmp_path / "cache.json",
        baseline_project_dir=tmp_path / "baseline",
        texture_probe_report=tmp_path / "texture_probe.json",
    )

    def fake_run_uploaded_media_preflight_job(
        uploads,
        *,
        work_dir,
        min_selected_frames,
        max_selected_frames,
        existing_upload_dirs,
        segmentation_cache_report,
        baseline_project_dir,
        texture_probe_report,
    ):
        captured["uploads"] = uploads
        captured["work_dir"] = work_dir
        captured["min_selected_frames"] = min_selected_frames
        captured["max_selected_frames"] = max_selected_frames
        captured["existing_upload_dirs"] = existing_upload_dirs
        captured["segmentation_cache_report"] = segmentation_cache_report
        captured["baseline_project_dir"] = baseline_project_dir
        captured["texture_probe_report"] = texture_probe_report
        return SimpleNamespace(job_id="preflight_http")

    monkeypatch.setattr(
        web,
        "run_uploaded_media_preflight_job",
        fake_run_uploaded_media_preflight_job,
    )
    monkeypatch.setattr(
        web,
        "_resolve_reconstruction_preflight_context",
        lambda root_dir: fake_context,
    )
    server = web.make_web_server(host="127.0.0.1", port=0, work_dir=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "OpenCat3DPreflightBoundary"
        body = _multipart_body(
            boundary=boundary,
            files=[
                ("b.png", _png_bytes(bbox=(2, 2, 6, 6))),
                ("a.png", _png_bytes(bbox=(1, 1, 5, 5))),
            ],
            fields={},
        )
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        connection.request(
            "POST",
            "/jobs/preflight",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 303
    assert response.headers["Location"] == "/preflight/jobs/preflight_http"
    assert [upload.filename for upload in captured["uploads"]] == ["b.png", "a.png"]
    assert captured["work_dir"] == tmp_path.resolve()
    assert captured["min_selected_frames"] == 2
    assert captured["max_selected_frames"] == 8
    assert captured["existing_upload_dirs"] == fake_context.existing_upload_dirs
    assert captured["segmentation_cache_report"] == fake_context.segmentation_cache_report
    assert captured["baseline_project_dir"] == fake_context.baseline_project_dir
    assert captured["texture_probe_report"] == fake_context.texture_probe_report


def test_http_post_parses_identity_uploads(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_identity_image_job(
        uploads,
        *,
        work_dir,
        captured_at,
        site_id,
        camera_id,
        uploader,
        model_path,
    ):
        captured["uploads"] = uploads
        captured["work_dir"] = work_dir
        captured["captured_at"] = captured_at
        captured["site_id"] = site_id
        captured["camera_id"] = camera_id
        captured["uploader"] = uploader
        captured["model_path"] = model_path
        return SimpleNamespace(job_id="idjob_http")

    monkeypatch.setattr(web, "run_identity_image_job", fake_run_identity_image_job)
    server = web.make_web_server(host="127.0.0.1", port=0, work_dir=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "CatNetIdentityBoundary"
        body = _multipart_body(
            boundary=boundary,
            files=[("cat.png", _png_bytes(bbox=(1, 1, 5, 5)))],
            fields={
                "captured_at": "2026-01-01T10:30",
                "site_id": "yard",
                "camera_id": "front",
                "uploader": "local",
                "model_path": "catnet_artifacts/catnet_reid_model.json",
            },
        )
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        connection.request(
            "POST",
            "/identity/jobs",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 303
    assert response.headers["Location"] == "/identity/jobs/idjob_http"
    assert [upload.filename for upload in captured["uploads"]] == ["cat.png"]
    assert captured["work_dir"] == tmp_path.resolve()
    assert captured["captured_at"] == datetime(2026, 1, 1, 10, 30, tzinfo=UTC)
    assert captured["site_id"] == "yard"
    assert captured["camera_id"] == "front"
    assert captured["uploader"] == "local"
    assert captured["model_path"].as_posix() == "catnet_artifacts/catnet_reid_model.json"


def test_serve_command_is_available_without_starting_server() -> None:
    result = CliRunner().invoke(app, ["serve", "--help"])

    assert result.exit_code == 0, result.output
    assert "local upload website" in result.output
    assert "--work-dir" in result.output


def test_catnet_identify_command_is_available_without_running(tmp_path) -> None:
    result = CliRunner().invoke(app, ["catnet", "identify", "--help"])

    assert result.exit_code == 0, result.output
    assert "CatNet identity" in result.output
    assert "--store-dir" in result.output


def test_catnet_review_queue_command_is_available_without_running() -> None:
    result = CliRunner().invoke(app, ["catnet", "review-queue", "--help"])

    assert result.exit_code == 0, result.output
    assert "active-learning review queue" in result.output
    assert "--output-path" in result.output


def _write_source_frame_acquisition_gap_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.evaluation.evidence_gap",
                "success": True,
                "metrics": {
                    "basecolor_observed_texel_fraction": 0.41481995582580566,
                    "new_duplicate_frame_count": 31,
                    "new_unique_frame_count": 0,
                    "observed_face_fraction": 0.3283261802575107,
                    "observed_vertex_fraction": 0.4394572025052192,
                    "overlay_mean_silhouette_iou": 0.35739877535217474,
                    "joint_projection_mean_keypoint_loss": 0.09013199753122704,
                },
                "metric_gaps": [
                    {
                        "metric": "basecolor_observed_texel_fraction",
                        "current": 0.41481995582580566,
                        "target": 0.55,
                        "gap": 0.13518004417419438,
                        "direction": "increase",
                    },
                    {
                        "metric": "observed_face_fraction",
                        "current": 0.3283261802575107,
                        "target": 0.45,
                        "gap": 0.12167381974248931,
                        "direction": "increase",
                    },
                    {
                        "metric": "new_unique_frame_count",
                        "current": 0,
                        "target": 6,
                        "gap": 6,
                        "direction": "increase",
                    },
                ],
                "view_gaps": [
                    {
                        "view_group": "frontal_face",
                        "current": 2,
                        "target": 3,
                        "missing": 1,
                    },
                    {
                        "view_group": "side_body",
                        "current": 1,
                        "target": 2,
                        "missing": 1,
                    },
                    {
                        "view_group": "rear_body",
                        "current": 0,
                        "target": 1,
                        "missing": 1,
                    },
                ],
                "recommendations": [
                    {
                        "id": "new_unique_evidence_required",
                        "reason": (
                            "The latest upload did not add enough non-duplicate "
                            "evidence to justify another reconstruction run."
                        ),
                        "capture_instructions": [
                            "Run preflight before full reconstruction and require "
                            "new_unique_frame_count > 0.",
                        ],
                    }
                ],
                "stop_condition": {
                    "reason": "insufficient_new_unique_evidence",
                    "run_reconstruction_now": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _png_bytes(*, bbox: tuple[int, int, int, int]) -> bytes:
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = np.asarray([222, 220, 208], dtype=np.uint8)
    if x1 - x0 >= 4 and y1 - y0 >= 4:
        image[y0, x0 + 1, :] = np.asarray([118, 104, 54], dtype=np.uint8)
        image[y0, x1 - 2, :] = np.asarray([118, 104, 54], dtype=np.uint8)
        image[y0 + 1, x0 + 1, :] = np.asarray([10, 9, 8], dtype=np.uint8)
        image[y0 + 1, x1 - 2, :] = np.asarray([10, 9, 8], dtype=np.uint8)
        image[y0 + 2, x0 + 2, :] = np.asarray([224, 136, 146], dtype=np.uint8)
        image[y0 + 3, x0 + 2, :] = np.asarray([52, 36, 38], dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _multipart_body(
    *,
    boundary: str,
    files: list[tuple[str, bytes]],
    fields: dict[str, str],
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )
    for filename, content in files:
        chunks.append(
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="files"; '
                f'filename="{filename}"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode()
        )
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)
