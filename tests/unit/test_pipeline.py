import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import opencat3d.pipeline as pipeline
from opencat3d.cli import app
from opencat3d.evaluation.raw import RAW_METRIC_THRESHOLDS
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.pipeline import (
    _raw_texture_export_refresh_needed,
    _stage_raw_pipeline_input_media,
    _write_baseline_geometry_texture_manifest,
    run_baseline_geometry_texture_bake,
    run_raw_pipeline,
    run_synthetic_pipeline,
    write_texture_source_admission_audit,
)
from opencat3d.template.cat_template import (
    CAT_TEMPLATE_FACE_COUNT,
    CAT_TEMPLATE_VERTEX_COUNT,
    create_placeholder_cat_template,
)
from opencat3d.textures.basecolor import (
    _exclude_protected_face_projection,
    _protected_face_region_mask,
    _source_patch_candidate_allowed,
)


def test_raw_texture_export_refresh_only_when_glb_texture_is_below_target() -> None:
    assert not _raw_texture_export_refresh_needed(
        {
            "raw_template_region_v2_texture_target_bake_used": 1.0,
            "raw_template_region_v2_texture_target_reported_size": 2048,
            "raw_glb_texture_size": 2048,
        }
    )
    assert _raw_texture_export_refresh_needed(
        {
            "raw_template_region_v2_texture_target_bake_used": 1.0,
            "raw_template_region_v2_texture_target_reported_size": 2048,
            "raw_glb_texture_size": 1024,
        }
    )
    assert not _raw_texture_export_refresh_needed(
        {
            "raw_template_region_v2_texture_target_bake_used": 0.0,
            "raw_template_region_v2_texture_target_reported_size": 2048,
            "raw_glb_texture_size": 1024,
        }
    )


def test_baseline_geometry_texture_bake_seeds_export_from_baseline_texture(
    tmp_path,
    monkeypatch,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    seed_dir = baseline / "outputs" / "textures"
    (baseline / "outputs" / "quality").mkdir(parents=True)
    seed_dir.mkdir(parents=True)
    source.mkdir()
    (baseline / "outputs" / "quality" / "raw_pipeline_summary.json").write_text(
        json.dumps({"steps": {}, "metrics": {}, "artifacts": {}}),
        encoding="utf-8",
    )

    def fake_manifest(*args, **kwargs):
        del args, kwargs
        return {
            "texture_frame_count": 2,
            "texture_media_counts": {"photo": 1, "video": 1},
            "added_unique_texture_frame_count": 1,
            "skipped_duplicate_texture_frame_count": 0,
            "rejected_low_detail_texture_frame_count": 0,
            "rejected_ablation_texture_frame_count": 0,
        }

    def fake_export(project_dir, *, mode, basecolor_seed_texture_dir, force):
        assert project_dir == project.resolve()
        assert mode == "raw"
        assert basecolor_seed_texture_dir == seed_dir
        assert force is True
        avatar_dir = project_dir / "outputs" / "avatar"
        return SimpleNamespace(
            obj_path=avatar_dir / "cat_canonical.obj",
            ply_path=avatar_dir / "cat_canonical.ply",
            glb_path=avatar_dir / "cat_avatar.glb",
            idle_animation_path=avatar_dir / "animations" / "idle.glb",
            skeleton_path=avatar_dir / "skeleton.json",
        )

    monkeypatch.setattr(pipeline, "_write_baseline_geometry_texture_manifest", fake_manifest)
    monkeypatch.setattr(pipeline, "export_fitted_canonical_mesh", fake_export)
    monkeypatch.setattr(
        pipeline,
        "evaluate_raw_reconstruction",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            metrics={"basecolor_observed_texel_fraction": 0.5},
            report_path=project / "outputs" / "quality" / "raw_evaluation.json",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_texture_only_promotion_guard",
        lambda *args, **kwargs: SimpleNamespace(
            passed=True,
            report_path=project
            / "outputs"
            / "quality"
            / "texture_only_promotion_guard.json",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_raw_objective_compliance_audit",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            report_path=project / "outputs" / "quality" / "compliance.json",
        ),
    )

    result = run_baseline_geometry_texture_bake(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        force=True,
    )

    assert result.success is True


def test_baseline_geometry_texture_bake_records_texture_only_evaluation_success(
    tmp_path,
    monkeypatch,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    seed_dir = baseline / "outputs" / "textures"
    (baseline / "outputs" / "quality").mkdir(parents=True)
    seed_dir.mkdir(parents=True)
    source.mkdir()
    (baseline / "outputs" / "quality" / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "steps": {},
                "metrics": {
                    "raw_quality_passed_check_count": 10,
                    "raw_quality_total_check_count": 10,
                },
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pipeline,
        "_write_baseline_geometry_texture_manifest",
        lambda *args, **kwargs: {
            "texture_frame_count": 3,
            "texture_media_counts": {"photo": 3, "video": 0},
            "added_unique_texture_frame_count": 1,
            "skipped_duplicate_texture_frame_count": 0,
            "rejected_low_detail_texture_frame_count": 0,
            "rejected_ablation_texture_frame_count": 0,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "export_fitted_canonical_mesh",
        lambda project_dir, **kwargs: SimpleNamespace(
            obj_path=project_dir / "outputs" / "avatar" / "cat_canonical.obj",
            ply_path=project_dir / "outputs" / "avatar" / "cat_canonical.ply",
            glb_path=project_dir / "outputs" / "avatar" / "cat_avatar.glb",
            idle_animation_path=project_dir
            / "outputs"
            / "avatar"
            / "animations"
            / "idle.glb",
            skeleton_path=project_dir / "outputs" / "avatar" / "skeleton.json",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_raw_reconstruction",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            metrics={
                "raw_quality_passed_check_count": 10,
                "raw_quality_total_check_count": 10,
            },
            report_path=project / "outputs" / "quality" / "raw_evaluation.json",
        ),
    )

    def fake_promotion_guard(*args, **kwargs):
        del args, kwargs
        summary = json.loads(
            (project / "outputs" / "quality" / "raw_pipeline_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["success"] is True
        assert summary["steps"]["evaluate"] is True
        assert summary["steps"]["raw_evaluate"] is False
        assert summary["metrics"]["raw_texture_only_evaluation_success"] == 1.0
        assert summary["metrics"]["raw_texture_only_raw_evaluation_success"] == 0.0
        return SimpleNamespace(
            passed=True,
            report_path=project
            / "outputs"
            / "quality"
            / "texture_only_promotion_guard.json",
        )

    monkeypatch.setattr(
        pipeline,
        "write_texture_only_promotion_guard",
        fake_promotion_guard,
    )
    monkeypatch.setattr(
        pipeline,
        "write_raw_objective_compliance_audit",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            report_path=project / "outputs" / "quality" / "compliance.json",
        ),
    )

    result = run_baseline_geometry_texture_bake(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        force=True,
    )

    assert result.success is False
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["success"] is True
    assert summary["texture_only_evaluation"]["quality_checks_passed"] is True


def test_baseline_geometry_texture_bake_recovers_fixed_geometry_alignment(
    tmp_path,
    monkeypatch,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    (baseline / "outputs" / "quality").mkdir(parents=True)
    (baseline / "outputs" / "textures").mkdir(parents=True)
    (baseline / "observations" / "raw").mkdir(parents=True)
    source.mkdir()
    (baseline / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "selected_frame_ids": ["frame_0001", "frame_0002"],
                "texture_frame_ids": ["frame_0001", "frame_0002"],
            }
        ),
        encoding="utf-8",
    )
    (baseline / "outputs" / "quality" / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "steps": {},
                "metrics": {
                    "overlay_mean_silhouette_iou": 0.42,
                    "silhouette_residual_mean_union_error": 0.58,
                    "joint_projection_mean_keypoint_loss": 0.09,
                },
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    (baseline / "outputs" / "quality" / "raw_silhouette_residuals.json").write_text(
        json.dumps(
            {
                "success": True,
                "metrics": {
                    "mean_silhouette_iou": 0.42,
                    "mean_union_error_fraction": 0.58,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pipeline,
        "_write_baseline_geometry_texture_manifest",
        lambda *args, **kwargs: {
            "texture_frame_count": 3,
            "texture_media_counts": {"photo": 3, "video": 0},
            "added_unique_texture_frame_count": 1,
            "skipped_duplicate_texture_frame_count": 0,
            "rejected_low_detail_texture_frame_count": 0,
            "rejected_ablation_texture_frame_count": 0,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "export_fitted_canonical_mesh",
        lambda project_dir, **kwargs: SimpleNamespace(
            obj_path=project_dir / "outputs" / "avatar" / "cat_canonical.obj",
            ply_path=project_dir / "outputs" / "avatar" / "cat_canonical.ply",
            glb_path=project_dir / "outputs" / "avatar" / "cat_avatar.glb",
            idle_animation_path=project_dir
            / "outputs"
            / "avatar"
            / "animations"
            / "idle.glb",
            skeleton_path=project_dir / "outputs" / "avatar" / "skeleton.json",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_raw_reconstruction",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            metrics={
                "overlay_mean_silhouette_iou": 0.0,
                "silhouette_residual_mean_union_error": 0.58,
                "joint_projection_mean_keypoint_loss": 0.09,
            },
            report_path=project / "outputs" / "quality" / "raw_evaluation.json",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_texture_only_promotion_guard",
        lambda *args, **kwargs: SimpleNamespace(
            passed=True,
            report_path=project
            / "outputs"
            / "quality"
            / "texture_only_promotion_guard.json",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_raw_objective_compliance_audit",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            report_path=project / "outputs" / "quality" / "compliance.json",
        ),
    )

    result = run_baseline_geometry_texture_bake(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        force=True,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["overlay_mean_silhouette_iou"] == pytest.approx(0.42)
    assert (
        summary["metrics"]["raw_texture_only_fixed_geometry_alignment_recovered"]
        == 1.0
    )
    assert (
        summary["texture_only_fixed_geometry_alignment"]["metric_overrides"][
            "overlay_mean_silhouette_iou"
        ]["source"]
        == "outputs/quality/raw_silhouette_residuals.json:metrics.mean_silhouette_iou"
    )


def test_stage_raw_pipeline_input_media_extracts_video_frames(tmp_path) -> None:
    project_dir = tmp_path / "project"
    photo_path = tmp_path / "cat.jpg"
    video_path = tmp_path / "cat.mp4"
    Image.fromarray(np.full((12, 16, 3), 220, dtype=np.uint8)).save(photo_path)
    _write_test_video(video_path, frame_count=6, size=(16, 12))

    staged_paths = _stage_raw_pipeline_input_media(
        project_dir,
        [photo_path, video_path],
        min_video_frames=2,
        max_video_frames=3,
        force=True,
    )

    assert staged_paths == (project_dir / "raw" / "staged_inputs",)
    media_manifest = json.loads(
        (staged_paths[0] / "media_manifest.json").read_text(encoding="utf-8")
    )
    assert len(media_manifest["frames"]) == 4
    assert media_manifest["frames"]["frame_0000.jpg"]["media_type"] == "photo"
    assert media_manifest["frames"]["frame_0000.jpg"]["source_media_id"] == "cat.jpg"
    video_frame = media_manifest["frames"]["frame_0001.jpg"]
    assert video_frame["media_type"] == "video"
    assert video_frame["source_type"] == "video_frame"
    assert video_frame["source_media_id"] == "cat.mp4"
    assert video_frame["source_frame_index"] == 0

    ingest_result = ingest_image_sources(project_dir, list(staged_paths), force=True)
    assert ingest_result.frame_count == 4
    assert ingest_result.frames[1].source_type.value == "video_frame"
    assert ingest_result.frames[1].source_media_id == "cat.mp4"


def test_write_baseline_geometry_texture_manifest_namespaces_new_frames(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    _write_texture_observation(baseline, "body_a", b"same-pixels", media_type="video")
    _write_texture_observation(source, "dup", b"same-pixels", media_type="video")
    _write_texture_observation(source, "body_a", b"new-face-pixels", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["dup", "body_a"])
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        force=True,
    )

    assert report["added_unique_texture_frame_count"] == 1
    assert report["skipped_duplicate_texture_frame_count"] == 1
    assert report["texture_source_preflight_accepted_frame_count"] == 1
    assert report["texture_source_preflight_duplicate_frame_count"] == 1
    assert report["texture_source_admission_audit_passed"] is True
    assert report["texture_source_admission_unique_accepted_frame_count"] == 1
    assert report["texture_source_admission_protected_face_frame_count"] == 1
    assert report["selected_frame_ids"] == ["body_a"]
    preflight = json.loads(
        (project / "outputs" / "quality" / "texture_source_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert preflight["accepted_frame_count"] == 1
    assert preflight["duplicate_frame_count"] == 1
    assert preflight["ranked_frames"][0]["source_frame_id"] == "body_a"
    assert preflight["ranked_frames"][0]["accepted_for_bake"] is True
    admission = json.loads(
        (
            project / "outputs" / "quality" / "texture_source_admission_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert admission["passed"] is True
    assert admission["unique_accepted_frame_count"] == 1
    assert admission["unique_accepted_frame_ids"] == ["body_a"]
    assert admission["protected_face_source_patch_frame_count"] == 1
    assert admission["protected_face_source_patch_frame_ids"] == ["body_a"]
    assert admission["ranked_frames"][0]["source_patch_allowed"] is True
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert texture_manifest["texture_frame_count"] == 2
    assert texture_manifest["texture_frame_ids"][0] == "body_a"
    assert texture_manifest["texture_frame_ids"][1].startswith("texture_source_body_a")
    assert len(set(texture_manifest["texture_frame_ids"])) == 2
    supplemental = texture_manifest["frames"][1]
    assert supplemental["texture_only_source_frame_id"] == "body_a"
    assert supplemental["texture_only_supplemental"] is True
    protection = supplemental["texture_only_face_region_projection_protection"]
    assert protection["broad_projection_excluded"] is True
    assert protection["source_patch_allowed"] is True
    assert "left_eye" in protection["protected_region_names"]
    assert report["face_region_projection_protection_applied_frame_count"] == 1
    assert texture_manifest["texture_only_bake"][
        "face_region_projection_protection_applied_frame_count"
    ] == 1
    assert texture_manifest["texture_only_bake"][
        "texture_source_admission_audit_passed"
    ] is True
    assert report["rejected_low_detail_texture_frame_count"] == 0


def test_write_baseline_geometry_texture_manifest_skips_stale_baseline_frames(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_manifest(baseline, ["body_a", "stale_body"])
    _write_texture_observation(source, "face_good", b"new-face-pixels", media_type="photo")
    _write_texture_manifest(source, ["face_good"])
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.82,
                "frame_id": "face_good",
                "reject_reasons": [],
                "texture_detail_candidate": True,
            },
        ],
    )
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        force=True,
    )

    assert report["baseline_texture_frame_count"] == 2
    assert report["baseline_texture_hash_frame_count"] == 1
    assert report["baseline_texture_missing_image_frame_count"] == 1
    assert report["baseline_texture_missing_image_frames"][0]["frame_id"] == "stale_body"
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "stale_body" not in texture_manifest["texture_frame_ids"]
    assert texture_manifest["texture_only_bake"][
        "baseline_texture_missing_image_frame_count"
    ] == 1


def test_texture_source_admission_audit_scopes_ablation_to_face_patches(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_good", b"new-face-pixels", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_good"])
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.82,
                "frame_id": "face_good",
                "reject_reasons": [],
                "texture_detail_candidate": True,
            },
        ],
    )

    report = write_texture_source_admission_audit(
        baseline,
        source,
        tmp_path / "admission.json",
        rejected_texture_source_frame_ids=["face_good"],
        force=True,
    )

    assert report["passed"] is True
    assert report["checks"]["top_score_meets_floor"] is True
    assert report["accepted_preflight_frame_count"] == 1
    assert report["checks"]["has_unique_accepted_texture_frames"] is True
    assert report["checks"]["has_unique_protected_face_source_patch_frames"] is False
    assert report["face_patch_passed"] is False
    assert report["face_patch_rejection_reasons"] == [
        "has_unique_protected_face_source_patch_frames"
    ]
    assert report["unique_accepted_frame_count"] == 1
    assert report["unique_accepted_frame_ids"] == ["face_good"]
    assert report["protected_face_source_patch_frame_count"] == 0
    assert report["protected_face_source_patch_frame_ids"] == []
    assert report["source_patch_blocked_ablation_frame_count"] == 1
    assert report["source_patch_blocked_ablation_frame_ids"] == ["face_good"]
    assert report["rejected_ablation_frame_count"] == 0
    assert report["rejected_ablation_frame_ids"] == []
    assert report["rejection_reasons"] == []
    assert report["ranked_frames"][0]["blocked_by_ablation"] is True
    assert report["ranked_frames"][0]["admission_reason"] == (
        "unique_texture_face_patch_blocked_by_ablation"
    )
    assert report["ranked_frames"][0]["source_patch_allowed"] is False


def test_texture_source_admission_audit_skips_stale_baseline_texture_frames(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_manifest(baseline, ["body_a", "stale_body"])
    _write_texture_observation(source, "face_good", b"new-face-pixels", media_type="photo")
    _write_texture_manifest(source, ["face_good"])
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.82,
                "frame_id": "face_good",
                "reject_reasons": [],
                "texture_detail_candidate": True,
            },
        ],
    )

    report = write_texture_source_admission_audit(
        baseline,
        source,
        tmp_path / "admission.json",
        force=True,
    )

    assert report["passed"] is True
    assert report["baseline_texture_frame_count"] == 2
    assert report["baseline_texture_hash_frame_count"] == 1
    assert report["baseline_texture_missing_image_frame_count"] == 1
    assert report["baseline_texture_missing_image_frames"][0]["frame_id"] == "stale_body"
    assert report["unique_accepted_frame_count"] == 1


def test_texture_source_admission_allows_observed_anchor_closeup_border_partial_body(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "white_face", b"new-white-face", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["white_face"])
    _write_face_anchor_manifest(
        source,
        "white_face",
        {
            "left_eye_center": (35.0, 36.0, 0.42),
            "right_eye_center": (52.0, 37.0, 0.41),
            "nose_tip": (45.0, 52.0, 0.36),
            "muzzle_center": (44.0, 62.0, 0.34),
            "left_ear_inner": (31.0, 18.0, 0.30),
            "right_ear_inner": (65.0, 20.0, 0.31),
        },
    )
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.10,
                "frame_id": "white_face",
                "reject_reasons": ["border_partial_body"],
                "texture_detail_candidate": False,
            },
        ],
    )

    report = write_texture_source_admission_audit(
        baseline,
        source,
        tmp_path / "admission.json",
        force=True,
    )

    assert report["passed"] is True
    assert report["accepted_preflight_frame_count"] == 1
    assert report["unique_accepted_frame_count"] == 1
    assert report["unique_accepted_frame_ids"] == ["white_face"]
    assert report["protected_face_source_patch_frame_count"] == 1
    assert report["protected_face_source_patch_frame_ids"] == ["white_face"]
    accepted = report["ranked_frames"][0]
    assert accepted["accepted_for_bake"] is True
    assert accepted["source_patch_allowed"] is True
    gate = accepted["quality_gate"]
    assert gate["reason"] == "observed_face_anchor_source_patch_candidate"
    assert gate["severe_reject_reasons"] == []
    assert gate["observed_face_anchor_source_patch_candidate"] is True
    assert gate["observed_face_anchor_source_patch_gate"][
        "allowed_border_partial_body"
    ] is True


def test_texture_source_admission_observed_anchor_closeup_keeps_sparse_mask_reject(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "sparse_face", b"new-sparse-face", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["sparse_face"])
    _write_face_anchor_manifest(
        source,
        "sparse_face",
        {
            "left_eye_center": (35.0, 36.0, 0.42),
            "right_eye_center": (52.0, 37.0, 0.41),
            "nose_tip": (45.0, 52.0, 0.36),
            "muzzle_center": (44.0, 62.0, 0.34),
            "left_ear_inner": (31.0, 18.0, 0.30),
            "right_ear_inner": (65.0, 20.0, 0.31),
        },
    )
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.90,
                "frame_id": "sparse_face",
                "reject_reasons": ["sparse_geometry_mask"],
                "texture_detail_candidate": True,
            },
        ],
    )

    report = write_texture_source_admission_audit(
        baseline,
        source,
        tmp_path / "admission.json",
        force=True,
    )

    assert report["passed"] is False
    assert report["accepted_preflight_frame_count"] == 0
    rejected = report["ranked_frames"][0]
    assert rejected["accepted_for_bake"] is False
    gate = rejected["quality_gate"]
    assert gate["reason"] == "severe_quality_reject"
    assert gate["observed_face_anchor_source_patch_candidate"] is False
    assert gate["observed_face_anchor_source_patch_gate"]["hard_reject_reasons"] == [
        "sparse_geometry_mask"
    ]


def test_write_baseline_geometry_texture_manifest_rejects_low_detail_sources(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_good", b"good-face-pixels", media_type="photo")
    _write_texture_observation(source, "body_bad", b"bad-body-pixels", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_good", "body_bad"])
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.72,
                "frame_id": "face_good",
                "reject_reasons": [],
                "texture_detail_candidate": True,
            },
            {
                "face_texture_detail_score": 0.12,
                "frame_id": "body_bad",
                "reject_reasons": ["border_partial_body"],
                "texture_detail_candidate": False,
            },
        ],
    )
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        force=True,
    )

    assert report["added_unique_texture_frame_count"] == 1
    assert report["rejected_low_detail_texture_frame_count"] == 1
    assert report["texture_source_preflight_accepted_frame_count"] == 1
    assert report["texture_source_preflight_rejected_frame_count"] == 1
    assert report["texture_source_preflight_top_score"] > 1.0
    assert report["added_texture_frames"][0]["source_frame_id"] == "face_good"
    assert report["added_texture_frames"][0]["preflight_rank"] == 1
    assert report["added_texture_frames"][0]["quality_gate"]["accepted"] is True
    assert report["rejected_low_detail_texture_frames"][0]["source_frame_id"] == "body_bad"
    assert report["rejected_low_detail_texture_frames"][0]["preflight_rank"] == 2
    assert (
        report["rejected_low_detail_texture_frames"][0]["quality_gate"]["reason"]
        == "severe_quality_reject"
    )
    preflight = json.loads(
        (project / "outputs" / "quality" / "texture_source_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert [frame["source_frame_id"] for frame in preflight["ranked_frames"]] == [
        "face_good",
        "body_bad",
    ]
    assert preflight["ranked_frames"][0]["accepted_for_bake"] is True
    assert preflight["ranked_frames"][1]["accepted_for_bake"] is False
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert texture_manifest["texture_frame_count"] == 2
    assert [
        frame.get("texture_only_source_frame_id")
        for frame in texture_manifest["frames"]
        if frame.get("texture_only_supplemental")
    ] == ["face_good"]
    assert (
        texture_manifest["texture_only_bake"]["rejected_low_detail_texture_frame_count"]
        == 1
    )
    assert texture_manifest["texture_only_bake"]["texture_source_preflight_top_score"] > 1.0


def test_write_baseline_geometry_texture_manifest_rejects_misaligned_face_anchors(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_good", b"good-face-pixels", media_type="photo")
    _write_texture_observation(source, "face_bad", b"bad-face-pixels", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_good", "face_bad"])
    _write_face_anchor_manifest(
        source,
        "face_good",
        {
            "left_eye_center": (34.0, 32.0, 0.86),
            "right_eye_center": (66.0, 33.0, 0.84),
            "nose_tip": (50.0, 55.0, 0.72),
            "muzzle_center": (50.0, 72.0, 0.64),
        },
    )
    _write_face_anchor_manifest(
        source,
        "face_bad",
        {
            "left_eye_center": (70.0, 32.0, 0.82),
            "right_eye_center": (30.0, 33.0, 0.81),
            "nose_tip": (50.0, 18.0, 0.72),
            "muzzle_center": (50.0, 22.0, 0.66),
        },
    )
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.82,
                "frame_id": "face_good",
                "reject_reasons": [],
                "texture_detail_candidate": True,
            },
            {
                "face_texture_detail_score": 0.81,
                "frame_id": "face_bad",
                "reject_reasons": [],
                "texture_detail_candidate": True,
            },
        ],
    )
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        force=True,
    )

    assert report["added_unique_texture_frame_count"] == 1
    assert report["added_texture_frames"][0]["source_frame_id"] == "face_good"
    assert report["rejected_low_detail_texture_frame_count"] == 1
    assert report["rejected_low_detail_texture_frames"][0]["source_frame_id"] == "face_bad"
    rejected_gate = report["rejected_low_detail_texture_frames"][0]["quality_gate"]
    assert rejected_gate["reason"] == "face_alignment_reject"
    assert rejected_gate["face_alignment"]["reason"] == "misaligned_face_anchors"
    assert "face_alignment_reject" in rejected_gate["severe_reject_reasons"]
    preflight = json.loads(
        (project / "outputs" / "quality" / "texture_source_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert preflight["accepted_frame_count"] == 1
    assert preflight["rejected_frame_count"] == 1
    assert preflight["ranked_frames"][1]["source_frame_id"] == "face_bad"
    assert preflight["ranked_frames"][1]["accepted_for_bake"] is False


def test_write_baseline_geometry_texture_manifest_probes_low_detail_source(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_probe", b"probe-face-pixels", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_probe"])
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.44,
                "frame_id": "face_probe",
                "reject_reasons": [],
                "texture_detail_candidate": False,
            }
        ],
    )
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        probe_texture_source_frame_ids=["face_probe"],
        force=True,
    )

    assert report["added_unique_texture_frame_count"] == 1
    assert report["rejected_low_detail_texture_frame_count"] == 0
    assert report["probe_texture_source_frame_count"] == 1
    assert report["probe_added_texture_frame_count"] == 1
    assert report["probe_added_texture_frames"][0]["source_frame_id"] == "face_probe"
    assert report["probe_added_texture_frames"][0]["probe"] is True
    assert report["probe_added_texture_frames"][0]["probe_forced_low_detail"] is True
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    supplemental = [
        frame
        for frame in texture_manifest["frames"]
        if frame.get("texture_only_supplemental")
    ][0]
    assert supplemental["texture_only_source_probe"] is True
    assert supplemental["texture_only_source_probe_forced_low_detail"] is True
    protection = supplemental["texture_only_face_region_projection_protection"]
    assert protection["broad_projection_excluded"] is True
    assert protection["source_patch_allowed"] is False
    assert supplemental["texture_only_source_patch_allowed"] is False
    assert texture_manifest["texture_only_bake"]["probe_added_texture_frame_count"] == 1


def test_write_baseline_geometry_texture_manifest_probe_skips_non_requested_sources(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_probe", b"probe-face-pixels", media_type="photo")
    _write_texture_observation(source, "face_other", b"other-face-pixels", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_probe", "face_other"])
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.82,
                "frame_id": "face_probe",
                "reject_reasons": [],
                "texture_detail_candidate": True,
            },
            {
                "face_texture_detail_score": 0.80,
                "frame_id": "face_other",
                "reject_reasons": [],
                "texture_detail_candidate": True,
            },
        ],
    )
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        probe_texture_source_frame_ids=["face_probe"],
        force=True,
    )

    assert report["added_unique_texture_frame_count"] == 1
    assert report["probe_added_texture_frame_count"] == 1
    assert report["probe_skipped_non_probe_texture_frame_count"] == 1
    assert report["probe_skipped_non_probe_texture_frames"][0]["source_frame_id"] == "face_other"
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    supplemental_source_ids = [
        frame["texture_only_source_frame_id"]
        for frame in texture_manifest["frames"]
        if frame.get("texture_only_supplemental")
    ]
    assert supplemental_source_ids == ["face_probe"]
    assert (
        texture_manifest["texture_only_bake"][
            "probe_skipped_non_probe_texture_frame_count"
        ]
        == 1
    )


def test_write_baseline_geometry_texture_manifest_tags_probe_sample_key_policy(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_probe", b"probe-face-pixels", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_probe"])
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.86,
                "frame_id": "face_probe",
                "reject_reasons": [],
                "texture_detail_candidate": True,
            },
        ],
    )
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        probe_texture_source_frame_ids=["face_probe"],
        probe_texture_source_sample_keys={
            "face_probe": ["left_iris", "right_pupil", "left_iris", "not_a_key"],
        },
        force=True,
    )

    assert report["probe_texture_source_sample_key_policy_frame_count"] == 1
    assert report["probe_texture_source_sample_key_policy"] == {
        "face_probe": ["left_iris", "right_pupil"],
    }
    assert report["probe_added_texture_frames"][0]["source_patch_allowed_sample_keys"] == [
        "left_iris",
        "right_pupil",
    ]
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    supplemental = [
        frame
        for frame in texture_manifest["frames"]
        if frame.get("texture_only_supplemental")
    ][0]
    assert supplemental["texture_only_source_patch_allowed_sample_keys"] == [
        "left_iris",
        "right_pupil",
    ]
    assert texture_manifest["texture_only_bake"][
        "probe_texture_source_sample_key_policy"
    ] == {"face_probe": ["left_iris", "right_pupil"]}


def test_supplemental_face_projection_policy_masks_protected_uv_regions() -> None:
    problem = SimpleNamespace(template=create_placeholder_cat_template())
    metadata = {
        "frame_id": "texture_source_face_probe",
        "texture_only_supplemental": True,
        "texture_only_face_region_projection_protection": {
            "applied": True,
            "broad_projection_excluded": True,
            "source_patch_allowed": False,
            "protected_region_names": [
                "left_eye",
                "right_eye",
                "nose",
                "muzzle",
                "left_ear_inner",
                "right_ear_inner",
            ],
        },
    }

    assert _exclude_protected_face_projection(metadata) is True
    assert _source_patch_candidate_allowed(metadata) is False
    protected = _protected_face_region_mask(
        problem,
        protected_region_names=tuple(
            metadata["texture_only_face_region_projection_protection"][
                "protected_region_names"
            ]
        ),
    )

    assert int(np.count_nonzero(protected)) > 0
    assert int(np.count_nonzero(protected)) < CAT_TEMPLATE_FACE_COUNT


def test_write_baseline_geometry_texture_manifest_does_not_probe_misaligned_face_anchors(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_probe", b"probe-face-pixels", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_probe"])
    _write_face_anchor_manifest(
        source,
        "face_probe",
        {
            "left_eye_center": (72.0, 32.0, 0.78),
            "right_eye_center": (28.0, 33.0, 0.78),
            "nose_tip": (50.0, 20.0, 0.66),
            "muzzle_center": (50.0, 21.0, 0.62),
        },
    )
    _write_frame_selection_report(
        source,
        [
            {
                "face_texture_detail_score": 0.44,
                "frame_id": "face_probe",
                "reject_reasons": [],
                "texture_detail_candidate": False,
            }
        ],
    )
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        probe_texture_source_frame_ids=["face_probe"],
        force=True,
    )

    assert report["added_unique_texture_frame_count"] == 0
    assert report["rejected_low_detail_texture_frame_count"] == 1
    assert report["probe_texture_source_frame_count"] == 1
    assert report["probe_added_texture_frame_count"] == 0
    rejected_gate = report["rejected_low_detail_texture_frames"][0]["quality_gate"]
    assert rejected_gate["reason"] == "face_alignment_reject"
    assert "face_alignment_reject" in rejected_gate["severe_reject_reasons"]
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert texture_manifest["texture_frame_count"] == 1
    assert texture_manifest["texture_only_bake"]["probe_added_texture_frame_count"] == 0


def test_write_baseline_geometry_texture_manifest_scopes_ablation_to_face_patches(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    rejected_payload = b"regressed-face-pixels"
    accepted_payload = b"better-face-pixels"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_bad", rejected_payload, media_type="photo")
    _write_texture_observation(source, "face_good", accepted_payload, media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_bad", "face_good"])
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )
    rejected_hash = hashlib.sha256(rejected_payload).hexdigest()

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        rejected_texture_source_hashes=[rejected_hash],
        force=True,
    )

    assert report["added_unique_texture_frame_count"] == 2
    assert report["rejected_ablation_texture_frame_count"] == 0
    assert report["rejected_ablation_texture_frames"] == []
    assert report["source_patch_blocked_ablation_texture_frame_count"] == 1
    blocked = report["source_patch_blocked_ablation_texture_frames"][0]
    assert blocked["source_frame_id"] == "face_bad"
    assert blocked["matched_rejected_sha256"] is True
    assert blocked["reason"] == (
        "face_source_patch_blocked_by_prior_regression"
    )
    added_by_source = {
        frame["source_frame_id"]: frame for frame in report["added_texture_frames"]
    }
    assert set(added_by_source) == {"face_bad", "face_good"}
    assert added_by_source["face_bad"]["source_patch_blocked_by_ablation"] is True
    assert added_by_source["face_bad"]["face_region_projection_protection"][
        "source_patch_allowed"
    ] is False
    assert added_by_source["face_good"]["face_region_projection_protection"][
        "source_patch_allowed"
    ] is True
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert texture_manifest["texture_frame_count"] == 3
    assert (
        texture_manifest["texture_only_bake"]["rejected_ablation_texture_frame_count"]
        == 0
    )
    assert (
        texture_manifest["texture_only_bake"][
            "source_patch_blocked_ablation_texture_frame_count"
        ]
        == 1
    )
    assert texture_manifest["texture_only_bake"]["rejected_texture_source_hashes"] == [
        rejected_hash
    ]


def test_write_baseline_geometry_texture_manifest_probes_source_id_rejection_with_new_hash(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_probe", b"new-probe-face-pixels", media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_probe"])
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        rejected_texture_source_hashes=["old-rejected-hash"],
        rejected_texture_source_frame_ids=["face_probe"],
        probe_texture_source_frame_ids=["face_probe"],
        force=True,
    )

    assert report["added_unique_texture_frame_count"] == 1
    assert report["rejected_ablation_texture_frame_count"] == 0
    assert report["probe_added_texture_frame_count"] == 1
    assert report["probe_bypassed_rejected_source_frame_id_count"] == 1
    assert (
        report["probe_added_texture_frames"][0][
            "probe_bypassed_rejected_source_frame_id"
        ]
        is True
    )
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    supplemental = [
        frame
        for frame in texture_manifest["frames"]
        if frame.get("texture_only_supplemental")
    ][0]
    assert (
        supplemental["texture_only_source_probe_bypassed_rejected_source_frame_id"]
        is True
    )


def test_write_baseline_geometry_texture_manifest_probe_override_unlocks_exact_hash(
    tmp_path,
) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "source"
    project = tmp_path / "project"
    payload = b"previously-regressed-but-explicitly-probed-face-pixels"
    _write_texture_observation(baseline, "body_a", b"baseline-pixels", media_type="video")
    _write_texture_observation(source, "face_probe", payload, media_type="photo")
    _write_texture_manifest(baseline, ["body_a"])
    _write_texture_manifest(source, ["face_probe"])
    (project / "observations" / "raw").mkdir(parents=True)
    (project / "outputs" / "quality").mkdir(parents=True)
    (project / "observations" / "raw" / "frame_selection.json").write_text(
        json.dumps(
            {
                "project_id": "project",
                "selected_frame_ids": ["body_a"],
                "texture_frame_ids": ["body_a"],
                "frames": [{"frame_id": "body_a", "texture_enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    (project / "observations" / "raw" / "manifest.json").write_text(
        json.dumps({"project_id": "project", "frames": [{"frame_id": "body_a"}]}),
        encoding="utf-8",
    )
    digest = hashlib.sha256(payload).hexdigest()

    report = _write_baseline_geometry_texture_manifest(
        project,
        baseline_project_dir=baseline,
        texture_source_project_dir=source,
        rejected_texture_source_hashes=[digest],
        rejected_texture_source_frame_ids=["face_probe"],
        probe_texture_source_frame_ids=["face_probe"],
        probe_texture_source_ablation_overrides={"face_probe": [digest]},
        force=True,
    )

    assert report["added_unique_texture_frame_count"] == 1
    assert report["source_patch_blocked_ablation_texture_frame_count"] == 0
    assert report["probe_bypassed_rejected_sha256_count"] == 1
    assert report["probe_bypassed_rejected_source_frame_id_count"] == 1
    assert report["face_region_source_patch_allowed_frame_count"] == 1
    probe = report["probe_added_texture_frames"][0]
    assert probe["probe_ablation_override_hash_match"] is True
    assert probe["source_patch_blocked_by_ablation"] is False
    assert probe["face_region_projection_protection"]["source_patch_allowed"] is True
    texture_manifest = json.loads(
        (project / "observations" / "raw" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert texture_manifest["texture_only_bake"][
        "probe_texture_source_ablation_overrides"
    ] == {"face_probe": [digest]}
    supplemental = [
        frame
        for frame in texture_manifest["frames"]
        if frame.get("texture_only_supplemental")
    ][0]
    assert supplemental["texture_only_source_probe_ablation_override_hash_match"] is True
    assert supplemental["texture_only_source_patch_blocked_by_ablation"] is False
    assert supplemental["texture_only_source_patch_allowed"] is True


def test_run_synthetic_pipeline_writes_acceptance_summary(tmp_path) -> None:
    project_dir = tmp_path / "demo"

    result = run_synthetic_pipeline(project_dir, num_views=3, image_size=64, seed=999)

    assert result.success
    assert result.summary_path.exists()
    assert result.evaluation_report_path.exists()
    assert result.acceptance_audit_path.exists()
    assert result.compliance_audit_path.exists()
    assert result.fit_report_path.exists()
    assert result.obj_path.exists()
    assert result.ply_path.exists()
    assert result.glb_path.exists()
    assert result.skeleton_path is not None
    assert result.skeleton_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["success"] is True
    assert summary["steps"] == {
        "evaluate": True,
        "export": True,
        "fit": True,
        "preprocess": True,
        "acceptance": True,
        "synthetic_create": True,
    }
    assert summary["metrics"]["segmentation_mean_iou_with_previous_masks"] == 1.0
    assert summary["metrics"]["keypoint_mean_confidence"] == 1.0
    assert summary["metrics"]["keypoint_mean_in_bbox_fraction"] >= 0.9
    assert summary["metrics"]["fit_init_prediction_used"] == 1.0
    assert summary["metrics"]["overlay_mean_silhouette_iou"] >= 0.0
    assert summary["metrics"]["observed_face_fraction"] > 0.0
    assert summary["metrics"]["observed_vertex_fraction"] > 0.0
    assert summary["metrics"]["generated_face_count"] == 0
    assert summary["metrics"]["basecolor_observed_texel_fraction"] > 0.0
    assert summary["metrics"]["basecolor_generated_texel_fraction"] == 0.0
    assert summary["metrics"]["pbr_observed_texel_fraction"] > 0.0
    assert summary["metrics"]["pbr_generated_texel_fraction"] == 0.0
    assert summary["metrics"]["joint_projection_mean_keypoint_loss"] <= 0.01
    assert summary["metrics"]["joint_projection_mean_pixel_error"] >= 0.0
    assert summary["metrics"]["silhouette_residual_mean_union_error"] <= 0.35
    assert summary["metrics"]["silhouette_residual_mean_false_negative"] >= 0.0
    assert summary["metrics"]["silhouette_residual_mean_false_positive"] >= 0.0
    assert summary["metrics"]["heldout_mean_silhouette_iou"] >= 0.80
    assert summary["metrics"]["heldout_mean_keypoint_reprojection_loss"] <= 0.01
    assert summary["metrics"]["final_loss"] < summary["metrics"]["initial_loss"]
    assert summary["metrics"]["final_keypoint_loss"] < summary["metrics"]["initial_keypoint_loss"]
    assert (
        summary["metrics"]["final_silhouette_loss"]
        < summary["metrics"]["initial_silhouette_loss"]
    )
    assert summary["metrics"]["final_beta_error"] <= summary["metrics"]["initial_beta_error"]
    assert summary["metrics"]["beta_error_reduction_fraction"] > 0.0
    assert summary["metrics"]["vertex_count"] == CAT_TEMPLATE_VERTEX_COUNT
    assert summary["metrics"]["face_count"] == CAT_TEMPLATE_FACE_COUNT
    assert (
        summary["metrics"]["acceptance_passed_check_count"]
        == summary["metrics"]["acceptance_total_check_count"]
    )
    assert summary["artifacts"]["init_manifest"].endswith("init/synthetic/manifest.json")
    assert summary["artifacts"]["fit_state"].endswith(
        "outputs/fitting/fit_state.safetensors"
    )
    assert summary["artifacts"]["fit_state"] == summary["artifacts"]["fit_state_safetensors"]
    assert summary["artifacts"]["fit_state_source_format"] == "safetensors"
    assert summary["artifacts"]["fit_state_safetensors"].endswith(
        "outputs/fitting/fit_state.safetensors"
    )
    assert not (project_dir / "outputs" / "fitting" / "fit_state.npz").exists()
    assert summary["artifacts"]["overlay_manifest"].endswith(
        "outputs/fitting/per_frame_overlay/manifest.json"
    )
    assert summary["artifacts"]["validation_report"].endswith("outputs/quality/validation.json")
    assert summary["artifacts"]["coverage_report"].endswith("outputs/quality/coverage.json")
    assert summary["artifacts"]["texture_manifest"].endswith(
        "outputs/textures/texture_manifest.json"
    )
    texture_manifest = json.loads(
        (project_dir / "outputs" / "textures" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert texture_manifest["facial_features"]["applied"] is False
    assert summary["artifacts"]["basecolor"].endswith("outputs/textures/basecolor.png")
    assert summary["artifacts"]["basecolor_confidence"].endswith(
        "outputs/textures/basecolor_confidence.png"
    )
    assert summary["artifacts"]["source_view_id"].endswith("outputs/textures/source_view_id.png")
    assert summary["artifacts"]["normal"].endswith("outputs/textures/normal.png")
    assert summary["artifacts"]["roughness"].endswith("outputs/textures/roughness.png")
    assert summary["artifacts"]["ao"].endswith("outputs/textures/ao.png")
    assert summary["artifacts"]["joint_projection"].endswith(
        "outputs/quality/synthetic_joint_projection.json"
    )
    assert summary["artifacts"]["silhouette_residuals"].endswith(
        "outputs/quality/synthetic_silhouette_residuals.json"
    )
    assert summary["artifacts"]["acceptance_audit"].endswith(
        "synthetic_acceptance_audit.json"
    )
    assert summary["artifacts"]["compliance_audit"].endswith(
        "opencat3d_compliance_audit.json"
    )
    assert summary["artifacts"]["idle_animation"].endswith(
        "outputs/avatar/animations/idle.glb"
    )
    assert summary["artifacts"]["skeleton"].endswith("outputs/avatar/skeleton.json")


def test_run_cli_executes_synthetic_pipeline(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "demo"

    result = runner.invoke(
        app,
        [
            "run",
            str(project_dir),
            "--mode",
            "synthetic",
            "--num-views",
            "3",
            "--image-size",
            "64",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "success=True" in result.output
    assert (project_dir / "outputs" / "quality" / "synthetic_pipeline_summary.json").exists()
    assert (project_dir / "outputs" / "quality" / "synthetic_acceptance_audit.json").exists()
    assert (project_dir / "outputs" / "quality" / "opencat3d_compliance_audit.json").exists()
    assert (project_dir / "outputs" / "quality" / "validation.json").exists()
    assert (project_dir / "outputs" / "quality" / "synthetic_joint_projection.json").exists()
    assert (project_dir / "outputs" / "quality" / "synthetic_silhouette_residuals.json").exists()
    assert (project_dir / "outputs" / "fitting" / "fit_state.safetensors").exists()
    assert (project_dir / "outputs" / "avatar" / "cat_avatar.glb").exists()
    assert (project_dir / "outputs" / "avatar" / "skeleton.json").exists()
    assert (project_dir / "outputs" / "textures" / "basecolor.png").exists()
    assert (project_dir / "outputs" / "textures" / "normal.png").exists()
    assert (project_dir / "outputs" / "textures" / "roughness.png").exists()
    assert (project_dir / "outputs" / "textures" / "ao.png").exists()


def _write_texture_observation(
    project_dir: Path,
    frame_id: str,
    payload: bytes,
    *,
    media_type: str,
) -> None:
    frame_dir = project_dir / "observations" / "raw" / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    image_path = frame_dir / "image_srgb.npz"
    image_path.write_bytes(payload)
    (frame_dir / "observation.json").write_text(
        json.dumps(
            {
                "frame_id": frame_id,
                "image_srgb_path": str(image_path),
                "media_type": media_type,
            }
        ),
        encoding="utf-8",
    )


def _write_texture_manifest(project_dir: Path, frame_ids: list[str]) -> None:
    frames = []
    for index, frame_id in enumerate(frame_ids):
        frame_dir = project_dir / "observations" / "raw" / frame_id
        frames.append(
            {
                "frame_id": frame_id,
                "media_type": "photo" if index else "video",
                "observation_path": str(frame_dir / "observation.json"),
                "image_srgb_path": str(frame_dir / "image_srgb.npz"),
            }
        )
    path = project_dir / "observations" / "raw" / "texture_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "project_id": project_dir.name,
                "texture_frame_count": len(frames),
                "texture_frame_ids": list(frame_ids),
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def _write_face_anchor_manifest(
    project_dir: Path,
    frame_id: str,
    targets: dict[str, tuple[float, float, float]],
) -> None:
    frame_dir = project_dir / "observations" / "raw" / frame_id
    anchor_path = frame_dir / "face_anchor_targets.json"
    target_objects = {
        name: {
            "name": name,
            "x": x,
            "y": y,
            "confidence": confidence,
            "pixel_count": 16,
        }
        for name, (x, y, confidence) in targets.items()
    }
    anchor_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.raw_observed_face_anchor_targets_v1",
                "frame_id": frame_id,
                "targets": target_objects,
                "target_count": len(target_objects),
                "mean_confidence": sum(item[2] for item in targets.values())
                / max(len(targets), 1),
            }
        ),
        encoding="utf-8",
    )
    manifest_path = project_dir / "observations" / "raw" / "texture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for frame in manifest["frames"]:
        if frame["frame_id"] == frame_id:
            frame["bbox_xyxy"] = [0.0, 0.0, 100.0, 100.0]
            frame["image_width"] = 100
            frame["image_height"] = 100
            frame["feature_anchors_xyc_path"] = str(anchor_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_frame_selection_report(
    project_dir: Path,
    frames: list[dict[str, object]],
) -> None:
    path = project_dir / "observations" / "raw" / "frame_selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "project_id": project_dir.name,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def test_run_raw_pipeline_writes_quality_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project_dir = Path("project")
    input_dir = Path("inputs")
    _write_raw_inputs(input_dir)

    result = run_raw_pipeline(
        project_dir,
        input_paths=[input_dir],
        color_delta_threshold=0.10,
        camera_root_steps=2,
    )

    assert result.success
    assert result.project_dir == (tmp_path / "project").resolve()
    assert result.summary_path.is_absolute()
    assert result.summary_path.exists()
    assert result.evaluation_report_path.exists()
    assert result.compliance_audit_path.exists()
    assert result.fit_report_path.exists()
    assert result.obj_path.exists()
    assert result.ply_path.exists()
    assert result.glb_path.exists()
    assert result.skeleton_path is not None
    assert result.skeleton_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["success"] is True
    assert summary["steps"] == {
        "evaluate": True,
        "export": True,
        "fit": True,
        "raw_frame_selection": True,
        "raw_ingest": True,
        "raw_keypoint_bootstrap": True,
        "raw_segmentation": True,
    }
    assert summary["metrics"]["raw_input_frame_count"] == 2
    assert summary["metrics"]["raw_frame_count"] == 2
    assert summary["metrics"]["raw_selected_frame_count"] == 2
    assert summary["metrics"]["raw_rejected_frame_count"] == 0
    assert summary["metrics"]["raw_texture_frame_count"] == 2
    assert summary["metrics"]["raw_texture_photo_frame_count"] == 2
    assert summary["metrics"]["raw_texture_video_frame_count"] == 0
    assert summary["metrics"]["raw_geometry_selected_body_priority_frame_count"] >= 0.0
    assert summary["metrics"]["raw_texture_only_detail_frame_count"] >= 0.0
    assert summary["metrics"]["segmentation_mean_mask_area_ratio"] > 0.0
    assert summary["metrics"]["segmentation_mean_bbox_area_ratio"] > 0.0
    assert summary["metrics"]["segmentation_full_frame_bbox_fraction"] == 0.0
    assert 0.0 < summary["metrics"]["keypoint_mean_confidence"] <= 0.5
    assert summary["metrics"]["keypoint_mean_in_bbox_fraction"] == 1.0
    assert 0.0 < summary["metrics"]["raw_keypoint_mean_frontal_face_confidence"] <= 1.0
    assert 0.35 <= summary["metrics"]["raw_keypoint_mean_view_weight"] <= 1.0
    assert summary["metrics"]["fit_init_prediction_used"] == 1.0
    assert summary["metrics"]["raw_fit_feature_anchor_weight"] == 0.0
    assert summary["metrics"]["raw_fit_initial_feature_anchor_loss"] == 0.0
    assert summary["metrics"]["raw_fit_final_feature_anchor_loss"] == 0.0
    assert summary["metrics"]["raw_fit_feature_anchor_loss_reduction"] == 0.0
    assert summary["metrics"]["raw_fit_mean_feature_anchor_target_count"] > 0.0
    assert summary["metrics"]["overlay_mean_silhouette_iou"] >= RAW_METRIC_THRESHOLDS[
        "min_overlay_mean_silhouette_iou"
    ]
    assert summary["metrics"]["raw_mean_mask_area_ratio"] <= RAW_METRIC_THRESHOLDS[
        "max_mean_mask_area_ratio"
    ]
    assert summary["metrics"]["raw_mean_bbox_area_ratio"] <= RAW_METRIC_THRESHOLDS[
        "max_mean_bbox_area_ratio"
    ]
    assert summary["metrics"]["raw_full_frame_bbox_fraction"] == 0.0
    assert summary["metrics"]["raw_border_touch_bbox_fraction"] == 0.0
    assert summary["metrics"]["raw_texture_unknown_luma_fraction_of_foreground"] >= (
        RAW_METRIC_THRESHOLDS["min_unknown_texture_luma_fraction_of_foreground"]
    )
    assert summary["metrics"]["raw_texture_width"] == RAW_METRIC_THRESHOLDS[
        "min_template_region_v2_texture_target_baked_size"
    ]
    assert summary["metrics"]["raw_texture_height"] == RAW_METRIC_THRESHOLDS[
        "min_template_region_v2_texture_target_baked_size"
    ]
    assert summary["metrics"]["raw_texture_resolution_scale"] == 8.0
    assert summary["metrics"]["raw_face_texture_target_bake_used"] == 0.0
    assert summary["metrics"]["raw_face_texture_target_reported_size"] == (
        summary["metrics"]["raw_texture_width"]
    )
    assert summary["metrics"]["raw_template_region_v2_texture_target_bake_used"] == 1.0
    assert summary["metrics"]["raw_template_region_v2_texture_target_reported_size"] == (
        summary["metrics"]["raw_texture_width"]
    )
    assert summary[
        "metrics"
    ]["raw_template_region_v2_texture_target_reported_allocated_texel_count"] >= (
        summary[
            "metrics"
        ]["raw_template_region_v2_texture_target_reported_requested_texel_count"]
    )
    assert summary["metrics"]["raw_pbr_texture_width"] == summary["metrics"][
        "raw_texture_width"
    ]
    assert summary["metrics"]["raw_pbr_texture_height"] == summary["metrics"][
        "raw_texture_height"
    ]
    assert summary["metrics"]["raw_glb_texture_size"] == summary["metrics"][
        "raw_texture_width"
    ]
    assert summary["metrics"]["raw_glb_texture_dimensions_valid"] == 1.0
    assert summary["metrics"]["raw_glb_pbr_texture_dimensions_valid"] == 1.0
    assert summary["metrics"]["raw_glb_basecolor_texture_valid"] == 1.0
    assert summary["metrics"]["raw_glb_pbr_textures_valid"] == 1.0
    assert summary["metrics"]["raw_glb_embedded_texture_dimension_gate_valid"] == 1.0
    assert summary["metrics"]["raw_glb_face_feature_required_group_count"] == (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_visible_group_count"]
    )
    assert summary["metrics"]["raw_glb_face_feature_visible_group_count"] == (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_visible_group_count"]
    )
    assert summary["metrics"]["raw_glb_face_feature_required_feature_count"] == (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_visible_feature_count"]
    )
    assert summary["metrics"]["raw_glb_face_feature_present_required_feature_count"] == (
        summary["metrics"]["raw_glb_face_feature_required_feature_count"]
    )
    assert summary["metrics"]["raw_glb_face_feature_visible_feature_count"] == (
        summary["metrics"]["raw_glb_face_feature_required_feature_count"]
    )
    assert summary["metrics"]["raw_glb_face_feature_sampled_texel_count"] > 0
    assert summary["metrics"]["raw_glb_face_feature_min_visible_texel_count"] > 0
    assert summary["metrics"]["raw_glb_face_feature_mean_rgb_error"] <= (
        RAW_METRIC_THRESHOLDS["max_glb_face_feature_mean_rgb_error"]
    )
    assert summary["metrics"]["raw_glb_face_feature_max_rgb_error"] <= (
        RAW_METRIC_THRESHOLDS["max_glb_face_feature_max_rgb_error"]
    )
    assert summary["metrics"]["raw_glb_face_feature_eye_visible"] == 1.0
    assert summary["metrics"]["raw_glb_face_feature_nose_visible"] == 1.0
    assert summary["metrics"]["raw_glb_face_feature_muzzle_visible"] == 1.0
    assert summary["metrics"]["raw_glb_face_feature_iris_to_pupil_rgb_distance"] >= (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_rgb_distance"]
    )
    assert summary["metrics"]["raw_glb_face_feature_eye_detail_luma_std"] > 0.0
    assert summary["metrics"]["raw_glb_face_feature_nose_to_muzzle_rgb_distance"] >= (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_rgb_distance"]
    )
    assert summary["metrics"]["raw_glb_face_feature_mouth_to_muzzle_rgb_distance"] >= (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_rgb_distance"]
    )
    assert summary["metrics"]["raw_glb_face_feature_material_visibility_gate_valid"] == 1.0
    assert summary["metrics"]["raw_face_priority_texture_used"] == 1.0
    assert (
        summary["metrics"][
            "raw_facial_feature_eye_source_detail_reinforcement_feature_count"
        ]
        > 0
    )
    assert (
        summary["metrics"][
            "raw_facial_feature_eye_source_detail_reinforcement_texel_count"
        ]
        > 0
    )
    assert (
        0.0
        < summary["metrics"][
            "raw_facial_feature_eye_source_detail_reinforcement_texel_fraction"
        ]
        <= 1.0
    )
    assert summary["metrics"]["raw_face_uv_region_count"] == 8
    assert summary["metrics"]["raw_face_uv_region_total_area"] > 0.0
    assert summary["metrics"]["raw_face_uv_region_min_density_weight"] > 0.0
    assert summary["metrics"]["raw_face_uv_region_contract_valid"] == 1.0
    assert summary["metrics"]["raw_template_region_v2_count"] == (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_count"]
    )
    assert summary["metrics"]["raw_template_region_v2_required_region_count"] == (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_count"]
    )
    assert summary["metrics"]["raw_template_region_v2_required_present_count"] == (
        summary["metrics"]["raw_template_region_v2_required_region_count"]
    )
    assert summary["metrics"]["raw_template_region_v2_contract_valid"] == 1.0
    assert summary["metrics"]["raw_glb_template_region_v2_contract_valid"] == 1.0
    assert summary["metrics"]["raw_template_region_v2_belly_vertex_count"] > 0
    assert summary["metrics"]["raw_face_uv_density_allocated_texel_count"] > 0
    assert summary["metrics"]["raw_face_uv_density_min_allocated_texels"] > 0
    assert summary["metrics"]["raw_face_uv_density_eye_min_allocated_texels"] >= (
        RAW_METRIC_THRESHOLDS["min_face_uv_eye_texels"]
    )
    assert summary["metrics"]["raw_face_uv_density_nose_allocated_texels"] >= (
        RAW_METRIC_THRESHOLDS["min_face_uv_nose_texels"]
    )
    assert summary["metrics"]["raw_face_uv_density_ear_inner_min_allocated_texels"] >= (
        RAW_METRIC_THRESHOLDS["min_face_uv_ear_inner_texels"]
    )
    assert summary["metrics"]["raw_template_region_v2_uv_region_count"] == (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_uv_region_count"]
    )
    assert summary["metrics"]["raw_template_region_v2_uv_required_present_count"] == (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_uv_region_count"]
    )
    assert summary["metrics"]["raw_template_region_v2_uv_island_count"] >= 13
    assert summary["metrics"]["raw_template_region_v2_region_scoped_uv_island_count"] > (
        summary["metrics"]["raw_template_region_v2_uv_island_count"]
    )
    assert (
        summary["metrics"]["raw_template_region_v2_uv_island_density_weighted_texel_count"]
        > summary["metrics"]["raw_template_region_v2_uv_island_allocated_texel_count"]
    )
    assert summary["metrics"]["raw_template_region_v2_face_uv_island_count"] == 8
    assert summary["metrics"]["raw_template_region_v2_facial_uv_island_count"] == 8
    assert (
        summary["metrics"]["raw_template_region_v2_facial_uv_island_allocated_texel_count"]
        > 0
    )
    assert summary["metrics"]["raw_template_region_v2_body_region_uv_island_count"] == 5
    assert summary["metrics"]["raw_template_region_v2_eye_min_allocated_texels"] == (
        summary["metrics"]["raw_face_uv_density_eye_min_allocated_texels"]
    )
    assert summary["metrics"]["raw_template_region_v2_nose_allocated_texels"] == (
        summary["metrics"]["raw_face_uv_density_nose_allocated_texels"]
    )
    assert summary["metrics"]["raw_template_region_v2_muzzle_allocated_texels"] > 0
    assert summary["metrics"]["raw_template_region_v2_ear_inner_min_allocated_texels"] == (
        summary["metrics"]["raw_face_uv_density_ear_inner_min_allocated_texels"]
    )
    assert summary["metrics"]["raw_template_region_v2_facial_density_weighted_texel_count"] > (
        summary["metrics"][
            "raw_template_region_v2_facial_uv_island_allocated_texel_count"
        ]
    )
    assert summary["metrics"]["raw_face_texture_target_size"] >= RAW_METRIC_THRESHOLDS[
        "min_face_texture_target_size"
    ]
    assert summary["metrics"]["raw_face_texture_target_scale"] >= RAW_METRIC_THRESHOLDS[
        "min_face_texture_target_scale"
    ]
    assert summary["metrics"]["raw_face_texture_target_allocated_texel_count"] >= (
        summary["metrics"]["raw_face_texture_target_requested_texel_count"]
    )
    assert summary["metrics"]["raw_face_texture_target_eye_min_texels"] >= (
        summary["metrics"]["raw_face_uv_density_eye_min_allocated_texels"]
    )
    assert summary["metrics"]["raw_face_texture_target_nose_texels"] >= (
        summary["metrics"]["raw_face_uv_density_nose_allocated_texels"]
    )
    assert summary["metrics"]["raw_face_texture_target_ear_inner_min_texels"] >= (
        summary["metrics"]["raw_face_uv_density_ear_inner_min_allocated_texels"]
    )
    assert summary["metrics"]["raw_template_region_v2_texture_target_size"] >= (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_texture_target_size"]
    )
    assert summary["metrics"]["raw_template_region_v2_texture_target_scale"] >= (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_texture_target_scale"]
    )
    assert summary["metrics"][
        "raw_template_region_v2_texture_target_facial_island_count"
    ] == RAW_METRIC_THRESHOLDS[
        "min_template_region_v2_texture_target_facial_island_count"
    ]
    assert summary["metrics"]["raw_template_region_v2_texture_target_island_count"] == (
        summary["metrics"]["raw_template_region_v2_facial_uv_island_count"]
    )
    assert summary["metrics"][
        "raw_template_region_v2_texture_target_allocated_texel_count"
    ] >= summary["metrics"][
        "raw_template_region_v2_texture_target_requested_texel_count"
    ]
    assert summary["metrics"][
        "raw_template_region_v2_texture_target_allocated_texel_count"
    ] >= summary["metrics"][
        "raw_template_region_v2_facial_uv_island_allocated_texel_count"
    ]
    assert summary["metrics"][
        "raw_template_region_v2_texture_target_eye_min_texels"
    ] >= summary["metrics"]["raw_template_region_v2_eye_min_allocated_texels"]
    assert summary["metrics"][
        "raw_template_region_v2_texture_target_nose_texels"
    ] >= summary["metrics"]["raw_template_region_v2_nose_allocated_texels"]
    assert summary["metrics"][
        "raw_template_region_v2_texture_target_muzzle_texels"
    ] >= summary["metrics"]["raw_template_region_v2_muzzle_allocated_texels"]
    assert summary["metrics"][
        "raw_template_region_v2_texture_target_ear_inner_min_texels"
    ] >= summary["metrics"]["raw_template_region_v2_ear_inner_min_allocated_texels"]
    assert summary["metrics"][
        "raw_template_region_v2_texture_target_max_density_scale"
    ] >= 1.0
    assert summary["metrics"]["raw_face_texture_crop_region_count"] == 6
    assert summary["metrics"]["raw_face_texture_crop_texture_width"] == summary["metrics"][
        "raw_texture_width"
    ]
    assert summary["metrics"]["raw_face_texture_crop_texture_height"] == summary["metrics"][
        "raw_texture_height"
    ]
    assert summary["metrics"]["raw_face_texture_crop_compared_region_count"] == 6
    assert summary["metrics"]["raw_face_texture_crop_exact_source_patch_region_count"] == 6
    assert summary["metrics"]["raw_face_texture_crop_exact_source_patch_fraction"] == 1.0
    assert summary["metrics"]["raw_face_texture_crop_detail_region_count"] == 6
    assert summary["metrics"]["raw_face_texture_crop_min_unique_rgb_count"] >= 2
    assert summary["metrics"]["raw_face_texture_crop_min_confident_texel_fraction"] > 0.0
    assert summary["metrics"]["raw_face_texture_crop_mean_confident_texel_fraction"] > 0.0
    assert summary["metrics"]["raw_face_texture_crop_mean_source_rgb_delta"] >= 0.0
    assert summary["metrics"]["raw_face_texture_crop_isolated_region_count"] >= (
        RAW_METRIC_THRESHOLDS["min_face_texture_isolated_crop_regions"]
    )
    assert summary["metrics"]["raw_face_texture_crop_overlap_pair_count"] <= (
        RAW_METRIC_THRESHOLDS["max_face_texture_crop_overlap_pair_count"]
    )
    assert summary["metrics"]["raw_face_texture_crop_overlap_fraction"] == 0.0
    assert summary["metrics"]["raw_face_texture_crop_visual_quality_region_count"] == 6
    assert summary["metrics"]["raw_face_texture_crop_visual_quality_passed_count"] == 6
    assert summary["metrics"]["raw_face_texture_crop_visual_quality_fraction"] == 1.0
    assert summary["metrics"]["raw_face_texture_crop_visual_quality_min_score"] >= (
        RAW_METRIC_THRESHOLDS["min_face_texture_visual_quality_score"]
    )
    assert summary["metrics"]["raw_face_texture_crop_visual_quality_mean_score"] >= summary[
        "metrics"
    ]["raw_face_texture_crop_visual_quality_min_score"]
    assert summary["metrics"]["raw_facial_feature_count"] >= 12
    assert summary["metrics"]["raw_facial_feature_painted_texel_count"] > 0
    assert summary["metrics"]["raw_facial_feature_source_patch_quality_passed_count"] == (
        summary["metrics"]["raw_facial_feature_source_patch_quality_total_count"]
    )
    assert summary["metrics"]["raw_facial_feature_source_patch_quality_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_quality_fraction"]
    )
    assert summary["metrics"]["raw_facial_feature_source_patch_min_quality_score"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_quality_score"]
    )
    assert summary["metrics"]["raw_facial_feature_source_patch_mean_quality_score"] >= (
        summary["metrics"]["raw_facial_feature_source_patch_min_quality_score"]
    )
    assert summary["metrics"]["raw_facial_feature_source_patch_fusion_min_weight"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_fusion_weight"]
    )
    assert summary["metrics"]["raw_facial_feature_source_patch_fusion_mean_weight"] >= (
        summary["metrics"]["raw_facial_feature_source_patch_fusion_min_weight"]
    )
    assert (
        summary["metrics"][
            "raw_facial_feature_source_patch_fusion_weighted_texel_count"
        ]
        > 0
    )
    assert summary["metrics"]["raw_facial_feature_source_patch_core_texel_count"] > 0
    assert summary["metrics"]["raw_facial_feature_source_patch_core_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_core_fraction"]
    )
    assert (
        summary["metrics"][
            "raw_facial_feature_source_patch_feathered_edge_texel_count"
        ]
        > 0
    )
    assert summary["metrics"]["raw_facial_feature_source_patch_feathered_edge_fraction"] >= (
        RAW_METRIC_THRESHOLDS[
            "min_facial_feature_source_patch_feathered_edge_fraction"
        ]
    )
    assert summary["metrics"]["raw_facial_feature_source_patch_mean_blend_alpha"] > 0.0
    assert summary["metrics"]["raw_face_feature_min_required_contrast"] >= (
        RAW_METRIC_THRESHOLDS["min_face_feature_required_contrast"]
    )
    assert summary["metrics"]["raw_face_feature_eye_to_muzzle_luma_delta"] > 0.0
    assert summary["metrics"]["raw_face_feature_mouth_to_muzzle_luma_delta"] > 0.0
    assert summary["metrics"]["raw_face_feature_nose_to_muzzle_rgb_distance"] > 0.0
    assert summary["metrics"]["raw_face_feature_iris_to_pupil_rgb_distance"] > 0.0
    assert summary["metrics"]["raw_face_feature_contrast_passed"] == 1.0
    assert summary["metrics"]["raw_face_detail_geometry_applied"] == 1.0
    assert summary["metrics"]["raw_face_detail_deformed_vertex_count"] > 0
    assert summary["metrics"]["raw_face_detail_max_displacement"] <= 0.040
    assert summary["metrics"]["raw_face_detail_mean_displacement"] > 0.0
    assert summary["metrics"]["raw_face_surface_anchor_count"] == 10
    assert summary["metrics"]["raw_face_surface_anchor_vertex_count"] > 0
    assert summary["metrics"]["raw_face_surface_anchor_max_displacement"] > 0.0
    assert summary["metrics"]["raw_face_turntable_view_count"] == 3
    assert summary["metrics"]["raw_face_turntable_front_visible_anchor_count"] >= 4
    assert summary["metrics"]["raw_face_turntable_front_anchor_pixel_count"] > 0
    assert summary["metrics"]["raw_face_turntable_required_anchor_visible_fraction"] == 1.0
    assert summary["metrics"]["raw_face_turntable_textured_view_count"] == 3
    assert summary["metrics"]["raw_face_turntable_textured_written_count"] == 3
    assert summary["metrics"]["raw_face_turntable_front_textured_unique_rgb_count"] >= 2
    assert summary["metrics"]["raw_face_turntable_front_textured_luma_std"] > 0.0
    assert (
        summary["metrics"]["raw_face_turntable_front_textured_mouth_line_pixel_count"]
        > 0
    )
    assert (
        summary["metrics"][
            "raw_face_turntable_front_textured_required_feature_fraction"
        ]
        > 0.0
    )
    assert summary["metrics"]["raw_face_turntable_mean_textured_unique_rgb_count"] >= 2
    assert summary["metrics"]["raw_face_turntable_mean_textured_luma_std"] > 0.0
    assert summary["metrics"]["raw_face_texture_closeup_width"] == 512
    assert summary["metrics"]["raw_face_texture_closeup_height"] == 512
    assert summary["metrics"]["raw_face_texture_closeup_feature_count"] >= 10
    assert (
        summary["metrics"]["raw_face_texture_closeup_required_present_count"]
        == summary["metrics"]["raw_face_texture_closeup_required_feature_count"]
    )
    assert summary["metrics"]["raw_face_texture_closeup_required_feature_fraction"] == 1.0
    assert summary["metrics"]["raw_face_texture_closeup_crop_texel_count"] > 0
    assert summary["metrics"]["raw_face_texture_closeup_unique_rgb_count"] >= 4
    assert summary["metrics"]["raw_face_texture_closeup_luma_std"] > 0.0
    assert (
        summary["metrics"][
            "raw_face_texture_source_compare_source_feature_available_count"
        ]
        >= 10
    )
    assert summary["metrics"]["raw_face_texture_source_compare_feature_count"] >= 10
    assert (
        summary["metrics"]["raw_face_texture_source_compare_required_present_count"]
        == summary["metrics"]["raw_face_texture_source_compare_required_feature_count"]
    )
    assert (
        summary["metrics"]["raw_face_texture_source_compare_required_feature_fraction"]
        == 1.0
    )
    assert summary["metrics"]["raw_face_texture_source_compare_mean_rgb_distance"] >= 0.0
    assert summary["metrics"]["raw_face_texture_source_compare_max_rgb_distance"] >= (
        summary["metrics"]["raw_face_texture_source_compare_mean_rgb_distance"]
    )
    assert summary["metrics"]["raw_face_texture_source_compare_mean_luma_delta"] >= 0.0
    assert summary["metrics"]["raw_face_texture_source_compare_max_luma_delta"] >= (
        summary["metrics"]["raw_face_texture_source_compare_mean_luma_delta"]
    )
    assert summary["metrics"]["raw_face_texture_source_compare_source_mean_luma_std"] >= 0.0
    assert summary["metrics"]["raw_face_texture_source_compare_baked_mean_luma_std"] >= 0.0
    iris_candidate_count = summary["metrics"][
        "raw_face_texture_source_compare_iris_candidate_texel_count"
    ]
    iris_accepted_count = summary["metrics"][
        "raw_face_texture_source_compare_iris_core_accepted_texel_count"
    ]
    iris_rejected_count = summary["metrics"][
        "raw_face_texture_source_compare_iris_annulus_rejected_texel_count"
    ]
    assert summary["metrics"]["raw_face_texture_source_compare_iris_feature_count"] == 2
    assert iris_candidate_count > 0
    assert iris_accepted_count > 0
    assert iris_candidate_count == iris_accepted_count + iris_rejected_count
    assert (
        summary["metrics"][
            "raw_face_texture_source_compare_iris_core_densified_texel_count"
        ]
        >= iris_accepted_count
    )
    assert (
        0.0
        <= summary["metrics"][
            "raw_face_texture_source_compare_iris_core_acceptance_fraction"
        ]
        <= 1.0
    )
    assert (
        0.0
        <= summary["metrics"][
            "raw_face_texture_source_compare_iris_annulus_rejected_fraction"
        ]
        <= 1.0
    )
    iris_halo_support = summary["metrics"][
        "raw_facial_feature_iris_halo_uv_support_texel_count"
    ]
    iris_halo_core = summary["metrics"][
        "raw_facial_feature_iris_halo_core_texel_count"
    ]
    iris_halo_edge = summary["metrics"][
        "raw_facial_feature_iris_halo_edge_texel_count"
    ]
    iris_halo_accepted_core = summary["metrics"][
        "raw_facial_feature_iris_halo_accepted_core_texel_count"
    ]
    assert summary["metrics"]["raw_facial_feature_iris_halo_feature_count"] == 2
    assert iris_halo_support > 0
    assert iris_halo_core > 0
    assert iris_halo_edge >= 0
    assert iris_halo_accepted_core > 0
    assert iris_halo_edge <= iris_halo_support
    assert iris_halo_core <= iris_halo_support
    assert (
        abs(
            summary["metrics"]["raw_facial_feature_iris_halo_edge_fraction"]
            - iris_halo_edge / max(iris_halo_support, 1)
        )
        < 1e-12
    )
    assert (
        abs(
            summary["metrics"][
                "raw_facial_feature_iris_halo_edge_to_core_ratio"
            ]
            - iris_halo_edge / max(iris_halo_core, 1)
        )
        < 1e-12
    )
    assert summary["metrics"][
        "raw_facial_feature_iris_halo_max_edge_to_core_ratio"
    ] >= summary["metrics"]["raw_facial_feature_iris_halo_edge_to_core_ratio"]
    assert (
        abs(
            summary["metrics"][
                "raw_facial_feature_iris_halo_support_to_accepted_core_ratio"
            ]
            - iris_halo_support / max(iris_halo_accepted_core, 1)
        )
        < 1e-12
    )
    assert summary["metrics"][
        "raw_facial_feature_left_iris_halo_edge_to_core_ratio"
    ] >= 0.0
    assert summary["metrics"][
        "raw_facial_feature_right_iris_halo_edge_to_core_ratio"
    ] >= 0.0
    assert summary["metrics"]["observed_face_fraction"] > 0.0
    assert summary["metrics"]["observed_vertex_fraction"] > 0.0
    assert summary["metrics"]["generated_face_count"] == 0
    assert summary["metrics"]["basecolor_observed_texel_fraction"] > 0.0
    assert summary["metrics"]["basecolor_generated_texel_fraction"] == 0.0
    assert summary["metrics"]["pbr_observed_texel_fraction"] > 0.0
    assert summary["metrics"]["pbr_generated_texel_fraction"] == 0.0
    assert summary["metrics"]["raw_pbr_face_material_detail_applied"] == 1.0
    assert summary["metrics"]["raw_pbr_face_material_detail_feature_count"] >= 12
    assert summary["metrics"]["raw_pbr_face_material_detail_texel_count"] >= (
        RAW_METRIC_THRESHOLDS["min_pbr_face_material_detail_texel_count"]
    )
    assert summary["metrics"]["raw_pbr_face_material_detail_normal_variance"] > 0.0
    assert summary["metrics"]["joint_projection_mean_keypoint_loss"] <= 0.35
    assert summary["metrics"]["joint_projection_mean_pixel_error"] >= 0.0
    assert summary["metrics"]["silhouette_residual_mean_union_error"] <= RAW_METRIC_THRESHOLDS[
        "max_silhouette_residual_mean_union_error"
    ]
    assert summary["metrics"]["silhouette_residual_mean_false_negative"] >= 0.0
    assert summary["metrics"]["silhouette_residual_mean_false_positive"] >= 0.0
    assert summary["metrics"]["final_loss"] <= summary["metrics"]["initial_loss"]
    assert summary["metrics"]["loss_reduction_fraction"] >= 0.0
    assert summary["metrics"]["keypoint_loss_reduction"] > 0.0
    assert summary["metrics"]["silhouette_loss_reduction"] > 0.0
    assert summary["metrics"]["vertex_count"] == CAT_TEMPLATE_VERTEX_COUNT
    assert summary["metrics"]["face_count"] == CAT_TEMPLATE_FACE_COUNT
    assert (
        summary["metrics"]["raw_quality_passed_check_count"]
        == summary["metrics"]["raw_quality_total_check_count"]
    )
    assert summary["metrics"]["raw_dense_evidence_manifest_present"] == 0.0
    assert summary["metrics"]["raw_dense_evidence_manifest_valid"] == 1.0
    assert summary["metrics"]["raw_dense_evidence_tensor_refs_valid"] == 1.0
    assert summary["artifacts"]["raw_manifest"].endswith("raw/frames.json")
    assert summary["artifacts"]["observation_manifest"].endswith("observations/raw/manifest.json")
    assert summary["artifacts"]["frame_selection_report"].endswith(
        "observations/raw/frame_selection.json"
    )
    assert summary["artifacts"]["texture_frame_manifest"].endswith(
        "observations/raw/texture_manifest.json"
    )
    assert summary["artifacts"]["init_manifest"].endswith("init/raw/manifest.json")
    assert summary["artifacts"]["fit_state"].endswith(
        "outputs/fitting/fit_state.safetensors"
    )
    assert summary["artifacts"]["fit_state"] == summary["artifacts"]["fit_state_safetensors"]
    assert summary["artifacts"]["fit_state_source_format"] == "safetensors"
    assert summary["artifacts"]["fit_state_safetensors"].endswith(
        "outputs/fitting/fit_state.safetensors"
    )
    assert not (project_dir / "outputs" / "fitting" / "fit_state.npz").exists()
    assert summary["artifacts"]["overlay_manifest"].endswith(
        "outputs/fitting/per_frame_overlay/manifest.json"
    )
    assert summary["artifacts"]["validation_report"].endswith("outputs/quality/validation.json")
    assert summary["artifacts"]["coverage_report"].endswith("outputs/quality/coverage.json")
    assert summary["artifacts"]["texture_manifest"].endswith(
        "outputs/textures/texture_manifest.json"
    )
    assert summary["artifacts"]["glb_face_feature_visibility"].endswith(
        "outputs/quality/raw_glb_face_feature_visibility.json"
    )
    texture_manifest = json.loads(
        (project_dir / "outputs" / "textures" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert texture_manifest["facial_features"]["applied"] is True
    assert texture_manifest["basecolor"]["width"] == RAW_METRIC_THRESHOLDS[
        "min_template_region_v2_texture_target_baked_size"
    ]
    assert texture_manifest["basecolor"]["height"] == RAW_METRIC_THRESHOLDS[
        "min_template_region_v2_texture_target_baked_size"
    ]
    assert texture_manifest["basecolor"]["texture_resolution_source"] == (
        "raw_template_region_v2_texture_target_2048"
    )
    assert texture_manifest["basecolor"]["template_region_v2_texture_target_bake_used"] == 1.0
    assert texture_manifest["basecolor"]["template_region_v2_texture_target_size"] == (
        texture_manifest["basecolor"]["width"]
    )
    assert texture_manifest["facial_features"]["source"] == (
        "observed_face_priority_texture_projection_v1"
    )
    assert texture_manifest["template_face_uv_regions"]["source"] == (
        "opencat3d.cat_template_face_uv_region_contract"
    )
    assert texture_manifest["template_face_uv_regions"]["region_count"] == 8
    uv_region_names = {
        region["name"] for region in texture_manifest["template_face_uv_regions"]["regions"]
    }
    assert {
        "left_eye",
        "right_eye",
        "nose",
        "muzzle",
        "mouth_line",
        "left_ear_inner",
        "right_ear_inner",
    }.issubset(uv_region_names)
    feature_names = {
        feature["name"] for feature in texture_manifest["facial_features"]["features"]
    }
    assert {
        "left_eye_core",
        "right_eye_core",
        "left_eye_iris",
        "right_eye_iris",
        "left_eye_pupil",
        "right_eye_pupil",
        "nose",
        "mouth_line",
        "left_muzzle",
        "right_muzzle",
        "left_whisker_pad",
        "right_whisker_pad",
        "left_whisker_root",
        "right_whisker_root",
        "left_ear_inner",
        "right_ear_inner",
    }.issubset(feature_names)
    features_by_name = {
        str(feature["name"]): feature
        for feature in texture_manifest["facial_features"]["features"]
    }
    assert features_by_name["left_eye_iris"]["sample_key"] == "left_iris"
    assert features_by_name["right_eye_iris"]["sample_key"] == "right_iris"
    assert features_by_name["left_eye_pupil"]["sample_key"] == "left_pupil"
    assert features_by_name["right_eye_pupil"]["sample_key"] == "right_pupil"
    assert texture_manifest["facial_features"]["feature_count"] >= 12
    assert texture_manifest["facial_features"]["whisker_pad_feature_count"] == 4
    assert texture_manifest["facial_features"]["whisker_pad_texel_count"] > 0
    assert texture_manifest["facial_features"]["whisker_pad_texel_fraction"] > 0.0
    assert texture_manifest["facial_features"]["painted_texel_count"] > 0
    assert texture_manifest["facial_features"]["observed_texel_count"] == (
        texture_manifest["facial_features"]["painted_texel_count"]
    )
    assert texture_manifest["facial_features"]["observed_texel_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_observed_texel_fraction"]
    )
    assert texture_manifest["facial_features"]["source_patch_count"] >= (
        texture_manifest["facial_features"]["feature_count"]
    )
    assert texture_manifest["facial_features"]["source_patch_bake_candidate_count"] >= 1
    assert texture_manifest["facial_features"]["source_patch_bake_selected_candidate"] in {
        "generic",
        "feedback",
    }
    assert isinstance(
        texture_manifest["facial_features"]["source_patch_bake_selected_feedback"],
        bool,
    )
    assert (
        texture_manifest["facial_features"][
            "source_patch_bake_visual_quality_passed_count"
        ]
        >= 0
    )
    assert (
        0.0
        <= texture_manifest["facial_features"][
            "source_patch_bake_visual_quality_min_score"
        ]
        <= texture_manifest["facial_features"][
            "source_patch_bake_visual_quality_mean_score"
        ]
        <= 1.0
    )
    assert texture_manifest["facial_features"]["source_patch_bake_candidates"]
    assert (
        texture_manifest["facial_features"][
            "eye_source_detail_reinforcement_feature_count"
        ]
        > 0
    )
    assert (
        texture_manifest["facial_features"][
            "eye_source_detail_reinforcement_texel_count"
        ]
        > 0
    )
    assert (
        0.0
        < texture_manifest["facial_features"][
            "eye_source_detail_reinforcement_texel_fraction"
        ]
        <= 1.0
    )
    assert texture_manifest["facial_features"]["source_patch_texel_count"] == (
        texture_manifest["facial_features"]["painted_texel_count"]
    )
    assert texture_manifest["facial_features"]["source_patch_nominal_texel_count"] >= (
        texture_manifest["facial_features"]["source_patch_texel_count"]
    )
    assert (
        0.0
        <= texture_manifest["facial_features"][
            "source_patch_footprint_reduction_fraction"
        ]
        <= 1.0
    )
    assert (
        texture_manifest["facial_features"]["source_patch_mean_footprint_area_scale"]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_mean_footprint_area_scale"
        ]
    )
    assert (
        texture_manifest["facial_features"]["source_patch_min_footprint_area_scale"]
        > 0.0
    )
    assert (
        texture_manifest["facial_features"]["source_patch_shape_mask_texel_count"]
        > 0
    )
    assert (
        texture_manifest["facial_features"]["source_patch_shape_mask_fraction"]
        >= RAW_METRIC_THRESHOLDS[
            "min_facial_feature_source_patch_shape_mask_fraction"
        ]
    )
    assert (
        texture_manifest["facial_features"]["source_patch_shape_mask_fraction"]
        <= 1.0
    )
    assert texture_manifest["facial_features"]["source_patch_shape_alpha_mean"] > 0.0
    assert (
        texture_manifest["facial_features"][
            "source_patch_source_mask_densified_texel_count"
        ]
        >= texture_manifest["facial_features"][
            "source_patch_source_mask_observed_texel_count"
        ]
    )
    assert (
        texture_manifest["facial_features"][
            "source_patch_source_mask_densification_gain"
        ]
        >= RAW_METRIC_THRESHOLDS[
            "min_facial_feature_source_patch_source_mask_densification_gain"
        ]
    )
    assert (
        texture_manifest["facial_features"][
            "source_patch_mean_source_mask_densification_gain"
        ]
        > 1.0
    )
    assert (
        texture_manifest["facial_features"][
            "source_patch_localization_candidate_texel_count"
        ]
        >= texture_manifest["facial_features"][
            "source_patch_localization_selected_texel_count"
        ]
        > 0
    )
    assert (
        0.0
        < texture_manifest["facial_features"][
            "source_patch_localization_fraction"
        ]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_localization_fraction"
        ]
    )
    assert (
        0.0
        < texture_manifest["facial_features"][
            "source_patch_mean_localization_fraction"
        ]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_localization_fraction"
        ]
    )
    assert texture_manifest["facial_features"][
        "source_patch_landmark_aligned_count"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_landmark_aligned_count"
    ]
    assert (
        0.0
        < texture_manifest["facial_features"][
            "source_patch_mean_landmark_confidence"
        ]
        <= 1.0
    )
    assert texture_manifest["facial_features"][
        "source_patch_landmark_destination_scaled_count"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_landmark_destination_scaled_count"
    ]
    assert (
        0.0
        < texture_manifest["facial_features"][
            "source_patch_mean_landmark_destination_scale"
        ]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_mean_landmark_destination_scale"
        ]
    )
    assert texture_manifest["facial_features"]["destination_anchor_source"] == (
        "face_anatomical_uv_anchor_v1"
    )
    assert texture_manifest["facial_features"][
        "destination_anchor_anatomical_count"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_destination_anatomical_anchor_count"
    ]
    assert texture_manifest["facial_features"]["destination_anchor_order_valid"] is True
    assert texture_manifest["facial_features"][
        "destination_anchor_eye_to_nose_v_gap"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_face_feature_destination_anchor_eye_to_nose_v_gap"
    ]
    assert texture_manifest["facial_features"][
        "destination_anchor_nose_to_mouth_v_gap"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_face_feature_destination_anchor_nose_to_mouth_v_gap"
    ]
    assert texture_manifest["facial_features"][
        "destination_anchor_eye_symmetry_u_error"
    ] <= RAW_METRIC_THRESHOLDS[
        "max_face_feature_destination_anchor_eye_symmetry_u_error"
    ]
    assert texture_manifest["facial_features"]["source_patch_texel_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_fraction"]
    )
    assert texture_manifest["facial_features"]["source_patch_quality_passed_count"] == (
        texture_manifest["facial_features"]["source_patch_quality_total_count"]
    )
    assert texture_manifest["facial_features"]["source_patch_quality_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_quality_fraction"]
    )
    assert texture_manifest["facial_features"]["source_patch_min_quality_score"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_quality_score"]
    )
    assert texture_manifest["facial_features"]["source_patch_view_weighting_source"] == (
        "raw_frontal_face_confidence_v1"
    )
    assert texture_manifest["facial_features"]["source_patch_view_weighting_used"] is True
    assert texture_manifest["facial_features"]["source_patch_view_weighted_count"] == (
        texture_manifest["facial_features"]["source_patch_count"]
    )
    min_view_weight = texture_manifest["facial_features"]["source_patch_min_view_weight"]
    mean_view_weight = texture_manifest["facial_features"]["source_patch_mean_view_weight"]
    assert 0.0 < min_view_weight <= mean_view_weight + 1e-12
    assert mean_view_weight <= 1.0
    min_frontal = texture_manifest["facial_features"][
        "source_patch_min_frontal_face_confidence"
    ]
    mean_frontal = texture_manifest["facial_features"][
        "source_patch_mean_frontal_face_confidence"
    ]
    assert 0.0 <= min_frontal <= mean_frontal + 1e-12
    assert mean_frontal <= 1.0
    min_view_rank = texture_manifest["facial_features"][
        "source_patch_min_view_weighted_rank"
    ]
    mean_view_rank = texture_manifest["facial_features"][
        "source_patch_mean_view_weighted_rank"
    ]
    assert 0.0 < min_view_rank <= mean_view_rank + 1e-12
    assert mean_view_rank <= 1.0
    assert texture_manifest["facial_features"]["source_patch_view_labels"]
    assert texture_manifest["facial_features"]["source_patch_fusion_min_weight"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_fusion_weight"]
    )
    assert texture_manifest["facial_features"]["source_patch_fusion_mean_weight"] >= (
        texture_manifest["facial_features"]["source_patch_fusion_min_weight"]
    )
    assert summary["metrics"]["raw_facial_feature_source_patch_nominal_texel_count"] >= (
        summary["metrics"]["raw_facial_feature_source_patch_texel_count"]
    )
    assert summary["metrics"][
        "raw_facial_feature_source_patch_bake_candidate_count"
    ] >= 1
    assert 0.0 <= summary["metrics"][
        "raw_facial_feature_source_patch_bake_selected_feedback"
    ] <= 1.0
    assert summary["metrics"][
        "raw_facial_feature_source_patch_view_weighting_used"
    ] == 1.0
    assert summary["metrics"][
        "raw_facial_feature_source_patch_view_weighted_count"
    ] == summary["metrics"]["raw_facial_feature_source_patch_count"]
    min_summary_view_weight = summary["metrics"][
        "raw_facial_feature_source_patch_min_view_weight"
    ]
    mean_summary_view_weight = summary["metrics"][
        "raw_facial_feature_source_patch_mean_view_weight"
    ]
    assert 0.0 < min_summary_view_weight <= mean_summary_view_weight + 1e-12
    assert mean_summary_view_weight <= 1.0
    min_summary_frontal = summary["metrics"][
        "raw_facial_feature_source_patch_min_frontal_face_confidence"
    ]
    mean_summary_frontal = summary["metrics"][
        "raw_facial_feature_source_patch_mean_frontal_face_confidence"
    ]
    assert 0.0 <= min_summary_frontal <= mean_summary_frontal + 1e-12
    assert mean_summary_frontal <= 1.0
    min_summary_view_rank = summary["metrics"][
        "raw_facial_feature_source_patch_min_view_weighted_rank"
    ]
    mean_summary_view_rank = summary["metrics"][
        "raw_facial_feature_source_patch_mean_view_weighted_rank"
    ]
    assert 0.0 < min_summary_view_rank <= mean_summary_view_rank + 1e-12
    assert mean_summary_view_rank <= 1.0
    assert summary["metrics"][
        "raw_facial_feature_source_patch_bake_visual_quality_passed_count"
    ] >= 0
    assert 0.0 <= summary["metrics"][
        "raw_facial_feature_source_patch_bake_visual_quality_min_score"
    ] <= summary["metrics"][
        "raw_facial_feature_source_patch_bake_visual_quality_mean_score"
    ] <= 1.0
    assert summary["metrics"]["raw_face_texture_bake_comparison_candidate_count"] >= 1
    assert summary["metrics"]["raw_face_texture_bake_comparison_region_count"] >= 6
    assert 0.0 <= summary["metrics"][
        "raw_face_texture_bake_comparison_selected_feedback"
    ] <= 1.0
    assert summary["metrics"][
        "raw_face_texture_bake_comparison_score_improved_region_count"
    ] >= 0
    assert summary["metrics"][
        "raw_face_texture_bake_comparison_coverage_improved_region_count"
    ] >= 0
    assert summary["metrics"][
        "raw_face_texture_bake_comparison_source_detail_improved_region_count"
    ] >= 0
    assert summary["metrics"]["raw_face_texture_bake_comparison_max_score_delta"] >= 0.0
    assert (
        summary["metrics"][
            "raw_face_texture_bake_comparison_selected_not_lower_rank"
        ]
        == 1.0
    )
    assert (
        0.0
        <= summary["metrics"][
            "raw_facial_feature_source_patch_footprint_reduction_fraction"
        ]
        <= 1.0
    )
    assert summary["metrics"][
        "raw_facial_feature_source_patch_mean_footprint_area_scale"
    ] <= RAW_METRIC_THRESHOLDS[
        "max_facial_feature_source_patch_mean_footprint_area_scale"
    ]
    assert summary["metrics"][
        "raw_facial_feature_source_patch_min_footprint_area_scale"
    ] > 0.0
    assert summary["metrics"][
        "raw_facial_feature_source_patch_shape_mask_texel_count"
    ] > 0
    assert summary["metrics"][
        "raw_facial_feature_source_patch_shape_mask_fraction"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_shape_mask_fraction"
    ]
    assert summary["metrics"][
        "raw_facial_feature_source_patch_shape_mask_fraction"
    ] <= 1.0
    assert summary["metrics"][
        "raw_facial_feature_source_patch_shape_alpha_mean"
    ] > 0.0
    assert summary["metrics"][
        "raw_facial_feature_source_patch_source_mask_densified_texel_count"
    ] >= summary["metrics"][
        "raw_facial_feature_source_patch_source_mask_observed_texel_count"
    ]
    assert summary["metrics"][
        "raw_facial_feature_source_patch_source_mask_densification_gain"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_source_mask_densification_gain"
    ]
    assert summary["metrics"][
        "raw_facial_feature_source_patch_mean_source_mask_densification_gain"
    ] > 1.0
    assert summary["metrics"][
        "raw_facial_feature_source_patch_localization_candidate_texel_count"
    ] >= summary["metrics"][
        "raw_facial_feature_source_patch_localization_selected_texel_count"
    ] > 0
    assert (
        0.0
        < summary["metrics"][
            "raw_facial_feature_source_patch_localization_fraction"
        ]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_localization_fraction"
        ]
    )
    assert (
        0.0
        < summary["metrics"][
            "raw_facial_feature_source_patch_mean_localization_fraction"
        ]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_localization_fraction"
        ]
    )
    assert summary["metrics"][
        "raw_facial_feature_source_patch_landmark_aligned_count"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_landmark_aligned_count"
    ]
    assert (
        0.0
        < summary["metrics"][
            "raw_facial_feature_source_patch_mean_landmark_confidence"
        ]
        <= 1.0
    )
    assert summary["metrics"][
        "raw_facial_feature_source_patch_landmark_destination_scaled_count"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_landmark_destination_scaled_count"
    ]
    assert (
        0.0
        < summary["metrics"][
            "raw_facial_feature_source_patch_mean_landmark_destination_scale"
        ]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_mean_landmark_destination_scale"
        ]
    )
    assert summary["metrics"][
        "raw_face_feature_destination_anchor_anatomical_count"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_destination_anatomical_anchor_count"
    ]
    assert summary["metrics"]["raw_face_feature_destination_anchor_order_valid"] == 1.0
    assert summary["metrics"][
        "raw_face_feature_destination_anchor_eye_to_nose_v_gap"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_face_feature_destination_anchor_eye_to_nose_v_gap"
    ]
    assert summary["metrics"][
        "raw_face_feature_destination_anchor_nose_to_mouth_v_gap"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_face_feature_destination_anchor_nose_to_mouth_v_gap"
    ]
    assert summary["metrics"][
        "raw_face_feature_destination_anchor_eye_symmetry_u_error"
    ] <= RAW_METRIC_THRESHOLDS[
        "max_face_feature_destination_anchor_eye_symmetry_u_error"
    ]
    assert texture_manifest["facial_features"]["source_patch_core_texel_count"] > 0
    assert texture_manifest["facial_features"]["source_patch_core_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_core_fraction"]
    )
    assert texture_manifest["facial_features"]["source_patch_feathered_edge_texel_count"] > 0
    assert texture_manifest["facial_features"]["source_patch_feathered_edge_fraction"] >= (
        RAW_METRIC_THRESHOLDS[
            "min_facial_feature_source_patch_feathered_edge_fraction"
        ]
    )
    assert texture_manifest["facial_features"]["source_patch_mean_blend_alpha"] > 0.0
    assert all(
        feature["provenance"] == "observed"
        and feature["source_view_id"] > 0
        and feature["observed_texel_count"] > 0
        and feature["source_patch_used"] is True
        and Path(str(feature["source_patch_path"])).exists()
        and len(str(feature["source_patch_sha256"])) == 64
        and Path(str(feature["source_patch_mask_path"])).exists()
        and len(str(feature["source_patch_mask_sha256"])) == 64
        and Path(str(feature["uv_support_mask_path"])).exists()
        and len(str(feature["uv_support_mask_sha256"])) == 64
        and feature["uv_support_mask_width"] == texture_manifest["basecolor"]["width"]
        and feature["uv_support_mask_height"] == texture_manifest["basecolor"]["height"]
        and feature["uv_support_mask_texel_count"] == feature["source_patch_texel_count"]
        and feature["source_patch_width"] > 0
        and feature["source_patch_height"] > 0
        and feature["source_patch_mask_width"] == feature["source_patch_width"]
        and feature["source_patch_mask_height"] == feature["source_patch_height"]
        and feature["source_patch_quality_passed"] is True
        and feature["source_patch_quality_score"]
        >= RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_quality_score"]
        and feature["source_patch_luma_std"] >= 0.0
        and feature["source_patch_unique_rgb_count"] > 0
        and feature["source_patch_view_weighting_used"] is True
        and 0.0 < feature["source_patch_view_weighted_rank"] <= 1.0
        and 0.0 <= feature["source_patch_frontal_face_confidence"] <= 1.0
        and 0.0 < feature["source_patch_view_weight"] <= 1.0
        and isinstance(feature["source_patch_view_label"], str)
        and feature["source_patch_fusion_weight"]
        >= RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_fusion_weight"]
        and feature["source_patch_fusion_weight"] <= 0.96
        and feature["source_patch_feather_width"] > 0.0
        and feature["source_patch_feather_x_scale"] > 0.0
        and feature["source_patch_feather_y_scale"] > 0.0
        and feature["nominal_radius_uv"][0] >= feature["radius_uv"][0] > 0.0
        and feature["nominal_radius_uv"][1] >= feature["radius_uv"][1] > 0.0
        and feature["source_patch_footprint_density"] > 0.0
        and 0.0 < feature["source_patch_footprint_area_scale"] <= 1.0
        and 0.0 < feature["source_patch_footprint_x_scale"] <= 1.0
        and 0.0 < feature["source_patch_footprint_y_scale"] <= 1.0
        and feature["source_patch_nominal_texel_count"]
        >= feature["source_patch_texel_count"]
        and feature["source_patch_ellipse_texel_count"]
        >= feature["source_patch_texel_count"]
        and 0.0 <= feature["source_patch_footprint_reduction_fraction"] <= 1.0
        and feature["source_patch_shape_mask_used"] is True
        and feature["source_patch_shape_mask_texel_count"] > 0
        and feature["source_patch_shape_alpha_mean"] > 0.0
        and feature["source_patch_source_mask_densified_texel_count"]
        >= feature["source_patch_source_mask_observed_texel_count"]
        and feature["source_patch_source_mask_densification_gain"] >= 1.0
        and feature["source_patch_localization_candidate_texel_count"]
        >= feature["source_patch_localization_selected_texel_count"]
        > 0
        and 0.0 < feature["source_patch_localization_fraction"] <= 1.0
        and 0.0 <= feature["source_patch_landmark_confidence"] <= 1.0
        and isinstance(feature["source_patch_landmark_aligned"], bool)
        and 0.0 < feature["source_patch_landmark_destination_scale"] <= 1.0
        and isinstance(feature["source_patch_landmark_destination_scaled"], bool)
        and feature["destination_anchor_source"] == "face_anatomical_uv_anchor_v1"
        and feature["source_patch_mean_blend_alpha"] > 0.0
        and feature["source_patch_core_texel_count"] > 0
        and feature["source_patch_feathered_edge_texel_count"] > 0
        and feature["source_patch_texel_count"] > 0
        for feature in texture_manifest["facial_features"]["features"]
    )
    contrast = texture_manifest["facial_features"]["contrast"]
    assert contrast["source"] == "observed_texture_face_feature_contrast_v1"
    assert contrast["min_required_contrast"] >= RAW_METRIC_THRESHOLDS[
        "min_face_feature_required_contrast"
    ]
    assert contrast["iris_to_pupil_rgb_distance"] > 0.0
    assert contrast["required_contrast_passed"] is True
    eye_detail_reinforced = [
        feature
        for feature in texture_manifest["facial_features"]["features"]
        if str(feature["name"]) in {
            "left_eye_iris",
            "right_eye_iris",
            "left_eye_pupil",
            "right_eye_pupil",
        }
        and feature["visibility_reinforcement_detail_preserved"] is True
    ]
    assert len(eye_detail_reinforced) >= 2
    assert all(
        feature["visibility_reinforcement_texel_count"]
        <= feature["source_patch_shape_mask_texel_count"]
        for feature in eye_detail_reinforced
    )
    assert all(
        str(feature["uv_support_mask_path"]).endswith("_uv_support.png")
        for feature in eye_detail_reinforced
    )
    assert texture_manifest["pbr"]["face_material_detail_applied"] is True
    assert texture_manifest["pbr"]["face_material_detail_source"] == (
        "inferred_from_observed_face_feature_anchors_v1"
    )
    assert texture_manifest["pbr"]["width"] == texture_manifest["basecolor"]["width"]
    assert texture_manifest["pbr"]["height"] == texture_manifest["basecolor"]["height"]
    assert texture_manifest["pbr"]["face_material_detail_feature_count"] >= 12
    assert texture_manifest["pbr"]["face_material_detail_texel_count"] >= (
        RAW_METRIC_THRESHOLDS["min_pbr_face_material_detail_texel_count"]
    )
    assert texture_manifest["pbr"]["face_material_detail_normal_variance"] > 0.0
    normal = np.asarray(Image.open(project_dir / "outputs" / "textures" / "normal.png"))
    roughness = np.asarray(
        Image.open(project_dir / "outputs" / "textures" / "roughness.png").convert("L")
    )
    ao = np.asarray(Image.open(project_dir / "outputs" / "textures" / "ao.png").convert("L"))
    assert np.unique(normal.reshape(-1, 3), axis=0).shape[0] > 1
    assert int(roughness.min()) < 204
    assert int(ao.min()) < 255
    assert summary["artifacts"]["basecolor"].endswith("outputs/textures/basecolor.png")
    assert summary["artifacts"]["basecolor_confidence"].endswith(
        "outputs/textures/basecolor_confidence.png"
    )
    assert summary["artifacts"]["source_view_id"].endswith("outputs/textures/source_view_id.png")
    assert summary["artifacts"]["normal"].endswith("outputs/textures/normal.png")
    assert summary["artifacts"]["roughness"].endswith("outputs/textures/roughness.png")
    assert summary["artifacts"]["ao"].endswith("outputs/textures/ao.png")
    assert summary["artifacts"]["joint_projection"].endswith(
        "outputs/quality/raw_joint_projection.json"
    )
    assert summary["artifacts"]["silhouette_residuals"].endswith(
        "outputs/quality/raw_silhouette_residuals.json"
    )
    assert summary["artifacts"]["face_turntable"].endswith(
        "outputs/quality/raw_face_turntable.json"
    )
    assert summary["artifacts"]["face_turntable_manifest"].endswith(
        "outputs/quality/raw_face_turntable/manifest.json"
    )
    assert summary["artifacts"]["face_turntable_front"].endswith(
        "outputs/quality/raw_face_turntable/front.png"
    )
    assert summary["artifacts"]["face_turntable_front_textured"].endswith(
        "outputs/quality/raw_face_turntable/front_textured.png"
    )
    assert summary["artifacts"]["face_texture_closeup"].endswith(
        "outputs/quality/raw_face_turntable/face_closeup.png"
    )
    assert summary["artifacts"]["face_texture_source_comparison"].endswith(
        "outputs/quality/raw_face_turntable/face_source_comparison.png"
    )
    assert summary["artifacts"]["face_uv_density"].endswith(
        "outputs/quality/raw_face_uv_density.json"
    )
    assert summary["artifacts"]["face_uv_density_overlay"].endswith(
        "outputs/quality/raw_face_uv_density.png"
    )
    assert summary["artifacts"]["face_texture_targets"].endswith(
        "outputs/quality/raw_face_texture_targets.json"
    )
    assert summary["artifacts"]["face_texture_diagnostics"].endswith(
        "outputs/quality/raw_face_texture_diagnostics.json"
    )
    assert summary["artifacts"]["face_texture_bake_comparison"].endswith(
        "outputs/quality/raw_face_texture_bake_comparison.json"
    )
    assert summary["artifacts"]["face_texture_crops"].endswith(
        "outputs/quality/raw_face_texture_crops.png"
    )
    assert summary["artifacts"]["face_texture_crop_dir"].endswith(
        "outputs/quality/raw_face_texture_crops"
    )
    assert summary["artifacts"]["compliance_audit"].endswith(
        "raw_opencat3d_compliance_audit.json"
    )
    assert summary["artifacts"]["idle_animation"].endswith(
        "outputs/avatar/animations/idle.glb"
    )
    assert summary["artifacts"]["skeleton"].endswith("outputs/avatar/skeleton.json")


def test_run_raw_pipeline_keeps_extra_photos_for_texture_projection(tmp_path) -> None:
    project_dir = tmp_path / "project"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_rgb_png(input_dir / "a.png", width=12, height=12, bbox=(2, 2, 8, 8))
    _write_rgb_png(input_dir / "b.png", width=12, height=12, bbox=(3, 2, 9, 8))
    _write_rgb_png(input_dir / "c.png", width=12, height=12, bbox=(2, 3, 8, 9))
    _write_rgb_png(input_dir / "d.png", width=12, height=12, bbox=(3, 3, 9, 9))

    result = run_raw_pipeline(
        project_dir,
        input_paths=[input_dir],
        color_delta_threshold=0.10,
        camera_root_steps=2,
        max_selected_frames=2,
    )

    assert result.success
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["raw_selected_frame_count"] == 2
    assert summary["metrics"]["raw_texture_frame_count"] == 4
    assert summary["metrics"]["raw_texture_photo_frame_count"] == 4
    texture_manifest = json.loads(
        (project_dir / "outputs" / "textures" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert texture_manifest["basecolor"]["fit_frame_count"] == 2
    assert texture_manifest["basecolor"]["projection_frame_count"] == 4
    assert texture_manifest["basecolor"]["supplemental_texture_frame_count"] == 2


def test_run_raw_pipeline_rejects_negative_feature_anchor_weight(tmp_path) -> None:
    with pytest.raises(ValueError, match="raw_feature_anchor_weight"):
        run_raw_pipeline(
            tmp_path / "project",
            input_paths=[tmp_path / "inputs"],
            raw_feature_anchor_weight=-0.1,
        )


def test_run_cli_executes_raw_pipeline(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "project"
    input_dir = tmp_path / "inputs"
    _write_raw_inputs(input_dir)

    result = runner.invoke(
        app,
        [
            "run",
            str(project_dir),
            "--mode",
            "raw",
            "--input",
            str(input_dir),
            "--color-delta-threshold",
            "0.10",
            "--camera-root-steps",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "success=True" in result.output
    assert (project_dir / "outputs" / "quality" / "raw_pipeline_summary.json").exists()
    assert (project_dir / "outputs" / "quality" / "raw_evaluation.json").exists()
    assert (project_dir / "outputs" / "quality" / "raw_opencat3d_compliance_audit.json").exists()
    assert (project_dir / "outputs" / "quality" / "validation.json").exists()
    assert (project_dir / "outputs" / "quality" / "raw_joint_projection.json").exists()
    assert (project_dir / "outputs" / "quality" / "raw_silhouette_residuals.json").exists()
    assert (project_dir / "outputs" / "fitting" / "fit_state.safetensors").exists()
    assert (project_dir / "outputs" / "avatar" / "cat_avatar.glb").exists()
    assert (project_dir / "outputs" / "avatar" / "skeleton.json").exists()
    assert (project_dir / "outputs" / "textures" / "basecolor.png").exists()
    assert (project_dir / "outputs" / "textures" / "normal.png").exists()
    assert (project_dir / "outputs" / "textures" / "roughness.png").exists()
    assert (project_dir / "outputs" / "textures" / "ao.png").exists()


def _write_raw_inputs(input_dir) -> None:
    input_dir.mkdir()
    _write_rgb_png(input_dir / "b.png", width=8, height=8, bbox=(2, 2, 6, 6))
    _write_rgb_png(input_dir / "a.png", width=8, height=8, bbox=(1, 1, 5, 5))


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = np.asarray([222, 220, 208], dtype=np.uint8)
    if x1 - x0 >= 4 and y1 - y0 >= 4:
        image[y0, x0 + 1, :] = np.asarray([118, 104, 54], dtype=np.uint8)
        image[y0, x1 - 2, :] = np.asarray([118, 104, 54], dtype=np.uint8)
        image[y0 + 1, x0 + 1, :] = np.asarray([10, 9, 8], dtype=np.uint8)
        image[y0 + 1, x1 - 2, :] = np.asarray([10, 9, 8], dtype=np.uint8)
        image[y0 + 2, x0 + 2, :] = np.asarray([224, 136, 146], dtype=np.uint8)
        image[y0 + 3, x0 + 2, :] = np.asarray([52, 36, 38], dtype=np.uint8)
    Image.fromarray(image).save(path)


def _write_test_video(path, *, frame_count: int, size: tuple[int, int]) -> None:
    width, height = size
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=size,
        fps=6,
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        for index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = 20 + index * 20
            frame[:, :, 1] = 60
            frame[:, :, 2] = 120
            writer.send(frame.tobytes())
    finally:
        writer.close()
