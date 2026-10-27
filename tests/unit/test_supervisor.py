import json

from opencat3d.evaluation.supervisor import (
    write_candidate_comparison_report,
    write_guarded_candidate_comparison_report,
)


def test_write_candidate_comparison_report_ranks_candidates(tmp_path) -> None:
    poor = _candidate(
        tmp_path,
        "poor",
        overlay=0.2,
        residual=0.8,
        texture=0.1,
        face=0.1,
    )
    better = _candidate(
        tmp_path,
        "better",
        overlay=0.45,
        residual=0.5,
        texture=0.4,
        face=0.3,
    )

    result = write_candidate_comparison_report(
        {"poor": poor, "better": better},
        tmp_path / "comparison.json",
    )

    assert result.best_candidate == "better"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["best_candidate"] == "better"
    assert [candidate["label"] for candidate in report["candidates"]] == ["better", "poor"]


def test_write_candidate_comparison_report_rewards_face_priority_metrics(tmp_path) -> None:
    generic = _candidate(
        tmp_path,
        "generic",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
    )
    face_detail = _candidate(
        tmp_path,
        "face_detail",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
        face_contrast=0.32,
        iris_pupil_contrast=0.24,
        texture_resolution_scale=2.0,
        pbr_detail_applied=1.0,
        pbr_detail_texels=4096,
        pbr_normal_variance=14.0,
        glb_face_feature_gate=1.0,
        glb_face_feature_visible_groups=3,
        glb_face_feature_visible_features=10,
        glb_face_feature_sampled_texels=4096,
        glb_face_feature_iris_pupil=0.24,
        glb_face_feature_eye_luma_std=0.08,
        glb_face_feature_nose_muzzle=0.32,
        glb_face_feature_mouth_muzzle=0.32,
        selected_frames=12,
        video_frames=8,
    )

    result = write_candidate_comparison_report(
        {"generic": generic, "face_detail": face_detail},
        tmp_path / "face_comparison.json",
    )

    assert result.best_candidate == "face_detail"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    best_terms = report["candidates"][0]["score_terms"]
    assert best_terms["face_priority_texture_used"] > 0.0
    assert best_terms["facial_feature_count"] > 0.0
    assert best_terms["face_feature_min_required_contrast"] > 0.0
    assert best_terms["face_feature_iris_to_pupil_rgb_distance"] > 0.0
    assert best_terms["glb_face_feature_eye_detail_luma_std"] > 0.0
    assert best_terms["glb_face_feature_iris_to_pupil_rgb_distance"] > 0.0
    assert best_terms["glb_face_feature_material_visibility"] > 0.0
    assert best_terms["glb_face_feature_visible_group_count"] > 0.0
    assert best_terms["raw_texture_resolution_scale"] > 0.0
    assert best_terms["pbr_face_material_detail"] > 0.0
    assert best_terms["pbr_face_material_detail_normal_variance"] > 0.0


def test_write_candidate_comparison_report_rewards_observed_whisker_pad_detail(
    tmp_path,
) -> None:
    baseline = _candidate(
        tmp_path,
        "baseline",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
    )
    whisker_detail = _candidate(
        tmp_path,
        "whisker_detail",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        face_priority=1.0,
        feature_count=18,
        painted_texels=4096,
        whisker_pad_features=4,
        whisker_pad_texels=2048,
        whisker_pad_fraction=0.10,
    )

    result = write_candidate_comparison_report(
        {"baseline": baseline, "whisker_detail": whisker_detail},
        tmp_path / "whisker_comparison.json",
    )

    assert result.best_candidate == "whisker_detail"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    best_terms = report["candidates"][0]["score_terms"]
    best_metrics = report["candidates"][0]["metrics"]
    assert best_terms["facial_feature_whisker_pad_feature_count"] > 0.0
    assert best_terms["facial_feature_whisker_pad_texel_fraction"] > 0.0
    assert best_metrics["raw_facial_feature_whisker_pad_feature_count"] == 4.0
    assert best_metrics["raw_facial_feature_whisker_pad_texel_count"] == 2048.0


def test_write_candidate_comparison_report_rewards_view_weighted_face_sources(
    tmp_path,
) -> None:
    weak_source = _candidate(
        tmp_path,
        "weak_source",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
        source_patch_quality=0.42,
    )
    frontal_source = _candidate(
        tmp_path,
        "frontal_source",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
        source_patch_quality=0.72,
        source_patch_view_weighting_used=1.0,
        source_patch_view_weighted_count=14,
        source_patch_mean_view_weight=0.76,
        source_patch_mean_frontal_face_confidence=0.64,
        source_patch_mean_view_weighted_rank=0.81,
    )

    result = write_candidate_comparison_report(
        {"weak_source": weak_source, "frontal_source": frontal_source},
        tmp_path / "source_view_comparison.json",
    )

    assert result.best_candidate == "frontal_source"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    best_terms = report["candidates"][0]["score_terms"]
    best_metrics = report["candidates"][0]["metrics"]
    assert best_terms["facial_feature_source_patch_view_weighting_used"] > 0.0
    assert best_terms["facial_feature_source_patch_view_weighted_count"] > 0.0
    assert best_terms["facial_feature_source_patch_mean_view_weight"] > 0.0
    assert best_terms["facial_feature_source_patch_mean_frontal_face_confidence"] > 0.0
    assert best_terms["facial_feature_source_patch_mean_view_weighted_rank"] > 0.0
    assert best_terms["facial_feature_source_patch_mean_quality_score"] > 0.0
    assert best_metrics["raw_facial_feature_source_patch_view_weighting_used"] == 1.0
    assert best_metrics["raw_facial_feature_source_patch_view_weighted_count"] == 14.0
    assert best_metrics["raw_facial_feature_source_patch_mean_view_weight"] == 0.76
    assert (
        best_metrics["raw_facial_feature_source_patch_mean_frontal_face_confidence"]
        == 0.64
    )
    assert best_metrics["raw_facial_feature_source_patch_mean_view_weighted_rank"] == 0.81


def test_write_candidate_comparison_report_rewards_glb_visible_eye_detail(
    tmp_path,
) -> None:
    high_source_quality = _candidate(
        tmp_path,
        "high_source_quality",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
        source_patch_quality=0.72,
        glb_face_feature_gate=1.0,
        glb_face_feature_visible_groups=3,
        glb_face_feature_visible_features=10,
        glb_face_feature_sampled_texels=4096,
        glb_face_feature_iris_pupil=0.18,
        glb_face_feature_eye_luma_std=0.08,
    )
    visible_eye_detail = _candidate(
        tmp_path,
        "visible_eye_detail",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
        source_patch_quality=0.58,
        glb_face_feature_gate=1.0,
        glb_face_feature_visible_groups=3,
        glb_face_feature_visible_features=10,
        glb_face_feature_sampled_texels=4096,
        glb_face_feature_iris_pupil=0.23,
        glb_face_feature_eye_luma_std=0.12,
    )

    result = write_candidate_comparison_report(
        {
            "high_source_quality": high_source_quality,
            "visible_eye_detail": visible_eye_detail,
        },
        tmp_path / "glb_eye_detail_comparison.json",
    )

    assert result.best_candidate == "visible_eye_detail"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    best_terms = report["candidates"][0]["score_terms"]
    assert best_terms["glb_face_feature_eye_detail_luma_std"] > 0.0
    assert best_terms["glb_face_feature_iris_to_pupil_rgb_distance"] > 0.0


def test_write_candidate_comparison_report_penalizes_failed_texture_guard(
    tmp_path,
) -> None:
    guard_failed = _candidate(
        tmp_path,
        "guard_failed",
        overlay=0.38,
        residual=0.5,
        texture=0.40,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
        glb_face_feature_gate=1.0,
        glb_face_feature_visible_groups=3,
        glb_face_feature_visible_features=10,
        glb_face_feature_sampled_texels=4096,
        glb_face_feature_iris_pupil=0.23,
        glb_face_feature_eye_luma_std=0.22,
        texture_guard_passed=False,
        texture_guard_regressed_metrics=("raw_glb_face_feature_eye_detail_luma_std",),
    )
    guard_passed = _candidate(
        tmp_path,
        "guard_passed",
        overlay=0.38,
        residual=0.5,
        texture=0.40,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
        glb_face_feature_gate=1.0,
        glb_face_feature_visible_groups=3,
        glb_face_feature_visible_features=10,
        glb_face_feature_sampled_texels=4096,
        glb_face_feature_iris_pupil=0.23,
        glb_face_feature_eye_luma_std=0.20,
        texture_guard_passed=True,
    )

    result = write_candidate_comparison_report(
        {"guard_failed": guard_failed, "guard_passed": guard_passed},
        tmp_path / "texture_guard_comparison.json",
    )

    assert result.best_candidate == "guard_passed"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    failed = next(item for item in report["candidates"] if item["label"] == "guard_failed")
    passed = next(item for item in report["candidates"] if item["label"] == "guard_passed")
    assert failed["score_terms"]["texture_only_promotion_guard_failed"] < 0.0
    assert passed["score_terms"]["texture_only_promotion_guard_failed"] == 0.0
    assert failed["metrics"]["raw_texture_only_promotion_guard_present"] == 1.0
    assert failed["metrics"]["raw_texture_only_promotion_guard_passed"] == 0.0
    assert failed["metrics"][
        "raw_texture_only_promotion_guard_strict_regressed_metric_count"
    ] == 1.0


def test_write_candidate_comparison_report_penalizes_failed_texture_admission(
    tmp_path,
) -> None:
    admission_failed = _candidate(
        tmp_path,
        "admission_failed",
        overlay=0.38,
        residual=0.5,
        texture=0.40,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
    )
    admission_absent = _candidate(
        tmp_path,
        "admission_absent",
        overlay=0.38,
        residual=0.5,
        texture=0.40,
        face=0.22,
        face_priority=1.0,
        feature_count=14,
        painted_texels=4096,
    )
    quality_dir = admission_failed / "outputs" / "quality"
    (quality_dir / "texture_source_admission_audit.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.texture_source_admission_audit",
                "passed": False,
                "rejection_reasons": [
                    "has_unique_accepted_texture_frames",
                    "has_unique_protected_face_source_patch_frames",
                ],
                "unique_accepted_frame_count": 0,
                "protected_face_source_patch_frame_count": 0,
                "duplicate_frame_count": 13,
                "rejected_low_detail_frame_count": 4,
                "rejected_ablation_frame_count": 7,
                "top_score": 1.848806969,
            }
        ),
        encoding="utf-8",
    )

    result = write_candidate_comparison_report(
        {
            "admission_absent": admission_absent,
            "admission_failed": admission_failed,
        },
        tmp_path / "texture_admission_comparison.json",
    )

    assert result.best_candidate == "admission_absent"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    failed = next(item for item in report["candidates"] if item["label"] == "admission_failed")
    absent = next(item for item in report["candidates"] if item["label"] == "admission_absent")
    assert failed["score_terms"]["texture_source_admission_failed"] < 0.0
    assert absent["score_terms"]["texture_source_admission_failed"] == 0.0
    assert failed["metrics"]["raw_texture_source_admission_audit_present"] == 1.0
    assert failed["metrics"]["raw_texture_source_admission_audit_passed"] == 0.0
    assert failed["metrics"]["raw_texture_source_admission_rejection_count"] == 2.0
    assert failed["metrics"]["raw_texture_source_admission_rejected_ablation_frame_count"] == 7.0


def test_write_candidate_comparison_report_penalizes_face_detail_loss(tmp_path) -> None:
    drifted = _candidate(
        tmp_path,
        "drifted",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        detail_loss=0.8,
    )
    stable = _candidate(
        tmp_path,
        "stable",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        detail_loss=0.1,
    )

    result = write_candidate_comparison_report(
        {"drifted": drifted, "stable": stable},
        tmp_path / "detail_comparison.json",
    )

    assert result.best_candidate == "stable"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["candidates"][0]["metrics"]["raw_face_region_detail_loss"] == 0.1
    assert report["candidates"][1]["score_terms"]["face_region_detail_loss"] < 0.0


def test_write_candidate_comparison_report_rewards_body_texture_selection_metrics(
    tmp_path,
) -> None:
    baseline = _candidate(
        tmp_path,
        "baseline",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
    )
    body_texture = _candidate(
        tmp_path,
        "body_texture",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        body_priority_candidates=2,
        selected_body_priority_frames=2,
        selected_mean_body_fit_score=0.62,
        texture_detail_candidates=2,
        texture_only_detail_frames=2,
    )

    result = write_candidate_comparison_report(
        {"baseline": baseline, "body_texture": body_texture},
        tmp_path / "body_texture_comparison.json",
    )

    assert result.best_candidate == "body_texture"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    best_terms = report["candidates"][0]["score_terms"]
    best_metrics = report["candidates"][0]["metrics"]
    assert best_terms["geometry_selected_body_priority_frame_count"] > 0.0
    assert best_terms["geometry_selected_mean_body_fit_score"] > 0.0
    assert best_terms["texture_only_detail_frame_count"] > 0.0
    assert best_metrics["raw_geometry_selected_body_priority_frame_count"] == 2.0
    assert best_metrics["raw_texture_only_detail_frame_count"] == 2.0


def test_write_guarded_candidate_comparison_report_rejects_body_fit_only_winner(
    tmp_path,
) -> None:
    baseline = _candidate(
        tmp_path,
        "baseline",
        overlay=0.36,
        residual=0.63,
        texture=0.30,
        face=0.20,
        selected_mean_body_fit_score=0.50,
    )
    body_fit_only = _candidate(
        tmp_path,
        "body_fit_only",
        overlay=0.36,
        residual=0.63,
        texture=0.42,
        face=0.28,
        selected_frames=8,
        video_frames=6,
        selected_body_priority_frames=4,
        selected_mean_body_fit_score=0.64,
        texture_only_detail_frames=4,
    )

    score_result = write_candidate_comparison_report(
        {"baseline": baseline, "body_fit_only": body_fit_only},
        tmp_path / "score.json",
    )
    guarded_result = write_guarded_candidate_comparison_report(
        {"baseline": baseline, "body_fit_only": body_fit_only},
        tmp_path / "guarded.json",
        baseline_label="baseline",
    )

    assert score_result.best_candidate == "body_fit_only"
    assert guarded_result.best_candidate == "baseline"
    assert guarded_result.rejected_candidate_count == 1
    report = json.loads(guarded_result.report_path.read_text(encoding="utf-8"))
    rejected = next(
        item for item in report["candidates"] if item["label"] == "body_fit_only"
    )
    assert rejected["promotion_guard"]["passed"] is False
    assert "body_fit_alignment_gate_passed" in rejected["promotion_guard"][
        "rejection_reasons"
    ]
    assert rejected["body_fit_alignment_gate"]["required"] is True


def test_write_guarded_candidate_comparison_report_keeps_aligned_body_fit_winner(
    tmp_path,
) -> None:
    baseline = _candidate(
        tmp_path,
        "baseline",
        overlay=0.36,
        residual=0.63,
        texture=0.30,
        face=0.20,
        selected_mean_body_fit_score=0.50,
    )
    aligned_body_fit = _candidate(
        tmp_path,
        "aligned_body_fit",
        overlay=0.38,
        residual=0.61,
        texture=0.42,
        face=0.28,
        selected_frames=8,
        video_frames=6,
        selected_body_priority_frames=4,
        selected_mean_body_fit_score=0.64,
        texture_only_detail_frames=4,
    )

    result = write_guarded_candidate_comparison_report(
        {"baseline": baseline, "aligned_body_fit": aligned_body_fit},
        tmp_path / "guarded.json",
        baseline_label="baseline",
    )

    assert result.best_candidate == "aligned_body_fit"
    assert result.rejected_candidate_count == 0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    winner = next(
        item for item in report["candidates"] if item["label"] == "aligned_body_fit"
    )
    assert winner["promotion_guard"]["passed"] is True
    assert winner["body_fit_alignment_gate"]["passed"] is True
    assert winner["body_fit_alignment_gate"]["improved_alignment_metric_count"] == 2


def test_write_candidate_comparison_report_penalizes_iris_halo_ratio(tmp_path) -> None:
    muddy_eye = _candidate(
        tmp_path,
        "muddy_eye",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        iris_halo_max_edge_to_core_ratio=9.33,
    )
    clean_eye = _candidate(
        tmp_path,
        "clean_eye",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        iris_halo_max_edge_to_core_ratio=1.10,
    )

    result = write_candidate_comparison_report(
        {"muddy_eye": muddy_eye, "clean_eye": clean_eye},
        tmp_path / "iris_halo_comparison.json",
    )

    assert result.best_candidate == "clean_eye"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    clean_terms = report["candidates"][0]["score_terms"]
    muddy_terms = report["candidates"][1]["score_terms"]
    assert clean_terms["iris_halo_edge_to_core_ratio"] == 0.0
    assert muddy_terms["iris_halo_edge_to_core_ratio"] < 0.0
    assert report["candidates"][1]["metrics"][
        "raw_facial_feature_iris_halo_max_edge_to_core_ratio"
    ] == 9.33


def test_write_candidate_comparison_report_rewards_face_detail_geometry(tmp_path) -> None:
    flat = _candidate(
        tmp_path,
        "flat",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
    )
    detailed = _candidate(
        tmp_path,
        "detailed",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        face_detail_applied=1.0,
        deformed_vertices=180,
        mean_displacement=0.018,
        max_displacement=0.032,
        surface_anchor_count=10,
        surface_anchor_vertices=160,
        surface_anchor_max_displacement=0.024,
        turntable_visible_fraction=1.0,
        turntable_anchor_pixels=2048,
    )

    result = write_candidate_comparison_report(
        {"flat": flat, "detailed": detailed},
        tmp_path / "geometry_comparison.json",
    )

    assert result.best_candidate == "detailed"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    best_terms = report["candidates"][0]["score_terms"]
    best_metrics = report["candidates"][0]["metrics"]
    assert best_terms["face_detail_geometry_applied"] > 0.0
    assert best_terms["face_detail_deformed_vertex_count"] > 0.0
    assert best_terms["face_detail_mean_displacement"] > 0.0
    assert best_terms["face_surface_anchor_count"] > 0.0
    assert best_terms["face_surface_anchor_vertex_count"] > 0.0
    assert best_terms["face_turntable_required_anchor_visible_fraction"] > 0.0
    assert best_terms["face_turntable_front_anchor_pixel_count"] > 0.0
    assert best_metrics["raw_face_detail_geometry_applied"] == 1.0
    assert best_metrics["raw_face_surface_anchor_count"] == 10.0
    assert best_metrics["raw_face_turntable_required_anchor_visible_fraction"] == 1.0
    assert best_metrics["raw_template_region_v2_contract_valid"] == 1.0
    assert best_metrics["raw_glb_template_region_v2_contract_valid"] == 1.0
    assert best_metrics["raw_glb_texture_size"] == 256.0
    assert best_metrics["raw_glb_texture_dimensions_valid"] == 1.0
    assert best_metrics["raw_glb_pbr_texture_dimensions_valid"] == 1.0
    assert best_metrics["raw_glb_embedded_texture_dimension_gate_valid"] == 1.0
    assert best_metrics["raw_template_region_v2_texture_target_size"] == 2048.0
    assert best_metrics["raw_template_region_v2_texture_target_bake_used"] == 1.0
    assert best_metrics["raw_template_region_v2_texture_target_reported_size"] == 2048.0
    assert (
        best_metrics["raw_template_region_v2_texture_target_allocated_texel_count"]
        >= best_metrics[
            "raw_template_region_v2_texture_target_requested_texel_count"
        ]
    )


def test_write_candidate_comparison_report_rewards_textured_turntable_detail(
    tmp_path,
) -> None:
    anchor_only = _candidate(
        tmp_path,
        "anchor_only",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        turntable_visible_fraction=1.0,
        turntable_anchor_pixels=2048,
    )
    textured = _candidate(
        tmp_path,
        "textured",
        overlay=0.38,
        residual=0.5,
        texture=0.32,
        face=0.22,
        turntable_visible_fraction=1.0,
        turntable_anchor_pixels=2048,
        textured_turntable_written=3,
        textured_turntable_unique=96,
        textured_turntable_luma_std=0.09,
        textured_turntable_mouth_pixels=24,
        textured_turntable_required_fraction=0.9,
    )

    result = write_candidate_comparison_report(
        {"anchor_only": anchor_only, "textured": textured},
        tmp_path / "textured_turntable_comparison.json",
    )

    assert result.best_candidate == "textured"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    best_terms = report["candidates"][0]["score_terms"]
    best_metrics = report["candidates"][0]["metrics"]
    assert best_terms["face_turntable_textured_written_count"] > 0.0
    assert best_terms["face_turntable_front_textured_unique_rgb_count"] > 0.0
    assert best_terms["face_turntable_front_textured_luma_std"] > 0.0
    assert best_terms["face_turntable_front_textured_mouth_line_pixel_count"] > 0.0
    assert best_terms["face_turntable_front_textured_required_feature_fraction"] > 0.0
    assert best_metrics["raw_face_turntable_textured_written_count"] == 3.0
    assert best_metrics["raw_face_turntable_front_textured_unique_rgb_count"] == 96.0
    assert best_metrics["raw_face_turntable_front_textured_luma_std"] == 0.09
    assert best_metrics["raw_face_turntable_front_textured_mouth_line_pixel_count"] == 24.0
    assert (
        best_metrics["raw_face_turntable_front_textured_required_feature_fraction"]
        == 0.9
    )


def _candidate(
    tmp_path,
    label: str,
    *,
    overlay: float,
    residual: float,
    texture: float,
    face: float,
    face_priority: float = 0.0,
    feature_count: int = 0,
    painted_texels: int = 0,
    selected_frames: int = 2,
    video_frames: int = 0,
    detail_loss: float | None = None,
    face_detail_applied: float = 0.0,
    deformed_vertices: int = 0,
    max_displacement: float = 0.0,
    mean_displacement: float = 0.0,
    surface_anchor_count: int = 0,
    surface_anchor_vertices: int = 0,
    surface_anchor_max_displacement: float = 0.0,
    turntable_visible_fraction: float = 0.0,
    turntable_anchor_pixels: int = 0,
    textured_turntable_written: int = 0,
    textured_turntable_unique: int = 0,
    textured_turntable_luma_std: float = 0.0,
    textured_turntable_mouth_pixels: int = 0,
    textured_turntable_required_fraction: float = 0.0,
    face_contrast: float = 0.0,
    iris_pupil_contrast: float = 0.0,
    texture_resolution_scale: float = 1.0,
    pbr_detail_applied: float = 0.0,
    pbr_detail_texels: int = 0,
    pbr_normal_variance: float = 0.0,
    glb_face_feature_gate: float = 0.0,
    glb_face_feature_visible_groups: int = 0,
    glb_face_feature_visible_features: int = 0,
    glb_face_feature_sampled_texels: int = 0,
    glb_face_feature_iris_pupil: float = 0.0,
    glb_face_feature_eye_luma_std: float = 0.0,
    glb_face_feature_nose_muzzle: float = 0.0,
    glb_face_feature_mouth_muzzle: float = 0.0,
    iris_halo_max_edge_to_core_ratio: float = 0.0,
    source_patch_quality: float = 0.0,
    source_patch_view_weighting_used: float = 0.0,
    source_patch_view_weighted_count: int = 0,
    source_patch_mean_view_weight: float = 0.0,
    source_patch_mean_frontal_face_confidence: float = 0.0,
    source_patch_mean_view_weighted_rank: float = 0.0,
    body_priority_candidates: int = 0,
    selected_body_priority_frames: int = 0,
    selected_mean_body_fit_score: float = 0.0,
    texture_detail_candidates: int = 0,
    texture_only_detail_frames: int = 0,
    texture_guard_passed: bool | None = None,
    texture_guard_regressed_metrics: tuple[str, ...] = (),
    whisker_pad_features: int = 0,
    whisker_pad_texels: int = 0,
    whisker_pad_fraction: float = 0.0,
):
    project_dir = tmp_path / label
    quality_dir = project_dir / "outputs" / "quality"
    quality_dir.mkdir(parents=True)
    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "success": True,
                "metrics": {
                    "raw_quality_passed_check_count": 40,
                    "raw_quality_total_check_count": 40,
                    "raw_selected_frame_count": selected_frames,
                    "raw_texture_video_frame_count": video_frames,
                    "raw_geometry_body_priority_candidate_count": body_priority_candidates,
                    "raw_geometry_selected_body_priority_frame_count": (
                        selected_body_priority_frames
                    ),
                    "raw_geometry_selected_mean_body_fit_score": (
                        selected_mean_body_fit_score
                    ),
                    "raw_geometry_selected_mean_face_texture_detail_score": 0.0,
                    "raw_texture_detail_candidate_count": texture_detail_candidates,
                    "raw_texture_only_detail_frame_count": texture_only_detail_frames,
                    "overlay_mean_silhouette_iou": overlay,
                    "silhouette_residual_mean_union_error": residual,
                    "basecolor_observed_texel_fraction": texture,
                    "observed_face_fraction": face,
                    "raw_face_priority_texture_used": face_priority,
                    "raw_facial_feature_count": feature_count,
                    "raw_facial_feature_painted_texel_count": painted_texels,
                    "raw_facial_feature_whisker_pad_feature_count": (
                        whisker_pad_features
                    ),
                    "raw_facial_feature_whisker_pad_texel_count": whisker_pad_texels,
                    "raw_facial_feature_whisker_pad_texel_fraction": (
                        whisker_pad_fraction
                    ),
                    "raw_facial_feature_source_patch_count": feature_count,
                    "raw_facial_feature_source_patch_quality_total_count": feature_count,
                    "raw_facial_feature_source_patch_quality_passed_count": int(
                        feature_count * source_patch_quality
                    ),
                    "raw_facial_feature_source_patch_mean_quality_score": (
                        source_patch_quality
                    ),
                    "raw_facial_feature_source_patch_view_weighting_used": (
                        source_patch_view_weighting_used
                    ),
                    "raw_facial_feature_source_patch_view_weighted_count": (
                        source_patch_view_weighted_count
                    ),
                    "raw_facial_feature_source_patch_mean_view_weight": (
                        source_patch_mean_view_weight
                    ),
                    "raw_facial_feature_source_patch_mean_frontal_face_confidence": (
                        source_patch_mean_frontal_face_confidence
                    ),
                    "raw_facial_feature_source_patch_mean_view_weighted_rank": (
                        source_patch_mean_view_weighted_rank
                    ),
                    "raw_face_feature_min_required_contrast": face_contrast,
                    "raw_face_feature_eye_to_muzzle_luma_delta": face_contrast,
                    "raw_face_feature_mouth_to_muzzle_luma_delta": face_contrast,
                    "raw_face_feature_nose_to_muzzle_rgb_distance": face_contrast,
                    "raw_face_feature_iris_to_pupil_rgb_distance": iris_pupil_contrast,
                    "raw_face_feature_contrast_passed": float(face_contrast >= 0.18),
                    "raw_facial_feature_iris_halo_feature_count": 2,
                    "raw_facial_feature_iris_halo_edge_to_core_ratio": (
                        iris_halo_max_edge_to_core_ratio
                    ),
                    "raw_facial_feature_iris_halo_max_edge_to_core_ratio": (
                        iris_halo_max_edge_to_core_ratio
                    ),
                    "raw_facial_feature_right_iris_halo_edge_to_core_ratio": (
                        iris_halo_max_edge_to_core_ratio
                    ),
                    "raw_texture_width": 256 * texture_resolution_scale,
                    "raw_texture_height": 256 * texture_resolution_scale,
                    "raw_texture_resolution_scale": texture_resolution_scale,
                    "raw_pbr_texture_width": 256 * texture_resolution_scale,
                    "raw_pbr_texture_height": 256 * texture_resolution_scale,
                    "raw_glb_texture_size": 256 * texture_resolution_scale,
                    "raw_glb_texture_dimensions_valid": 1.0,
                    "raw_glb_pbr_texture_dimensions_valid": 1.0,
                    "raw_glb_basecolor_texture_valid": 1.0,
                    "raw_glb_pbr_textures_valid": 1.0,
                    "raw_glb_embedded_texture_dimension_gate_valid": 1.0,
                    "raw_glb_face_feature_required_group_count": 3,
                    "raw_glb_face_feature_visible_group_count": (
                        glb_face_feature_visible_groups
                    ),
                    "raw_glb_face_feature_required_feature_count": 10,
                    "raw_glb_face_feature_present_required_feature_count": (
                        glb_face_feature_visible_features
                    ),
                    "raw_glb_face_feature_visible_feature_count": (
                        glb_face_feature_visible_features
                    ),
                    "raw_glb_face_feature_sampled_texel_count": (
                        glb_face_feature_sampled_texels
                    ),
                    "raw_glb_face_feature_min_visible_texel_count": (
                        1 if glb_face_feature_visible_features else 0
                    ),
                    "raw_glb_face_feature_mean_rgb_error": 0.0,
                    "raw_glb_face_feature_max_rgb_error": 0.0,
                    "raw_glb_face_feature_eye_visible": float(
                        glb_face_feature_visible_groups >= 1
                    ),
                    "raw_glb_face_feature_nose_visible": float(
                        glb_face_feature_visible_groups >= 2
                    ),
                    "raw_glb_face_feature_muzzle_visible": float(
                        glb_face_feature_visible_groups >= 3
                    ),
                    "raw_glb_face_feature_iris_to_pupil_rgb_distance": (
                        glb_face_feature_iris_pupil
                    ),
                    "raw_glb_face_feature_eye_detail_luma_std": (
                        glb_face_feature_eye_luma_std
                    ),
                    "raw_glb_face_feature_nose_to_muzzle_rgb_distance": (
                        glb_face_feature_nose_muzzle
                    ),
                    "raw_glb_face_feature_mouth_to_muzzle_rgb_distance": (
                        glb_face_feature_mouth_muzzle
                    ),
                    "raw_glb_face_feature_material_visibility_gate_valid": (
                        glb_face_feature_gate
                    ),
                    "raw_pbr_face_material_detail_applied": pbr_detail_applied,
                    "raw_pbr_face_material_detail_feature_count": feature_count,
                    "raw_pbr_face_material_detail_texel_count": pbr_detail_texels,
                    "raw_pbr_face_material_detail_normal_variance": pbr_normal_variance,
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
                    "raw_face_detail_geometry_applied": face_detail_applied,
                    "raw_face_detail_deformed_vertex_count": deformed_vertices,
                    "raw_face_detail_max_displacement": max_displacement,
                    "raw_face_detail_mean_displacement": mean_displacement,
                    "raw_face_surface_anchor_count": surface_anchor_count,
                    "raw_face_surface_anchor_vertex_count": surface_anchor_vertices,
                    "raw_face_surface_anchor_max_displacement": (
                        surface_anchor_max_displacement
                    ),
                    "raw_face_turntable_view_count": 3,
                    "raw_face_turntable_front_visible_anchor_count": int(
                        4 * turntable_visible_fraction
                    ),
                    "raw_face_turntable_front_anchor_pixel_count": turntable_anchor_pixels,
                    "raw_face_turntable_required_anchor_visible_fraction": (
                        turntable_visible_fraction
                    ),
                    "raw_face_turntable_textured_view_count": 3,
                    "raw_face_turntable_textured_written_count": (
                        textured_turntable_written
                    ),
                    "raw_face_turntable_front_textured_unique_rgb_count": (
                        textured_turntable_unique
                    ),
                    "raw_face_turntable_front_textured_luma_std": (
                        textured_turntable_luma_std
                    ),
                    "raw_face_turntable_front_textured_mouth_line_pixel_count": (
                        textured_turntable_mouth_pixels
                    ),
                    "raw_face_turntable_front_textured_required_feature_visible_count": int(
                        12 * textured_turntable_required_fraction
                    ),
                    "raw_face_turntable_front_textured_required_feature_fraction": (
                        textured_turntable_required_fraction
                    ),
                    "raw_face_turntable_mean_textured_unique_rgb_count": (
                        textured_turntable_unique
                    ),
                    "raw_face_turntable_mean_textured_luma_std": (
                        textured_turntable_luma_std
                    ),
                    "joint_projection_mean_keypoint_loss": 0.08,
                    **(
                        {"raw_face_region_detail_loss": detail_loss}
                        if detail_loss is not None
                        else {}
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    if texture_guard_passed is not None:
        (quality_dir / "texture_only_promotion_guard.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "source": "opencat3d.texture_only_promotion_guard",
                    "passed": texture_guard_passed,
                    "regressed_metric_names": list(texture_guard_regressed_metrics),
                    "strict_regressed_metric_names": list(texture_guard_regressed_metrics),
                    "rejection_reasons": []
                    if texture_guard_passed
                    else ["strict_face_detail_metrics_not_regressed"],
                }
            ),
            encoding="utf-8",
        )
    return project_dir
