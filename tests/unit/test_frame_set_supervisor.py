from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from opencat3d.evaluation.frame_set_supervisor import (
    FRAME_SET_SUPERVISOR_VERSION,
    write_alignment_gated_frame_set_report,
    write_body_fit_alignment_gate_report,
    write_frame_set_candidate_report,
    write_texture_only_promotion_guard,
)


def test_write_frame_set_candidate_report_is_report_only(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    selection_path = project_dir / "observations" / "raw" / "frame_selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_report = _selection_report()
    selection_path.write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = selection_path.read_text(encoding="utf-8")

    result = write_frame_set_candidate_report(project_dir)

    assert selection_path.read_text(encoding="utf-8") == before
    assert result.report_path == project_dir / "outputs" / "quality" / "frame_set_candidates.json"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["version"] == FRAME_SET_SUPERVISOR_VERSION
    assert report["best_candidate"] == result.best_candidate
    assert report["candidate_count"] == result.candidate_count
    assert report["candidate_count"] >= 3

    labels = {candidate["label"] for candidate in report["candidates"]}
    assert "current_selection" in labels
    assert "score_only_geometry" in labels
    assert "current_geometry_expanded_texture" in labels

    scores = [candidate["score"] for candidate in report["candidates"]]
    assert scores == sorted(scores, reverse=True)
    current = next(
        candidate for candidate in report["candidates"] if candidate["label"] == "current_selection"
    )
    assert current["geometry_frame_ids"] == selection_report["selected_frame_ids"]
    assert current["metrics"]["geometry_photo_frame_count"] == 1.0
    assert current["metrics"]["geometry_video_frame_count"] == 2.0

    expanded = next(
        candidate
        for candidate in report["candidates"]
        if candidate["label"] == "current_geometry_expanded_texture"
    )
    assert "photo_body" in expanded["texture_frame_ids"]
    assert expanded["metrics"]["texture_face_detail_fraction"] > 0.0


def test_write_frame_set_candidate_report_requires_selection_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_frame_set_candidate_report(tmp_path)


def test_write_frame_set_candidate_report_adds_view_diverse_body_geometry(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    selection_path = project_dir / "observations" / "raw" / "frame_selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_report = {
        "schema_version": "1.0",
        "project_id": "demo_project",
        "min_frames": 2,
        "max_selected_frames": 3,
        "selected_frame_ids": ["center_a", "center_b"],
        "texture_frame_ids": ["center_a"],
        "frames": [
            _frame(
                "center_a",
                media_type="video",
                source_index=0,
                score=0.10,
                bbox_center=(0.50, 0.50),
                bbox_aspect=1.0,
            ),
            _frame(
                "center_b",
                media_type="video",
                source_index=1,
                score=0.11,
                bbox_center=(0.52, 0.50),
                bbox_aspect=1.0,
            ),
            _frame(
                "left_body",
                media_type="video",
                source_index=2,
                score=0.12,
                bbox_center=(0.25, 0.50),
                bbox_aspect=1.6,
            ),
            _frame(
                "right_body",
                media_type="video",
                source_index=3,
                score=0.13,
                bbox_center=(0.78, 0.50),
                bbox_aspect=1.55,
            ),
        ],
    }
    selection_path.write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = write_frame_set_candidate_report(project_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    view_diverse = next(
        item for item in report["candidates"] if item["label"] == "view_diverse_body_geometry"
    )
    assert view_diverse["geometry_frame_ids"] == [
        "center_a",
        "left_body",
        "right_body",
    ]
    assert view_diverse["metrics"]["geometry_view_bucket_count"] == 3.0
    assert view_diverse["metrics"]["geometry_view_spread_fraction"] <= 1.0
    assert view_diverse["metrics"]["geometry_horizontal_center_span"] > 0.5
    assert view_diverse["score_terms"]["view_spread"] > 0.0


def test_write_frame_set_candidate_report_adds_full_body_upload_geometry(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    selection_path = project_dir / "observations" / "raw" / "frame_selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_report = {
        "schema_version": "1.0",
        "project_id": "demo_project",
        "min_frames": 2,
        "max_selected_frames": 4,
        "selected_frame_ids": ["video_front", "video_side"],
        "texture_frame_ids": ["video_front", "photo_closeup"],
        "frames": [
            _frame(
                "photo_full_front",
                media_type="photo",
                source_index=0,
                score=0.13,
                mask_area_ratio=0.16,
                bbox_area_ratio=0.28,
            ),
            _frame(
                "photo_full_side",
                media_type="photo",
                source_index=1,
                score=0.14,
                mask_area_ratio=0.15,
                bbox_area_ratio=0.44,
                bbox_aspect=0.55,
                touches_image_border=1.0,
            ),
            _frame(
                "photo_closeup",
                media_type="photo",
                source_index=2,
                score=0.12,
                mask_area_ratio=0.30,
                bbox_area_ratio=0.70,
                touches_image_border=1.0,
            ),
            _frame("video_front", media_type="video", source_index=3, score=0.15),
            _frame(
                "video_side",
                media_type="video",
                source_index=4,
                score=0.16,
                bbox_center=(0.72, 0.50),
                bbox_aspect=1.55,
            ),
        ],
    }
    selection_path.write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = write_frame_set_candidate_report(project_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    full_body = next(
        item for item in report["candidates"] if item["label"] == "full_body_upload_geometry"
    )
    assert "photo_full_front" in full_body["geometry_frame_ids"]
    assert "photo_full_side" in full_body["geometry_frame_ids"]
    assert "photo_closeup" not in full_body["geometry_frame_ids"]
    assert full_body["metrics"]["geometry_full_body_photo_frame_count"] == 2.0
    assert full_body["metrics"]["texture_full_body_photo_frame_count"] == 2.0
    assert full_body["score_terms"]["full_body_photo_evidence"] > 0.0


def test_write_frame_set_candidate_report_prefilters_alignment_risk(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    selection_path = project_dir / "observations" / "raw" / "frame_selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_report = {
        "schema_version": "1.0",
        "project_id": "demo_project",
        "min_frames": 2,
        "max_selected_frames": 3,
        "selected_frame_ids": [
            "border_wide",
            "border_narrow",
            "center_body",
        ],
        "texture_frame_ids": ["border_wide", "border_narrow", "center_body"],
        "frames": [
            _frame(
                "border_wide",
                media_type="video",
                source_index=0,
                score=0.05,
                touches_image_border=1.0,
                bbox_center=(0.50, 0.50),
                bbox_aspect=2.0,
            ),
            _frame(
                "border_narrow",
                media_type="video",
                source_index=1,
                score=0.06,
                touches_image_border=1.0,
                bbox_center=(0.52, 0.50),
                bbox_aspect=0.5,
            ),
            _frame(
                "center_body",
                media_type="video",
                source_index=2,
                score=0.10,
                bbox_center=(0.50, 0.50),
                bbox_aspect=1.0,
            ),
            _frame(
                "left_body",
                media_type="video",
                source_index=3,
                score=0.11,
                bbox_center=(0.28, 0.50),
                bbox_aspect=1.05,
            ),
            _frame(
                "right_body",
                media_type="video",
                source_index=4,
                score=0.12,
                bbox_center=(0.74, 0.50),
                bbox_aspect=0.95,
            ),
        ],
    }
    selection_path.write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = write_frame_set_candidate_report(project_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    low_border = next(
        item for item in report["candidates"] if item["label"] == "low_border_body_geometry"
    )
    assert low_border["alignment_risk_prefilter"]["passed"] is False
    assert (
        "border_profile_not_overcorrected"
        in low_border["alignment_risk_prefilter"]["rejection_reasons"]
    )
    assert "aspect_span_preserved" in low_border["alignment_risk_prefilter"]["rejection_reasons"]
    assert low_border["score_terms"]["alignment_risk_prefilter"] < 0.0
    assert report["best_candidate"] != "low_border_body_geometry"
    assert report["alignment_risk_prefiltered_candidate_count"] >= 1


def test_write_frame_set_candidate_report_uses_projection_residuals(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    selection_path = project_dir / "observations" / "raw" / "frame_selection.json"
    residual_path = project_dir / "outputs" / "quality" / "raw_silhouette_residuals.json"
    selection_path.parent.mkdir(parents=True)
    residual_path.parent.mkdir(parents=True)
    selection_report = {
        "schema_version": "1.0",
        "project_id": "demo_project",
        "min_frames": 2,
        "max_selected_frames": 2,
        "selected_frame_ids": ["good_fit", "bad_fit"],
        "texture_frame_ids": ["good_fit", "bad_fit"],
        "frames": [
            _frame("good_fit", media_type="video", source_index=0, score=0.10),
            _frame("bad_fit", media_type="video", source_index=1, score=0.11),
            _frame("unknown_a", media_type="video", source_index=2, score=0.09),
            _frame("unknown_b", media_type="video", source_index=3, score=0.12),
        ],
    }
    selection_path.write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    residual_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.silhouette_residuals",
                "frames": [
                    {
                        "frame_id": "good_fit",
                        "union_error_fraction": 0.25,
                        "silhouette_iou": 0.75,
                    },
                    {
                        "frame_id": "bad_fit",
                        "union_error_fraction": 0.85,
                        "silhouette_iou": 0.15,
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = write_frame_set_candidate_report(project_dir)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["projection_residual_report"] == str(residual_path)
    assert report["projection_residual_frame_count"] == 2
    current = next(
        candidate for candidate in report["candidates"] if candidate["label"] == "current_selection"
    )
    assert current["metrics"]["geometry_projection_residual_frame_count"] == 2.0
    assert current["metrics"]["geometry_projection_residual_coverage_fraction"] == 1.0
    assert current["metrics"]["mean_known_projection_union_error_fraction"] == 0.55
    assert current["metrics"]["max_known_projection_union_error_fraction"] == 0.85
    assert current["score_terms"]["projection_residual_coverage"] > 0.0
    assert current["score_terms"]["projection_residual_consistency"] > 0.0
    assert current["score_terms"]["projection_residual_risk"] == 0.0

    score_only = next(
        candidate
        for candidate in report["candidates"]
        if candidate["label"] == "score_only_geometry"
    )
    assert score_only["metrics"]["geometry_projection_residual_frame_count"] == 1.0
    assert score_only["metrics"]["geometry_unknown_projection_residual_fraction"] == 0.5


def test_write_frame_set_candidate_report_maps_reference_residuals_by_image_hash(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "reference"
    target_dir = tmp_path / "target"
    _write_observation_bytes(reference_dir, "ref_seen", b"same pixels")
    _write_observation_bytes(reference_dir, "ref_other", b"other pixels")
    _write_observation_bytes(target_dir, "target_seen", b"same pixels")
    _write_observation_bytes(target_dir, "target_new", b"new pixels")
    _write_selection_report(
        target_dir,
        ["target_seen", "target_new"],
        [
            _frame("target_seen", media_type="video", source_index=0, score=0.10),
            _frame("target_new", media_type="video", source_index=1, score=0.11),
        ],
    )
    residual_path = reference_dir / "outputs" / "quality" / "raw_silhouette_residuals.json"
    residual_path.parent.mkdir(parents=True)
    residual_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frames": [
                    {
                        "frame_id": "ref_seen",
                        "union_error_fraction": 0.40,
                        "silhouette_iou": 0.60,
                    },
                    {
                        "frame_id": "ref_other",
                        "union_error_fraction": 0.90,
                        "silhouette_iou": 0.10,
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = write_frame_set_candidate_report(
        target_dir,
        reference_project_dir=reference_dir,
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    current = next(
        candidate for candidate in report["candidates"] if candidate["label"] == "current_selection"
    )
    assert report["projection_residual_report"] == str(residual_path)
    assert report["projection_residual_reference_project"] == str(reference_dir)
    assert report["projection_residual_frame_count"] == 1
    assert current["metrics"]["geometry_projection_residual_frame_count"] == 1.0
    assert current["metrics"]["geometry_projection_residual_coverage_fraction"] == 0.5
    assert current["metrics"]["mean_known_projection_union_error_fraction"] == 0.4
    assert current["metrics"]["mean_known_projection_silhouette_iou"] == 0.6


def test_write_alignment_gated_frame_set_report_rejects_alignment_regression(
    tmp_path: Path,
) -> None:
    baseline = _pipeline_project(
        tmp_path,
        "baseline",
        overlay=0.36,
        residual=0.63,
        keypoint=0.08,
        observed=0.43,
        texture_frames=31,
    )
    drifted = _pipeline_project(
        tmp_path,
        "drifted",
        overlay=0.24,
        residual=0.75,
        keypoint=0.14,
        observed=0.45,
        texture_frames=34,
        selected_ids=["new_body", "new_side", "video_a", "video_b"],
    )
    texture_only = _pipeline_project(
        tmp_path,
        "texture_only",
        overlay=0.36,
        residual=0.63,
        keypoint=0.08,
        observed=0.431,
        texture_frames=33,
    )

    result = write_alignment_gated_frame_set_report(
        baseline,
        {
            "drifted_full_body_geometry": drifted,
            "texture_only_new_media": texture_only,
        },
        tmp_path / "alignment_gate.json",
    )

    assert result.best_candidate == "texture_only_new_media"
    assert result.rejected_candidate_count == 1
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    drifted_candidate = next(
        item for item in report["candidates"] if item["label"] == "drifted_full_body_geometry"
    )
    assert drifted_candidate["alignment_gate"]["passed"] is False
    assert "overlay_not_lower" in drifted_candidate["alignment_gate"]["rejection_reasons"]
    assert (
        "silhouette_residual_not_higher" in drifted_candidate["alignment_gate"]["rejection_reasons"]
    )
    texture_candidate = next(
        item for item in report["candidates"] if item["label"] == "texture_only_new_media"
    )
    assert texture_candidate["alignment_gate"]["passed"] is True
    assert (
        texture_candidate["score"]
        > next(item for item in report["candidates"] if item["label"] == "baseline")["score"]
    )


def test_write_alignment_gated_frame_set_report_falls_back_to_baseline(
    tmp_path: Path,
) -> None:
    baseline = _pipeline_project(
        tmp_path,
        "baseline",
        overlay=0.36,
        residual=0.63,
        keypoint=0.08,
        observed=0.43,
        texture_frames=31,
    )
    drifted = _pipeline_project(
        tmp_path,
        "drifted",
        overlay=0.35,
        residual=0.64,
        keypoint=0.08,
        observed=0.45,
        texture_frames=34,
    )

    result = write_alignment_gated_frame_set_report(
        baseline,
        {"drifted": drifted},
        tmp_path / "alignment_gate.json",
    )

    assert result.best_candidate == "baseline"
    assert result.rejected_candidate_count == 1


def test_write_alignment_gated_frame_set_report_applies_body_fit_gate(
    tmp_path: Path,
) -> None:
    baseline = _pipeline_project(
        tmp_path,
        "baseline",
        overlay=0.36,
        residual=0.63,
        keypoint=0.08,
        observed=0.43,
        texture_frames=31,
        mean_body_fit=0.55,
    )
    body_fit_only = _pipeline_project(
        tmp_path,
        "body_fit_only",
        overlay=0.36,
        residual=0.63,
        keypoint=0.08,
        observed=0.45,
        texture_frames=34,
        selected_ids=["better_body", "side_body"],
        mean_body_fit=0.61,
    )

    result = write_alignment_gated_frame_set_report(
        baseline,
        {"body_fit_only": body_fit_only},
        tmp_path / "alignment_gate.json",
    )

    assert result.best_candidate == "baseline"
    assert result.rejected_candidate_count == 1
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    candidate = next(item for item in report["candidates"] if item["label"] == "body_fit_only")
    assert candidate["alignment_gate"]["passed"] is True
    assert candidate["body_fit_alignment_gate"]["required"] is True
    assert candidate["body_fit_alignment_gate"]["passed"] is False
    assert "alignment_metric_improved" in candidate["body_fit_alignment_gate"]["rejection_reasons"]


def test_write_alignment_gated_frame_set_report_accepts_aligned_body_fit_gain(
    tmp_path: Path,
) -> None:
    baseline = _pipeline_project(
        tmp_path,
        "baseline",
        overlay=0.36,
        residual=0.63,
        keypoint=0.08,
        observed=0.43,
        texture_frames=31,
        mean_body_fit=0.55,
    )
    aligned_body_fit = _pipeline_project(
        tmp_path,
        "aligned_body_fit",
        overlay=0.37,
        residual=0.62,
        keypoint=0.08,
        observed=0.44,
        texture_frames=31,
        selected_ids=["better_body", "side_body"],
        mean_body_fit=0.61,
    )

    result = write_alignment_gated_frame_set_report(
        baseline,
        {"aligned_body_fit": aligned_body_fit},
        tmp_path / "alignment_gate.json",
    )

    assert result.best_candidate == "aligned_body_fit"
    assert result.rejected_candidate_count == 0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    candidate = next(item for item in report["candidates"] if item["label"] == "aligned_body_fit")
    assert candidate["body_fit_alignment_gate"]["required"] is True
    assert candidate["body_fit_alignment_gate"]["passed"] is True
    assert candidate["body_fit_alignment_gate"]["improved_alignment_metric_count"] == 2


def test_write_body_fit_alignment_gate_report_accepts_aligned_body_fit_gain(
    tmp_path: Path,
) -> None:
    baseline = _pipeline_project(
        tmp_path,
        "baseline",
        overlay=0.36,
        residual=0.63,
        keypoint=0.08,
        observed=0.43,
        texture_frames=31,
        mean_body_fit=0.55,
    )
    candidate = _pipeline_project(
        tmp_path,
        "candidate",
        overlay=0.37,
        residual=0.62,
        keypoint=0.08,
        observed=0.43,
        texture_frames=31,
        selected_ids=["body_front", "body_side"],
        mean_body_fit=0.61,
    )

    result = write_body_fit_alignment_gate_report(
        baseline,
        candidate,
        tmp_path / "body_fit_gate.json",
    )

    assert result.passed is True
    assert result.rejection_reasons == ()
    assert result.body_fit_delta > 0.0
    assert result.improved_alignment_metric_count == 2
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["promotion_recommendation"] == "eligible_body_fit_frame_change"
    assert report["checks"]["body_fit_improved"] is True
    assert report["checks"]["alignment_metric_improved"] is True


def test_write_body_fit_alignment_gate_report_rejects_task190_style_regression(
    tmp_path: Path,
) -> None:
    baseline = _pipeline_project(
        tmp_path,
        "baseline",
        overlay=0.25150424471654087,
        residual=0.7484957552834591,
        keypoint=0.1473209660021366,
        observed=0.41205334663391113,
        texture_frames=63,
        mean_body_fit=0.5491777794285715,
    )
    candidate = _pipeline_project(
        tmp_path,
        "bodyfit_candidate",
        overlay=0.22129628863332362,
        residual=0.7787037113666763,
        keypoint=0.1610707594020039,
        observed=0.38594698905944824,
        texture_frames=22,
        selected_ids=[
            "frame_0000",
            "frame_0003",
            "frame_0009",
            "frame_0010",
            "frame_0011",
            "frame_0026",
            "frame_0028",
        ],
        mean_body_fit=0.604305496857143,
    )

    result = write_body_fit_alignment_gate_report(
        baseline,
        candidate,
        tmp_path / "body_fit_gate.json",
    )

    assert result.passed is False
    assert "alignment_not_regressed" in result.rejection_reasons
    assert "alignment_metric_improved" in result.rejection_reasons
    assert result.body_fit_delta > 0.0
    assert result.improved_alignment_metric_count == 0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["checks"]["body_fit_improved"] is True
    assert report["alignment_gate"]["checks"]["overlay_not_lower"] is False
    assert report["promotion_recommendation"] == "keep_baseline_or_try_smaller_frame_change"


def test_write_texture_only_promotion_guard_accepts_preserved_geometry_gain(
    tmp_path: Path,
) -> None:
    baseline = _texture_guard_project(
        tmp_path,
        "baseline",
        selected=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        texture=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        basecolor=0.43,
        face_fraction=0.32,
        painted_texels=100,
    )
    candidate = _texture_guard_project(
        tmp_path,
        "candidate",
        selected=[("new_a", b"geometry-a"), ("new_b", b"geometry-b")],
        texture=[
            ("new_a", b"geometry-a"),
            ("new_b", b"geometry-b"),
            ("face_closeup", b"new-face-texture"),
        ],
        basecolor=0.44,
        face_fraction=0.32,
        painted_texels=100,
    )

    result = write_texture_only_promotion_guard(
        baseline,
        candidate,
        tmp_path / "texture_guard.json",
    )

    assert result.passed is True
    assert result.rejection_reasons == ()
    assert result.improved_metric_count >= 1
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["checks"]["geometry_hashes_preserved"] is True
    assert report["texture"]["added_unique_texture_hash_count"] == 1


def test_write_texture_only_promotion_guard_hashes_copied_local_observations(
    tmp_path: Path,
) -> None:
    baseline = _texture_guard_project(
        tmp_path,
        "baseline",
        selected=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        texture=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        basecolor=0.43,
        face_fraction=0.32,
        painted_texels=100,
    )
    candidate = _texture_guard_project(
        tmp_path,
        "candidate",
        selected=[("new_a", b"geometry-a"), ("new_b", b"geometry-b")],
        texture=[
            ("new_a", b"geometry-a"),
            ("new_b", b"geometry-b"),
            ("face_closeup", b"new-face-texture"),
        ],
        basecolor=0.44,
        face_fraction=0.32,
        painted_texels=100,
    )
    for project_dir in (baseline, candidate):
        _rewrite_observation_paths_to_stale_absolute(project_dir)
        frame_paths = [
            (
                frame_id,
                Path("/missing/copied/source") / frame_id / "image_srgb.npz",
            )
            for frame_id in _all_observation_frame_ids(project_dir)
        ]
        _write_texture_manifest_for_guard(project_dir, frame_paths)

    result = write_texture_only_promotion_guard(
        baseline,
        candidate,
        tmp_path / "texture_guard.json",
    )

    assert result.passed is True
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["checks"]["geometry_hashes_preserved"] is True
    assert report["checks"]["texture_evidence_preserved"] is True
    assert report["geometry"]["missing_baseline_hash_frame_ids"] == []
    assert report["geometry"]["missing_candidate_hash_frame_ids"] == []
    assert report["texture"]["missing_baseline_hash_frame_ids"] == []
    assert report["texture"]["missing_candidate_hash_frame_ids"] == []


def test_write_texture_only_promotion_guard_accepts_baked_seed_texture_evidence(
    tmp_path: Path,
) -> None:
    baseline = _texture_guard_project(
        tmp_path,
        "baseline",
        selected=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        texture=[
            ("body_a", b"geometry-a"),
            ("body_b", b"geometry-b"),
            ("old_face", b"baseline-face-texture"),
        ],
        basecolor=0.43,
        face_fraction=0.32,
        painted_texels=100,
    )
    candidate = _texture_guard_project(
        tmp_path,
        "candidate",
        selected=[("new_a", b"geometry-a"), ("new_b", b"geometry-b")],
        texture=[
            ("new_a", b"geometry-a"),
            ("new_b", b"geometry-b"),
            ("face_closeup", b"new-face-texture"),
        ],
        basecolor=0.44,
        face_fraction=0.32,
        painted_texels=110,
    )
    _write_baked_basecolor_seed_for_guard(baseline, candidate)

    result = write_texture_only_promotion_guard(
        baseline,
        candidate,
        tmp_path / "texture_guard.json",
    )

    assert result.passed is True
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["checks"]["texture_evidence_preserved"] is True
    assert report["texture"]["frame_hashes_preserved"] is False
    seed_evidence = report["texture"]["baseline_baked_seed_evidence"]
    assert seed_evidence["preserved"] is True
    assert seed_evidence["artifact_hashes_preserved"] is True
    assert seed_evidence["baseline_texture_hashes_preserved_by_seed_count"] == 3
    assert seed_evidence["seed_observed_texel_count"] == 16


def test_write_texture_only_promotion_guard_rejects_geometry_drift(
    tmp_path: Path,
) -> None:
    baseline = _texture_guard_project(
        tmp_path,
        "baseline",
        selected=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        texture=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        basecolor=0.43,
    )
    candidate = _texture_guard_project(
        tmp_path,
        "candidate",
        selected=[("body_a", b"geometry-a"), ("different", b"changed-geometry")],
        texture=[
            ("body_a", b"geometry-a"),
            ("different", b"changed-geometry"),
            ("face_closeup", b"new-face-texture"),
        ],
        basecolor=0.44,
    )

    result = write_texture_only_promotion_guard(
        baseline,
        candidate,
        tmp_path / "texture_guard.json",
    )

    assert result.passed is False
    assert "geometry_hashes_preserved" in result.rejection_reasons


def test_write_texture_only_promotion_guard_separates_summary_failure_from_alignment(
    tmp_path: Path,
) -> None:
    baseline = _texture_guard_project(
        tmp_path,
        "baseline",
        selected=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        texture=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        basecolor=0.43,
    )
    candidate = _texture_guard_project(
        tmp_path,
        "candidate",
        selected=[("new_a", b"geometry-a"), ("new_b", b"geometry-b")],
        texture=[
            ("new_a", b"geometry-a"),
            ("new_b", b"geometry-b"),
            ("face_closeup", b"new-face-texture"),
        ],
        basecolor=0.44,
        success=False,
    )

    result = write_texture_only_promotion_guard(
        baseline,
        candidate,
        tmp_path / "texture_guard.json",
    )

    assert result.passed is False
    assert result.rejection_reasons == ("candidate_success",)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["checks"]["alignment_not_regressed"] is True
    assert report["alignment_gate"]["summary_success_required"] is False
    assert report["alignment_gate"]["observed_summary_success"] is False


def test_write_texture_only_promotion_guard_accepts_texture_only_evaluation_success(
    tmp_path: Path,
) -> None:
    baseline = _texture_guard_project(
        tmp_path,
        "baseline",
        selected=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        texture=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        basecolor=0.43,
        face_fraction=0.32,
        painted_texels=100,
    )
    candidate = _texture_guard_project(
        tmp_path,
        "candidate",
        selected=[("new_a", b"geometry-a"), ("new_b", b"geometry-b")],
        texture=[
            ("new_a", b"geometry-a"),
            ("new_b", b"geometry-b"),
            ("face_closeup", b"new-face-texture"),
        ],
        basecolor=0.44,
        face_fraction=0.32,
        painted_texels=100,
        success=False,
        summary_source="opencat3d.baseline_geometry_texture_bake",
        extra_metrics={"raw_texture_only_evaluation_success": 1.0},
    )

    result = write_texture_only_promotion_guard(
        baseline,
        candidate,
        tmp_path / "texture_guard.json",
    )

    assert result.passed is True
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["checks"]["candidate_success"] is True
    assert (
        report["candidate_success_policy"]["source"]
        == "raw_texture_only_evaluation_success"
    )
    assert report["candidate_success_policy"]["summary_success"] is False


def test_write_texture_only_promotion_guard_rejects_detail_regression(
    tmp_path: Path,
) -> None:
    baseline = _texture_guard_project(
        tmp_path,
        "baseline",
        selected=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        texture=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        basecolor=0.43,
        face_fraction=0.32,
        painted_texels=100,
    )
    candidate = _texture_guard_project(
        tmp_path,
        "candidate",
        selected=[("new_a", b"geometry-a"), ("new_b", b"geometry-b")],
        texture=[
            ("new_a", b"geometry-a"),
            ("new_b", b"geometry-b"),
            ("face_closeup", b"new-face-texture"),
        ],
        basecolor=0.44,
        face_fraction=0.31,
        painted_texels=90,
    )

    result = write_texture_only_promotion_guard(
        baseline,
        candidate,
        tmp_path / "texture_guard.json",
    )

    assert result.passed is False
    assert "detail_metrics_not_regressed" in result.rejection_reasons
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert "observed_face_fraction" in report["regressed_metric_names"]
    assert "raw_facial_feature_painted_texel_count" in report["regressed_metric_names"]


def test_write_texture_only_promotion_guard_rejects_supplemental_eye_regression(
    tmp_path: Path,
) -> None:
    baseline = _texture_guard_project(
        tmp_path,
        "baseline",
        selected=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        texture=[("body_a", b"geometry-a"), ("body_b", b"geometry-b")],
        basecolor=0.43,
        extra_metrics={
            "raw_glb_face_feature_eye_detail_luma_std": 0.21,
            "raw_glb_face_feature_iris_to_pupil_rgb_distance": 0.25,
        },
    )
    candidate = _texture_guard_project(
        tmp_path,
        "candidate",
        selected=[("new_a", b"geometry-a"), ("new_b", b"geometry-b")],
        texture=[
            ("new_a", b"geometry-a"),
            ("new_b", b"geometry-b"),
            ("face_closeup", b"new-face-texture"),
        ],
        basecolor=0.44,
        extra_metrics={
            "raw_glb_face_feature_eye_detail_luma_std": 0.17,
            "raw_glb_face_feature_iris_to_pupil_rgb_distance": 0.19,
        },
    )
    supplemental_path = tmp_path / "external_face_closeup.npz"
    supplemental_path.write_bytes(b"new-face-texture")
    _write_texture_manifest_for_guard(
        candidate,
        [
            ("new_a", candidate / "observations" / "raw" / "new_a" / "image_srgb.npz"),
            ("new_b", candidate / "observations" / "raw" / "new_b" / "image_srgb.npz"),
            ("face_closeup", supplemental_path),
        ],
    )
    shutil.rmtree(candidate / "observations" / "raw" / "face_closeup")
    bake_path = candidate / "outputs" / "quality" / "baseline_geometry_texture_bake.json"
    bake_path.write_text(
        json.dumps(
            {
                "added_texture_frames": [
                    {
                        "frame_id": "face_closeup",
                        "source_frame_id": "source_face",
                        "sha256": "sha-face",
                        "media_type": "photo",
                        "preflight_rank": 1,
                        "preflight_score": 1.25,
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = write_texture_only_promotion_guard(
        baseline,
        candidate,
        tmp_path / "texture_guard.json",
    )

    assert result.passed is False
    assert "strict_face_detail_metrics_not_regressed" in result.rejection_reasons
    assert "texture_evidence_preserved" not in result.rejection_reasons
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["texture"]["added_unique_texture_hash_count"] == 1
    assert report["texture"]["missing_candidate_hash_frame_ids"] == []
    assert report["strict_regressed_metric_names"] == [
        "raw_glb_face_feature_eye_detail_luma_std",
        "raw_glb_face_feature_iris_to_pupil_rgb_distance",
    ]
    assert report["supplemental_source_rejection_count"] == 1
    rejection = report["supplemental_source_rejections"][0]
    assert rejection["frame_id"] == "face_closeup"
    assert rejection["source_frame_id"] == "source_face"
    assert rejection["reason"] == "strict_face_detail_metric_regression"


def _selection_report() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "project_id": "demo_project",
        "min_frames": 2,
        "max_selected_frames": 5,
        "selected_frame_ids": ["photo_body", "video_front", "video_side"],
        "texture_frame_ids": ["photo_closeup", "video_front"],
        "frames": [
            _frame("photo_body", media_type="photo", source_index=0, score=0.12),
            _frame(
                "photo_closeup",
                media_type="photo",
                source_index=1,
                score=0.45,
                mask_area_ratio=0.30,
                bbox_area_ratio=0.62,
                touches_image_border=1.0,
                rejected_by_quality=True,
                reject_reasons=["bbox_too_large"],
            ),
            _frame(
                "photo_detail",
                media_type="photo",
                source_index=2,
                score=0.30,
                mask_area_ratio=0.22,
                bbox_area_ratio=0.42,
                touches_image_border=1.0,
            ),
            _frame("video_front", media_type="video", source_index=3, score=0.14),
            _frame(
                "video_side",
                media_type="video",
                source_index=4,
                score=0.16,
                mask_area_ratio=0.20,
                bbox_area_ratio=0.36,
            ),
            _frame(
                "video_border",
                media_type="video",
                source_index=5,
                score=0.26,
                mask_area_ratio=0.34,
                bbox_area_ratio=0.58,
                touches_image_border=1.0,
            ),
            _frame(
                "tiny_rejected",
                media_type="video",
                source_index=6,
                score=0.80,
                mask_area_ratio=0.01,
                bbox_area_ratio=0.02,
                rejected_by_quality=True,
                reject_reasons=["mask_too_small", "bbox_too_small"],
            ),
        ],
    }


def _write_selection_report(
    project_dir: Path,
    selected_ids: list[str],
    frames: list[dict[str, Any]],
) -> None:
    selection_path = project_dir / "observations" / "raw" / "frame_selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_id": project_dir.name,
                "min_frames": 2,
                "max_selected_frames": 2,
                "selected_frame_ids": selected_ids,
                "texture_frame_ids": selected_ids,
                "frames": frames,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_observation_bytes(project_dir: Path, frame_id: str, payload: bytes) -> None:
    frame_dir = project_dir / "observations" / "raw" / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    image_path = frame_dir / "image_srgb.npz"
    image_path.write_bytes(payload)
    (frame_dir / "observation.json").write_text(
        json.dumps(
            {
                "frame_id": frame_id,
                "image_srgb_path": str(image_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _texture_guard_project(
    tmp_path: Path,
    label: str,
    *,
    selected: list[tuple[str, bytes]],
    texture: list[tuple[str, bytes]],
    basecolor: float,
    face_fraction: float = 0.32,
    observed_vertex: float = 0.42,
    painted_texels: int = 100,
    overlay: float = 0.36,
    residual: float = 0.63,
    keypoint: float = 0.08,
    success: bool = True,
    summary_source: str | None = None,
    extra_metrics: dict[str, float | int] | None = None,
) -> Path:
    project_dir = tmp_path / label
    quality_dir = project_dir / "outputs" / "quality"
    selection_dir = project_dir / "observations" / "raw"
    quality_dir.mkdir(parents=True)
    selection_dir.mkdir(parents=True)
    payloads = {frame_id: payload for frame_id, payload in [*selected, *texture]}
    for frame_id, payload in payloads.items():
        _write_observation_bytes(project_dir, frame_id, payload)
    metrics = {
        "raw_quality_passed_check_count": 74,
        "raw_quality_total_check_count": 74,
        "overlay_mean_silhouette_iou": overlay,
        "silhouette_residual_mean_union_error": residual,
        "joint_projection_mean_keypoint_loss": keypoint,
        "basecolor_observed_texel_fraction": basecolor,
        "observed_face_fraction": face_fraction,
        "observed_vertex_fraction": observed_vertex,
        "raw_facial_feature_painted_texel_count": painted_texels,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "success": success,
                **({"source": summary_source} if summary_source is not None else {}),
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    (selection_dir / "frame_selection.json").write_text(
        json.dumps(
            {
                "selected_frame_ids": [frame_id for frame_id, _payload in selected],
                "texture_frame_ids": [frame_id for frame_id, _payload in texture],
            }
        ),
        encoding="utf-8",
    )
    return project_dir


def _write_texture_manifest_for_guard(
    project_dir: Path,
    frames: list[tuple[str, Path]],
) -> None:
    manifest_path = project_dir / "observations" / "raw" / "texture_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_id": frame_id, "image_srgb_path": str(path)}
                    for frame_id, path in frames
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _rewrite_observation_paths_to_stale_absolute(project_dir: Path) -> None:
    for observation_path in sorted(
        (project_dir / "observations" / "raw").glob("*/observation.json")
    ):
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        frame_id = str(observation.get("frame_id", observation_path.parent.name))
        observation["image_srgb_path"] = str(
            Path("/missing/copied/source") / frame_id / "image_srgb.npz"
        )
        observation_path.write_text(
            json.dumps(observation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _all_observation_frame_ids(project_dir: Path) -> list[str]:
    return [
        path.parent.name
        for path in sorted((project_dir / "observations" / "raw").glob("*/observation.json"))
    ]


def _write_baked_basecolor_seed_for_guard(
    baseline_project_dir: Path,
    candidate_project_dir: Path,
) -> None:
    baseline_texture_dir = baseline_project_dir / "outputs" / "textures"
    candidate_texture_dir = candidate_project_dir / "outputs" / "textures"
    baseline_texture_dir.mkdir(parents=True)
    candidate_texture_dir.mkdir(parents=True)
    for name, payload in (
        ("basecolor.png", b"baseline-basecolor"),
        ("basecolor_confidence.png", b"baseline-confidence"),
        ("source_view_id.png", b"baseline-source-view"),
    ):
        (baseline_texture_dir / name).write_bytes(payload)
    (candidate_texture_dir / "texture_manifest.json").write_text(
        json.dumps(
            {
                "basecolor": {
                    "baked_basecolor_seed": {
                        "applied": True,
                        "basecolor_path": str(baseline_texture_dir / "basecolor.png"),
                        "confidence_path": str(baseline_texture_dir / "basecolor_confidence.png"),
                        "source_view_id_path": str(baseline_texture_dir / "source_view_id.png"),
                        "seed_observed_texel_count": 16,
                        "seed_texture_dir": str(baseline_texture_dir),
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _frame(
    frame_id: str,
    *,
    media_type: str,
    source_index: int,
    score: float,
    mask_area_ratio: float = 0.18,
    bbox_area_ratio: float = 0.34,
    touches_image_border: float = 0.0,
    rejected_by_quality: bool = False,
    reject_reasons: list[str] | None = None,
    bbox_center: tuple[float, float] = (0.5, 0.5),
    bbox_aspect: float = 1.0,
) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "media_type": media_type,
        "source_index": source_index,
        "score": score,
        "rejected_by_quality": rejected_by_quality,
        "reject_reasons": reject_reasons or [],
        "mask_area_ratio": mask_area_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "touches_image_border": touches_image_border,
        "bbox_center": list(bbox_center),
        "bbox_aspect": bbox_aspect,
    }


def _pipeline_project(
    tmp_path: Path,
    label: str,
    *,
    overlay: float,
    residual: float,
    keypoint: float,
    observed: float,
    texture_frames: int,
    selected_ids: list[str] | None = None,
    mean_body_fit: float = 0.0,
) -> Path:
    project_dir = tmp_path / label
    quality_dir = project_dir / "outputs" / "quality"
    selection_dir = project_dir / "observations" / "raw"
    quality_dir.mkdir(parents=True)
    selection_dir.mkdir(parents=True)
    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "success": True,
                "metrics": {
                    "raw_quality_passed_check_count": 74,
                    "raw_quality_total_check_count": 74,
                    "overlay_mean_silhouette_iou": overlay,
                    "silhouette_residual_mean_union_error": residual,
                    "joint_projection_mean_keypoint_loss": keypoint,
                    "basecolor_observed_texel_fraction": observed,
                    "observed_face_fraction": 0.32,
                    "raw_texture_frame_count": texture_frames,
                    "raw_face_turntable_front_textured_mouth_line_pixel_count": 8,
                    "raw_geometry_selected_mean_body_fit_score": mean_body_fit,
                },
            }
        ),
        encoding="utf-8",
    )
    selected = selected_ids or ["video_a", "video_b", "photo_body"]
    texture = [*selected, "photo_face"]
    (selection_dir / "frame_selection.json").write_text(
        json.dumps(
            {
                "selected_frame_ids": selected,
                "texture_frame_ids": texture,
            }
        ),
        encoding="utf-8",
    )
    return project_dir
