import json

from opencat3d.evaluation.evidence_gap import (
    write_source_frame_acquisition_gap_report,
)


def test_source_frame_acquisition_gap_report_recommends_new_views(tmp_path) -> None:
    project_dir = tmp_path / "project"
    quality_dir = project_dir / "outputs" / "quality"
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
    media_preflight_path = tmp_path / "raw_media_preflight_report.json"
    media_preflight_path.write_text(
        json.dumps(
            {
                "face_texture_frame_count": 2,
                "geometry_safe_frame_count": 8,
                "texture_usable_frame_count": 18,
                "view_group_counts": {
                    "frontal_face": 1,
                    "oblique_body": 4,
                    "side_body": 0,
                    "vertical_body": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    media_batch_path = tmp_path / "media_preflight_report.json"
    media_batch_path.write_text(
        json.dumps(
            {
                "duplicate_frame_count": 31,
                "new_duplicate_frame_count": 31,
                "new_unique_frame_count": 0,
                "staged_upload_frame_count": 31,
                "upload_dir": str(tmp_path / "preflight_uploads"),
            }
        ),
        encoding="utf-8",
    )
    texture_probe_path = tmp_path / "texture_probe_loop_report.json"
    texture_probe_path.write_text(
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

    result = write_source_frame_acquisition_gap_report(
        project_dir,
        media_preflight_report_path=media_preflight_path,
        media_batch_report_path=media_batch_path,
        texture_probe_report_path=texture_probe_path,
    )

    assert result.success
    assert result.recommendation_count == 4
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["stop_condition"] == {
        "reason": "insufficient_new_unique_evidence",
        "run_reconstruction_now": False,
    }
    assert {item["id"] for item in report["recommendations"]} == {
        "geometry_alignment_views",
        "new_unique_evidence_required",
        "preflight_gate_before_reconstruction",
        "texture_coverage_views",
    }
    view_gaps = {item["view_group"]: item["missing"] for item in report["view_gaps"]}
    assert view_gaps["frontal_face"] == 2
    assert view_gaps["side_body"] == 2
    assert view_gaps["rear_body"] == 1
    assert report["metrics"]["texture_probe_added_frame_count"] == 0


def test_source_frame_acquisition_gap_report_allows_new_unique_batch(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    quality_dir = project_dir / "outputs" / "quality"
    quality_dir.mkdir(parents=True)
    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "basecolor_observed_texel_fraction": 0.62,
                    "joint_projection_mean_keypoint_loss": 0.07,
                    "observed_face_fraction": 0.55,
                    "observed_vertex_fraction": 0.7,
                    "overlay_mean_silhouette_iou": 0.48,
                    "silhouette_residual_mean_union_error": 0.52,
                }
            }
        ),
        encoding="utf-8",
    )
    media_batch_path = tmp_path / "media_preflight_report.json"
    media_batch_path.write_text(
        json.dumps(
            {
                "new_duplicate_frame_count": 1,
                "new_unique_frame_count": 8,
                "staged_upload_frame_count": 9,
            }
        ),
        encoding="utf-8",
    )

    result = write_source_frame_acquisition_gap_report(
        project_dir,
        media_batch_report_path=media_batch_path,
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["stop_condition"] == {
        "reason": "new_unique_evidence_available",
        "run_reconstruction_now": True,
    }
