import json

from opencat3d.evaluation.face_texture_bake_comparison import (
    evaluate_face_texture_bake_comparison_report,
)


def test_evaluate_face_texture_bake_comparison_report_writes_region_deltas(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    texture_dir = project_dir / "outputs" / "textures"
    texture_dir.mkdir(parents=True)
    (texture_dir / "texture_manifest.json").write_text(
        json.dumps(
            {
                "facial_features": {
                    "source_patch_bake_selected_candidate": "feedback",
                    "source_patch_bake_candidates": [
                        {
                            "label": "generic",
                            "selected": False,
                            "region_count": 2,
                            "visual_quality_passed_count": 2,
                            "visual_quality_min_score": 0.920578,
                            "visual_quality_mean_score": 0.960289,
                            "min_confident_texel_fraction": 0.1197,
                            "mean_source_luma_std": 0.014,
                            "feedback_used_count": 0,
                            "regions": [
                                {
                                    "name": "left_eye",
                                    "score": 1.0,
                                    "passed": True,
                                    "confident_texel_fraction": 0.401,
                                    "source_luma_std": 0.016,
                                    "unique_rgb_count": 900,
                                    "weakest_component": "confident_texel_score",
                                },
                                {
                                    "name": "left_ear_inner",
                                    "score": 0.920578,
                                    "passed": True,
                                    "confident_texel_fraction": 0.1197,
                                    "source_luma_std": 0.012,
                                    "unique_rgb_count": 250,
                                    "weakest_component": "source_luma_score",
                                },
                            ],
                        },
                        {
                            "label": "feedback",
                            "selected": True,
                            "region_count": 2,
                            "visual_quality_passed_count": 2,
                            "visual_quality_min_score": 1.0,
                            "visual_quality_mean_score": 1.0,
                            "min_confident_texel_fraction": 0.452,
                            "mean_source_luma_std": 0.021,
                            "feedback_used_count": 13,
                            "regions": [
                                {
                                    "name": "left_eye",
                                    "score": 1.0,
                                    "passed": True,
                                    "confident_texel_fraction": 0.401,
                                    "source_luma_std": 0.016,
                                    "unique_rgb_count": 900,
                                    "weakest_component": "confident_texel_score",
                                },
                                {
                                    "name": "left_ear_inner",
                                    "score": 1.0,
                                    "passed": True,
                                    "confident_texel_fraction": 0.452,
                                    "source_luma_std": 0.021,
                                    "unique_rgb_count": 1004,
                                    "weakest_component": "confident_texel_score",
                                },
                            ],
                        },
                    ],
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_face_texture_bake_comparison_report(project_dir, mode="raw")

    assert result.success
    assert result.report_path.name == "raw_face_texture_bake_comparison.json"
    assert all(result.checks.values())
    assert result.metrics["candidate_count"] == 2
    assert result.metrics["compared_region_count"] == 2
    assert result.metrics["selected_feedback"] == 1.0
    assert result.metrics["score_improved_region_count"] == 1
    assert result.metrics["coverage_improved_region_count"] == 1
    assert result.metrics["source_detail_improved_region_count"] == 1
    assert result.metrics["max_score_delta"] == 0.079422
    assert result.metrics["max_coverage_delta"] == 0.3323
    assert result.metrics["selected_candidate_not_lower_rank"] == 1.0

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["source"] == "opencat3d.face_texture_bake_comparison_v1"
    assert report["baseline_candidate"] == "generic"
    assert report["selected_candidate"] == "feedback"
    assert report["output_contract"] == {
        "target_representation": "metadata_only_quality_artifact",
        "texture_changed": False,
        "topology_changed": False,
    }
    left_ear = next(region for region in report["regions"] if region["name"] == "left_ear_inner")
    assert left_ear["score_delta"] == 0.079422
    assert left_ear["decision_driver"] == "score"
    assert left_ear["selected_unique_rgb_count"] == 1004
    assert report["artifacts"]["texture_manifest"].endswith(
        "outputs/textures/texture_manifest.json"
    )
