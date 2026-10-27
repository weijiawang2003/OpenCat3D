import json
from pathlib import Path

from opencat3d.evaluation.texture_probe_regression import (
    write_texture_probe_regression_audit,
)


def test_write_texture_probe_regression_audit_classifies_global_face_smoothing(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    probe = tmp_path / "probe"
    _write_project(
        baseline,
        luma_std={
            "left_eye": 0.11,
            "right_eye": 0.16,
            "nose": 0.13,
            "muzzle": 0.12,
            "left_ear_inner": 0.08,
            "right_ear_inner": 0.08,
        },
        basecolor_observed=0.41,
    )
    _write_project(
        probe,
        luma_std={
            "left_eye": 0.10,
            "right_eye": 0.15,
            "nose": 0.12,
            "muzzle": 0.11,
            "left_ear_inner": 0.07,
            "right_ear_inner": 0.07,
        },
        basecolor_observed=0.42,
        guard={
            "passed": False,
            "rejection_reasons": [
                "detail_metrics_not_regressed",
                "strict_face_detail_metrics_not_regressed",
            ],
            "strict_regressed_metric_names": [
                "raw_face_turntable_front_textured_luma_std",
                "raw_glb_face_feature_eye_detail_luma_std",
                "raw_glb_face_feature_iris_to_pupil_rgb_distance",
            ],
            "regressed_metric_names": [
                "raw_face_turntable_front_textured_luma_std",
            ],
            "improved_metric_count": 1,
            "detail_metric_comparisons": [
                {
                    "baseline": 0.41,
                    "candidate": 0.42,
                    "delta": 0.01,
                    "metric": "basecolor_observed_texel_fraction",
                    "status": "improved",
                },
                {
                    "baseline": 0.08,
                    "candidate": 0.07,
                    "delta": -0.01,
                    "metric": "raw_face_turntable_front_textured_luma_std",
                    "status": "regressed",
                },
            ],
        },
        bake={
            "probe_added_texture_frame_count": 1,
            "probe_bypassed_rejected_source_frame_id_count": 1,
            "probe_added_texture_frames": [
                {
                    "source_frame_id": "frame_0016",
                    "sha256": "abc",
                    "preflight_score": 2.0,
                }
            ],
        },
    )

    result = write_texture_probe_regression_audit(
        baseline_project_dir=baseline,
        probe_project_dirs=[probe],
        output_path=tmp_path / "audit.json",
        force=True,
    )

    assert result.success is True
    assert result.metrics["probe_count"] == 1
    assert result.metrics["strict_regressed_probe_count"] == 1
    assert result.metrics["all_face_region_luma_regressed_probe_count"] == 1
    assert result.metrics["coverage_improved_probe_count"] == 1
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    probe_report = report["probes"][0]
    assert probe_report["failure_mode"] == (
        "global_face_texture_smoothing_over_existing_source_patches"
    )
    assert probe_report["probe_source_frame_ids"] == ["frame_0016"]
    assert len(report["recommendations"]) == 3
    assert report["strategy"]["continue_texture_source_frame_addition"] is False


def _write_project(
    project_dir: Path,
    *,
    luma_std: dict[str, float],
    basecolor_observed: float,
    guard: dict | None = None,
    bake: dict | None = None,
) -> None:
    quality_dir = project_dir / "outputs" / "quality"
    quality_dir.mkdir(parents=True)
    (quality_dir / "raw_pipeline_summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "basecolor_observed_texel_fraction": basecolor_observed,
                    "raw_face_texture_closeup_luma_std": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    (quality_dir / "raw_face_texture_diagnostics.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "name": name,
                        "texture_luma_std": value,
                        "texture_unique_rgb_count": 100,
                        "texture_visual_quality_score": 1.0,
                        "source_frame_id": f"{name}_source",
                        "source_crop_exact_patch_used": True,
                    }
                    for name, value in luma_std.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    if guard is not None:
        (quality_dir / "texture_only_promotion_guard.json").write_text(
            json.dumps(guard),
            encoding="utf-8",
        )
    if bake is not None:
        (quality_dir / "baseline_geometry_texture_bake.json").write_text(
            json.dumps(bake),
            encoding="utf-8",
        )
