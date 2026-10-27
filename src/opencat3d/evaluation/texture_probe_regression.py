"""Audit rejected texture-only probes against protected face-detail metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STRICT_FACE_DETAIL_METRICS: tuple[str, ...] = (
    "raw_face_turntable_front_textured_luma_std",
    "raw_glb_face_feature_eye_detail_luma_std",
    "raw_glb_face_feature_iris_to_pupil_rgb_distance",
)


@dataclass(frozen=True)
class TextureProbeRegressionAuditResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]
    recommendation_count: int


def write_texture_probe_regression_audit(
    *,
    baseline_project_dir: Path,
    probe_project_dirs: list[Path],
    output_path: Path,
    force: bool = False,
) -> TextureProbeRegressionAuditResult:
    """Write an evidence audit for texture probes rejected by face-detail guards."""

    baseline_project_dir = Path(baseline_project_dir).expanduser().resolve()
    probe_project_dirs = [Path(path).expanduser().resolve() for path in probe_project_dirs]
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True")

    baseline_summary_path = (
        baseline_project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
    )
    baseline_diagnostics_path = (
        baseline_project_dir / "outputs" / "quality" / "raw_face_texture_diagnostics.json"
    )
    baseline_summary = _read_json(baseline_summary_path)
    baseline_diagnostics = _read_json(baseline_diagnostics_path)
    baseline_regions = _regions_by_name(baseline_diagnostics)

    probe_reports = [
        _probe_report(
            baseline_regions=baseline_regions,
            probe_project_dir=probe_project_dir,
        )
        for probe_project_dir in probe_project_dirs
    ]
    metrics = _audit_metrics(probe_reports)
    recommendations = _recommendations(metrics)
    checks = {
        "baseline_project_exists": baseline_project_dir.is_dir(),
        "baseline_summary_exists": baseline_summary_path.is_file(),
        "baseline_face_texture_diagnostics_exists": baseline_diagnostics_path.is_file(),
        "baseline_regions_present": bool(baseline_regions),
        "probe_project_count_positive": bool(probe_project_dirs),
        "probe_reports_loaded": all(
            bool(report["checks"]["probe_project_exists"])
            for report in probe_reports
        ),
        "probe_guards_loaded": all(
            bool(report["checks"]["promotion_guard_exists"])
            for report in probe_reports
        ),
        "probe_bakes_loaded": all(
            bool(report["checks"]["bake_report_exists"]) for report in probe_reports
        ),
        "probe_face_diagnostics_loaded": all(
            bool(report["checks"]["face_texture_diagnostics_exists"])
            for report in probe_reports
        ),
    }
    success = all(checks.values())
    report = {
        "schema_version": "texture_probe_regression_audit_v1",
        "source": "opencat3d.evaluation.texture_probe_regression",
        "success": success,
        "checks": checks,
        "baseline_project_dir": str(baseline_project_dir),
        "baseline_summary_path": str(baseline_summary_path),
        "baseline_face_texture_diagnostics_path": str(baseline_diagnostics_path),
        "strict_face_detail_metrics": list(STRICT_FACE_DETAIL_METRICS),
        "metrics": metrics,
        "probes": probe_reports,
        "recommendations": recommendations,
        "strategy": {
            "continue_texture_source_frame_addition": False,
            "reason": (
                "all_loaded_probes_regressed_strict_face_detail_metrics"
                if metrics["strict_regressed_probe_count"] == len(probe_reports)
                else "some_probes_need_manual_review"
            ),
            "next_strategy": (
                "face_region_protected_weighting_or_template_uv_detail"
                if metrics["strict_regressed_probe_count"] > 0
                else "no_strategy_change_required"
            ),
        },
        "artifacts": {
            "baseline_summary": str(baseline_summary_path),
            "baseline_face_texture_diagnostics": str(baseline_diagnostics_path),
            "baseline_raw_pipeline_summary_metrics": baseline_summary.get("metrics", {}),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return TextureProbeRegressionAuditResult(
        report_path=output_path,
        success=success,
        checks=checks,
        metrics=metrics,
        recommendation_count=len(recommendations),
    )


def _probe_report(
    *,
    baseline_regions: dict[str, dict[str, Any]],
    probe_project_dir: Path,
) -> dict[str, Any]:
    quality_dir = probe_project_dir / "outputs" / "quality"
    bake_path = quality_dir / "baseline_geometry_texture_bake.json"
    guard_path = quality_dir / "texture_only_promotion_guard.json"
    summary_path = quality_dir / "raw_pipeline_summary.json"
    diagnostics_path = quality_dir / "raw_face_texture_diagnostics.json"
    bake = _read_optional_json(bake_path)
    guard = _read_optional_json(guard_path)
    summary = _read_optional_json(summary_path)
    diagnostics = _read_optional_json(diagnostics_path)
    probe_regions = _regions_by_name(diagnostics)
    region_deltas = _region_deltas(baseline_regions, probe_regions)
    added_frames = bake.get("probe_added_texture_frames", [])
    if not isinstance(added_frames, list):
        added_frames = []
    strict_regressed = [
        str(metric)
        for metric in guard.get("strict_regressed_metric_names", [])
        if str(metric)
    ]
    metric_deltas = _metric_deltas(guard)
    luma_regressed_count = sum(
        1 for region in region_deltas if float(region["texture_luma_std_delta"]) < 0.0
    )
    same_source_patch_count = sum(
        1 for region in region_deltas if bool(region["same_exact_source_patch"])
    )
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    return {
        "project_dir": str(probe_project_dir),
        "checks": {
            "probe_project_exists": probe_project_dir.is_dir(),
            "bake_report_exists": bake_path.is_file(),
            "promotion_guard_exists": guard_path.is_file(),
            "raw_pipeline_summary_exists": summary_path.is_file(),
            "face_texture_diagnostics_exists": diagnostics_path.is_file(),
        },
        "probe_source_frame_ids": [
            str(frame.get("source_frame_id", ""))
            for frame in added_frames
            if isinstance(frame, dict)
        ],
        "probe_added_texture_frame_count": int(
            _numeric(bake.get("probe_added_texture_frame_count"))
        ),
        "probe_bypassed_rejected_source_frame_id_count": int(
            _numeric(bake.get("probe_bypassed_rejected_source_frame_id_count"))
        ),
        "promotion_guard_passed": bool(guard.get("passed", False)),
        "promotion_guard_rejection_reasons": list(
            guard.get("rejection_reasons", [])
            if isinstance(guard.get("rejection_reasons"), list)
            else []
        ),
        "strict_regressed_metric_names": strict_regressed,
        "strict_regressed_metric_count": int(len(strict_regressed)),
        "regressed_metric_names": list(
            guard.get("regressed_metric_names", [])
            if isinstance(guard.get("regressed_metric_names"), list)
            else []
        ),
        "improved_metric_count": int(_numeric(guard.get("improved_metric_count"))),
        "basecolor_observed_texel_fraction": _numeric(
            metrics.get("basecolor_observed_texel_fraction")
        ),
        "raw_face_texture_closeup_luma_std": _numeric(
            metrics.get("raw_face_texture_closeup_luma_std")
        ),
        "metric_deltas": metric_deltas,
        "region_deltas": region_deltas,
        "face_region_luma_regressed_count": int(luma_regressed_count),
        "same_exact_source_patch_region_count": int(same_source_patch_count),
        "failure_mode": _failure_mode(
            strict_regressed_count=len(strict_regressed),
            region_count=len(region_deltas),
            luma_regressed_count=luma_regressed_count,
            same_source_patch_count=same_source_patch_count,
        ),
        "artifacts": {
            "bake_report": str(bake_path),
            "promotion_guard": str(guard_path),
            "raw_pipeline_summary": str(summary_path),
            "face_texture_diagnostics": str(diagnostics_path),
        },
    }


def _metric_deltas(guard: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = guard.get("detail_metric_comparisons", [])
    if not isinstance(comparisons, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in comparisons:
        if not isinstance(raw, dict):
            continue
        metric = str(raw.get("metric", ""))
        if not metric:
            continue
        result.append(
            {
                "metric": metric,
                "baseline": _numeric(raw.get("baseline")),
                "candidate": _numeric(raw.get("candidate")),
                "delta": _numeric(raw.get("delta")),
                "status": str(raw.get("status", "")),
                "strict_face_detail_metric": metric in STRICT_FACE_DETAIL_METRICS,
            }
        )
    return result


def _region_deltas(
    baseline_regions: dict[str, dict[str, Any]],
    probe_regions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for name in sorted(set(baseline_regions) & set(probe_regions)):
        baseline = baseline_regions[name]
        probe = probe_regions[name]
        reports.append(
            {
                "name": name,
                "baseline_texture_luma_std": _numeric(
                    baseline.get("texture_luma_std")
                ),
                "probe_texture_luma_std": _numeric(probe.get("texture_luma_std")),
                "texture_luma_std_delta": _numeric(probe.get("texture_luma_std"))
                - _numeric(baseline.get("texture_luma_std")),
                "baseline_texture_unique_rgb_count": int(
                    _numeric(baseline.get("texture_unique_rgb_count"))
                ),
                "probe_texture_unique_rgb_count": int(
                    _numeric(probe.get("texture_unique_rgb_count"))
                ),
                "texture_unique_rgb_delta": int(
                    _numeric(probe.get("texture_unique_rgb_count"))
                    - _numeric(baseline.get("texture_unique_rgb_count"))
                ),
                "baseline_visual_quality_score": _numeric(
                    baseline.get("texture_visual_quality_score")
                ),
                "probe_visual_quality_score": _numeric(
                    probe.get("texture_visual_quality_score")
                ),
                "visual_quality_score_delta": _numeric(
                    probe.get("texture_visual_quality_score")
                )
                - _numeric(baseline.get("texture_visual_quality_score")),
                "baseline_source_frame_id": str(baseline.get("source_frame_id", "")),
                "probe_source_frame_id": str(probe.get("source_frame_id", "")),
                "same_exact_source_patch": bool(
                    baseline.get("source_crop_exact_patch_used", False)
                    and probe.get("source_crop_exact_patch_used", False)
                    and str(baseline.get("source_frame_id", ""))
                    == str(probe.get("source_frame_id", ""))
                ),
            }
        )
    return reports


def _regions_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    regions = payload.get("regions", [])
    if not isinstance(regions, list):
        return {}
    return {
        str(region.get("name", "")): dict(region)
        for region in regions
        if isinstance(region, dict) and str(region.get("name", ""))
    }


def _audit_metrics(probe_reports: list[dict[str, Any]]) -> dict[str, float | int]:
    probe_count = len(probe_reports)
    strict_regressed_probe_count = sum(
        1 for report in probe_reports if int(report["strict_regressed_metric_count"]) > 0
    )
    promoted_probe_count = sum(
        1 for report in probe_reports if bool(report["promotion_guard_passed"])
    )
    all_region_luma_regressed_probe_count = sum(
        1
        for report in probe_reports
        if int(report["face_region_luma_regressed_count"])
        == len(report["region_deltas"])
        and len(report["region_deltas"]) > 0
    )
    same_patch_all_region_probe_count = sum(
        1
        for report in probe_reports
        if int(report["same_exact_source_patch_region_count"])
        == len(report["region_deltas"])
        and len(report["region_deltas"]) > 0
    )
    coverage_improved_probe_count = sum(
        1
        for report in probe_reports
        if any(
            item["metric"] == "basecolor_observed_texel_fraction"
            and float(item["delta"]) > 0.0
            for item in report["metric_deltas"]
        )
    )
    return {
        "probe_count": int(probe_count),
        "promoted_probe_count": int(promoted_probe_count),
        "rejected_probe_count": int(probe_count - promoted_probe_count),
        "strict_regressed_probe_count": int(strict_regressed_probe_count),
        "all_face_region_luma_regressed_probe_count": int(
            all_region_luma_regressed_probe_count
        ),
        "all_face_region_same_exact_source_patch_probe_count": int(
            same_patch_all_region_probe_count
        ),
        "coverage_improved_probe_count": int(coverage_improved_probe_count),
        "stop_same_texture_source_probe_fraction": float(
            strict_regressed_probe_count / max(probe_count, 1)
        ),
    }


def _failure_mode(
    *,
    strict_regressed_count: int,
    region_count: int,
    luma_regressed_count: int,
    same_source_patch_count: int,
) -> str:
    if (
        strict_regressed_count > 0
        and region_count > 0
        and luma_regressed_count == region_count
        and same_source_patch_count == region_count
    ):
        return "global_face_texture_smoothing_over_existing_source_patches"
    if strict_regressed_count > 0:
        return "strict_face_detail_regression"
    return "no_strict_face_detail_regression"


def _recommendations(metrics: dict[str, float | int]) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    if int(metrics["strict_regressed_probe_count"]) > 0:
        recommendations.append(
            {
                "id": "stop_same_texture_source_frame_addition",
                "reason": "all accepted task249 video probes regressed strict face detail",
                "next_action": "do not spend more runs adding frames from this texture source",
            }
        )
        recommendations.append(
            {
                "id": "protect_face_regions_before_supplemental_texture_fusion",
                "reason": (
                    "strict face metrics decreased even when exact face source patches "
                    "stayed selected"
                ),
                "next_action": "prefer face-region protected weighting or a UV/detail-map change",
            }
        )
    if int(metrics["coverage_improved_probe_count"]) > 0:
        recommendations.append(
            {
                "id": "separate_body_coverage_from_face_detail",
                "reason": "some probes improved basecolor coverage while still hurting face detail",
                "next_action": (
                    "allow supplemental frames to influence body/unknown texels without "
                    "overwriting protected face texels"
                ),
            }
        )
    return recommendations


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return _read_json(path)
    except json.JSONDecodeError:
        return {}


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
