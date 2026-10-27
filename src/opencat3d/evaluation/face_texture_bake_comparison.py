"""Face texture bake candidate comparison reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FaceTextureBakeComparisonResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


def evaluate_face_texture_bake_comparison_report(
    project_dir: Path,
    *,
    mode: str = "raw",
    force: bool = False,
) -> FaceTextureBakeComparisonResult:
    """Write a deterministic generic-vs-selected face bake comparison report."""

    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported face texture bake comparison mode: {mode}")

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / f"{mode}_face_texture_bake_comparison.json"
    texture_manifest_path = project_dir / "outputs" / "textures" / "texture_manifest.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    manifest = _read_json(texture_manifest_path)
    facial_features = _object(manifest.get("facial_features"))
    candidates = [
        candidate
        for candidate in _list(facial_features.get("source_patch_bake_candidates"))
        if isinstance(candidate, dict)
    ]
    selected_label = str(
        facial_features.get("source_patch_bake_selected_candidate", "none")
    )
    baseline = _candidate_by_label(candidates, "generic") or (candidates[0] if candidates else {})
    selected = (
        _candidate_by_label(candidates, selected_label)
        or next((candidate for candidate in candidates if candidate.get("selected")), {})
        or (candidates[-1] if candidates else {})
    )
    baseline_label = str(baseline.get("label", "none"))
    selected_label = str(selected.get("label", selected_label))
    baseline_regions = _regions_by_name(baseline)
    selected_regions = _regions_by_name(selected)
    compared_region_names = [
        name
        for name in [
            *selected_regions.keys(),
            *baseline_regions.keys(),
        ]
        if name in selected_regions and name in baseline_regions
    ]
    compared_region_names = list(dict.fromkeys(compared_region_names))
    region_comparisons = [
        _region_comparison(
            name,
            baseline_regions[name],
            selected_regions[name],
        )
        for name in compared_region_names
    ]

    score_deltas = [float(region["score_delta"]) for region in region_comparisons]
    coverage_deltas = [
        float(region["confident_texel_fraction_delta"])
        for region in region_comparisons
    ]
    source_detail_deltas = [
        float(region["source_luma_std_delta"]) for region in region_comparisons
    ]
    selected_rank = _candidate_rank(selected)
    baseline_rank = _candidate_rank(baseline)
    metrics: dict[str, float | int] = {
        "candidate_count": int(len(candidates)),
        "compared_region_count": int(len(region_comparisons)),
        "selected_feedback": float(selected_label == "feedback"),
        "score_improved_region_count": int(
            sum(1 for value in score_deltas if value > 0.001)
        ),
        "score_regressed_region_count": int(
            sum(1 for value in score_deltas if value < -0.001)
        ),
        "coverage_improved_region_count": int(
            sum(1 for value in coverage_deltas if value > 0.001)
        ),
        "source_detail_improved_region_count": int(
            sum(1 for value in source_detail_deltas if value > 0.001)
        ),
        "max_score_delta": round(max(score_deltas, default=0.0), 6),
        "min_score_delta": round(min(score_deltas, default=0.0), 6),
        "max_coverage_delta": round(max(coverage_deltas, default=0.0), 6),
        "max_source_detail_delta": round(max(source_detail_deltas, default=0.0), 6),
        "selected_candidate_not_lower_rank": float(selected_rank >= baseline_rank),
    }
    checks = {
        "texture_manifest_exists": texture_manifest_path.exists(),
        "facial_features_present": bool(facial_features),
        "bake_candidates_present": len(candidates) >= 1,
        "comparison_candidates_present": len(candidates) >= 1,
        "baseline_candidate_present": bool(baseline),
        "selected_candidate_present": bool(selected),
        "compared_regions_present": bool(region_comparisons),
        "selected_candidate_not_lower_rank": selected_rank >= baseline_rank,
    }
    success = all(checks.values())
    report = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "source": "opencat3d.face_texture_bake_comparison_v1",
        "success": success,
        "checks": checks,
        "metrics": metrics,
        "baseline_candidate": baseline_label,
        "selected_candidate": selected_label,
        "selected_feedback": selected_label == "feedback",
        "candidates": [_candidate_summary(candidate) for candidate in candidates],
        "regions": region_comparisons,
        "output_contract": {
            "topology_changed": False,
            "texture_changed": False,
            "target_representation": "metadata_only_quality_artifact",
        },
        "artifacts": {
            "texture_manifest": str(texture_manifest_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return FaceTextureBakeComparisonResult(
        report_path=report_path,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, object]:
    return {
        "label": str(candidate.get("label", "")),
        "selected": bool(candidate.get("selected", False)),
        "region_count": int(candidate.get("region_count", 0)),
        "visual_quality_passed_count": int(
            candidate.get("visual_quality_passed_count", 0)
        ),
        "visual_quality_min_score": round(
            _numeric(candidate.get("visual_quality_min_score")),
            6,
        ),
        "visual_quality_mean_score": round(
            _numeric(candidate.get("visual_quality_mean_score")),
            6,
        ),
        "min_confident_texel_fraction": round(
            _numeric(candidate.get("min_confident_texel_fraction")),
            6,
        ),
        "mean_source_luma_std": round(
            _numeric(candidate.get("mean_source_luma_std")),
            6,
        ),
        "feedback_used_count": int(candidate.get("feedback_used_count", 0)),
    }


def _region_comparison(
    name: str,
    baseline: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, object]:
    baseline_score = _numeric(baseline.get("score"))
    selected_score = _numeric(selected.get("score"))
    baseline_coverage = _numeric(baseline.get("confident_texel_fraction"))
    selected_coverage = _numeric(selected.get("confident_texel_fraction"))
    baseline_source_detail = _numeric(baseline.get("source_luma_std"))
    selected_source_detail = _numeric(selected.get("source_luma_std"))
    score_delta = selected_score - baseline_score
    coverage_delta = selected_coverage - baseline_coverage
    source_detail_delta = selected_source_detail - baseline_source_detail
    return {
        "name": name,
        "baseline_score": round(baseline_score, 6),
        "selected_score": round(selected_score, 6),
        "score_delta": round(score_delta, 6),
        "baseline_confident_texel_fraction": round(baseline_coverage, 6),
        "selected_confident_texel_fraction": round(selected_coverage, 6),
        "confident_texel_fraction_delta": round(coverage_delta, 6),
        "baseline_source_luma_std": round(baseline_source_detail, 6),
        "selected_source_luma_std": round(selected_source_detail, 6),
        "source_luma_std_delta": round(source_detail_delta, 6),
        "baseline_unique_rgb_count": int(baseline.get("unique_rgb_count", 0)),
        "selected_unique_rgb_count": int(selected.get("unique_rgb_count", 0)),
        "baseline_weakest_component": str(baseline.get("weakest_component", "")),
        "selected_weakest_component": str(selected.get("weakest_component", "")),
        "baseline_passed": bool(baseline.get("passed", False)),
        "selected_passed": bool(selected.get("passed", False)),
        "decision_driver": _decision_driver(
            score_delta,
            coverage_delta,
            source_detail_delta,
        ),
    }


def _decision_driver(
    score_delta: float,
    coverage_delta: float,
    source_detail_delta: float,
) -> str:
    if score_delta > 0.001:
        return "score"
    if coverage_delta > 0.001:
        return "coverage"
    if source_detail_delta > 0.001:
        return "source_detail"
    if score_delta < -0.001:
        return "score_regression"
    if coverage_delta < -0.001:
        return "coverage_regression"
    if source_detail_delta < -0.001:
        return "source_detail_regression"
    return "no_change"


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    return (
        _numeric(candidate.get("visual_quality_passed_count")),
        _numeric(candidate.get("visual_quality_min_score")),
        _numeric(candidate.get("visual_quality_mean_score")),
        _numeric(candidate.get("min_confident_texel_fraction")),
        _numeric(candidate.get("mean_source_luma_std")),
        _numeric(candidate.get("feedback_used_count")),
    )


def _candidate_by_label(
    candidates: list[dict[str, Any]],
    label: str,
) -> dict[str, Any] | None:
    for candidate in candidates:
        if str(candidate.get("label", "")) == label:
            return candidate
    return None


def _regions_by_name(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    regions = {}
    for region in _list(candidate.get("regions")):
        if not isinstance(region, dict):
            continue
        name = str(region.get("name", ""))
        if name:
            regions[name] = region
    return regions


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _numeric(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
