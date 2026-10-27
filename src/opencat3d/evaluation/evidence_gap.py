"""Capture guidance from observed reconstruction evidence gaps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CAPTURE_TARGETS: dict[str, float | int] = {
    "min_basecolor_observed_texel_fraction": 0.55,
    "min_observed_face_fraction": 0.45,
    "min_observed_vertex_fraction": 0.60,
    "min_overlay_mean_silhouette_iou": 0.40,
    "max_silhouette_residual_mean_union_error": 0.60,
    "max_joint_projection_mean_keypoint_loss": 0.085,
    "min_texture_usable_frame_count": 24,
    "min_face_texture_frame_count": 4,
    "min_new_unique_frame_count_for_reconstruction": 6,
    "min_frontal_face_views": 3,
    "min_side_body_views": 2,
    "min_oblique_body_views": 4,
    "min_vertical_body_views": 2,
    "min_rear_body_views": 1,
}


@dataclass(frozen=True)
class SourceFrameAcquisitionGapResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]
    recommendation_count: int


def write_source_frame_acquisition_gap_report(
    project_dir: Path,
    *,
    media_preflight_report_path: Path | None = None,
    media_batch_report_path: Path | None = None,
    texture_probe_report_path: Path | None = None,
    output_path: Path | None = None,
    force: bool = False,
) -> SourceFrameAcquisitionGapResult:
    """Write deterministic capture guidance before another reconstruction run."""

    project_dir = Path(project_dir).expanduser().resolve()
    output_dir = project_dir / "outputs" / "quality"
    report_path = output_path or output_dir / "source_frame_acquisition_gap.json"
    report_path = Path(report_path).expanduser().resolve()
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True")

    summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
    summary = _read_json(summary_path)
    current_metrics = dict(summary.get("metrics", {})) if isinstance(summary, dict) else {}
    media_preflight = _read_optional_json(media_preflight_report_path)
    media_batch = _read_optional_json(media_batch_report_path)
    texture_probe = _read_optional_json(texture_probe_report_path)

    metrics = _gap_metrics(
        current_metrics=current_metrics,
        media_preflight=media_preflight,
        media_batch=media_batch,
        texture_probe=texture_probe,
    )
    metric_gaps = _metric_gaps(metrics)
    view_gaps = _view_gaps(media_preflight)
    recommendations = _capture_recommendations(
        metrics=metrics,
        metric_gaps=metric_gaps,
        view_gaps=view_gaps,
        media_batch=media_batch,
        texture_probe=texture_probe,
    )
    checks = {
        "project_dir_exists": project_dir.is_dir(),
        "raw_pipeline_summary_exists": summary_path.is_file(),
        "current_metrics_present": bool(current_metrics),
        "media_preflight_loaded": media_preflight_report_path is None
        or bool(media_preflight),
        "media_batch_loaded": media_batch_report_path is None or bool(media_batch),
        "texture_probe_loaded": texture_probe_report_path is None or bool(texture_probe),
        "recommendations_written": bool(recommendations),
    }
    success = all(checks.values())
    report = {
        "schema_version": "source_frame_acquisition_gap_v1",
        "source": "opencat3d.evaluation.evidence_gap",
        "project_dir": str(project_dir),
        "success": success,
        "checks": checks,
        "targets": CAPTURE_TARGETS,
        "metrics": metrics,
        "metric_gaps": metric_gaps,
        "view_gaps": view_gaps,
        "recommendations": recommendations,
        "stop_condition": {
            "run_reconstruction_now": _should_run_reconstruction_now(metrics),
            "reason": _reconstruction_stop_reason(metrics),
        },
        "artifacts": {
            "raw_pipeline_summary": str(summary_path),
            "media_preflight_report": (
                str(Path(media_preflight_report_path).expanduser().resolve())
                if media_preflight_report_path is not None
                else None
            ),
            "media_batch_report": (
                str(Path(media_batch_report_path).expanduser().resolve())
                if media_batch_report_path is not None
                else None
            ),
            "texture_probe_report": (
                str(Path(texture_probe_report_path).expanduser().resolve())
                if texture_probe_report_path is not None
                else None
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return SourceFrameAcquisitionGapResult(
        report_path=report_path,
        success=success,
        checks=checks,
        metrics=metrics,
        recommendation_count=len(recommendations),
    )


def _gap_metrics(
    *,
    current_metrics: dict[str, Any],
    media_preflight: dict[str, Any],
    media_batch: dict[str, Any],
    texture_probe: dict[str, Any],
) -> dict[str, float | int]:
    texture_candidates = texture_probe.get("pipeline_reports", [])
    if not isinstance(texture_candidates, list):
        texture_candidates = []
    return {
        "basecolor_observed_texel_fraction": _numeric(
            current_metrics.get("basecolor_observed_texel_fraction")
        ),
        "observed_face_fraction": _numeric(
            current_metrics.get("observed_face_fraction")
        ),
        "observed_vertex_fraction": _numeric(
            current_metrics.get("observed_vertex_fraction")
        ),
        "overlay_mean_silhouette_iou": _numeric(
            current_metrics.get("overlay_mean_silhouette_iou")
        ),
        "silhouette_residual_mean_union_error": _numeric(
            current_metrics.get("silhouette_residual_mean_union_error")
        ),
        "joint_projection_mean_keypoint_loss": _numeric(
            current_metrics.get("joint_projection_mean_keypoint_loss")
        ),
        "raw_facial_feature_count": int(
            _numeric(current_metrics.get("raw_facial_feature_count"))
        ),
        "raw_quality_passed_check_count": int(
            _numeric(current_metrics.get("raw_quality_passed_check_count"))
        ),
        "raw_quality_total_check_count": int(
            _numeric(current_metrics.get("raw_quality_total_check_count"))
        ),
        "geometry_safe_frame_count": int(
            _numeric(media_preflight.get("geometry_safe_frame_count"))
        ),
        "texture_usable_frame_count": int(
            _numeric(media_preflight.get("texture_usable_frame_count"))
        ),
        "face_texture_frame_count": int(
            _numeric(media_preflight.get("face_texture_frame_count"))
        ),
        "new_unique_frame_count": int(
            _numeric(media_batch.get("new_unique_frame_count"))
        ),
        "new_duplicate_frame_count": int(
            _numeric(media_batch.get("new_duplicate_frame_count"))
        ),
        "duplicate_frame_count": int(_numeric(media_batch.get("duplicate_frame_count"))),
        "staged_upload_frame_count": int(
            _numeric(media_batch.get("staged_upload_frame_count"))
        ),
        "texture_probe_candidate_count": len(texture_candidates),
        "texture_probe_added_frame_count": int(
            sum(
                _numeric(item.get("texture_only_added_unique_frame_count"))
                for item in texture_candidates
                if isinstance(item, dict)
            )
        ),
    }


def _metric_gaps(metrics: dict[str, float | int]) -> list[dict[str, object]]:
    gaps: list[dict[str, object]] = []
    _append_min_gap(
        gaps,
        metrics,
        "basecolor_observed_texel_fraction",
        "min_basecolor_observed_texel_fraction",
    )
    _append_min_gap(gaps, metrics, "observed_face_fraction", "min_observed_face_fraction")
    _append_min_gap(
        gaps,
        metrics,
        "observed_vertex_fraction",
        "min_observed_vertex_fraction",
    )
    _append_min_gap(
        gaps,
        metrics,
        "overlay_mean_silhouette_iou",
        "min_overlay_mean_silhouette_iou",
    )
    _append_max_gap(
        gaps,
        metrics,
        "silhouette_residual_mean_union_error",
        "max_silhouette_residual_mean_union_error",
    )
    _append_max_gap(
        gaps,
        metrics,
        "joint_projection_mean_keypoint_loss",
        "max_joint_projection_mean_keypoint_loss",
    )
    _append_min_gap(
        gaps,
        metrics,
        "texture_usable_frame_count",
        "min_texture_usable_frame_count",
    )
    _append_min_gap(
        gaps,
        metrics,
        "face_texture_frame_count",
        "min_face_texture_frame_count",
    )
    _append_min_gap(
        gaps,
        metrics,
        "new_unique_frame_count",
        "min_new_unique_frame_count_for_reconstruction",
    )
    return gaps


def _view_gaps(media_preflight: dict[str, Any]) -> list[dict[str, object]]:
    counts = media_preflight.get("view_group_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    targets = {
        "frontal_face": int(CAPTURE_TARGETS["min_frontal_face_views"]),
        "side_body": int(CAPTURE_TARGETS["min_side_body_views"]),
        "oblique_body": int(CAPTURE_TARGETS["min_oblique_body_views"]),
        "vertical_body": int(CAPTURE_TARGETS["min_vertical_body_views"]),
        "rear_body": int(CAPTURE_TARGETS["min_rear_body_views"]),
    }
    gaps: list[dict[str, object]] = []
    for view_group, target in targets.items():
        current = int(_numeric(counts.get(view_group)))
        if current < target:
            gaps.append(
                {
                    "view_group": view_group,
                    "current": current,
                    "target": target,
                    "missing": target - current,
                }
            )
    return gaps


def _capture_recommendations(
    *,
    metrics: dict[str, float | int],
    metric_gaps: list[dict[str, object]],
    view_gaps: list[dict[str, object]],
    media_batch: dict[str, Any],
    texture_probe: dict[str, Any],
) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    if int(metrics["new_unique_frame_count"]) < int(
        CAPTURE_TARGETS["min_new_unique_frame_count_for_reconstruction"]
    ):
        recommendations.append(
            {
                "id": "new_unique_evidence_required",
                "priority": 1,
                "category": "deduplication",
                "reason": (
                    "The latest upload did not add enough non-duplicate evidence "
                    "to justify another reconstruction run."
                ),
                "current_evidence": {
                    "new_unique_frame_count": metrics["new_unique_frame_count"],
                    "new_duplicate_frame_count": metrics["new_duplicate_frame_count"],
                    "duplicate_frame_count": metrics["duplicate_frame_count"],
                    "staged_upload_frame_count": metrics["staged_upload_frame_count"],
                },
                "target_evidence": {
                    "new_unique_frame_count": CAPTURE_TARGETS[
                        "min_new_unique_frame_count_for_reconstruction"
                    ],
                },
                "capture_instructions": [
                    "Capture new photos or video from angles not already in the evidence set.",
                    "Avoid re-exporting the same video segment or the same still frames.",
                    (
                        "Run preflight before full reconstruction and require "
                        "new_unique_frame_count > 0."
                    ),
                ],
            }
        )
    if any(gap["metric"] in _GEOMETRY_GAP_METRICS for gap in metric_gaps):
        recommendations.append(
            {
                "id": "geometry_alignment_views",
                "priority": 2,
                "category": "geometry",
                "reason": (
                    "Silhouette/keypoint alignment and observed mesh coverage "
                    "still need stronger full-body evidence."
                ),
                "current_evidence": {
                    "overlay_mean_silhouette_iou": metrics["overlay_mean_silhouette_iou"],
                    "silhouette_residual_mean_union_error": metrics[
                        "silhouette_residual_mean_union_error"
                    ],
                    "observed_vertex_fraction": metrics["observed_vertex_fraction"],
                    "view_gaps": [
                        gap for gap in view_gaps if gap["view_group"] != "frontal_face"
                    ],
                },
                "target_evidence": {
                    "overlay_mean_silhouette_iou": CAPTURE_TARGETS[
                        "min_overlay_mean_silhouette_iou"
                    ],
                    "observed_vertex_fraction": CAPTURE_TARGETS[
                        "min_observed_vertex_fraction"
                    ],
                },
                "capture_instructions": [
                    (
                        "Record slow, sharp full-body passes with the whole cat, "
                        "paws, and tail in frame."
                    ),
                    "Prioritize true left/right side views and a rear or back/tail-base view.",
                    (
                        "Keep the cat away from image borders; avoid cropped body, "
                        "cropped tail, and motion blur."
                    ),
                ],
            }
        )
    if any(gap["metric"] in _TEXTURE_GAP_METRICS for gap in metric_gaps):
        recommendations.append(
            {
                "id": "texture_coverage_views",
                "priority": 3,
                "category": "appearance",
                "reason": (
                    "Texture-source probing found no additional safe texels, "
                    "so more observed surface coverage is needed."
                ),
                "current_evidence": {
                    "basecolor_observed_texel_fraction": metrics[
                        "basecolor_observed_texel_fraction"
                    ],
                    "observed_face_fraction": metrics["observed_face_fraction"],
                    "texture_usable_frame_count": metrics["texture_usable_frame_count"],
                    "face_texture_frame_count": metrics["face_texture_frame_count"],
                    "texture_probe_added_frame_count": metrics[
                        "texture_probe_added_frame_count"
                    ],
                    "texture_probe_best_is_baseline": texture_probe.get(
                        "best_is_baseline"
                    ),
                },
                "target_evidence": {
                    "basecolor_observed_texel_fraction": CAPTURE_TARGETS[
                        "min_basecolor_observed_texel_fraction"
                    ],
                    "observed_face_fraction": CAPTURE_TARGETS[
                        "min_observed_face_fraction"
                    ],
                },
                "capture_instructions": [
                    (
                        "Add close frontal face photos with both eyes, nose, "
                        "muzzle, and ear interiors visible."
                    ),
                    (
                        "Add full-body side and three-quarter photos under even "
                        "exposure so white fur is not blown out."
                    ),
                    "Use still photos when possible; for video, move slowly and keep focus locked.",
                ],
            }
        )
    if media_batch:
        recommendations.append(
            {
                "id": "preflight_gate_before_reconstruction",
                "priority": 4,
                "category": "workflow",
                "reason": (
                    "Preflight can reject duplicate or low-value uploads before "
                    "expensive reconstruction."
                ),
                "current_evidence": {
                    "latest_media_batch_report": media_batch.get("upload_dir"),
                    "new_unique_frame_count": metrics["new_unique_frame_count"],
                },
                "target_evidence": {
                    "preflight_required": True,
                    "full_reconstruction_allowed_only_if": (
                        "new_unique_frame_count >= 6 or an operator intentionally "
                        "overrides the stop condition"
                    ),
                },
                "capture_instructions": [
                    "Always run media preflight first.",
                    (
                        "Only run full reconstruction when the acquisition gap "
                        "report says run_reconstruction_now is true."
                    ),
                ],
            }
        )
    return recommendations


def _should_run_reconstruction_now(metrics: dict[str, float | int]) -> bool:
    return int(metrics["new_unique_frame_count"]) >= int(
        CAPTURE_TARGETS["min_new_unique_frame_count_for_reconstruction"]
    )


def _reconstruction_stop_reason(metrics: dict[str, float | int]) -> str:
    if _should_run_reconstruction_now(metrics):
        return "new_unique_evidence_available"
    return "insufficient_new_unique_evidence"


def _append_min_gap(
    gaps: list[dict[str, object]],
    metrics: dict[str, float | int],
    metric_name: str,
    target_name: str,
) -> None:
    value = float(metrics.get(metric_name, 0.0))
    target = float(CAPTURE_TARGETS[target_name])
    if value < target:
        gaps.append(
            {
                "metric": metric_name,
                "direction": "increase",
                "current": value,
                "target": target,
                "gap": target - value,
            }
        )


def _append_max_gap(
    gaps: list[dict[str, object]],
    metrics: dict[str, float | int],
    metric_name: str,
    target_name: str,
) -> None:
    value = float(metrics.get(metric_name, 0.0))
    target = float(CAPTURE_TARGETS[target_name])
    if value > target:
        gaps.append(
            {
                "metric": metric_name,
                "direction": "decrease",
                "current": value,
                "target": target,
                "gap": value - target,
            }
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return _read_json(Path(path).expanduser().resolve())


def _numeric(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_GEOMETRY_GAP_METRICS = {
    "observed_vertex_fraction",
    "overlay_mean_silhouette_iou",
    "silhouette_residual_mean_union_error",
    "joint_projection_mean_keypoint_loss",
}
_TEXTURE_GAP_METRICS = {
    "basecolor_observed_texel_fraction",
    "observed_face_fraction",
    "texture_usable_frame_count",
    "face_texture_frame_count",
}
