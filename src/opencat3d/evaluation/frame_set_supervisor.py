"""Deterministic frame-set candidate supervisor for raw reconstruction jobs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

FRAME_SET_SUPERVISOR_VERSION = "frame_set_supervisor_v1"


@dataclass(frozen=True)
class FrameSetSupervisorResult:
    report_path: Path
    best_candidate: str
    candidate_count: int


@dataclass(frozen=True)
class AlignmentGatedFrameSetResult:
    report_path: Path
    best_candidate: str
    candidate_count: int
    rejected_candidate_count: int


@dataclass(frozen=True)
class TextureOnlyPromotionGuardResult:
    report_path: Path
    passed: bool
    rejection_reasons: tuple[str, ...]
    improved_metric_count: int


@dataclass(frozen=True)
class BodyFitAlignmentGateResult:
    report_path: Path
    passed: bool
    rejection_reasons: tuple[str, ...]
    body_fit_delta: float
    improved_alignment_metric_count: int


TEXTURE_ONLY_DETAIL_METRICS: tuple[str, ...] = (
    "basecolor_observed_texel_fraction",
    "observed_face_fraction",
    "observed_vertex_fraction",
    "raw_facial_feature_painted_texel_count",
    "raw_face_turntable_front_textured_unique_rgb_count",
    "raw_face_turntable_front_textured_luma_std",
    "raw_face_turntable_front_textured_mouth_line_pixel_count",
    "raw_face_turntable_front_textured_required_feature_fraction",
    "raw_face_texture_closeup_required_feature_fraction",
    "raw_glb_face_feature_eye_detail_luma_std",
    "raw_glb_face_feature_iris_to_pupil_rgb_distance",
)

TEXTURE_ONLY_STRICT_FACE_DETAIL_METRICS: tuple[str, ...] = (
    "raw_face_turntable_front_textured_luma_std",
    "raw_glb_face_feature_eye_detail_luma_std",
    "raw_glb_face_feature_iris_to_pupil_rgb_distance",
)


def write_frame_set_candidate_report(
    project_dir: Path,
    output_path: Path | None = None,
    *,
    reference_project_dir: Path | None = None,
    force: bool = False,
) -> FrameSetSupervisorResult:
    """Write deterministic frame-set proposals from an existing frame-selection report."""

    project_dir = Path(project_dir)
    selection_path = project_dir / "observations" / "raw" / "frame_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"{selection_path} is required")
    if output_path is None:
        output_path = project_dir / "outputs" / "quality" / "frame_set_candidates.json"
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True to overwrite")

    selection_report = _read_json(selection_path)
    frames = selection_report.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("frame selection report contains no frames")
    frame_records = [_normalise_frame(frame) for frame in frames]
    min_frames = int(selection_report.get("min_frames", 2))
    max_frames = int(selection_report.get("max_selected_frames", 12) or len(frame_records))
    max_frames = max(min_frames, min(max_frames, len(frame_records)))
    current_ids = tuple(str(item) for item in selection_report.get("selected_frame_ids", []))
    current_texture_ids = tuple(str(item) for item in selection_report.get("texture_frame_ids", []))
    (
        residual_metrics_by_frame_id,
        residual_report_path,
        residual_reference_project,
    ) = _load_projection_residual_metrics(
        project_dir,
        reference_project_dir=reference_project_dir,
    )

    proposals = _candidate_proposals(
        frame_records,
        current_ids=current_ids,
        current_texture_ids=current_texture_ids,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    if not proposals:
        raise ValueError("frame-set supervisor could not produce candidates")
    reference_metrics = _reference_candidate_metrics(
        frame_records,
        current_ids=current_ids,
        current_texture_ids=current_texture_ids,
        residual_metrics_by_frame_id=residual_metrics_by_frame_id,
    )
    candidates = [
        _candidate_json(
            label,
            geometry_ids,
            texture_ids,
            frame_records,
            reference_metrics=reference_metrics,
            residual_metrics_by_frame_id=residual_metrics_by_frame_id,
        )
        for label, geometry_ids, texture_ids in proposals
    ]
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["label"])))
    prefiltered_count = sum(
        1 for candidate in candidates if not bool(candidate["alignment_risk_prefilter"]["passed"])
    )

    report = {
        "schema_version": "1.0",
        "source": "opencat3d.frame_set_supervisor",
        "version": FRAME_SET_SUPERVISOR_VERSION,
        "project_id": selection_report.get("project_id", project_dir.name),
        "selection_report": str(selection_path),
        "input_frame_count": len(frame_records),
        "min_frames": min_frames,
        "max_selected_frames": max_frames,
        "best_candidate": str(candidates[0]["label"]),
        "candidate_count": len(candidates),
        "alignment_risk_prefiltered_candidate_count": int(prefiltered_count),
        "projection_residual_report": (
            str(residual_report_path) if residual_report_path is not None else None
        ),
        "projection_residual_reference_project": (
            str(residual_reference_project) if residual_reference_project is not None else None
        ),
        "projection_residual_frame_count": len(residual_metrics_by_frame_id),
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return FrameSetSupervisorResult(
        report_path=output_path,
        best_candidate=str(candidates[0]["label"]),
        candidate_count=len(candidates),
    )


def write_alignment_gated_frame_set_report(
    baseline_project_dir: Path,
    candidate_project_dirs: dict[str, Path],
    output_path: Path,
    *,
    max_overlay_iou_drop: float = 1e-9,
    max_silhouette_residual_increase: float = 1e-9,
    max_keypoint_loss_increase: float = 1e-9,
    require_body_fit_alignment: bool = True,
    min_body_fit_delta: float = 1e-9,
    metric_epsilon: float = 1e-9,
    force: bool = False,
) -> AlignmentGatedFrameSetResult:
    """Select evaluated frame-set candidates only when alignment does not regress."""

    if not candidate_project_dirs:
        raise ValueError("at least one candidate is required")
    if max_overlay_iou_drop < 0.0:
        raise ValueError("max_overlay_iou_drop must be non-negative")
    if max_silhouette_residual_increase < 0.0:
        raise ValueError("max_silhouette_residual_increase must be non-negative")
    if max_keypoint_loss_increase < 0.0:
        raise ValueError("max_keypoint_loss_increase must be non-negative")
    if min_body_fit_delta < 0.0:
        raise ValueError("min_body_fit_delta must be non-negative")
    if metric_epsilon < 0.0:
        raise ValueError("metric_epsilon must be non-negative")

    baseline_project_dir = Path(baseline_project_dir)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True to overwrite")

    baseline_summary_path = _summary_path(baseline_project_dir)
    baseline_summary = _read_json(baseline_summary_path)
    baseline_metrics = _metrics(baseline_summary, baseline_summary_path)
    baseline_alignment = _alignment_metrics(baseline_metrics)
    candidates = [
        _alignment_candidate_json(
            "baseline",
            baseline_project_dir,
            baseline_summary_path,
            baseline_summary,
            baseline_metrics,
            baseline_alignment=baseline_alignment,
            baseline_metrics=baseline_metrics,
            max_overlay_iou_drop=max_overlay_iou_drop,
            max_silhouette_residual_increase=max_silhouette_residual_increase,
            max_keypoint_loss_increase=max_keypoint_loss_increase,
            require_body_fit_alignment=require_body_fit_alignment,
            min_body_fit_delta=min_body_fit_delta,
            metric_epsilon=metric_epsilon,
            is_baseline=True,
        )
    ]
    for label, project_dir in sorted(candidate_project_dirs.items()):
        candidate_project_dir = Path(project_dir)
        summary_path = _summary_path(candidate_project_dir)
        summary = _read_json(summary_path)
        metrics = _metrics(summary, summary_path)
        candidates.append(
            _alignment_candidate_json(
                str(label),
                candidate_project_dir,
                summary_path,
                summary,
                metrics,
                baseline_alignment=baseline_alignment,
                baseline_metrics=baseline_metrics,
                max_overlay_iou_drop=max_overlay_iou_drop,
                max_silhouette_residual_increase=max_silhouette_residual_increase,
                max_keypoint_loss_increase=max_keypoint_loss_increase,
                require_body_fit_alignment=require_body_fit_alignment,
                min_body_fit_delta=min_body_fit_delta,
                metric_epsilon=metric_epsilon,
                is_baseline=False,
            )
        )

    candidates.sort(key=lambda item: (-float(item["score"]), str(item["label"])))
    rejected_count = sum(
        1
        for item in candidates
        if str(item["label"]) != "baseline"
        and (
            not bool(item["alignment_gate"]["passed"])
            or not bool(item["body_fit_alignment_gate"]["passed"])
        )
    )
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.frame_set_alignment_gate",
        "version": FRAME_SET_SUPERVISOR_VERSION,
        "baseline_project_dir": str(baseline_project_dir),
        "baseline_summary_path": str(baseline_summary_path),
        "baseline_alignment_metrics": baseline_alignment,
        "thresholds": {
            "max_overlay_iou_drop": float(max_overlay_iou_drop),
            "max_silhouette_residual_increase": float(max_silhouette_residual_increase),
            "max_keypoint_loss_increase": float(max_keypoint_loss_increase),
            "require_body_fit_alignment": bool(require_body_fit_alignment),
            "min_body_fit_delta": float(min_body_fit_delta),
            "metric_epsilon": float(metric_epsilon),
        },
        "best_candidate": str(candidates[0]["label"]),
        "candidate_count": len(candidates),
        "rejected_candidate_count": int(rejected_count),
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return AlignmentGatedFrameSetResult(
        report_path=output_path,
        best_candidate=str(candidates[0]["label"]),
        candidate_count=len(candidates),
        rejected_candidate_count=int(rejected_count),
    )


def write_body_fit_alignment_gate_report(
    baseline_project_dir: Path,
    candidate_project_dir: Path,
    output_path: Path,
    *,
    min_body_fit_delta: float = 1e-9,
    metric_epsilon: float = 1e-9,
    max_overlay_iou_drop: float = 0.0,
    max_silhouette_residual_increase: float = 0.0,
    max_keypoint_loss_increase: float = 0.0,
    force: bool = False,
) -> BodyFitAlignmentGateResult:
    """Gate body-fit frame changes on measured alignment improvement."""

    if min_body_fit_delta < 0.0:
        raise ValueError("min_body_fit_delta must be non-negative")
    if metric_epsilon < 0.0:
        raise ValueError("metric_epsilon must be non-negative")
    if max_overlay_iou_drop < 0.0:
        raise ValueError("max_overlay_iou_drop must be non-negative")
    if max_silhouette_residual_increase < 0.0:
        raise ValueError("max_silhouette_residual_increase must be non-negative")
    if max_keypoint_loss_increase < 0.0:
        raise ValueError("max_keypoint_loss_increase must be non-negative")

    baseline_project_dir = Path(baseline_project_dir)
    candidate_project_dir = Path(candidate_project_dir)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True to overwrite")

    baseline_summary_path = _summary_path(baseline_project_dir)
    candidate_summary_path = _summary_path(candidate_project_dir)
    baseline_summary = _read_json(baseline_summary_path)
    candidate_summary = _read_json(candidate_summary_path)
    baseline_metrics = _metrics(baseline_summary, baseline_summary_path)
    candidate_metrics = _metrics(candidate_summary, candidate_summary_path)

    baseline_alignment = _alignment_metrics(baseline_metrics)
    candidate_alignment = _alignment_metrics(candidate_metrics)
    alignment_gate = _alignment_gate(
        candidate_alignment,
        baseline_alignment=baseline_alignment,
        max_overlay_iou_drop=max_overlay_iou_drop,
        max_silhouette_residual_increase=max_silhouette_residual_increase,
        max_keypoint_loss_increase=max_keypoint_loss_increase,
        summary_success=bool(candidate_summary.get("success", False)),
        is_baseline=False,
    )
    baseline_body_fit = _optional_numeric_metric(
        baseline_metrics,
        "raw_geometry_selected_mean_body_fit_score",
    )
    candidate_body_fit = _optional_numeric_metric(
        candidate_metrics,
        "raw_geometry_selected_mean_body_fit_score",
    )
    body_fit_delta = (
        float(candidate_body_fit - baseline_body_fit)
        if baseline_body_fit is not None and candidate_body_fit is not None
        else 0.0
    )
    alignment_deltas = {
        "overlay_mean_silhouette_iou": float(
            candidate_alignment["overlay_mean_silhouette_iou"]
            - baseline_alignment["overlay_mean_silhouette_iou"]
        ),
        "silhouette_residual_mean_union_error": float(
            candidate_alignment["silhouette_residual_mean_union_error"]
            - baseline_alignment["silhouette_residual_mean_union_error"]
        ),
        "joint_projection_mean_keypoint_loss": float(
            candidate_alignment["joint_projection_mean_keypoint_loss"]
            - baseline_alignment["joint_projection_mean_keypoint_loss"]
        ),
    }
    alignment_improvements = {
        "overlay_mean_silhouette_iou": (
            alignment_deltas["overlay_mean_silhouette_iou"] > metric_epsilon
        ),
        "silhouette_residual_mean_union_error": (
            alignment_deltas["silhouette_residual_mean_union_error"] < -metric_epsilon
        ),
        "joint_projection_mean_keypoint_loss": (
            alignment_deltas["joint_projection_mean_keypoint_loss"] < -metric_epsilon
        ),
    }
    improved_alignment_metric_count = sum(1 for value in alignment_improvements.values() if value)
    baseline_selected_ids, baseline_texture_ids = _selection_ids(baseline_project_dir)
    candidate_selected_ids, candidate_texture_ids = _selection_ids(candidate_project_dir)
    checks = {
        "candidate_success": bool(candidate_summary.get("success", False)),
        "baseline_body_fit_present": baseline_body_fit is not None,
        "candidate_body_fit_present": candidate_body_fit is not None,
        "body_fit_improved": body_fit_delta > min_body_fit_delta,
        "alignment_not_regressed": bool(alignment_gate["passed"]),
        "alignment_metric_improved": improved_alignment_metric_count > 0,
    }
    rejection_reasons = [key for key, value in checks.items() if not value]
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.body_fit_alignment_gate",
        "version": f"{FRAME_SET_SUPERVISOR_VERSION}:body_fit_alignment_gate_v1",
        "baseline_project_dir": str(baseline_project_dir),
        "candidate_project_dir": str(candidate_project_dir),
        "baseline_summary_path": str(baseline_summary_path),
        "candidate_summary_path": str(candidate_summary_path),
        "passed": not rejection_reasons,
        "checks": checks,
        "rejection_reasons": rejection_reasons,
        "thresholds": {
            "min_body_fit_delta": float(min_body_fit_delta),
            "metric_epsilon": float(metric_epsilon),
            "max_overlay_iou_drop": float(max_overlay_iou_drop),
            "max_silhouette_residual_increase": float(max_silhouette_residual_increase),
            "max_keypoint_loss_increase": float(max_keypoint_loss_increase),
        },
        "body_fit": {
            "metric": "raw_geometry_selected_mean_body_fit_score",
            "baseline": baseline_body_fit,
            "candidate": candidate_body_fit,
            "delta": body_fit_delta,
        },
        "alignment_gate": alignment_gate,
        "alignment_metrics": {
            "baseline": baseline_alignment,
            "candidate": candidate_alignment,
            "deltas": alignment_deltas,
            "improvements": alignment_improvements,
            "improved_metric_count": int(improved_alignment_metric_count),
        },
        "frame_selection": {
            "baseline_selected_frame_ids": list(baseline_selected_ids),
            "candidate_selected_frame_ids": list(candidate_selected_ids),
            "baseline_texture_frame_ids": list(baseline_texture_ids),
            "candidate_texture_frame_ids": list(candidate_texture_ids),
        },
        "promotion_recommendation": (
            "eligible_body_fit_frame_change"
            if not rejection_reasons
            else "keep_baseline_or_try_smaller_frame_change"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BodyFitAlignmentGateResult(
        report_path=output_path,
        passed=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
        body_fit_delta=float(body_fit_delta),
        improved_alignment_metric_count=int(improved_alignment_metric_count),
    )


def write_texture_only_promotion_guard(
    baseline_project_dir: Path,
    candidate_project_dir: Path,
    output_path: Path,
    *,
    metric_epsilon: float = 1e-9,
    force: bool = False,
) -> TextureOnlyPromotionGuardResult:
    """Gate texture-only candidates so texture gains cannot hide geometry regressions."""

    if metric_epsilon < 0.0:
        raise ValueError("metric_epsilon must be non-negative")
    baseline_project_dir = Path(baseline_project_dir)
    candidate_project_dir = Path(candidate_project_dir)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True to overwrite")

    baseline_summary_path = _summary_path(baseline_project_dir)
    candidate_summary_path = _summary_path(candidate_project_dir)
    baseline_summary = _read_json(baseline_summary_path)
    candidate_summary = _read_json(candidate_summary_path)
    baseline_metrics = _metrics(baseline_summary, baseline_summary_path)
    candidate_metrics = _metrics(candidate_summary, candidate_summary_path)
    candidate_success = _texture_only_candidate_success(candidate_summary, candidate_metrics)

    baseline_selected_ids, baseline_texture_ids = _selection_ids(baseline_project_dir)
    candidate_selected_ids, candidate_texture_ids = _selection_ids(candidate_project_dir)
    baseline_hashes = _observation_image_hashes(baseline_project_dir)
    candidate_hashes = _observation_image_hashes(candidate_project_dir)
    baseline_texture_hashes_by_frame_id = _texture_image_hashes(baseline_project_dir)
    candidate_texture_hashes_by_frame_id = _texture_image_hashes(candidate_project_dir)
    baseline_geometry_hashes = _frame_hash_sequence(
        baseline_selected_ids,
        baseline_hashes,
    )
    candidate_geometry_hashes = _frame_hash_sequence(
        candidate_selected_ids,
        candidate_hashes,
    )
    baseline_texture_hashes = _frame_hash_set(
        baseline_texture_ids,
        baseline_texture_hashes_by_frame_id,
    )
    candidate_texture_hashes = _frame_hash_set(
        candidate_texture_ids,
        candidate_texture_hashes_by_frame_id,
    )
    added_texture_hashes = sorted(candidate_texture_hashes - baseline_texture_hashes)
    baked_seed_texture_evidence = _baked_seed_texture_evidence_report(
        baseline_project_dir,
        candidate_project_dir,
        baseline_texture_hashes=baseline_texture_hashes,
    )
    missing_baseline_geometry_hash_ids = _missing_hash_ids(
        baseline_selected_ids,
        baseline_hashes,
    )
    missing_candidate_geometry_hash_ids = _missing_hash_ids(
        candidate_selected_ids,
        candidate_hashes,
    )
    missing_baseline_texture_hash_ids = _missing_hash_ids(
        baseline_texture_ids,
        baseline_texture_hashes_by_frame_id,
    )
    missing_candidate_texture_hash_ids = _missing_hash_ids(
        candidate_texture_ids,
        candidate_texture_hashes_by_frame_id,
    )

    baseline_alignment = _alignment_metrics(baseline_metrics)
    candidate_alignment = _alignment_metrics(candidate_metrics)
    alignment_gate = _alignment_gate(
        candidate_alignment,
        baseline_alignment=baseline_alignment,
        max_overlay_iou_drop=0.0,
        max_silhouette_residual_increase=0.0,
        max_keypoint_loss_increase=0.0,
        summary_success=bool(candidate_summary.get("success", False)),
        require_summary_success=False,
        is_baseline=False,
    )
    metric_comparisons = _texture_only_metric_comparisons(
        baseline_metrics,
        candidate_metrics,
        metric_epsilon=metric_epsilon,
    )
    regressed_metrics = [
        item["metric"]
        for item in metric_comparisons
        if item["status"] in {"regressed", "missing_candidate"}
    ]
    improved_metric_count = sum(1 for item in metric_comparisons if item["status"] == "improved")
    strict_regressed_metric_names = [
        item["metric"]
        for item in metric_comparisons
        if item["metric"] in TEXTURE_ONLY_STRICT_FACE_DETAIL_METRICS
        and item["status"] in {"regressed", "missing_candidate"}
    ]
    supplemental_source_rejections = _supplemental_texture_source_rejections(
        candidate_project_dir,
        strict_regressed_metric_names=strict_regressed_metric_names,
    )
    geometry_hashes_preserved = (
        bool(baseline_geometry_hashes)
        and baseline_geometry_hashes == candidate_geometry_hashes
        and not missing_baseline_geometry_hash_ids
        and not missing_candidate_geometry_hash_ids
    )
    texture_frame_hashes_preserved = baseline_texture_hashes <= candidate_texture_hashes
    texture_evidence_preserved = (
        (texture_frame_hashes_preserved or bool(baked_seed_texture_evidence["preserved"]))
        and not missing_baseline_texture_hash_ids
        and not missing_candidate_texture_hash_ids
    )
    has_texture_or_detail_improvement = bool(added_texture_hashes or improved_metric_count > 0)
    checks = {
        "candidate_success": candidate_success,
        "geometry_hashes_preserved": geometry_hashes_preserved,
        "texture_evidence_preserved": texture_evidence_preserved,
        "alignment_not_regressed": bool(alignment_gate["passed"]),
        "detail_metrics_not_regressed": not regressed_metrics,
        "strict_face_detail_metrics_not_regressed": not strict_regressed_metric_names,
        "has_texture_or_detail_improvement": has_texture_or_detail_improvement,
    }
    rejection_reasons = [key for key, value in checks.items() if not value]
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.texture_only_promotion_guard",
        "version": f"{FRAME_SET_SUPERVISOR_VERSION}:texture_only_guard_v1",
        "baseline_project_dir": str(baseline_project_dir),
        "candidate_project_dir": str(candidate_project_dir),
        "baseline_summary_path": str(baseline_summary_path),
        "candidate_summary_path": str(candidate_summary_path),
        "passed": not rejection_reasons,
        "checks": checks,
        "rejection_reasons": rejection_reasons,
        "alignment_gate": alignment_gate,
        "alignment_deltas": {
            "overlay_mean_silhouette_iou": float(
                candidate_alignment["overlay_mean_silhouette_iou"]
                - baseline_alignment["overlay_mean_silhouette_iou"]
            ),
            "silhouette_residual_mean_union_error": float(
                candidate_alignment["silhouette_residual_mean_union_error"]
                - baseline_alignment["silhouette_residual_mean_union_error"]
            ),
            "joint_projection_mean_keypoint_loss": float(
                candidate_alignment["joint_projection_mean_keypoint_loss"]
                - baseline_alignment["joint_projection_mean_keypoint_loss"]
            ),
        },
        "geometry": {
            "baseline_selected_frame_ids": list(baseline_selected_ids),
            "candidate_selected_frame_ids": list(candidate_selected_ids),
            "baseline_selected_hashes": list(baseline_geometry_hashes),
            "candidate_selected_hashes": list(candidate_geometry_hashes),
            "missing_baseline_hash_frame_ids": list(missing_baseline_geometry_hash_ids),
            "missing_candidate_hash_frame_ids": list(missing_candidate_geometry_hash_ids),
        },
        "texture": {
            "baseline_texture_frame_ids": list(baseline_texture_ids),
            "candidate_texture_frame_ids": list(candidate_texture_ids),
            "baseline_unique_texture_hash_count": len(baseline_texture_hashes),
            "candidate_unique_texture_hash_count": len(candidate_texture_hashes),
            "added_unique_texture_hash_count": len(added_texture_hashes),
            "hash_source": "observations/raw/texture_manifest.json",
            "frame_hashes_preserved": texture_frame_hashes_preserved,
            "baseline_baked_seed_evidence": baked_seed_texture_evidence,
            "missing_baseline_hash_frame_ids": list(missing_baseline_texture_hash_ids),
            "missing_candidate_hash_frame_ids": list(missing_candidate_texture_hash_ids),
        },
        "detail_metric_names": list(TEXTURE_ONLY_DETAIL_METRICS),
        "strict_face_detail_metric_names": list(TEXTURE_ONLY_STRICT_FACE_DETAIL_METRICS),
        "detail_metric_comparisons": metric_comparisons,
        "improved_metric_count": int(improved_metric_count),
        "regressed_metric_names": regressed_metrics,
        "strict_regressed_metric_names": strict_regressed_metric_names,
        "candidate_success_policy": _texture_only_candidate_success_policy(
            candidate_summary,
            candidate_metrics,
        ),
        "supplemental_source_rejection_count": len(supplemental_source_rejections),
        "supplemental_source_rejections": supplemental_source_rejections,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TextureOnlyPromotionGuardResult(
        report_path=output_path,
        passed=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
        improved_metric_count=int(improved_metric_count),
    )


def _candidate_proposals(
    frames: list[dict[str, Any]],
    *,
    current_ids: tuple[str, ...],
    current_texture_ids: tuple[str, ...],
    min_frames: int,
    max_frames: int,
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    eligible = [frame for frame in frames if not bool(frame["rejected_by_quality"])]
    usable = eligible if len(eligible) >= min_frames else frames
    score_ranked = _rank_geometry_frames(usable)
    proposals: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    if current_ids:
        proposals.append(
            (
                "current_selection",
                _ordered_existing_ids(frames, current_ids),
                _ordered_existing_ids(frames, current_texture_ids),
            )
        )
    proposals.append(
        (
            "score_only_geometry",
            _ids(score_ranked[:max_frames]),
            _texture_ids(frames, _ids(score_ranked[:max_frames])),
        )
    )
    balanced = _balanced_frame_ids(score_ranked, max_frames=max_frames, min_frames=min_frames)
    proposals.append(("balanced_photo_video_geometry", balanced, _texture_ids(frames, balanced)))

    full_body_upload = _full_body_photo_video_ids(
        score_ranked,
        max_frames=max_frames,
        min_frames=min_frames,
    )
    if len(full_body_upload) >= min_frames:
        proposals.append(
            (
                "full_body_upload_geometry",
                full_body_upload,
                _texture_ids(frames, full_body_upload),
            )
        )

    low_risk = _rank_geometry_frames(
        [
            frame
            for frame in usable
            if float(frame["touches_image_border"]) < 0.5
            and float(frame["bbox_area_ratio"]) <= 0.58
        ]
    )
    if len(low_risk) >= min_frames:
        low_risk_ids = _top_up_ids(_ids(low_risk[:max_frames]), score_ranked, max_frames)
        proposals.append(
            (
                "low_border_body_geometry",
                low_risk_ids,
                _texture_ids(frames, low_risk_ids),
            )
        )

    relaxed_body = _rank_geometry_frames(
        [frame for frame in frames if _relaxed_body_candidate(frame)]
    )
    if len(relaxed_body) >= min_frames:
        relaxed_ids = _balanced_frame_ids(
            relaxed_body,
            max_frames=max_frames,
            min_frames=min_frames,
        )
        proposals.append(
            (
                "relaxed_body_photo_geometry",
                relaxed_ids,
                _texture_ids(frames, relaxed_ids),
            )
        )

    view_diverse = _view_diverse_body_ids(
        score_ranked,
        max_frames=max_frames,
        min_frames=min_frames,
    )
    if len(view_diverse) >= min_frames:
        proposals.append(
            (
                "view_diverse_body_geometry",
                view_diverse,
                _texture_ids(frames, view_diverse),
            )
        )

    video_dense = _video_dense_ids(score_ranked, max_frames=max_frames, min_frames=min_frames)
    if len(video_dense) >= min_frames:
        proposals.append(
            ("video_dense_body_geometry", video_dense, _texture_ids(frames, video_dense))
        )

    texture_expanded = _texture_ids(frames, current_ids or _ids(score_ranked[:max_frames]))
    if texture_expanded:
        proposals.append(
            (
                "current_geometry_expanded_texture",
                _ordered_existing_ids(frames, current_ids or _ids(score_ranked[:max_frames])),
                texture_expanded,
            )
        )
    return _dedupe_proposals(proposals)


def _candidate_json(
    label: str,
    geometry_ids: tuple[str, ...],
    texture_ids: tuple[str, ...],
    frames: list[dict[str, Any]],
    *,
    reference_metrics: dict[str, float] | None,
    residual_metrics_by_frame_id: dict[str, dict[str, float]],
) -> dict[str, Any]:
    by_id = {str(frame["frame_id"]): frame for frame in frames}
    geometry = [by_id[frame_id] for frame_id in geometry_ids if frame_id in by_id]
    texture = [by_id[frame_id] for frame_id in texture_ids if frame_id in by_id]
    metrics = _candidate_metrics(
        geometry,
        texture,
        residual_metrics_by_frame_id=residual_metrics_by_frame_id,
    )
    prefilter = _alignment_risk_prefilter(
        metrics,
        reference_metrics=reference_metrics,
        is_reference=label == "current_selection",
    )
    terms = _score_terms(metrics)
    terms["alignment_risk_prefilter"] = _alignment_risk_score_term(prefilter)
    score = sum(terms.values())
    return {
        "label": label,
        "score": round(float(score), 9),
        "score_terms": {key: round(float(value), 9) for key, value in terms.items()},
        "geometry_frame_ids": list(geometry_ids),
        "texture_frame_ids": list(texture_ids),
        "geometry_media_counts": _media_counts(geometry),
        "texture_media_counts": _media_counts(texture),
        "geometry_reject_reasons": _reject_reason_counts(geometry),
        "alignment_risk_prefilter": prefilter,
        "metrics": metrics,
    }


def _candidate_metrics(
    geometry: list[dict[str, Any]],
    texture: list[dict[str, Any]],
    *,
    residual_metrics_by_frame_id: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    geometry_count = len(geometry)
    texture_count = len(texture)
    residual_metrics_by_frame_id = residual_metrics_by_frame_id or {}
    residuals = [
        residual_metrics_by_frame_id[str(frame["frame_id"])]
        for frame in geometry
        if str(frame["frame_id"]) in residual_metrics_by_frame_id
    ]
    residual_count = len(residuals)
    residual_coverage = residual_count / max(geometry_count, 1)
    geometry_quality = _mean(max(0.0, 1.0 - min(float(frame["score"]), 1.0)) for frame in geometry)
    bbox_mean = _mean(float(frame["bbox_area_ratio"]) for frame in geometry)
    mask_mean = _mean(float(frame["mask_area_ratio"]) for frame in geometry)
    border_mean = _mean(float(frame["touches_image_border"]) for frame in geometry)
    geometry_rejected_fraction = _mean(
        float(bool(frame["rejected_by_quality"])) for frame in geometry
    )
    relaxed_bbox_fraction = _mean(
        float("bbox_too_large" in frame["reject_reasons"]) for frame in geometry
    )
    closeup_photo_fraction = _mean(float(_is_closeup_photo(frame)) for frame in geometry)
    view_buckets = [_frame_view_bucket(frame) for frame in geometry]
    view_bucket_count = len(set(view_buckets))
    view_bucket_denom = max(1.0, min(4.0, float(geometry_count)))
    view_spread_fraction = min(float(view_bucket_count) / view_bucket_denom, 1.0)
    centers_x = [float(frame["bbox_center"][0]) for frame in geometry]
    aspects = [float(frame["bbox_aspect"]) for frame in geometry]
    texture_photo_count = sum(1 for frame in texture if frame["media_type"] == "photo")
    texture_closeup_photo_count = sum(1 for frame in texture if _is_closeup_photo(frame))
    geometry_full_body_photo_count = sum(1 for frame in geometry if _is_full_body_photo(frame))
    texture_full_body_photo_count = sum(1 for frame in texture if _is_full_body_photo(frame))
    texture_face_detail_fraction = texture_closeup_photo_count / max(texture_photo_count, 1)
    return {
        "geometry_frame_count": float(geometry_count),
        "texture_frame_count": float(texture_count),
        "geometry_photo_frame_count": float(
            sum(1 for frame in geometry if frame["media_type"] == "photo")
        ),
        "geometry_video_frame_count": float(
            sum(1 for frame in geometry if frame["media_type"] == "video")
        ),
        "texture_photo_frame_count": float(texture_photo_count),
        "texture_video_frame_count": float(
            sum(1 for frame in texture if frame["media_type"] == "video")
        ),
        "mean_geometry_quality": round(geometry_quality, 9),
        "mean_mask_area_ratio": round(mask_mean, 9),
        "mean_bbox_area_ratio": round(bbox_mean, 9),
        "mean_border_touch_fraction": round(border_mean, 9),
        "geometry_rejected_fraction": round(geometry_rejected_fraction, 9),
        "relaxed_bbox_fraction": round(relaxed_bbox_fraction, 9),
        "geometry_closeup_photo_fraction": round(closeup_photo_fraction, 9),
        "geometry_full_body_photo_frame_count": float(geometry_full_body_photo_count),
        "geometry_full_body_photo_fraction": round(
            geometry_full_body_photo_count / max(geometry_count, 1),
            9,
        ),
        "texture_full_body_photo_frame_count": float(texture_full_body_photo_count),
        "geometry_view_bucket_count": float(view_bucket_count),
        "geometry_view_spread_fraction": round(view_spread_fraction, 9),
        "geometry_horizontal_center_span": round(_span(centers_x), 9),
        "geometry_bbox_aspect_span": round(_span(aspects), 9),
        "texture_face_detail_fraction": round(texture_face_detail_fraction, 9),
        "texture_closeup_photo_count": float(texture_closeup_photo_count),
        "media_balance_fraction": round(_media_balance_fraction(geometry), 9),
        "geometry_projection_residual_frame_count": float(residual_count),
        "geometry_projection_residual_coverage_fraction": round(residual_coverage, 9),
        "geometry_unknown_projection_residual_fraction": round(
            1.0 - residual_coverage,
            9,
        ),
        "mean_known_projection_union_error_fraction": round(
            _mean(item["union_error_fraction"] for item in residuals),
            9,
        ),
        "max_known_projection_union_error_fraction": round(
            _max(item["union_error_fraction"] for item in residuals),
            9,
        ),
        "mean_known_projection_silhouette_iou": round(
            _mean(item["silhouette_iou"] for item in residuals),
            9,
        ),
    }


def _score_terms(metrics: dict[str, float]) -> dict[str, float]:
    geometry_count = min(metrics["geometry_frame_count"] / 12.0, 1.0)
    texture_count = min(metrics["texture_frame_count"] / 14.0, 1.0)
    body_bbox = 1.0 - min(abs(metrics["mean_bbox_area_ratio"] - 0.34) / 0.34, 1.0)
    body_mask = 1.0 - min(abs(metrics["mean_mask_area_ratio"] - 0.18) / 0.18, 1.0)
    residual_coverage = metrics["geometry_projection_residual_coverage_fraction"]
    residual_mean = metrics["mean_known_projection_union_error_fraction"]
    residual_consistency = max(0.0, 1.0 - residual_mean) * residual_coverage
    residual_risk = max(0.0, residual_mean - 0.65) * residual_coverage
    return {
        "geometry_count": geometry_count * 0.75,
        "geometry_quality": metrics["mean_geometry_quality"] * 2.0,
        "body_bbox_fit": body_bbox * 1.0,
        "body_mask_fit": body_mask * 0.75,
        "photo_video_balance": metrics["media_balance_fraction"] * 0.75,
        "view_spread": metrics["geometry_view_spread_fraction"] * 0.65,
        "full_body_photo_evidence": min(
            metrics["geometry_full_body_photo_frame_count"] / 4.0,
            1.0,
        ),
        "texture_count": texture_count * 0.5,
        "face_texture_detail": metrics["texture_face_detail_fraction"] * 1.0,
        "full_body_photo_texture": min(
            metrics["texture_full_body_photo_frame_count"] / 2.0,
            1.0,
        )
        * 0.2,
        "border_risk": -metrics["mean_border_touch_fraction"] * 0.75,
        "rejected_geometry_risk": -metrics["geometry_rejected_fraction"] * 1.5,
        "relaxed_bbox_risk": -metrics["relaxed_bbox_fraction"] * 0.75,
        "geometry_closeup_risk": -metrics["geometry_closeup_photo_fraction"] * 0.75,
        "projection_residual_coverage": residual_coverage * 0.25,
        "projection_residual_consistency": residual_consistency * 0.75,
        "projection_residual_risk": -residual_risk * 2.0,
    }


def _reference_candidate_metrics(
    frames: list[dict[str, Any]],
    *,
    current_ids: tuple[str, ...],
    current_texture_ids: tuple[str, ...],
    residual_metrics_by_frame_id: dict[str, dict[str, float]],
) -> dict[str, float] | None:
    if not current_ids:
        return None
    by_id = {str(frame["frame_id"]): frame for frame in frames}
    geometry_ids = _ordered_existing_ids(frames, current_ids)
    texture_ids = _ordered_existing_ids(frames, current_texture_ids)
    geometry = [by_id[frame_id] for frame_id in geometry_ids if frame_id in by_id]
    texture = [by_id[frame_id] for frame_id in texture_ids if frame_id in by_id]
    if not geometry:
        return None
    return _candidate_metrics(
        geometry,
        texture,
        residual_metrics_by_frame_id=residual_metrics_by_frame_id,
    )


def _load_projection_residual_metrics(
    project_dir: Path,
    *,
    reference_project_dir: Path | None,
) -> tuple[dict[str, dict[str, float]], Path | None, Path | None]:
    residuals, report_path = _load_local_projection_residual_metrics(project_dir)
    if residuals or reference_project_dir is None:
        return (residuals, report_path, None)

    reference_project_dir = Path(reference_project_dir)
    reference_residuals, reference_report_path = _load_local_projection_residual_metrics(
        reference_project_dir,
    )
    if not reference_residuals:
        return ({}, reference_report_path, reference_project_dir)
    return (
        _map_projection_residual_metrics_by_observation_hash(
            reference_project_dir,
            project_dir,
            reference_residuals,
        ),
        reference_report_path,
        reference_project_dir,
    )


def _load_local_projection_residual_metrics(
    project_dir: Path,
) -> tuple[dict[str, dict[str, float]], Path | None]:
    quality_dir = Path(project_dir) / "outputs" / "quality"
    candidates = (
        quality_dir / "raw_silhouette_residuals.json",
        quality_dir / "raw_silhouette_residuals" / "manifest.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        report = _read_json(path)
        frames = report.get("frames", [])
        if not isinstance(frames, list):
            return ({}, path)
        residuals: dict[str, dict[str, float]] = {}
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_id = str(frame.get("frame_id", ""))
            if not frame_id:
                continue
            union_error = _finite_float(frame.get("union_error_fraction"))
            silhouette_iou = _finite_float(frame.get("silhouette_iou"))
            if union_error is None or silhouette_iou is None:
                continue
            residuals[frame_id] = {
                "union_error_fraction": union_error,
                "silhouette_iou": silhouette_iou,
            }
        return (residuals, path)
    return ({}, None)


def _map_projection_residual_metrics_by_observation_hash(
    reference_project_dir: Path,
    target_project_dir: Path,
    reference_residuals: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    reference_hashes = _observation_image_hashes(reference_project_dir)
    target_hashes = _observation_image_hashes(target_project_dir)
    residuals_by_hash: dict[str, dict[str, float]] = {}
    for frame_id in sorted(reference_residuals):
        digest = reference_hashes.get(frame_id)
        if digest is None:
            continue
        residuals_by_hash.setdefault(digest, reference_residuals[frame_id])
    return {
        frame_id: residuals_by_hash[digest]
        for frame_id, digest in sorted(target_hashes.items())
        if digest in residuals_by_hash
    }


def _observation_image_hashes(project_dir: Path) -> dict[str, str]:
    observations_dir = Path(project_dir) / "observations" / "raw"
    hashes: dict[str, str] = {}
    for observation_path in sorted(observations_dir.glob("*/observation.json")):
        observation = _read_json(observation_path)
        frame_id = str(observation.get("frame_id") or observation_path.parent.name)
        image_path_raw = observation.get("image_srgb_path")
        if not image_path_raw:
            continue
        image_path = _existing_manifest_image_path(
            image_path_raw,
            base_dir=observation_path.parent,
            frame_id=frame_id,
        )
        if image_path is None:
            continue
        hashes[frame_id] = _sha256_file(image_path)
    return hashes


def _texture_image_hashes(project_dir: Path) -> dict[str, str]:
    """Return hashes for texture frames, including texture-only manifest frames."""

    manifest_path = Path(project_dir) / "observations" / "raw" / "texture_manifest.json"
    if not manifest_path.is_file():
        return _observation_image_hashes(project_dir)
    manifest = _read_json(manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list):
        return _observation_image_hashes(project_dir)
    hashes = _observation_image_hashes(project_dir)
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_id = str(frame.get("frame_id", ""))
        if not frame_id:
            continue
        digest = _texture_frame_hash(frame, manifest_path=manifest_path)
        if digest is not None:
            hashes[frame_id] = digest
    return hashes


def _texture_frame_hash(frame: dict[str, Any], *, manifest_path: Path) -> str | None:
    image_path_raw = (
        frame.get("image_srgb_path")
        or frame.get("image_linear_path")
        or frame.get("raw_source_path")
    )
    if image_path_raw:
        image_path = _existing_manifest_image_path(
            image_path_raw,
            base_dir=manifest_path.parent,
            frame_id=str(frame.get("frame_id", "")),
        )
        if image_path is not None:
            return _sha256_file(image_path)
    digest = frame.get("texture_only_image_sha256") or frame.get("sha256")
    if isinstance(digest, str) and digest:
        return digest
    return None


def _baked_seed_texture_evidence_report(
    baseline_project_dir: Path,
    candidate_project_dir: Path,
    *,
    baseline_texture_hashes: set[str],
) -> dict[str, Any]:
    candidate_manifest_path = (
        Path(candidate_project_dir) / "outputs" / "textures" / "texture_manifest.json"
    )
    baseline_texture_dir = Path(baseline_project_dir) / "outputs" / "textures"
    if not candidate_manifest_path.is_file():
        return {
            "preserved": False,
            "reason": "missing_candidate_texture_manifest",
            "candidate_texture_manifest_path": str(candidate_manifest_path),
        }
    candidate_manifest = _read_json(candidate_manifest_path)
    basecolor = candidate_manifest.get("basecolor", {})
    if not isinstance(basecolor, dict):
        return {
            "preserved": False,
            "reason": "missing_candidate_basecolor_section",
            "candidate_texture_manifest_path": str(candidate_manifest_path),
        }
    seed = basecolor.get("baked_basecolor_seed", {})
    if not isinstance(seed, dict) or not bool(seed.get("applied", False)):
        return {
            "preserved": False,
            "reason": "baked_basecolor_seed_not_applied",
            "candidate_texture_manifest_path": str(candidate_manifest_path),
        }

    artifact_reports: list[dict[str, Any]] = []
    all_artifacts_preserved = True
    for seed_key, baseline_name in (
        ("basecolor_path", "basecolor.png"),
        ("confidence_path", "basecolor_confidence.png"),
        ("source_view_id_path", "source_view_id.png"),
    ):
        seed_path = _resolve_report_path(
            seed.get(seed_key),
            base_dir=candidate_manifest_path.parent,
        )
        baseline_path = baseline_texture_dir / baseline_name
        seed_exists = seed_path is not None and seed_path.is_file()
        baseline_exists = baseline_path.is_file()
        seed_sha = _sha256_file(seed_path) if seed_exists and seed_path is not None else ""
        baseline_sha = _sha256_file(baseline_path) if baseline_exists else ""
        hash_matches = bool(seed_sha and seed_sha == baseline_sha)
        all_artifacts_preserved = (
            all_artifacts_preserved and seed_exists and baseline_exists and hash_matches
        )
        artifact_reports.append(
            {
                "seed_key": seed_key,
                "baseline_name": baseline_name,
                "seed_path": str(seed_path) if seed_path is not None else "",
                "baseline_path": str(baseline_path),
                "seed_exists": bool(seed_exists),
                "baseline_exists": bool(baseline_exists),
                "sha256_matches_baseline": bool(hash_matches),
            }
        )

    seed_observed_texel_count = _optional_int(seed.get("seed_observed_texel_count")) or 0
    preserved = bool(all_artifacts_preserved and seed_observed_texel_count > 0)
    return {
        "preserved": preserved,
        "reason": "baseline_baked_texture_seed_preserved"
        if preserved
        else "baseline_baked_texture_seed_not_preserved",
        "version": "baked_basecolor_seed_texture_evidence_v1",
        "candidate_texture_manifest_path": str(candidate_manifest_path),
        "baseline_texture_dir": str(baseline_texture_dir),
        "seed_texture_dir": str(seed.get("seed_texture_dir", "")),
        "seed_observed_texel_count": int(seed_observed_texel_count),
        "baseline_unique_texture_hash_count": int(len(baseline_texture_hashes)),
        "baseline_texture_hashes_preserved_by_seed_count": int(
            len(baseline_texture_hashes) if preserved else 0
        ),
        "artifact_count": len(artifact_reports),
        "artifact_hashes_preserved": bool(all_artifacts_preserved),
        "artifacts": artifact_reports,
    }


def _resolve_report_path(path_raw: object, *, base_dir: Path) -> Path | None:
    if not isinstance(path_raw, str) or not path_raw:
        return None
    path = Path(path_raw)
    if path.is_absolute():
        return path
    return base_dir / path


def _existing_manifest_image_path(
    image_path_raw: object,
    *,
    base_dir: Path,
    frame_id: str,
) -> Path | None:
    image_path = Path(str(image_path_raw))
    candidates = [image_path]
    if not image_path.is_absolute():
        candidates.append(base_dir / image_path)
    candidates.append(base_dir / image_path.name)
    if frame_id:
        candidates.append(base_dir / frame_id / image_path.name)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _supplemental_texture_source_rejections(
    candidate_project_dir: Path,
    *,
    strict_regressed_metric_names: list[str],
) -> list[dict[str, Any]]:
    if not strict_regressed_metric_names:
        return []
    bake_path = (
        Path(candidate_project_dir) / "outputs" / "quality" / "baseline_geometry_texture_bake.json"
    )
    if not bake_path.is_file():
        return []
    bake_report = _read_json(bake_path)
    added_frames = bake_report.get("added_texture_frames", [])
    if not isinstance(added_frames, list):
        return []
    rejections: list[dict[str, Any]] = []
    for item in added_frames:
        if not isinstance(item, dict):
            continue
        rejections.append(
            {
                "frame_id": str(item.get("frame_id", "")),
                "source_frame_id": str(item.get("source_frame_id", "")),
                "sha256": str(item.get("sha256", "")),
                "media_type": str(item.get("media_type", "unknown")),
                "preflight_rank": _optional_int(item.get("preflight_rank")),
                "preflight_score": _optional_numeric_metric(item, "preflight_score"),
                "rejected": True,
                "reason": "strict_face_detail_metric_regression",
                "regressed_metric_names": list(strict_regressed_metric_names),
            }
        )
    return rejections


def _frame_hash_sequence(
    frame_ids: tuple[str, ...],
    hashes_by_frame_id: dict[str, str],
) -> tuple[str, ...]:
    return tuple(
        hashes_by_frame_id[frame_id] for frame_id in frame_ids if frame_id in hashes_by_frame_id
    )


def _frame_hash_set(
    frame_ids: tuple[str, ...],
    hashes_by_frame_id: dict[str, str],
) -> set[str]:
    return {
        hashes_by_frame_id[frame_id] for frame_id in frame_ids if frame_id in hashes_by_frame_id
    }


def _missing_hash_ids(
    frame_ids: tuple[str, ...],
    hashes_by_frame_id: dict[str, str],
) -> tuple[str, ...]:
    return tuple(frame_id for frame_id in frame_ids if frame_id not in hashes_by_frame_id)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alignment_risk_prefilter(
    metrics: dict[str, float],
    *,
    reference_metrics: dict[str, float] | None,
    is_reference: bool,
) -> dict[str, Any]:
    if reference_metrics is None or is_reference:
        return {
            "passed": True,
            "checks": {
                "border_profile_not_overcorrected": True,
                "center_span_preserved": True,
                "center_span_not_excessive": True,
                "aspect_span_preserved": True,
            },
            "rejection_reasons": [],
            "deltas": {},
            "thresholds": {},
        }

    reference_border = float(reference_metrics["mean_border_touch_fraction"])
    candidate_border = float(metrics["mean_border_touch_fraction"])
    reference_center_span = float(reference_metrics["geometry_horizontal_center_span"])
    candidate_center_span = float(metrics["geometry_horizontal_center_span"])
    reference_aspect_span = float(reference_metrics["geometry_bbox_aspect_span"])
    candidate_aspect_span = float(metrics["geometry_bbox_aspect_span"])

    min_center_span = reference_center_span * 0.75
    max_center_span = max(reference_center_span * 1.6, reference_center_span + 0.20)
    min_aspect_span = reference_aspect_span * 0.65
    overcorrected_border = (
        reference_border >= 0.45
        and candidate_border <= 0.10
        and candidate_aspect_span < min_aspect_span
    )
    checks = {
        "border_profile_not_overcorrected": not overcorrected_border,
        "center_span_preserved": (
            reference_center_span <= 1e-9 or candidate_center_span >= min_center_span
        ),
        "center_span_not_excessive": candidate_center_span <= max_center_span,
        "aspect_span_preserved": (
            reference_aspect_span <= 1e-9 or candidate_aspect_span >= min_aspect_span
        ),
    }
    reasons = [key for key, value in checks.items() if not value]
    return {
        "passed": not reasons,
        "checks": checks,
        "rejection_reasons": reasons,
        "deltas": {
            "mean_border_touch_fraction": round(candidate_border - reference_border, 9),
            "geometry_horizontal_center_span": round(
                candidate_center_span - reference_center_span,
                9,
            ),
            "geometry_bbox_aspect_span": round(
                candidate_aspect_span - reference_aspect_span,
                9,
            ),
        },
        "thresholds": {
            "reference_mean_border_touch_fraction": round(reference_border, 9),
            "overcorrection_border_floor": 0.45,
            "overcorrection_candidate_border_ceiling": 0.10,
            "min_geometry_horizontal_center_span": round(min_center_span, 9),
            "max_geometry_horizontal_center_span": round(max_center_span, 9),
            "min_geometry_bbox_aspect_span": round(min_aspect_span, 9),
        },
    }


def _alignment_risk_score_term(prefilter: dict[str, Any]) -> float:
    reasons = prefilter.get("rejection_reasons", [])
    if not isinstance(reasons, list) or not reasons:
        return 0.0
    return -2.0 * float(len(reasons))


def _rank_geometry_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        frames,
        key=lambda frame: (
            float(frame["score"]),
            float(frame["touches_image_border"]),
            float(frame["bbox_area_ratio"]),
            int(frame["source_index"]),
            str(frame["frame_id"]),
        ),
    )


def _balanced_frame_ids(
    ranked: list[dict[str, Any]],
    *,
    max_frames: int,
    min_frames: int,
) -> tuple[str, ...]:
    photos = [frame for frame in ranked if frame["media_type"] == "photo"]
    videos = [frame for frame in ranked if frame["media_type"] == "video"]
    if not photos or not videos:
        return _ids(ranked[:max_frames])
    photo_quota = max(2, min(4, max_frames // 3))
    photo_quota = min(photo_quota, len(photos), max_frames)
    video_quota = min(len(videos), max_frames - photo_quota)
    selected = _ids(photos[:photo_quota] + videos[:video_quota])
    selected = _top_up_ids(selected, ranked, max_frames)
    if len(selected) < min_frames:
        return _ids(ranked[:max_frames])
    return selected


def _view_diverse_body_ids(
    ranked: list[dict[str, Any]],
    *,
    max_frames: int,
    min_frames: int,
) -> tuple[str, ...]:
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for frame in ranked:
        by_bucket.setdefault(_frame_view_bucket(frame), []).append(frame)
    if len(by_bucket) < 2:
        return ()

    selected: list[str] = []
    selected_set: set[str] = set()
    bucket_items = sorted(
        by_bucket.items(),
        key=lambda item: (float(item[1][0]["score"]), str(item[0])),
    )
    for _bucket, bucket_frames in bucket_items:
        if len(selected) >= max_frames:
            break
        frame_id = str(bucket_frames[0]["frame_id"])
        if frame_id in selected_set:
            continue
        selected.append(frame_id)
        selected_set.add(frame_id)

    selected_tuple = _top_up_ids(tuple(selected), ranked, max_frames)
    selected_frames = [frame for frame in ranked if str(frame["frame_id"]) in set(selected_tuple)]
    if len(selected_tuple) < min_frames:
        return ()
    if len({_frame_view_bucket(frame) for frame in selected_frames}) < 2:
        return ()
    return selected_tuple


def _video_dense_ids(
    ranked: list[dict[str, Any]],
    *,
    max_frames: int,
    min_frames: int,
) -> tuple[str, ...]:
    photos = [frame for frame in ranked if frame["media_type"] == "photo"]
    videos = [frame for frame in ranked if frame["media_type"] == "video"]
    if not videos:
        return ()
    photo_quota = min(max(1, max_frames // 4), len(photos))
    selected = _ids(photos[:photo_quota] + videos[: max_frames - photo_quota])
    selected = _top_up_ids(selected, ranked, max_frames)
    return selected if len(selected) >= min_frames else ()


def _full_body_photo_video_ids(
    ranked: list[dict[str, Any]],
    *,
    max_frames: int,
    min_frames: int,
) -> tuple[str, ...]:
    body_photos = [frame for frame in ranked if _is_full_body_photo(frame)]
    videos = [frame for frame in ranked if frame["media_type"] == "video"]
    if not body_photos:
        return ()
    photo_quota = min(max(2, max_frames // 3), len(body_photos), max_frames)
    video_quota = min(len(videos), max_frames - photo_quota)
    selected = _ids(body_photos[:photo_quota] + videos[:video_quota])
    selected = _top_up_ids(selected, ranked, max_frames)
    return selected if len(selected) >= min_frames else ()


def _texture_ids(
    frames: list[dict[str, Any]],
    geometry_ids: tuple[str, ...],
) -> tuple[str, ...]:
    ids = set(geometry_ids)
    for frame in frames:
        if frame["media_type"] != "photo":
            continue
        if bool(frame["rejected_by_quality"]):
            continue
        ids.add(str(frame["frame_id"]))
    return _ordered_existing_ids(frames, tuple(ids))


def _top_up_ids(
    selected_ids: tuple[str, ...],
    ranked: list[dict[str, Any]],
    max_frames: int,
) -> tuple[str, ...]:
    selected = list(selected_ids)
    selected_set = set(selected)
    for frame in ranked:
        if len(selected) >= max_frames:
            break
        frame_id = str(frame["frame_id"])
        if frame_id in selected_set:
            continue
        selected.append(frame_id)
        selected_set.add(frame_id)
    return tuple(selected)


def _ids(frames: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(frame["frame_id"]) for frame in frames)


def _ordered_existing_ids(
    frames: list[dict[str, Any]],
    frame_ids: tuple[str, ...],
) -> tuple[str, ...]:
    frame_id_set = set(frame_ids)
    return tuple(
        str(frame["frame_id"]) for frame in frames if str(frame["frame_id"]) in frame_id_set
    )


def _dedupe_proposals(
    proposals: list[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    seen: set[tuple[frozenset[str], frozenset[str]]] = set()
    deduped = []
    for label, geometry_ids, texture_ids in proposals:
        key = (frozenset(geometry_ids), frozenset(texture_ids))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, geometry_ids, texture_ids))
    return deduped


def _relaxed_body_candidate(frame: dict[str, Any]) -> bool:
    reasons = set(frame["reject_reasons"])
    if {"mask_too_small", "bbox_too_small", "full_frame_bbox"} & reasons:
        return False
    if float(frame["mask_area_ratio"]) <= 0.05:
        return False
    if float(frame["bbox_area_ratio"]) <= 0.0:
        return False
    return True


def _is_closeup_photo(frame: dict[str, Any]) -> bool:
    return (
        frame["media_type"] == "photo"
        and not _is_full_body_photo(frame)
        and float(frame["bbox_area_ratio"]) >= 0.32
        and float(frame["touches_image_border"]) >= 0.5
    )


def _is_full_body_photo(frame: dict[str, Any]) -> bool:
    if frame["media_type"] != "photo":
        return False
    if not _relaxed_body_candidate(frame):
        return False
    bbox_area = float(frame["bbox_area_ratio"])
    mask_area = float(frame["mask_area_ratio"])
    aspect = float(frame["bbox_aspect"])
    touches_border = float(frame["touches_image_border"])
    if bbox_area >= 0.58 and 0.75 <= aspect <= 1.25:
        return False
    compact_body = bbox_area <= 0.32 and mask_area <= 0.20
    elongated_body = aspect <= 0.75 or aspect >= 1.35
    border_body = touches_border >= 0.5 and elongated_body and bbox_area <= 0.52
    return bool(compact_body or elongated_body or border_body)


def _media_counts(frames: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"photo": 0, "video": 0}
    for frame in frames:
        media_type = str(frame.get("media_type", "photo"))
        if media_type not in counts:
            counts[media_type] = 0
        counts[media_type] += 1
    return counts


def _reject_reason_counts(frames: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        for reason in frame["reject_reasons"]:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


def _media_balance_fraction(frames: list[dict[str, Any]]) -> float:
    counts = _media_counts(frames)
    photo = float(counts.get("photo", 0))
    video = float(counts.get("video", 0))
    total = photo + video
    if total <= 0.0:
        return 0.0
    if photo <= 0.0 or video <= 0.0:
        return 0.0
    return min(photo, video) / total


def _frame_view_bucket(frame: dict[str, Any]) -> str:
    center_x = float(frame["bbox_center"][0])
    aspect = float(frame["bbox_aspect"])
    if center_x <= 0.42:
        position = "left"
    elif center_x >= 0.58:
        position = "right"
    else:
        position = "center"

    if aspect >= 1.25:
        shape = "wide"
    elif aspect <= 0.75:
        shape = "narrow"
    else:
        shape = "balanced"
    return f"{position}_{shape}"


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(float(value) for value in items) / len(items))


def _max(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return max(items)


def _span(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(max(items) - min(items))


def _normalise_frame(frame: object) -> dict[str, Any]:
    if not isinstance(frame, dict):
        raise ValueError("frame selection frames must be objects")
    frame_id = str(frame.get("frame_id", ""))
    if not frame_id:
        raise ValueError("frame selection frame is missing frame_id")
    reject_reasons = frame.get("reject_reasons", [])
    if not isinstance(reject_reasons, list):
        reject_reasons = []
    bbox_center = _bbox_center(frame.get("bbox_center", [0.5, 0.5]))
    return {
        "frame_id": frame_id,
        "media_type": str(frame.get("media_type", "photo")),
        "source_index": int(frame.get("source_index", 0)),
        "score": float(frame.get("score", 1.0)),
        "rejected_by_quality": bool(frame.get("rejected_by_quality", False)),
        "reject_reasons": tuple(str(reason) for reason in reject_reasons),
        "mask_area_ratio": float(frame.get("mask_area_ratio", 0.0)),
        "bbox_area_ratio": float(frame.get("bbox_area_ratio", 0.0)),
        "touches_image_border": float(frame.get("touches_image_border", 0.0)),
        "bbox_center": bbox_center,
        "bbox_aspect": float(frame.get("bbox_aspect", 1.0)),
    }


def _bbox_center(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return (0.5, 0.5)
    return (float(value[0]), float(value[1]))


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _summary_path(project_dir: Path) -> Path:
    return Path(project_dir) / "outputs" / "quality" / "raw_pipeline_summary.json"


def _metrics(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"{summary_path} metrics must be a JSON object")
    return metrics


def _alignment_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "overlay_mean_silhouette_iou": _numeric_metric(
            metrics,
            "overlay_mean_silhouette_iou",
        ),
        "silhouette_residual_mean_union_error": _numeric_metric(
            metrics,
            "silhouette_residual_mean_union_error",
        ),
        "joint_projection_mean_keypoint_loss": _numeric_metric(
            metrics,
            "joint_projection_mean_keypoint_loss",
        ),
    }


def _alignment_deltas(
    alignment: dict[str, float],
    *,
    baseline_alignment: dict[str, float],
) -> dict[str, float]:
    return {
        "overlay_mean_silhouette_iou": float(
            alignment["overlay_mean_silhouette_iou"]
            - baseline_alignment["overlay_mean_silhouette_iou"]
        ),
        "silhouette_residual_mean_union_error": float(
            alignment["silhouette_residual_mean_union_error"]
            - baseline_alignment["silhouette_residual_mean_union_error"]
        ),
        "joint_projection_mean_keypoint_loss": float(
            alignment["joint_projection_mean_keypoint_loss"]
            - baseline_alignment["joint_projection_mean_keypoint_loss"]
        ),
    }


def _alignment_improvements(
    alignment_deltas: dict[str, float],
    *,
    metric_epsilon: float,
) -> dict[str, bool]:
    return {
        "overlay_mean_silhouette_iou": (
            alignment_deltas["overlay_mean_silhouette_iou"] > metric_epsilon
        ),
        "silhouette_residual_mean_union_error": (
            alignment_deltas["silhouette_residual_mean_union_error"] < -metric_epsilon
        ),
        "joint_projection_mean_keypoint_loss": (
            alignment_deltas["joint_projection_mean_keypoint_loss"] < -metric_epsilon
        ),
    }


def _body_fit_alignment_gate(
    metrics: dict[str, Any],
    *,
    baseline_metrics: dict[str, Any],
    alignment_gate: dict[str, Any],
    alignment_deltas: dict[str, float],
    require_body_fit_alignment: bool,
    min_body_fit_delta: float,
    metric_epsilon: float,
    is_baseline: bool,
) -> dict[str, Any]:
    baseline_body_fit = _optional_numeric_metric(
        baseline_metrics,
        "raw_geometry_selected_mean_body_fit_score",
    )
    candidate_body_fit = _optional_numeric_metric(
        metrics,
        "raw_geometry_selected_mean_body_fit_score",
    )
    body_fit_delta = (
        float(candidate_body_fit - baseline_body_fit)
        if baseline_body_fit is not None and candidate_body_fit is not None
        else 0.0
    )
    body_fit_improved = bool(body_fit_delta > min_body_fit_delta)
    improvements = _alignment_improvements(
        alignment_deltas,
        metric_epsilon=metric_epsilon,
    )
    improved_alignment_metric_count = sum(1 for value in improvements.values() if value)
    gate_required = bool(require_body_fit_alignment and body_fit_improved and not is_baseline)
    checks = {
        "body_fit_alignment_required": gate_required,
        "baseline_body_fit_present": baseline_body_fit is not None,
        "candidate_body_fit_present": candidate_body_fit is not None,
        "body_fit_improved": body_fit_improved,
        "alignment_not_regressed": (bool(alignment_gate["passed"]) if gate_required else True),
        "alignment_metric_improved": (
            improved_alignment_metric_count > 0 if gate_required else True
        ),
    }
    rejection_reasons = [
        key
        for key in ("alignment_not_regressed", "alignment_metric_improved")
        if not bool(checks[key])
    ]
    return {
        "passed": not rejection_reasons,
        "required": gate_required,
        "checks": checks,
        "rejection_reasons": rejection_reasons,
        "body_fit": {
            "metric": "raw_geometry_selected_mean_body_fit_score",
            "baseline": baseline_body_fit,
            "candidate": candidate_body_fit,
            "delta": body_fit_delta,
        },
        "alignment_improvements": improvements,
        "improved_alignment_metric_count": int(improved_alignment_metric_count),
        "promotion_recommendation": (
            "not_body_fit_geometry_change"
            if not gate_required
            else (
                "eligible_body_fit_frame_change"
                if not rejection_reasons
                else "keep_baseline_or_try_smaller_frame_change"
            )
        ),
    }


def _alignment_candidate_json(
    label: str,
    project_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    *,
    baseline_alignment: dict[str, float],
    baseline_metrics: dict[str, Any],
    max_overlay_iou_drop: float,
    max_silhouette_residual_increase: float,
    max_keypoint_loss_increase: float,
    require_body_fit_alignment: bool,
    min_body_fit_delta: float,
    metric_epsilon: float,
    is_baseline: bool,
) -> dict[str, Any]:
    alignment = _alignment_metrics(metrics)
    gate = _alignment_gate(
        alignment,
        baseline_alignment=baseline_alignment,
        max_overlay_iou_drop=max_overlay_iou_drop,
        max_silhouette_residual_increase=max_silhouette_residual_increase,
        max_keypoint_loss_increase=max_keypoint_loss_increase,
        summary_success=bool(summary.get("success", False)),
        is_baseline=is_baseline,
    )
    alignment_deltas = _alignment_deltas(
        alignment,
        baseline_alignment=baseline_alignment,
    )
    body_fit_gate = _body_fit_alignment_gate(
        metrics,
        baseline_metrics=baseline_metrics,
        alignment_gate=gate,
        alignment_deltas=alignment_deltas,
        require_body_fit_alignment=require_body_fit_alignment,
        min_body_fit_delta=min_body_fit_delta,
        metric_epsilon=metric_epsilon,
        is_baseline=is_baseline,
    )
    selected_frame_ids, texture_frame_ids = _selection_ids(project_dir)
    score_terms = _alignment_score_terms(
        metrics,
        gate_passed=bool(gate["passed"]) and bool(body_fit_gate["passed"]),
    )
    score = sum(score_terms.values())
    return {
        "label": label,
        "project_dir": str(project_dir),
        "summary_path": str(summary_path),
        "success": bool(summary.get("success", False)),
        "score": round(float(score), 9),
        "score_terms": {key: round(float(value), 9) for key, value in score_terms.items()},
        "alignment_gate": gate,
        "body_fit_alignment_gate": body_fit_gate,
        "alignment_metrics": alignment,
        "alignment_deltas": alignment_deltas,
        "selected_frame_ids": list(selected_frame_ids),
        "texture_frame_ids": list(texture_frame_ids),
        "selected_frame_count": len(selected_frame_ids),
        "texture_frame_count": len(texture_frame_ids),
        "metrics": {
            "raw_quality_passed_check_count": _numeric_metric(
                metrics,
                "raw_quality_passed_check_count",
            ),
            "raw_quality_total_check_count": _numeric_metric(
                metrics,
                "raw_quality_total_check_count",
            ),
            "basecolor_observed_texel_fraction": _numeric_metric(
                metrics,
                "basecolor_observed_texel_fraction",
            ),
            "observed_face_fraction": _numeric_metric(
                metrics,
                "observed_face_fraction",
            ),
            "raw_texture_frame_count": _numeric_metric(
                metrics,
                "raw_texture_frame_count",
            ),
            "raw_face_turntable_front_textured_mouth_line_pixel_count": (
                _numeric_metric(
                    metrics,
                    "raw_face_turntable_front_textured_mouth_line_pixel_count",
                )
            ),
        },
    }


def _alignment_gate(
    alignment: dict[str, float],
    *,
    baseline_alignment: dict[str, float],
    max_overlay_iou_drop: float,
    max_silhouette_residual_increase: float,
    max_keypoint_loss_increase: float,
    summary_success: bool,
    require_summary_success: bool = True,
    is_baseline: bool,
) -> dict[str, Any]:
    overlay_floor = baseline_alignment["overlay_mean_silhouette_iou"] - max_overlay_iou_drop
    residual_ceiling = (
        baseline_alignment["silhouette_residual_mean_union_error"]
        + max_silhouette_residual_increase
    )
    keypoint_ceiling = (
        baseline_alignment["joint_projection_mean_keypoint_loss"] + max_keypoint_loss_increase
    )
    checks = {
        "summary_success": (bool(summary_success) if require_summary_success else True),
        "overlay_not_lower": alignment["overlay_mean_silhouette_iou"] >= overlay_floor,
        "silhouette_residual_not_higher": (
            alignment["silhouette_residual_mean_union_error"] <= residual_ceiling
        ),
        "keypoint_loss_not_higher": (
            alignment["joint_projection_mean_keypoint_loss"] <= keypoint_ceiling
        ),
    }
    if is_baseline:
        checks = {key: True for key in checks}
    reasons = [key for key, value in checks.items() if not value]
    return {
        "passed": not reasons,
        "checks": checks,
        "rejection_reasons": reasons,
        "summary_success_required": bool(require_summary_success),
        "observed_summary_success": bool(summary_success),
        "thresholds": {
            "overlay_floor": float(overlay_floor),
            "silhouette_residual_ceiling": float(residual_ceiling),
            "keypoint_loss_ceiling": float(keypoint_ceiling),
        },
    }


def _alignment_score_terms(metrics: dict[str, Any], *, gate_passed: bool) -> dict[str, float]:
    quality_total = _numeric_metric(metrics, "raw_quality_total_check_count")
    quality_passed = _numeric_metric(metrics, "raw_quality_passed_check_count")
    quality_fraction = quality_passed / quality_total if quality_total > 0.0 else 0.0
    return {
        "alignment_gate": 100.0 if gate_passed else -100.0,
        "quality_pass_fraction": quality_fraction * 10.0,
        "basecolor_observed_texel_fraction": _numeric_metric(
            metrics,
            "basecolor_observed_texel_fraction",
        )
        * 1.5,
        "observed_face_fraction": _numeric_metric(metrics, "observed_face_fraction") * 0.75,
        "texture_frame_count": min(
            _numeric_metric(metrics, "raw_texture_frame_count") / 32.0,
            1.0,
        )
        * 0.3,
        "front_mouth_visibility": min(
            _numeric_metric(
                metrics,
                "raw_face_turntable_front_textured_mouth_line_pixel_count",
            )
            / 16.0,
            1.0,
        )
        * 0.2,
    }


def _selection_ids(project_dir: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selection_path = Path(project_dir) / "observations" / "raw" / "frame_selection.json"
    if not selection_path.is_file():
        return ((), ())
    selection = _read_json(selection_path)
    selected = tuple(str(item) for item in selection.get("selected_frame_ids", []))
    texture = tuple(str(item) for item in selection.get("texture_frame_ids", []))
    return selected, texture


def _numeric_metric(metrics: dict[str, Any], name: str) -> float:
    try:
        return float(metrics.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _texture_only_candidate_success(
    summary: dict[str, Any],
    metrics: dict[str, Any],
) -> bool:
    if summary.get("source") == "opencat3d.baseline_geometry_texture_bake":
        evaluation_success = _optional_numeric_metric(
            metrics,
            "raw_texture_only_evaluation_success",
        )
        if evaluation_success is not None:
            return evaluation_success > 0.5
    return bool(summary.get("success", False))


def _texture_only_candidate_success_policy(
    summary: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    evaluation_success = _optional_numeric_metric(
        metrics,
        "raw_texture_only_evaluation_success",
    )
    uses_texture_only_metric = (
        summary.get("source") == "opencat3d.baseline_geometry_texture_bake"
        and evaluation_success is not None
    )
    return {
        "source": (
            "raw_texture_only_evaluation_success"
            if uses_texture_only_metric
            else "summary.success"
        ),
        "summary_success": bool(summary.get("success", False)),
        "raw_texture_only_evaluation_success": evaluation_success,
    }


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _texture_only_metric_comparisons(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    *,
    metric_epsilon: float,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for metric in TEXTURE_ONLY_DETAIL_METRICS:
        baseline_value = _optional_numeric_metric(baseline_metrics, metric)
        candidate_value = _optional_numeric_metric(candidate_metrics, metric)
        if baseline_value is None and candidate_value is None:
            continue
        if baseline_value is None:
            status = "new"
            delta = None
        elif candidate_value is None:
            status = "missing_candidate"
            delta = None
        else:
            delta = float(candidate_value - baseline_value)
            if candidate_value + metric_epsilon < baseline_value:
                status = "regressed"
            elif candidate_value > baseline_value + metric_epsilon:
                status = "improved"
            else:
                status = "unchanged"
        comparisons.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
                "status": status,
            }
        )
    return comparisons


def _optional_numeric_metric(metrics: dict[str, Any], name: str) -> float | None:
    if name not in metrics:
        return None
    try:
        value = float(metrics[name])
    except (TypeError, ValueError):
        return None
    if not isfinite(value):
        return None
    return value


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric
