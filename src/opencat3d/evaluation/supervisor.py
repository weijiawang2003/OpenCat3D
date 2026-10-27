"""Deterministic candidate comparison for raw reconstruction loops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opencat3d.evaluation.frame_set_supervisor import (
    write_alignment_gated_frame_set_report,
)
from opencat3d.fitting.losses import face_region_detail_loss
from opencat3d.fitting.optimizer import load_fit_state, load_raw_fit_problem


@dataclass(frozen=True)
class CandidateComparisonResult:
    report_path: Path
    best_candidate: str
    candidate_count: int


@dataclass(frozen=True)
class GuardedCandidateComparisonResult:
    report_path: Path
    best_candidate: str
    candidate_count: int
    rejected_candidate_count: int
    score_report_path: Path
    alignment_gate_report_path: Path


SCORE_WEIGHTS = {
    "quality_pass_fraction": 10.0,
    "overlay_mean_silhouette_iou": 2.0,
    "basecolor_observed_texel_fraction": 1.5,
    "raw_texture_resolution_scale": 0.25,
    "observed_face_fraction": 0.75,
    "face_priority_texture_used": 1.0,
    "facial_feature_source_patch_view_weighting_used": 0.25,
    "facial_feature_source_patch_view_weighted_count": 0.25,
    "facial_feature_source_patch_mean_view_weight": 0.35,
    "facial_feature_source_patch_mean_frontal_face_confidence": 0.35,
    "facial_feature_source_patch_mean_view_weighted_rank": 0.45,
    "facial_feature_source_patch_mean_frontal_photo_priority": 0.45,
    "facial_feature_source_patch_mean_quality_score": 0.35,
    "facial_feature_count": 1.0,
    "facial_feature_painted_texel_count": 0.75,
    "facial_feature_whisker_pad_feature_count": 0.25,
    "facial_feature_whisker_pad_texel_fraction": 0.25,
    "face_feature_min_required_contrast": 0.75,
    "face_feature_iris_to_pupil_rgb_distance": 0.25,
    "glb_face_feature_eye_detail_luma_std": 0.45,
    "glb_face_feature_iris_to_pupil_rgb_distance": 0.35,
    "iris_halo_edge_to_core_ratio": -0.6,
    "glb_face_feature_material_visibility": 1.0,
    "glb_face_feature_visible_group_count": 0.5,
    "pbr_face_material_detail": 0.5,
    "pbr_face_material_detail_normal_variance": 0.25,
    "face_detail_geometry_applied": 0.75,
    "face_detail_deformed_vertex_count": 0.4,
    "face_detail_mean_displacement": 0.25,
    "face_surface_anchor_count": 0.5,
    "face_surface_anchor_vertex_count": 0.25,
    "face_turntable_required_anchor_visible_fraction": 0.75,
    "face_turntable_front_anchor_pixel_count": 0.25,
    "face_turntable_textured_written_count": 0.20,
    "face_turntable_front_textured_unique_rgb_count": 0.25,
    "face_turntable_front_textured_luma_std": 0.35,
    "face_turntable_front_textured_mouth_line_pixel_count": 0.25,
    "face_turntable_front_textured_required_feature_fraction": 0.30,
    "selected_frame_count": 0.5,
    "texture_video_frame_count": 0.5,
    "geometry_selected_body_priority_frame_count": 0.25,
    "geometry_selected_mean_body_fit_score": 0.25,
    "texture_only_detail_frame_count": 0.35,
    "texture_only_promotion_guard_failed": -3.0,
    "texture_source_admission_failed": -2.0,
    "face_region_detail_loss": -0.75,
    "silhouette_residual_mean_union_error": -1.0,
    "joint_projection_mean_keypoint_loss": -0.5,
}


def write_candidate_comparison_report(
    candidate_project_dirs: dict[str, Path],
    output_path: Path,
    *,
    force: bool = False,
) -> CandidateComparisonResult:
    """Rank raw reconstruction candidates by deterministic quality metrics."""

    if not candidate_project_dirs:
        raise ValueError("at least one candidate is required")
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True to overwrite")

    candidates = []
    for label, project_dir in sorted(candidate_project_dirs.items()):
        summary_path = Path(project_dir) / "outputs" / "quality" / "raw_pipeline_summary.json"
        summary = _read_json(summary_path)
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metrics = _candidate_metrics(Path(project_dir), metrics)
        score_terms = _score_terms(summary, metrics)
        score = sum(score_terms.values())
        candidates.append(
            {
                "label": label,
                "project_dir": str(project_dir),
                "summary_path": str(summary_path),
                "success": bool(summary.get("success", False)),
                "score": round(float(score), 9),
                "score_terms": {key: round(float(value), 9) for key, value in score_terms.items()},
                "metrics": _selected_metrics(metrics),
            }
        )

    candidates.sort(key=lambda item: (-float(item["score"]), str(item["label"])))
    best_candidate = str(candidates[0]["label"])
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.reconstruction_supervisor",
        "best_candidate": best_candidate,
        "candidate_count": len(candidates),
        "score_weights": SCORE_WEIGHTS,
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CandidateComparisonResult(
        report_path=output_path,
        best_candidate=best_candidate,
        candidate_count=len(candidates),
    )


def write_guarded_candidate_comparison_report(
    candidate_project_dirs: dict[str, Path],
    output_path: Path,
    *,
    baseline_label: str,
    score_report_path: Path | None = None,
    alignment_gate_report_path: Path | None = None,
    force: bool = False,
) -> GuardedCandidateComparisonResult:
    """Rank candidates, then gate promotion by alignment and body-fit evidence."""

    if not candidate_project_dirs:
        raise ValueError("at least one candidate is required")
    if baseline_label not in candidate_project_dirs:
        raise ValueError(f"baseline_label {baseline_label!r} is not a candidate label")
    non_baseline_dirs = {
        label: project_dir
        for label, project_dir in candidate_project_dirs.items()
        if label != baseline_label
    }
    if not non_baseline_dirs:
        raise ValueError("at least one non-baseline candidate is required")

    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True to overwrite")
    if score_report_path is None:
        score_report_path = output_path.with_name(f"{output_path.stem}_score.json")
    if alignment_gate_report_path is None:
        alignment_gate_report_path = output_path.with_name(
            f"{output_path.stem}_alignment_gate.json"
        )
    score_report_path = Path(score_report_path)
    alignment_gate_report_path = Path(alignment_gate_report_path)

    score_result = write_candidate_comparison_report(
        candidate_project_dirs,
        score_report_path,
        force=force,
    )
    alignment_result = write_alignment_gated_frame_set_report(
        Path(candidate_project_dirs[baseline_label]),
        {label: Path(project_dir) for label, project_dir in non_baseline_dirs.items()},
        alignment_gate_report_path,
        force=force,
    )
    score_report = _read_json(score_result.report_path)
    alignment_report = _read_json(alignment_result.report_path)
    gate_by_label = {
        str(candidate["label"]): candidate
        for candidate in alignment_report.get("candidates", [])
        if isinstance(candidate, dict)
    }

    guarded_candidates = []
    for rank, candidate in enumerate(score_report.get("candidates", [])):
        if not isinstance(candidate, dict):
            continue
        label = str(candidate.get("label", ""))
        gate_label = "baseline" if label == baseline_label else label
        gate = gate_by_label.get(gate_label)
        guard = _candidate_promotion_guard(candidate, gate)
        guarded_item = dict(candidate)
        guarded_item["score_rank"] = rank + 1
        guarded_item["promotion_guard"] = guard
        if gate is not None:
            guarded_item["alignment_gate_candidate_label"] = gate_label
            guarded_item["alignment_gate"] = gate.get("alignment_gate")
            guarded_item["body_fit_alignment_gate"] = gate.get(
                "body_fit_alignment_gate"
            )
        guarded_candidates.append(guarded_item)

    guarded_candidates.sort(
        key=lambda item: (
            not bool(item["promotion_guard"]["passed"]),
            int(item["score_rank"]),
            str(item["label"]),
        )
    )
    best_candidate = str(guarded_candidates[0]["label"])
    rejected_count = sum(
        1
        for item in guarded_candidates
        if not bool(item["promotion_guard"]["passed"])
    )
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.guarded_reconstruction_supervisor",
        "baseline_label": baseline_label,
        "best_candidate": best_candidate,
        "score_best_candidate": score_result.best_candidate,
        "alignment_gate_best_candidate": alignment_result.best_candidate,
        "candidate_count": len(guarded_candidates),
        "rejected_candidate_count": int(rejected_count),
        "score_report_path": str(score_result.report_path),
        "alignment_gate_report_path": str(alignment_result.report_path),
        "candidates": guarded_candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return GuardedCandidateComparisonResult(
        report_path=output_path,
        best_candidate=best_candidate,
        candidate_count=len(guarded_candidates),
        rejected_candidate_count=int(rejected_count),
        score_report_path=score_result.report_path,
        alignment_gate_report_path=alignment_result.report_path,
    )


def _candidate_promotion_guard(
    candidate: dict[str, Any],
    gate: dict[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "candidate_success": bool(candidate.get("success", False)),
        "alignment_gate_present": gate is not None,
        "alignment_gate_passed": bool(
            gate is not None
            and isinstance(gate.get("alignment_gate"), dict)
            and gate["alignment_gate"].get("passed", False)
        ),
        "body_fit_alignment_gate_passed": bool(
            gate is not None
            and isinstance(gate.get("body_fit_alignment_gate"), dict)
            and gate["body_fit_alignment_gate"].get("passed", False)
        ),
    }
    rejection_reasons = [key for key, value in checks.items() if not value]
    return {
        "passed": not rejection_reasons,
        "checks": checks,
        "rejection_reasons": rejection_reasons,
    }


def _score_terms(summary: dict[str, Any], metrics: dict[str, Any]) -> dict[str, float]:
    quality_total = float(metrics.get("raw_quality_total_check_count", 0.0))
    quality_passed = float(metrics.get("raw_quality_passed_check_count", 0.0))
    quality_fraction = quality_passed / quality_total if quality_total > 0.0 else 0.0
    if not bool(summary.get("success", False)):
        quality_fraction -= 1.0

    return {
        "quality_pass_fraction": quality_fraction * SCORE_WEIGHTS["quality_pass_fraction"],
        "overlay_mean_silhouette_iou": _metric(metrics, "overlay_mean_silhouette_iou")
        * SCORE_WEIGHTS["overlay_mean_silhouette_iou"],
        "basecolor_observed_texel_fraction": _metric(
            metrics,
            "basecolor_observed_texel_fraction",
        )
        * SCORE_WEIGHTS["basecolor_observed_texel_fraction"],
        "raw_texture_resolution_scale": min(
            _metric(metrics, "raw_texture_resolution_scale") / 2.0,
            1.0,
        )
        * SCORE_WEIGHTS["raw_texture_resolution_scale"],
        "observed_face_fraction": _metric(metrics, "observed_face_fraction")
        * SCORE_WEIGHTS["observed_face_fraction"],
        "face_priority_texture_used": _metric(metrics, "raw_face_priority_texture_used")
        * SCORE_WEIGHTS["face_priority_texture_used"],
        "facial_feature_source_patch_view_weighting_used": _metric(
            metrics,
            "raw_facial_feature_source_patch_view_weighting_used",
        )
        * SCORE_WEIGHTS["facial_feature_source_patch_view_weighting_used"],
        "facial_feature_source_patch_view_weighted_count": min(
            _metric(metrics, "raw_facial_feature_source_patch_view_weighted_count") / 12.0,
            1.0,
        )
        * SCORE_WEIGHTS["facial_feature_source_patch_view_weighted_count"],
        "facial_feature_source_patch_mean_view_weight": _metric(
            metrics,
            "raw_facial_feature_source_patch_mean_view_weight",
        )
        * SCORE_WEIGHTS["facial_feature_source_patch_mean_view_weight"],
        "facial_feature_source_patch_mean_frontal_face_confidence": _metric(
            metrics,
            "raw_facial_feature_source_patch_mean_frontal_face_confidence",
        )
        * SCORE_WEIGHTS["facial_feature_source_patch_mean_frontal_face_confidence"],
        "facial_feature_source_patch_mean_view_weighted_rank": _metric(
            metrics,
            "raw_facial_feature_source_patch_mean_view_weighted_rank",
        )
        * SCORE_WEIGHTS["facial_feature_source_patch_mean_view_weighted_rank"],
        "facial_feature_source_patch_mean_frontal_photo_priority": _metric(
            metrics,
            "raw_facial_feature_source_patch_mean_frontal_photo_priority",
        )
        * SCORE_WEIGHTS["facial_feature_source_patch_mean_frontal_photo_priority"],
        "facial_feature_source_patch_mean_quality_score": _metric(
            metrics,
            "raw_facial_feature_source_patch_mean_quality_score",
        )
        * SCORE_WEIGHTS["facial_feature_source_patch_mean_quality_score"],
        "facial_feature_count": min(
            _metric(metrics, "raw_facial_feature_count") / 10.0,
            1.0,
        )
        * SCORE_WEIGHTS["facial_feature_count"],
        "facial_feature_painted_texel_count": min(
            _metric(metrics, "raw_facial_feature_painted_texel_count") / 2048.0,
            1.0,
        )
        * SCORE_WEIGHTS["facial_feature_painted_texel_count"],
        "facial_feature_whisker_pad_feature_count": min(
            _metric(metrics, "raw_facial_feature_whisker_pad_feature_count") / 4.0,
            1.0,
        )
        * SCORE_WEIGHTS["facial_feature_whisker_pad_feature_count"],
        "facial_feature_whisker_pad_texel_fraction": min(
            _metric(metrics, "raw_facial_feature_whisker_pad_texel_fraction") / 0.08,
            1.0,
        )
        * SCORE_WEIGHTS["facial_feature_whisker_pad_texel_fraction"],
        "face_feature_min_required_contrast": min(
            _metric(metrics, "raw_face_feature_min_required_contrast") / 0.35,
            1.0,
        )
        * SCORE_WEIGHTS["face_feature_min_required_contrast"],
        "face_feature_iris_to_pupil_rgb_distance": min(
            _metric(metrics, "raw_face_feature_iris_to_pupil_rgb_distance") / 0.20,
            1.0,
        )
        * SCORE_WEIGHTS["face_feature_iris_to_pupil_rgb_distance"],
        "glb_face_feature_eye_detail_luma_std": min(
            _metric(metrics, "raw_glb_face_feature_eye_detail_luma_std") / 0.12,
            1.0,
        )
        * SCORE_WEIGHTS["glb_face_feature_eye_detail_luma_std"],
        "glb_face_feature_iris_to_pupil_rgb_distance": min(
            _metric(metrics, "raw_glb_face_feature_iris_to_pupil_rgb_distance") / 0.22,
            1.0,
        )
        * SCORE_WEIGHTS["glb_face_feature_iris_to_pupil_rgb_distance"],
        "iris_halo_edge_to_core_ratio": min(
            max(
                _metric(metrics, "raw_facial_feature_iris_halo_max_edge_to_core_ratio")
                - 1.25,
                0.0,
            )
            / 4.0,
            1.0,
        )
        * SCORE_WEIGHTS["iris_halo_edge_to_core_ratio"],
        "glb_face_feature_material_visibility": _metric(
            metrics,
            "raw_glb_face_feature_material_visibility_gate_valid",
        )
        * SCORE_WEIGHTS["glb_face_feature_material_visibility"],
        "glb_face_feature_visible_group_count": min(
            _metric(metrics, "raw_glb_face_feature_visible_group_count") / 3.0,
            1.0,
        )
        * SCORE_WEIGHTS["glb_face_feature_visible_group_count"],
        "pbr_face_material_detail": min(
            _metric(metrics, "raw_pbr_face_material_detail_texel_count") / 2048.0,
            1.0,
        )
        * _metric(metrics, "raw_pbr_face_material_detail_applied")
        * SCORE_WEIGHTS["pbr_face_material_detail"],
        "pbr_face_material_detail_normal_variance": min(
            _metric(metrics, "raw_pbr_face_material_detail_normal_variance") / 12.0,
            1.0,
        )
        * SCORE_WEIGHTS["pbr_face_material_detail_normal_variance"],
        "face_detail_geometry_applied": _metric(
            metrics,
            "raw_face_detail_geometry_applied",
        )
        * SCORE_WEIGHTS["face_detail_geometry_applied"],
        "face_detail_deformed_vertex_count": min(
            _metric(metrics, "raw_face_detail_deformed_vertex_count") / 200.0,
            1.0,
        )
        * SCORE_WEIGHTS["face_detail_deformed_vertex_count"],
        "face_detail_mean_displacement": min(
            _metric(metrics, "raw_face_detail_mean_displacement") / 0.025,
            1.0,
        )
        * SCORE_WEIGHTS["face_detail_mean_displacement"],
        "face_surface_anchor_count": min(
            _metric(metrics, "raw_face_surface_anchor_count") / 10.0,
            1.0,
        )
        * SCORE_WEIGHTS["face_surface_anchor_count"],
        "face_surface_anchor_vertex_count": min(
            _metric(metrics, "raw_face_surface_anchor_vertex_count") / 160.0,
            1.0,
        )
        * SCORE_WEIGHTS["face_surface_anchor_vertex_count"],
        "face_turntable_required_anchor_visible_fraction": _metric(
            metrics,
            "raw_face_turntable_required_anchor_visible_fraction",
        )
        * SCORE_WEIGHTS["face_turntable_required_anchor_visible_fraction"],
        "face_turntable_front_anchor_pixel_count": min(
            _metric(metrics, "raw_face_turntable_front_anchor_pixel_count") / 2048.0,
            1.0,
        )
        * SCORE_WEIGHTS["face_turntable_front_anchor_pixel_count"],
        "face_turntable_textured_written_count": min(
            _metric(metrics, "raw_face_turntable_textured_written_count") / 3.0,
            1.0,
        )
        * SCORE_WEIGHTS["face_turntable_textured_written_count"],
        "face_turntable_front_textured_unique_rgb_count": min(
            _metric(metrics, "raw_face_turntable_front_textured_unique_rgb_count") / 64.0,
            1.0,
        )
        * SCORE_WEIGHTS["face_turntable_front_textured_unique_rgb_count"],
        "face_turntable_front_textured_luma_std": min(
            _metric(metrics, "raw_face_turntable_front_textured_luma_std") / 0.08,
            1.0,
        )
        * SCORE_WEIGHTS["face_turntable_front_textured_luma_std"],
        "face_turntable_front_textured_mouth_line_pixel_count": min(
            _metric(
                metrics,
                "raw_face_turntable_front_textured_mouth_line_pixel_count",
            )
            / 16.0,
            1.0,
        )
        * SCORE_WEIGHTS["face_turntable_front_textured_mouth_line_pixel_count"],
        "face_turntable_front_textured_required_feature_fraction": _metric(
            metrics,
            "raw_face_turntable_front_textured_required_feature_fraction",
        )
        * SCORE_WEIGHTS["face_turntable_front_textured_required_feature_fraction"],
        "selected_frame_count": min(_metric(metrics, "raw_selected_frame_count") / 12.0, 1.0)
        * SCORE_WEIGHTS["selected_frame_count"],
        "texture_video_frame_count": min(
            _metric(metrics, "raw_texture_video_frame_count") / 8.0,
            1.0,
        )
        * SCORE_WEIGHTS["texture_video_frame_count"],
        "geometry_selected_body_priority_frame_count": min(
            _metric(metrics, "raw_geometry_selected_body_priority_frame_count") / 4.0,
            1.0,
        )
        * SCORE_WEIGHTS["geometry_selected_body_priority_frame_count"],
        "geometry_selected_mean_body_fit_score": _metric(
            metrics,
            "raw_geometry_selected_mean_body_fit_score",
        )
        * SCORE_WEIGHTS["geometry_selected_mean_body_fit_score"],
        "texture_only_detail_frame_count": min(
            _metric(metrics, "raw_texture_only_detail_frame_count") / 4.0,
            1.0,
        )
        * SCORE_WEIGHTS["texture_only_detail_frame_count"],
        "texture_only_promotion_guard_failed": (
            _metric(metrics, "raw_texture_only_promotion_guard_present")
            * (1.0 - _metric(metrics, "raw_texture_only_promotion_guard_passed"))
            * SCORE_WEIGHTS["texture_only_promotion_guard_failed"]
        ),
        "texture_source_admission_failed": (
            _metric(metrics, "raw_texture_source_admission_audit_present")
            * (1.0 - _metric(metrics, "raw_texture_source_admission_audit_passed"))
            * SCORE_WEIGHTS["texture_source_admission_failed"]
        ),
        "face_region_detail_loss": min(
            _metric(metrics, "raw_face_region_detail_loss"),
            1.0,
        )
        * SCORE_WEIGHTS["face_region_detail_loss"],
        "silhouette_residual_mean_union_error": _metric(
            metrics,
            "silhouette_residual_mean_union_error",
        )
        * SCORE_WEIGHTS["silhouette_residual_mean_union_error"],
        "joint_projection_mean_keypoint_loss": _metric(
            metrics,
            "joint_projection_mean_keypoint_loss",
        )
        * SCORE_WEIGHTS["joint_projection_mean_keypoint_loss"],
    }


def _selected_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    names = (
        "raw_quality_passed_check_count",
        "raw_quality_total_check_count",
        "raw_input_frame_count",
        "raw_selected_frame_count",
        "raw_texture_frame_count",
        "raw_texture_photo_frame_count",
        "raw_texture_video_frame_count",
        "raw_geometry_body_priority_candidate_count",
        "raw_geometry_selected_body_priority_frame_count",
        "raw_geometry_selected_mean_body_fit_score",
        "raw_geometry_selected_mean_face_texture_detail_score",
        "raw_texture_detail_candidate_count",
        "raw_texture_only_detail_frame_count",
        "raw_texture_only_promotion_guard_present",
        "raw_texture_only_promotion_guard_passed",
        "raw_texture_only_promotion_guard_regressed_metric_count",
        "raw_texture_only_promotion_guard_strict_regressed_metric_count",
        "raw_texture_only_promotion_guard_rejection_count",
        "raw_texture_source_admission_audit_present",
        "raw_texture_source_admission_audit_passed",
        "raw_texture_source_admission_rejection_count",
        "raw_texture_source_admission_unique_accepted_frame_count",
        "raw_texture_source_admission_protected_face_frame_count",
        "raw_texture_source_admission_rejected_ablation_frame_count",
        "raw_texture_source_admission_duplicate_frame_count",
        "raw_texture_source_admission_rejected_low_detail_frame_count",
        "raw_texture_source_admission_top_score",
        "overlay_mean_silhouette_iou",
        "silhouette_residual_mean_union_error",
        "joint_projection_mean_keypoint_loss",
        "basecolor_observed_texel_fraction",
        "raw_texture_width",
        "raw_texture_height",
        "raw_texture_resolution_scale",
        "raw_face_texture_target_bake_used",
        "raw_face_texture_target_reported_size",
        "raw_template_region_v2_texture_target_bake_used",
        "raw_template_region_v2_texture_target_reported_size",
        "raw_template_region_v2_texture_target_reported_scale",
        "raw_template_region_v2_texture_target_reported_island_count",
        "raw_template_region_v2_texture_target_reported_facial_island_count",
        "raw_template_region_v2_texture_target_reported_requested_texel_count",
        "raw_template_region_v2_texture_target_reported_allocated_texel_count",
        "raw_pbr_texture_width",
        "raw_pbr_texture_height",
        "raw_glb_texture_size",
        "raw_glb_texture_dimensions_valid",
        "raw_glb_pbr_texture_dimensions_valid",
        "raw_glb_basecolor_texture_valid",
        "raw_glb_pbr_textures_valid",
        "raw_glb_embedded_texture_dimension_gate_valid",
        "raw_glb_face_feature_required_group_count",
        "raw_glb_face_feature_visible_group_count",
        "raw_glb_face_feature_required_feature_count",
        "raw_glb_face_feature_present_required_feature_count",
        "raw_glb_face_feature_visible_feature_count",
        "raw_glb_face_feature_sampled_texel_count",
        "raw_glb_face_feature_min_visible_texel_count",
        "raw_glb_face_feature_mean_rgb_error",
        "raw_glb_face_feature_max_rgb_error",
        "raw_glb_face_feature_eye_visible",
        "raw_glb_face_feature_nose_visible",
        "raw_glb_face_feature_muzzle_visible",
        "raw_glb_face_feature_iris_to_pupil_rgb_distance",
        "raw_glb_face_feature_eye_detail_luma_std",
        "raw_glb_face_feature_nose_to_muzzle_rgb_distance",
        "raw_glb_face_feature_mouth_to_muzzle_rgb_distance",
        "raw_glb_face_feature_material_visibility_gate_valid",
        "observed_face_fraction",
        "observed_vertex_fraction",
        "raw_face_priority_texture_used",
        "raw_face_uv_region_count",
        "raw_face_uv_region_total_area",
        "raw_face_uv_region_min_density_weight",
        "raw_face_uv_region_contract_valid",
        "raw_template_region_v2_count",
        "raw_template_region_v2_required_region_count",
        "raw_template_region_v2_required_present_count",
        "raw_template_region_v2_contract_valid",
        "raw_glb_template_region_v2_contract_valid",
        "raw_template_region_v2_belly_vertex_count",
        "raw_face_uv_density_allocated_texel_count",
        "raw_face_uv_density_min_allocated_texels",
        "raw_face_uv_density_eye_min_allocated_texels",
        "raw_face_uv_density_nose_allocated_texels",
        "raw_face_uv_density_ear_inner_min_allocated_texels",
        "raw_template_region_v2_uv_region_count",
        "raw_template_region_v2_uv_required_present_count",
        "raw_template_region_v2_uv_island_count",
        "raw_template_region_v2_region_scoped_uv_island_count",
        "raw_template_region_v2_uv_island_allocated_texel_count",
        "raw_template_region_v2_uv_island_density_weighted_texel_count",
        "raw_template_region_v2_face_uv_island_count",
        "raw_template_region_v2_facial_uv_island_count",
        "raw_template_region_v2_facial_uv_island_allocated_texel_count",
        "raw_template_region_v2_body_region_uv_island_count",
        "raw_template_region_v2_eye_min_allocated_texels",
        "raw_template_region_v2_nose_allocated_texels",
        "raw_template_region_v2_muzzle_allocated_texels",
        "raw_template_region_v2_ear_inner_min_allocated_texels",
        "raw_template_region_v2_facial_density_weighted_texel_count",
        "raw_face_texture_target_size",
        "raw_face_texture_target_scale",
        "raw_face_texture_target_requested_texel_count",
        "raw_face_texture_target_allocated_texel_count",
        "raw_face_texture_target_eye_min_texels",
        "raw_face_texture_target_nose_texels",
        "raw_face_texture_target_ear_inner_min_texels",
        "raw_template_region_v2_texture_target_size",
        "raw_template_region_v2_texture_target_scale",
        "raw_template_region_v2_texture_target_island_count",
        "raw_template_region_v2_texture_target_facial_island_count",
        "raw_template_region_v2_texture_target_requested_texel_count",
        "raw_template_region_v2_texture_target_allocated_texel_count",
        "raw_template_region_v2_texture_target_eye_min_texels",
        "raw_template_region_v2_texture_target_nose_texels",
        "raw_template_region_v2_texture_target_muzzle_texels",
        "raw_template_region_v2_texture_target_ear_inner_min_texels",
        "raw_template_region_v2_texture_target_max_density_scale",
        "raw_face_texture_crop_region_count",
        "raw_face_texture_crop_texture_width",
        "raw_face_texture_crop_texture_height",
        "raw_face_texture_crop_compared_region_count",
        "raw_face_texture_crop_exact_source_patch_region_count",
        "raw_face_texture_crop_exact_source_patch_fraction",
        "raw_face_texture_crop_detail_region_count",
        "raw_face_texture_crop_min_unique_rgb_count",
        "raw_face_texture_crop_min_confident_texel_fraction",
        "raw_face_texture_crop_mean_confident_texel_fraction",
        "raw_face_texture_crop_mean_source_rgb_delta",
        "raw_face_texture_crop_isolated_region_count",
        "raw_face_texture_crop_overlap_pair_count",
        "raw_face_texture_crop_overlap_fraction",
        "raw_face_texture_crop_visual_quality_region_count",
        "raw_face_texture_crop_visual_quality_passed_count",
        "raw_face_texture_crop_visual_quality_fraction",
        "raw_face_texture_crop_visual_quality_min_score",
        "raw_face_texture_crop_visual_quality_mean_score",
        "raw_facial_feature_count",
        "raw_facial_feature_painted_texel_count",
        "raw_facial_feature_observed_texel_count",
        "raw_facial_feature_observed_texel_fraction",
        "raw_facial_feature_source_patch_count",
        "raw_facial_feature_source_patch_feedback_sample_count",
        "raw_facial_feature_source_patch_feedback_used_count",
        "raw_facial_feature_source_patch_bake_candidate_count",
        "raw_facial_feature_source_patch_bake_selected_feedback",
        "raw_facial_feature_source_patch_bake_visual_quality_passed_count",
        "raw_facial_feature_source_patch_bake_visual_quality_min_score",
        "raw_facial_feature_source_patch_bake_visual_quality_mean_score",
        "raw_facial_feature_whisker_pad_feature_count",
        "raw_facial_feature_whisker_pad_texel_count",
        "raw_facial_feature_whisker_pad_texel_fraction",
        "raw_facial_feature_iris_halo_feature_count",
        "raw_facial_feature_iris_halo_uv_support_texel_count",
        "raw_facial_feature_iris_halo_core_texel_count",
        "raw_facial_feature_iris_halo_edge_texel_count",
        "raw_facial_feature_iris_halo_accepted_core_texel_count",
        "raw_facial_feature_iris_halo_edge_fraction",
        "raw_facial_feature_iris_halo_edge_to_core_ratio",
        "raw_facial_feature_iris_halo_max_edge_to_core_ratio",
        "raw_facial_feature_iris_halo_support_to_accepted_core_ratio",
        "raw_facial_feature_left_iris_halo_edge_to_core_ratio",
        "raw_facial_feature_left_iris_halo_support_to_accepted_core_ratio",
        "raw_facial_feature_right_iris_halo_edge_to_core_ratio",
        "raw_facial_feature_right_iris_halo_support_to_accepted_core_ratio",
        "raw_face_texture_bake_comparison_candidate_count",
        "raw_face_texture_bake_comparison_region_count",
        "raw_face_texture_bake_comparison_selected_feedback",
        "raw_face_texture_bake_comparison_score_improved_region_count",
        "raw_face_texture_bake_comparison_coverage_improved_region_count",
        "raw_face_texture_bake_comparison_source_detail_improved_region_count",
        "raw_face_texture_bake_comparison_max_score_delta",
        "raw_face_texture_bake_comparison_max_coverage_delta",
        "raw_face_texture_bake_comparison_max_source_detail_delta",
        "raw_face_texture_bake_comparison_selected_not_lower_rank",
        "raw_facial_feature_source_patch_texel_count",
        "raw_facial_feature_source_patch_texel_fraction",
        "raw_facial_feature_source_patch_quality_passed_count",
        "raw_facial_feature_source_patch_quality_total_count",
        "raw_facial_feature_source_patch_quality_fraction",
        "raw_facial_feature_source_patch_min_quality_score",
        "raw_facial_feature_source_patch_mean_quality_score",
        "raw_facial_feature_source_patch_view_weighting_used",
        "raw_facial_feature_source_patch_view_weighted_count",
        "raw_facial_feature_source_patch_min_view_weight",
        "raw_facial_feature_source_patch_mean_view_weight",
        "raw_facial_feature_source_patch_min_frontal_face_confidence",
        "raw_facial_feature_source_patch_mean_frontal_face_confidence",
        "raw_facial_feature_source_patch_min_view_weighted_rank",
        "raw_facial_feature_source_patch_mean_view_weighted_rank",
        "raw_facial_feature_source_patch_frontal_photo_priority_used",
        "raw_facial_feature_source_patch_frontal_photo_priority_count",
        "raw_facial_feature_source_patch_max_frontal_photo_priority",
        "raw_facial_feature_source_patch_mean_frontal_photo_priority",
        "raw_facial_feature_source_patch_fusion_min_weight",
        "raw_facial_feature_source_patch_fusion_mean_weight",
        "raw_facial_feature_source_patch_fusion_weighted_texel_count",
        "raw_facial_feature_source_patch_nominal_texel_count",
        "raw_facial_feature_source_patch_footprint_reduction_fraction",
        "raw_facial_feature_source_patch_mean_footprint_area_scale",
        "raw_facial_feature_source_patch_min_footprint_area_scale",
        "raw_facial_feature_source_patch_shape_mask_texel_count",
        "raw_facial_feature_source_patch_shape_mask_fraction",
        "raw_facial_feature_source_patch_shape_alpha_mean",
        "raw_facial_feature_source_patch_source_mask_observed_texel_count",
        "raw_facial_feature_source_patch_source_mask_densified_texel_count",
        "raw_facial_feature_source_patch_source_mask_densification_gain",
        "raw_facial_feature_source_patch_mean_source_mask_densification_gain",
        "raw_facial_feature_source_patch_localization_candidate_texel_count",
        "raw_facial_feature_source_patch_localization_selected_texel_count",
        "raw_facial_feature_source_patch_localization_fraction",
        "raw_facial_feature_source_patch_mean_localization_fraction",
        "raw_facial_feature_source_patch_landmark_aligned_count",
        "raw_facial_feature_source_patch_mean_landmark_confidence",
        "raw_facial_feature_source_patch_observed_anchor_guided_count",
        "raw_facial_feature_source_patch_observed_anchor_guided_fraction",
        "raw_facial_feature_source_patch_landmark_destination_scaled_count",
        "raw_facial_feature_source_patch_mean_landmark_destination_scale",
        "raw_face_feature_destination_anchor_anatomical_count",
        "raw_face_feature_destination_anchor_order_valid",
        "raw_face_feature_destination_anchor_eye_to_nose_v_gap",
        "raw_face_feature_destination_anchor_nose_to_mouth_v_gap",
        "raw_face_feature_destination_anchor_nose_to_muzzle_v_gap",
        "raw_face_feature_destination_anchor_ear_to_eye_v_gap",
        "raw_face_feature_destination_anchor_eye_symmetry_u_error",
        "raw_facial_feature_source_patch_core_texel_count",
        "raw_facial_feature_source_patch_core_fraction",
        "raw_facial_feature_source_patch_feathered_edge_texel_count",
        "raw_facial_feature_source_patch_feathered_edge_fraction",
        "raw_facial_feature_source_patch_mean_blend_alpha",
        "raw_face_feature_min_required_contrast",
        "raw_face_feature_eye_to_muzzle_luma_delta",
        "raw_face_feature_mouth_to_muzzle_luma_delta",
        "raw_face_feature_nose_to_muzzle_rgb_distance",
        "raw_face_feature_iris_to_pupil_rgb_distance",
        "raw_face_feature_contrast_passed",
        "raw_pbr_face_material_detail_applied",
        "raw_pbr_face_material_detail_feature_count",
        "raw_pbr_face_material_detail_texel_count",
        "raw_pbr_face_material_detail_normal_variance",
        "raw_face_detail_geometry_applied",
        "raw_face_detail_deformed_vertex_count",
        "raw_face_detail_max_displacement",
        "raw_face_detail_mean_displacement",
        "raw_face_surface_anchor_count",
        "raw_face_surface_anchor_vertex_count",
        "raw_face_surface_anchor_max_displacement",
        "raw_face_turntable_view_count",
        "raw_face_turntable_front_visible_anchor_count",
        "raw_face_turntable_front_anchor_pixel_count",
        "raw_face_turntable_required_anchor_visible_fraction",
        "raw_face_turntable_textured_view_count",
        "raw_face_turntable_textured_written_count",
        "raw_face_turntable_front_textured_unique_rgb_count",
        "raw_face_turntable_front_textured_luma_std",
        "raw_face_turntable_front_textured_mouth_line_pixel_count",
        "raw_face_turntable_front_textured_required_feature_visible_count",
        "raw_face_turntable_front_textured_required_feature_fraction",
        "raw_face_turntable_mean_textured_unique_rgb_count",
        "raw_face_turntable_mean_textured_luma_std",
        "raw_face_texture_closeup_width",
        "raw_face_texture_closeup_height",
        "raw_face_texture_closeup_feature_count",
        "raw_face_texture_closeup_required_feature_count",
        "raw_face_texture_closeup_required_present_count",
        "raw_face_texture_closeup_required_feature_fraction",
        "raw_face_texture_closeup_crop_texel_count",
        "raw_face_texture_closeup_unique_rgb_count",
        "raw_face_texture_closeup_luma_std",
        "raw_face_texture_source_compare_feature_count",
        "raw_face_texture_source_compare_source_feature_available_count",
        "raw_face_texture_source_compare_required_feature_count",
        "raw_face_texture_source_compare_required_present_count",
        "raw_face_texture_source_compare_required_feature_fraction",
        "raw_face_texture_source_compare_mean_rgb_distance",
        "raw_face_texture_source_compare_max_rgb_distance",
        "raw_face_texture_source_compare_mean_luma_delta",
        "raw_face_texture_source_compare_max_luma_delta",
        "raw_face_texture_source_compare_source_mean_luma_std",
        "raw_face_texture_source_compare_baked_mean_luma_std",
        "raw_face_texture_source_compare_iris_feature_count",
        "raw_face_texture_source_compare_iris_candidate_texel_count",
        "raw_face_texture_source_compare_iris_core_accepted_texel_count",
        "raw_face_texture_source_compare_iris_core_densified_texel_count",
        "raw_face_texture_source_compare_iris_annulus_rejected_texel_count",
        "raw_face_texture_source_compare_iris_core_acceptance_fraction",
        "raw_face_texture_source_compare_iris_annulus_rejected_fraction",
        "raw_face_region_detail_loss",
        "final_loss",
        "loss_reduction_fraction",
    )
    return {name: _metric(metrics, name) for name in names if name in metrics}


def _candidate_metrics(project_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    merged = dict(metrics)
    merged.update(_texture_only_promotion_guard_metrics(project_dir))
    merged.update(_texture_source_admission_metrics(project_dir))
    if "raw_face_region_detail_loss" not in merged:
        merged.update(_computed_face_detail_metrics(project_dir, merged))
    return merged


def _texture_only_promotion_guard_metrics(project_dir: Path) -> dict[str, float]:
    guard_path = project_dir / "outputs" / "quality" / "texture_only_promotion_guard.json"
    if not guard_path.exists():
        return {}
    try:
        guard = _read_json(guard_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "raw_texture_only_promotion_guard_present": 1.0,
            "raw_texture_only_promotion_guard_passed": 0.0,
            "raw_texture_only_promotion_guard_regressed_metric_count": 1.0,
            "raw_texture_only_promotion_guard_strict_regressed_metric_count": 1.0,
            "raw_texture_only_promotion_guard_rejection_count": 1.0,
        }
    return {
        "raw_texture_only_promotion_guard_present": 1.0,
        "raw_texture_only_promotion_guard_passed": float(bool(guard.get("passed", False))),
        "raw_texture_only_promotion_guard_regressed_metric_count": float(
            len(_guard_string_list(guard.get("regressed_metric_names")))
        ),
        "raw_texture_only_promotion_guard_strict_regressed_metric_count": float(
            len(_guard_string_list(guard.get("strict_regressed_metric_names")))
        ),
        "raw_texture_only_promotion_guard_rejection_count": float(
            len(_guard_string_list(guard.get("rejection_reasons")))
        ),
    }


def _guard_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _texture_source_admission_metrics(project_dir: Path) -> dict[str, float]:
    audit_path = project_dir / "outputs" / "quality" / "texture_source_admission_audit.json"
    if not audit_path.exists():
        return {}
    try:
        audit = _read_json(audit_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "raw_texture_source_admission_audit_present": 1.0,
            "raw_texture_source_admission_audit_passed": 0.0,
            "raw_texture_source_admission_rejection_count": 1.0,
        }
    return {
        "raw_texture_source_admission_audit_present": 1.0,
        "raw_texture_source_admission_audit_passed": float(
            bool(audit.get("passed", False))
        ),
        "raw_texture_source_admission_rejection_count": float(
            len(_guard_string_list(audit.get("rejection_reasons")))
        ),
        "raw_texture_source_admission_unique_accepted_frame_count": float(
            _metric(audit, "unique_accepted_frame_count")
        ),
        "raw_texture_source_admission_protected_face_frame_count": float(
            _metric(audit, "protected_face_source_patch_frame_count")
        ),
        "raw_texture_source_admission_rejected_ablation_frame_count": float(
            _metric(audit, "rejected_ablation_frame_count")
        ),
        "raw_texture_source_admission_duplicate_frame_count": float(
            _metric(audit, "duplicate_frame_count")
        ),
        "raw_texture_source_admission_rejected_low_detail_frame_count": float(
            _metric(audit, "rejected_low_detail_frame_count")
        ),
        "raw_texture_source_admission_top_score": float(_metric(audit, "top_score")),
    }


def _computed_face_detail_metrics(
    project_dir: Path,
    metrics: dict[str, Any],
) -> dict[str, float]:
    try:
        problem = load_raw_fit_problem(project_dir)
        fit_state = load_fit_state(
            project_dir,
            mode="raw",
            expected_frame_count=int(_metric(metrics, "raw_selected_frame_count")),
            expected_joint_count=int(problem.template.joints_rest.shape[0]),
        )
        vertices = problem.template.shaped_vertices(fit_state.beta)
        target_vertices = problem.template.shaped_vertices(fit_state.target_beta)
        return {
            "raw_face_region_detail_loss": face_region_detail_loss(
                vertices,
                target_vertices,
                problem.template,
            )
        }
    except (FileNotFoundError, KeyError, ValueError):
        return {}


def _metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name, 0.0)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload
