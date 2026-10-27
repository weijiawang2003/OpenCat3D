"""Raw reconstruction quality report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opencat3d.adapters.dense_evidence import DENSE_EVIDENCE_MANIFEST_FILENAME
from opencat3d.evaluation.coverage import evaluate_mesh_coverage
from opencat3d.evaluation.face_texture_bake_comparison import (
    evaluate_face_texture_bake_comparison_report,
)
from opencat3d.evaluation.face_texture_diagnostics import (
    REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS,
    evaluate_direct_photo_face_candidate_report,
    evaluate_face_texture_diagnostics,
)
from opencat3d.evaluation.face_texture_targets import (
    evaluate_face_texture_targets_report,
)
from opencat3d.evaluation.face_uv_density import (
    REQUIRED_TEMPLATE_REGION_V2_UV_REGIONS,
    evaluate_face_uv_density_report,
)
from opencat3d.evaluation.glb_face_features import (
    GLB_FACE_FEATURE_MAX_FEATURE_RGB_ERROR,
    GLB_FACE_FEATURE_MAX_MEAN_RGB_ERROR,
    GLB_FACE_FEATURE_MIN_EYE_LUMA_STD,
    GLB_FACE_FEATURE_MIN_RGB_DISTANCE,
    GLB_FACE_FEATURE_REQUIRED_FEATURE_COUNT,
    GLB_FACE_FEATURE_REQUIRED_GROUP_COUNT,
    evaluate_glb_face_feature_visibility_report,
)
from opencat3d.evaluation.projection import evaluate_joint_projection_report
from opencat3d.evaluation.residuals import evaluate_silhouette_residual_maps
from opencat3d.evaluation.turntable import evaluate_face_turntable_report
from opencat3d.evaluation.validation import write_quality_validation_report
from opencat3d.export.mesh import (
    export_fitted_canonical_mesh,
    read_glb_metadata,
    read_obj,
    read_ply,
    validate_glb_structure,
    validate_reloaded_mesh,
)
from opencat3d.export.skeleton import validate_skeleton_metadata_file
from opencat3d.fitting.optimizer import FitState, load_fit_state
from opencat3d.io.cache import sha256_file
from opencat3d.schemas import (
    DenseEvidenceFrame,
    DenseEvidenceKind,
    DenseEvidenceManifest,
    TensorRef,
    read_json_model,
)
from opencat3d.template.cat_template import (
    CAT_TEMPLATE_FACE_COUNT,
    CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES,
    CAT_TEMPLATE_VERTEX_COUNT,
)

RAW_METRIC_THRESHOLDS: dict[str, float | int] = {
    "min_num_frames": 2,
    "expected_vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
    "expected_face_count": CAT_TEMPLATE_FACE_COUNT,
    "expected_joint_count": 11,
    "min_keypoint_loss_reduction": 1e-12,
    "min_silhouette_loss_reduction": 1e-12,
    "max_mean_mask_area_ratio": 0.80,
    "max_mean_bbox_area_ratio": 0.75,
    "max_full_frame_bbox_fraction": 0.35,
    "min_overlay_mean_silhouette_iou": 0.18,
    "max_silhouette_residual_mean_union_error": 0.85,
    "min_unknown_texture_luma_fraction_of_foreground": 0.65,
    "min_face_feature_required_contrast": 0.18,
    "min_pbr_face_material_detail_texel_count": 128,
    "min_raw_texture_size": 512,
    "min_face_uv_region_count": 8,
    "min_template_region_v2_count": len(CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES),
    "min_face_uv_eye_texels": 512,
    "min_face_uv_nose_texels": 128,
    "min_face_uv_ear_inner_texels": 512,
    "min_template_region_v2_uv_region_count": len(REQUIRED_TEMPLATE_REGION_V2_UV_REGIONS),
    "min_template_region_v2_face_uv_island_count": 8,
    "min_template_region_v2_eye_texels": 512,
    "min_template_region_v2_nose_texels": 128,
    "min_template_region_v2_ear_inner_texels": 512,
    "min_face_texture_target_size": 1024,
    "min_face_texture_target_scale": 1.0,
    "min_template_region_v2_texture_target_size": 2048,
    "min_template_region_v2_texture_target_scale": 1.0,
    "min_template_region_v2_texture_target_facial_island_count": 8,
    "min_template_region_v2_texture_target_baked_size": 2048,
    "min_face_texture_crop_detail_regions": len(REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS),
    "min_face_texture_exact_source_patch_regions": len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    ),
    "min_face_texture_isolated_crop_regions": len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    ),
    "max_face_texture_crop_overlap_pair_count": 0,
    "min_face_texture_visual_quality_regions": len(
        REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS
    ),
    "min_face_texture_visual_quality_score": 0.70,
    "min_facial_feature_observed_texel_fraction": 0.95,
    "min_facial_feature_source_patch_fraction": 0.95,
    "min_facial_feature_source_patch_quality_score": 0.04,
    "min_facial_feature_source_patch_quality_fraction": 1.0,
    "min_facial_feature_source_patch_fusion_weight": 0.35,
    "min_facial_feature_source_patch_feathered_edge_fraction": 0.05,
    "min_facial_feature_source_patch_core_fraction": 0.25,
    "max_facial_feature_source_patch_mean_footprint_area_scale": 1.0,
    "min_facial_feature_source_patch_shape_mask_fraction": 0.05,
    "min_facial_feature_source_patch_source_mask_densification_gain": 1.05,
    "max_facial_feature_source_patch_localization_fraction": 1.0,
    "min_facial_feature_source_patch_landmark_aligned_count": 1,
    "min_facial_feature_source_patch_landmark_destination_scaled_count": 1,
    "max_facial_feature_source_patch_mean_landmark_destination_scale": 1.0,
    "min_facial_feature_destination_anatomical_anchor_count": 12,
    "min_face_feature_destination_anchor_eye_to_nose_v_gap": 0.025,
    "min_face_feature_destination_anchor_nose_to_mouth_v_gap": 0.035,
    "max_face_feature_destination_anchor_eye_symmetry_u_error": 0.035,
    "min_glb_face_feature_visible_group_count": GLB_FACE_FEATURE_REQUIRED_GROUP_COUNT,
    "min_glb_face_feature_visible_feature_count": GLB_FACE_FEATURE_REQUIRED_FEATURE_COUNT,
    "max_glb_face_feature_mean_rgb_error": GLB_FACE_FEATURE_MAX_MEAN_RGB_ERROR,
    "max_glb_face_feature_max_rgb_error": GLB_FACE_FEATURE_MAX_FEATURE_RGB_ERROR,
    "min_glb_face_feature_rgb_distance": GLB_FACE_FEATURE_MIN_RGB_DISTANCE,
    "min_glb_face_feature_eye_luma_std": GLB_FACE_FEATURE_MIN_EYE_LUMA_STD,
}


@dataclass(frozen=True)
class RawEvaluationResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


def evaluate_raw_reconstruction(
    project_dir: Path,
    *,
    force: bool = False,
) -> RawEvaluationResult:
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"
    observation_manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    mesh_validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    obj_path = project_dir / "outputs" / "avatar" / "cat_canonical.obj"
    ply_path = project_dir / "outputs" / "avatar" / "cat_canonical.ply"
    glb_path = project_dir / "outputs" / "avatar" / "cat_avatar.glb"
    skeleton_path = project_dir / "outputs" / "avatar" / "skeleton.json"
    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / "raw_evaluation.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    fit_report = _read_json(fit_report_path)
    mesh_validation = _read_json(mesh_validation_path)
    coverage_result = evaluate_mesh_coverage(
        project_dir,
        mode="raw",
        reuse_existing_basecolor=True,
        force=force,
    )
    projection_result = evaluate_joint_projection_report(project_dir, mode="raw", force=force)
    residual_result = evaluate_silhouette_residual_maps(project_dir, mode="raw", force=force)
    turntable_result = evaluate_face_turntable_report(project_dir, mode="raw", force=force)
    face_uv_density_result = evaluate_face_uv_density_report(
        project_dir,
        mode="raw",
        force=force,
    )
    face_texture_targets_result = evaluate_face_texture_targets_report(
        project_dir,
        mode="raw",
        force=force,
    )
    (
        coverage_result,
        face_uv_density_result,
        face_texture_targets_result,
        texture_rebaked,
    ) = (
        _maybe_apply_template_region_v2_texture_target(
            project_dir,
            coverage_result=coverage_result,
            face_uv_density_result=face_uv_density_result,
            face_texture_targets_result=face_texture_targets_result,
        )
    )
    if texture_rebaked and _mesh_validation_allows_texture_export_refresh(
        mesh_validation,
    ):
        export_fitted_canonical_mesh(project_dir, mode="raw", force=True)
        mesh_validation = _read_json(mesh_validation_path)
    validation_result = write_quality_validation_report(project_dir, mode="raw", force=force)
    face_texture_diagnostic_result = evaluate_face_texture_diagnostics(
        project_dir,
        mode="raw",
        force=force,
    )
    direct_photo_face_candidate_result = evaluate_direct_photo_face_candidate_report(
        project_dir,
        mode="raw",
        force=force,
    )
    face_texture_bake_comparison_result = evaluate_face_texture_bake_comparison_report(
        project_dir,
        mode="raw",
        force=force,
    )
    glb_face_feature_visibility_result = evaluate_glb_face_feature_visibility_report(
        project_dir,
        mode="raw",
        force=force,
    )
    observation_metrics = _raw_observation_quality_metrics(observation_manifest_path)
    dense_evidence_metrics = _dense_evidence_metrics(
        project_dir,
        expected_frame_count=int(fit_report["num_frames"]),
    )
    texture_color_metrics = _texture_color_metrics(coverage_result.texture_manifest_path)
    expected_glb_texture_size = int(texture_color_metrics["basecolor_width"])
    metrics_raw = _as_object(fit_report.get("metrics"), "fit_report.metrics")
    output_paths = _as_object(fit_report.get("output_paths"), "fit_report.output_paths")
    overlay_manifest_path = Path(
        str(
            output_paths.get(
                "overlay_manifest",
                project_dir / "outputs" / "fitting" / "per_frame_overlay" / "manifest.json",
            )
        )
    )
    initial_loss = float(metrics_raw["initial_loss"])
    final_loss = float(metrics_raw["final_loss"])
    initial_keypoint_loss = float(metrics_raw["initial_keypoint_loss"])
    final_keypoint_loss = float(metrics_raw["final_keypoint_loss"])
    initial_silhouette_loss = float(metrics_raw["initial_silhouette_loss"])
    final_silhouette_loss = float(metrics_raw["final_silhouette_loss"])
    keypoint_loss_reduction = initial_keypoint_loss - final_keypoint_loss
    silhouette_loss_reduction = initial_silhouette_loss - final_silhouette_loss
    vertex_count = int(mesh_validation["vertex_count"])
    face_count = int(mesh_validation["face_count"])
    geometry_detail_metrics = _geometry_detail_metrics(mesh_validation)
    template_region_v2_metrics = _template_region_v2_metrics(mesh_validation)
    thresholds = dict(RAW_METRIC_THRESHOLDS)
    fit_state = load_fit_state(
        project_dir,
        mode="raw",
        expected_frame_count=int(fit_report["num_frames"]),
        expected_joint_count=int(mesh_validation["joint_count"]),
    )

    obj_mesh = read_obj(obj_path)
    ply_mesh = read_ply(ply_path)
    validate_reloaded_mesh(obj_mesh, ply_mesh)
    glb_metadata = read_glb_metadata(glb_path)
    glb_structure = validate_glb_structure(
        glb_path,
        expected_vertex_count=vertex_count,
        expected_face_count=face_count,
        expected_textured=True,
        expected_texture_size=expected_glb_texture_size,
        expected_pbr_textures=True,
        expected_skinned=True,
        expected_animated=True,
    )
    skeleton_report = validate_skeleton_metadata_file(
        skeleton_path,
        expected_mode="raw",
        expected_vertex_count=vertex_count,
        expected_face_count=face_count,
        expected_joint_count=int(thresholds["expected_joint_count"]),
    )
    fit_state_finite = _fit_state_finite(fit_state)
    overlay_valid, overlay_mean_silhouette_iou = _overlay_manifest_valid(
        overlay_manifest_path,
        expected_frame_count=int(fit_report["num_frames"]),
    )

    checks = {
        "fit_report_success": bool(fit_report["success"]),
        "raw_fit_mode": float(metrics_raw.get("fit_mode_raw", 0.0)) == 1.0,
        "multi_frame_input": int(fit_report["num_frames"]) >= int(thresholds["min_num_frames"]),
        "init_prediction_used": float(metrics_raw.get("init_prediction_used", 0.0)) == 1.0,
        "loss_finite": _is_finite(initial_loss) and _is_finite(final_loss),
        "loss_not_increased": final_loss <= initial_loss + 1e-12,
        "keypoint_loss_reduced": keypoint_loss_reduction
        >= float(thresholds["min_keypoint_loss_reduction"]),
        "silhouette_loss_reduced": silhouette_loss_reduction
        >= float(thresholds["min_silhouette_loss_reduction"]),
        "fit_state_finite": fit_state_finite,
        "obj_exists": obj_path.exists(),
        "ply_exists": ply_path.exists(),
        "glb_exists": glb_path.exists(),
        "skeleton_exists": skeleton_path.exists(),
        "obj_ply_reload_match": True,
        "glb_metadata_valid": (
            glb_metadata["vertex_count"] == vertex_count
            and glb_metadata["face_count"] == face_count
        ),
        "glb_structure_valid": glb_structure.success
        and bool(mesh_validation.get("glb_structure_valid", False)),
        "glb_basecolor_texture_valid": bool(
            mesh_validation.get("glb_basecolor_texture_valid", False)
            and glb_structure.checks.get("expected_textured", False)
            and glb_structure.checks.get("texture_material_valid", False)
            and glb_structure.checks.get("texture_dimensions_valid", False)
        ),
        "glb_pbr_textures_valid": bool(
            mesh_validation.get("glb_pbr_textures_valid", False)
            and glb_structure.checks.get("expected_pbr_textures", False)
            and glb_structure.checks.get("pbr_material_textures_valid", False)
            and glb_structure.checks.get("pbr_texture_dimensions_valid", False)
            and glb_structure.checks.get("pbr_metallic_factor_zero", False)
        ),
        "glb_face_detail_geometry_valid": bool(
            mesh_validation.get("glb_face_detail_geometry_valid", False)
        ),
        "raw_face_uv_regions_valid": bool(
            mesh_validation.get("glb_face_uv_regions_valid", False)
            and texture_color_metrics["face_uv_region_contract_valid"] == 1.0
            and int(texture_color_metrics["face_uv_region_count"])
            >= int(thresholds["min_face_uv_region_count"])
        ),
        "raw_face_uv_texture_density_report_valid": bool(
            face_uv_density_result.success
            and int(face_uv_density_result.metrics["face_uv_region_count"])
            >= int(thresholds["min_face_uv_region_count"])
            and int(face_uv_density_result.metrics["face_uv_eye_min_allocated_texels"])
            >= int(thresholds["min_face_uv_eye_texels"])
            and int(face_uv_density_result.metrics["face_uv_nose_allocated_texels"])
            >= int(thresholds["min_face_uv_nose_texels"])
            and int(
                face_uv_density_result.metrics["face_uv_ear_inner_min_allocated_texels"]
            )
            >= int(thresholds["min_face_uv_ear_inner_texels"])
        ),
        "raw_template_region_v2_uv_island_density_report_valid": bool(
            face_uv_density_result.success
            and int(face_uv_density_result.metrics["template_region_v2_uv_region_count"])
            >= int(thresholds["min_template_region_v2_uv_region_count"])
            and int(
                face_uv_density_result.metrics[
                    "template_region_v2_uv_required_present_count"
                ]
            )
            == int(
                face_uv_density_result.metrics[
                    "template_region_v2_uv_required_region_count"
                ]
            )
            and int(
                face_uv_density_result.metrics[
                    "template_region_v2_face_uv_island_count"
                ]
            )
            >= int(thresholds["min_template_region_v2_face_uv_island_count"])
            and int(
                face_uv_density_result.metrics[
                    "template_region_v2_eye_min_allocated_texels"
                ]
            )
            >= int(thresholds["min_template_region_v2_eye_texels"])
            and int(
                face_uv_density_result.metrics[
                    "template_region_v2_nose_allocated_texels"
                ]
            )
            >= int(thresholds["min_template_region_v2_nose_texels"])
            and int(
                face_uv_density_result.metrics[
                    "template_region_v2_ear_inner_min_allocated_texels"
                ]
            )
            >= int(thresholds["min_template_region_v2_ear_inner_texels"])
        ),
        "raw_template_region_v2_contract_valid": bool(
            template_region_v2_metrics["contract_valid"] == 1.0
            and template_region_v2_metrics["glb_contract_valid"] == 1.0
            and int(template_region_v2_metrics["region_count"])
            >= int(thresholds["min_template_region_v2_count"])
            and int(template_region_v2_metrics["required_present_count"])
            == int(template_region_v2_metrics["required_region_count"])
        ),
        "raw_face_texture_targets_report_valid": bool(
            face_texture_targets_result.success
            and int(face_texture_targets_result.metrics["target_texture_size"])
            >= int(thresholds["min_face_texture_target_size"])
            and float(face_texture_targets_result.metrics["target_texture_scale"])
            >= float(thresholds["min_face_texture_target_scale"])
            and int(
                face_texture_targets_result.metrics[
                    "face_texture_target_allocated_texel_count"
                ]
            )
            >= int(
                face_texture_targets_result.metrics[
                    "face_texture_target_requested_texel_count"
                ]
            )
        ),
        "raw_template_region_v2_texture_targets_report_valid": bool(
            face_texture_targets_result.success
            and int(
                face_texture_targets_result.metrics[
                    "template_region_v2_texture_target_size"
                ]
            )
            >= int(thresholds["min_template_region_v2_texture_target_size"])
            and float(
                face_texture_targets_result.metrics[
                    "template_region_v2_texture_target_scale"
                ]
            )
            >= float(thresholds["min_template_region_v2_texture_target_scale"])
            and int(
                face_texture_targets_result.metrics[
                    "template_region_v2_texture_target_facial_island_count"
                ]
            )
            >= int(
                thresholds[
                    "min_template_region_v2_texture_target_facial_island_count"
                ]
            )
            and int(
                face_texture_targets_result.metrics[
                    "template_region_v2_texture_target_allocated_texel_count"
                ]
            )
            >= int(
                face_texture_targets_result.metrics[
                    "template_region_v2_texture_target_requested_texel_count"
                ]
            )
            and int(
                face_texture_targets_result.metrics[
                    "template_region_v2_texture_target_eye_min_texels"
                ]
            )
            >= int(
                face_uv_density_result.metrics[
                    "template_region_v2_eye_min_allocated_texels"
                ]
            )
            and int(
                face_texture_targets_result.metrics[
                    "template_region_v2_texture_target_nose_texels"
                ]
            )
            >= int(
                face_uv_density_result.metrics[
                    "template_region_v2_nose_allocated_texels"
                ]
            )
            and int(
                face_texture_targets_result.metrics[
                    "template_region_v2_texture_target_muzzle_texels"
                ]
            )
            >= int(
                face_uv_density_result.metrics[
                    "template_region_v2_muzzle_allocated_texels"
                ]
            )
            and int(
                face_texture_targets_result.metrics[
                    "template_region_v2_texture_target_ear_inner_min_texels"
                ]
            )
            >= int(
                face_uv_density_result.metrics[
                    "template_region_v2_ear_inner_min_allocated_texels"
                ]
            )
        ),
        "raw_face_texture_target_bake_consistent": bool(
            texture_color_metrics["face_texture_target_bake_used"] == 0.0
            or (
                int(texture_color_metrics["basecolor_width"])
                >= int(texture_color_metrics["face_texture_target_size"])
                >= int(thresholds["min_face_texture_target_size"])
                and int(texture_color_metrics["pbr_width"])
                == int(texture_color_metrics["basecolor_width"])
                and float(texture_color_metrics["basecolor_generated_texel_fraction"]) == 0.0
            )
        ),
        "raw_template_region_v2_texture_target_bake_consistent": bool(
            texture_color_metrics["template_region_v2_texture_target_bake_used"] == 1.0
            and int(texture_color_metrics["basecolor_width"])
            >= int(texture_color_metrics["template_region_v2_texture_target_size"])
            >= int(thresholds["min_template_region_v2_texture_target_baked_size"])
            and int(texture_color_metrics["pbr_width"])
            == int(texture_color_metrics["basecolor_width"])
            and float(texture_color_metrics["basecolor_generated_texel_fraction"]) == 0.0
            and int(
                texture_color_metrics[
                    "template_region_v2_texture_target_allocated_texel_count"
                ]
            )
            >= int(
                texture_color_metrics[
                    "template_region_v2_texture_target_requested_texel_count"
                ]
            )
        ),
        "raw_face_texture_crop_diagnostic_valid": bool(
            face_texture_diagnostic_result.success
            and int(
                face_texture_diagnostic_result.metrics[
                    "face_texture_crop_detail_region_count"
                ]
            )
            >= int(thresholds["min_face_texture_crop_detail_regions"])
            and int(
                face_texture_diagnostic_result.metrics[
                    "face_texture_crop_compared_region_count"
                ]
            )
            == len(REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS)
        ),
        "raw_face_texture_crop_exact_source_patches_valid": bool(
            int(
                face_texture_diagnostic_result.metrics[
                    "face_texture_crop_exact_source_patch_region_count"
                ]
            )
            >= int(thresholds["min_face_texture_exact_source_patch_regions"])
        ),
        "raw_face_texture_crop_isolation_valid": bool(
            int(
                face_texture_diagnostic_result.metrics[
                    "face_texture_crop_isolated_region_count"
                ]
            )
            >= int(thresholds["min_face_texture_isolated_crop_regions"])
            and int(
                face_texture_diagnostic_result.metrics[
                    "face_texture_crop_overlap_pair_count"
                ]
            )
            <= int(thresholds["max_face_texture_crop_overlap_pair_count"])
        ),
        "raw_face_texture_crop_visual_quality_valid": bool(
            int(
                face_texture_diagnostic_result.metrics[
                    "face_texture_crop_visual_quality_passed_count"
                ]
            )
            >= int(thresholds["min_face_texture_visual_quality_regions"])
            and float(
                face_texture_diagnostic_result.metrics[
                    "face_texture_crop_visual_quality_min_score"
                ]
            )
            >= float(thresholds["min_face_texture_visual_quality_score"])
        ),
        "raw_direct_photo_face_candidate_diagnostic_valid": bool(
            direct_photo_face_candidate_result.success
        ),
        "raw_face_texture_bake_comparison_valid": bool(
            face_texture_bake_comparison_result.success
            and int(
                face_texture_bake_comparison_result.metrics[
                    "compared_region_count"
                ]
            )
            >= len(REQUIRED_FACE_TEXTURE_DIAGNOSTIC_REGIONS)
            and float(
                face_texture_bake_comparison_result.metrics[
                    "selected_candidate_not_lower_rank"
                ]
            )
            == 1.0
        ),
        "raw_glb_face_feature_material_visibility": bool(
            glb_face_feature_visibility_result.success
            and int(
                glb_face_feature_visibility_result.metrics[
                    "visible_group_count"
                ]
            )
            >= int(thresholds["min_glb_face_feature_visible_group_count"])
            and int(
                glb_face_feature_visibility_result.metrics[
                    "visible_feature_count"
                ]
            )
            >= int(thresholds["min_glb_face_feature_visible_feature_count"])
            and float(
                glb_face_feature_visibility_result.metrics["mean_rgb_error"]
            )
            <= float(thresholds["max_glb_face_feature_mean_rgb_error"])
            and float(
                glb_face_feature_visibility_result.metrics["max_rgb_error"]
            )
            <= float(thresholds["max_glb_face_feature_max_rgb_error"])
            and float(
                glb_face_feature_visibility_result.metrics[
                    "iris_to_pupil_rgb_distance"
                ]
            )
            >= float(thresholds["min_glb_face_feature_rgb_distance"])
            and float(
                glb_face_feature_visibility_result.metrics[
                    "nose_to_muzzle_rgb_distance"
                ]
            )
            >= float(thresholds["min_glb_face_feature_rgb_distance"])
            and float(
                glb_face_feature_visibility_result.metrics[
                    "mouth_to_muzzle_rgb_distance"
                ]
            )
            >= float(thresholds["min_glb_face_feature_rgb_distance"])
        ),
        "glb_skinning_valid": bool(
            mesh_validation.get("glb_skinning_valid", False)
            and glb_structure.checks.get("expected_skinned", False)
            and glb_structure.checks.get("skin_valid", False)
            and glb_structure.checks.get("weights_normalized", False)
        ),
        "glb_idle_animation_valid": bool(
            mesh_validation.get("glb_idle_animation_valid", False)
            and glb_structure.checks.get("expected_animated", False)
            and glb_structure.checks.get("animation_valid", False)
            and glb_structure.checks.get("animation_rotations_normalized", False)
        ),
        "template_vertex_count": vertex_count == int(thresholds["expected_vertex_count"]),
        "template_face_count": face_count == int(thresholds["expected_face_count"]),
        "mesh_vertices_finite": bool(mesh_validation["finite_vertices"]),
        "mesh_faces_valid": bool(mesh_validation["valid_faces"]),
        "skeleton_metadata_valid": skeleton_report.success
        and bool(mesh_validation.get("skeleton_metadata_valid", False)),
        "skeleton_skin_weights_valid": bool(
            skeleton_report.checks.get("skin_weight_row_sums_valid", False)
            and skeleton_report.checks.get("skin_weight_indices_valid", False)
            and skeleton_report.checks.get("skin_weight_values_valid", False)
        ),
        "fit_overlays_present": overlay_valid,
        "raw_mask_area_reasonable": observation_metrics["mean_mask_area_ratio"]
        <= float(thresholds["max_mean_mask_area_ratio"]),
        "raw_bbox_area_reasonable": observation_metrics["mean_bbox_area_ratio"]
        <= float(thresholds["max_mean_bbox_area_ratio"]),
        "raw_full_frame_bbox_fraction_reasonable": observation_metrics[
            "full_frame_bbox_fraction"
        ]
        <= float(thresholds["max_full_frame_bbox_fraction"]),
        "raw_overlay_alignment_quality": overlay_mean_silhouette_iou
        >= float(thresholds["min_overlay_mean_silhouette_iou"]),
        "raw_silhouette_residual_quality": float(
            residual_result.metrics["mean_union_error_fraction"]
        )
        <= float(thresholds["max_silhouette_residual_mean_union_error"]),
        "raw_texture_unknown_color_reasonable": (
            texture_color_metrics["foreground_fallback_luma"] <= 0.50
            or texture_color_metrics["unknown_luma_fraction_of_foreground"]
            >= float(thresholds["min_unknown_texture_luma_fraction_of_foreground"])
        ),
        "raw_face_feature_contrast_quality": texture_color_metrics[
            "face_feature_min_required_contrast"
        ]
        >= float(thresholds["min_face_feature_required_contrast"])
        and texture_color_metrics["face_feature_required_contrast_passed"] == 1.0,
        "raw_face_feature_observed_provenance_quality": (
            texture_color_metrics["facial_feature_observed_texel_fraction"]
            >= float(thresholds["min_facial_feature_observed_texel_fraction"])
            and texture_color_metrics["facial_feature_observed_texel_count"]
            >= texture_color_metrics["facial_feature_painted_texel_count"]
        ),
        "raw_face_feature_source_patch_quality": (
            texture_color_metrics["facial_feature_source_patch_texel_fraction"]
            >= float(thresholds["min_facial_feature_source_patch_fraction"])
            and texture_color_metrics["facial_feature_source_patch_texel_count"]
            >= texture_color_metrics["facial_feature_painted_texel_count"]
        ),
        "raw_face_feature_source_patch_quality_score": (
            texture_color_metrics["facial_feature_source_patch_min_quality_score"]
            >= float(thresholds["min_facial_feature_source_patch_quality_score"])
            and texture_color_metrics["facial_feature_source_patch_quality_fraction"]
            >= float(thresholds["min_facial_feature_source_patch_quality_fraction"])
        ),
        "raw_face_feature_source_patch_fusion_weight": (
            texture_color_metrics["facial_feature_source_patch_fusion_min_weight"]
            >= float(thresholds["min_facial_feature_source_patch_fusion_weight"])
            and texture_color_metrics[
                "facial_feature_source_patch_fusion_weighted_texel_count"
            ]
            > 0.0
        ),
        "raw_face_feature_source_patch_feathering": (
            texture_color_metrics[
                "facial_feature_source_patch_feathered_edge_fraction"
            ]
            >= float(
                thresholds["min_facial_feature_source_patch_feathered_edge_fraction"]
            )
            and texture_color_metrics["facial_feature_source_patch_mean_blend_alpha"]
            > 0.0
        ),
        "raw_face_feature_source_patch_core_preserved": (
            texture_color_metrics["facial_feature_source_patch_core_fraction"]
            >= float(thresholds["min_facial_feature_source_patch_core_fraction"])
        ),
        "raw_face_feature_source_patch_footprint_adaptive": (
            texture_color_metrics["facial_feature_source_patch_nominal_texel_count"]
            >= texture_color_metrics["facial_feature_source_patch_texel_count"]
            and texture_color_metrics[
                "facial_feature_source_patch_mean_footprint_area_scale"
            ]
            <= float(
                thresholds[
                    "max_facial_feature_source_patch_mean_footprint_area_scale"
                ]
            )
            and texture_color_metrics[
                "facial_feature_source_patch_min_footprint_area_scale"
            ]
            > 0.0
        ),
        "raw_face_feature_source_patch_shape_masked": (
            texture_color_metrics["facial_feature_source_patch_shape_mask_texel_count"]
            > 0.0
            and texture_color_metrics[
                "facial_feature_source_patch_shape_mask_fraction"
            ]
            >= float(
                thresholds["min_facial_feature_source_patch_shape_mask_fraction"]
            )
            and texture_color_metrics[
                "facial_feature_source_patch_shape_mask_fraction"
            ]
            <= 1.0
            and texture_color_metrics["facial_feature_source_patch_shape_alpha_mean"]
            > 0.0
        ),
        "raw_face_feature_source_patch_source_mask_densified": (
            texture_color_metrics[
                "facial_feature_source_patch_source_mask_densified_texel_count"
            ]
            >= texture_color_metrics[
                "facial_feature_source_patch_source_mask_observed_texel_count"
            ]
            and texture_color_metrics[
                "facial_feature_source_patch_source_mask_densification_gain"
            ]
            >= float(
                thresholds[
                    "min_facial_feature_source_patch_source_mask_densification_gain"
                ]
            )
        ),
        "raw_face_feature_source_patch_component_localized": (
            texture_color_metrics[
                "facial_feature_source_patch_localization_candidate_texel_count"
            ]
            >= texture_color_metrics[
                "facial_feature_source_patch_localization_selected_texel_count"
            ]
            > 0.0
            and texture_color_metrics[
                "facial_feature_source_patch_localization_fraction"
            ]
            <= float(
                thresholds[
                    "max_facial_feature_source_patch_localization_fraction"
                ]
            )
            and texture_color_metrics[
                "facial_feature_source_patch_mean_localization_fraction"
            ]
            <= float(
                thresholds[
                    "max_facial_feature_source_patch_localization_fraction"
                ]
            )
        ),
        "raw_face_feature_source_patch_landmark_aligned": (
            texture_color_metrics[
                "facial_feature_source_patch_landmark_aligned_count"
            ]
            >= float(
                thresholds[
                    "min_facial_feature_source_patch_landmark_aligned_count"
                ]
            )
            and texture_color_metrics[
                "facial_feature_source_patch_mean_landmark_confidence"
            ]
            > 0.0
        ),
        "raw_face_feature_source_patch_landmark_destination_scaled": (
            texture_color_metrics[
                "facial_feature_source_patch_landmark_destination_scaled_count"
            ]
            >= float(
                thresholds[
                    "min_facial_feature_source_patch_landmark_destination_scaled_count"
                ]
            )
            and 0.0
            < texture_color_metrics[
                "facial_feature_source_patch_mean_landmark_destination_scale"
            ]
            <= float(
                thresholds[
                    "max_facial_feature_source_patch_mean_landmark_destination_scale"
                ]
            )
        ),
        "raw_face_feature_destination_anchor_anatomical": (
            texture_color_metrics[
                "facial_feature_destination_anchor_anatomical_count"
            ]
            >= float(
                thresholds[
                    "min_facial_feature_destination_anatomical_anchor_count"
                ]
            )
            and texture_color_metrics[
                "facial_feature_destination_anchor_order_valid"
            ]
            == 1.0
            and texture_color_metrics[
                "facial_feature_destination_anchor_eye_to_nose_v_gap"
            ]
            >= float(
                thresholds[
                    "min_face_feature_destination_anchor_eye_to_nose_v_gap"
                ]
            )
            and texture_color_metrics[
                "facial_feature_destination_anchor_nose_to_mouth_v_gap"
            ]
            >= float(
                thresholds[
                    "min_face_feature_destination_anchor_nose_to_mouth_v_gap"
                ]
            )
            and texture_color_metrics[
                "facial_feature_destination_anchor_eye_symmetry_u_error"
            ]
            <= float(
                thresholds[
                    "max_face_feature_destination_anchor_eye_symmetry_u_error"
                ]
            )
        ),
        "raw_texture_resolution_quality": (
            int(texture_color_metrics["basecolor_width"])
            >= int(thresholds["min_raw_texture_size"])
            and int(texture_color_metrics["basecolor_height"])
            >= int(thresholds["min_raw_texture_size"])
            and int(texture_color_metrics["pbr_width"])
            == int(texture_color_metrics["basecolor_width"])
            and int(texture_color_metrics["pbr_height"])
            == int(texture_color_metrics["basecolor_height"])
        ),
        "coverage_report_valid": coverage_result.success,
        "texture_basecolor_valid": bool(
            coverage_result.checks.get("texture_basecolor_written", False)
            and float(coverage_result.metrics["basecolor_observed_texel_fraction"]) > 0.0
            and float(coverage_result.metrics["basecolor_generated_texel_fraction"]) == 0.0
        ),
        "pbr_texture_outputs_valid": bool(
            coverage_result.checks.get("pbr_textures_written", False)
            and float(coverage_result.metrics["pbr_observed_texel_fraction"]) > 0.0
            and float(coverage_result.metrics["pbr_generated_texel_fraction"]) == 0.0
        ),
        "raw_pbr_face_material_detail_quality": bool(
            float(coverage_result.metrics["pbr_face_material_detail_applied"]) == 1.0
            and int(coverage_result.metrics["pbr_face_material_detail_texel_count"])
            >= int(thresholds["min_pbr_face_material_detail_texel_count"])
            and float(coverage_result.metrics["pbr_face_material_detail_normal_variance"])
            > 0.0
        ),
        "joint_projection_report_valid": projection_result.success,
        "silhouette_residual_maps_valid": residual_result.success,
        "artifact_validation_report_valid": validation_result.success,
        "face_turntable_report_valid": turntable_result.success,
        "raw_dense_evidence_manifest_valid": (
            dense_evidence_metrics["raw_dense_evidence_manifest_valid"] == 1.0
        ),
        "raw_dense_evidence_tensor_refs_valid": (
            dense_evidence_metrics["raw_dense_evidence_tensor_refs_valid"] == 1.0
        ),
        "raw_export_mode": mesh_validation.get("mode") == "raw",
    }
    metrics: dict[str, float | int] = {
        "num_frames": int(fit_report["num_frames"]),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": initial_loss - final_loss,
        "loss_reduction_fraction": (initial_loss - final_loss) / max(initial_loss, 1e-12),
        "initial_keypoint_loss": initial_keypoint_loss,
        "final_keypoint_loss": final_keypoint_loss,
        "keypoint_loss_reduction": keypoint_loss_reduction,
        "initial_silhouette_loss": initial_silhouette_loss,
        "final_silhouette_loss": final_silhouette_loss,
        "silhouette_loss_reduction": silhouette_loss_reduction,
        "init_prediction_mean_confidence": float(
            metrics_raw.get("init_prediction_mean_confidence", 0.0)
        ),
        "raw_mean_mask_area_ratio": float(observation_metrics["mean_mask_area_ratio"]),
        "raw_mean_bbox_area_ratio": float(observation_metrics["mean_bbox_area_ratio"]),
        "raw_full_frame_bbox_fraction": float(
            observation_metrics["full_frame_bbox_fraction"]
        ),
        "raw_border_touch_bbox_fraction": float(
            observation_metrics["border_touch_bbox_fraction"]
        ),
        "raw_texture_unknown_luma": float(texture_color_metrics["unknown_luma"]),
        "raw_texture_foreground_fallback_luma": float(
            texture_color_metrics["foreground_fallback_luma"]
        ),
        "raw_texture_unknown_luma_fraction_of_foreground": float(
            texture_color_metrics["unknown_luma_fraction_of_foreground"]
        ),
        "raw_texture_width": int(texture_color_metrics["basecolor_width"]),
        "raw_texture_height": int(texture_color_metrics["basecolor_height"]),
        "raw_texture_resolution_scale": float(
            texture_color_metrics["basecolor_width"] / 256.0
        ),
        "raw_face_texture_target_bake_used": float(
            texture_color_metrics["face_texture_target_bake_used"]
        ),
        "raw_face_texture_target_reported_size": int(
            texture_color_metrics["face_texture_target_size"]
        ),
        "raw_template_region_v2_texture_target_bake_used": float(
            texture_color_metrics["template_region_v2_texture_target_bake_used"]
        ),
        "raw_template_region_v2_texture_target_reported_size": int(
            texture_color_metrics["template_region_v2_texture_target_size"]
        ),
        "raw_template_region_v2_texture_target_reported_scale": float(
            texture_color_metrics["template_region_v2_texture_target_scale"]
        ),
        "raw_template_region_v2_texture_target_reported_island_count": int(
            texture_color_metrics["template_region_v2_texture_target_island_count"]
        ),
        "raw_template_region_v2_texture_target_reported_facial_island_count": int(
            texture_color_metrics[
                "template_region_v2_texture_target_facial_island_count"
            ]
        ),
        "raw_template_region_v2_texture_target_reported_requested_texel_count": int(
            texture_color_metrics[
                "template_region_v2_texture_target_requested_texel_count"
            ]
        ),
        "raw_template_region_v2_texture_target_reported_allocated_texel_count": int(
            texture_color_metrics[
                "template_region_v2_texture_target_allocated_texel_count"
            ]
        ),
        "raw_pbr_texture_width": int(texture_color_metrics["pbr_width"]),
        "raw_pbr_texture_height": int(texture_color_metrics["pbr_height"]),
        "raw_glb_texture_size": int(
            mesh_validation.get("glb_texture_size", expected_glb_texture_size)
        ),
        "raw_glb_texture_dimensions_valid": float(
            glb_structure.checks.get("texture_dimensions_valid", False)
        ),
        "raw_glb_pbr_texture_dimensions_valid": float(
            glb_structure.checks.get("pbr_texture_dimensions_valid", False)
        ),
        "raw_glb_basecolor_texture_valid": float(checks["glb_basecolor_texture_valid"]),
        "raw_glb_pbr_textures_valid": float(checks["glb_pbr_textures_valid"]),
        "raw_glb_embedded_texture_dimension_gate_valid": float(
            glb_structure.checks.get("texture_dimensions_valid", False)
            and glb_structure.checks.get("pbr_texture_dimensions_valid", False)
        ),
        "raw_glb_face_feature_required_group_count": int(
            glb_face_feature_visibility_result.metrics["required_group_count"]
        ),
        "raw_glb_face_feature_visible_group_count": int(
            glb_face_feature_visibility_result.metrics["visible_group_count"]
        ),
        "raw_glb_face_feature_required_feature_count": int(
            glb_face_feature_visibility_result.metrics["required_feature_count"]
        ),
        "raw_glb_face_feature_present_required_feature_count": int(
            glb_face_feature_visibility_result.metrics[
                "present_required_feature_count"
            ]
        ),
        "raw_glb_face_feature_visible_feature_count": int(
            glb_face_feature_visibility_result.metrics["visible_feature_count"]
        ),
        "raw_glb_face_feature_sampled_texel_count": int(
            glb_face_feature_visibility_result.metrics["sampled_texel_count"]
        ),
        "raw_glb_face_feature_min_visible_texel_count": int(
            glb_face_feature_visibility_result.metrics["min_visible_texel_count"]
        ),
        "raw_glb_face_feature_mean_rgb_error": float(
            glb_face_feature_visibility_result.metrics["mean_rgb_error"]
        ),
        "raw_glb_face_feature_max_rgb_error": float(
            glb_face_feature_visibility_result.metrics["max_rgb_error"]
        ),
        "raw_glb_face_feature_eye_visible": float(
            glb_face_feature_visibility_result.metrics["eye_visible"]
        ),
        "raw_glb_face_feature_nose_visible": float(
            glb_face_feature_visibility_result.metrics["nose_visible"]
        ),
        "raw_glb_face_feature_muzzle_visible": float(
            glb_face_feature_visibility_result.metrics["muzzle_visible"]
        ),
        "raw_glb_face_feature_iris_to_pupil_rgb_distance": float(
            glb_face_feature_visibility_result.metrics[
                "iris_to_pupil_rgb_distance"
            ]
        ),
        "raw_glb_face_feature_eye_detail_luma_std": float(
            glb_face_feature_visibility_result.metrics["eye_detail_luma_std"]
        ),
        "raw_glb_face_feature_nose_to_muzzle_rgb_distance": float(
            glb_face_feature_visibility_result.metrics[
                "nose_to_muzzle_rgb_distance"
            ]
        ),
        "raw_glb_face_feature_mouth_to_muzzle_rgb_distance": float(
            glb_face_feature_visibility_result.metrics[
                "mouth_to_muzzle_rgb_distance"
            ]
        ),
        "raw_glb_face_feature_material_visibility_gate_valid": float(
            glb_face_feature_visibility_result.metrics[
                "material_visibility_gate_valid"
            ]
        ),
        "raw_facial_feature_count": int(texture_color_metrics["facial_feature_count"]),
        "raw_facial_feature_painted_texel_count": int(
            texture_color_metrics["facial_feature_painted_texel_count"]
        ),
        "raw_facial_feature_observed_texel_count": int(
            texture_color_metrics["facial_feature_observed_texel_count"]
        ),
        "raw_facial_feature_observed_texel_fraction": float(
            texture_color_metrics["facial_feature_observed_texel_fraction"]
        ),
        "raw_facial_feature_source_patch_count": int(
            texture_color_metrics["facial_feature_source_patch_count"]
        ),
        "raw_facial_feature_source_patch_feedback_sample_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_feedback_sample_count"
            ]
        ),
        "raw_facial_feature_source_patch_feedback_used_count": int(
            texture_color_metrics["facial_feature_source_patch_feedback_used_count"]
        ),
        "raw_facial_feature_source_patch_bake_candidate_count": int(
            texture_color_metrics["facial_feature_source_patch_bake_candidate_count"]
        ),
        "raw_facial_feature_source_patch_bake_selected_feedback": float(
            texture_color_metrics[
                "facial_feature_source_patch_bake_selected_feedback"
            ]
        ),
        "raw_facial_feature_source_patch_bake_visual_quality_passed_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_bake_visual_quality_passed_count"
            ]
        ),
        "raw_facial_feature_source_patch_bake_visual_quality_min_score": float(
            texture_color_metrics[
                "facial_feature_source_patch_bake_visual_quality_min_score"
            ]
        ),
        "raw_facial_feature_source_patch_bake_visual_quality_mean_score": float(
            texture_color_metrics[
                "facial_feature_source_patch_bake_visual_quality_mean_score"
            ]
        ),
        "raw_facial_feature_eye_source_detail_reinforcement_feature_count": int(
            texture_color_metrics[
                "facial_feature_eye_source_detail_reinforcement_feature_count"
            ]
        ),
        "raw_facial_feature_eye_source_detail_reinforcement_texel_count": int(
            texture_color_metrics[
                "facial_feature_eye_source_detail_reinforcement_texel_count"
            ]
        ),
        "raw_facial_feature_eye_source_detail_reinforcement_texel_fraction": float(
            texture_color_metrics[
                "facial_feature_eye_source_detail_reinforcement_texel_fraction"
            ]
        ),
        "raw_facial_feature_whisker_pad_feature_count": int(
            texture_color_metrics["facial_feature_whisker_pad_feature_count"]
        ),
        "raw_facial_feature_whisker_pad_texel_count": int(
            texture_color_metrics["facial_feature_whisker_pad_texel_count"]
        ),
        "raw_facial_feature_whisker_pad_texel_fraction": float(
            texture_color_metrics["facial_feature_whisker_pad_texel_fraction"]
        ),
        "raw_facial_feature_iris_halo_feature_count": int(
            texture_color_metrics["facial_feature_iris_halo_feature_count"]
        ),
        "raw_facial_feature_iris_halo_uv_support_texel_count": int(
            texture_color_metrics[
                "facial_feature_iris_halo_uv_support_texel_count"
            ]
        ),
        "raw_facial_feature_iris_halo_core_texel_count": int(
            texture_color_metrics["facial_feature_iris_halo_core_texel_count"]
        ),
        "raw_facial_feature_iris_halo_edge_texel_count": int(
            texture_color_metrics["facial_feature_iris_halo_edge_texel_count"]
        ),
        "raw_facial_feature_iris_halo_accepted_core_texel_count": int(
            texture_color_metrics[
                "facial_feature_iris_halo_accepted_core_texel_count"
            ]
        ),
        "raw_facial_feature_iris_halo_edge_fraction": float(
            texture_color_metrics["facial_feature_iris_halo_edge_fraction"]
        ),
        "raw_facial_feature_iris_halo_edge_to_core_ratio": float(
            texture_color_metrics["facial_feature_iris_halo_edge_to_core_ratio"]
        ),
        "raw_facial_feature_iris_halo_max_edge_to_core_ratio": float(
            texture_color_metrics[
                "facial_feature_iris_halo_max_edge_to_core_ratio"
            ]
        ),
        "raw_facial_feature_iris_halo_support_to_accepted_core_ratio": float(
            texture_color_metrics[
                "facial_feature_iris_halo_support_to_accepted_core_ratio"
            ]
        ),
        "raw_facial_feature_left_iris_halo_uv_support_texel_count": int(
            texture_color_metrics[
                "facial_feature_left_iris_halo_uv_support_texel_count"
            ]
        ),
        "raw_facial_feature_left_iris_halo_core_texel_count": int(
            texture_color_metrics[
                "facial_feature_left_iris_halo_core_texel_count"
            ]
        ),
        "raw_facial_feature_left_iris_halo_edge_texel_count": int(
            texture_color_metrics[
                "facial_feature_left_iris_halo_edge_texel_count"
            ]
        ),
        "raw_facial_feature_left_iris_halo_accepted_core_texel_count": int(
            texture_color_metrics[
                "facial_feature_left_iris_halo_accepted_core_texel_count"
            ]
        ),
        "raw_facial_feature_left_iris_halo_edge_fraction": float(
            texture_color_metrics["facial_feature_left_iris_halo_edge_fraction"]
        ),
        "raw_facial_feature_left_iris_halo_edge_to_core_ratio": float(
            texture_color_metrics[
                "facial_feature_left_iris_halo_edge_to_core_ratio"
            ]
        ),
        "raw_facial_feature_left_iris_halo_support_to_accepted_core_ratio": float(
            texture_color_metrics[
                "facial_feature_left_iris_halo_support_to_accepted_core_ratio"
            ]
        ),
        "raw_facial_feature_right_iris_halo_uv_support_texel_count": int(
            texture_color_metrics[
                "facial_feature_right_iris_halo_uv_support_texel_count"
            ]
        ),
        "raw_facial_feature_right_iris_halo_core_texel_count": int(
            texture_color_metrics[
                "facial_feature_right_iris_halo_core_texel_count"
            ]
        ),
        "raw_facial_feature_right_iris_halo_edge_texel_count": int(
            texture_color_metrics[
                "facial_feature_right_iris_halo_edge_texel_count"
            ]
        ),
        "raw_facial_feature_right_iris_halo_accepted_core_texel_count": int(
            texture_color_metrics[
                "facial_feature_right_iris_halo_accepted_core_texel_count"
            ]
        ),
        "raw_facial_feature_right_iris_halo_edge_fraction": float(
            texture_color_metrics["facial_feature_right_iris_halo_edge_fraction"]
        ),
        "raw_facial_feature_right_iris_halo_edge_to_core_ratio": float(
            texture_color_metrics[
                "facial_feature_right_iris_halo_edge_to_core_ratio"
            ]
        ),
        "raw_facial_feature_right_iris_halo_support_to_accepted_core_ratio": float(
            texture_color_metrics[
                "facial_feature_right_iris_halo_support_to_accepted_core_ratio"
            ]
        ),
        "raw_face_texture_bake_comparison_candidate_count": int(
            face_texture_bake_comparison_result.metrics["candidate_count"]
        ),
        "raw_face_texture_bake_comparison_region_count": int(
            face_texture_bake_comparison_result.metrics["compared_region_count"]
        ),
        "raw_face_texture_bake_comparison_selected_feedback": float(
            face_texture_bake_comparison_result.metrics["selected_feedback"]
        ),
        "raw_face_texture_bake_comparison_score_improved_region_count": int(
            face_texture_bake_comparison_result.metrics[
                "score_improved_region_count"
            ]
        ),
        "raw_face_texture_bake_comparison_coverage_improved_region_count": int(
            face_texture_bake_comparison_result.metrics[
                "coverage_improved_region_count"
            ]
        ),
        "raw_face_texture_bake_comparison_source_detail_improved_region_count": int(
            face_texture_bake_comparison_result.metrics[
                "source_detail_improved_region_count"
            ]
        ),
        "raw_face_texture_bake_comparison_max_score_delta": float(
            face_texture_bake_comparison_result.metrics["max_score_delta"]
        ),
        "raw_face_texture_bake_comparison_max_coverage_delta": float(
            face_texture_bake_comparison_result.metrics["max_coverage_delta"]
        ),
        "raw_face_texture_bake_comparison_max_source_detail_delta": float(
            face_texture_bake_comparison_result.metrics["max_source_detail_delta"]
        ),
        "raw_face_texture_bake_comparison_selected_not_lower_rank": float(
            face_texture_bake_comparison_result.metrics[
                "selected_candidate_not_lower_rank"
            ]
        ),
        "raw_facial_feature_source_patch_texel_count": int(
            texture_color_metrics["facial_feature_source_patch_texel_count"]
        ),
        "raw_facial_feature_source_patch_texel_fraction": float(
            texture_color_metrics["facial_feature_source_patch_texel_fraction"]
        ),
        "raw_facial_feature_source_patch_quality_passed_count": int(
            texture_color_metrics["facial_feature_source_patch_quality_passed_count"]
        ),
        "raw_facial_feature_source_patch_quality_total_count": int(
            texture_color_metrics["facial_feature_source_patch_quality_total_count"]
        ),
        "raw_facial_feature_source_patch_quality_fraction": float(
            texture_color_metrics["facial_feature_source_patch_quality_fraction"]
        ),
        "raw_facial_feature_source_patch_min_quality_score": float(
            texture_color_metrics["facial_feature_source_patch_min_quality_score"]
        ),
        "raw_facial_feature_source_patch_mean_quality_score": float(
            texture_color_metrics["facial_feature_source_patch_mean_quality_score"]
        ),
        "raw_facial_feature_source_patch_view_weighting_used": float(
            texture_color_metrics[
                "facial_feature_source_patch_view_weighting_used"
            ]
        ),
        "raw_facial_feature_source_patch_view_weighted_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_view_weighted_count"
            ]
        ),
        "raw_facial_feature_source_patch_min_view_weight": float(
            texture_color_metrics["facial_feature_source_patch_min_view_weight"]
        ),
        "raw_facial_feature_source_patch_mean_view_weight": float(
            texture_color_metrics["facial_feature_source_patch_mean_view_weight"]
        ),
        "raw_facial_feature_source_patch_min_frontal_face_confidence": float(
            texture_color_metrics[
                "facial_feature_source_patch_min_frontal_face_confidence"
            ]
        ),
        "raw_facial_feature_source_patch_mean_frontal_face_confidence": float(
            texture_color_metrics[
                "facial_feature_source_patch_mean_frontal_face_confidence"
            ]
        ),
        "raw_facial_feature_source_patch_min_view_weighted_rank": float(
            texture_color_metrics[
                "facial_feature_source_patch_min_view_weighted_rank"
            ]
        ),
        "raw_facial_feature_source_patch_mean_view_weighted_rank": float(
            texture_color_metrics[
                "facial_feature_source_patch_mean_view_weighted_rank"
            ]
        ),
        "raw_facial_feature_source_patch_frontal_photo_priority_used": float(
            texture_color_metrics[
                "facial_feature_source_patch_frontal_photo_priority_used"
            ]
        ),
        "raw_facial_feature_source_patch_frontal_photo_priority_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_frontal_photo_priority_count"
            ]
        ),
        "raw_facial_feature_source_patch_max_frontal_photo_priority": float(
            texture_color_metrics[
                "facial_feature_source_patch_max_frontal_photo_priority"
            ]
        ),
        "raw_facial_feature_source_patch_mean_frontal_photo_priority": float(
            texture_color_metrics[
                "facial_feature_source_patch_mean_frontal_photo_priority"
            ]
        ),
        "raw_facial_feature_source_patch_direct_photo_candidate_diagnostic_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_direct_photo_candidate_diagnostic_count"
            ]
        ),
        "raw_facial_feature_source_patch_direct_photo_zero_priority_candidate_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_direct_photo_zero_priority_candidate_count"
            ]
        ),
        "raw_facial_feature_source_patch_direct_photo_quality_below_min_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_direct_photo_quality_below_min_count"
            ]
        ),
        "raw_facial_feature_source_patch_direct_photo_detail_below_min_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_direct_photo_detail_below_min_count"
            ]
        ),
        "raw_facial_feature_source_patch_ear_inner_direct_photo_quality_below_min_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_ear_inner_direct_photo_quality_below_min_count"
            ]
        ),
        "raw_facial_feature_source_patch_direct_photo_max_quality_gap": float(
            texture_color_metrics[
                "facial_feature_source_patch_direct_photo_max_quality_gap"
            ]
        ),
        "raw_facial_feature_source_patch_direct_photo_max_detail_gap": float(
            texture_color_metrics[
                "facial_feature_source_patch_direct_photo_max_detail_gap"
            ]
        ),
        "raw_facial_feature_source_patch_fusion_min_weight": float(
            texture_color_metrics["facial_feature_source_patch_fusion_min_weight"]
        ),
        "raw_facial_feature_source_patch_fusion_mean_weight": float(
            texture_color_metrics["facial_feature_source_patch_fusion_mean_weight"]
        ),
        "raw_facial_feature_source_patch_fusion_weighted_texel_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_fusion_weighted_texel_count"
            ]
        ),
        "raw_facial_feature_source_patch_nominal_texel_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_nominal_texel_count"
            ]
        ),
        "raw_facial_feature_source_patch_footprint_reduction_fraction": float(
            texture_color_metrics[
                "facial_feature_source_patch_footprint_reduction_fraction"
            ]
        ),
        "raw_facial_feature_source_patch_mean_footprint_area_scale": float(
            texture_color_metrics[
                "facial_feature_source_patch_mean_footprint_area_scale"
            ]
        ),
        "raw_facial_feature_source_patch_min_footprint_area_scale": float(
            texture_color_metrics[
                "facial_feature_source_patch_min_footprint_area_scale"
            ]
        ),
        "raw_facial_feature_source_patch_shape_mask_texel_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_shape_mask_texel_count"
            ]
        ),
        "raw_facial_feature_source_patch_shape_mask_fraction": float(
            texture_color_metrics[
                "facial_feature_source_patch_shape_mask_fraction"
            ]
        ),
        "raw_facial_feature_source_patch_shape_alpha_mean": float(
            texture_color_metrics[
                "facial_feature_source_patch_shape_alpha_mean"
            ]
        ),
        "raw_facial_feature_source_patch_source_mask_observed_texel_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_source_mask_observed_texel_count"
            ]
        ),
        "raw_facial_feature_source_patch_source_mask_densified_texel_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_source_mask_densified_texel_count"
            ]
        ),
        "raw_facial_feature_source_patch_source_mask_densification_gain": float(
            texture_color_metrics[
                "facial_feature_source_patch_source_mask_densification_gain"
            ]
        ),
        "raw_facial_feature_source_patch_mean_source_mask_densification_gain": float(
            texture_color_metrics[
                "facial_feature_source_patch_mean_source_mask_densification_gain"
            ]
        ),
        "raw_facial_feature_source_patch_localization_candidate_texel_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_localization_candidate_texel_count"
            ]
        ),
        "raw_facial_feature_source_patch_localization_selected_texel_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_localization_selected_texel_count"
            ]
        ),
        "raw_facial_feature_source_patch_localization_fraction": float(
            texture_color_metrics[
                "facial_feature_source_patch_localization_fraction"
            ]
        ),
        "raw_facial_feature_source_patch_mean_localization_fraction": float(
            texture_color_metrics[
                "facial_feature_source_patch_mean_localization_fraction"
            ]
        ),
        "raw_facial_feature_source_patch_landmark_aligned_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_landmark_aligned_count"
            ]
        ),
        "raw_facial_feature_source_patch_mean_landmark_confidence": float(
            texture_color_metrics[
                "facial_feature_source_patch_mean_landmark_confidence"
            ]
        ),
        "raw_facial_feature_source_patch_observed_anchor_guided_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_observed_anchor_guided_count"
            ]
        ),
        "raw_facial_feature_source_patch_observed_anchor_guided_fraction": float(
            texture_color_metrics[
                "facial_feature_source_patch_observed_anchor_guided_fraction"
            ]
        ),
        "raw_facial_feature_source_patch_landmark_destination_scaled_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_landmark_destination_scaled_count"
            ]
        ),
        "raw_facial_feature_source_patch_mean_landmark_destination_scale": float(
            texture_color_metrics[
                "facial_feature_source_patch_mean_landmark_destination_scale"
            ]
        ),
        "raw_face_feature_destination_anchor_anatomical_count": int(
            texture_color_metrics[
                "facial_feature_destination_anchor_anatomical_count"
            ]
        ),
        "raw_face_feature_destination_anchor_order_valid": float(
            texture_color_metrics[
                "facial_feature_destination_anchor_order_valid"
            ]
        ),
        "raw_face_feature_destination_anchor_eye_to_nose_v_gap": float(
            texture_color_metrics[
                "facial_feature_destination_anchor_eye_to_nose_v_gap"
            ]
        ),
        "raw_face_feature_destination_anchor_nose_to_mouth_v_gap": float(
            texture_color_metrics[
                "facial_feature_destination_anchor_nose_to_mouth_v_gap"
            ]
        ),
        "raw_face_feature_destination_anchor_nose_to_muzzle_v_gap": float(
            texture_color_metrics[
                "facial_feature_destination_anchor_nose_to_muzzle_v_gap"
            ]
        ),
        "raw_face_feature_destination_anchor_ear_to_eye_v_gap": float(
            texture_color_metrics[
                "facial_feature_destination_anchor_ear_to_eye_v_gap"
            ]
        ),
        "raw_face_feature_destination_anchor_eye_symmetry_u_error": float(
            texture_color_metrics[
                "facial_feature_destination_anchor_eye_symmetry_u_error"
            ]
        ),
        "raw_facial_feature_source_patch_core_texel_count": int(
            texture_color_metrics["facial_feature_source_patch_core_texel_count"]
        ),
        "raw_facial_feature_source_patch_core_fraction": float(
            texture_color_metrics["facial_feature_source_patch_core_fraction"]
        ),
        "raw_facial_feature_source_patch_feathered_edge_texel_count": int(
            texture_color_metrics[
                "facial_feature_source_patch_feathered_edge_texel_count"
            ]
        ),
        "raw_facial_feature_source_patch_feathered_edge_fraction": float(
            texture_color_metrics[
                "facial_feature_source_patch_feathered_edge_fraction"
            ]
        ),
        "raw_facial_feature_source_patch_mean_blend_alpha": float(
            texture_color_metrics["facial_feature_source_patch_mean_blend_alpha"]
        ),
        "raw_face_priority_texture_used": float(
            texture_color_metrics["face_priority_texture_used"]
        ),
        "raw_face_uv_region_count": int(texture_color_metrics["face_uv_region_count"]),
        "raw_face_uv_region_total_area": float(
            texture_color_metrics["face_uv_region_total_area"]
        ),
        "raw_face_uv_region_min_density_weight": float(
            texture_color_metrics["face_uv_region_min_density_weight"]
        ),
        "raw_face_uv_region_contract_valid": float(
            texture_color_metrics["face_uv_region_contract_valid"]
        ),
        "raw_template_region_v2_count": int(template_region_v2_metrics["region_count"]),
        "raw_template_region_v2_required_region_count": int(
            template_region_v2_metrics["required_region_count"]
        ),
        "raw_template_region_v2_required_present_count": int(
            template_region_v2_metrics["required_present_count"]
        ),
        "raw_template_region_v2_contract_valid": float(
            template_region_v2_metrics["contract_valid"]
        ),
        "raw_glb_template_region_v2_contract_valid": float(
            template_region_v2_metrics["glb_contract_valid"]
        ),
        "raw_template_region_v2_belly_vertex_count": int(
            template_region_v2_metrics["belly_vertex_count"]
        ),
        "raw_face_uv_density_allocated_texel_count": int(
            face_uv_density_result.metrics["face_uv_region_allocated_texel_count"]
        ),
        "raw_face_uv_density_min_allocated_texels": int(
            face_uv_density_result.metrics["face_uv_region_min_allocated_texels"]
        ),
        "raw_face_uv_density_eye_min_allocated_texels": int(
            face_uv_density_result.metrics["face_uv_eye_min_allocated_texels"]
        ),
        "raw_face_uv_density_nose_allocated_texels": int(
            face_uv_density_result.metrics["face_uv_nose_allocated_texels"]
        ),
        "raw_face_uv_density_ear_inner_min_allocated_texels": int(
            face_uv_density_result.metrics["face_uv_ear_inner_min_allocated_texels"]
        ),
        "raw_template_region_v2_uv_region_count": int(
            face_uv_density_result.metrics["template_region_v2_uv_region_count"]
        ),
        "raw_template_region_v2_uv_required_present_count": int(
            face_uv_density_result.metrics[
                "template_region_v2_uv_required_present_count"
            ]
        ),
        "raw_template_region_v2_uv_island_count": int(
            face_uv_density_result.metrics["template_region_v2_uv_island_count"]
        ),
        "raw_template_region_v2_region_scoped_uv_island_count": int(
            face_uv_density_result.metrics[
                "template_region_v2_region_scoped_uv_island_count"
            ]
        ),
        "raw_template_region_v2_uv_island_allocated_texel_count": int(
            face_uv_density_result.metrics[
                "template_region_v2_uv_island_allocated_texel_count"
            ]
        ),
        "raw_template_region_v2_uv_island_density_weighted_texel_count": int(
            face_uv_density_result.metrics[
                "template_region_v2_uv_island_density_weighted_texel_count"
            ]
        ),
        "raw_template_region_v2_face_uv_island_count": int(
            face_uv_density_result.metrics["template_region_v2_face_uv_island_count"]
        ),
        "raw_template_region_v2_facial_uv_island_count": int(
            face_uv_density_result.metrics[
                "template_region_v2_facial_uv_island_count"
            ]
        ),
        "raw_template_region_v2_facial_uv_island_allocated_texel_count": int(
            face_uv_density_result.metrics[
                "template_region_v2_facial_uv_island_allocated_texel_count"
            ]
        ),
        "raw_template_region_v2_body_region_uv_island_count": int(
            face_uv_density_result.metrics[
                "template_region_v2_body_region_uv_island_count"
            ]
        ),
        "raw_template_region_v2_eye_min_allocated_texels": int(
            face_uv_density_result.metrics[
                "template_region_v2_eye_min_allocated_texels"
            ]
        ),
        "raw_template_region_v2_nose_allocated_texels": int(
            face_uv_density_result.metrics["template_region_v2_nose_allocated_texels"]
        ),
        "raw_template_region_v2_muzzle_allocated_texels": int(
            face_uv_density_result.metrics[
                "template_region_v2_muzzle_allocated_texels"
            ]
        ),
        "raw_template_region_v2_ear_inner_min_allocated_texels": int(
            face_uv_density_result.metrics[
                "template_region_v2_ear_inner_min_allocated_texels"
            ]
        ),
        "raw_template_region_v2_facial_density_weighted_texel_count": int(
            face_uv_density_result.metrics[
                "template_region_v2_facial_density_weighted_texel_count"
            ]
        ),
        "raw_face_texture_target_size": int(
            face_texture_targets_result.metrics["target_texture_size"]
        ),
        "raw_face_texture_target_scale": float(
            face_texture_targets_result.metrics["target_texture_scale"]
        ),
        "raw_face_texture_target_requested_texel_count": int(
            face_texture_targets_result.metrics[
                "face_texture_target_requested_texel_count"
            ]
        ),
        "raw_face_texture_target_allocated_texel_count": int(
            face_texture_targets_result.metrics[
                "face_texture_target_allocated_texel_count"
            ]
        ),
        "raw_face_texture_target_eye_min_texels": int(
            face_texture_targets_result.metrics["face_texture_target_eye_min_texels"]
        ),
        "raw_face_texture_target_nose_texels": int(
            face_texture_targets_result.metrics["face_texture_target_nose_texels"]
        ),
        "raw_face_texture_target_ear_inner_min_texels": int(
            face_texture_targets_result.metrics[
                "face_texture_target_ear_inner_min_texels"
            ]
        ),
        "raw_template_region_v2_texture_target_size": int(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_size"
            ]
        ),
        "raw_template_region_v2_texture_target_scale": float(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_scale"
            ]
        ),
        "raw_template_region_v2_texture_target_island_count": int(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_island_count"
            ]
        ),
        "raw_template_region_v2_texture_target_facial_island_count": int(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_facial_island_count"
            ]
        ),
        "raw_template_region_v2_texture_target_requested_texel_count": int(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_requested_texel_count"
            ]
        ),
        "raw_template_region_v2_texture_target_allocated_texel_count": int(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_allocated_texel_count"
            ]
        ),
        "raw_template_region_v2_texture_target_eye_min_texels": int(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_eye_min_texels"
            ]
        ),
        "raw_template_region_v2_texture_target_nose_texels": int(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_nose_texels"
            ]
        ),
        "raw_template_region_v2_texture_target_muzzle_texels": int(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_muzzle_texels"
            ]
        ),
        "raw_template_region_v2_texture_target_ear_inner_min_texels": int(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_ear_inner_min_texels"
            ]
        ),
        "raw_template_region_v2_texture_target_max_density_scale": float(
            face_texture_targets_result.metrics[
                "template_region_v2_texture_target_max_density_scale"
            ]
        ),
        "raw_face_texture_crop_region_count": int(
            face_texture_diagnostic_result.metrics["face_texture_crop_region_count"]
        ),
        "raw_face_texture_crop_texture_width": int(
            face_texture_diagnostic_result.metrics["face_texture_crop_texture_width"]
        ),
        "raw_face_texture_crop_texture_height": int(
            face_texture_diagnostic_result.metrics["face_texture_crop_texture_height"]
        ),
        "raw_face_texture_crop_compared_region_count": int(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_compared_region_count"
            ]
        ),
        "raw_face_texture_crop_exact_source_patch_region_count": int(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_exact_source_patch_region_count"
            ]
        ),
        "raw_face_texture_crop_exact_source_patch_fraction": float(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_exact_source_patch_fraction"
            ]
        ),
        "raw_face_texture_crop_detail_region_count": int(
            face_texture_diagnostic_result.metrics["face_texture_crop_detail_region_count"]
        ),
        "raw_face_texture_crop_min_unique_rgb_count": int(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_min_unique_rgb_count"
            ]
        ),
        "raw_face_texture_crop_min_confident_texel_fraction": float(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_min_confident_texel_fraction"
            ]
        ),
        "raw_face_texture_crop_mean_confident_texel_fraction": float(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_mean_confident_texel_fraction"
            ]
        ),
        "raw_face_texture_crop_mean_source_rgb_delta": float(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_mean_source_rgb_delta"
            ]
        ),
        "raw_face_texture_crop_isolated_region_count": int(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_isolated_region_count"
            ]
        ),
        "raw_face_texture_crop_overlap_pair_count": int(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_overlap_pair_count"
            ]
        ),
        "raw_face_texture_crop_overlap_fraction": float(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_overlap_fraction"
            ]
        ),
        "raw_face_texture_crop_visual_quality_region_count": int(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_visual_quality_region_count"
            ]
        ),
        "raw_face_texture_crop_visual_quality_passed_count": int(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_visual_quality_passed_count"
            ]
        ),
        "raw_face_texture_crop_visual_quality_fraction": float(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_visual_quality_fraction"
            ]
        ),
        "raw_face_texture_crop_visual_quality_min_score": float(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_visual_quality_min_score"
            ]
        ),
        "raw_face_texture_crop_visual_quality_mean_score": float(
            face_texture_diagnostic_result.metrics[
                "face_texture_crop_visual_quality_mean_score"
            ]
        ),
        "raw_direct_photo_face_candidate_sample_key_count": int(
            direct_photo_face_candidate_result.metrics[
                "direct_photo_face_candidate_sample_key_count"
            ]
        ),
        "raw_direct_photo_face_candidate_total_count": int(
            direct_photo_face_candidate_result.metrics[
                "direct_photo_face_candidate_total_count"
            ]
        ),
        "raw_direct_photo_face_candidate_direct_count": int(
            direct_photo_face_candidate_result.metrics[
                "direct_photo_face_candidate_direct_count"
            ]
        ),
        "raw_direct_photo_face_candidate_eligible_count": int(
            direct_photo_face_candidate_result.metrics[
                "direct_photo_face_candidate_eligible_count"
            ]
        ),
        "raw_direct_photo_face_candidate_selected_count": int(
            direct_photo_face_candidate_result.metrics[
                "direct_photo_face_candidate_selected_count"
            ]
        ),
        "raw_direct_photo_face_candidate_rejected_count": int(
            direct_photo_face_candidate_result.metrics[
                "direct_photo_face_candidate_rejected_count"
            ]
        ),
        "raw_direct_photo_face_candidate_missing_metric_feature_count": int(
            direct_photo_face_candidate_result.metrics[
                "direct_photo_face_candidate_missing_metric_feature_count"
            ]
        ),
        "raw_direct_photo_face_candidate_max_priority": float(
            direct_photo_face_candidate_result.metrics[
                "direct_photo_face_candidate_max_priority"
            ]
        ),
        "raw_face_feature_min_required_contrast": float(
            texture_color_metrics["face_feature_min_required_contrast"]
        ),
        "raw_face_feature_eye_to_muzzle_luma_delta": float(
            texture_color_metrics["face_feature_eye_to_muzzle_luma_delta"]
        ),
        "raw_face_feature_mouth_to_muzzle_luma_delta": float(
            texture_color_metrics["face_feature_mouth_to_muzzle_luma_delta"]
        ),
        "raw_face_feature_nose_to_muzzle_rgb_distance": float(
            texture_color_metrics["face_feature_nose_to_muzzle_rgb_distance"]
        ),
        "raw_face_feature_iris_to_pupil_rgb_distance": float(
            texture_color_metrics["face_feature_iris_to_pupil_rgb_distance"]
        ),
        "raw_face_feature_contrast_passed": float(
            texture_color_metrics["face_feature_required_contrast_passed"]
        ),
        "raw_face_detail_geometry_applied": float(
            geometry_detail_metrics["face_detail_geometry_applied"]
        ),
        "raw_face_detail_deformed_vertex_count": int(
            geometry_detail_metrics["face_detail_deformed_vertex_count"]
        ),
        "raw_face_detail_max_displacement": float(
            geometry_detail_metrics["face_detail_max_displacement"]
        ),
        "raw_face_detail_mean_displacement": float(
            geometry_detail_metrics["face_detail_mean_displacement"]
        ),
        "raw_face_surface_anchor_count": int(
            geometry_detail_metrics["face_surface_anchor_count"]
        ),
        "raw_face_surface_anchor_vertex_count": int(
            geometry_detail_metrics["face_surface_anchor_vertex_count"]
        ),
        "raw_face_surface_anchor_max_displacement": float(
            geometry_detail_metrics["face_surface_anchor_max_displacement"]
        ),
        "raw_face_turntable_view_count": int(
            turntable_result.metrics["turntable_view_count"]
        ),
        "raw_face_turntable_front_visible_anchor_count": int(
            turntable_result.metrics["front_visible_anchor_count"]
        ),
        "raw_face_turntable_front_anchor_pixel_count": int(
            turntable_result.metrics["front_anchor_pixel_count"]
        ),
        "raw_face_turntable_required_anchor_visible_fraction": float(
            turntable_result.metrics["required_anchor_visible_fraction"]
        ),
        "raw_face_turntable_textured_view_count": int(
            turntable_result.metrics["textured_turntable_view_count"]
        ),
        "raw_face_turntable_textured_written_count": int(
            turntable_result.metrics["textured_turntable_written_count"]
        ),
        "raw_face_turntable_front_textured_unique_rgb_count": int(
            turntable_result.metrics["textured_turntable_front_unique_rgb_count"]
        ),
        "raw_face_turntable_front_textured_luma_std": float(
            turntable_result.metrics["textured_turntable_front_luma_std"]
        ),
        "raw_face_turntable_front_textured_mouth_line_pixel_count": int(
            turntable_result.metrics[
                "textured_turntable_front_mouth_line_pixel_count"
            ]
        ),
        "raw_face_turntable_front_textured_required_feature_visible_count": int(
            turntable_result.metrics[
                "textured_turntable_front_required_feature_visible_count"
            ]
        ),
        "raw_face_turntable_front_textured_required_feature_fraction": float(
            turntable_result.metrics[
                "textured_turntable_front_required_feature_fraction"
            ]
        ),
        "raw_face_turntable_mean_textured_unique_rgb_count": float(
            turntable_result.metrics["textured_turntable_mean_unique_rgb_count"]
        ),
        "raw_face_turntable_mean_textured_luma_std": float(
            turntable_result.metrics["textured_turntable_mean_luma_std"]
        ),
        "raw_face_texture_closeup_width": int(
            turntable_result.metrics["face_texture_closeup_width"]
        ),
        "raw_face_texture_closeup_height": int(
            turntable_result.metrics["face_texture_closeup_height"]
        ),
        "raw_face_texture_closeup_feature_count": int(
            turntable_result.metrics["face_texture_closeup_feature_count"]
        ),
        "raw_face_texture_closeup_required_feature_count": int(
            turntable_result.metrics["face_texture_closeup_required_feature_count"]
        ),
        "raw_face_texture_closeup_required_present_count": int(
            turntable_result.metrics["face_texture_closeup_required_present_count"]
        ),
        "raw_face_texture_closeup_required_feature_fraction": float(
            turntable_result.metrics["face_texture_closeup_required_feature_fraction"]
        ),
        "raw_face_texture_closeup_crop_texel_count": int(
            turntable_result.metrics["face_texture_closeup_crop_texel_count"]
        ),
        "raw_face_texture_closeup_unique_rgb_count": int(
            turntable_result.metrics["face_texture_closeup_unique_rgb_count"]
        ),
        "raw_face_texture_closeup_luma_std": float(
            turntable_result.metrics["face_texture_closeup_luma_std"]
        ),
        "raw_face_texture_source_compare_feature_count": int(
            turntable_result.metrics["face_texture_source_compare_feature_count"]
        ),
        "raw_face_texture_source_compare_source_feature_available_count": int(
            turntable_result.metrics[
                "face_texture_source_compare_source_feature_available_count"
            ]
        ),
        "raw_face_texture_source_compare_required_feature_count": int(
            turntable_result.metrics[
                "face_texture_source_compare_required_feature_count"
            ]
        ),
        "raw_face_texture_source_compare_required_present_count": int(
            turntable_result.metrics[
                "face_texture_source_compare_required_present_count"
            ]
        ),
        "raw_face_texture_source_compare_required_feature_fraction": float(
            turntable_result.metrics[
                "face_texture_source_compare_required_feature_fraction"
            ]
        ),
        "raw_face_texture_source_compare_mean_rgb_distance": float(
            turntable_result.metrics[
                "face_texture_source_compare_mean_rgb_distance"
            ]
        ),
        "raw_face_texture_source_compare_max_rgb_distance": float(
            turntable_result.metrics["face_texture_source_compare_max_rgb_distance"]
        ),
        "raw_face_texture_source_compare_mean_luma_delta": float(
            turntable_result.metrics["face_texture_source_compare_mean_luma_delta"]
        ),
        "raw_face_texture_source_compare_max_luma_delta": float(
            turntable_result.metrics["face_texture_source_compare_max_luma_delta"]
        ),
        "raw_face_texture_source_compare_source_mean_luma_std": float(
            turntable_result.metrics[
                "face_texture_source_compare_source_mean_luma_std"
            ]
        ),
        "raw_face_texture_source_compare_baked_mean_luma_std": float(
            turntable_result.metrics[
                "face_texture_source_compare_baked_mean_luma_std"
            ]
        ),
        "raw_face_texture_source_compare_iris_feature_count": int(
            turntable_result.metrics["face_texture_source_compare_iris_feature_count"]
        ),
        "raw_face_texture_source_compare_iris_candidate_texel_count": int(
            turntable_result.metrics[
                "face_texture_source_compare_iris_candidate_texel_count"
            ]
        ),
        "raw_face_texture_source_compare_iris_core_accepted_texel_count": int(
            turntable_result.metrics[
                "face_texture_source_compare_iris_core_accepted_texel_count"
            ]
        ),
        "raw_face_texture_source_compare_iris_core_densified_texel_count": int(
            turntable_result.metrics[
                "face_texture_source_compare_iris_core_densified_texel_count"
            ]
        ),
        "raw_face_texture_source_compare_iris_annulus_rejected_texel_count": int(
            turntable_result.metrics[
                "face_texture_source_compare_iris_annulus_rejected_texel_count"
            ]
        ),
        "raw_face_texture_source_compare_iris_core_acceptance_fraction": float(
            turntable_result.metrics[
                "face_texture_source_compare_iris_core_acceptance_fraction"
            ]
        ),
        "raw_face_texture_source_compare_iris_annulus_rejected_fraction": float(
            turntable_result.metrics[
                "face_texture_source_compare_iris_annulus_rejected_fraction"
            ]
        ),
        "overlay_mean_silhouette_iou": overlay_mean_silhouette_iou,
        "observed_face_fraction": float(coverage_result.metrics["observed_face_fraction"]),
        "observed_vertex_fraction": float(
            coverage_result.metrics["observed_vertex_fraction"]
        ),
        "generated_face_count": int(coverage_result.metrics["generated_face_count"]),
        "basecolor_observed_texel_fraction": float(
            coverage_result.metrics["basecolor_observed_texel_fraction"]
        ),
        "basecolor_generated_texel_fraction": float(
            coverage_result.metrics["basecolor_generated_texel_fraction"]
        ),
        "pbr_observed_texel_fraction": float(
            coverage_result.metrics["pbr_observed_texel_fraction"]
        ),
        "pbr_generated_texel_fraction": float(
            coverage_result.metrics["pbr_generated_texel_fraction"]
        ),
        "raw_pbr_face_material_detail_applied": float(
            coverage_result.metrics["pbr_face_material_detail_applied"]
        ),
        "raw_pbr_face_material_detail_feature_count": int(
            coverage_result.metrics["pbr_face_material_detail_feature_count"]
        ),
        "raw_pbr_face_material_detail_texel_count": int(
            coverage_result.metrics["pbr_face_material_detail_texel_count"]
        ),
        "raw_pbr_face_material_detail_normal_variance": float(
            coverage_result.metrics["pbr_face_material_detail_normal_variance"]
        ),
        "joint_projection_mean_keypoint_loss": float(
            projection_result.metrics["mean_keypoint_reprojection_loss"]
        ),
        "joint_projection_mean_pixel_error": float(
            projection_result.metrics["mean_pixel_error"]
        ),
        "joint_projection_observed_keypoint_frame_count": int(
            projection_result.metrics["observed_keypoint_frame_count"]
        ),
        "joint_projection_sparse_keypoint_frame_count": int(
            projection_result.metrics["sparse_keypoint_frame_count"]
        ),
        "joint_projection_missing_keypoint_frame_count": int(
            projection_result.metrics["missing_keypoint_frame_count"]
        ),
        "silhouette_residual_mean_union_error": float(
            residual_result.metrics["mean_union_error_fraction"]
        ),
        "silhouette_residual_mean_false_negative": float(
            residual_result.metrics["mean_false_negative_fraction"]
        ),
        "silhouette_residual_mean_false_positive": float(
            residual_result.metrics["mean_false_positive_fraction"]
        ),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "joint_count": skeleton_report.joint_count,
    }
    metrics.update(dense_evidence_metrics)
    success = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "success": success,
        "checks": checks,
        "thresholds": thresholds,
        "metrics": metrics,
        "artifacts": {
            "fit_report": str(fit_report_path),
            "fit_state": str(fit_state.path),
            "fit_state_source_format": fit_state.source_format,
            "fit_state_safetensors": str(
                project_dir / "outputs" / "fitting" / "fit_state.safetensors"
            ),
            "observation_manifest": str(observation_manifest_path),
            "mesh_validation": str(mesh_validation_path),
            "overlay_manifest": str(overlay_manifest_path),
            "coverage_report": str(coverage_result.report_path),
            "coverage_provenance": str(coverage_result.provenance_path),
            "texture_manifest": str(coverage_result.texture_manifest_path),
            "basecolor": str(project_dir / "outputs" / "textures" / "basecolor.png"),
            "basecolor_confidence": str(
                project_dir / "outputs" / "textures" / "basecolor_confidence.png"
            ),
            "source_view_id": str(project_dir / "outputs" / "textures" / "source_view_id.png"),
            "normal": str(project_dir / "outputs" / "textures" / "normal.png"),
            "roughness": str(project_dir / "outputs" / "textures" / "roughness.png"),
            "ao": str(project_dir / "outputs" / "textures" / "ao.png"),
            "obj": str(obj_path),
            "ply": str(ply_path),
            "glb": str(glb_path),
            "idle_animation": str(
                project_dir / "outputs" / "avatar" / "animations" / "idle.glb"
            ),
            "skeleton": str(skeleton_path),
            "joint_projection": str(projection_result.report_path),
            "silhouette_residuals": str(residual_result.report_path),
            "silhouette_residual_manifest": str(residual_result.manifest_path),
            "validation_report": str(validation_result.report_path),
            "face_turntable": str(turntable_result.report_path),
            "face_turntable_manifest": str(turntable_result.manifest_path),
            "face_turntable_front": str(
                project_dir / "outputs" / "quality" / "raw_face_turntable" / "front.png"
            ),
            "face_turntable_front_textured": str(
                project_dir
                / "outputs"
                / "quality"
                / "raw_face_turntable"
                / "front_textured.png"
            ),
            "face_texture_closeup": str(
                project_dir
                / "outputs"
                / "quality"
                / "raw_face_turntable"
                / "face_closeup.png"
            ),
            "face_texture_source_comparison": str(
                project_dir
                / "outputs"
                / "quality"
                / "raw_face_turntable"
                / "face_source_comparison.png"
            ),
            "face_uv_density": str(face_uv_density_result.report_path),
            "face_uv_density_overlay": str(face_uv_density_result.overlay_path),
            "face_texture_targets": str(face_texture_targets_result.report_path),
            "face_texture_diagnostics": str(face_texture_diagnostic_result.report_path),
            "direct_photo_face_candidates": str(
                direct_photo_face_candidate_result.report_path
            ),
            "face_texture_bake_comparison": str(
                face_texture_bake_comparison_result.report_path
            ),
            "glb_face_feature_visibility": str(
                glb_face_feature_visibility_result.report_path
            ),
            "face_texture_crops": str(face_texture_diagnostic_result.contact_sheet_path),
            "face_texture_crop_dir": str(face_texture_diagnostic_result.crop_dir),
        },
        "glb_structure_checks": glb_structure.checks,
        "skeleton_checks": skeleton_report.checks,
        "joint_projection_checks": projection_result.checks,
        "silhouette_residual_checks": residual_result.checks,
        "face_turntable_checks": turntable_result.checks,
        "face_uv_density_checks": face_uv_density_result.checks,
        "face_texture_targets_checks": face_texture_targets_result.checks,
        "face_texture_diagnostic_checks": face_texture_diagnostic_result.checks,
        "direct_photo_face_candidate_checks": (
            direct_photo_face_candidate_result.checks
        ),
        "face_texture_bake_comparison_checks": (
            face_texture_bake_comparison_result.checks
        ),
        "glb_face_feature_visibility_checks": (
            glb_face_feature_visibility_result.checks
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RawEvaluationResult(
        report_path=report_path,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _as_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _dense_evidence_metrics(
    project_dir: Path,
    *,
    expected_frame_count: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "raw_dense_evidence_manifest_present": 0.0,
        "raw_dense_evidence_manifest_count": 0,
        "raw_dense_evidence_manifest_parse_error_count": 0,
        "raw_dense_evidence_manifest_schema_valid_fraction": 1.0,
        "raw_dense_evidence_manifest_valid": 1.0,
        "raw_dense_evidence_deterministic_manifest_count": 0,
        "raw_dense_evidence_deterministic_valid": 1.0,
        "raw_dense_evidence_frame_count": 0,
        "raw_dense_evidence_frame_fraction": 0.0,
        "raw_dense_evidence_payload_count": 0,
        "raw_dense_evidence_tensor_ref_count": 0,
        "raw_dense_evidence_tensor_files_exist_count": 0,
        "raw_dense_evidence_tensor_sha_valid_count": 0,
        "raw_dense_evidence_tensor_files_exist_fraction": 1.0,
        "raw_dense_evidence_tensor_sha_valid_fraction": 1.0,
        "raw_dense_evidence_tensor_refs_valid": 1.0,
        "raw_dense_evidence_depth_frame_count": 0,
        "raw_dense_evidence_normal_frame_count": 0,
        "raw_dense_evidence_tracks_frame_count": 0,
        "raw_dense_evidence_correspondence_frame_count": 0,
        "raw_dense_evidence_camera_prior_frame_count": 0,
        "raw_dense_evidence_render_probe_frame_count": 0,
        "raw_dense_evidence_observed_frame_count": 0,
        "raw_dense_evidence_inferred_frame_count": 0,
        "raw_dense_evidence_generated_frame_count": 0,
        "raw_dense_evidence_unknown_frame_count": 0,
        "raw_dense_evidence_generated_evidence_fraction": 0.0,
        "raw_dense_evidence_error_count": 0,
    }
    manifest_paths = _find_dense_evidence_manifest_paths(project_dir)
    if not manifest_paths:
        return metrics

    metrics["raw_dense_evidence_manifest_present"] = 1.0
    metrics["raw_dense_evidence_manifest_count"] = len(manifest_paths)
    valid_manifest_count = 0
    generated_weighted_payloads = 0.0

    for manifest_path in manifest_paths:
        try:
            manifest = read_json_model(DenseEvidenceManifest, manifest_path)
        except Exception:
            metrics["raw_dense_evidence_manifest_parse_error_count"] += 1
            continue

        valid_manifest_count += 1
        if manifest.deterministic:
            metrics["raw_dense_evidence_deterministic_manifest_count"] += 1

        metrics["raw_dense_evidence_frame_count"] += len(manifest.frames)
        for frame in manifest.frames:
            payloads = _dense_evidence_frame_payloads(frame)
            payload_count = len(payloads)
            metrics["raw_dense_evidence_payload_count"] += payload_count
            generated_weighted_payloads += (
                float(manifest.generated_evidence_fraction) * payload_count
            )
            provenance_key = f"raw_dense_evidence_{frame.provenance.value}_frame_count"
            if provenance_key in metrics:
                metrics[provenance_key] += 1

            for kind, ref in payloads:
                metrics[f"raw_dense_evidence_{kind.value}_frame_count"] += 1
                if ref is None:
                    continue
                metrics["raw_dense_evidence_tensor_ref_count"] += 1
                ref_path = _dense_evidence_tensor_path(manifest_path, ref)
                if not ref_path.exists():
                    continue
                metrics["raw_dense_evidence_tensor_files_exist_count"] += 1
                if sha256_file(ref_path) == ref.sha256:
                    metrics["raw_dense_evidence_tensor_sha_valid_count"] += 1

    manifest_count = int(metrics["raw_dense_evidence_manifest_count"])
    parse_errors = int(metrics["raw_dense_evidence_manifest_parse_error_count"])
    tensor_ref_count = int(metrics["raw_dense_evidence_tensor_ref_count"])
    tensor_file_count = int(metrics["raw_dense_evidence_tensor_files_exist_count"])
    tensor_sha_count = int(metrics["raw_dense_evidence_tensor_sha_valid_count"])
    payload_count = int(metrics["raw_dense_evidence_payload_count"])
    metrics["raw_dense_evidence_manifest_schema_valid_fraction"] = (
        valid_manifest_count / manifest_count
    )
    metrics["raw_dense_evidence_deterministic_valid"] = float(
        valid_manifest_count == manifest_count
        and int(metrics["raw_dense_evidence_deterministic_manifest_count"])
        == valid_manifest_count
    )
    metrics["raw_dense_evidence_frame_fraction"] = (
        float(metrics["raw_dense_evidence_frame_count"]) / max(expected_frame_count, 1)
    )
    if tensor_ref_count > 0:
        metrics["raw_dense_evidence_tensor_files_exist_fraction"] = (
            tensor_file_count / tensor_ref_count
        )
        metrics["raw_dense_evidence_tensor_sha_valid_fraction"] = (
            tensor_sha_count / tensor_ref_count
        )
    if payload_count > 0:
        metrics["raw_dense_evidence_generated_evidence_fraction"] = (
            generated_weighted_payloads / payload_count
        )
    tensor_refs_valid = (
        tensor_ref_count == 0
        or (tensor_file_count == tensor_ref_count and tensor_sha_count == tensor_ref_count)
    )
    metrics["raw_dense_evidence_tensor_refs_valid"] = float(tensor_refs_valid)
    metrics["raw_dense_evidence_error_count"] = (
        parse_errors
        + (tensor_ref_count - tensor_file_count)
        + (tensor_ref_count - tensor_sha_count)
    )
    metrics["raw_dense_evidence_manifest_valid"] = float(
        parse_errors == 0
        and metrics["raw_dense_evidence_deterministic_valid"] == 1.0
        and tensor_refs_valid
    )
    return metrics


def _find_dense_evidence_manifest_paths(project_dir: Path) -> list[Path]:
    candidates = [
        project_dir / "dense_evidence" / DENSE_EVIDENCE_MANIFEST_FILENAME,
        project_dir / "outputs" / "dense_evidence" / DENSE_EVIDENCE_MANIFEST_FILENAME,
        project_dir / "outputs" / "evidence" / DENSE_EVIDENCE_MANIFEST_FILENAME,
    ]
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    for path in sorted(project_dir.rglob(DENSE_EVIDENCE_MANIFEST_FILENAME)):
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _dense_evidence_frame_payloads(
    frame: DenseEvidenceFrame,
) -> list[tuple[DenseEvidenceKind, TensorRef | None]]:
    payloads: list[tuple[DenseEvidenceKind, TensorRef | None]] = []
    if frame.depth is not None:
        payloads.append((DenseEvidenceKind.DEPTH, frame.depth))
    if frame.normal is not None:
        payloads.append((DenseEvidenceKind.NORMAL, frame.normal))
    if frame.tracks_xy is not None:
        payloads.append((DenseEvidenceKind.TRACKS, frame.tracks_xy))
    if frame.correspondences is not None:
        payloads.append((DenseEvidenceKind.CORRESPONDENCE, frame.correspondences))
    if frame.camera is not None:
        payloads.append((DenseEvidenceKind.CAMERA_PRIOR, None))
    if frame.render_rgb is not None:
        payloads.append((DenseEvidenceKind.RENDER_PROBE, frame.render_rgb))
    return payloads


def _dense_evidence_tensor_path(manifest_path: Path, ref: TensorRef) -> Path:
    path = Path(ref.path).expanduser()
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _facial_feature_iris_halo_metrics(
    facial_features: dict[str, Any],
) -> dict[str, float]:
    raw_features = facial_features.get("features", [])
    if isinstance(raw_features, dict):
        feature_items = raw_features.values()
    elif isinstance(raw_features, list):
        feature_items = raw_features
    else:
        feature_items = []

    metrics: dict[str, float] = {
        "facial_feature_iris_halo_feature_count": 0.0,
        "facial_feature_iris_halo_uv_support_texel_count": 0.0,
        "facial_feature_iris_halo_core_texel_count": 0.0,
        "facial_feature_iris_halo_edge_texel_count": 0.0,
        "facial_feature_iris_halo_accepted_core_texel_count": 0.0,
        "facial_feature_iris_halo_edge_fraction": 0.0,
        "facial_feature_iris_halo_edge_to_core_ratio": 0.0,
        "facial_feature_iris_halo_max_edge_to_core_ratio": 0.0,
        "facial_feature_iris_halo_support_to_accepted_core_ratio": 0.0,
    }
    per_eye_names = {
        "left_eye_iris": "left",
        "right_eye_iris": "right",
    }
    for prefix in per_eye_names.values():
        metrics.update(
            {
                f"facial_feature_{prefix}_iris_halo_uv_support_texel_count": 0.0,
                f"facial_feature_{prefix}_iris_halo_core_texel_count": 0.0,
                f"facial_feature_{prefix}_iris_halo_edge_texel_count": 0.0,
                f"facial_feature_{prefix}_iris_halo_accepted_core_texel_count": 0.0,
                f"facial_feature_{prefix}_iris_halo_edge_fraction": 0.0,
                f"facial_feature_{prefix}_iris_halo_edge_to_core_ratio": 0.0,
                (
                    f"facial_feature_{prefix}_iris_halo_"
                    "support_to_accepted_core_ratio"
                ): 0.0,
            }
        )

    total_support = 0.0
    total_core = 0.0
    total_edge = 0.0
    total_accepted_core = 0.0
    max_edge_to_core_ratio = 0.0
    feature_count = 0.0
    for raw_feature in feature_items:
        if not isinstance(raw_feature, dict):
            continue
        feature_name = raw_feature.get("name")
        if not isinstance(feature_name, str) or feature_name not in per_eye_names:
            continue
        support = float(raw_feature.get("uv_support_mask_texel_count", 0.0))
        core = float(raw_feature.get("source_patch_core_texel_count", 0.0))
        edge = float(raw_feature.get("source_patch_feathered_edge_texel_count", 0.0))
        accepted_core = float(
            raw_feature.get("source_patch_source_mask_observed_texel_count", 0.0)
        )
        if max(support, core, edge, accepted_core) <= 0.0:
            continue

        prefix = per_eye_names[feature_name]
        edge_fraction = edge / max(support, 1.0)
        edge_to_core_ratio = edge / max(core, 1.0)
        support_to_accepted_core_ratio = support / max(accepted_core, 1.0)
        metrics[f"facial_feature_{prefix}_iris_halo_uv_support_texel_count"] = support
        metrics[f"facial_feature_{prefix}_iris_halo_core_texel_count"] = core
        metrics[f"facial_feature_{prefix}_iris_halo_edge_texel_count"] = edge
        metrics[
            f"facial_feature_{prefix}_iris_halo_accepted_core_texel_count"
        ] = accepted_core
        metrics[f"facial_feature_{prefix}_iris_halo_edge_fraction"] = edge_fraction
        metrics[
            f"facial_feature_{prefix}_iris_halo_edge_to_core_ratio"
        ] = edge_to_core_ratio
        metrics[
            f"facial_feature_{prefix}_iris_halo_support_to_accepted_core_ratio"
        ] = support_to_accepted_core_ratio

        total_support += support
        total_core += core
        total_edge += edge
        total_accepted_core += accepted_core
        max_edge_to_core_ratio = max(max_edge_to_core_ratio, edge_to_core_ratio)
        feature_count += 1.0

    metrics["facial_feature_iris_halo_feature_count"] = feature_count
    metrics["facial_feature_iris_halo_uv_support_texel_count"] = total_support
    metrics["facial_feature_iris_halo_core_texel_count"] = total_core
    metrics["facial_feature_iris_halo_edge_texel_count"] = total_edge
    metrics["facial_feature_iris_halo_accepted_core_texel_count"] = total_accepted_core
    metrics["facial_feature_iris_halo_edge_fraction"] = (
        total_edge / max(total_support, 1.0)
    )
    metrics["facial_feature_iris_halo_edge_to_core_ratio"] = (
        total_edge / max(total_core, 1.0)
    )
    metrics["facial_feature_iris_halo_max_edge_to_core_ratio"] = (
        max_edge_to_core_ratio
    )
    metrics["facial_feature_iris_halo_support_to_accepted_core_ratio"] = (
        total_support / max(total_accepted_core, 1.0)
    )
    return metrics


def _maybe_apply_template_region_v2_texture_target(
    project_dir: Path,
    *,
    coverage_result: Any,
    face_uv_density_result: Any,
    face_texture_targets_result: Any,
) -> tuple[Any, Any, Any, bool]:
    """Rebake raw textures once when the v2 target demands a larger atlas."""

    target_size = int(
        face_texture_targets_result.metrics.get(
            "template_region_v2_texture_target_size",
            0,
        )
    )
    if target_size <= 0:
        return coverage_result, face_uv_density_result, face_texture_targets_result, False
    texture_manifest = _read_json(coverage_result.texture_manifest_path)
    basecolor = _as_object(
        texture_manifest.get("basecolor"),
        "texture_manifest.basecolor",
    )
    current_size = max(
        int(basecolor.get("width", 0)),
        int(basecolor.get("height", 0)),
    )
    if current_size >= target_size:
        return coverage_result, face_uv_density_result, face_texture_targets_result, False

    coverage_result = evaluate_mesh_coverage(project_dir, mode="raw", force=True)
    face_uv_density_result = evaluate_face_uv_density_report(
        project_dir,
        mode="raw",
        force=True,
    )
    face_texture_targets_result = evaluate_face_texture_targets_report(
        project_dir,
        mode="raw",
        force=True,
    )
    return coverage_result, face_uv_density_result, face_texture_targets_result, True


def _mesh_validation_allows_texture_export_refresh(mesh_validation: dict[str, Any]) -> bool:
    return bool(
        mesh_validation.get("mode") == "raw"
        and mesh_validation.get("glb_template_regions_v2_valid", True)
    )


def _raw_observation_quality_metrics(manifest_path: Path) -> dict[str, float]:
    manifest = _read_json(manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("raw observation manifest must contain frames")

    mask_area_ratios: list[float] = []
    bbox_area_ratios: list[float] = []
    border_touch_bboxes: list[float] = []
    full_frame_bboxes: list[float] = []
    for index, frame_raw in enumerate(frames):
        frame = _as_object(frame_raw, f"raw observation frame {index}")
        quality = _as_object(frame.get("quality", {}), f"raw observation frame {index}.quality")
        mask_area_ratios.append(
            _frame_mask_area_ratio(frame)
            if "mask_area_ratio" not in quality
            else float(quality["mask_area_ratio"])
        )
        image_width, image_height = _frame_dimensions(frame)
        x0, y0, x1, y1 = _bbox_xyxy(
            frame.get("bbox_xyxy"),
            f"raw observation frame {index}.bbox_xyxy",
        )
        bbox_width = max(0.0, x1 - x0)
        bbox_height = max(0.0, y1 - y0)
        bbox_area_ratio = (bbox_width * bbox_height) / max(
            float(image_width * image_height), 1.0
        )
        touches_image_border = (
            x0 <= 0.0
            or y0 <= 0.0
            or x1 >= float(image_width)
            or y1 >= float(image_height)
        )
        full_frame_bbox = touches_image_border and bbox_area_ratio >= 0.95
        bbox_area_ratios.append(float(bbox_area_ratio))
        border_touch_bboxes.append(float(touches_image_border))
        full_frame_bboxes.append(float(full_frame_bbox))

    return {
        "mean_mask_area_ratio": float(np.mean(mask_area_ratios)),
        "mean_bbox_area_ratio": float(np.mean(bbox_area_ratios)),
        "border_touch_bbox_fraction": float(np.mean(border_touch_bboxes)),
        "full_frame_bbox_fraction": float(np.mean(full_frame_bboxes)),
    }


def _source_patch_priority_diagnostic_metrics(
    facial_features: dict[str, Any],
) -> dict[str, float]:
    diagnostics = facial_features.get("source_patch_candidate_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    direct_rows: list[dict[str, Any]] = []
    for sample_key, candidates in diagnostics.items():
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if not bool(candidate.get("source_patch_direct_photo_candidate", False)):
                continue
            row = dict(candidate)
            row["_sample_key"] = str(row.get("sample_key") or sample_key)
            direct_rows.append(row)

    zero_priority_rows = [
        row
        for row in direct_rows
        if _float_metric(row.get("source_patch_frontal_photo_priority", 0.0)) <= 0.0
    ]
    quality_below_min_rows = [
        row
        for row in zero_priority_rows
        if row.get("source_patch_frontal_photo_priority_rejection_reason")
        == "quality_below_min"
    ]
    detail_below_min_rows = [
        row
        for row in zero_priority_rows
        if row.get("source_patch_frontal_photo_priority_rejection_reason")
        == "detail_below_min"
    ]
    ear_quality_below_min_rows = [
        row
        for row in quality_below_min_rows
        if row.get("_sample_key") == "ear_inner"
    ]
    return {
        "facial_feature_source_patch_direct_photo_candidate_diagnostic_count": float(
            len(direct_rows)
        ),
        "facial_feature_source_patch_direct_photo_zero_priority_candidate_count": float(
            len(zero_priority_rows)
        ),
        "facial_feature_source_patch_direct_photo_quality_below_min_count": float(
            len(quality_below_min_rows)
        ),
        "facial_feature_source_patch_direct_photo_detail_below_min_count": float(
            len(detail_below_min_rows)
        ),
        "facial_feature_source_patch_ear_inner_direct_photo_quality_below_min_count": float(
            len(ear_quality_below_min_rows)
        ),
        "facial_feature_source_patch_direct_photo_max_quality_gap": round(
            max(
                (
                    max(
                        _float_metric(
                            row.get(
                                "source_patch_frontal_photo_priority_quality_min",
                                0.0,
                            )
                        )
                        - _float_metric(row.get("source_patch_quality_score", 0.0)),
                        0.0,
                    )
                    for row in zero_priority_rows
                ),
                default=0.0,
            ),
            6,
        ),
        "facial_feature_source_patch_direct_photo_max_detail_gap": round(
            max(
                (
                    max(
                        _float_metric(
                            row.get(
                                "source_patch_frontal_photo_priority_detail_min",
                                0.0,
                            )
                        )
                        - _float_metric(row.get("source_patch_detail_score", 0.0)),
                        0.0,
                    )
                    for row in zero_priority_rows
                ),
                default=0.0,
            ),
            6,
        ),
    }


def _float_metric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _texture_color_metrics(manifest_path: Path) -> dict[str, float]:
    manifest = _read_json(manifest_path)
    basecolor = _as_object(manifest.get("basecolor"), "texture_manifest.basecolor")
    unknown_rgb = _rgb_triplet_255(basecolor.get("unknown_rgb"), "basecolor.unknown_rgb")
    fallback_rgb = _rgb_triplet_255(
        basecolor.get("foreground_fallback_rgb", unknown_rgb),
        "basecolor.foreground_fallback_rgb",
    )
    unknown_luma = _rgb_luma_255(unknown_rgb)
    fallback_luma = _rgb_luma_255(fallback_rgb)
    facial_features = manifest.get("facial_features", {})
    if not isinstance(facial_features, dict):
        facial_features = {}
    contrast = facial_features.get("contrast", {})
    if not isinstance(contrast, dict):
        contrast = {}
    pbr = manifest.get("pbr", {})
    if not isinstance(pbr, dict):
        pbr = {}
    face_uv_region_metrics = _face_uv_region_metrics(manifest)
    iris_halo_metrics = _facial_feature_iris_halo_metrics(facial_features)
    priority_diagnostic_metrics = _source_patch_priority_diagnostic_metrics(
        facial_features
    )
    return {
        "basecolor_width": float(basecolor.get("width", 0.0)),
        "basecolor_height": float(basecolor.get("height", 0.0)),
        "basecolor_generated_texel_fraction": float(
            basecolor.get("generated_texel_fraction", 1.0)
        ),
        "face_texture_target_bake_used": float(
            basecolor.get("texture_resolution_source") == "raw_face_texture_target_1024"
        ),
        "face_texture_target_size": float(basecolor.get("face_texture_target_size", 0.0)),
        "template_region_v2_texture_target_bake_used": float(
            basecolor.get("texture_resolution_source")
            == "raw_template_region_v2_texture_target_2048"
            and float(basecolor.get("template_region_v2_texture_target_bake_used", 0.0))
            == 1.0
        ),
        "template_region_v2_texture_target_size": float(
            basecolor.get("template_region_v2_texture_target_size", 0.0)
        ),
        "template_region_v2_texture_target_scale": float(
            basecolor.get("template_region_v2_texture_target_scale", 0.0)
        ),
        "template_region_v2_texture_target_island_count": float(
            basecolor.get("template_region_v2_texture_target_island_count", 0.0)
        ),
        "template_region_v2_texture_target_facial_island_count": float(
            basecolor.get("template_region_v2_texture_target_facial_island_count", 0.0)
        ),
        "template_region_v2_texture_target_requested_texel_count": float(
            basecolor.get(
                "template_region_v2_texture_target_requested_texel_count",
                0.0,
            )
        ),
        "template_region_v2_texture_target_allocated_texel_count": float(
            basecolor.get(
                "template_region_v2_texture_target_allocated_texel_count",
                0.0,
            )
        ),
        "pbr_width": float(pbr.get("width", 0.0)),
        "pbr_height": float(pbr.get("height", 0.0)),
        "unknown_luma": unknown_luma,
        "foreground_fallback_luma": fallback_luma,
        "unknown_luma_fraction_of_foreground": unknown_luma / max(fallback_luma, 1e-12),
        "facial_feature_count": float(facial_features.get("feature_count", 0.0)),
        "facial_feature_painted_texel_count": float(
            facial_features.get("painted_texel_count", 0.0)
        ),
        "facial_feature_observed_texel_count": float(
            facial_features.get("observed_texel_count", 0.0)
        ),
        "facial_feature_observed_texel_fraction": float(
            facial_features.get("observed_texel_fraction", 0.0)
        ),
        "facial_feature_source_patch_count": float(
            facial_features.get("source_patch_count", 0.0)
        ),
        "facial_feature_source_patch_feedback_sample_count": float(
            facial_features.get("source_patch_feedback_sample_count", 0.0)
        ),
        "facial_feature_source_patch_feedback_used_count": float(
            facial_features.get("source_patch_feedback_used_count", 0.0)
        ),
        "facial_feature_source_patch_bake_candidate_count": float(
            facial_features.get("source_patch_bake_candidate_count", 0.0)
        ),
        "facial_feature_source_patch_bake_selected_feedback": float(
            facial_features.get("source_patch_bake_selected_candidate") == "feedback"
        ),
        "facial_feature_source_patch_bake_visual_quality_passed_count": float(
            facial_features.get("source_patch_bake_visual_quality_passed_count", 0.0)
        ),
        "facial_feature_source_patch_bake_visual_quality_min_score": float(
            facial_features.get("source_patch_bake_visual_quality_min_score", 0.0)
        ),
        "facial_feature_source_patch_bake_visual_quality_mean_score": float(
            facial_features.get("source_patch_bake_visual_quality_mean_score", 0.0)
        ),
        "facial_feature_eye_source_detail_reinforcement_feature_count": float(
            facial_features.get("eye_source_detail_reinforcement_feature_count", 0.0)
        ),
        "facial_feature_eye_source_detail_reinforcement_texel_count": float(
            facial_features.get("eye_source_detail_reinforcement_texel_count", 0.0)
        ),
        "facial_feature_eye_source_detail_reinforcement_texel_fraction": float(
            facial_features.get("eye_source_detail_reinforcement_texel_fraction", 0.0)
        ),
        "facial_feature_whisker_pad_feature_count": float(
            facial_features.get("whisker_pad_feature_count", 0.0)
        ),
        "facial_feature_whisker_pad_texel_count": float(
            facial_features.get("whisker_pad_texel_count", 0.0)
        ),
        "facial_feature_whisker_pad_texel_fraction": float(
            facial_features.get("whisker_pad_texel_fraction", 0.0)
        ),
        "facial_feature_iris_halo_feature_count": float(
            iris_halo_metrics["facial_feature_iris_halo_feature_count"]
        ),
        "facial_feature_iris_halo_uv_support_texel_count": float(
            iris_halo_metrics["facial_feature_iris_halo_uv_support_texel_count"]
        ),
        "facial_feature_iris_halo_core_texel_count": float(
            iris_halo_metrics["facial_feature_iris_halo_core_texel_count"]
        ),
        "facial_feature_iris_halo_edge_texel_count": float(
            iris_halo_metrics["facial_feature_iris_halo_edge_texel_count"]
        ),
        "facial_feature_iris_halo_accepted_core_texel_count": float(
            iris_halo_metrics["facial_feature_iris_halo_accepted_core_texel_count"]
        ),
        "facial_feature_iris_halo_edge_fraction": float(
            iris_halo_metrics["facial_feature_iris_halo_edge_fraction"]
        ),
        "facial_feature_iris_halo_edge_to_core_ratio": float(
            iris_halo_metrics["facial_feature_iris_halo_edge_to_core_ratio"]
        ),
        "facial_feature_iris_halo_max_edge_to_core_ratio": float(
            iris_halo_metrics["facial_feature_iris_halo_max_edge_to_core_ratio"]
        ),
        "facial_feature_iris_halo_support_to_accepted_core_ratio": float(
            iris_halo_metrics[
                "facial_feature_iris_halo_support_to_accepted_core_ratio"
            ]
        ),
        "facial_feature_left_iris_halo_uv_support_texel_count": float(
            iris_halo_metrics[
                "facial_feature_left_iris_halo_uv_support_texel_count"
            ]
        ),
        "facial_feature_left_iris_halo_core_texel_count": float(
            iris_halo_metrics["facial_feature_left_iris_halo_core_texel_count"]
        ),
        "facial_feature_left_iris_halo_edge_texel_count": float(
            iris_halo_metrics["facial_feature_left_iris_halo_edge_texel_count"]
        ),
        "facial_feature_left_iris_halo_accepted_core_texel_count": float(
            iris_halo_metrics[
                "facial_feature_left_iris_halo_accepted_core_texel_count"
            ]
        ),
        "facial_feature_left_iris_halo_edge_fraction": float(
            iris_halo_metrics["facial_feature_left_iris_halo_edge_fraction"]
        ),
        "facial_feature_left_iris_halo_edge_to_core_ratio": float(
            iris_halo_metrics[
                "facial_feature_left_iris_halo_edge_to_core_ratio"
            ]
        ),
        "facial_feature_left_iris_halo_support_to_accepted_core_ratio": float(
            iris_halo_metrics[
                "facial_feature_left_iris_halo_support_to_accepted_core_ratio"
            ]
        ),
        "facial_feature_right_iris_halo_uv_support_texel_count": float(
            iris_halo_metrics[
                "facial_feature_right_iris_halo_uv_support_texel_count"
            ]
        ),
        "facial_feature_right_iris_halo_core_texel_count": float(
            iris_halo_metrics["facial_feature_right_iris_halo_core_texel_count"]
        ),
        "facial_feature_right_iris_halo_edge_texel_count": float(
            iris_halo_metrics["facial_feature_right_iris_halo_edge_texel_count"]
        ),
        "facial_feature_right_iris_halo_accepted_core_texel_count": float(
            iris_halo_metrics[
                "facial_feature_right_iris_halo_accepted_core_texel_count"
            ]
        ),
        "facial_feature_right_iris_halo_edge_fraction": float(
            iris_halo_metrics["facial_feature_right_iris_halo_edge_fraction"]
        ),
        "facial_feature_right_iris_halo_edge_to_core_ratio": float(
            iris_halo_metrics[
                "facial_feature_right_iris_halo_edge_to_core_ratio"
            ]
        ),
        "facial_feature_right_iris_halo_support_to_accepted_core_ratio": float(
            iris_halo_metrics[
                "facial_feature_right_iris_halo_support_to_accepted_core_ratio"
            ]
        ),
        "facial_feature_source_patch_texel_count": float(
            facial_features.get("source_patch_texel_count", 0.0)
        ),
        "facial_feature_source_patch_texel_fraction": float(
            facial_features.get("source_patch_texel_fraction", 0.0)
        ),
        "facial_feature_source_patch_quality_passed_count": float(
            facial_features.get("source_patch_quality_passed_count", 0.0)
        ),
        "facial_feature_source_patch_quality_total_count": float(
            facial_features.get("source_patch_quality_total_count", 0.0)
        ),
        "facial_feature_source_patch_quality_fraction": float(
            facial_features.get("source_patch_quality_fraction", 0.0)
        ),
        "facial_feature_source_patch_min_quality_score": float(
            facial_features.get("source_patch_min_quality_score", 0.0)
        ),
        "facial_feature_source_patch_mean_quality_score": float(
            facial_features.get("source_patch_mean_quality_score", 0.0)
        ),
        "facial_feature_source_patch_view_weighting_used": float(
            bool(facial_features.get("source_patch_view_weighting_used", False))
        ),
        "facial_feature_source_patch_view_weighted_count": float(
            facial_features.get("source_patch_view_weighted_count", 0.0)
        ),
        "facial_feature_source_patch_min_view_weight": float(
            facial_features.get("source_patch_min_view_weight", 0.0)
        ),
        "facial_feature_source_patch_mean_view_weight": float(
            facial_features.get("source_patch_mean_view_weight", 0.0)
        ),
        "facial_feature_source_patch_min_frontal_face_confidence": float(
            facial_features.get("source_patch_min_frontal_face_confidence", 0.0)
        ),
        "facial_feature_source_patch_mean_frontal_face_confidence": float(
            facial_features.get("source_patch_mean_frontal_face_confidence", 0.0)
        ),
        "facial_feature_source_patch_min_view_weighted_rank": float(
            facial_features.get("source_patch_min_view_weighted_rank", 0.0)
        ),
        "facial_feature_source_patch_mean_view_weighted_rank": float(
            facial_features.get("source_patch_mean_view_weighted_rank", 0.0)
        ),
        "facial_feature_source_patch_frontal_photo_priority_used": float(
            bool(facial_features.get("source_patch_frontal_photo_priority_used", False))
        ),
        "facial_feature_source_patch_frontal_photo_priority_count": float(
            facial_features.get("source_patch_frontal_photo_priority_count", 0.0)
        ),
        "facial_feature_source_patch_max_frontal_photo_priority": float(
            facial_features.get("source_patch_max_frontal_photo_priority", 0.0)
        ),
        "facial_feature_source_patch_mean_frontal_photo_priority": float(
            facial_features.get("source_patch_mean_frontal_photo_priority", 0.0)
        ),
        **priority_diagnostic_metrics,
        "facial_feature_source_patch_fusion_min_weight": float(
            facial_features.get("source_patch_fusion_min_weight", 0.0)
        ),
        "facial_feature_source_patch_fusion_mean_weight": float(
            facial_features.get("source_patch_fusion_mean_weight", 0.0)
        ),
        "facial_feature_source_patch_fusion_weighted_texel_count": float(
            facial_features.get("source_patch_fusion_weighted_texel_count", 0.0)
        ),
        "facial_feature_source_patch_nominal_texel_count": float(
            facial_features.get("source_patch_nominal_texel_count", 0.0)
        ),
        "facial_feature_source_patch_footprint_reduction_fraction": float(
            facial_features.get("source_patch_footprint_reduction_fraction", 0.0)
        ),
        "facial_feature_source_patch_mean_footprint_area_scale": float(
            facial_features.get("source_patch_mean_footprint_area_scale", 1.0)
        ),
        "facial_feature_source_patch_min_footprint_area_scale": float(
            facial_features.get("source_patch_min_footprint_area_scale", 1.0)
        ),
        "facial_feature_source_patch_shape_mask_texel_count": float(
            facial_features.get("source_patch_shape_mask_texel_count", 0.0)
        ),
        "facial_feature_source_patch_shape_mask_fraction": float(
            facial_features.get("source_patch_shape_mask_fraction", 0.0)
        ),
        "facial_feature_source_patch_shape_alpha_mean": float(
            facial_features.get("source_patch_shape_alpha_mean", 0.0)
        ),
        "facial_feature_source_patch_source_mask_observed_texel_count": float(
            facial_features.get("source_patch_source_mask_observed_texel_count", 0.0)
        ),
        "facial_feature_source_patch_source_mask_densified_texel_count": float(
            facial_features.get("source_patch_source_mask_densified_texel_count", 0.0)
        ),
        "facial_feature_source_patch_source_mask_densification_gain": float(
            facial_features.get("source_patch_source_mask_densification_gain", 0.0)
        ),
        "facial_feature_source_patch_mean_source_mask_densification_gain": float(
            facial_features.get(
                "source_patch_mean_source_mask_densification_gain",
                0.0,
            )
        ),
        "facial_feature_source_patch_localization_candidate_texel_count": float(
            facial_features.get(
                "source_patch_localization_candidate_texel_count",
                0.0,
            )
        ),
        "facial_feature_source_patch_localization_selected_texel_count": float(
            facial_features.get(
                "source_patch_localization_selected_texel_count",
                0.0,
            )
        ),
        "facial_feature_source_patch_localization_fraction": float(
            facial_features.get("source_patch_localization_fraction", 0.0)
        ),
        "facial_feature_source_patch_mean_localization_fraction": float(
            facial_features.get("source_patch_mean_localization_fraction", 0.0)
        ),
        "facial_feature_source_patch_landmark_aligned_count": float(
            facial_features.get("source_patch_landmark_aligned_count", 0.0)
        ),
        "facial_feature_source_patch_mean_landmark_confidence": float(
            facial_features.get("source_patch_mean_landmark_confidence", 0.0)
        ),
        "facial_feature_source_patch_observed_anchor_guided_count": float(
            facial_features.get("source_patch_observed_anchor_guided_count", 0.0)
        ),
        "facial_feature_source_patch_observed_anchor_guided_fraction": float(
            facial_features.get("source_patch_observed_anchor_guided_fraction", 0.0)
        ),
        "facial_feature_source_patch_landmark_destination_scaled_count": float(
            facial_features.get(
                "source_patch_landmark_destination_scaled_count",
                0.0,
            )
        ),
        "facial_feature_source_patch_mean_landmark_destination_scale": float(
            facial_features.get(
                "source_patch_mean_landmark_destination_scale",
                1.0,
            )
        ),
        "facial_feature_destination_anchor_anatomical_count": float(
            facial_features.get("destination_anchor_anatomical_count", 0.0)
        ),
        "facial_feature_destination_anchor_order_valid": float(
            bool(facial_features.get("destination_anchor_order_valid", False))
        ),
        "facial_feature_destination_anchor_eye_to_nose_v_gap": float(
            facial_features.get("destination_anchor_eye_to_nose_v_gap", 0.0)
        ),
        "facial_feature_destination_anchor_nose_to_mouth_v_gap": float(
            facial_features.get("destination_anchor_nose_to_mouth_v_gap", 0.0)
        ),
        "facial_feature_destination_anchor_nose_to_muzzle_v_gap": float(
            facial_features.get("destination_anchor_nose_to_muzzle_v_gap", 0.0)
        ),
        "facial_feature_destination_anchor_ear_to_eye_v_gap": float(
            facial_features.get("destination_anchor_ear_to_eye_v_gap", 0.0)
        ),
        "facial_feature_destination_anchor_eye_symmetry_u_error": float(
            facial_features.get("destination_anchor_eye_symmetry_u_error", 1.0)
        ),
        "facial_feature_source_patch_core_texel_count": float(
            facial_features.get("source_patch_core_texel_count", 0.0)
        ),
        "facial_feature_source_patch_core_fraction": float(
            facial_features.get("source_patch_core_fraction", 0.0)
        ),
        "facial_feature_source_patch_feathered_edge_texel_count": float(
            facial_features.get("source_patch_feathered_edge_texel_count", 0.0)
        ),
        "facial_feature_source_patch_feathered_edge_fraction": float(
            facial_features.get("source_patch_feathered_edge_fraction", 0.0)
        ),
        "facial_feature_source_patch_mean_blend_alpha": float(
            facial_features.get("source_patch_mean_blend_alpha", 0.0)
        ),
        "face_priority_texture_used": float(
            facial_features.get("source") == "observed_face_priority_texture_projection_v1"
        ),
        "face_uv_region_count": float(face_uv_region_metrics["region_count"]),
        "face_uv_region_total_area": float(face_uv_region_metrics["total_area"]),
        "face_uv_region_min_density_weight": float(
            face_uv_region_metrics["min_density_weight"]
        ),
        "face_uv_region_contract_valid": float(face_uv_region_metrics["contract_valid"]),
        "face_feature_min_required_contrast": float(
            contrast.get("min_required_contrast", 0.0)
        ),
        "face_feature_eye_to_muzzle_luma_delta": float(
            contrast.get("eye_to_muzzle_luma_delta", 0.0)
        ),
        "face_feature_mouth_to_muzzle_luma_delta": float(
            contrast.get("mouth_to_muzzle_luma_delta", 0.0)
        ),
        "face_feature_nose_to_muzzle_rgb_distance": float(
            contrast.get("nose_to_muzzle_rgb_distance", 0.0)
        ),
        "face_feature_iris_to_pupil_rgb_distance": float(
            contrast.get("iris_to_pupil_rgb_distance", 0.0)
        ),
        "face_feature_required_contrast_passed": float(
            bool(contrast.get("required_contrast_passed", False))
        ),
    }


def _face_uv_region_metrics(manifest: dict[str, Any]) -> dict[str, float | int]:
    face_uv_regions = manifest.get("template_face_uv_regions", {})
    if not isinstance(face_uv_regions, dict):
        face_uv_regions = {}
    regions = face_uv_regions.get("regions", [])
    if not isinstance(regions, list):
        regions = []
    required_names = {
        "face",
        "left_eye",
        "right_eye",
        "nose",
        "muzzle",
        "mouth_line",
        "left_ear_inner",
        "right_ear_inner",
    }
    names: set[str] = set()
    total_area = 0.0
    density_weights: list[float] = []
    bounds_valid = True
    for region in regions:
        if not isinstance(region, dict):
            bounds_valid = False
            continue
        names.add(str(region.get("name", "")))
        bounds = region.get("uv_bounds", [])
        if not isinstance(bounds, list) or len(bounds) != 4:
            bounds_valid = False
            continue
        u0, v0, u1, v1 = (float(value) for value in bounds)
        bounds_valid = bounds_valid and 0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0
        total_area += max(0.0, u1 - u0) * max(0.0, v1 - v0)
        density_weights.append(float(region.get("target_texel_density", 0.0)))
    contract_valid = (
        face_uv_regions.get("schema_version") == "1.0"
        and face_uv_regions.get("source") == "opencat3d.cat_template_face_uv_region_contract"
        and int(face_uv_regions.get("region_count", -1)) == len(regions)
        and required_names.issubset(names)
        and bounds_valid
        and all(weight > 0.0 for weight in density_weights)
    )
    return {
        "region_count": int(len(regions)),
        "total_area": float(total_area),
        "min_density_weight": float(min(density_weights, default=0.0)),
        "contract_valid": float(contract_valid),
    }


def _template_region_v2_metrics(mesh_validation: dict[str, Any]) -> dict[str, float | int]:
    manifest = mesh_validation.get("template_regions_v2", {})
    if not isinstance(manifest, dict):
        manifest = {}
    regions = manifest.get("regions", [])
    if not isinstance(regions, list):
        regions = []
    required_names = set(CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES)
    names: set[str] = set()
    belly_vertex_count = 0
    positive_counts = True
    for region in regions:
        if not isinstance(region, dict):
            positive_counts = False
            continue
        name = str(region.get("name", ""))
        names.add(name)
        vertex_count = int(region.get("vertex_count", 0))
        face_count = int(region.get("face_count", 0))
        positive_counts = positive_counts and vertex_count > 0 and face_count > 0
        if name == "belly":
            belly_vertex_count = vertex_count
    validation = manifest.get("validation", {})
    if not isinstance(validation, dict):
        validation = {}
    required_present_count = len(required_names.intersection(names))
    contract_valid = (
        manifest.get("schema_version") == "2.0"
        and manifest.get("source") == "opencat3d.cat_template_region_contract_v2"
        and manifest.get("target_representation") == "fixed_topology_cat_template"
        and manifest.get("topology_changed") is False
        and int(manifest.get("region_count", -1)) == len(regions)
        and required_present_count == len(required_names)
        and positive_counts
        and bool(validation.get("valid", False))
        and bool(mesh_validation.get("template_region_v2_contract_valid", False))
    )
    return {
        "region_count": int(len(regions)),
        "required_region_count": int(len(required_names)),
        "required_present_count": int(required_present_count),
        "contract_valid": float(contract_valid),
        "glb_contract_valid": float(
            bool(mesh_validation.get("glb_template_regions_v2_valid", False))
        ),
        "belly_vertex_count": int(belly_vertex_count),
    }


def _geometry_detail_metrics(mesh_validation: dict[str, Any]) -> dict[str, float | int]:
    detail = mesh_validation.get("geometry_detail", {})
    if not isinstance(detail, dict):
        detail = {}
    return {
        "face_detail_geometry_applied": float(bool(detail.get("applied", False))),
        "face_detail_deformed_vertex_count": int(detail.get("deformed_vertex_count", 0)),
        "face_detail_max_displacement": float(detail.get("max_displacement", 0.0)),
        "face_detail_mean_displacement": float(detail.get("mean_displacement", 0.0)),
        "face_surface_anchor_count": int(detail.get("surface_anchor_count", 0)),
        "face_surface_anchor_vertex_count": int(
            detail.get("surface_anchor_vertex_count", 0)
        ),
        "face_surface_anchor_max_displacement": float(
            detail.get("surface_anchor_max_displacement", 0.0)
        ),
    }


def _rgb_triplet_255(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three RGB values")
    rgb = tuple(float(channel) for channel in value)
    if any(channel < 0.0 or channel > 255.0 for channel in rgb):
        raise ValueError(f"{name} values must be in [0,255]")
    return rgb


def _rgb_luma_255(rgb: tuple[float, float, float]) -> float:
    return float((0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0)


def _frame_dimensions(frame: dict[str, Any]) -> tuple[int, int]:
    if "image_width" in frame and "image_height" in frame:
        return int(frame["image_width"]), int(frame["image_height"])

    image_path = Path(str(frame.get("image_srgb_path", "")))
    if not image_path.exists():
        raise FileNotFoundError(f"{image_path} is required for raw observation quality")
    with np.load(image_path) as data:
        image = np.asarray(data["image_srgb"])
    if image.ndim != 3:
        raise ValueError("image_srgb must have three dimensions")
    if image.shape[0] in {1, 3}:
        height, width = int(image.shape[1]), int(image.shape[2])
    elif image.shape[2] in {1, 3}:
        height, width = int(image.shape[0]), int(image.shape[1])
    else:
        raise ValueError("image_srgb must contain a channel dimension")
    return width, height


def _frame_mask_area_ratio(frame: dict[str, Any]) -> float:
    hard_mask = _as_object(frame.get("hard_mask"), "raw observation frame.hard_mask")
    mask_path = Path(str(hard_mask.get("path", "")))
    if not mask_path.exists():
        raise FileNotFoundError(f"{mask_path} is required for raw observation quality")
    with np.load(mask_path) as data:
        mask = np.asarray(data["hard_mask"], dtype=np.float32)
    return float(np.mean(mask > 0.0))


def _bbox_xyxy(value: object, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must contain four numbers")
    x0, y0, x1, y1 = (float(item) for item in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{name} must satisfy x1 > x0 and y1 > y0")
    return x0, y0, x1, y1


def _overlay_manifest_valid(path: Path, *, expected_frame_count: int) -> tuple[bool, float]:
    if not path.exists():
        return False, 0.0
    try:
        payload = _read_json(path)
        frames = payload.get("frames", [])
        if not isinstance(frames, list):
            return False, 0.0
        overlays_exist = all(
            isinstance(frame, dict) and Path(str(frame.get("overlay_path", ""))).exists()
            for frame in frames
        )
        frame_count = int(payload.get("frame_count", -1))
        mean_iou = float(payload.get("mean_silhouette_iou", 0.0))
    except (OSError, ValueError, TypeError):
        return False, 0.0
    valid = (
        frame_count == expected_frame_count
        and len(frames) == expected_frame_count
        and overlays_exist
    )
    return valid, mean_iou


def _fit_state_finite(fit_state: FitState) -> bool:
    arrays = [
        fit_state.beta,
        fit_state.target_beta,
        fit_state.initial_beta,
        fit_state.root_translations,
        fit_state.initial_root_translations,
        fit_state.poses_6d,
        fit_state.initial_poses_6d,
    ]
    return all(np.asarray(array).size > 0 and np.all(np.isfinite(array)) for array in arrays)


def _is_finite(value: float) -> bool:
    return value not in {float("inf"), float("-inf")} and value == value
