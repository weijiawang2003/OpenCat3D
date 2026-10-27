import json
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from opencat3d.adapters.dense_evidence import (
    DenseEvidencePayload,
    DenseEvidenceRequest,
    LocalFileDenseEvidenceProvider,
)
from opencat3d.cli import app
from opencat3d.evaluation.raw import (
    RAW_METRIC_THRESHOLDS,
    _dense_evidence_metrics,
    _source_patch_priority_diagnostic_metrics,
    evaluate_raw_reconstruction,
)
from opencat3d.export.mesh import export_fitted_canonical_mesh
from opencat3d.fitting.optimizer import OptimizerConfig, run_raw_fit
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.preprocess.keypoints import bootstrap_raw_keypoints_and_pose
from opencat3d.preprocess.segmentation import segment_raw_image_observations
from opencat3d.schemas import DenseEvidenceKind, EvidenceProvenance


def test_evaluate_raw_reconstruction_writes_quality_report(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_exported_project(project_dir)
    _write_dense_evidence_fixture(project_dir)

    result = evaluate_raw_reconstruction(project_dir)

    assert result.success
    assert result.report_path.exists()
    assert all(result.checks.values())
    assert result.metrics["num_frames"] == 2
    assert result.metrics["loss_reduction"] >= 0.0
    assert result.metrics["vertex_count"] == RAW_METRIC_THRESHOLDS["expected_vertex_count"]
    assert result.metrics["face_count"] == RAW_METRIC_THRESHOLDS["expected_face_count"]
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["thresholds"] == RAW_METRIC_THRESHOLDS
    assert report["checks"]["raw_fit_mode"]
    assert report["checks"]["raw_export_mode"]
    assert report["checks"]["glb_structure_valid"]
    assert report["checks"]["glb_basecolor_texture_valid"]
    assert report["checks"]["glb_pbr_textures_valid"]
    assert report["checks"]["glb_face_detail_geometry_valid"]
    assert report["checks"]["raw_face_uv_regions_valid"]
    assert report["checks"]["raw_face_uv_texture_density_report_valid"]
    assert report["checks"]["raw_template_region_v2_contract_valid"]
    assert report["checks"]["raw_face_texture_targets_report_valid"]
    assert report["checks"]["raw_template_region_v2_texture_targets_report_valid"]
    assert report["checks"]["raw_face_texture_target_bake_consistent"]
    assert report["checks"]["raw_face_texture_crop_diagnostic_valid"]
    assert report["checks"]["raw_face_texture_crop_exact_source_patches_valid"]
    assert report["checks"]["raw_face_texture_crop_isolation_valid"]
    assert report["checks"]["raw_face_texture_crop_visual_quality_valid"]
    assert report["checks"]["raw_face_texture_bake_comparison_valid"]
    assert report["checks"]["raw_glb_face_feature_material_visibility"]
    assert report["checks"]["glb_skinning_valid"]
    assert report["checks"]["glb_idle_animation_valid"]
    assert report["checks"]["skeleton_metadata_valid"]
    assert report["checks"]["skeleton_skin_weights_valid"]
    assert report["checks"]["fit_overlays_present"]
    assert report["checks"]["raw_mask_area_reasonable"]
    assert report["checks"]["raw_bbox_area_reasonable"]
    assert report["checks"]["raw_full_frame_bbox_fraction_reasonable"]
    assert report["checks"]["raw_overlay_alignment_quality"]
    assert report["checks"]["raw_silhouette_residual_quality"]
    assert report["checks"]["raw_texture_unknown_color_reasonable"]
    assert report["checks"]["raw_face_feature_contrast_quality"]
    assert report["checks"]["raw_face_feature_observed_provenance_quality"]
    assert report["checks"]["raw_face_feature_source_patch_quality"]
    assert report["checks"]["raw_face_feature_source_patch_quality_score"]
    assert report["checks"]["raw_face_feature_source_patch_fusion_weight"]
    assert report["checks"]["raw_face_feature_source_patch_feathering"]
    assert report["checks"]["raw_face_feature_source_patch_core_preserved"]
    assert report["checks"]["raw_face_feature_source_patch_footprint_adaptive"]
    assert report["checks"]["raw_face_feature_source_patch_shape_masked"]
    assert report["checks"]["raw_face_feature_source_patch_source_mask_densified"]
    assert report["checks"]["raw_face_feature_source_patch_component_localized"]
    assert report["checks"]["raw_face_feature_source_patch_landmark_aligned"]
    assert report["checks"]["raw_face_feature_source_patch_landmark_destination_scaled"]
    assert report["checks"]["raw_face_feature_destination_anchor_anatomical"]
    assert report["checks"]["coverage_report_valid"]
    assert report["checks"]["texture_basecolor_valid"]
    assert report["checks"]["pbr_texture_outputs_valid"]
    assert report["checks"]["raw_pbr_face_material_detail_quality"]
    assert report["checks"]["joint_projection_report_valid"]
    assert report["checks"]["silhouette_residual_maps_valid"]
    assert report["checks"]["artifact_validation_report_valid"]
    assert report["checks"]["face_turntable_report_valid"]
    assert report["checks"]["raw_dense_evidence_manifest_valid"]
    assert report["checks"]["raw_dense_evidence_tensor_refs_valid"]
    assert report["checks"]["keypoint_loss_reduced"]
    assert report["checks"]["silhouette_loss_reduced"]
    assert report["metrics"]["keypoint_loss_reduction"] > 0.0
    assert report["metrics"]["silhouette_loss_reduction"] > 0.0
    assert report["metrics"]["joint_count"] == 11
    assert report["metrics"]["raw_mean_mask_area_ratio"] <= RAW_METRIC_THRESHOLDS[
        "max_mean_mask_area_ratio"
    ]
    assert report["metrics"]["raw_mean_bbox_area_ratio"] <= RAW_METRIC_THRESHOLDS[
        "max_mean_bbox_area_ratio"
    ]
    assert report["metrics"]["raw_full_frame_bbox_fraction"] == 0.0
    assert report["metrics"]["raw_border_touch_bbox_fraction"] == 0.0
    assert report["metrics"]["raw_dense_evidence_manifest_present"] == 1.0
    assert report["metrics"]["raw_dense_evidence_manifest_count"] == 1
    assert report["metrics"]["raw_dense_evidence_manifest_valid"] == 1.0
    assert report["metrics"]["raw_dense_evidence_frame_count"] == 1
    assert report["metrics"]["raw_dense_evidence_frame_fraction"] == 0.5
    assert report["metrics"]["raw_dense_evidence_payload_count"] == 2
    assert report["metrics"]["raw_dense_evidence_depth_frame_count"] == 1
    assert report["metrics"]["raw_dense_evidence_tracks_frame_count"] == 1
    assert report["metrics"]["raw_dense_evidence_tensor_files_exist_fraction"] == 1.0
    assert report["metrics"]["raw_dense_evidence_tensor_sha_valid_fraction"] == 1.0
    assert report["metrics"]["raw_dense_evidence_generated_evidence_fraction"] == 0.0
    assert report["metrics"]["raw_texture_unknown_luma_fraction_of_foreground"] >= (
        RAW_METRIC_THRESHOLDS["min_unknown_texture_luma_fraction_of_foreground"]
    )
    assert report["checks"]["raw_texture_resolution_quality"]
    assert report["checks"]["raw_template_region_v2_texture_target_bake_consistent"]
    assert report["metrics"]["raw_texture_width"] == RAW_METRIC_THRESHOLDS[
        "min_template_region_v2_texture_target_baked_size"
    ]
    assert report["metrics"]["raw_texture_height"] == RAW_METRIC_THRESHOLDS[
        "min_template_region_v2_texture_target_baked_size"
    ]
    assert report["metrics"]["raw_texture_resolution_scale"] == 8.0
    assert report["metrics"]["raw_face_texture_target_bake_used"] == 0.0
    assert report["metrics"]["raw_face_texture_target_reported_size"] == (
        report["metrics"]["raw_texture_width"]
    )
    assert report["metrics"]["raw_template_region_v2_texture_target_bake_used"] == 1.0
    assert report["metrics"]["raw_template_region_v2_texture_target_reported_size"] == (
        report["metrics"]["raw_texture_width"]
    )
    assert report[
        "metrics"
    ]["raw_template_region_v2_texture_target_reported_allocated_texel_count"] >= (
        report[
            "metrics"
        ]["raw_template_region_v2_texture_target_reported_requested_texel_count"]
    )
    assert report["metrics"]["raw_pbr_texture_width"] == report["metrics"]["raw_texture_width"]
    assert report["metrics"]["raw_pbr_texture_height"] == report["metrics"]["raw_texture_height"]
    assert report["metrics"]["raw_glb_texture_size"] == report["metrics"]["raw_texture_width"]
    assert report["metrics"]["raw_glb_texture_dimensions_valid"] == 1.0
    assert report["metrics"]["raw_glb_pbr_texture_dimensions_valid"] == 1.0
    assert report["metrics"]["raw_glb_basecolor_texture_valid"] == 1.0
    assert report["metrics"]["raw_glb_pbr_textures_valid"] == 1.0
    assert report["metrics"]["raw_glb_embedded_texture_dimension_gate_valid"] == 1.0
    assert report["metrics"]["raw_glb_face_feature_required_group_count"] == (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_visible_group_count"]
    )
    assert report["metrics"]["raw_glb_face_feature_visible_group_count"] == (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_visible_group_count"]
    )
    assert report["metrics"]["raw_glb_face_feature_required_feature_count"] == (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_visible_feature_count"]
    )
    assert report["metrics"]["raw_glb_face_feature_present_required_feature_count"] == (
        report["metrics"]["raw_glb_face_feature_required_feature_count"]
    )
    assert report["metrics"]["raw_glb_face_feature_visible_feature_count"] == (
        report["metrics"]["raw_glb_face_feature_required_feature_count"]
    )
    assert report["metrics"]["raw_glb_face_feature_sampled_texel_count"] > 0
    assert report["metrics"]["raw_glb_face_feature_min_visible_texel_count"] > 0
    assert report["metrics"]["raw_glb_face_feature_mean_rgb_error"] <= (
        RAW_METRIC_THRESHOLDS["max_glb_face_feature_mean_rgb_error"]
    )
    assert report["metrics"]["raw_glb_face_feature_max_rgb_error"] <= (
        RAW_METRIC_THRESHOLDS["max_glb_face_feature_max_rgb_error"]
    )
    assert report["metrics"]["raw_glb_face_feature_eye_visible"] == 1.0
    assert report["metrics"]["raw_glb_face_feature_nose_visible"] == 1.0

    assert report["metrics"]["raw_glb_face_feature_muzzle_visible"] == 1.0
    assert report["metrics"]["raw_glb_face_feature_iris_to_pupil_rgb_distance"] >= (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_rgb_distance"]
    )
    assert report["metrics"]["raw_glb_face_feature_eye_detail_luma_std"] > 0.0
    assert report["metrics"]["raw_glb_face_feature_nose_to_muzzle_rgb_distance"] >= (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_rgb_distance"]
    )
    assert report["metrics"]["raw_glb_face_feature_mouth_to_muzzle_rgb_distance"] >= (
        RAW_METRIC_THRESHOLDS["min_glb_face_feature_rgb_distance"]
    )
    assert report["metrics"]["raw_glb_face_feature_material_visibility_gate_valid"] == 1.0
    assert report["metrics"]["raw_face_uv_region_count"] == RAW_METRIC_THRESHOLDS[
        "min_face_uv_region_count"
    ]
    assert report["metrics"]["raw_face_uv_region_total_area"] > 0.0
    assert report["metrics"]["raw_face_uv_region_min_density_weight"] > 0.0
    assert report["metrics"]["raw_face_uv_region_contract_valid"] == 1.0
    assert report["metrics"]["raw_template_region_v2_count"] == RAW_METRIC_THRESHOLDS[
        "min_template_region_v2_count"
    ]
    assert report["metrics"]["raw_template_region_v2_required_region_count"] == (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_count"]
    )
    assert report["metrics"]["raw_template_region_v2_required_present_count"] == (
        report["metrics"]["raw_template_region_v2_required_region_count"]
    )
    assert report["metrics"]["raw_template_region_v2_contract_valid"] == 1.0
    assert report["metrics"]["raw_glb_template_region_v2_contract_valid"] == 1.0
    assert report["metrics"]["raw_template_region_v2_belly_vertex_count"] > 0
    assert report["metrics"]["raw_face_uv_density_allocated_texel_count"] > 0
    assert report["metrics"]["raw_face_uv_density_min_allocated_texels"] > 0
    assert report["metrics"]["raw_face_uv_density_eye_min_allocated_texels"] >= (
        RAW_METRIC_THRESHOLDS["min_face_uv_eye_texels"]
    )
    assert report["metrics"]["raw_face_uv_density_nose_allocated_texels"] >= (
        RAW_METRIC_THRESHOLDS["min_face_uv_nose_texels"]
    )
    assert report["metrics"]["raw_face_uv_density_ear_inner_min_allocated_texels"] >= (
        RAW_METRIC_THRESHOLDS["min_face_uv_ear_inner_texels"]
    )
    assert report["checks"]["raw_template_region_v2_uv_island_density_report_valid"]
    assert report["metrics"]["raw_template_region_v2_uv_region_count"] == (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_uv_region_count"]
    )
    assert report["metrics"]["raw_template_region_v2_uv_required_present_count"] == (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_uv_region_count"]
    )
    assert report["metrics"]["raw_template_region_v2_uv_island_count"] >= 13
    assert report["metrics"]["raw_template_region_v2_region_scoped_uv_island_count"] > (
        report["metrics"]["raw_template_region_v2_uv_island_count"]
    )
    assert report["metrics"]["raw_template_region_v2_uv_island_allocated_texel_count"] > 0
    assert report[
        "metrics"
    ]["raw_template_region_v2_uv_island_density_weighted_texel_count"] > (
        report["metrics"]["raw_template_region_v2_uv_island_allocated_texel_count"]
    )
    assert report["metrics"]["raw_template_region_v2_face_uv_island_count"] == 8
    assert report["metrics"]["raw_template_region_v2_facial_uv_island_count"] == 8
    assert report[
        "metrics"
    ]["raw_template_region_v2_facial_uv_island_allocated_texel_count"] > 0
    assert report["metrics"]["raw_template_region_v2_body_region_uv_island_count"] == 5
    assert report["metrics"]["raw_template_region_v2_eye_min_allocated_texels"] == (
        report["metrics"]["raw_face_uv_density_eye_min_allocated_texels"]
    )
    assert report["metrics"]["raw_template_region_v2_nose_allocated_texels"] == (
        report["metrics"]["raw_face_uv_density_nose_allocated_texels"]
    )
    assert report["metrics"]["raw_template_region_v2_muzzle_allocated_texels"] > 0
    assert report["metrics"]["raw_template_region_v2_ear_inner_min_allocated_texels"] == (
        report["metrics"]["raw_face_uv_density_ear_inner_min_allocated_texels"]
    )
    assert report["metrics"]["raw_template_region_v2_facial_density_weighted_texel_count"] > (
        report["metrics"][
            "raw_template_region_v2_facial_uv_island_allocated_texel_count"
        ]
    )
    assert report["metrics"]["raw_face_texture_target_size"] >= RAW_METRIC_THRESHOLDS[
        "min_face_texture_target_size"
    ]
    assert report["metrics"]["raw_face_texture_target_scale"] >= RAW_METRIC_THRESHOLDS[
        "min_face_texture_target_scale"
    ]
    assert report["metrics"]["raw_face_texture_target_allocated_texel_count"] >= (
        report["metrics"]["raw_face_texture_target_requested_texel_count"]
    )
    assert report["metrics"]["raw_face_texture_target_eye_min_texels"] >= (
        report["metrics"]["raw_face_uv_density_eye_min_allocated_texels"]
    )
    assert report["metrics"]["raw_face_texture_target_nose_texels"] >= (
        report["metrics"]["raw_face_uv_density_nose_allocated_texels"]
    )
    assert report["metrics"]["raw_face_texture_target_ear_inner_min_texels"] >= (
        report["metrics"]["raw_face_uv_density_ear_inner_min_allocated_texels"]
    )
    assert report["metrics"]["raw_template_region_v2_texture_target_size"] >= (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_texture_target_size"]
    )
    assert report["metrics"]["raw_template_region_v2_texture_target_scale"] >= (
        RAW_METRIC_THRESHOLDS["min_template_region_v2_texture_target_scale"]
    )
    assert report["metrics"][
        "raw_template_region_v2_texture_target_facial_island_count"
    ] == RAW_METRIC_THRESHOLDS[
        "min_template_region_v2_texture_target_facial_island_count"
    ]
    assert report["metrics"]["raw_template_region_v2_texture_target_island_count"] == (
        report["metrics"]["raw_template_region_v2_facial_uv_island_count"]
    )
    assert report["metrics"][
        "raw_template_region_v2_texture_target_allocated_texel_count"
    ] >= report["metrics"][
        "raw_template_region_v2_texture_target_requested_texel_count"
    ]
    assert report["metrics"][
        "raw_template_region_v2_texture_target_allocated_texel_count"
    ] >= report["metrics"][
        "raw_template_region_v2_facial_uv_island_allocated_texel_count"
    ]
    assert report["metrics"]["raw_template_region_v2_texture_target_eye_min_texels"] >= (
        report["metrics"]["raw_template_region_v2_eye_min_allocated_texels"]
    )
    assert report["metrics"]["raw_template_region_v2_texture_target_nose_texels"] >= (
        report["metrics"]["raw_template_region_v2_nose_allocated_texels"]
    )
    assert report["metrics"]["raw_template_region_v2_texture_target_muzzle_texels"] >= (
        report["metrics"]["raw_template_region_v2_muzzle_allocated_texels"]
    )
    assert report["metrics"][
        "raw_template_region_v2_texture_target_ear_inner_min_texels"
    ] >= report["metrics"]["raw_template_region_v2_ear_inner_min_allocated_texels"]
    assert report["metrics"][
        "raw_template_region_v2_texture_target_max_density_scale"
    ] >= 1.0
    assert report["metrics"]["raw_face_texture_crop_region_count"] == 6
    assert report["metrics"]["raw_face_texture_crop_texture_width"] == report["metrics"][
        "raw_texture_width"
    ]
    assert report["metrics"]["raw_face_texture_crop_texture_height"] == report["metrics"][
        "raw_texture_height"
    ]
    assert report["metrics"]["raw_face_texture_crop_compared_region_count"] == 6
    assert report["metrics"]["raw_face_texture_crop_exact_source_patch_region_count"] == 6
    assert report["metrics"]["raw_face_texture_crop_exact_source_patch_fraction"] == 1.0
    assert report["metrics"]["raw_face_texture_crop_detail_region_count"] == 6
    assert report["metrics"]["raw_face_texture_crop_min_unique_rgb_count"] >= 2
    assert report["metrics"]["raw_face_texture_crop_min_confident_texel_fraction"] > 0.0
    assert report["metrics"]["raw_face_texture_crop_mean_confident_texel_fraction"] > 0.0
    assert report["metrics"]["raw_face_texture_crop_mean_source_rgb_delta"] >= 0.0
    assert report["metrics"]["raw_face_texture_crop_isolated_region_count"] >= (
        RAW_METRIC_THRESHOLDS["min_face_texture_isolated_crop_regions"]
    )
    assert report["metrics"]["raw_face_texture_crop_overlap_pair_count"] <= (
        RAW_METRIC_THRESHOLDS["max_face_texture_crop_overlap_pair_count"]
    )
    assert report["metrics"]["raw_face_texture_crop_overlap_fraction"] == 0.0
    assert report["metrics"]["raw_face_texture_crop_visual_quality_region_count"] == 6
    assert report["metrics"]["raw_face_texture_crop_visual_quality_passed_count"] == 6
    assert report["metrics"]["raw_face_texture_crop_visual_quality_fraction"] == 1.0
    assert report["metrics"]["raw_face_texture_crop_visual_quality_min_score"] >= (
        RAW_METRIC_THRESHOLDS["min_face_texture_visual_quality_score"]
    )
    assert report["metrics"]["raw_face_texture_crop_visual_quality_mean_score"] >= report[
        "metrics"
    ]["raw_face_texture_crop_visual_quality_min_score"]
    assert report["metrics"]["raw_face_texture_bake_comparison_candidate_count"] >= 1
    assert report["metrics"]["raw_face_texture_bake_comparison_region_count"] >= 6
    assert 0.0 <= report["metrics"][
        "raw_face_texture_bake_comparison_selected_feedback"
    ] <= 1.0
    assert report["metrics"][
        "raw_face_texture_bake_comparison_score_improved_region_count"
    ] >= 0
    assert report["metrics"][
        "raw_face_texture_bake_comparison_coverage_improved_region_count"
    ] >= 0
    assert report["metrics"][
        "raw_face_texture_bake_comparison_source_detail_improved_region_count"
    ] >= 0
    assert report["metrics"]["raw_face_texture_bake_comparison_max_score_delta"] >= 0.0
    assert (
        report["metrics"][
            "raw_face_texture_bake_comparison_selected_not_lower_rank"
        ]
        == 1.0
    )
    assert report["metrics"]["raw_facial_feature_observed_texel_count"] == report[
        "metrics"
    ]["raw_facial_feature_painted_texel_count"]
    assert report["metrics"]["raw_facial_feature_observed_texel_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_observed_texel_fraction"]
    )
    assert report["metrics"]["raw_facial_feature_source_patch_count"] >= report[
        "metrics"
    ]["raw_facial_feature_count"]
    assert report["metrics"]["raw_facial_feature_source_patch_texel_count"] == report[
        "metrics"
    ]["raw_facial_feature_painted_texel_count"]
    assert report["metrics"]["raw_facial_feature_source_patch_nominal_texel_count"] >= (
        report["metrics"]["raw_facial_feature_source_patch_texel_count"]
    )
    assert (
        0.0
        <= report["metrics"][
            "raw_facial_feature_source_patch_footprint_reduction_fraction"
        ]
        <= 1.0
    )
    assert report["metrics"][
        "raw_facial_feature_source_patch_mean_footprint_area_scale"
    ] <= RAW_METRIC_THRESHOLDS[
        "max_facial_feature_source_patch_mean_footprint_area_scale"
    ]
    assert report["metrics"][
        "raw_facial_feature_source_patch_min_footprint_area_scale"
    ] > 0.0
    assert report["metrics"][
        "raw_facial_feature_source_patch_shape_mask_texel_count"
    ] > 0
    assert report["metrics"][
        "raw_facial_feature_source_patch_shape_mask_fraction"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_shape_mask_fraction"
    ]
    assert report["metrics"][
        "raw_facial_feature_source_patch_shape_mask_fraction"
    ] <= 1.0
    assert report["metrics"][
        "raw_facial_feature_source_patch_shape_alpha_mean"
    ] > 0.0
    assert report["metrics"][
        "raw_facial_feature_source_patch_source_mask_densified_texel_count"
    ] >= report["metrics"][
        "raw_facial_feature_source_patch_source_mask_observed_texel_count"
    ]
    assert report["metrics"][
        "raw_facial_feature_source_patch_source_mask_densification_gain"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_source_mask_densification_gain"
    ]
    assert report["metrics"][
        "raw_facial_feature_source_patch_mean_source_mask_densification_gain"
    ] > 1.0
    assert report["metrics"][
        "raw_facial_feature_source_patch_localization_candidate_texel_count"
    ] >= report["metrics"][
        "raw_facial_feature_source_patch_localization_selected_texel_count"
    ] > 0
    assert (
        0.0
        < report["metrics"][
            "raw_facial_feature_source_patch_localization_fraction"
        ]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_localization_fraction"
        ]
    )
    assert (
        0.0
        < report["metrics"][
            "raw_facial_feature_source_patch_mean_localization_fraction"
        ]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_localization_fraction"
        ]
    )
    assert report["metrics"][
        "raw_facial_feature_source_patch_landmark_aligned_count"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_landmark_aligned_count"
    ]
    assert (
        0.0
        < report["metrics"][
            "raw_facial_feature_source_patch_mean_landmark_confidence"
        ]
        <= 1.0
    )
    assert report["metrics"][
        "raw_facial_feature_source_patch_landmark_destination_scaled_count"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_source_patch_landmark_destination_scaled_count"
    ]
    assert (
        0.0
        < report["metrics"][
            "raw_facial_feature_source_patch_mean_landmark_destination_scale"
        ]
        <= RAW_METRIC_THRESHOLDS[
            "max_facial_feature_source_patch_mean_landmark_destination_scale"
        ]
    )
    assert report["metrics"][
        "raw_face_feature_destination_anchor_anatomical_count"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_facial_feature_destination_anatomical_anchor_count"
    ]
    assert report["metrics"]["raw_face_feature_destination_anchor_order_valid"] == 1.0
    assert report["metrics"][
        "raw_face_feature_destination_anchor_eye_to_nose_v_gap"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_face_feature_destination_anchor_eye_to_nose_v_gap"
    ]
    assert report["metrics"][
        "raw_face_feature_destination_anchor_nose_to_mouth_v_gap"
    ] >= RAW_METRIC_THRESHOLDS[
        "min_face_feature_destination_anchor_nose_to_mouth_v_gap"
    ]
    assert report["metrics"][
        "raw_face_feature_destination_anchor_eye_symmetry_u_error"
    ] <= RAW_METRIC_THRESHOLDS[
        "max_face_feature_destination_anchor_eye_symmetry_u_error"
    ]
    assert report["metrics"]["raw_facial_feature_source_patch_texel_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_fraction"]
    )
    assert report["metrics"]["raw_facial_feature_source_patch_quality_passed_count"] == (
        report["metrics"]["raw_facial_feature_source_patch_quality_total_count"]
    )
    assert report["metrics"]["raw_facial_feature_source_patch_quality_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_quality_fraction"]
    )
    assert report["metrics"]["raw_facial_feature_source_patch_min_quality_score"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_quality_score"]
    )
    assert report["metrics"]["raw_facial_feature_source_patch_mean_quality_score"] >= (
        report["metrics"]["raw_facial_feature_source_patch_min_quality_score"]
    )
    assert report["metrics"]["raw_facial_feature_source_patch_fusion_min_weight"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_fusion_weight"]
    )
    assert report["metrics"]["raw_facial_feature_source_patch_fusion_mean_weight"] >= (
        report["metrics"]["raw_facial_feature_source_patch_fusion_min_weight"]
    )
    assert report[
        "metrics"
    ]["raw_facial_feature_source_patch_fusion_weighted_texel_count"] > 0
    assert report["metrics"]["raw_facial_feature_source_patch_core_texel_count"] > 0
    assert report["metrics"]["raw_facial_feature_source_patch_core_fraction"] >= (
        RAW_METRIC_THRESHOLDS["min_facial_feature_source_patch_core_fraction"]
    )
    assert report[
        "metrics"
    ]["raw_facial_feature_source_patch_feathered_edge_texel_count"] > 0
    assert report["metrics"]["raw_facial_feature_source_patch_feathered_edge_fraction"] >= (
        RAW_METRIC_THRESHOLDS[
            "min_facial_feature_source_patch_feathered_edge_fraction"
        ]
    )
    assert report["metrics"]["raw_facial_feature_source_patch_mean_blend_alpha"] > 0.0
    assert (
        report["metrics"][
            "raw_facial_feature_eye_source_detail_reinforcement_feature_count"
        ]
        > 0
    )
    assert (
        report["metrics"][
            "raw_facial_feature_eye_source_detail_reinforcement_texel_count"
        ]
        > 0
    )
    assert (
        0.0
        < report["metrics"][
            "raw_facial_feature_eye_source_detail_reinforcement_texel_fraction"
        ]
        <= 1.0
    )
    assert report["metrics"]["raw_face_feature_min_required_contrast"] >= (
        RAW_METRIC_THRESHOLDS["min_face_feature_required_contrast"]
    )
    assert report["metrics"]["raw_face_feature_eye_to_muzzle_luma_delta"] > 0.0
    assert report["metrics"]["raw_face_feature_mouth_to_muzzle_luma_delta"] > 0.0
    assert report["metrics"]["raw_face_feature_nose_to_muzzle_rgb_distance"] > 0.0
    assert report["metrics"]["raw_face_feature_iris_to_pupil_rgb_distance"] > 0.0
    assert report["metrics"]["raw_face_feature_contrast_passed"] == 1.0
    assert report["metrics"]["raw_face_detail_geometry_applied"] == 1.0
    assert report["metrics"]["raw_face_detail_deformed_vertex_count"] > 0
    assert report["metrics"]["raw_face_detail_max_displacement"] <= 0.040
    assert report["metrics"]["raw_face_detail_mean_displacement"] > 0.0
    assert report["metrics"]["raw_face_surface_anchor_count"] == 10
    assert report["metrics"]["raw_face_surface_anchor_vertex_count"] > 0
    assert report["metrics"]["raw_face_surface_anchor_max_displacement"] > 0.0
    assert report["metrics"]["raw_face_turntable_view_count"] == 3
    assert report["metrics"]["raw_face_turntable_front_visible_anchor_count"] >= 4
    assert report["metrics"]["raw_face_turntable_front_anchor_pixel_count"] > 0
    assert report["metrics"]["raw_face_turntable_required_anchor_visible_fraction"] == 1.0
    assert report["metrics"]["raw_face_turntable_textured_view_count"] == 3
    assert report["metrics"]["raw_face_turntable_textured_written_count"] == 3
    assert report["metrics"]["raw_face_turntable_front_textured_unique_rgb_count"] >= 2
    assert report["metrics"]["raw_face_turntable_front_textured_luma_std"] > 0.0
    assert (
        report["metrics"]["raw_face_turntable_front_textured_mouth_line_pixel_count"]
        > 0
    )
    assert (
        report["metrics"][
            "raw_face_turntable_front_textured_required_feature_fraction"
        ]
        > 0.0
    )
    assert report["metrics"]["raw_face_turntable_mean_textured_unique_rgb_count"] >= 2
    assert report["metrics"]["raw_face_turntable_mean_textured_luma_std"] > 0.0
    assert report["metrics"]["raw_face_texture_closeup_width"] == 512
    assert report["metrics"]["raw_face_texture_closeup_height"] == 512
    assert report["metrics"]["raw_face_texture_closeup_feature_count"] >= 10
    assert (
        report["metrics"]["raw_face_texture_closeup_required_present_count"]
        == report["metrics"]["raw_face_texture_closeup_required_feature_count"]
    )
    assert report["metrics"]["raw_face_texture_closeup_required_feature_fraction"] == 1.0
    assert report["metrics"]["raw_face_texture_closeup_crop_texel_count"] > 0
    assert report["metrics"]["raw_face_texture_closeup_unique_rgb_count"] >= 4
    assert report["metrics"]["raw_face_texture_closeup_luma_std"] > 0.0
    assert (
        report["metrics"][
            "raw_face_texture_source_compare_source_feature_available_count"
        ]
        >= 10
    )
    assert report["metrics"]["raw_face_texture_source_compare_feature_count"] >= 10
    assert (
        report["metrics"]["raw_face_texture_source_compare_required_present_count"]
        == report["metrics"]["raw_face_texture_source_compare_required_feature_count"]
    )
    assert (
        report["metrics"]["raw_face_texture_source_compare_required_feature_fraction"]
        == 1.0
    )
    assert report["metrics"]["raw_face_texture_source_compare_mean_rgb_distance"] >= 0.0
    assert report["metrics"]["raw_face_texture_source_compare_max_rgb_distance"] >= (
        report["metrics"]["raw_face_texture_source_compare_mean_rgb_distance"]
    )
    assert report["metrics"]["raw_face_texture_source_compare_mean_luma_delta"] >= 0.0
    assert report["metrics"]["raw_face_texture_source_compare_max_luma_delta"] >= (
        report["metrics"]["raw_face_texture_source_compare_mean_luma_delta"]
    )
    assert report["metrics"]["raw_face_texture_source_compare_source_mean_luma_std"] >= 0.0
    assert report["metrics"]["raw_face_texture_source_compare_baked_mean_luma_std"] >= 0.0
    iris_candidate_count = report["metrics"][
        "raw_face_texture_source_compare_iris_candidate_texel_count"
    ]
    iris_accepted_count = report["metrics"][
        "raw_face_texture_source_compare_iris_core_accepted_texel_count"
    ]
    iris_rejected_count = report["metrics"][
        "raw_face_texture_source_compare_iris_annulus_rejected_texel_count"
    ]
    assert report["metrics"]["raw_face_texture_source_compare_iris_feature_count"] == 2
    assert iris_candidate_count > 0
    assert iris_accepted_count > 0
    assert iris_candidate_count == iris_accepted_count + iris_rejected_count
    assert (
        report["metrics"][
            "raw_face_texture_source_compare_iris_core_densified_texel_count"
        ]
        >= iris_accepted_count
    )
    assert (
        0.0
        <= report["metrics"][
            "raw_face_texture_source_compare_iris_core_acceptance_fraction"
        ]
        <= 1.0
    )
    assert (
        0.0
        <= report["metrics"][
            "raw_face_texture_source_compare_iris_annulus_rejected_fraction"
        ]
        <= 1.0
    )
    iris_halo_support = report["metrics"][
        "raw_facial_feature_iris_halo_uv_support_texel_count"
    ]
    iris_halo_core = report["metrics"][
        "raw_facial_feature_iris_halo_core_texel_count"
    ]
    iris_halo_edge = report["metrics"][
        "raw_facial_feature_iris_halo_edge_texel_count"
    ]
    iris_halo_accepted_core = report["metrics"][
        "raw_facial_feature_iris_halo_accepted_core_texel_count"
    ]
    assert report["metrics"]["raw_facial_feature_iris_halo_feature_count"] == 2
    assert iris_halo_support > 0
    assert iris_halo_core > 0
    assert iris_halo_edge >= 0
    assert iris_halo_accepted_core > 0
    assert iris_halo_edge <= iris_halo_support
    assert iris_halo_core <= iris_halo_support
    assert (
        abs(
            report["metrics"]["raw_facial_feature_iris_halo_edge_fraction"]
            - iris_halo_edge / max(iris_halo_support, 1)
        )
        < 1e-12
    )
    assert (
        abs(
            report["metrics"]["raw_facial_feature_iris_halo_edge_to_core_ratio"]
            - iris_halo_edge / max(iris_halo_core, 1)
        )
        < 1e-12
    )
    assert report["metrics"][
        "raw_facial_feature_iris_halo_max_edge_to_core_ratio"
    ] >= report["metrics"]["raw_facial_feature_iris_halo_edge_to_core_ratio"]
    assert (
        abs(
            report["metrics"][
                "raw_facial_feature_iris_halo_support_to_accepted_core_ratio"
            ]
            - iris_halo_support / max(iris_halo_accepted_core, 1)
        )
        < 1e-12
    )
    assert report["metrics"][
        "raw_facial_feature_left_iris_halo_edge_to_core_ratio"
    ] >= 0.0
    assert report["metrics"][
        "raw_facial_feature_right_iris_halo_edge_to_core_ratio"
    ] >= 0.0
    assert report["metrics"]["overlay_mean_silhouette_iou"] >= RAW_METRIC_THRESHOLDS[
        "min_overlay_mean_silhouette_iou"
    ]
    assert report["metrics"]["observed_face_fraction"] > 0.0
    assert report["metrics"]["observed_vertex_fraction"] > 0.0
    assert report["metrics"]["generated_face_count"] == 0
    assert report["metrics"]["basecolor_observed_texel_fraction"] > 0.0
    assert report["metrics"]["basecolor_generated_texel_fraction"] == 0.0
    assert report["metrics"]["pbr_observed_texel_fraction"] > 0.0
    assert report["metrics"]["pbr_generated_texel_fraction"] == 0.0
    assert report["metrics"]["raw_pbr_face_material_detail_applied"] == 1.0
    assert report["metrics"]["raw_pbr_face_material_detail_feature_count"] >= 12
    assert report["metrics"]["raw_pbr_face_material_detail_texel_count"] >= (
        RAW_METRIC_THRESHOLDS["min_pbr_face_material_detail_texel_count"]
    )
    assert report["metrics"]["raw_pbr_face_material_detail_normal_variance"] > 0.0
    assert report["metrics"]["joint_projection_mean_keypoint_loss"] <= 0.35
    assert report["metrics"]["joint_projection_mean_pixel_error"] >= 0.0
    assert report["metrics"]["silhouette_residual_mean_union_error"] <= RAW_METRIC_THRESHOLDS[
        "max_silhouette_residual_mean_union_error"
    ]
    assert report["metrics"]["silhouette_residual_mean_false_negative"] >= 0.0
    assert report["metrics"]["silhouette_residual_mean_false_positive"] >= 0.0
    assert all(report["glb_structure_checks"].values())
    assert report["glb_structure_checks"]["expected_textured"]
    assert report["glb_structure_checks"]["expected_pbr_textures"]
    assert report["glb_structure_checks"]["expected_skinned"]
    assert report["glb_structure_checks"]["expected_animated"]
    assert report["glb_structure_checks"]["texture_material_valid"]
    assert report["glb_structure_checks"]["texture_dimensions_valid"]
    assert report["glb_structure_checks"]["pbr_material_textures_valid"]
    assert report["glb_structure_checks"]["pbr_texture_dimensions_valid"]
    assert report["glb_structure_checks"]["pbr_metallic_factor_zero"]
    assert report["glb_structure_checks"]["skin_valid"]
    assert report["glb_structure_checks"]["animation_valid"]
    assert all(report["skeleton_checks"].values())
    assert all(report["joint_projection_checks"].values())
    assert all(report["silhouette_residual_checks"].values())
    assert report["artifacts"]["glb"].endswith("cat_avatar.glb")
    assert report["artifacts"]["fit_state"].endswith("outputs/fitting/fit_state.safetensors")
    assert report["artifacts"]["fit_state_source_format"] == "safetensors"
    assert report["artifacts"]["fit_state_safetensors"].endswith(
        "outputs/fitting/fit_state.safetensors"
    )
    assert report["artifacts"]["observation_manifest"].endswith(
        "observations/raw/manifest.json"
    )
    assert report["artifacts"]["idle_animation"].endswith("outputs/avatar/animations/idle.glb")
    assert report["artifacts"]["skeleton"].endswith("outputs/avatar/skeleton.json")
    assert report["artifacts"]["overlay_manifest"].endswith(
        "outputs/fitting/per_frame_overlay/manifest.json"
    )
    assert report["artifacts"]["coverage_report"].endswith("outputs/quality/coverage.json")
    assert report["artifacts"]["coverage_provenance"].endswith(
        "outputs/quality/coverage_provenance.npz"
    )
    assert report["artifacts"]["texture_manifest"].endswith(
        "outputs/textures/texture_manifest.json"
    )
    assert report["artifacts"]["basecolor"].endswith("outputs/textures/basecolor.png")
    assert report["artifacts"]["basecolor_confidence"].endswith(
        "outputs/textures/basecolor_confidence.png"
    )
    assert report["artifacts"]["source_view_id"].endswith(
        "outputs/textures/source_view_id.png"
    )
    assert report["artifacts"]["normal"].endswith("outputs/textures/normal.png")
    assert report["artifacts"]["roughness"].endswith("outputs/textures/roughness.png")
    assert report["artifacts"]["ao"].endswith("outputs/textures/ao.png")
    assert report["artifacts"]["joint_projection"].endswith(
        "outputs/quality/raw_joint_projection.json"
    )
    assert report["artifacts"]["silhouette_residuals"].endswith(
        "outputs/quality/raw_silhouette_residuals.json"
    )
    assert report["artifacts"]["silhouette_residual_manifest"].endswith(
        "outputs/quality/raw_silhouette_residuals/manifest.json"
    )
    assert report["artifacts"]["validation_report"].endswith("outputs/quality/validation.json")
    assert report["artifacts"]["face_turntable"].endswith(
        "outputs/quality/raw_face_turntable.json"
    )
    assert report["artifacts"]["face_turntable_manifest"].endswith(
        "outputs/quality/raw_face_turntable/manifest.json"
    )
    assert report["artifacts"]["face_turntable_front"].endswith(
        "outputs/quality/raw_face_turntable/front.png"
    )
    assert report["artifacts"]["face_turntable_front_textured"].endswith(
        "outputs/quality/raw_face_turntable/front_textured.png"
    )
    assert report["artifacts"]["face_texture_closeup"].endswith(
        "outputs/quality/raw_face_turntable/face_closeup.png"
    )
    assert report["artifacts"]["face_texture_source_comparison"].endswith(
        "outputs/quality/raw_face_turntable/face_source_comparison.png"
    )
    assert report["artifacts"]["face_uv_density"].endswith(
        "outputs/quality/raw_face_uv_density.json"
    )
    assert report["artifacts"]["face_uv_density_overlay"].endswith(
        "outputs/quality/raw_face_uv_density.png"
    )
    assert report["artifacts"]["face_texture_targets"].endswith(
        "outputs/quality/raw_face_texture_targets.json"
    )
    assert report["artifacts"]["face_texture_diagnostics"].endswith(
        "outputs/quality/raw_face_texture_diagnostics.json"
    )
    assert report["artifacts"]["face_texture_bake_comparison"].endswith(
        "outputs/quality/raw_face_texture_bake_comparison.json"
    )
    assert report["artifacts"]["glb_face_feature_visibility"].endswith(
        "outputs/quality/raw_glb_face_feature_visibility.json"
    )
    assert report["artifacts"]["face_texture_crops"].endswith(
        "outputs/quality/raw_face_texture_crops.png"
    )
    assert report["artifacts"]["face_texture_crop_dir"].endswith(
        "outputs/quality/raw_face_texture_crops"
    )
    assert all(report["face_uv_density_checks"].values())
    assert all(report["face_turntable_checks"].values())
    assert all(report["face_texture_targets_checks"].values())
    assert all(report["face_texture_diagnostic_checks"].values())
    assert all(report["face_texture_bake_comparison_checks"].values())
    assert all(report["glb_face_feature_visibility_checks"].values())
    validation = json.loads(
        Path(report["artifacts"]["validation_report"]).read_text(encoding="utf-8")
    )
    assert validation["success"] is True
    assert validation["checks"]["glb_exists"]
    assert validation["checks"]["glb_structure_valid"]
    assert validation["checks"]["fit_state_safetensors_valid"]
    assert validation["checks"]["texture_artifact_hashes_valid"]
    assert validation["fit_state_checks"]["translation_shapes_valid"]
    assert validation["texture_manifest_checks"]["basecolor_source_observed"]
    assert "fit_state_safetensors" in validation["sha256"]
    assert "texture_manifest" in validation["sha256"]


def test_source_patch_priority_diagnostic_metrics_count_direct_photo_rejections() -> None:
    metrics = _source_patch_priority_diagnostic_metrics(
        {
            "source_patch_candidate_diagnostics": {
                "ear_inner": [
                    {
                        "sample_key": "ear_inner",
                        "source_patch_direct_photo_candidate": True,
                        "source_patch_frontal_photo_priority": 0.0,
                        "source_patch_frontal_photo_priority_rejection_reason": (
                            "quality_below_min"
                        ),
                        "source_patch_frontal_photo_priority_quality_min": 0.45,
                        "source_patch_quality_score": 0.36,
                        "source_patch_frontal_photo_priority_detail_min": 0.70,
                        "source_patch_detail_score": 0.90,
                    },
                    {
                        "sample_key": "ear_inner",
                        "source_patch_direct_photo_candidate": True,
                        "source_patch_frontal_photo_priority": 0.0,
                        "source_patch_frontal_photo_priority_rejection_reason": (
                            "detail_below_min"
                        ),
                        "source_patch_frontal_photo_priority_quality_min": 0.45,
                        "source_patch_quality_score": 0.50,
                        "source_patch_frontal_photo_priority_detail_min": 0.70,
                        "source_patch_detail_score": 0.40,
                    },
                ],
                "muzzle": [
                    {
                        "sample_key": "muzzle",
                        "source_patch_direct_photo_candidate": False,
                        "source_patch_frontal_photo_priority": 0.0,
                    },
                    {
                        "sample_key": "muzzle",
                        "source_patch_direct_photo_candidate": True,
                        "source_patch_frontal_photo_priority": 0.2,
                    },
                ],
            }
        }
    )

    assert (
        metrics["facial_feature_source_patch_direct_photo_candidate_diagnostic_count"]
        == 3.0
    )
    assert (
        metrics["facial_feature_source_patch_direct_photo_zero_priority_candidate_count"]
        == 2.0
    )
    assert (
        metrics["facial_feature_source_patch_direct_photo_quality_below_min_count"]
        == 1.0
    )
    assert (
        metrics["facial_feature_source_patch_direct_photo_detail_below_min_count"]
        == 1.0
    )
    assert (
        metrics[
            "facial_feature_source_patch_ear_inner_direct_photo_quality_below_min_count"
        ]
        == 1.0
    )
    assert metrics["facial_feature_source_patch_direct_photo_max_quality_gap"] == 0.09
    assert metrics["facial_feature_source_patch_direct_photo_max_detail_gap"] == 0.30


def test_evaluate_raw_reconstruction_fails_when_export_mode_is_not_raw(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_exported_project(project_dir)
    validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["mode"] = "synthetic"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = evaluate_raw_reconstruction(project_dir, force=True)

    assert not result.success
    assert not result.checks["raw_export_mode"]
    assert result.checks["glb_metadata_valid"]


def test_evaluate_raw_reconstruction_fails_when_component_loss_does_not_reduce(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_exported_project(project_dir)
    fit_report_path = project_dir / "outputs" / "fitting" / "fit_report.json"
    fit_report = json.loads(fit_report_path.read_text(encoding="utf-8"))
    fit_report["metrics"]["final_keypoint_loss"] = fit_report["metrics"][
        "initial_keypoint_loss"
    ]
    fit_report["metrics"]["final_silhouette_loss"] = fit_report["metrics"][
        "initial_silhouette_loss"
    ]
    fit_report_path.write_text(
        json.dumps(fit_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = evaluate_raw_reconstruction(project_dir, force=True)

    assert not result.success
    assert not result.checks["keypoint_loss_reduced"]
    assert not result.checks["silhouette_loss_reduced"]


def test_evaluate_raw_reconstruction_fails_when_fit_overlays_are_missing(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_exported_project(project_dir)
    overlay_manifest = project_dir / "outputs" / "fitting" / "per_frame_overlay" / "manifest.json"
    overlay_manifest.unlink()

    result = evaluate_raw_reconstruction(project_dir, force=True)

    assert not result.success
    assert not result.checks["fit_overlays_present"]


def test_evaluate_raw_reconstruction_fails_when_template_region_v2_glb_contract_is_invalid(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_exported_project(project_dir)
    validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["glb_template_regions_v2_valid"] = False
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = evaluate_raw_reconstruction(project_dir, force=True)

    assert not result.success
    assert not result.checks["raw_template_region_v2_contract_valid"]
    assert result.metrics["raw_template_region_v2_contract_valid"] == 1.0
    assert result.metrics["raw_glb_template_region_v2_contract_valid"] == 0.0


def test_evaluate_raw_reconstruction_fails_when_raw_masks_are_full_frame(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_exported_project(project_dir)
    observation_manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    manifest = json.loads(observation_manifest_path.read_text(encoding="utf-8"))
    for frame in manifest["frames"]:
        image_width = float(frame["image_width"])
        image_height = float(frame["image_height"])
        frame["bbox_xyxy"] = [0.0, 0.0, image_width, image_height]
        frame["quality"]["mask_area_ratio"] = 0.95
        frame["quality"]["bbox_area_ratio"] = 1.0
        frame["quality"]["touches_image_border"] = 1.0
        frame["quality"]["full_frame_bbox"] = 1.0
    manifest["mean_mask_area_ratio"] = 0.95
    manifest["mean_bbox_area_ratio"] = 1.0
    manifest["full_frame_bbox_fraction"] = 1.0
    observation_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = evaluate_raw_reconstruction(project_dir, force=True)

    assert not result.success
    assert not result.checks["raw_mask_area_reasonable"]
    assert not result.checks["raw_bbox_area_reasonable"]
    assert not result.checks["raw_full_frame_bbox_fraction_reasonable"]


def test_dense_evidence_metrics_fail_when_tensor_hash_changes(tmp_path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    observation_manifest = project_dir / "observations" / "raw" / "manifest.json"
    observation_manifest.parent.mkdir(parents=True)
    observation_manifest.write_text('{"schema_version":"1.0","frames":[]}\n', encoding="utf-8")
    depth_path = project_dir / "depth.npy"
    np.save(depth_path, np.ones((4, 3), dtype=np.float32))
    provider = LocalFileDenseEvidenceProvider(
        payloads=(
            DenseEvidencePayload(
                frame_id="frame_0001",
                kind=DenseEvidenceKind.DEPTH,
                path=depth_path,
                provenance=EvidenceProvenance.OBSERVED,
            ),
        )
    )
    provider.estimate(
        DenseEvidenceRequest(
            project_id="demo_cat",
            project_dir=project_dir,
            observation_manifest_path=observation_manifest,
            output_dir=project_dir / "outputs" / "dense_evidence",
            frame_ids=("frame_0001",),
            evidence_kinds=(DenseEvidenceKind.DEPTH,),
        )
    )
    np.save(
        project_dir / "outputs" / "dense_evidence" / "payloads" / "frame_0001" / "depth.npy",
        np.zeros((4, 3), dtype=np.float32),
    )

    metrics = _dense_evidence_metrics(project_dir, expected_frame_count=1)

    assert metrics["raw_dense_evidence_manifest_present"] == 1.0
    assert metrics["raw_dense_evidence_manifest_valid"] == 0.0
    assert metrics["raw_dense_evidence_tensor_refs_valid"] == 0.0
    assert metrics["raw_dense_evidence_tensor_files_exist_fraction"] == 1.0
    assert metrics["raw_dense_evidence_tensor_sha_valid_fraction"] == 0.0
    assert metrics["raw_dense_evidence_error_count"] == 1


def test_evaluate_cli_runs_raw_chain(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "project"
    _create_raw_initialized_project(project_dir)

    fit_result = runner.invoke(
        app,
        [
            "fit",
            str(project_dir),
            "--mode",
            "raw",
            "--camera-root-steps",
            "2",
        ],
    )
    export_result = runner.invoke(app, ["export", str(project_dir), "--mode", "raw"])
    evaluate_result = runner.invoke(app, ["evaluate", str(project_dir), "--mode", "raw"])

    assert fit_result.exit_code == 0, fit_result.output
    assert export_result.exit_code == 0, export_result.output
    assert evaluate_result.exit_code == 0, evaluate_result.output
    assert "success=True" in evaluate_result.output
    assert (project_dir / "outputs" / "quality" / "raw_evaluation.json").exists()


def _create_raw_fitted_exported_project(project_dir) -> None:
    _create_raw_initialized_project(project_dir)
    run_raw_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=2, shape_steps=0, silhouette_weight=1.0),
        initial_root_offset=np.asarray([0.0, 0.0, 0.0]),
    )
    export_fitted_canonical_mesh(project_dir, mode="raw")


def _create_raw_initialized_project(project_dir) -> None:
    input_dir = project_dir.parent / "inputs"
    input_dir.mkdir()
    _write_rgb_png(input_dir / "b.png", width=8, height=8, bbox=(2, 2, 6, 6))
    _write_rgb_png(input_dir / "a.png", width=8, height=8, bbox=(1, 1, 5, 5))
    ingest_image_sources(project_dir, [input_dir], min_frames=2)
    segment_raw_image_observations(project_dir, color_delta_threshold=0.10)
    bootstrap_raw_keypoints_and_pose(project_dir)


def _write_dense_evidence_fixture(project_dir) -> None:
    observation_manifest = project_dir / "observations" / "raw" / "manifest.json"
    manifest = json.loads(observation_manifest.read_text(encoding="utf-8"))
    frame_id = manifest["frames"][0]["frame_id"]
    depth_path = project_dir / "fixture_depth.npy"
    tracks_path = project_dir / "fixture_tracks.npy"
    np.save(depth_path, np.ones((4, 3), dtype=np.float32))
    np.save(tracks_path, np.zeros((1, 3, 2), dtype=np.float32))
    provider = LocalFileDenseEvidenceProvider(
        payloads=(
            DenseEvidencePayload(
                frame_id=frame_id,
                kind=DenseEvidenceKind.DEPTH,
                path=depth_path,
                provenance=EvidenceProvenance.OBSERVED,
            ),
            DenseEvidencePayload(
                frame_id=frame_id,
                kind=DenseEvidenceKind.TRACKS,
                path=tracks_path,
                provenance=EvidenceProvenance.INFERRED,
            ),
        )
    )
    provider.estimate(
        DenseEvidenceRequest(
            project_id=project_dir.name,
            project_dir=project_dir,
            observation_manifest_path=observation_manifest,
            output_dir=project_dir / "outputs" / "dense_evidence",
            frame_ids=(frame_id,),
            evidence_kinds=(DenseEvidenceKind.DEPTH, DenseEvidenceKind.TRACKS),
        )
    )


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = np.asarray([222, 220, 208], dtype=np.uint8)
    if x1 - x0 >= 4 and y1 - y0 >= 4:
        image[y0, x0 + 1, :] = np.asarray([118, 104, 54], dtype=np.uint8)
        image[y0, x1 - 2, :] = np.asarray([118, 104, 54], dtype=np.uint8)
        image[y0 + 1, x0 + 1, :] = np.asarray([10, 9, 8], dtype=np.uint8)
        image[y0 + 1, x1 - 2, :] = np.asarray([10, 9, 8], dtype=np.uint8)
        image[y0 + 2, x0 + 2, :] = np.asarray([224, 136, 146], dtype=np.uint8)
        image[y0 + 3, x0 + 2, :] = np.asarray([52, 36, 38], dtype=np.uint8)
    Image.fromarray(image).save(path)
