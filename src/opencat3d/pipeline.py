"""End-to-end deterministic pipeline entry points."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opencat3d.evaluation.acceptance import evaluate_synthetic_acceptance
from opencat3d.evaluation.compliance import (
    write_objective_compliance_audit,
    write_raw_objective_compliance_audit,
)
from opencat3d.evaluation.frame_set_supervisor import (
    write_texture_only_promotion_guard,
)
from opencat3d.evaluation.raw import evaluate_raw_reconstruction
from opencat3d.evaluation.synthetic import evaluate_synthetic_reconstruction
from opencat3d.export.mesh import export_fitted_canonical_mesh
from opencat3d.fitting.optimizer import OptimizerConfig, run_raw_fit, run_synthetic_fit
from opencat3d.ingest.images import SUPPORTED_IMAGE_EXTENSIONS, ingest_image_sources
from opencat3d.ingest.video import SUPPORTED_VIDEO_EXTENSIONS, extract_video_frames
from opencat3d.io.cache import sha256_file
from opencat3d.preprocess.frame_selection import select_raw_observation_frames
from opencat3d.preprocess.keypoints import (
    bootstrap_raw_keypoints_and_pose,
    bootstrap_synthetic_keypoints_and_pose,
)
from opencat3d.preprocess.segmentation import (
    RawSegmentationBackend,
    segment_raw_image_observations,
    segment_synthetic_rgb_observations,
)
from opencat3d.synthetic.generator import create_synthetic_project

TEXTURE_ONLY_SOURCE_QUALITY_GATE_VERSION = "face_detail_positive_v2"
TEXTURE_SOURCE_ADMISSION_AUDIT_VERSION = "unique_protected_face_evidence_v1"
TEXTURE_ONLY_FACE_ALIGNMENT_GATE_VERSION = "observed_face_anchor_alignment_v1"
TEXTURE_ONLY_FACE_REGION_PROTECTION_VERSION = "supplemental_face_region_protection_v1"
TEXTURE_ONLY_FIXED_GEOMETRY_ALIGNMENT_VERSION = "fixed_geometry_alignment_recovery_v1"
TEXTURE_ONLY_PROTECTED_FACE_REGION_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "muzzle",
    "left_ear_inner",
    "right_ear_inner",
)
TEXTURE_ONLY_SOURCE_PATCH_SAMPLE_KEYS = (
    "left_eye",
    "right_eye",
    "eye",
    "iris",
    "left_iris",
    "right_iris",
    "left_pupil",
    "right_pupil",
    "nose",
    "muzzle",
    "ear_inner",
    "mouth",
)
TEXTURE_ONLY_FACE_DETAIL_SCORE_FLOOR = 0.48
TEXTURE_ONLY_SEVERE_REJECT_REASONS = frozenset(
    {
        "mask_too_small",
        "bbox_too_small",
        "full_frame_bbox",
        "border_partial_body",
        "sparse_geometry_mask",
    }
)


@dataclass(frozen=True)
class SyntheticPipelineResult:
    project_dir: Path
    success: bool
    summary_path: Path
    evaluation_report_path: Path
    acceptance_audit_path: Path
    compliance_audit_path: Path
    fit_report_path: Path
    obj_path: Path
    ply_path: Path
    glb_path: Path
    skeleton_path: Path | None = None


@dataclass(frozen=True)
class RawPipelineResult:
    project_dir: Path
    success: bool
    summary_path: Path
    evaluation_report_path: Path
    compliance_audit_path: Path
    fit_report_path: Path
    obj_path: Path
    ply_path: Path
    glb_path: Path
    skeleton_path: Path | None = None


@dataclass(frozen=True)
class RawTextureOnlyBakeResult:
    project_dir: Path
    success: bool
    summary_path: Path
    evaluation_report_path: Path
    compliance_audit_path: Path | None
    promotion_guard_path: Path
    obj_path: Path
    ply_path: Path
    glb_path: Path
    added_unique_texture_frame_count: int
    promotion_guard_passed: bool
    skeleton_path: Path | None = None


def _stage_raw_pipeline_input_media(
    project_dir: Path,
    input_paths: list[Path] | tuple[Path, ...] | None,
    *,
    min_video_frames: int = 2,
    max_video_frames: int = 24,
    force: bool = False,
) -> tuple[Path, ...] | None:
    """Stage mixed image/video paths for raw pipeline ingestion.

    `ingest_image_sources` intentionally consumes image frames only. This helper keeps
    the public raw pipeline entry point video-capable by converting videos to
    deterministic frame images and writing the same media metadata contract used by
    the web uploader.
    """

    if not input_paths:
        return None
    sources = tuple(Path(path).expanduser() for path in input_paths)
    media_sources = _raw_pipeline_media_sources(sources)
    if not any(path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS for path in media_sources):
        return sources

    stage_dir = project_dir / "raw" / "staged_inputs"
    if stage_dir.exists() and force:
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged_frame_index = 0
    media_frames: dict[str, dict[str, object]] = {}
    for media_index, source_path in enumerate(media_sources):
        suffix = source_path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            target = stage_dir / f"frame_{staged_frame_index:04d}{suffix}"
            if target.exists() and not force:
                raise FileExistsError(f"{target} already exists; pass force=True")
            shutil.copyfile(source_path, target)
            media_frames[target.name] = {
                "media_type": "photo",
                "source_type": "image",
                "source_media_id": source_path.name,
                "source_path": str(source_path),
            }
            staged_frame_index += 1
            continue

        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported raw media extension: {source_path.suffix}")
        video_dir = stage_dir / f"video_{media_index:04d}"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_target = video_dir / f"source{suffix}"
        if video_target.exists() and not force:
            raise FileExistsError(f"{video_target} already exists; pass force=True")
        shutil.copyfile(source_path, video_target)
        extraction = extract_video_frames(
            video_target,
            stage_dir / f"video_{media_index:04d}_frames",
            min_frames=min_video_frames,
            max_frames=max_video_frames,
            force=force,
        )
        for frame_path, source_frame_index in zip(
            extraction.frame_paths,
            extraction.selected_indices,
            strict=True,
        ):
            staged_target = stage_dir / f"frame_{staged_frame_index:04d}.jpg"
            if staged_target.exists() and not force:
                raise FileExistsError(f"{staged_target} already exists; pass force=True")
            shutil.copyfile(frame_path, staged_target)
            media_frames[staged_target.name] = {
                "media_type": "video",
                "source_type": "video_frame",
                "source_media_id": source_path.name,
                "source_frame_index": int(source_frame_index),
                "source_path": str(source_path),
                "source_video_path": str(video_target),
            }
            staged_frame_index += 1

    media_manifest_path = stage_dir / "media_manifest.json"
    media_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.raw_pipeline_media_staging",
                "frames": media_frames,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (stage_dir,)


def _raw_pipeline_media_sources(input_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    sources: list[Path] = []
    supported = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
    for input_path in input_paths:
        path = input_path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        if path.is_dir():
            sources.extend(
                child
                for child in sorted(path.iterdir(), key=lambda item: item.name)
                if child.is_file() and child.suffix.lower() in supported
            )
            continue
        if path.suffix.lower() not in supported:
            raise ValueError(f"unsupported raw media extension: {path.suffix}")
        sources.append(path)

    unique = {path.resolve(): path for path in sources}
    return tuple(unique[key] for key in sorted(unique))


def run_synthetic_pipeline(
    project_dir: Path,
    *,
    num_views: int = 3,
    image_size: int = 64,
    seed: int = 20260625,
    force: bool = False,
) -> SyntheticPipelineResult:
    project_dir = Path(project_dir).resolve()
    create_synthetic_project(
        project_dir,
        num_views=num_views,
        image_size=image_size,
        seed=seed,
        force=force,
    )
    segmentation_result = segment_synthetic_rgb_observations(project_dir, force=force)
    keypoint_result = bootstrap_synthetic_keypoints_and_pose(project_dir, force=force)
    fit_report = run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=30),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
        force=force,
    )
    export_result = export_fitted_canonical_mesh(project_dir, force=force)
    eval_result = evaluate_synthetic_reconstruction(project_dir, force=force)
    acceptance_result = evaluate_synthetic_acceptance(project_dir, force=force)
    if (
        export_result.obj_path is None
        or export_result.ply_path is None
        or export_result.glb_path is None
    ):
        raise ValueError("synthetic pipeline requires OBJ, PLY, and GLB outputs")

    summary_path = project_dir / "outputs" / "quality" / "synthetic_pipeline_summary.json"
    compliance_audit_path = (
        project_dir / "outputs" / "quality" / "opencat3d_compliance_audit.json"
    )
    summary = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "success": bool(eval_result.success and acceptance_result.success),
        "steps": {
            "synthetic_create": True,
            "preprocess": True,
            "fit": bool(fit_report.success),
            "export": True,
            "evaluate": bool(eval_result.success),
            "acceptance": bool(acceptance_result.success),
        },
        "metrics": {
            "segmentation_mean_iou_with_previous_masks": float(
                segmentation_result.mean_iou_with_previous_masks
            ),
            "keypoint_mean_confidence": float(keypoint_result.mean_keypoint_confidence),
            "keypoint_mean_in_bbox_fraction": float(keypoint_result.mean_in_bbox_fraction),
            "fit_init_prediction_used": float(fit_report.metrics["init_prediction_used"]),
            "overlay_mean_silhouette_iou": float(
                fit_report.metrics["overlay_mean_silhouette_iou"]
            ),
            "observed_face_fraction": float(eval_result.metrics["observed_face_fraction"]),
            "observed_vertex_fraction": float(
                eval_result.metrics["observed_vertex_fraction"]
            ),
            "generated_face_count": int(eval_result.metrics["generated_face_count"]),
            "basecolor_observed_texel_fraction": float(
                eval_result.metrics["basecolor_observed_texel_fraction"]
            ),
            "basecolor_generated_texel_fraction": float(
                eval_result.metrics["basecolor_generated_texel_fraction"]
            ),
            "pbr_observed_texel_fraction": float(
                eval_result.metrics["pbr_observed_texel_fraction"]
            ),
            "pbr_generated_texel_fraction": float(
                eval_result.metrics["pbr_generated_texel_fraction"]
            ),
            "joint_projection_mean_keypoint_loss": float(
                eval_result.metrics["joint_projection_mean_keypoint_loss"]
            ),
            "joint_projection_mean_pixel_error": float(
                eval_result.metrics["joint_projection_mean_pixel_error"]
            ),
            "silhouette_residual_mean_union_error": float(
                eval_result.metrics["silhouette_residual_mean_union_error"]
            ),
            "silhouette_residual_mean_false_negative": float(
                eval_result.metrics["silhouette_residual_mean_false_negative"]
            ),
            "silhouette_residual_mean_false_positive": float(
                eval_result.metrics["silhouette_residual_mean_false_positive"]
            ),
            "initial_loss": float(eval_result.metrics["initial_loss"]),
            "final_loss": float(eval_result.metrics["final_loss"]),
            "loss_reduction_fraction": float(eval_result.metrics["loss_reduction_fraction"]),
            "initial_keypoint_loss": float(eval_result.metrics["initial_keypoint_loss"]),
            "final_keypoint_loss": float(eval_result.metrics["final_keypoint_loss"]),
            "initial_silhouette_loss": float(eval_result.metrics["initial_silhouette_loss"]),
            "final_silhouette_loss": float(eval_result.metrics["final_silhouette_loss"]),
            "heldout_mean_silhouette_iou": float(
                eval_result.metrics["heldout_mean_silhouette_iou"]
            ),
            "heldout_mean_keypoint_reprojection_loss": float(
                eval_result.metrics["heldout_mean_keypoint_reprojection_loss"]
            ),
            "initial_beta_error": float(eval_result.metrics["initial_beta_error"]),
            "final_beta_error": float(eval_result.metrics["final_beta_error"]),
            "beta_error_reduction": float(eval_result.metrics["beta_error_reduction"]),
            "beta_error_reduction_fraction": float(
                eval_result.metrics["beta_error_reduction_fraction"]
            ),
            "vertex_count": int(eval_result.metrics["vertex_count"]),
            "face_count": int(eval_result.metrics["face_count"]),
            "acceptance_passed_check_count": int(
                acceptance_result.metrics["passed_check_count"]
            ),
            "acceptance_total_check_count": int(
                acceptance_result.metrics["total_check_count"]
            ),
        },
        "artifacts": {
            "fit_report": fit_report.output_paths["fit_report"],
            "fit_state": fit_report.output_paths["fit_state"],
            "fit_state_source_format": "safetensors",
            "fit_state_safetensors": fit_report.output_paths["fit_state_safetensors"],
            "overlay_manifest": fit_report.output_paths["overlay_manifest"],
            "init_manifest": str(keypoint_result.manifest_path),
            "evaluation_report": str(eval_result.report_path),
            "validation_report": str(
                project_dir / "outputs" / "quality" / "validation.json"
            ),
            "coverage_report": str(
                project_dir / "outputs" / "quality" / "coverage.json"
            ),
            "texture_manifest": str(
                project_dir / "outputs" / "textures" / "texture_manifest.json"
            ),
            "basecolor": str(project_dir / "outputs" / "textures" / "basecolor.png"),
            "basecolor_confidence": str(
                project_dir / "outputs" / "textures" / "basecolor_confidence.png"
            ),
            "source_view_id": str(project_dir / "outputs" / "textures" / "source_view_id.png"),
            "normal": str(project_dir / "outputs" / "textures" / "normal.png"),
            "roughness": str(project_dir / "outputs" / "textures" / "roughness.png"),
            "ao": str(project_dir / "outputs" / "textures" / "ao.png"),
            "joint_projection": str(
                project_dir / "outputs" / "quality" / "synthetic_joint_projection.json"
            ),
            "silhouette_residuals": str(
                project_dir / "outputs" / "quality" / "synthetic_silhouette_residuals.json"
            ),
            "acceptance_audit": str(acceptance_result.report_path),
            "compliance_audit": str(compliance_audit_path),
            "obj": str(export_result.obj_path),
            "ply": str(export_result.ply_path),
            "glb": str(export_result.glb_path),
            "idle_animation": str(export_result.idle_animation_path),
            "skeleton": str(export_result.skeleton_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compliance_result = write_objective_compliance_audit(project_dir, force=force)
    return SyntheticPipelineResult(
        project_dir=project_dir,
        success=bool(
            eval_result.success
            and acceptance_result.success
            and compliance_result.success
        ),
        summary_path=summary_path,
        evaluation_report_path=eval_result.report_path,
        acceptance_audit_path=acceptance_result.report_path,
        compliance_audit_path=compliance_result.report_path,
        fit_report_path=Path(fit_report.output_paths["fit_report"]),
        obj_path=export_result.obj_path,
        ply_path=export_result.ply_path,
        glb_path=export_result.glb_path,
        skeleton_path=export_result.skeleton_path,
    )


def run_baseline_geometry_texture_bake(
    project_dir: Path,
    *,
    baseline_project_dir: Path,
    texture_source_project_dir: Path,
    rejected_texture_source_hashes: Sequence[str] = (),
    rejected_texture_source_frame_ids: Sequence[str] = (),
    probe_texture_source_frame_ids: Sequence[str] = (),
    probe_texture_source_sample_keys: Mapping[str, Sequence[str]] | None = None,
    probe_texture_source_ablation_overrides: Mapping[str, Sequence[str]] | None = None,
    force: bool = False,
) -> RawTextureOnlyBakeResult:
    """Re-bake raw texture evidence while preserving a baseline geometry fit."""

    project_dir = Path(project_dir).resolve()
    baseline_project_dir = Path(baseline_project_dir).resolve()
    texture_source_project_dir = Path(texture_source_project_dir).resolve()
    if project_dir.exists():
        if not force:
            raise FileExistsError(f"{project_dir} already exists; pass force=True to overwrite")
        shutil.rmtree(project_dir)
    shutil.copytree(baseline_project_dir, project_dir)

    texture_manifest_report = _write_baseline_geometry_texture_manifest(
        project_dir,
        baseline_project_dir=baseline_project_dir,
        texture_source_project_dir=texture_source_project_dir,
        rejected_texture_source_hashes=rejected_texture_source_hashes,
        rejected_texture_source_frame_ids=rejected_texture_source_frame_ids,
        probe_texture_source_frame_ids=probe_texture_source_frame_ids,
        probe_texture_source_sample_keys=probe_texture_source_sample_keys,
        probe_texture_source_ablation_overrides=probe_texture_source_ablation_overrides,
        force=True,
    )
    if int(texture_manifest_report["added_unique_texture_frame_count"]) == 0:
        summary_path = _write_texture_only_pipeline_summary(
            project_dir,
            baseline_project_dir=baseline_project_dir,
            texture_source_project_dir=texture_source_project_dir,
            texture_manifest_report=texture_manifest_report,
            export_result=None,
            eval_result=None,
            promotion_guard_report_path=None,
            promotion_guard_passed=None,
        )
        promotion_guard_path = (
            project_dir / "outputs" / "quality" / "texture_only_promotion_guard.json"
        )
        promotion_guard = write_texture_only_promotion_guard(
            baseline_project_dir,
            project_dir,
            promotion_guard_path,
            force=True,
        )
        _write_texture_only_pipeline_summary(
            project_dir,
            baseline_project_dir=baseline_project_dir,
            texture_source_project_dir=texture_source_project_dir,
            texture_manifest_report=texture_manifest_report,
            export_result=None,
            eval_result=None,
            promotion_guard_report_path=promotion_guard.report_path,
            promotion_guard_passed=promotion_guard.passed,
        )
        return RawTextureOnlyBakeResult(
            project_dir=project_dir,
            success=False,
            summary_path=summary_path,
            evaluation_report_path=project_dir / "outputs" / "quality" / "raw_evaluation.json",
            compliance_audit_path=None,
            promotion_guard_path=promotion_guard.report_path,
            obj_path=project_dir / "outputs" / "avatar" / "cat_canonical.obj",
            ply_path=project_dir / "outputs" / "avatar" / "cat_canonical.ply",
            glb_path=project_dir / "outputs" / "avatar" / "cat_avatar.glb",
            added_unique_texture_frame_count=0,
            promotion_guard_passed=bool(promotion_guard.passed),
            skeleton_path=project_dir / "outputs" / "avatar" / "skeleton.json",
        )

    export_result = export_fitted_canonical_mesh(
        project_dir,
        mode="raw",
        basecolor_seed_texture_dir=baseline_project_dir / "outputs" / "textures",
        force=True,
    )
    eval_result = evaluate_raw_reconstruction(project_dir, force=True)
    if (
        export_result.obj_path is None
        or export_result.ply_path is None
        or export_result.glb_path is None
    ):
        raise ValueError("texture-only bake requires OBJ, PLY, and GLB outputs")

    summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
    _write_texture_only_pipeline_summary(
        project_dir,
        baseline_project_dir=baseline_project_dir,
        texture_source_project_dir=texture_source_project_dir,
        texture_manifest_report=texture_manifest_report,
        export_result=export_result,
        eval_result=eval_result,
        promotion_guard_report_path=None,
        promotion_guard_passed=None,
    )
    promotion_guard_path = (
        project_dir / "outputs" / "quality" / "texture_only_promotion_guard.json"
    )
    promotion_guard = write_texture_only_promotion_guard(
        baseline_project_dir,
        project_dir,
        promotion_guard_path,
        force=True,
    )
    _write_texture_only_pipeline_summary(
        project_dir,
        baseline_project_dir=baseline_project_dir,
        texture_source_project_dir=texture_source_project_dir,
        texture_manifest_report=texture_manifest_report,
        export_result=export_result,
        eval_result=eval_result,
        promotion_guard_report_path=promotion_guard.report_path,
        promotion_guard_passed=promotion_guard.passed,
    )
    compliance_result = write_raw_objective_compliance_audit(project_dir, force=True)
    return RawTextureOnlyBakeResult(
        project_dir=project_dir,
        success=bool(eval_result.success and compliance_result.success and promotion_guard.passed),
        summary_path=summary_path,
        evaluation_report_path=eval_result.report_path,
        compliance_audit_path=compliance_result.report_path,
        promotion_guard_path=promotion_guard.report_path,
        obj_path=export_result.obj_path,
        ply_path=export_result.ply_path,
        glb_path=export_result.glb_path,
        added_unique_texture_frame_count=int(
            texture_manifest_report["added_unique_texture_frame_count"]
        ),
        promotion_guard_passed=bool(promotion_guard.passed),
        skeleton_path=export_result.skeleton_path,
    )


def run_raw_pipeline(
    project_dir: Path,
    *,
    input_paths: list[Path] | tuple[Path, ...] | None = None,
    min_frames: int = 2,
    alpha_threshold: float = 0.05,
    color_delta_threshold: float = 0.08,
    segmentation_backend: RawSegmentationBackend = "auto_subject_v1",
    keypoint_confidence: float = 0.5,
    camera_root_steps: int = 2,
    raw_feature_anchor_weight: float = 0.0,
    max_selected_frames: int | None = 5,
    selected_frame_ids: Sequence[str] | None = None,
    texture_frame_ids: Sequence[str] | None = None,
    frame_selection_label: str | None = None,
    force: bool = False,
) -> RawPipelineResult:
    project_dir = Path(project_dir).resolve()
    if raw_feature_anchor_weight < 0.0:
        raise ValueError("raw_feature_anchor_weight must be non-negative")
    staged_input_paths = _stage_raw_pipeline_input_media(
        project_dir,
        input_paths,
        force=force,
    )
    ingest_result = ingest_image_sources(
        project_dir,
        input_paths=staged_input_paths,
        min_frames=min_frames,
        force=force,
    )
    segmentation_result = segment_raw_image_observations(
        project_dir,
        alpha_threshold=alpha_threshold,
        color_delta_threshold=color_delta_threshold,
        segmentation_backend=segmentation_backend,
        force=force,
    )
    frame_selection_result = select_raw_observation_frames(
        project_dir,
        min_frames=min_frames,
        max_selected_frames=max_selected_frames,
        selected_frame_ids=selected_frame_ids,
        texture_frame_ids=texture_frame_ids,
        selection_label=frame_selection_label,
        force=force,
    )
    keypoint_result = bootstrap_raw_keypoints_and_pose(
        project_dir,
        keypoint_confidence=keypoint_confidence,
        force=force,
    )
    fit_report = run_raw_fit(
        project_dir,
        config=OptimizerConfig(
            camera_root_steps=camera_root_steps,
            shape_steps=0,
            silhouette_weight=1.0,
            feature_anchor_weight=raw_feature_anchor_weight,
        ),
        initial_root_offset=np.zeros(3, dtype=np.float64),
        force=force,
    )
    export_result = export_fitted_canonical_mesh(project_dir, mode="raw", force=force)
    eval_result = evaluate_raw_reconstruction(project_dir, force=force)
    if _raw_texture_export_refresh_needed(eval_result.metrics):
        export_result = export_fitted_canonical_mesh(project_dir, mode="raw", force=True)
        eval_result = evaluate_raw_reconstruction(project_dir, force=True)
    if (
        export_result.obj_path is None
        or export_result.ply_path is None
        or export_result.glb_path is None
    ):
        raise ValueError("raw pipeline requires OBJ, PLY, and GLB outputs")

    summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
    compliance_audit_path = (
        project_dir / "outputs" / "quality" / "raw_opencat3d_compliance_audit.json"
    )
    summary = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "success": bool(eval_result.success),
        "steps": {
            "raw_ingest": ingest_result.frame_count >= min_frames,
            "raw_segmentation": segmentation_result.frame_count == ingest_result.frame_count,
            "raw_frame_selection": frame_selection_result.selected_frame_count >= min_frames,
            "raw_keypoint_bootstrap": (
                keypoint_result.frame_count == frame_selection_result.selected_frame_count
            ),
            "fit": bool(fit_report.success),
            "export": True,
            "evaluate": bool(eval_result.success),
        },
        "metrics": {
            "raw_input_frame_count": int(frame_selection_result.input_frame_count),
            "raw_frame_count": int(frame_selection_result.selected_frame_count),
            "raw_selected_frame_count": int(frame_selection_result.selected_frame_count),
            "raw_rejected_frame_count": int(frame_selection_result.rejected_frame_count),
            "raw_input_photo_frame_count": int(
                frame_selection_result.input_media_counts.get("photo", 0)
            ),
            "raw_input_video_frame_count": int(
                frame_selection_result.input_media_counts.get("video", 0)
            ),
            "raw_selected_photo_frame_count": int(
                frame_selection_result.selected_media_counts.get("photo", 0)
            ),
            "raw_selected_video_frame_count": int(
                frame_selection_result.selected_media_counts.get("video", 0)
            ),
            "raw_texture_frame_count": int(frame_selection_result.texture_frame_count),
            "raw_texture_photo_frame_count": int(
                frame_selection_result.texture_media_counts.get("photo", 0)
            ),
            "raw_texture_video_frame_count": int(
                frame_selection_result.texture_media_counts.get("video", 0)
            ),
            "raw_balanced_media_selection_used": float(
                frame_selection_result.media_strategy == "balanced_photo_video"
            ),
            **_raw_frame_selection_detail_summary_metrics(
                frame_selection_result.detail_metrics
            ),
            "segmentation_mean_mask_area_ratio": float(
                frame_selection_result.mean_mask_area_ratio
            ),
            "segmentation_mean_bbox_area_ratio": float(
                frame_selection_result.mean_bbox_area_ratio
            ),
            "segmentation_full_frame_bbox_fraction": float(
                frame_selection_result.full_frame_bbox_fraction
            ),
            "keypoint_mean_confidence": float(keypoint_result.mean_keypoint_confidence),
            "keypoint_mean_in_bbox_fraction": float(keypoint_result.mean_in_bbox_fraction),
            "raw_keypoint_photo_frame_count": int(keypoint_result.photo_frame_count),
            "raw_keypoint_video_frame_count": int(keypoint_result.video_frame_count),
            "raw_keypoint_enriched_frame_count": int(
                keypoint_result.enriched_keypoint_frame_count
            ),
            "raw_keypoint_enriched_video_frame_count": int(
                keypoint_result.enriched_video_keypoint_frame_count
            ),
            "raw_observed_face_anchor_frame_count": int(
                keypoint_result.observed_face_anchor_frame_count
            ),
            "raw_observed_face_anchor_count": int(keypoint_result.observed_face_anchor_count),
            "raw_observed_face_anchor_mean_confidence": float(
                keypoint_result.mean_face_anchor_confidence
            ),
            "raw_keypoint_mean_frontal_face_confidence": float(
                keypoint_result.mean_frontal_face_confidence
            ),
            "raw_keypoint_mean_view_weight": float(
                keypoint_result.mean_keypoint_weight_scale
            ),
            "fit_init_prediction_used": float(fit_report.metrics["init_prediction_used"]),
            "raw_fit_feature_anchor_weight": float(
                fit_report.metrics["feature_anchor_weight"]
            ),
            "raw_fit_initial_feature_anchor_loss": float(
                fit_report.metrics["initial_feature_anchor_loss"]
            ),
            "raw_fit_final_feature_anchor_loss": float(
                fit_report.metrics["final_feature_anchor_loss"]
            ),
            "raw_fit_feature_anchor_loss_reduction": float(
                fit_report.metrics["feature_anchor_loss_reduction"]
            ),
            "raw_fit_mean_feature_anchor_target_count": float(
                fit_report.metrics["mean_feature_anchor_target_count"]
            ),
            "overlay_mean_silhouette_iou": float(
                fit_report.metrics["overlay_mean_silhouette_iou"]
            ),
            "raw_mean_mask_area_ratio": float(
                eval_result.metrics["raw_mean_mask_area_ratio"]
            ),
            "raw_mean_bbox_area_ratio": float(
                eval_result.metrics["raw_mean_bbox_area_ratio"]
            ),
            "raw_full_frame_bbox_fraction": float(
                eval_result.metrics["raw_full_frame_bbox_fraction"]
            ),
            "raw_border_touch_bbox_fraction": float(
                eval_result.metrics["raw_border_touch_bbox_fraction"]
            ),
            "raw_texture_unknown_luma": float(
                eval_result.metrics["raw_texture_unknown_luma"]
            ),
            "raw_texture_foreground_fallback_luma": float(
                eval_result.metrics["raw_texture_foreground_fallback_luma"]
            ),
            "raw_texture_unknown_luma_fraction_of_foreground": float(
                eval_result.metrics["raw_texture_unknown_luma_fraction_of_foreground"]
            ),
            "raw_texture_width": int(eval_result.metrics["raw_texture_width"]),
            "raw_texture_height": int(eval_result.metrics["raw_texture_height"]),
            "raw_texture_resolution_scale": float(
                eval_result.metrics["raw_texture_resolution_scale"]
            ),
            "raw_face_texture_target_bake_used": float(
                eval_result.metrics["raw_face_texture_target_bake_used"]
            ),
            "raw_face_texture_target_reported_size": int(
                eval_result.metrics["raw_face_texture_target_reported_size"]
            ),
            "raw_template_region_v2_texture_target_bake_used": float(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_bake_used"
                ]
            ),
            "raw_template_region_v2_texture_target_reported_size": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_reported_size"
                ]
            ),
            "raw_template_region_v2_texture_target_reported_scale": float(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_reported_scale"
                ]
            ),
            "raw_template_region_v2_texture_target_reported_island_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_reported_island_count"
                ]
            ),
            "raw_template_region_v2_texture_target_reported_facial_island_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_reported_facial_island_count"
                ]
            ),
            "raw_template_region_v2_texture_target_reported_requested_texel_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_reported_requested_texel_count"
                ]
            ),
            "raw_template_region_v2_texture_target_reported_allocated_texel_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_reported_allocated_texel_count"
                ]
            ),
            "raw_pbr_texture_width": int(eval_result.metrics["raw_pbr_texture_width"]),
            "raw_pbr_texture_height": int(eval_result.metrics["raw_pbr_texture_height"]),
            "raw_glb_texture_size": int(eval_result.metrics["raw_glb_texture_size"]),
            "raw_glb_texture_dimensions_valid": float(
                eval_result.metrics["raw_glb_texture_dimensions_valid"]
            ),
            "raw_glb_pbr_texture_dimensions_valid": float(
                eval_result.metrics["raw_glb_pbr_texture_dimensions_valid"]
            ),
            "raw_glb_basecolor_texture_valid": float(
                eval_result.metrics["raw_glb_basecolor_texture_valid"]
            ),
            "raw_glb_pbr_textures_valid": float(
                eval_result.metrics["raw_glb_pbr_textures_valid"]
            ),
            "raw_glb_embedded_texture_dimension_gate_valid": float(
                eval_result.metrics[
                    "raw_glb_embedded_texture_dimension_gate_valid"
                ]
            ),
            "raw_glb_face_feature_required_group_count": int(
                eval_result.metrics["raw_glb_face_feature_required_group_count"]
            ),
            "raw_glb_face_feature_visible_group_count": int(
                eval_result.metrics["raw_glb_face_feature_visible_group_count"]
            ),
            "raw_glb_face_feature_required_feature_count": int(
                eval_result.metrics["raw_glb_face_feature_required_feature_count"]
            ),
            "raw_glb_face_feature_present_required_feature_count": int(
                eval_result.metrics[
                    "raw_glb_face_feature_present_required_feature_count"
                ]
            ),
            "raw_glb_face_feature_visible_feature_count": int(
                eval_result.metrics["raw_glb_face_feature_visible_feature_count"]
            ),
            "raw_glb_face_feature_sampled_texel_count": int(
                eval_result.metrics["raw_glb_face_feature_sampled_texel_count"]
            ),
            "raw_glb_face_feature_min_visible_texel_count": int(
                eval_result.metrics["raw_glb_face_feature_min_visible_texel_count"]
            ),
            "raw_glb_face_feature_mean_rgb_error": float(
                eval_result.metrics["raw_glb_face_feature_mean_rgb_error"]
            ),
            "raw_glb_face_feature_max_rgb_error": float(
                eval_result.metrics["raw_glb_face_feature_max_rgb_error"]
            ),
            "raw_glb_face_feature_eye_visible": float(
                eval_result.metrics["raw_glb_face_feature_eye_visible"]
            ),
            "raw_glb_face_feature_nose_visible": float(
                eval_result.metrics["raw_glb_face_feature_nose_visible"]
            ),
            "raw_glb_face_feature_muzzle_visible": float(
                eval_result.metrics["raw_glb_face_feature_muzzle_visible"]
            ),
            "raw_glb_face_feature_iris_to_pupil_rgb_distance": float(
                eval_result.metrics[
                    "raw_glb_face_feature_iris_to_pupil_rgb_distance"
                ]
            ),
            "raw_glb_face_feature_eye_detail_luma_std": float(
                eval_result.metrics["raw_glb_face_feature_eye_detail_luma_std"]
            ),
            "raw_glb_face_feature_nose_to_muzzle_rgb_distance": float(
                eval_result.metrics[
                    "raw_glb_face_feature_nose_to_muzzle_rgb_distance"
                ]
            ),
            "raw_glb_face_feature_mouth_to_muzzle_rgb_distance": float(
                eval_result.metrics[
                    "raw_glb_face_feature_mouth_to_muzzle_rgb_distance"
                ]
            ),
            "raw_glb_face_feature_material_visibility_gate_valid": float(
                eval_result.metrics[
                    "raw_glb_face_feature_material_visibility_gate_valid"
                ]
            ),
            "raw_facial_feature_count": int(
                eval_result.metrics["raw_facial_feature_count"]
            ),
            "raw_facial_feature_painted_texel_count": int(
                eval_result.metrics["raw_facial_feature_painted_texel_count"]
            ),
            "raw_facial_feature_observed_texel_count": int(
                eval_result.metrics["raw_facial_feature_observed_texel_count"]
            ),
            "raw_facial_feature_observed_texel_fraction": float(
                eval_result.metrics["raw_facial_feature_observed_texel_fraction"]
            ),
            "raw_facial_feature_source_patch_count": int(
                eval_result.metrics["raw_facial_feature_source_patch_count"]
            ),
            "raw_facial_feature_source_patch_bake_candidate_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_bake_candidate_count"
                ]
            ),
            "raw_facial_feature_source_patch_bake_selected_feedback": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_bake_selected_feedback"
                ]
            ),
            "raw_facial_feature_source_patch_bake_visual_quality_passed_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_bake_visual_quality_passed_count"
                ]
            ),
            "raw_facial_feature_source_patch_bake_visual_quality_min_score": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_bake_visual_quality_min_score"
                ]
            ),
            "raw_facial_feature_source_patch_bake_visual_quality_mean_score": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_bake_visual_quality_mean_score"
                ]
            ),
            "raw_facial_feature_eye_source_detail_reinforcement_feature_count": int(
                eval_result.metrics[
                    "raw_facial_feature_eye_source_detail_reinforcement_feature_count"
                ]
            ),
            "raw_facial_feature_eye_source_detail_reinforcement_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_eye_source_detail_reinforcement_texel_count"
                ]
            ),
            "raw_facial_feature_eye_source_detail_reinforcement_texel_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_eye_source_detail_reinforcement_texel_fraction"
                ]
            ),
            "raw_facial_feature_whisker_pad_feature_count": int(
                eval_result.metrics[
                    "raw_facial_feature_whisker_pad_feature_count"
                ]
            ),
            "raw_facial_feature_whisker_pad_texel_count": int(
                eval_result.metrics["raw_facial_feature_whisker_pad_texel_count"]
            ),
            "raw_facial_feature_whisker_pad_texel_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_whisker_pad_texel_fraction"
                ]
            ),
            "raw_facial_feature_iris_halo_feature_count": int(
                eval_result.metrics["raw_facial_feature_iris_halo_feature_count"]
            ),
            "raw_facial_feature_iris_halo_uv_support_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_iris_halo_uv_support_texel_count"
                ]
            ),
            "raw_facial_feature_iris_halo_core_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_iris_halo_core_texel_count"
                ]
            ),
            "raw_facial_feature_iris_halo_edge_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_iris_halo_edge_texel_count"
                ]
            ),
            "raw_facial_feature_iris_halo_accepted_core_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_iris_halo_accepted_core_texel_count"
                ]
            ),
            "raw_facial_feature_iris_halo_edge_fraction": float(
                eval_result.metrics["raw_facial_feature_iris_halo_edge_fraction"]
            ),
            "raw_facial_feature_iris_halo_edge_to_core_ratio": float(
                eval_result.metrics[
                    "raw_facial_feature_iris_halo_edge_to_core_ratio"
                ]
            ),
            "raw_facial_feature_iris_halo_max_edge_to_core_ratio": float(
                eval_result.metrics[
                    "raw_facial_feature_iris_halo_max_edge_to_core_ratio"
                ]
            ),
            "raw_facial_feature_iris_halo_support_to_accepted_core_ratio": float(
                eval_result.metrics[
                    "raw_facial_feature_iris_halo_support_to_accepted_core_ratio"
                ]
            ),
            "raw_facial_feature_left_iris_halo_uv_support_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_left_iris_halo_uv_support_texel_count"
                ]
            ),
            "raw_facial_feature_left_iris_halo_core_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_left_iris_halo_core_texel_count"
                ]
            ),
            "raw_facial_feature_left_iris_halo_edge_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_left_iris_halo_edge_texel_count"
                ]
            ),
            "raw_facial_feature_left_iris_halo_accepted_core_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_left_iris_halo_accepted_core_texel_count"
                ]
            ),
            "raw_facial_feature_left_iris_halo_edge_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_left_iris_halo_edge_fraction"
                ]
            ),
            "raw_facial_feature_left_iris_halo_edge_to_core_ratio": float(
                eval_result.metrics[
                    "raw_facial_feature_left_iris_halo_edge_to_core_ratio"
                ]
            ),
            "raw_facial_feature_left_iris_halo_support_to_accepted_core_ratio": float(
                eval_result.metrics[
                    "raw_facial_feature_left_iris_halo_support_to_accepted_core_ratio"
                ]
            ),
            "raw_facial_feature_right_iris_halo_uv_support_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_right_iris_halo_uv_support_texel_count"
                ]
            ),
            "raw_facial_feature_right_iris_halo_core_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_right_iris_halo_core_texel_count"
                ]
            ),
            "raw_facial_feature_right_iris_halo_edge_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_right_iris_halo_edge_texel_count"
                ]
            ),
            "raw_facial_feature_right_iris_halo_accepted_core_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_right_iris_halo_accepted_core_texel_count"
                ]
            ),
            "raw_facial_feature_right_iris_halo_edge_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_right_iris_halo_edge_fraction"
                ]
            ),
            "raw_facial_feature_right_iris_halo_edge_to_core_ratio": float(
                eval_result.metrics[
                    "raw_facial_feature_right_iris_halo_edge_to_core_ratio"
                ]
            ),
            "raw_facial_feature_right_iris_halo_support_to_accepted_core_ratio": float(
                eval_result.metrics[
                    "raw_facial_feature_right_iris_halo_support_to_accepted_core_ratio"
                ]
            ),
            "raw_face_texture_bake_comparison_candidate_count": int(
                eval_result.metrics["raw_face_texture_bake_comparison_candidate_count"]
            ),
            "raw_face_texture_bake_comparison_region_count": int(
                eval_result.metrics["raw_face_texture_bake_comparison_region_count"]
            ),
            "raw_face_texture_bake_comparison_selected_feedback": float(
                eval_result.metrics["raw_face_texture_bake_comparison_selected_feedback"]
            ),
            "raw_face_texture_bake_comparison_score_improved_region_count": int(
                eval_result.metrics[
                    "raw_face_texture_bake_comparison_score_improved_region_count"
                ]
            ),
            "raw_face_texture_bake_comparison_coverage_improved_region_count": int(
                eval_result.metrics[
                    "raw_face_texture_bake_comparison_coverage_improved_region_count"
                ]
            ),
            "raw_face_texture_bake_comparison_source_detail_improved_region_count": int(
                eval_result.metrics[
                    "raw_face_texture_bake_comparison_source_detail_improved_region_count"
                ]
            ),
            "raw_face_texture_bake_comparison_max_score_delta": float(
                eval_result.metrics["raw_face_texture_bake_comparison_max_score_delta"]
            ),
            "raw_face_texture_bake_comparison_max_coverage_delta": float(
                eval_result.metrics[
                    "raw_face_texture_bake_comparison_max_coverage_delta"
                ]
            ),
            "raw_face_texture_bake_comparison_max_source_detail_delta": float(
                eval_result.metrics[
                    "raw_face_texture_bake_comparison_max_source_detail_delta"
                ]
            ),
            "raw_face_texture_bake_comparison_selected_not_lower_rank": float(
                eval_result.metrics[
                    "raw_face_texture_bake_comparison_selected_not_lower_rank"
                ]
            ),
            "raw_facial_feature_source_patch_texel_count": int(
                eval_result.metrics["raw_facial_feature_source_patch_texel_count"]
            ),
            "raw_facial_feature_source_patch_texel_fraction": float(
                eval_result.metrics["raw_facial_feature_source_patch_texel_fraction"]
            ),
            "raw_facial_feature_source_patch_quality_passed_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_quality_passed_count"
                ]
            ),
            "raw_facial_feature_source_patch_quality_total_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_quality_total_count"
                ]
            ),
            "raw_facial_feature_source_patch_quality_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_quality_fraction"
                ]
            ),
            "raw_facial_feature_source_patch_min_quality_score": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_min_quality_score"
                ]
            ),
            "raw_facial_feature_source_patch_mean_quality_score": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_quality_score"
                ]
            ),
            "raw_facial_feature_source_patch_view_weighting_used": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_view_weighting_used"
                ]
            ),
            "raw_facial_feature_source_patch_view_weighted_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_view_weighted_count"
                ]
            ),
            "raw_facial_feature_source_patch_min_view_weight": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_min_view_weight"
                ]
            ),
            "raw_facial_feature_source_patch_mean_view_weight": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_view_weight"
                ]
            ),
            "raw_facial_feature_source_patch_min_frontal_face_confidence": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_min_frontal_face_confidence"
                ]
            ),
            "raw_facial_feature_source_patch_mean_frontal_face_confidence": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_frontal_face_confidence"
                ]
            ),
            "raw_facial_feature_source_patch_min_view_weighted_rank": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_min_view_weighted_rank"
                ]
            ),
            "raw_facial_feature_source_patch_mean_view_weighted_rank": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_view_weighted_rank"
                ]
            ),
            "raw_facial_feature_source_patch_frontal_photo_priority_used": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_frontal_photo_priority_used"
                ]
            ),
            "raw_facial_feature_source_patch_frontal_photo_priority_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_frontal_photo_priority_count"
                ]
            ),
            "raw_facial_feature_source_patch_max_frontal_photo_priority": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_max_frontal_photo_priority"
                ]
            ),
            "raw_facial_feature_source_patch_mean_frontal_photo_priority": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_frontal_photo_priority"
                ]
            ),
            "raw_facial_feature_source_patch_fusion_min_weight": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_fusion_min_weight"
                ]
            ),
            "raw_facial_feature_source_patch_fusion_mean_weight": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_fusion_mean_weight"
                ]
            ),
            "raw_facial_feature_source_patch_fusion_weighted_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_fusion_weighted_texel_count"
                ]
            ),
            "raw_facial_feature_source_patch_nominal_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_nominal_texel_count"
                ]
            ),
            "raw_facial_feature_source_patch_footprint_reduction_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_footprint_reduction_fraction"
                ]
            ),
            "raw_facial_feature_source_patch_mean_footprint_area_scale": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_footprint_area_scale"
                ]
            ),
            "raw_facial_feature_source_patch_min_footprint_area_scale": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_min_footprint_area_scale"
                ]
            ),
            "raw_facial_feature_source_patch_shape_mask_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_shape_mask_texel_count"
                ]
            ),
            "raw_facial_feature_source_patch_shape_mask_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_shape_mask_fraction"
                ]
            ),
            "raw_facial_feature_source_patch_shape_alpha_mean": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_shape_alpha_mean"
                ]
            ),
            "raw_facial_feature_source_patch_source_mask_observed_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_source_mask_observed_texel_count"
                ]
            ),
            "raw_facial_feature_source_patch_source_mask_densified_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_source_mask_densified_texel_count"
                ]
            ),
            "raw_facial_feature_source_patch_source_mask_densification_gain": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_source_mask_densification_gain"
                ]
            ),
            "raw_facial_feature_source_patch_mean_source_mask_densification_gain": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_source_mask_densification_gain"
                ]
            ),
            "raw_facial_feature_source_patch_localization_candidate_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_localization_candidate_texel_count"
                ]
            ),
            "raw_facial_feature_source_patch_localization_selected_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_localization_selected_texel_count"
                ]
            ),
            "raw_facial_feature_source_patch_localization_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_localization_fraction"
                ]
            ),
            "raw_facial_feature_source_patch_mean_localization_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_localization_fraction"
                ]
            ),
            "raw_facial_feature_source_patch_landmark_aligned_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_landmark_aligned_count"
                ]
            ),
            "raw_facial_feature_source_patch_mean_landmark_confidence": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_landmark_confidence"
                ]
            ),
            "raw_facial_feature_source_patch_observed_anchor_guided_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_observed_anchor_guided_count"
                ]
            ),
            "raw_facial_feature_source_patch_observed_anchor_guided_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_observed_anchor_guided_fraction"
                ]
            ),
            "raw_facial_feature_source_patch_landmark_destination_scaled_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_landmark_destination_scaled_count"
                ]
            ),
            "raw_facial_feature_source_patch_mean_landmark_destination_scale": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_landmark_destination_scale"
                ]
            ),
            "raw_face_feature_destination_anchor_anatomical_count": int(
                eval_result.metrics[
                    "raw_face_feature_destination_anchor_anatomical_count"
                ]
            ),
            "raw_face_feature_destination_anchor_order_valid": float(
                eval_result.metrics[
                    "raw_face_feature_destination_anchor_order_valid"
                ]
            ),
            "raw_face_feature_destination_anchor_eye_to_nose_v_gap": float(
                eval_result.metrics[
                    "raw_face_feature_destination_anchor_eye_to_nose_v_gap"
                ]
            ),
            "raw_face_feature_destination_anchor_nose_to_mouth_v_gap": float(
                eval_result.metrics[
                    "raw_face_feature_destination_anchor_nose_to_mouth_v_gap"
                ]
            ),
            "raw_face_feature_destination_anchor_nose_to_muzzle_v_gap": float(
                eval_result.metrics[
                    "raw_face_feature_destination_anchor_nose_to_muzzle_v_gap"
                ]
            ),
            "raw_face_feature_destination_anchor_ear_to_eye_v_gap": float(
                eval_result.metrics[
                    "raw_face_feature_destination_anchor_ear_to_eye_v_gap"
                ]
            ),
            "raw_face_feature_destination_anchor_eye_symmetry_u_error": float(
                eval_result.metrics[
                    "raw_face_feature_destination_anchor_eye_symmetry_u_error"
                ]
            ),
            "raw_facial_feature_source_patch_core_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_core_texel_count"
                ]
            ),
            "raw_facial_feature_source_patch_core_fraction": float(
                eval_result.metrics["raw_facial_feature_source_patch_core_fraction"]
            ),
            "raw_facial_feature_source_patch_feathered_edge_texel_count": int(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_feathered_edge_texel_count"
                ]
            ),
            "raw_facial_feature_source_patch_feathered_edge_fraction": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_feathered_edge_fraction"
                ]
            ),
            "raw_facial_feature_source_patch_mean_blend_alpha": float(
                eval_result.metrics[
                    "raw_facial_feature_source_patch_mean_blend_alpha"
                ]
            ),
            "raw_face_priority_texture_used": float(
                eval_result.metrics["raw_face_priority_texture_used"]
            ),
            "raw_face_uv_region_count": int(
                eval_result.metrics["raw_face_uv_region_count"]
            ),
            "raw_face_uv_region_total_area": float(
                eval_result.metrics["raw_face_uv_region_total_area"]
            ),
            "raw_face_uv_region_min_density_weight": float(
                eval_result.metrics["raw_face_uv_region_min_density_weight"]
            ),
            "raw_face_uv_region_contract_valid": float(
                eval_result.metrics["raw_face_uv_region_contract_valid"]
            ),
            "raw_template_region_v2_count": int(
                eval_result.metrics["raw_template_region_v2_count"]
            ),
            "raw_template_region_v2_required_region_count": int(
                eval_result.metrics["raw_template_region_v2_required_region_count"]
            ),
            "raw_template_region_v2_required_present_count": int(
                eval_result.metrics["raw_template_region_v2_required_present_count"]
            ),
            "raw_template_region_v2_contract_valid": float(
                eval_result.metrics["raw_template_region_v2_contract_valid"]
            ),
            "raw_glb_template_region_v2_contract_valid": float(
                eval_result.metrics["raw_glb_template_region_v2_contract_valid"]
            ),
            "raw_template_region_v2_belly_vertex_count": int(
                eval_result.metrics["raw_template_region_v2_belly_vertex_count"]
            ),
            "raw_face_uv_density_allocated_texel_count": int(
                eval_result.metrics["raw_face_uv_density_allocated_texel_count"]
            ),
            "raw_face_uv_density_min_allocated_texels": int(
                eval_result.metrics["raw_face_uv_density_min_allocated_texels"]
            ),
            "raw_face_uv_density_eye_min_allocated_texels": int(
                eval_result.metrics["raw_face_uv_density_eye_min_allocated_texels"]
            ),
            "raw_face_uv_density_nose_allocated_texels": int(
                eval_result.metrics["raw_face_uv_density_nose_allocated_texels"]
            ),
            "raw_face_uv_density_ear_inner_min_allocated_texels": int(
                eval_result.metrics["raw_face_uv_density_ear_inner_min_allocated_texels"]
            ),
            "raw_template_region_v2_uv_region_count": int(
                eval_result.metrics["raw_template_region_v2_uv_region_count"]
            ),
            "raw_template_region_v2_uv_required_present_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_uv_required_present_count"
                ]
            ),
            "raw_template_region_v2_uv_island_count": int(
                eval_result.metrics["raw_template_region_v2_uv_island_count"]
            ),
            "raw_template_region_v2_region_scoped_uv_island_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_region_scoped_uv_island_count"
                ]
            ),
            "raw_template_region_v2_uv_island_allocated_texel_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_uv_island_allocated_texel_count"
                ]
            ),
            "raw_template_region_v2_uv_island_density_weighted_texel_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_uv_island_density_weighted_texel_count"
                ]
            ),
            "raw_template_region_v2_face_uv_island_count": int(
                eval_result.metrics["raw_template_region_v2_face_uv_island_count"]
            ),
            "raw_template_region_v2_facial_uv_island_count": int(
                eval_result.metrics["raw_template_region_v2_facial_uv_island_count"]
            ),
            "raw_template_region_v2_facial_uv_island_allocated_texel_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_facial_uv_island_allocated_texel_count"
                ]
            ),
            "raw_template_region_v2_body_region_uv_island_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_body_region_uv_island_count"
                ]
            ),
            "raw_template_region_v2_eye_min_allocated_texels": int(
                eval_result.metrics["raw_template_region_v2_eye_min_allocated_texels"]
            ),
            "raw_template_region_v2_nose_allocated_texels": int(
                eval_result.metrics["raw_template_region_v2_nose_allocated_texels"]
            ),
            "raw_template_region_v2_muzzle_allocated_texels": int(
                eval_result.metrics["raw_template_region_v2_muzzle_allocated_texels"]
            ),
            "raw_template_region_v2_ear_inner_min_allocated_texels": int(
                eval_result.metrics[
                    "raw_template_region_v2_ear_inner_min_allocated_texels"
                ]
            ),
            "raw_template_region_v2_facial_density_weighted_texel_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_facial_density_weighted_texel_count"
                ]
            ),
            "raw_face_texture_target_size": int(
                eval_result.metrics["raw_face_texture_target_size"]
            ),
            "raw_face_texture_target_scale": float(
                eval_result.metrics["raw_face_texture_target_scale"]
            ),
            "raw_face_texture_target_requested_texel_count": int(
                eval_result.metrics["raw_face_texture_target_requested_texel_count"]
            ),
            "raw_face_texture_target_allocated_texel_count": int(
                eval_result.metrics["raw_face_texture_target_allocated_texel_count"]
            ),
            "raw_face_texture_target_eye_min_texels": int(
                eval_result.metrics["raw_face_texture_target_eye_min_texels"]
            ),
            "raw_face_texture_target_nose_texels": int(
                eval_result.metrics["raw_face_texture_target_nose_texels"]
            ),
            "raw_face_texture_target_ear_inner_min_texels": int(
                eval_result.metrics["raw_face_texture_target_ear_inner_min_texels"]
            ),
            "raw_template_region_v2_texture_target_size": int(
                eval_result.metrics["raw_template_region_v2_texture_target_size"]
            ),
            "raw_template_region_v2_texture_target_scale": float(
                eval_result.metrics["raw_template_region_v2_texture_target_scale"]
            ),
            "raw_template_region_v2_texture_target_island_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_island_count"
                ]
            ),
            "raw_template_region_v2_texture_target_facial_island_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_facial_island_count"
                ]
            ),
            "raw_template_region_v2_texture_target_requested_texel_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_requested_texel_count"
                ]
            ),
            "raw_template_region_v2_texture_target_allocated_texel_count": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_allocated_texel_count"
                ]
            ),
            "raw_template_region_v2_texture_target_eye_min_texels": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_eye_min_texels"
                ]
            ),
            "raw_template_region_v2_texture_target_nose_texels": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_nose_texels"
                ]
            ),
            "raw_template_region_v2_texture_target_muzzle_texels": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_muzzle_texels"
                ]
            ),
            "raw_template_region_v2_texture_target_ear_inner_min_texels": int(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_ear_inner_min_texels"
                ]
            ),
            "raw_template_region_v2_texture_target_max_density_scale": float(
                eval_result.metrics[
                    "raw_template_region_v2_texture_target_max_density_scale"
                ]
            ),
            "raw_face_texture_crop_region_count": int(
                eval_result.metrics["raw_face_texture_crop_region_count"]
            ),
            "raw_face_texture_crop_texture_width": int(
                eval_result.metrics["raw_face_texture_crop_texture_width"]
            ),
            "raw_face_texture_crop_texture_height": int(
                eval_result.metrics["raw_face_texture_crop_texture_height"]
            ),
            "raw_face_texture_crop_compared_region_count": int(
                eval_result.metrics["raw_face_texture_crop_compared_region_count"]
            ),
            "raw_face_texture_crop_exact_source_patch_region_count": int(
                eval_result.metrics[
                    "raw_face_texture_crop_exact_source_patch_region_count"
                ]
            ),
            "raw_face_texture_crop_exact_source_patch_fraction": float(
                eval_result.metrics[
                    "raw_face_texture_crop_exact_source_patch_fraction"
                ]
            ),
            "raw_face_texture_crop_detail_region_count": int(
                eval_result.metrics["raw_face_texture_crop_detail_region_count"]
            ),
            "raw_face_texture_crop_min_unique_rgb_count": int(
                eval_result.metrics["raw_face_texture_crop_min_unique_rgb_count"]
            ),
            "raw_face_texture_crop_min_confident_texel_fraction": float(
                eval_result.metrics["raw_face_texture_crop_min_confident_texel_fraction"]
            ),
            "raw_face_texture_crop_mean_confident_texel_fraction": float(
                eval_result.metrics[
                    "raw_face_texture_crop_mean_confident_texel_fraction"
                ]
            ),
            "raw_face_texture_crop_mean_source_rgb_delta": float(
                eval_result.metrics["raw_face_texture_crop_mean_source_rgb_delta"]
            ),
            "raw_face_texture_crop_isolated_region_count": int(
                eval_result.metrics["raw_face_texture_crop_isolated_region_count"]
            ),
            "raw_face_texture_crop_overlap_pair_count": int(
                eval_result.metrics["raw_face_texture_crop_overlap_pair_count"]
            ),
            "raw_face_texture_crop_overlap_fraction": float(
                eval_result.metrics["raw_face_texture_crop_overlap_fraction"]
            ),
            "raw_face_texture_crop_visual_quality_region_count": int(
                eval_result.metrics[
                    "raw_face_texture_crop_visual_quality_region_count"
                ]
            ),
            "raw_face_texture_crop_visual_quality_passed_count": int(
                eval_result.metrics[
                    "raw_face_texture_crop_visual_quality_passed_count"
                ]
            ),
            "raw_face_texture_crop_visual_quality_fraction": float(
                eval_result.metrics[
                    "raw_face_texture_crop_visual_quality_fraction"
                ]
            ),
            "raw_face_texture_crop_visual_quality_min_score": float(
                eval_result.metrics[
                    "raw_face_texture_crop_visual_quality_min_score"
                ]
            ),
            "raw_face_texture_crop_visual_quality_mean_score": float(
                eval_result.metrics[
                    "raw_face_texture_crop_visual_quality_mean_score"
                ]
            ),
            "raw_direct_photo_face_candidate_sample_key_count": int(
                eval_result.metrics[
                    "raw_direct_photo_face_candidate_sample_key_count"
                ]
            ),
            "raw_direct_photo_face_candidate_total_count": int(
                eval_result.metrics["raw_direct_photo_face_candidate_total_count"]
            ),
            "raw_direct_photo_face_candidate_direct_count": int(
                eval_result.metrics["raw_direct_photo_face_candidate_direct_count"]
            ),
            "raw_direct_photo_face_candidate_eligible_count": int(
                eval_result.metrics["raw_direct_photo_face_candidate_eligible_count"]
            ),
            "raw_direct_photo_face_candidate_selected_count": int(
                eval_result.metrics["raw_direct_photo_face_candidate_selected_count"]
            ),
            "raw_direct_photo_face_candidate_rejected_count": int(
                eval_result.metrics["raw_direct_photo_face_candidate_rejected_count"]
            ),
            "raw_direct_photo_face_candidate_missing_metric_feature_count": int(
                eval_result.metrics[
                    "raw_direct_photo_face_candidate_missing_metric_feature_count"
                ]
            ),
            "raw_direct_photo_face_candidate_max_priority": float(
                eval_result.metrics["raw_direct_photo_face_candidate_max_priority"]
            ),
            "raw_face_feature_min_required_contrast": float(
                eval_result.metrics["raw_face_feature_min_required_contrast"]
            ),
            "raw_face_feature_eye_to_muzzle_luma_delta": float(
                eval_result.metrics["raw_face_feature_eye_to_muzzle_luma_delta"]
            ),
            "raw_face_feature_mouth_to_muzzle_luma_delta": float(
                eval_result.metrics["raw_face_feature_mouth_to_muzzle_luma_delta"]
            ),
            "raw_face_feature_nose_to_muzzle_rgb_distance": float(
                eval_result.metrics["raw_face_feature_nose_to_muzzle_rgb_distance"]
            ),
            "raw_face_feature_iris_to_pupil_rgb_distance": float(
                eval_result.metrics["raw_face_feature_iris_to_pupil_rgb_distance"]
            ),
            "raw_face_feature_contrast_passed": float(
                eval_result.metrics["raw_face_feature_contrast_passed"]
            ),
            "raw_face_detail_geometry_applied": float(
                eval_result.metrics["raw_face_detail_geometry_applied"]
            ),
            "raw_face_detail_deformed_vertex_count": int(
                eval_result.metrics["raw_face_detail_deformed_vertex_count"]
            ),
            "raw_face_detail_max_displacement": float(
                eval_result.metrics["raw_face_detail_max_displacement"]
            ),
            "raw_face_detail_mean_displacement": float(
                eval_result.metrics["raw_face_detail_mean_displacement"]
            ),
            "raw_face_surface_anchor_count": int(
                eval_result.metrics["raw_face_surface_anchor_count"]
            ),
            "raw_face_surface_anchor_vertex_count": int(
                eval_result.metrics["raw_face_surface_anchor_vertex_count"]
            ),
            "raw_face_surface_anchor_max_displacement": float(
                eval_result.metrics["raw_face_surface_anchor_max_displacement"]
            ),
            "raw_face_turntable_view_count": int(
                eval_result.metrics["raw_face_turntable_view_count"]
            ),
            "raw_face_turntable_front_visible_anchor_count": int(
                eval_result.metrics["raw_face_turntable_front_visible_anchor_count"]
            ),
            "raw_face_turntable_front_anchor_pixel_count": int(
                eval_result.metrics["raw_face_turntable_front_anchor_pixel_count"]
            ),
            "raw_face_turntable_required_anchor_visible_fraction": float(
                eval_result.metrics["raw_face_turntable_required_anchor_visible_fraction"]
            ),
            "raw_face_turntable_textured_view_count": int(
                eval_result.metrics["raw_face_turntable_textured_view_count"]
            ),
            "raw_face_turntable_textured_written_count": int(
                eval_result.metrics["raw_face_turntable_textured_written_count"]
            ),
            "raw_face_turntable_front_textured_unique_rgb_count": int(
                eval_result.metrics[
                    "raw_face_turntable_front_textured_unique_rgb_count"
                ]
            ),
            "raw_face_turntable_front_textured_luma_std": float(
                eval_result.metrics["raw_face_turntable_front_textured_luma_std"]
            ),
            "raw_face_turntable_front_textured_mouth_line_pixel_count": int(
                eval_result.metrics[
                    "raw_face_turntable_front_textured_mouth_line_pixel_count"
                ]
            ),
            "raw_face_turntable_front_textured_required_feature_visible_count": int(
                eval_result.metrics[
                    "raw_face_turntable_front_textured_required_feature_visible_count"
                ]
            ),
            "raw_face_turntable_front_textured_required_feature_fraction": float(
                eval_result.metrics[
                    "raw_face_turntable_front_textured_required_feature_fraction"
                ]
            ),
            "raw_face_turntable_mean_textured_unique_rgb_count": float(
                eval_result.metrics[
                    "raw_face_turntable_mean_textured_unique_rgb_count"
                ]
            ),
            "raw_face_turntable_mean_textured_luma_std": float(
                eval_result.metrics["raw_face_turntable_mean_textured_luma_std"]
            ),
            "raw_face_texture_closeup_width": int(
                eval_result.metrics["raw_face_texture_closeup_width"]
            ),
            "raw_face_texture_closeup_height": int(
                eval_result.metrics["raw_face_texture_closeup_height"]
            ),
            "raw_face_texture_closeup_feature_count": int(
                eval_result.metrics["raw_face_texture_closeup_feature_count"]
            ),
            "raw_face_texture_closeup_required_feature_count": int(
                eval_result.metrics["raw_face_texture_closeup_required_feature_count"]
            ),
            "raw_face_texture_closeup_required_present_count": int(
                eval_result.metrics["raw_face_texture_closeup_required_present_count"]
            ),
            "raw_face_texture_closeup_required_feature_fraction": float(
                eval_result.metrics["raw_face_texture_closeup_required_feature_fraction"]
            ),
            "raw_face_texture_closeup_crop_texel_count": int(
                eval_result.metrics["raw_face_texture_closeup_crop_texel_count"]
            ),
            "raw_face_texture_closeup_unique_rgb_count": int(
                eval_result.metrics["raw_face_texture_closeup_unique_rgb_count"]
            ),
            "raw_face_texture_closeup_luma_std": float(
                eval_result.metrics["raw_face_texture_closeup_luma_std"]
            ),
            "raw_face_texture_source_compare_feature_count": int(
                eval_result.metrics["raw_face_texture_source_compare_feature_count"]
            ),
            "raw_face_texture_source_compare_source_feature_available_count": int(
                eval_result.metrics[
                    "raw_face_texture_source_compare_source_feature_available_count"
                ]
            ),
            "raw_face_texture_source_compare_required_feature_count": int(
                eval_result.metrics[
                    "raw_face_texture_source_compare_required_feature_count"
                ]
            ),
            "raw_face_texture_source_compare_required_present_count": int(
                eval_result.metrics[
                    "raw_face_texture_source_compare_required_present_count"
                ]
            ),
            "raw_face_texture_source_compare_required_feature_fraction": float(
                eval_result.metrics[
                    "raw_face_texture_source_compare_required_feature_fraction"
                ]
            ),
            "raw_face_texture_source_compare_mean_rgb_distance": float(
                eval_result.metrics[
                    "raw_face_texture_source_compare_mean_rgb_distance"
                ]
            ),
            "raw_face_texture_source_compare_max_rgb_distance": float(
                eval_result.metrics[
                    "raw_face_texture_source_compare_max_rgb_distance"
                ]
            ),
            "raw_face_texture_source_compare_mean_luma_delta": float(
                eval_result.metrics[
                    "raw_face_texture_source_compare_mean_luma_delta"
                ]
            ),
            "raw_face_texture_source_compare_max_luma_delta": float(
                eval_result.metrics[
                    "raw_face_texture_source_compare_max_luma_delta"
                ]
            ),
            "raw_face_texture_source_compare_source_mean_luma_std": float(
                eval_result.metrics[
                    "raw_face_texture_source_compare_source_mean_luma_std"
                ]
            ),
            "raw_face_texture_source_compare_baked_mean_luma_std": float(
                eval_result.metrics[
                    "raw_face_texture_source_compare_baked_mean_luma_std"
                ]
            ),
            "raw_face_texture_source_compare_iris_feature_count": int(
                eval_result.metrics[
                    "raw_face_texture_source_compare_iris_feature_count"
                ]
            ),
            "raw_face_texture_source_compare_iris_candidate_texel_count": int(
                eval_result.metrics[
                    "raw_face_texture_source_compare_iris_candidate_texel_count"
                ]
            ),
            "raw_face_texture_source_compare_iris_core_accepted_texel_count": int(
                eval_result.metrics[
                    "raw_face_texture_source_compare_iris_core_accepted_texel_count"
                ]
            ),
            "raw_face_texture_source_compare_iris_core_densified_texel_count": int(
                eval_result.metrics[
                    "raw_face_texture_source_compare_iris_core_densified_texel_count"
                ]
            ),
            "raw_face_texture_source_compare_iris_annulus_rejected_texel_count": int(
                eval_result.metrics[
                    "raw_face_texture_source_compare_iris_annulus_rejected_texel_count"
                ]
            ),
            "raw_face_texture_source_compare_iris_core_acceptance_fraction": float(
                eval_result.metrics[
                    "raw_face_texture_source_compare_iris_core_acceptance_fraction"
                ]
            ),
            "raw_face_texture_source_compare_iris_annulus_rejected_fraction": float(
                eval_result.metrics[
                    "raw_face_texture_source_compare_iris_annulus_rejected_fraction"
                ]
            ),
            "observed_face_fraction": float(eval_result.metrics["observed_face_fraction"]),
            "observed_vertex_fraction": float(
                eval_result.metrics["observed_vertex_fraction"]
            ),
            "generated_face_count": int(eval_result.metrics["generated_face_count"]),
            "basecolor_observed_texel_fraction": float(
                eval_result.metrics["basecolor_observed_texel_fraction"]
            ),
            "basecolor_generated_texel_fraction": float(
                eval_result.metrics["basecolor_generated_texel_fraction"]
            ),
            "pbr_observed_texel_fraction": float(
                eval_result.metrics["pbr_observed_texel_fraction"]
            ),
            "pbr_generated_texel_fraction": float(
                eval_result.metrics["pbr_generated_texel_fraction"]
            ),
            "raw_pbr_face_material_detail_applied": float(
                eval_result.metrics["raw_pbr_face_material_detail_applied"]
            ),
            "raw_pbr_face_material_detail_feature_count": int(
                eval_result.metrics["raw_pbr_face_material_detail_feature_count"]
            ),
            "raw_pbr_face_material_detail_texel_count": int(
                eval_result.metrics["raw_pbr_face_material_detail_texel_count"]
            ),
            "raw_pbr_face_material_detail_normal_variance": float(
                eval_result.metrics["raw_pbr_face_material_detail_normal_variance"]
            ),
            "joint_projection_mean_keypoint_loss": float(
                eval_result.metrics["joint_projection_mean_keypoint_loss"]
            ),
            "joint_projection_mean_pixel_error": float(
                eval_result.metrics["joint_projection_mean_pixel_error"]
            ),
            "joint_projection_observed_keypoint_frame_count": int(
                eval_result.metrics["joint_projection_observed_keypoint_frame_count"]
            ),
            "joint_projection_sparse_keypoint_frame_count": int(
                eval_result.metrics["joint_projection_sparse_keypoint_frame_count"]
            ),
            "joint_projection_missing_keypoint_frame_count": int(
                eval_result.metrics["joint_projection_missing_keypoint_frame_count"]
            ),
            "silhouette_residual_mean_union_error": float(
                eval_result.metrics["silhouette_residual_mean_union_error"]
            ),
            "silhouette_residual_mean_false_negative": float(
                eval_result.metrics["silhouette_residual_mean_false_negative"]
            ),
            "silhouette_residual_mean_false_positive": float(
                eval_result.metrics["silhouette_residual_mean_false_positive"]
            ),
            "initial_loss": float(eval_result.metrics["initial_loss"]),
            "final_loss": float(eval_result.metrics["final_loss"]),
            "loss_reduction_fraction": float(eval_result.metrics["loss_reduction_fraction"]),
            "initial_keypoint_loss": float(eval_result.metrics["initial_keypoint_loss"]),
            "final_keypoint_loss": float(eval_result.metrics["final_keypoint_loss"]),
            "keypoint_loss_reduction": float(eval_result.metrics["keypoint_loss_reduction"]),
            "initial_silhouette_loss": float(eval_result.metrics["initial_silhouette_loss"]),
            "final_silhouette_loss": float(eval_result.metrics["final_silhouette_loss"]),
            "silhouette_loss_reduction": float(
                eval_result.metrics["silhouette_loss_reduction"]
            ),
            "vertex_count": int(eval_result.metrics["vertex_count"]),
            "face_count": int(eval_result.metrics["face_count"]),
            "raw_quality_passed_check_count": int(sum(eval_result.checks.values())),
            "raw_quality_total_check_count": int(len(eval_result.checks)),
        },
        "artifacts": {
            "raw_manifest": str(ingest_result.manifest_path),
            "observation_manifest": str(segmentation_result.manifest_path),
            "frame_selection_report": str(frame_selection_result.report_path),
            "texture_frame_manifest": str(frame_selection_result.texture_manifest_path),
            "init_manifest": str(keypoint_result.manifest_path),
            "fit_report": fit_report.output_paths["fit_report"],
            "fit_state": fit_report.output_paths["fit_state"],
            "fit_state_source_format": "safetensors",
            "fit_state_safetensors": fit_report.output_paths["fit_state_safetensors"],
            "overlay_manifest": fit_report.output_paths["overlay_manifest"],
            "evaluation_report": str(eval_result.report_path),
            "validation_report": str(
                project_dir / "outputs" / "quality" / "validation.json"
            ),
            "coverage_report": str(
                project_dir / "outputs" / "quality" / "coverage.json"
            ),
            "texture_manifest": str(
                project_dir / "outputs" / "textures" / "texture_manifest.json"
            ),
            "basecolor": str(project_dir / "outputs" / "textures" / "basecolor.png"),
            "basecolor_confidence": str(
                project_dir / "outputs" / "textures" / "basecolor_confidence.png"
            ),
            "source_view_id": str(project_dir / "outputs" / "textures" / "source_view_id.png"),
            "normal": str(project_dir / "outputs" / "textures" / "normal.png"),
            "roughness": str(project_dir / "outputs" / "textures" / "roughness.png"),
            "ao": str(project_dir / "outputs" / "textures" / "ao.png"),
            "joint_projection": str(
                project_dir / "outputs" / "quality" / "raw_joint_projection.json"
            ),
            "silhouette_residuals": str(
                project_dir / "outputs" / "quality" / "raw_silhouette_residuals.json"
            ),
            "face_turntable": str(
                project_dir / "outputs" / "quality" / "raw_face_turntable.json"
            ),
            "face_turntable_manifest": str(
                project_dir
                / "outputs"
                / "quality"
                / "raw_face_turntable"
                / "manifest.json"
            ),
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
            "face_uv_density": str(
                project_dir / "outputs" / "quality" / "raw_face_uv_density.json"
            ),
            "face_uv_density_overlay": str(
                project_dir / "outputs" / "quality" / "raw_face_uv_density.png"
            ),
            "face_texture_targets": str(
                project_dir / "outputs" / "quality" / "raw_face_texture_targets.json"
            ),
            "face_texture_diagnostics": str(
                project_dir / "outputs" / "quality" / "raw_face_texture_diagnostics.json"
            ),
            "direct_photo_face_candidates": str(
                project_dir
                / "outputs"
                / "quality"
                / "raw_direct_photo_face_candidates.json"
            ),
            "face_texture_bake_comparison": str(
                project_dir
                / "outputs"
                / "quality"
                / "raw_face_texture_bake_comparison.json"
            ),
            "glb_face_feature_visibility": str(
                project_dir
                / "outputs"
                / "quality"
                / "raw_glb_face_feature_visibility.json"
            ),
            "face_texture_crops": str(
                project_dir / "outputs" / "quality" / "raw_face_texture_crops.png"
            ),
            "face_texture_crop_dir": str(
                project_dir / "outputs" / "quality" / "raw_face_texture_crops"
            ),
            "compliance_audit": str(compliance_audit_path),
            "obj": str(export_result.obj_path),
            "ply": str(export_result.ply_path),
            "glb": str(export_result.glb_path),
            "idle_animation": str(export_result.idle_animation_path),
            "skeleton": str(export_result.skeleton_path),
        },
    }
    summary["metrics"].update(_raw_dense_evidence_summary_metrics(eval_result.metrics))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compliance_result = write_raw_objective_compliance_audit(project_dir, force=force)
    return RawPipelineResult(
        project_dir=project_dir,
        success=bool(eval_result.success and compliance_result.success),
        summary_path=summary_path,
        evaluation_report_path=eval_result.report_path,
        compliance_audit_path=compliance_result.report_path,
        fit_report_path=Path(fit_report.output_paths["fit_report"]),
        obj_path=export_result.obj_path,
        ply_path=export_result.ply_path,
        glb_path=export_result.glb_path,
        skeleton_path=export_result.skeleton_path,
    )


def _raw_texture_export_refresh_needed(metrics: dict[str, float | int]) -> bool:
    if float(metrics.get("raw_template_region_v2_texture_target_bake_used", 0.0)) != 1.0:
        return False
    target_size = int(metrics.get("raw_template_region_v2_texture_target_reported_size", 0))
    glb_size = int(metrics.get("raw_glb_texture_size", 0))
    return target_size > 0 and glb_size < target_size


def _normalize_probe_texture_source_sample_keys(
    raw_policy: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    if not raw_policy:
        return {}
    valid_keys = set(TEXTURE_ONLY_SOURCE_PATCH_SAMPLE_KEYS)
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_frame_id, raw_keys in raw_policy.items():
        frame_id = str(raw_frame_id)
        if not frame_id:
            continue
        keys: list[str] = []
        for raw_key in raw_keys:
            key = str(raw_key)
            if key not in valid_keys or key in keys:
                continue
            keys.append(key)
        if keys:
            normalized[frame_id] = tuple(keys)
    return normalized


def _normalize_probe_texture_source_ablation_overrides(
    raw_policy: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    if not raw_policy:
        return {}
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_frame_id, raw_hashes in raw_policy.items():
        frame_id = str(raw_frame_id)
        if not frame_id:
            continue
        hashes = tuple(
            sorted({str(item).lower() for item in raw_hashes if str(item)})
        )
        if hashes:
            normalized[frame_id] = hashes
    return normalized


def _write_baseline_geometry_texture_manifest(
    project_dir: Path,
    *,
    baseline_project_dir: Path,
    texture_source_project_dir: Path,
    rejected_texture_source_hashes: Sequence[str] = (),
    rejected_texture_source_frame_ids: Sequence[str] = (),
    probe_texture_source_frame_ids: Sequence[str] = (),
    probe_texture_source_sample_keys: Mapping[str, Sequence[str]] | None = None,
    probe_texture_source_ablation_overrides: Mapping[str, Sequence[str]] | None = None,
    force: bool,
) -> dict[str, Any]:
    project_dir = Path(project_dir)
    baseline_project_dir = Path(baseline_project_dir)
    texture_source_project_dir = Path(texture_source_project_dir)
    output_path = project_dir / "outputs" / "quality" / "baseline_geometry_texture_bake.json"
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True to overwrite")

    baseline_texture_manifest_path = (
        baseline_project_dir / "observations" / "raw" / "texture_manifest.json"
    )
    source_texture_manifest_path = (
        texture_source_project_dir / "observations" / "raw" / "texture_manifest.json"
    )
    project_texture_manifest_path = project_dir / "observations" / "raw" / "texture_manifest.json"
    project_frame_selection_path = project_dir / "observations" / "raw" / "frame_selection.json"
    project_observation_manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    texture_source_preflight_path = (
        project_dir / "outputs" / "quality" / "texture_source_preflight.json"
    )
    texture_source_admission_audit_path = (
        project_dir / "outputs" / "quality" / "texture_source_admission_audit.json"
    )
    baseline_texture_manifest = _read_json(baseline_texture_manifest_path)
    source_texture_manifest = _read_json(source_texture_manifest_path)
    project_frame_selection = _read_json(project_frame_selection_path)
    baseline_frames = _frame_objects(baseline_texture_manifest_path, baseline_texture_manifest)
    source_frames = _frame_objects(source_texture_manifest_path, source_texture_manifest)
    rejected_hashes = {
        str(item) for item in rejected_texture_source_hashes if str(item)
    }
    rejected_source_frame_ids = {
        str(item) for item in rejected_texture_source_frame_ids if str(item)
    }
    probe_source_frame_ids = {
        str(item) for item in probe_texture_source_frame_ids if str(item)
    }
    probe_source_sample_key_policy = _normalize_probe_texture_source_sample_keys(
        probe_texture_source_sample_keys,
    )
    probe_source_ablation_override_policy = (
        _normalize_probe_texture_source_ablation_overrides(
            probe_texture_source_ablation_overrides,
        )
    )
    source_quality_by_frame_id = _source_texture_quality_by_frame_id(
        texture_source_project_dir,
    )

    seen_hashes: dict[str, str] = {}
    texture_frames: list[dict[str, Any]] = []
    duplicate_reports: list[dict[str, Any]] = []
    added_reports: list[dict[str, Any]] = []
    probe_reports: list[dict[str, Any]] = []
    skipped_baseline_texture_reports: list[dict[str, Any]] = []
    skipped_non_probe_reports: list[dict[str, Any]] = []
    rejected_low_detail_reports: list[dict[str, Any]] = []
    rejected_ablation_reports: list[dict[str, Any]] = []
    used_frame_ids: set[str] = set()
    for frame in baseline_frames:
        frame_obj = dict(frame)
        frame_id = str(frame_obj["frame_id"])
        try:
            digest = _texture_frame_image_hash(frame_obj)
        except FileNotFoundError as exc:
            skipped_baseline_texture_reports.append(
                {
                    "frame_id": frame_id,
                    "reason": str(exc),
                }
            )
            continue
        seen_hashes[digest] = frame_id
        used_frame_ids.add(frame_id)
        texture_frames.append(frame_obj)
    baseline_texture_hash_frame_count = len(seen_hashes)

    source_preflight = _texture_source_preflight_report(
        source_frames,
        texture_source_project_dir=texture_source_project_dir,
        source_quality_by_frame_id=source_quality_by_frame_id,
        baseline_hashes=seen_hashes,
    )
    source_preflight["report_path"] = str(texture_source_preflight_path)
    texture_source_preflight_path.parent.mkdir(parents=True, exist_ok=True)
    texture_source_preflight_path.write_text(
        json.dumps(source_preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    texture_source_admission_audit = _texture_source_admission_audit_report(
        source_preflight,
        rejected_texture_source_hashes=rejected_hashes,
        rejected_texture_source_frame_ids=rejected_source_frame_ids,
    )
    texture_source_admission_audit["report_path"] = str(
        texture_source_admission_audit_path
    )
    texture_source_admission_audit_path.write_text(
        json.dumps(texture_source_admission_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_frames_by_id = {str(frame["frame_id"]): frame for frame in source_frames}
    missing_probe_source_frame_ids = sorted(
        probe_source_frame_ids - set(source_frames_by_id)
    )

    for decision in source_preflight["ranked_frames"]:
        digest = str(decision["sha256"])
        original_frame_id = str(decision["source_frame_id"])
        frame = source_frames_by_id[original_frame_id]
        is_probe_frame = original_frame_id in probe_source_frame_ids
        if probe_source_frame_ids and not is_probe_frame:
            skipped_non_probe_reports.append(
                {
                    "source_frame_id": original_frame_id,
                    "sha256": digest,
                    "media_type": str(frame.get("media_type", "photo")),
                    "preflight_rank": int(decision["rank"]),
                    "preflight_score": float(decision["rank_score"]),
                    "reason": "not_requested_probe_frame",
                }
            )
            continue
        if digest in seen_hashes:
            duplicate_reports.append(
                {
                    "source_frame_id": original_frame_id,
                    "matching_baseline_frame_id": seen_hashes[digest],
                    "sha256": digest,
                    "preflight_rank": int(decision["rank"]),
                }
            )
            continue
        quality_gate = dict(decision["quality_gate"])
        matched_rejected_sha256 = digest in rejected_hashes
        matched_rejected_source_frame_id = original_frame_id in rejected_source_frame_ids
        probe_override_hashes = probe_source_ablation_override_policy.get(
            original_frame_id,
            (),
        )
        probe_bypassed_rejected_sha256 = bool(
            is_probe_frame
            and matched_rejected_sha256
            and digest.lower() in probe_override_hashes
        )
        probe_bypassed_rejected_source_frame_id = bool(
            is_probe_frame
            and matched_rejected_source_frame_id
            and (not matched_rejected_sha256 or probe_bypassed_rejected_sha256)
        )
        source_patch_blocked_by_ablation = bool(
            (matched_rejected_sha256 and not probe_bypassed_rejected_sha256)
            or (
                matched_rejected_source_frame_id
                and not probe_bypassed_rejected_source_frame_id
            )
        )
        accepted_for_bake = bool(decision["accepted_for_bake"])
        severe_reject_reasons = tuple(
            str(reason)
            for reason in quality_gate.get("severe_reject_reasons", [])
            if str(reason)
        )
        probe_forced_low_detail = bool(
            is_probe_frame and not accepted_for_bake and not severe_reject_reasons
        )
        if not accepted_for_bake and not probe_forced_low_detail:
            rejected_low_detail_reports.append(
                {
                    "source_frame_id": original_frame_id,
                    "sha256": digest,
                    "media_type": str(frame.get("media_type", "photo")),
                    "preflight_rank": int(decision["rank"]),
                    "preflight_score": float(decision["rank_score"]),
                    "quality_gate": quality_gate,
                }
            )
            continue
        face_region_protection = _supplemental_texture_face_region_protection(
            quality_gate,
            probe_forced_low_detail=probe_forced_low_detail,
            probe_bypassed_rejected_source_frame_id=(
                probe_bypassed_rejected_source_frame_id
            ),
            probe_bypassed_rejected_sha256=probe_bypassed_rejected_sha256,
            source_patch_blocked_by_ablation=source_patch_blocked_by_ablation,
        )
        new_frame_id = _supplemental_texture_frame_id(
            texture_source_project_dir.name,
            original_frame_id,
            used_frame_ids,
        )
        used_frame_ids.add(new_frame_id)
        frame_obj = dict(frame)
        frame_obj["frame_id"] = new_frame_id
        frame_obj["texture_only_source_project_id"] = texture_source_project_dir.name
        frame_obj["texture_only_source_frame_id"] = original_frame_id
        frame_obj["texture_only_source_index"] = int(decision["source_index"])
        frame_obj["texture_only_source_preflight_rank"] = int(decision["rank"])
        frame_obj["texture_only_source_preflight_score"] = float(decision["rank_score"])
        frame_obj["texture_only_supplemental"] = True
        frame_obj["texture_only_source_probe"] = bool(is_probe_frame)
        frame_obj["texture_only_source_probe_forced_low_detail"] = bool(
            probe_forced_low_detail
        )
        frame_obj["texture_only_source_probe_bypassed_rejected_source_frame_id"] = bool(
            probe_bypassed_rejected_source_frame_id
        )
        frame_obj["texture_only_source_probe_bypassed_rejected_sha256"] = bool(
            probe_bypassed_rejected_sha256
        )
        frame_obj["texture_only_source_probe_ablation_override_hash_match"] = bool(
            probe_bypassed_rejected_sha256
        )
        frame_obj["texture_only_source_patch_blocked_by_ablation"] = bool(
            source_patch_blocked_by_ablation
        )
        frame_obj["texture_only_source_patch_blocked_matched_sha256"] = bool(
            matched_rejected_sha256
        )
        frame_obj["texture_only_source_patch_blocked_matched_source_frame_id"] = bool(
            matched_rejected_source_frame_id
        )
        frame_obj["texture_only_image_sha256"] = digest
        frame_obj["texture_only_source_quality_gate"] = quality_gate
        frame_obj["texture_only_face_region_projection_protection"] = dict(
            face_region_protection
        )
        frame_obj["texture_only_face_region_projection_protected"] = bool(
            face_region_protection["broad_projection_excluded"]
        )
        frame_obj["texture_only_protected_face_region_names"] = list(
            face_region_protection["protected_region_names"]
        )
        frame_obj["texture_only_source_patch_allowed"] = bool(
            face_region_protection["source_patch_allowed"]
        )
        allowed_source_patch_sample_keys = probe_source_sample_key_policy.get(
            original_frame_id,
            (),
        )
        if allowed_source_patch_sample_keys:
            frame_obj["texture_only_source_patch_allowed_sample_keys"] = list(
                allowed_source_patch_sample_keys
            )
        _absolutize_texture_frame_paths(frame_obj, texture_source_project_dir)
        seen_hashes[digest] = new_frame_id
        texture_frames.append(frame_obj)
        added_report = {
            "frame_id": new_frame_id,
            "source_frame_id": original_frame_id,
            "sha256": digest,
            "media_type": str(frame_obj.get("media_type", "photo")),
            "preflight_rank": int(decision["rank"]),
            "preflight_score": float(decision["rank_score"]),
            "quality_gate": quality_gate,
            "probe": bool(is_probe_frame),
            "probe_forced_low_detail": bool(probe_forced_low_detail),
            "probe_bypassed_rejected_source_frame_id": bool(
                probe_bypassed_rejected_source_frame_id
            ),
            "probe_bypassed_rejected_sha256": bool(
                probe_bypassed_rejected_sha256
            ),
            "probe_ablation_override_hash_match": bool(
                probe_bypassed_rejected_sha256
            ),
            "source_patch_blocked_by_ablation": bool(
                source_patch_blocked_by_ablation
            ),
            "source_patch_blocked_matched_sha256": bool(matched_rejected_sha256),
            "source_patch_blocked_matched_source_frame_id": bool(
                matched_rejected_source_frame_id
            ),
            "face_region_projection_protection": dict(face_region_protection),
            "source_patch_allowed_sample_keys": list(
                allowed_source_patch_sample_keys
            ),
        }
        added_reports.append(added_report)
        if source_patch_blocked_by_ablation:
            rejected_ablation_reports.append(
                {
                    "source_frame_id": original_frame_id,
                    "sha256": digest,
                    "media_type": str(frame.get("media_type", "photo")),
                    "preflight_rank": int(decision["rank"]),
                    "preflight_score": float(decision["rank_score"]),
                    "quality_gate": quality_gate,
                    "reason": "face_source_patch_blocked_by_prior_regression",
                    "matched_rejected_sha256": matched_rejected_sha256,
                    "matched_rejected_source_frame_id": matched_rejected_source_frame_id,
                    "probe": bool(is_probe_frame),
                    "added_as_general_texture_evidence": True,
                }
            )
        if is_probe_frame:
            probe_reports.append(dict(added_report))

    texture_frame_ids = [str(frame["frame_id"]) for frame in texture_frames]
    texture_media_counts = _media_counts_from_frames(texture_frames)
    protected_added_reports = [
        report
        for report in added_reports
        if bool(
            report.get("face_region_projection_protection", {}).get(
                "broad_projection_excluded",
                False,
            )
        )
    ]
    source_patch_allowed_reports = [
        report
        for report in added_reports
        if bool(
            report.get("face_region_projection_protection", {}).get(
                "source_patch_allowed",
                False,
            )
        )
    ]
    texture_manifest = dict(baseline_texture_manifest)
    texture_manifest["project_id"] = project_dir.name
    texture_manifest["source"] = "opencat3d.baseline_geometry_texture_bake"
    texture_manifest["baseline_project_id"] = baseline_project_dir.name
    texture_manifest["texture_source_project_id"] = texture_source_project_dir.name
    texture_manifest["input_frame_count"] = len(texture_frames)
    texture_manifest["texture_frame_count"] = len(texture_frames)
    texture_manifest["texture_frame_ids"] = texture_frame_ids
    texture_manifest["texture_media_counts"] = texture_media_counts
    texture_manifest["frames"] = texture_frames
    texture_manifest["texture_only_bake"] = {
        "applied": True,
        "version": "baseline_geometry_texture_bake_v1",
        "baseline_project_dir": str(baseline_project_dir),
        "texture_source_project_dir": str(texture_source_project_dir),
        "baseline_texture_frame_count": len(baseline_frames),
        "baseline_texture_hash_frame_count": baseline_texture_hash_frame_count,
        "baseline_texture_missing_image_frame_count": len(
            skipped_baseline_texture_reports
        ),
        "baseline_texture_missing_image_frames": skipped_baseline_texture_reports,
        "source_texture_frame_count": len(source_frames),
        "added_unique_texture_frame_count": len(added_reports),
        "skipped_duplicate_texture_frame_count": len(duplicate_reports),
        "rejected_low_detail_texture_frame_count": len(rejected_low_detail_reports),
        "source_patch_blocked_ablation_texture_frame_count": len(
            rejected_ablation_reports
        ),
        "rejected_ablation_texture_frame_count": 0,
        "probe_texture_source_frame_count": len(probe_source_frame_ids),
        "probe_added_texture_frame_count": len(probe_reports),
        "probe_bypassed_rejected_source_frame_id_count": sum(
            1
            for report in probe_reports
            if bool(report.get("probe_bypassed_rejected_source_frame_id", False))
        ),
        "probe_missing_texture_source_frame_count": len(missing_probe_source_frame_ids),
        "probe_texture_source_frame_ids": sorted(probe_source_frame_ids),
        "probe_missing_texture_source_frame_ids": missing_probe_source_frame_ids,
        "probe_texture_source_sample_key_policy_frame_count": len(
            probe_source_sample_key_policy
        ),
        "probe_texture_source_sample_key_policy": {
            frame_id: list(sample_keys)
            for frame_id, sample_keys in sorted(
                probe_source_sample_key_policy.items()
            )
        },
        "probe_texture_source_ablation_override_frame_count": len(
            probe_source_ablation_override_policy
        ),
        "probe_texture_source_ablation_overrides": {
            frame_id: list(hashes)
            for frame_id, hashes in sorted(
                probe_source_ablation_override_policy.items()
            )
        },
        "probe_bypassed_rejected_sha256_count": sum(
            1
            for report in probe_reports
            if bool(report.get("probe_bypassed_rejected_sha256", False))
        ),
        "probe_skipped_non_probe_texture_frame_count": len(skipped_non_probe_reports),
        "probe_skipped_non_probe_texture_frames": skipped_non_probe_reports,
        "source_quality_gate_version": TEXTURE_ONLY_SOURCE_QUALITY_GATE_VERSION,
        "face_region_projection_protection_version": (
            TEXTURE_ONLY_FACE_REGION_PROTECTION_VERSION
        ),
        "face_region_projection_protection_applied_frame_count": len(
            protected_added_reports
        ),
        "face_region_projection_protected_region_names": list(
            TEXTURE_ONLY_PROTECTED_FACE_REGION_NAMES
        ),
        "face_region_projection_protected_region_count": len(
            TEXTURE_ONLY_PROTECTED_FACE_REGION_NAMES
        ),
        "face_region_source_patch_allowed_frame_count": len(
            source_patch_allowed_reports
        ),
        "rejected_texture_source_hashes": sorted(rejected_hashes),
        "rejected_texture_source_frame_ids": sorted(rejected_source_frame_ids),
        "texture_source_preflight_path": str(texture_source_preflight_path),
        "texture_source_admission_audit_path": str(
            texture_source_admission_audit_path
        ),
        "texture_source_admission_audit_passed": bool(
            texture_source_admission_audit["passed"]
        ),
        "texture_source_admission_unique_accepted_frame_count": int(
            texture_source_admission_audit["unique_accepted_frame_count"]
        ),
        "texture_source_admission_protected_face_frame_count": int(
            texture_source_admission_audit["protected_face_source_patch_frame_count"]
        ),
        "texture_source_admission_rejected_ablation_frame_count": int(
            texture_source_admission_audit["rejected_ablation_frame_count"]
        ),
        "texture_source_admission_source_patch_blocked_ablation_frame_count": int(
            texture_source_admission_audit[
                "source_patch_blocked_ablation_frame_count"
            ]
        ),
        "texture_source_preflight_accepted_frame_count": int(
            source_preflight["accepted_frame_count"]
        ),
        "texture_source_preflight_duplicate_frame_count": int(
            source_preflight["duplicate_frame_count"]
        ),
        "texture_source_preflight_rejected_frame_count": int(
            source_preflight["rejected_frame_count"]
        ),
        "texture_source_preflight_top_score": float(source_preflight["top_score"]),
    }
    project_texture_manifest_path.write_text(
        json.dumps(texture_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    project_frame_selection["project_id"] = project_dir.name
    project_frame_selection["texture_frame_count"] = len(texture_frames)
    project_frame_selection["texture_frame_ids"] = texture_frame_ids
    project_frame_selection["texture_media_counts"] = texture_media_counts
    project_frame_selection["texture_only_bake"] = dict(texture_manifest["texture_only_bake"])
    project_frame_selection_path.write_text(
        json.dumps(project_frame_selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    observation_manifest = _read_json(project_observation_manifest_path)
    observation_manifest["project_id"] = project_dir.name
    project_observation_manifest_path.write_text(
        json.dumps(observation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": "1.0",
        "source": "opencat3d.baseline_geometry_texture_bake",
        "version": "baseline_geometry_texture_bake_v1",
        "project_id": project_dir.name,
        "baseline_project_dir": str(baseline_project_dir),
        "texture_source_project_dir": str(texture_source_project_dir),
        "texture_manifest_path": str(project_texture_manifest_path),
        "selected_frame_ids": list(project_frame_selection.get("selected_frame_ids", [])),
        "texture_frame_ids": texture_frame_ids,
        "baseline_texture_frame_count": len(baseline_frames),
        "baseline_texture_hash_frame_count": baseline_texture_hash_frame_count,
        "baseline_texture_missing_image_frame_count": len(
            skipped_baseline_texture_reports
        ),
        "baseline_texture_missing_image_frames": skipped_baseline_texture_reports,
        "source_texture_frame_count": len(source_frames),
        "texture_frame_count": len(texture_frames),
        "added_unique_texture_frame_count": len(added_reports),
        "skipped_duplicate_texture_frame_count": len(duplicate_reports),
        "rejected_low_detail_texture_frame_count": len(rejected_low_detail_reports),
        "source_patch_blocked_ablation_texture_frame_count": len(
            rejected_ablation_reports
        ),
        "rejected_ablation_texture_frame_count": 0,
        "texture_media_counts": texture_media_counts,
        "added_texture_frames": added_reports,
        "probe_added_texture_frames": probe_reports,
        "skipped_duplicate_texture_frames": duplicate_reports,
        "rejected_low_detail_texture_frames": rejected_low_detail_reports,
        "source_patch_blocked_ablation_texture_frames": rejected_ablation_reports,
        "rejected_ablation_texture_frames": [],
        "probe_texture_source_frame_count": len(probe_source_frame_ids),
        "probe_added_texture_frame_count": len(probe_reports),
        "probe_bypassed_rejected_source_frame_id_count": sum(
            1
            for report in probe_reports
            if bool(report.get("probe_bypassed_rejected_source_frame_id", False))
        ),
        "probe_missing_texture_source_frame_count": len(missing_probe_source_frame_ids),
        "probe_texture_source_frame_ids": sorted(probe_source_frame_ids),
        "probe_missing_texture_source_frame_ids": missing_probe_source_frame_ids,
        "probe_texture_source_sample_key_policy_frame_count": len(
            probe_source_sample_key_policy
        ),
        "probe_texture_source_sample_key_policy": {
            frame_id: list(sample_keys)
            for frame_id, sample_keys in sorted(
                probe_source_sample_key_policy.items()
            )
        },
        "probe_texture_source_ablation_override_frame_count": len(
            probe_source_ablation_override_policy
        ),
        "probe_texture_source_ablation_overrides": {
            frame_id: list(hashes)
            for frame_id, hashes in sorted(
                probe_source_ablation_override_policy.items()
            )
        },
        "probe_bypassed_rejected_sha256_count": sum(
            1
            for report in probe_reports
            if bool(report.get("probe_bypassed_rejected_sha256", False))
        ),
        "probe_skipped_non_probe_texture_frame_count": len(skipped_non_probe_reports),
        "probe_skipped_non_probe_texture_frames": skipped_non_probe_reports,
        "source_quality_gate_version": TEXTURE_ONLY_SOURCE_QUALITY_GATE_VERSION,
        "face_region_projection_protection_version": (
            TEXTURE_ONLY_FACE_REGION_PROTECTION_VERSION
        ),
        "face_region_projection_protection_applied_frame_count": len(
            protected_added_reports
        ),
        "face_region_projection_protected_region_names": list(
            TEXTURE_ONLY_PROTECTED_FACE_REGION_NAMES
        ),
        "face_region_projection_protected_region_count": len(
            TEXTURE_ONLY_PROTECTED_FACE_REGION_NAMES
        ),
        "face_region_source_patch_allowed_frame_count": len(
            source_patch_allowed_reports
        ),
        "rejected_texture_source_hashes": sorted(rejected_hashes),
        "rejected_texture_source_frame_ids": sorted(rejected_source_frame_ids),
        "texture_source_preflight_path": str(texture_source_preflight_path),
        "texture_source_admission_audit_path": str(
            texture_source_admission_audit_path
        ),
        "texture_source_admission_audit_passed": bool(
            texture_source_admission_audit["passed"]
        ),
        "texture_source_admission_unique_accepted_frame_count": int(
            texture_source_admission_audit["unique_accepted_frame_count"]
        ),
        "texture_source_admission_protected_face_frame_count": int(
            texture_source_admission_audit["protected_face_source_patch_frame_count"]
        ),
        "texture_source_admission_rejected_ablation_frame_count": int(
            texture_source_admission_audit["rejected_ablation_frame_count"]
        ),
        "texture_source_admission_source_patch_blocked_ablation_frame_count": int(
            texture_source_admission_audit[
                "source_patch_blocked_ablation_frame_count"
            ]
        ),
        "texture_source_preflight_accepted_frame_count": int(
            source_preflight["accepted_frame_count"]
        ),
        "texture_source_preflight_duplicate_frame_count": int(
            source_preflight["duplicate_frame_count"]
        ),
        "texture_source_preflight_rejected_frame_count": int(
            source_preflight["rejected_frame_count"]
        ),
        "texture_source_preflight_top_score": float(source_preflight["top_score"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_texture_source_admission_audit(
    baseline_project_dir: Path,
    texture_source_project_dir: Path,
    output_path: Path,
    *,
    rejected_texture_source_hashes: Sequence[str] = (),
    rejected_texture_source_frame_ids: Sequence[str] = (),
    force: bool = False,
) -> dict[str, Any]:
    """Write a deterministic gate for whether a source can improve face texture."""

    baseline_project_dir = Path(baseline_project_dir)
    texture_source_project_dir = Path(texture_source_project_dir)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True to overwrite")

    baseline_texture_manifest_path = (
        baseline_project_dir / "observations" / "raw" / "texture_manifest.json"
    )
    source_texture_manifest_path = (
        texture_source_project_dir / "observations" / "raw" / "texture_manifest.json"
    )
    baseline_texture_manifest = _read_json(baseline_texture_manifest_path)
    source_texture_manifest = _read_json(source_texture_manifest_path)
    baseline_frames = _frame_objects(baseline_texture_manifest_path, baseline_texture_manifest)
    source_frames = _frame_objects(source_texture_manifest_path, source_texture_manifest)
    baseline_hashes: dict[str, str] = {}
    skipped_baseline_frames: list[dict[str, str]] = []
    for frame in baseline_frames:
        frame_id = str(frame["frame_id"])
        try:
            baseline_hashes[_texture_frame_image_hash(frame)] = frame_id
        except FileNotFoundError as exc:
            skipped_baseline_frames.append(
                {
                    "frame_id": frame_id,
                    "reason": str(exc),
                }
            )
    source_preflight = _texture_source_preflight_report(
        source_frames,
        texture_source_project_dir=texture_source_project_dir,
        source_quality_by_frame_id=_source_texture_quality_by_frame_id(
            texture_source_project_dir
        ),
        baseline_hashes=baseline_hashes,
    )
    report = _texture_source_admission_audit_report(
        source_preflight,
        rejected_texture_source_hashes=(
            str(item) for item in rejected_texture_source_hashes if str(item)
        ),
        rejected_texture_source_frame_ids=(
            str(item) for item in rejected_texture_source_frame_ids if str(item)
        ),
    )
    report["baseline_project_dir"] = str(baseline_project_dir)
    report["texture_source_project_dir"] = str(texture_source_project_dir)
    report["baseline_texture_frame_count"] = len(baseline_frames)
    report["baseline_texture_hash_frame_count"] = len(baseline_hashes)
    report["baseline_texture_missing_image_frame_count"] = len(skipped_baseline_frames)
    report["baseline_texture_missing_image_frames"] = skipped_baseline_frames
    report["source_texture_frame_count"] = len(source_frames)
    report["report_path"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _texture_source_admission_audit_report(
    source_preflight: dict[str, Any],
    *,
    rejected_texture_source_hashes: Sequence[str] | set[str] = (),
    rejected_texture_source_frame_ids: Sequence[str] | set[str] = (),
    min_unique_accepted_frames: int = 1,
    min_protected_face_source_patch_frames: int = 1,
    min_top_score: float = 1.0,
) -> dict[str, Any]:
    rejected_hashes = {str(item) for item in rejected_texture_source_hashes if str(item)}
    rejected_frame_ids = {
        str(item) for item in rejected_texture_source_frame_ids if str(item)
    }
    ranked_frames: list[dict[str, Any]] = []
    duplicate_count = 0
    low_detail_count = 0
    ablation_count = 0
    unique_accepted_count = 0
    protected_projection_count = 0
    protected_source_patch_count = 0
    duplicate_frame_ids: list[str] = []
    rejected_low_detail_frame_ids: list[str] = []
    rejected_ablation_frame_ids: list[str] = []
    source_patch_blocked_ablation_frame_ids: list[str] = []
    unique_accepted_frame_ids: list[str] = []
    protected_projection_frame_ids: list[str] = []
    protected_source_patch_frame_ids: list[str] = []
    for decision in source_preflight.get("ranked_frames", []):
        if not isinstance(decision, dict):
            continue
        frame_id = str(decision.get("source_frame_id", ""))
        digest = str(decision.get("sha256", ""))
        quality_gate = (
            dict(decision.get("quality_gate", {}))
            if isinstance(decision.get("quality_gate"), dict)
            else {}
        )
        duplicate = bool(decision.get("duplicate", False))
        matched_rejected_sha256 = bool(digest and digest in rejected_hashes)
        matched_rejected_source_frame_id = bool(frame_id and frame_id in rejected_frame_ids)
        blocked_by_ablation = matched_rejected_sha256 or matched_rejected_source_frame_id
        accepted_for_bake = bool(decision.get("accepted_for_bake", False))
        unique_accepted = bool(accepted_for_bake and not duplicate)
        protection = (
            _supplemental_texture_face_region_protection(
                quality_gate,
                probe_forced_low_detail=False,
                probe_bypassed_rejected_source_frame_id=False,
                source_patch_blocked_by_ablation=blocked_by_ablation,
            )
            if unique_accepted
            else {}
        )
        protected_projection = bool(
            unique_accepted and protection.get("broad_projection_excluded", False)
        )
        source_patch_allowed = bool(
            unique_accepted and protection.get("source_patch_allowed", False)
        )
        if duplicate:
            duplicate_count += 1
            duplicate_frame_ids.append(frame_id)
            reason = "duplicate_baseline_texture"
        elif not accepted_for_bake:
            low_detail_count += 1
            rejected_low_detail_frame_ids.append(frame_id)
            reason = str(quality_gate.get("reason", "low_face_detail_score"))
        elif blocked_by_ablation:
            ablation_count += 1
            source_patch_blocked_ablation_frame_ids.append(frame_id)
            reason = "unique_texture_face_patch_blocked_by_ablation"
        else:
            unique_accepted_count += 1
            unique_accepted_frame_ids.append(frame_id)
            reason = "unique_protected_face_evidence"
        if unique_accepted and frame_id not in unique_accepted_frame_ids:
            unique_accepted_count += 1
            unique_accepted_frame_ids.append(frame_id)
        if protected_projection:
            protected_projection_count += 1
            protected_projection_frame_ids.append(frame_id)
        if source_patch_allowed:
            protected_source_patch_count += 1
            protected_source_patch_frame_ids.append(frame_id)
        ranked_frames.append(
            {
                "source_frame_id": frame_id,
                "rank": int(decision.get("rank", 0)),
                "media_type": str(decision.get("media_type", "photo")),
                "sha256": digest,
                "rank_score": float(decision.get("rank_score", 0.0)),
                "accepted_for_bake": accepted_for_bake,
                "unique_accepted": unique_accepted,
                "duplicate": duplicate,
                "blocked_by_ablation": blocked_by_ablation,
                "matched_rejected_sha256": matched_rejected_sha256,
                "matched_rejected_source_frame_id": matched_rejected_source_frame_id,
                "source_patch_allowed": source_patch_allowed,
                "protected_projection": protected_projection,
                "admission_reason": reason,
                "quality_gate": quality_gate,
            }
        )

    top_score = float(source_preflight.get("top_score", 0.0))
    checks = {
        "source_frames_present": bool(source_preflight.get("source_frame_count", 0)),
        "top_score_meets_floor": top_score >= min_top_score,
        "has_unique_accepted_texture_frames": (
            unique_accepted_count >= min_unique_accepted_frames
        ),
        "has_unique_protected_face_source_patch_frames": (
            protected_source_patch_count >= min_protected_face_source_patch_frames
        ),
    }
    required_checks = {
        key: value
        for key, value in checks.items()
        if key != "has_unique_protected_face_source_patch_frames"
    }
    rejection_reasons = [key for key, value in required_checks.items() if not value]
    face_patch_rejection_reasons = [
        key
        for key, value in {
            "has_unique_protected_face_source_patch_frames": checks[
                "has_unique_protected_face_source_patch_frames"
            ]
        }.items()
        if not value
    ]
    return {
        "schema_version": "1.0",
        "source": "opencat3d.texture_source_admission_audit",
        "version": TEXTURE_SOURCE_ADMISSION_AUDIT_VERSION,
        "passed": not rejection_reasons,
        "checks": checks,
        "rejection_reasons": rejection_reasons,
        "min_unique_accepted_frames": int(min_unique_accepted_frames),
        "min_protected_face_source_patch_frames": int(
            min_protected_face_source_patch_frames
        ),
        "min_top_score": float(min_top_score),
        "source_frame_count": int(source_preflight.get("source_frame_count", 0)),
        "accepted_preflight_frame_count": int(
            source_preflight.get("accepted_frame_count", 0)
        ),
        "duplicate_frame_count": int(duplicate_count),
        "duplicate_frame_ids": duplicate_frame_ids,
        "rejected_low_detail_frame_count": int(low_detail_count),
        "rejected_low_detail_frame_ids": rejected_low_detail_frame_ids,
        "rejected_ablation_frame_count": 0,
        "rejected_ablation_frame_ids": rejected_ablation_frame_ids,
        "source_patch_blocked_ablation_frame_count": int(ablation_count),
        "source_patch_blocked_ablation_frame_ids": (
            source_patch_blocked_ablation_frame_ids
        ),
        "unique_accepted_frame_count": int(unique_accepted_count),
        "unique_accepted_frame_ids": unique_accepted_frame_ids,
        "protected_projection_frame_count": int(protected_projection_count),
        "protected_projection_frame_ids": protected_projection_frame_ids,
        "protected_face_source_patch_frame_count": int(protected_source_patch_count),
        "protected_face_source_patch_frame_ids": protected_source_patch_frame_ids,
        "protected_region_names": list(TEXTURE_ONLY_PROTECTED_FACE_REGION_NAMES),
        "protected_region_count": len(TEXTURE_ONLY_PROTECTED_FACE_REGION_NAMES),
        "top_score": top_score,
        "face_patch_passed": not face_patch_rejection_reasons,
        "face_patch_rejection_reasons": face_patch_rejection_reasons,
        "ranked_frames": ranked_frames,
    }


def _supplemental_texture_face_region_protection(
    quality_gate: dict[str, Any],
    *,
    probe_forced_low_detail: bool,
    probe_bypassed_rejected_source_frame_id: bool,
    probe_bypassed_rejected_sha256: bool = False,
    source_patch_blocked_by_ablation: bool = False,
) -> dict[str, Any]:
    severe_reasons = quality_gate.get("severe_reject_reasons", [])
    if not isinstance(severe_reasons, list):
        severe_reasons = []
    generic_source_id_reprobe = bool(
        probe_bypassed_rejected_source_frame_id
        and not probe_bypassed_rejected_sha256
    )
    source_patch_allowed = bool(
        quality_gate.get("accepted", False)
        and not probe_forced_low_detail
        and not generic_source_id_reprobe
        and not source_patch_blocked_by_ablation
        and not severe_reasons
    )
    reason = (
        "accepted_face_detail_source"
        if source_patch_allowed
        else "face_source_patch_blocked_by_prior_regression"
        if source_patch_blocked_by_ablation
        else "supplemental_body_or_probe_source"
    )
    return {
        "applied": True,
        "version": TEXTURE_ONLY_FACE_REGION_PROTECTION_VERSION,
        "policy": "preserve_baseline_face_regions",
        "reason": reason,
        "broad_projection_excluded": True,
        "broad_projection_allowed": False,
        "source_patch_allowed": bool(source_patch_allowed),
        "protected_region_names": list(TEXTURE_ONLY_PROTECTED_FACE_REGION_NAMES),
    }


def _write_texture_only_pipeline_summary(
    project_dir: Path,
    *,
    baseline_project_dir: Path,
    texture_source_project_dir: Path,
    texture_manifest_report: dict[str, Any],
    export_result: Any | None,
    eval_result: Any | None,
    promotion_guard_report_path: Path | None,
    promotion_guard_passed: bool | None,
) -> Path:
    project_dir = Path(project_dir)
    baseline_summary_path = (
        Path(baseline_project_dir) / "outputs" / "quality" / "raw_pipeline_summary.json"
    )
    baseline_summary = _read_json(baseline_summary_path)
    baseline_metrics = dict(baseline_summary.get("metrics", {}))
    eval_metrics = (
        {str(key): _json_metric_value(value) for key, value in eval_result.metrics.items()}
        if eval_result is not None
        else {}
    )
    metrics = {**baseline_metrics, **eval_metrics}
    fixed_geometry_alignment_report = _recover_texture_only_fixed_geometry_alignment_metrics(
        project_dir,
        baseline_project_dir=baseline_project_dir,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        eval_metrics=eval_metrics,
    )
    metrics.update(_raw_keypoint_summary_metrics(project_dir))
    metrics.update(
        {
            "raw_texture_frame_count": int(texture_manifest_report["texture_frame_count"]),
            "raw_texture_photo_frame_count": int(
                texture_manifest_report["texture_media_counts"].get("photo", 0)
            ),
            "raw_texture_video_frame_count": int(
                texture_manifest_report["texture_media_counts"].get("video", 0)
            ),
            "raw_baseline_geometry_texture_bake_used": 1.0,
            "raw_texture_only_added_unique_frame_count": int(
                texture_manifest_report["added_unique_texture_frame_count"]
            ),
            "raw_texture_only_skipped_duplicate_frame_count": int(
                texture_manifest_report["skipped_duplicate_texture_frame_count"]
            ),
            "raw_texture_only_rejected_low_detail_frame_count": int(
                texture_manifest_report.get("rejected_low_detail_texture_frame_count", 0)
            ),
            "raw_texture_only_rejected_ablation_frame_count": int(
                texture_manifest_report.get("rejected_ablation_texture_frame_count", 0)
            ),
            "raw_texture_only_probe_frame_count": int(
                texture_manifest_report.get("probe_texture_source_frame_count", 0)
            ),
            "raw_texture_only_probe_added_frame_count": int(
                texture_manifest_report.get("probe_added_texture_frame_count", 0)
            ),
            "raw_texture_only_face_projection_protection_used": float(
                int(
                    texture_manifest_report.get(
                        "face_region_projection_protection_applied_frame_count",
                        0,
                    )
                )
                > 0
            ),
            "raw_texture_only_face_projection_protected_frame_count": int(
                texture_manifest_report.get(
                    "face_region_projection_protection_applied_frame_count",
                    0,
                )
            ),
            "raw_texture_only_face_projection_protected_region_count": int(
                texture_manifest_report.get(
                    "face_region_projection_protected_region_count",
                    0,
                )
            ),
            "raw_texture_only_face_source_patch_allowed_frame_count": int(
                texture_manifest_report.get(
                    "face_region_source_patch_allowed_frame_count",
                    0,
                )
            ),
            "raw_texture_source_preflight_accepted_frame_count": int(
                texture_manifest_report.get(
                    "texture_source_preflight_accepted_frame_count",
                    0,
                )
            ),
            "raw_texture_source_preflight_duplicate_frame_count": int(
                texture_manifest_report.get(
                    "texture_source_preflight_duplicate_frame_count",
                    0,
                )
            ),
            "raw_texture_source_preflight_rejected_frame_count": int(
                texture_manifest_report.get(
                    "texture_source_preflight_rejected_frame_count",
                    0,
                )
            ),
            "raw_texture_source_preflight_top_score": float(
                texture_manifest_report.get("texture_source_preflight_top_score", 0.0)
            ),
            "raw_texture_source_admission_audit_present": float(
                "texture_source_admission_audit_passed" in texture_manifest_report
            ),
            "raw_texture_source_admission_audit_passed": float(
                bool(texture_manifest_report.get("texture_source_admission_audit_passed", False))
            ),
            "raw_texture_source_admission_unique_accepted_frame_count": int(
                texture_manifest_report.get(
                    "texture_source_admission_unique_accepted_frame_count",
                    0,
                )
            ),
            "raw_texture_source_admission_protected_face_frame_count": int(
                texture_manifest_report.get(
                    "texture_source_admission_protected_face_frame_count",
                    0,
                )
            ),
            "raw_texture_source_admission_rejected_ablation_frame_count": int(
                texture_manifest_report.get(
                    "texture_source_admission_rejected_ablation_frame_count",
                    0,
                )
            ),
            "raw_texture_source_admission_source_patch_blocked_ablation_frame_count": int(
                texture_manifest_report.get(
                    "texture_source_admission_source_patch_blocked_ablation_frame_count",
                    0,
                )
            ),
        }
    )
    metrics.update(_texture_only_basecolor_seed_metrics(project_dir))
    if promotion_guard_passed is not None:
        metrics["raw_texture_only_promotion_guard_passed"] = float(
            bool(promotion_guard_passed)
        )
    texture_only_evaluation = _texture_only_evaluation_report(
        metrics,
        eval_result=eval_result,
    )
    metrics.update(
        {
            "raw_texture_only_evaluation_success": float(
                bool(texture_only_evaluation["success"])
            ),
            "raw_texture_only_raw_evaluation_success": float(
                bool(texture_only_evaluation["raw_evaluation_success"])
            ),
            "raw_texture_only_quality_checks_passed": float(
                bool(texture_only_evaluation["quality_checks_passed"])
            ),
        }
    )

    artifacts = dict(baseline_summary.get("artifacts", {}))
    artifacts.update(
        {
            "raw_manifest": str(project_dir / "raw" / "frames.json"),
            "observation_manifest": str(project_dir / "observations" / "raw" / "manifest.json"),
            "frame_selection_report": str(
                project_dir / "observations" / "raw" / "frame_selection.json"
            ),
            "texture_frame_manifest": str(
                project_dir / "observations" / "raw" / "texture_manifest.json"
            ),
            "baseline_geometry_texture_bake": str(
                project_dir / "outputs" / "quality" / "baseline_geometry_texture_bake.json"
            ),
            "texture_source_preflight": str(
                project_dir / "outputs" / "quality" / "texture_source_preflight.json"
            ),
            "texture_source_admission_audit": str(
                project_dir
                / "outputs"
                / "quality"
                / "texture_source_admission_audit.json"
            ),
            "evaluation_report": str(
                eval_result.report_path
                if eval_result is not None
                else project_dir / "outputs" / "quality" / "raw_evaluation.json"
            ),
            "validation_report": str(project_dir / "outputs" / "quality" / "validation.json"),
            "coverage_report": str(project_dir / "outputs" / "quality" / "coverage.json"),
            "texture_manifest": str(project_dir / "outputs" / "textures" / "texture_manifest.json"),
            "basecolor": str(project_dir / "outputs" / "textures" / "basecolor.png"),
            "basecolor_confidence": str(
                project_dir / "outputs" / "textures" / "basecolor_confidence.png"
            ),
            "source_view_id": str(project_dir / "outputs" / "textures" / "source_view_id.png"),
            "normal": str(project_dir / "outputs" / "textures" / "normal.png"),
            "roughness": str(project_dir / "outputs" / "textures" / "roughness.png"),
            "ao": str(project_dir / "outputs" / "textures" / "ao.png"),
            "obj": str(
                export_result.obj_path
                if export_result is not None
                else project_dir / "outputs" / "avatar" / "cat_canonical.obj"
            ),
            "ply": str(
                export_result.ply_path
                if export_result is not None
                else project_dir / "outputs" / "avatar" / "cat_canonical.ply"
            ),
            "glb": str(
                export_result.glb_path
                if export_result is not None
                else project_dir / "outputs" / "avatar" / "cat_avatar.glb"
            ),
            "idle_animation": str(
                export_result.idle_animation_path
                if export_result is not None
                else project_dir / "outputs" / "avatar" / "animations" / "idle.glb"
            ),
            "skeleton": str(
                export_result.skeleton_path
                if export_result is not None
                else project_dir / "outputs" / "avatar" / "skeleton.json"
            ),
        }
    )
    if promotion_guard_report_path is not None:
        artifacts["texture_only_promotion_guard"] = str(promotion_guard_report_path)

    steps = dict(baseline_summary.get("steps", {}))
    steps.update(
        {
            "fit": True,
            "export": True,
            "evaluate": bool(texture_only_evaluation["success"]),
            "raw_evaluate": bool(eval_result.success) if eval_result is not None else True,
            "texture_only_bake": True,
            "texture_only_promotion_guard": (
                bool(promotion_guard_passed)
                if promotion_guard_passed is not None
                else False
            ),
        }
    )
    summary = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "source": "opencat3d.baseline_geometry_texture_bake",
        "success": bool(texture_only_evaluation["success"]),
        "baseline_project_dir": str(baseline_project_dir),
        "texture_source_project_dir": str(texture_source_project_dir),
        "steps": steps,
        "metrics": metrics,
        "artifacts": artifacts,
        "texture_only_evaluation": texture_only_evaluation,
        "texture_only_fixed_geometry_alignment": fixed_geometry_alignment_report,
    }
    summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def _texture_only_evaluation_report(
    metrics: dict[str, Any],
    *,
    eval_result: Any | None,
) -> dict[str, Any]:
    quality_passed = _optional_float_metric(metrics, "raw_quality_passed_check_count")
    quality_total = _optional_float_metric(metrics, "raw_quality_total_check_count")
    quality_checks_passed = (
        quality_passed is not None
        and quality_total is not None
        and quality_total > 0.0
        and quality_passed >= quality_total
    )
    raw_evaluation_success = bool(eval_result.success) if eval_result is not None else True
    return {
        "success": bool(raw_evaluation_success or quality_checks_passed),
        "raw_evaluation_success": raw_evaluation_success,
        "quality_checks_passed": bool(quality_checks_passed),
        "raw_quality_passed_check_count": quality_passed,
        "raw_quality_total_check_count": quality_total,
        "policy": "raw_success_or_preserved_quality_checks",
    }


def _optional_float_metric(metrics: dict[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    try:
        value = float(metrics[key])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _recover_texture_only_fixed_geometry_alignment_metrics(
    project_dir: Path,
    *,
    baseline_project_dir: Path,
    metrics: dict[str, Any],
    baseline_metrics: Mapping[str, Any],
    eval_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_selected_ids = _texture_only_selected_frame_ids(baseline_project_dir)
    candidate_selected_ids = _texture_only_selected_frame_ids(project_dir)
    selected_ids_preserved = bool(baseline_selected_ids) and (
        baseline_selected_ids == candidate_selected_ids
    )
    report: dict[str, Any] = {
        "version": TEXTURE_ONLY_FIXED_GEOMETRY_ALIGNMENT_VERSION,
        "applied": False,
        "selected_frame_ids_preserved": selected_ids_preserved,
        "baseline_selected_frame_ids": list(baseline_selected_ids),
        "candidate_selected_frame_ids": list(candidate_selected_ids),
        "metric_overrides": {},
    }
    metrics["raw_texture_only_fixed_geometry_selected_frame_ids_preserved"] = float(
        selected_ids_preserved
    )
    metrics["raw_texture_only_fixed_geometry_alignment_recovered"] = 0.0
    metrics["raw_texture_only_fixed_geometry_alignment_recovered_metric_count"] = 0
    if not selected_ids_preserved:
        return report

    residual_alignment = _texture_only_silhouette_residual_alignment_metrics(project_dir)
    overlay_candidate = _finite_metric(eval_metrics.get("overlay_mean_silhouette_iou"))
    overlay_baseline = _finite_metric(baseline_metrics.get("overlay_mean_silhouette_iou"))
    overlay_residual = residual_alignment.get("overlay_mean_silhouette_iou")
    if (
        overlay_candidate is not None
        and overlay_baseline is not None
        and overlay_residual is not None
        and overlay_candidate <= 1e-9
        and overlay_baseline > 1e-9
        and overlay_residual >= overlay_baseline - 1e-9
    ):
        metrics["overlay_mean_silhouette_iou"] = float(overlay_residual)
        report["metric_overrides"]["overlay_mean_silhouette_iou"] = {
            "from": float(overlay_candidate),
            "to": float(overlay_residual),
            "baseline": float(overlay_baseline),
            "source": "outputs/quality/raw_silhouette_residuals.json:metrics.mean_silhouette_iou",
        }

    residual_candidate = _finite_metric(
        eval_metrics.get("silhouette_residual_mean_union_error")
    )
    residual_baseline = _finite_metric(
        baseline_metrics.get("silhouette_residual_mean_union_error")
    )
    residual_report_value = residual_alignment.get("silhouette_residual_mean_union_error")
    if (
        residual_baseline is not None
        and residual_report_value is not None
        and residual_report_value <= residual_baseline + 1e-9
        and (
            residual_candidate is None
            or abs(residual_candidate - residual_report_value) > 1e-9
        )
    ):
        metrics["silhouette_residual_mean_union_error"] = float(residual_report_value)
        report["metric_overrides"]["silhouette_residual_mean_union_error"] = {
            "from": None if residual_candidate is None else float(residual_candidate),
            "to": float(residual_report_value),
            "baseline": float(residual_baseline),
            "source": (
                "outputs/quality/raw_silhouette_residuals.json:"
                "metrics.mean_union_error_fraction"
            ),
        }

    override_count = len(report["metric_overrides"])
    if override_count:
        report["applied"] = True
        metrics["raw_texture_only_fixed_geometry_alignment_recovered"] = 1.0
        metrics["raw_texture_only_fixed_geometry_alignment_recovered_metric_count"] = (
            override_count
        )
    return report


def _texture_only_selected_frame_ids(project_dir: Path) -> tuple[str, ...]:
    report_path = Path(project_dir) / "observations" / "raw" / "frame_selection.json"
    if not report_path.exists():
        return ()
    report = _read_json(report_path)
    selected_ids = report.get("selected_frame_ids")
    if isinstance(selected_ids, list):
        return tuple(str(frame_id) for frame_id in selected_ids if str(frame_id))
    frames = report.get("frames", [])
    if not isinstance(frames, list):
        return ()
    return tuple(
        str(frame.get("frame_id"))
        for frame in frames
        if isinstance(frame, dict) and frame.get("selected") and str(frame.get("frame_id"))
    )


def _texture_only_silhouette_residual_alignment_metrics(project_dir: Path) -> dict[str, float]:
    report_path = Path(project_dir) / "outputs" / "quality" / "raw_silhouette_residuals.json"
    if not report_path.exists():
        return {}
    report = _read_json(report_path)
    residual_metrics = report.get("metrics", {})
    if not isinstance(residual_metrics, Mapping):
        return {}
    metrics: dict[str, float] = {}
    overlay = _finite_metric(residual_metrics.get("mean_silhouette_iou"))
    if overlay is not None:
        metrics["overlay_mean_silhouette_iou"] = overlay
    union_error = _finite_metric(residual_metrics.get("mean_union_error_fraction"))
    if union_error is not None:
        metrics["silhouette_residual_mean_union_error"] = union_error
    return metrics


def _finite_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(metric):
        return None
    return metric


def _texture_only_basecolor_seed_metrics(project_dir: Path) -> dict[str, float | int]:
    texture_manifest_path = project_dir / "outputs" / "textures" / "texture_manifest.json"
    if not texture_manifest_path.exists():
        return {}
    manifest = _read_json(texture_manifest_path)
    basecolor = manifest.get("basecolor", {})
    if not isinstance(basecolor, dict):
        return {}
    seed = basecolor.get("baked_basecolor_seed", {})
    if not isinstance(seed, dict):
        return {}
    return {
        "raw_texture_only_baked_basecolor_seed_used": float(
            bool(seed.get("applied", False))
        ),
        "raw_texture_only_baked_basecolor_seed_filled_texel_count": int(
            seed.get("filled_texel_count", 0)
        ),
        "raw_texture_only_baked_basecolor_seed_observed_texel_count": int(
            seed.get("seed_observed_texel_count", 0)
        ),
        "raw_texture_only_baked_basecolor_seed_candidate_unknown_texel_count": int(
            seed.get("candidate_unknown_texel_count_before_seed", 0)
        ),
    }


def _frame_objects(path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path} contains no frames")
    objects: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError(f"{path} frame entries must be objects")
        frame_obj = dict(frame)
        if not str(frame_obj.get("frame_id", "")):
            raise ValueError(f"{path} frame is missing frame_id")
        objects.append(frame_obj)
    return objects


def _source_texture_quality_by_frame_id(project_dir: Path) -> dict[str, dict[str, Any]]:
    frame_selection_path = Path(project_dir) / "observations" / "raw" / "frame_selection.json"
    if not frame_selection_path.exists():
        return {}
    report = _read_json(frame_selection_path)
    frames = report.get("frames", [])
    if not isinstance(frames, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_id = str(frame.get("frame_id", ""))
        if frame_id:
            by_id[frame_id] = dict(frame)
    return by_id


def _texture_source_preflight_report(
    source_frames: list[dict[str, Any]],
    *,
    texture_source_project_dir: Path,
    source_quality_by_frame_id: dict[str, dict[str, Any]],
    baseline_hashes: dict[str, str],
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for index, frame in enumerate(source_frames):
        frame_id = str(frame["frame_id"])
        digest = _texture_frame_image_hash(frame)
        duplicate_baseline_frame_id = baseline_hashes.get(digest)
        quality_gate = _texture_only_source_quality_gate(
            frame_id,
            source_quality_by_frame_id=source_quality_by_frame_id,
            source_frame=frame,
            texture_source_project_dir=texture_source_project_dir,
        )
        duplicate = duplicate_baseline_frame_id is not None
        accepted = bool(quality_gate["accepted"]) and not duplicate
        rank_score = _texture_source_preflight_rank_score(
            quality_gate,
            duplicate=duplicate,
        )
        decisions.append(
            {
                "source_frame_id": frame_id,
                "source_index": index,
                "media_type": str(frame.get("media_type", "photo")),
                "sha256": digest,
                "duplicate": duplicate,
                "duplicate_baseline_frame_id": duplicate_baseline_frame_id,
                "accepted_for_bake": accepted,
                "rank_score": rank_score,
                "quality_gate": quality_gate,
            }
        )
    ranked = sorted(
        decisions,
        key=lambda item: (
            not bool(item["accepted_for_bake"]),
            bool(item["duplicate"]),
            -float(item["rank_score"]),
            int(item["source_index"]),
            str(item["source_frame_id"]),
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    accepted_count = sum(1 for item in ranked if bool(item["accepted_for_bake"]))
    duplicate_count = sum(1 for item in ranked if bool(item["duplicate"]))
    rejected_count = sum(
        1
        for item in ranked
        if not bool(item["accepted_for_bake"]) and not bool(item["duplicate"])
    )
    top_score = float(ranked[0]["rank_score"]) if ranked else 0.0
    return {
        "schema_version": "1.0",
        "source": "opencat3d.texture_source_preflight",
        "version": TEXTURE_ONLY_SOURCE_QUALITY_GATE_VERSION,
        "texture_source_project_dir": str(texture_source_project_dir),
        "source_frame_count": len(source_frames),
        "accepted_frame_count": int(accepted_count),
        "duplicate_frame_count": int(duplicate_count),
        "rejected_frame_count": int(rejected_count),
        "top_score": top_score,
        "ranked_frames": ranked,
    }


def _texture_source_preflight_rank_score(
    quality_gate: dict[str, Any],
    *,
    duplicate: bool,
) -> float:
    score = _optional_float(quality_gate.get("face_texture_detail_score"))
    base_score = 0.0 if score is None else score
    if bool(quality_gate.get("texture_detail_candidate", False)):
        base_score += 0.25
    if bool(quality_gate.get("accepted", False)):
        base_score += 1.0
    if duplicate:
        base_score -= 2.0
    return round(float(base_score), 9)


def _texture_only_source_quality_gate(
    frame_id: str,
    *,
    source_quality_by_frame_id: dict[str, dict[str, Any]],
    source_frame: dict[str, Any] | None = None,
    texture_source_project_dir: Path | None = None,
) -> dict[str, Any]:
    quality = source_quality_by_frame_id.get(frame_id)
    if quality is None:
        face_alignment = _texture_source_face_alignment_gate(
            source_frame,
            texture_source_project_dir=texture_source_project_dir,
        )
        if face_alignment is not None and not bool(face_alignment["accepted"]):
            return {
                "accepted": False,
                "reason": "face_alignment_reject",
                "version": TEXTURE_ONLY_SOURCE_QUALITY_GATE_VERSION,
                "face_alignment": face_alignment,
                "severe_reject_reasons": ["face_alignment_reject"],
            }
        return {
            "accepted": True,
            "reason": "missing_source_quality_report",
            "version": TEXTURE_ONLY_SOURCE_QUALITY_GATE_VERSION,
            "face_alignment": face_alignment,
        }

    score = _optional_float(quality.get("face_texture_detail_score"))
    texture_detail_candidate = bool(quality.get("texture_detail_candidate", False))
    raw_reasons = quality.get("reject_reasons", [])
    reject_reasons = (
        [str(reason) for reason in raw_reasons]
        if isinstance(raw_reasons, list)
        else []
    )
    severe_reasons = sorted(
        reason for reason in reject_reasons if reason in TEXTURE_ONLY_SEVERE_REJECT_REASONS
    )
    face_alignment = _texture_source_face_alignment_gate(
        source_frame,
        texture_source_project_dir=texture_source_project_dir,
    )
    face_alignment_accepted = (
        face_alignment is None or bool(face_alignment.get("accepted", False))
    )
    anchor_patch_gate = _observed_face_anchor_source_patch_gate(
        face_alignment,
        score=score,
        reject_reasons=reject_reasons,
    )
    anchor_patch_candidate = bool(anchor_patch_gate.get("accepted", False))
    if anchor_patch_candidate:
        severe_reasons = [
            reason for reason in severe_reasons if reason != "border_partial_body"
        ]
    effective_face_alignment_accepted = bool(
        face_alignment_accepted or anchor_patch_candidate
    )
    if (
        face_alignment is not None
        and not face_alignment_accepted
        and not anchor_patch_candidate
    ):
        severe_reasons = sorted({*severe_reasons, "face_alignment_reject"})
    score_passed = (
        score is not None
        and score >= TEXTURE_ONLY_FACE_DETAIL_SCORE_FLOOR
        and not severe_reasons
        and effective_face_alignment_accepted
    )
    accepted = bool(
        (texture_detail_candidate or score_passed or anchor_patch_candidate)
        and effective_face_alignment_accepted
        and not severe_reasons
    )
    reason = (
        "face_alignment_reject"
        if not effective_face_alignment_accepted
        else
        "observed_face_anchor_source_patch_candidate"
        if anchor_patch_candidate
        else
        "severe_quality_reject"
        if severe_reasons
        else
        "texture_detail_candidate"
        if texture_detail_candidate
        else "face_detail_score_passed"
        if score_passed
        else "low_face_detail_score"
    )
    return {
        "accepted": accepted,
        "reason": reason,
        "version": TEXTURE_ONLY_SOURCE_QUALITY_GATE_VERSION,
        "face_texture_detail_score": score,
        "face_texture_detail_score_floor": TEXTURE_ONLY_FACE_DETAIL_SCORE_FLOOR,
        "texture_detail_candidate": texture_detail_candidate,
        "observed_face_anchor_source_patch_candidate": anchor_patch_candidate,
        "observed_face_anchor_source_patch_gate": anchor_patch_gate,
        "reject_reasons": reject_reasons,
        "severe_reject_reasons": severe_reasons,
        "face_alignment": face_alignment,
    }


def _observed_face_anchor_source_patch_gate(
    face_alignment: dict[str, Any] | None,
    *,
    score: float | None,
    reject_reasons: list[str],
) -> dict[str, Any]:
    hard_rejects = sorted(
        {
            reason
            for reason in reject_reasons
            if reason
            in {
                "mask_too_small",
                "bbox_too_small",
                "full_frame_bbox",
                "sparse_geometry_mask",
            }
        }
    )
    if not isinstance(face_alignment, dict):
        return {
            "accepted": False,
            "reason": "missing_face_alignment",
            "hard_reject_reasons": hard_rejects,
        }
    points_raw = face_alignment.get("points", {})
    points = points_raw if isinstance(points_raw, dict) else {}
    confidence_values = [
        float(point.get("confidence", 0.0))
        for point in points.values()
        if isinstance(point, dict)
    ]
    mean_confidence = float(
        face_alignment.get(
            "mean_confidence",
            (
                sum(confidence_values) / max(len(confidence_values), 1)
                if confidence_values
                else 0.0
            ),
        )
    )
    available = bool(face_alignment.get("available", False))
    aligned = bool(face_alignment.get("accepted", False))
    has_left_eye = "left_eye_center" in points
    has_right_eye = "right_eye_center" in points
    eye_count = int(has_left_eye) + int(has_right_eye)
    has_eye_pair = eye_count == 2
    has_muzzle_axis = "nose_tip" in points or "muzzle_center" in points
    ear_count = int("left_ear_inner" in points) + int("right_ear_inner" in points)
    score_value = 0.0 if score is None else float(score)
    aligned_closeup = bool(
        aligned
        and has_eye_pair
        and has_muzzle_axis
        and mean_confidence >= 0.18
        and len(points) >= 4
        and score_value >= 0.08
    )
    partial_closeup = bool(
        not aligned
        and str(face_alignment.get("reason")) == "missing_required_face_anchors"
        and eye_count >= 1
        and has_muzzle_axis
        and ear_count >= 1
        and len(points) >= 4
        and mean_confidence >= 0.18
        and score_value >= 0.20
    )
    accepted = bool(available and not hard_rejects and (aligned_closeup or partial_closeup))
    reason = (
        "aligned_observed_face_anchors"
        if accepted and aligned_closeup
        else "partial_observed_face_anchors"
        if accepted and partial_closeup
        else "hard_reject_reasons"
        if hard_rejects
        else "insufficient_observed_face_anchors"
    )
    return {
        "accepted": accepted,
        "reason": reason,
        "available": available,
        "aligned": aligned,
        "aligned_closeup": aligned_closeup,
        "partial_closeup": partial_closeup,
        "mean_confidence": mean_confidence,
        "face_texture_detail_score": score,
        "eye_count": eye_count,
        "has_eye_pair": has_eye_pair,
        "has_muzzle_axis": has_muzzle_axis,
        "ear_inner_count": ear_count,
        "target_count": len(points),
        "allowed_border_partial_body": bool(
            accepted and "border_partial_body" in reject_reasons
        ),
        "hard_reject_reasons": hard_rejects,
    }


def _texture_source_face_alignment_gate(
    source_frame: dict[str, Any] | None,
    *,
    texture_source_project_dir: Path | None,
) -> dict[str, Any] | None:
    if not isinstance(source_frame, dict):
        return None
    anchor_path_raw = source_frame.get("feature_anchors_xyc_path")
    if not anchor_path_raw:
        return None
    anchor_path = Path(str(anchor_path_raw))
    if not anchor_path.is_absolute() and texture_source_project_dir is not None:
        anchor_path = Path(texture_source_project_dir) / anchor_path
    if not anchor_path.is_file():
        return {
            "available": True,
            "accepted": False,
            "reason": "missing_face_anchor_manifest",
            "version": TEXTURE_ONLY_FACE_ALIGNMENT_GATE_VERSION,
            "manifest_path": str(anchor_path),
        }
    try:
        manifest = _read_json(anchor_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {
            "available": True,
            "accepted": False,
            "reason": "invalid_face_anchor_manifest",
            "version": TEXTURE_ONLY_FACE_ALIGNMENT_GATE_VERSION,
            "manifest_path": str(anchor_path),
        }
    targets = manifest.get("targets", {})
    if not isinstance(targets, dict):
        targets = {}
    bbox = _source_frame_bbox(source_frame)
    if bbox is None:
        return {
            "available": True,
            "accepted": False,
            "reason": "missing_source_bbox",
            "version": TEXTURE_ONLY_FACE_ALIGNMENT_GATE_VERSION,
            "manifest_path": str(anchor_path),
            "target_count": len(targets),
        }
    x0, y0, x1, y1 = bbox
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    points = {
        name: _normalised_anchor_point(target, bbox=bbox)
        for name, target in targets.items()
        if isinstance(target, dict)
    }
    required = {"left_eye_center", "right_eye_center"}
    has_eye_pair = required.issubset(points)
    has_muzzle_axis = "nose_tip" in points or "muzzle_center" in points
    if not has_eye_pair or not has_muzzle_axis:
        return _face_alignment_report(
            accepted=False,
            reason="missing_required_face_anchors",
            anchor_path=anchor_path,
            points=points,
            extra={
                "target_count": len(targets),
                "has_eye_pair": bool(has_eye_pair),
                "has_muzzle_axis": bool(has_muzzle_axis),
            },
        )

    left_eye = points["left_eye_center"]
    right_eye = points["right_eye_center"]
    eye_span = right_eye["x"] - left_eye["x"]
    eye_mid_x = 0.5 * (left_eye["x"] + right_eye["x"])
    eye_mid_y = 0.5 * (left_eye["y"] + right_eye["y"])
    eye_level_delta = abs(left_eye["y"] - right_eye["y"])
    nose = points.get("nose_tip")
    muzzle = points.get("muzzle_center")
    axis = nose or muzzle
    axis_dx = abs(float(axis["x"]) - eye_mid_x) if axis is not None else 999.0
    axis_dy = float(axis["y"]) - eye_mid_y if axis is not None else -999.0
    mean_confidence = float(
        sum(point["confidence"] for point in points.values()) / max(len(points), 1)
    )
    checks = {
        "eye_order": eye_span > 0.0,
        "eye_span_in_range": 0.12 <= eye_span <= 0.72,
        "eye_level_consistent": eye_level_delta <= 0.18,
        "eyes_in_upper_face": 0.04 <= eye_mid_y <= 0.58,
        "muzzle_axis_centered": axis_dx <= max(0.28, 0.90 * max(eye_span, 0.01)),
        "muzzle_axis_below_eyes": 0.04 <= axis_dy <= 0.62,
        "mean_confidence": mean_confidence >= 0.18,
    }
    if nose is not None and muzzle is not None:
        checks["nose_above_muzzle"] = float(nose["y"]) <= float(muzzle["y"]) + 0.10
    accepted = all(bool(value) for value in checks.values())
    reason = "aligned_face_anchors" if accepted else "misaligned_face_anchors"
    return _face_alignment_report(
        accepted=accepted,
        reason=reason,
        anchor_path=anchor_path,
        points=points,
        extra={
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "eye_span": eye_span,
            "eye_level_delta": eye_level_delta,
            "muzzle_axis_dx": axis_dx,
            "muzzle_axis_dy": axis_dy,
            "mean_confidence": mean_confidence,
            "checks": checks,
            "target_count": len(targets),
        },
    )


def _face_alignment_report(
    *,
    accepted: bool,
    reason: str,
    anchor_path: Path,
    points: dict[str, dict[str, float]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "available": True,
        "accepted": bool(accepted),
        "reason": reason,
        "version": TEXTURE_ONLY_FACE_ALIGNMENT_GATE_VERSION,
        "manifest_path": str(anchor_path),
        "points": points,
        **extra,
    }


def _normalised_anchor_point(
    target: dict[str, Any],
    *,
    bbox: tuple[float, float, float, float],
) -> dict[str, float]:
    x0, y0, x1, y1 = bbox
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    return {
        "x": round((float(target.get("x", 0.0)) - x0) / bbox_width, 9),
        "y": round((float(target.get("y", 0.0)) - y0) / bbox_height, 9),
        "confidence": round(float(target.get("confidence", 0.0)), 9),
    }


def _source_frame_bbox(frame: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox_raw = frame.get("bbox_xyxy")
    if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        return None
    bbox = tuple(float(value) for value in bbox_raw)
    if not all(math.isfinite(value) for value in bbox):
        return None
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _texture_frame_image_hash(frame: dict[str, Any]) -> str:
    image_path = _texture_frame_image_path(frame)
    return sha256_file(image_path)


def _texture_frame_image_path(frame: dict[str, Any]) -> Path:
    observation_path_raw = frame.get("observation_path")
    if observation_path_raw:
        observation_path = Path(str(observation_path_raw))
        if observation_path.is_file():
            observation = _read_json(observation_path)
            image_path_raw = observation.get("image_srgb_path")
            if image_path_raw:
                image_path = Path(str(image_path_raw))
                if image_path.is_file():
                    return image_path
    image_path_raw = frame.get("image_srgb_path")
    if image_path_raw:
        image_path = Path(str(image_path_raw))
        if image_path.is_file():
            return image_path
    raise FileNotFoundError(f"texture frame {frame.get('frame_id')} is missing image_srgb")


def _supplemental_texture_frame_id(
    source_project_id: str,
    source_frame_id: str,
    used_frame_ids: set[str],
) -> str:
    sanitized_project = "".join(
        char if char.isalnum() else "_" for char in source_project_id
    ).strip("_")
    sanitized_frame = "".join(
        char if char.isalnum() else "_" for char in source_frame_id
    ).strip("_")
    base = f"texture_{sanitized_project}_{sanitized_frame}"
    candidate = base
    suffix = 1
    while candidate in used_frame_ids:
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _absolutize_texture_frame_paths(
    frame: dict[str, Any],
    source_project_dir: Path,
) -> None:
    for key in (
        "observation_path",
        "image_srgb_path",
        "image_linear_path",
    ):
        raw = frame.get(key)
        if raw is None:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = source_project_dir / path
        frame[key] = str(path.resolve())
    for key in ("hard_mask", "soft_mask", "keypoints_xyc"):
        raw = frame.get(key)
        if not isinstance(raw, dict) or "path" not in raw:
            continue
        path = Path(str(raw["path"]))
        if not path.is_absolute():
            path = source_project_dir / path
        raw["path"] = str(path.resolve())


def _media_counts_from_frames(frames: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        media_type = str(frame.get("media_type", "photo"))
        counts[media_type] = counts.get(media_type, 0) + 1
    return counts


def _raw_keypoint_summary_metrics(project_dir: Path) -> dict[str, float]:
    manifest_path = Path(project_dir) / "init" / "raw" / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = _read_json(manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return {}
    confidences: list[float] = []
    in_bbox_fractions: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            confidences.append(float(frame.get("mean_confidence", 0.0)))
            in_bbox_fractions.append(float(frame.get("in_bbox_fraction", 0.0)))
        except (TypeError, ValueError):
            continue
    if not confidences or not in_bbox_fractions:
        return {}
    return {
        "keypoint_mean_confidence": float(np.mean(confidences)),
        "keypoint_mean_in_bbox_fraction": float(np.mean(in_bbox_fractions)),
    }


def _raw_frame_selection_detail_summary_metrics(
    metrics: dict[str, float],
) -> dict[str, float]:
    return {
        f"raw_{key}": float(value)
        for key, value in sorted(metrics.items())
        if key.startswith(("geometry_", "texture_"))
    }


def _json_metric_value(value: Any) -> float | int:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, int):
        return value
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _raw_dense_evidence_summary_metrics(
    metrics: dict[str, float | int],
) -> dict[str, float | int]:
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith("raw_dense_evidence_")
    }
