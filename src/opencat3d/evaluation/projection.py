"""Fitted skeleton joint projection quality reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opencat3d.camera.conventions import project_world_points
from opencat3d.export.skeleton import validate_skeleton_metadata_file
from opencat3d.fitting.losses import bbox_diagonal, keypoint_reprojection_loss
from opencat3d.fitting.optimizer import FitProblem, load_fit_problem, load_fit_state

JOINT_PROJECTION_THRESHOLDS: dict[str, float | int] = {
    "min_frame_count": 2,
    "min_visible_joint_count": 1,
    "max_skeleton_position_delta": 1e-8,
    "synthetic_max_mean_keypoint_reprojection_loss": 0.01,
    "synthetic_max_frame_keypoint_reprojection_loss": 0.02,
    "raw_max_mean_keypoint_reprojection_loss": 0.35,
    "raw_max_frame_keypoint_reprojection_loss": 0.50,
}


@dataclass(frozen=True)
class JointProjectionFrameMetric:
    frame_id: str
    source_keypoint_count: int
    evaluated_keypoint_count: int
    visible_joint_count: int
    visible_joint_fraction: float
    mean_pixel_error: float
    max_pixel_error: float
    mean_normalized_error: float
    keypoint_reprojection_loss: float
    robust_keypoint_reprojection_loss: float
    positive_depth_fraction: float


@dataclass(frozen=True)
class JointProjectionEvaluationResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]
    frames: tuple[JointProjectionFrameMetric, ...]


def evaluate_joint_projection_report(
    project_dir: Path,
    *,
    mode: str = "synthetic",
    force: bool = False,
) -> JointProjectionEvaluationResult:
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported joint projection mode: {mode}")

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / f"{mode}_joint_projection.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    problem = load_fit_problem(project_dir, mode=mode)
    skeleton_path = project_dir / "outputs" / "avatar" / "skeleton.json"
    fit_state = load_fit_state(
        project_dir,
        mode=mode,
        expected_frame_count=len(problem.observations),
        expected_joint_count=int(problem.template.joints_rest.shape[0]),
    )
    beta = fit_state.beta
    root_translations = fit_state.root_translations
    skeleton_report = validate_skeleton_metadata_file(
        skeleton_path,
        expected_mode=mode,
        expected_vertex_count=int(problem.template.vertices_canonical.shape[0]),
        expected_face_count=int(problem.template.faces.shape[0]),
        expected_joint_count=int(problem.template.joints_rest.shape[0]),
    )

    joints = problem.template.joints(beta)
    projected_frames = _evaluate_frames(problem, joints, root_translations)
    skeleton_delta = _skeleton_position_delta(skeleton_path, joints)
    thresholds = dict(JOINT_PROJECTION_THRESHOLDS)
    all_visible_counts = [frame.visible_joint_count for frame in projected_frames]
    frames_with_keypoints = [
        frame for frame in projected_frames if frame.visible_joint_count > 0
    ]
    mean_loss = _mean_metric(frames_with_keypoints, "keypoint_reprojection_loss")
    max_frame_loss = _max_metric(frames_with_keypoints, "keypoint_reprojection_loss")
    mode_mean_key = f"{mode}_max_mean_keypoint_reprojection_loss"
    mode_frame_key = f"{mode}_max_frame_keypoint_reprojection_loss"

    visible_counts = [frame.visible_joint_count for frame in frames_with_keypoints]
    visible_depth_fractions = [
        frame.positive_depth_fraction for frame in frames_with_keypoints
    ]
    sparse_keypoint_frame_count = sum(
        frame.source_keypoint_count != int(joints.shape[0])
        for frame in projected_frames
    )
    missing_keypoint_frame_count = sum(
        frame.visible_joint_count == 0 for frame in projected_frames
    )
    checks = {
        "fit_state_exists": fit_state.path.exists(),
        "fit_state_shapes_valid": root_translations.shape == (len(problem.observations), 3),
        "skeleton_metadata_valid": skeleton_report.success,
        "frame_count_threshold": len(projected_frames) >= int(thresholds["min_frame_count"]),
        "observed_visible_joints": bool(all_visible_counts)
        and bool(visible_counts)
        and min(visible_counts) >= int(thresholds["min_visible_joint_count"]),
        "projected_points_finite": all(
            np.isfinite(
                [
                    frame.mean_pixel_error,
                    frame.max_pixel_error,
                    frame.mean_normalized_error,
                    frame.keypoint_reprojection_loss,
                    frame.robust_keypoint_reprojection_loss,
                ]
            ).all()
            for frame in projected_frames
        ),
        "projected_depths_positive": bool(visible_depth_fractions)
        and min(visible_depth_fractions) >= 1.0,
        "skeleton_matches_fit_state": skeleton_delta
        <= float(thresholds["max_skeleton_position_delta"]),
        "mean_keypoint_reprojection_threshold": mean_loss <= float(thresholds[mode_mean_key]),
        "frame_keypoint_reprojection_threshold": max_frame_loss
        <= float(thresholds[mode_frame_key]),
    }
    metrics: dict[str, float | int] = {
        "frame_count": len(projected_frames),
        "joint_count": int(joints.shape[0]),
        "observed_keypoint_frame_count": len(frames_with_keypoints),
        "sparse_keypoint_frame_count": int(sparse_keypoint_frame_count),
        "missing_keypoint_frame_count": int(missing_keypoint_frame_count),
        "mean_visible_joint_count": _mean_values(all_visible_counts),
        "mean_visible_joint_fraction": _mean_metric(projected_frames, "visible_joint_fraction"),
        "mean_pixel_error": _mean_metric(frames_with_keypoints, "mean_pixel_error"),
        "max_pixel_error": _max_metric(frames_with_keypoints, "max_pixel_error"),
        "mean_normalized_error": _mean_metric(
            frames_with_keypoints,
            "mean_normalized_error",
        ),
        "mean_keypoint_reprojection_loss": mean_loss,
        "max_frame_keypoint_reprojection_loss": max_frame_loss,
        "mean_robust_keypoint_reprojection_loss": _mean_metric(
            frames_with_keypoints,
            "robust_keypoint_reprojection_loss",
        ),
        "skeleton_fit_max_abs_position_delta": skeleton_delta,
    }
    success = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "success": success,
        "checks": checks,
        "thresholds": thresholds,
        "metrics": metrics,
        "frames": [
            {
                "frame_id": frame.frame_id,
                "source_keypoint_count": frame.source_keypoint_count,
                "evaluated_keypoint_count": frame.evaluated_keypoint_count,
                "visible_joint_count": frame.visible_joint_count,
                "visible_joint_fraction": frame.visible_joint_fraction,
                "mean_pixel_error": frame.mean_pixel_error,
                "max_pixel_error": frame.max_pixel_error,
                "mean_normalized_error": frame.mean_normalized_error,
                "keypoint_reprojection_loss": frame.keypoint_reprojection_loss,
                "robust_keypoint_reprojection_loss": frame.robust_keypoint_reprojection_loss,
                "positive_depth_fraction": frame.positive_depth_fraction,
            }
            for frame in projected_frames
        ],
        "artifacts": {
            "fit_state": str(fit_state.path),
            "fit_state_source_format": fit_state.source_format,
            "skeleton": str(skeleton_path),
        },
        "skeleton_checks": skeleton_report.checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return JointProjectionEvaluationResult(
        report_path=report_path,
        success=success,
        checks=checks,
        metrics=metrics,
        frames=tuple(projected_frames),
    )


def _evaluate_frames(
    problem: FitProblem,
    joints: NDArray[np.float64],
    root_translations: NDArray[np.float64],
) -> list[JointProjectionFrameMetric]:
    frame_metrics: list[JointProjectionFrameMetric] = []
    for observation, translation in zip(
        problem.observations,
        root_translations,
        strict=True,
    ):
        moved_joints = joints + translation[None, :]
        projected = project_world_points(moved_joints, observation.camera)
        observed_source = np.asarray(observation.keypoints_xyc, dtype=np.float64)
        projected_eval, observed = _paired_projected_observed(projected, observed_source)
        confidence = np.clip(observed[:, 2], 0.0, 1.0)
        visible = confidence > 0.0
        if not np.any(visible):
            residual = np.zeros((observed.shape[0],), dtype=np.float64)
            visible_count = 0
            mean_pixel_error = 0.0
            max_pixel_error = 0.0
            mean_normalized_error = 0.0
            positive_depth_fraction = 0.0
        else:
            residual = np.linalg.norm(projected_eval[:, :2] - observed[:, :2], axis=1)
            visible_count = int(np.count_nonzero(visible))
            visible_weights = confidence[visible]
            mean_pixel_error = _weighted_mean(residual[visible], visible_weights)
            max_pixel_error = float(np.max(residual[visible]))
            normalized = residual[visible] / bbox_diagonal(observation.bbox_xyxy)
            mean_normalized_error = _weighted_mean(normalized, visible_weights)
            positive_depth_fraction = float(np.mean(projected_eval[visible, 2] > 1e-9))
        frame_metrics.append(
            JointProjectionFrameMetric(
                frame_id=observation.frame_id,
                source_keypoint_count=int(observed_source.shape[0]),
                evaluated_keypoint_count=int(observed.shape[0]),
                visible_joint_count=visible_count,
                visible_joint_fraction=float(visible_count / max(projected.shape[0], 1)),
                mean_pixel_error=float(mean_pixel_error),
                max_pixel_error=float(max_pixel_error),
                mean_normalized_error=float(mean_normalized_error),
                keypoint_reprojection_loss=keypoint_reprojection_loss(
                    projected_eval,
                    observed,
                    observation.bbox_xyxy,
                    robust=False,
                ),
                robust_keypoint_reprojection_loss=keypoint_reprojection_loss(
                    projected_eval,
                    observed,
                    observation.bbox_xyxy,
                    robust=True,
                ),
                positive_depth_fraction=positive_depth_fraction,
            )
        )
    return frame_metrics


def _paired_projected_observed(
    projected_xy_depth: NDArray[np.float64],
    observed_xyc: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    projected = np.asarray(projected_xy_depth, dtype=np.float64)
    observed = np.asarray(observed_xyc, dtype=np.float64)
    if projected.ndim != 2 or projected.shape[1] < 2:
        raise ValueError("projected joints must have shape [J,>=2]")
    if observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("observed keypoints must have shape [N,3]")
    pair_count = min(int(projected.shape[0]), int(observed.shape[0]))
    return projected[:pair_count], observed[:pair_count]


def _skeleton_position_delta(path: Path, fitted_joints: NDArray[np.float64]) -> float:
    if not path.exists():
        return float("inf")
    payload = json.loads(path.read_text(encoding="utf-8"))
    joints = payload.get("joints", [])
    if not isinstance(joints, list) or len(joints) != fitted_joints.shape[0]:
        return float("inf")
    positions: list[list[float]] = []
    for joint in joints:
        if not isinstance(joint, dict):
            return float("inf")
        position = joint.get("position")
        if not isinstance(position, list) or len(position) != 3:
            return float("inf")
        positions.append([float(value) for value in position])
    exported = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(exported).all():
        return float("inf")
    return float(np.max(np.abs(exported - fitted_joints)))


def _weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-12))


def _mean_metric(frames: list[JointProjectionFrameMetric], field_name: str) -> float:
    return _mean_values([float(getattr(frame, field_name)) for frame in frames])


def _max_metric(frames: list[JointProjectionFrameMetric], field_name: str) -> float:
    if not frames:
        return 0.0
    return float(max(float(getattr(frame, field_name)) for frame in frames))


def _mean_values(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0
