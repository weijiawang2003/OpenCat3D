"""Held-out synthetic view consistency evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opencat3d.camera.conventions import project_world_points
from opencat3d.fitting.losses import keypoint_reprojection_loss, soft_iou_loss
from opencat3d.fitting.optimizer import FitProblem, load_fit_state, load_synthetic_fit_problem
from opencat3d.synthetic.renderer import render_silhouette

SYNTHETIC_HELDOUT_THRESHOLDS: dict[str, float | int] = {
    "heldout_fraction": 1.0 / 3.0,
    "min_heldout_silhouette_iou": 0.80,
    "max_heldout_keypoint_reprojection_loss": 0.01,
    "min_heldout_frames": 1,
}


@dataclass(frozen=True)
class HeldoutFrameMetric:
    frame_id: str
    silhouette_iou: float
    keypoint_reprojection_loss: float


@dataclass(frozen=True)
class SyntheticHeldoutEvaluationResult:
    report_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]
    frames: tuple[HeldoutFrameMetric, ...]


def evaluate_synthetic_heldout_views(
    project_dir: Path,
    *,
    heldout_fraction: float = float(SYNTHETIC_HELDOUT_THRESHOLDS["heldout_fraction"]),
    force: bool = False,
) -> SyntheticHeldoutEvaluationResult:
    if not 0.0 < heldout_fraction <= 1.0:
        raise ValueError("heldout_fraction must be in (0,1]")

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / "synthetic_heldout_evaluation.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    problem = load_synthetic_fit_problem(project_dir)
    fit_state = load_fit_state(
        project_dir,
        mode="synthetic",
        expected_frame_count=len(problem.observations),
        expected_joint_count=int(problem.template.joints_rest.shape[0]),
    )
    beta = fit_state.beta
    root_translations = fit_state.root_translations
    frame_indices = _heldout_indices(len(problem.observations), heldout_fraction)
    frame_metrics = _evaluate_frames(problem, beta, root_translations, frame_indices)

    heldout_silhouette_iou = float(np.mean([frame.silhouette_iou for frame in frame_metrics]))
    heldout_keypoint_loss = float(
        np.mean([frame.keypoint_reprojection_loss for frame in frame_metrics])
    )
    thresholds = {
        **SYNTHETIC_HELDOUT_THRESHOLDS,
        "heldout_fraction": float(heldout_fraction),
    }
    checks = {
        "fit_state_exists": fit_state.path.exists(),
        "heldout_frame_count": len(frame_metrics) >= int(thresholds["min_heldout_frames"]),
        "heldout_silhouette_iou_threshold": (
            heldout_silhouette_iou >= float(thresholds["min_heldout_silhouette_iou"])
        ),
        "heldout_keypoint_reprojection_threshold": (
            heldout_keypoint_loss
            <= float(thresholds["max_heldout_keypoint_reprojection_loss"])
        ),
    }
    metrics: dict[str, float | int] = {
        "heldout_fraction": float(heldout_fraction),
        "heldout_frame_count": len(frame_metrics),
        "heldout_mean_silhouette_iou": heldout_silhouette_iou,
        "heldout_mean_keypoint_reprojection_loss": heldout_keypoint_loss,
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "success": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "metrics": metrics,
        "heldout_frame_ids": [frame.frame_id for frame in frame_metrics],
        "frames": [
            {
                "frame_id": frame.frame_id,
                "silhouette_iou": frame.silhouette_iou,
                "keypoint_reprojection_loss": frame.keypoint_reprojection_loss,
            }
            for frame in frame_metrics
        ],
        "artifacts": {
            "fit_state": str(fit_state.path),
            "fit_state_source_format": fit_state.source_format,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SyntheticHeldoutEvaluationResult(
        report_path=report_path,
        success=bool(report["success"]),
        checks=checks,
        metrics=metrics,
        frames=tuple(frame_metrics),
    )


def _heldout_indices(frame_count: int, heldout_fraction: float) -> tuple[int, ...]:
    if frame_count <= 0:
        raise ValueError("at least one frame is required")
    heldout_count = max(1, int(ceil(frame_count * heldout_fraction)))
    start = max(frame_count - heldout_count, 0)
    return tuple(range(start, frame_count))


def _evaluate_frames(
    problem: FitProblem,
    beta: NDArray[np.float64],
    root_translations: NDArray[np.float64],
    frame_indices: tuple[int, ...],
) -> list[HeldoutFrameMetric]:
    vertices = problem.template.shaped_vertices(beta)
    joints = problem.template.joints(beta)
    frame_metrics: list[HeldoutFrameMetric] = []
    for index in frame_indices:
        observation = problem.observations[index]
        translation = root_translations[index]
        moved_vertices = vertices + translation[None, :]
        predicted_mask = render_silhouette(
            moved_vertices,
            problem.template.faces,
            observation.camera,
        )
        silhouette_iou = 1.0 - soft_iou_loss(predicted_mask, observation.hard_mask)
        moved_joints = joints + translation[None, :]
        projected = project_world_points(moved_joints, observation.camera)
        keypoint_loss = keypoint_reprojection_loss(
            projected,
            observation.keypoints_xyc,
            observation.bbox_xyxy,
            robust=False,
        )
        frame_metrics.append(
            HeldoutFrameMetric(
                frame_id=observation.frame_id,
                silhouette_iou=float(silhouette_iou),
                keypoint_reprojection_loss=float(keypoint_loss),
            )
        )
    return frame_metrics
