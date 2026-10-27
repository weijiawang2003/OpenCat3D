"""Per-view fitted silhouette residual map artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from opencat3d.fitting.optimizer import FitProblem, load_fit_problem, load_fit_state
from opencat3d.io.cache import sha256_file
from opencat3d.synthetic.renderer import render_silhouette

SILHOUETTE_RESIDUAL_THRESHOLDS: dict[str, float | int] = {
    "min_frame_count": 2,
    "synthetic_max_mean_union_error_fraction": 0.35,
    "synthetic_max_frame_union_error_fraction": 0.60,
    "raw_max_mean_union_error_fraction": 0.90,
    "raw_max_frame_union_error_fraction": 0.98,
}

RESIDUAL_LEGEND: dict[str, list[int]] = {
    "background_rgb": [16, 16, 16],
    "overlap_rgb": [255, 216, 32],
    "observed_only_rgb": [32, 112, 255],
    "fitted_only_rgb": [255, 72, 32],
}


@dataclass(frozen=True)
class SilhouetteResidualFrameMetric:
    frame_id: str
    residual_path: Path
    silhouette_iou: float
    union_error_fraction: float
    false_negative_fraction: float
    false_positive_fraction: float
    observed_area_fraction: float
    fitted_area_fraction: float


@dataclass(frozen=True)
class SilhouetteResidualEvaluationResult:
    report_path: Path
    manifest_path: Path
    residual_dir: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]
    frames: tuple[SilhouetteResidualFrameMetric, ...]


def evaluate_silhouette_residual_maps(
    project_dir: Path,
    *,
    mode: str = "synthetic",
    force: bool = False,
) -> SilhouetteResidualEvaluationResult:
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported silhouette residual mode: {mode}")

    output_dir = project_dir / "outputs" / "quality"
    residual_dir = output_dir / f"{mode}_silhouette_residuals"
    manifest_path = residual_dir / "manifest.json"
    report_path = output_dir / f"{mode}_silhouette_residuals.json"
    if (report_path.exists() or manifest_path.exists()) and not force:
        raise FileExistsError(
            f"{mode} silhouette residual outputs already exist; pass force=True to overwrite"
        )

    problem = load_fit_problem(project_dir, mode=mode)
    fit_state = load_fit_state(
        project_dir,
        mode=mode,
        expected_frame_count=len(problem.observations),
        expected_joint_count=int(problem.template.joints_rest.shape[0]),
    )
    beta = fit_state.beta
    root_translations = fit_state.root_translations
    vertices = problem.template.shaped_vertices(beta)
    frame_metrics = _write_frame_residuals(
        problem,
        vertices,
        root_translations,
        residual_dir,
        force=force,
    )

    thresholds = dict(SILHOUETTE_RESIDUAL_THRESHOLDS)
    mode_mean_key = f"{mode}_max_mean_union_error_fraction"
    mode_frame_key = f"{mode}_max_frame_union_error_fraction"
    mean_union_error = _mean_metric(frame_metrics, "union_error_fraction")
    max_union_error = _max_metric(frame_metrics, "union_error_fraction")
    checks = {
        "fit_state_exists": fit_state.path.exists(),
        "frame_count_threshold": len(frame_metrics) >= int(thresholds["min_frame_count"]),
        "residual_maps_exist": all(frame.residual_path.exists() for frame in frame_metrics),
        "residual_metrics_finite": all(_frame_metrics_are_finite(frame) for frame in frame_metrics),
        "mean_union_error_threshold": mean_union_error <= float(thresholds[mode_mean_key]),
        "frame_union_error_threshold": max_union_error <= float(thresholds[mode_frame_key]),
    }
    metrics: dict[str, float | int] = {
        "frame_count": len(frame_metrics),
        "mean_silhouette_iou": _mean_metric(frame_metrics, "silhouette_iou"),
        "min_silhouette_iou": _min_metric(frame_metrics, "silhouette_iou"),
        "mean_union_error_fraction": mean_union_error,
        "max_union_error_fraction": max_union_error,
        "mean_false_negative_fraction": _mean_metric(
            frame_metrics,
            "false_negative_fraction",
        ),
        "mean_false_positive_fraction": _mean_metric(
            frame_metrics,
            "false_positive_fraction",
        ),
        "mean_observed_area_fraction": _mean_metric(
            frame_metrics,
            "observed_area_fraction",
        ),
        "mean_fitted_area_fraction": _mean_metric(frame_metrics, "fitted_area_fraction"),
    }
    success = all(checks.values())
    manifest = {
        "schema_version": "1.0",
        "source": "opencat3d.silhouette_residuals",
        "mode": mode,
        "frame_count": len(frame_metrics),
        "legend": RESIDUAL_LEGEND,
        "frames": [
            {
                "frame_id": frame.frame_id,
                "residual_path": str(frame.residual_path),
                "sha256": sha256_file(frame.residual_path),
                "silhouette_iou": frame.silhouette_iou,
                "union_error_fraction": frame.union_error_fraction,
                "false_negative_fraction": frame.false_negative_fraction,
                "false_positive_fraction": frame.false_positive_fraction,
                "observed_area_fraction": frame.observed_area_fraction,
                "fitted_area_fraction": frame.fitted_area_fraction,
            }
            for frame in frame_metrics
        ],
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "success": success,
        "checks": checks,
        "thresholds": thresholds,
        "metrics": metrics,
        "artifacts": {
            "fit_state": str(fit_state.path),
            "fit_state_source_format": fit_state.source_format,
            "manifest": str(manifest_path),
            "residual_dir": str(residual_dir),
        },
        "frames": manifest["frames"],
    }
    residual_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SilhouetteResidualEvaluationResult(
        report_path=report_path,
        manifest_path=manifest_path,
        residual_dir=residual_dir,
        success=success,
        checks=checks,
        metrics=metrics,
        frames=tuple(frame_metrics),
    )


def _write_frame_residuals(
    problem: FitProblem,
    vertices: NDArray[np.float64],
    root_translations: NDArray[np.float64],
    residual_dir: Path,
    *,
    force: bool,
) -> list[SilhouetteResidualFrameMetric]:
    residual_dir.mkdir(parents=True, exist_ok=True)
    frame_metrics: list[SilhouetteResidualFrameMetric] = []
    for observation, translation in zip(
        problem.observations,
        root_translations,
        strict=True,
    ):
        fitted_mask = render_silhouette(
            vertices + translation[None, :],
            problem.template.faces,
            observation.camera,
        )
        observed_mask = np.asarray(observation.hard_mask, dtype=np.float32)
        residual = _compose_residual_map(observed_mask, fitted_mask)
        residual_path = residual_dir / f"{observation.frame_id}_residual.png"
        if residual_path.exists() and not force:
            raise FileExistsError(f"{residual_path} already exists; pass force=True to overwrite")
        Image.fromarray(residual).save(residual_path)
        frame_metrics.append(
            _frame_metric(
                observation.frame_id,
                residual_path,
                observed_mask,
                fitted_mask,
            )
        )
    return frame_metrics


def _compose_residual_map(
    observed_mask: NDArray[np.floating],
    fitted_mask: NDArray[np.floating],
) -> NDArray[np.uint8]:
    observed = np.asarray(observed_mask, dtype=np.float32) >= 0.5
    fitted = np.asarray(fitted_mask, dtype=np.float32) >= 0.5
    if observed.shape != fitted.shape:
        raise ValueError("observed and fitted masks must have the same shape")

    residual = np.full((*observed.shape, 3), RESIDUAL_LEGEND["background_rgb"], dtype=np.uint8)
    residual[observed & fitted] = np.asarray(RESIDUAL_LEGEND["overlap_rgb"], dtype=np.uint8)
    residual[observed & ~fitted] = np.asarray(
        RESIDUAL_LEGEND["observed_only_rgb"],
        dtype=np.uint8,
    )
    residual[fitted & ~observed] = np.asarray(RESIDUAL_LEGEND["fitted_only_rgb"], dtype=np.uint8)
    return residual


def _frame_metric(
    frame_id: str,
    residual_path: Path,
    observed_mask: NDArray[np.floating],
    fitted_mask: NDArray[np.floating],
) -> SilhouetteResidualFrameMetric:
    observed = np.asarray(observed_mask, dtype=np.float32) >= 0.5
    fitted = np.asarray(fitted_mask, dtype=np.float32) >= 0.5
    if observed.shape != fitted.shape:
        raise ValueError("observed and fitted masks must have the same shape")

    observed_only = observed & ~fitted
    fitted_only = fitted & ~observed
    overlap = observed & fitted
    union = observed | fitted
    observed_count = int(np.count_nonzero(observed))
    fitted_count = int(np.count_nonzero(fitted))
    union_count = int(np.count_nonzero(union))
    overlap_count = int(np.count_nonzero(overlap))
    return SilhouetteResidualFrameMetric(
        frame_id=frame_id,
        residual_path=residual_path,
        silhouette_iou=float(overlap_count / max(union_count, 1)),
        union_error_fraction=float(
            (np.count_nonzero(observed_only) + np.count_nonzero(fitted_only))
            / max(union_count, 1)
        ),
        false_negative_fraction=float(np.count_nonzero(observed_only) / max(observed_count, 1)),
        false_positive_fraction=float(np.count_nonzero(fitted_only) / max(fitted_count, 1)),
        observed_area_fraction=float(np.mean(observed)),
        fitted_area_fraction=float(np.mean(fitted)),
    )


def _frame_metrics_are_finite(frame: SilhouetteResidualFrameMetric) -> bool:
    return bool(
        np.isfinite(
            [
                frame.silhouette_iou,
                frame.union_error_fraction,
                frame.false_negative_fraction,
                frame.false_positive_fraction,
                frame.observed_area_fraction,
                frame.fitted_area_fraction,
            ]
        ).all()
    )


def _mean_metric(frames: list[SilhouetteResidualFrameMetric], field_name: str) -> float:
    if not frames:
        return 0.0
    values = [float(getattr(frame, field_name)) for frame in frames]
    return float(np.mean(values))


def _max_metric(frames: list[SilhouetteResidualFrameMetric], field_name: str) -> float:
    return float(max(float(getattr(frame, field_name)) for frame in frames)) if frames else 0.0


def _min_metric(frames: list[SilhouetteResidualFrameMetric], field_name: str) -> float:
    return float(min(float(getattr(frame, field_name)) for frame in frames)) if frames else 0.0
