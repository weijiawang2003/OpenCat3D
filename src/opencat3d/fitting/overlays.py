"""Deterministic per-frame fitting overlay artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from opencat3d.io.cache import sha256_file
from opencat3d.synthetic.renderer import render_silhouette

if TYPE_CHECKING:
    from opencat3d.fitting.optimizer import FitProblem


@dataclass(frozen=True)
class FitOverlayResult:
    overlay_dir: Path
    manifest_path: Path
    frame_count: int
    mean_silhouette_iou: float


def write_fit_overlays(
    problem: FitProblem,
    *,
    beta: NDArray[np.floating],
    root_translations: NDArray[np.floating],
    output_dir: Path,
    force: bool = False,
) -> FitOverlayResult:
    """Write deterministic observed-vs-fitted silhouette overlays for each frame."""

    translations = np.asarray(root_translations, dtype=np.float64)
    if translations.shape != (len(problem.observations), 3):
        raise ValueError("root_translations must have shape [num_frames,3]")
    shaped_vertices = problem.template.shaped_vertices(np.asarray(beta, dtype=np.float64))

    overlay_dir = output_dir / "per_frame_overlay"
    manifest_path = overlay_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} already exists; pass force=True to overwrite")
    overlay_dir.mkdir(parents=True, exist_ok=True)

    frame_records: list[dict[str, object]] = []
    silhouette_ious: list[float] = []
    for observation, translation in zip(problem.observations, translations, strict=True):
        fitted_mask = render_silhouette(
            shaped_vertices + translation[None, :],
            problem.template.faces,
            observation.camera,
        )
        observed_mask = np.asarray(observation.hard_mask, dtype=np.float32)
        overlay = _compose_overlay(observed_mask, fitted_mask)
        overlay_path = overlay_dir / f"{observation.frame_id}_overlay.png"
        if overlay_path.exists() and not force:
            raise FileExistsError(f"{overlay_path} already exists; pass force=True to overwrite")
        Image.fromarray(overlay).save(overlay_path)

        silhouette_iou = _binary_iou(observed_mask, fitted_mask)
        silhouette_ious.append(silhouette_iou)
        frame_records.append(
            {
                "frame_id": observation.frame_id,
                "overlay_path": str(overlay_path),
                "sha256": sha256_file(overlay_path),
                "image_width": int(observation.camera.image_width),
                "image_height": int(observation.camera.image_height),
                "observed_mask_area": float(np.mean(observed_mask >= 0.5)),
                "fitted_mask_area": float(np.mean(fitted_mask >= 0.5)),
                "silhouette_iou": silhouette_iou,
            }
        )

    mean_silhouette_iou = float(np.mean(silhouette_ious)) if silhouette_ious else 0.0
    manifest = {
        "schema_version": "1.0",
        "source": "opencat3d.fit_overlay",
        "frame_count": len(frame_records),
        "mean_silhouette_iou": mean_silhouette_iou,
        "legend": {
            "observed_only_rgb": [32, 112, 255],
            "fitted_only_rgb": [255, 72, 32],
            "overlap_rgb": [255, 216, 32],
            "background_rgb": [16, 16, 16],
        },
        "frames": frame_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FitOverlayResult(
        overlay_dir=overlay_dir,
        manifest_path=manifest_path,
        frame_count=len(frame_records),
        mean_silhouette_iou=mean_silhouette_iou,
    )


def _compose_overlay(
    observed_mask: NDArray[np.floating],
    fitted_mask: NDArray[np.floating],
) -> NDArray[np.uint8]:
    observed = np.asarray(observed_mask, dtype=np.float32) >= 0.5
    fitted = np.asarray(fitted_mask, dtype=np.float32) >= 0.5
    if observed.shape != fitted.shape:
        raise ValueError("observed and fitted masks must have the same shape")

    overlay = np.full((*observed.shape, 3), 16, dtype=np.uint8)
    observed_only = observed & ~fitted
    fitted_only = fitted & ~observed
    overlap = observed & fitted
    overlay[observed_only] = np.asarray([32, 112, 255], dtype=np.uint8)
    overlay[fitted_only] = np.asarray([255, 72, 32], dtype=np.uint8)
    overlay[overlap] = np.asarray([255, 216, 32], dtype=np.uint8)
    return overlay


def _binary_iou(
    observed_mask: NDArray[np.floating],
    fitted_mask: NDArray[np.floating],
) -> float:
    observed = np.asarray(observed_mask, dtype=np.float32) >= 0.5
    fitted = np.asarray(fitted_mask, dtype=np.float32) >= 0.5
    if observed.shape != fitted.shape:
        raise ValueError("observed and fitted masks must have the same shape")
    union = np.logical_or(observed, fitted)
    if not np.any(union):
        return 1.0
    intersection = np.logical_and(observed, fitted)
    return float(np.count_nonzero(intersection) / np.count_nonzero(union))
