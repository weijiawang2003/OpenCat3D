"""Synthetic multi-view observation generator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from opencat3d.camera.conventions import make_perspective_camera, project_world_points
from opencat3d.io.cache import sha256_file
from opencat3d.schemas import (
    FrameObservationMeta,
    PerspectiveCameraState,
    TensorRef,
    TensorStorageFormat,
    write_json_model,
)
from opencat3d.synthetic.renderer import render_silhouette
from opencat3d.template.cat_template import create_placeholder_cat_template


@dataclass(frozen=True)
class SyntheticCreateResult:
    project_dir: Path
    observation_manifest_path: Path
    template_path: Path
    frame_count: int


def create_synthetic_project(
    project_dir: Path,
    *,
    num_views: int = 6,
    image_size: int = 128,
    seed: int = 20260625,
    force: bool = False,
) -> SyntheticCreateResult:
    if num_views < 3:
        raise ValueError("num_views must be >= 3")
    if image_size < 32:
        raise ValueError("image_size must be >= 32")

    observations_dir = project_dir / "observations" / "synthetic"
    template_dir = project_dir / "assets" / "cat_template"
    if observations_dir.exists() and any(observations_dir.iterdir()) and not force:
        raise FileExistsError(f"{observations_dir} already exists; pass force=True to overwrite")

    observations_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    template = create_placeholder_cat_template()
    template_path = template_dir / "placeholder_cat_template.npz"
    template.save_npz(template_path)

    rng = np.random.default_rng(seed)
    beta = np.asarray([0.10, -0.05, 0.08], dtype=np.float64)
    vertices = template.shaped_vertices(beta)
    joints = template.joints(beta)
    frame_metas = []
    cameras: dict[str, dict[str, object]] = {}

    for index, camera in enumerate(_synthetic_cameras(num_views, image_size)):
        frame_id = f"synthetic_{index:03d}"
        frame_dir = observations_dir / frame_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        hard_mask = render_silhouette(vertices, template.faces, camera)
        if index == num_views - 1:
            hard_mask = _apply_occluder(hard_mask, rng)
        soft_mask = hard_mask.copy()
        keypoints = _project_keypoints(joints, camera)
        image = np.stack([hard_mask * 0.7, hard_mask * 0.7, hard_mask * 0.7], axis=0).astype(
            np.float32
        )

        image_ref = _save_array_ref(
            frame_dir / "image_srgb.npz",
            "image_srgb",
            image,
            "image_srgb",
        )
        hard_ref = _save_array_ref(
            frame_dir / "hard_mask.npz",
            "hard_mask",
            hard_mask[None],
            "hard_mask",
        )
        soft_ref = _save_array_ref(
            frame_dir / "soft_mask.npz",
            "soft_mask",
            soft_mask[None],
            "soft_mask",
        )
        kp_ref = _save_array_ref(
            frame_dir / "keypoints_xyc.npz",
            "keypoints_xyc",
            keypoints,
            "keypoints",
        )
        bbox = _bbox_from_mask(hard_mask)
        meta = FrameObservationMeta(
            frame_id=frame_id,
            image_srgb_path=image_ref.path,
            image_linear_path=None,
            hard_mask=hard_ref,
            soft_mask=soft_ref,
            bbox_xyxy=bbox,
            keypoints_xyc=kp_ref,
            quality={
                "synthetic": 1.0,
                "mask_area_ratio": float(hard_mask.mean()),
                "keypoint_confidence": float(keypoints[:, 2].mean()),
            },
        )
        write_json_model(meta, frame_dir / "observation.json")
        frame_metas.append(meta.model_dump(mode="json"))
        cameras[frame_id] = camera.model_dump(mode="json")

    observation_manifest_path = observations_dir / "manifest.json"
    observation_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generator": "opencat3d.synthetic",
                "seed": seed,
                "num_views": num_views,
                "image_size": image_size,
                "beta": beta.tolist(),
                "template_path": str(template_path),
                "frames": frame_metas,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (observations_dir / "cameras.json").write_text(
        json.dumps(cameras, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SyntheticCreateResult(
        project_dir=project_dir,
        observation_manifest_path=observation_manifest_path,
        template_path=template_path,
        frame_count=num_views,
    )


def _synthetic_cameras(num_views: int, image_size: int) -> list[PerspectiveCameraState]:
    cameras = []
    radius = 4.2
    for index in range(num_views):
        angle = 2.0 * np.pi * index / num_views
        eye = np.asarray([radius * np.sin(angle), 1.0, radius * np.cos(angle)], dtype=np.float64)
        target = np.asarray([0.0, 0.65, 0.0], dtype=np.float64)
        cameras.append(
            make_perspective_camera(
                image_width=image_size,
                image_height=image_size,
                fov_deg=42.0,
                eye=eye,
                target=target,
            )
        )
    return cameras


def _save_array_ref(path: Path, key: str, array: NDArray[np.floating], semantic: str) -> TensorRef:
    np.savez_compressed(path, **{key: array})
    return TensorRef(
        path=str(path),
        shape=list(array.shape),
        dtype=str(array.dtype),
        sha256=sha256_file(path),
        storage=TensorStorageFormat.NPZ,
        semantic=semantic,
    )


def _project_keypoints(
    joints_world: NDArray[np.floating],
    camera: PerspectiveCameraState,
) -> NDArray[np.float32]:
    projected = project_world_points(joints_world, camera)
    visible = (
        (projected[:, 2] > 1e-6)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < camera.image_width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < camera.image_height)
    )
    return np.stack([projected[:, 0], projected[:, 1], visible.astype(np.float64)], axis=-1).astype(
        np.float32
    )


def _bbox_from_mask(mask: NDArray[np.floating]) -> list[float]:
    ys, xs = np.nonzero(mask > 0.0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("synthetic renderer produced an empty mask")
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _apply_occluder(mask: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    out = mask.copy()
    height, width = out.shape
    occ_w = max(width // 8, 1)
    occ_h = max(height // 3, 1)
    x0 = int(rng.integers(width // 3, max(width // 3 + 1, width // 2)))
    y0 = int(rng.integers(height // 4, max(height // 4 + 1, height // 2)))
    out[y0 : min(y0 + occ_h, height), x0 : min(x0 + occ_w, width)] = 0.0
    return out
