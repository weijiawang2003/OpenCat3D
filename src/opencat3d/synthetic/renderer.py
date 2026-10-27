"""Small deterministic CPU rasterizer for synthetic sandbox tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opencat3d.camera.conventions import project_world_points
from opencat3d.schemas import PerspectiveCameraState


@dataclass(frozen=True)
class RenderBuffers:
    mask: NDArray[np.float32]
    depth: NDArray[np.float32]
    face_index: NDArray[np.int64]


def render_silhouette(
    vertices_world: NDArray[np.floating],
    faces: NDArray[np.integer],
    camera: PerspectiveCameraState,
) -> NDArray[np.float32]:
    return render_buffers(vertices_world, faces, camera).mask


def render_buffers(
    vertices_world: NDArray[np.floating],
    faces: NDArray[np.integer],
    camera: PerspectiveCameraState,
) -> RenderBuffers:
    projected = project_world_points(vertices_world, camera)
    mask = np.zeros((camera.image_height, camera.image_width), dtype=np.float32)
    depth = np.full((camera.image_height, camera.image_width), np.inf, dtype=np.float32)
    face_index = np.full((camera.image_height, camera.image_width), -1, dtype=np.int64)

    for index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        tri = projected[face]
        if np.any(tri[:, 2] <= 1e-6):
            continue
        _rasterize_triangle(mask, depth, face_index, tri, index)
    return RenderBuffers(mask=mask, depth=depth, face_index=face_index)


def _rasterize_triangle(
    mask: NDArray[np.float32],
    depth: NDArray[np.float32],
    face_index: NDArray[np.int64],
    tri_xyz: NDArray[np.float64],
    index: int,
) -> None:
    height, width = mask.shape
    tri_xy = tri_xyz[:, :2]
    min_xy = np.floor(tri_xy.min(axis=0)).astype(np.int64)
    max_xy = np.ceil(tri_xy.max(axis=0)).astype(np.int64)
    x0 = max(int(min_xy[0]), 0)
    y0 = max(int(min_xy[1]), 0)
    x1 = min(int(max_xy[0]), width - 1)
    y1 = min(int(max_xy[1]), height - 1)
    if x1 < x0 or y1 < y0:
        return

    a, b, c = tri_xy
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(denom)) < 1e-12:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    w0 = ((b[1] - c[1]) * (grid_x - c[0]) + (c[0] - b[0]) * (grid_y - c[1])) / denom
    w1 = ((c[1] - a[1]) * (grid_x - c[0]) + (a[0] - c[0]) * (grid_y - c[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    z = w0 * tri_xyz[0, 2] + w1 * tri_xyz[1, 2] + w2 * tri_xyz[2, 2]

    depth_view = depth[y0 : y1 + 1, x0 : x1 + 1]
    mask_view = mask[y0 : y1 + 1, x0 : x1 + 1]
    face_view = face_index[y0 : y1 + 1, x0 : x1 + 1]
    update = inside & np.isfinite(z) & (z < depth_view - 1e-9)
    depth_view[update] = z[update].astype(np.float32)
    mask_view[update] = 1.0
    face_view[update] = index
