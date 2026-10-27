"""Coordinate conventions, rotations, and projection helpers.

Internal world coordinates are right-handed, +Y up, and the canonical cat faces +Z.
Camera states are stored in OpenCV convention: +X image-right, +Y image-down, +Z forward.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from opencat3d.schemas import PerspectiveCameraState

FloatArray = NDArray[np.float64]

OPENCV_CAMERA_TO_GLTF_CAMERA: FloatArray = np.diag([1.0, -1.0, -1.0])
GLTF_CAMERA_TO_OPENCV_CAMERA: FloatArray = OPENCV_CAMERA_TO_GLTF_CAMERA.copy()


def normalize_vector(vector: NDArray[np.floating], eps: float = 1e-12) -> FloatArray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norm, eps)


def look_at_opencv(
    eye: NDArray[np.floating],
    target: NDArray[np.floating],
    up: NDArray[np.floating] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return OpenCV world-to-camera rotation and translation for an internal-world camera."""

    eye_f = np.asarray(eye, dtype=np.float64)
    target_f = np.asarray(target, dtype=np.float64)
    up_f = np.asarray([0.0, 1.0, 0.0] if up is None else up, dtype=np.float64)

    forward = normalize_vector(target_f - eye_f)
    right = normalize_vector(np.cross(up_f, forward))
    down = normalize_vector(np.cross(forward, right))
    rotation_world_to_camera = np.stack([right, down, forward], axis=0)
    translation_world_to_camera = -rotation_world_to_camera @ eye_f
    return rotation_world_to_camera, translation_world_to_camera


def make_perspective_camera(
    image_width: int,
    image_height: int,
    fov_deg: float,
    eye: NDArray[np.floating],
    target: NDArray[np.floating] | None = None,
) -> PerspectiveCameraState:
    target_arr = np.asarray([0.0, 0.0, 0.0] if target is None else target, dtype=np.float64)
    rotation, translation = look_at_opencv(np.asarray(eye, dtype=np.float64), target_arr)
    focal = 0.5 * float(image_width) / math.tan(math.radians(fov_deg) * 0.5)
    return PerspectiveCameraState(
        image_width=image_width,
        image_height=image_height,
        fx=focal,
        fy=focal,
        cx=(image_width - 1) * 0.5,
        cy=(image_height - 1) * 0.5,
        rotation_world_to_camera=rotation.tolist(),
        translation_world_to_camera=translation.tolist(),
    )


def camera_center_world(camera: PerspectiveCameraState) -> FloatArray:
    rotation = np.asarray(camera.rotation_world_to_camera, dtype=np.float64)
    translation = np.asarray(camera.translation_world_to_camera, dtype=np.float64)
    return -rotation.T @ translation


def world_to_camera(
    points_world: NDArray[np.floating],
    camera: PerspectiveCameraState,
) -> FloatArray:
    points = np.asarray(points_world, dtype=np.float64)
    rotation = np.asarray(camera.rotation_world_to_camera, dtype=np.float64)
    translation = np.asarray(camera.translation_world_to_camera, dtype=np.float64)
    return points @ rotation.T + translation


def camera_to_world(
    points_camera: NDArray[np.floating],
    camera: PerspectiveCameraState,
) -> FloatArray:
    points = np.asarray(points_camera, dtype=np.float64)
    rotation = np.asarray(camera.rotation_world_to_camera, dtype=np.float64)
    translation = np.asarray(camera.translation_world_to_camera, dtype=np.float64)
    return (points - translation) @ rotation


def project_camera_points(
    points_camera: NDArray[np.floating],
    camera: PerspectiveCameraState,
    eps: float = 1e-9,
) -> FloatArray:
    points = np.asarray(points_camera, dtype=np.float64)
    z = np.maximum(points[..., 2], eps)
    u = camera.fx * (points[..., 0] / z) + camera.cx
    v = camera.fy * (points[..., 1] / z) + camera.cy
    return np.stack([u, v, points[..., 2]], axis=-1)


def project_world_points(
    points_world: NDArray[np.floating],
    camera: PerspectiveCameraState,
) -> FloatArray:
    return project_camera_points(world_to_camera(points_world, camera), camera)


def opencv_camera_to_gltf_camera(points_camera: NDArray[np.floating]) -> FloatArray:
    points = np.asarray(points_camera, dtype=np.float64)
    return points @ OPENCV_CAMERA_TO_GLTF_CAMERA.T


def gltf_camera_to_opencv_camera(points_camera: NDArray[np.floating]) -> FloatArray:
    points = np.asarray(points_camera, dtype=np.float64)
    return points @ GLTF_CAMERA_TO_OPENCV_CAMERA.T


def internal_points_to_gltf(points_world: NDArray[np.floating]) -> FloatArray:
    return np.asarray(points_world, dtype=np.float64).copy()


def gltf_points_to_internal(points_world: NDArray[np.floating]) -> FloatArray:
    return np.asarray(points_world, dtype=np.float64).copy()


def rotation_6d_to_matrix(rotation_6d: NDArray[np.floating]) -> FloatArray:
    arr = np.asarray(rotation_6d, dtype=np.float64)
    if arr.shape[-1] != 6:
        raise ValueError("rotation_6d must have last dimension 6")
    a1 = arr[..., 0:3]
    a2 = arr[..., 3:6]
    b1 = normalize_vector(a1)
    b2 = normalize_vector(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rotation_6d(matrix: NDArray[np.floating]) -> FloatArray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape[-2:] != (3, 3):
        raise ValueError("matrix must have trailing shape 3x3")
    return np.concatenate([arr[..., :, 0], arr[..., :, 1]], axis=-1)


def axis_angle_to_matrix(axis_angle: NDArray[np.floating]) -> FloatArray:
    vector = np.asarray(axis_angle, dtype=np.float64)
    if vector.shape[-1] != 3:
        raise ValueError("axis_angle must have last dimension 3")
    angle = np.linalg.norm(vector, axis=-1, keepdims=True)
    axis = vector / np.maximum(angle, 1e-12)
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    zeros = np.zeros_like(x)
    skew = np.stack(
        [
            np.stack([zeros, -z, y], axis=-1),
            np.stack([z, zeros, -x], axis=-1),
            np.stack([-y, x, zeros], axis=-1),
        ],
        axis=-2,
    )
    identity = np.broadcast_to(np.eye(3), skew.shape)
    sin = np.sin(angle)[..., None]
    cos = np.cos(angle)[..., None]
    return identity + sin * skew + (1.0 - cos) * (skew @ skew)


def quaternion_to_matrix(quaternion: NDArray[np.floating]) -> FloatArray:
    quat = normalize_vector(np.asarray(quaternion, dtype=np.float64))
    if quat.shape[-1] != 4:
        raise ValueError("quaternion must have last dimension 4 in wxyz order")
    w, x, y, z = np.moveaxis(quat, -1, 0)
    two = 2.0
    return np.stack(
        [
            np.stack(
                [
                    1 - two * (y * y + z * z),
                    two * (x * y - z * w),
                    two * (x * z + y * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    two * (x * y + z * w),
                    1 - two * (x * x + z * z),
                    two * (y * z - x * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    two * (x * z - y * w),
                    two * (y * z + x * w),
                    1 - two * (x * x + y * y),
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def matrix_to_quaternion(matrix: NDArray[np.floating]) -> FloatArray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape[-2:] != (3, 3):
        raise ValueError("matrix must have trailing shape 3x3")

    flat = arr.reshape((-1, 3, 3))
    quats: list[FloatArray] = []
    for mat in flat:
        trace = float(np.trace(mat))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quat = np.array(
                [
                    0.25 * scale,
                    (mat[2, 1] - mat[1, 2]) / scale,
                    (mat[0, 2] - mat[2, 0]) / scale,
                    (mat[1, 0] - mat[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            idx = int(np.argmax(np.diag(mat)))
            if idx == 0:
                scale = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
                quat = np.array(
                    [
                        (mat[2, 1] - mat[1, 2]) / scale,
                        0.25 * scale,
                        (mat[0, 1] + mat[1, 0]) / scale,
                        (mat[0, 2] + mat[2, 0]) / scale,
                    ],
                    dtype=np.float64,
                )
            elif idx == 1:
                scale = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
                quat = np.array(
                    [
                        (mat[0, 2] - mat[2, 0]) / scale,
                        (mat[0, 1] + mat[1, 0]) / scale,
                        0.25 * scale,
                        (mat[1, 2] + mat[2, 1]) / scale,
                    ],
                    dtype=np.float64,
                )
            else:
                scale = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
                quat = np.array(
                    [
                        (mat[1, 0] - mat[0, 1]) / scale,
                        (mat[0, 2] + mat[2, 0]) / scale,
                        (mat[1, 2] + mat[2, 1]) / scale,
                        0.25 * scale,
                    ],
                    dtype=np.float64,
                )
        quats.append(normalize_vector(quat))
    return np.asarray(quats, dtype=np.float64).reshape(arr.shape[:-2] + (4,))
