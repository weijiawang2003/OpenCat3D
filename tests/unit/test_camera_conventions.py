import numpy as np

from opencat3d.camera.conventions import (
    axis_angle_to_matrix,
    camera_to_world,
    gltf_camera_to_opencv_camera,
    internal_points_to_gltf,
    make_perspective_camera,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    opencv_camera_to_gltf_camera,
    project_world_points,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    world_to_camera,
)


def test_world_camera_round_trip_and_projection_center() -> None:
    camera = make_perspective_camera(
        image_width=101,
        image_height=101,
        fov_deg=45.0,
        eye=np.asarray([0.0, 0.6, -4.0]),
        target=np.asarray([0.0, 0.6, 0.0]),
    )
    points = np.asarray([[0.0, 0.6, 0.0], [0.2, 0.8, 0.5]], dtype=np.float64)

    camera_points = world_to_camera(points, camera)
    recovered = camera_to_world(camera_points, camera)
    projected = project_world_points(points[:1], camera)

    np.testing.assert_allclose(recovered, points, atol=1e-9)
    np.testing.assert_allclose(projected[0, :2], [50.0, 50.0], atol=1e-9)
    assert camera_points[0, 2] > 0.0


def test_opencv_gltf_camera_round_trip() -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, 4.0]], dtype=np.float64)

    converted = opencv_camera_to_gltf_camera(points)
    recovered = gltf_camera_to_opencv_camera(converted)

    np.testing.assert_allclose(converted, [[1.0, -2.0, -3.0], [-1.0, 2.0, -4.0]])
    np.testing.assert_allclose(recovered, points)
    np.testing.assert_allclose(internal_points_to_gltf(points), points)


def test_rotation_6d_matrix_round_trip_is_orthonormal() -> None:
    matrix = axis_angle_to_matrix(np.asarray([0.15, -0.30, 0.05], dtype=np.float64))
    rot6d = matrix_to_rotation_6d(matrix)
    recovered = rotation_6d_to_matrix(rot6d)

    np.testing.assert_allclose(recovered.T @ recovered, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(recovered, matrix, atol=1e-9)


def test_quaternion_matrix_round_trip() -> None:
    matrix = axis_angle_to_matrix(np.asarray([0.0, 0.4, 0.0], dtype=np.float64))
    quat = matrix_to_quaternion(matrix)
    recovered = quaternion_to_matrix(quat)

    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-9)
    np.testing.assert_allclose(recovered, matrix, atol=1e-9)

