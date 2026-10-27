import numpy as np
import pytest

from opencat3d.camera.conventions import project_world_points
from opencat3d.fitting.losses import (
    FACE_DETAIL_REGION_NAMES,
    FEATURE_ANCHOR_TARGET_NAMES,
    face_region_detail_loss,
    feature_anchor_observation_weights,
    feature_anchor_projection_loss,
    feature_anchor_world_points,
    region_detail_vertex_weights,
)
from opencat3d.fitting.optimizer import load_synthetic_fit_problem
from opencat3d.synthetic.generator import create_synthetic_project
from opencat3d.template.cat_template import create_placeholder_cat_template


def test_region_detail_vertex_weights_prioritize_face_detail_regions() -> None:
    template = create_placeholder_cat_template()

    weights = region_detail_vertex_weights(template)

    assert weights.shape == (template.vertices_canonical.shape[0],)
    assert np.isfinite(weights).all()
    assert weights.max() == 1.0
    assert weights[template.region("nose").vertex_indices].mean() > 0.0
    assert weights[template.region("left_eye").vertex_indices].mean() > 0.0
    assert weights[template.region("torso").vertex_indices].max() == 0.0


def test_face_region_detail_loss_prefers_target_geometry() -> None:
    template = create_placeholder_cat_template()
    target = template.vertices_canonical.copy()
    candidate = target.copy()
    candidate[template.region("nose").vertex_indices, 2] += 0.025
    candidate[template.region("left_eye").vertex_indices, 2] -= 0.020
    candidate[template.region("right_eye").vertex_indices, 2] -= 0.020

    target_loss = face_region_detail_loss(target, target, template)
    candidate_loss = face_region_detail_loss(candidate, target, template)

    assert target_loss == 0.0
    assert np.isfinite(candidate_loss)
    assert candidate_loss > target_loss


def test_face_region_detail_loss_ignores_unselected_body_regions() -> None:
    template = create_placeholder_cat_template()
    target = template.vertices_canonical.copy()
    candidate = target.copy()
    candidate[template.region("torso").vertex_indices, 1] += 0.20

    loss = face_region_detail_loss(candidate, target, template)

    assert loss == 0.0


def test_face_region_detail_loss_uses_confidence_gate() -> None:
    template = create_placeholder_cat_template()
    target = template.vertices_canonical.copy()
    candidate = target.copy()
    candidate[template.region("muzzle").vertex_indices, 2] += 0.05
    confidence = np.zeros(template.vertices_canonical.shape[0], dtype=np.float64)

    suppressed = face_region_detail_loss(
        candidate,
        target,
        template,
        vertex_confidence=confidence,
    )
    confidence[template.region("muzzle").vertex_indices] = 1.0
    active = face_region_detail_loss(
        candidate,
        target,
        template,
        vertex_confidence=confidence,
    )

    assert suppressed == 0.0
    assert active > suppressed


def test_feature_anchor_observation_weights_prioritize_facial_identity() -> None:
    template = create_placeholder_cat_template()

    weights = feature_anchor_observation_weights(template)

    assert set(weights) == set(FEATURE_ANCHOR_TARGET_NAMES)
    assert weights["left_eye_center"] == 1.0
    assert weights["right_eye_center"] == 1.0
    assert weights["nose_tip"] == 1.0
    assert weights["muzzle_center"] > weights["tail_tip"]
    assert weights["left_eye_center"] > weights["left_front_paw"]


def test_feature_anchor_projection_loss_prefers_observed_projection(tmp_path) -> None:
    project_dir = tmp_path / "project"
    create_synthetic_project(project_dir, num_views=3, image_size=96, seed=123)
    problem = load_synthetic_fit_problem(project_dir)
    observation = problem.observations[0]
    vertices = problem.template.shaped_vertices(problem.beta)
    anchor_names = ("left_eye_center", "nose_tip", "left_front_paw", "tail_tip")
    points = feature_anchor_world_points(
        problem.template,
        vertices,
        anchor_names=anchor_names,
    )
    projected = project_world_points(
        np.asarray([points[name] for name in anchor_names], dtype=np.float64),
        observation.camera,
    )
    observed = {
        name: (float(projected[index, 0]), float(projected[index, 1]), 1.0)
        for index, name in enumerate(anchor_names)
    }

    true_loss = feature_anchor_projection_loss(
        problem.template,
        vertices,
        observation.camera,
        observed,
        observation.bbox_xyxy,
        anchor_names=anchor_names,
        robust=False,
    )
    shifted_loss = feature_anchor_projection_loss(
        problem.template,
        vertices + np.asarray([0.18, 0.0, 0.0], dtype=np.float64),
        observation.camera,
        observed,
        observation.bbox_xyxy,
        anchor_names=anchor_names,
        robust=False,
    )

    assert true_loss < 1e-12
    assert shifted_loss > true_loss


def test_feature_anchor_projection_loss_uses_confidence_gate(tmp_path) -> None:
    project_dir = tmp_path / "project"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=321)
    problem = load_synthetic_fit_problem(project_dir)
    observation = problem.observations[0]
    vertices = problem.template.shaped_vertices(problem.beta)
    anchor_names = ("nose_tip",)
    points = feature_anchor_world_points(problem.template, vertices, anchor_names=anchor_names)
    projected = project_world_points(
        np.asarray([points["nose_tip"]], dtype=np.float64),
        observation.camera,
    )
    observed = {
        "nose_tip": (
            float(projected[0, 0] + 12.0),
            float(projected[0, 1]),
            0.0,
        )
    }

    loss = feature_anchor_projection_loss(
        problem.template,
        vertices,
        observation.camera,
        observed,
        observation.bbox_xyxy,
        anchor_names=anchor_names,
    )

    assert loss == 0.0


def test_feature_anchor_projection_loss_rejects_unknown_observed_anchor(tmp_path) -> None:
    project_dir = tmp_path / "project"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=456)
    problem = load_synthetic_fit_problem(project_dir)
    observation = problem.observations[0]
    vertices = problem.template.shaped_vertices(problem.beta)

    with pytest.raises(ValueError, match="unknown feature anchor"):
        feature_anchor_projection_loss(
            problem.template,
            vertices,
            observation.camera,
            {"generated_anchor": (0.0, 0.0, 1.0)},
            observation.bbox_xyxy,
        )


def test_face_region_detail_loss_validates_shapes() -> None:
    template = create_placeholder_cat_template()
    vertices = template.vertices_canonical

    with pytest.raises(ValueError):
        face_region_detail_loss(vertices[:-1], vertices, template)
    with pytest.raises(ValueError):
        face_region_detail_loss(
            vertices,
            vertices,
            template,
            region_names=(),
        )
    with pytest.raises(ValueError):
        feature_anchor_world_points(template, vertices[:-1])
    with pytest.raises(ValueError):
        feature_anchor_observation_weights(template, anchor_names=("not_an_anchor",))


def test_default_face_detail_regions_are_stable() -> None:
    assert FACE_DETAIL_REGION_NAMES == (
        "left_eye",
        "right_eye",
        "nose",
        "muzzle",
        "left_ear_inner",
        "right_ear_inner",
    )


def test_default_feature_anchor_targets_are_stable() -> None:
    assert FEATURE_ANCHOR_TARGET_NAMES == (
        "left_eye_center",
        "right_eye_center",
        "nose_tip",
        "muzzle_center",
        "mouth_line",
        "left_ear_inner",
        "right_ear_inner",
        "left_front_paw",
        "right_front_paw",
        "left_hind_paw",
        "right_hind_paw",
        "tail_base",
        "tail_tip",
    )
