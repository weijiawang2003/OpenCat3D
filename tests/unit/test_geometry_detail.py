import numpy as np
import pytest

from opencat3d.geometry.detail import (
    FACE_DETAIL_DEFORMATION_VERSION,
    FACE_SURFACE_ANCHOR_VERSION,
    apply_bounded_face_detail_deformation,
    face_surface_anchor_vertex_groups,
)
from opencat3d.template.cat_template import create_placeholder_cat_template


def test_bounded_face_detail_deformation_preserves_topology_and_is_deterministic() -> None:
    template = create_placeholder_cat_template()
    vertices = template.shaped_vertices()

    first = apply_bounded_face_detail_deformation(template, vertices)
    second = apply_bounded_face_detail_deformation(template, vertices)

    assert first.vertices.shape == vertices.shape
    np.testing.assert_allclose(first.vertices, second.vertices)
    assert first.manifest["source"] == FACE_DETAIL_DEFORMATION_VERSION
    assert first.manifest["surface_anchor_source"] == FACE_SURFACE_ANCHOR_VERSION
    assert first.manifest["applied"] is True
    assert first.manifest["deformed_vertex_count"] > 0
    assert first.manifest["deformed_vertex_count"] < vertices.shape[0]
    assert first.manifest["surface_anchor_count"] == 10
    assert first.manifest["surface_anchor_vertex_count"] > 0

    displacement = np.linalg.norm(first.vertices - vertices, axis=1)
    assert float(np.max(displacement)) <= 0.040 + 1e-12
    assert int(np.count_nonzero(displacement > 0.0)) == first.manifest[
        "deformed_vertex_count"
    ]


def test_bounded_face_detail_deformation_has_named_surface_anchors() -> None:
    template = create_placeholder_cat_template()
    vertices = template.shaped_vertices()

    result = apply_bounded_face_detail_deformation(template, vertices)
    anchors = {anchor["name"]: anchor for anchor in result.manifest["surface_anchors"]}

    assert {
        "left_eye_core",
        "right_eye_core",
        "left_eye_socket",
        "right_eye_socket",
        "nose_tip",
        "left_muzzle_pad",
        "right_muzzle_pad",
        "mouth_groove",
    }.issubset(anchors)
    assert anchors["mouth_groove"]["role"] == "mouth_line_indent"
    assert anchors["mouth_groove"]["vertex_count"] > 0

    muzzle_indices = template.region("muzzle").vertex_indices
    muzzle_delta = result.vertices[muzzle_indices] - vertices[muzzle_indices]
    assert np.min(muzzle_delta[:, 2]) < 0.0

    groups = {group.name: group for group in face_surface_anchor_vertex_groups(template, vertices)}
    assert groups["mouth_groove"].role == "mouth_line_indent"
    np.testing.assert_array_equal(
        groups["mouth_groove"].vertex_indices,
        groups["mouth_groove"].vertex_indices.astype(np.int64),
    )
    assert groups["mouth_groove"].vertex_indices.size == anchors["mouth_groove"][
        "vertex_count"
    ]


def test_bounded_face_detail_deformation_targets_face_regions_only() -> None:
    template = create_placeholder_cat_template()
    vertices = template.shaped_vertices()

    result = apply_bounded_face_detail_deformation(template, vertices)
    displacement = np.linalg.norm(result.vertices - vertices, axis=1)

    for region_name in (
        "left_eye",
        "right_eye",
        "nose",
        "muzzle",
        "left_ear_inner",
        "right_ear_inner",
    ):
        region_displacement = displacement[template.region(region_name).vertex_indices]
        assert np.count_nonzero(region_displacement > 0.0) > 0

    deformed = displacement > 0.0
    untouched_vertices = np.setdiff1d(
        np.arange(vertices.shape[0]),
        np.nonzero(deformed)[0],
    )
    assert untouched_vertices.size > 0
    np.testing.assert_allclose(result.vertices[untouched_vertices], vertices[untouched_vertices])


def test_bounded_face_detail_deformation_rejects_invalid_inputs() -> None:
    template = create_placeholder_cat_template()

    with pytest.raises(ValueError, match="max_total_displacement"):
        apply_bounded_face_detail_deformation(
            template,
            template.shaped_vertices(),
            max_total_displacement=0.0,
        )

    with pytest.raises(ValueError, match="vertices must have shape"):
        apply_bounded_face_detail_deformation(template, template.shaped_vertices()[:10])
