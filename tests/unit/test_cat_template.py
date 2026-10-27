import numpy as np
import pytest

from opencat3d.template.cat_template import (
    CAT_TEMPLATE_FACE_COUNT,
    CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES,
    CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES,
    CAT_TEMPLATE_VERTEX_COUNT,
    create_placeholder_cat_template,
    load_cat_template_npz,
    validate_cat_template_feature_anchor_manifest,
    validate_cat_template_region_manifest_v2,
)


def test_placeholder_template_has_stable_topology_and_weights() -> None:
    template = create_placeholder_cat_template()

    assert template.vertices_canonical.shape == (CAT_TEMPLATE_VERTEX_COUNT, 3)
    assert template.faces.shape == (CAT_TEMPLATE_FACE_COUNT, 3)
    assert template.joints_rest.shape == (11, 3)
    assert template.skin_weights.shape == (CAT_TEMPLATE_VERTEX_COUNT, 11)
    np.testing.assert_allclose(template.skin_weights.sum(axis=1), 1.0)
    assert np.all(template.faces >= 0)
    assert np.all(template.faces < template.vertices_canonical.shape[0])


def test_placeholder_template_has_explicit_facial_components() -> None:
    template = create_placeholder_cat_template()
    vertices = template.vertices_canonical

    expected_anchors = np.asarray(
        [
            [-0.12, 1.02, 1.18],
            [0.12, 1.02, 1.18],
            [0.0, 0.89, 1.36],
        ],
        dtype=np.float64,
    )
    for anchor in expected_anchors:
        distances = np.linalg.norm(vertices - anchor[None, :], axis=1)
        assert np.count_nonzero(distances <= 0.060) >= 12


def test_placeholder_template_has_stable_region_contract() -> None:
    template = create_placeholder_cat_template()
    regions = {region.name: region for region in template.regions}

    required = {
        "torso",
        "head",
        "muzzle",
        "left_eye",
        "right_eye",
        "nose",
        "left_ear_inner",
        "right_ear_inner",
        "left_front_paw",
        "right_front_paw",
        "left_hind_paw",
        "right_hind_paw",
        "tail",
        "ears",
        "paws",
        "limbs",
        "face",
        "body",
    }
    assert required.issubset(regions)

    for region in regions.values():
        assert region.vertex_indices.ndim == 1
        assert region.face_indices.ndim == 1
        assert region.vertex_indices.size > 0
        assert region.face_indices.size > 0
        assert np.all(region.vertex_indices[:-1] <= region.vertex_indices[1:])
        assert np.all(region.face_indices[:-1] <= region.face_indices[1:])
        assert int(region.vertex_indices[-1]) < CAT_TEMPLATE_VERTEX_COUNT
        assert int(region.face_indices[-1]) < CAT_TEMPLATE_FACE_COUNT
        u0, v0, u1, v1 = region.uv_bounds
        assert 0.0 <= u0 <= u1 <= 1.0
        assert 0.0 <= v0 <= v1 <= 1.0
        assert region.detail_weight > 0.0
        assert region.texture_priority >= 0

    assert regions["face"].detail_weight > regions["torso"].detail_weight
    assert regions["left_eye"].texture_priority > regions["torso"].texture_priority
    assert regions["paws"].semantic == "paw"


def test_template_region_manifest_is_serializable_and_stable_after_npz_round_trip(tmp_path) -> None:
    template = create_placeholder_cat_template()
    path = tmp_path / "cat_template.npz"
    template.save_npz(path)

    loaded = load_cat_template_npz(path)
    manifest = loaded.region_manifest(include_indices=True)

    assert manifest["schema_version"] == "1.0"
    assert manifest["topology"] == {
        "vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
        "face_count": CAT_TEMPLATE_FACE_COUNT,
        "uv_vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
        "uv_face_count": CAT_TEMPLATE_FACE_COUNT,
    }
    region_names = [region["name"] for region in manifest["regions"]]
    assert region_names == [region.name for region in template.regions]
    face_entry = next(region for region in manifest["regions"] if region["name"] == "face")
    assert face_entry["vertex_count"] >= template.region("head").vertex_indices.size
    assert face_entry["texture_priority"] == 100
    assert face_entry["vertex_indices"]
    assert face_entry["face_indices"]


def test_template_region_manifest_v2_has_required_addressable_regions() -> None:
    template = create_placeholder_cat_template()
    manifest = template.region_manifest_v2(include_indices=True)

    assert manifest["schema_version"] == "2.0"
    assert manifest["source"] == "opencat3d.cat_template_region_contract_v2"
    assert manifest["target_representation"] == "fixed_topology_cat_template"
    assert manifest["topology_changed"] is False
    assert manifest["reference_contract"] == {
        "schema_version": "1.0",
        "source": "opencat3d.cat_template_region_v2_reference_contract",
        "fixed_topology": True,
        "base_region_names": [region.name for region in template.regions],
        "face_uv_island_names": [region.name for region in template.face_uv_regions],
        "region_bounds_uv_island_source": (
            "opencat3d.cat_template_region_contract_v2_bounds"
        ),
    }
    assert manifest["uv_island_contract"] == {
        "schema_version": "1.0",
        "source": "opencat3d.cat_template_region_uv_island_contract_v1",
        "region_scoped": True,
        "fixed_topology": True,
    }
    assert manifest["topology"] == {
        "vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
        "face_count": CAT_TEMPLATE_FACE_COUNT,
        "uv_vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
        "uv_face_count": CAT_TEMPLATE_FACE_COUNT,
    }
    assert manifest["required_region_names"] == list(CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES)
    assert manifest["validation"]["valid"] is True

    regions = manifest["regions"]
    by_name = {region["name"]: region for region in regions}
    assert set(CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES).issubset(by_name)
    assert manifest["region_count"] == len(regions)

    for region in regions:
        assert region["vertex_count"] > 0
        assert region["face_count"] > 0
        assert region["source_regions"]
        assert region["vertex_indices"]
        assert region["face_indices"]
        u0, v0, u1, v1 = region["uv_bounds"]
        assert 0.0 <= u0 <= u1 <= 1.0
        assert 0.0 <= v0 <= v1 <= 1.0
        assert region["uv_island_count"] == len(region["uv_islands"])
        assert region["uv_island_names"] == [
            island["name"] for island in region["uv_islands"]
        ]
        assert region["primary_uv_island"] in region["uv_island_names"]
        for island in region["uv_islands"]:
            iu0, iv0, iu1, iv1 = island["uv_bounds"]
            assert 0.0 <= iu0 < iu1 <= 1.0
            assert 0.0 <= iv0 < iv1 <= 1.0
            assert island["uv_area"] > 0.0
            assert island["source"]
            assert island["target_texel_density"] > 0.0
        assert region["detail_weight"] > 0.0
        assert region["texture_priority"] >= 0

    assert by_name["face"]["uv_island_names"] == [
        "face",
        "left_eye",
        "right_eye",
        "nose",
        "muzzle",
        "mouth_line",
        "left_ear_inner",
        "right_ear_inner",
    ]
    assert by_name["eyes"]["source_regions"] == ["left_eye", "right_eye"]
    assert by_name["eyes"]["uv_island_names"] == ["left_eye", "right_eye"]
    assert by_name["eyes"]["symmetry_group"] == "bilateral_eye_pair"
    assert by_name["ear_interiors"]["source_regions"] == [
        "left_ear_inner",
        "right_ear_inner",
    ]
    assert by_name["ear_interiors"]["uv_island_names"] == [
        "left_ear_inner",
        "right_ear_inner",
    ]
    assert by_name["muzzle"]["uv_island_names"] == ["muzzle", "mouth_line"]
    assert by_name["nose"]["uv_island_names"] == ["nose"]
    assert by_name["belly"]["source_regions"] == ["torso"]
    assert by_name["belly"]["uv_island_names"] == ["belly_region"]
    assert by_name["belly"]["uv_islands"][0]["source"] == (
        "opencat3d.cat_template_region_contract_v2_bounds"
    )
    assert by_name["paws"]["uv_island_names"] == ["paws_region"]
    assert by_name["tail"]["uv_island_names"] == ["tail_region"]
    assert by_name["torso"]["uv_island_names"] == ["torso_region"]
    assert by_name["limbs"]["uv_island_names"] == ["limbs_region"]
    assert by_name["belly"]["semantic"] == "belly"
    assert by_name["belly"]["vertex_count"] < by_name["torso"]["vertex_count"]
    assert by_name["face"]["detail_weight"] > by_name["torso"]["detail_weight"]
    assert by_name["nose"]["texture_priority"] >= by_name["muzzle"]["texture_priority"]


def test_template_region_manifest_v2_validator_reports_missing_required_region() -> None:
    template = create_placeholder_cat_template()
    manifest = template.region_manifest_v2()
    manifest["regions"] = [
        region for region in manifest["regions"] if region["name"] != "nose"
    ]
    manifest["region_count"] = len(manifest["regions"])

    validation = validate_cat_template_region_manifest_v2(manifest)

    assert validation["valid"] is False
    assert validation["required_regions_present"] is False
    assert validation["missing_required_regions"] == ["nose"]


def test_template_region_manifest_v2_validator_rejects_missing_uv_islands() -> None:
    template = create_placeholder_cat_template()
    manifest = template.region_manifest_v2()
    first_region = manifest["regions"][0]
    assert isinstance(first_region, dict)
    first_region["uv_island_count"] = 0
    first_region["uv_island_names"] = []
    first_region["uv_islands"] = []

    validation = validate_cat_template_region_manifest_v2(manifest)

    assert validation["valid"] is False
    assert validation["uv_islands_declared"] is False
    assert validation["primary_uv_islands_declared"] is False


def test_template_region_manifest_v2_validator_rejects_unknown_source_region() -> None:
    template = create_placeholder_cat_template()
    manifest = template.region_manifest_v2()
    nose = next(region for region in manifest["regions"] if region["name"] == "nose")
    assert isinstance(nose, dict)
    nose["source_regions"] = ["synthetic_guess_region"]

    validation = validate_cat_template_region_manifest_v2(manifest)

    assert validation["valid"] is False
    assert validation["source_regions_declared"] is True
    assert validation["reference_contract_declared"] is True
    assert validation["source_region_references_valid"] is False


def test_template_region_manifest_v2_validator_rejects_unknown_uv_island() -> None:
    template = create_placeholder_cat_template()
    manifest = template.region_manifest_v2()
    eyes = next(region for region in manifest["regions"] if region["name"] == "eyes")
    assert isinstance(eyes, dict)
    left_eye = eyes["uv_islands"][0]
    assert isinstance(left_eye, dict)
    left_eye["name"] = "generated_eye_texture"
    eyes["uv_island_names"] = ["generated_eye_texture", "right_eye"]

    validation = validate_cat_template_region_manifest_v2(manifest)

    assert validation["valid"] is False
    assert validation["uv_islands_declared"] is True
    assert validation["reference_contract_declared"] is True
    assert validation["uv_island_references_valid"] is False


def test_template_face_uv_region_manifest_is_stable_and_feature_addressable() -> None:
    template = create_placeholder_cat_template()
    manifest = template.face_uv_region_manifest()

    assert manifest["schema_version"] == "1.0"
    assert manifest["source"] == "opencat3d.cat_template_face_uv_region_contract"
    assert manifest["topology"] == {
        "vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
        "face_count": CAT_TEMPLATE_FACE_COUNT,
        "uv_vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
        "uv_face_count": CAT_TEMPLATE_FACE_COUNT,
    }
    regions = manifest["regions"]
    assert isinstance(regions, list)
    assert manifest["region_count"] == len(regions)
    region_names = [region["name"] for region in regions]
    assert region_names == [
        "face",
        "left_eye",
        "right_eye",
        "nose",
        "muzzle",
        "mouth_line",
        "left_ear_inner",
        "right_ear_inner",
    ]
    total_area = 0.0
    for region in regions:
        u0, v0, u1, v1 = region["uv_bounds"]
        assert 0.0 <= u0 < u1 <= 1.0
        assert 0.0 <= v0 < v1 <= 1.0
        total_area += (u1 - u0) * (v1 - v0)
        assert region["detail_weight"] > 0.0
        assert region["target_texel_density"] > 0.0
        assert region["source_vertex_count"] > 0
        assert region["source_face_count"] > 0

    assert manifest["total_uv_area"] == pytest.approx(total_area)
    by_name = {region["name"]: region for region in regions}
    assert by_name["left_eye"]["target_texel_density"] > by_name["face"][
        "target_texel_density"
    ]
    assert by_name["nose"]["feature_names"] == ["nose"]
    assert "mouth_line" in by_name["muzzle"]["feature_names"]
    assert by_name["left_ear_inner"]["semantic"] == "ear_inner"
    eye_v = 0.5 * (
        by_name["left_eye"]["uv_center"][1] + by_name["right_eye"]["uv_center"][1]
    )
    nose_v = by_name["nose"]["uv_center"][1]
    muzzle_v = by_name["muzzle"]["uv_center"][1]
    mouth_v = by_name["mouth_line"]["uv_center"][1]
    ear_v = 0.5 * (
        by_name["left_ear_inner"]["uv_center"][1]
        + by_name["right_ear_inner"]["uv_center"][1]
    )
    assert ear_v > eye_v > nose_v > muzzle_v > mouth_v


def test_template_feature_anchor_manifest_is_stable_and_addressable() -> None:
    template = create_placeholder_cat_template()
    manifest = template.feature_anchor_manifest(include_indices=True)

    assert manifest["schema_version"] == "1.0"
    assert manifest["source"] == "opencat3d.cat_template_feature_anchor_contract_v1"
    assert manifest["target_representation"] == "fixed_topology_cat_template"
    assert manifest["fixed_topology"] is True
    assert manifest["topology"] == {
        "vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
        "face_count": CAT_TEMPLATE_FACE_COUNT,
        "uv_vertex_count": CAT_TEMPLATE_VERTEX_COUNT,
        "uv_face_count": CAT_TEMPLATE_FACE_COUNT,
    }
    assert manifest["required_anchor_names"] == list(
        CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES
    )
    assert manifest["anchor_count"] == len(manifest["anchors"])
    assert manifest["validation"]["valid"] is True

    anchors = manifest["anchors"]
    by_name = {anchor["name"]: anchor for anchor in anchors}
    assert set(CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES).issubset(by_name)
    for anchor in anchors:
        assert anchor["region_name"] in manifest["region_reference"]["region_names"]
        assert anchor["reconstruction_role"]
        assert anchor["support_radius"] > 0.0
        assert anchor["support_vertex_count"] == len(anchor["support_vertex_indices"])
        assert anchor["support_vertex_count"] > 0
        assert 0 <= anchor["closest_vertex_index"] < CAT_TEMPLATE_VERTEX_COUNT
        assert len(anchor["anchor_xyz"]) == 3
        assert len(anchor["closest_vertex_xyz"]) == 3

    assert by_name["left_eye_center"]["symmetry_group"] == "bilateral_eye_pair"
    assert by_name["right_eye_center"]["symmetry_group"] == "bilateral_eye_pair"
    assert by_name["left_eye_center"]["closest_vertex_xyz"][0] < 0.0
    assert by_name["right_eye_center"]["closest_vertex_xyz"][0] > 0.0
    assert by_name["tail_tip"]["anchor_xyz"][2] < by_name["tail_base"]["anchor_xyz"][2]


def test_template_feature_anchor_manifest_validator_rejects_unknown_region() -> None:
    template = create_placeholder_cat_template()
    manifest = template.feature_anchor_manifest()
    nose = next(anchor for anchor in manifest["anchors"] if anchor["name"] == "nose_tip")
    assert isinstance(nose, dict)
    nose["region_name"] = "generated_region"

    validation = validate_cat_template_feature_anchor_manifest(manifest)

    assert validation["valid"] is False
    assert validation["region_references_valid"] is False


def test_placeholder_template_faces_are_outward_wound_by_component() -> None:
    template = create_placeholder_cat_template()
    vertices = template.vertices_canonical
    faces = template.faces

    for face_indices in _face_components(faces, vertices.shape[0]):
        component_faces = faces[face_indices]
        component_vertices = np.unique(component_faces.reshape(-1))
        center = np.mean(vertices[component_vertices], axis=0)
        tri = vertices[component_faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        face_centers = np.mean(tri, axis=1)
        outward = np.sum(normals * (face_centers - center[None, :]), axis=1)
        assert np.all(outward > 0.0)


def test_template_shape_basis_is_deterministic_and_validates_beta() -> None:
    template = create_placeholder_cat_template()
    beta = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)

    shaped_a = template.shaped_vertices(beta)
    shaped_b = template.shaped_vertices(beta)

    np.testing.assert_allclose(shaped_a, shaped_b)
    assert not np.allclose(shaped_a, template.vertices_canonical)
    with pytest.raises(ValueError):
        template.shaped_vertices(np.asarray([1.0, 2.0]))


def test_template_npz_round_trip(tmp_path) -> None:
    template = create_placeholder_cat_template()
    path = tmp_path / "cat_template.npz"

    template.save_npz(path)
    loaded = load_cat_template_npz(path)

    np.testing.assert_allclose(loaded.vertices_canonical, template.vertices_canonical)
    np.testing.assert_array_equal(loaded.faces, template.faces)
    np.testing.assert_allclose(loaded.shape_basis, template.shape_basis)


def _face_components(faces: np.ndarray, vertex_count: int) -> list[np.ndarray]:
    vertex_to_faces: list[list[int]] = [[] for _ in range(vertex_count)]
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_to_faces[int(vertex_index)].append(face_index)

    seen = np.zeros(faces.shape[0], dtype=np.bool_)
    components: list[np.ndarray] = []
    for start in range(faces.shape[0]):
        if bool(seen[start]):
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            face_index = stack.pop()
            component.append(face_index)
            for vertex_index in faces[face_index]:
                for neighbor in vertex_to_faces[int(vertex_index)]:
                    if not bool(seen[neighbor]):
                        seen[neighbor] = True
                        stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components
