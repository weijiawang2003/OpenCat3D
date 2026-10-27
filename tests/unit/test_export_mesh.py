import json
import struct

import numpy as np
from PIL import Image
from safetensors.numpy import load_file
from typer.testing import CliRunner

from opencat3d.cli import app
from opencat3d.export.mesh import (
    Mesh,
    export_fitted_canonical_mesh,
    read_glb_metadata,
    read_obj,
    read_ply,
    validate_glb_structure,
    validate_reloaded_mesh,
    write_glb,
    write_obj,
    write_ply,
)
from opencat3d.export.skeleton import validate_skeleton_metadata_file
from opencat3d.fitting.optimizer import (
    OptimizerConfig,
    load_fit_state,
    load_raw_fit_problem,
    load_synthetic_fit_problem,
    run_raw_fit,
    run_synthetic_fit,
)
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.preprocess.keypoints import bootstrap_raw_keypoints_and_pose
from opencat3d.preprocess.segmentation import segment_raw_image_observations
from opencat3d.synthetic.generator import create_synthetic_project
from opencat3d.template.cat_template import (
    CAT_TEMPLATE_FACE_COUNT,
    CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES,
    CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES,
    CAT_TEMPLATE_VERTEX_COUNT,
)


def test_obj_and_ply_round_trip(tmp_path) -> None:
    mesh = Mesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    obj_path = tmp_path / "mesh.obj"
    ply_path = tmp_path / "mesh.ply"

    write_obj(mesh, obj_path)
    write_ply(mesh, ply_path)

    validate_reloaded_mesh(mesh, read_obj(obj_path))
    validate_reloaded_mesh(mesh, read_ply(ply_path))


def test_export_fitted_canonical_mesh_writes_obj_ply_and_validation(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=20),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )
    _write_legacy_fit_state_npz(project_dir, beta=np.asarray([9.0, 9.0, 9.0]))
    safetensors_beta = load_file(
        project_dir / "outputs" / "fitting" / "fit_state.safetensors"
    )["beta"]

    result = export_fitted_canonical_mesh(project_dir)

    assert result.obj_path is not None
    assert result.ply_path is not None
    assert result.glb_path is not None
    assert result.skeleton_path is not None
    assert result.idle_animation_path is not None
    assert result.obj_path.exists()
    assert result.ply_path.exists()
    assert result.glb_path.exists()
    assert result.skeleton_path.exists()
    assert result.idle_animation_path.exists()
    assert result.vertex_count == CAT_TEMPLATE_VERTEX_COUNT
    assert result.face_count == CAT_TEMPLATE_FACE_COUNT
    obj_mesh = read_obj(result.obj_path)
    ply_mesh = read_ply(result.ply_path)
    validate_reloaded_mesh(obj_mesh, ply_mesh)
    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert validation["mode"] == "synthetic"
    assert validation["fit_state"].endswith("outputs/fitting/fit_state.safetensors")
    assert validation["fit_state_source_format"] == "safetensors"
    assert validation["finite_vertices"] is True
    assert validation["valid_faces"] is True
    assert validation["joint_count"] == 11
    assert validation["skeleton"] == str(result.skeleton_path)
    assert validation["skeleton_metadata_valid"] is True
    assert all(validation["skeleton_checks"].values())
    assert validation["glb_structure_valid"] is True
    assert validation["glb_basecolor_texture_valid"] is True
    assert validation["glb_pbr_textures_valid"] is True
    assert validation["glb_skinning_valid"] is True
    assert validation["glb_idle_animation_valid"] is True
    assert validation["pbr_textures_valid"] is True
    assert validation["face_uv_region_count"] == 8
    assert validation["glb_face_uv_regions_valid"] is True
    assert validation["template_region_v2_count"] == 10
    assert validation["template_region_v2_contract_valid"] is True
    assert validation["glb_template_regions_v2_valid"] is True
    assert validation["template_feature_anchor_count"] == len(
        CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES
    )
    assert validation["template_feature_anchor_contract_valid"] is True
    assert validation["glb_template_feature_anchors_valid"] is True
    assert validation["template_regions_v2"]["source"] == (
        "opencat3d.cat_template_region_contract_v2"
    )
    assert validation["template_regions_v2"]["required_region_names"] == list(
        CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES
    )
    assert validation["template_regions_v2"]["validation"]["valid"] is True
    assert validation["template_feature_anchors"]["source"] == (
        "opencat3d.cat_template_feature_anchor_contract_v1"
    )
    assert validation["template_feature_anchors"]["required_anchor_names"] == list(
        CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES
    )
    assert validation["template_feature_anchors"]["validation"]["valid"] is True
    assert validation["face_uv_regions"]["source"] == (
        "opencat3d.cat_template_face_uv_region_contract"
    )
    assert [region["name"] for region in validation["face_uv_regions"]["regions"]] == [
        "face",
        "left_eye",
        "right_eye",
        "nose",
        "muzzle",
        "mouth_line",
        "left_ear_inner",
        "right_ear_inner",
    ]
    assert validation["glb_basecolor_texture"].endswith("outputs/textures/basecolor.png")
    assert validation["pbr_textures"]["normal"].endswith("outputs/textures/normal.png")
    assert validation["pbr_textures"]["roughness"].endswith("outputs/textures/roughness.png")
    assert validation["pbr_textures"]["ao"].endswith("outputs/textures/ao.png")
    assert validation["animations"]["idle_glb"].endswith(
        "outputs/avatar/animations/idle.glb"
    )
    assert all(validation["glb_structure_checks"].values())
    assert validation["formats"]["obj"] == str(result.obj_path)
    assert validation["formats"]["ply"] == str(result.ply_path)
    assert validation["formats"]["glb"] == str(result.glb_path)
    expected_mesh = load_synthetic_fit_problem(project_dir).template.shaped_vertices(
        safetensors_beta
    )
    np.testing.assert_allclose(obj_mesh.vertices, expected_mesh, atol=1e-8)
    glb_metadata = read_glb_metadata(result.glb_path)
    assert glb_metadata["vertex_count"] == CAT_TEMPLATE_VERTEX_COUNT
    assert glb_metadata["face_count"] == CAT_TEMPLATE_FACE_COUNT
    assert glb_metadata["json"]["asset"]["version"] == "2.0"
    assert glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "basecolor_texture_source"
    ] == "observed"
    assert glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "pbr_texture_source"
    ] == "neutral_priors"
    assert glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "unknown_texels_black"
    ] is False
    assert glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "front_face_winding"
    ] == "counter_clockwise"
    face_uv_regions = glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "face_uv_regions"
    ]
    assert face_uv_regions["region_count"] == 8
    assert face_uv_regions["regions"][1]["name"] == "left_eye"
    assert face_uv_regions["regions"][1]["target_texel_density"] > 1.0
    template_regions_v2 = glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "template_regions_v2"
    ]
    assert template_regions_v2["region_count"] == 10
    assert template_regions_v2["validation"]["valid"] is True
    assert template_regions_v2["required_region_names"] == list(
        CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES
    )
    assert [region["name"] for region in template_regions_v2["regions"]] == list(
        CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES
    )
    assert template_regions_v2["regions"][2]["uv_island_names"] == [
        "left_eye",
        "right_eye",
    ]
    template_feature_anchors = glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "template_feature_anchors"
    ]
    assert template_feature_anchors["anchor_count"] == len(
        CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES
    )
    assert template_feature_anchors["validation"]["valid"] is True
    assert template_feature_anchors["anchors"][0]["name"] == "left_eye_center"
    assert template_regions_v2["regions"][4]["uv_island_names"] == [
        "left_ear_inner",
        "right_ear_inner",
    ]
    assert template_regions_v2["regions"][8]["name"] == "belly"
    assert template_regions_v2["regions"][8]["uv_island_names"] == ["belly_region"]
    assert glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "fixed_topology_skinning"
    ] is True
    assert glb_metadata["json"]["asset"]["extras"]["opencat3d"]["idle_animation"] is True
    assert glb_metadata["json"]["asset"]["extras"]["opencat3d"]["joint_count"] == 11
    primitive = glb_metadata["json"]["meshes"][0]["primitives"][0]
    assert primitive["attributes"]["NORMAL"] == 2
    assert primitive["attributes"]["TEXCOORD_0"] == 3
    assert "JOINTS_0" in primitive["attributes"]
    assert "WEIGHTS_0" in primitive["attributes"]
    assert primitive["material"] == 0
    assert glb_metadata["json"]["nodes"][0]["skin"] == 0
    assert glb_metadata["json"]["skins"][0]["skeleton"] == 1
    assert len(glb_metadata["json"]["skins"][0]["joints"]) == 11
    assert glb_metadata["json"]["nodes"][1]["name"] == "root"
    animation = glb_metadata["json"]["animations"][0]
    assert animation["name"] == "idle"
    assert animation["channels"][0]["target"]["path"] == "rotation"
    assert animation["channels"][0]["target"]["node"] in glb_metadata["json"]["skins"][0]["joints"]
    assert animation["samplers"][0]["interpolation"] == "LINEAR"
    inverse_bind_accessor = glb_metadata["json"]["accessors"][
        glb_metadata["json"]["skins"][0]["inverseBindMatrices"]
    ]
    assert inverse_bind_accessor["type"] == "MAT4"
    assert inverse_bind_accessor["count"] == 11
    assert [image["name"] for image in glb_metadata["json"]["images"]] == [
        "observed_basecolor",
        "neutral_normal",
        "neutral_roughness_metallic",
        "neutral_ao",
    ]
    assert all(image["mimeType"] == "image/png" for image in glb_metadata["json"]["images"])
    material = glb_metadata["json"]["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    assert material["alphaMode"] == "OPAQUE"
    assert material["doubleSided"] is False
    assert pbr["baseColorTexture"] == {"index": 0}
    assert pbr["metallicRoughnessTexture"] == {"index": 2}
    assert pbr["metallicFactor"] == 0.0
    assert pbr["roughnessFactor"] == 1.0
    assert material["normalTexture"] == {"index": 1, "scale": 1.0}
    assert material["occlusionTexture"] == {"index": 3, "strength": 1.0}
    glb_structure = validate_glb_structure(
        result.glb_path,
        expected_vertex_count=CAT_TEMPLATE_VERTEX_COUNT,
        expected_face_count=CAT_TEMPLATE_FACE_COUNT,
        expected_textured=True,
        expected_texture_size=256,
        expected_pbr_textures=True,
        expected_skinned=True,
        expected_animated=True,
    )
    assert glb_structure.success
    assert all(glb_structure.checks.values())
    assert glb_structure.checks["texture_dimensions_valid"]
    assert glb_structure.checks["pbr_texture_dimensions_valid"]
    assert glb_structure.checks["normal_accessor_valid"]
    assert glb_structure.checks["normals_finite_normalized"]
    assert glb_structure.checks["material_surface_opaque_front_facing"]
    wrong_texture_size = validate_glb_structure(
        result.glb_path,
        expected_vertex_count=CAT_TEMPLATE_VERTEX_COUNT,
        expected_face_count=CAT_TEMPLATE_FACE_COUNT,
        expected_textured=True,
        expected_texture_size=512,
        expected_pbr_textures=True,
        expected_skinned=True,
        expected_animated=True,
    )
    assert not wrong_texture_size.success
    assert not wrong_texture_size.checks["texture_dimensions_valid"]
    assert not wrong_texture_size.checks["pbr_texture_dimensions_valid"]
    skeleton_report = validate_skeleton_metadata_file(
        result.skeleton_path,
        expected_mode="synthetic",
        expected_vertex_count=CAT_TEMPLATE_VERTEX_COUNT,
        expected_face_count=CAT_TEMPLATE_FACE_COUNT,
        expected_joint_count=11,
    )
    assert skeleton_report.success
    assert all(skeleton_report.checks.values())
    skeleton = json.loads(result.skeleton_path.read_text(encoding="utf-8"))
    assert skeleton["root_joint_index"] == 0
    assert skeleton["parents"] == [-1, 0, 1, 2, 2, 1, 1, 0, 0, 0, 9]
    assert skeleton["joints"][0]["name"] == "root"
    assert skeleton["skin_weights"]["row_sum_min"] == 1.0
    assert skeleton["skin_weights"]["row_sum_max"] == 1.0
    assert len(skeleton["skin_weights"]["entries"]) == CAT_TEMPLATE_VERTEX_COUNT


def test_export_fitted_canonical_mesh_runs_for_raw_fit(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_project(project_dir)

    result = export_fitted_canonical_mesh(project_dir, mode="raw")

    assert result.obj_path is not None
    assert result.ply_path is not None
    assert result.glb_path is not None
    assert result.skeleton_path is not None
    assert result.idle_animation_path is not None
    assert result.vertex_count == CAT_TEMPLATE_VERTEX_COUNT
    assert result.face_count == CAT_TEMPLATE_FACE_COUNT
    assert result.obj_path is not None
    obj_mesh = read_obj(result.obj_path)
    problem = load_raw_fit_problem(project_dir)
    fit_state = load_fit_state(
        project_dir,
        mode="raw",
        expected_frame_count=len(problem.observations),
        expected_joint_count=int(problem.template.joints_rest.shape[0]),
    )
    base_vertices = problem.template.shaped_vertices(fit_state.beta)
    displacement = np.linalg.norm(obj_mesh.vertices - base_vertices, axis=1)
    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert validation["mode"] == "raw"
    assert validation["geometry_detail"]["source"] == "face_detail_region_deformation_v1"
    assert validation["geometry_detail"]["applied"] is True
    assert validation["geometry_detail"]["deformed_vertex_count"] > 0
    assert validation["geometry_detail"]["max_displacement"] <= 0.040
    assert validation["geometry_detail"]["surface_anchor_source"] == (
        "face_surface_anchor_deformation_v1"
    )
    assert validation["geometry_detail"]["surface_anchor_count"] == 10
    assert validation["geometry_detail"]["surface_anchor_vertex_count"] > 0
    assert validation["glb_face_detail_geometry_valid"] is True
    assert int(np.count_nonzero(displacement > 1e-9)) == validation["geometry_detail"][
        "deformed_vertex_count"
    ]
    assert float(np.max(displacement)) <= 0.040 + 1e-8
    assert validation["finite_vertices"] is True
    assert validation["valid_faces"] is True
    assert validation["joint_count"] == 11
    assert validation["skeleton_metadata_valid"] is True
    assert validation["glb_structure_valid"] is True
    assert validation["glb_basecolor_texture_valid"] is True
    assert validation["glb_pbr_textures_valid"] is True
    assert validation["glb_skinning_valid"] is True
    assert validation["glb_idle_animation_valid"] is True
    assert validation["pbr_textures_valid"] is True
    assert validation["face_uv_region_count"] == 8
    assert validation["glb_face_uv_regions_valid"] is True
    assert validation["template_region_v2_count"] == 10
    assert validation["template_region_v2_contract_valid"] is True
    assert validation["glb_template_regions_v2_valid"] is True
    assert validation["template_feature_anchor_count"] == len(
        CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES
    )
    assert validation["template_feature_anchor_contract_valid"] is True
    assert validation["glb_template_feature_anchors_valid"] is True
    assert result.idle_animation_path.exists()
    glb_metadata = read_glb_metadata(result.glb_path)
    assert glb_metadata["face_count"] == CAT_TEMPLATE_FACE_COUNT
    assert glb_metadata["json"]["skins"][0]["name"] == "cat_template_skin"
    assert glb_metadata["json"]["animations"][0]["name"] == "idle"
    assert glb_metadata["json"]["images"][0]["mimeType"] == "image/png"
    glb_face_detail = glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "face_detail_geometry"
    ]
    assert glb_face_detail["surface_anchor_count"] == 10
    glb_face_uv_regions = glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "face_uv_regions"
    ]
    assert glb_face_uv_regions["regions"][3]["name"] == "nose"
    assert glb_face_uv_regions["regions"][3]["feature_names"] == ["nose"]
    glb_template_regions_v2 = glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "template_regions_v2"
    ]
    assert glb_template_regions_v2["validation"]["valid"] is True
    assert glb_template_regions_v2["regions"][0]["name"] == "face"
    assert glb_template_regions_v2["regions"][0]["uv_island_count"] == 8
    assert glb_template_regions_v2["regions"][4]["name"] == "ear_interiors"
    assert glb_template_regions_v2["regions"][4]["uv_island_names"] == [
        "left_ear_inner",
        "right_ear_inner",
    ]
    glb_feature_anchors = glb_metadata["json"]["asset"]["extras"]["opencat3d"][
        "template_feature_anchors"
    ]
    assert glb_feature_anchors["anchor_count"] == len(
        CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES
    )
    assert glb_feature_anchors["anchors"][0]["name"] == "left_eye_center"
    assert any(
        anchor["name"] == "mouth_groove"
        for anchor in glb_face_detail["surface_anchors"]
    )
    assert validate_skeleton_metadata_file(
        result.skeleton_path,
        expected_mode="raw",
        expected_vertex_count=CAT_TEMPLATE_VERTEX_COUNT,
        expected_face_count=CAT_TEMPLATE_FACE_COUNT,
        expected_joint_count=11,
    ).success


def test_export_fitted_canonical_mesh_uses_valid_raw_face_texture_target(tmp_path) -> None:
    project_dir = tmp_path / "project"
    _create_raw_fitted_project(project_dir)
    _write_raw_face_texture_target_report(project_dir, target_size=1024)

    result = export_fitted_canonical_mesh(project_dir, mode="raw")

    assert result.glb_path is not None
    texture_manifest = json.loads(
        (project_dir / "outputs" / "textures" / "texture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert texture_manifest["basecolor"]["width"] == 1024
    assert texture_manifest["basecolor"]["height"] == 1024
    assert texture_manifest["basecolor"]["texture_resolution_source"] == (
        "raw_face_texture_target_1024"
    )
    assert texture_manifest["basecolor"]["face_texture_target_size"] == 1024
    assert texture_manifest["basecolor"]["generated_texel_fraction"] == 0.0
    assert texture_manifest["pbr"]["width"] == 1024
    assert texture_manifest["pbr"]["height"] == 1024
    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert validation["glb_texture_size"] == 1024
    assert validation["glb_structure_checks"]["texture_dimensions_valid"] is True
    assert validation["glb_structure_checks"]["pbr_texture_dimensions_valid"] is True
    assert validation["glb_basecolor_texture_valid"] is True
    assert validation["glb_pbr_textures_valid"] is True


def test_export_fitted_canonical_mesh_falls_back_to_legacy_npz_state(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=2, shape_steps=0),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )
    _write_legacy_fit_state_npz(project_dir)
    (project_dir / "outputs" / "fitting" / "fit_state.safetensors").unlink()

    result = export_fitted_canonical_mesh(project_dir, formats=("obj",))

    assert result.obj_path is not None
    assert result.obj_path.exists()
    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert validation["fit_state"].endswith("outputs/fitting/fit_state.npz")
    assert validation["fit_state_source_format"] == "npz"


def test_validate_glb_structure_rejects_out_of_bounds_indices(tmp_path) -> None:
    mesh = Mesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    glb_path = tmp_path / "mesh.glb"
    write_glb(mesh, glb_path)
    _overwrite_first_glb_index(glb_path, value=99)

    structure = validate_glb_structure(
        glb_path,
        expected_vertex_count=3,
        expected_face_count=1,
    )

    assert not structure.success
    assert not structure.checks["indices_in_bounds"]
    assert structure.checks["expected_vertex_count"]
    assert structure.checks["expected_face_count"]


def test_export_cli_runs_after_synthetic_fit(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "demo"

    create_result = runner.invoke(
        app,
        ["synthetic", "create", str(project_dir), "--num-views", "3", "--image-size", "64"],
    )
    fit_result = runner.invoke(app, ["fit", str(project_dir), "--mode", "synthetic"])
    export_result = runner.invoke(app, ["export", str(project_dir), "--mode", "synthetic"])

    assert create_result.exit_code == 0, create_result.output
    assert fit_result.exit_code == 0, fit_result.output
    assert export_result.exit_code == 0, export_result.output
    assert (project_dir / "outputs" / "avatar" / "cat_canonical.obj").exists()
    assert (project_dir / "outputs" / "avatar" / "cat_canonical.ply").exists()
    assert (project_dir / "outputs" / "avatar" / "cat_avatar.glb").exists()
    assert (project_dir / "outputs" / "avatar" / "skeleton.json").exists()
    assert (project_dir / "outputs" / "textures" / "basecolor.png").exists()
    assert (project_dir / "outputs" / "textures" / "normal.png").exists()
    assert (project_dir / "outputs" / "textures" / "roughness.png").exists()
    assert (project_dir / "outputs" / "textures" / "ao.png").exists()
    assert f"vertices={CAT_TEMPLATE_VERTEX_COUNT}" in export_result.output
    assert "glb=" in export_result.output
    assert "skeleton=" in export_result.output


def test_export_cli_runs_after_raw_fit(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "project"
    _create_raw_initialized_project(project_dir)

    fit_result = runner.invoke(
        app,
        [
            "fit",
            str(project_dir),
            "--mode",
            "raw",
            "--camera-root-steps",
            "2",
        ],
    )
    export_result = runner.invoke(app, ["export", str(project_dir), "--mode", "raw"])

    assert fit_result.exit_code == 0, fit_result.output
    assert export_result.exit_code == 0, export_result.output
    assert (project_dir / "outputs" / "avatar" / "cat_canonical.obj").exists()
    assert (project_dir / "outputs" / "avatar" / "cat_canonical.ply").exists()
    assert (project_dir / "outputs" / "avatar" / "cat_avatar.glb").exists()
    assert (project_dir / "outputs" / "avatar" / "skeleton.json").exists()
    assert (project_dir / "outputs" / "textures" / "basecolor.png").exists()
    assert (project_dir / "outputs" / "textures" / "normal.png").exists()
    assert (project_dir / "outputs" / "textures" / "roughness.png").exists()
    assert (project_dir / "outputs" / "textures" / "ao.png").exists()
    assert f"vertices={CAT_TEMPLATE_VERTEX_COUNT}" in export_result.output


def _create_raw_fitted_project(project_dir) -> None:
    _create_raw_initialized_project(project_dir)
    run_raw_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=2, shape_steps=0, silhouette_weight=1.0),
        initial_root_offset=np.asarray([0.0, 0.0, 0.0]),
    )


def _create_raw_initialized_project(project_dir) -> None:
    input_dir = project_dir.parent / "inputs"
    input_dir.mkdir()
    _write_rgb_png(input_dir / "b.png", width=8, height=8, bbox=(2, 2, 6, 6))
    _write_rgb_png(input_dir / "a.png", width=8, height=8, bbox=(1, 1, 5, 5))
    ingest_image_sources(project_dir, [input_dir], min_frames=2)
    segment_raw_image_observations(project_dir, color_delta_threshold=0.10)
    bootstrap_raw_keypoints_and_pose(project_dir)


def _write_rgb_png(path, *, width: int, height: int, bbox: tuple[int, int, int, int]) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    image[y0:y1, x0:x1, :] = 51
    Image.fromarray(image).save(path)


def _write_raw_face_texture_target_report(project_dir, *, target_size: int) -> None:
    quality_dir = project_dir / "outputs" / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.0",
        "success": True,
        "source": "opencat3d.face_texture_upsampling_targets_v1",
        "output_contract": {
            "target_representation": "metadata_only_future_texture_bake",
            "texture_resampled": False,
            "topology_changed": False,
        },
        "metrics": {
            "target_texture_size": target_size,
            "target_texture_scale": target_size / 512.0,
            "face_texture_target_requested_texel_count": 1000,
            "face_texture_target_allocated_texel_count": 4000,
        },
    }
    (quality_dir / "raw_face_texture_targets.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _overwrite_first_glb_index(path, *, value: int) -> None:
    payload = bytearray(path.read_bytes())
    json_length = struct.unpack_from("<I", payload, 12)[0]
    json_start = 20
    gltf = json.loads(bytes(payload[json_start : json_start + json_length]).decode("utf-8"))
    bin_header_offset = json_start + json_length
    bin_start = bin_header_offset + 8
    index_view = gltf["bufferViews"][gltf["accessors"][1]["bufferView"]]
    index_offset = bin_start + int(index_view["byteOffset"])
    component_type = int(gltf["accessors"][1]["componentType"])
    if component_type == 5123:
        struct.pack_into("<H", payload, index_offset, value)
    else:
        struct.pack_into("<I", payload, index_offset, value)
    path.write_bytes(payload)


def _write_legacy_fit_state_npz(project_dir, *, beta: np.ndarray | None = None) -> None:
    fitting_dir = project_dir / "outputs" / "fitting"
    safetensors_state = load_file(fitting_dir / "fit_state.safetensors")
    arrays = {
        key: np.asarray(value, dtype=np.float64).copy()
        for key, value in safetensors_state.items()
    }
    if beta is not None:
        arrays["beta"] = np.asarray(beta, dtype=np.float64)
    arrays["mode"] = np.asarray("synthetic")
    fit_state_path = fitting_dir / "fit_state.npz"
    np.savez_compressed(fit_state_path, **arrays)
