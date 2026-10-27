"""Deterministic mesh export and reload validation."""

from __future__ import annotations

import json
import shutil
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opencat3d.export.skeleton import write_skeleton_metadata
from opencat3d.fitting.optimizer import load_fit_problem, load_fit_state
from opencat3d.geometry.detail import apply_bounded_face_detail_deformation
from opencat3d.template.cat_template import (
    validate_cat_template_feature_anchor_manifest,
    validate_cat_template_region_manifest_v2,
)
from opencat3d.template.skeleton import JOINT_NAMES
from opencat3d.textures.basecolor import texture_size_for_mode, write_observed_basecolor_texture
from opencat3d.textures.pbr import write_neutral_pbr_textures


@dataclass(frozen=True)
class Mesh:
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]


@dataclass(frozen=True)
class MeshExportResult:
    obj_path: Path | None
    ply_path: Path | None
    glb_path: Path | None
    validation_path: Path
    vertex_count: int
    face_count: int
    skeleton_path: Path | None = None
    idle_animation_path: Path | None = None


@dataclass(frozen=True)
class GlbStructureReport:
    path: Path
    success: bool
    checks: dict[str, bool]
    vertex_count: int
    face_count: int


def export_fitted_canonical_mesh(
    project_dir: Path,
    *,
    mode: str = "synthetic",
    formats: tuple[str, ...] = ("obj", "ply", "glb"),
    basecolor_seed_texture_dir: Path | None = None,
    force: bool = False,
) -> MeshExportResult:
    project_dir = Path(project_dir).resolve()
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported export mode: {mode}")
    normalized_formats = tuple(format_name.lower() for format_name in formats)
    unsupported = sorted(set(normalized_formats).difference({"obj", "ply", "glb"}))
    if unsupported:
        raise ValueError(f"unsupported mesh export format(s): {unsupported}")
    if not normalized_formats:
        raise ValueError("at least one export format is required")

    problem = load_fit_problem(project_dir, mode=mode)
    fit_state = load_fit_state(
        project_dir,
        mode=mode,
        expected_frame_count=len(problem.observations),
        expected_joint_count=int(problem.template.joints_rest.shape[0]),
    )
    beta = np.asarray(fit_state.beta, dtype=np.float64)
    root_translations = np.asarray(fit_state.root_translations, dtype=np.float64)

    base_vertices = problem.template.shaped_vertices(beta)
    geometry_detail: dict[str, object] = {
        "applied": False,
        "source": "none",
        "vertex_count": int(base_vertices.shape[0]),
        "deformed_vertex_count": 0,
        "max_displacement": 0.0,
        "mean_displacement": 0.0,
    }
    if mode == "raw":
        detail_result = apply_bounded_face_detail_deformation(problem.template, base_vertices)
        base_vertices = detail_result.vertices
        geometry_detail = detail_result.manifest
    face_uv_regions = problem.template.face_uv_region_manifest()
    template_regions_v2 = problem.template.region_manifest_v2()
    template_feature_anchors = problem.template.feature_anchor_manifest()

    mesh = Mesh(vertices=base_vertices, faces=problem.template.faces)
    validate_mesh(mesh)

    output_dir = project_dir / "outputs" / "avatar"
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} already exists; pass force=True to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_path: Path | None = None
    ply_path: Path | None = None
    glb_path: Path | None = None
    basecolor_path: Path | None = None
    idle_animation_path: Path | None = None
    pbr_texture_paths: dict[str, str] = {}
    validation: dict[str, object] = {
        "schema_version": "1.0",
        "mode": mode,
        "fit_state": str(fit_state.path),
        "fit_state_source_format": fit_state.source_format,
        "vertex_count": int(mesh.vertices.shape[0]),
        "face_count": int(mesh.faces.shape[0]),
        "geometry_detail": geometry_detail,
        "face_uv_regions": face_uv_regions,
        "face_uv_region_count": int(face_uv_regions["region_count"]),
        "template_regions_v2": template_regions_v2,
        "template_region_v2_count": int(template_regions_v2["region_count"]),
        "template_region_v2_contract_valid": bool(template_regions_v2["validation"]["valid"]),
        "template_feature_anchors": template_feature_anchors,
        "template_feature_anchor_count": int(template_feature_anchors["anchor_count"]),
        "template_feature_anchor_contract_valid": bool(
            template_feature_anchors["validation"]["valid"]
        ),
        "formats": {},
    }
    skeleton_path = output_dir / "skeleton.json"
    skeleton_report = write_skeleton_metadata(
        problem.template,
        beta,
        skeleton_path,
        mode=mode,
    )
    validation["skeleton"] = str(skeleton_path)
    validation["skeleton_metadata_valid"] = skeleton_report.success
    validation["skeleton_checks"] = skeleton_report.checks
    validation["joint_count"] = skeleton_report.joint_count

    if "obj" in normalized_formats:
        obj_path = output_dir / "cat_canonical.obj"
        write_obj(mesh, obj_path)
        reloaded = read_obj(obj_path)
        validate_reloaded_mesh(mesh, reloaded)
        validation["formats"]["obj"] = str(obj_path)  # type: ignore[index]
    if "ply" in normalized_formats:
        ply_path = output_dir / "cat_canonical.ply"
        write_ply(mesh, ply_path)
        reloaded = read_ply(ply_path)
        validate_reloaded_mesh(mesh, reloaded)
        validation["formats"]["ply"] = str(ply_path)  # type: ignore[index]
    if "glb" in normalized_formats:
        glb_path = output_dir / "cat_avatar.glb"
        basecolor_result = write_observed_basecolor_texture(
            project_dir,
            problem,
            mode=mode,
            beta=beta,
            root_translations=root_translations,
            texture_size=texture_size_for_mode(mode, project_dir=project_dir),
            seed_texture_dir=basecolor_seed_texture_dir,
            force=force,
        )
        basecolor_path = basecolor_result.basecolor_path
        pbr_result = write_neutral_pbr_textures(project_dir, force=force)
        pbr_texture_paths = {
            "normal": str(pbr_result.normal_path),
            "roughness": str(pbr_result.roughness_path),
            "ao": str(pbr_result.ao_path),
        }
        write_glb(
            mesh,
            glb_path,
            uv_vertices=problem.template.uv_vertices,
            uv_faces=problem.template.uv_faces,
            basecolor_path=basecolor_path,
            normal_path=pbr_result.normal_path,
            roughness_path=pbr_result.roughness_path,
            ao_path=pbr_result.ao_path,
            joints=problem.template.joints(beta),
            joint_parents=problem.template.joint_parents,
            skin_weights=problem.template.skin_weights,
            joint_names=JOINT_NAMES,
            include_idle_animation=True,
            opencat3d_extras={
                "face_detail_geometry": geometry_detail,
                "face_uv_regions": face_uv_regions,
                "template_regions_v2": template_regions_v2,
                "template_feature_anchors": template_feature_anchors,
            },
        )
        animation_dir = output_dir / "animations"
        animation_dir.mkdir(parents=True, exist_ok=True)
        idle_animation_path = animation_dir / "idle.glb"
        shutil.copyfile(glb_path, idle_animation_path)
        metadata = read_glb_metadata(glb_path)
        structure = validate_glb_structure(
            glb_path,
            expected_vertex_count=int(mesh.vertices.shape[0]),
            expected_face_count=int(mesh.faces.shape[0]),
            expected_textured=True,
            expected_texture_size=basecolor_result.width,
            expected_pbr_textures=True,
            expected_skinned=True,
            expected_animated=True,
        )
        if (
            metadata["vertex_count"] != mesh.vertices.shape[0]
            or metadata["face_count"] != mesh.faces.shape[0]
            or not structure.success
        ):
            raise ValueError("GLB validation failed")
        validation["formats"]["glb"] = str(glb_path)  # type: ignore[index]
        validation["glb_structure_valid"] = structure.success
        validation["glb_structure_checks"] = structure.checks
        validation["glb_face_detail_geometry_valid"] = (
            _glb_face_detail_geometry_valid(metadata, geometry_detail)
            if mode == "raw"
            else True
        )
        validation["glb_face_uv_regions_valid"] = _glb_face_uv_regions_valid(
            metadata,
            face_uv_regions,
        )
        validation["glb_template_regions_v2_valid"] = _glb_template_regions_v2_valid(
            metadata,
            template_regions_v2,
        )
        validation["glb_template_feature_anchors_valid"] = (
            _glb_template_feature_anchors_valid(metadata, template_feature_anchors)
        )
        validation["glb_basecolor_texture"] = str(basecolor_path)
        validation["glb_texture_size"] = int(basecolor_result.width)
        validation["pbr_textures"] = pbr_texture_paths
        validation["pbr_textures_valid"] = bool(
            pbr_result.normal_path.exists()
            and pbr_result.roughness_path.exists()
            and pbr_result.ao_path.exists()
            and pbr_result.generated_texel_fraction == 0.0
        )
        validation["glb_basecolor_texture_valid"] = bool(
            structure.checks.get("expected_textured", False)
            and structure.checks.get("texture_material_valid", False)
            and structure.checks.get("texture_dimensions_valid", False)
            and structure.checks.get("texcoord_accessor_valid", False)
        )
        validation["glb_pbr_textures_valid"] = bool(
            structure.checks.get("expected_pbr_textures", False)
            and structure.checks.get("pbr_material_textures_valid", False)
            and structure.checks.get("pbr_texture_dimensions_valid", False)
            and structure.checks.get("pbr_metallic_factor_zero", False)
        )
        validation["glb_skinning_valid"] = bool(
            structure.checks.get("expected_skinned", False)
            and structure.checks.get("skin_valid", False)
            and structure.checks.get("joint_accessor_valid", False)
            and structure.checks.get("weight_accessor_valid", False)
            and structure.checks.get("inverse_bind_accessor_valid", False)
            and structure.checks.get("weights_normalized", False)
        )
        validation["animations"] = {"idle_glb": str(idle_animation_path)}
        validation["glb_idle_animation_valid"] = bool(
            idle_animation_path.exists()
            and structure.checks.get("expected_animated", False)
            and structure.checks.get("animation_valid", False)
            and structure.checks.get("animation_accessor_valid", False)
            and structure.checks.get("animation_rotations_normalized", False)
        )

    validation_path = output_dir / "mesh_validation.json"
    validation["finite_vertices"] = bool(np.isfinite(mesh.vertices).all())
    validation["valid_faces"] = bool(_faces_are_valid(mesh.faces, mesh.vertices.shape[0]))
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MeshExportResult(
        obj_path=obj_path,
        ply_path=ply_path,
        glb_path=glb_path,
        validation_path=validation_path,
        vertex_count=int(mesh.vertices.shape[0]),
        face_count=int(mesh.faces.shape[0]),
        skeleton_path=skeleton_path,
        idle_animation_path=idle_animation_path,
    )


def write_obj(mesh: Mesh, path: Path) -> None:
    validate_mesh(mesh)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# OpenCat3D canonical mesh\n"]
    for vertex in mesh.vertices:
        lines.append(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
    for face in mesh.faces:
        a, b, c = (int(index) + 1 for index in face)
        lines.append(f"f {a} {b} {c}\n")
    path.write_text("".join(lines), encoding="utf-8")


def read_obj(path: Path) -> Mesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            faces.append([_parse_obj_index(part) for part in parts[1:4]])
    return Mesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )


def _glb_face_detail_geometry_valid(
    metadata: dict[str, Any],
    geometry_detail: dict[str, object],
) -> bool:
    opencat3d = (
        metadata.get("json", {})
        .get("asset", {})
        .get("extras", {})
        .get("opencat3d", {})
    )
    if not isinstance(opencat3d, dict):
        return False
    detail = opencat3d.get("face_detail_geometry", {})
    if not isinstance(detail, dict):
        return False
    expected_anchors = geometry_detail.get("surface_anchors", [])
    actual_anchors = detail.get("surface_anchors", [])
    return (
        detail.get("source") == geometry_detail.get("source")
        and detail.get("surface_anchor_source")
        == geometry_detail.get("surface_anchor_source")
        and int(detail.get("surface_anchor_count", -1))
        == int(geometry_detail.get("surface_anchor_count", -2))
        and isinstance(actual_anchors, list)
        and isinstance(expected_anchors, list)
        and len(actual_anchors) == len(expected_anchors)
    )


def _glb_face_uv_regions_valid(
    metadata: dict[str, Any],
    expected_manifest: dict[str, object],
) -> bool:
    opencat3d = (
        metadata.get("json", {})
        .get("asset", {})
        .get("extras", {})
        .get("opencat3d", {})
    )
    if not isinstance(opencat3d, dict):
        return False
    manifest = opencat3d.get("face_uv_regions", {})
    if not isinstance(manifest, dict):
        return False
    expected_regions = expected_manifest.get("regions", [])
    actual_regions = manifest.get("regions", [])
    if not isinstance(expected_regions, list) or not isinstance(actual_regions, list):
        return False
    expected_names = [
        str(region.get("name"))
        for region in expected_regions
        if isinstance(region, dict)
    ]
    actual_names = [
        str(region.get("name"))
        for region in actual_regions
        if isinstance(region, dict)
    ]
    return (
        manifest.get("schema_version") == expected_manifest.get("schema_version")
        and manifest.get("source") == expected_manifest.get("source")
        and int(manifest.get("region_count", -1))
        == int(expected_manifest.get("region_count", -2))
        and len(actual_regions) == len(expected_regions)
        and actual_names == expected_names
    )


def _glb_template_regions_v2_valid(
    metadata: dict[str, Any],
    expected_manifest: dict[str, object],
) -> bool:
    opencat3d = (
        metadata.get("json", {})
        .get("asset", {})
        .get("extras", {})
        .get("opencat3d", {})
    )
    if not isinstance(opencat3d, dict):
        return False
    manifest = opencat3d.get("template_regions_v2", {})
    if not isinstance(manifest, dict):
        return False
    actual_validation = validate_cat_template_region_manifest_v2(manifest)
    expected_regions = expected_manifest.get("regions", [])
    actual_regions = manifest.get("regions", [])
    if not isinstance(expected_regions, list) or not isinstance(actual_regions, list):
        return False
    expected_names = [
        str(region.get("name"))
        for region in expected_regions
        if isinstance(region, dict)
    ]
    actual_names = [
        str(region.get("name"))
        for region in actual_regions
        if isinstance(region, dict)
    ]
    return (
        manifest.get("schema_version") == expected_manifest.get("schema_version")
        and manifest.get("source") == expected_manifest.get("source")
        and manifest.get("target_representation")
        == expected_manifest.get("target_representation")
        and manifest.get("topology_changed") == expected_manifest.get("topology_changed")
        and manifest.get("required_region_names")
        == expected_manifest.get("required_region_names")
        and int(manifest.get("region_count", -1))
        == int(expected_manifest.get("region_count", -2))
        and len(actual_regions) == len(expected_regions)
        and actual_names == expected_names
        and actual_validation.get("valid") is True
    )


def _glb_template_feature_anchors_valid(
    metadata: dict[str, Any],
    expected_manifest: dict[str, object],
) -> bool:
    opencat3d = (
        metadata.get("json", {})
        .get("asset", {})
        .get("extras", {})
        .get("opencat3d", {})
    )
    if not isinstance(opencat3d, dict):
        return False
    manifest = opencat3d.get("template_feature_anchors", {})
    if not isinstance(manifest, dict):
        return False
    actual_validation = validate_cat_template_feature_anchor_manifest(manifest)
    expected_anchors = expected_manifest.get("anchors", [])
    actual_anchors = manifest.get("anchors", [])
    if not isinstance(expected_anchors, list) or not isinstance(actual_anchors, list):
        return False
    expected_names = [
        str(anchor.get("name")) for anchor in expected_anchors if isinstance(anchor, dict)
    ]
    actual_names = [
        str(anchor.get("name")) for anchor in actual_anchors if isinstance(anchor, dict)
    ]
    return (
        manifest.get("schema_version") == expected_manifest.get("schema_version")
        and manifest.get("source") == expected_manifest.get("source")
        and manifest.get("target_representation")
        == expected_manifest.get("target_representation")
        and manifest.get("fixed_topology") == expected_manifest.get("fixed_topology")
        and manifest.get("required_anchor_names")
        == expected_manifest.get("required_anchor_names")
        and int(manifest.get("anchor_count", -1))
        == int(expected_manifest.get("anchor_count", -2))
        and len(actual_anchors) == len(expected_anchors)
        and actual_names == expected_names
        and actual_validation.get("valid") is True
    )


def write_ply(mesh: Mesh, path: Path) -> None:
    validate_mesh(mesh)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply\n",
        "format ascii 1.0\n",
        "comment OpenCat3D canonical mesh\n",
        f"element vertex {mesh.vertices.shape[0]}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        f"element face {mesh.faces.shape[0]}\n",
        "property list uchar int vertex_indices\n",
        "end_header\n",
    ]
    for vertex in mesh.vertices:
        lines.append(f"{vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
    for face in mesh.faces:
        lines.append(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    path.write_text("".join(lines), encoding="utf-8")


def read_ply(path: Path) -> Mesh:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "ply":
        raise ValueError("not an ASCII PLY file")
    vertex_count = 0
    face_count = 0
    header_end = -1
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
        elif len(parts) == 3 and parts[:2] == ["element", "face"]:
            face_count = int(parts[2])
        elif line == "end_header":
            header_end = index
            break
    if header_end < 0:
        raise ValueError("PLY header missing end_header")

    vertex_lines = lines[header_end + 1 : header_end + 1 + vertex_count]
    face_lines = lines[header_end + 1 + vertex_count : header_end + 1 + vertex_count + face_count]
    vertices = np.asarray([[float(value) for value in line.split()[:3]] for line in vertex_lines])
    faces = []
    for line in face_lines:
        parts = line.split()
        if not parts or int(parts[0]) != 3:
            raise ValueError("only triangular PLY faces are supported")
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return Mesh(vertices=vertices.astype(np.float64), faces=np.asarray(faces, dtype=np.int64))


def _vertex_normals(
    vertices: NDArray[np.floating],
    faces: NDArray[np.integer],
) -> NDArray[np.float32]:
    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices_array, dtype=np.float64)
    triangles = vertices_array[faces_array]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    for corner in range(3):
        np.add.at(normals, faces_array[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = lengths[:, 0] <= 1e-12
    lengths[degenerate] = 1.0
    normals = normals / lengths
    normals[degenerate] = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    return normals.astype(np.float32)


def write_glb(
    mesh: Mesh,
    path: Path,
    *,
    uv_vertices: NDArray[np.floating] | None = None,
    uv_faces: NDArray[np.integer] | None = None,
    basecolor_path: Path | None = None,
    normal_path: Path | None = None,
    roughness_path: Path | None = None,
    ao_path: Path | None = None,
    joints: NDArray[np.floating] | None = None,
    joint_parents: NDArray[np.integer] | None = None,
    skin_weights: NDArray[np.floating] | None = None,
    joint_names: Sequence[str] | None = None,
    include_idle_animation: bool = False,
    opencat3d_extras: dict[str, object] | None = None,
) -> None:
    validate_mesh(mesh)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(mesh.vertices, dtype="<f4")
    normals = _vertex_normals(mesh.vertices, mesh.faces).astype("<f4", copy=False)
    flat_faces = np.asarray(mesh.faces.reshape(-1), dtype=np.uint32)
    index_component_type = 5125
    index_bytes = flat_faces.astype("<u4").tobytes()
    if mesh.vertices.shape[0] <= np.iinfo(np.uint16).max:
        index_component_type = 5123
        index_bytes = flat_faces.astype("<u2").tobytes()

    textured = basecolor_path is not None
    pbr_textured = any(path is not None for path in (normal_path, roughness_path, ao_path))
    texcoords: NDArray[np.float32] | None = None
    image_bytes = b""
    normal_image_bytes = b""
    roughness_image_bytes = b""
    ao_image_bytes = b""
    pbr_paths_complete = all(path is not None for path in (normal_path, roughness_path, ao_path))
    if pbr_textured and not pbr_paths_complete:
        raise ValueError("normal_path, roughness_path, and ao_path are required together")
    if pbr_textured and not textured:
        raise ValueError("PBR texture export requires basecolor_path")
    if textured or pbr_textured:
        if uv_vertices is None or uv_faces is None:
            raise ValueError("uv_vertices and uv_faces are required for textured GLB export")
        uv_faces_array = np.asarray(uv_faces, dtype=np.int64)
        if uv_faces_array.shape != mesh.faces.shape or not np.array_equal(
            uv_faces_array,
            mesh.faces,
        ):
            raise ValueError("textured GLB export requires uv_faces to match mesh faces")
        texcoords = np.asarray(uv_vertices, dtype="<f4")
        if texcoords.shape != (mesh.vertices.shape[0], 2):
            raise ValueError("uv_vertices must have shape [V,2]")
        if not np.isfinite(texcoords).all():
            raise ValueError("uv_vertices must be finite")
    if textured:
        image_path = Path(basecolor_path)
        image_bytes = _read_png_bytes(image_path, label="basecolor texture")
    if pbr_textured:
        normal_image_bytes = _read_png_bytes(Path(normal_path), label="normal texture")
        roughness_image_bytes = _read_png_bytes(
            Path(roughness_path),
            label="roughness texture",
        )
        ao_image_bytes = _read_png_bytes(Path(ao_path), label="AO texture")

    skinned = any(
        value is not None for value in (joints, joint_parents, skin_weights, joint_names)
    )
    joint_positions: NDArray[np.float32] | None = None
    joint_parent_indices: NDArray[np.int64] | None = None
    skin_joint_indices: NDArray[np.uint16] | None = None
    skin_joint_weights: NDArray[np.float32] | None = None
    inverse_bind_matrices: NDArray[np.float32] | None = None
    normalized_joint_names: list[str] = []
    if skinned:
        if joints is None or joint_parents is None or skin_weights is None:
            raise ValueError("joints, joint_parents, and skin_weights are required together")
        (
            joint_positions,
            joint_parent_indices,
            skin_joint_indices,
            skin_joint_weights,
            inverse_bind_matrices,
            normalized_joint_names,
        ) = _prepare_skinning_payload(
            vertex_count=int(mesh.vertices.shape[0]),
            joints=joints,
            joint_parents=joint_parents,
            skin_weights=skin_weights,
            joint_names=joint_names,
        )
    if include_idle_animation and not skinned:
        raise ValueError("idle animation requires skinned GLB export")
    animation_times: NDArray[np.float32] | None = None
    animation_rotations: NDArray[np.float32] | None = None
    animation_target_joint_index = -1
    if include_idle_animation:
        (
            animation_times,
            animation_rotations,
            animation_target_joint_index,
        ) = _build_idle_animation_payload(normalized_joint_names)

    position_bytes = positions.tobytes()
    normal_bytes = normals.tobytes()
    joint_bytes = skin_joint_indices.tobytes() if skin_joint_indices is not None else b""
    weight_bytes = skin_joint_weights.tobytes() if skin_joint_weights is not None else b""
    inverse_bind_bytes = (
        inverse_bind_matrices.tobytes() if inverse_bind_matrices is not None else b""
    )
    animation_time_bytes = animation_times.tobytes() if animation_times is not None else b""
    animation_rotation_bytes = (
        animation_rotations.tobytes() if animation_rotations is not None else b""
    )
    (
        position_offset,
        index_offset,
        normal_offset,
        texcoord_offset,
        joint_offset,
        weight_offset,
        inverse_bind_offset,
        animation_time_offset,
        animation_rotation_offset,
        image_offset,
        normal_image_offset,
        roughness_image_offset,
        ao_image_offset,
        bin_blob,
    ) = _build_glb_bin_blob(
        position_bytes=position_bytes,
        index_bytes=index_bytes,
        normal_bytes=normal_bytes,
        texcoord_bytes=texcoords.tobytes() if texcoords is not None else b"",
        joint_bytes=joint_bytes,
        weight_bytes=weight_bytes,
        inverse_bind_bytes=inverse_bind_bytes,
        animation_time_bytes=animation_time_bytes,
        animation_rotation_bytes=animation_rotation_bytes,
        image_bytes=image_bytes,
        normal_image_bytes=normal_image_bytes,
        roughness_image_bytes=roughness_image_bytes,
        ao_image_bytes=ao_image_bytes,
    )

    buffer_views: list[dict[str, object]] = [
        {
            "buffer": 0,
            "byteOffset": position_offset,
            "byteLength": len(position_bytes),
            "target": 34962,
        },
        {
            "buffer": 0,
            "byteOffset": index_offset,
            "byteLength": len(index_bytes),
            "target": 34963,
        },
    ]
    accessors: list[dict[str, object]] = [
        {
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126,
            "count": int(mesh.vertices.shape[0]),
            "type": "VEC3",
            "min": np.min(mesh.vertices, axis=0).astype(float).tolist(),
            "max": np.max(mesh.vertices, axis=0).astype(float).tolist(),
        },
        {
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": index_component_type,
            "count": int(mesh.faces.size),
            "type": "SCALAR",
        },
    ]
    attributes: dict[str, int] = {"POSITION": 0}
    primitive: dict[str, object] = {
        "attributes": attributes,
        "indices": 1,
        "mode": 4,
    }
    texture_payload: dict[str, object] = {}
    skin_payload: dict[str, object] = {}
    animation_payload: dict[str, object] = {}
    if normal_offset >= 0:
        normal_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": normal_offset,
                "byteLength": int(normals.nbytes),
                "target": 34962,
            }
        )
        normal_accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": normal_view_index,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(normals.shape[0]),
                "type": "VEC3",
                "min": np.min(normals, axis=0).astype(float).tolist(),
                "max": np.max(normals, axis=0).astype(float).tolist(),
            }
        )
        attributes["NORMAL"] = normal_accessor_index

    if textured and texcoords is not None and texcoord_offset >= 0 and image_offset >= 0:
        texcoord_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": texcoord_offset,
                "byteLength": int(texcoords.nbytes),
                "target": 34962,
            }
        )
        texcoord_accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": texcoord_view_index,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(texcoords.shape[0]),
                "type": "VEC2",
                "min": np.min(texcoords, axis=0).astype(float).tolist(),
                "max": np.max(texcoords, axis=0).astype(float).tolist(),
            }
        )
        attributes["TEXCOORD_0"] = texcoord_accessor_index

    if (
        skinned
        and joint_positions is not None
        and joint_parent_indices is not None
        and skin_joint_indices is not None
        and skin_joint_weights is not None
        and inverse_bind_matrices is not None
        and joint_offset >= 0
        and weight_offset >= 0
        and inverse_bind_offset >= 0
    ):
        joint_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": joint_offset,
                "byteLength": int(skin_joint_indices.nbytes),
                "target": 34962,
            }
        )
        joint_accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": joint_view_index,
                "byteOffset": 0,
                "componentType": 5123,
                "count": int(skin_joint_indices.shape[0]),
                "type": "VEC4",
                "min": np.min(skin_joint_indices, axis=0).astype(int).tolist(),
                "max": np.max(skin_joint_indices, axis=0).astype(int).tolist(),
            }
        )
        weight_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": weight_offset,
                "byteLength": int(skin_joint_weights.nbytes),
                "target": 34962,
            }
        )
        weight_accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": weight_view_index,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(skin_joint_weights.shape[0]),
                "type": "VEC4",
                "min": np.min(skin_joint_weights, axis=0).astype(float).tolist(),
                "max": np.max(skin_joint_weights, axis=0).astype(float).tolist(),
            }
        )
        inverse_bind_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": inverse_bind_offset,
                "byteLength": int(inverse_bind_matrices.nbytes),
            }
        )
        inverse_bind_accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": inverse_bind_view_index,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(inverse_bind_matrices.shape[0]),
                "type": "MAT4",
            }
        )
        attributes["JOINTS_0"] = joint_accessor_index
        attributes["WEIGHTS_0"] = weight_accessor_index

        joint_node_start = 1
        joint_nodes = _build_joint_nodes(
            joints=joint_positions,
            joint_parents=joint_parent_indices,
            joint_names=normalized_joint_names,
            node_index_offset=joint_node_start,
        )
        root_joint_nodes = [
            joint_node_start + int(index)
            for index, parent in enumerate(joint_parent_indices)
            if int(parent) < 0
        ]
        skin_payload = {
            "joint_nodes": joint_nodes,
            "root_joint_nodes": root_joint_nodes,
            "skins": [
                {
                    "inverseBindMatrices": inverse_bind_accessor_index,
                    "joints": [
                        joint_node_start + index for index in range(int(joint_positions.shape[0]))
                    ],
                    "name": "cat_template_skin",
                    "skeleton": root_joint_nodes[0],
                }
            ],
        }

    if (
        include_idle_animation
        and animation_times is not None
        and animation_rotations is not None
        and animation_time_offset >= 0
        and animation_rotation_offset >= 0
    ):
        animation_time_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": animation_time_offset,
                "byteLength": int(animation_times.nbytes),
            }
        )
        animation_time_accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": animation_time_view_index,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(animation_times.shape[0]),
                "type": "SCALAR",
                "min": [float(np.min(animation_times))],
                "max": [float(np.max(animation_times))],
            }
        )
        animation_rotation_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": animation_rotation_offset,
                "byteLength": int(animation_rotations.nbytes),
            }
        )
        animation_rotation_accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": animation_rotation_view_index,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(animation_rotations.shape[0]),
                "type": "VEC4",
                "min": np.min(animation_rotations, axis=0).astype(float).tolist(),
                "max": np.max(animation_rotations, axis=0).astype(float).tolist(),
            }
        )
        animation_payload = {
            "animations": [
                {
                    "name": "idle",
                    "channels": [
                        {
                            "sampler": 0,
                            "target": {
                                "node": 1 + animation_target_joint_index,
                                "path": "rotation",
                            },
                        }
                    ],
                    "samplers": [
                        {
                            "input": animation_time_accessor_index,
                            "interpolation": "LINEAR",
                            "output": animation_rotation_accessor_index,
                        }
                    ],
                }
            ]
        }

    if textured and image_offset >= 0:
        image_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": image_offset,
                "byteLength": len(image_bytes),
            }
        )
        primitive["material"] = 0
        images: list[dict[str, object]] = [
            {
                "bufferView": image_view_index,
                "mimeType": "image/png",
                "name": "observed_basecolor",
            }
        ]
        textures: list[dict[str, int]] = [{"sampler": 0, "source": 0}]
        pbr_material: dict[str, object] = {
            "baseColorTexture": {"index": 0},
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.8,
        }
        material: dict[str, object] = {
            "name": "observed_pbr_material" if pbr_textured else "observed_basecolor_material",
            "alphaMode": "OPAQUE",
            "doubleSided": False,
            "pbrMetallicRoughness": pbr_material,
        }
        if (
            pbr_textured
            and normal_image_offset >= 0
            and roughness_image_offset >= 0
            and ao_image_offset >= 0
        ):
            normal_view_index = len(buffer_views)
            buffer_views.append(
                {
                    "buffer": 0,
                    "byteOffset": normal_image_offset,
                    "byteLength": len(normal_image_bytes),
                }
            )
            roughness_view_index = len(buffer_views)
            buffer_views.append(
                {
                    "buffer": 0,
                    "byteOffset": roughness_image_offset,
                    "byteLength": len(roughness_image_bytes),
                }
            )
            ao_view_index = len(buffer_views)
            buffer_views.append(
                {
                    "buffer": 0,
                    "byteOffset": ao_image_offset,
                    "byteLength": len(ao_image_bytes),
                }
            )
            images.extend(
                [
                    {
                        "bufferView": normal_view_index,
                        "mimeType": "image/png",
                        "name": "neutral_normal",
                    },
                    {
                        "bufferView": roughness_view_index,
                        "mimeType": "image/png",
                        "name": "neutral_roughness_metallic",
                    },
                    {
                        "bufferView": ao_view_index,
                        "mimeType": "image/png",
                        "name": "neutral_ao",
                    },
                ]
            )
            textures.extend(
                [
                    {"sampler": 0, "source": 1},
                    {"sampler": 0, "source": 2},
                    {"sampler": 0, "source": 3},
                ]
            )
            pbr_material["metallicRoughnessTexture"] = {"index": 2}
            pbr_material["roughnessFactor"] = 1.0
            material["normalTexture"] = {"index": 1, "scale": 1.0}
            material["occlusionTexture"] = {"index": 3, "strength": 1.0}
        texture_payload = {
            "images": images,
            "samplers": [
                {
                    "magFilter": 9729,
                    "minFilter": 9729,
                    "wrapS": 10497,
                    "wrapT": 10497,
                }
            ],
            "textures": textures,
            "materials": [material],
        }

    opencat3d_payload: dict[str, object] = {
        "metric_scale_known": False,
        "canonical_body_length": 1.0,
        "scale_source": "normalized",
        "basecolor_texture_source": "observed" if textured else "none",
        "pbr_texture_source": "neutral_priors" if pbr_textured else "none",
        "unknown_texels_black": False,
        "front_face_winding": "counter_clockwise",
    }
    if opencat3d_extras:
        opencat3d_payload.update(opencat3d_extras)

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "OpenCat3D",
            "extras": {
                "opencat3d": opencat3d_payload
            },
        },
        "scene": 0,
        "scenes": [
            {
                "nodes": [0] + list(skin_payload.get("root_joint_nodes", [])),
            }
        ],
        "nodes": [
            {
                "mesh": 0,
                "name": "cat_canonical",
                **({"skin": 0} if skinned else {}),
            },
            *skin_payload.get("joint_nodes", []),
        ],
        "meshes": [
            {
                "name": "cat_canonical",
                "primitives": [
                    primitive
                ],
            }
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    gltf.update(texture_payload)
    gltf.update(animation_payload)
    if skin_payload:
        gltf["skins"] = skin_payload["skins"]
        gltf["asset"]["extras"]["opencat3d"].update(  # type: ignore[index]
            {
                "fixed_topology_skinning": True,
                "joint_count": len(skin_payload["joint_nodes"]),
                "skin_weight_encoding": "JOINTS_0_WEIGHTS_0",
                "idle_animation": bool(animation_payload),
            }
        )
    json_blob = json.dumps(gltf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    json_blob += b" " * (_align4(len(json_blob)) - len(json_blob))
    total_length = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, total_length))
        handle.write(struct.pack("<I4s", len(json_blob), b"JSON"))
        handle.write(json_blob)
        handle.write(struct.pack("<I4s", len(bin_blob), b"BIN\x00"))
        handle.write(bin_blob)


def _read_png_bytes(path: Path, *, label: str) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"{path} is required for {label}")
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{label} must be a PNG image")
    return data


def _prepare_skinning_payload(
    *,
    vertex_count: int,
    joints: NDArray[np.floating],
    joint_parents: NDArray[np.integer],
    skin_weights: NDArray[np.floating],
    joint_names: Sequence[str] | None,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int64],
    NDArray[np.uint16],
    NDArray[np.float32],
    NDArray[np.float32],
    list[str],
]:
    joint_positions = np.asarray(joints, dtype=np.float32)
    joint_parent_indices = np.asarray(joint_parents, dtype=np.int64)
    weights = np.asarray(skin_weights, dtype=np.float64)
    if joint_positions.ndim != 2 or joint_positions.shape[1] != 3:
        raise ValueError("joints must have shape [J,3]")
    joint_count = int(joint_positions.shape[0])
    if joint_count == 0:
        raise ValueError("at least one joint is required for skinned GLB export")
    if joint_count > np.iinfo(np.uint16).max:
        raise ValueError("skinned GLB export supports at most 65535 joints")
    if joint_parent_indices.shape != (joint_count,):
        raise ValueError("joint_parents must have shape [J]")
    if weights.shape != (vertex_count, joint_count):
        raise ValueError("skin_weights must have shape [V,J]")
    if not np.isfinite(joint_positions).all():
        raise ValueError("joints must be finite")
    if not np.isfinite(weights).all():
        raise ValueError("skin_weights must be finite")
    _validate_joint_parent_tree(joint_parent_indices)
    names = (
        [f"joint_{index:02d}" for index in range(joint_count)]
        if joint_names is None
        else [str(name) for name in joint_names]
    )
    if len(names) != joint_count:
        raise ValueError("joint_names must match joint count")
    joint_indices, joint_weights = _pack_skin_weights(weights)
    inverse_bind_matrices = _inverse_bind_matrices(joint_positions)
    return (
        joint_positions.astype("<f4", copy=False),
        joint_parent_indices,
        joint_indices,
        joint_weights,
        inverse_bind_matrices,
        names,
    )


def _pack_skin_weights(
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.uint16], NDArray[np.float32]]:
    if np.any(weights < -1e-8):
        raise ValueError("skin_weights must be non-negative")
    clipped = np.maximum(weights, 0.0)
    vertex_count, joint_count = clipped.shape
    joint_indices = np.zeros((vertex_count, 4), dtype=np.uint16)
    joint_weights = np.zeros((vertex_count, 4), dtype="<f4")
    tie_breakers = np.arange(joint_count)
    for vertex_index, row in enumerate(clipped):
        if row.sum() <= 0.0:
            raise ValueError("each skin_weights row must have positive mass")
        order = np.lexsort((tie_breakers, -row))[:4]
        selected = row[order]
        selected_sum = selected.sum()
        if selected_sum <= 0.0:
            raise ValueError("each skin_weights row must include a nonzero joint")
        normalized = (selected / selected_sum).astype(np.float32)
        joint_indices[vertex_index, : order.shape[0]] = order.astype(np.uint16)
        joint_weights[vertex_index, : normalized.shape[0]] = normalized
        joint_weights[vertex_index, 0] += np.float32(
            1.0 - joint_weights[vertex_index].sum(dtype=np.float64)
        )
    return joint_indices, joint_weights


def _inverse_bind_matrices(joints: NDArray[np.float32]) -> NDArray[np.float32]:
    matrices = np.zeros((joints.shape[0], 16), dtype="<f4")
    matrices[:, 0] = 1.0
    matrices[:, 5] = 1.0
    matrices[:, 10] = 1.0
    matrices[:, 15] = 1.0
    matrices[:, 12:15] = -joints
    return matrices


def _build_idle_animation_payload(
    joint_names: Sequence[str],
) -> tuple[NDArray[np.float32], NDArray[np.float32], int]:
    if not joint_names:
        raise ValueError("idle animation requires at least one joint")
    target_joint_index = (
        joint_names.index("tail_tip") if "tail_tip" in joint_names else len(joint_names) - 1
    )
    angle = np.deg2rad(8.0)
    half_angle = angle * 0.5
    neutral = np.asarray([0.0, 0.0, 0.0, 1.0], dtype="<f4")
    tail_sway = np.asarray(
        [0.0, np.sin(half_angle), 0.0, np.cos(half_angle)],
        dtype="<f4",
    )
    times = np.asarray([0.0, 0.5, 1.0], dtype="<f4")
    rotations = np.stack([neutral, tail_sway, neutral], axis=0).astype("<f4")
    return times, rotations, target_joint_index


def _build_joint_nodes(
    *,
    joints: NDArray[np.float32],
    joint_parents: NDArray[np.int64],
    joint_names: Sequence[str],
    node_index_offset: int,
) -> list[dict[str, object]]:
    children_by_parent = _children_by_parent(joint_parents)
    nodes: list[dict[str, object]] = []
    for joint_index, parent_index in enumerate(joint_parents):
        parent = int(parent_index)
        translation = joints[joint_index] if parent < 0 else joints[joint_index] - joints[parent]
        node: dict[str, object] = {
            "name": str(joint_names[joint_index]),
            "translation": translation.astype(float).tolist(),
        }
        children = children_by_parent[joint_index]
        if children:
            node["children"] = [node_index_offset + child for child in children]
        nodes.append(node)
    return nodes


def _children_by_parent(joint_parents: NDArray[np.int64]) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(int(joint_parents.shape[0]))]
    for joint_index, parent_index in enumerate(joint_parents):
        parent = int(parent_index)
        if parent >= 0:
            children[parent].append(joint_index)
    return children


def _validate_joint_parent_tree(joint_parents: NDArray[np.int64]) -> None:
    root_count = int(np.count_nonzero(joint_parents < 0))
    if root_count != 1:
        raise ValueError("joint_parents must contain exactly one root")
    joint_count = int(joint_parents.shape[0])
    for joint_index, parent_index in enumerate(joint_parents):
        parent = int(parent_index)
        if parent < 0:
            continue
        if parent >= joint_count or parent == joint_index:
            raise ValueError("joint_parents contains invalid parent indices")
        visited = {joint_index}
        while parent >= 0:
            if parent in visited:
                raise ValueError("joint_parents must be acyclic")
            visited.add(parent)
            parent = int(joint_parents[parent])


def read_glb_metadata(path: Path) -> dict[str, object]:
    gltf, bin_chunk = _read_glb_payload(path)
    accessors = gltf.get("accessors", [])
    if len(accessors) < 2:
        raise ValueError("GLB must contain position and index accessors")
    vertex_count = int(accessors[0]["count"])
    index_count = int(accessors[1]["count"])
    if index_count % 3 != 0:
        raise ValueError("GLB index count must be divisible by 3")
    buffers = gltf.get("buffers", [])
    if not buffers or int(buffers[0]["byteLength"]) != len(bin_chunk):
        raise ValueError("GLB buffer byteLength mismatch")
    return {
        "vertex_count": vertex_count,
        "face_count": index_count // 3,
        "json": gltf,
    }


def validate_glb_structure(
    path: Path,
    *,
    expected_vertex_count: int | None = None,
    expected_face_count: int | None = None,
    expected_textured: bool | None = None,
    expected_texture_size: int | None = None,
    expected_pbr_textures: bool | None = None,
    expected_skinned: bool | None = None,
    expected_animated: bool | None = None,
) -> GlbStructureReport:
    gltf, bin_chunk = _read_glb_payload(path)
    accessors = _list_field(gltf, "accessors")
    buffer_views = _list_field(gltf, "bufferViews")
    buffers = _list_field(gltf, "buffers")
    scenes = _list_field(gltf, "scenes")
    nodes = _list_field(gltf, "nodes")
    meshes = _list_field(gltf, "meshes")
    skins = _list_field(gltf, "skins")
    animations = _list_field(gltf, "animations")
    asset = _dict_field(gltf, "asset")

    primitive = _first_mesh_primitive(meshes)
    position_accessor_index = _primitive_position_accessor_index(primitive)
    index_accessor_index = _int_field(primitive, "indices", default=-1)
    normal_accessor_index = _primitive_attribute_accessor_index(primitive, "NORMAL")
    texcoord_accessor_index = _primitive_texcoord_accessor_index(primitive)
    joint_accessor_index = _primitive_attribute_accessor_index(primitive, "JOINTS_0")
    weight_accessor_index = _primitive_attribute_accessor_index(primitive, "WEIGHTS_0")
    position_accessor = (
        accessors[position_accessor_index]
        if 0 <= position_accessor_index < len(accessors)
        else {}
    )
    index_accessor = (
        accessors[index_accessor_index] if 0 <= index_accessor_index < len(accessors) else {}
    )

    vertex_count = int(position_accessor.get("count", 0)) if position_accessor else 0
    index_count = int(index_accessor.get("count", 0)) if index_accessor else 0
    face_count = index_count // 3 if index_count % 3 == 0 else 0
    decoded_positions = _decode_position_accessor(position_accessor, buffer_views, bin_chunk)
    decoded_indices = _decode_index_accessor(index_accessor, buffer_views, bin_chunk)
    normal_accessor = (
        accessors[normal_accessor_index] if 0 <= normal_accessor_index < len(accessors) else {}
    )
    decoded_normals = _decode_normal_accessor(normal_accessor, buffer_views, bin_chunk)
    texcoord_accessor = (
        accessors[texcoord_accessor_index]
        if 0 <= texcoord_accessor_index < len(accessors)
        else {}
    )
    decoded_texcoords = _decode_texcoord_accessor(texcoord_accessor, buffer_views, bin_chunk)
    texture_present = _gltf_texture_material_present(gltf, primitive)
    texture_valid = _gltf_texture_material_valid(gltf, primitive, buffer_views, len(bin_chunk))
    texture_dimensions_valid = _gltf_material_texture_dimensions_valid(
        gltf,
        primitive,
        buffer_views,
        bin_chunk,
        expected_size=expected_texture_size,
    )
    material_surface_valid = _gltf_material_surface_flags_valid(gltf, primitive)
    texture_checks_required = texture_present or expected_textured is True
    pbr_textures_present = _gltf_pbr_textures_present(gltf, primitive)
    pbr_texture_checks_required = pbr_textures_present or expected_pbr_textures is True
    pbr_texture_dimensions_valid = _gltf_pbr_texture_dimensions_valid(
        gltf,
        primitive,
        buffer_views,
        bin_chunk,
        expected_size=expected_texture_size,
    )
    skin_present = _gltf_skin_present(gltf, primitive, nodes)
    skin_checks_required = skin_present or expected_skinned is True
    skin = skins[0] if skins else {}
    skin_joint_count = _skin_joint_count(skin)
    inverse_bind_accessor_index = _int_field(skin, "inverseBindMatrices", default=-1)
    joint_accessor = (
        accessors[joint_accessor_index] if 0 <= joint_accessor_index < len(accessors) else {}
    )
    weight_accessor = (
        accessors[weight_accessor_index] if 0 <= weight_accessor_index < len(accessors) else {}
    )
    inverse_bind_accessor = (
        accessors[inverse_bind_accessor_index]
        if 0 <= inverse_bind_accessor_index < len(accessors)
        else {}
    )
    decoded_joints = _decode_joint_accessor(joint_accessor, buffer_views, bin_chunk)
    decoded_weights = _decode_weight_accessor(weight_accessor, buffer_views, bin_chunk)
    decoded_inverse_binds = _decode_inverse_bind_accessor(
        inverse_bind_accessor,
        buffer_views,
        bin_chunk,
    )
    animation_present = bool(animations)
    animation_checks_required = animation_present or expected_animated is True
    animation_accessors = _first_animation_accessors(gltf, accessors)
    animation_time_accessor = animation_accessors[0]
    animation_rotation_accessor = animation_accessors[1]
    decoded_animation_times = _decode_animation_time_accessor(
        animation_time_accessor,
        buffer_views,
        bin_chunk,
    )
    decoded_animation_rotations = _decode_animation_rotation_accessor(
        animation_rotation_accessor,
        buffer_views,
        bin_chunk,
    )
    checks = {
        "asset_version_2": asset.get("version") == "2.0",
        "default_scene_valid": _default_scene_is_valid(gltf, scenes, nodes),
        "single_mesh_triangle_primitive": (
            bool(primitive)
            and int(primitive.get("mode", 4)) == 4
            and position_accessor_index >= 0
            and index_accessor_index >= 0
        ),
        "buffer_count_valid": len(buffers) == 1
        and int(buffers[0].get("byteLength", -1)) == len(bin_chunk),
        "buffer_views_valid": _buffer_views_valid(buffer_views, len(bin_chunk)),
        "position_accessor_valid": _position_accessor_valid(position_accessor, buffer_views),
        "index_accessor_valid": _index_accessor_valid(index_accessor, buffer_views),
        "accessor_byte_ranges_valid": (
            _accessor_range_valid(position_accessor, buffer_views, len(bin_chunk))
            and _accessor_range_valid(index_accessor, buffer_views, len(bin_chunk))
            and _accessor_range_valid(normal_accessor, buffer_views, len(bin_chunk))
        ),
        "positions_finite": decoded_positions.size > 0
        and bool(np.isfinite(decoded_positions).all()),
        "indices_in_bounds": decoded_indices.size > 0
        and vertex_count > 0
        and bool(np.all(decoded_indices < vertex_count)),
        "normal_accessor_valid": _normal_accessor_valid(
            normal_accessor,
            buffer_views,
            vertex_count,
        ),
        "normals_finite_normalized": decoded_normals.size > 0
        and bool(np.isfinite(decoded_normals).all())
        and bool(np.allclose(np.linalg.norm(decoded_normals, axis=1), 1.0, atol=1e-5)),
        "triangle_index_count": index_count > 0 and index_count % 3 == 0,
        "expected_vertex_count": expected_vertex_count is None
        or vertex_count == expected_vertex_count,
        "expected_face_count": expected_face_count is None or face_count == expected_face_count,
        "expected_textured": expected_textured is None or texture_present == expected_textured,
        "texture_material_valid": (not texture_checks_required) or texture_valid,
        "texture_dimensions_valid": (
            not texture_checks_required or expected_texture_size is None
        )
        or texture_dimensions_valid,
        "material_surface_opaque_front_facing": (not texture_checks_required)
        or material_surface_valid,
        "expected_pbr_textures": expected_pbr_textures is None
        or pbr_textures_present == expected_pbr_textures,
        "pbr_material_textures_valid": (not pbr_texture_checks_required)
        or _gltf_pbr_textures_valid(gltf, primitive, buffer_views, len(bin_chunk)),
        "pbr_texture_dimensions_valid": (
            not pbr_texture_checks_required or expected_texture_size is None
        )
        or pbr_texture_dimensions_valid,
        "pbr_metallic_factor_zero": (not pbr_texture_checks_required)
        or _gltf_material_metallic_factor_zero(gltf, primitive),
        "texcoord_accessor_valid": (not texture_checks_required)
        or _texcoord_accessor_valid(texcoord_accessor, buffer_views, vertex_count),
        "texcoords_finite": (not texture_checks_required)
        or (decoded_texcoords.size > 0 and bool(np.isfinite(decoded_texcoords).all())),
        "texcoords_in_range": (not texture_checks_required)
        or (
            decoded_texcoords.size > 0
            and bool(np.all(decoded_texcoords >= -1e-6))
            and bool(np.all(decoded_texcoords <= 1.0 + 1e-6))
        ),
        "expected_skinned": expected_skinned is None or skin_present == expected_skinned,
        "skin_valid": (not skin_checks_required)
        or _gltf_skin_valid(gltf, primitive, nodes, accessors),
        "joint_accessor_valid": (not skin_checks_required)
        or _joint_accessor_valid(joint_accessor, buffer_views, vertex_count),
        "weight_accessor_valid": (not skin_checks_required)
        or _weight_accessor_valid(weight_accessor, buffer_views, vertex_count),
        "inverse_bind_accessor_valid": (not skin_checks_required)
        or _inverse_bind_accessor_valid(inverse_bind_accessor, buffer_views, skin_joint_count),
        "skin_accessor_byte_ranges_valid": (not skin_checks_required)
        or (
            _accessor_range_valid(joint_accessor, buffer_views, len(bin_chunk))
            and _accessor_range_valid(weight_accessor, buffer_views, len(bin_chunk))
            and _accessor_range_valid(inverse_bind_accessor, buffer_views, len(bin_chunk))
        ),
        "joint_nodes_valid": (not skin_checks_required)
        or _skin_joint_nodes_valid(gltf, nodes),
        "joints_in_bounds": (not skin_checks_required)
        or (
            decoded_joints.size > 0
            and skin_joint_count > 0
            and bool(np.all(decoded_joints < skin_joint_count))
        ),
        "weights_finite": (not skin_checks_required)
        or (
            decoded_weights.size > 0
            and bool(np.isfinite(decoded_weights).all())
            and bool(np.all(decoded_weights >= -1e-6))
        ),
        "weights_normalized": (not skin_checks_required)
        or (
            decoded_weights.size > 0
            and bool(np.allclose(decoded_weights.sum(axis=1), 1.0, atol=1e-5))
        ),
        "inverse_bind_matrices_finite": (not skin_checks_required)
        or (
            decoded_inverse_binds.size > 0
            and bool(np.isfinite(decoded_inverse_binds).all())
        ),
        "expected_animated": expected_animated is None
        or animation_present == expected_animated,
        "animation_valid": (not animation_checks_required)
        or _gltf_animation_valid(gltf, nodes, accessors),
        "animation_accessor_valid": (not animation_checks_required)
        or (
            _animation_time_accessor_valid(animation_time_accessor, buffer_views)
            and _animation_rotation_accessor_valid(animation_rotation_accessor, buffer_views)
            and _int_field(animation_time_accessor, "count")
            == _int_field(animation_rotation_accessor, "count")
        ),
        "animation_accessor_byte_ranges_valid": (not animation_checks_required)
        or (
            _accessor_range_valid(animation_time_accessor, buffer_views, len(bin_chunk))
            and _accessor_range_valid(animation_rotation_accessor, buffer_views, len(bin_chunk))
        ),
        "animation_times_finite_monotonic": (not animation_checks_required)
        or (
            decoded_animation_times.size >= 2
            and bool(np.isfinite(decoded_animation_times).all())
            and bool(np.all(np.diff(decoded_animation_times) > 0.0))
            and float(decoded_animation_times[0]) >= 0.0
        ),
        "animation_rotations_normalized": (not animation_checks_required)
        or (
            decoded_animation_rotations.size > 0
            and bool(np.isfinite(decoded_animation_rotations).all())
            and bool(
                np.allclose(
                    np.linalg.norm(decoded_animation_rotations, axis=1),
                    1.0,
                    atol=1e-5,
                )
            )
        ),
    }
    return GlbStructureReport(
        path=path,
        success=all(checks.values()),
        checks=checks,
        vertex_count=vertex_count,
        face_count=face_count,
    )


def _read_glb_payload(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as handle:
        payload = handle.read()
    if len(payload) < 20:
        raise ValueError("GLB file is too small")
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError("not a glTF 2.0 GLB")
    if total_length != len(payload):
        raise ValueError("GLB total length mismatch")

    offset = 12
    chunks: dict[str, bytes] = {}
    chunk_order: list[str] = []
    while offset < total_length:
        if offset + 8 > total_length:
            raise ValueError("GLB chunk header exceeds total length")
        chunk_length, chunk_type = struct.unpack_from("<I4s", payload, offset)
        offset += 8
        if offset + chunk_length > total_length:
            raise ValueError("GLB chunk payload exceeds total length")
        chunk_data = payload[offset : offset + chunk_length]
        offset += chunk_length
        name = chunk_type.decode("ascii").rstrip("\x00")
        if name in chunks:
            raise ValueError("GLB contains duplicate chunks")
        chunk_order.append(name)
        chunks[name] = chunk_data
    if offset != total_length:
        raise ValueError("GLB chunk parsing did not consume total length")
    if "JSON" not in chunks or "BIN" not in chunks:
        raise ValueError("GLB must contain JSON and BIN chunks")
    if chunk_order[:2] != ["JSON", "BIN"]:
        raise ValueError("GLB chunks must be ordered as JSON then BIN")

    gltf = json.loads(chunks["JSON"].decode("utf-8").rstrip(" "))
    if not isinstance(gltf, dict):
        raise ValueError("GLB JSON chunk must contain a glTF object")
    return gltf, chunks["BIN"]


def _list_field(value: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = value.get(key, [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _dict_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    raw = value.get(key, {})
    return raw if isinstance(raw, dict) else {}


def _int_field(value: dict[str, Any], key: str, *, default: int = 0) -> int:
    try:
        return int(value.get(key, default))
    except (TypeError, ValueError):
        return default


def _first_mesh_primitive(meshes: list[dict[str, Any]]) -> dict[str, Any]:
    if not meshes:
        return {}
    primitives = meshes[0].get("primitives", [])
    if not isinstance(primitives, list) or not primitives:
        return {}
    primitive = primitives[0]
    return primitive if isinstance(primitive, dict) else {}


def _primitive_position_accessor_index(primitive: dict[str, Any]) -> int:
    return _primitive_attribute_accessor_index(primitive, "POSITION")


def _primitive_texcoord_accessor_index(primitive: dict[str, Any]) -> int:
    return _primitive_attribute_accessor_index(primitive, "TEXCOORD_0")


def _primitive_attribute_accessor_index(primitive: dict[str, Any], name: str) -> int:
    attributes = primitive.get("attributes", {})
    if not isinstance(attributes, dict):
        return -1
    try:
        return int(attributes.get(name, -1))
    except (TypeError, ValueError):
        return -1


def _default_scene_is_valid(
    gltf: dict[str, Any],
    scenes: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> bool:
    scene_index = _int_field(gltf, "scene", default=-1)
    if scene_index < 0 or scene_index >= len(scenes):
        return False
    scene_nodes = scenes[scene_index].get("nodes", [])
    if not isinstance(scene_nodes, list) or not scene_nodes:
        return False
    try:
        node_indices = [int(index) for index in scene_nodes]
    except (TypeError, ValueError):
        return False
    return all(0 <= index < len(nodes) for index in node_indices) and any(
        _int_field(nodes[index], "mesh", default=-1) == 0 for index in node_indices
    )


def _buffer_views_valid(buffer_views: list[dict[str, Any]], bin_length: int) -> bool:
    if len(buffer_views) < 2:
        return False
    for view in buffer_views:
        byte_offset = _int_field(view, "byteOffset")
        byte_length = _int_field(view, "byteLength", default=-1)
        if _int_field(view, "buffer", default=-1) != 0:
            return False
        if byte_offset < 0 or byte_length <= 0 or byte_offset + byte_length > bin_length:
            return False
        if byte_offset % 4 != 0:
            return False
    return True


def _position_accessor_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    return (
        0 <= view_index < len(buffer_views)
        and _int_field(accessor, "componentType", default=-1) == 5126
        and _int_field(accessor, "count") > 0
        and accessor.get("type") == "VEC3"
        and _int_field(buffer_views[view_index], "target", default=-1) == 34962
    )


def _index_accessor_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    return (
        0 <= view_index < len(buffer_views)
        and _int_field(accessor, "componentType", default=-1) in {5123, 5125}
        and _int_field(accessor, "count") > 0
        and accessor.get("type") == "SCALAR"
        and _int_field(buffer_views[view_index], "target", default=-1) == 34963
    )


def _texcoord_accessor_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    expected_vertex_count: int,
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    return (
        0 <= view_index < len(buffer_views)
        and _int_field(accessor, "componentType", default=-1) == 5126
        and _int_field(accessor, "count") == expected_vertex_count
        and accessor.get("type") == "VEC2"
        and _int_field(buffer_views[view_index], "target", default=-1) == 34962
    )


def _normal_accessor_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    expected_vertex_count: int,
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    return (
        0 <= view_index < len(buffer_views)
        and _int_field(accessor, "componentType", default=-1) == 5126
        and _int_field(accessor, "count") == expected_vertex_count
        and accessor.get("type") == "VEC3"
        and _int_field(buffer_views[view_index], "target", default=-1) == 34962
    )


def _joint_accessor_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    expected_vertex_count: int,
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    return (
        0 <= view_index < len(buffer_views)
        and _int_field(accessor, "componentType", default=-1) in {5121, 5123}
        and _int_field(accessor, "count") == expected_vertex_count
        and accessor.get("type") == "VEC4"
        and _int_field(buffer_views[view_index], "target", default=-1) == 34962
    )


def _weight_accessor_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    expected_vertex_count: int,
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    return (
        0 <= view_index < len(buffer_views)
        and _int_field(accessor, "componentType", default=-1) == 5126
        and _int_field(accessor, "count") == expected_vertex_count
        and accessor.get("type") == "VEC4"
        and _int_field(buffer_views[view_index], "target", default=-1) == 34962
    )


def _inverse_bind_accessor_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    expected_joint_count: int,
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    return (
        expected_joint_count > 0
        and 0 <= view_index < len(buffer_views)
        and _int_field(accessor, "componentType", default=-1) == 5126
        and _int_field(accessor, "count") == expected_joint_count
        and accessor.get("type") == "MAT4"
    )


def _animation_time_accessor_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    return (
        0 <= view_index < len(buffer_views)
        and _int_field(accessor, "componentType", default=-1) == 5126
        and _int_field(accessor, "count") >= 2
        and accessor.get("type") == "SCALAR"
    )


def _animation_rotation_accessor_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    return (
        0 <= view_index < len(buffer_views)
        and _int_field(accessor, "componentType", default=-1) == 5126
        and _int_field(accessor, "count") >= 2
        and accessor.get("type") == "VEC4"
    )


def _accessor_range_valid(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_length: int,
) -> bool:
    view_index = _int_field(accessor, "bufferView", default=-1)
    if view_index < 0 or view_index >= len(buffer_views):
        return False
    view = buffer_views[view_index]
    component_size = _component_size(_int_field(accessor, "componentType", default=-1))
    element_count = _element_component_count(str(accessor.get("type", "")))
    count = _int_field(accessor, "count", default=-1)
    if component_size <= 0 or element_count <= 0 or count <= 0:
        return False
    accessor_offset = _int_field(accessor, "byteOffset")
    view_offset = _int_field(view, "byteOffset")
    view_length = _int_field(view, "byteLength", default=-1)
    required_length = accessor_offset + count * element_count * component_size
    return (
        accessor_offset >= 0
        and view_length >= required_length
        and view_offset + required_length <= bin_length
    )


def _decode_position_accessor(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
) -> NDArray[np.float32]:
    if not _position_accessor_valid(accessor, buffer_views) or not _accessor_range_valid(
        accessor,
        buffer_views,
        len(bin_chunk),
    ):
        return np.empty((0, 3), dtype=np.float32)
    view = buffer_views[_int_field(accessor, "bufferView", default=-1)]
    start = _int_field(view, "byteOffset") + _int_field(accessor, "byteOffset")
    count = _int_field(accessor, "count")
    data = np.frombuffer(bin_chunk, dtype="<f4", count=count * 3, offset=start)
    return np.asarray(data, dtype=np.float32).reshape(count, 3)


def _decode_index_accessor(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
) -> NDArray[np.uint32]:
    if not _index_accessor_valid(accessor, buffer_views) or not _accessor_range_valid(
        accessor,
        buffer_views,
        len(bin_chunk),
    ):
        return np.empty((0,), dtype=np.uint32)
    view = buffer_views[_int_field(accessor, "bufferView", default=-1)]
    start = _int_field(view, "byteOffset") + _int_field(accessor, "byteOffset")
    count = _int_field(accessor, "count")
    component_type = _int_field(accessor, "componentType", default=-1)
    dtype = "<u2" if component_type == 5123 else "<u4"
    data = np.frombuffer(bin_chunk, dtype=dtype, count=count, offset=start)
    return np.asarray(data, dtype=np.uint32)


def _decode_texcoord_accessor(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
) -> NDArray[np.float32]:
    count = _int_field(accessor, "count")
    if (
        count <= 0
        or not _texcoord_accessor_valid(accessor, buffer_views, count)
        or not _accessor_range_valid(accessor, buffer_views, len(bin_chunk))
    ):
        return np.empty((0, 2), dtype=np.float32)
    view = buffer_views[_int_field(accessor, "bufferView", default=-1)]
    start = _int_field(view, "byteOffset") + _int_field(accessor, "byteOffset")
    data = np.frombuffer(bin_chunk, dtype="<f4", count=count * 2, offset=start)
    return np.asarray(data, dtype=np.float32).reshape(count, 2)


def _decode_normal_accessor(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
) -> NDArray[np.float32]:
    count = _int_field(accessor, "count")
    if (
        count <= 0
        or not _normal_accessor_valid(accessor, buffer_views, count)
        or not _accessor_range_valid(accessor, buffer_views, len(bin_chunk))
    ):
        return np.empty((0, 3), dtype=np.float32)
    view = buffer_views[_int_field(accessor, "bufferView", default=-1)]
    start = _int_field(view, "byteOffset") + _int_field(accessor, "byteOffset")
    data = np.frombuffer(bin_chunk, dtype="<f4", count=count * 3, offset=start)
    return np.asarray(data, dtype=np.float32).reshape(count, 3)


def _decode_joint_accessor(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
) -> NDArray[np.uint32]:
    count = _int_field(accessor, "count")
    if (
        count <= 0
        or not _joint_accessor_valid(accessor, buffer_views, count)
        or not _accessor_range_valid(accessor, buffer_views, len(bin_chunk))
    ):
        return np.empty((0, 4), dtype=np.uint32)
    view = buffer_views[_int_field(accessor, "bufferView", default=-1)]
    start = _int_field(view, "byteOffset") + _int_field(accessor, "byteOffset")
    component_type = _int_field(accessor, "componentType", default=-1)
    dtype = "<u1" if component_type == 5121 else "<u2"
    data = np.frombuffer(bin_chunk, dtype=dtype, count=count * 4, offset=start)
    return np.asarray(data, dtype=np.uint32).reshape(count, 4)


def _decode_weight_accessor(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
) -> NDArray[np.float32]:
    count = _int_field(accessor, "count")
    if (
        count <= 0
        or not _weight_accessor_valid(accessor, buffer_views, count)
        or not _accessor_range_valid(accessor, buffer_views, len(bin_chunk))
    ):
        return np.empty((0, 4), dtype=np.float32)
    view = buffer_views[_int_field(accessor, "bufferView", default=-1)]
    start = _int_field(view, "byteOffset") + _int_field(accessor, "byteOffset")
    data = np.frombuffer(bin_chunk, dtype="<f4", count=count * 4, offset=start)
    return np.asarray(data, dtype=np.float32).reshape(count, 4)


def _decode_inverse_bind_accessor(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
) -> NDArray[np.float32]:
    count = _int_field(accessor, "count")
    if (
        count <= 0
        or not _inverse_bind_accessor_valid(accessor, buffer_views, count)
        or not _accessor_range_valid(accessor, buffer_views, len(bin_chunk))
    ):
        return np.empty((0, 16), dtype=np.float32)
    view = buffer_views[_int_field(accessor, "bufferView", default=-1)]
    start = _int_field(view, "byteOffset") + _int_field(accessor, "byteOffset")
    data = np.frombuffer(bin_chunk, dtype="<f4", count=count * 16, offset=start)
    return np.asarray(data, dtype=np.float32).reshape(count, 16)


def _decode_animation_time_accessor(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
) -> NDArray[np.float32]:
    count = _int_field(accessor, "count")
    if (
        count <= 0
        or not _animation_time_accessor_valid(accessor, buffer_views)
        or not _accessor_range_valid(accessor, buffer_views, len(bin_chunk))
    ):
        return np.empty((0,), dtype=np.float32)
    view = buffer_views[_int_field(accessor, "bufferView", default=-1)]
    start = _int_field(view, "byteOffset") + _int_field(accessor, "byteOffset")
    data = np.frombuffer(bin_chunk, dtype="<f4", count=count, offset=start)
    return np.asarray(data, dtype=np.float32)


def _decode_animation_rotation_accessor(
    accessor: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
) -> NDArray[np.float32]:
    count = _int_field(accessor, "count")
    if (
        count <= 0
        or not _animation_rotation_accessor_valid(accessor, buffer_views)
        or not _accessor_range_valid(accessor, buffer_views, len(bin_chunk))
    ):
        return np.empty((0, 4), dtype=np.float32)
    view = buffer_views[_int_field(accessor, "bufferView", default=-1)]
    start = _int_field(view, "byteOffset") + _int_field(accessor, "byteOffset")
    data = np.frombuffer(bin_chunk, dtype="<f4", count=count * 4, offset=start)
    return np.asarray(data, dtype=np.float32).reshape(count, 4)


def _gltf_texture_material_present(gltf: dict[str, Any], primitive: dict[str, Any]) -> bool:
    if _int_field(primitive, "material", default=-1) < 0:
        return False
    return bool(_list_field(gltf, "materials") and _list_field(gltf, "textures"))


def _gltf_texture_material_valid(
    gltf: dict[str, Any],
    primitive: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_length: int,
) -> bool:
    materials = _list_field(gltf, "materials")
    textures = _list_field(gltf, "textures")
    images = _list_field(gltf, "images")
    material_index = _int_field(primitive, "material", default=-1)
    if material_index < 0 or material_index >= len(materials):
        return False
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    if not isinstance(pbr, dict):
        return False
    texture_info = pbr.get("baseColorTexture", {})
    return _texture_info_image_valid(
        texture_info,
        textures,
        images,
        buffer_views,
        bin_length,
    )


def _gltf_material_texture_dimensions_valid(
    gltf: dict[str, Any],
    primitive: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
    *,
    expected_size: int | None,
) -> bool:
    if expected_size is None:
        return True
    materials = _list_field(gltf, "materials")
    material_index = _int_field(primitive, "material", default=-1)
    if material_index < 0 or material_index >= len(materials):
        return False
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    if not isinstance(pbr, dict):
        return False
    return _texture_info_image_dimensions_valid(
        pbr.get("baseColorTexture", {}),
        _list_field(gltf, "textures"),
        _list_field(gltf, "images"),
        buffer_views,
        bin_chunk,
        expected_size=expected_size,
    )


def _gltf_material_surface_flags_valid(
    gltf: dict[str, Any],
    primitive: dict[str, Any],
) -> bool:
    materials = _list_field(gltf, "materials")
    material_index = _int_field(primitive, "material", default=-1)
    if material_index < 0 or material_index >= len(materials):
        return False
    material = materials[material_index]
    return material.get("alphaMode") == "OPAQUE" and material.get("doubleSided") is False


def _gltf_pbr_textures_present(gltf: dict[str, Any], primitive: dict[str, Any]) -> bool:
    materials = _list_field(gltf, "materials")
    material_index = _int_field(primitive, "material", default=-1)
    if material_index < 0 or material_index >= len(materials):
        return False
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    if not isinstance(pbr, dict):
        return False
    return (
        isinstance(material.get("normalTexture"), dict)
        or isinstance(material.get("occlusionTexture"), dict)
        or isinstance(pbr.get("metallicRoughnessTexture"), dict)
    )


def _gltf_pbr_textures_valid(
    gltf: dict[str, Any],
    primitive: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_length: int,
) -> bool:
    materials = _list_field(gltf, "materials")
    textures = _list_field(gltf, "textures")
    images = _list_field(gltf, "images")
    material_index = _int_field(primitive, "material", default=-1)
    if material_index < 0 or material_index >= len(materials):
        return False
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    if not isinstance(pbr, dict):
        return False
    texture_infos = (
        material.get("normalTexture"),
        pbr.get("metallicRoughnessTexture"),
        material.get("occlusionTexture"),
    )
    if not all(isinstance(info, dict) for info in texture_infos):
        return False
    texture_indices = [_int_field(info, "index", default=-1) for info in texture_infos]
    if len(set(texture_indices)) != 3:
        return False
    return all(
        _texture_info_image_valid(info, textures, images, buffer_views, bin_length)
        for info in texture_infos
        if isinstance(info, dict)
    )


def _gltf_pbr_texture_dimensions_valid(
    gltf: dict[str, Any],
    primitive: dict[str, Any],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
    *,
    expected_size: int | None,
) -> bool:
    if expected_size is None:
        return True
    materials = _list_field(gltf, "materials")
    textures = _list_field(gltf, "textures")
    images = _list_field(gltf, "images")
    material_index = _int_field(primitive, "material", default=-1)
    if material_index < 0 or material_index >= len(materials):
        return False
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    if not isinstance(pbr, dict):
        return False
    texture_infos = (
        material.get("normalTexture"),
        pbr.get("metallicRoughnessTexture"),
        material.get("occlusionTexture"),
    )
    if not all(isinstance(info, dict) for info in texture_infos):
        return False
    return all(
        _texture_info_image_dimensions_valid(
            info,
            textures,
            images,
            buffer_views,
            bin_chunk,
            expected_size=expected_size,
        )
        for info in texture_infos
        if isinstance(info, dict)
    )


def _gltf_material_metallic_factor_zero(
    gltf: dict[str, Any],
    primitive: dict[str, Any],
) -> bool:
    materials = _list_field(gltf, "materials")
    material_index = _int_field(primitive, "material", default=-1)
    if material_index < 0 or material_index >= len(materials):
        return False
    pbr = materials[material_index].get("pbrMetallicRoughness", {})
    if not isinstance(pbr, dict):
        return False
    try:
        return float(pbr.get("metallicFactor", 1.0)) == 0.0
    except (TypeError, ValueError):
        return False


def _texture_info_image_valid(
    texture_info: object,
    textures: list[dict[str, Any]],
    images: list[dict[str, Any]],
    buffer_views: list[dict[str, Any]],
    bin_length: int,
) -> bool:
    if not isinstance(texture_info, dict):
        return False
    texture_index = _int_field(texture_info, "index", default=-1)
    if texture_index < 0 or texture_index >= len(textures):
        return False
    texture = textures[texture_index]
    image_index = _int_field(texture, "source", default=-1)
    if image_index < 0 or image_index >= len(images):
        return False
    image = images[image_index]
    if image.get("mimeType") != "image/png":
        return False
    view_index = _int_field(image, "bufferView", default=-1)
    if view_index < 0 or view_index >= len(buffer_views):
        return False
    view = buffer_views[view_index]
    byte_offset = _int_field(view, "byteOffset")
    byte_length = _int_field(view, "byteLength", default=-1)
    if byte_offset < 0 or byte_length <= 8 or byte_offset + byte_length > bin_length:
        return False
    return _int_field(view, "buffer", default=-1) == 0


def _texture_info_image_dimensions_valid(
    texture_info: object,
    textures: list[dict[str, Any]],
    images: list[dict[str, Any]],
    buffer_views: list[dict[str, Any]],
    bin_chunk: bytes,
    *,
    expected_size: int,
) -> bool:
    bounds = _texture_info_image_bounds(
        texture_info,
        textures,
        images,
        buffer_views,
        len(bin_chunk),
    )
    if bounds is None:
        return False
    byte_offset, byte_length = bounds
    dimensions = _png_dimensions(bin_chunk[byte_offset : byte_offset + byte_length])
    return dimensions == (expected_size, expected_size)


def _texture_info_image_bounds(
    texture_info: object,
    textures: list[dict[str, Any]],
    images: list[dict[str, Any]],
    buffer_views: list[dict[str, Any]],
    bin_length: int,
) -> tuple[int, int] | None:
    if not isinstance(texture_info, dict):
        return None
    texture_index = _int_field(texture_info, "index", default=-1)
    if texture_index < 0 or texture_index >= len(textures):
        return None
    texture = textures[texture_index]
    image_index = _int_field(texture, "source", default=-1)
    if image_index < 0 or image_index >= len(images):
        return None
    image = images[image_index]
    if image.get("mimeType") != "image/png":
        return None
    view_index = _int_field(image, "bufferView", default=-1)
    if view_index < 0 or view_index >= len(buffer_views):
        return None
    view = buffer_views[view_index]
    byte_offset = _int_field(view, "byteOffset")
    byte_length = _int_field(view, "byteLength", default=-1)
    if byte_offset < 0 or byte_length <= 24 or byte_offset + byte_length > bin_length:
        return None
    if _int_field(view, "buffer", default=-1) != 0:
        return None
    return byte_offset, byte_length


def _png_dimensions(payload: bytes) -> tuple[int, int] | None:
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _gltf_skin_present(
    gltf: dict[str, Any],
    primitive: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> bool:
    return bool(
        _list_field(gltf, "skins")
        or _primitive_attribute_accessor_index(primitive, "JOINTS_0") >= 0
        or _primitive_attribute_accessor_index(primitive, "WEIGHTS_0") >= 0
        or any(_int_field(node, "skin", default=-1) >= 0 for node in nodes)
    )


def _gltf_skin_valid(
    gltf: dict[str, Any],
    primitive: dict[str, Any],
    nodes: list[dict[str, Any]],
    accessors: list[dict[str, Any]],
) -> bool:
    skins = _list_field(gltf, "skins")
    if len(skins) != 1:
        return False
    skin = skins[0]
    joint_nodes = _skin_joint_indices(skin)
    if not joint_nodes:
        return False
    if any(index < 0 or index >= len(nodes) for index in joint_nodes):
        return False
    skeleton = _int_field(skin, "skeleton", default=-1)
    if skeleton not in joint_nodes:
        return False
    inverse_bind_accessor = _int_field(skin, "inverseBindMatrices", default=-1)
    if inverse_bind_accessor < 0 or inverse_bind_accessor >= len(accessors):
        return False
    if _primitive_attribute_accessor_index(primitive, "JOINTS_0") < 0:
        return False
    if _primitive_attribute_accessor_index(primitive, "WEIGHTS_0") < 0:
        return False
    return any(
        _int_field(node, "mesh", default=-1) == 0
        and _int_field(node, "skin", default=-1) == 0
        for node in nodes
    )


def _skin_joint_nodes_valid(gltf: dict[str, Any], nodes: list[dict[str, Any]]) -> bool:
    skins = _list_field(gltf, "skins")
    if len(skins) != 1:
        return False
    skin = skins[0]
    joint_nodes = _skin_joint_indices(skin)
    if not joint_nodes or len(set(joint_nodes)) != len(joint_nodes):
        return False
    for node_index in joint_nodes:
        if node_index < 0 or node_index >= len(nodes):
            return False
        node = nodes[node_index]
        if _int_field(node, "mesh", default=-1) >= 0:
            return False
        translation = node.get("translation", [])
        if not isinstance(translation, list) or len(translation) != 3:
            return False
        try:
            translation_values = np.asarray([float(value) for value in translation])
        except (TypeError, ValueError):
            return False
        if not np.isfinite(translation_values).all():
            return False
        children = node.get("children", [])
        if children is None:
            continue
        if not isinstance(children, list):
            return False
        try:
            child_indices = [int(child) for child in children]
        except (TypeError, ValueError):
            return False
        if any(child not in joint_nodes for child in child_indices):
            return False
    scene_index = _int_field(gltf, "scene", default=-1)
    scenes = _list_field(gltf, "scenes")
    if scene_index < 0 or scene_index >= len(scenes):
        return False
    scene_nodes = scenes[scene_index].get("nodes", [])
    if not isinstance(scene_nodes, list):
        return False
    try:
        scene_node_indices = [int(index) for index in scene_nodes]
    except (TypeError, ValueError):
        return False
    return _int_field(skin, "skeleton", default=-1) in scene_node_indices


def _skin_joint_count(skin: dict[str, Any]) -> int:
    return len(_skin_joint_indices(skin))


def _skin_joint_indices(skin: dict[str, Any]) -> list[int]:
    raw_joints = skin.get("joints", [])
    if not isinstance(raw_joints, list):
        return []
    try:
        return [int(index) for index in raw_joints]
    except (TypeError, ValueError):
        return []


def _gltf_animation_valid(
    gltf: dict[str, Any],
    nodes: list[dict[str, Any]],
    accessors: list[dict[str, Any]],
) -> bool:
    animations = _list_field(gltf, "animations")
    if len(animations) != 1:
        return False
    animation = animations[0]
    samplers = animation.get("samplers", [])
    channels = animation.get("channels", [])
    if not isinstance(samplers, list) or not isinstance(channels, list):
        return False
    if not samplers or not channels:
        return False
    has_rotation_channel = False
    for channel in channels:
        if not isinstance(channel, dict):
            return False
        sampler_index = _int_field(channel, "sampler", default=-1)
        if sampler_index < 0 or sampler_index >= len(samplers):
            return False
        target = channel.get("target", {})
        if not isinstance(target, dict):
            return False
        target_node = _int_field(target, "node", default=-1)
        if target_node < 0 or target_node >= len(nodes):
            return False
        if target.get("path") != "rotation":
            return False
        has_rotation_channel = True
    for sampler in samplers:
        if not isinstance(sampler, dict):
            return False
        input_accessor = _int_field(sampler, "input", default=-1)
        output_accessor = _int_field(sampler, "output", default=-1)
        if input_accessor < 0 or input_accessor >= len(accessors):
            return False
        if output_accessor < 0 or output_accessor >= len(accessors):
            return False
        if sampler.get("interpolation", "LINEAR") != "LINEAR":
            return False
    return has_rotation_channel


def _first_animation_accessors(
    gltf: dict[str, Any],
    accessors: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    animations = _list_field(gltf, "animations")
    if not animations:
        return {}, {}
    samplers = animations[0].get("samplers", [])
    if not isinstance(samplers, list) or not samplers or not isinstance(samplers[0], dict):
        return {}, {}
    sampler = samplers[0]
    input_index = _int_field(sampler, "input", default=-1)
    output_index = _int_field(sampler, "output", default=-1)
    time_accessor = accessors[input_index] if 0 <= input_index < len(accessors) else {}
    rotation_accessor = accessors[output_index] if 0 <= output_index < len(accessors) else {}
    return time_accessor, rotation_accessor


def _component_size(component_type: int) -> int:
    return {5121: 1, 5123: 2, 5125: 4, 5126: 4}.get(component_type, 0)


def _element_component_count(accessor_type: str) -> int:
    return {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}.get(
        accessor_type,
        0,
    )


def validate_mesh(mesh: Mesh) -> None:
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V,3]")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("faces must have shape [F,3]")
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError("mesh must contain vertices and faces")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh vertices must be finite")
    if not _faces_are_valid(mesh.faces, mesh.vertices.shape[0]):
        raise ValueError("mesh faces contain invalid or degenerate indices")


def validate_reloaded_mesh(expected: Mesh, actual: Mesh) -> None:
    validate_mesh(actual)
    if (
        expected.vertices.shape != actual.vertices.shape
        or expected.faces.shape != actual.faces.shape
    ):
        raise ValueError("reloaded mesh shape mismatch")
    np.testing.assert_allclose(actual.vertices, expected.vertices, atol=1e-8)
    np.testing.assert_array_equal(actual.faces, expected.faces)


def _vertex_normals(
    vertices: NDArray[np.floating],
    faces: NDArray[np.integer],
) -> NDArray[np.float32]:
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertex_array, dtype=np.float64)
    tri = vertex_array[face_array]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, face_array[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    fallback = lengths <= 1e-12
    normals[~fallback] /= lengths[~fallback, None]
    if np.any(fallback):
        centered = vertex_array[fallback] - np.mean(vertex_array, axis=0)[None, :]
        centered_lengths = np.linalg.norm(centered, axis=1)
        usable = centered_lengths > 1e-12
        fallback_normals = np.zeros_like(centered, dtype=np.float64)
        fallback_normals[usable] = centered[usable] / centered_lengths[usable, None]
        fallback_normals[~usable, 1] = 1.0
        normals[fallback] = fallback_normals
    return normals.astype(np.float32)


def _parse_obj_index(token: str) -> int:
    raw = token.split("/")[0]
    index = int(raw)
    if index <= 0:
        raise ValueError("only positive OBJ indices are supported")
    return index - 1


def _faces_are_valid(faces: NDArray[np.integer], vertex_count: int) -> bool:
    if faces.size == 0:
        return False
    if np.any(faces < 0) or np.any(faces >= vertex_count):
        return False
    unique_counts = np.asarray([len(set(map(int, face))) for face in faces])
    return bool(np.all(unique_counts == 3))


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _build_glb_bin_blob(
    *,
    position_bytes: bytes,
    index_bytes: bytes,
    normal_bytes: bytes = b"",
    texcoord_bytes: bytes = b"",
    joint_bytes: bytes = b"",
    weight_bytes: bytes = b"",
    inverse_bind_bytes: bytes = b"",
    animation_time_bytes: bytes = b"",
    animation_rotation_bytes: bytes = b"",
    image_bytes: bytes = b"",
    normal_image_bytes: bytes = b"",
    roughness_image_bytes: bytes = b"",
    ao_image_bytes: bytes = b"",
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, bytes]:
    chunks: list[bytes] = []
    offset = 0

    def append(data: bytes) -> int:
        nonlocal offset
        aligned_offset = _align4(offset)
        if aligned_offset > offset:
            chunks.append(b"\x00" * (aligned_offset - offset))
        chunks.append(data)
        offset = aligned_offset + len(data)
        return aligned_offset

    position_offset = append(position_bytes)
    index_offset = append(index_bytes)
    normal_offset = append(normal_bytes) if normal_bytes else -1
    texcoord_offset = append(texcoord_bytes) if texcoord_bytes else -1
    joint_offset = append(joint_bytes) if joint_bytes else -1
    weight_offset = append(weight_bytes) if weight_bytes else -1
    inverse_bind_offset = append(inverse_bind_bytes) if inverse_bind_bytes else -1
    animation_time_offset = append(animation_time_bytes) if animation_time_bytes else -1
    animation_rotation_offset = (
        append(animation_rotation_bytes) if animation_rotation_bytes else -1
    )
    image_offset = append(image_bytes) if image_bytes else -1
    normal_image_offset = append(normal_image_bytes) if normal_image_bytes else -1
    roughness_image_offset = append(roughness_image_bytes) if roughness_image_bytes else -1
    ao_image_offset = append(ao_image_bytes) if ao_image_bytes else -1
    end = _align4(offset)
    if end > offset:
        chunks.append(b"\x00" * (end - offset))
    return (
        position_offset,
        index_offset,
        normal_offset,
        texcoord_offset,
        joint_offset,
        weight_offset,
        inverse_bind_offset,
        animation_time_offset,
        animation_rotation_offset,
        image_offset,
        normal_image_offset,
        roughness_image_offset,
        ao_image_offset,
        b"".join(chunks),
    )
