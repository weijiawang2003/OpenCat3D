"""Fixed-topology CatTemplate mesh used by fitting and export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from opencat3d.camera.conventions import rotation_6d_to_matrix
from opencat3d.template.skeleton import JOINT_PARENTS

CAT_TEMPLATE_VERTEX_COUNT = 958
CAT_TEMPLATE_FACE_COUNT = 1864
CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES = (
    "face",
    "muzzle",
    "eyes",
    "nose",
    "ear_interiors",
    "paws",
    "tail",
    "torso",
    "belly",
    "limbs",
)
CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES = (
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


@dataclass(frozen=True)
class CatTemplateRegion:
    name: str
    semantic: str
    vertex_indices: NDArray[np.int64]
    face_indices: NDArray[np.int64]
    uv_bounds: tuple[float, float, float, float]
    detail_weight: float
    texture_priority: int

    def model_dump(self, *, include_indices: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "semantic": self.semantic,
            "vertex_count": int(self.vertex_indices.size),
            "face_count": int(self.face_indices.size),
            "uv_bounds": [float(value) for value in self.uv_bounds],
            "detail_weight": float(self.detail_weight),
            "texture_priority": int(self.texture_priority),
        }
        if include_indices:
            payload["vertex_indices"] = [int(value) for value in self.vertex_indices]
            payload["face_indices"] = [int(value) for value in self.face_indices]
        return payload


@dataclass(frozen=True)
class CatTemplateRegionV2:
    name: str
    semantic: str
    source_regions: tuple[str, ...]
    vertex_indices: NDArray[np.int64]
    face_indices: NDArray[np.int64]
    uv_bounds: tuple[float, float, float, float]
    uv_islands: tuple[CatTemplateRegionUvIsland, ...]
    detail_weight: float
    texture_priority: int
    reconstruction_role: str
    symmetry_group: str | None = None

    def model_dump(self, *, include_indices: bool = False) -> dict[str, object]:
        uv_island_names = [island.name for island in self.uv_islands]
        payload: dict[str, object] = {
            "name": self.name,
            "semantic": self.semantic,
            "source_regions": list(self.source_regions),
            "vertex_count": int(self.vertex_indices.size),
            "face_count": int(self.face_indices.size),
            "uv_bounds": [float(value) for value in self.uv_bounds],
            "primary_uv_island": uv_island_names[0] if uv_island_names else None,
            "uv_island_count": int(len(self.uv_islands)),
            "uv_island_names": uv_island_names,
            "uv_islands": [island.model_dump() for island in self.uv_islands],
            "detail_weight": float(self.detail_weight),
            "texture_priority": int(self.texture_priority),
            "reconstruction_role": self.reconstruction_role,
            "symmetry_group": self.symmetry_group,
        }
        if include_indices:
            payload["vertex_indices"] = [int(value) for value in self.vertex_indices]
            payload["face_indices"] = [int(value) for value in self.face_indices]
        return payload


@dataclass(frozen=True)
class CatTemplateRegionUvIsland:
    name: str
    semantic: str
    source: str
    source_region: str
    uv_bounds: tuple[float, float, float, float]
    detail_weight: float
    texture_priority: int
    target_texel_density: float
    allocation_role: str
    feature_names: tuple[str, ...]

    def model_dump(self) -> dict[str, object]:
        u0, v0, u1, v1 = self.uv_bounds
        return {
            "name": self.name,
            "semantic": self.semantic,
            "source": self.source,
            "source_region": self.source_region,
            "uv_bounds": [float(value) for value in self.uv_bounds],
            "uv_area": float(max(0.0, u1 - u0) * max(0.0, v1 - v0)),
            "detail_weight": float(self.detail_weight),
            "texture_priority": int(self.texture_priority),
            "target_texel_density": float(self.target_texel_density),
            "allocation_role": self.allocation_role,
            "feature_names": list(self.feature_names),
        }


@dataclass(frozen=True)
class CatTemplateFaceUvRegion:
    name: str
    semantic: str
    source_region: str
    anchor_xyz: tuple[float, float, float]
    uv_center: tuple[float, float]
    uv_radius: tuple[float, float]
    uv_bounds: tuple[float, float, float, float]
    detail_weight: float
    texture_priority: int
    target_texel_density: float
    feature_names: tuple[str, ...]
    source_vertex_count: int
    source_face_count: int

    def model_dump(self) -> dict[str, object]:
        u0, v0, u1, v1 = self.uv_bounds
        return {
            "name": self.name,
            "semantic": self.semantic,
            "source_region": self.source_region,
            "anchor_xyz": [float(value) for value in self.anchor_xyz],
            "uv_center": [float(value) for value in self.uv_center],
            "uv_radius": [float(value) for value in self.uv_radius],
            "uv_bounds": [float(value) for value in self.uv_bounds],
            "uv_area": float(max(0.0, u1 - u0) * max(0.0, v1 - v0)),
            "detail_weight": float(self.detail_weight),
            "texture_priority": int(self.texture_priority),
            "target_texel_density": float(self.target_texel_density),
            "feature_names": list(self.feature_names),
            "source_vertex_count": int(self.source_vertex_count),
            "source_face_count": int(self.source_face_count),
        }


@dataclass(frozen=True)
class CatTemplateFeatureAnchor:
    name: str
    semantic: str
    region_name: str
    reconstruction_role: str
    anchor_xyz: tuple[float, float, float]
    support_radius: float
    closest_vertex_index: int
    closest_vertex_xyz: tuple[float, float, float]
    support_vertex_indices: NDArray[np.int64]
    symmetry_group: str | None = None

    def model_dump(self, *, include_indices: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "semantic": self.semantic,
            "region_name": self.region_name,
            "reconstruction_role": self.reconstruction_role,
            "anchor_xyz": [float(value) for value in self.anchor_xyz],
            "support_radius": float(self.support_radius),
            "closest_vertex_index": int(self.closest_vertex_index),
            "closest_vertex_xyz": [float(value) for value in self.closest_vertex_xyz],
            "support_vertex_count": int(self.support_vertex_indices.size),
            "symmetry_group": self.symmetry_group,
        }
        if include_indices:
            payload["support_vertex_indices"] = [
                int(value) for value in self.support_vertex_indices
            ]
        return payload


@dataclass(frozen=True)
class _RegionSpec:
    name: str
    semantic: str
    center: tuple[float, float, float]
    radii: tuple[float, float, float]
    detail_weight: float
    texture_priority: int


@dataclass(frozen=True)
class _RegionV2Spec:
    name: str
    semantic: str
    source_regions: tuple[str, ...]
    detail_weight: float
    texture_priority: int
    reconstruction_role: str
    symmetry_group: str | None = None
    uv_island_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FaceUvRegionSpec:
    name: str
    semantic: str
    source_region: str
    anchor_xyz: tuple[float, float, float]
    uv_radius: tuple[float, float]
    detail_weight: float
    texture_priority: int
    target_texel_density: float
    feature_names: tuple[str, ...]
    uv_center: tuple[float, float] | None = None


@dataclass(frozen=True)
class _FeatureAnchorSpec:
    name: str
    semantic: str
    region_name: str
    anchor_xyz: tuple[float, float, float]
    support_radius: float
    reconstruction_role: str
    symmetry_group: str | None = None


@dataclass(frozen=True)
class CatTemplate:
    vertices_canonical: NDArray[np.float64]
    faces: NDArray[np.int64]
    joints_rest: NDArray[np.float64]
    skin_weights: NDArray[np.float64]
    joint_parents: NDArray[np.int64]
    uv_vertices: NDArray[np.float64]
    uv_faces: NDArray[np.int64]
    shape_basis: NDArray[np.float64]

    @property
    def regions(self) -> tuple[CatTemplateRegion, ...]:
        return _cat_template_regions(self)

    @property
    def regions_v2(self) -> tuple[CatTemplateRegionV2, ...]:
        return _cat_template_regions_v2(self)

    @property
    def face_uv_regions(self) -> tuple[CatTemplateFaceUvRegion, ...]:
        return _cat_template_face_uv_regions(self)

    @property
    def feature_anchors(self) -> tuple[CatTemplateFeatureAnchor, ...]:
        return _cat_template_feature_anchors(self)

    def region(self, name: str) -> CatTemplateRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"unknown CatTemplate region: {name}")

    def region_manifest(self, *, include_indices: bool = False) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "source": "opencat3d.cat_template_region_contract",
            "topology": {
                "vertex_count": int(self.vertices_canonical.shape[0]),
                "face_count": int(self.faces.shape[0]),
                "uv_vertex_count": int(self.uv_vertices.shape[0]),
                "uv_face_count": int(self.uv_faces.shape[0]),
            },
            "regions": [
                region.model_dump(include_indices=include_indices) for region in self.regions
            ],
        }

    def region_manifest_v2(self, *, include_indices: bool = False) -> dict[str, object]:
        regions = self.regions_v2
        base_regions = self.regions
        face_uv_regions = self.face_uv_regions
        manifest: dict[str, object] = {
            "schema_version": "2.0",
            "source": "opencat3d.cat_template_region_contract_v2",
            "target_representation": "fixed_topology_cat_template",
            "topology_changed": False,
            "reference_contract": {
                "schema_version": "1.0",
                "source": "opencat3d.cat_template_region_v2_reference_contract",
                "fixed_topology": True,
                "base_region_names": [region.name for region in base_regions],
                "face_uv_island_names": [region.name for region in face_uv_regions],
                "region_bounds_uv_island_source": (
                    "opencat3d.cat_template_region_contract_v2_bounds"
                ),
            },
            "uv_island_contract": {
                "schema_version": "1.0",
                "source": "opencat3d.cat_template_region_uv_island_contract_v1",
                "region_scoped": True,
                "fixed_topology": True,
            },
            "topology": {
                "vertex_count": int(self.vertices_canonical.shape[0]),
                "face_count": int(self.faces.shape[0]),
                "uv_vertex_count": int(self.uv_vertices.shape[0]),
                "uv_face_count": int(self.uv_faces.shape[0]),
            },
            "required_region_names": list(CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES),
            "region_count": int(len(regions)),
            "regions": [
                region.model_dump(include_indices=include_indices) for region in regions
            ],
        }
        manifest["validation"] = validate_cat_template_region_manifest_v2(manifest)
        return manifest

    def face_uv_region_manifest(self) -> dict[str, object]:
        regions = self.face_uv_regions
        total_area = sum(
            max(0.0, region.uv_bounds[2] - region.uv_bounds[0])
            * max(0.0, region.uv_bounds[3] - region.uv_bounds[1])
            for region in regions
        )
        return {
            "schema_version": "1.0",
            "source": "opencat3d.cat_template_face_uv_region_contract",
            "topology": {
                "vertex_count": int(self.vertices_canonical.shape[0]),
                "face_count": int(self.faces.shape[0]),
                "uv_vertex_count": int(self.uv_vertices.shape[0]),
                "uv_face_count": int(self.uv_faces.shape[0]),
            },
            "region_count": int(len(regions)),
            "total_uv_area": float(total_area),
            "regions": [region.model_dump() for region in regions],
        }

    def feature_anchor_manifest(self, *, include_indices: bool = False) -> dict[str, object]:
        anchors = self.feature_anchors
        manifest: dict[str, object] = {
            "schema_version": "1.0",
            "source": "opencat3d.cat_template_feature_anchor_contract_v1",
            "target_representation": "fixed_topology_cat_template",
            "fixed_topology": True,
            "topology": {
                "vertex_count": int(self.vertices_canonical.shape[0]),
                "face_count": int(self.faces.shape[0]),
                "uv_vertex_count": int(self.uv_vertices.shape[0]),
                "uv_face_count": int(self.uv_faces.shape[0]),
            },
            "region_reference": {
                "source": "opencat3d.cat_template_region_contract",
                "region_names": [region.name for region in self.regions],
            },
            "required_anchor_names": list(CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES),
            "anchor_count": int(len(anchors)),
            "anchors": [
                anchor.model_dump(include_indices=include_indices) for anchor in anchors
            ],
        }
        manifest["validation"] = validate_cat_template_feature_anchor_manifest(manifest)
        return manifest

    def shaped_vertices(self, beta: NDArray[np.floating] | None = None) -> NDArray[np.float64]:
        vertices = self.vertices_canonical.copy()
        if beta is None:
            return vertices
        coeffs = np.asarray(beta, dtype=np.float64)
        if coeffs.ndim != 1 or coeffs.shape[0] != self.shape_basis.shape[0]:
            raise ValueError(f"beta must have shape ({self.shape_basis.shape[0]},)")
        return vertices + np.einsum("b,bvc->vc", coeffs, self.shape_basis)

    def posed_vertices(
        self,
        beta: NDArray[np.floating] | None = None,
        pose_6d: NDArray[np.floating] | None = None,
        global_orient_6d: NDArray[np.floating] | None = None,
    ) -> NDArray[np.float64]:
        vertices = self.shaped_vertices(beta)
        if pose_6d is not None:
            pose = np.asarray(pose_6d, dtype=np.float64)
            if pose.shape != (self.joints_rest.shape[0], 6):
                raise ValueError(f"pose_6d must have shape ({self.joints_rest.shape[0]}, 6)")
        if global_orient_6d is None:
            return vertices
        rotation = rotation_6d_to_matrix(np.asarray(global_orient_6d, dtype=np.float64))
        return vertices @ rotation.T

    def joints(
        self,
        beta: NDArray[np.floating] | None = None,
        pose_6d: NDArray[np.floating] | None = None,
    ) -> NDArray[np.float64]:
        if pose_6d is not None:
            pose = np.asarray(pose_6d, dtype=np.float64)
            if pose.shape != (self.joints_rest.shape[0], 6):
                raise ValueError(f"pose_6d must have shape ({self.joints_rest.shape[0]}, 6)")
        joints = self.joints_rest.copy()
        if beta is None:
            return joints
        coeffs = np.asarray(beta, dtype=np.float64)
        if coeffs.shape != (self.shape_basis.shape[0],):
            raise ValueError(f"beta must have shape ({self.shape_basis.shape[0]},)")
        scale_x = 1.0 + 0.20 * coeffs[0]
        scale_y = 1.0 + 0.15 * coeffs[1]
        scale_z = 1.0 + 0.25 * coeffs[2]
        joints[:, 0] *= scale_x
        joints[:, 1] *= scale_y
        joints[:, 2] *= scale_z
        return joints

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vertices_canonical=self.vertices_canonical,
            faces=self.faces,
            joints_rest=self.joints_rest,
            skin_weights=self.skin_weights,
            joint_parents=self.joint_parents,
            uv_vertices=self.uv_vertices,
            uv_faces=self.uv_faces,
            shape_basis=self.shape_basis,
        )


def load_cat_template_npz(path: Path) -> CatTemplate:
    with np.load(path) as data:
        vertices = np.asarray(data["vertices_canonical"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
        uv_faces = np.asarray(data["uv_faces"], dtype=np.int64)
        faces, uv_faces = _orient_faces_outward_by_component(
            vertices,
            faces,
            uv_faces=uv_faces,
        )
        return CatTemplate(
            vertices_canonical=vertices,
            faces=faces,
            joints_rest=np.asarray(data["joints_rest"], dtype=np.float64),
            skin_weights=np.asarray(data["skin_weights"], dtype=np.float64),
            joint_parents=np.asarray(data["joint_parents"], dtype=np.int64),
            uv_vertices=np.asarray(data["uv_vertices"], dtype=np.float64),
            uv_faces=uv_faces,
            shape_basis=np.asarray(data["shape_basis"], dtype=np.float64),
        )


def validate_cat_template_region_manifest_v2(manifest: dict[str, object]) -> dict[str, object]:
    regions = manifest.get("regions", [])
    if not isinstance(regions, list):
        return {
            "valid": False,
            "schema_version_valid": manifest.get("schema_version") == "2.0",
            "required_regions_present": False,
            "missing_required_regions": list(CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES),
            "duplicate_region_names": [],
            "region_count_matches": False,
            "positive_region_counts": False,
            "uv_bounds_valid": False,
            "uv_islands_declared": False,
            "uv_island_bounds_valid": False,
            "uv_island_density_positive": False,
            "primary_uv_islands_declared": False,
            "source_regions_declared": False,
            "reference_contract_declared": False,
            "source_region_references_valid": False,
            "uv_island_references_valid": False,
            "fixed_topology_contract": False,
        }

    names = [
        str(region.get("name"))
        for region in regions
        if isinstance(region, dict) and region.get("name") is not None
    ]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    missing = [
        name for name in CAT_TEMPLATE_REGION_V2_REQUIRED_NAMES if name not in set(names)
    ]
    region_count_matches = int(manifest.get("region_count", -1)) == len(regions)
    positive_region_counts = all(
        isinstance(region, dict)
        and int(region.get("vertex_count", 0)) > 0
        and int(region.get("face_count", 0)) > 0
        for region in regions
    )
    uv_bounds_valid = all(
        isinstance(region, dict) and _manifest_uv_bounds_valid(region.get("uv_bounds"))
        for region in regions
    )
    uv_island_checks = [
        _manifest_region_uv_islands_valid(region) if isinstance(region, dict) else {}
        for region in regions
    ]
    uv_islands_declared = all(
        bool(check.get("declared", False)) for check in uv_island_checks
    )
    uv_island_bounds_valid = all(
        bool(check.get("bounds_valid", False)) for check in uv_island_checks
    )
    uv_island_density_positive = all(
        bool(check.get("density_positive", False)) for check in uv_island_checks
    )
    primary_uv_islands_declared = all(
        bool(check.get("primary_declared", False)) for check in uv_island_checks
    )
    source_regions_declared = all(
        isinstance(region, dict)
        and isinstance(region.get("source_regions"), list)
        and len(region.get("source_regions", [])) > 0
        for region in regions
    )
    reference_contract = _manifest_reference_contract(manifest.get("reference_contract"))
    reference_contract_declared = bool(reference_contract["declared"])
    source_region_references_valid = _manifest_source_region_references_valid(
        regions,
        allowed_source_region_names=reference_contract["base_region_names"],
    )
    uv_island_references_valid = _manifest_uv_island_references_valid(
        regions,
        allowed_source_region_names=reference_contract["base_region_names"],
        allowed_face_uv_island_names=reference_contract["face_uv_island_names"],
        region_bounds_uv_island_source=reference_contract["region_bounds_uv_island_source"],
    )
    fixed_topology_contract = (
        manifest.get("target_representation") == "fixed_topology_cat_template"
        and manifest.get("topology_changed") is False
    )
    checks = {
        "schema_version_valid": manifest.get("schema_version") == "2.0",
        "required_regions_present": not missing,
        "missing_required_regions": missing,
        "duplicate_region_names": duplicate_names,
        "region_count_matches": region_count_matches,
        "positive_region_counts": positive_region_counts,
        "uv_bounds_valid": uv_bounds_valid,
        "uv_islands_declared": uv_islands_declared,
        "uv_island_bounds_valid": uv_island_bounds_valid,
        "uv_island_density_positive": uv_island_density_positive,
        "primary_uv_islands_declared": primary_uv_islands_declared,
        "source_regions_declared": source_regions_declared,
        "reference_contract_declared": reference_contract_declared,
        "source_region_references_valid": source_region_references_valid,
        "uv_island_references_valid": uv_island_references_valid,
        "fixed_topology_contract": fixed_topology_contract,
    }
    checks["valid"] = bool(
        checks["schema_version_valid"]
        and checks["required_regions_present"]
        and not duplicate_names
        and region_count_matches
        and positive_region_counts
        and uv_bounds_valid
        and uv_islands_declared
        and uv_island_bounds_valid
        and uv_island_density_positive
        and primary_uv_islands_declared
        and source_regions_declared
        and reference_contract_declared
        and source_region_references_valid
        and uv_island_references_valid
        and fixed_topology_contract
    )
    return checks


def validate_cat_template_feature_anchor_manifest(
    manifest: dict[str, object],
) -> dict[str, object]:
    anchors = manifest.get("anchors", [])
    if not isinstance(anchors, list):
        return {
            "valid": False,
            "schema_version_valid": manifest.get("schema_version") == "1.0",
            "required_anchors_present": False,
            "missing_required_anchors": list(CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES),
            "duplicate_anchor_names": [],
            "anchor_count_matches": False,
            "region_references_valid": False,
            "anchor_positions_valid": False,
            "support_radius_positive": False,
            "support_vertices_declared": False,
            "closest_vertices_valid": False,
            "roles_declared": False,
            "fixed_topology_contract": False,
        }

    names = [
        str(anchor.get("name"))
        for anchor in anchors
        if isinstance(anchor, dict) and anchor.get("name") is not None
    ]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    missing = [
        name for name in CAT_TEMPLATE_FEATURE_ANCHOR_REQUIRED_NAMES if name not in set(names)
    ]
    region_reference = manifest.get("region_reference", {})
    region_names = (
        _manifest_string_set(region_reference.get("region_names"))
        if isinstance(region_reference, dict)
        else set()
    )
    vertex_count = _manifest_topology_vertex_count(manifest.get("topology"))
    anchor_count_matches = int(manifest.get("anchor_count", -1)) == len(anchors)
    region_references_valid = bool(region_names) and all(
        isinstance(anchor, dict) and anchor.get("region_name") in region_names
        for anchor in anchors
    )
    anchor_positions_valid = all(
        isinstance(anchor, dict)
        and _manifest_xyz_valid(anchor.get("anchor_xyz"))
        and _manifest_xyz_valid(anchor.get("closest_vertex_xyz"))
        for anchor in anchors
    )
    support_radius_positive = all(
        isinstance(anchor, dict)
        and _manifest_positive_float(anchor.get("support_radius"))
        for anchor in anchors
    )
    support_vertices_declared = all(
        isinstance(anchor, dict) and int(anchor.get("support_vertex_count", 0)) > 0
        for anchor in anchors
    )
    closest_vertices_valid = vertex_count > 0 and all(
        isinstance(anchor, dict)
        and _manifest_vertex_index_valid(anchor.get("closest_vertex_index"), vertex_count)
        for anchor in anchors
    )
    roles_declared = all(
        isinstance(anchor, dict)
        and bool(str(anchor.get("semantic", "")))
        and bool(str(anchor.get("reconstruction_role", "")))
        for anchor in anchors
    )
    fixed_topology_contract = (
        manifest.get("target_representation") == "fixed_topology_cat_template"
        and manifest.get("fixed_topology") is True
    )
    checks = {
        "schema_version_valid": manifest.get("schema_version") == "1.0",
        "required_anchors_present": not missing,
        "missing_required_anchors": missing,
        "duplicate_anchor_names": duplicate_names,
        "anchor_count_matches": anchor_count_matches,
        "region_references_valid": region_references_valid,
        "anchor_positions_valid": anchor_positions_valid,
        "support_radius_positive": support_radius_positive,
        "support_vertices_declared": support_vertices_declared,
        "closest_vertices_valid": closest_vertices_valid,
        "roles_declared": roles_declared,
        "fixed_topology_contract": fixed_topology_contract,
    }
    checks["valid"] = bool(
        checks["schema_version_valid"]
        and checks["required_anchors_present"]
        and not duplicate_names
        and anchor_count_matches
        and region_references_valid
        and anchor_positions_valid
        and support_radius_positive
        and support_vertices_declared
        and closest_vertices_valid
        and roles_declared
        and fixed_topology_contract
    )
    return checks


def create_placeholder_cat_template() -> CatTemplate:
    parts = [
        _ellipsoid(
            center=(0.0, 0.70, -0.10),
            radii=(0.48, 0.30, 0.82),
            rings=9,
            segments=32,
            joint_index=0,
        ),
        _ellipsoid(
            center=(0.0, 0.96, 0.92),
            radii=(0.34, 0.27, 0.34),
            rings=7,
            segments=28,
            joint_index=2,
        ),
        _ellipsoid(
            center=(0.0, 0.87, 1.20),
            radii=(0.20, 0.13, 0.18),
            rings=5,
            segments=20,
            joint_index=2,
        ),
        _ellipsoid(
            center=(-0.12, 1.02, 1.18),
            radii=(0.045, 0.035, 0.025),
            rings=3,
            segments=12,
            joint_index=2,
        ),
        _ellipsoid(
            center=(0.12, 1.02, 1.18),
            radii=(0.045, 0.035, 0.025),
            rings=3,
            segments=12,
            joint_index=2,
        ),
        _ellipsoid(
            center=(0.0, 0.89, 1.36),
            radii=(0.055, 0.035, 0.035),
            rings=3,
            segments=12,
            joint_index=2,
        ),
        _cone(
            base_center=(-0.20, 1.16, 0.96),
            radius_x=0.11,
            radius_z=0.08,
            height=0.34,
            segments=16,
            joint_index=3,
        ),
        _cone(
            base_center=(0.20, 1.16, 0.96),
            radius_x=0.11,
            radius_z=0.08,
            height=0.34,
            segments=16,
            joint_index=4,
        ),
        _cylinder_y(
            center=(-0.30, 0.18, 0.46),
            radius_x=0.085,
            radius_z=0.10,
            height=0.62,
            segments=16,
            joint_index=5,
        ),
        _cylinder_y(
            center=(0.30, 0.18, 0.46),
            radius_x=0.085,
            radius_z=0.10,
            height=0.62,
            segments=16,
            joint_index=6,
        ),
        _cylinder_y(
            center=(-0.31, 0.18, -0.55),
            radius_x=0.10,
            radius_z=0.12,
            height=0.66,
            segments=16,
            joint_index=7,
        ),
        _cylinder_y(
            center=(0.31, 0.18, -0.55),
            radius_x=0.10,
            radius_z=0.12,
            height=0.66,
            segments=16,
            joint_index=8,
        ),
        _tail_tube(
            centers=(
                (0.0, 0.76, -0.88),
                (0.0, 0.80, -1.02),
                (0.0, 0.88, -1.18),
                (0.0, 1.04, -1.42),
                (0.0, 1.22, -1.56),
            ),
            radii=(0.095, 0.090, 0.080, 0.066, 0.050),
            segments=16,
            joint_index=10,
        ),
    ]
    vertices, faces, weights = _merge_parts(parts, joint_count=len(JOINT_PARENTS))
    if vertices.shape != (CAT_TEMPLATE_VERTEX_COUNT, 3):
        raise AssertionError(f"unexpected CatTemplate vertex count: {vertices.shape[0]}")
    if faces.shape != (CAT_TEMPLATE_FACE_COUNT, 3):
        raise AssertionError(f"unexpected CatTemplate face count: {faces.shape[0]}")
    joints = np.asarray(
        [
            [0.0, 0.69, -0.15],
            [0.0, 0.86, 0.52],
            [0.0, 0.94, 1.12],
            [-0.20, 1.36, 0.96],
            [0.20, 1.36, 0.96],
            [-0.30, -0.12, 0.46],
            [0.30, -0.12, 0.46],
            [-0.31, -0.14, -0.55],
            [0.31, -0.14, -0.55],
            [0.0, 0.76, -0.86],
            [0.0, 1.10, -1.36],
        ],
        dtype=np.float64,
    )
    uv_vertices = _uv_from_vertices(vertices)
    shape_basis = _shape_basis(vertices)
    return CatTemplate(
        vertices_canonical=vertices,
        faces=faces,
        joints_rest=joints,
        skin_weights=weights,
        joint_parents=np.asarray(JOINT_PARENTS, dtype=np.int64),
        uv_vertices=uv_vertices,
        uv_faces=faces.copy(),
        shape_basis=shape_basis,
    )


_REGION_SPECS = (
    _RegionSpec("torso", "body", (0.0, 0.70, -0.10), (0.50, 0.32, 0.84), 1.0, 20),
    _RegionSpec("head", "face", (0.0, 0.96, 0.92), (0.36, 0.29, 0.36), 2.0, 70),
    _RegionSpec("muzzle", "face", (0.0, 0.87, 1.20), (0.22, 0.15, 0.20), 2.6, 90),
    _RegionSpec("left_eye", "face_detail", (-0.12, 1.02, 1.18), (0.055, 0.045, 0.035), 3.0, 100),
    _RegionSpec("right_eye", "face_detail", (0.12, 1.02, 1.18), (0.055, 0.045, 0.035), 3.0, 100),
    _RegionSpec("nose", "face_detail", (0.0, 0.89, 1.36), (0.070, 0.050, 0.050), 3.0, 100),
    _RegionSpec(
        "left_ear_inner",
        "ear",
        (-0.20, 1.30, 0.96),
        (0.14, 0.24, 0.10),
        2.4,
        85,
    ),
    _RegionSpec(
        "right_ear_inner",
        "ear",
        (0.20, 1.30, 0.96),
        (0.14, 0.24, 0.10),
        2.4,
        85,
    ),
    _RegionSpec(
        "left_front_limb",
        "limb",
        (-0.30, 0.18, 0.46),
        (0.10, 0.33, 0.12),
        1.2,
        35,
    ),
    _RegionSpec(
        "right_front_limb",
        "limb",
        (0.30, 0.18, 0.46),
        (0.10, 0.33, 0.12),
        1.2,
        35,
    ),
    _RegionSpec(
        "left_hind_limb",
        "limb",
        (-0.31, 0.18, -0.55),
        (0.12, 0.35, 0.14),
        1.2,
        35,
    ),
    _RegionSpec(
        "right_hind_limb",
        "limb",
        (0.31, 0.18, -0.55),
        (0.12, 0.35, 0.14),
        1.2,
        35,
    ),
    _RegionSpec(
        "left_front_paw",
        "paw",
        (-0.30, -0.12, 0.46),
        (0.10, 0.08, 0.12),
        1.6,
        50,
    ),
    _RegionSpec(
        "right_front_paw",
        "paw",
        (0.30, -0.12, 0.46),
        (0.10, 0.08, 0.12),
        1.6,
        50,
    ),
    _RegionSpec(
        "left_hind_paw",
        "paw",
        (-0.31, -0.13, -0.55),
        (0.12, 0.08, 0.14),
        1.6,
        50,
    ),
    _RegionSpec(
        "right_hind_paw",
        "paw",
        (0.31, -0.13, -0.55),
        (0.12, 0.08, 0.14),
        1.6,
        50,
    ),
    _RegionSpec("tail", "tail", (0.0, 0.94, -1.22), (0.13, 0.37, 0.46), 1.8, 60),
)

_COMPOSITE_REGIONS = (
    (
        "ears",
        "ear",
        ("left_ear_inner", "right_ear_inner"),
        2.4,
        85,
    ),
    (
        "paws",
        "paw",
        ("left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"),
        1.6,
        50,
    ),
    (
        "limbs",
        "limb",
        ("left_front_limb", "right_front_limb", "left_hind_limb", "right_hind_limb"),
        1.2,
        35,
    ),
    (
        "face",
        "face",
        (
            "head",
            "muzzle",
            "left_eye",
            "right_eye",
            "nose",
            "left_ear_inner",
            "right_ear_inner",
        ),
        3.0,
        100,
    ),
    (
        "body",
        "body",
        (
            "torso",
            "left_front_limb",
            "right_front_limb",
            "left_hind_limb",
            "right_hind_limb",
            "tail",
        ),
        1.0,
        20,
    ),
)

_REGION_V2_SPECS = (
    _RegionV2Spec(
        "face",
        "face",
        ("face",),
        3.0,
        100,
        "identity_shape_and_texture",
        uv_island_names=(
            "face",
            "left_eye",
            "right_eye",
            "nose",
            "muzzle",
            "mouth_line",
            "left_ear_inner",
            "right_ear_inner",
        ),
    ),
    _RegionV2Spec(
        "muzzle",
        "muzzle",
        ("muzzle",),
        3.0,
        96,
        "facial_feature",
        uv_island_names=("muzzle", "mouth_line"),
    ),
    _RegionV2Spec(
        "eyes",
        "eyes",
        ("left_eye", "right_eye"),
        3.2,
        100,
        "facial_feature",
        symmetry_group="bilateral_eye_pair",
        uv_island_names=("left_eye", "right_eye"),
    ),
    _RegionV2Spec(
        "nose",
        "nose",
        ("nose",),
        3.4,
        100,
        "facial_feature",
        uv_island_names=("nose",),
    ),
    _RegionV2Spec(
        "ear_interiors",
        "ear_inner",
        ("left_ear_inner", "right_ear_inner"),
        2.8,
        92,
        "facial_feature",
        symmetry_group="bilateral_ear_pair",
        uv_island_names=("left_ear_inner", "right_ear_inner"),
    ),
    _RegionV2Spec(
        "paws",
        "paw",
        ("paws",),
        1.6,
        50,
        "contact_and_scale",
    ),
    _RegionV2Spec(
        "tail",
        "tail",
        ("tail",),
        1.8,
        60,
        "silhouette_and_pose",
    ),
    _RegionV2Spec(
        "torso",
        "torso",
        ("torso",),
        1.0,
        20,
        "body_identity",
    ),
    _RegionV2Spec(
        "belly",
        "belly",
        ("torso",),
        1.4,
        45,
        "ventral_markings",
    ),
    _RegionV2Spec(
        "limbs",
        "limb",
        ("limbs",),
        1.2,
        35,
        "pose_and_silhouette",
    ),
)

_FACE_UV_REGION_SPECS = (
    _FaceUvRegionSpec(
        "face",
        "face",
        "face",
        (0.0, 0.96, 1.12),
        (0.30, 0.16),
        2.4,
        90,
        2.0,
        (),
    ),
    _FaceUvRegionSpec(
        "left_eye",
        "eye",
        "left_eye",
        (-0.12, 1.02, 1.188),
        (0.078, 0.048),
        3.2,
        100,
        6.0,
        (
            "left_eye_shadow",
            "left_eye_core",
            "left_eye_iris",
            "left_eye_pupil",
        ),
        uv_center=(0.375, 0.930),
    ),
    _FaceUvRegionSpec(
        "right_eye",
        "eye",
        "right_eye",
        (0.12, 1.02, 1.188),
        (0.078, 0.048),
        3.2,
        100,
        6.0,
        (
            "right_eye_shadow",
            "right_eye_core",
            "right_eye_iris",
            "right_eye_pupil",
        ),
        uv_center=(0.625, 0.930),
    ),
    _FaceUvRegionSpec(
        "nose",
        "nose",
        "nose",
        (0.0, 0.90, 1.36),
        (0.056, 0.034),
        3.4,
        100,
        6.0,
        ("nose",),
        uv_center=(0.500, 0.885),
    ),
    _FaceUvRegionSpec(
        "muzzle",
        "muzzle",
        "muzzle",
        (0.0, 0.87, 1.285),
        (0.150, 0.070),
        3.0,
        95,
        4.0,
        ("left_muzzle", "right_muzzle", "mouth_line"),
        uv_center=(0.500, 0.855),
    ),
    _FaceUvRegionSpec(
        "mouth_line",
        "mouth",
        "muzzle",
        (0.0, 0.85, 1.315),
        (0.050, 0.010),
        3.2,
        96,
        5.0,
        ("mouth_line",),
        uv_center=(0.500, 0.825),
    ),
    _FaceUvRegionSpec(
        "left_ear_inner",
        "ear_inner",
        "left_ear_inner",
        (-0.20, 1.20, 0.98),
        (0.070, 0.055),
        2.8,
        92,
        4.0,
        ("left_ear_inner",),
        uv_center=(0.292, 0.955),
    ),
    _FaceUvRegionSpec(
        "right_ear_inner",
        "ear_inner",
        "right_ear_inner",
        (0.20, 1.20, 0.98),
        (0.070, 0.055),
        2.8,
        92,
        4.0,
        ("right_ear_inner",),
        uv_center=(0.708, 0.955),
    ),
)

_FEATURE_ANCHOR_SPECS = (
    _FeatureAnchorSpec(
        "left_eye_center",
        "eye",
        "left_eye",
        (-0.12, 1.02, 1.188),
        0.080,
        "facial_identity",
        symmetry_group="bilateral_eye_pair",
    ),
    _FeatureAnchorSpec(
        "right_eye_center",
        "eye",
        "right_eye",
        (0.12, 1.02, 1.188),
        0.080,
        "facial_identity",
        symmetry_group="bilateral_eye_pair",
    ),
    _FeatureAnchorSpec(
        "nose_tip",
        "nose",
        "nose",
        (0.0, 0.90, 1.36),
        0.085,
        "facial_identity",
    ),
    _FeatureAnchorSpec(
        "muzzle_center",
        "muzzle",
        "muzzle",
        (0.0, 0.87, 1.285),
        0.170,
        "facial_identity",
    ),
    _FeatureAnchorSpec(
        "mouth_line",
        "mouth",
        "muzzle",
        (0.0, 0.85, 1.315),
        0.110,
        "facial_identity",
    ),
    _FeatureAnchorSpec(
        "left_ear_inner",
        "ear_inner",
        "left_ear_inner",
        (-0.20, 1.20, 0.98),
        0.180,
        "ear_shape_and_interior_texture",
        symmetry_group="bilateral_ear_pair",
    ),
    _FeatureAnchorSpec(
        "right_ear_inner",
        "ear_inner",
        "right_ear_inner",
        (0.20, 1.20, 0.98),
        0.180,
        "ear_shape_and_interior_texture",
        symmetry_group="bilateral_ear_pair",
    ),
    _FeatureAnchorSpec(
        "left_front_paw",
        "paw",
        "left_front_paw",
        (-0.30, -0.12, 0.46),
        0.130,
        "contact_and_scale",
        symmetry_group="front_paw_pair",
    ),
    _FeatureAnchorSpec(
        "right_front_paw",
        "paw",
        "right_front_paw",
        (0.30, -0.12, 0.46),
        0.130,
        "contact_and_scale",
        symmetry_group="front_paw_pair",
    ),
    _FeatureAnchorSpec(
        "left_hind_paw",
        "paw",
        "left_hind_paw",
        (-0.31, -0.14, -0.55),
        0.145,
        "contact_and_scale",
        symmetry_group="hind_paw_pair",
    ),
    _FeatureAnchorSpec(
        "right_hind_paw",
        "paw",
        "right_hind_paw",
        (0.31, -0.14, -0.55),
        0.145,
        "contact_and_scale",
        symmetry_group="hind_paw_pair",
    ),
    _FeatureAnchorSpec(
        "tail_base",
        "tail",
        "tail",
        (0.0, 0.76, -0.88),
        0.160,
        "tail_pose_and_silhouette",
    ),
    _FeatureAnchorSpec(
        "tail_tip",
        "tail",
        "tail",
        (0.0, 1.22, -1.56),
        0.120,
        "tail_pose_and_silhouette",
    ),
)


def _cat_template_regions(template: CatTemplate) -> tuple[CatTemplateRegion, ...]:
    vertices = np.asarray(template.vertices_canonical, dtype=np.float64)
    faces = np.asarray(template.faces, dtype=np.int64)
    uv_vertices = np.asarray(template.uv_vertices, dtype=np.float64)
    if uv_vertices.shape != (vertices.shape[0], 2):
        raise ValueError("uv_vertices must have shape [vertex_count,2]")

    regions_by_name: dict[str, CatTemplateRegion] = {}
    ordered_regions: list[CatTemplateRegion] = []
    for spec in _REGION_SPECS:
        vertex_mask = _ellipsoid_vertex_mask(vertices, center=spec.center, radii=spec.radii)
        region = _region_from_mask(
            spec.name,
            spec.semantic,
            vertex_mask,
            faces=faces,
            uv_vertices=uv_vertices,
            detail_weight=spec.detail_weight,
            texture_priority=spec.texture_priority,
        )
        regions_by_name[region.name] = region
        ordered_regions.append(region)

    for name, semantic, child_names, detail_weight, texture_priority in _COMPOSITE_REGIONS:
        vertex_indices = np.unique(
            np.concatenate(
                [regions_by_name[child_name].vertex_indices for child_name in child_names]
            )
        )
        vertex_mask = np.zeros(vertices.shape[0], dtype=np.bool_)
        vertex_mask[vertex_indices] = True
        region = _region_from_mask(
            name,
            semantic,
            vertex_mask,
            faces=faces,
            uv_vertices=uv_vertices,
            detail_weight=detail_weight,
            texture_priority=texture_priority,
        )
        regions_by_name[region.name] = region
        ordered_regions.append(region)

    return tuple(ordered_regions)


def _cat_template_regions_v2(template: CatTemplate) -> tuple[CatTemplateRegionV2, ...]:
    vertices = np.asarray(template.vertices_canonical, dtype=np.float64)
    faces = np.asarray(template.faces, dtype=np.int64)
    uv_vertices = np.asarray(template.uv_vertices, dtype=np.float64)
    if uv_vertices.shape != (vertices.shape[0], 2):
        raise ValueError("uv_vertices must have shape [vertex_count,2]")

    base_regions = {region.name: region for region in template.regions}
    face_uv_regions = {region.name: region for region in template.face_uv_regions}
    regions: list[CatTemplateRegionV2] = []
    for spec in _REGION_V2_SPECS:
        missing_sources = [
            source_name for source_name in spec.source_regions if source_name not in base_regions
        ]
        if missing_sources:
            raise ValueError(
                f"CatTemplate v2 region {spec.name!r} has unknown sources: {missing_sources}"
            )
        if spec.name == "belly":
            vertex_mask = _belly_vertex_mask(vertices, base_regions["torso"].vertex_indices)
        else:
            source_indices = np.unique(
                np.concatenate(
                    [
                        base_regions[source_name].vertex_indices
                        for source_name in spec.source_regions
                    ]
                )
            )
            vertex_mask = np.zeros(vertices.shape[0], dtype=np.bool_)
            vertex_mask[source_indices] = True
        regions.append(
            _region_v2_from_mask(
                spec,
                vertex_mask,
                faces=faces,
                uv_vertices=uv_vertices,
                face_uv_regions=face_uv_regions,
            )
        )
    return tuple(regions)


def _cat_template_face_uv_regions(template: CatTemplate) -> tuple[CatTemplateFaceUvRegion, ...]:
    vertices = np.asarray(template.vertices_canonical, dtype=np.float64)
    regions: list[CatTemplateFaceUvRegion] = []
    for spec in _FACE_UV_REGION_SPECS:
        source = template.region(spec.source_region)
        if spec.uv_center is None:
            uv_center_array = _uv_for_canonical_xz(
                vertices,
                x=spec.anchor_xyz[0],
                z=spec.anchor_xyz[2],
            )
        else:
            uv_center_array = np.asarray(spec.uv_center, dtype=np.float64)
        radius_u, radius_v = spec.uv_radius
        uv_center = (
            float(np.clip(uv_center_array[0], 0.0, 1.0)),
            float(np.clip(uv_center_array[1], 0.0, 1.0)),
        )
        uv_bounds = (
            float(np.clip(uv_center[0] - radius_u, 0.0, 1.0)),
            float(np.clip(uv_center[1] - radius_v, 0.0, 1.0)),
            float(np.clip(uv_center[0] + radius_u, 0.0, 1.0)),
            float(np.clip(uv_center[1] + radius_v, 0.0, 1.0)),
        )
        if uv_bounds[0] >= uv_bounds[2] or uv_bounds[1] >= uv_bounds[3]:
            raise ValueError(f"CatTemplate face UV region {spec.name!r} has empty bounds")
        regions.append(
            CatTemplateFaceUvRegion(
                name=spec.name,
                semantic=spec.semantic,
                source_region=spec.source_region,
                anchor_xyz=spec.anchor_xyz,
                uv_center=uv_center,
                uv_radius=spec.uv_radius,
                uv_bounds=uv_bounds,
                detail_weight=spec.detail_weight,
                texture_priority=spec.texture_priority,
                target_texel_density=spec.target_texel_density,
                feature_names=spec.feature_names,
                source_vertex_count=int(source.vertex_indices.size),
                source_face_count=int(source.face_indices.size),
            )
        )
    return tuple(regions)


def _cat_template_feature_anchors(template: CatTemplate) -> tuple[CatTemplateFeatureAnchor, ...]:
    vertices = np.asarray(template.vertices_canonical, dtype=np.float64)
    regions = {region.name: region for region in template.regions}
    anchors: list[CatTemplateFeatureAnchor] = []
    for spec in _FEATURE_ANCHOR_SPECS:
        region = regions.get(spec.region_name)
        if region is None:
            raise ValueError(
                f"CatTemplate feature anchor {spec.name!r} references unknown "
                f"region {spec.region_name!r}"
            )
        region_indices = region.vertex_indices
        anchor_xyz = np.asarray(spec.anchor_xyz, dtype=np.float64)
        distances = np.linalg.norm(vertices[region_indices] - anchor_xyz[None, :], axis=1)
        closest_local = int(np.argmin(distances))
        closest_vertex_index = int(region_indices[closest_local])
        support = region_indices[distances <= float(spec.support_radius)]
        if support.size == 0:
            support = np.asarray([closest_vertex_index], dtype=np.int64)
        anchors.append(
            CatTemplateFeatureAnchor(
                name=spec.name,
                semantic=spec.semantic,
                region_name=spec.region_name,
                reconstruction_role=spec.reconstruction_role,
                anchor_xyz=spec.anchor_xyz,
                support_radius=float(spec.support_radius),
                closest_vertex_index=closest_vertex_index,
                closest_vertex_xyz=tuple(
                    float(value) for value in vertices[closest_vertex_index]
                ),
                support_vertex_indices=np.asarray(np.unique(support), dtype=np.int64),
                symmetry_group=spec.symmetry_group,
            )
        )
    return tuple(anchors)


def _region_v2_from_mask(
    spec: _RegionV2Spec,
    vertex_mask: NDArray[np.bool_],
    *,
    faces: NDArray[np.int64],
    uv_vertices: NDArray[np.float64],
    face_uv_regions: dict[str, CatTemplateFaceUvRegion],
) -> CatTemplateRegionV2:
    vertex_indices = np.nonzero(vertex_mask)[0].astype(np.int64)
    if vertex_indices.size == 0:
        raise ValueError(f"CatTemplate v2 region {spec.name!r} has no vertices")
    face_hits = np.count_nonzero(vertex_mask[faces], axis=1) > 0
    face_indices = np.nonzero(face_hits)[0].astype(np.int64)
    if face_indices.size == 0:
        raise ValueError(f"CatTemplate v2 region {spec.name!r} has no faces")
    uv = uv_vertices[vertex_indices]
    uv_min = np.min(uv, axis=0)
    uv_max = np.max(uv, axis=0)
    uv_bounds = (float(uv_min[0]), float(uv_min[1]), float(uv_max[0]), float(uv_max[1]))
    uv_islands = _region_v2_uv_islands(
        spec,
        uv_bounds=uv_bounds,
        face_uv_regions=face_uv_regions,
    )
    return CatTemplateRegionV2(
        name=spec.name,
        semantic=spec.semantic,
        source_regions=spec.source_regions,
        vertex_indices=vertex_indices,
        face_indices=face_indices,
        uv_bounds=uv_bounds,
        uv_islands=uv_islands,
        detail_weight=float(spec.detail_weight),
        texture_priority=int(spec.texture_priority),
        reconstruction_role=spec.reconstruction_role,
        symmetry_group=spec.symmetry_group,
    )


def _region_v2_uv_islands(
    spec: _RegionV2Spec,
    *,
    uv_bounds: tuple[float, float, float, float],
    face_uv_regions: dict[str, CatTemplateFaceUvRegion],
) -> tuple[CatTemplateRegionUvIsland, ...]:
    if spec.uv_island_names:
        islands: list[CatTemplateRegionUvIsland] = []
        for island_name in spec.uv_island_names:
            face_region = face_uv_regions.get(island_name)
            if face_region is None:
                raise ValueError(
                    f"CatTemplate v2 region {spec.name!r} references unknown UV island "
                    f"{island_name!r}"
                )
            islands.append(
                CatTemplateRegionUvIsland(
                    name=face_region.name,
                    semantic=face_region.semantic,
                    source="opencat3d.cat_template_face_uv_region_contract",
                    source_region=face_region.source_region,
                    uv_bounds=face_region.uv_bounds,
                    detail_weight=face_region.detail_weight,
                    texture_priority=face_region.texture_priority,
                    target_texel_density=face_region.target_texel_density,
                    allocation_role="facial_detail",
                    feature_names=face_region.feature_names,
                )
            )
        return tuple(islands)

    return (
        CatTemplateRegionUvIsland(
            name=f"{spec.name}_region",
            semantic=spec.semantic,
            source="opencat3d.cat_template_region_contract_v2_bounds",
            source_region=spec.name,
            uv_bounds=uv_bounds,
            detail_weight=spec.detail_weight,
            texture_priority=spec.texture_priority,
            target_texel_density=max(1.0, float(spec.detail_weight)),
            allocation_role=spec.reconstruction_role,
            feature_names=(spec.name,),
        ),
    )


def _belly_vertex_mask(
    vertices: NDArray[np.float64],
    torso_vertex_indices: NDArray[np.int64],
) -> NDArray[np.bool_]:
    torso_vertices = vertices[torso_vertex_indices]
    torso_min = np.min(torso_vertices, axis=0)
    torso_max = np.max(torso_vertices, axis=0)
    torso_span = np.maximum(torso_max - torso_min, 1e-9)
    x_limit = max(abs(float(torso_min[0])), abs(float(torso_max[0]))) * 0.88
    y_limit = float(torso_min[1] + 0.42 * torso_span[1])
    z_min = float(torso_min[2] + 0.14 * torso_span[2])
    z_max = float(torso_max[2] - 0.16 * torso_span[2])

    selected = (
        (np.abs(torso_vertices[:, 0]) <= x_limit)
        & (torso_vertices[:, 1] <= y_limit)
        & (torso_vertices[:, 2] >= z_min)
        & (torso_vertices[:, 2] <= z_max)
    )
    vertex_mask = np.zeros(vertices.shape[0], dtype=np.bool_)
    vertex_mask[torso_vertex_indices[selected]] = True
    return vertex_mask


def _ellipsoid_vertex_mask(
    vertices: NDArray[np.float64],
    *,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> NDArray[np.bool_]:
    center_array = np.asarray(center, dtype=np.float64)
    radii_array = np.maximum(np.asarray(radii, dtype=np.float64), 1e-9)
    normalized = (vertices - center_array[None, :]) / radii_array[None, :]
    return np.sum(normalized * normalized, axis=1) <= 1.0 + 1e-12


def _region_from_mask(
    name: str,
    semantic: str,
    vertex_mask: NDArray[np.bool_],
    *,
    faces: NDArray[np.int64],
    uv_vertices: NDArray[np.float64],
    detail_weight: float,
    texture_priority: int,
) -> CatTemplateRegion:
    vertex_indices = np.nonzero(vertex_mask)[0].astype(np.int64)
    if vertex_indices.size == 0:
        raise ValueError(f"CatTemplate region {name!r} has no vertices")
    face_hits = np.count_nonzero(vertex_mask[faces], axis=1) > 0
    face_indices = np.nonzero(face_hits)[0].astype(np.int64)
    if face_indices.size == 0:
        raise ValueError(f"CatTemplate region {name!r} has no faces")
    uv = uv_vertices[vertex_indices]
    uv_min = np.min(uv, axis=0)
    uv_max = np.max(uv, axis=0)
    uv_bounds = (
        float(uv_min[0]),
        float(uv_min[1]),
        float(uv_max[0]),
        float(uv_max[1]),
    )
    return CatTemplateRegion(
        name=name,
        semantic=semantic,
        vertex_indices=vertex_indices,
        face_indices=face_indices,
        uv_bounds=uv_bounds,
        detail_weight=float(detail_weight),
        texture_priority=int(texture_priority),
    )


def _uv_for_canonical_xz(
    vertices: NDArray[np.float64],
    *,
    x: float,
    z: float,
) -> NDArray[np.float64]:
    mins = vertices.min(axis=0)
    spans = np.maximum(vertices.max(axis=0) - mins, 1e-9)
    return np.asarray(
        [
            np.clip((x - mins[0]) / spans[0], 0.0, 1.0),
            np.clip((z - mins[2]) / spans[2], 0.0, 1.0),
        ],
        dtype=np.float64,
    )


def _manifest_reference_contract(value: object) -> dict[str, object]:
    empty: dict[str, object] = {
        "declared": False,
        "base_region_names": set(),
        "face_uv_island_names": set(),
        "region_bounds_uv_island_source": "",
    }
    if not isinstance(value, dict):
        return empty
    base_region_names = _manifest_string_set(value.get("base_region_names"))
    face_uv_island_names = _manifest_string_set(value.get("face_uv_island_names"))
    region_bounds_source = str(value.get("region_bounds_uv_island_source", ""))
    declared = bool(
        value.get("schema_version") == "1.0"
        and value.get("source") == "opencat3d.cat_template_region_v2_reference_contract"
        and value.get("fixed_topology") is True
        and base_region_names
        and face_uv_island_names
        and region_bounds_source
    )
    return {
        "declared": declared,
        "base_region_names": base_region_names,
        "face_uv_island_names": face_uv_island_names,
        "region_bounds_uv_island_source": region_bounds_source,
    }


def _manifest_string_set(value: object) -> set[str]:
    if not isinstance(value, list | tuple):
        return set()
    names: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item:
            return set()
        names.add(item)
    return names


def _manifest_source_region_references_valid(
    regions: list[object],
    *,
    allowed_source_region_names: object,
) -> bool:
    if not isinstance(allowed_source_region_names, set) or not allowed_source_region_names:
        return False
    for region in regions:
        if not isinstance(region, dict):
            return False
        source_regions = region.get("source_regions", [])
        if not isinstance(source_regions, list) or not source_regions:
            return False
        for source_region in source_regions:
            if source_region not in allowed_source_region_names:
                return False
    return True


def _manifest_uv_island_references_valid(
    regions: list[object],
    *,
    allowed_source_region_names: object,
    allowed_face_uv_island_names: object,
    region_bounds_uv_island_source: object,
) -> bool:
    if not isinstance(allowed_source_region_names, set) or not allowed_source_region_names:
        return False
    if not isinstance(allowed_face_uv_island_names, set) or not allowed_face_uv_island_names:
        return False
    if not isinstance(region_bounds_uv_island_source, str) or not region_bounds_uv_island_source:
        return False
    for region in regions:
        if not isinstance(region, dict):
            return False
        region_name = region.get("name")
        islands = region.get("uv_islands", [])
        if not isinstance(region_name, str) or not isinstance(islands, list) or not islands:
            return False
        for island in islands:
            if not isinstance(island, dict):
                return False
            source = island.get("source")
            source_region = island.get("source_region")
            name = island.get("name")
            if source == "opencat3d.cat_template_face_uv_region_contract":
                if name not in allowed_face_uv_island_names:
                    return False
                if source_region not in allowed_source_region_names:
                    return False
            elif source == region_bounds_uv_island_source:
                if name != f"{region_name}_region" or source_region != region_name:
                    return False
            else:
                return False
    return True


def _manifest_topology_vertex_count(value: object) -> int:
    if not isinstance(value, dict):
        return 0
    try:
        return int(value.get("vertex_count", 0))
    except (TypeError, ValueError):
        return 0


def _manifest_xyz_valid(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return False
    try:
        xyz = np.asarray([float(component) for component in value], dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(xyz).all())


def _manifest_positive_float(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric > 0.0)


def _manifest_vertex_index_valid(value: object, vertex_count: int) -> bool:
    try:
        vertex_index = int(value)
    except (TypeError, ValueError):
        return False
    return 0 <= vertex_index < vertex_count


def _manifest_uv_bounds_valid(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return False
    try:
        u0, v0, u1, v1 = (float(component) for component in value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= u0 <= u1 <= 1.0 and 0.0 <= v0 <= v1 <= 1.0


def _manifest_region_uv_islands_valid(region: dict[str, object]) -> dict[str, bool]:
    islands = region.get("uv_islands", [])
    primary = region.get("primary_uv_island")
    island_count = region.get("uv_island_count", -1)
    island_names = region.get("uv_island_names", [])
    if not isinstance(islands, list) or not islands:
        return {
            "declared": False,
            "bounds_valid": False,
            "density_positive": False,
            "primary_declared": False,
        }
    actual_names = [
        str(island.get("name"))
        for island in islands
        if isinstance(island, dict) and island.get("name") is not None
    ]
    try:
        count_matches = int(island_count) == len(islands)
    except (TypeError, ValueError):
        count_matches = False
    bounds_valid = all(
        isinstance(island, dict)
        and _manifest_uv_bounds_strictly_positive(island.get("uv_bounds"))
        for island in islands
    )
    density_positive = all(
        isinstance(island, dict) and _manifest_uv_island_density_positive(island)
        for island in islands
    )
    names_declared = isinstance(island_names, list) and [
        str(name) for name in island_names
    ] == actual_names
    return {
        "declared": count_matches and names_declared,
        "bounds_valid": bounds_valid,
        "density_positive": density_positive,
        "primary_declared": isinstance(primary, str) and primary in actual_names,
    }


def _manifest_uv_bounds_strictly_positive(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return False
    try:
        u0, v0, u1, v1 = (float(component) for component in value)
    except (TypeError, ValueError):
        return False
    if not (0.0 <= u0 <= u1 <= 1.0 and 0.0 <= v0 <= v1 <= 1.0):
        return False
    return u0 < u1 and v0 < v1


def _manifest_uv_island_density_positive(island: dict[str, object]) -> bool:
    try:
        target_texel_density = float(island.get("target_texel_density", 0.0))
        detail_weight = float(island.get("detail_weight", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        target_texel_density > 0.0
        and detail_weight > 0.0
        and bool(str(island.get("source", "")))
    )


def _ellipsoid(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    rings: int,
    segments: int,
    joint_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    if rings < 1 or segments < 3:
        raise ValueError("ellipsoid requires at least one ring and three segments")
    cx, cy, cz = center
    rx, ry, rz = radii
    vertices: list[list[float]] = [[cx, cy + ry, cz]]
    for ring in range(1, rings + 1):
        theta = np.pi * ring / (rings + 1)
        ring_y = cy + ry * np.cos(theta)
        ring_radius = np.sin(theta)
        for segment in range(segments):
            phi = 2.0 * np.pi * segment / segments
            vertices.append(
                [
                    cx + rx * ring_radius * np.cos(phi),
                    ring_y,
                    cz + rz * ring_radius * np.sin(phi),
                ]
            )
    bottom_index = len(vertices)
    vertices.append([cx, cy - ry, cz])

    faces: list[list[int]] = []
    first_ring = 1
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        faces.append([0, first_ring + segment, first_ring + next_segment])
    for ring in range(rings - 1):
        ring_a = 1 + ring * segments
        ring_b = ring_a + segments
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a0 = ring_a + segment
            a1 = ring_a + next_segment
            b0 = ring_b + segment
            b1 = ring_b + next_segment
            faces.append([a0, b0, b1])
            faces.append([a0, b1, a1])
    last_ring = 1 + (rings - 1) * segments
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        faces.append([bottom_index, last_ring + next_segment, last_ring + segment])
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        joint_index,
    )


def _cone(
    base_center: tuple[float, float, float],
    radius_x: float,
    radius_z: float,
    height: float,
    segments: int,
    joint_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    if segments < 3:
        raise ValueError("cone requires at least three segments")
    cx, cy, cz = base_center
    vertices: list[list[float]] = []
    for segment in range(segments):
        phi = 2.0 * np.pi * segment / segments
        vertices.append([cx + radius_x * np.cos(phi), cy, cz + radius_z * np.sin(phi)])
    apex = len(vertices)
    vertices.append([cx, cy + height, cz])
    base = len(vertices)
    vertices.append([cx, cy, cz])

    faces: list[list[int]] = []
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        faces.append([segment, next_segment, apex])
        faces.append([base, segment, next_segment])
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        joint_index,
    )


def _cylinder_y(
    center: tuple[float, float, float],
    radius_x: float,
    radius_z: float,
    height: float,
    segments: int,
    joint_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    if segments < 3:
        raise ValueError("cylinder requires at least three segments")
    cx, cy, cz = center
    y0 = cy - 0.5 * height
    y1 = cy + 0.5 * height
    vertices: list[list[float]] = []
    for y in (y0, y1):
        for segment in range(segments):
            phi = 2.0 * np.pi * segment / segments
            vertices.append([cx + radius_x * np.cos(phi), y, cz + radius_z * np.sin(phi)])
    bottom_center = len(vertices)
    vertices.append([cx, y0, cz])
    top_center = len(vertices)
    vertices.append([cx, y1, cz])

    faces: list[list[int]] = []
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        bottom_0 = segment
        bottom_1 = next_segment
        top_0 = segments + segment
        top_1 = segments + next_segment
        faces.append([bottom_0, top_0, top_1])
        faces.append([bottom_0, top_1, bottom_1])
        faces.append([bottom_center, bottom_1, bottom_0])
        faces.append([top_center, top_0, top_1])
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        joint_index,
    )


def _tail_tube(
    centers: tuple[tuple[float, float, float], ...],
    radii: tuple[float, ...],
    segments: int,
    joint_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    if len(centers) != len(radii) or len(centers) < 2 or segments < 3:
        raise ValueError("tail tube requires matching centers/radii and at least two rings")
    vertices: list[list[float]] = []
    for center, radius in zip(centers, radii, strict=True):
        cx, cy, cz = center
        for segment in range(segments):
            phi = 2.0 * np.pi * segment / segments
            vertices.append([cx + radius * np.cos(phi), cy + radius * np.sin(phi), cz])
    root_center = len(vertices)
    vertices.append(list(centers[0]))
    tip_center = len(vertices)
    vertices.append(list(centers[-1]))

    faces: list[list[int]] = []
    for ring in range(len(centers) - 1):
        ring_a = ring * segments
        ring_b = (ring + 1) * segments
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a0 = ring_a + segment
            a1 = ring_a + next_segment
            b0 = ring_b + segment
            b1 = ring_b + next_segment
            faces.append([a0, b0, b1])
            faces.append([a0, b1, a1])
    last_ring = (len(centers) - 1) * segments
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        faces.append([root_center, next_segment, segment])
        faces.append([tip_center, last_ring + segment, last_ring + next_segment])
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        joint_index,
    )


def _merge_parts(
    parts: list[tuple[NDArray[np.float64], NDArray[np.int64], int]],
    joint_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    vertices_out: list[NDArray[np.float64]] = []
    faces_out: list[NDArray[np.int64]] = []
    weights_out: list[NDArray[np.float64]] = []
    offset = 0
    for vertices, faces, joint_index in parts:
        vertices_out.append(vertices)
        faces_out.append(faces + offset)
        weights = np.zeros((vertices.shape[0], joint_count), dtype=np.float64)
        weights[:, joint_index] = 1.0
        weights_out.append(weights)
        offset += vertices.shape[0]
    vertices = np.vstack(vertices_out)
    faces = np.vstack(faces_out)
    faces, _ = _orient_faces_outward_by_component(vertices, faces)
    return vertices, faces, np.vstack(weights_out)


def _orient_faces_outward_by_component(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    *,
    uv_faces: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64] | None]:
    oriented_faces = np.asarray(faces, dtype=np.int64).copy()
    oriented_uv_faces = None if uv_faces is None else np.asarray(uv_faces, dtype=np.int64).copy()
    vertex_count = int(vertices.shape[0])
    components = _face_components(oriented_faces, vertex_count)

    for face_indices in components:
        component_faces = oriented_faces[face_indices]
        component_vertices = np.unique(component_faces.reshape(-1))
        center = np.mean(vertices[component_vertices], axis=0)
        tri = vertices[component_faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        face_centers = np.mean(tri, axis=1)
        outward = np.sum(normals * (face_centers - center[None, :]), axis=1)
        flip_indices = face_indices[outward < 0.0]
        if flip_indices.size == 0:
            continue
        oriented_faces[flip_indices] = oriented_faces[flip_indices][:, [0, 2, 1]]
        if oriented_uv_faces is not None:
            oriented_uv_faces[flip_indices] = oriented_uv_faces[flip_indices][:, [0, 2, 1]]
    return oriented_faces, oriented_uv_faces


def _face_components(
    faces: NDArray[np.int64],
    vertex_count: int,
) -> list[NDArray[np.int64]]:
    vertex_to_faces: list[list[int]] = [[] for _ in range(vertex_count)]
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_to_faces[int(vertex_index)].append(face_index)

    seen = np.zeros(faces.shape[0], dtype=np.bool_)
    components: list[NDArray[np.int64]] = []
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


def _uv_from_vertices(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    mins = vertices.min(axis=0)
    spans = np.maximum(vertices.max(axis=0) - mins, 1e-9)
    u = (vertices[:, 0] - mins[0]) / spans[0]
    v = (vertices[:, 2] - mins[2]) / spans[2]
    return np.stack([u, v], axis=-1)


def _shape_basis(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    basis = np.zeros((3, vertices.shape[0], 3), dtype=np.float64)
    basis[0, :, 0] = vertices[:, 0] * 0.20
    basis[1, :, 1] = np.maximum(vertices[:, 1], 0.0) * 0.15
    basis[2, :, 2] = vertices[:, 2] * 0.20
    return basis
