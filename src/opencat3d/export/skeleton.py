"""Deterministic skeleton metadata export for fitted CatTemplate meshes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opencat3d.template.cat_template import CatTemplate
from opencat3d.template.skeleton import JOINT_NAMES


@dataclass(frozen=True)
class SkeletonMetadataReport:
    path: Path
    success: bool
    checks: dict[str, bool]
    joint_count: int
    vertex_count: int


def write_skeleton_metadata(
    template: CatTemplate,
    beta: NDArray[np.floating],
    path: Path,
    *,
    mode: str,
) -> SkeletonMetadataReport:
    """Write fitted canonical skeleton and skinning metadata next to mesh exports."""

    payload = build_skeleton_metadata(template, beta, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validate_skeleton_metadata_file(
        path,
        expected_mode=mode,
        expected_vertex_count=int(template.vertices_canonical.shape[0]),
        expected_face_count=int(template.faces.shape[0]),
        expected_joint_count=int(template.joints_rest.shape[0]),
    )


def build_skeleton_metadata(
    template: CatTemplate,
    beta: NDArray[np.floating],
    *,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported skeleton export mode: {mode}")

    beta_array = np.asarray(beta, dtype=np.float64)
    if beta_array.shape != (template.shape_basis.shape[0],):
        raise ValueError(f"beta must have shape ({template.shape_basis.shape[0]},)")

    parents = np.asarray(template.joint_parents, dtype=np.int64)
    joints = np.asarray(template.joints(beta_array), dtype=np.float64)
    rest_joints = np.asarray(template.joints_rest, dtype=np.float64)
    skin_weights = np.asarray(template.skin_weights, dtype=np.float64)
    if joints.shape != rest_joints.shape or joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError("template joints must have shape [J,3]")
    if parents.shape != (joints.shape[0],):
        raise ValueError("joint parent array must match joint count")
    if skin_weights.shape != (template.vertices_canonical.shape[0], joints.shape[0]):
        raise ValueError("skin weights must have shape [V,J]")

    joint_names = _joint_names(int(joints.shape[0]))
    weight_row_sums = skin_weights.sum(axis=1)
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": mode,
        "topology": {
            "vertex_count": int(template.vertices_canonical.shape[0]),
            "face_count": int(template.faces.shape[0]),
            "joint_count": int(joints.shape[0]),
        },
        "identity_beta": [_rounded_float(value) for value in beta_array],
        "root_joint_index": _root_joint_index(parents),
        "parents": [int(parent) for parent in parents],
        "joints": [
            {
                "index": index,
                "name": joint_names[index],
                "parent_index": int(parents[index]),
                "position": _rounded_vector(joints[index]),
                "rest_position": _rounded_vector(rest_joints[index]),
            }
            for index in range(joints.shape[0])
        ],
        "skin_weights": {
            "format": "sparse_per_vertex",
            "vertex_count": int(skin_weights.shape[0]),
            "joint_count": int(skin_weights.shape[1]),
            "row_sum_min": _rounded_float(np.min(weight_row_sums)),
            "row_sum_max": _rounded_float(np.max(weight_row_sums)),
            "entries": _sparse_skin_weight_entries(skin_weights),
        },
    }
    report = validate_skeleton_metadata_payload(
        metadata,
        expected_mode=mode,
        expected_vertex_count=int(template.vertices_canonical.shape[0]),
        expected_face_count=int(template.faces.shape[0]),
        expected_joint_count=int(joints.shape[0]),
        path=Path("skeleton.json"),
    )
    metadata["validation"] = {
        "success": report.success,
        "checks": report.checks,
    }
    return metadata


def validate_skeleton_metadata_file(
    path: Path,
    *,
    expected_mode: str | None = None,
    expected_vertex_count: int | None = None,
    expected_face_count: int | None = None,
    expected_joint_count: int | None = None,
) -> SkeletonMetadataReport:
    if not path.exists():
        return SkeletonMetadataReport(
            path=path,
            success=False,
            checks={"path_exists": False},
            joint_count=0,
            vertex_count=0,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SkeletonMetadataReport(
            path=path,
            success=False,
            checks={"path_exists": True, "json_object": False},
            joint_count=0,
            vertex_count=0,
        )
    return validate_skeleton_metadata_payload(
        payload,
        expected_mode=expected_mode,
        expected_vertex_count=expected_vertex_count,
        expected_face_count=expected_face_count,
        expected_joint_count=expected_joint_count,
        path=path,
    )


def validate_skeleton_metadata_payload(
    payload: object,
    *,
    expected_mode: str | None = None,
    expected_vertex_count: int | None = None,
    expected_face_count: int | None = None,
    expected_joint_count: int | None = None,
    path: Path,
) -> SkeletonMetadataReport:
    if not isinstance(payload, Mapping):
        return SkeletonMetadataReport(
            path=path,
            success=False,
            checks={"json_object": False},
            joint_count=0,
            vertex_count=0,
        )

    topology = _mapping(payload.get("topology"))
    joints = payload.get("joints")
    parents = payload.get("parents")
    skin_weights = _mapping(payload.get("skin_weights"))
    entries = skin_weights.get("entries")
    vertex_count = _int(topology.get("vertex_count"), default=0)
    face_count = _int(topology.get("face_count"), default=0)
    joint_count = _int(topology.get("joint_count"), default=0)

    joint_rows = joints if isinstance(joints, list) else []
    parent_rows = parents if isinstance(parents, list) else []
    skin_entries = entries if isinstance(entries, list) else []
    weight_stats = _skin_weight_stats(skin_entries, vertex_count, joint_count)

    checks = {
        "json_object": True,
        "schema_version": payload.get("schema_version") == "1.0",
        "mode": expected_mode is None or payload.get("mode") == expected_mode,
        "topology_counts_positive": vertex_count > 0 and face_count > 0 and joint_count > 0,
        "expected_vertex_count": expected_vertex_count is None
        or vertex_count == expected_vertex_count,
        "expected_face_count": expected_face_count is None or face_count == expected_face_count,
        "expected_joint_count": expected_joint_count is None
        or joint_count == expected_joint_count,
        "joint_count_matches_payload": len(joint_rows) == joint_count
        and len(parent_rows) == joint_count,
        "joint_indices_ordered": _joint_indices_are_ordered(joint_rows),
        "joint_names_unique": _joint_names_are_unique(joint_rows),
        "joint_positions_finite": _joint_positions_are_finite(joint_rows),
        "single_root_joint": _single_root_joint(parent_rows),
        "parent_indices_valid": _parent_indices_are_valid(parent_rows, joint_count),
        "skeleton_acyclic": _parents_are_acyclic(parent_rows, joint_count),
        "skin_weight_format": skin_weights.get("format") == "sparse_per_vertex",
        "skin_weight_shape": _int(skin_weights.get("vertex_count"), default=-1)
        == vertex_count
        and _int(skin_weights.get("joint_count"), default=-1) == joint_count,
        "skin_weight_covers_vertices": weight_stats["covers_vertices"],
        "skin_weight_indices_valid": weight_stats["indices_valid"],
        "skin_weight_values_valid": weight_stats["values_valid"],
        "skin_weight_row_sums_valid": weight_stats["row_sums_valid"],
    }
    return SkeletonMetadataReport(
        path=path,
        success=all(checks.values()),
        checks=checks,
        joint_count=joint_count,
        vertex_count=vertex_count,
    )


def _joint_names(joint_count: int) -> list[str]:
    if len(JOINT_NAMES) == joint_count:
        return list(JOINT_NAMES)
    return [f"joint_{index:02d}" for index in range(joint_count)]


def _root_joint_index(parents: NDArray[np.int64]) -> int:
    roots = np.flatnonzero(parents == -1)
    return int(roots[0]) if roots.size else -1


def _sparse_skin_weight_entries(skin_weights: NDArray[np.float64]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for vertex_index, row in enumerate(skin_weights):
        joint_indices = np.flatnonzero(np.abs(row) > 1e-12)
        entries.append(
            {
                "vertex_index": vertex_index,
                "weights": [
                    {
                        "joint_index": int(joint_index),
                        "weight": _rounded_float(row[joint_index]),
                    }
                    for joint_index in joint_indices
                ],
            }
        )
    return entries


def _skin_weight_stats(
    entries: list[object],
    vertex_count: int,
    joint_count: int,
) -> dict[str, bool]:
    if vertex_count <= 0 or joint_count <= 0 or len(entries) != vertex_count:
        return {
            "covers_vertices": False,
            "indices_valid": False,
            "values_valid": False,
            "row_sums_valid": False,
        }

    seen_vertices: set[int] = set()
    indices_valid = True
    values_valid = True
    row_sums_valid = True
    for raw_entry in entries:
        entry = _mapping(raw_entry)
        vertex_index = _int(entry.get("vertex_index"), default=-1)
        if vertex_index < 0 or vertex_index >= vertex_count or vertex_index in seen_vertices:
            indices_valid = False
        seen_vertices.add(vertex_index)

        weights = entry.get("weights")
        if not isinstance(weights, list) or not weights:
            indices_valid = False
            row_sums_valid = False
            continue

        row_sum = 0.0
        seen_joints: set[int] = set()
        for raw_weight in weights:
            weight_entry = _mapping(raw_weight)
            joint_index = _int(weight_entry.get("joint_index"), default=-1)
            weight = _float(weight_entry.get("weight"))
            if joint_index < 0 or joint_index >= joint_count or joint_index in seen_joints:
                indices_valid = False
            seen_joints.add(joint_index)
            if weight is None or not np.isfinite(weight) or weight < -1e-12 or weight > 1.0 + 1e-12:
                values_valid = False
                continue
            row_sum += weight
        if abs(row_sum - 1.0) > 1e-8:
            row_sums_valid = False

    return {
        "covers_vertices": seen_vertices == set(range(vertex_count)),
        "indices_valid": indices_valid,
        "values_valid": values_valid,
        "row_sums_valid": row_sums_valid,
    }


def _joint_indices_are_ordered(joints: list[object]) -> bool:
    for expected_index, raw_joint in enumerate(joints):
        joint = _mapping(raw_joint)
        if _int(joint.get("index"), default=-1) != expected_index:
            return False
    return bool(joints)


def _joint_names_are_unique(joints: list[object]) -> bool:
    names = [str(_mapping(joint).get("name", "")) for joint in joints]
    return bool(names) and "" not in names and len(names) == len(set(names))


def _joint_positions_are_finite(joints: list[object]) -> bool:
    for raw_joint in joints:
        joint = _mapping(raw_joint)
        if not _finite_vector3(joint.get("position")):
            return False
        if not _finite_vector3(joint.get("rest_position")):
            return False
    return bool(joints)


def _single_root_joint(parents: list[object]) -> bool:
    return sum(1 for parent in parents if _int(parent, default=-2) == -1) == 1


def _parent_indices_are_valid(parents: list[object], joint_count: int) -> bool:
    if len(parents) != joint_count or joint_count <= 0:
        return False
    for index, raw_parent in enumerate(parents):
        parent = _int(raw_parent, default=-2)
        if parent == index:
            return False
        if parent != -1 and not 0 <= parent < joint_count:
            return False
    return True


def _parents_are_acyclic(parents: list[object], joint_count: int) -> bool:
    if not _parent_indices_are_valid(parents, joint_count):
        return False
    parent_values = [_int(parent, default=-2) for parent in parents]
    for index in range(joint_count):
        seen: set[int] = set()
        current = index
        while current != -1:
            if current in seen:
                return False
            seen.add(current)
            current = parent_values[current]
    return True


def _finite_vector3(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    numbers = [_float(item) for item in value]
    return all(number is not None and np.isfinite(number) for number in numbers)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: object, *, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _rounded_vector(values: NDArray[np.float64]) -> list[float]:
    return [_rounded_float(value) for value in values]


def _rounded_float(value: float | np.floating[Any]) -> float:
    return round(float(value), 9)
