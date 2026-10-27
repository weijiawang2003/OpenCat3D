"""Deterministic observed/unknown mesh coverage report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opencat3d.fitting.optimizer import load_fit_problem, load_fit_state
from opencat3d.synthetic.renderer import render_buffers
from opencat3d.textures.basecolor import texture_size_for_mode, write_observed_basecolor_texture
from opencat3d.textures.pbr import write_neutral_pbr_textures
from opencat3d.textures.provenance import write_texture_provenance_images

PROVENANCE_LABELS: dict[str, int] = {
    "unknown": 0,
    "observed": 1,
    "inferred": 2,
    "generated": 3,
}


@dataclass(frozen=True)
class MeshCoverageResult:
    report_path: Path
    provenance_path: Path
    texture_manifest_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


def evaluate_mesh_coverage(
    project_dir: Path,
    *,
    mode: str = "synthetic",
    reuse_existing_basecolor: bool = False,
    force: bool = False,
) -> MeshCoverageResult:
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported coverage mode: {mode}")

    output_dir = project_dir / "outputs" / "quality"
    report_path = output_dir / "coverage.json"
    provenance_path = output_dir / "coverage_provenance.npz"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")
    if provenance_path.exists() and not force:
        raise FileExistsError(f"{provenance_path} already exists; pass force=True to overwrite")

    problem = load_fit_problem(project_dir, mode=mode)
    fit_state = load_fit_state(
        project_dir,
        mode=mode,
        expected_frame_count=len(problem.observations),
        expected_joint_count=int(problem.template.joints_rest.shape[0]),
    )
    beta = fit_state.beta
    root_translations = fit_state.root_translations

    face_count = int(problem.template.faces.shape[0])
    vertex_count = int(problem.template.vertices_canonical.shape[0])
    face_view_counts = np.zeros(face_count, dtype=np.int32)
    face_pixel_counts = np.zeros(face_count, dtype=np.int64)
    visible_faces_per_frame: list[int] = []

    shaped_vertices = problem.template.shaped_vertices(beta)
    for observation, translation in zip(problem.observations, root_translations, strict=True):
        buffers = render_buffers(
            shaped_vertices + translation[None, :],
            problem.template.faces,
            observation.camera,
        )
        visible_faces = np.unique(buffers.face_index[buffers.face_index >= 0])
        visible_faces = visible_faces.astype(np.int64)
        visible_faces_per_frame.append(int(visible_faces.size))
        face_view_counts[visible_faces] += 1
        for face_index in visible_faces:
            face_pixel_counts[face_index] += int(np.count_nonzero(buffers.face_index == face_index))

    face_provenance = np.full(face_count, PROVENANCE_LABELS["unknown"], dtype=np.int8)
    observed_faces = face_view_counts > 0
    face_provenance[observed_faces] = PROVENANCE_LABELS["observed"]
    vertex_provenance = _vertex_provenance(problem.template.faces, face_provenance, vertex_count)

    observed_face_count = int(np.count_nonzero(face_provenance == PROVENANCE_LABELS["observed"]))
    unknown_face_count = int(np.count_nonzero(face_provenance == PROVENANCE_LABELS["unknown"]))
    observed_vertex_count = int(
        np.count_nonzero(vertex_provenance == PROVENANCE_LABELS["observed"])
    )
    unknown_vertex_count = int(
        np.count_nonzero(vertex_provenance == PROVENANCE_LABELS["unknown"])
    )
    metrics: dict[str, float | int] = {
        "num_frames": int(len(problem.observations)),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "observed_face_count": observed_face_count,
        "unknown_face_count": unknown_face_count,
        "inferred_face_count": 0,
        "generated_face_count": 0,
        "observed_face_fraction": observed_face_count / max(face_count, 1),
        "observed_vertex_count": observed_vertex_count,
        "unknown_vertex_count": unknown_vertex_count,
        "inferred_vertex_count": 0,
        "generated_vertex_count": 0,
        "observed_vertex_fraction": observed_vertex_count / max(vertex_count, 1),
        "mean_visible_faces_per_frame": float(np.mean(visible_faces_per_frame))
        if visible_faces_per_frame
        else 0.0,
        "max_visible_faces_per_frame": int(max(visible_faces_per_frame, default=0)),
        "observed_face_pixel_count": int(face_pixel_counts.sum()),
    }
    checks = {
        "fit_state_exists": fit_state.path.exists(),
        "fit_state_arrays_finite": bool(
            np.all(np.isfinite(beta)) and np.all(np.isfinite(root_translations))
        ),
        "frame_count_matches": root_translations.shape == (len(problem.observations), 3),
        "template_vertex_count": vertex_count > 0,
        "template_face_count": face_count > 0,
        "observed_faces_present": observed_face_count > 0,
        "no_generated_regions_claimed": int(metrics["generated_face_count"]) == 0
        and int(metrics["generated_vertex_count"]) == 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        provenance_path,
        face_provenance=face_provenance,
        vertex_provenance=vertex_provenance,
        face_view_counts=face_view_counts,
        face_pixel_counts=face_pixel_counts,
    )
    texture_result = write_texture_provenance_images(
        project_dir,
        face_provenance=face_provenance,
        provenance_labels=PROVENANCE_LABELS,
        force=force,
    )
    basecolor_result = write_observed_basecolor_texture(
        project_dir,
        problem,
        mode=mode,
        beta=beta,
        root_translations=root_translations,
        texture_size=texture_size_for_mode(mode, project_dir=project_dir),
        force=force and not reuse_existing_basecolor,
    )
    pbr_result = write_neutral_pbr_textures(project_dir, force=force)
    metrics.update(
        {
            "basecolor_observed_face_count": basecolor_result.observed_face_count,
            "basecolor_unknown_face_count": basecolor_result.unknown_face_count,
            "basecolor_observed_texel_fraction": basecolor_result.observed_texel_fraction,
            "basecolor_generated_texel_fraction": 0.0,
            "pbr_observed_texel_fraction": pbr_result.observed_texel_fraction,
            "pbr_unknown_texel_fraction": pbr_result.unknown_texel_fraction,
            "pbr_generated_texel_fraction": pbr_result.generated_texel_fraction,
            "pbr_face_material_detail_applied": float(
                pbr_result.face_material_detail_applied
            ),
            "pbr_face_material_detail_feature_count": int(
                pbr_result.face_material_detail_feature_count
            ),
            "pbr_face_material_detail_texel_count": int(
                pbr_result.face_material_detail_texel_count
            ),
            "pbr_face_material_detail_normal_variance": float(
                pbr_result.face_material_detail_normal_variance
            ),
        }
    )
    checks["provenance_arrays_written"] = provenance_path.exists()
    checks["texture_provenance_written"] = (
        texture_result.manifest_path.exists()
        and texture_result.provenance_path.exists()
        and texture_result.confidence_path.exists()
    )
    checks["texture_basecolor_written"] = (
        basecolor_result.manifest_path.exists()
        and basecolor_result.basecolor_path.exists()
        and basecolor_result.confidence_path.exists()
        and basecolor_result.source_view_id_path.exists()
        and basecolor_result.observed_texel_fraction > 0.0
    )
    checks["pbr_textures_written"] = (
        pbr_result.manifest_path.exists()
        and pbr_result.normal_path.exists()
        and pbr_result.roughness_path.exists()
        and pbr_result.ao_path.exists()
        and pbr_result.observed_texel_fraction > 0.0
        and pbr_result.generated_texel_fraction == 0.0
    )
    success = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "success": success,
        "checks": checks,
        "metrics": metrics,
        "provenance_labels": PROVENANCE_LABELS,
        "artifacts": {
            "fit_state": str(fit_state.path),
            "fit_state_source_format": fit_state.source_format,
            "provenance_arrays": str(provenance_path),
            "texture_manifest": str(texture_result.manifest_path),
            "provenance_png": str(texture_result.provenance_path),
            "confidence_png": str(texture_result.confidence_path),
            "basecolor_png": str(basecolor_result.basecolor_path),
            "basecolor_confidence_png": str(basecolor_result.confidence_path),
            "source_view_id_png": str(basecolor_result.source_view_id_path),
            "normal_png": str(pbr_result.normal_path),
            "roughness_png": str(pbr_result.roughness_path),
            "ao_png": str(pbr_result.ao_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MeshCoverageResult(
        report_path=report_path,
        provenance_path=provenance_path,
        texture_manifest_path=texture_result.manifest_path,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def _vertex_provenance(
    faces: NDArray[np.integer],
    face_provenance: NDArray[np.integer],
    vertex_count: int,
) -> NDArray[np.int8]:
    vertex_provenance = np.full(vertex_count, PROVENANCE_LABELS["unknown"], dtype=np.int8)
    observed_faces = np.asarray(faces, dtype=np.int64)[
        face_provenance == PROVENANCE_LABELS["observed"]
    ]
    if observed_faces.size:
        vertex_provenance[np.unique(observed_faces)] = PROVENANCE_LABELS["observed"]
    return vertex_provenance
