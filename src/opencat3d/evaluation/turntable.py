"""Deterministic face-anchor turntable render checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from opencat3d.camera.conventions import make_perspective_camera, project_world_points
from opencat3d.evaluation.glb_face_features import REQUIRED_GLB_FACE_FEATURE_GROUPS
from opencat3d.export.mesh import read_glb_metadata, read_obj
from opencat3d.fitting.optimizer import load_fit_problem, load_fit_state
from opencat3d.geometry.detail import face_surface_anchor_vertex_groups
from opencat3d.synthetic.renderer import render_buffers

REQUIRED_FACE_TURNTABLE_ANCHORS = (
    "left_eye_core",
    "right_eye_core",
    "nose_tip",
    "mouth_groove",
)

_ANCHOR_COLORS: dict[str, tuple[int, int, int]] = {
    "left_eye_core": (32, 72, 112),
    "right_eye_core": (32, 72, 112),
    "left_eye_socket": (70, 92, 118),
    "right_eye_socket": (70, 92, 118),
    "nose_tip": (212, 136, 132),
    "left_muzzle_pad": (236, 226, 200),
    "right_muzzle_pad": (236, 226, 200),
    "mouth_groove": (72, 50, 48),
    "left_ear_inner_bowl": (214, 166, 156),
    "right_ear_inner_bowl": (214, 166, 156),
}
_FACE_CLOSEUP_SIZE = 512
_FACE_CLOSEUP_PADDING = 2.25
_MIN_FACE_CLOSEUP_UNIQUE_RGB = 4
_MIN_FACE_CLOSEUP_LUMA_STD = 0.001
_FACE_SOURCE_COMPARE_TILE_SIZE = 96
_IRIS_SOURCE_COMPARISON_FEATURES = ("left_eye_iris", "right_eye_iris")
_FACE_CLOSEUP_FEATURE_COLORS: dict[str, tuple[int, int, int]] = {
    "left_eye_shadow": (60, 120, 255),
    "right_eye_shadow": (60, 120, 255),
    "left_eye_core": (32, 72, 180),
    "right_eye_core": (32, 72, 180),
    "left_eye_iris": (0, 220, 255),
    "right_eye_iris": (0, 220, 255),
    "left_eye_pupil": (255, 255, 255),
    "right_eye_pupil": (255, 255, 255),
    "nose": (255, 96, 132),
    "mouth_line": (255, 190, 64),
    "left_muzzle": (255, 255, 128),
    "right_muzzle": (255, 255, 128),
    "left_ear_inner": (220, 80, 220),
    "right_ear_inner": (220, 80, 220),
}


@dataclass(frozen=True)
class FaceTurntableEvaluationResult:
    report_path: Path
    manifest_path: Path
    success: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int]


def evaluate_face_turntable_report(
    project_dir: Path,
    *,
    mode: str = "raw",
    image_size: int = 256,
    force: bool = False,
) -> FaceTurntableEvaluationResult:
    """Render deterministic turntable overlays and score face-anchor visibility."""

    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported turntable mode: {mode}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    output_dir = project_dir / "outputs" / "quality"
    turntable_dir = output_dir / f"{mode}_face_turntable"
    report_path = output_dir / f"{mode}_face_turntable.json"
    manifest_path = turntable_dir / "manifest.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")
    if turntable_dir.exists() and any(turntable_dir.iterdir()) and not force:
        raise FileExistsError(f"{turntable_dir} already exists; pass force=True to overwrite")
    turntable_dir.mkdir(parents=True, exist_ok=True)

    obj_path = project_dir / "outputs" / "avatar" / "cat_canonical.obj"
    glb_path = project_dir / "outputs" / "avatar" / "cat_avatar.glb"
    mesh = read_obj(obj_path)
    problem = load_fit_problem(project_dir, mode=mode)
    fit_state = load_fit_state(
        project_dir,
        mode=mode,
        expected_frame_count=len(problem.observations),
        expected_joint_count=int(problem.template.joints_rest.shape[0]),
    )
    base_vertices = problem.template.shaped_vertices(fit_state.beta)
    anchor_groups = face_surface_anchor_vertex_groups(problem.template, base_vertices)
    anchor_faces = _anchor_faces(mesh.faces, anchor_groups)
    texture_feature_items = _texture_manifest_feature_items(project_dir)

    views = _turntable_views(image_size)
    frame_reports = []
    for view_name, camera in views:
        buffers = render_buffers(mesh.vertices, mesh.faces, camera)
        anchor_pixel_counts = _anchor_pixel_counts(buffers.face_index, anchor_faces)
        overlay_path = turntable_dir / f"{view_name}.png"
        _write_anchor_overlay(buffers.face_index, anchor_faces, overlay_path)
        textured_path = turntable_dir / f"{view_name}_textured.png"
        textured_metrics = _write_textured_turntable_frame(
            mesh.vertices,
            mesh.faces,
            problem.template.uv_vertices,
            problem.template.uv_faces,
            camera,
            project_dir / "outputs" / "textures" / "basecolor.png",
            textured_path,
            feature_items=texture_feature_items,
        )
        visible_anchor_names = [
            name for name, count in sorted(anchor_pixel_counts.items()) if count > 0
        ]
        frame_reports.append(
            {
                "view": view_name,
                "overlay_path": str(overlay_path),
                "textured_path": str(textured_path),
                "silhouette_pixel_count": int(np.count_nonzero(buffers.face_index >= 0)),
                "visible_anchor_count": len(visible_anchor_names),
                "visible_anchor_names": visible_anchor_names,
                "anchor_pixel_counts": anchor_pixel_counts,
                **textured_metrics,
            }
        )

    face_closeup_path = turntable_dir / "face_closeup.png"
    face_source_comparison_path = turntable_dir / "face_source_comparison.png"
    face_closeup = _write_face_texture_closeup(
        project_dir,
        face_closeup_path,
        source_comparison_path=face_source_comparison_path,
        image_size=_FACE_CLOSEUP_SIZE,
    )
    front = next(frame for frame in frame_reports if frame["view"] == "front")
    front_counts = front["anchor_pixel_counts"]
    required_visible = {
        name: int(front_counts.get(name, 0)) > 0 for name in REQUIRED_FACE_TURNTABLE_ANCHORS
    }
    glb_face_detail = _glb_face_detail_payload(glb_path)
    glb_anchor_names = {
        str(anchor.get("name"))
        for anchor in glb_face_detail.get("surface_anchors", [])
        if isinstance(anchor, dict)
    }
    required_visible_count = int(sum(required_visible.values()))
    textured_written_count = int(
        sum(1 for frame in frame_reports if bool(frame["textured_written"]))
    )
    textured_luma_stds = [
        float(frame["textured_luma_std"])
        for frame in frame_reports
        if bool(frame["textured_written"])
    ]
    textured_unique_counts = [
        int(frame["textured_unique_rgb_count"])
        for frame in frame_reports
        if bool(frame["textured_written"])
    ]
    source_comparison_expected = (
        int(face_closeup["source_comparison_source_feature_available_count"]) > 0
    )
    source_comparison_all_required_expected = (
        int(face_closeup["source_comparison_source_feature_available_count"])
        >= int(face_closeup["source_comparison_required_feature_count"])
    )
    metrics: dict[str, float | int] = {
        "turntable_view_count": len(frame_reports),
        "front_visible_anchor_count": int(front["visible_anchor_count"]),
        "front_anchor_pixel_count": int(sum(int(value) for value in front_counts.values())),
        "required_anchor_count": len(REQUIRED_FACE_TURNTABLE_ANCHORS),
        "required_anchor_visible_count": required_visible_count,
        "required_anchor_visible_fraction": required_visible_count
        / max(len(REQUIRED_FACE_TURNTABLE_ANCHORS), 1),
        "glb_surface_anchor_count": int(glb_face_detail.get("surface_anchor_count", 0)),
        "textured_turntable_view_count": len(frame_reports),
        "textured_turntable_written_count": textured_written_count,
        "textured_turntable_front_unique_rgb_count": int(
            front["textured_unique_rgb_count"]
        ),
        "textured_turntable_front_luma_std": float(front["textured_luma_std"]),
        "textured_turntable_front_mouth_line_pixel_count": int(
            front["textured_feature_pixel_counts"].get("mouth_line", 0)
        ),
        "textured_turntable_front_required_feature_visible_count": int(
            front["textured_required_feature_visible_count"]
        ),
        "textured_turntable_front_required_feature_fraction": float(
            front["textured_required_feature_fraction"]
        ),
        "textured_turntable_mean_unique_rgb_count": float(
            np.mean(textured_unique_counts) if textured_unique_counts else 0.0
        ),
        "textured_turntable_mean_luma_std": float(
            np.mean(textured_luma_stds) if textured_luma_stds else 0.0
        ),
        "face_texture_closeup_width": int(face_closeup["width"]),
        "face_texture_closeup_height": int(face_closeup["height"]),
        "face_texture_closeup_feature_count": int(face_closeup["feature_count"]),
        "face_texture_closeup_required_feature_count": int(
            face_closeup["required_feature_count"]
        ),
        "face_texture_closeup_required_present_count": int(
            face_closeup["required_present_count"]
        ),
        "face_texture_closeup_required_feature_fraction": float(
            face_closeup["required_feature_fraction"]
        ),
        "face_texture_closeup_crop_texel_count": int(
            face_closeup["crop_texel_count"]
        ),
        "face_texture_closeup_unique_rgb_count": int(
            face_closeup["unique_rgb_count"]
        ),
        "face_texture_closeup_luma_std": float(face_closeup["luma_std"]),
        "face_texture_source_compare_feature_count": int(
            face_closeup["source_comparison_feature_count"]
        ),
        "face_texture_source_compare_source_feature_available_count": int(
            face_closeup["source_comparison_source_feature_available_count"]
        ),
        "face_texture_source_compare_required_feature_count": int(
            face_closeup["source_comparison_required_feature_count"]
        ),
        "face_texture_source_compare_required_present_count": int(
            face_closeup["source_comparison_required_present_count"]
        ),
        "face_texture_source_compare_required_feature_fraction": float(
            face_closeup["source_comparison_required_feature_fraction"]
        ),
        "face_texture_source_compare_mean_rgb_distance": float(
            face_closeup["source_comparison_mean_rgb_distance"]
        ),
        "face_texture_source_compare_max_rgb_distance": float(
            face_closeup["source_comparison_max_rgb_distance"]
        ),
        "face_texture_source_compare_mean_luma_delta": float(
            face_closeup["source_comparison_mean_luma_delta"]
        ),
        "face_texture_source_compare_max_luma_delta": float(
            face_closeup["source_comparison_max_luma_delta"]
        ),
        "face_texture_source_compare_source_mean_luma_std": float(
            face_closeup["source_comparison_source_mean_luma_std"]
        ),
        "face_texture_source_compare_baked_mean_luma_std": float(
            face_closeup["source_comparison_baked_mean_luma_std"]
        ),
        "face_texture_source_compare_iris_feature_count": int(
            face_closeup["source_comparison_iris_feature_count"]
        ),
        "face_texture_source_compare_iris_candidate_texel_count": int(
            face_closeup["source_comparison_iris_candidate_texel_count"]
        ),
        "face_texture_source_compare_iris_core_accepted_texel_count": int(
            face_closeup["source_comparison_iris_core_accepted_texel_count"]
        ),
        "face_texture_source_compare_iris_core_densified_texel_count": int(
            face_closeup["source_comparison_iris_core_densified_texel_count"]
        ),
        "face_texture_source_compare_iris_annulus_rejected_texel_count": int(
            face_closeup["source_comparison_iris_annulus_rejected_texel_count"]
        ),
        "face_texture_source_compare_iris_core_acceptance_fraction": float(
            face_closeup["source_comparison_iris_core_acceptance_fraction"]
        ),
        "face_texture_source_compare_iris_annulus_rejected_fraction": float(
            face_closeup["source_comparison_iris_annulus_rejected_fraction"]
        ),
    }
    checks = {
        "obj_exists": obj_path.exists(),
        "glb_exists": glb_path.exists(),
        "turntable_overlays_written": all(
            Path(str(frame["overlay_path"])).exists() for frame in frame_reports
        ),
        "turntable_textured_written": textured_written_count == len(frame_reports),
        "front_textured_has_detail": (
            int(front["textured_unique_rgb_count"]) >= 2
            and float(front["textured_luma_std"]) > 0.0
        ),
        "front_textured_mouth_line_visible": (
            int(front["textured_feature_pixel_counts"].get("mouth_line", 0)) > 0
        ),
        "front_silhouette_visible": int(front["silhouette_pixel_count"]) > 0,
        "front_required_anchors_visible": all(required_visible.values()),
        "glb_face_detail_geometry_present": bool(glb_face_detail),
        "glb_required_anchors_declared": set(REQUIRED_FACE_TURNTABLE_ANCHORS).issubset(
            glb_anchor_names
        ),
        "face_texture_manifest_exists": bool(face_closeup["texture_manifest_exists"]),
        "face_texture_basecolor_exists": bool(face_closeup["basecolor_exists"]),
        "face_texture_closeup_written": bool(face_closeup["written"]),
        "face_texture_closeup_required_features_present": bool(
            face_closeup["required_features_present"]
        ),
        "face_texture_closeup_has_detail": bool(face_closeup["has_detail"]),
        "face_texture_source_comparison_written": bool(
            not source_comparison_expected or face_closeup["source_comparison_written"]
        ),
        "face_texture_source_comparison_required_features_present": bool(
            not source_comparison_all_required_expected
            or face_closeup["source_comparison_required_features_present"]
        ),
        "face_texture_source_comparison_has_samples": bool(
            not source_comparison_expected or face_closeup["source_comparison_has_samples"]
        ),
    }
    success = all(checks.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "mode": mode,
        "source": "opencat3d.face_anchor_turntable_v1",
        "success": success,
        "checks": checks,
        "metrics": metrics,
        "required_anchors": list(REQUIRED_FACE_TURNTABLE_ANCHORS),
        "front_required_anchor_visible": required_visible,
        "frames": frame_reports,
        "face_texture_closeup": face_closeup,
        "artifacts": {
            "obj": str(obj_path),
            "glb": str(glb_path),
            "manifest": str(manifest_path),
            "face_closeup": str(face_closeup_path),
            "face_source_comparison": str(face_source_comparison_path),
            "front_textured": str(turntable_dir / "front_textured.png"),
        },
    }
    manifest = {
        "schema_version": "1.0",
        "source": "opencat3d.face_anchor_turntable_v1",
        "image_size": image_size,
        "frames": frame_reports,
        "face_texture_closeup": face_closeup,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FaceTurntableEvaluationResult(
        report_path=report_path,
        manifest_path=manifest_path,
        success=success,
        checks=checks,
        metrics=metrics,
    )


def _turntable_views(image_size: int) -> tuple[tuple[str, Any], ...]:
    target = np.asarray([0.0, 0.88, 0.95], dtype=np.float64)
    return (
        (
            "front",
            make_perspective_camera(
                image_size,
                image_size,
                30.0,
                eye=np.asarray([0.0, 0.92, 3.8], dtype=np.float64),
                target=target,
            ),
        ),
        (
            "left_quarter",
            make_perspective_camera(
                image_size,
                image_size,
                34.0,
                eye=np.asarray([-1.8, 0.92, 3.4], dtype=np.float64),
                target=target,
            ),
        ),
        (
            "right_quarter",
            make_perspective_camera(
                image_size,
                image_size,
                34.0,
                eye=np.asarray([1.8, 0.92, 3.4], dtype=np.float64),
                target=target,
            ),
        ),
    )


def _anchor_faces(
    faces: NDArray[np.integer],
    anchor_groups: tuple[Any, ...],
) -> dict[str, NDArray[np.int64]]:
    faces_array = np.asarray(faces, dtype=np.int64)
    result: dict[str, NDArray[np.int64]] = {}
    for group in anchor_groups:
        mask = np.any(np.isin(faces_array, group.vertex_indices), axis=1)
        result[group.name] = np.nonzero(mask)[0].astype(np.int64)
    return result


def _anchor_pixel_counts(
    face_index: NDArray[np.integer],
    anchor_faces: dict[str, NDArray[np.int64]],
) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(np.isin(face_index, faces)))
        for name, faces in sorted(anchor_faces.items())
    }


def _write_anchor_overlay(
    face_index: NDArray[np.integer],
    anchor_faces: dict[str, NDArray[np.int64]],
    path: Path,
) -> None:
    image = np.full((*face_index.shape, 3), 255, dtype=np.uint8)
    silhouette = face_index >= 0
    image[silhouette] = np.asarray([214, 216, 210], dtype=np.uint8)
    for name, faces in sorted(anchor_faces.items()):
        mask = np.isin(face_index, faces)
        if not np.any(mask):
            continue
        image[mask] = np.asarray(_ANCHOR_COLORS.get(name, (40, 120, 200)), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _write_textured_turntable_frame(
    vertices: NDArray[np.floating],
    faces: NDArray[np.integer],
    uv_vertices: NDArray[np.floating],
    uv_faces: NDArray[np.integer],
    camera: Any,
    basecolor_path: Path,
    path: Path,
    *,
    feature_items: dict[str, Any],
) -> dict[str, Any]:
    required_names = _required_face_feature_names()
    metrics: dict[str, Any] = {
        "textured_written": False,
        "textured_silhouette_pixel_count": 0,
        "textured_unique_rgb_count": 0,
        "textured_luma_std": 0.0,
        "textured_mean_rgb": [0.0, 0.0, 0.0],
        "textured_feature_pixel_counts": {},
        "textured_required_feature_visible_count": 0,
        "textured_required_feature_fraction": 0.0,
        "textured_mouth_line_pixel_count": 0,
    }
    if not basecolor_path.exists():
        return metrics
    uv_faces_array = np.asarray(uv_faces, dtype=np.int64)
    if uv_faces_array.ndim != 2 or uv_faces_array.shape[1] != 3:
        return metrics
    face_count = int(uv_faces_array.shape[0])
    faces_array = np.asarray(faces, dtype=np.int64)
    if faces_array.shape != uv_faces_array.shape:
        return metrics
    texture = np.asarray(Image.open(basecolor_path).convert("RGB"), dtype=np.uint8)
    image = np.full((camera.image_height, camera.image_width, 3), 255, dtype=np.uint8)
    visible, uv_image = _rasterize_textured_mesh(
        image,
        np.asarray(vertices, dtype=np.float64),
        faces_array[:face_count],
        np.asarray(uv_vertices, dtype=np.float64),
        uv_faces_array,
        texture,
        camera,
    )
    feature_pixel_counts = _textured_feature_pixel_counts(
        uv_image,
        visible,
        feature_items,
    )
    required_feature_count = int(
        sum(1 for name in required_names if feature_pixel_counts.get(name, 0) > 0)
    )
    metrics["textured_feature_pixel_counts"] = feature_pixel_counts
    metrics["textured_required_feature_visible_count"] = required_feature_count
    metrics["textured_required_feature_fraction"] = required_feature_count / max(
        len(required_names),
        1,
    )
    metrics["textured_mouth_line_pixel_count"] = int(
        feature_pixel_counts.get("mouth_line", 0)
    )
    metrics["textured_silhouette_pixel_count"] = int(np.count_nonzero(visible))
    if np.any(visible):
        visible_rgb = image[visible].reshape(-1, 3)
        metrics["textured_unique_rgb_count"] = int(
            np.unique(visible_rgb, axis=0).shape[0]
        )
        visible_rgb_f = visible_rgb.astype(np.float64)
        luma = (
            visible_rgb_f
            @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
        ) / 255.0
        metrics["textured_luma_std"] = float(np.std(luma))
        metrics["textured_mean_rgb"] = [
            round(float(value), 6) for value in visible_rgb_f.mean(axis=0)
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    metrics["textured_written"] = path.exists()
    return metrics


def _rasterize_textured_mesh(
    image: NDArray[np.uint8],
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    uv_vertices: NDArray[np.float64],
    uv_faces: NDArray[np.int64],
    texture: NDArray[np.uint8],
    camera: Any,
) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    projected = project_world_points(vertices, camera)
    depth = np.full((camera.image_height, camera.image_width), np.inf, dtype=np.float32)
    visible = np.zeros((camera.image_height, camera.image_width), dtype=np.bool_)
    uv_image = np.zeros((camera.image_height, camera.image_width, 2), dtype=np.float32)
    for face, uv_face in zip(faces, uv_faces, strict=True):
        tri = projected[face]
        if np.any(tri[:, 2] <= 1e-6):
            continue
        _rasterize_textured_triangle(
            image,
            depth,
            visible,
            uv_image,
            tri,
            uv_vertices[uv_face],
            texture,
        )
    return visible, uv_image


def _rasterize_textured_triangle(
    image: NDArray[np.uint8],
    depth: NDArray[np.float32],
    visible: NDArray[np.bool_],
    uv_image: NDArray[np.float32],
    tri_xyz: NDArray[np.float64],
    tri_uv: NDArray[np.float64],
    texture: NDArray[np.uint8],
) -> None:
    height, width = depth.shape
    tri_xy = tri_xyz[:, :2]
    min_xy = np.floor(tri_xy.min(axis=0)).astype(np.int64)
    max_xy = np.ceil(tri_xy.max(axis=0)).astype(np.int64)
    x0 = max(int(min_xy[0]), 0)
    y0 = max(int(min_xy[1]), 0)
    x1 = min(int(max_xy[0]), width - 1)
    y1 = min(int(max_xy[1]), height - 1)
    if x1 < x0 or y1 < y0:
        return

    a, b, c = tri_xy
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(float(denom)) < 1e-12:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
    ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    w0 = ((b[1] - c[1]) * (grid_x - c[0]) + (c[0] - b[0]) * (grid_y - c[1])) / denom
    w1 = ((c[1] - a[1]) * (grid_x - c[0]) + (a[0] - c[0]) * (grid_y - c[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    z = w0 * tri_xyz[0, 2] + w1 * tri_xyz[1, 2] + w2 * tri_xyz[2, 2]

    depth_view = depth[y0 : y1 + 1, x0 : x1 + 1]
    update = inside & np.isfinite(z) & (z < depth_view - 1e-9)
    if not np.any(update):
        return

    uv = (
        w0[..., None] * tri_uv[0]
        + w1[..., None] * tri_uv[1]
        + w2[..., None] * tri_uv[2]
    )
    texture_height, texture_width = texture.shape[:2]
    clipped = np.clip(uv, 0.0, 1.0)
    tex_x = np.rint(clipped[..., 0] * float(texture_width - 1)).astype(np.int64)
    tex_y = np.rint((1.0 - clipped[..., 1]) * float(texture_height - 1)).astype(np.int64)
    sampled = texture[tex_y, tex_x]

    image_view = image[y0 : y1 + 1, x0 : x1 + 1]
    visible_view = visible[y0 : y1 + 1, x0 : x1 + 1]
    uv_view = uv_image[y0 : y1 + 1, x0 : x1 + 1]
    depth_view[update] = z[update].astype(np.float32)
    image_view[update] = sampled[update]
    visible_view[update] = True
    uv_view[update] = clipped[update].astype(np.float32)


def _texture_manifest_feature_items(project_dir: Path) -> dict[str, Any]:
    manifest_path = project_dir / "outputs" / "textures" / "texture_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    facial_features = _object(manifest.get("facial_features"))
    return {
        str(item.get("name", "")): item
        for item in _list(facial_features.get("features"))
        if isinstance(item, dict) and str(item.get("name", ""))
    }


def _textured_feature_pixel_counts(
    uv_image: NDArray[np.float32],
    visible: NDArray[np.bool_],
    feature_items: dict[str, Any],
) -> dict[str, int]:
    if not feature_items or not np.any(visible):
        return {}
    uv_pixels = uv_image.astype(np.float64, copy=False)
    counts: dict[str, int] = {}
    for name, item in sorted(feature_items.items()):
        if not isinstance(item, dict):
            continue
        uv = _float_pair(item.get("uv"), default=(0.5, 0.5))
        radius_uv = _float_pair(item.get("radius_uv"), default=(0.01, 0.01))
        radius_u = max(float(radius_uv[0]), 1e-6)
        radius_v = max(float(radius_uv[1]), 1e-6)
        normalized = (
            ((uv_pixels[..., 0] - uv[0]) / radius_u) ** 2
            + ((uv_pixels[..., 1] - uv[1]) / radius_v) ** 2
        )
        counts[name] = int(np.count_nonzero(visible & (normalized <= 1.0)))
    return counts


def _write_face_texture_closeup(
    project_dir: Path,
    path: Path,
    *,
    source_comparison_path: Path,
    image_size: int,
) -> dict[str, Any]:
    texture_manifest_path = project_dir / "outputs" / "textures" / "texture_manifest.json"
    basecolor_path = project_dir / "outputs" / "textures" / "basecolor.png"
    manifest = _read_json(texture_manifest_path) if texture_manifest_path.exists() else {}
    facial_features = _object(manifest.get("facial_features"))
    feature_items = {
        str(item.get("name", "")): item
        for item in _list(facial_features.get("features"))
        if isinstance(item, dict) and str(item.get("name", ""))
    }
    required_names = _required_face_feature_names()
    required_present = [name for name in required_names if name in feature_items]

    metrics: dict[str, Any] = {
        "path": str(path),
        "texture_manifest_path": str(texture_manifest_path),
        "basecolor_path": str(basecolor_path),
        "texture_manifest_exists": texture_manifest_path.exists(),
        "basecolor_exists": basecolor_path.exists(),
        "width": image_size,
        "height": image_size,
        "feature_count": len(feature_items),
        "required_feature_count": len(required_names),
        "required_present_count": len(required_present),
        "required_feature_fraction": len(required_present) / max(len(required_names), 1),
        "required_features_present": set(required_names).issubset(feature_items),
        "crop_texel_count": 0,
        "unique_rgb_count": 0,
        "luma_std": 0.0,
        "written": False,
        "has_detail": False,
        "crop_uv_bounds": [0.0, 0.0, 1.0, 1.0],
        "feature_names": sorted(feature_items),
        "required_feature_names": list(required_names),
        "source_comparison_path": str(source_comparison_path),
        "source_comparison_written": False,
        "source_comparison_source_feature_available_count": 0,
        "source_comparison_feature_count": 0,
        "source_comparison_required_feature_count": len(required_names),
        "source_comparison_required_present_count": 0,
        "source_comparison_required_feature_fraction": 0.0,
        "source_comparison_required_features_present": False,
        "source_comparison_has_samples": False,
        "source_comparison_mean_rgb_distance": 0.0,
        "source_comparison_max_rgb_distance": 0.0,
        "source_comparison_mean_luma_delta": 0.0,
        "source_comparison_max_luma_delta": 0.0,
        "source_comparison_source_mean_luma_std": 0.0,
        "source_comparison_baked_mean_luma_std": 0.0,
        "source_comparison_iris_feature_count": 0,
        "source_comparison_iris_candidate_texel_count": 0,
        "source_comparison_iris_core_accepted_texel_count": 0,
        "source_comparison_iris_core_densified_texel_count": 0,
        "source_comparison_iris_annulus_rejected_texel_count": 0,
        "source_comparison_iris_core_acceptance_fraction": 0.0,
        "source_comparison_iris_annulus_rejected_fraction": 0.0,
        "source_comparison_iris_features": [],
        "source_comparison_features": [],
    }
    if not basecolor_path.exists() or not feature_items:
        return metrics

    image = Image.open(basecolor_path).convert("RGB")
    width, height = image.size
    crop_bounds = _face_closeup_uv_bounds(feature_items.values())
    left, top, right, bottom = _uv_bounds_to_pixel_box(crop_bounds, width, height)
    crop = image.crop((left, top, right, bottom))
    crop_array = np.asarray(crop, dtype=np.uint8)
    metrics["crop_texel_count"] = int(crop_array.shape[0] * crop_array.shape[1])
    if crop_array.size:
        rgb = crop_array.reshape(-1, 3)
        metrics["unique_rgb_count"] = int(np.unique(rgb, axis=0).shape[0])
        luma = (
            rgb.astype(np.float64)
            @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
        ) / 255.0
        metrics["luma_std"] = float(np.std(luma))

    resampling = getattr(Image, "Resampling", Image)
    closeup = crop.resize((image_size, image_size), resampling.NEAREST)
    draw = ImageDraw.Draw(closeup)
    crop_width = max(right - left, 1)
    crop_height = max(bottom - top, 1)
    for name, item in sorted(feature_items.items()):
        uv = _float_pair(item.get("uv"), default=(0.5, 0.5))
        radius_uv = _float_pair(item.get("radius_uv"), default=(0.01, 0.01))
        center_x = ((uv[0] * (width - 1)) - left) / crop_width * image_size
        center_y = (((1.0 - uv[1]) * (height - 1)) - top) / crop_height * image_size
        radius_x = max(radius_uv[0] * width / crop_width * image_size, 3.0)
        radius_y = max(radius_uv[1] * height / crop_height * image_size, 3.0)
        color = _FACE_CLOSEUP_FEATURE_COLORS.get(name, (255, 255, 255))
        draw.ellipse(
            (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ),
            outline=color,
            width=3,
        )
        draw.line(
            ((center_x - 3, center_y), (center_x + 3, center_y)),
            fill=color,
            width=1,
        )
        draw.line(
            ((center_x, center_y - 3), (center_x, center_y + 3)),
            fill=color,
            width=1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    closeup.save(path)
    source_comparison = _write_face_source_comparison(
        project_dir,
        image,
        feature_items,
        source_comparison_path,
        required_names=required_names,
    )
    metrics.update(source_comparison)
    metrics["written"] = path.exists()
    metrics["has_detail"] = bool(
        int(metrics["unique_rgb_count"]) >= _MIN_FACE_CLOSEUP_UNIQUE_RGB
        and float(metrics["luma_std"]) >= _MIN_FACE_CLOSEUP_LUMA_STD
    )
    metrics["crop_uv_bounds"] = [float(value) for value in crop_bounds]
    metrics["crop_pixel_bounds"] = [left, top, right, bottom]
    return metrics


def _write_face_source_comparison(
    project_dir: Path,
    basecolor: Image.Image,
    feature_items: dict[str, Any],
    path: Path,
    *,
    required_names: tuple[str, ...],
) -> dict[str, Any]:
    base_array = np.asarray(basecolor, dtype=np.uint8)
    comparisons: list[dict[str, Any]] = []
    source_tiles: list[Image.Image] = []
    baked_tiles: list[Image.Image] = []
    source_feature_available_count = 0
    iris_metrics = _iris_source_comparison_metrics(feature_items)
    iris_feature_metrics = {
        str(item["name"]): item for item in iris_metrics["source_comparison_iris_features"]
    }
    for name, item in sorted(feature_items.items()):
        if not isinstance(item, dict):
            continue
        source_path = _resolve_project_path(project_dir, item.get("source_patch_path"))
        if source_path is not None and source_path.exists():
            source_feature_available_count += 1
        source_stats = _source_patch_stats(
            source_path,
            item.get("source_patch_mask_path"),
            project_dir,
        )
        baked_stats = _baked_feature_stats(base_array, item, project_dir=project_dir)
        if not source_stats["sampled"] or not baked_stats["sampled"]:
            continue
        rgb_distance = _rgb_distance(source_stats["mean_rgb"], baked_stats["mean_rgb"])
        luma_delta = abs(float(source_stats["mean_luma"]) - float(baked_stats["mean_luma"]))
        comparison = {
            "name": name,
            "source_path": str(source_path) if source_path is not None else "",
            "source_sample_count": int(source_stats["sample_count"]),
            "baked_sample_count": int(baked_stats["sample_count"]),
            "source_mean_rgb": _rounded_rgb(source_stats["mean_rgb"]),
            "baked_mean_rgb": _rounded_rgb(baked_stats["mean_rgb"]),
            "rgb_distance": round(float(rgb_distance), 6),
            "luma_delta": round(float(luma_delta), 6),
            "source_luma_std": round(float(source_stats["luma_std"]), 6),
            "baked_luma_std": round(float(baked_stats["luma_std"]), 6),
        }
        if name in iris_feature_metrics:
            comparison.update(iris_feature_metrics[name])
        comparisons.append(comparison)
        source_tiles.append(_comparison_tile(source_stats["tile"], name, "source"))
        baked_tiles.append(_comparison_tile(baked_stats["tile"], name, "baked"))

    if comparisons:
        _write_source_comparison_sheet(source_tiles, baked_tiles, path)

    required_present = [
        item["name"] for item in comparisons if str(item["name"]) in set(required_names)
    ]
    rgb_distances = [float(item["rgb_distance"]) for item in comparisons]
    luma_deltas = [float(item["luma_delta"]) for item in comparisons]
    source_luma_stds = [float(item["source_luma_std"]) for item in comparisons]
    baked_luma_stds = [float(item["baked_luma_std"]) for item in comparisons]
    return {
        "source_comparison_written": path.exists(),
        "source_comparison_source_feature_available_count": source_feature_available_count,
        "source_comparison_feature_count": len(comparisons),
        "source_comparison_required_feature_count": len(required_names),
        "source_comparison_required_present_count": len(required_present),
        "source_comparison_required_feature_fraction": len(required_present)
        / max(len(required_names), 1),
        "source_comparison_required_features_present": set(required_names).issubset(
            {str(item["name"]) for item in comparisons}
        ),
        "source_comparison_has_samples": bool(comparisons),
        "source_comparison_mean_rgb_distance": float(np.mean(rgb_distances))
        if rgb_distances
        else 0.0,
        "source_comparison_max_rgb_distance": float(np.max(rgb_distances))
        if rgb_distances
        else 0.0,
        "source_comparison_mean_luma_delta": float(np.mean(luma_deltas))
        if luma_deltas
        else 0.0,
        "source_comparison_max_luma_delta": float(np.max(luma_deltas))
        if luma_deltas
        else 0.0,
        "source_comparison_source_mean_luma_std": float(np.mean(source_luma_stds))
        if source_luma_stds
        else 0.0,
        "source_comparison_baked_mean_luma_std": float(np.mean(baked_luma_stds))
        if baked_luma_stds
        else 0.0,
        "source_comparison_features": comparisons,
        **iris_metrics,
    }


def _iris_source_comparison_metrics(
    feature_items: dict[str, Any],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    candidate_total = 0
    accepted_total = 0
    densified_total = 0
    rejected_total = 0
    for name in _IRIS_SOURCE_COMPARISON_FEATURES:
        item = _object(feature_items.get(name))
        if not item:
            continue
        accepted = _nonnegative_int(
            item.get("source_patch_source_mask_observed_texel_count")
        )
        densified = _nonnegative_int(
            item.get("source_patch_source_mask_densified_texel_count")
        )
        candidate = max(
            _nonnegative_int(item.get("sample_count")),
            accepted,
            densified,
        )
        rejected = max(candidate - accepted, 0)
        core_fraction = accepted / max(candidate, 1) if candidate else 0.0
        rejected_fraction = rejected / max(candidate, 1) if candidate else 0.0
        feature = {
            "name": name,
            "iris_candidate_texel_count": int(candidate),
            "iris_core_accepted_texel_count": int(accepted),
            "iris_core_densified_texel_count": int(densified),
            "iris_annulus_rejected_texel_count": int(rejected),
            "iris_core_acceptance_fraction": float(core_fraction),
            "iris_annulus_rejected_fraction": float(rejected_fraction),
        }
        features.append(feature)
        candidate_total += candidate
        accepted_total += accepted
        densified_total += densified
        rejected_total += rejected
    return {
        "source_comparison_iris_feature_count": len(features),
        "source_comparison_iris_candidate_texel_count": int(candidate_total),
        "source_comparison_iris_core_accepted_texel_count": int(accepted_total),
        "source_comparison_iris_core_densified_texel_count": int(densified_total),
        "source_comparison_iris_annulus_rejected_texel_count": int(rejected_total),
        "source_comparison_iris_core_acceptance_fraction": (
            accepted_total / max(candidate_total, 1) if candidate_total else 0.0
        ),
        "source_comparison_iris_annulus_rejected_fraction": (
            rejected_total / max(candidate_total, 1) if candidate_total else 0.0
        ),
        "source_comparison_iris_features": features,
    }


def _source_patch_stats(
    source_path: Path | None,
    mask_path_value: Any,
    project_dir: Path,
) -> dict[str, Any]:
    empty = _empty_sample_stats()
    if source_path is None or not source_path.exists():
        return empty
    source = Image.open(source_path).convert("RGB")
    source_array = np.asarray(source, dtype=np.uint8)
    mask_path = _resolve_project_path(project_dir, mask_path_value)
    if mask_path is not None and mask_path.exists():
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        if mask.shape != source_array.shape[:2]:
            mask = np.ones(source_array.shape[:2], dtype=bool)
    else:
        mask = np.ones(source_array.shape[:2], dtype=bool)
    return _sample_stats(source_array, mask, source)


def _baked_feature_stats(
    base_array: NDArray[np.uint8],
    item: dict[str, Any],
    *,
    project_dir: Path,
) -> dict[str, Any]:
    support_mask = _uv_support_mask(
        project_dir,
        item.get("uv_support_mask_path"),
        expected_shape=base_array.shape[:2],
    )
    if support_mask is not None and np.any(support_mask):
        ys, xs = np.nonzero(support_mask)
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        left = int(xs.min())
        right = int(xs.max()) + 1
        crop = base_array[top:bottom, left:right]
        tile = Image.fromarray(crop)
        return _sample_stats(crop, support_mask[top:bottom, left:right], tile)

    uv = _float_pair(item.get("uv"), default=(0.5, 0.5))
    radius_uv = _float_pair(item.get("radius_uv"), default=(0.01, 0.01))
    height, width = base_array.shape[:2]
    center_x = uv[0] * (width - 1)
    center_y = (1.0 - uv[1]) * (height - 1)
    radius_x = max(radius_uv[0] * width, 1.0)
    radius_y = max(radius_uv[1] * height, 1.0)
    left = max(0, int(np.floor(center_x - radius_x)))
    right = min(width, int(np.ceil(center_x + radius_x)) + 1)
    top = max(0, int(np.floor(center_y - radius_y)))
    bottom = min(height, int(np.ceil(center_y + radius_y)) + 1)
    crop = base_array[top:bottom, left:right]
    if crop.size == 0:
        return _empty_sample_stats()
    yy, xx = np.mgrid[top:bottom, left:right]
    mask = ((xx - center_x) / max(radius_x, 1.0)) ** 2 + (
        (yy - center_y) / max(radius_y, 1.0)
    ) ** 2 <= 1.0
    tile = Image.fromarray(crop)
    return _sample_stats(crop, mask, tile)


def _uv_support_mask(
    project_dir: Path,
    mask_path_value: Any,
    *,
    expected_shape: tuple[int, int],
) -> NDArray[np.bool_] | None:
    mask_path = _resolve_project_path(project_dir, mask_path_value)
    if mask_path is None or not mask_path.exists():
        return None
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    if mask.shape != expected_shape:
        return None
    return mask


def _sample_stats(
    image: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    tile: Image.Image,
) -> dict[str, Any]:
    pixels = image[mask]
    if pixels.size == 0:
        return _empty_sample_stats()
    pixels_f = pixels.astype(np.float64)
    pixels_rgb = pixels_f / 255.0
    luma = _luma(pixels_rgb)
    return {
        "sampled": True,
        "sample_count": int(pixels.shape[0]),
        "mean_rgb": np.mean(pixels_rgb, axis=0),
        "mean_luma": float(np.mean(luma)),
        "luma_std": float(np.std(luma)),
        "tile": tile,
    }


def _empty_sample_stats() -> dict[str, Any]:
    return {
        "sampled": False,
        "sample_count": 0,
        "mean_rgb": np.zeros(3, dtype=np.float64),
        "mean_luma": 0.0,
        "luma_std": 0.0,
        "tile": Image.new("RGB", (1, 1), (255, 255, 255)),
    }


def _write_source_comparison_sheet(
    source_tiles: list[Image.Image],
    baked_tiles: list[Image.Image],
    path: Path,
) -> None:
    tile_size = _FACE_SOURCE_COMPARE_TILE_SIZE
    label_height = 18
    row_height = tile_size + label_height
    width = tile_size * 2
    height = max(len(source_tiles), 1) * row_height
    sheet = Image.new("RGB", (width, height), (246, 246, 242))
    for row, (source, baked) in enumerate(zip(source_tiles, baked_tiles, strict=True)):
        y = row * row_height
        sheet.paste(source, (0, y))
        sheet.paste(baked, (tile_size, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _comparison_tile(tile: Image.Image, name: str, label: str) -> Image.Image:
    tile_size = _FACE_SOURCE_COMPARE_TILE_SIZE
    label_height = 18
    resampling = getattr(Image, "Resampling", Image)
    canvas = Image.new("RGB", (tile_size, tile_size + label_height), (246, 246, 242))
    resized = tile.convert("RGB").resize((tile_size, tile_size), resampling.BILINEAR)
    canvas.paste(resized, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((3, tile_size + 3), f"{name[:8]} {label}", fill=(24, 30, 34))
    return canvas


def _resolve_project_path(project_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return project_dir / path


def _luma(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    return rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _rgb_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(a - b))


def _rounded_rgb(rgb: NDArray[np.float64]) -> list[float]:
    return [round(float(value), 6) for value in rgb]


def _face_closeup_uv_bounds(features: Any) -> tuple[float, float, float, float]:
    u_min = 1.0
    v_min = 1.0
    u_max = 0.0
    v_max = 0.0
    for item in features:
        if not isinstance(item, dict):
            continue
        uv = _float_pair(item.get("uv"), default=(0.5, 0.5))
        radius_uv = _float_pair(item.get("radius_uv"), default=(0.01, 0.01))
        u_min = min(u_min, uv[0] - radius_uv[0] * _FACE_CLOSEUP_PADDING)
        u_max = max(u_max, uv[0] + radius_uv[0] * _FACE_CLOSEUP_PADDING)
        v_min = min(v_min, uv[1] - radius_uv[1] * _FACE_CLOSEUP_PADDING)
        v_max = max(v_max, uv[1] + radius_uv[1] * _FACE_CLOSEUP_PADDING)
    u_min = max(0.0, u_min)
    v_min = max(0.0, v_min)
    u_max = min(1.0, u_max)
    v_max = min(1.0, v_max)
    if u_max <= u_min or v_max <= v_min:
        return (0.0, 0.0, 1.0, 1.0)
    return (u_min, v_min, u_max, v_max)


def _uv_bounds_to_pixel_box(
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    u_min, v_min, u_max, v_max = bounds
    left = int(np.floor(u_min * (width - 1)))
    right = int(np.ceil(u_max * (width - 1))) + 1
    top = int(np.floor((1.0 - v_max) * (height - 1)))
    bottom = int(np.ceil((1.0 - v_min) * (height - 1))) + 1
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return (left, top, right, bottom)


def _required_face_feature_names() -> tuple[str, ...]:
    return tuple(
        name
        for group_names in REQUIRED_GLB_FACE_FEATURE_GROUPS.values()
        for name in group_names
    )


def _glb_face_detail_payload(glb_path: Path) -> dict[str, Any]:
    try:
        metadata = read_glb_metadata(glb_path)
        payload = (
            metadata.get("json", {})
            .get("asset", {})
            .get("extras", {})
            .get("opencat3d", {})
            .get("face_detail_geometry", {})
        )
    except (FileNotFoundError, KeyError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float_pair(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return default
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return default


def _nonnegative_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0
