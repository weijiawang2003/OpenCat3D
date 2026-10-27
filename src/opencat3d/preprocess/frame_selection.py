"""Deterministic raw frame selection before multi-view fitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FRAME_SELECTION_VERSION = "raw_template_compat_body_texture_v2"


@dataclass(frozen=True)
class RawFrameSelectionResult:
    project_dir: Path
    report_path: Path
    manifest_path: Path
    texture_manifest_path: Path
    input_frame_count: int
    selected_frame_count: int
    rejected_frame_count: int
    texture_frame_count: int
    selected_frame_ids: tuple[str, ...]
    mean_mask_area_ratio: float
    mean_bbox_area_ratio: float
    full_frame_bbox_fraction: float
    media_strategy: str
    input_media_counts: dict[str, int]
    selected_media_counts: dict[str, int]
    texture_media_counts: dict[str, int]
    detail_metrics: dict[str, float]


@dataclass(frozen=True)
class RawFrameSelectionHashMapResult:
    reference_project_dir: Path
    target_project_dir: Path
    selected_frame_ids: tuple[str, ...]
    texture_frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class RawTextureManifestDedupResult:
    project_dir: Path
    manifest_path: Path
    report_path: Path
    input_texture_frame_count: int
    deduplicated_texture_frame_count: int
    removed_duplicate_frame_ids: tuple[str, ...]
    duplicate_group_count: int


TEXTURE_MANIFEST_DEDUP_VERSION = "raw_texture_image_hash_v1"


def deduplicate_raw_texture_manifest_by_image_hash(
    project_dir: Path,
    *,
    force: bool = False,
) -> RawTextureManifestDedupResult:
    """Remove duplicate texture frames while preserving selected geometry frames."""

    project_dir = Path(project_dir)
    observations_dir = project_dir / "observations" / "raw"
    texture_manifest_path = observations_dir / "texture_manifest.json"
    frame_selection_path = observations_dir / "frame_selection.json"
    geometry_manifest_path = observations_dir / "manifest.json"
    raw_frames_path = project_dir / "raw" / "frames.json"
    report_path = observations_dir / "texture_manifest_deduplication.json"
    if not texture_manifest_path.exists():
        raise FileNotFoundError(f"{texture_manifest_path} is required")
    if not raw_frames_path.exists():
        raise FileNotFoundError(f"{raw_frames_path} is required")
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")

    texture_manifest = _read_json(texture_manifest_path)
    raw_frames = _raw_frame_records_by_id(raw_frames_path)
    texture_frames = [_as_frame(frame) for frame in texture_manifest.get("frames", [])]
    if not texture_frames:
        raise ValueError("texture manifest contains no frames")
    geometry_selected_ids = {
        str(frame_id)
        for frame_id in texture_manifest.get("geometry_selected_frame_ids", [])
    }
    if not geometry_selected_ids and geometry_manifest_path.exists():
        geometry_manifest = _read_json(geometry_manifest_path)
        frame_selection = geometry_manifest.get("frame_selection", {})
        if isinstance(frame_selection, dict):
            geometry_selected_ids = {
                str(frame_id)
                for frame_id in frame_selection.get("selected_frame_ids", [])
            }

    duplicate_groups: dict[str, list[dict[str, Any]]] = {}
    for frame in texture_frames:
        frame_id = str(frame["frame_id"])
        digest = _raw_frame_sha256(raw_frames, frame_id)
        duplicate_groups.setdefault(digest, []).append(frame)

    kept_ids: set[str] = set()
    duplicate_reports: list[dict[str, Any]] = []
    for digest, group in sorted(duplicate_groups.items()):
        chosen = min(
            group,
            key=lambda frame: _texture_dedup_rank(frame, geometry_selected_ids),
        )
        chosen_id = str(chosen["frame_id"])
        kept_ids.add(chosen_id)
        removed_ids = sorted(
            str(frame["frame_id"])
            for frame in group
            if str(frame["frame_id"]) != chosen_id
        )
        if removed_ids:
            duplicate_reports.append(
                {
                    "sha256": digest,
                    "kept_frame_id": chosen_id,
                    "removed_frame_ids": removed_ids,
                    "geometry_frame_preserved": chosen_id in geometry_selected_ids,
                    "source_media_ids": sorted(
                        {
                            str(raw_frames[str(frame["frame_id"])].get("source_media_id", ""))
                            for frame in group
                        }
                    ),
                }
            )

    deduplicated_frames = [
        frame for frame in texture_frames if str(frame["frame_id"]) in kept_ids
    ]
    deduplicated_frame_ids = [str(frame["frame_id"]) for frame in deduplicated_frames]
    removed_duplicate_frame_ids = tuple(
        sorted(
            frame_id
            for report in duplicate_reports
            for frame_id in report["removed_frame_ids"]
        )
    )
    texture_manifest["texture_frame_count"] = len(deduplicated_frames)
    texture_manifest["texture_frame_ids"] = deduplicated_frame_ids
    texture_manifest["frames"] = deduplicated_frames
    texture_manifest["texture_media_counts"] = _media_counts(deduplicated_frames)
    texture_manifest["deduplication"] = {
        "applied": bool(removed_duplicate_frame_ids),
        "version": TEXTURE_MANIFEST_DEDUP_VERSION,
        "input_texture_frame_count": len(texture_frames),
        "deduplicated_texture_frame_count": len(deduplicated_frames),
        "removed_duplicate_frame_count": len(removed_duplicate_frame_ids),
        "duplicate_group_count": len(duplicate_reports),
        "geometry_selected_frame_ids_preserved": sorted(geometry_selected_ids),
        "report_path": str(report_path),
    }
    texture_manifest_path.write_text(
        json.dumps(texture_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _update_texture_manifest_selection_metadata(
        geometry_manifest_path,
        texture_frame_ids=deduplicated_frame_ids,
        texture_media_counts=texture_manifest["texture_media_counts"],
        deduplication=texture_manifest["deduplication"],
    )
    _update_texture_manifest_selection_metadata(
        frame_selection_path,
        texture_frame_ids=deduplicated_frame_ids,
        texture_media_counts=texture_manifest["texture_media_counts"],
        deduplication=texture_manifest["deduplication"],
    )

    report = {
        "schema_version": "1.0",
        "source": "opencat3d.raw_texture_manifest_deduplication",
        "version": TEXTURE_MANIFEST_DEDUP_VERSION,
        "project_id": texture_manifest.get("project_id", project_dir.name),
        "texture_manifest_path": str(texture_manifest_path),
        "input_texture_frame_count": len(texture_frames),
        "deduplicated_texture_frame_count": len(deduplicated_frames),
        "removed_duplicate_frame_count": len(removed_duplicate_frame_ids),
        "duplicate_group_count": len(duplicate_reports),
        "geometry_selected_frame_ids": sorted(geometry_selected_ids),
        "texture_frame_ids": deduplicated_frame_ids,
        "removed_duplicate_frame_ids": list(removed_duplicate_frame_ids),
        "duplicate_groups": duplicate_reports,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RawTextureManifestDedupResult(
        project_dir=project_dir,
        manifest_path=texture_manifest_path,
        report_path=report_path,
        input_texture_frame_count=len(texture_frames),
        deduplicated_texture_frame_count=len(deduplicated_frames),
        removed_duplicate_frame_ids=removed_duplicate_frame_ids,
        duplicate_group_count=len(duplicate_reports),
    )


def map_raw_frame_selection_by_image_hash(
    reference_project_dir: Path,
    target_project_dir: Path,
    *,
    selected_frame_ids: Sequence[str] | None = None,
    texture_frame_ids: Sequence[str] | None = None,
) -> RawFrameSelectionHashMapResult:
    """Map reference frame IDs into another project using image content hashes.

    Frame IDs are local to one ingest run. When multiple uploads are combined, the
    same source frame can receive a different ID, so raw ID reuse can silently
    select the wrong geometry frames. This keeps overrides tied to observed
    pixels instead of ingest order.
    """

    reference_project_dir = Path(reference_project_dir)
    target_project_dir = Path(target_project_dir)
    reference_selection = _read_json(
        reference_project_dir / "observations" / "raw" / "frame_selection.json"
    )
    if selected_frame_ids is None:
        selected_frame_ids = [
            str(item) for item in reference_selection.get("selected_frame_ids", [])
        ]
    if texture_frame_ids is None:
        texture_frame_ids = [
            str(item) for item in reference_selection.get("texture_frame_ids", [])
        ]

    reference_hashes = _observation_image_hashes(reference_project_dir)
    target_by_hash = _target_frame_ids_by_image_hash(target_project_dir)
    return RawFrameSelectionHashMapResult(
        reference_project_dir=reference_project_dir,
        target_project_dir=target_project_dir,
        selected_frame_ids=_map_frame_ids_by_hash(
            selected_frame_ids,
            reference_hashes=reference_hashes,
            target_by_hash=target_by_hash,
            label="selected_frame_ids",
        ),
        texture_frame_ids=_map_frame_ids_by_hash(
            texture_frame_ids,
            reference_hashes=reference_hashes,
            target_by_hash=target_by_hash,
            label="texture_frame_ids",
        ),
    )


def select_raw_observation_frames(
    project_dir: Path,
    *,
    min_frames: int = 2,
    max_selected_frames: int | None = 5,
    selected_frame_ids: Sequence[str] | None = None,
    texture_frame_ids: Sequence[str] | None = None,
    selection_label: str | None = None,
    force: bool = False,
) -> RawFrameSelectionResult:
    """Select template-compatible raw frames and rewrite the raw observation manifest."""

    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    if max_selected_frames is not None and max_selected_frames < min_frames:
        raise ValueError("max_selected_frames must be None or >= min_frames")

    observations_dir = project_dir / "observations" / "raw"
    manifest_path = observations_dir / "manifest.json"
    report_path = observations_dir / "frame_selection.json"
    texture_manifest_path = observations_dir / "texture_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} is required")
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")
    if texture_manifest_path.exists() and not force:
        raise FileExistsError(
            f"{texture_manifest_path} already exists; pass force=True to overwrite"
        )

    manifest = _read_json(manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("raw observation manifest contains no frames")
    frame_objects = [_as_frame(frame) for frame in frames]
    override_applied = selected_frame_ids is not None or texture_frame_ids is not None
    limit = len(frame_objects) if max_selected_frames is None else min(
        len(frame_objects),
        int(max_selected_frames),
    )
    limit = max(limit, min_frames)

    decisions = [_score_frame(frame, index=index) for index, frame in enumerate(frame_objects)]
    eligible = [decision for decision in decisions if not decision["rejected_by_quality"]]
    ranked_pool = eligible if len(eligible) >= min_frames else decisions
    ranked = sorted(
        ranked_pool,
        key=lambda item: (
            float(item["geometry_fit_score"]),
            int(item["source_index"]),
            str(item["frame_id"]),
        ),
    )
    if selected_frame_ids is None:
        selected_ids, media_strategy = _select_ranked_frame_ids(
            ranked,
            limit=limit,
            min_frames=min_frames,
        )
    else:
        selected_ids = _validate_override_ids(
            selected_frame_ids,
            frame_objects,
            min_frames=min_frames,
            max_frames=limit,
            label="selected_frame_ids",
        )
        media_strategy = f"override:{selection_label or 'manual'}"
    if len(selected_ids) < min_frames:
        raise ValueError("raw frame selection could not retain enough frames")

    selected_frames = [
        frame for frame in frame_objects if str(frame.get("frame_id")) in selected_ids
    ]
    selected_frame_ids = tuple(str(frame["frame_id"]) for frame in selected_frames)
    selected_metrics = _selection_metrics(selected_frames)
    selected_id_set = set(selected_frame_ids)
    input_media_counts = _media_counts(decisions)
    selected_media_counts = _media_counts(
        [decision for decision in decisions if str(decision["frame_id"]) in selected_id_set]
    )
    selection_detail_metrics = _selection_detail_metrics(
        decisions,
        selected_id_set=selected_id_set,
        texture_id_set=None,
    )
    if texture_frame_ids is None:
        texture_frames = _texture_frames(frame_objects, decisions, selected_id_set)
        texture_policy = "selected_frames_plus_quality_media"
    else:
        texture_ids = _validate_override_ids(
            texture_frame_ids,
            frame_objects,
            min_frames=1,
            max_frames=len(frame_objects),
            label="texture_frame_ids",
        )
        texture_ids.update(selected_id_set)
        texture_frames = [
            frame for frame in frame_objects if str(frame.get("frame_id")) in texture_ids
        ]
        texture_policy = "override_plus_selected_geometry"
    texture_frame_ids = tuple(str(frame["frame_id"]) for frame in texture_frames)
    texture_id_set = set(texture_frame_ids)
    texture_media_counts = _media_counts(
        [decision for decision in decisions if str(decision["frame_id"]) in texture_id_set]
    )
    selection_detail_metrics = _selection_detail_metrics(
        decisions,
        selected_id_set=selected_id_set,
        texture_id_set=texture_id_set,
    )
    report_decisions = []
    for decision in decisions:
        selected = str(decision["frame_id"]) in selected_id_set
        texture_enabled = str(decision["frame_id"]) in texture_id_set
        report_decisions.append(
            {**decision, "selected": selected, "texture_enabled": texture_enabled}
        )

    manifest["frame_count"] = len(selected_frames)
    manifest["input_frame_count"] = len(frame_objects)
    manifest["frame_selection"] = {
        "applied": len(selected_frames) != len(frame_objects),
        "version": FRAME_SELECTION_VERSION,
        "input_frame_count": len(frame_objects),
        "selected_frame_count": len(selected_frames),
        "rejected_frame_count": len(frame_objects) - len(selected_frames),
        "selected_frame_ids": list(selected_frame_ids),
        "media_strategy": media_strategy,
        "input_media_counts": input_media_counts,
        "selected_media_counts": selected_media_counts,
        "texture_frame_count": len(texture_frames),
        "texture_frame_ids": list(texture_frame_ids),
        "texture_media_counts": texture_media_counts,
        "texture_manifest_path": str(texture_manifest_path),
        "selection_label": selection_label,
        "report_path": str(report_path),
    }
    manifest["mean_mask_area_ratio"] = selected_metrics["mean_mask_area_ratio"]
    manifest["mean_bbox_area_ratio"] = selected_metrics["mean_bbox_area_ratio"]
    manifest["full_frame_bbox_fraction"] = selected_metrics["full_frame_bbox_fraction"]
    manifest["frames"] = selected_frames
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": "1.0",
        "project_id": manifest.get("project_id", project_dir.name),
        "source": "opencat3d.raw_frame_selection",
        "version": FRAME_SELECTION_VERSION,
        "min_frames": min_frames,
        "max_selected_frames": max_selected_frames,
        "input_frame_count": len(frame_objects),
        "selected_frame_count": len(selected_frames),
        "rejected_frame_count": len(frame_objects) - len(selected_frames),
        "texture_frame_count": len(texture_frames),
        "selected_frame_ids": list(selected_frame_ids),
        "texture_frame_ids": list(texture_frame_ids),
        "media_strategy": media_strategy,
        "input_media_counts": input_media_counts,
        "selected_media_counts": selected_media_counts,
        "texture_media_counts": texture_media_counts,
        "body_priority_metrics": selection_detail_metrics,
        "texture_policy": texture_policy,
        "texture_manifest_path": str(texture_manifest_path),
        "selection_label": selection_label,
        "override_applied": override_applied,
        "fallback_to_rejected_frames": len(eligible) < min_frames,
        "metrics": {**selected_metrics, **selection_detail_metrics},
        "frames": report_decisions,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    texture_manifest = {
        "schema_version": "1.0",
        "project_id": manifest.get("project_id", project_dir.name),
        "source": "opencat3d.raw_texture_frame_selection",
        "version": FRAME_SELECTION_VERSION,
        "texture_policy": texture_policy,
        "input_frame_count": len(frame_objects),
        "geometry_selected_frame_count": len(selected_frames),
        "texture_frame_count": len(texture_frames),
        "geometry_selected_frame_ids": list(selected_frame_ids),
        "texture_frame_ids": list(texture_frame_ids),
        "input_media_counts": input_media_counts,
        "geometry_selected_media_counts": selected_media_counts,
        "texture_media_counts": texture_media_counts,
        "geometry_manifest_path": str(manifest_path),
        "frame_selection_report_path": str(report_path),
        "selection_label": selection_label,
        "override_applied": override_applied,
        "frames": texture_frames,
    }
    texture_manifest_path.write_text(
        json.dumps(texture_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return RawFrameSelectionResult(
        project_dir=project_dir,
        report_path=report_path,
        manifest_path=manifest_path,
        texture_manifest_path=texture_manifest_path,
        input_frame_count=len(frame_objects),
        selected_frame_count=len(selected_frames),
        rejected_frame_count=len(frame_objects) - len(selected_frames),
        texture_frame_count=len(texture_frames),
        selected_frame_ids=selected_frame_ids,
        mean_mask_area_ratio=float(selected_metrics["mean_mask_area_ratio"]),
        mean_bbox_area_ratio=float(selected_metrics["mean_bbox_area_ratio"]),
        full_frame_bbox_fraction=float(selected_metrics["full_frame_bbox_fraction"]),
        media_strategy=media_strategy,
        input_media_counts=input_media_counts,
        selected_media_counts=selected_media_counts,
        texture_media_counts=texture_media_counts,
        detail_metrics=selection_detail_metrics,
    )


def _score_frame(frame: dict[str, Any], *, index: int) -> dict[str, Any]:
    frame_id = str(frame.get("frame_id", f"frame_{index:04d}"))
    media_type = _frame_media_type(frame)
    quality = frame.get("quality", {})
    if not isinstance(quality, dict):
        quality = {}
    image_width = max(float(frame.get("image_width", 1.0)), 1.0)
    image_height = max(float(frame.get("image_height", 1.0)), 1.0)
    bbox = frame.get("bbox_xyxy", [0.0, 0.0, image_width, image_height])
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{frame_id} bbox_xyxy must contain four values")

    x0, y0, x1, y1 = (float(value) for value in bbox)
    bbox_width_fraction = max(0.0, x1 - x0) / image_width
    bbox_height_fraction = max(0.0, y1 - y0) / image_height
    bbox_area_ratio = float(
        quality.get("bbox_area_ratio", bbox_width_fraction * bbox_height_fraction)
    )
    mask_area_ratio = float(quality.get("mask_area_ratio", 0.0))
    touches_border = float(quality.get("touches_image_border", 0.0))
    full_frame_bbox = float(quality.get("full_frame_bbox", 0.0))
    center_x = (x0 + x1) / (2.0 * image_width)
    center_y = (y0 + y1) / (2.0 * image_height)
    aspect = bbox_width_fraction / max(bbox_height_fraction, 1e-6)

    reject_reasons: list[str] = []
    if mask_area_ratio < 0.05:
        reject_reasons.append("mask_too_small")
    if bbox_area_ratio < 0.08:
        reject_reasons.append("bbox_too_small")
    if bbox_area_ratio > 0.58:
        reject_reasons.append("bbox_too_large")
    if full_frame_bbox >= 0.5:
        reject_reasons.append("full_frame_bbox")
    if (
        touches_border >= 0.5
        and bbox_height_fraction >= 0.95
        and bbox_width_fraction <= 0.48
        and aspect <= 0.52
    ):
        reject_reasons.append("border_partial_body")
    if mask_area_ratio < 0.06 and bbox_area_ratio < 0.20:
        reject_reasons.append("sparse_geometry_mask")

    score = 0.0
    score += abs(mask_area_ratio - 0.18) * 1.6
    score += abs(bbox_area_ratio - 0.34) * 0.9
    score += math.hypot(center_x - 0.5, center_y - 0.5) * 0.35
    score += max(0.0, 0.95 - bbox_width_fraction) * 0.05
    score += max(0.0, abs(aspect - 1.8) - 1.2) * 0.08
    score += touches_border * 0.04
    body_fit_score = _body_fit_score(
        mask_area_ratio=mask_area_ratio,
        bbox_area_ratio=bbox_area_ratio,
        bbox_width_fraction=bbox_width_fraction,
        bbox_height_fraction=bbox_height_fraction,
        aspect=aspect,
        center_x=center_x,
        center_y=center_y,
        touches_border=touches_border,
    )
    face_texture_score = _face_texture_detail_score(
        mask_area_ratio=mask_area_ratio,
        bbox_area_ratio=bbox_area_ratio,
        bbox_width_fraction=bbox_width_fraction,
        bbox_height_fraction=bbox_height_fraction,
        aspect=aspect,
        center_x=center_x,
        center_y=center_y,
        touches_border=touches_border,
    )
    geometry_fit_score = score - body_fit_score * 0.22 + face_texture_score * 0.34
    geometry_fit_candidate = body_fit_score >= 0.48 and not reject_reasons
    texture_detail_candidate = face_texture_score >= 0.52 and not any(
        reason in {"mask_too_small", "bbox_too_small", "full_frame_bbox"}
        for reason in reject_reasons
    )

    return {
        "frame_id": frame_id,
        "media_type": media_type,
        "source_index": index,
        "score": round(float(score), 9),
        "geometry_fit_score": round(float(geometry_fit_score), 9),
        "body_fit_score": round(float(body_fit_score), 9),
        "face_texture_detail_score": round(float(face_texture_score), 9),
        "geometry_fit_candidate": bool(geometry_fit_candidate),
        "texture_detail_candidate": bool(texture_detail_candidate),
        "rejected_by_quality": bool(reject_reasons),
        "reject_reasons": reject_reasons,
        "mask_area_ratio": mask_area_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_width_fraction": bbox_width_fraction,
        "bbox_height_fraction": bbox_height_fraction,
        "bbox_center": [center_x, center_y],
        "bbox_aspect": aspect,
        "touches_image_border": touches_border,
        "full_frame_bbox": full_frame_bbox,
    }


def _select_ranked_frame_ids(
    ranked: list[dict[str, Any]],
    *,
    limit: int,
    min_frames: int,
) -> tuple[set[str], str]:
    media_types = {str(item.get("media_type", "photo")) for item in ranked}
    if {"photo", "video"}.issubset(media_types):
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        photos = [item for item in ranked if item.get("media_type") == "photo"]
        videos = [item for item in ranked if item.get("media_type") == "video"]
        photo_quota = 1 if limit < 4 else max(2, min(4, limit // 3))
        photo_quota = min(photo_quota, len(photos), limit)
        video_quota = min(len(videos), max(1, limit - photo_quota))
        if photo_quota + video_quota > limit:
            photo_quota = max(0, limit - video_quota)

        for group, quota in ((photos, photo_quota), (videos, video_quota)):
            for item in group[:quota]:
                frame_id = str(item["frame_id"])
                if frame_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(frame_id)

        for item in ranked:
            if len(selected) >= limit:
                break
            frame_id = str(item["frame_id"])
            if frame_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(frame_id)

        if len(selected_ids) >= min_frames:
            return selected_ids, "balanced_photo_video"

    return {str(item["frame_id"]) for item in ranked[:limit]}, "score_only"


def _validate_override_ids(
    frame_ids: Sequence[str],
    frames: list[dict[str, Any]],
    *,
    min_frames: int,
    max_frames: int,
    label: str,
) -> set[str]:
    if not frame_ids:
        raise ValueError(f"{label} override must contain at least {min_frames} frame ids")
    known_ids = {str(frame.get("frame_id")) for frame in frames}
    selected_ids: set[str] = set()
    unknown_ids: list[str] = []
    for frame_id in frame_ids:
        frame_id_text = str(frame_id)
        if frame_id_text not in known_ids:
            unknown_ids.append(frame_id_text)
            continue
        selected_ids.add(frame_id_text)
    if unknown_ids:
        raise ValueError(f"{label} override contains unknown frame ids: {sorted(unknown_ids)}")
    if len(selected_ids) < min_frames:
        raise ValueError(f"{label} override must retain at least {min_frames} unique frames")
    if len(selected_ids) > max_frames:
        raise ValueError(f"{label} override must contain no more than {max_frames} frames")
    return selected_ids


def _frame_media_type(frame: dict[str, Any]) -> str:
    if frame.get("media_type") == "video" or frame.get("source_type") == "video_frame":
        return "video"
    return "photo"


def _media_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"photo": 0, "video": 0}
    for decision in decisions:
        media_type = str(decision.get("media_type", "photo"))
        if media_type not in counts:
            counts[media_type] = 0
        counts[media_type] += 1
    return counts


def _texture_frames(
    frames: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    texture_ids = set(selected_ids)
    for decision in decisions:
        rejected = bool(decision.get("rejected_by_quality", False))
        detail_safe = bool(decision.get("texture_detail_candidate", False))
        if rejected and not detail_safe:
            continue
        texture_ids.add(str(decision["frame_id"]))
    return [frame for frame in frames if str(frame.get("frame_id")) in texture_ids]


def _body_fit_score(
    *,
    mask_area_ratio: float,
    bbox_area_ratio: float,
    bbox_width_fraction: float,
    bbox_height_fraction: float,
    aspect: float,
    center_x: float,
    center_y: float,
    touches_border: float,
) -> float:
    area_score = _triangular_score(bbox_area_ratio, lower=0.14, peak=0.34, upper=0.56)
    mask_score = _triangular_score(mask_area_ratio, lower=0.05, peak=0.18, upper=0.34)
    width_score = _triangular_score(bbox_width_fraction, lower=0.28, peak=0.68, upper=0.96)
    height_score = _triangular_score(bbox_height_fraction, lower=0.22, peak=0.56, upper=1.00)
    aspect_score = _triangular_score(aspect, lower=0.42, peak=1.25, upper=3.20)
    center_score = _clamp01(1.0 - math.hypot(center_x - 0.5, center_y - 0.5) / 0.70)
    border_score = 1.0 - min(0.55, _clamp01(touches_border) * 0.55)
    return _clamp01(
        (
            0.24 * area_score
            + 0.20 * mask_score
            + 0.20 * width_score
            + 0.14 * height_score
            + 0.14 * aspect_score
            + 0.08 * center_score
        )
        * border_score
    )


def _face_texture_detail_score(
    *,
    mask_area_ratio: float,
    bbox_area_ratio: float,
    bbox_width_fraction: float,
    bbox_height_fraction: float,
    aspect: float,
    center_x: float,
    center_y: float,
    touches_border: float,
) -> float:
    closeup_score = _triangular_score(bbox_area_ratio, lower=0.28, peak=0.62, upper=0.86)
    compact_score = _triangular_score(aspect, lower=0.55, peak=1.00, upper=1.70)
    size_score = _clamp01((bbox_width_fraction - 0.34) / 0.34) * _clamp01(
        (bbox_height_fraction - 0.34) / 0.34
    )
    mask_score = _triangular_score(mask_area_ratio, lower=0.05, peak=0.25, upper=0.48)
    center_score = _clamp01(1.0 - math.hypot(center_x - 0.5, center_y - 0.5) / 0.56)
    border_score = 1.0 - min(0.35, _clamp01(touches_border) * 0.35)
    return _clamp01(
        (
            0.30 * closeup_score
            + 0.22 * compact_score
            + 0.20 * size_score
            + 0.16 * mask_score
            + 0.12 * center_score
        )
        * border_score
    )


def _triangular_score(value: float, *, lower: float, peak: float, upper: float) -> float:
    if value <= lower or value >= upper:
        return 0.0
    if value == peak:
        return 1.0
    if value < peak:
        return _clamp01((value - lower) / max(peak - lower, 1e-9))
    return _clamp01((upper - value) / max(upper - peak, 1e-9))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _selection_detail_metrics(
    decisions: list[dict[str, Any]],
    *,
    selected_id_set: set[str],
    texture_id_set: set[str] | None,
) -> dict[str, float]:
    texture_ids = texture_id_set or set()
    selected = [
        decision for decision in decisions if str(decision["frame_id"]) in selected_id_set
    ]
    texture_only = [
        decision
        for decision in decisions
        if str(decision["frame_id"]) in texture_ids
        and str(decision["frame_id"]) not in selected_id_set
    ]
    return {
        "geometry_body_priority_candidate_count": float(
            sum(1 for decision in decisions if bool(decision.get("geometry_fit_candidate")))
        ),
        "geometry_selected_body_priority_frame_count": float(
            sum(1 for decision in selected if bool(decision.get("geometry_fit_candidate")))
        ),
        "texture_detail_candidate_count": float(
            sum(1 for decision in decisions if bool(decision.get("texture_detail_candidate")))
        ),
        "texture_only_detail_frame_count": float(
            sum(1 for decision in texture_only if bool(decision.get("texture_detail_candidate")))
        ),
        "geometry_selected_mean_body_fit_score": float(
            np.mean([float(decision.get("body_fit_score", 0.0)) for decision in selected])
            if selected
            else 0.0
        ),
        "geometry_selected_mean_face_texture_detail_score": float(
            np.mean(
                [
                    float(decision.get("face_texture_detail_score", 0.0))
                    for decision in selected
                ]
            )
            if selected
            else 0.0
        ),
    }


def _selection_metrics(frames: list[dict[str, Any]]) -> dict[str, float]:
    mask_area_ratios = []
    bbox_area_ratios = []
    full_frame_bboxes = []
    for frame in frames:
        quality = frame.get("quality", {})
        if not isinstance(quality, dict):
            quality = {}
        mask_area_ratios.append(float(quality.get("mask_area_ratio", 0.0)))
        bbox_area_ratios.append(float(quality.get("bbox_area_ratio", 0.0)))
        full_frame_bboxes.append(float(quality.get("full_frame_bbox", 0.0)))
    return {
        "mean_mask_area_ratio": float(np.mean(mask_area_ratios)),
        "mean_bbox_area_ratio": float(np.mean(bbox_area_ratios)),
        "full_frame_bbox_fraction": float(np.mean(full_frame_bboxes)),
    }


def _as_frame(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("raw observation manifest frame must be an object")
    return dict(value)


def _raw_frame_records_by_id(raw_frames_path: Path) -> dict[str, dict[str, Any]]:
    report = _read_json(raw_frames_path)
    frames = report.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("raw frame manifest contains no frames")
    by_id: dict[str, dict[str, Any]] = {}
    for frame in frames:
        frame_obj = _as_frame(frame)
        frame_id = str(frame_obj.get("frame_id", ""))
        if not frame_id:
            raise ValueError("raw frame is missing frame_id")
        by_id[frame_id] = frame_obj
    return by_id


def _raw_frame_sha256(raw_frames: dict[str, dict[str, Any]], frame_id: str) -> str:
    frame = raw_frames.get(frame_id)
    if frame is None:
        raise ValueError(f"texture frame {frame_id} is missing from raw frame manifest")
    digest = str(frame.get("sha256", "")).strip()
    if not digest:
        raise ValueError(f"raw frame {frame_id} is missing sha256")
    return digest


def _texture_dedup_rank(
    frame: dict[str, Any],
    geometry_selected_ids: set[str],
) -> tuple[int, int, str]:
    frame_id = str(frame["frame_id"])
    source_index = int(frame.get("source_index", 0))
    return (
        0 if frame_id in geometry_selected_ids else 1,
        source_index,
        frame_id,
    )


def _update_texture_manifest_selection_metadata(
    path: Path,
    *,
    texture_frame_ids: list[str],
    texture_media_counts: dict[str, int],
    deduplication: dict[str, Any],
) -> None:
    if not path.exists():
        return
    report = _read_json(path)
    report["texture_frame_count"] = len(texture_frame_ids)
    report["texture_frame_ids"] = list(texture_frame_ids)
    report["texture_media_counts"] = dict(texture_media_counts)
    report["texture_deduplication"] = dict(deduplication)
    frame_selection = report.get("frame_selection")
    if isinstance(frame_selection, dict):
        frame_selection["texture_frame_count"] = len(texture_frame_ids)
        frame_selection["texture_frame_ids"] = list(texture_frame_ids)
        frame_selection["texture_media_counts"] = dict(texture_media_counts)
        frame_selection["texture_deduplication"] = dict(deduplication)
    frames = report.get("frames")
    if isinstance(frames, list):
        texture_id_set = set(texture_frame_ids)
        updated_frames = []
        for frame in frames:
            if not isinstance(frame, dict):
                updated_frames.append(frame)
                continue
            frame_id = str(frame.get("frame_id", ""))
            updated_frames.append(
                {
                    **frame,
                    "texture_enabled": frame_id in texture_id_set,
                }
            )
        report["frames"] = updated_frames
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _map_frame_ids_by_hash(
    frame_ids: Sequence[str],
    *,
    reference_hashes: dict[str, str],
    target_by_hash: dict[str, str],
    label: str,
) -> tuple[str, ...]:
    mapped: list[str] = []
    missing_reference_ids: list[str] = []
    missing_target_ids: list[str] = []
    for frame_id in frame_ids:
        frame_id_text = str(frame_id)
        digest = reference_hashes.get(frame_id_text)
        if digest is None:
            missing_reference_ids.append(frame_id_text)
            continue
        target_id = target_by_hash.get(digest)
        if target_id is None:
            missing_target_ids.append(frame_id_text)
            continue
        mapped.append(target_id)
    if missing_reference_ids:
        raise ValueError(
            f"{label} contains unknown reference frame ids: {sorted(missing_reference_ids)}"
        )
    if missing_target_ids:
        raise ValueError(
            f"{label} could not map reference frames into target project: "
            f"{sorted(missing_target_ids)}"
        )
    return tuple(mapped)


def _target_frame_ids_by_image_hash(project_dir: Path) -> dict[str, str]:
    by_hash: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for frame_id, digest in _observation_image_hashes(project_dir).items():
        if digest in by_hash:
            duplicates.setdefault(digest, [by_hash[digest]]).append(frame_id)
            continue
        by_hash[digest] = frame_id
    if duplicates:
        duplicate_ids = {
            digest: sorted(frame_ids) for digest, frame_ids in sorted(duplicates.items())
        }
        raise ValueError(f"target project contains duplicate frame image hashes: {duplicate_ids}")
    return by_hash


def _observation_image_hashes(project_dir: Path) -> dict[str, str]:
    observations_dir = Path(project_dir) / "observations" / "raw"
    hashes: dict[str, str] = {}
    for observation_path in sorted(observations_dir.glob("frame_*/observation.json")):
        observation = _read_json(observation_path)
        frame_id = str(observation.get("frame_id") or observation_path.parent.name)
        image_path_raw = observation.get("image_srgb_path")
        if not image_path_raw:
            raise ValueError(f"{observation_path} is missing image_srgb_path")
        image_path = Path(str(image_path_raw))
        if not image_path.is_file():
            raise FileNotFoundError(f"{image_path} is required for frame hash mapping")
        hashes[frame_id] = _sha256_file(image_path)
    if not hashes:
        raise ValueError(f"{observations_dir} contains no frame observations")
    return hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
