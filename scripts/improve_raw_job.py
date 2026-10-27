"""Deterministic raw-job improvement loop.

This script does not call external APIs and does not train models. It sweeps local
raw segmentation backends, selects reliable frames, runs the best full pipelines,
and writes a reproducible JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opencat3d.evaluation.supervisor import (  # noqa: E402
    write_candidate_comparison_report,
    write_guarded_candidate_comparison_report,
)
from opencat3d.ingest.images import SUPPORTED_IMAGE_EXTENSIONS, ingest_image_sources  # noqa: E402
from opencat3d.ingest.video import (  # noqa: E402
    SUPPORTED_VIDEO_EXTENSIONS,
    extract_video_frames,
)
from opencat3d.pipeline import (  # noqa: E402
    run_baseline_geometry_texture_bake,
    run_raw_pipeline,
    write_texture_source_admission_audit,
)
from opencat3d.preprocess.frame_selection import select_raw_observation_frames  # noqa: E402
from opencat3d.preprocess.keypoints import bootstrap_raw_keypoints_and_pose  # noqa: E402
from opencat3d.preprocess.segmentation import (  # noqa: E402
    RawSegmentationBackend,
    segment_raw_image_observations,
)

SOLID_THRESHOLDS: dict[str, float] = {
    "max_mean_bbox_area_ratio": 0.65,
    "max_full_frame_bbox_fraction": 0.20,
    "min_overlay_mean_silhouette_iou": 0.18,
    "max_silhouette_residual_mean_union_error": 0.85,
    "min_unknown_texture_luma_fraction_of_foreground": 0.65,
}
BYTES_PER_GIB = 1024**3
DISK_PREFLIGHT_VERSION = "improve_raw_job_disk_preflight_v1"
TEXTURE_SOURCE_ABLATION_BLOCKLIST_VERSION = "texture_source_ablation_blocklist_v1"
SEGMENTATION_CANDIDATE_CACHE_VERSION = "improve_raw_job_segmentation_candidate_cache_v1"


def _parse_probe_ablation_overrides(
    raw_items: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    overrides: dict[str, set[str]] = {}
    for raw_item in raw_items:
        item = str(raw_item).strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(
                "--texture-source-probe-ablation-override must be frame_id:sha256"
            )
        frame_id, digest = item.split(":", 1)
        frame_id = frame_id.strip()
        digest = digest.strip().lower()
        if not frame_id or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise SystemExit(
                "--texture-source-probe-ablation-override must be frame_id:sha256"
            )
        overrides.setdefault(frame_id, set()).add(digest)
    return {
        frame_id: tuple(sorted(hashes))
        for frame_id, hashes in sorted(overrides.items())
    }


@dataclass(frozen=True)
class FrameSelection:
    frame_id: str
    source_path: Path
    media_type: str
    source_type: str
    source_media_id: str | None
    source_frame_index: int | None
    geometry_reject_reasons: tuple[str, ...]
    geometry_body_fit_score: float
    score: float
    mask_area_ratio: float
    bbox_area_ratio: float
    full_frame_bbox: bool
    raw_view_label: str = "unknown_view"
    raw_view_group: str = "unknown"
    raw_pose_label: str = "unknown_pose"
    raw_frontal_face_confidence: float = 0.0
    raw_view_weight: float = 0.0


@dataclass(frozen=True)
class SegmentationCandidate:
    candidate_id: str
    backend: RawSegmentationBackend
    color_delta_threshold: float
    project_dir: Path
    selected_frames: tuple[FrameSelection, ...]
    frame_count: int
    selected_frame_count: int
    mean_mask_area_ratio: float
    mean_bbox_area_ratio: float
    full_frame_bbox_fraction: float
    score: float
    texture_frames: tuple[FrameSelection, ...] = ()


@dataclass(frozen=True)
class GeometryVariant:
    variant_id: str
    geometry_frames: tuple[FrameSelection, ...]
    require_min_frames: bool = True


@dataclass(frozen=True)
class RawViewPoseQuality:
    view_label: str
    view_group: str
    pose_label: str
    frontal_face_confidence: float
    view_weight: float


@dataclass(frozen=True)
class SegmentationCacheLoadResult:
    candidates: list[SegmentationCandidate]
    summary: dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-dir", type=Path, default=None)
    parser.add_argument(
        "--extra-media",
        type=Path,
        action="append",
        default=[],
        help="Additional image/video files or folders to merge with --upload-dir.",
    )
    parser.add_argument("--uploads-root", type=Path, default=ROOT / "web_runs" / "uploads")
    parser.add_argument("--loop-dir", type=Path, default=ROOT / "web_runs" / "loop")
    parser.add_argument("--projects-dir", type=Path, default=ROOT / "web_runs" / "projects")
    parser.add_argument(
        "--project-id-prefix",
        type=str,
        default=None,
        help=(
            "Prefix generated project directories. Defaults to the loop directory "
            "name so repeated automation runs do not overwrite prior evidence."
        ),
    )
    parser.add_argument("--camera-root-steps", type=int, default=30)
    parser.add_argument("--min-selected-frames", type=int, default=4)
    parser.add_argument("--max-selected-frames", type=int, default=8)
    parser.add_argument("--max-full-pipelines", type=int, default=2)
    parser.add_argument("--max-geometry-variants", type=int, default=3)
    parser.add_argument(
        "--geometry-frame-sweep",
        action="store_true",
        help=(
            "For each segmentation candidate, evaluate compact-to-rich geometry "
            "frame counts before broad variant families. This is useful when "
            "additional frames may dilute alignment."
        ),
    )
    parser.add_argument("--max-video-frames", type=int, default=24)
    parser.add_argument("--baseline-project-dir", type=Path, default=None)
    parser.add_argument(
        "--texture-only-source-project-dir",
        type=Path,
        default=None,
        help=(
            "Skip full geometry candidates and run baseline-geometry texture "
            "probes from this existing project only. Requires --baseline-project-dir."
        ),
    )
    parser.add_argument(
        "--texture-source-blocklist",
        type=Path,
        default=None,
        help=(
            "Stable JSON blocklist for supplemental texture sources that regressed "
            "strict face-detail metrics."
        ),
    )
    parser.add_argument(
        "--max-texture-source-probes",
        type=int,
        default=3,
        help=(
            "When the baseline-geometry texture bake adds no safe frames, try up "
            "to this many next-ranked non-duplicate source frames one at a time."
        ),
    )
    parser.add_argument(
        "--texture-source-probe-frame-id",
        type=str,
        action="append",
        default=[],
        help=(
            "Run texture-only source mode as explicit single-frame probes for the "
            "given source frame IDs, skipping the broad normal texture bake."
        ),
    )
    parser.add_argument(
        "--texture-source-probe-ablation-override",
        type=str,
        action="append",
        default=[],
        help=(
            "Allow one explicit texture-source probe to bypass a prior face-source "
            "ablation block only when the value is frame_id:sha256 and the probed "
            "frame/hash both match."
        ),
    )
    parser.add_argument(
        "--max-texture-source-fallbacks",
        type=int,
        default=3,
        help=(
            "Try this many ranked full-pipeline projects as supplemental texture "
            "sources before giving up on texture-only improvement."
        ),
    )
    parser.add_argument(
        "--skip-upload-texture-source",
        action="store_true",
        help=(
            "Do not build a fresh supplemental texture source from the current "
            "upload/extra media before falling back to older full-pipeline runs."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Stage, segment, and write media preflight reports without running "
            "reconstruction candidates."
        ),
    )
    parser.add_argument(
        "--admission-only",
        action="store_true",
        help=(
            "Stage and segment media, build the uploaded texture-source project, "
            "run texture-source admission against --baseline-project-dir, and stop "
            "before any geometry reconstruction or texture bake."
        ),
    )
    parser.add_argument(
        "--segmentation-cache-report",
        type=Path,
        default=None,
        help=(
            "Optional previous raw_improvement_loop_report.json to reuse "
            "segmentation candidates when staged media and frame limits match."
        ),
    )
    parser.add_argument(
        "--max-upload-texture-source-frames",
        type=int,
        default=18,
        help="Maximum uploaded frames to expose through the supplemental texture manifest.",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=5.0,
        help="Abort before staging when the loop filesystem has less free space.",
    )
    parser.add_argument(
        "--cleanup-intermediates",
        action="store_true",
        help=(
            "Remove repeatable seg_* and staged_upload_frames from --loop-dir "
            "before disk preflight."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.admission_only:
        raise SystemExit("--admission-only cannot be combined with --preflight-only")
    if args.admission_only and args.baseline_project_dir is None:
        raise SystemExit("--admission-only requires --baseline-project-dir")
    explicit_probe_frame_ids = tuple(
        str(item) for item in args.texture_source_probe_frame_id if str(item)
    )
    explicit_probe_ablation_overrides = _parse_probe_ablation_overrides(
        tuple(str(item) for item in args.texture_source_probe_ablation_override),
    )
    if explicit_probe_ablation_overrides and not explicit_probe_frame_ids:
        raise SystemExit(
            "--texture-source-probe-ablation-override requires "
            "--texture-source-probe-frame-id"
        )
    unknown_override_frames = sorted(
        set(explicit_probe_ablation_overrides) - set(explicit_probe_frame_ids)
    )
    if unknown_override_frames:
        raise SystemExit(
            "--texture-source-probe-ablation-override frame IDs must also be "
            "--texture-source-probe-frame-id values: "
            + ", ".join(unknown_override_frames)
        )

    upload_dir = args.upload_dir or _latest_upload_dir(args.uploads_root)
    loop_dir = args.loop_dir.resolve()
    projects_dir = args.projects_dir.resolve()
    project_id_prefix = _safe_candidate_token(args.project_id_prefix or loop_dir.name)
    texture_source_blocklist_path = (
        args.texture_source_blocklist.resolve()
        if args.texture_source_blocklist is not None
        else loop_dir.parent / "texture_source_ablation_blocklist.json"
    )
    loop_dir.mkdir(parents=True, exist_ok=True)
    projects_dir.mkdir(parents=True, exist_ok=True)
    disk_preflight = _write_disk_preflight_report(
        loop_dir,
        projects_dir=projects_dir,
        min_free_gib=args.min_free_gib,
        cleanup_intermediates=args.cleanup_intermediates,
        force=args.force,
    )
    if not bool(disk_preflight["passed"]):
        raise SystemExit(
            "insufficient disk space for raw improvement loop: "
            f"{disk_preflight['free_gib']:.2f} GiB available, "
            f"{disk_preflight['min_free_gib']:.2f} GiB required; "
            f"report={disk_preflight['report_path']}"
        )
    if args.texture_only_source_project_dir is not None:
        if args.baseline_project_dir is None:
            raise SystemExit(
                "--texture-only-source-project-dir requires --baseline-project-dir"
            )
        report_path, best_is_baseline, best_report = _run_texture_only_source_mode(
            texture_source_project_dir=args.texture_only_source_project_dir,
            baseline_project_dir=args.baseline_project_dir,
            loop_dir=loop_dir,
            projects_dir=projects_dir,
            project_id_prefix=project_id_prefix,
            texture_source_blocklist_path=texture_source_blocklist_path,
            max_texture_source_probes=max(0, args.max_texture_source_probes),
            explicit_texture_source_probe_frame_ids=explicit_probe_frame_ids,
            explicit_texture_source_ablation_overrides=(
                explicit_probe_ablation_overrides
            ),
            force=args.force,
        )
        print(f"report={report_path}")
        if best_is_baseline:
            print(
                "best "
                "baseline=true "
                f"project={args.baseline_project_dir.resolve()}"
            )
        elif best_report is not None:
            print(
                "best "
                f"solid={best_report['solid']} "
                f"success={best_report['success']} "
                f"project={best_report['project_dir']} "
                f"glb={best_report['artifacts'].get('glb', '')}"
            )
        return 0
    media_staging = _stage_upload_media_frames(
        upload_dir,
        stage_dir=loop_dir / "staged_upload_frames",
        extra_media_paths=tuple(args.extra_media),
        max_video_frames=max(2, args.max_video_frames),
        force=args.force,
    )
    raw_frame_paths = media_staging.frame_paths

    cache_load = _load_segmentation_candidate_cache(
        args.segmentation_cache_report,
        current_media_staging=media_staging,
        min_selected_frames=max(2, args.min_selected_frames),
        max_selected_frames=max(args.min_selected_frames, args.max_selected_frames),
    )
    segmentation_candidates = cache_load.candidates
    if not segmentation_candidates:
        segmentation_candidates = _sweep_segmenters(
            upload_dir=upload_dir.resolve(),
            raw_frame_paths=raw_frame_paths,
            loop_dir=loop_dir,
            min_selected_frames=max(2, args.min_selected_frames),
            max_selected_frames=max(args.min_selected_frames, args.max_selected_frames),
            backends=(
                ("white_subject_v1", "auto_subject_v1")
                if args.admission_only
                else None
            ),
            color_thresholds=((0.08,) if args.admission_only else None),
        )
        cache_load = SegmentationCacheLoadResult(
            candidates=segmentation_candidates,
            summary={
                **cache_load.summary,
                "used": False,
                "fallback_sweep_candidate_count": len(segmentation_candidates),
            },
        )
    if not segmentation_candidates:
        raise SystemExit("no segmentation candidates produced enough selected frames")
    media_preflight_report = _write_media_preflight_report(
        segmentation_candidates,
        loop_dir=loop_dir,
        min_geometry_frames=max(2, args.min_selected_frames),
        max_geometry_frames=max(args.min_selected_frames, args.max_selected_frames),
        max_texture_frames=max(2, args.max_upload_texture_source_frames),
    )
    media_preflight_promotion_gate = _write_media_preflight_promotion_gate(
        media_preflight_report,
        loop_dir=loop_dir,
    )
    if args.preflight_only:
        report_path = _write_preflight_only_loop_report(
            upload_dir=upload_dir,
            baseline_project_dir=args.baseline_project_dir,
            media_staging=media_staging,
            texture_source_blocklist_path=texture_source_blocklist_path,
            media_preflight_report=media_preflight_report,
            media_preflight_promotion_gate=media_preflight_promotion_gate,
            segmentation_candidates=segmentation_candidates,
            segmentation_cache=cache_load.summary,
            loop_dir=loop_dir,
        )
        print(f"report={report_path}")
        print(
            "preflight_only=true "
            f"ready_for_reconstruction={media_preflight_report['ready_for_reconstruction']} "
            f"ready_for_promotion={media_preflight_report['ready_for_promotion']} "
            f"report={media_preflight_report['report_path']}"
        )
        return 0
    if args.admission_only:
        report_path, best_is_baseline, best_report = _run_admission_only_mode(
            upload_dir=upload_dir,
            baseline_project_dir=args.baseline_project_dir,
            raw_frame_paths=raw_frame_paths,
            media_staging=media_staging,
            media_preflight_report=media_preflight_report,
            media_preflight_promotion_gate=media_preflight_promotion_gate,
            segmentation_candidates=segmentation_candidates,
            segmentation_cache=cache_load.summary,
            loop_dir=loop_dir,
            projects_dir=projects_dir,
            project_id_prefix=project_id_prefix,
            texture_source_blocklist_path=texture_source_blocklist_path,
            max_texture_frames=max(2, args.max_upload_texture_source_frames),
            force=args.force,
        )
        print(f"report={report_path}")
        if best_is_baseline:
            print(
                "best "
                "baseline=true "
                f"project={args.baseline_project_dir.resolve()}"
            )
        elif best_report is not None:
            print(
                "best "
                f"solid={best_report['solid']} "
                f"success={best_report['success']} "
                f"project={best_report['project_dir']} "
                f"admission={best_report.get('texture_source_admission_audit_passed')}"
            )
        return 0

    upload_texture_source_report: dict[str, Any] | None = None
    texture_only_source_project: dict[str, Any] | None = None
    pipeline_reports: list[dict[str, Any]] = []
    for rank, candidate in enumerate(
        segmentation_candidates[: max(1, args.max_full_pipelines)],
        start=1,
    ):
        reports = _run_full_candidate_variants(
            candidate,
            projects_dir=projects_dir,
            rank=rank,
            min_selected_frames=max(2, args.min_selected_frames),
            max_selected_frames=max(args.min_selected_frames, args.max_selected_frames),
            camera_root_steps=args.camera_root_steps,
            max_geometry_variants=max(1, args.max_geometry_variants),
            project_id_prefix=project_id_prefix,
            geometry_frame_sweep=args.geometry_frame_sweep,
            force=args.force,
        )
        pipeline_reports.extend(reports)
    if (
        args.baseline_project_dir is not None
        and pipeline_reports
        and not args.skip_upload_texture_source
        and texture_only_source_project is None
    ):
        upload_texture_source_report = _build_uploaded_texture_source_report(
            raw_frame_paths=raw_frame_paths,
            source_candidate=segmentation_candidates[0],
            loop_dir=loop_dir,
            min_geometry_frames=max(2, min(args.min_selected_frames, 4)),
            max_texture_frames=max(2, args.max_upload_texture_source_frames),
            force=args.force,
        )
    if _should_run_baseline_texture_fallback(
        baseline_project_dir=args.baseline_project_dir,
        pipeline_reports=pipeline_reports,
        texture_only_source_project=texture_only_source_project,
    ):
        texture_source_reports = _ordered_texture_source_reports(
            pipeline_reports,
            uploaded_texture_source_report=upload_texture_source_report,
            max_reports=max(1, args.max_texture_source_fallbacks),
        )
        if texture_source_reports:
            pipeline_reports.extend(
                _run_baseline_geometry_texture_fallback_candidates(
                    texture_source_reports,
                    baseline_project_dir=args.baseline_project_dir,
                    projects_dir=projects_dir,
                    texture_source_blocklist_path=texture_source_blocklist_path,
                    max_texture_source_probes=max(0, args.max_texture_source_probes),
                    explicit_texture_source_probe_frame_ids=explicit_probe_frame_ids,
                    explicit_texture_source_ablation_overrides=(
                        explicit_probe_ablation_overrides
                    ),
                    project_id_prefix=project_id_prefix,
                    force=args.force,
                )
            )
    supervisor_report: dict[str, Any] | None = None
    best_report: dict[str, Any] | None = None
    best_candidate_id: str | None = None
    best_is_baseline = False
    if pipeline_reports:
        candidate_project_dirs = {
            str(report["candidate_id"]): Path(str(report["project_dir"]))
            for report in pipeline_reports
        }
        if args.baseline_project_dir is not None:
            baseline_label = "baseline"
            candidate_project_dirs[baseline_label] = args.baseline_project_dir
            comparison_result = write_guarded_candidate_comparison_report(
                candidate_project_dirs,
                loop_dir / "raw_candidate_guarded_supervisor_report.json",
                baseline_label=baseline_label,
                force=True,
            )
        else:
            baseline_label = None
            comparison_result = write_candidate_comparison_report(
                candidate_project_dirs,
                loop_dir / "raw_candidate_supervisor_report.json",
                force=True,
            )
        supervisor_report = json.loads(
            comparison_result.report_path.read_text(encoding="utf-8")
        )
        _merge_supervisor_results(pipeline_reports, supervisor_report)
        best_candidate_id = comparison_result.best_candidate
        best_is_baseline = baseline_label is not None and best_candidate_id == baseline_label
        if not best_is_baseline:
            best_report = _pipeline_report_by_candidate_id(
                pipeline_reports,
                best_candidate_id,
            )
        if baseline_label is not None and not best_is_baseline:
            _attach_media_preflight_promotion_gate(
                supervisor_report,
                candidate_id=str(best_candidate_id),
                gate=media_preflight_promotion_gate,
            )
            if not bool(media_preflight_promotion_gate["passed"]):
                if best_report is not None:
                    best_report["media_preflight_promotion_gate"] = dict(
                        media_preflight_promotion_gate
                    )
                supervisor_report["best_candidate_before_media_preflight_gate"] = (
                    best_candidate_id
                )
                supervisor_report["best_candidate"] = baseline_label
                supervisor_report["media_preflight_promotion_gate"] = dict(
                    media_preflight_promotion_gate
                )
                best_candidate_id = baseline_label
                best_is_baseline = True
                best_report = None

    output_report = {
        "schema_version": "1.0",
        "upload_dir": str(upload_dir.resolve()),
        "baseline_project_dir": (
            str(args.baseline_project_dir.resolve())
            if args.baseline_project_dir is not None
            else None
        ),
        "media_staging": {
            "report_path": (
                str(media_staging.report_path)
                if media_staging.report_path is not None
                else None
            ),
            "input_photo_count": media_staging.input_photo_count,
            "input_video_count": media_staging.input_video_count,
            "extracted_video_frame_count": media_staging.extracted_video_frame_count,
            "staged_frame_count": len(media_staging.frame_paths),
        },
        "solid_thresholds": SOLID_THRESHOLDS,
        "texture_source_ablation_blocklist": _blocklist_report_summary(
            texture_source_blocklist_path
        ),
        "supervisor_report": supervisor_report,
        "media_preflight": media_preflight_report,
        "media_preflight_promotion_gate": media_preflight_promotion_gate,
        "segmentation_cache": cache_load.summary,
        "segmentation_candidates": [
            _segmentation_candidate_json(candidate) for candidate in segmentation_candidates
        ],
        "uploaded_texture_source": upload_texture_source_report,
        "texture_only_source_project": texture_only_source_project,
        "explicit_texture_source_probe_frame_ids": list(explicit_probe_frame_ids),
        "explicit_texture_source_ablation_overrides": {
            frame_id: list(hashes)
            for frame_id, hashes in sorted(explicit_probe_ablation_overrides.items())
        },
        "pipeline_reports": pipeline_reports,
        "best_candidate_id": best_candidate_id,
        "best_is_baseline": best_is_baseline,
        "best": best_report,
    }
    report_path = loop_dir / "raw_improvement_loop_report.json"
    report_path.write_text(
        json.dumps(output_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report={report_path}")
    if best_is_baseline:
        print(
            "best "
            "baseline=true "
            f"project={args.baseline_project_dir.resolve() if args.baseline_project_dir else ''}"
        )
    elif best_report is not None:
        print(
            "best "
            f"solid={best_report['solid']} "
            f"success={best_report['success']} "
            f"project={best_report['project_dir']} "
            f"glb={best_report['artifacts'].get('glb', '')}"
        )
    return 0


def _write_preflight_only_loop_report(
    *,
    upload_dir: Path,
    baseline_project_dir: Path | None,
    media_staging: UploadMediaStagingResult,
    texture_source_blocklist_path: Path | None,
    media_preflight_report: dict[str, Any],
    media_preflight_promotion_gate: dict[str, Any],
    segmentation_candidates: list[SegmentationCandidate],
    segmentation_cache: dict[str, Any],
    loop_dir: Path,
) -> Path:
    output_report = {
        "schema_version": "1.0",
        "mode": "preflight_only",
        "upload_dir": str(upload_dir.resolve()),
        "baseline_project_dir": (
            str(baseline_project_dir.resolve())
            if baseline_project_dir is not None
            else None
        ),
        "media_staging": {
            "report_path": (
                str(media_staging.report_path)
                if media_staging.report_path is not None
                else None
            ),
            "input_photo_count": media_staging.input_photo_count,
            "input_video_count": media_staging.input_video_count,
            "extracted_video_frame_count": media_staging.extracted_video_frame_count,
            "staged_frame_count": len(media_staging.frame_paths),
        },
        "solid_thresholds": SOLID_THRESHOLDS,
        "texture_source_ablation_blocklist": _blocklist_report_summary(
            texture_source_blocklist_path
        ),
        "supervisor_report": None,
        "media_preflight": media_preflight_report,
        "media_preflight_promotion_gate": media_preflight_promotion_gate,
        "segmentation_cache": segmentation_cache,
        "segmentation_candidates": [
            _segmentation_candidate_json(candidate) for candidate in segmentation_candidates
        ],
        "uploaded_texture_source": None,
        "pipeline_reports": [],
        "best_candidate_id": None,
        "best_is_baseline": False,
        "best": None,
    }
    report_path = loop_dir / "raw_improvement_loop_report.json"
    report_path.write_text(
        json.dumps(output_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _run_admission_only_mode(
    *,
    upload_dir: Path,
    baseline_project_dir: Path,
    raw_frame_paths: list[Path],
    media_staging: UploadMediaStagingResult,
    media_preflight_report: dict[str, Any],
    media_preflight_promotion_gate: dict[str, Any],
    segmentation_candidates: list[SegmentationCandidate],
    segmentation_cache: dict[str, Any],
    loop_dir: Path,
    projects_dir: Path,
    project_id_prefix: str,
    texture_source_blocklist_path: Path | None,
    max_texture_frames: int,
    force: bool,
) -> tuple[Path, bool, dict[str, Any] | None]:
    if not segmentation_candidates:
        raise ValueError("admission-only mode requires at least one segmentation candidate")

    uploaded_texture_source_report = _build_uploaded_texture_source_report(
        raw_frame_paths=raw_frame_paths,
        source_candidate=segmentation_candidates[0],
        loop_dir=loop_dir,
        min_geometry_frames=max(2, min(len(segmentation_candidates[0].selected_frames), 4)),
        max_texture_frames=max_texture_frames,
        force=force,
    )
    source_candidate_id = str(uploaded_texture_source_report["candidate_id"])
    report_candidate_id = f"{source_candidate_id}_texture_source_admission"
    project_dir = projects_dir / _scoped_project_name(
        project_id_prefix,
        f"loop_texture_{report_candidate_id}",
    )
    project_guard_path = (
        project_dir / "outputs" / "quality" / "texture_only_promotion_guard.json"
    )
    prior_rejections = _texture_source_ablation_rejection_keys(
        _load_texture_source_ablation_blocklist(texture_source_blocklist_path)
    )
    admission_audit = _write_texture_source_admission_for_candidate(
        project_dir,
        baseline_project_dir=baseline_project_dir,
        texture_source_project_dir=Path(str(uploaded_texture_source_report["project_dir"])),
        rejected_texture_source_hashes=tuple(prior_rejections["sha256"]),
        rejected_texture_source_frame_ids=tuple(prior_rejections["source_frame_id"]),
        force=force,
    )
    if admission_audit is None:
        raise SystemExit("failed to write texture-source admission audit")

    pipeline_reports = [
        _texture_source_admission_only_report(
            uploaded_texture_source_report,
            project_dir=project_dir,
            baseline_project_dir=baseline_project_dir,
            project_guard_path=project_guard_path,
            admission_audit=admission_audit,
            report_candidate_id=report_candidate_id,
            source_candidate_id=source_candidate_id,
            texture_source_blocklist_path=texture_source_blocklist_path,
            prior_rejection_hash_count=len(prior_rejections["sha256"]),
            prior_rejection_source_frame_count=len(prior_rejections["source_frame_id"]),
        )
    ]

    candidate_project_dirs = {
        str(report["candidate_id"]): Path(str(report["project_dir"]))
        for report in pipeline_reports
    }
    baseline_label = "baseline"
    candidate_project_dirs[baseline_label] = baseline_project_dir
    comparison_result = write_guarded_candidate_comparison_report(
        candidate_project_dirs,
        loop_dir / "raw_candidate_guarded_supervisor_report.json",
        baseline_label=baseline_label,
        force=True,
    )
    supervisor_report = json.loads(
        comparison_result.report_path.read_text(encoding="utf-8")
    )
    _merge_supervisor_results(pipeline_reports, supervisor_report)
    best_candidate_id = comparison_result.best_candidate
    best_is_baseline = best_candidate_id == baseline_label
    best_report = (
        None
        if best_is_baseline
        else _pipeline_report_by_candidate_id(pipeline_reports, best_candidate_id)
    )

    output_report = {
        "schema_version": "1.0",
        "mode": "admission_only",
        "upload_dir": str(upload_dir.resolve()),
        "baseline_project_dir": str(baseline_project_dir.resolve()),
        "media_staging": {
            "report_path": (
                str(media_staging.report_path)
                if media_staging.report_path is not None
                else None
            ),
            "input_photo_count": media_staging.input_photo_count,
            "input_video_count": media_staging.input_video_count,
            "extracted_video_frame_count": media_staging.extracted_video_frame_count,
            "staged_frame_count": len(media_staging.frame_paths),
        },
        "solid_thresholds": SOLID_THRESHOLDS,
        "texture_source_ablation_blocklist": _blocklist_report_summary(
            texture_source_blocklist_path
        ),
        "supervisor_report": supervisor_report,
        "media_preflight": media_preflight_report,
        "media_preflight_promotion_gate": media_preflight_promotion_gate,
        "segmentation_cache": segmentation_cache,
        "segmentation_candidates": [
            _segmentation_candidate_json(candidate) for candidate in segmentation_candidates
        ],
        "uploaded_texture_source": uploaded_texture_source_report,
        "texture_only_source_project": None,
        "texture_source_admission_audit": admission_audit,
        "pipeline_reports": pipeline_reports,
        "best_candidate_id": best_candidate_id,
        "best_is_baseline": best_is_baseline,
        "best": best_report,
    }
    report_path = loop_dir / "raw_improvement_loop_report.json"
    report_path.write_text(
        json.dumps(output_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, best_is_baseline, best_report


def _run_texture_only_source_mode(
    *,
    texture_source_project_dir: Path,
    baseline_project_dir: Path,
    loop_dir: Path,
    projects_dir: Path,
    project_id_prefix: str,
    texture_source_blocklist_path: Path | None,
    max_texture_source_probes: int,
    explicit_texture_source_probe_frame_ids: tuple[str, ...] = (),
    explicit_texture_source_ablation_overrides: dict[str, tuple[str, ...]] | None = None,
    force: bool,
) -> tuple[Path, bool, dict[str, Any] | None]:
    texture_only_source_project = _texture_source_project_report(texture_source_project_dir)
    pipeline_reports = _run_baseline_geometry_texture_fallback_candidates(
        [texture_only_source_project],
        baseline_project_dir=baseline_project_dir,
        projects_dir=projects_dir,
        texture_source_blocklist_path=texture_source_blocklist_path,
        max_texture_source_probes=max_texture_source_probes,
        explicit_texture_source_probe_frame_ids=explicit_texture_source_probe_frame_ids,
        explicit_texture_source_ablation_overrides=(
            explicit_texture_source_ablation_overrides or {}
        ),
        project_id_prefix=project_id_prefix,
        force=force,
    )
    supervisor_report: dict[str, Any] | None = None
    best_report: dict[str, Any] | None = None
    best_candidate_id: str | None = None
    best_is_baseline = False
    if pipeline_reports:
        candidate_project_dirs = {
            str(report["candidate_id"]): Path(str(report["project_dir"]))
            for report in pipeline_reports
        }
        baseline_label = "baseline"
        candidate_project_dirs[baseline_label] = baseline_project_dir
        comparison_result = write_guarded_candidate_comparison_report(
            candidate_project_dirs,
            loop_dir / "raw_candidate_guarded_supervisor_report.json",
            baseline_label=baseline_label,
            force=True,
        )
        supervisor_report = json.loads(
            comparison_result.report_path.read_text(encoding="utf-8")
        )
        _merge_supervisor_results(pipeline_reports, supervisor_report)
        best_candidate_id = comparison_result.best_candidate
        best_is_baseline = best_candidate_id == baseline_label
        if not best_is_baseline:
            best_report = _pipeline_report_by_candidate_id(
                pipeline_reports,
                best_candidate_id,
            )

    output_report = {
        "schema_version": "1.0",
        "mode": "texture_only_source_project",
        "upload_dir": None,
        "baseline_project_dir": str(baseline_project_dir.resolve()),
        "media_staging": {
            "report_path": None,
            "input_photo_count": 0,
            "input_video_count": 0,
            "extracted_video_frame_count": 0,
            "staged_frame_count": 0,
        },
        "solid_thresholds": SOLID_THRESHOLDS,
        "texture_source_ablation_blocklist": _blocklist_report_summary(
            texture_source_blocklist_path
        ),
        "supervisor_report": supervisor_report,
        "media_preflight": None,
        "media_preflight_promotion_gate": None,
        "segmentation_cache": {"used": False, "skipped": True},
        "segmentation_candidates": [],
        "uploaded_texture_source": None,
        "texture_only_source_project": texture_only_source_project,
        "explicit_texture_source_probe_frame_ids": list(
            explicit_texture_source_probe_frame_ids
        ),
        "explicit_texture_source_ablation_overrides": {
            frame_id: list(hashes)
            for frame_id, hashes in sorted(
                (explicit_texture_source_ablation_overrides or {}).items()
            )
        },
        "pipeline_reports": pipeline_reports,
        "best_candidate_id": best_candidate_id,
        "best_is_baseline": best_is_baseline,
        "best": best_report,
    }
    report_path = loop_dir / "raw_improvement_loop_report.json"
    report_path.write_text(
        json.dumps(output_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, best_is_baseline, best_report


def _sweep_segmenters(
    *,
    upload_dir: Path,
    raw_frame_paths: list[Path],
    loop_dir: Path,
    min_selected_frames: int,
    max_selected_frames: int,
    backends: tuple[RawSegmentationBackend, ...] | None = None,
    color_thresholds: tuple[float, ...] | None = None,
) -> list[SegmentationCandidate]:
    selected_backends: tuple[RawSegmentationBackend, ...] = backends or (
        "auto_subject_v1",
        "white_subject_v1",
        "alpha_or_border_color_v1",
    )
    selected_color_thresholds = color_thresholds or (0.08, 0.10, 0.12)
    candidates: list[SegmentationCandidate] = []
    for backend in selected_backends:
        for color_threshold in selected_color_thresholds:
            candidate_id = _candidate_id(
                upload_dir=upload_dir,
                backend=backend,
                color_delta_threshold=color_threshold,
            )
            project_dir = loop_dir / f"seg_{candidate_id}"
            ingest_image_sources(
                project_dir,
                input_paths=raw_frame_paths,
                min_frames=2,
                force=True,
            )
            result = segment_raw_image_observations(
                project_dir,
                color_delta_threshold=color_threshold,
                segmentation_backend=backend,
                force=True,
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            frames = _frame_selection_candidates(manifest)
            selected = _select_frames_from_candidates(
                frames,
                min_selected_frames=min_selected_frames,
                max_selected_frames=max_selected_frames,
            )
            if len(selected) < 2:
                continue
            texture_frames = _texture_source_frames(
                tuple(frames),
                selected_frames=tuple(selected),
                max_texture_frames=min(len(frames), max_selected_frames + 6),
            )
            candidate = SegmentationCandidate(
                candidate_id=candidate_id,
                backend=backend,
                color_delta_threshold=color_threshold,
                project_dir=project_dir,
                selected_frames=tuple(selected),
                frame_count=int(result.frame_count),
                selected_frame_count=len(selected),
                mean_mask_area_ratio=float(result.mean_mask_area_ratio),
                mean_bbox_area_ratio=float(result.mean_bbox_area_ratio),
                full_frame_bbox_fraction=float(result.full_frame_bbox_fraction),
                score=_segmentation_score(
                    selected,
                    mean_bbox_area_ratio=float(result.mean_bbox_area_ratio),
                    full_frame_bbox_fraction=float(result.full_frame_bbox_fraction),
                ),
                texture_frames=texture_frames,
            )
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _load_segmentation_candidate_cache(
    cache_report_path: Path | None,
    *,
    current_media_staging: UploadMediaStagingResult,
    min_selected_frames: int,
    max_selected_frames: int,
) -> SegmentationCacheLoadResult:
    summary: dict[str, Any] = {
        "version": SEGMENTATION_CANDIDATE_CACHE_VERSION,
        "requested": cache_report_path is not None,
        "used": False,
        "report_path": str(cache_report_path.resolve()) if cache_report_path else None,
        "reason": "not_requested",
    }
    if cache_report_path is None:
        return SegmentationCacheLoadResult([], summary)

    cache_report_path = Path(cache_report_path).expanduser().resolve()
    summary["report_path"] = str(cache_report_path)
    if not cache_report_path.is_file():
        summary["reason"] = "cache_report_missing"
        return SegmentationCacheLoadResult([], summary)
    try:
        cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        summary["reason"] = "cache_report_unreadable"
        return SegmentationCacheLoadResult([], summary)

    current_fingerprint = _media_staging_fingerprint(current_media_staging.report_path)
    cache_media_staging = cache_report.get("media_staging", {})
    cache_manifest_path = None
    if isinstance(cache_media_staging, dict):
        cache_manifest_path = cache_media_staging.get("report_path")
    cache_fingerprint = _media_staging_fingerprint(
        _resolve_report_path(cache_manifest_path, base_dir=cache_report_path.parent)
    )
    summary["current_media_fingerprint"] = current_fingerprint
    summary["cache_media_fingerprint"] = cache_fingerprint
    if current_fingerprint is None or cache_fingerprint is None:
        summary["reason"] = "media_fingerprint_missing"
        return SegmentationCacheLoadResult([], summary)
    if current_fingerprint["digest"] != cache_fingerprint["digest"]:
        summary["reason"] = "media_fingerprint_mismatch"
        return SegmentationCacheLoadResult([], summary)

    cache_media_preflight = cache_report.get("media_preflight", {})
    if not isinstance(cache_media_preflight, dict):
        summary["reason"] = "cache_media_preflight_missing"
        return SegmentationCacheLoadResult([], summary)
    cache_preflight_full = _load_cached_preflight_report(
        cache_media_preflight,
        base_dir=cache_report_path.parent,
    )
    cache_min = int(
        cache_media_preflight.get(
            "min_geometry_frames",
            cache_preflight_full.get("min_geometry_frames", -1),
        )
    )
    cache_max = int(
        cache_media_preflight.get(
            "max_geometry_frames",
            cache_preflight_full.get("max_geometry_frames", -1),
        )
    )
    summary["requested_min_selected_frames"] = min_selected_frames
    summary["requested_max_selected_frames"] = max_selected_frames
    summary["cache_min_geometry_frames"] = cache_min
    summary["cache_max_geometry_frames"] = cache_max
    if cache_min != min_selected_frames or cache_max != max_selected_frames:
        summary["reason"] = "frame_limit_mismatch"
        return SegmentationCacheLoadResult([], summary)

    candidate_payloads = cache_report.get("segmentation_candidates", [])
    if not isinstance(candidate_payloads, list) or not candidate_payloads:
        summary["reason"] = "cache_candidates_missing"
        return SegmentationCacheLoadResult([], summary)
    frame_lookup = _cached_preflight_frame_lookup(
        cache_media_preflight,
        base_dir=cache_report_path.parent,
    )
    current_frame_lookup = _staged_frame_lookup(current_media_staging.report_path)
    try:
        candidates = [
            _segmentation_candidate_from_json(
                candidate,
                frame_lookup=frame_lookup,
                current_frame_lookup=current_frame_lookup,
            )
            for candidate in candidate_payloads
            if isinstance(candidate, dict)
        ]
    except (KeyError, TypeError, ValueError, OSError) as exc:
        summary["reason"] = "cache_candidate_unreadable"
        summary["error"] = str(exc)
        return SegmentationCacheLoadResult([], summary)
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    if not candidates:
        summary["reason"] = "cache_candidates_empty"
        return SegmentationCacheLoadResult([], summary)
    summary["used"] = True
    summary["reason"] = "cache_hit"
    summary["candidate_count"] = len(candidates)
    summary["best_candidate_id"] = candidates[0].candidate_id
    return SegmentationCacheLoadResult(candidates, summary)


def _media_staging_fingerprint(report_path: Path | None) -> dict[str, Any] | None:
    if report_path is None:
        return None
    report_path = Path(report_path)
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    frames = payload.get("frames", {})
    if not isinstance(frames, dict) or not frames:
        return None
    manifest_dir = report_path.parent
    records: list[dict[str, Any]] = []
    for frame_name, metadata in sorted(frames.items()):
        if not isinstance(frame_name, str) or not isinstance(metadata, dict):
            continue
        frame_path = manifest_dir / frame_name
        if not frame_path.is_file():
            return None
        records.append(
            {
                "frame_name": frame_name,
                "sha256": _file_sha256(frame_path),
                "media_type": str(metadata.get("media_type", "photo")),
                "source_type": str(metadata.get("source_type", "image")),
                "source_media_id": _optional_str(metadata.get("source_media_id")),
                "source_frame_index": _optional_int(metadata.get("source_frame_index")),
            }
        )
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    media_type_counts: dict[str, int] = {}
    for record in records:
        media_type = str(record["media_type"])
        media_type_counts[media_type] = media_type_counts.get(media_type, 0) + 1
    return {
        "version": "staged_media_fingerprint_v1",
        "digest": digest,
        "frame_count": len(records),
        "media_type_counts": dict(sorted(media_type_counts.items())),
    }


def _staged_frame_lookup(report_path: Path | None) -> dict[str, FrameSelection]:
    if report_path is None:
        return {}
    report_path = Path(report_path)
    if not report_path.is_file():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    frames = payload.get("frames", {})
    if not isinstance(frames, dict):
        return {}
    lookup: dict[str, FrameSelection] = {}
    manifest_dir = report_path.parent
    for filename, metadata in sorted(frames.items()):
        if not isinstance(filename, str) or not isinstance(metadata, dict):
            continue
        source_path = manifest_dir / filename
        if not source_path.is_file():
            continue
        frame_id = Path(filename).stem
        lookup[frame_id] = FrameSelection(
            frame_id=frame_id,
            source_path=source_path,
            media_type=str(metadata.get("media_type", "photo")),
            source_type=str(metadata.get("source_type", "image")),
            source_media_id=_optional_str(metadata.get("source_media_id")),
            source_frame_index=_optional_int(metadata.get("source_frame_index")),
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.0,
            score=0.0,
            mask_area_ratio=0.0,
            bbox_area_ratio=0.0,
            full_frame_bbox=False,
        )
    return lookup


def _cached_preflight_frame_lookup(
    media_preflight: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, FrameSelection]:
    report = _load_cached_preflight_report(media_preflight, base_dir=base_dir)
    frames = report.get("frames", [])
    if not isinstance(frames, list):
        return {}
    lookup: dict[str, FrameSelection] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            lookup[str(frame["frame_id"])] = _frame_selection_from_json(frame)
        except FileNotFoundError:
            continue
    return lookup


def _load_cached_preflight_report(
    media_preflight: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    report_path = _resolve_report_path(media_preflight.get("report_path"), base_dir=base_dir)
    if report_path is None or not report_path.is_file():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return report if isinstance(report, dict) else {}


def _segmentation_candidate_from_json(
    payload: dict[str, Any],
    *,
    frame_lookup: dict[str, FrameSelection],
    current_frame_lookup: dict[str, FrameSelection],
) -> SegmentationCandidate:
    selected_frames = tuple(
        _frame_selection_from_json(
            frame,
            fallback=current_frame_lookup.get(str(frame.get("frame_id", ""))),
            prefer_fallback_source=True,
        )
        for frame in payload.get("selected_frames", [])
        if isinstance(frame, dict)
    )
    selected_by_id = {frame.frame_id: frame for frame in selected_frames}
    texture_frames_raw = payload.get("texture_source_frames", [])
    texture_frames = tuple(
        _frame_selection_from_json(
            frame,
            fallback=current_frame_lookup.get(str(frame.get("frame_id", "")))
            or selected_by_id.get(str(frame.get("frame_id", "")))
            or frame_lookup.get(str(frame.get("frame_id", ""))),
            prefer_fallback_source=True,
        )
        for frame in texture_frames_raw
        if isinstance(frame, dict)
    )
    if not selected_frames:
        raise ValueError("cache candidate has no selected frames")
    if not texture_frames:
        texture_frames = selected_frames
    backend = str(payload["backend"])
    if backend not in {"auto_subject_v1", "white_subject_v1", "alpha_or_border_color_v1"}:
        raise ValueError(f"unsupported cached segmentation backend: {backend}")
    return SegmentationCandidate(
        candidate_id=str(payload["candidate_id"]),
        backend=backend,  # type: ignore[arg-type]
        color_delta_threshold=float(payload["color_delta_threshold"]),
        project_dir=Path(str(payload["project_dir"])),
        selected_frames=selected_frames,
        frame_count=int(payload["frame_count"]),
        selected_frame_count=int(payload["selected_frame_count"]),
        mean_mask_area_ratio=float(payload["mean_mask_area_ratio"]),
        mean_bbox_area_ratio=float(payload["mean_bbox_area_ratio"]),
        full_frame_bbox_fraction=float(payload["full_frame_bbox_fraction"]),
        score=float(payload["score"]),
        texture_frames=texture_frames,
    )


def _frame_selection_from_json(
    payload: dict[str, Any],
    *,
    fallback: FrameSelection | None = None,
    prefer_fallback_source: bool = False,
) -> FrameSelection:
    if fallback is not None:
        merged = _frame_selection_json(fallback)
        fallback_source = {
            key: merged[key]
            for key in (
                "source_path",
                "media_type",
                "source_type",
                "source_media_id",
                "source_frame_index",
            )
        }
        merged.update(payload)
        if prefer_fallback_source:
            merged.update(fallback_source)
        payload = merged
    source_path = Path(str(payload["source_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(f"cached frame source is missing: {source_path}")
    return FrameSelection(
        frame_id=str(payload["frame_id"]),
        source_path=source_path,
        media_type=str(payload.get("media_type", "photo")),
        source_type=str(payload.get("source_type", "image")),
        source_media_id=_optional_str(payload.get("source_media_id")),
        source_frame_index=_optional_int(payload.get("source_frame_index")),
        geometry_reject_reasons=tuple(
            str(reason) for reason in payload.get("geometry_reject_reasons", [])
        ),
        geometry_body_fit_score=float(payload.get("geometry_body_fit_score", 0.0)),
        score=float(payload.get("score", 0.0)),
        mask_area_ratio=float(payload.get("mask_area_ratio", 0.0)),
        bbox_area_ratio=float(payload.get("bbox_area_ratio", 0.0)),
        full_frame_bbox=bool(payload.get("full_frame_bbox", False)),
        raw_view_label=str(payload.get("raw_view_label", "unknown_view")),
        raw_view_group=str(payload.get("raw_view_group", "unknown")),
        raw_pose_label=str(payload.get("raw_pose_label", "unknown_pose")),
        raw_frontal_face_confidence=float(payload.get("raw_frontal_face_confidence", 0.0)),
        raw_view_weight=float(payload.get("raw_view_weight", 0.0)),
    )


def _resolve_report_path(value: object, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _write_media_preflight_report(
    candidates: list[SegmentationCandidate],
    *,
    loop_dir: Path,
    min_geometry_frames: int,
    max_geometry_frames: int,
    max_texture_frames: int,
) -> dict[str, Any]:
    report = _build_media_preflight_report(
        candidates,
        min_geometry_frames=min_geometry_frames,
        max_geometry_frames=max_geometry_frames,
        max_texture_frames=max_texture_frames,
    )
    report_path = loop_dir / "raw_media_preflight_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "report_path": str(report_path),
        "best_candidate_id": report["best_candidate_id"],
        "candidate_count": report["candidate_count"],
        "ready_for_reconstruction": report["ready_for_reconstruction"],
        "ready_for_promotion": report["ready_for_promotion"],
        "warnings": report["warnings"],
        "promotion_rejection_reasons": report["promotion_rejection_reasons"],
        "face_texture_frame_count": report["face_texture_frame_count"],
        "photo_frame_count": report["photo_frame_count"],
        "video_frame_count": report["video_frame_count"],
        "geometry_video_frame_count": report["geometry_video_frame_count"],
        "texture_video_frame_count": report["texture_video_frame_count"],
        "media_type_counts": report["media_type_counts"],
        "view_group_counts": report["view_group_counts"],
        "min_geometry_frames": report["min_geometry_frames"],
        "max_geometry_frames": report["max_geometry_frames"],
        "recommended_geometry_frame_ids": report["recommended_geometry_frame_ids"],
        "recommended_texture_frame_ids": report["recommended_texture_frame_ids"],
        "geometry_safe_frame_count": report["geometry_safe_frame_count"],
        "texture_usable_frame_count": report["texture_usable_frame_count"],
    }


def _build_media_preflight_report(
    candidates: list[SegmentationCandidate],
    *,
    min_geometry_frames: int,
    max_geometry_frames: int,
    max_texture_frames: int,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("media preflight requires at least one segmentation candidate")
    min_geometry_frames = max(1, int(min_geometry_frames))
    max_geometry_frames = max(min_geometry_frames, int(max_geometry_frames))
    max_texture_frames = max(1, int(max_texture_frames))

    best = candidates[0]
    geometry_ranked = _rank_geometry_frames(_usable_geometry_frames(best.selected_frames))
    texture_ranked = _rank_preflight_texture_frames(_candidate_texture_frames(best))
    face_texture_frame_count = sum(1 for frame in texture_ranked if _is_face_texture_frame(frame))
    recommended_geometry = geometry_ranked[:max_geometry_frames]
    recommended_texture = texture_ranked[:max_texture_frames]
    geometry_ids = {frame.frame_id for frame in recommended_geometry}
    texture_ids = {frame.frame_id for frame in recommended_texture}
    all_frames = _unique_frames(best.selected_frames + _candidate_texture_frames(best))
    media_type_counts = _frame_count_by_attr(all_frames, "media_type")

    warnings: list[str] = []
    if len(geometry_ranked) < min_geometry_frames:
        warnings.append("insufficient_geometry_safe_frames")
    if len(texture_ranked) < 2:
        warnings.append("insufficient_texture_usable_frames")
    if not any(_is_face_texture_frame(frame) for frame in texture_ranked):
        warnings.append("missing_high_confidence_face_texture_frame")
    if not any(frame.media_type == "photo" for frame in all_frames):
        warnings.append("missing_photo_frames")
    if not any(frame.media_type == "video" for frame in all_frames):
        warnings.append("missing_video_frames")
    promotion_rejection_reasons = []
    if len(geometry_ranked) < min_geometry_frames:
        promotion_rejection_reasons.append("insufficient_geometry_safe_frames")
    if len(texture_ranked) < 2:
        promotion_rejection_reasons.append("insufficient_texture_usable_frames")
    if face_texture_frame_count < 1:
        promotion_rejection_reasons.append("missing_high_confidence_face_texture_frame")

    frame_reports = []
    for frame in all_frames:
        if frame.frame_id in geometry_ids and frame.frame_id in texture_ids:
            role = "geometry_and_texture"
        elif frame.frame_id in geometry_ids:
            role = "geometry"
        elif frame.frame_id in texture_ids:
            role = "texture"
        elif _is_geometry_fit_frame(frame):
            role = "geometry_reserve"
        elif not _has_hard_texture_source_reject(frame.geometry_reject_reasons):
            role = "texture_reserve"
        else:
            role = "rejected"
        frame_reports.append(
            {
                "frame_id": frame.frame_id,
                "source_path": str(frame.source_path),
                "media_type": frame.media_type,
                "source_type": frame.source_type,
                "source_media_id": frame.source_media_id,
                "source_frame_index": frame.source_frame_index,
                "recommended_role": role,
                "geometry_safe": _is_geometry_fit_frame(frame),
                "texture_usable": not _has_hard_texture_source_reject(
                    frame.geometry_reject_reasons
                ),
                "geometry_reject_reasons": list(frame.geometry_reject_reasons),
                "geometry_body_fit_score": frame.geometry_body_fit_score,
                "score": frame.score,
                "mask_area_ratio": frame.mask_area_ratio,
                "bbox_area_ratio": frame.bbox_area_ratio,
                "full_frame_bbox": frame.full_frame_bbox,
                "raw_view_label": frame.raw_view_label,
                "raw_view_group": frame.raw_view_group,
                "raw_pose_label": frame.raw_pose_label,
                "raw_frontal_face_confidence": frame.raw_frontal_face_confidence,
                "raw_view_weight": frame.raw_view_weight,
            }
        )

    return {
        "schema_version": "raw_media_preflight_v1",
        "source": "opencat3d.improve_raw_job_media_preflight",
        "best_candidate_id": best.candidate_id,
        "best_candidate_backend": best.backend,
        "best_candidate_color_delta_threshold": best.color_delta_threshold,
        "candidate_count": len(candidates),
        "ready_for_reconstruction": (
            len(geometry_ranked) >= min_geometry_frames and len(texture_ranked) >= 2
        ),
        "ready_for_promotion": not promotion_rejection_reasons,
        "warnings": warnings,
        "promotion_rejection_reasons": promotion_rejection_reasons,
        "min_geometry_frames": min_geometry_frames,
        "max_geometry_frames": max_geometry_frames,
        "max_texture_frames": max_texture_frames,
        "geometry_safe_frame_count": len(geometry_ranked),
        "texture_usable_frame_count": len(texture_ranked),
        "face_texture_frame_count": face_texture_frame_count,
        "photo_frame_count": int(media_type_counts.get("photo", 0)),
        "video_frame_count": int(media_type_counts.get("video", 0)),
        "geometry_video_frame_count": _media_count(tuple(geometry_ranked), "video"),
        "texture_video_frame_count": _media_count(tuple(texture_ranked), "video"),
        "recommended_geometry_frame_ids": [frame.frame_id for frame in recommended_geometry],
        "recommended_texture_frame_ids": [frame.frame_id for frame in recommended_texture],
        "media_type_counts": media_type_counts,
        "view_group_counts": _frame_count_by_attr(all_frames, "raw_view_group"),
        "candidate_summaries": [
            _media_preflight_candidate_summary(candidate) for candidate in candidates
        ],
        "frames": frame_reports,
    }


def _write_media_preflight_promotion_gate(
    media_preflight_report: dict[str, Any],
    *,
    loop_dir: Path,
) -> dict[str, Any]:
    gate = _media_preflight_promotion_gate(media_preflight_report)
    report_path = loop_dir / "raw_media_preflight_promotion_gate.json"
    gate["report_path"] = str(report_path)
    report_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def _media_preflight_promotion_gate(
    media_preflight_report: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "ready_for_reconstruction": bool(
            media_preflight_report.get("ready_for_reconstruction", False)
        ),
        "geometry_safe_frame_count_passed": int(
            media_preflight_report.get("geometry_safe_frame_count", 0)
        )
        >= int(media_preflight_report.get("min_geometry_frames", 2)),
        "texture_usable_frame_count_passed": int(
            media_preflight_report.get("texture_usable_frame_count", 0)
        )
        >= 2,
        "face_texture_frame_present": int(
            media_preflight_report.get("face_texture_frame_count", 0)
        )
        >= 1,
    }
    rejection_reasons = [key for key, value in checks.items() if not value]
    return {
        "version": "raw_media_preflight_promotion_gate_v1",
        "passed": not rejection_reasons,
        "checks": checks,
        "rejection_reasons": rejection_reasons,
        "media_preflight_report_path": media_preflight_report.get("report_path"),
        "recommended_geometry_frame_ids": list(
            media_preflight_report.get("recommended_geometry_frame_ids", [])
        ),
        "recommended_texture_frame_ids": list(
            media_preflight_report.get("recommended_texture_frame_ids", [])
        ),
    }


def _attach_media_preflight_promotion_gate(
    supervisor_report: dict[str, Any],
    *,
    candidate_id: str,
    gate: dict[str, Any],
) -> None:
    candidates = supervisor_report.get("candidates", [])
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict) or str(candidate.get("label", "")) != candidate_id:
            continue
        candidate["media_preflight_promotion_gate"] = dict(gate)
        promotion_guard = candidate.get("promotion_guard")
        if isinstance(promotion_guard, dict):
            checks = dict(promotion_guard.get("checks", {}))
            checks["media_preflight_promotion_gate_passed"] = bool(gate["passed"])
            rejection_reasons = list(promotion_guard.get("rejection_reasons", []))
            if not bool(gate["passed"]):
                rejection_reasons.append("media_preflight_promotion_gate_passed")
            promotion_guard["checks"] = checks
            promotion_guard["rejection_reasons"] = rejection_reasons
            promotion_guard["passed"] = not rejection_reasons
        break


def _media_preflight_candidate_summary(candidate: SegmentationCandidate) -> dict[str, Any]:
    frames = _unique_frames(candidate.selected_frames + _candidate_texture_frames(candidate))
    return {
        "candidate_id": candidate.candidate_id,
        "backend": candidate.backend,
        "color_delta_threshold": candidate.color_delta_threshold,
        "score": candidate.score,
        "selected_frame_count": candidate.selected_frame_count,
        "texture_source_frame_count": len(_candidate_texture_frames(candidate)),
        "geometry_safe_frame_count": len(_usable_geometry_frames(candidate.selected_frames)),
        "texture_usable_frame_count": len(
            _rank_preflight_texture_frames(_candidate_texture_frames(candidate))
        ),
        "media_type_counts": _frame_count_by_attr(frames, "media_type"),
        "view_group_counts": _frame_count_by_attr(frames, "raw_view_group"),
    }


def _rank_preflight_texture_frames(
    frames: tuple[FrameSelection, ...],
) -> tuple[FrameSelection, ...]:
    usable = [
        frame
        for frame in frames
        if not _has_hard_texture_source_reject(frame.geometry_reject_reasons)
    ]
    return tuple(
        sorted(
            usable,
            key=lambda frame: (
                not _is_face_texture_frame(frame),
                -float(frame.raw_frontal_face_confidence),
                -float(frame.raw_view_weight),
                -float(frame.score),
                frame.media_type != "photo",
                frame.frame_id,
            ),
        )
    )


def _is_face_texture_frame(frame: FrameSelection) -> bool:
    return (
        frame.raw_view_group == "frontal_face"
        or frame.raw_pose_label.endswith("closeup_face")
        or frame.raw_frontal_face_confidence >= 0.68
    )


def _unique_frames(frames: tuple[FrameSelection, ...]) -> tuple[FrameSelection, ...]:
    unique: dict[str, FrameSelection] = {}
    for frame in frames:
        unique.setdefault(frame.frame_id, frame)
    return tuple(unique.values())


def _frame_count_by_attr(frames: tuple[FrameSelection, ...], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        value = str(getattr(frame, attr))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _build_uploaded_texture_source_report(
    *,
    raw_frame_paths: list[Path],
    source_candidate: SegmentationCandidate,
    loop_dir: Path,
    min_geometry_frames: int,
    max_texture_frames: int,
    force: bool,
) -> dict[str, Any] | None:
    """Build a lightweight texture source from current upload media.

    This source is only used for supplemental texture evidence. It intentionally
    does not run fitting/export and never changes the baseline geometry.
    """

    if len(raw_frame_paths) < 2:
        return None
    min_geometry_frames = max(2, int(min_geometry_frames))
    max_texture_frames = max(min_geometry_frames, int(max_texture_frames))
    candidate_id = f"uploaded_texture_source_{source_candidate.candidate_id}"
    project_dir = loop_dir / f"texture_source_upload_{source_candidate.candidate_id}"
    if project_dir.exists() and force:
        shutil.rmtree(project_dir)

    ingest_image_sources(
        project_dir,
        input_paths=raw_frame_paths,
        min_frames=2,
        force=force,
    )
    result = segment_raw_image_observations(
        project_dir,
        color_delta_threshold=source_candidate.color_delta_threshold,
        segmentation_backend=source_candidate.backend,
        force=force,
    )
    keypoint_result = bootstrap_raw_keypoints_and_pose(
        project_dir,
        keypoint_confidence=0.5,
        force=force,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    frames = tuple(_frame_selection_candidates(manifest))
    if len(frames) < 2:
        return None

    geometry_frames = _uploaded_texture_source_geometry_frames(
        frames,
        min_geometry_frames=min(min_geometry_frames, len(frames)),
    )
    if len(geometry_frames) < 2:
        return None
    texture_frames = _uploaded_texture_source_texture_frames(
        frames,
        geometry_frames=geometry_frames,
        max_texture_frames=max_texture_frames,
    )
    if len(texture_frames) < len(geometry_frames):
        texture_frames = geometry_frames

    selection = select_raw_observation_frames(
        project_dir,
        min_frames=len(geometry_frames),
        max_selected_frames=len(geometry_frames),
        selected_frame_ids=[frame.frame_id for frame in geometry_frames],
        texture_frame_ids=[frame.frame_id for frame in texture_frames],
        selection_label="uploaded_closeup_texture_source",
        force=True,
    )
    texture_ids = {frame.frame_id for frame in texture_frames}
    closeup_texture_frames = tuple(
        frame
        for frame in texture_frames
        if frame.raw_pose_label.endswith("closeup_face")
        or frame.raw_view_group == "frontal_face"
        or frame.raw_frontal_face_confidence >= 0.68
    )
    report_path = project_dir / "observations" / "raw" / "uploaded_texture_source_report.json"
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.improve_raw_job.uploaded_texture_source",
        "version": "uploaded_closeup_texture_source_v1",
        "candidate_id": candidate_id,
        "base_candidate_id": source_candidate.candidate_id,
        "backend": source_candidate.backend,
        "color_delta_threshold": source_candidate.color_delta_threshold,
        "project_dir": str(project_dir),
        "success": True,
        "solid": False,
        "frame_count": len(frames),
        "selected_frame_count": len(geometry_frames),
        "texture_source_frame_count": len(texture_frames),
        "closeup_texture_frame_count": len(closeup_texture_frames),
        "selected_frames": [frame.frame_id for frame in geometry_frames],
        "texture_source_frames": [frame.frame_id for frame in texture_frames],
        "closeup_texture_frames": [frame.frame_id for frame in closeup_texture_frames],
        "geometry_view_counts": _view_group_counts(geometry_frames),
        "geometry_pose_counts": _pose_label_counts(geometry_frames),
        "texture_view_counts": _view_group_counts(texture_frames),
        "texture_pose_counts": _pose_label_counts(texture_frames),
        "frame_selection_report": str(selection.report_path),
        "texture_manifest_path": str(selection.texture_manifest_path),
        "report_path": str(report_path),
        "metrics": {
            "raw_selected_frame_count": len(geometry_frames),
            "raw_texture_frame_count": len(texture_frames),
            "raw_selected_video_frame_count": _media_count(geometry_frames, "video"),
            "raw_texture_video_frame_count": _media_count(texture_frames, "video"),
            "raw_face_priority_texture_used": float(bool(closeup_texture_frames)),
            "raw_uploaded_closeup_texture_frame_count": len(closeup_texture_frames),
            "raw_uploaded_source_observed_face_anchor_count": int(
                keypoint_result.observed_face_anchor_count
            ),
            "raw_uploaded_source_observed_face_anchor_frame_count": int(
                keypoint_result.observed_face_anchor_frame_count
            ),
            "raw_uploaded_source_mean_face_anchor_confidence": float(
                keypoint_result.mean_face_anchor_confidence
            ),
            "raw_mean_mask_area_ratio": float(result.mean_mask_area_ratio),
            "raw_mean_bbox_area_ratio": float(result.mean_bbox_area_ratio),
            "raw_full_frame_bbox_fraction": float(result.full_frame_bbox_fraction),
        },
        "frames": [
            _uploaded_texture_source_frame_json(
                frame,
                selected=frame.frame_id in {item.frame_id for item in geometry_frames},
                texture=frame.frame_id in texture_ids,
            )
            for frame in frames
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _uploaded_texture_source_geometry_frames(
    frames: tuple[FrameSelection, ...],
    *,
    min_geometry_frames: int,
) -> tuple[FrameSelection, ...]:
    body_like = tuple(
        frame
        for frame in frames
        if not frame.raw_pose_label.endswith("closeup_face")
        and frame.raw_view_group != "frontal_face"
        and not _has_severe_geometry_reject(frame.geometry_reject_reasons)
    )
    if len(body_like) >= min_geometry_frames:
        return _media_balanced_geometry_frames(body_like, min_frames=min_geometry_frames)
    safe = tuple(
        frame for frame in frames if not _has_severe_geometry_reject(frame.geometry_reject_reasons)
    )
    return _media_balanced_geometry_frames(
        safe or frames,
        min_frames=min(min_geometry_frames, len(frames)),
    )


def _uploaded_texture_source_texture_frames(
    frames: tuple[FrameSelection, ...],
    *,
    geometry_frames: tuple[FrameSelection, ...],
    max_texture_frames: int,
) -> tuple[FrameSelection, ...]:
    selected: list[FrameSelection] = []
    selected_ids: set[str] = set()
    geometry_ids = {frame.frame_id for frame in geometry_frames}
    priority_limit = max(0, max_texture_frames - len(geometry_frames))

    def add(frame: FrameSelection) -> None:
        if len(selected) >= max_texture_frames or frame.frame_id in selected_ids:
            return
        selected.append(frame)
        selected_ids.add(frame.frame_id)

    for frame in _rank_texture_source_reserve_frames(frames):
        if len(selected) >= priority_limit:
            break
        if frame.frame_id in geometry_ids:
            continue
        add(frame)
    for frame in _rank_score_frames(frames):
        if len(selected) >= priority_limit:
            break
        if frame.frame_id in geometry_ids:
            continue
        add(frame)
    for frame in geometry_frames:
        add(frame)
    return tuple(selected)


def _uploaded_texture_source_frame_json(
    frame: FrameSelection,
    *,
    selected: bool,
    texture: bool,
) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "source_path": str(frame.source_path),
        "media_type": frame.media_type,
        "source_type": frame.source_type,
        "source_media_id": frame.source_media_id,
        "source_frame_index": frame.source_frame_index,
        "selected": selected,
        "texture_enabled": texture,
        "geometry_reject_reasons": list(frame.geometry_reject_reasons),
        "geometry_body_fit_score": frame.geometry_body_fit_score,
        "score": frame.score,
        "mask_area_ratio": frame.mask_area_ratio,
        "bbox_area_ratio": frame.bbox_area_ratio,
        "full_frame_bbox": frame.full_frame_bbox,
        "raw_view_label": frame.raw_view_label,
        "raw_view_group": frame.raw_view_group,
        "raw_pose_label": frame.raw_pose_label,
        "raw_frontal_face_confidence": frame.raw_frontal_face_confidence,
        "raw_view_weight": frame.raw_view_weight,
    }


def _run_full_candidate_variants(
    candidate: SegmentationCandidate,
    *,
    projects_dir: Path,
    rank: int,
    min_selected_frames: int,
    max_selected_frames: int | None = None,
    camera_root_steps: int,
    max_geometry_variants: int,
    project_id_prefix: str = "",
    geometry_frame_sweep: bool = False,
    force: bool,
) -> list[dict[str, Any]]:
    geometry_frame_target = _geometry_variant_frame_target(
        candidate.selected_frames,
        min_selected_frames=min_selected_frames,
        max_selected_frames=max_selected_frames,
    )
    variants = (
        _geometry_frame_sweep_variants(
            candidate.selected_frames,
            min_selected_frames=min_selected_frames,
            max_selected_frames=geometry_frame_target,
            max_variants=max_geometry_variants,
        )
        if geometry_frame_sweep
        else _geometry_variants(
            candidate.selected_frames,
            min_frames=geometry_frame_target,
            max_variants=max_geometry_variants,
        )
    )
    return [
        _run_full_candidate(
            candidate,
            geometry_variant=variant,
            projects_dir=projects_dir,
            rank=rank,
            min_selected_frames=min_selected_frames,
            camera_root_steps=camera_root_steps,
            project_id_prefix=project_id_prefix,
            force=force,
        )
        for variant in variants
    ]


def _run_full_candidate(
    candidate: SegmentationCandidate,
    *,
    geometry_variant: GeometryVariant | None = None,
    projects_dir: Path,
    rank: int,
    min_selected_frames: int,
    camera_root_steps: int,
    project_id_prefix: str = "",
    force: bool,
) -> dict[str, Any]:
    if geometry_variant is None:
        geometry_variant = _geometry_variants(
            candidate.selected_frames,
            min_frames=min(min_selected_frames, len(candidate.selected_frames)),
            max_variants=1,
        )[0]
    texture_frames = _candidate_texture_frames(candidate)
    report_candidate_id = f"{candidate.candidate_id}_{geometry_variant.variant_id}"
    job_id = _scoped_project_name(
        project_id_prefix,
        f"loop_{rank:02d}_{report_candidate_id}",
    )
    project_dir = projects_dir / job_id
    selected_paths = _stage_selected_candidate_frames(
        texture_frames,
        project_dir=project_dir,
        force=force,
    )
    geometry_frames = geometry_variant.geometry_frames
    geometry_frame_ids = _staged_frame_ids_for_frames(
        texture_frames,
        geometry_frames,
    )
    texture_frame_ids = _staged_frame_ids(len(selected_paths))
    pipeline_min_frames = max(2, min(min_selected_frames, len(geometry_frame_ids)))
    result = run_raw_pipeline(
        project_dir,
        input_paths=selected_paths,
        min_frames=pipeline_min_frames,
        color_delta_threshold=candidate.color_delta_threshold,
        segmentation_backend=candidate.backend,
        camera_root_steps=camera_root_steps,
        max_selected_frames=len(geometry_frame_ids),
        selected_frame_ids=geometry_frame_ids,
        texture_frame_ids=texture_frame_ids,
        frame_selection_label="improve_raw_job_geometry_texture_split",
        force=force,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    solid = (
        bool(result.success)
        and float(metrics["raw_mean_bbox_area_ratio"])
        <= SOLID_THRESHOLDS["max_mean_bbox_area_ratio"]
        and float(metrics["raw_full_frame_bbox_fraction"])
        <= SOLID_THRESHOLDS["max_full_frame_bbox_fraction"]
        and float(metrics["overlay_mean_silhouette_iou"])
        >= SOLID_THRESHOLDS["min_overlay_mean_silhouette_iou"]
        and float(metrics["silhouette_residual_mean_union_error"])
        <= SOLID_THRESHOLDS["max_silhouette_residual_mean_union_error"]
        and float(metrics["raw_texture_unknown_luma_fraction_of_foreground"])
        >= SOLID_THRESHOLDS["min_unknown_texture_luma_fraction_of_foreground"]
    )
    return {
        "candidate_id": report_candidate_id,
        "base_candidate_id": candidate.candidate_id,
        "geometry_variant_id": geometry_variant.variant_id,
        "backend": candidate.backend,
        "color_delta_threshold": candidate.color_delta_threshold,
        "project_dir": str(project_dir),
        "selected_frame_count": len(selected_paths),
        "selected_frames": [frame.frame_id for frame in candidate.selected_frames],
        "texture_source_frame_count": len(texture_frames),
        "texture_source_frames": [frame.frame_id for frame in texture_frames],
        "texture_source_added_reserve_frames": [
            frame.frame_id
            for frame in texture_frames
            if frame.frame_id not in {item.frame_id for item in candidate.selected_frames}
        ],
        "geometry_frame_count": len(geometry_frame_ids),
        "geometry_min_frame_count": pipeline_min_frames,
        "geometry_frames": [frame.frame_id for frame in geometry_frames],
        "geometry_view_counts": _view_group_counts(geometry_frames),
        "geometry_pose_counts": _pose_label_counts(geometry_frames),
        "geometry_pose_gate": _pose_gate_report(texture_frames, geometry_frames),
        "texture_frame_count": len(texture_frame_ids),
        "texture_frames": [frame.frame_id for frame in texture_frames],
        "texture_view_counts": _view_group_counts(texture_frames),
        "texture_pose_counts": _pose_label_counts(texture_frames),
        "success": bool(result.success),
        "solid": solid,
        "metrics": _report_metric_subset(metrics),
        "artifacts": summary["artifacts"],
    }


def _run_baseline_geometry_texture_fallback_candidates(
    texture_source_reports: list[dict[str, Any]],
    *,
    baseline_project_dir: Path,
    projects_dir: Path,
    texture_source_blocklist_path: Path | None,
    max_texture_source_probes: int,
    explicit_texture_source_probe_frame_ids: tuple[str, ...] = (),
    explicit_texture_source_ablation_overrides: dict[str, tuple[str, ...]] | None = None,
    project_id_prefix: str = "",
    force: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    fallback_count = len(texture_source_reports)
    for fallback_rank, texture_source_report in enumerate(texture_source_reports, start=1):
        source_reports = _run_baseline_geometry_texture_candidates(
            texture_source_report,
            baseline_project_dir=baseline_project_dir,
            projects_dir=projects_dir,
            texture_source_blocklist_path=texture_source_blocklist_path,
            max_texture_source_probes=max_texture_source_probes,
            explicit_texture_source_probe_frame_ids=explicit_texture_source_probe_frame_ids,
            explicit_texture_source_ablation_overrides=(
                explicit_texture_source_ablation_overrides or {}
            ),
            project_id_prefix=project_id_prefix,
            force=force,
        )
        for report in source_reports:
            report["texture_source_fallback_rank"] = fallback_rank
            report["texture_source_fallback_count"] = fallback_count
        reports.extend(source_reports)
        if _texture_source_reports_have_guarded_success(source_reports):
            break
    return reports


def _texture_source_project_report(project_dir: Path) -> dict[str, Any]:
    project_dir = Path(project_dir).expanduser().resolve()
    return {
        "candidate_id": _safe_candidate_token(project_dir.name),
        "base_candidate_id": _safe_candidate_token(project_dir.name),
        "geometry_variant_id": "texture_only_source_project",
        "backend": "texture_only_source_project",
        "color_delta_threshold": None,
        "project_dir": str(project_dir),
        "success": True,
        "solid": True,
        "metrics": {},
    }


def _should_run_baseline_texture_fallback(
    *,
    baseline_project_dir: Path | None,
    pipeline_reports: list[dict[str, Any]],
    texture_only_source_project: dict[str, Any] | None,
) -> bool:
    return (
        baseline_project_dir is not None
        and bool(pipeline_reports)
        and texture_only_source_project is None
    )


def _run_baseline_geometry_texture_candidates(
    texture_source_report: dict[str, Any],
    *,
    baseline_project_dir: Path,
    projects_dir: Path,
    texture_source_blocklist_path: Path | None,
    max_texture_source_probes: int,
    explicit_texture_source_probe_frame_ids: tuple[str, ...] = (),
    explicit_texture_source_ablation_overrides: dict[str, tuple[str, ...]] | None = None,
    project_id_prefix: str = "",
    force: bool,
) -> list[dict[str, Any]]:
    ablation_overrides = explicit_texture_source_ablation_overrides or {}
    if explicit_texture_source_probe_frame_ids:
        reports: list[dict[str, Any]] = []
        for index, source_frame_id in enumerate(
            explicit_texture_source_probe_frame_ids,
            start=1,
        ):
            probe_suffix = (
                f"explicit_probe_{index:02d}_{_safe_candidate_token(source_frame_id)}"
            )
            reports.append(
                _run_baseline_geometry_texture_candidate(
                    texture_source_report,
                    baseline_project_dir=baseline_project_dir,
                    projects_dir=projects_dir,
                    texture_source_blocklist_path=texture_source_blocklist_path,
                    probe_texture_source_frame_ids=(source_frame_id,),
                    probe_texture_source_ablation_overrides=ablation_overrides,
                    candidate_suffix=probe_suffix,
                    project_id_prefix=project_id_prefix,
                    force=force,
                )
            )
        return reports

    normal_report = _run_baseline_geometry_texture_candidate(
        texture_source_report,
        baseline_project_dir=baseline_project_dir,
        projects_dir=projects_dir,
        texture_source_blocklist_path=texture_source_blocklist_path,
        probe_texture_source_frame_ids=(),
        probe_texture_source_ablation_overrides={},
        candidate_suffix="",
        project_id_prefix=project_id_prefix,
        force=force,
    )
    reports = [normal_report]
    if (
        _texture_source_reports_have_guarded_success([normal_report])
        or max_texture_source_probes <= 0
    ):
        return reports
    normal_report["texture_only_probe_retry_reason"] = (
        "guard_rejected_added_frames"
        if int(normal_report.get("texture_only_added_unique_frame_count", 0)) > 0
        else "no_unique_texture_frames"
    )

    rejected_keys = _texture_source_ablation_rejection_keys(
        _load_texture_source_ablation_blocklist(texture_source_blocklist_path)
    )
    probe_search = _texture_source_probe_search_from_bake_report(
        Path(str(normal_report["texture_only_bake_report"])),
        rejected_texture_source_hashes=tuple(rejected_keys["sha256"]),
        rejected_texture_source_frame_ids=tuple(rejected_keys["source_frame_id"]),
        max_probe_count=max_texture_source_probes,
        allow_source_id_reprobe_when_hash_differs=True,
    )
    normal_report["texture_only_probe_search"] = {
        "probe_source_frame_ids": list(probe_search["probe_source_frame_ids"]),
        "skipped_probe_frames": list(probe_search["skipped_probe_frames"]),
    }
    probe_source_frame_ids = tuple(str(item) for item in probe_search["probe_source_frame_ids"])
    for index, source_frame_id in enumerate(probe_source_frame_ids, start=1):
        probe_suffix = f"probe_{index:02d}_{_safe_candidate_token(source_frame_id)}"
        reports.append(
            _run_baseline_geometry_texture_candidate(
                texture_source_report,
                baseline_project_dir=baseline_project_dir,
                projects_dir=projects_dir,
                texture_source_blocklist_path=texture_source_blocklist_path,
                probe_texture_source_frame_ids=(source_frame_id,),
                probe_texture_source_ablation_overrides=ablation_overrides,
                candidate_suffix=probe_suffix,
                project_id_prefix=project_id_prefix,
                force=force,
            )
        )
    return reports


def _texture_source_reports_have_guarded_success(
    reports: list[dict[str, Any]],
) -> bool:
    return any(
        bool(report.get("texture_only_guarded_result_success", False))
        and bool(report.get("texture_only_promotion_guard_passed", False))
        for report in reports
    )


def _run_baseline_geometry_texture_candidate(
    texture_source_report: dict[str, Any],
    *,
    baseline_project_dir: Path,
    projects_dir: Path,
    texture_source_blocklist_path: Path | None,
    probe_texture_source_frame_ids: tuple[str, ...] = (),
    probe_texture_source_ablation_overrides: dict[str, tuple[str, ...]] | None = None,
    candidate_suffix: str = "",
    project_id_prefix: str = "",
    force: bool,
) -> dict[str, Any]:
    source_candidate_id = str(texture_source_report["candidate_id"])
    base_report_candidate_id = f"{source_candidate_id}_baseline_geometry_texture"
    suffix = f"_{_safe_candidate_token(candidate_suffix)}" if candidate_suffix else ""
    report_candidate_id = f"{base_report_candidate_id}{suffix}"
    project_dir = projects_dir / _scoped_project_name(
        project_id_prefix,
        f"loop_texture_{report_candidate_id}",
    )
    project_guard_path = (
        project_dir / "outputs" / "quality" / "texture_only_promotion_guard.json"
    )
    prior_rejection_entries = _merged_texture_source_ablation_rejections(
        _load_texture_source_ablation_blocklist(texture_source_blocklist_path),
        _supplemental_texture_source_rejections_from_guard(project_guard_path),
    )
    if texture_source_blocklist_path is not None:
        _write_texture_source_ablation_blocklist(
            texture_source_blocklist_path,
            prior_rejection_entries,
            source_paths=(project_guard_path,),
        )
    prior_rejections = _texture_source_ablation_rejection_keys(prior_rejection_entries)
    admission_audit = _write_texture_source_admission_for_candidate(
        project_dir,
        baseline_project_dir=baseline_project_dir,
        texture_source_project_dir=Path(str(texture_source_report["project_dir"])),
        rejected_texture_source_hashes=tuple(prior_rejections["sha256"]),
        rejected_texture_source_frame_ids=tuple(prior_rejections["source_frame_id"]),
        force=force,
    )
    if (
        admission_audit is not None
        and not bool(admission_audit.get("passed", False))
        and not probe_texture_source_frame_ids
    ):
        return _texture_source_admission_skipped_report(
            texture_source_report,
            project_dir=project_dir,
            baseline_project_dir=baseline_project_dir,
            project_guard_path=project_guard_path,
            admission_audit=admission_audit,
            report_candidate_id=report_candidate_id,
            source_candidate_id=source_candidate_id,
        )
    result = run_baseline_geometry_texture_bake(
        project_dir,
        baseline_project_dir=baseline_project_dir,
        texture_source_project_dir=Path(str(texture_source_report["project_dir"])),
        rejected_texture_source_hashes=tuple(prior_rejections["sha256"]),
        rejected_texture_source_frame_ids=tuple(prior_rejections["source_frame_id"]),
        probe_texture_source_frame_ids=probe_texture_source_frame_ids,
        probe_texture_source_ablation_overrides=(
            probe_texture_source_ablation_overrides or {}
        ),
        force=force,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    summary_success = bool(summary.get("success", result.success))
    metrics = summary["metrics"]
    bake_report_path = (
        project_dir / "outputs" / "quality" / "baseline_geometry_texture_bake.json"
    )
    bake_report = json.loads(bake_report_path.read_text(encoding="utf-8"))
    current_rejection_entries = _supplemental_texture_source_rejections_from_guard(
        Path(result.promotion_guard_path)
    )
    merged_rejection_entries = _merged_texture_source_ablation_rejections(
        prior_rejection_entries,
        current_rejection_entries,
    )
    if texture_source_blocklist_path is not None:
        _write_texture_source_ablation_blocklist(
            texture_source_blocklist_path,
            merged_rejection_entries,
            source_paths=(project_guard_path, Path(result.promotion_guard_path)),
        )
    selected_frame_ids = tuple(str(frame_id) for frame_id in bake_report["selected_frame_ids"])
    texture_frame_ids = tuple(str(frame_id) for frame_id in bake_report["texture_frame_ids"])
    solid = (
        summary_success
        and float(metrics["raw_mean_bbox_area_ratio"])
        <= SOLID_THRESHOLDS["max_mean_bbox_area_ratio"]
        and float(metrics["raw_full_frame_bbox_fraction"])
        <= SOLID_THRESHOLDS["max_full_frame_bbox_fraction"]
        and float(metrics["overlay_mean_silhouette_iou"])
        >= SOLID_THRESHOLDS["min_overlay_mean_silhouette_iou"]
        and float(metrics["silhouette_residual_mean_union_error"])
        <= SOLID_THRESHOLDS["max_silhouette_residual_mean_union_error"]
        and float(metrics["raw_texture_unknown_luma_fraction_of_foreground"])
        >= SOLID_THRESHOLDS["min_unknown_texture_luma_fraction_of_foreground"]
    )
    return {
        "candidate_id": report_candidate_id,
        "base_candidate_id": source_candidate_id,
        "geometry_variant_id": "baseline_geometry_texture_only",
        "backend": texture_source_report.get("backend", "unknown"),
        "color_delta_threshold": texture_source_report.get("color_delta_threshold"),
        "project_dir": str(project_dir),
        "baseline_project_dir": str(Path(baseline_project_dir).resolve()),
        "texture_source_project_dir": str(texture_source_report["project_dir"]),
        "selected_frame_count": len(selected_frame_ids),
        "selected_frames": list(selected_frame_ids),
        "geometry_frame_count": len(selected_frame_ids),
        "geometry_min_frame_count": len(selected_frame_ids),
        "geometry_frames": list(selected_frame_ids),
        "geometry_view_counts": {},
        "geometry_pose_counts": {},
        "geometry_pose_gate": {
            "version": "baseline_geometry_texture_bake_v1",
            "geometry_frame_ids": list(selected_frame_ids),
            "gated_texture_only_frame_ids": [
                frame_id for frame_id in texture_frame_ids if frame_id not in selected_frame_ids
            ],
            "passed_risky_geometry_frame_ids": [],
            "frame_reasons": {
                frame_id: (
                    "baseline_geometry_frame"
                    if frame_id in selected_frame_ids
                    else "texture_only_supplemental"
                )
                for frame_id in texture_frame_ids
            },
        },
        "texture_frame_count": len(texture_frame_ids),
        "texture_frames": list(texture_frame_ids),
        "texture_view_counts": {},
        "texture_pose_counts": {},
        "texture_only_added_unique_frame_count": result.added_unique_texture_frame_count,
        "texture_only_probe_source_frame_ids": list(probe_texture_source_frame_ids),
        "texture_only_probe_ablation_overrides": {
            frame_id: list(hashes)
            for frame_id, hashes in sorted(
                (probe_texture_source_ablation_overrides or {}).items()
            )
        },
        "texture_only_probe_added_frame_count": int(
            bake_report.get("probe_added_texture_frame_count", 0)
        ),
        "texture_only_probe_bypassed_rejected_sha256_count": int(
            bake_report.get("probe_bypassed_rejected_sha256_count", 0)
        ),
        "texture_only_probe_added_frames": list(
            bake_report.get("probe_added_texture_frames", [])
        ),
        "texture_only_rejected_ablation_frame_count": int(
            bake_report.get("rejected_ablation_texture_frame_count", 0)
        ),
        "texture_only_rejected_ablation_frames": list(
            bake_report.get("rejected_ablation_texture_frames", [])
        ),
        "texture_only_prior_rejection_hash_count": len(prior_rejections["sha256"]),
        "texture_only_prior_rejection_source_frame_count": len(
            prior_rejections["source_frame_id"]
        ),
        "texture_only_ablation_blocklist": (
            str(texture_source_blocklist_path) if texture_source_blocklist_path else None
        ),
        "texture_only_ablation_blocklist_entry_count": len(merged_rejection_entries),
        "texture_source_admission_audit": bake_report.get(
            "texture_source_admission_audit_path",
            (
                str(admission_audit.get("report_path"))
                if admission_audit is not None
                else None
            ),
        ),
        "texture_source_admission_audit_passed": bool(
            bake_report.get(
                "texture_source_admission_audit_passed",
                admission_audit.get("passed") if admission_audit is not None else False,
            )
        ),
        "texture_source_admission_unique_accepted_frame_count": int(
            bake_report.get(
                "texture_source_admission_unique_accepted_frame_count",
                admission_audit.get("unique_accepted_frame_count", 0)
                if admission_audit is not None
                else 0,
            )
        ),
        "texture_source_admission_protected_face_frame_count": int(
            bake_report.get(
                "texture_source_admission_protected_face_frame_count",
                admission_audit.get("protected_face_source_patch_frame_count", 0)
                if admission_audit is not None
                else 0,
            )
        ),
        "texture_source_admission_rejected_ablation_frame_count": int(
            bake_report.get(
                "texture_source_admission_rejected_ablation_frame_count",
                admission_audit.get("rejected_ablation_frame_count", 0)
                if admission_audit is not None
                else 0,
            )
        ),
        "texture_source_admission_source_patch_blocked_ablation_frame_count": int(
            bake_report.get(
                "texture_source_admission_source_patch_blocked_ablation_frame_count",
                admission_audit.get("source_patch_blocked_ablation_frame_count", 0)
                if admission_audit is not None
                else 0,
            )
        ),
        "texture_only_promotion_guard_passed": result.promotion_guard_passed,
        "texture_only_guarded_result_success": bool(result.success),
        "texture_only_bake_report": str(bake_report_path),
        "texture_only_promotion_guard": str(result.promotion_guard_path),
        "success": summary_success,
        "solid": solid,
        "metrics": _report_metric_subset(metrics),
        "artifacts": summary["artifacts"],
    }


def _write_texture_source_admission_for_candidate(
    project_dir: Path,
    *,
    baseline_project_dir: Path,
    texture_source_project_dir: Path,
    rejected_texture_source_hashes: tuple[str, ...],
    rejected_texture_source_frame_ids: tuple[str, ...],
    force: bool,
) -> dict[str, Any] | None:
    output_path = project_dir / "outputs" / "quality" / "texture_source_admission_audit.json"
    try:
        return write_texture_source_admission_audit(
            baseline_project_dir,
            texture_source_project_dir,
            output_path,
            rejected_texture_source_hashes=rejected_texture_source_hashes,
            rejected_texture_source_frame_ids=rejected_texture_source_frame_ids,
            force=force,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
        return None


def _texture_source_admission_only_report(
    texture_source_report: dict[str, Any],
    *,
    project_dir: Path,
    baseline_project_dir: Path,
    project_guard_path: Path,
    admission_audit: dict[str, Any],
    report_candidate_id: str,
    source_candidate_id: str,
    texture_source_blocklist_path: Path | None,
    prior_rejection_hash_count: int,
    prior_rejection_source_frame_count: int,
) -> dict[str, Any]:
    quality_dir = project_dir / "outputs" / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    metrics = _texture_source_admission_metrics(admission_audit)
    summary_path = quality_dir / "raw_pipeline_summary.json"
    summary = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "source": "opencat3d.texture_source_admission_only",
        "success": False,
        "baseline_project_dir": str(Path(baseline_project_dir).resolve()),
        "texture_source_project_dir": str(texture_source_report["project_dir"]),
        "steps": {
            "texture_source_project_build": True,
            "texture_source_admission_audit": True,
            "texture_only_bake": False,
            "texture_only_promotion_guard": False,
        },
        "metrics": metrics,
        "artifacts": {
            "texture_source_admission_audit": str(admission_audit["report_path"]),
            "texture_only_promotion_guard": str(project_guard_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    admission_passed = bool(admission_audit.get("passed", False))
    return {
        "candidate_id": report_candidate_id,
        "base_candidate_id": source_candidate_id,
        "geometry_variant_id": "texture_source_admission_only",
        "backend": texture_source_report.get("backend", "unknown"),
        "color_delta_threshold": texture_source_report.get("color_delta_threshold"),
        "project_dir": str(project_dir),
        "baseline_project_dir": str(Path(baseline_project_dir).resolve()),
        "texture_source_project_dir": str(texture_source_report["project_dir"]),
        "selected_frame_count": 0,
        "selected_frames": [],
        "geometry_frame_count": 0,
        "geometry_min_frame_count": 0,
        "geometry_frames": [],
        "geometry_view_counts": {},
        "geometry_pose_counts": {},
        "geometry_pose_gate": {
            "version": "texture_source_admission_only_v1",
            "geometry_frame_ids": [],
            "gated_texture_only_frame_ids": [],
            "passed_risky_geometry_frame_ids": [],
            "frame_reasons": {},
        },
        "texture_frame_count": 0,
        "texture_frames": [],
        "texture_view_counts": {},
        "texture_pose_counts": {},
        "texture_only_added_unique_frame_count": 0,
        "texture_only_probe_source_frame_ids": [],
        "texture_only_probe_added_frame_count": 0,
        "texture_only_probe_added_frames": [],
        "texture_only_rejected_ablation_frame_count": int(
            admission_audit.get("rejected_ablation_frame_count", 0)
        ),
        "texture_only_rejected_ablation_frames": [],
        "texture_only_prior_rejection_hash_count": int(prior_rejection_hash_count),
        "texture_only_prior_rejection_source_frame_count": int(
            prior_rejection_source_frame_count
        ),
        "texture_only_ablation_blocklist": (
            str(texture_source_blocklist_path) if texture_source_blocklist_path else None
        ),
        "texture_only_ablation_blocklist_entry_count": (
            int(prior_rejection_hash_count) + int(prior_rejection_source_frame_count)
        ),
        "texture_source_admission_audit": str(admission_audit["report_path"]),
        "texture_source_admission_audit_passed": admission_passed,
        "texture_source_admission_rejection_reasons": list(
            admission_audit.get("rejection_reasons", [])
        ),
        "texture_source_admission_unique_accepted_frame_count": int(
            admission_audit.get("unique_accepted_frame_count", 0)
        ),
        "texture_source_admission_protected_face_frame_count": int(
            admission_audit.get("protected_face_source_patch_frame_count", 0)
        ),
        "texture_source_admission_rejected_ablation_frame_count": int(
            admission_audit.get("rejected_ablation_frame_count", 0)
        ),
        "texture_source_admission_source_patch_blocked_ablation_frame_count": int(
            admission_audit.get("source_patch_blocked_ablation_frame_count", 0)
        ),
        "texture_only_promotion_guard_passed": False,
        "texture_only_guarded_result_success": False,
        "texture_only_bake_report": None,
        "texture_only_promotion_guard": str(project_guard_path),
        "success": False,
        "solid": False,
        "metrics": _report_metric_subset(metrics),
        "artifacts": summary["artifacts"],
    }


def _texture_source_admission_skipped_report(
    texture_source_report: dict[str, Any],
    *,
    project_dir: Path,
    baseline_project_dir: Path,
    project_guard_path: Path,
    admission_audit: dict[str, Any],
    report_candidate_id: str,
    source_candidate_id: str,
) -> dict[str, Any]:
    quality_dir = project_dir / "outputs" / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    metrics = _texture_source_admission_metrics(admission_audit)
    summary_path = quality_dir / "raw_pipeline_summary.json"
    summary = {
        "schema_version": "1.0",
        "project_id": project_dir.name,
        "source": "opencat3d.texture_source_admission_skip",
        "success": False,
        "baseline_project_dir": str(Path(baseline_project_dir).resolve()),
        "texture_source_project_dir": str(texture_source_report["project_dir"]),
        "steps": {
            "texture_source_admission_audit": True,
            "texture_only_bake": False,
            "texture_only_promotion_guard": False,
        },
        "metrics": metrics,
        "artifacts": {
            "texture_source_admission_audit": str(admission_audit["report_path"]),
            "texture_only_promotion_guard": str(project_guard_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "candidate_id": report_candidate_id,
        "base_candidate_id": source_candidate_id,
        "geometry_variant_id": "baseline_geometry_texture_only",
        "backend": texture_source_report.get("backend", "unknown"),
        "color_delta_threshold": texture_source_report.get("color_delta_threshold"),
        "project_dir": str(project_dir),
        "baseline_project_dir": str(Path(baseline_project_dir).resolve()),
        "texture_source_project_dir": str(texture_source_report["project_dir"]),
        "selected_frame_count": 0,
        "selected_frames": [],
        "geometry_frame_count": 0,
        "geometry_min_frame_count": 0,
        "geometry_frames": [],
        "geometry_view_counts": {},
        "geometry_pose_counts": {},
        "geometry_pose_gate": {
            "version": "texture_source_admission_skip_v1",
            "geometry_frame_ids": [],
            "gated_texture_only_frame_ids": [],
            "passed_risky_geometry_frame_ids": [],
            "frame_reasons": {},
        },
        "texture_frame_count": 0,
        "texture_frames": [],
        "texture_view_counts": {},
        "texture_pose_counts": {},
        "texture_only_added_unique_frame_count": 0,
        "texture_only_probe_source_frame_ids": [],
        "texture_only_probe_added_frame_count": 0,
        "texture_only_probe_added_frames": [],
        "texture_only_rejected_ablation_frame_count": int(
            admission_audit.get("rejected_ablation_frame_count", 0)
        ),
        "texture_only_rejected_ablation_frames": [],
        "texture_only_prior_rejection_hash_count": 0,
        "texture_only_prior_rejection_source_frame_count": 0,
        "texture_only_ablation_blocklist": None,
        "texture_only_ablation_blocklist_entry_count": 0,
        "texture_source_admission_audit": str(admission_audit["report_path"]),
        "texture_source_admission_audit_passed": False,
        "texture_source_admission_rejection_reasons": list(
            admission_audit.get("rejection_reasons", [])
        ),
        "texture_source_admission_unique_accepted_frame_count": int(
            admission_audit.get("unique_accepted_frame_count", 0)
        ),
        "texture_source_admission_protected_face_frame_count": int(
            admission_audit.get("protected_face_source_patch_frame_count", 0)
        ),
        "texture_source_admission_rejected_ablation_frame_count": int(
            admission_audit.get("rejected_ablation_frame_count", 0)
        ),
        "texture_source_admission_source_patch_blocked_ablation_frame_count": int(
            admission_audit.get("source_patch_blocked_ablation_frame_count", 0)
        ),
        "texture_only_promotion_guard_passed": False,
        "texture_only_guarded_result_success": False,
        "texture_only_bake_report": None,
        "texture_only_promotion_guard": str(project_guard_path),
        "success": False,
        "solid": False,
        "metrics": _report_metric_subset(metrics),
        "artifacts": summary["artifacts"],
    }


def _texture_source_admission_metrics(admission_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_texture_source_admission_audit_present": 1.0,
        "raw_texture_source_admission_audit_passed": float(
            bool(admission_audit.get("passed", False))
        ),
        "raw_texture_source_admission_rejection_count": int(
            len(admission_audit.get("rejection_reasons", []))
            if isinstance(admission_audit.get("rejection_reasons"), list)
            else 0
        ),
        "raw_texture_source_admission_unique_accepted_frame_count": int(
            admission_audit.get("unique_accepted_frame_count", 0)
        ),
        "raw_texture_source_admission_protected_face_frame_count": int(
            admission_audit.get("protected_face_source_patch_frame_count", 0)
        ),
        "raw_texture_source_admission_rejected_ablation_frame_count": int(
            admission_audit.get("rejected_ablation_frame_count", 0)
        ),
        "raw_texture_source_admission_source_patch_blocked_ablation_frame_count": int(
            admission_audit.get("source_patch_blocked_ablation_frame_count", 0)
        ),
        "raw_texture_source_preflight_accepted_frame_count": int(
            admission_audit.get("accepted_preflight_frame_count", 0)
        ),
        "raw_texture_source_preflight_duplicate_frame_count": int(
            admission_audit.get("duplicate_frame_count", 0)
        ),
        "raw_texture_source_preflight_rejected_frame_count": int(
            admission_audit.get("rejected_low_detail_frame_count", 0)
        ),
        "raw_texture_source_preflight_top_score": float(
            admission_audit.get("top_score", 0.0)
        ),
    }


def _texture_source_probe_frame_ids_from_bake_report(
    bake_report_path: Path,
    *,
    rejected_texture_source_hashes: tuple[str, ...],
    rejected_texture_source_frame_ids: tuple[str, ...],
    max_probe_count: int,
    allow_source_id_reprobe_when_hash_differs: bool = False,
) -> tuple[str, ...]:
    search = _texture_source_probe_search_from_bake_report(
        bake_report_path,
        rejected_texture_source_hashes=rejected_texture_source_hashes,
        rejected_texture_source_frame_ids=rejected_texture_source_frame_ids,
        max_probe_count=max_probe_count,
        allow_source_id_reprobe_when_hash_differs=(
            allow_source_id_reprobe_when_hash_differs
        ),
    )
    return tuple(search["probe_source_frame_ids"])


def _texture_source_probe_search_from_bake_report(
    bake_report_path: Path,
    *,
    rejected_texture_source_hashes: tuple[str, ...],
    rejected_texture_source_frame_ids: tuple[str, ...],
    max_probe_count: int,
    allow_source_id_reprobe_when_hash_differs: bool = False,
) -> dict[str, object]:
    if max_probe_count <= 0 or not bake_report_path.is_file():
        return {"probe_source_frame_ids": (), "skipped_probe_frames": []}
    try:
        bake_report = json.loads(bake_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"probe_source_frame_ids": (), "skipped_probe_frames": []}
    preflight_path_raw = bake_report.get("texture_source_preflight_path")
    if not preflight_path_raw:
        return {"probe_source_frame_ids": (), "skipped_probe_frames": []}
    preflight_path = Path(str(preflight_path_raw))
    if not preflight_path.is_file():
        return {"probe_source_frame_ids": (), "skipped_probe_frames": []}
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"probe_source_frame_ids": (), "skipped_probe_frames": []}
    rejected_hashes = {str(item) for item in rejected_texture_source_hashes if item}
    rejected_source_frame_ids = {
        str(item) for item in rejected_texture_source_frame_ids if item
    }
    probe_ids: list[str] = []
    skipped_probe_frames: list[dict[str, object]] = []
    ranked_frames = preflight.get("ranked_frames", [])
    if not isinstance(ranked_frames, list):
        return {"probe_source_frame_ids": (), "skipped_probe_frames": []}
    for item in ranked_frames:
        if not isinstance(item, dict):
            continue
        if bool(item.get("duplicate", False)):
            continue
        source_frame_id = str(item.get("source_frame_id", ""))
        digest = str(item.get("sha256", ""))
        if not source_frame_id:
            continue
        matched_rejected_hash = digest in rejected_hashes
        matched_rejected_source_frame_id = source_frame_id in rejected_source_frame_ids
        accepted_for_bake = bool(item.get("accepted_for_bake", False))
        source_id_reprobe_candidate = bool(
            allow_source_id_reprobe_when_hash_differs
            and matched_rejected_source_frame_id
            and not matched_rejected_hash
        )
        if matched_rejected_hash or (
            matched_rejected_source_frame_id and not source_id_reprobe_candidate
        ):
            skipped_probe_frames.append(
                {
                    "source_frame_id": source_frame_id,
                    "sha256": digest,
                    "preflight_rank": item.get("rank"),
                    "preflight_score": item.get("rank_score"),
                    "reason": "prior_ablation_rejection",
                    "matched_rejected_sha256": matched_rejected_hash,
                    "matched_rejected_source_frame_id": matched_rejected_source_frame_id,
                }
            )
            continue
        quality_gate = item.get("quality_gate", {})
        severe_reject_reasons = (
            quality_gate.get("severe_reject_reasons", [])
            if isinstance(quality_gate, dict)
            else []
        )
        severe_reject_reasons = (
            [str(reason) for reason in severe_reject_reasons]
            if isinstance(severe_reject_reasons, list)
            else []
        )
        if severe_reject_reasons:
            skipped_probe_frames.append(
                {
                    "source_frame_id": source_frame_id,
                    "sha256": digest,
                    "preflight_rank": item.get("rank"),
                    "preflight_score": item.get("rank_score"),
                    "reason": "severe_quality_reject",
                    "severe_reject_reasons": sorted(severe_reject_reasons),
                }
            )
            continue
        if accepted_for_bake and not source_id_reprobe_candidate:
            continue
        probe_ids.append(source_frame_id)
        if len(probe_ids) >= max_probe_count:
            break
    return {
        "probe_source_frame_ids": tuple(probe_ids),
        "skipped_probe_frames": skipped_probe_frames,
    }


def _blocklist_report_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    entries = _load_texture_source_ablation_blocklist(path)
    return {
        "path": str(path),
        "entry_count": len(entries),
        "sha256_count": len({item["sha256"] for item in entries if item.get("sha256")}),
        "source_frame_id_count": len(
            {item["source_frame_id"] for item in entries if item.get("source_frame_id")}
        ),
        "exists": Path(path).is_file(),
    }


def _load_texture_source_ablation_blocklist(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return []
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    entries = report.get("rejections", [])
    if not isinstance(entries, list):
        return []
    return _merged_texture_source_ablation_rejections(entries)


def _write_texture_source_ablation_blocklist(
    path: Path,
    entries: list[dict[str, Any]],
    *,
    source_paths: tuple[Path, ...],
) -> Path:
    path = Path(path)
    merged = _merged_texture_source_ablation_rejections(entries)
    hashes = sorted({item["sha256"] for item in merged if item.get("sha256")})
    source_frame_ids = sorted(
        {item["source_frame_id"] for item in merged if item.get("source_frame_id")}
    )
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.texture_source_ablation_blocklist",
        "version": TEXTURE_SOURCE_ABLATION_BLOCKLIST_VERSION,
        "path": str(path),
        "entry_count": len(merged),
        "sha256_count": len(hashes),
        "source_frame_id_count": len(source_frame_ids),
        "sha256": hashes,
        "source_frame_ids": source_frame_ids,
        "source_paths": [str(item) for item in source_paths],
        "rejections": merged,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _supplemental_texture_source_rejections_from_guard(
    guard_path: Path,
) -> list[dict[str, Any]]:
    guard_path = Path(guard_path)
    if not guard_path.is_file():
        return []
    try:
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rejections = guard.get("supplemental_source_rejections", [])
    if not isinstance(rejections, list):
        return []
    return _merged_texture_source_ablation_rejections(rejections)


def _merged_texture_source_ablation_rejections(
    *groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        for item in group:
            normalised = _normalise_texture_source_ablation_rejection(item)
            if normalised is None:
                continue
            key = (
                str(normalised.get("sha256", "")),
                str(normalised.get("source_frame_id", "")),
            )
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = normalised
                continue
            existing_metrics = set(existing.get("regressed_metric_names", []))
            incoming_metrics = set(normalised.get("regressed_metric_names", []))
            existing["regressed_metric_names"] = sorted(existing_metrics | incoming_metrics)
    return sorted(
        by_key.values(),
        key=lambda item: (
            str(item.get("sha256", "")),
            str(item.get("source_frame_id", "")),
        ),
    )


def _normalise_texture_source_ablation_rejection(
    item: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(item, dict) or not bool(item.get("rejected", False)):
        return None
    if item.get("reason") != "strict_face_detail_metric_regression":
        return None
    digest = str(item.get("sha256", ""))
    source_frame_id = str(item.get("source_frame_id", ""))
    if not digest and not source_frame_id:
        return None
    metric_names = item.get("regressed_metric_names", [])
    if not isinstance(metric_names, list):
        metric_names = []
    return {
        "frame_id": str(item.get("frame_id", "")),
        "source_frame_id": source_frame_id,
        "sha256": digest,
        "media_type": str(item.get("media_type", "unknown")),
        "preflight_rank": item.get("preflight_rank"),
        "preflight_score": item.get("preflight_score"),
        "reason": "strict_face_detail_metric_regression",
        "regressed_metric_names": sorted(str(metric) for metric in metric_names),
        "rejected": True,
    }


def _texture_source_ablation_rejection_keys(
    entries: list[dict[str, Any]],
) -> dict[str, list[str]]:
    return {
        "sha256": sorted({str(item.get("sha256", "")) for item in entries if item.get("sha256")}),
        "source_frame_id": sorted(
            {
                str(item.get("source_frame_id", ""))
                for item in entries
                if item.get("source_frame_id")
            }
        ),
    }


def _texture_source_report_for_baseline_bake(
    pipeline_reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    reports = _texture_source_reports_for_baseline_bake(pipeline_reports, max_reports=1)
    return reports[0] if reports else None


def _ordered_texture_source_reports(
    pipeline_reports: list[dict[str, Any]],
    *,
    uploaded_texture_source_report: dict[str, Any] | None,
    max_reports: int,
) -> list[dict[str, Any]]:
    if max_reports <= 0:
        return []
    reports: list[dict[str, Any]] = []
    seen_project_dirs: set[str] = set()
    if uploaded_texture_source_report is not None and _texture_source_project_available(
        Path(str(uploaded_texture_source_report.get("project_dir", "")))
    ):
        reports.append(uploaded_texture_source_report)
        seen_project_dirs.add(
            str(Path(str(uploaded_texture_source_report["project_dir"])).resolve())
        )
    for report in _texture_source_reports_for_baseline_bake(
        pipeline_reports,
        max_reports=max_reports,
    ):
        project_dir = str(Path(str(report["project_dir"])).resolve())
        if project_dir in seen_project_dirs:
            continue
        reports.append(report)
        seen_project_dirs.add(project_dir)
        if len(reports) >= max_reports:
            break
    return reports


def _texture_source_reports_for_baseline_bake(
    pipeline_reports: list[dict[str, Any]],
    *,
    max_reports: int,
) -> list[dict[str, Any]]:
    if max_reports <= 0:
        return []
    candidates = [
        report
        for report in pipeline_reports
        if _texture_source_project_available(Path(str(report.get("project_dir", ""))))
    ]
    if not candidates:
        return []
    ranked = sorted(candidates, key=_texture_source_report_score, reverse=True)
    unique: list[dict[str, Any]] = []
    seen_project_dirs: set[str] = set()
    for report in ranked:
        project_dir = str(Path(str(report["project_dir"])).resolve())
        if project_dir in seen_project_dirs:
            continue
        unique.append(report)
        seen_project_dirs.add(project_dir)
        if len(unique) >= max_reports:
            break
    return unique


def _texture_source_project_available(project_dir: Path) -> bool:
    return (
        project_dir.exists()
        and (project_dir / "observations" / "raw" / "texture_manifest.json").is_file()
    )


def _texture_source_report_score(report: dict[str, Any]) -> tuple[float, ...]:
    metrics = report.get("metrics", {})
    return (
        float(bool(report.get("success", False))),
        float(bool(report.get("solid", False))),
        _quality_fraction(metrics),
        float(metrics.get("raw_facial_feature_painted_texel_count", 0.0)) / 100_000.0,
        float(metrics.get("raw_texture_video_frame_count", 0.0)) / 10.0,
        float(metrics.get("raw_face_priority_texture_used", 0.0)),
        float(metrics.get("overlay_mean_silhouette_iou", 0.0)),
    )


def _quality_fraction(metrics: dict[str, Any]) -> float:
    total = float(metrics.get("raw_quality_total_check_count", 0.0))
    if total <= 0.0:
        return 0.0
    return float(metrics.get("raw_quality_passed_check_count", 0.0)) / total


def _report_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    names = (
        "raw_mean_mask_area_ratio",
        "raw_mean_bbox_area_ratio",
        "raw_full_frame_bbox_fraction",
        "raw_texture_unknown_luma",
        "raw_texture_foreground_fallback_luma",
        "raw_texture_unknown_luma_fraction_of_foreground",
        "overlay_mean_silhouette_iou",
        "silhouette_residual_mean_union_error",
        "joint_projection_mean_keypoint_loss",
        "keypoint_loss_reduction",
        "silhouette_loss_reduction",
        "raw_quality_passed_check_count",
        "raw_quality_total_check_count",
        "raw_selected_frame_count",
        "raw_texture_frame_count",
        "raw_selected_video_frame_count",
        "raw_texture_video_frame_count",
        "raw_face_priority_texture_used",
        "raw_facial_feature_count",
        "raw_facial_feature_painted_texel_count",
        "raw_baseline_geometry_texture_bake_used",
        "raw_texture_only_added_unique_frame_count",
        "raw_texture_only_skipped_duplicate_frame_count",
        "raw_texture_only_rejected_low_detail_frame_count",
        "raw_texture_only_rejected_ablation_frame_count",
        "raw_texture_only_probe_frame_count",
        "raw_texture_only_probe_added_frame_count",
        "raw_texture_only_promotion_guard_passed",
        "raw_texture_only_face_projection_protection_used",
        "raw_texture_only_face_projection_protected_frame_count",
        "raw_texture_only_face_projection_protected_region_count",
        "raw_texture_only_face_source_patch_allowed_frame_count",
        "raw_texture_source_preflight_accepted_frame_count",
        "raw_texture_source_preflight_duplicate_frame_count",
        "raw_texture_source_preflight_rejected_frame_count",
        "raw_texture_source_preflight_top_score",
        "raw_texture_source_admission_audit_present",
        "raw_texture_source_admission_audit_passed",
        "raw_texture_source_admission_rejection_count",
        "raw_texture_source_admission_unique_accepted_frame_count",
        "raw_texture_source_admission_protected_face_frame_count",
        "raw_texture_source_admission_rejected_ablation_frame_count",
        "raw_texture_source_admission_source_patch_blocked_ablation_frame_count",
        "observed_face_fraction",
        "observed_vertex_fraction",
        "basecolor_observed_texel_fraction",
        "raw_glb_face_feature_eye_detail_luma_std",
        "raw_glb_face_feature_iris_to_pupil_rgb_distance",
        "vertex_count",
        "face_count",
    )
    return {key: metrics[key] for key in names if key in metrics}


def _select_frames(
    manifest: dict[str, Any],
    *,
    min_selected_frames: int,
    max_selected_frames: int,
) -> list[FrameSelection]:
    return _select_frames_from_candidates(
        _frame_selection_candidates(manifest),
        min_selected_frames=min_selected_frames,
        max_selected_frames=max_selected_frames,
    )


def _frame_selection_candidates(manifest: dict[str, Any]) -> list[FrameSelection]:
    frames = []
    for frame in manifest["frames"]:
        quality = frame["quality"]
        source_path = Path(str(frame["raw_source_path"]))
        mask_area_ratio = float(quality["mask_area_ratio"])
        bbox_area_ratio = float(quality["bbox_area_ratio"])
        full_frame_bbox = bool(float(quality["full_frame_bbox"]))
        geometry_quality = _geometry_quality(frame, quality)
        view_pose_quality = _raw_view_pose_quality(frame, quality)
        score = _frame_score(
            mask_area_ratio=mask_area_ratio,
            bbox_area_ratio=bbox_area_ratio,
            full_frame_bbox=full_frame_bbox,
        )
        frames.append(
            FrameSelection(
                frame_id=str(frame["frame_id"]),
                source_path=source_path,
                media_type=str(frame.get("media_type", "photo")),
                source_type=str(frame.get("source_type", "image")),
                source_media_id=_optional_str(frame.get("source_media_id")),
                source_frame_index=_optional_int(frame.get("source_frame_index")),
                geometry_reject_reasons=geometry_quality["reject_reasons"],
                geometry_body_fit_score=float(geometry_quality["body_fit_score"]),
                score=score,
                mask_area_ratio=mask_area_ratio,
                bbox_area_ratio=bbox_area_ratio,
                full_frame_bbox=full_frame_bbox,
                raw_view_label=view_pose_quality.view_label,
                raw_view_group=view_pose_quality.view_group,
                raw_pose_label=view_pose_quality.pose_label,
                raw_frontal_face_confidence=view_pose_quality.frontal_face_confidence,
                raw_view_weight=view_pose_quality.view_weight,
            )
        )
    return frames


def _select_frames_from_candidates(
    frames: list[FrameSelection],
    *,
    min_selected_frames: int,
    max_selected_frames: int,
) -> list[FrameSelection]:
    filtered = [
        frame
        for frame in frames
        if frame.score > 0.0
        and not frame.full_frame_bbox
        and 0.002 <= frame.mask_area_ratio <= 0.55
        and 0.002 <= frame.bbox_area_ratio <= 0.75
    ]
    selected = filtered if len(filtered) >= min_selected_frames else frames
    selected = sorted(selected, key=lambda item: item.score, reverse=True)
    if len(selected) > max_selected_frames:
        return selected[:max_selected_frames]
    return selected


def _candidate_texture_frames(candidate: SegmentationCandidate) -> tuple[FrameSelection, ...]:
    return candidate.texture_frames or candidate.selected_frames


def _texture_source_frames(
    frames: tuple[FrameSelection, ...],
    *,
    selected_frames: tuple[FrameSelection, ...],
    max_texture_frames: int,
) -> tuple[FrameSelection, ...]:
    selected_ids = {frame.frame_id for frame in selected_frames}
    texture_frames = list(selected_frames)
    for frame in _rank_texture_source_reserve_frames(frames):
        if len(texture_frames) >= max_texture_frames:
            break
        if frame.frame_id in selected_ids:
            continue
        texture_frames.append(frame)
        selected_ids.add(frame.frame_id)
    return tuple(texture_frames)


def _rank_texture_source_reserve_frames(
    frames: tuple[FrameSelection, ...],
) -> tuple[FrameSelection, ...]:
    candidates = [
        frame
        for frame in frames
        if not frame.full_frame_bbox
        and frame.score > 0.0
        and frame.bbox_area_ratio <= 0.80
        and frame.mask_area_ratio >= 0.035
        and (
            frame.raw_view_group == "frontal_face"
            or frame.raw_frontal_face_confidence >= 0.50
            or frame.raw_view_weight >= 0.65
        )
        and not _has_hard_texture_source_reject(frame.geometry_reject_reasons)
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda frame: (
                -float(frame.raw_frontal_face_confidence),
                -float(frame.raw_view_weight),
                -float(frame.score),
                frame.media_type != "photo",
                frame.frame_id,
            ),
        )
    )


def _has_hard_texture_source_reject(reject_reasons: tuple[str, ...]) -> bool:
    hard_reasons = {"mask_too_small", "bbox_too_small", "full_frame_bbox"}
    return any(reason in hard_reasons for reason in reject_reasons)


def _geometry_quality(
    frame: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    image_width = max(float(frame.get("image_width", 1.0)), 1.0)
    image_height = max(float(frame.get("image_height", 1.0)), 1.0)
    bbox = frame.get("bbox_xyxy", [0.0, 0.0, image_width, image_height])
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{frame.get('frame_id', 'frame')} bbox_xyxy must contain four values")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    bbox_width_fraction = max(0.0, x1 - x0) / image_width
    bbox_height_fraction = max(0.0, y1 - y0) / image_height
    bbox_area_ratio = float(
        quality.get("bbox_area_ratio", bbox_width_fraction * bbox_height_fraction)
    )
    mask_area_ratio = float(quality.get("mask_area_ratio", 0.0))
    touches_border = float(quality.get("touches_image_border", 0.0))
    full_frame_bbox = float(quality.get("full_frame_bbox", 0.0))
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

    body_fit_score = _body_fit_score(
        mask_area_ratio=mask_area_ratio,
        bbox_area_ratio=bbox_area_ratio,
        bbox_width_fraction=bbox_width_fraction,
        bbox_height_fraction=bbox_height_fraction,
        aspect=aspect,
        center_x=(x0 + x1) / (2.0 * image_width),
        center_y=(y0 + y1) / (2.0 * image_height),
        touches_border=touches_border,
    )
    return {
        "reject_reasons": tuple(reject_reasons),
        "body_fit_score": body_fit_score,
    }


def _raw_view_pose_quality(
    frame: dict[str, Any],
    quality: dict[str, Any],
) -> RawViewPoseQuality:
    image_width = max(float(frame.get("image_width", 1.0)), 1.0)
    image_height = max(float(frame.get("image_height", 1.0)), 1.0)
    bbox = frame.get("bbox_xyxy", [0.0, 0.0, image_width, image_height])
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{frame.get('frame_id', 'frame')} bbox_xyxy must contain four values")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    bbox_width_fraction = max(0.0, x1 - x0) / image_width
    bbox_height_fraction = max(0.0, y1 - y0) / image_height
    bbox_area_ratio = float(
        quality.get("bbox_area_ratio", bbox_width_fraction * bbox_height_fraction)
    )
    mask_area_ratio = float(quality.get("mask_area_ratio", 0.0))
    touches_border = float(quality.get("touches_image_border", 0.0))
    aspect = bbox_width_fraction / max(bbox_height_fraction, 1e-6)
    fill_score = _clamp01(mask_area_ratio / max(bbox_area_ratio, 1e-6))
    center_x = (x0 + x1) / (2.0 * image_width)
    center_y = (y0 + y1) / (2.0 * image_height)
    center_score = _clamp01(1.0 - math.hypot(center_x - 0.5, center_y - 0.5) / 0.55)
    square_score = _clamp01(1.0 - abs(math.log(max(aspect, 1e-6))) / math.log(2.5))
    area_score = _clamp01(
        1.0 - abs(math.log(max(bbox_area_ratio, 1e-6) / 0.24)) / math.log(4.0)
    )
    border_score = 1.0 - _clamp01(touches_border)
    frontal_face_confidence = _clamp01(
        (0.42 * square_score + 0.24 * center_score + 0.20 * fill_score + 0.14 * area_score)
        * (0.75 + 0.25 * border_score)
    )
    view_weight = _clamp01(0.35 + 0.65 * frontal_face_confidence)
    if frontal_face_confidence >= 0.72:
        view_group = "frontal_face"
        view_label = "frontal_face_candidate"
    elif aspect >= 1.55:
        view_group = "side_body"
        view_label = "side_body_candidate"
    elif aspect <= 0.75:
        view_group = "vertical_body"
        view_label = "vertical_body_candidate"
    else:
        view_group = "oblique_body"
        view_label = "oblique_body_candidate"
    if bbox_area_ratio < 0.16 and frontal_face_confidence < 0.52:
        view_group = "distant_body"
        view_label = "distant_body_candidate"

    if frontal_face_confidence >= 0.72 and bbox_area_ratio >= 0.28:
        pose_label = "closeup_face"
    elif aspect >= 1.65 and bbox_height_fraction <= 0.72:
        pose_label = "side_standing_body"
    elif aspect >= 1.35:
        pose_label = "side_body"
    elif aspect <= 0.75 and bbox_height_fraction >= 0.72:
        pose_label = "vertical_body"
    elif bbox_area_ratio < 0.16:
        pose_label = "distant_body"
    else:
        pose_label = "oblique_body"

    if frame.get("media_type") == "video" or frame.get("source_type") == "video_frame":
        view_label = f"video_{view_label}"
        pose_label = f"video_{pose_label}"
    return RawViewPoseQuality(
        view_label=view_label,
        view_group=view_group,
        pose_label=pose_label,
        frontal_face_confidence=round(float(frontal_face_confidence), 9),
        view_weight=round(float(view_weight), 9),
    )


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


def _stage_selected_candidate_frames(
    selected_frames: tuple[FrameSelection, ...],
    *,
    project_dir: Path,
    force: bool,
) -> list[Path]:
    selected_dir = project_dir / "raw" / "selected_inputs"
    if selected_dir.exists() and force:
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)

    staged_paths: list[Path] = []
    media_frames: dict[str, dict[str, object]] = {}
    for index, frame in enumerate(selected_frames):
        suffix = frame.source_path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            suffix = ".jpg"
        target = selected_dir / f"frame_{index:04d}{suffix}"
        if target.exists() and not force:
            raise FileExistsError(f"{target} already exists; pass --force")
        shutil.copyfile(frame.source_path, target)

        metadata: dict[str, object] = {
            "media_type": frame.media_type,
            "source_type": frame.source_type,
            "source_media_id": frame.source_media_id or frame.source_path.name,
            "source_path": str(frame.source_path),
            "source_frame_id": frame.frame_id,
        }
        if frame.source_frame_index is not None:
            metadata["source_frame_index"] = int(frame.source_frame_index)
        metadata["raw_view_label"] = frame.raw_view_label
        metadata["raw_view_group"] = frame.raw_view_group
        metadata["raw_pose_label"] = frame.raw_pose_label
        metadata["raw_frontal_face_confidence"] = frame.raw_frontal_face_confidence
        metadata["raw_view_weight"] = frame.raw_view_weight
        media_frames[target.name] = metadata
        staged_paths.append(target)

    (selected_dir / "media_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.improve_raw_job_selected_candidate_staging",
                "frames": media_frames,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return staged_paths


def _staged_frame_ids(frame_count: int) -> tuple[str, ...]:
    return tuple(f"frame_{index:04d}" for index in range(frame_count))


def _geometry_variants(
    selected_frames: tuple[FrameSelection, ...],
    *,
    min_frames: int,
    max_variants: int,
) -> tuple[GeometryVariant, ...]:
    min_frames = max(1, min(min_frames, len(selected_frames)))
    usable_frames = _usable_geometry_frames(selected_frames)
    candidates = (
        GeometryVariant(
            "pose_gated_bodyfit",
            _pose_gated_geometry_frames(usable_frames, min_frames=min_frames),
            require_min_frames=False,
        ),
        GeometryVariant(
            "view_balanced_bodyfit",
            _view_balanced_geometry_frames(usable_frames, min_frames=min_frames),
        ),
        GeometryVariant(
            "media_balanced_bodyfit",
            _media_balanced_geometry_frames(usable_frames, min_frames=min_frames),
        ),
        GeometryVariant(
            "media_balanced_score",
            _media_balanced_score_frames(usable_frames, min_frames=min_frames),
        ),
        GeometryVariant(
            "bodyfit_only",
            _rank_geometry_frames(usable_frames or selected_frames)[:min_frames],
        ),
        GeometryVariant(
            "score_only",
            _rank_score_frames(usable_frames or selected_frames)[:min_frames],
        ),
    )
    variants: list[GeometryVariant] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        frame_ids = tuple(frame.frame_id for frame in candidate.geometry_frames)
        minimum_required = min_frames if candidate.require_min_frames else min(2, min_frames)
        if len(frame_ids) < minimum_required or frame_ids in seen:
            continue
        seen.add(frame_ids)
        variants.append(candidate)
        if len(variants) >= max_variants:
            break
    if variants:
        return tuple(variants)
    return (
        GeometryVariant(
            "fallback_score",
            _rank_score_frames(selected_frames)[:min_frames],
        ),
    )


def _geometry_frame_sweep_variants(
    selected_frames: tuple[FrameSelection, ...],
    *,
    min_selected_frames: int,
    max_selected_frames: int,
    max_variants: int,
) -> tuple[GeometryVariant, ...]:
    if not selected_frames:
        return ()
    max_variants = max(1, int(max_variants))
    usable_frames = _usable_geometry_frames(selected_frames)
    max_target = max(1, min(max_selected_frames, len(usable_frames or selected_frames)))
    min_target = max(1, min(min_selected_frames, max_target))
    target_candidates = (min_target, 4, 6, 8, max_target)
    targets: list[int] = []
    for target in target_candidates:
        target = max(min_target, min(int(target), max_target))
        if target not in targets:
            targets.append(target)
    variants: list[GeometryVariant] = []
    seen: set[tuple[str, ...]] = set()
    for target in targets:
        for variant in _geometry_variants(
            selected_frames,
            min_frames=target,
            max_variants=1,
        ):
            frame_ids = tuple(frame.frame_id for frame in variant.geometry_frames)
            if not frame_ids or frame_ids in seen:
                continue
            seen.add(frame_ids)
            variants.append(
                GeometryVariant(
                    f"frames_{len(frame_ids):02d}_{variant.variant_id}",
                    variant.geometry_frames,
                    require_min_frames=variant.require_min_frames,
                )
            )
            break
        if len(variants) >= max_variants:
            return tuple(variants)
    for variant in _geometry_variants(
        selected_frames,
        min_frames=max_target,
        max_variants=max_variants,
    ):
        frame_ids = tuple(frame.frame_id for frame in variant.geometry_frames)
        if not frame_ids or frame_ids in seen:
            continue
        seen.add(frame_ids)
        variants.append(variant)
        if len(variants) >= max_variants:
            break
    return tuple(variants)


def _geometry_variant_frame_target(
    selected_frames: tuple[FrameSelection, ...],
    *,
    min_selected_frames: int,
    max_selected_frames: int | None,
) -> int:
    if not selected_frames:
        return 1
    minimum = max(1, min(min_selected_frames, len(selected_frames)))
    if max_selected_frames is None:
        return minimum
    usable_count = len(_usable_geometry_frames(selected_frames))
    target = max(minimum, min(max_selected_frames, usable_count or len(selected_frames)))
    return max(1, min(target, len(selected_frames)))


def _geometry_safe_frames(
    selected_frames: tuple[FrameSelection, ...],
    *,
    min_frames: int,
) -> tuple[FrameSelection, ...]:
    usable_frames = _usable_geometry_frames(selected_frames)
    return _media_balanced_geometry_frames(
        usable_frames or selected_frames,
        min_frames=min_frames,
    )


def _usable_geometry_frames(
    selected_frames: tuple[FrameSelection, ...],
) -> tuple[FrameSelection, ...]:
    return tuple(
        frame
        for frame in selected_frames
        if _is_geometry_fit_frame(frame)
    )


def _media_balanced_geometry_frames(
    selected_frames: tuple[FrameSelection, ...],
    *,
    min_frames: int,
) -> tuple[FrameSelection, ...]:
    min_frames = max(1, min(min_frames, len(selected_frames)))
    ranked = _rank_geometry_frames(selected_frames)
    if _has_mixed_media(ranked) and len(ranked) >= min_frames:
        balanced = _balanced_geometry_frames(ranked, min_frames=min_frames)
        if len(balanced) >= min_frames:
            return balanced
    if len(ranked) >= min_frames:
        return ranked[:min_frames]
    return tuple(
        sorted(
            selected_frames,
            key=lambda frame: (
                _has_severe_geometry_reject(frame.geometry_reject_reasons),
                len(frame.geometry_reject_reasons),
                -frame.geometry_body_fit_score,
                -frame.score,
                frame.frame_id,
            ),
        )[:min_frames]
    )


def _pose_gated_geometry_frames(
    selected_frames: tuple[FrameSelection, ...],
    *,
    min_frames: int,
) -> tuple[FrameSelection, ...]:
    min_frames = max(1, min(min_frames, len(selected_frames)))
    ranked = _rank_geometry_frames(selected_frames)
    gated = tuple(frame for frame in ranked if _pose_geometry_gate_passes(frame))
    if len(gated) >= min_frames:
        return gated[:min_frames]
    return gated


def _view_balanced_geometry_frames(
    selected_frames: tuple[FrameSelection, ...],
    *,
    min_frames: int,
) -> tuple[FrameSelection, ...]:
    min_frames = max(1, min(min_frames, len(selected_frames)))
    ranked = _rank_geometry_frames(selected_frames)
    if len(ranked) < min_frames:
        return ranked
    buckets = ("frontal_face", "side_body", "oblique_body", "vertical_body", "distant_body")
    selected: list[FrameSelection] = []
    selected_ids: set[str] = set()
    for bucket in buckets:
        match = next(
            (
                frame
                for frame in ranked
                if frame.raw_view_group == bucket and frame.frame_id not in selected_ids
            ),
            None,
        )
        if match is None:
            continue
        selected.append(match)
        selected_ids.add(match.frame_id)
        if len(selected) >= min_frames:
            break
    for frame in ranked:
        if len(selected) >= min_frames:
            break
        if frame.frame_id in selected_ids:
            continue
        selected.append(frame)
        selected_ids.add(frame.frame_id)
    return tuple(selected)


def _media_balanced_score_frames(
    selected_frames: tuple[FrameSelection, ...],
    *,
    min_frames: int,
) -> tuple[FrameSelection, ...]:
    min_frames = max(1, min(min_frames, len(selected_frames)))
    ranked = _rank_score_frames(selected_frames)
    if _has_mixed_media(ranked) and len(ranked) >= min_frames:
        balanced = _balanced_geometry_frames(ranked, min_frames=min_frames)
        if len(balanced) >= min_frames:
            return balanced
    return ranked[:min_frames]


def _rank_geometry_frames(
    frames: tuple[FrameSelection, ...],
) -> tuple[FrameSelection, ...]:
    return tuple(
        sorted(
            frames,
            key=lambda frame: (
                -frame.geometry_body_fit_score,
                len(frame.geometry_reject_reasons),
                -frame.score,
                frame.frame_id,
            ),
        )
    )


def _rank_score_frames(
    frames: tuple[FrameSelection, ...],
) -> tuple[FrameSelection, ...]:
    return tuple(
        sorted(
            frames,
            key=lambda frame: (
                _has_severe_geometry_reject(frame.geometry_reject_reasons),
                len(frame.geometry_reject_reasons),
                -frame.score,
                -frame.geometry_body_fit_score,
                frame.frame_id,
            ),
        )
    )


def _has_mixed_media(frames: tuple[FrameSelection, ...]) -> bool:
    media_types = {frame.media_type for frame in frames}
    return {"photo", "video"}.issubset(media_types)


def _balanced_geometry_frames(
    ranked_frames: tuple[FrameSelection, ...],
    *,
    min_frames: int,
) -> tuple[FrameSelection, ...]:
    photos = tuple(frame for frame in ranked_frames if frame.media_type == "photo")
    videos = tuple(frame for frame in ranked_frames if frame.media_type == "video")
    photo_quota = min(len(photos), max(1, min_frames // 2))
    video_quota = min(len(videos), max(1, min_frames - photo_quota))
    if photo_quota + video_quota < min_frames:
        photo_quota = min(len(photos), min_frames - video_quota)
    selected = list(photos[:photo_quota]) + list(videos[:video_quota])
    selected_ids = {frame.frame_id for frame in selected}
    for frame in ranked_frames:
        if len(selected) >= min_frames:
            break
        if frame.frame_id in selected_ids:
            continue
        selected.append(frame)
        selected_ids.add(frame.frame_id)
    return tuple(selected)


def _has_severe_geometry_reject(reject_reasons: tuple[str, ...]) -> bool:
    severe_reasons = {"mask_too_small", "bbox_too_small", "bbox_too_large", "full_frame_bbox"}
    return any(reason in severe_reasons for reason in reject_reasons)


def _has_texture_only_geometry_reject(reject_reasons: tuple[str, ...]) -> bool:
    texture_only_reasons = {"border_partial_body", "sparse_geometry_mask"}
    return any(reason in texture_only_reasons for reason in reject_reasons)


def _is_geometry_fit_frame(frame: FrameSelection) -> bool:
    return (
        not _has_severe_geometry_reject(frame.geometry_reject_reasons)
        and not _has_texture_only_geometry_reject(frame.geometry_reject_reasons)
        and not frame.raw_pose_label.endswith("closeup_face")
    )


def _pose_geometry_gate_passes(frame: FrameSelection) -> bool:
    if (
        _has_severe_geometry_reject(frame.geometry_reject_reasons)
        or _has_texture_only_geometry_reject(frame.geometry_reject_reasons)
    ):
        return False
    view_group = frame.raw_view_group
    pose_label = frame.raw_pose_label
    risky_vertical = view_group in {"vertical_body", "distant_body"} or pose_label.endswith(
        "vertical_body"
    )
    risky_closeup = pose_label.endswith("closeup_face")
    if not risky_vertical and not risky_closeup:
        return True
    if risky_closeup:
        return False
    return (
        frame.geometry_body_fit_score >= 0.52
        and frame.score >= 0.70
        and frame.raw_view_weight >= 0.62
    )


def _pose_gate_report(
    selected_frames: tuple[FrameSelection, ...],
    geometry_frames: tuple[FrameSelection, ...],
) -> dict[str, Any]:
    geometry_ids = {frame.frame_id for frame in geometry_frames}
    risky_frames = [
        frame
        for frame in selected_frames
        if _pose_gate_risk_label(frame) != "geometry_safe"
    ]
    return {
        "version": "pose_geometry_gate_v1",
        "geometry_frame_ids": [frame.frame_id for frame in geometry_frames],
        "gated_texture_only_frame_ids": [
            frame.frame_id
            for frame in risky_frames
            if frame.frame_id not in geometry_ids
        ],
        "passed_risky_geometry_frame_ids": [
            frame.frame_id
            for frame in risky_frames
            if frame.frame_id in geometry_ids and _pose_geometry_gate_passes(frame)
        ],
        "frame_reasons": {
            frame.frame_id: _pose_gate_risk_label(frame)
            for frame in selected_frames
        },
    }


def _pose_gate_risk_label(frame: FrameSelection) -> str:
    if _has_severe_geometry_reject(frame.geometry_reject_reasons):
        return "severe_geometry_reject"
    if _has_texture_only_geometry_reject(frame.geometry_reject_reasons):
        return "partial_or_sparse_geometry_texture_only"
    if frame.raw_pose_label.endswith("closeup_face"):
        return "closeup_face_texture_only"
    if frame.raw_view_group in {"vertical_body", "distant_body"} or frame.raw_pose_label.endswith(
        "vertical_body"
    ):
        return "vertical_or_distant_body_requires_bodyfit_prior"
    return "geometry_safe"


def _view_group_counts(frames: tuple[FrameSelection, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        label = frame.raw_view_group or "unknown"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _pose_label_counts(frames: tuple[FrameSelection, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        label = frame.raw_pose_label or "unknown_pose"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _media_count(frames: tuple[FrameSelection, ...], media_type: str) -> int:
    return sum(1 for frame in frames if frame.media_type == media_type)


def _staged_frame_ids_for_frames(
    selected_frames: tuple[FrameSelection, ...],
    target_frames: tuple[FrameSelection, ...],
) -> tuple[str, ...]:
    target_ids = {frame.frame_id for frame in target_frames}
    return tuple(
        f"frame_{index:04d}"
        for index, frame in enumerate(selected_frames)
        if frame.frame_id in target_ids
    )


def _frame_score(
    *,
    mask_area_ratio: float,
    bbox_area_ratio: float,
    full_frame_bbox: bool,
) -> float:
    if mask_area_ratio <= 0.0 or bbox_area_ratio <= 0.0:
        return -10.0
    target_bbox = 0.28
    target_mask = 0.16
    return (
        1.0
        - abs(bbox_area_ratio - target_bbox) * 1.6
        - abs(mask_area_ratio - target_mask) * 0.9
        - max(0.0, bbox_area_ratio - 0.65) * 3.0
        - float(full_frame_bbox) * 3.0
    )


def _segmentation_score(
    frames: list[FrameSelection],
    *,
    mean_bbox_area_ratio: float,
    full_frame_bbox_fraction: float,
) -> float:
    return (
        sum(frame.score for frame in frames) / max(len(frames), 1)
        + min(len(frames), 10) * 0.03
        - mean_bbox_area_ratio * 0.7
        - full_frame_bbox_fraction * 1.2
    )


def _pipeline_score(report: dict[str, Any]) -> float:
    metrics = report["metrics"]
    return (
        float(report["success"]) * 2.0
        + float(report["solid"]) * 3.0
        + float(metrics["overlay_mean_silhouette_iou"]) * 4.0
        - float(metrics["silhouette_residual_mean_union_error"]) * 1.5
        - float(metrics["raw_full_frame_bbox_fraction"]) * 1.5
        - max(0.0, float(metrics["raw_mean_bbox_area_ratio"]) - 0.55)
    )


def _pipeline_report_by_candidate_id(
    pipeline_reports: list[dict[str, Any]],
    candidate_id: str,
) -> dict[str, Any] | None:
    for report in pipeline_reports:
        if str(report.get("candidate_id")) == candidate_id:
            return report
    return None


def _merge_supervisor_results(
    pipeline_reports: list[dict[str, Any]],
    supervisor_report: dict[str, Any],
) -> None:
    reports_by_id = {str(report.get("candidate_id")): report for report in pipeline_reports}
    candidates = supervisor_report.get("candidates", [])
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        report = reports_by_id.get(str(candidate.get("label")))
        if report is None:
            continue
        report["supervisor_score"] = float(candidate.get("score", 0.0))
        report["supervisor_score_terms"] = dict(candidate.get("score_terms", {}))
        promotion_guard = candidate.get("promotion_guard")
        if isinstance(promotion_guard, dict):
            report["promotion_guard"] = dict(promotion_guard)
        alignment_gate = candidate.get("alignment_gate")
        if isinstance(alignment_gate, dict):
            report["alignment_gate"] = dict(alignment_gate)
        body_fit_alignment_gate = candidate.get("body_fit_alignment_gate")
        if isinstance(body_fit_alignment_gate, dict):
            report["body_fit_alignment_gate"] = dict(body_fit_alignment_gate)
        supervisor_metrics = candidate.get("metrics", {})
        if isinstance(supervisor_metrics, dict):
            report["metrics"].update(supervisor_metrics)


def _segmentation_candidate_json(candidate: SegmentationCandidate) -> dict[str, Any]:
    selected_ids = {item.frame_id for item in candidate.selected_frames}
    return {
        "candidate_id": candidate.candidate_id,
        "backend": candidate.backend,
        "color_delta_threshold": candidate.color_delta_threshold,
        "project_dir": str(candidate.project_dir),
        "frame_count": candidate.frame_count,
        "selected_frame_count": candidate.selected_frame_count,
        "texture_source_frame_count": len(_candidate_texture_frames(candidate)),
        "mean_mask_area_ratio": candidate.mean_mask_area_ratio,
        "mean_bbox_area_ratio": candidate.mean_bbox_area_ratio,
        "full_frame_bbox_fraction": candidate.full_frame_bbox_fraction,
        "score": candidate.score,
        "selected_frames": [
            _frame_selection_json(frame)
            for frame in candidate.selected_frames
        ],
        "texture_source_frames": [
            {
                **_frame_selection_json(frame),
                "reserve_only": frame.frame_id not in selected_ids,
            }
            for frame in _candidate_texture_frames(candidate)
        ],
    }


def _frame_selection_json(frame: FrameSelection) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "source_path": str(frame.source_path),
        "media_type": frame.media_type,
        "source_type": frame.source_type,
        "source_media_id": frame.source_media_id,
        "source_frame_index": frame.source_frame_index,
        "geometry_reject_reasons": list(frame.geometry_reject_reasons),
        "geometry_body_fit_score": frame.geometry_body_fit_score,
        "score": frame.score,
        "mask_area_ratio": frame.mask_area_ratio,
        "bbox_area_ratio": frame.bbox_area_ratio,
        "full_frame_bbox": frame.full_frame_bbox,
        "raw_view_label": frame.raw_view_label,
        "raw_view_group": frame.raw_view_group,
        "raw_pose_label": frame.raw_pose_label,
        "raw_frontal_face_confidence": frame.raw_frontal_face_confidence,
        "raw_view_weight": frame.raw_view_weight,
    }


def _candidate_id(
    *,
    upload_dir: Path,
    backend: RawSegmentationBackend,
    color_delta_threshold: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(upload_dir.resolve()).encode("utf-8"))
    digest.update(b"\0")
    digest.update(backend.encode("utf-8"))
    digest.update(b"\0")
    digest.update(f"{color_delta_threshold:.6f}".encode())
    return digest.hexdigest()[:12]


def _safe_candidate_token(value: str, *, max_length: int = 80) -> str:
    token = "".join(char if char.isalnum() else "_" for char in value).strip("_")
    token = "_".join(part for part in token.split("_") if part)
    return (token[:max_length] if token else "probe")


def _scoped_project_name(project_id_prefix: str, base_name: str) -> str:
    prefix = _safe_candidate_token(project_id_prefix, max_length=48)
    if prefix == "probe":
        return base_name
    if base_name.startswith(f"{prefix}_"):
        return base_name
    return f"{prefix}_{base_name}"


def _latest_upload_dir(root: Path) -> Path:
    candidates = [path for path in root.glob("job_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no upload jobs found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _write_disk_preflight_report(
    loop_dir: Path,
    *,
    projects_dir: Path,
    min_free_gib: float,
    cleanup_intermediates: bool,
    force: bool,
) -> dict[str, Any]:
    if min_free_gib < 0.0:
        raise ValueError("min_free_gib must be non-negative")
    loop_dir = Path(loop_dir)
    projects_dir = Path(projects_dir)
    report_path = loop_dir / "disk_preflight.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")
    cleanup_report = (
        _cleanup_loop_intermediates(loop_dir) if cleanup_intermediates else _empty_cleanup_report()
    )
    usage = shutil.disk_usage(loop_dir)
    min_free_bytes = int(min_free_gib * BYTES_PER_GIB)
    passed = int(usage.free) >= min_free_bytes
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.improve_raw_job.disk_preflight",
        "version": DISK_PREFLIGHT_VERSION,
        "loop_dir": str(loop_dir),
        "projects_dir": str(projects_dir),
        "report_path": str(report_path),
        "min_free_gib": float(min_free_gib),
        "min_free_bytes": int(min_free_bytes),
        "free_bytes": int(usage.free),
        "free_gib": float(int(usage.free) / BYTES_PER_GIB),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "cleanup_intermediates_requested": bool(cleanup_intermediates),
        "cleanup": cleanup_report,
        "passed": bool(passed),
        "recommendation": (
            "continue"
            if passed
            else (
                "free disk space or rerun with --cleanup-intermediates after "
                "reviewing generated loop artifacts"
            )
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _cleanup_loop_intermediates(loop_dir: Path) -> dict[str, Any]:
    loop_dir = Path(loop_dir)
    removed: list[dict[str, Any]] = []
    for child in sorted(loop_dir.iterdir() if loop_dir.exists() else []):
        if not child.is_dir():
            continue
        if not (child.name == "staged_upload_frames" or child.name.startswith("seg_")):
            continue
        size_bytes = _path_size_bytes(child)
        shutil.rmtree(child)
        removed.append(
            {
                "path": str(child),
                "size_bytes": int(size_bytes),
            }
        )
    return {
        "removed_path_count": len(removed),
        "removed_bytes": int(sum(item["size_bytes"] for item in removed)),
        "removed_paths": removed,
    }


def _empty_cleanup_report() -> dict[str, Any]:
    return {
        "removed_path_count": 0,
        "removed_bytes": 0,
        "removed_paths": [],
    }


def _path_size_bytes(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return int(total)


def _raw_frame_paths(
    upload_dir: Path,
    *,
    stage_dir: Path | None = None,
    extra_media_paths: tuple[Path, ...] = (),
    min_video_frames: int = 2,
    max_video_frames: int = 24,
    force: bool = False,
) -> list[Path]:
    return _stage_upload_media_frames(
        upload_dir,
        stage_dir=stage_dir,
        extra_media_paths=extra_media_paths,
        min_video_frames=min_video_frames,
        max_video_frames=max_video_frames,
        force=force,
    ).frame_paths


@dataclass(frozen=True)
class UploadMediaStagingResult:
    frame_paths: list[Path]
    report_path: Path | None
    input_photo_count: int
    input_video_count: int
    extracted_video_frame_count: int


def _stage_upload_media_frames(
    upload_dir: Path,
    *,
    stage_dir: Path | None,
    extra_media_paths: tuple[Path, ...] = (),
    min_video_frames: int = 2,
    max_video_frames: int = 24,
    force: bool = False,
) -> UploadMediaStagingResult:
    media_paths = _raw_media_paths(upload_dir, extra_media_paths=extra_media_paths)
    if not media_paths:
        raise ValueError(f"{upload_dir} must contain image or video media")
    media_metadata = _load_stage_media_metadata(tuple(media_paths))
    has_video = any(path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS for path in media_paths)
    if not has_video and stage_dir is None:
        frame_paths = [
            path
            for path in media_paths
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if len(frame_paths) < 2:
            raise ValueError(f"{upload_dir} must contain at least two image frames")
        return UploadMediaStagingResult(
            frame_paths=frame_paths,
            report_path=None,
            input_photo_count=len(frame_paths),
            input_video_count=0,
            extracted_video_frame_count=0,
        )
    if stage_dir is None:
        stage_dir = upload_dir / ".opencat3d_staged_frames"
    stage_dir = Path(stage_dir)
    if stage_dir.exists() and force:
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    frames: dict[str, dict[str, object]] = {}
    input_photo_count = 0
    input_video_source_ids: set[str] = set()
    extracted_video_frame_count = 0
    for media_index, media_path in enumerate(media_paths):
        suffix = media_path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            target = stage_dir / f"frame_{len(frame_paths):04d}{suffix}"
            if target.exists() and not force:
                raise FileExistsError(f"{target} already exists; pass --force")
            shutil.copyfile(media_path, target)
            metadata = dict(media_metadata.get(media_path.resolve(), {}))
            media_type = str(metadata.get("media_type", "photo"))
            source_type = str(
                metadata.get(
                    "source_type",
                    "video_frame" if media_type == "video" else "image",
                )
            )
            source_media_id = str(metadata.get("source_media_id") or media_path.name)
            frame_metadata: dict[str, object] = dict(metadata)
            frame_metadata.update(
                {
                "media_type": media_type,
                "source_type": source_type,
                "source_media_id": source_media_id,
                    "source_path": str(
                        metadata.get("source_path")
                        or metadata.get("source_video_path")
                        or media_path
                    ),
                }
            )
            if metadata.get("source_frame_index") is not None:
                frame_metadata["source_frame_index"] = int(metadata["source_frame_index"])
            frames[target.name] = frame_metadata
            frame_paths.append(target)
            if media_type == "video" or source_type == "video_frame":
                input_video_source_ids.add(source_media_id)
                extracted_video_frame_count += 1
            else:
                input_photo_count += 1
            continue
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        input_video_source_ids.add(str(media_path.resolve()))
        extraction = extract_video_frames(
            media_path,
            stage_dir / f"video_{media_index:04d}_frames",
            min_frames=min_video_frames,
            max_frames=max_video_frames,
            force=force,
        )
        for extracted_path, source_frame_index in zip(
            extraction.frame_paths,
            extraction.selected_indices,
            strict=True,
        ):
            target = stage_dir / f"frame_{len(frame_paths):04d}.jpg"
            if target.exists() and not force:
                raise FileExistsError(f"{target} already exists; pass --force")
            shutil.copyfile(extracted_path, target)
            frames[target.name] = {
                "media_type": "video",
                "source_type": "video_frame",
                "source_media_id": media_path.name,
                "source_path": str(media_path),
                "source_frame_index": int(source_frame_index),
            }
            frame_paths.append(target)
            extracted_video_frame_count += 1
    if len(frame_paths) < 2:
        raise ValueError(f"{upload_dir} must yield at least two staged frames")
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.improve_raw_job_upload_media_staging",
        "upload_dir": str(upload_dir),
        "extra_media_paths": [str(path) for path in extra_media_paths],
        "stage_dir": str(stage_dir),
        "input_photo_count": input_photo_count,
        "input_video_count": len(input_video_source_ids),
        "extracted_video_frame_count": extracted_video_frame_count,
        "staged_frame_count": len(frame_paths),
        "frames": frames,
    }
    report_path = stage_dir / "media_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return UploadMediaStagingResult(
        frame_paths=frame_paths,
        report_path=report_path,
        input_photo_count=input_photo_count,
        input_video_count=len(input_video_source_ids),
        extracted_video_frame_count=extracted_video_frame_count,
    )


def _raw_media_paths(
    upload_dir: Path,
    *,
    extra_media_paths: tuple[Path, ...] = (),
) -> list[Path]:
    supported = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
    sources: list[Path] = sorted(
        path
        for path in Path(upload_dir).iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )
    for extra_path in extra_media_paths:
        path = Path(extra_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        if path.is_dir():
            sources.extend(
                sorted(
                    child
                    for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() in supported
                )
            )
            continue
        if path.suffix.lower() not in supported:
            raise ValueError(f"unsupported extra media extension: {path.suffix}")
        sources.append(path)
    unique: dict[str, Path] = {}
    for path in sources:
        unique.setdefault(_media_content_key(path), path)
    return list(unique.values())


def _media_content_key(path: Path) -> str:
    return f"{Path(path).suffix.lower()}:{_file_sha256(path)}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stage_media_metadata(media_paths: tuple[Path, ...]) -> dict[Path, dict[str, Any]]:
    metadata: dict[Path, dict[str, Any]] = {}
    manifest_dirs = {path.parent.resolve() for path in media_paths}
    for manifest_dir in sorted(manifest_dirs):
        manifest_path = manifest_dir / "media_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        frames = payload.get("frames", {})
        if not isinstance(frames, dict):
            continue
        for filename, frame_metadata in frames.items():
            if not isinstance(filename, str) or not isinstance(frame_metadata, dict):
                continue
            metadata[(manifest_dir / filename).resolve()] = dict(frame_metadata)
    return metadata


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
