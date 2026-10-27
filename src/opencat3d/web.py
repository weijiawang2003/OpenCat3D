"""Small local web interface for OpenCat3D uploads."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from catnet.active_learning import build_ambiguity_queue
from catnet.embedding.service import EmbeddingService
from catnet.feedback import add_feedback_constraint, closed_review_task_ids
from catnet.index_registry import IndexRegistry
from catnet.matcher.decision import OpenSetDecider, OpenSetDecisionConfig
from catnet.matcher.faiss_index import FaissIndex, SearchCandidate
from catnet.matcher.rerank import LocalTextureReranker
from catnet.reid.data import ReIDSample
from catnet.reid.model import CatReIDModel
from catnet.schema import (
    BranchType,
    CatIdentity,
    DecisionType,
    EmbeddingVector,
    FeedbackConstraintType,
    IdentityEventType,
    IdentityStatus,
    MatchEdge,
    MediaAsset,
    ModelVersion,
    Observation,
    stable_id,
)
from catnet.storage import CatNetStore
from opencat3d.evaluation.evidence_gap import (
    SourceFrameAcquisitionGapResult,
    write_source_frame_acquisition_gap_report,
)
from opencat3d.ingest.images import SUPPORTED_IMAGE_EXTENSIONS
from opencat3d.ingest.video import SUPPORTED_VIDEO_EXTENSIONS, extract_video_frames
from opencat3d.pipeline import RawPipelineResult, run_raw_pipeline
from opencat3d.preprocess.segmentation import RawSegmentationBackend

SUPPORTED_SCAN_EXTENSIONS = frozenset({".obj", ".ply", ".glb", ".gltf", ".zip"})
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    content: bytes


@dataclass(frozen=True)
class WebJobResult:
    job_id: str
    project_dir: Path
    upload_dir: Path
    result: RawPipelineResult


@dataclass(frozen=True)
class StagedReconstructionUploadResult:
    upload_dir: Path
    media_manifest_path: Path
    frame_paths: tuple[Path, ...]
    frame_count: int
    photo_count: int
    video_count: int
    has_video: bool


@dataclass(frozen=True)
class ReconstructionMediaPreflightResult:
    upload_dir: Path
    report_path: Path
    media_manifest_path: Path
    frame_paths: tuple[Path, ...]
    input_frame_count: int
    existing_frame_count: int
    staged_upload_frame_count: int
    unique_frame_count: int
    duplicate_frame_count: int
    duplicate_group_count: int
    new_unique_frame_count: int
    new_duplicate_frame_count: int
    existing_unique_frame_count: int
    existing_duplicate_frame_count: int
    photo_frame_count: int
    video_frame_count: int


@dataclass(frozen=True)
class ReconstructionPreflightJobResult:
    job_id: str
    upload_dir: Path
    loop_dir: Path
    report_path: Path
    media_preflight_report_path: Path
    promotion_gate_report_path: Path
    ready_for_reconstruction: bool
    ready_for_promotion: bool
    warnings: tuple[str, ...]
    source_frame_acquisition_gap_report_path: Path | None = None


@dataclass(frozen=True)
class ReconstructionPreflightContext:
    existing_upload_dirs: tuple[Path, ...] = ()
    segmentation_cache_report: Path | None = None
    baseline_project_dir: Path | None = None
    texture_probe_report: Path | None = None


@dataclass(frozen=True)
class IdentityObservationResult:
    filename: str
    obs_id: str
    cat_id: str
    decision: str
    confidence: float
    fused_score: float
    top_candidate_cat_id: str | None
    top_candidate_obs_id: str | None
    edge_id: str


@dataclass(frozen=True)
class IdentityJobResult:
    job_id: str
    upload_dir: Path
    store_dir: Path
    results: list[IdentityObservationResult]


def serve_web(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    work_dir: Path = Path("web_runs"),
) -> None:
    """Run the local OpenCat3D web UI until interrupted."""

    server = make_web_server(host=host, port=port, work_dir=work_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def make_web_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    work_dir: Path = Path("web_runs"),
) -> ThreadingHTTPServer:
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    class OpenCat3DHandler(_WebHandler):
        root_dir = work_dir

    return ThreadingHTTPServer((host, port), OpenCat3DHandler)


def run_uploaded_image_job(
    uploads: list[UploadedFile],
    *,
    work_dir: Path,
    color_delta_threshold: float = 0.10,
    camera_root_steps: int = 30,
    segmentation_backend: str = "auto_subject_v1",
) -> WebJobResult:
    reconstruction_uploads = _validate_reconstruction_uploads(uploads)
    has_video = any(
        Path(upload.filename).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        for upload in reconstruction_uploads
    )
    max_selected_frames = 12 if has_video else 5
    job_id = _job_id(
        reconstruction_uploads,
        color_delta_threshold=color_delta_threshold,
        camera_root_steps=camera_root_steps,
        segmentation_backend=segmentation_backend,
        max_selected_frames=max_selected_frames,
    )
    root = Path(work_dir).resolve()
    upload_dir = root / "uploads" / job_id
    project_dir = root / "projects" / job_id
    stage_result = stage_reconstruction_uploads(
        reconstruction_uploads,
        upload_dir=upload_dir,
        force=True,
    )

    result = run_raw_pipeline(
        project_dir,
        input_paths=[stage_result.upload_dir],
        color_delta_threshold=color_delta_threshold,
        segmentation_backend=cast(RawSegmentationBackend, segmentation_backend),
        camera_root_steps=camera_root_steps,
        max_selected_frames=max_selected_frames,
        force=True,
    )
    return WebJobResult(
        job_id=job_id,
        project_dir=project_dir,
        upload_dir=stage_result.upload_dir,
        result=result,
    )


def run_uploaded_media_preflight_job(
    uploads: list[UploadedFile],
    *,
    work_dir: Path,
    min_selected_frames: int = 2,
    max_selected_frames: int = 8,
    existing_upload_dirs: tuple[Path, ...] | list[Path] = (),
    segmentation_cache_report: Path | None = None,
    baseline_project_dir: Path | None = None,
    texture_probe_report: Path | None = None,
) -> ReconstructionPreflightJobResult:
    reconstruction_uploads = _validate_reconstruction_uploads(uploads)
    job_id = _preflight_job_id(
        reconstruction_uploads,
        min_selected_frames=min_selected_frames,
        max_selected_frames=max_selected_frames,
        existing_upload_dirs=existing_upload_dirs,
    )
    root = Path(work_dir).resolve()
    upload_dir = root / "preflight_uploads" / job_id
    loop_dir = root / "preflight_jobs" / job_id
    stage_result = preflight_reconstruction_media_batch(
        reconstruction_uploads,
        upload_dir=upload_dir,
        existing_upload_dirs=existing_upload_dirs,
        force=True,
    )
    loop_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "improve_raw_job.py"),
        "--upload-dir",
        str(stage_result.upload_dir),
        "--loop-dir",
        str(loop_dir),
        "--projects-dir",
        str(root / "projects"),
        "--project-id-prefix",
        job_id,
        "--min-selected-frames",
        str(max(2, min_selected_frames)),
        "--max-selected-frames",
        str(max(max_selected_frames, min_selected_frames)),
        "--preflight-only",
        "--force",
    ]
    if segmentation_cache_report is not None:
        command.extend(
            [
                "--segmentation-cache-report",
                str(Path(segmentation_cache_report).expanduser().resolve()),
            ]
        )
    completed = _run_improve_raw_job_preflight_command(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(
            "media preflight failed: "
            f"{completed.stderr.strip() or completed.stdout.strip() or completed.returncode}"
        )
    report_path = loop_dir / "raw_improvement_loop_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    media_preflight = cast("dict[str, object]", report.get("media_preflight", {}))
    promotion_gate = cast(
        "dict[str, object]",
        report.get("media_preflight_promotion_gate", {}),
    )
    source_gap_report_path = _write_preflight_source_frame_acquisition_gap(
        report_path,
        baseline_project_dir=baseline_project_dir,
        media_preflight_report_path=Path(str(media_preflight["report_path"])),
        media_batch_report_path=stage_result.report_path,
        texture_probe_report_path=texture_probe_report,
    )
    warnings = tuple(str(item) for item in media_preflight.get("warnings", []))
    return ReconstructionPreflightJobResult(
        job_id=job_id,
        upload_dir=stage_result.upload_dir,
        loop_dir=loop_dir,
        report_path=report_path,
        media_preflight_report_path=Path(str(media_preflight["report_path"])),
        promotion_gate_report_path=Path(str(promotion_gate["report_path"])),
        ready_for_reconstruction=bool(media_preflight.get("ready_for_reconstruction", False)),
        ready_for_promotion=bool(media_preflight.get("ready_for_promotion", False)),
        warnings=warnings,
        source_frame_acquisition_gap_report_path=source_gap_report_path,
    )


def _write_preflight_source_frame_acquisition_gap(
    loop_report_path: Path,
    *,
    baseline_project_dir: Path | None,
    media_preflight_report_path: Path,
    media_batch_report_path: Path,
    texture_probe_report_path: Path | None,
) -> Path | None:
    if baseline_project_dir is None:
        return None
    output_path = loop_report_path.parent / "source_frame_acquisition_gap.json"
    result = write_source_frame_acquisition_gap_report(
        baseline_project_dir,
        media_preflight_report_path=media_preflight_report_path,
        media_batch_report_path=media_batch_report_path,
        texture_probe_report_path=texture_probe_report_path,
        output_path=output_path,
        force=True,
    )
    _record_source_frame_acquisition_gap(loop_report_path, result)
    return result.report_path


def _record_source_frame_acquisition_gap(
    loop_report_path: Path,
    result: SourceFrameAcquisitionGapResult,
) -> None:
    report = json.loads(loop_report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        return
    report["source_frame_acquisition_gap"] = {
        "report_path": str(result.report_path),
        "success": result.success,
        "recommendation_count": result.recommendation_count,
        "checks": result.checks,
        "metrics": result.metrics,
    }
    artifacts = report.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    artifacts["source_frame_acquisition_gap"] = str(result.report_path)
    report["artifacts"] = artifacts
    loop_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_reconstruction_preflight_context(
    root_dir: Path,
    *,
    state_path: Path | None = None,
) -> ReconstructionPreflightContext:
    root = Path(root_dir).expanduser().resolve()
    state = _read_json_object(state_path or PROJECT_ROOT / "state" / "state.json")
    if state is None:
        return ReconstructionPreflightContext()

    baseline_project_dir = _state_project_dir(
        state.get("current_best_project"),
        root_dir=root,
    )
    segmentation_cache_report = _first_state_file(
        state,
        root_dir=root,
        keys=(
            "last_new_downloads_preflight_raw_report",
            "last_new_uploaded_media_preflight_raw_report",
            "last_real_media_preflight_loop_report",
        ),
    )
    existing_upload_dir = _preflight_upload_dir_for_report(
        segmentation_cache_report,
        root_dir=root,
    )
    texture_probe_report = _first_state_file(
        state,
        root_dir=root,
        keys=(
            "last_texture_only_probe_project_raw_report",
            "last_texture_only_probe_project_report",
        ),
    )
    return ReconstructionPreflightContext(
        existing_upload_dirs=(existing_upload_dir,) if existing_upload_dir else (),
        segmentation_cache_report=segmentation_cache_report,
        baseline_project_dir=baseline_project_dir,
        texture_probe_report=texture_probe_report,
    )


def _state_project_dir(value: object, *, root_dir: Path) -> Path | None:
    path = _state_path(value)
    if path is None:
        return None
    projects_dir = root_dir / "projects"
    summary_path = path / "outputs" / "quality" / "raw_pipeline_summary.json"
    if path.is_dir() and summary_path.is_file() and _is_relative_to(path, projects_dir):
        return path
    return None


def _first_state_file(
    state: dict[str, object],
    *,
    root_dir: Path,
    keys: tuple[str, ...],
) -> Path | None:
    for key in keys:
        path = _state_path(state.get(key))
        if path is None:
            continue
        if path.is_file() and _is_relative_to(path, root_dir):
            return path
    return None


def _state_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _preflight_upload_dir_for_report(
    report_path: Path | None,
    *,
    root_dir: Path,
) -> Path | None:
    if report_path is None:
        return None
    preflight_jobs_dir = root_dir / "preflight_jobs"
    try:
        relative = report_path.parent.relative_to(preflight_jobs_dir)
    except ValueError:
        return None
    if len(relative.parts) != 1:
        return None
    upload_dir = root_dir / "preflight_uploads" / relative.parts[0]
    if (upload_dir / "media_manifest.json").is_file():
        return upload_dir.resolve()
    return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _run_improve_raw_job_preflight_command(
    command: list[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def stage_reconstruction_uploads(
    uploads: list[UploadedFile],
    *,
    upload_dir: Path,
    force: bool = False,
) -> StagedReconstructionUploadResult:
    """Stage photos/video frames in the deterministic upload directory format."""

    reconstruction_uploads = _validate_reconstruction_uploads(uploads)
    upload_dir = Path(upload_dir).resolve()
    if upload_dir.exists() and force:
        _clear_staged_reconstruction_uploads(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    staged_frame_index = 0
    media_frames: dict[str, dict[str, object]] = {}
    staged_frame_paths: list[Path] = []
    photo_count = 0
    video_count = 0
    for upload_index, upload in enumerate(reconstruction_uploads):
        suffix = Path(upload.filename).suffix.lower()
        if suffix in SUPPORTED_VIDEO_EXTENSIONS:
            video_count += 1
            video_dir = upload_dir / f"video_{upload_index:04d}"
            video_dir.mkdir(parents=True, exist_ok=True)
            target = video_dir / f"source{suffix}"
            target.write_bytes(upload.content)
            extraction = extract_video_frames(
                target,
                upload_dir / f"video_{upload_index:04d}_frames",
                min_frames=2,
                max_frames=24,
                force=True,
            )
            selected_indices = getattr(extraction, "selected_indices", None)
            if selected_indices is None:
                selected_indices = tuple(range(len(extraction.frame_paths)))
            for frame_path, source_frame_index in zip(
                extraction.frame_paths,
                selected_indices,
                strict=True,
            ):
                staged_target = upload_dir / f"frame_{staged_frame_index:04d}.jpg"
                staged_target.write_bytes(frame_path.read_bytes())
                staged_frame_paths.append(staged_target)
                media_frames[staged_target.name] = {
                    "media_type": "video",
                    "source_type": "video_frame",
                    "source_media_id": upload.filename,
                    "source_frame_index": int(source_frame_index),
                    "source_video_path": str(target),
                }
                staged_frame_index += 1
            continue
        photo_count += 1
        target = upload_dir / f"frame_{staged_frame_index:04d}{suffix}"
        target.write_bytes(upload.content)
        staged_frame_paths.append(target)
        media_frames[target.name] = {
            "media_type": "photo",
            "source_type": "image",
            "source_media_id": upload.filename,
        }
        staged_frame_index += 1
    media_manifest_path = upload_dir / "media_manifest.json"
    media_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.web.upload_staging",
                "frames": media_frames,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return StagedReconstructionUploadResult(
        upload_dir=upload_dir,
        media_manifest_path=media_manifest_path,
        frame_paths=tuple(staged_frame_paths),
        frame_count=len(staged_frame_paths),
        photo_count=photo_count,
        video_count=video_count,
        has_video=video_count > 0,
    )


def preflight_reconstruction_media_batch(
    uploads: list[UploadedFile],
    *,
    upload_dir: Path,
    existing_upload_dirs: tuple[Path, ...] | list[Path] = (),
    force: bool = False,
) -> ReconstructionMediaPreflightResult:
    """Create a deduplicated reconstruction input folder from new and existing media."""

    reconstruction_uploads = _validate_reconstruction_uploads(uploads)
    upload_dir = Path(upload_dir).resolve()
    report_path = upload_dir / "media_preflight_report.json"
    media_manifest_path = upload_dir / "media_manifest.json"
    if upload_dir.exists() and force:
        _clear_media_preflight_dir(upload_dir)
    if (report_path.exists() or media_manifest_path.exists()) and not force:
        raise FileExistsError(f"{upload_dir} already contains a preflight; pass force=True")
    upload_dir.mkdir(parents=True, exist_ok=True)

    new_stage_dir = upload_dir / "_new_upload_staging"
    new_stage = stage_reconstruction_uploads(
        reconstruction_uploads,
        upload_dir=new_stage_dir,
        force=True,
    )
    existing_records: list[dict[str, object]] = []
    for existing_upload_dir in existing_upload_dirs:
        existing_records.extend(
            _staged_upload_frame_records(
                Path(existing_upload_dir),
                origin="existing_upload",
            )
        )
    new_records = _staged_upload_frame_records(
        new_stage.upload_dir,
        origin="new_upload",
    )
    records = existing_records + new_records
    if not records:
        raise ValueError("media preflight has no staged frames")

    digest_groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        digest_groups.setdefault(str(record["sha256"]), []).append(record)

    kept_records: list[dict[str, object]] = []
    duplicate_groups: list[dict[str, object]] = []
    duplicate_origin_counts = {"existing_upload": 0, "new_upload": 0}
    for digest, group in sorted(digest_groups.items()):
        ordered_group = sorted(
            group,
            key=lambda item: (
                0 if item["origin"] == "existing_upload" else 1,
                str(item["origin_upload_dir"]),
                str(item["source_frame_name"]),
            ),
        )
        kept = ordered_group[0]
        kept_records.append(kept)
        duplicate_records = ordered_group[1:]
        if duplicate_records:
            for record in duplicate_records:
                origin = str(record["origin"])
                duplicate_origin_counts[origin] = duplicate_origin_counts.get(origin, 0) + 1
            duplicate_groups.append(
                {
                    "sha256": digest,
                    "kept_origin": kept["origin"],
                    "kept_origin_upload_dir": kept["origin_upload_dir"],
                    "kept_source_frame_name": kept["source_frame_name"],
                    "kept_source_media_id": kept["metadata"].get("source_media_id", ""),
                    "duplicate_records": [
                        _preflight_duplicate_record_json(record)
                        for record in duplicate_records
                    ],
                }
            )

    kept_records.sort(
        key=lambda item: (
            0 if item["origin"] == "existing_upload" else 1,
            str(item["origin_upload_dir"]),
            str(item["source_frame_name"]),
        )
    )
    frame_paths: list[Path] = []
    media_frames: dict[str, dict[str, object]] = {}
    kept_report_frames: list[dict[str, object]] = []
    for output_index, record in enumerate(kept_records):
        source_path = Path(str(record["source_path"]))
        suffix = source_path.suffix.lower()
        target = upload_dir / f"frame_{output_index:04d}{suffix}"
        if target.resolve() != source_path.resolve():
            shutil.copyfile(source_path, target)
        metadata = dict(record["metadata"])
        metadata["origin"] = record["origin"]
        metadata["origin_upload_dir"] = record["origin_upload_dir"]
        metadata["origin_frame_name"] = record["source_frame_name"]
        metadata["sha256"] = record["sha256"]
        media_frames[target.name] = metadata
        frame_paths.append(target)
        kept_report_frames.append(
            {
                "frame_name": target.name,
                "source_path": str(target),
                "sha256": record["sha256"],
                "origin": record["origin"],
                "origin_upload_dir": record["origin_upload_dir"],
                "origin_frame_name": record["source_frame_name"],
                "media_type": metadata.get("media_type", "photo"),
                "source_media_id": metadata.get("source_media_id", ""),
            }
        )

    media_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.web.media_batch_preflight",
                "frames": media_frames,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    photo_frame_count = sum(
        1 for value in media_frames.values() if value.get("media_type") == "photo"
    )
    video_frame_count = sum(
        1 for value in media_frames.values() if value.get("media_type") == "video"
    )
    unique_origin_counts = {"existing_upload": 0, "new_upload": 0}
    for record in kept_records:
        origin = str(record["origin"])
        unique_origin_counts[origin] = unique_origin_counts.get(origin, 0) + 1
    report = {
        "schema_version": "1.0",
        "source": "opencat3d.web.media_batch_preflight",
        "upload_dir": str(upload_dir),
        "media_manifest_path": str(media_manifest_path),
        "new_upload_staging_dir": str(new_stage.upload_dir),
        "existing_upload_dirs": [
            str(Path(path).resolve()) for path in existing_upload_dirs
        ],
        "input_frame_count": len(records),
        "existing_frame_count": len(existing_records),
        "staged_upload_frame_count": len(new_records),
        "unique_frame_count": len(kept_records),
        "duplicate_frame_count": len(records) - len(kept_records),
        "duplicate_group_count": len(duplicate_groups),
        "new_unique_frame_count": unique_origin_counts.get("new_upload", 0),
        "new_duplicate_frame_count": duplicate_origin_counts.get("new_upload", 0),
        "existing_unique_frame_count": unique_origin_counts.get("existing_upload", 0),
        "existing_duplicate_frame_count": duplicate_origin_counts.get(
            "existing_upload",
            0,
        ),
        "photo_frame_count": photo_frame_count,
        "video_frame_count": video_frame_count,
        "kept_frames": kept_report_frames,
        "duplicate_groups": duplicate_groups,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ReconstructionMediaPreflightResult(
        upload_dir=upload_dir,
        report_path=report_path,
        media_manifest_path=media_manifest_path,
        frame_paths=tuple(frame_paths),
        input_frame_count=len(records),
        existing_frame_count=len(existing_records),
        staged_upload_frame_count=len(new_records),
        unique_frame_count=len(kept_records),
        duplicate_frame_count=len(records) - len(kept_records),
        duplicate_group_count=len(duplicate_groups),
        new_unique_frame_count=unique_origin_counts.get("new_upload", 0),
        new_duplicate_frame_count=duplicate_origin_counts.get("new_upload", 0),
        existing_unique_frame_count=unique_origin_counts.get("existing_upload", 0),
        existing_duplicate_frame_count=duplicate_origin_counts.get(
            "existing_upload",
            0,
        ),
        photo_frame_count=photo_frame_count,
        video_frame_count=video_frame_count,
    )


def run_identity_image_job(
    uploads: list[UploadedFile],
    *,
    work_dir: Path,
    captured_at: datetime | None = None,
    site_id: str | None = None,
    camera_id: str | None = None,
    uploader: str | None = None,
    model_path: Path | None = None,
) -> IdentityJobResult:
    image_uploads = _validate_identity_uploads(uploads)
    job_id = _identity_job_id(
        image_uploads,
        captured_at=captured_at,
        site_id=site_id,
        camera_id=camera_id,
        uploader=uploader,
        model_path=model_path,
    )
    root = Path(work_dir).resolve()
    upload_dir = root / "identity_uploads" / job_id
    store_dir = root / "catnet"
    upload_dir.mkdir(parents=True, exist_ok=True)

    store = CatNetStore(store_dir)
    service = EmbeddingService()
    trained_model = CatReIDModel.load(model_path) if model_path else None
    results: list[IdentityObservationResult] = []
    for index, upload in enumerate(image_uploads):
        suffix = Path(upload.filename).suffix.lower()
        target = upload_dir / f"observation_{index:04d}{suffix}"
        target.write_bytes(upload.content)

        sha256 = hashlib.sha256(upload.content).hexdigest()
        width, height = _image_size(upload.content)
        media = MediaAsset(
            media_id=stable_id("media", sha256),
            storage_uri=str(target),
            sha256=sha256,
            width=width,
            height=height,
            mime_type=_upload_mime_type(upload.filename),
            uploader=uploader,
            metadata={"original_filename": upload.filename},
        )
        obs = Observation(
            obs_id=stable_id("obs", media.media_id, job_id, index),
            observation_version=stable_id("obv", media.media_id, job_id, index),
            media_id=media.media_id,
            captured_at=captured_at,
            site_id=site_id,
            camera_id=camera_id,
            metadata={
                "identity_job_id": job_id,
                "upload_filename": upload.filename,
                **({"uploader": uploader} if uploader else {}),
            },
        )
        store.upsert_media(media)
        store.upsert_observation(obs)

        if trained_model is not None:
            embeddings = [
                _trained_embedding(
                    trained_model,
                    target,
                    observation=obs,
                    model_path=model_path,
                )
            ]
            model_version = _trained_model_version(trained_model, model_path)
            direct_prediction = trained_model.predict_embedding(
                np.asarray(embeddings[0].vector, dtype=np.float64),
                top_k=5,
            )
        else:
            embeddings = service.extract(upload.content, observation=obs, obs_id=obs.obs_id)
            model_version = service.model_version
            direct_prediction = None

        fused = next(
            embedding for embedding in embeddings if embedding.branch_type == BranchType.FUSED
        )
        existing_embeddings = store.list_embeddings(
            branch_type=BranchType.FUSED,
            model_version=fused.model_version,
        )
        existing_observations = store.observations_for_embeddings(existing_embeddings)
        indexer = _identity_indexer(
            store_dir,
            existing_embeddings=existing_embeddings,
            existing_observations=existing_observations,
            query_embedding=fused,
        )
        candidates = indexer.search(
            fused.vector or [],
            top_k=5,
            filters={"species": "cat"},
            exclude_obs_ids={obs.obs_id},
        )
        if candidates:
            rerank_result = LocalTextureReranker().rerank(
                query_image_path=target,
                candidates=candidates,
                candidate_image_paths=_candidate_image_paths(store, candidates),
                top_k=5,
            )
            candidates = rerank_result.candidates
        if (
            direct_prediction is not None
            and direct_prediction.is_known
            and direct_prediction.cat_id
        ):
            cat_id = direct_prediction.cat_id
            _ensure_identity(
                store,
                cat_id=cat_id,
                first_seen_at=obs.captured_at,
                reason="catnet trained model recognized identity",
            )
            decision_type = DecisionType.MATCH_EXISTING
            confidence = direct_prediction.confidence
            visual_score = direct_prediction.candidates[0][1]
            global_score = visual_score
            local_score = None
            temporal_score = 0.5
            metadata_score = None
            fused_score = visual_score
            top_candidate_cat_id = cat_id
            top_candidate_obs_id = None
            explanation = {
                "source": "trained_prototype",
                "threshold": direct_prediction.threshold,
                "candidates": direct_prediction.candidates,
            }
            decision_rule_version = "catnet-trained-prototype-v1"
            reason = "catnet trained prototype accepted existing identity"
        else:
            decider_config = None
            if trained_model is not None:
                threshold = trained_model.config.threshold
                decider_config = OpenSetDecisionConfig(
                    match_threshold=threshold,
                    review_threshold=max(-1.0, threshold - 0.08),
                    decision_rule_version="catnet-trained-local-index-rule-v1",
                )
            decider = OpenSetDecider(
                model_version=model_version,
                index_version=indexer.index_version.index_version,
                config=decider_config,
            )
            decision = decider.decide(
                candidates,
                query_observation=obs,
                observations_by_id=existing_observations,
            )
            if decision.decision == DecisionType.MATCH_EXISTING and decision.cat_id:
                cat_id = decision.cat_id
                reason = "catnet open-set accepted existing identity"
            else:
                identity = store.create_identity(
                    seed=f"{obs.obs_id}:{decision.decision.value}",
                    first_seen_at=obs.captured_at,
                    reason="catnet open-set provisional identity",
                )
                cat_id = identity.cat_id
                reason = f"catnet open-set {decision.decision.value}"
            decision_type = decision.decision
            confidence = decision.confidence
            visual_score = decision.visual_score
            global_score = _float_from_explanation(
                decision.explanation,
                "top_global_score",
                fallback=visual_score,
            )
            local_score = decision.local_score
            temporal_score = decision.temporal_score
            metadata_score = decision.metadata_score
            fused_score = decision.fused_score
            top_candidate_cat_id = decision.top_candidate_cat_id
            top_candidate_obs_id = decision.top_candidate_obs_id
            explanation = decision.explanation
            decision_rule_version = decision.decision_rule_version
        assigned = store.assign_observation(obs.obs_id, cat_id, reason=reason)
        for embedding in embeddings:
            store.upsert_embedding(embedding)

        edge = MatchEdge(
            edge_id=stable_id(
                "edge",
                assigned.obs_id,
                top_candidate_obs_id or "none",
                decision_type.value,
                f"{fused_score:.6f}",
            ),
            src_obs_id=assigned.obs_id,
            dst_obs_id=top_candidate_obs_id,
            src_cat_id=cat_id,
            dst_cat_id=top_candidate_cat_id,
            global_score=global_score,
            local_score=local_score,
            temporal_score=temporal_score,
            metadata_score=metadata_score,
            fused_score=fused_score,
            decision=decision_type,
            confidence=confidence,
            model_version=model_version.model_version,
            index_version=indexer.index_version.index_version,
            calibration_version=model_version.calibration_version,
            decision_rule_version=decision_rule_version,
            explanation=explanation,
        )
        store.add_match_edge(edge)
        results.append(
            IdentityObservationResult(
                filename=upload.filename,
                obs_id=assigned.obs_id,
                cat_id=cat_id,
                decision=decision_type.value,
                confidence=confidence,
                fused_score=fused_score,
                top_candidate_cat_id=top_candidate_cat_id,
                top_candidate_obs_id=top_candidate_obs_id,
                edge_id=edge.edge_id,
            )
        )
    store.save()
    _write_identity_job_summary(
        root,
        IdentityJobResult(job_id, upload_dir, store_dir, results),
    )
    return IdentityJobResult(
        job_id=job_id,
        upload_dir=upload_dir,
        store_dir=store_dir,
        results=results,
    )


def render_index(*, message: str | None = None) -> str:
    escaped_message = "" if message is None else html.escape(message)
    notice = (
        f'<div class="notice" role="status">{escaped_message}</div>'
        if escaped_message
        else ""
    )
    image_exts = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
    video_exts = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
    scan_exts = ", ".join(sorted(SUPPORTED_SCAN_EXTENSIONS))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenCat3D Upload</title>
  {STYLE}
</head>
<body>
  <main class="app-shell">
    <section class="toolbar">
      <div>
        <h1>OpenCat3D</h1>
        <p>Submit cat image sets to generate a fixed-topology 3D avatar.</p>
      </div>
      <a class="health" href="/healthz">Health</a>
    </section>
    {notice}
    <section class="workspace">
      <form
        class="panel primary"
        method="post"
        action="/jobs"
        enctype="multipart/form-data"
        data-upload-form
      >
        <div class="panel-header">
          <h2>Images</h2>
          <span>Enabled</span>
        </div>
        <label class="drop-zone">
          <input
            class="native-file-input"
            type="file"
            name="files"
            accept=".jpg,.jpeg,.png,.avi,.m4v,.mov,.mp4"
            multiple
            data-file-input
          >
          <strong>Add cat photos or video</strong>
          <small data-file-count>0 selected. Supported: {image_exts}, {video_exts}</small>
        </label>
        <div class="staged-files">
          <div class="staged-file-header">
            <strong>Selected media</strong>
            <button class="secondary-button" type="button" data-clear-files>Clear</button>
          </div>
          <ul class="file-list" data-file-list aria-live="polite">
            <li class="empty-file-row">No photos selected yet.</li>
          </ul>
        </div>
        <div class="controls">
          <label>
            Color threshold
            <input
              name="color_delta_threshold"
              type="number"
              value="0.10"
              min="0"
              max="1"
              step="0.01"
            >
          </label>
          <label>
            Root steps
            <input name="camera_root_steps" type="number" value="30" min="1" max="80" step="1">
          </label>
        </div>
        <p class="subtle">Use 30-50 root steps for quality checks. 2 is only a smoke test.</p>
        <div class="action-row">
          <button
            class="secondary-button"
            type="submit"
            formaction="/jobs/preflight"
            data-submit
            data-ready-label="Check Media"
            disabled
          >Add Photos Or Video</button>
          <button
            type="submit"
            data-submit
            data-ready-label="Generate 3D Model"
            disabled
          >Add Photos Or Video</button>
        </div>
      </form>
      <section class="panel">
        <div class="panel-header">
          <h2>Videos</h2>
          <span>Enabled</span>
        </div>
        <p>Video upload uses deterministic frame extraction before reconstruction.</p>
        <small>Supported formats: {video_exts}</small>
      </section>
      <section class="panel disabled" aria-disabled="true">
        <div class="panel-header">
          <h2>Scans</h2>
          <span>Adapter needed</span>
        </div>
        <p>Scan upload will require a registration adapter into the fixed CatTemplate topology.</p>
        <small>Planned formats: {scan_exts}</small>
      </section>
    </section>
    <section class="identity-workspace">
      <form
        class="panel primary"
        method="post"
        action="/identity/jobs"
        enctype="multipart/form-data"
      >
        <div class="panel-header">
          <h2>Cat Identity</h2>
          <span>Baseline</span>
        </div>
        <label class="drop-zone compact">
          <input
            class="native-file-input"
            type="file"
            name="files"
            accept=".jpg,.jpeg,.png"
            multiple
          >
          <strong>Add identity photos</strong>
          <small>Supported: {image_exts}</small>
        </label>
        <div class="controls">
          <label>
            Model path
            <input
              name="model_path"
              type="text"
              placeholder="catnet_artifacts/catnet_reid_model.json"
            >
          </label>
          <label>
            Captured at
            <input name="captured_at" type="datetime-local">
          </label>
          <label>
            Site
            <input name="site_id" type="text" placeholder="home-yard">
          </label>
          <label>
            Camera
            <input name="camera_id" type="text" placeholder="front-door">
          </label>
          <label>
            Uploader
            <input name="uploader" type="text" placeholder="local">
          </label>
        </div>
        <button type="submit">Identify Cat</button>
      </form>
    </section>
  </main>
  {STAGED_UPLOAD_SCRIPT}
</body>
</html>"""


def render_job(project_dir: Path, *, job_id: str) -> str:
    summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
    if not summary_path.exists():
        return render_index(message=f"Job {job_id} was not found.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = summary.get("artifacts", {})
    metrics = summary.get("metrics", {})
    topology = (
        f"{int(metrics['vertex_count'])} vertices / {int(metrics['face_count'])} faces"
        if "vertex_count" in metrics and "face_count" in metrics
        else "Topology unavailable"
    )
    links = [
        ("GLB", "glb"),
        ("OBJ", "obj"),
        ("PLY", "ply"),
        ("Summary", "evaluation_report"),
        ("Compliance", "compliance_audit"),
        ("Frame selection", "frame_selection_report"),
        ("Face turntable", "face_turntable"),
        ("Face front", "face_turntable_front"),
        ("Face close-up", "face_texture_closeup"),
        ("Face source comparison", "face_texture_source_comparison"),
        ("Face bake comparison", "face_texture_bake_comparison"),
        ("GLB face features", "glb_face_feature_visibility"),
        ("Face crops", "face_texture_crops"),
        ("Texture source preflight", "texture_source_preflight"),
        ("Acquisition gap", "source_frame_acquisition_gap"),
    ]
    artifact_links = "\n".join(
        _artifact_link(project_dir, job_id, label, str(artifacts.get(key, "")))
        for label, key in links
    )
    face_preview_panel = _face_texture_preview_panel(
        project_dir,
        job_id=job_id,
        artifacts=artifacts,
    )
    face_bake_panel = _face_texture_bake_panel(project_dir, job_id=job_id)
    template_region_panel = _template_region_v2_panel(project_dir, job_id=job_id)
    source_gap_panel = _source_frame_acquisition_gap_panel(
        project_dir,
        job_id=job_id,
        artifacts=artifacts,
    )
    acquisition_panel = _texture_source_acquisition_panel(
        project_dir,
        job_id=job_id,
        artifacts=artifacts,
    )
    metric_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(metrics.items())
        if key
        in {
            "raw_frame_count",
            "raw_input_frame_count",
            "raw_selected_frame_count",
            "raw_rejected_frame_count",
            "raw_input_photo_frame_count",
            "raw_input_video_frame_count",
            "raw_selected_photo_frame_count",
            "raw_selected_video_frame_count",
            "raw_texture_frame_count",
            "raw_texture_photo_frame_count",
            "raw_texture_video_frame_count",
            "raw_balanced_media_selection_used",
            "segmentation_mean_mask_area_ratio",
            "segmentation_mean_bbox_area_ratio",
            "segmentation_full_frame_bbox_fraction",
            "overlay_mean_silhouette_iou",
            "silhouette_residual_mean_union_error",
            "basecolor_observed_texel_fraction",
            "observed_face_fraction",
            "observed_vertex_fraction",
            "raw_texture_unknown_luma",
            "raw_texture_foreground_fallback_luma",
            "raw_texture_unknown_luma_fraction_of_foreground",
            "raw_texture_width",
            "raw_texture_height",
            "raw_texture_resolution_scale",
            "raw_face_texture_target_bake_used",
            "raw_face_texture_target_reported_size",
            "raw_template_region_v2_texture_target_bake_used",
            "raw_template_region_v2_texture_target_reported_size",
            "raw_template_region_v2_texture_target_reported_scale",
            "raw_template_region_v2_texture_target_reported_island_count",
            "raw_template_region_v2_texture_target_reported_facial_island_count",
            "raw_template_region_v2_texture_target_reported_requested_texel_count",
            "raw_template_region_v2_texture_target_reported_allocated_texel_count",
            "raw_pbr_texture_width",
            "raw_pbr_texture_height",
            "raw_glb_texture_size",
            "raw_glb_texture_dimensions_valid",
            "raw_glb_pbr_texture_dimensions_valid",
            "raw_glb_basecolor_texture_valid",
            "raw_glb_pbr_textures_valid",
            "raw_glb_embedded_texture_dimension_gate_valid",
            "raw_glb_face_feature_required_group_count",
            "raw_glb_face_feature_visible_group_count",
            "raw_glb_face_feature_required_feature_count",
            "raw_glb_face_feature_present_required_feature_count",
            "raw_glb_face_feature_visible_feature_count",
            "raw_glb_face_feature_sampled_texel_count",
            "raw_glb_face_feature_min_visible_texel_count",
            "raw_glb_face_feature_mean_rgb_error",
            "raw_glb_face_feature_max_rgb_error",
            "raw_glb_face_feature_eye_visible",
            "raw_glb_face_feature_nose_visible",
            "raw_glb_face_feature_muzzle_visible",
            "raw_glb_face_feature_iris_to_pupil_rgb_distance",
            "raw_glb_face_feature_eye_detail_luma_std",
            "raw_glb_face_feature_nose_to_muzzle_rgb_distance",
            "raw_glb_face_feature_mouth_to_muzzle_rgb_distance",
            "raw_glb_face_feature_material_visibility_gate_valid",
            "raw_facial_feature_count",
            "raw_facial_feature_painted_texel_count",
            "raw_facial_feature_observed_texel_count",
            "raw_facial_feature_observed_texel_fraction",
            "raw_facial_feature_source_patch_count",
            "raw_facial_feature_source_patch_feedback_sample_count",
            "raw_facial_feature_source_patch_feedback_used_count",
            "raw_facial_feature_source_patch_bake_candidate_count",
            "raw_facial_feature_source_patch_bake_selected_feedback",
            "raw_facial_feature_source_patch_bake_visual_quality_passed_count",
            "raw_facial_feature_source_patch_bake_visual_quality_min_score",
            "raw_facial_feature_source_patch_bake_visual_quality_mean_score",
            "raw_facial_feature_iris_halo_feature_count",
            "raw_facial_feature_iris_halo_uv_support_texel_count",
            "raw_facial_feature_iris_halo_core_texel_count",
            "raw_facial_feature_iris_halo_edge_texel_count",
            "raw_facial_feature_iris_halo_accepted_core_texel_count",
            "raw_facial_feature_iris_halo_edge_fraction",
            "raw_facial_feature_iris_halo_edge_to_core_ratio",
            "raw_facial_feature_iris_halo_max_edge_to_core_ratio",
            "raw_facial_feature_iris_halo_support_to_accepted_core_ratio",
            "raw_facial_feature_left_iris_halo_edge_to_core_ratio",
            "raw_facial_feature_left_iris_halo_support_to_accepted_core_ratio",
            "raw_facial_feature_right_iris_halo_edge_to_core_ratio",
            "raw_facial_feature_right_iris_halo_support_to_accepted_core_ratio",
            "raw_facial_feature_source_patch_texel_count",
            "raw_facial_feature_source_patch_texel_fraction",
            "raw_facial_feature_source_patch_quality_passed_count",
            "raw_facial_feature_source_patch_quality_total_count",
            "raw_facial_feature_source_patch_quality_fraction",
            "raw_facial_feature_source_patch_min_quality_score",
            "raw_facial_feature_source_patch_mean_quality_score",
            "raw_facial_feature_source_patch_fusion_min_weight",
            "raw_facial_feature_source_patch_fusion_mean_weight",
            "raw_facial_feature_source_patch_fusion_weighted_texel_count",
            "raw_facial_feature_source_patch_nominal_texel_count",
            "raw_facial_feature_source_patch_footprint_reduction_fraction",
            "raw_facial_feature_source_patch_mean_footprint_area_scale",
            "raw_facial_feature_source_patch_min_footprint_area_scale",
            "raw_facial_feature_source_patch_shape_mask_texel_count",
            "raw_facial_feature_source_patch_shape_mask_fraction",
            "raw_facial_feature_source_patch_shape_alpha_mean",
            "raw_facial_feature_source_patch_source_mask_observed_texel_count",
            "raw_facial_feature_source_patch_source_mask_densified_texel_count",
            "raw_facial_feature_source_patch_source_mask_densification_gain",
            "raw_facial_feature_source_patch_mean_source_mask_densification_gain",
            "raw_facial_feature_source_patch_localization_candidate_texel_count",
            "raw_facial_feature_source_patch_localization_selected_texel_count",
            "raw_facial_feature_source_patch_localization_fraction",
            "raw_facial_feature_source_patch_mean_localization_fraction",
            "raw_facial_feature_source_patch_landmark_aligned_count",
            "raw_facial_feature_source_patch_mean_landmark_confidence",
            "raw_facial_feature_source_patch_observed_anchor_guided_count",
            "raw_facial_feature_source_patch_observed_anchor_guided_fraction",
            "raw_facial_feature_source_patch_landmark_destination_scaled_count",
            "raw_facial_feature_source_patch_mean_landmark_destination_scale",
            "raw_face_feature_destination_anchor_anatomical_count",
            "raw_face_feature_destination_anchor_order_valid",
            "raw_face_feature_destination_anchor_eye_to_nose_v_gap",
            "raw_face_feature_destination_anchor_nose_to_mouth_v_gap",
            "raw_face_feature_destination_anchor_nose_to_muzzle_v_gap",
            "raw_face_feature_destination_anchor_ear_to_eye_v_gap",
            "raw_face_feature_destination_anchor_eye_symmetry_u_error",
            "raw_facial_feature_source_patch_core_texel_count",
            "raw_facial_feature_source_patch_core_fraction",
            "raw_facial_feature_source_patch_feathered_edge_texel_count",
            "raw_facial_feature_source_patch_feathered_edge_fraction",
            "raw_facial_feature_source_patch_mean_blend_alpha",
            "raw_face_priority_texture_used",
            "raw_face_uv_region_count",
            "raw_face_uv_region_total_area",
            "raw_face_uv_region_min_density_weight",
            "raw_face_uv_region_contract_valid",
            "raw_template_region_v2_count",
            "raw_template_region_v2_required_region_count",
            "raw_template_region_v2_required_present_count",
            "raw_template_region_v2_contract_valid",
            "raw_glb_template_region_v2_contract_valid",
            "raw_template_region_v2_belly_vertex_count",
            "raw_face_uv_density_allocated_texel_count",
            "raw_face_uv_density_min_allocated_texels",
            "raw_face_uv_density_eye_min_allocated_texels",
            "raw_face_uv_density_nose_allocated_texels",
            "raw_face_uv_density_ear_inner_min_allocated_texels",
            "raw_template_region_v2_uv_region_count",
            "raw_template_region_v2_uv_required_present_count",
            "raw_template_region_v2_uv_island_count",
            "raw_template_region_v2_region_scoped_uv_island_count",
            "raw_template_region_v2_uv_island_allocated_texel_count",
            "raw_template_region_v2_uv_island_density_weighted_texel_count",
            "raw_template_region_v2_face_uv_island_count",
            "raw_template_region_v2_facial_uv_island_count",
            "raw_template_region_v2_facial_uv_island_allocated_texel_count",
            "raw_template_region_v2_body_region_uv_island_count",
            "raw_template_region_v2_eye_min_allocated_texels",
            "raw_template_region_v2_nose_allocated_texels",
            "raw_template_region_v2_muzzle_allocated_texels",
            "raw_template_region_v2_ear_inner_min_allocated_texels",
            "raw_template_region_v2_facial_density_weighted_texel_count",
            "raw_face_texture_target_size",
            "raw_face_texture_target_scale",
            "raw_face_texture_target_requested_texel_count",
            "raw_face_texture_target_allocated_texel_count",
            "raw_face_texture_target_eye_min_texels",
            "raw_face_texture_target_nose_texels",
            "raw_face_texture_target_ear_inner_min_texels",
            "raw_template_region_v2_texture_target_size",
            "raw_template_region_v2_texture_target_scale",
            "raw_template_region_v2_texture_target_island_count",
            "raw_template_region_v2_texture_target_facial_island_count",
            "raw_template_region_v2_texture_target_requested_texel_count",
            "raw_template_region_v2_texture_target_allocated_texel_count",
            "raw_template_region_v2_texture_target_eye_min_texels",
            "raw_template_region_v2_texture_target_nose_texels",
            "raw_template_region_v2_texture_target_muzzle_texels",
            "raw_template_region_v2_texture_target_ear_inner_min_texels",
            "raw_template_region_v2_texture_target_max_density_scale",
            "raw_face_texture_crop_region_count",
            "raw_face_texture_crop_texture_width",
            "raw_face_texture_crop_texture_height",
            "raw_face_texture_crop_compared_region_count",
            "raw_face_texture_crop_exact_source_patch_region_count",
            "raw_face_texture_crop_exact_source_patch_fraction",
            "raw_face_texture_crop_detail_region_count",
            "raw_face_texture_crop_min_unique_rgb_count",
            "raw_face_texture_crop_min_confident_texel_fraction",
            "raw_face_texture_crop_mean_confident_texel_fraction",
            "raw_face_texture_crop_mean_source_rgb_delta",
            "raw_face_texture_crop_isolated_region_count",
            "raw_face_texture_crop_overlap_pair_count",
            "raw_face_texture_crop_overlap_fraction",
            "raw_face_texture_crop_visual_quality_region_count",
            "raw_face_texture_crop_visual_quality_passed_count",
            "raw_face_texture_crop_visual_quality_fraction",
            "raw_face_texture_crop_visual_quality_min_score",
            "raw_face_texture_crop_visual_quality_mean_score",
            "raw_direct_photo_face_candidate_sample_key_count",
            "raw_direct_photo_face_candidate_total_count",
            "raw_direct_photo_face_candidate_direct_count",
            "raw_direct_photo_face_candidate_eligible_count",
            "raw_direct_photo_face_candidate_selected_count",
            "raw_direct_photo_face_candidate_rejected_count",
            "raw_direct_photo_face_candidate_missing_metric_feature_count",
            "raw_direct_photo_face_candidate_max_priority",
            "raw_face_feature_min_required_contrast",
            "raw_face_feature_eye_to_muzzle_luma_delta",
            "raw_face_feature_mouth_to_muzzle_luma_delta",
            "raw_face_feature_nose_to_muzzle_rgb_distance",
            "raw_face_feature_iris_to_pupil_rgb_distance",
            "raw_face_feature_contrast_passed",
            "raw_pbr_face_material_detail_applied",
            "raw_pbr_face_material_detail_feature_count",
            "raw_pbr_face_material_detail_texel_count",
            "raw_pbr_face_material_detail_normal_variance",
            "raw_face_detail_geometry_applied",
            "raw_face_detail_deformed_vertex_count",
            "raw_face_detail_max_displacement",
            "raw_face_detail_mean_displacement",
            "raw_face_surface_anchor_count",
            "raw_face_surface_anchor_vertex_count",
            "raw_face_surface_anchor_max_displacement",
            "raw_face_turntable_view_count",
            "raw_face_turntable_front_visible_anchor_count",
            "raw_face_turntable_front_anchor_pixel_count",
            "raw_face_turntable_required_anchor_visible_fraction",
            "raw_face_texture_closeup_width",
            "raw_face_texture_closeup_height",
            "raw_face_texture_closeup_feature_count",
            "raw_face_texture_closeup_required_feature_count",
            "raw_face_texture_closeup_required_present_count",
            "raw_face_texture_closeup_required_feature_fraction",
            "raw_face_texture_closeup_crop_texel_count",
            "raw_face_texture_closeup_unique_rgb_count",
            "raw_face_texture_closeup_luma_std",
            "raw_face_texture_source_compare_feature_count",
            "raw_face_texture_source_compare_source_feature_available_count",
            "raw_face_texture_source_compare_required_feature_count",
            "raw_face_texture_source_compare_required_present_count",
            "raw_face_texture_source_compare_required_feature_fraction",
            "raw_face_texture_source_compare_mean_rgb_distance",
            "raw_face_texture_source_compare_max_rgb_distance",
            "raw_face_texture_source_compare_mean_luma_delta",
            "raw_face_texture_source_compare_max_luma_delta",
            "raw_face_texture_source_compare_source_mean_luma_std",
            "raw_face_texture_source_compare_baked_mean_luma_std",
            "raw_face_texture_source_compare_iris_feature_count",
            "raw_face_texture_source_compare_iris_candidate_texel_count",
            "raw_face_texture_source_compare_iris_core_accepted_texel_count",
            "raw_face_texture_source_compare_iris_core_densified_texel_count",
            "raw_face_texture_source_compare_iris_annulus_rejected_texel_count",
            "raw_face_texture_source_compare_iris_core_acceptance_fraction",
            "raw_face_texture_source_compare_iris_annulus_rejected_fraction",
            "raw_texture_source_preflight_accepted_frame_count",
            "raw_texture_source_preflight_duplicate_frame_count",
            "raw_texture_source_preflight_rejected_frame_count",
            "raw_texture_source_preflight_top_score",
            "final_loss",
            "initial_loss",
            "keypoint_loss_reduction",
            "silhouette_loss_reduction",
            "vertex_count",
            "face_count",
            "raw_quality_passed_check_count",
            "raw_quality_total_check_count",
        }
    )
    success = "Passed" if bool(summary.get("success")) else "Failed"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenCat3D Result</title>
  {STYLE}
</head>
<body>
  <main class="app-shell">
    <section class="toolbar">
      <div>
        <h1>OpenCat3D</h1>
        <p>Job {html.escape(job_id)} - {success}</p>
      </div>
      <a class="health" href="/">New Job</a>
    </section>
    <section class="result-grid">
      <div class="panel primary">
        <div class="panel-header">
          <h2>Artifacts</h2>
          <span>{success}</span>
        </div>
        <p class="subtle">{html.escape(topology)}</p>
        <div class="artifact-list">{artifact_links}</div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <h2>Metrics</h2>
          <span>Raw pipeline</span>
        </div>
        <table>{metric_rows}</table>
      </div>
    </section>
    {template_region_panel}
    {source_gap_panel}
    {acquisition_panel}
    {face_bake_panel}
    {face_preview_panel}
  </main>
</body>
</html>"""


def render_reconstruction_preflight_job(root_dir: Path, *, job_id: str) -> str:
    loop_dir = Path(root_dir).resolve() / "preflight_jobs" / job_id
    upload_dir = Path(root_dir).resolve() / "preflight_uploads" / job_id
    report_path = loop_dir / "raw_improvement_loop_report.json"
    if not report_path.exists():
        return render_index(message=f"Preflight {job_id} was not found.")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return render_index(message=f"Preflight {job_id} has no readable report.")
    media_preflight = report.get("media_preflight", {})
    if not isinstance(media_preflight, dict):
        media_preflight = {}
    gate = report.get("media_preflight_promotion_gate", {})
    if not isinstance(gate, dict):
        gate = {}
    ready_reconstruction = bool(media_preflight.get("ready_for_reconstruction", False))
    ready_promotion = bool(media_preflight.get("ready_for_promotion", False))
    warnings = media_preflight.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    rejection_reasons = gate.get("rejection_reasons", [])
    if not isinstance(rejection_reasons, list):
        rejection_reasons = []
    geometry_ids = media_preflight.get("recommended_geometry_frame_ids", [])
    if not isinstance(geometry_ids, list):
        geometry_ids = []
    texture_ids = media_preflight.get("recommended_texture_frame_ids", [])
    if not isinstance(texture_ids, list):
        texture_ids = []
    status = "Ready" if ready_promotion else "Needs more evidence"
    warning_rows = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in warnings
    ) or "<li>none</li>"
    rejection_rows = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in rejection_reasons
    ) or "<li>none</li>"
    geometry_rows = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in geometry_ids
    ) or "<li>none</li>"
    texture_rows = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in texture_ids
    ) or "<li>none</li>"
    metrics = "".join(
        [
            _face_preview_metric("Reconstruction", "yes" if ready_reconstruction else "no"),
            _face_preview_metric("Promotion", "yes" if ready_promotion else "no"),
            _face_preview_metric(
                "Geometry Safe",
                media_preflight.get("geometry_safe_frame_count", 0),
            ),
            _face_preview_metric(
                "Face Texture",
                media_preflight.get("face_texture_frame_count", 0),
            ),
            _face_preview_metric(
                "Texture Usable",
                media_preflight.get("texture_usable_frame_count", 0),
            ),
        ]
    )
    media_batch_panel = _media_batch_preflight_panel(upload_dir)
    source_gap_panel = _source_frame_acquisition_gap_preflight_panel(
        report,
        loop_dir=loop_dir,
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenCat3D Preflight</title>
  {STYLE}
</head>
<body>
  <main>
    <section class="toolbar">
      <div>
        <h1>OpenCat3D</h1>
        <p>Media preflight for {html.escape(job_id)}.</p>
      </div>
      <a class="health" href="/">New Upload</a>
    </section>
    <section class="panel primary">
      <div class="panel-header">
        <h2>Preflight</h2>
        <span>{html.escape(status)}</span>
      </div>
      <dl class="face-preview-metrics">{metrics}</dl>
      <div class="result-grid">
        <section>
          <h3>Geometry Frames</h3>
          <ul>{geometry_rows}</ul>
        </section>
        <section>
          <h3>Texture Frames</h3>
          <ul>{texture_rows}</ul>
        </section>
      </div>
      {media_batch_panel}
      {source_gap_panel}
      <div class="result-grid">
        <section>
          <h3>Warnings</h3>
          <ul>{warning_rows}</ul>
        </section>
        <section>
          <h3>Promotion Gate</h3>
          <ul>{rejection_rows}</ul>
        </section>
      </div>
    </section>
  </main>
</body>
</html>"""


def _media_batch_preflight_panel(upload_dir: Path) -> str:
    report_path = upload_dir / "media_preflight_report.json"
    if not report_path.is_file():
        return ""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    metric_rows = "".join(
        _face_preview_metric(label, report.get(key, 0))
        for label, key in [
            ("Input Frames", "input_frame_count"),
            ("Unique Frames", "unique_frame_count"),
            ("Duplicate Frames", "duplicate_frame_count"),
            ("Existing Frames", "existing_frame_count"),
            ("New Unique", "new_unique_frame_count"),
            ("New Duplicates", "new_duplicate_frame_count"),
            ("Photos", "photo_frame_count"),
            ("Video Frames", "video_frame_count"),
        ]
    )
    duplicate_groups = report.get("duplicate_groups", [])
    if not isinstance(duplicate_groups, list):
        duplicate_groups = []
    duplicate_rows = "".join(
        "<li>"
        f"{html.escape(str(group.get('kept_source_media_id', 'unknown')))} kept, "
        f"{len(group.get('duplicate_records', [])) if isinstance(group, dict) else 0} duplicate"
        "</li>"
        for group in duplicate_groups[:8]
        if isinstance(group, dict)
    ) or "<li>none</li>"
    return f"""
      <section>
        <h3>Media Batch</h3>
        <dl class="face-preview-metrics">{metric_rows}</dl>
        <ul>{duplicate_rows}</ul>
      </section>
    """


def render_identity_job(work_dir: Path, *, job_id: str) -> str:
    summary_path = Path(work_dir).resolve() / "catnet" / "jobs" / f"{job_id}.json"
    if not summary_path.exists():
        return render_index(message=f"Identity job {job_id} was not found.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['filename'])}</td>"
        f"<td>{html.escape(item['cat_id'])}</td>"
        f"<td>{html.escape(item['decision'])}</td>"
        f"<td>{float(item['confidence']):.3f}</td>"
        f"<td>{float(item['fused_score']):.3f}</td>"
        "</tr>"
        for item in summary.get("results", [])
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CatNet Identity Result</title>
  {STYLE}
</head>
<body>
  <main class="app-shell">
    <section class="toolbar">
      <div>
        <h1>CatNet</h1>
        <p>Identity job {html.escape(job_id)}</p>
      </div>
      <a class="health" href="/">New Job</a>
    </section>
    <section class="panel primary">
      <div class="panel-header">
        <h2>Identity Decisions</h2>
        <span>{len(summary.get("results", []))} observations</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>File</th>
            <th>Cat ID</th>
            <th>Decision</th>
            <th>Confidence</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""


def render_identity_review(work_dir: Path) -> str:
    store = CatNetStore(Path(work_dir).resolve() / "catnet")
    tasks = _open_review_tasks(store)
    if tasks:
        rows = "\n".join(
            "<tr>"
            f'<td><a href="/identity/review/{html.escape(task.task_id)}">'
            f"{html.escape(task.task_id)}</a></td>"
            f"<td>{html.escape(task.decision.value)}</td>"
            f"<td>{task.priority_score:.3f}</td>"
            f"<td>{task.fused_score:.3f}</td>"
            f"<td>{', '.join(html.escape(reason) for reason in task.reason_codes)}</td>"
            "</tr>"
            for task in tasks
        )
    else:
        rows = '<tr><td colspan="5">No open identity review tasks.</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CatNet Identity Review</title>
  {STYLE}
</head>
<body>
  <main class="app-shell">
    <section class="toolbar">
      <div>
        <h1>CatNet Review</h1>
        <p>Reviewer feedback writes durable identity constraints.</p>
      </div>
      <a class="health" href="/">New Job</a>
    </section>
    <section class="panel primary">
      <div class="panel-header">
        <h2>Open Tasks</h2>
        <span>{len(tasks)} tasks</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Task</th>
            <th>Decision</th>
            <th>Priority</th>
            <th>Fused</th>
            <th>Reasons</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""


def render_identity_review_task(work_dir: Path, *, task_id: str) -> str:
    store = CatNetStore(Path(work_dir).resolve() / "catnet")
    task = _find_review_task(store, task_id=task_id)
    edge = store.catalog.match_edges[task.edge_id]
    src_obs = store.catalog.observations.get(task.src_obs_id)
    dst_obs = store.catalog.observations.get(task.dst_obs_id) if task.dst_obs_id else None
    score_rows = "\n".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>"
        for label, value in [
            ("Global", f"{edge.global_score:.3f}"),
            ("Local", _score_text(edge.local_score)),
            ("Temporal", f"{edge.temporal_score:.3f}"),
            ("Metadata", _score_text(edge.metadata_score)),
            ("Fused", f"{edge.fused_score:.3f}"),
            ("Confidence", f"{edge.confidence:.3f}"),
        ]
    )
    explanation = html.escape(json.dumps(edge.explanation, sort_keys=True)[:1200])
    same_disabled = " disabled" if dst_obs is None else ""
    different_disabled = " disabled" if dst_obs is None else ""
    task_url = f"/identity/review/{html.escape(task.task_id)}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CatNet Review Task</title>
  {STYLE}
</head>
<body>
  <main class="app-shell">
    <section class="toolbar">
      <div>
        <h1>CatNet Review</h1>
        <p>Task {html.escape(task.task_id)}</p>
      </div>
      <a class="health" href="/identity/review">Review Queue</a>
    </section>
    <section class="review-grid">
      {_review_image_panel(store, "Query image", src_obs)}
      {_review_image_panel(store, "Candidate image", dst_obs)}
      <section class="panel">
        <div class="panel-header">
          <h2>Scores</h2>
          <span>{html.escape(edge.decision.value)}</span>
        </div>
        <table>{score_rows}</table>
        <p class="compact-explanation">{explanation}</p>
        <form class="feedback-actions" method="post" action="{task_url}">
          <button type="submit" name="feedback_action" value="same"{same_disabled}>Same Cat</button>
          <button
            type="submit"
            name="feedback_action"
            value="different"{different_disabled}
          >Different Cat</button>
          <button type="submit" name="feedback_action" value="unsure">Unsure</button>
        </form>
      </section>
    </section>
  </main>
</body>
</html>"""


class _WebHandler(BaseHTTPRequestHandler):
    root_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_index())
            return
        if parsed.path == "/healthz":
            self._send_text("ok\n")
            return
        if parsed.path == "/identity/review":
            self._send_html(render_identity_review(self.root_dir))
            return
        if parsed.path.startswith("/identity/review/"):
            try:
                task_id = _safe_segment(parsed.path.removeprefix("/identity/review/"))
                payload = render_identity_review_task(self.root_dir, task_id=task_id)
            except (FileNotFoundError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            self._send_html(payload)
            return
        if parsed.path.startswith("/identity/media/"):
            self._send_identity_media(parsed.path.removeprefix("/identity/media/"))
            return
        if parsed.path.startswith("/jobs/"):
            try:
                job_id = _safe_segment(parsed.path.removeprefix("/jobs/"))
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            self._send_html(render_job(self.root_dir / "projects" / job_id, job_id=job_id))
            return
        if parsed.path.startswith("/preflight/jobs/"):
            try:
                job_id = _safe_segment(parsed.path.removeprefix("/preflight/jobs/"))
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            self._send_html(render_reconstruction_preflight_job(self.root_dir, job_id=job_id))
            return
        if parsed.path.startswith("/identity/jobs/"):
            try:
                job_id = _safe_segment(parsed.path.removeprefix("/identity/jobs/"))
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            self._send_html(render_identity_job(self.root_dir, job_id=job_id))
            return
        if parsed.path.startswith("/files/"):
            self._send_project_file(parsed.path.removeprefix("/files/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/identity/review/"):
            try:
                task_id = _safe_segment(parsed.path.removeprefix("/identity/review/"))
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                self._handle_identity_review_post(task_id)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
            except ValueError as exc:
                self._send_text(f"{exc}\n", status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/identity/jobs":
            self._handle_identity_post()
            return
        if parsed.path == "/jobs/preflight":
            self._handle_reconstruction_preflight_post()
            return
        if parsed.path != "/jobs":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            uploads = self._read_uploads()
            color_delta_threshold = float(self._form_value("color_delta_threshold", "0.10"))
            camera_root_steps = int(float(self._form_value("camera_root_steps", "2")))
            result = run_uploaded_image_job(
                uploads,
                work_dir=self.root_dir,
                color_delta_threshold=color_delta_threshold,
                camera_root_steps=camera_root_steps,
                segmentation_backend="auto_subject_v1",
            )
        except ValueError as exc:
            self._send_html(render_index(message=str(exc)), status=HTTPStatus.BAD_REQUEST)
            return
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/jobs/{result.job_id}")
        self.end_headers()

    def _handle_reconstruction_preflight_post(self) -> None:
        try:
            uploads = self._read_uploads()
            context = _resolve_reconstruction_preflight_context(self.root_dir)
            result = run_uploaded_media_preflight_job(
                uploads,
                work_dir=self.root_dir,
                min_selected_frames=2,
                max_selected_frames=8,
                existing_upload_dirs=context.existing_upload_dirs,
                segmentation_cache_report=context.segmentation_cache_report,
                baseline_project_dir=context.baseline_project_dir,
                texture_probe_report=context.texture_probe_report,
            )
        except (RuntimeError, ValueError) as exc:
            self._send_html(render_index(message=str(exc)), status=HTTPStatus.BAD_REQUEST)
            return
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/preflight/jobs/{result.job_id}")
        self.end_headers()

    def _handle_identity_post(self) -> None:
        try:
            uploads = self._read_uploads()
            result = run_identity_image_job(
                uploads,
                work_dir=self.root_dir,
                captured_at=_parse_optional_datetime(self._form_value("captured_at", "")),
                site_id=_clean_optional(self._form_value("site_id", "")),
                camera_id=_clean_optional(self._form_value("camera_id", "")),
                uploader=_clean_optional(self._form_value("uploader", "")),
                model_path=_optional_path(self._form_value("model_path", "")),
            )
        except ValueError as exc:
            self._send_html(render_index(message=str(exc)), status=HTTPStatus.BAD_REQUEST)
            return
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/identity/jobs/{result.job_id}")
        self.end_headers()

    def _handle_identity_review_post(self, task_id: str) -> None:
        self._read_form_fields()
        action = self._form_value("feedback_action", "").strip().lower()
        if action not in {"same", "different", "unsure"}:
            raise ValueError("invalid review action")
        store = CatNetStore(self.root_dir / "catnet")
        task = _find_review_task(store, task_id=task_id)
        if action != "unsure" and task.dst_obs_id is None:
            raise ValueError("review task has no candidate observation")
        constraint_type = {
            "same": FeedbackConstraintType.MUST_LINK,
            "different": FeedbackConstraintType.CANNOT_LINK,
            "unsure": FeedbackConstraintType.UNSURE,
        }[action]
        add_feedback_constraint(
            store,
            constraint_type=constraint_type,
            left_obs_id=task.src_obs_id if action != "unsure" else None,
            right_obs_id=task.dst_obs_id if action != "unsure" else None,
            review_task_id=task.task_id,
            reviewer=_clean_optional(self._form_value("reviewer", "")),
            actor="catweb",
            source="identity_review_ui",
            reason=f"reviewer selected {action}",
            metadata={
                "edge_id": task.edge_id,
                "global_score": task.global_score,
                "local_score": task.local_score,
                "temporal_score": task.temporal_score,
                "metadata_score": task.metadata_score,
                "fused_score": task.fused_score,
            },
        )
        store.save()
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/identity/review")
        self.end_headers()

    def _read_uploads(self) -> list[UploadedFile]:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            self._form_values = {}
            return []
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("Expected a multipart form upload.")

        body = self.rfile.read(content_length)
        message = BytesParser(policy=default).parsebytes(
            b"Content-Type: "
            + content_type.encode("utf-8")
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + body
        )
        if not message.is_multipart():
            self._form_values = {}
            return []

        uploads = []
        self._form_values = {}
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = str(part.get_param("name", header="content-disposition") or "")
            filename = part.get_filename()
            content = part.get_payload(decode=True) or b""
            if filename and name == "files" and content:
                uploads.append(UploadedFile(filename=str(filename), content=content))
                continue
            if name:
                charset = part.get_content_charset() or "utf-8"
                self._form_values[name] = content.decode(charset, errors="replace")
        return uploads

    def _read_form_fields(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        self._form_values = {}
        if content_length <= 0:
            return
        if content_type.lower().startswith("multipart/form-data"):
            self._read_uploads()
            return
        body = self.rfile.read(content_length)
        if content_type.lower().startswith("application/x-www-form-urlencoded"):
            decoded = body.decode("utf-8", errors="replace")
            self._form_values = {
                key: values[-1] if values else ""
                for key, values in parse_qs(decoded, keep_blank_values=True).items()
            }
            return
        raise ValueError("Expected a form submission.")

    def _form_value(self, name: str, default: str) -> str:
        form_values = getattr(self, "_form_values", {})
        return str(form_values.get(name, default))

    def _send_project_file(self, raw_path: str) -> None:
        parts = [part for part in raw_path.split("/") if part]
        if len(parts) < 2:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            job_id = _safe_segment(parts[0])
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        relative = Path(unquote("/".join(parts[1:])))
        project_dir = (self.root_dir / "projects" / job_id).resolve()
        target = (project_dir / relative).resolve()
        if not target.is_file() or not target.is_relative_to(project_dir):
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", _content_type(target))
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        with target.open("rb") as handle:
            self.wfile.write(handle.read())

    def _send_identity_media(self, raw_media_id: str) -> None:
        try:
            media_id = _safe_segment(raw_media_id)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        store = CatNetStore(self.root_dir / "catnet")
        media = store.catalog.media_assets.get(media_id)
        if media is None:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        target = Path(media.storage_uri).resolve()
        root = self.root_dir.resolve()
        if not target.is_file() or not target.is_relative_to(root):
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media.mime_type or _content_type(target))
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        with target.open("rb") as handle:
            self.wfile.write(handle.read())

    def _send_html(self, payload: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, payload: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _validate_reconstruction_uploads(uploads: list[UploadedFile]) -> list[UploadedFile]:
    if not uploads:
        raise ValueError("Upload at least two photos or one video.")
    invalid = [
        upload.filename
        for upload in uploads
        if Path(upload.filename).suffix.lower()
        not in SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
    ]
    if invalid:
        suffixes = sorted({Path(filename).suffix.lower() for filename in invalid})
        if any(suffix in SUPPORTED_SCAN_EXTENSIONS for suffix in suffixes):
            raise ValueError("Scan upload is not implemented yet; upload photos for fitting.")
        raise ValueError(f"Unsupported upload type: {', '.join(suffixes)}")
    image_count = sum(
        Path(upload.filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS for upload in uploads
    )
    video_count = sum(
        Path(upload.filename).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS for upload in uploads
    )
    if video_count == 0 and image_count < 2:
        raise ValueError("Upload at least two JPG or PNG cat photos, or one video.")
    return sorted(uploads, key=lambda upload: upload.filename)


def _clear_staged_reconstruction_uploads(upload_dir: Path) -> None:
    for path in upload_dir.iterdir():
        if path.name == "media_manifest.json" or path.name.startswith("frame_"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            continue
        if path.name.startswith("video_"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _clear_media_preflight_dir(upload_dir: Path) -> None:
    if not upload_dir.exists():
        return
    for path in upload_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _staged_upload_frame_records(
    upload_dir: Path,
    *,
    origin: str,
) -> list[dict[str, object]]:
    upload_dir = upload_dir.resolve()
    if not upload_dir.exists():
        raise FileNotFoundError(f"{upload_dir} does not exist")
    media_manifest_path = upload_dir / "media_manifest.json"
    manifest_frames: dict[str, object] = {}
    if media_manifest_path.exists():
        manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))
        frames = manifest.get("frames", {})
        if isinstance(frames, dict):
            manifest_frames = dict(frames)
    records: list[dict[str, object]] = []
    for frame_path in sorted(upload_dir.glob("frame_*"), key=lambda path: path.name):
        if not frame_path.is_file():
            continue
        raw_metadata = manifest_frames.get(frame_path.name, {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata.setdefault("media_type", "photo")
        metadata.setdefault("source_type", "image")
        metadata.setdefault("source_media_id", frame_path.name)
        records.append(
            {
                "source_path": str(frame_path),
                "source_frame_name": frame_path.name,
                "origin_upload_dir": str(upload_dir),
                "origin": origin,
                "sha256": _sha256_path(frame_path),
                "metadata": metadata,
            }
        )
    return records


def _preflight_duplicate_record_json(record: dict[str, object]) -> dict[str, object]:
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "origin": record["origin"],
        "origin_upload_dir": record["origin_upload_dir"],
        "source_frame_name": record["source_frame_name"],
        "source_media_id": metadata.get("source_media_id", ""),
        "media_type": metadata.get("media_type", "photo"),
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identity_uploads(uploads: list[UploadedFile]) -> list[UploadedFile]:
    if not uploads:
        raise ValueError("Upload at least one JPG or PNG cat photo for identity matching.")
    invalid = [
        upload.filename
        for upload in uploads
        if Path(upload.filename).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if invalid:
        suffixes = sorted({Path(filename).suffix.lower() for filename in invalid})
        if any(suffix in SUPPORTED_VIDEO_EXTENSIONS for suffix in suffixes):
            raise ValueError("Video identity matching needs frame extraction first.")
        if any(suffix in SUPPORTED_SCAN_EXTENSIONS for suffix in suffixes):
            raise ValueError("Scan identity matching is not implemented yet.")
        raise ValueError(f"Unsupported identity upload type: {', '.join(suffixes)}")
    return sorted(uploads, key=lambda upload: upload.filename)


def _job_id(
    uploads: list[UploadedFile],
    *,
    color_delta_threshold: float,
    camera_root_steps: int,
    segmentation_backend: str,
    max_selected_frames: int = 5,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"color_delta_threshold={color_delta_threshold:.6f}".encode())
    digest.update(b"\0")
    digest.update(f"camera_root_steps={camera_root_steps}".encode())
    digest.update(b"\0")
    digest.update(f"max_selected_frames={max_selected_frames}".encode())
    digest.update(b"\0")
    digest.update(f"segmentation_backend={segmentation_backend}".encode())
    digest.update(b"\0")
    for upload in uploads:
        digest.update(upload.filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(upload.content)
        digest.update(b"\0")
    return f"job_{digest.hexdigest()[:16]}"


def _preflight_job_id(
    uploads: list[UploadedFile],
    *,
    min_selected_frames: int,
    max_selected_frames: int,
    existing_upload_dirs: tuple[Path, ...] | list[Path] = (),
) -> str:
    digest = hashlib.sha256()
    digest.update(f"min_selected_frames={min_selected_frames}".encode())
    digest.update(b"\0")
    digest.update(f"max_selected_frames={max_selected_frames}".encode())
    digest.update(b"\0")
    for existing_upload_dir in sorted(
        (Path(path).expanduser().resolve() for path in existing_upload_dirs),
        key=str,
    ):
        digest.update(f"existing_upload_dir={existing_upload_dir}".encode())
        digest.update(b"\0")
        for record in _staged_upload_frame_records(
            existing_upload_dir,
            origin="existing_upload",
        ):
            metadata = record["metadata"] if isinstance(record["metadata"], dict) else {}
            digest.update(str(record["source_frame_name"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(record["sha256"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(metadata.get("source_media_id", "")).encode("utf-8"))
            digest.update(b"\0")
    for upload in uploads:
        digest.update(upload.filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(upload.content)
        digest.update(b"\0")
    return f"preflight_{digest.hexdigest()[:16]}"


def _identity_job_id(
    uploads: list[UploadedFile],
    *,
    captured_at: datetime | None,
    site_id: str | None,
    camera_id: str | None,
    uploader: str | None,
    model_path: Path | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"captured_at={captured_at.isoformat() if captured_at else ''}".encode())
    digest.update(b"\0")
    digest.update(f"site_id={site_id or ''}".encode())
    digest.update(b"\0")
    digest.update(f"camera_id={camera_id or ''}".encode())
    digest.update(b"\0")
    digest.update(f"uploader={uploader or ''}".encode())
    digest.update(b"\0")
    digest.update(f"model_path={model_path or ''}".encode())
    digest.update(b"\0")
    for upload in uploads:
        digest.update(upload.filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(upload.content)
        digest.update(b"\0")
    return f"idjob_{digest.hexdigest()[:16]}"


def _safe_segment(value: str) -> str:
    segment = unquote(value).strip("/")
    if not segment or "/" in segment or "\\" in segment or segment in {".", ".."}:
        raise ValueError("invalid path segment")
    return segment


def _write_identity_job_summary(root: Path, result: IdentityJobResult) -> None:
    jobs_dir = root / "catnet" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "1.0",
        "job_id": result.job_id,
        "upload_dir": str(result.upload_dir),
        "store_dir": str(result.store_dir),
        "results": [item.__dict__ for item in result.results],
    }
    (jobs_dir / f"{result.job_id}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _open_review_tasks(store: CatNetStore) -> list[object]:
    closed = closed_review_task_ids(store)
    return [
        task
        for task in build_ambiguity_queue(
            list(store.catalog.match_edges.values()),
            max_tasks=100,
        )
        if task.task_id not in closed
    ]


def _find_review_task(store: CatNetStore, *, task_id: str) -> object:
    _safe_segment(task_id)
    for task in _open_review_tasks(store):
        if task.task_id == task_id:
            return task
    raise FileNotFoundError(f"review task not found: {task_id}")


def _review_image_panel(store: CatNetStore, label: str, observation: Observation | None) -> str:
    if observation is None:
        return f"""<section class="panel">
        <div class="panel-header"><h2>{html.escape(label)}</h2><span>Missing</span></div>
        <p>No candidate image is attached to this review task.</p>
      </section>"""
    media = store.catalog.media_assets.get(observation.media_id)
    if media is None:
        return f"""<section class="panel">
        <div class="panel-header"><h2>{html.escape(label)}</h2><span>Missing</span></div>
        <p>Media record was not found for observation {html.escape(observation.obs_id)}.</p>
      </section>"""
    media_url = f"/identity/media/{html.escape(media.media_id)}"
    escaped_label = html.escape(label)
    captured_at = html.escape(
        observation.captured_at.isoformat() if observation.captured_at else "unknown"
    )
    return f"""<section class="panel">
        <div class="panel-header">
          <h2>{escaped_label}</h2>
          <span>{html.escape(observation.cat_id or "unassigned")}</span>
        </div>
        <img class="review-image" src="{media_url}" alt="{escaped_label}">
        <table>
          <tr><th>Observation</th><td>{html.escape(observation.obs_id)}</td></tr>
          <tr><th>Captured</th><td>{captured_at}</td></tr>
          <tr><th>Site</th><td>{html.escape(observation.site_id or "")}</td></tr>
        </table>
      </section>"""


def _score_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _candidate_image_paths(
    store: CatNetStore,
    candidates: list[SearchCandidate],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for candidate in candidates:
        observation = store.catalog.observations.get(candidate.obs_id)
        if observation is None:
            continue
        media = store.catalog.media_assets.get(observation.media_id)
        if media is None:
            continue
        paths[candidate.obs_id] = Path(media.storage_uri)
    return paths


def _identity_indexer(
    store_dir: Path,
    *,
    existing_embeddings: list[EmbeddingVector],
    existing_observations: dict[str, Observation],
    query_embedding: EmbeddingVector,
) -> FaissIndex:
    registry = IndexRegistry(store_dir / "index_registry")
    for alias in ("hot", "main"):
        index = registry.load_current_index(alias)
        if index is None:
            continue
        if (
            index.index_version.model_version == query_embedding.model_version
            and index.dim == query_embedding.dim
        ):
            return index
    return FaissIndex.from_embeddings(existing_embeddings, existing_observations)


def _float_from_explanation(
    explanation: dict[str, object],
    key: str,
    *,
    fallback: float,
) -> float:
    try:
        return float(explanation.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _trained_embedding(
    model: CatReIDModel,
    image_path: Path,
    *,
    observation: Observation,
    model_path: Path | None,
) -> EmbeddingVector:
    sample = ReIDSample(
        image_path=str(image_path),
        cat_id="query",
        obs_id=observation.obs_id,
        captured_at=observation.captured_at,
        site_id=observation.site_id,
        camera_id=observation.camera_id,
        age_bin=observation.age_bin,
        size_bin=observation.size_bin,
        face_view=observation.face_view,
        body_view=observation.body_view,
        quality_score=observation.quality_score,
        metadata=observation.metadata,
    )
    vector = model.embed_sample(sample)
    return EmbeddingVector(
        emb_id=stable_id(
            "emb",
            observation.obs_id,
            BranchType.FUSED.value,
            model.config.model_version,
        ),
        obs_id=observation.obs_id,
        model_version=model.config.model_version,
        branch_type=BranchType.FUSED,
        dim=int(vector.shape[0]),
        vector=[float(value) for value in vector],
        norm=float(np.linalg.norm(vector)),
        metadata={
            "backbone": model.backbone_name,
            "trained_model_path": str(model_path) if model_path else "",
        },
    )


def _trained_model_version(model: CatReIDModel, model_path: Path | None) -> ModelVersion:
    return ModelVersion(
        model_version=model.config.model_version,
        backbone=model.backbone_name,
        train_dataset_snapshot=str(model_path) if model_path else "unknown",
        calibration_version=model.config.calibration_version,
        embedding_dim=int(model.projection.shape[1]),
        branch_types=[BranchType.FUSED],
    )


def _ensure_identity(
    store: CatNetStore,
    *,
    cat_id: str,
    first_seen_at: datetime | None,
    reason: str,
) -> CatIdentity:
    existing = store.catalog.identities.get(cat_id)
    if existing is not None:
        return existing
    version = store.record_identity_event(
        cat_id=cat_id,
        event_type=IdentityEventType.CREATED,
        reason=reason,
        payload={"source": "trained_reid_model"},
    )
    identity = CatIdentity(
        cat_id=cat_id,
        identity_version=version.identity_version,
        status=IdentityStatus.ACTIVE,
        first_seen_at=first_seen_at,
        last_seen_at=first_seen_at,
    )
    return store.upsert_identity(identity)


def _image_size(content: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(content)) as image:
        return image.size


def _upload_mime_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def _parse_optional_datetime(raw: str) -> datetime | None:
    value = raw.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _clean_optional(raw: str) -> str | None:
    value = raw.strip()
    return value or None


def _optional_path(raw: str) -> Path | None:
    value = raw.strip()
    return Path(value).expanduser() if value else None


def _artifact_link(project_dir: Path, job_id: str, label: str, raw_path: str) -> str:
    href = _project_file_href(project_dir, job_id=job_id, raw_path=raw_path)
    if not href:
        return ""
    return f'<a href="{href}">{html.escape(label)}</a>'


def _project_file_href(project_dir: Path, *, job_id: str, raw_path: str) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_dir / path
    if not path.is_file():
        return ""
    try:
        relative = path.resolve().relative_to(project_dir.resolve())
    except ValueError:
        return ""
    return f"/files/{html.escape(job_id)}/{html.escape(relative.as_posix())}"


def _template_region_v2_panel(project_dir: Path, *, job_id: str) -> str:
    validation_path = project_dir / "outputs" / "avatar" / "mesh_validation.json"
    validation_href = _project_file_href(
        project_dir,
        job_id=job_id,
        raw_path=str(validation_path),
    )
    if not validation_href:
        return ""
    try:
        mesh_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(mesh_validation, dict):
        return ""
    manifest = mesh_validation.get("template_regions_v2", {})
    if not isinstance(manifest, dict):
        return ""
    regions_raw = manifest.get("regions", [])
    if not isinstance(regions_raw, list) or not regions_raw:
        return ""
    validation = manifest.get("validation", {})
    if not isinstance(validation, dict):
        validation = {}
    required_names = manifest.get("required_region_names", [])
    if not isinstance(required_names, list):
        required_names = []
    region_names = {
        str(region.get("name", ""))
        for region in regions_raw
        if isinstance(region, dict)
    }
    required_present_count = len(
        {str(name) for name in required_names}.intersection(region_names)
    )
    region_count = int(mesh_validation.get("template_region_v2_count", len(regions_raw)))
    contract_valid = bool(
        mesh_validation.get("template_region_v2_contract_valid", False)
        and mesh_validation.get("glb_template_regions_v2_valid", False)
        and validation.get("valid", False)
    )
    status = "Valid" if contract_valid else "Invalid"
    summary = "".join(
        [
            _face_preview_metric("Contract", status),
            _face_preview_metric("Regions", region_count),
            _face_preview_metric("Required", len(required_names)),
            _face_preview_metric("Required Present", required_present_count),
            _face_preview_metric(
                "GLB Contract",
                "Valid" if mesh_validation.get("glb_template_regions_v2_valid") else "Invalid",
            ),
        ]
    )
    rows = []
    for raw_region in regions_raw:
        if not isinstance(raw_region, dict):
            continue
        region = cast("dict[str, object]", raw_region)
        name = str(region.get("name", "region"))
        source_regions = region.get("source_regions", [])
        if isinstance(source_regions, list):
            source_label = ", ".join(str(value) for value in source_regions)
        else:
            source_label = str(source_regions)
        rows.append(
            "<tr>"
            f"<td>{html.escape(name.replace('_', ' ').title())}</td>"
            f"<td>{html.escape(str(region.get('semantic', '')))}</td>"
            f"<td>{html.escape(str(region.get('vertex_count', '')))}</td>"
            f"<td>{html.escape(str(region.get('face_count', '')))}</td>"
            f"<td>{html.escape(str(region.get('reconstruction_role', '')))}</td>"
            f"<td>{html.escape(source_label)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    missing = validation.get("missing_required_regions", [])
    missing_note = ""
    if isinstance(missing, list) and missing:
        missing_note = (
            "<p class=\"face-preview-notes fail\">"
            f"Missing required regions: {html.escape(', '.join(str(value) for value in missing))}"
            "</p>"
        )
    return f"""
    <section class="panel face-preview-panel">
      <div class="panel-header">
        <h2>CatTemplate v2 Regions</h2>
        <span>{html.escape(status)}</span>
      </div>
      <p class="subtle">
        Fixed-topology anatomical contract embedded in the exported GLB.
        <a href="{validation_href}">Mesh Validation JSON</a>
      </p>
      <dl class="face-preview-metrics template-region-summary">{summary}</dl>
      {missing_note}
      <table class="face-bake-table">
        <thead>
          <tr>
            <th>Region</th>
            <th>Semantic</th>
            <th>Vertices</th>
            <th>Faces</th>
            <th>Role</th>
            <th>Source Regions</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _source_frame_acquisition_gap_panel(
    project_dir: Path,
    *,
    job_id: str,
    artifacts: dict[str, object],
) -> str:
    report_path = str(
        artifacts.get(
            "source_frame_acquisition_gap",
            project_dir / "outputs" / "quality" / "source_frame_acquisition_gap.json",
        )
    )
    report_href = _project_file_href(project_dir, job_id=job_id, raw_path=report_path)
    if not report_href:
        return ""
    path = Path(report_path)
    if not path.is_absolute():
        path = project_dir / path
    report = _read_json_object(path)
    if report is None:
        return ""
    return _source_frame_acquisition_gap_report_panel(
        report,
        report_href=report_href,
    )


def _source_frame_acquisition_gap_preflight_panel(
    report: dict[str, object],
    *,
    loop_dir: Path,
) -> str:
    report_path = _source_frame_acquisition_gap_path_from_loop_report(
        report,
        loop_dir=loop_dir,
    )
    if report_path is None:
        return ""
    gap_report = _read_json_object(report_path)
    if gap_report is None:
        return ""
    return _source_frame_acquisition_gap_report_panel(gap_report, report_href="")


def _source_frame_acquisition_gap_path_from_loop_report(
    report: dict[str, object],
    *,
    loop_dir: Path,
) -> Path | None:
    candidates: list[object] = []
    for key in (
        "source_frame_acquisition_gap",
        "source_frame_acquisition_gap_report",
    ):
        value = report.get(key)
        if isinstance(value, dict):
            for path_key in ("report_path", "path", "artifact_path"):
                if value.get(path_key):
                    candidates.append(value[path_key])
        elif value:
            candidates.append(value)
    artifacts = report.get("artifacts", {})
    if isinstance(artifacts, dict):
        for key in (
            "source_frame_acquisition_gap",
            "source_frame_acquisition_gap_report",
        ):
            if artifacts.get(key):
                candidates.append(artifacts[key])
    candidates.extend(
        [
            loop_dir / "source_frame_acquisition_gap.json",
            loop_dir / "source_frame_acquisition_gap_report.json",
        ]
    )
    for raw_path in candidates:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = loop_dir / path
        if path.is_file():
            return path
    return None


def _read_json_object(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _source_frame_acquisition_gap_report_panel(
    report: dict[str, object],
    *,
    report_href: str,
) -> str:
    if str(report.get("source", "")) != "opencat3d.evaluation.evidence_gap":
        return ""
    stop_condition = report.get("stop_condition", {})
    if not isinstance(stop_condition, dict):
        stop_condition = {}
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    run_now = bool(stop_condition.get("run_reconstruction_now", False))
    reason = str(stop_condition.get("reason", "unknown"))
    status = "Ready" if run_now else "Run Later"
    note_class = "pass" if run_now else "fail"
    report_link = (
        f' <a href="{report_href}">Gap JSON</a>' if report_href else ""
    )
    summary = "".join(
        [
            _face_preview_metric("Run Reconstruction", "yes" if run_now else "no"),
            _face_preview_metric("Stop Reason", reason),
            _face_preview_metric("New Unique", metrics.get("new_unique_frame_count")),
            _face_preview_metric("New Duplicates", metrics.get("new_duplicate_frame_count")),
            _face_preview_metric(
                "Face Coverage",
                _format_percent(metrics.get("observed_face_fraction")),
            ),
            _face_preview_metric(
                "Vertex Coverage",
                _format_percent(metrics.get("observed_vertex_fraction")),
            ),
            _face_preview_metric(
                "Silhouette IoU",
                _format_score(metrics.get("overlay_mean_silhouette_iou")),
            ),
            _face_preview_metric(
                "Keypoint Loss",
                _format_score(metrics.get("joint_projection_mean_keypoint_loss")),
            ),
        ]
    )
    metric_rows = _source_frame_acquisition_metric_gap_rows(report)
    view_rows = _source_frame_acquisition_view_gap_rows(report)
    recommendation_rows = _source_frame_acquisition_recommendation_rows(report)
    return f"""
    <section class="panel face-preview-panel">
      <div class="panel-header">
        <h2>Source Frame Acquisition Gap</h2>
        <span>{html.escape(status)}</span>
      </div>
      <p class="subtle">
        Current observed-evidence gaps before spending another full reconstruction run.
        {report_link}
      </p>
      <dl class="face-preview-metrics template-region-summary">{summary}</dl>
      <p class="face-preview-notes {note_class}">
        {html.escape(_source_frame_acquisition_status_message(run_now, reason))}
      </p>
      {metric_rows}
      {view_rows}
      {recommendation_rows}
    </section>"""


def _source_frame_acquisition_status_message(run_now: bool, reason: str) -> str:
    if run_now:
        return "The current capture set has enough new evidence for a reconstruction run."
    return (
        "Do not run full reconstruction yet; collect the missing observed views first. "
        f"Reason: {reason}."
    )


def _source_frame_acquisition_metric_gap_rows(report: dict[str, object]) -> str:
    gaps = report.get("metric_gaps", [])
    if not isinstance(gaps, list) or not gaps:
        return ""
    rows = []
    for raw_gap in gaps[:12]:
        if not isinstance(raw_gap, dict):
            continue
        gap = cast("dict[str, object]", raw_gap)
        metric = str(gap.get("metric", "metric"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(metric)}</td>"
            f"<td>{html.escape(_format_acquisition_gap_value(metric, gap.get('current')))}</td>"
            f"<td>{html.escape(_format_acquisition_gap_value(metric, gap.get('target')))}</td>"
            f"<td>{html.escape(_format_acquisition_gap_value(metric, gap.get('gap')))}</td>"
            f"<td>{html.escape(str(gap.get('direction', '')))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
      <h3 class="face-bake-subheading">Metric Gaps</h3>
      <table class="face-bake-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Current</th>
            <th>Target</th>
            <th>Gap</th>
            <th>Direction</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>"""


def _source_frame_acquisition_view_gap_rows(report: dict[str, object]) -> str:
    gaps = report.get("view_gaps", [])
    if not isinstance(gaps, list) or not gaps:
        return ""
    rows = []
    for raw_gap in gaps:
        if not isinstance(raw_gap, dict):
            continue
        gap = cast("dict[str, object]", raw_gap)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(gap.get('view_group', 'view')))}</td>"
            f"<td>{html.escape(str(gap.get('current', '')))}</td>"
            f"<td>{html.escape(str(gap.get('target', '')))}</td>"
            f"<td>{html.escape(str(gap.get('missing', '')))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
      <h3 class="face-bake-subheading">View Gaps</h3>
      <table class="face-bake-table">
        <thead>
          <tr>
            <th>View</th>
            <th>Current</th>
            <th>Target</th>
            <th>Missing</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>"""


def _source_frame_acquisition_recommendation_rows(report: dict[str, object]) -> str:
    recommendations = report.get("recommendations", [])
    if not isinstance(recommendations, list) or not recommendations:
        return ""
    items = []
    for raw_recommendation in recommendations[:4]:
        if not isinstance(raw_recommendation, dict):
            continue
        recommendation = cast("dict[str, object]", raw_recommendation)
        instructions = recommendation.get("capture_instructions", [])
        first_instruction = ""
        if isinstance(instructions, list) and instructions:
            first_instruction = f" {instructions[0]}"
        items.append(
            "<li>"
            f"<strong>{html.escape(str(recommendation.get('id', 'recommendation')))}</strong>: "
            f"{html.escape(str(recommendation.get('reason', '')))}"
            f"{html.escape(first_instruction)}"
            "</li>"
        )
    if not items:
        return ""
    return f"""
      <h3 class="face-bake-subheading">Capture Recommendations</h3>
      <ul>{''.join(items)}</ul>"""


def _face_texture_preview_panel(
    project_dir: Path,
    *,
    job_id: str,
    artifacts: dict[str, object],
) -> str:
    report_path = str(
        artifacts.get(
            "face_texture_diagnostics",
            project_dir / "outputs" / "quality" / "raw_face_texture_diagnostics.json",
        )
    )
    report_href = _project_file_href(project_dir, job_id=job_id, raw_path=report_path)
    if not report_href:
        return ""
    path = Path(report_path)
    if not path.is_absolute():
        path = project_dir / path
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    regions = report.get("regions", [])
    if not isinstance(regions, list):
        return ""

    cards = []
    for raw_region in regions:
        if not isinstance(raw_region, dict):
            continue
        region = cast("dict[str, object]", raw_region)
        name = str(region.get("name", "region"))
        texture_href = _project_file_href(
            project_dir,
            job_id=job_id,
            raw_path=str(region.get("texture_crop_path", "")),
        )
        source_href = _project_file_href(
            project_dir,
            job_id=job_id,
            raw_path=str(region.get("source_crop_path", "")),
        )
        if not texture_href and not source_href:
            continue
        confident_texels = _format_percent(
            region.get("texture_confident_texel_fraction")
        )
        source_frame = region.get("dominant_source_frame_id") or region.get(
            "source_frame_id"
        )
        visual_score = _format_score(region.get("texture_visual_quality_score"))
        cards.append(
            f"""
        <article class="face-preview-card">
          <h3>{html.escape(name.replace("_", " ").title())}</h3>
          <div class="face-preview-images">
            {_face_preview_image(texture_href, f"{name} texture crop")}
            {_face_preview_image(source_href, f"{name} source crop")}
          </div>
          <dl class="face-preview-metrics">
            {_face_preview_metric("Visual Score", visual_score)}
            {_face_preview_metric("Visual Gate", _face_visual_quality_status(region))}
            {_face_preview_metric("Unique RGB", region.get("texture_unique_rgb_count"))}
            {_face_preview_metric("Confident Texels", confident_texels)}
            {_face_preview_metric("Exact Source", region.get("source_crop_exact_patch_used"))}
            {_face_preview_metric("Isolated", region.get("texture_crop_isolated"))}
            {_face_preview_metric("Frame", source_frame)}
          </dl>
          {_face_preview_quality_notes(region)}
        </article>"""
        )
    if not cards:
        return ""
    return f"""
    <section class="panel face-preview-panel">
      <div class="panel-header">
        <h2>Face Diagnostics</h2>
        <a href="{report_href}">Diagnostics JSON</a>
      </div>
      <p class="subtle">Observed texture crops beside their selected source patches.</p>
      <div class="face-preview-grid">{''.join(cards)}
      </div>
    </section>"""


def _texture_source_acquisition_panel(
    project_dir: Path,
    *,
    job_id: str,
    artifacts: dict[str, object],
) -> str:
    report_path = str(
        artifacts.get(
            "texture_source_preflight",
            project_dir / "outputs" / "quality" / "texture_source_preflight.json",
        )
    )
    report_href = _project_file_href(project_dir, job_id=job_id, raw_path=report_path)
    if not report_href:
        return ""
    path = Path(report_path)
    if not path.is_absolute():
        path = project_dir / path
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(report, dict):
        return ""
    hint = _texture_source_acquisition_hint(report)
    if not hint:
        return ""

    status = "Ready" if hint["accepted_frame_count"] > 0 else "Needs media"
    metrics = "".join(
        [
            _face_preview_metric("Source Frames", hint["source_frame_count"]),
            _face_preview_metric("Accepted Unique", hint["accepted_frame_count"]),
            _face_preview_metric("Duplicates", hint["duplicate_frame_count"]),
            _face_preview_metric("Rejected", hint["rejected_frame_count"]),
            _face_preview_metric("Best Non-Duplicate", hint["best_non_duplicate_score"]),
            _face_preview_metric("Required Score", hint["required_score"]),
            _face_preview_metric("Best Frame", hint["best_non_duplicate_frame_id"]),
            _face_preview_metric("Missing View", hint["missing_view_type"]),
        ]
    )
    note_class = "pass" if hint["accepted_frame_count"] > 0 else "fail"
    return f"""
    <section class="panel face-preview-panel">
      <div class="panel-header">
        <h2>Media Needed</h2>
        <span>{html.escape(status)}</span>
      </div>
      <p class="subtle">
        Texture-only candidates are gated by observed face detail before they can update
        eyes, nose, muzzle, and ear interiors.
        <a href="{report_href}">Texture Source Preflight JSON</a>
      </p>
      <dl class="face-preview-metrics template-region-summary">{metrics}</dl>
      <p class="face-preview-notes {note_class}">
        {html.escape(hint["message"])}
      </p>
    </section>"""


def _texture_source_acquisition_hint(report: dict[str, object]) -> dict[str, object] | None:
    ranked_frames = report.get("ranked_frames", [])
    if not isinstance(ranked_frames, list):
        ranked_frames = []
    accepted_count = _non_negative_int(report.get("accepted_frame_count"))
    duplicate_count = _non_negative_int(report.get("duplicate_frame_count"))
    rejected_count = _non_negative_int(report.get("rejected_frame_count"))
    source_count = _non_negative_int(report.get("source_frame_count"))
    best_non_duplicate = _best_ranked_frame(ranked_frames, duplicate=False)
    best_duplicate = _best_ranked_frame(ranked_frames, duplicate=True)
    best_frame = best_non_duplicate or best_duplicate
    if best_frame is None and accepted_count > 0:
        return None

    required_score = _required_texture_source_score(best_frame)
    best_score = _numeric(best_non_duplicate.get("rank_score") if best_non_duplicate else None)
    best_frame_id = str(best_non_duplicate.get("source_frame_id", "")) if best_non_duplicate else ""
    missing_view_type = (
        "non-duplicate frontal close-up of eyes, nose, muzzle, and ear interiors"
    )
    if accepted_count > 0:
        message = (
            "Texture preflight has accepted unique face-detail evidence. "
            "Another texture-only bake can use the accepted sources."
        )
    else:
        message = (
            "Before another texture-only bake, add non-duplicate frontal close-up "
            "photos or a slow face video where both eyes, nose, muzzle, and ear "
            "interiors fill the frame."
        )
        if best_non_duplicate is not None:
            message += (
                f" Best non-duplicate frame {best_frame_id} scored "
                f"{best_score:.3f}, below the required {required_score:.3f}."
            )
        if duplicate_count:
            message += (
                f" {duplicate_count} high-overlap frame"
                f"{'s were' if duplicate_count != 1 else ' was'} already covered by "
                "the baseline, so new angles must not be repeated uploads."
            )
        if rejected_count:
            message += (
                f" {rejected_count} frame"
                f"{'s' if rejected_count != 1 else ''} failed face-detail quality."
            )

    return {
        "accepted_frame_count": accepted_count,
        "best_non_duplicate_frame_id": best_frame_id,
        "best_non_duplicate_score": _format_score(best_score),
        "duplicate_frame_count": duplicate_count,
        "message": message,
        "missing_view_type": missing_view_type,
        "rejected_frame_count": rejected_count,
        "required_score": _format_score(required_score),
        "source_frame_count": source_count,
    }


def _best_ranked_frame(
    ranked_frames: list[object],
    *,
    duplicate: bool,
) -> dict[str, object] | None:
    for raw_frame in ranked_frames:
        if not isinstance(raw_frame, dict):
            continue
        frame = cast("dict[str, object]", raw_frame)
        if bool(frame.get("duplicate", False)) == duplicate:
            return frame
    return None


def _required_texture_source_score(frame: dict[str, object] | None) -> float:
    if frame is None:
        return 0.48
    quality_gate = frame.get("quality_gate", {})
    if not isinstance(quality_gate, dict):
        return 0.48
    return _numeric(quality_gate.get("face_texture_detail_score_floor")) or 0.48


def _face_texture_bake_panel(project_dir: Path, *, job_id: str) -> str:
    manifest_path = project_dir / "outputs" / "textures" / "texture_manifest.json"
    report_href = _project_file_href(
        project_dir,
        job_id=job_id,
        raw_path=str(manifest_path),
    )
    if not report_href:
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(manifest, dict):
        return ""
    facial_features = manifest.get("facial_features", {})
    if not isinstance(facial_features, dict):
        return ""
    candidates_raw = facial_features.get("source_patch_bake_candidates", [])
    if not isinstance(candidates_raw, list) or not candidates_raw:
        return ""

    rows = []
    candidates = []
    for raw_candidate in candidates_raw:
        if not isinstance(raw_candidate, dict):
            continue
        candidate = cast("dict[str, object]", raw_candidate)
        candidates.append(candidate)
        label = str(candidate.get("label", "candidate"))
        selected = bool(candidate.get("selected", False))
        passed_count = html.escape(
            str(candidate.get("visual_quality_passed_count", ""))
        )
        min_score = html.escape(
            _format_score(candidate.get("visual_quality_min_score"))
        )
        mean_score = html.escape(
            _format_score(candidate.get("visual_quality_mean_score"))
        )
        min_coverage = html.escape(
            _format_percent(candidate.get("min_confident_texel_fraction"))
        )
        source_detail = html.escape(
            _format_score(candidate.get("mean_source_luma_std"))
        )
        feedback_used = html.escape(str(candidate.get("feedback_used_count", 0)))
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{'Selected' if selected else 'Compared'}</td>"
            f"<td>{passed_count}</td>"
            f"<td>{min_score}</td>"
            f"<td>{mean_score}</td>"
            f"<td>{min_coverage}</td>"
            f"<td>{source_detail}</td>"
            f"<td>{feedback_used}</td>"
            "</tr>"
        )
    if not rows:
        return ""

    selected_label = str(
        facial_features.get("source_patch_bake_selected_candidate", "unknown")
    )
    selected_feedback = bool(
        facial_features.get("source_patch_bake_selected_feedback", False)
    )
    selected_summary = (
        f"{selected_label} / feedback"
        if selected_feedback
        else f"{selected_label} / no feedback"
    )
    region_comparison = _face_bake_region_comparison(candidates)
    return f"""
    <section class="panel face-preview-panel">
      <div class="panel-header">
        <h2>Face Bake Candidates</h2>
        <span>{html.escape(selected_summary)}</span>
      </div>
      <p class="subtle">
        Observed-pixel source patch candidates compared before the final face atlas write.
        <a href="{report_href}">Texture Manifest JSON</a>
      </p>
      <table class="face-bake-table">
        <thead>
          <tr>
            <th>Candidate</th>
            <th>Status</th>
            <th>Regions Passed</th>
            <th>Min Score</th>
            <th>Mean Score</th>
            <th>Min Coverage</th>
            <th>Source Detail</th>
            <th>Feedback Used</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      {region_comparison}
    </section>"""


def _face_bake_region_comparison(candidates: list[dict[str, object]]) -> str:
    if len(candidates) < 2:
        return ""
    baseline = _face_bake_candidate_by_label(candidates, "generic") or candidates[0]
    selected = next(
        (candidate for candidate in candidates if bool(candidate.get("selected", False))),
        candidates[-1],
    )
    if selected is baseline:
        return ""

    baseline_label = str(baseline.get("label", "baseline"))
    selected_label = str(selected.get("label", "selected"))
    baseline_regions = _face_bake_regions_by_name(baseline)
    selected_regions = _face_bake_regions_by_name(selected)
    ordered_names = [
        name
        for name in [
            *selected_regions.keys(),
            *baseline_regions.keys(),
        ]
        if name in selected_regions and name in baseline_regions
    ]
    ordered_names = list(dict.fromkeys(ordered_names))
    rows = []
    for name in ordered_names:
        before = baseline_regions[name]
        after = selected_regions[name]
        score_delta = _numeric(after.get("score")) - _numeric(before.get("score"))
        coverage_delta = _numeric(after.get("confident_texel_fraction")) - _numeric(
            before.get("confident_texel_fraction")
        )
        source_detail_delta = _numeric(after.get("source_luma_std")) - _numeric(
            before.get("source_luma_std")
        )
        decision_driver = html.escape(
            _face_bake_region_driver(
                score_delta,
                coverage_delta,
                source_detail_delta,
            )
        )
        weakest_signal = html.escape(
            _face_reason_label(str(after.get("weakest_component", "")))
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(name.replace('_', ' ').title())}</td>"
            f"<td>{html.escape(_format_score(before.get('score')))}</td>"
            f"<td>{html.escape(_format_score(after.get('score')))}</td>"
            f"<td>{html.escape(_format_signed_score_delta(score_delta))}</td>"
            f"<td>{html.escape(_format_signed_percent_delta(coverage_delta))}</td>"
            f"<td>{html.escape(_format_signed_score_delta(source_detail_delta))}</td>"
            f"<td>{decision_driver}</td>"
            f"<td>{weakest_signal}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
      <h3 class="face-bake-subheading">Region Comparison</h3>
      <p class="subtle">
        {html.escape(selected_label)} compared against {html.escape(baseline_label)}.
      </p>
      <table class="face-bake-table">
        <thead>
          <tr>
            <th>Region</th>
            <th>{html.escape(baseline_label)} Score</th>
            <th>{html.escape(selected_label)} Score</th>
            <th>Score Delta</th>
            <th>Coverage Delta</th>
            <th>Source Detail Delta</th>
            <th>Decision Driver</th>
            <th>Weakest Signal</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>"""


def _face_bake_candidate_by_label(
    candidates: list[dict[str, object]],
    label: str,
) -> dict[str, object] | None:
    for candidate in candidates:
        if str(candidate.get("label", "")) == label:
            return candidate
    return None


def _face_bake_regions_by_name(
    candidate: dict[str, object],
) -> dict[str, dict[str, object]]:
    regions_raw = candidate.get("regions", [])
    if not isinstance(regions_raw, list):
        return {}
    regions = {}
    for raw_region in regions_raw:
        if not isinstance(raw_region, dict):
            continue
        region = cast("dict[str, object]", raw_region)
        name = str(region.get("name", ""))
        if name:
            regions[name] = region
    return regions


def _face_bake_region_driver(
    score_delta: float,
    coverage_delta: float,
    source_detail_delta: float,
) -> str:
    if score_delta > 0.001:
        return f"Score {_format_signed_score_delta(score_delta)}"
    if coverage_delta > 0.001:
        return f"Coverage {_format_signed_percent_delta(coverage_delta)}"
    if source_detail_delta > 0.001:
        return f"Source detail {_format_signed_score_delta(source_detail_delta)}"
    if score_delta < -0.001:
        return f"Score {_format_signed_score_delta(score_delta)}"
    if coverage_delta < -0.001:
        return f"Coverage {_format_signed_percent_delta(coverage_delta)}"
    if source_detail_delta < -0.001:
        return f"Source detail {_format_signed_score_delta(source_detail_delta)}"
    return "No change"


def _face_preview_image(href: str, label: str) -> str:
    if not href:
        return ""
    return f"""
            <figure>
              <img src="{href}" alt="{html.escape(label)}" loading="lazy">
              <figcaption>{html.escape(label)}</figcaption>
            </figure>"""


def _face_preview_metric(label: str, value: object) -> str:
    if value is None or value == "":
        return ""
    return (
        f"<div><dt>{html.escape(label)}</dt>"
        f"<dd>{html.escape(str(value))}</dd></div>"
    )


def _face_preview_quality_notes(region: dict[str, object]) -> str:
    reasons = _face_visual_quality_reasons(region)
    weakest = _face_reason_label(
        str(region.get("texture_visual_quality_weakest_component", ""))
    )
    if not reasons and bool(region.get("texture_visual_quality_passed", False)):
        message = f"Pass. Weakest signal: {weakest}." if weakest else "Pass."
        return f'<p class="face-preview-notes pass">{html.escape(message)}</p>'
    if not reasons:
        return ""
    items = "".join(f"<li>{html.escape(_face_reason_label(reason))}</li>" for reason in reasons)
    weakest_line = (
        f"<p>Weakest signal: {html.escape(weakest)}.</p>" if weakest else ""
    )
    return f"""
          <div class="face-preview-notes fail">
            <strong>Quality Issues</strong>
            <ul>{items}</ul>
            {weakest_line}
          </div>"""


def _face_visual_quality_reasons(region: dict[str, object]) -> list[str]:
    raw_reasons = region.get("texture_visual_quality_failure_reasons", [])
    if not isinstance(raw_reasons, list):
        return []
    return [str(reason) for reason in raw_reasons if reason]


def _face_visual_quality_status(region: dict[str, object]) -> str:
    value = region.get("texture_visual_quality_passed")
    if value is True:
        return "Passed"
    if value is False:
        return "Failed"
    return ""


def _face_reason_label(reason: str) -> str:
    return {
        "coverage": "Coverage too sparse",
        "flat_texture": "Texture is too flat",
        "source_detail": "Source crop lacks detail",
        "source_evidence": "Source crop missing",
        "combined_score": "Combined score too low",
        "confident_texel_score": "Coverage",
        "texture_luma_score": "Texture contrast",
        "unique_rgb_score": "Color variety",
        "source_luma_score": "Source detail",
        "source_evidence_score": "Source evidence",
    }.get(reason, reason.replace("_", " ").title())


def _format_percent(value: object) -> str:
    if not isinstance(value, int | float):
        return ""
    return f"{value:.1%}"


def _format_score(value: object) -> str:
    if not isinstance(value, int | float):
        return ""
    return f"{value:.3f}"


def _format_acquisition_gap_value(metric: str, value: object) -> str:
    if not isinstance(value, int | float):
        return ""
    if metric in {
        "basecolor_observed_texel_fraction",
        "observed_face_fraction",
        "observed_vertex_fraction",
        "overlay_mean_silhouette_iou",
        "silhouette_residual_mean_union_error",
    }:
        return _format_percent(value)
    if float(value).is_integer():
        return str(int(value))
    return _format_score(value)


def _format_signed_score_delta(value: float) -> str:
    if abs(value) < 0.0005:
        return "0.000"
    return f"{value:+.3f}"


def _format_signed_percent_delta(value: float) -> str:
    if abs(value) < 0.0005:
        return "0.0 pp"
    return f"{value * 100:+.1f} pp"


def _numeric(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _non_negative_int(value: object) -> int:
    if isinstance(value, int | float):
        return max(0, int(value))
    return 0


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".glb":
        return "model/gltf-binary"
    if suffix in {".obj", ".ply"}:
        return "text/plain"
    return "application/octet-stream"


STYLE = """<style>
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  background: #f5f7f8;
  color: #162024;
}
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; }
.app-shell { max-width: 1120px; margin: 0 auto; padding: 28px; }
.toolbar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  border-bottom: 1px solid #d8e0e3; padding-bottom: 18px; margin-bottom: 20px;
}
h1, h2, p { margin: 0; }
h1 { font-size: 28px; line-height: 1.1; }
h2 { font-size: 18px; }
p, small { color: #55666d; }
.subtle { color: #6b7b81; font-size: 13px; margin-bottom: 12px; }
.health, button, .artifact-list a {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  min-height: 38px; padding: 0 14px; border-radius: 6px; border: 1px solid #9ab0b8;
  background: #ffffff; color: #162024; text-decoration: none; font-weight: 650;
}
.workspace, .result-grid { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 14px; }
.result-grid { grid-template-columns: 1fr 1fr; }
.review-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
.panel {
  background: #ffffff; border: 1px solid #d8e0e3; border-radius: 8px; padding: 16px;
  min-width: 0;
}
.primary { border-color: #7aa2ad; }
.panel-header {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  margin-bottom: 14px;
}
.panel-header span {
  font-size: 12px; border: 1px solid #d8e0e3; border-radius: 999px; padding: 4px 8px;
  color: #44545a; white-space: nowrap;
}
.drop-zone {
  display: grid; gap: 4px; min-height: 138px; padding: 20px; place-items: center;
  border: 1px dashed #7aa2ad; border-radius: 8px; background: #f8fbfb; text-align: center;
  cursor: pointer;
}
.drop-zone.compact { min-height: 96px; }
.native-file-input {
  position: absolute; width: 1px; height: 1px; overflow: hidden; opacity: 0;
}
.staged-files { display: grid; gap: 8px; margin-top: 12px; }
.staged-file-header {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
}
.file-list {
  display: grid; gap: 6px; list-style: none; margin: 0; max-height: 180px; overflow: auto;
  padding: 0;
}
.file-row, .empty-file-row {
  align-items: center; border: 1px solid #d8e0e3; border-radius: 6px; display: grid;
  gap: 8px; min-height: 34px; padding: 6px 8px;
}
.file-row { grid-template-columns: minmax(0, 1fr) auto auto; }
.file-row span {
  min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.file-row small { white-space: nowrap; }
.empty-file-row { color: #6b7b81; }
.controls { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 14px 0; }
.action-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
label { display: grid; gap: 6px; font-size: 13px; font-weight: 650; color: #34444a; }
input[type="number"] {
  width: 100%; min-height: 36px; border: 1px solid #b9c7cc; border-radius: 6px;
  padding: 0 10px;
}
input[type="text"], input[type="datetime-local"] {
  width: 100%; min-height: 36px; border: 1px solid #b9c7cc; border-radius: 6px;
  padding: 0 10px;
}
button { width: 100%; background: #244f5a; color: #ffffff; border-color: #244f5a; cursor: pointer; }
button:disabled { cursor: not-allowed; opacity: 0.62; }
.secondary-button, .remove-file {
  background: #ffffff; color: #162024; width: auto;
}
.remove-file {
  min-height: 28px; min-width: 28px; padding: 0;
}
.disabled { opacity: 0.70; }
.identity-workspace { margin-top: 14px; }
.notice {
  border: 1px solid #e1a545; background: #fff8e8; padding: 12px 14px;
  border-radius: 8px; margin-bottom: 14px;
}
.artifact-list { display: flex; flex-wrap: wrap; gap: 10px; }
.face-preview-panel { margin-top: 14px; }
.face-preview-grid {
  display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.face-preview-card {
  background: #fbfdfd; border: 1px solid #d8e0e3; border-radius: 6px; display: grid;
  gap: 10px; padding: 10px;
}
.face-preview-card h3 { font-size: 14px; margin: 0; }
.face-preview-images { display: grid; gap: 8px; grid-template-columns: 1fr 1fr; }
.face-preview-images figure { margin: 0; min-width: 0; }
.face-preview-images img {
  width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #f0f4f5;
  border: 1px solid #d8e0e3; border-radius: 6px;
}
.face-preview-images figcaption {
  color: #55666d; font-size: 11px; margin-top: 4px; overflow-wrap: anywhere;
}
.face-preview-metrics {
  display: grid; gap: 4px; font-size: 12px; line-height: 1.35; margin: 0;
}
.face-preview-metrics div { display: flex; gap: 8px; justify-content: space-between; }
.face-preview-metrics dt { color: #44545a; font-weight: 650; }
.face-preview-metrics dd { margin: 0; overflow-wrap: anywhere; text-align: right; }
.face-preview-notes {
  border-radius: 6px; font-size: 12px; line-height: 1.35; margin: 0; padding: 8px;
}
.face-preview-notes.pass { background: #eef7f1; color: #244d34; }
.face-preview-notes.fail { background: #fff4e8; border: 1px solid #e1a545; color: #5c3d12; }
.face-preview-notes strong { display: block; margin-bottom: 4px; }
.face-preview-notes ul { margin: 0; padding-left: 16px; }
.face-preview-notes p { color: inherit; margin-top: 6px; }
.face-bake-subheading { font-size: 15px; margin: 14px 0 8px; }
.face-bake-table { display: block; font-size: 12px; overflow-x: auto; white-space: nowrap; }
.face-bake-table th { width: auto; }
.review-image {
  width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #f0f4f5;
  border: 1px solid #d8e0e3; border-radius: 6px; margin-bottom: 12px;
}
.compact-explanation {
  color: #44545a; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px; line-height: 1.4; margin-top: 12px; overflow-wrap: anywhere;
}
.feedback-actions { display: grid; gap: 8px; margin-top: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; border-bottom: 1px solid #e8edef; padding: 8px 4px; }
th { color: #44545a; font-weight: 650; width: 55%; }
@media (max-width: 780px) {
  .workspace, .result-grid, .review-grid { grid-template-columns: 1fr; }
  .toolbar { align-items: flex-start; flex-direction: column; }
  .controls { grid-template-columns: 1fr; }
  .action-row { grid-template-columns: 1fr; }
}
</style>"""


STAGED_UPLOAD_SCRIPT = """<script>
(() => {
  const form = document.querySelector("[data-upload-form]");
  const input = document.querySelector("[data-file-input]");
  const list = document.querySelector("[data-file-list]");
  const count = document.querySelector("[data-file-count]");
  const clearButton = document.querySelector("[data-clear-files]");
  const submitButtons = Array.from(document.querySelectorAll("[data-submit]"));
  if (!form || !input || !list || !count || !clearButton || submitButtons.length === 0) {
    return;
  }

  const stagedFiles = [];
  const imageSuffix = /\\.(jpe?g|png)$/i;
  const videoSuffix = /\\.(avi|m4v|mov|mp4)$/i;
  const supportedSuffix = /\\.(avi|jpe?g|m4v|mov|mp4|png)$/i;
  const fileKey = (file) => `${file.name}:${file.size}:${file.lastModified}`;
  const fileSize = (file) => {
    const kib = Math.max(1, Math.ceil(file.size / 1024));
    return `${kib} KB`;
  };

  const render = () => {
    list.replaceChildren();
    if (stagedFiles.length === 0) {
      const row = document.createElement("li");
      row.className = "empty-file-row";
      row.textContent = "No media selected yet.";
      list.append(row);
    }
    stagedFiles.forEach((file, index) => {
      const row = document.createElement("li");
      row.className = "file-row";

      const name = document.createElement("span");
      name.textContent = file.name;
      row.append(name);

      const size = document.createElement("small");
      size.textContent = fileSize(file);
      row.append(size);

      const remove = document.createElement("button");
      remove.className = "remove-file";
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${file.name}`);
      remove.textContent = "x";
      remove.addEventListener("click", () => {
        stagedFiles.splice(index, 1);
        render();
      });
      row.append(remove);
      list.append(row);
    });

    const selectedText = stagedFiles.length === 1 ? "1 selected" : `${stagedFiles.length} selected`;
    const imageCount = stagedFiles.filter((file) => imageSuffix.test(file.name)).length;
    const videoCount = stagedFiles.filter((file) => videoSuffix.test(file.name)).length;
    const canSubmit = videoCount >= 1 || imageCount >= 2;
    count.textContent = `${selectedText}. Supported: .avi, .jpeg, .jpg, .m4v, .mov, .mp4, .png`;
    submitButtons.forEach((button) => {
      button.disabled = !canSubmit;
      button.textContent = !canSubmit
        ? "Add Photos Or Video"
        : (button.dataset.readyLabel || "Generate 3D Model");
    });
  };

  input.addEventListener("change", () => {
    const existingKeys = new Set(stagedFiles.map(fileKey));
    Array.from(input.files || []).forEach((file) => {
      if (!supportedSuffix.test(file.name)) {
        return;
      }
      const key = fileKey(file);
      if (!existingKeys.has(key)) {
        stagedFiles.push(file);
        existingKeys.add(key);
      }
    });
    input.value = "";
    render();
  });

  clearButton.addEventListener("click", () => {
    stagedFiles.splice(0, stagedFiles.length);
    input.value = "";
    render();
  });

  form.addEventListener("submit", (event) => {
    const imageCount = stagedFiles.filter((file) => imageSuffix.test(file.name)).length;
    const videoCount = stagedFiles.filter((file) => videoSuffix.test(file.name)).length;
    if (videoCount < 1 && imageCount < 2) {
      event.preventDefault();
      count.textContent = "Add at least two photos or one video before generating.";
      return;
    }
    const transfer = new DataTransfer();
    stagedFiles.forEach((file) => transfer.items.add(file));
    input.files = transfer.files;
    const submitter = event.submitter || submitButtons[0];
    submitButtons.forEach((button) => {
      button.disabled = true;
    });
    submitter.textContent = submitter.getAttribute("formaction") === "/jobs/preflight"
      ? "Checking..."
      : "Generating...";
  });

  render();
})();
</script>"""
