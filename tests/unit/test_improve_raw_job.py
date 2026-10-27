import json
import sys
from types import SimpleNamespace

from PIL import Image

from opencat3d.ingest.images import ingest_image_sources
from scripts import improve_raw_job
from scripts.improve_raw_job import _merge_supervisor_results


def _write_stage_manifest(tmp_path, stage_name: str, *, variant: int = 0):
    stage_dir = tmp_path / stage_name
    stage_dir.mkdir(parents=True)
    frame_specs = (
        ("frame_0000.jpg", (235 + variant, 235, 235), "photo", "image", "14.jpg", None),
        ("frame_0001.jpg", (225 + variant, 225, 225), "photo", "image", "11.jpg", None),
        ("frame_0002.jpg", (70 + variant, 90, 110), "video", "video_frame", "v1.mp4", 17),
    )
    frames = {}
    frame_paths = []
    for filename, color, media_type, source_type, source_media_id, frame_index in frame_specs:
        path = stage_dir / filename
        Image.new("RGB", (8, 6), color=color).save(path)
        frame_paths.append(path)
        metadata = {
            "media_type": media_type,
            "source_type": source_type,
            "source_media_id": source_media_id,
        }
        if frame_index is not None:
            metadata["source_frame_index"] = frame_index
        frames[filename] = metadata
    manifest_path = stage_dir / "media_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "test",
                "frames": frames,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return stage_dir, frame_paths, manifest_path


def _copy_stage_manifest_with_bytes(
    source_stage_dir,
    target_stage_dir,
    source_manifest_path,
):
    target_stage_dir.mkdir(parents=True)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    for filename in manifest["frames"]:
        (target_stage_dir / filename).write_bytes((source_stage_dir / filename).read_bytes())
    target_manifest_path = target_stage_dir / "media_manifest.json"
    target_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_manifest_path


def _candidate_from_stage(stage_dir, candidate_id: str):
    frames = (
        improve_raw_job.FrameSelection(
            frame_id="frame_0000",
            source_path=stage_dir / "frame_0000.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="14.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.20,
            score=0.96,
            mask_area_ratio=0.16,
            bbox_area_ratio=0.30,
            full_frame_bbox=False,
            raw_view_group="frontal_face",
            raw_pose_label="closeup_face",
            raw_frontal_face_confidence=0.92,
            raw_view_weight=0.92,
        ),
        improve_raw_job.FrameSelection(
            frame_id="frame_0001",
            source_path=stage_dir / "frame_0001.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="11.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.88,
            score=0.80,
            mask_area_ratio=0.18,
            bbox_area_ratio=0.34,
            full_frame_bbox=False,
            raw_view_group="side_body",
            raw_pose_label="side_body",
            raw_frontal_face_confidence=0.20,
            raw_view_weight=0.50,
        ),
        improve_raw_job.FrameSelection(
            frame_id="frame_0002",
            source_path=stage_dir / "frame_0002.jpg",
            media_type="video",
            source_type="video_frame",
            source_media_id="v1.mp4",
            source_frame_index=17,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.78,
            score=0.76,
            mask_area_ratio=0.17,
            bbox_area_ratio=0.32,
            full_frame_bbox=False,
            raw_view_group="oblique_body",
            raw_pose_label="video_oblique_body",
            raw_frontal_face_confidence=0.35,
            raw_view_weight=0.58,
        ),
    )
    return improve_raw_job.SegmentationCandidate(
        candidate_id=candidate_id,
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=stage_dir / "seg_candidate",
        selected_frames=frames,
        frame_count=len(frames),
        selected_frame_count=len(frames),
        mean_mask_area_ratio=0.17,
        mean_bbox_area_ratio=0.32,
        full_frame_bbox_fraction=0.0,
        score=1.0,
        texture_frames=frames,
    )


def _write_segmentation_cache_report(tmp_path, cache_manifest_path, candidate):
    loop_dir = tmp_path / "cache_loop"
    loop_dir.mkdir(parents=True)
    preflight_report_path = loop_dir / "raw_media_preflight_report.json"
    preflight_report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frames": [
                    improve_raw_job._frame_selection_json(frame)
                    for frame in candidate.selected_frames
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cache_report_path = loop_dir / "raw_improvement_loop_report.json"
    cache_report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "media_staging": {"report_path": str(cache_manifest_path)},
                "media_preflight": {
                    "report_path": str(preflight_report_path),
                    "min_geometry_frames": 2,
                    "max_geometry_frames": 3,
                },
                "segmentation_candidates": [
                    improve_raw_job._segmentation_candidate_json(candidate)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return cache_report_path


def test_merge_supervisor_results_adds_scores_and_metrics() -> None:
    pipeline_reports = [
        {
            "candidate_id": "candidate_a",
            "metrics": {"overlay_mean_silhouette_iou": 0.2},
        },
        {
            "candidate_id": "candidate_b",
            "metrics": {"overlay_mean_silhouette_iou": 0.3},
        },
    ]
    supervisor_report = {
        "candidates": [
            {
                "label": "candidate_b",
                "score": 12.5,
                "score_terms": {"face_region_detail_loss": -0.1},
                "metrics": {"raw_face_region_detail_loss": 0.13},
            },
            {
                "label": "candidate_a",
                "score": 9.0,
                "score_terms": {"face_region_detail_loss": -0.4},
                "metrics": {"raw_face_region_detail_loss": 0.44},
            },
        ]
    }

    _merge_supervisor_results(pipeline_reports, supervisor_report)

    assert pipeline_reports[1]["supervisor_score"] == 12.5
    assert pipeline_reports[1]["supervisor_score_terms"] == {"face_region_detail_loss": -0.1}
    assert pipeline_reports[1]["metrics"]["raw_face_region_detail_loss"] == 0.13
    assert pipeline_reports[0]["metrics"]["raw_face_region_detail_loss"] == 0.44


def test_merge_supervisor_results_adds_guard_reports() -> None:
    pipeline_reports = [
        {
            "candidate_id": "candidate_a",
            "metrics": {"overlay_mean_silhouette_iou": 0.2},
        },
    ]
    promotion_guard = {
        "passed": False,
        "rejection_reasons": ["body_fit_alignment_gate_passed"],
    }
    alignment_gate = {
        "passed": False,
        "rejection_reasons": ["overlay_not_lower"],
    }
    body_fit_alignment_gate = {
        "required": True,
        "passed": False,
        "rejection_reasons": ["alignment_metric_improved"],
    }
    supervisor_report = {
        "candidates": [
            {
                "label": "candidate_a",
                "score": 9.0,
                "score_terms": {},
                "metrics": {},
                "promotion_guard": promotion_guard,
                "alignment_gate": alignment_gate,
                "body_fit_alignment_gate": body_fit_alignment_gate,
            },
        ]
    }

    _merge_supervisor_results(pipeline_reports, supervisor_report)

    assert pipeline_reports[0]["promotion_guard"] == promotion_guard
    assert pipeline_reports[0]["alignment_gate"] == alignment_gate
    assert pipeline_reports[0]["body_fit_alignment_gate"] == body_fit_alignment_gate


def test_raw_frame_paths_stages_mixed_photo_video_upload(tmp_path, monkeypatch) -> None:
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    photo_path = upload_dir / "11.jpg"
    video_path = upload_dir / "v1.mp4"
    Image.new("RGB", (8, 6), color=(240, 240, 240)).save(photo_path)
    video_path.write_bytes(b"not-a-real-video")

    def fake_extract_video_frames(video_path, output_dir, *, min_frames, max_frames, force):
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for index in range(2):
            frame_path = output_dir / f"video_frame_{index:04d}.jpg"
            Image.new("RGB", (8, 6), color=(20 + index, 60, 120)).save(frame_path)
            frame_paths.append(frame_path)
        return SimpleNamespace(
            frame_paths=tuple(frame_paths),
            selected_indices=(0, 5),
        )

    monkeypatch.setattr(improve_raw_job, "extract_video_frames", fake_extract_video_frames)

    staged_paths = improve_raw_job._raw_frame_paths(
        upload_dir,
        stage_dir=tmp_path / "stage",
        min_video_frames=2,
        max_video_frames=2,
        force=True,
    )

    assert [path.name for path in staged_paths] == [
        "frame_0000.jpg",
        "frame_0001.jpg",
        "frame_0002.jpg",
    ]
    assert all(path.exists() for path in staged_paths)
    manifest = json.loads((tmp_path / "stage" / "media_manifest.json").read_text())
    assert manifest["input_photo_count"] == 1
    assert manifest["input_video_count"] == 1
    assert manifest["extracted_video_frame_count"] == 2
    assert manifest["frames"]["frame_0000.jpg"]["media_type"] == "photo"
    assert manifest["frames"]["frame_0001.jpg"]["media_type"] == "video"
    assert manifest["frames"]["frame_0001.jpg"]["source_frame_index"] == 0


def test_stage_upload_media_frames_preserves_preextracted_video_metadata(tmp_path) -> None:
    upload_dir = tmp_path / "web_upload"
    upload_dir.mkdir()
    Image.new("RGB", (8, 6), color=(240, 240, 240)).save(upload_dir / "frame_0000.jpg")
    Image.new("RGB", (8, 6), color=(80, 90, 100)).save(upload_dir / "frame_0001.jpg")
    (upload_dir / "media_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "opencat3d.web.upload_staging",
                "frames": {
                    "frame_0000.jpg": {
                        "media_type": "photo",
                        "source_type": "image",
                        "source_media_id": "11.jpg",
                    },
                    "frame_0001.jpg": {
                        "media_type": "video",
                        "source_type": "video_frame",
                        "source_media_id": "v1.mp4",
                        "source_frame_index": 17,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    staged = improve_raw_job._stage_upload_media_frames(
        upload_dir,
        stage_dir=tmp_path / "stage",
        force=True,
    )

    manifest = json.loads((tmp_path / "stage" / "media_manifest.json").read_text())
    assert staged.input_photo_count == 1
    assert staged.input_video_count == 1
    assert staged.extracted_video_frame_count == 1
    assert manifest["frames"]["frame_0000.jpg"]["media_type"] == "photo"
    video_frame = manifest["frames"]["frame_0001.jpg"]
    assert video_frame["media_type"] == "video"
    assert video_frame["source_type"] == "video_frame"
    assert video_frame["source_media_id"] == "v1.mp4"
    assert video_frame["source_frame_index"] == 17

    ingest = ingest_image_sources(tmp_path / "project", [tmp_path / "stage"], force=True)
    assert ingest.frames[1].source_type.value == "video_frame"
    assert ingest.frames[1].source_media_id == "v1.mp4"
    assert ingest.frames[1].source_frame_index == 17


def test_raw_frame_paths_merges_extra_media_with_upload(tmp_path, monkeypatch) -> None:
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    upload_photo = upload_dir / "11.jpg"
    duplicate_photo = extra_dir / "11_duplicate.jpg"
    extra_photo = extra_dir / "12.jpg"
    video_path = extra_dir / "v1.mp4"
    Image.new("RGB", (8, 6), color=(240, 240, 240)).save(upload_photo)
    Image.new("RGB", (8, 6), color=(240, 240, 240)).save(duplicate_photo)
    Image.new("RGB", (8, 6), color=(230, 230, 230)).save(extra_photo)
    video_path.write_bytes(b"not-a-real-video")

    def fake_extract_video_frames(video_path, output_dir, *, min_frames, max_frames, force):
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_path = output_dir / "video_frame_0000.jpg"
        Image.new("RGB", (8, 6), color=(20, 60, 120)).save(frame_path)
        return SimpleNamespace(
            frame_paths=(frame_path,),
            selected_indices=(7,),
        )

    monkeypatch.setattr(improve_raw_job, "extract_video_frames", fake_extract_video_frames)

    staged_paths = improve_raw_job._raw_frame_paths(
        upload_dir,
        stage_dir=tmp_path / "stage",
        extra_media_paths=(extra_dir,),
        min_video_frames=1,
        max_video_frames=1,
        force=True,
    )

    assert [path.name for path in staged_paths] == [
        "frame_0000.jpg",
        "frame_0001.jpg",
        "frame_0002.jpg",
    ]
    manifest = json.loads((tmp_path / "stage" / "media_manifest.json").read_text())
    assert manifest["input_photo_count"] == 2
    assert manifest["input_video_count"] == 1
    assert manifest["extracted_video_frame_count"] == 1
    assert manifest["frames"]["frame_0000.jpg"]["source_media_id"] == "11.jpg"
    assert manifest["frames"]["frame_0001.jpg"]["source_media_id"] == "12.jpg"
    assert manifest["frames"]["frame_0002.jpg"]["media_type"] == "video"
    assert manifest["frames"]["frame_0002.jpg"]["source_frame_index"] == 7


def test_raw_frame_paths_keeps_image_only_upload_without_stage(tmp_path) -> None:
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    first = upload_dir / "11.jpg"
    second = upload_dir / "12.png"
    Image.new("RGB", (8, 6), color=(240, 240, 240)).save(first)
    Image.new("RGB", (8, 6), color=(230, 230, 230)).save(second)

    assert improve_raw_job._raw_frame_paths(upload_dir) == [first, second]


def test_disk_preflight_report_rejects_low_space(tmp_path, monkeypatch) -> None:
    loop_dir = tmp_path / "loop"
    projects_dir = tmp_path / "projects"
    loop_dir.mkdir()
    projects_dir.mkdir()

    monkeypatch.setattr(
        improve_raw_job.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=10 * improve_raw_job.BYTES_PER_GIB,
            used=9 * improve_raw_job.BYTES_PER_GIB,
            free=1 * improve_raw_job.BYTES_PER_GIB,
        ),
    )

    report = improve_raw_job._write_disk_preflight_report(
        loop_dir,
        projects_dir=projects_dir,
        min_free_gib=5.0,
        cleanup_intermediates=False,
        force=True,
    )

    assert report["passed"] is False
    assert report["free_gib"] == 1.0
    assert report["min_free_gib"] == 5.0
    assert report["cleanup"]["removed_path_count"] == 0
    persisted = json.loads((loop_dir / "disk_preflight.json").read_text())
    assert persisted["passed"] is False


def test_disk_preflight_cleanup_removes_only_repeatable_loop_intermediates(
    tmp_path,
    monkeypatch,
) -> None:
    loop_dir = tmp_path / "loop"
    projects_dir = tmp_path / "projects"
    seg_dir = loop_dir / "seg_demo"
    stage_dir = loop_dir / "staged_upload_frames"
    keep_dir = loop_dir / "keep_project"
    for directory in (seg_dir, stage_dir, keep_dir, projects_dir):
        directory.mkdir(parents=True)
    (seg_dir / "mask.npz").write_bytes(b"mask")
    (stage_dir / "frame.jpg").write_bytes(b"frame")
    (keep_dir / "artifact.json").write_text("{}", encoding="utf-8")
    (loop_dir / "raw_improvement_loop_report.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        improve_raw_job.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=10 * improve_raw_job.BYTES_PER_GIB,
            used=3 * improve_raw_job.BYTES_PER_GIB,
            free=7 * improve_raw_job.BYTES_PER_GIB,
        ),
    )

    report = improve_raw_job._write_disk_preflight_report(
        loop_dir,
        projects_dir=projects_dir,
        min_free_gib=5.0,
        cleanup_intermediates=True,
        force=True,
    )

    assert report["passed"] is True
    assert report["cleanup"]["removed_path_count"] == 2
    assert report["cleanup"]["removed_bytes"] == len(b"mask") + len(b"frame")
    assert not seg_dir.exists()
    assert not stage_dir.exists()
    assert keep_dir.exists()
    assert (loop_dir / "raw_improvement_loop_report.json").exists()


def test_geometry_safe_frames_excludes_closeup_and_partial_frames(tmp_path) -> None:
    frames = (
        improve_raw_job.FrameSelection(
            frame_id="clean_closeup",
            source_path=tmp_path / "clean.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="clean.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.20,
            score=0.95,
            mask_area_ratio=0.3,
            bbox_area_ratio=0.5,
            full_frame_bbox=False,
            raw_view_group="frontal_face",
            raw_pose_label="closeup_face",
            raw_frontal_face_confidence=0.86,
            raw_view_weight=0.90,
        ),
        improve_raw_job.FrameSelection(
            frame_id="body_photo",
            source_path=tmp_path / "body.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="body.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.86,
            score=0.80,
            mask_area_ratio=0.16,
            bbox_area_ratio=0.32,
            full_frame_bbox=False,
            raw_view_group="side_body",
            raw_pose_label="side_body",
        ),
        improve_raw_job.FrameSelection(
            frame_id="body_video",
            source_path=tmp_path / "body_video.jpg",
            media_type="video",
            source_type="video_frame",
            source_media_id="v1.mp4",
            source_frame_index=12,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.72,
            score=0.75,
            mask_area_ratio=0.18,
            bbox_area_ratio=0.36,
            full_frame_bbox=False,
            raw_view_group="oblique_body",
            raw_pose_label="video_oblique_body",
        ),
        improve_raw_job.FrameSelection(
            frame_id="mild_reject_body",
            source_path=tmp_path / "mild.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="mild.jpg",
            source_frame_index=None,
            geometry_reject_reasons=("sparse_geometry_mask",),
            geometry_body_fit_score=0.58,
            score=0.70,
            mask_area_ratio=0.06,
            bbox_area_ratio=0.18,
            full_frame_bbox=False,
        ),
        improve_raw_job.FrameSelection(
            frame_id="severe_tiny",
            source_path=tmp_path / "tiny.jpg",
            media_type="video",
            source_type="video_frame",
            source_media_id="v1.mp4",
            source_frame_index=40,
            geometry_reject_reasons=("mask_too_small", "bbox_too_small"),
            geometry_body_fit_score=0.99,
            score=0.99,
            mask_area_ratio=0.01,
            bbox_area_ratio=0.01,
            full_frame_bbox=False,
        ),
    )

    selected = improve_raw_job._geometry_safe_frames(frames, min_frames=2)

    assert [frame.frame_id for frame in selected] == ["body_photo", "body_video"]


def test_media_preflight_ranks_face_texture_and_body_geometry(tmp_path) -> None:
    def frame(
        frame_id: str,
        *,
        media_type: str,
        geometry_body_fit_score: float,
        score: float,
        raw_view_group: str,
        raw_pose_label: str,
        raw_frontal_face_confidence: float,
        geometry_reject_reasons: tuple[str, ...] = (),
    ) -> improve_raw_job.FrameSelection:
        return improve_raw_job.FrameSelection(
            frame_id=frame_id,
            source_path=tmp_path / f"{frame_id}.jpg",
            media_type=media_type,
            source_type="video_frame" if media_type == "video" else "image",
            source_media_id="v1.mp4" if media_type == "video" else f"{frame_id}.jpg",
            source_frame_index=12 if media_type == "video" else None,
            geometry_reject_reasons=geometry_reject_reasons,
            geometry_body_fit_score=geometry_body_fit_score,
            score=score,
            mask_area_ratio=0.16,
            bbox_area_ratio=0.32,
            full_frame_bbox=False,
            raw_view_label=raw_view_group,
            raw_view_group=raw_view_group,
            raw_pose_label=raw_pose_label,
            raw_frontal_face_confidence=raw_frontal_face_confidence,
            raw_view_weight=0.35 + 0.65 * raw_frontal_face_confidence,
        )

    closeup = frame(
        "closeup_face",
        media_type="photo",
        geometry_body_fit_score=0.20,
        score=0.96,
        raw_view_group="frontal_face",
        raw_pose_label="closeup_face",
        raw_frontal_face_confidence=0.92,
    )
    body_photo = frame(
        "body_photo",
        media_type="photo",
        geometry_body_fit_score=0.88,
        score=0.78,
        raw_view_group="side_body",
        raw_pose_label="side_body",
        raw_frontal_face_confidence=0.20,
    )
    body_video = frame(
        "body_video",
        media_type="video",
        geometry_body_fit_score=0.76,
        score=0.74,
        raw_view_group="oblique_body",
        raw_pose_label="video_oblique_body",
        raw_frontal_face_confidence=0.38,
    )
    rejected = frame(
        "tiny_reject",
        media_type="video",
        geometry_body_fit_score=0.05,
        score=0.10,
        raw_view_group="distant_body",
        raw_pose_label="video_distant_body",
        raw_frontal_face_confidence=0.10,
        geometry_reject_reasons=("mask_too_small", "bbox_too_small"),
    )
    candidate = improve_raw_job.SegmentationCandidate(
        candidate_id="candidate_preflight",
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=tmp_path / "seg_candidate",
        selected_frames=(closeup, body_photo, body_video, rejected),
        frame_count=4,
        selected_frame_count=4,
        mean_mask_area_ratio=0.16,
        mean_bbox_area_ratio=0.32,
        full_frame_bbox_fraction=0.0,
        score=1.0,
        texture_frames=(closeup, body_photo, body_video, rejected),
    )

    report = improve_raw_job._build_media_preflight_report(
        [candidate],
        min_geometry_frames=2,
        max_geometry_frames=3,
        max_texture_frames=4,
    )

    assert report["ready_for_reconstruction"] is True
    assert report["ready_for_promotion"] is True
    assert report["warnings"] == []
    assert report["promotion_rejection_reasons"] == []
    assert report["face_texture_frame_count"] == 1
    assert report["photo_frame_count"] == 2
    assert report["video_frame_count"] == 2
    assert report["geometry_video_frame_count"] == 1
    assert report["texture_video_frame_count"] == 1
    assert report["recommended_geometry_frame_ids"] == ["body_photo", "body_video"]
    assert report["recommended_texture_frame_ids"][:3] == [
        "closeup_face",
        "body_video",
        "body_photo",
    ]
    by_id = {frame["frame_id"]: frame for frame in report["frames"]}
    assert by_id["closeup_face"]["recommended_role"] == "texture"
    assert by_id["closeup_face"]["geometry_safe"] is False
    assert by_id["body_photo"]["recommended_role"] == "geometry_and_texture"
    assert by_id["tiny_reject"]["recommended_role"] == "rejected"
    assert report["candidate_summaries"][0]["geometry_safe_frame_count"] == 2

    summary = improve_raw_job._write_media_preflight_report(
        [candidate],
        loop_dir=tmp_path,
        min_geometry_frames=2,
        max_geometry_frames=3,
        max_texture_frames=4,
    )
    assert summary["candidate_count"] == 1
    assert summary["photo_frame_count"] == 2
    assert summary["video_frame_count"] == 2
    assert summary["geometry_video_frame_count"] == 1
    assert summary["texture_video_frame_count"] == 1
    assert summary["media_type_counts"] == {"photo": 2, "video": 2}


def test_media_preflight_blocks_promotion_without_face_texture(tmp_path) -> None:
    frames = (
        improve_raw_job.FrameSelection(
            frame_id="body_photo",
            source_path=tmp_path / "body_photo.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="body_photo.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.90,
            score=0.80,
            mask_area_ratio=0.18,
            bbox_area_ratio=0.34,
            full_frame_bbox=False,
            raw_view_group="side_body",
            raw_pose_label="side_body",
            raw_frontal_face_confidence=0.20,
            raw_view_weight=0.48,
        ),
        improve_raw_job.FrameSelection(
            frame_id="body_video",
            source_path=tmp_path / "body_video.jpg",
            media_type="video",
            source_type="video_frame",
            source_media_id="v1.mp4",
            source_frame_index=24,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.78,
            score=0.74,
            mask_area_ratio=0.16,
            bbox_area_ratio=0.31,
            full_frame_bbox=False,
            raw_view_group="oblique_body",
            raw_pose_label="video_oblique_body",
            raw_frontal_face_confidence=0.30,
            raw_view_weight=0.54,
        ),
    )
    candidate = improve_raw_job.SegmentationCandidate(
        candidate_id="candidate_body_only",
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=tmp_path / "seg_candidate",
        selected_frames=frames,
        frame_count=2,
        selected_frame_count=2,
        mean_mask_area_ratio=0.17,
        mean_bbox_area_ratio=0.33,
        full_frame_bbox_fraction=0.0,
        score=1.0,
        texture_frames=frames,
    )

    report = improve_raw_job._build_media_preflight_report(
        [candidate],
        min_geometry_frames=2,
        max_geometry_frames=4,
        max_texture_frames=4,
    )
    gate = improve_raw_job._media_preflight_promotion_gate(report)

    assert report["ready_for_reconstruction"] is True
    assert report["ready_for_promotion"] is False
    assert report["face_texture_frame_count"] == 0
    assert report["promotion_rejection_reasons"] == [
        "missing_high_confidence_face_texture_frame"
    ]
    assert gate["passed"] is False
    assert "face_texture_frame_present" in gate["rejection_reasons"]


def test_attach_media_preflight_gate_rejects_candidate_promotion() -> None:
    supervisor_report = {
        "best_candidate": "candidate_a",
        "candidates": [
            {
                "label": "candidate_a",
                "promotion_guard": {
                    "passed": True,
                    "checks": {"candidate_success": True},
                    "rejection_reasons": [],
                },
            }
        ],
    }
    gate = {
        "passed": False,
        "rejection_reasons": ["face_texture_frame_present"],
    }

    improve_raw_job._attach_media_preflight_promotion_gate(
        supervisor_report,
        candidate_id="candidate_a",
        gate=gate,
    )

    candidate = supervisor_report["candidates"][0]
    assert candidate["media_preflight_promotion_gate"] == gate
    assert candidate["promotion_guard"]["passed"] is False
    assert candidate["promotion_guard"]["checks"][
        "media_preflight_promotion_gate_passed"
    ] is False
    assert "media_preflight_promotion_gate_passed" in candidate["promotion_guard"][
        "rejection_reasons"
    ]


def test_main_preflight_only_skips_reconstruction_pipeline(tmp_path, monkeypatch) -> None:
    upload_dir = tmp_path / "upload"
    loop_dir = tmp_path / "loop"
    projects_dir = tmp_path / "projects"
    upload_dir.mkdir()
    frame_path = loop_dir / "staged_upload_frames" / "frame_0000.jpg"
    media_report_path = loop_dir / "staged_upload_frames" / "media_manifest.json"

    frames = (
        improve_raw_job.FrameSelection(
            frame_id="face",
            source_path=tmp_path / "face.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="face.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.20,
            score=0.95,
            mask_area_ratio=0.16,
            bbox_area_ratio=0.30,
            full_frame_bbox=False,
            raw_view_group="frontal_face",
            raw_pose_label="closeup_face",
            raw_frontal_face_confidence=0.9,
            raw_view_weight=0.9,
        ),
        improve_raw_job.FrameSelection(
            frame_id="body_a",
            source_path=tmp_path / "body_a.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="body_a.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.85,
            score=0.80,
            mask_area_ratio=0.18,
            bbox_area_ratio=0.34,
            full_frame_bbox=False,
            raw_view_group="side_body",
            raw_pose_label="side_body",
            raw_frontal_face_confidence=0.2,
            raw_view_weight=0.5,
        ),
        improve_raw_job.FrameSelection(
            frame_id="body_b",
            source_path=tmp_path / "body_b.jpg",
            media_type="video",
            source_type="video_frame",
            source_media_id="v1.mp4",
            source_frame_index=10,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.78,
            score=0.74,
            mask_area_ratio=0.17,
            bbox_area_ratio=0.32,
            full_frame_bbox=False,
            raw_view_group="oblique_body",
            raw_pose_label="video_oblique_body",
            raw_frontal_face_confidence=0.35,
            raw_view_weight=0.58,
        ),
    )
    candidate = improve_raw_job.SegmentationCandidate(
        candidate_id="candidate_preflight_only",
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=loop_dir / "seg_candidate",
        selected_frames=frames,
        frame_count=len(frames),
        selected_frame_count=len(frames),
        mean_mask_area_ratio=0.17,
        mean_bbox_area_ratio=0.32,
        full_frame_bbox_fraction=0.0,
        score=1.0,
        texture_frames=frames,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "improve_raw_job.py",
            "--upload-dir",
            str(upload_dir),
            "--loop-dir",
            str(loop_dir),
            "--projects-dir",
            str(projects_dir),
            "--min-selected-frames",
            "2",
            "--preflight-only",
            "--force",
        ],
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_write_disk_preflight_report",
        lambda *args, **kwargs: {
            "passed": True,
            "free_gib": 10.0,
            "min_free_gib": 5.0,
            "report_path": str(loop_dir / "disk_preflight.json"),
        },
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_stage_upload_media_frames",
        lambda *args, **kwargs: improve_raw_job.UploadMediaStagingResult(
            frame_paths=[frame_path],
            report_path=media_report_path,
            input_photo_count=2,
            input_video_count=1,
            extracted_video_frame_count=1,
        ),
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_sweep_segmenters",
        lambda *args, **kwargs: [candidate],
    )

    def fail_run_raw_pipeline(*args, **kwargs):
        raise AssertionError("preflight-only must not run reconstruction")

    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fail_run_raw_pipeline)

    assert improve_raw_job.main() == 0

    report = json.loads((loop_dir / "raw_improvement_loop_report.json").read_text())
    assert report["mode"] == "preflight_only"
    assert report["pipeline_reports"] == []
    assert report["best_candidate_id"] is None
    assert report["media_preflight"]["ready_for_promotion"] is True
    assert report["media_preflight_promotion_gate"]["passed"] is True


def test_main_texture_only_source_mode_skips_media_staging_and_segmentation(
    tmp_path,
    monkeypatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    source_dir = tmp_path / "source"
    loop_dir = tmp_path / "loop"
    projects_dir = tmp_path / "projects"
    baseline_dir.mkdir()
    source_dir.mkdir()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "improve_raw_job.py",
            "--texture-only-source-project-dir",
            str(source_dir),
            "--baseline-project-dir",
            str(baseline_dir),
            "--loop-dir",
            str(loop_dir),
            "--projects-dir",
            str(projects_dir),
            "--min-free-gib",
            "0",
            "--force",
        ],
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_write_disk_preflight_report",
        lambda *args, **kwargs: {
            "passed": True,
            "free_gib": 1.0,
            "min_free_gib": 0.0,
            "report_path": str(loop_dir / "disk_preflight.json"),
        },
    )

    def fail_stage(*args, **kwargs):
        raise AssertionError("texture-only mode must not stage upload media")

    def fail_sweep(*args, **kwargs):
        raise AssertionError("texture-only mode must not sweep segmenters")

    def fake_fallback_candidates(
        texture_source_reports,
        *,
        baseline_project_dir,
        projects_dir,
        texture_source_blocklist_path,
        max_texture_source_probes,
        explicit_texture_source_probe_frame_ids,
        explicit_texture_source_ablation_overrides,
        project_id_prefix,
        force,
    ):
        del texture_source_blocklist_path, max_texture_source_probes
        del explicit_texture_source_probe_frame_ids
        del explicit_texture_source_ablation_overrides
        del project_id_prefix, force
        calls.append(
            {
                "baseline_project_dir": baseline_project_dir,
                "projects_dir": projects_dir,
                "texture_source_reports": texture_source_reports,
            }
        )
        project_dir = projects_dir / "texture_skip"
        quality_dir = project_dir / "outputs" / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        summary_path = quality_dir / "raw_pipeline_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {},
                    "metrics": {
                        "raw_texture_source_admission_audit_passed": 0.0,
                    },
                    "success": False,
                }
            ),
            encoding="utf-8",
        )
        return [
            {
                "artifacts": {},
                "backend": "texture_only_source_project",
                "candidate_id": "source_baseline_geometry_texture",
                "project_dir": str(project_dir),
                "solid": False,
                "success": False,
                "metrics": {
                    "raw_texture_source_admission_audit_passed": 0.0,
                },
            }
        ]

    monkeypatch.setattr(improve_raw_job, "_stage_upload_media_frames", fail_stage)
    monkeypatch.setattr(improve_raw_job, "_sweep_segmenters", fail_sweep)
    monkeypatch.setattr(
        improve_raw_job,
        "_run_baseline_geometry_texture_fallback_candidates",
        fake_fallback_candidates,
    )
    monkeypatch.setattr(
        improve_raw_job,
        "write_guarded_candidate_comparison_report",
        lambda *args, **kwargs: SimpleNamespace(
            report_path=loop_dir / "raw_candidate_guarded_supervisor_report.json",
            best_candidate="baseline",
        ),
    )
    (loop_dir / "raw_candidate_guarded_supervisor_report.json").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (loop_dir / "raw_candidate_guarded_supervisor_report.json").write_text(
        json.dumps(
            {
                "best_candidate": "baseline",
                "candidates": [
                    {
                        "label": "baseline",
                        "score": 1.0,
                        "score_terms": {},
                        "metrics": {},
                    },
                    {
                        "label": "source_baseline_geometry_texture",
                        "score": 0.0,
                        "score_terms": {},
                        "metrics": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert improve_raw_job.main() == 0

    assert len(calls) == 1
    assert calls[0]["texture_source_reports"][0]["project_dir"] == str(source_dir.resolve())
    report = json.loads((loop_dir / "raw_improvement_loop_report.json").read_text())
    assert report["mode"] == "texture_only_source_project"
    assert report["media_staging"]["staged_frame_count"] == 0
    assert report["media_preflight"] is None
    assert report["segmentation_candidates"] == []
    assert report["best_is_baseline"] is True


def test_main_admission_only_builds_source_without_reconstruction(
    tmp_path,
    monkeypatch,
) -> None:
    upload_dir = tmp_path / "upload"
    baseline_dir = tmp_path / "baseline"
    loop_dir = tmp_path / "loop"
    projects_dir = tmp_path / "projects"
    upload_dir.mkdir()
    baseline_dir.mkdir()
    stage_dir, frame_paths, manifest_path = _write_stage_manifest(tmp_path, "stage")
    candidate = _candidate_from_stage(stage_dir, "candidate_admission_only")
    source_project_dir = loop_dir / "texture_source_upload_fake"
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "improve_raw_job.py",
            "--upload-dir",
            str(upload_dir),
            "--baseline-project-dir",
            str(baseline_dir),
            "--loop-dir",
            str(loop_dir),
            "--projects-dir",
            str(projects_dir),
            "--min-selected-frames",
            "2",
            "--admission-only",
            "--min-free-gib",
            "0",
            "--force",
        ],
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_write_disk_preflight_report",
        lambda *args, **kwargs: {
            "passed": True,
            "free_gib": 1.0,
            "min_free_gib": 0.0,
            "report_path": str(loop_dir / "disk_preflight.json"),
        },
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_stage_upload_media_frames",
        lambda *args, **kwargs: improve_raw_job.UploadMediaStagingResult(
            frame_paths=frame_paths,
            report_path=manifest_path,
            input_photo_count=2,
            input_video_count=1,
            extracted_video_frame_count=1,
        ),
    )
    monkeypatch.setattr(improve_raw_job, "_sweep_segmenters", lambda *args, **kwargs: [candidate])

    def fake_build_uploaded_texture_source_report(**kwargs):
        calls.append({"raw_frame_paths": tuple(kwargs["raw_frame_paths"])})
        return {
            "candidate_id": "uploaded_texture_source_fake",
            "backend": "white_subject_v1",
            "color_delta_threshold": 0.08,
            "project_dir": str(source_project_dir),
            "metrics": {"raw_uploaded_closeup_texture_frame_count": 3},
        }

    def fake_admission(
        project_dir,
        *,
        baseline_project_dir,
        texture_source_project_dir,
        rejected_texture_source_hashes,
        rejected_texture_source_frame_ids,
        force,
    ):
        del rejected_texture_source_hashes, rejected_texture_source_frame_ids, force
        assert baseline_project_dir == baseline_dir
        assert texture_source_project_dir == source_project_dir
        output_path = project_dir / "outputs" / "quality" / "texture_source_admission_audit.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audit = {
            "passed": True,
            "report_path": str(output_path),
            "rejection_reasons": [],
            "unique_accepted_frame_count": 2,
            "protected_face_source_patch_frame_count": 1,
            "rejected_ablation_frame_count": 0,
            "accepted_preflight_frame_count": 2,
            "duplicate_frame_count": 1,
            "rejected_low_detail_frame_count": 0,
            "top_score": 2.5,
        }
        output_path.write_text(json.dumps(audit), encoding="utf-8")
        return audit

    def fake_supervisor(candidate_project_dirs, output_path, *, baseline_label, force):
        del force
        assert baseline_label == "baseline"
        assert "uploaded_texture_source_fake_texture_source_admission" in candidate_project_dirs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "best_candidate": "baseline",
                    "candidates": [
                        {
                            "label": "baseline",
                            "score": 1.0,
                            "score_terms": {},
                            "metrics": {},
                        },
                        {
                            "label": "uploaded_texture_source_fake_texture_source_admission",
                            "score": 0.0,
                            "score_terms": {
                                "texture_source_admission_failed": 0.0,
                            },
                            "metrics": {
                                "raw_texture_source_admission_audit_present": 1.0,
                                "raw_texture_source_admission_audit_passed": 1.0,
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(report_path=output_path, best_candidate="baseline")

    def fail_reconstruction(*args, **kwargs):
        raise AssertionError("admission-only must not run reconstruction or texture bake")

    monkeypatch.setattr(
        improve_raw_job,
        "_build_uploaded_texture_source_report",
        fake_build_uploaded_texture_source_report,
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_write_texture_source_admission_for_candidate",
        fake_admission,
    )
    monkeypatch.setattr(
        improve_raw_job,
        "write_guarded_candidate_comparison_report",
        fake_supervisor,
    )
    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fail_reconstruction)
    monkeypatch.setattr(
        improve_raw_job,
        "run_baseline_geometry_texture_bake",
        fail_reconstruction,
    )

    assert improve_raw_job.main() == 0

    assert calls and calls[0]["raw_frame_paths"] == tuple(frame_paths)
    report = json.loads((loop_dir / "raw_improvement_loop_report.json").read_text())
    assert report["mode"] == "admission_only"
    assert report["media_staging"]["staged_frame_count"] == 3
    assert report["uploaded_texture_source"]["project_dir"] == str(source_project_dir)
    assert report["texture_source_admission_audit"]["passed"] is True
    assert len(report["pipeline_reports"]) == 1
    pipeline_report = report["pipeline_reports"][0]
    assert pipeline_report["success"] is False
    assert pipeline_report["texture_source_admission_audit_passed"] is True
    assert pipeline_report["metrics"]["raw_texture_source_admission_audit_present"] == 1.0
    assert pipeline_report["metrics"]["raw_texture_source_admission_audit_passed"] == 1.0
    assert report["best_is_baseline"] is True


def test_main_reuses_segmentation_cache_when_media_and_limits_match(
    tmp_path,
    monkeypatch,
) -> None:
    upload_dir = tmp_path / "upload"
    loop_dir = tmp_path / "loop"
    projects_dir = tmp_path / "projects"
    upload_dir.mkdir()
    current_stage_dir, current_frame_paths, current_manifest_path = _write_stage_manifest(
        tmp_path,
        "current_stage",
    )
    cache_stage_dir = tmp_path / "cache_stage"
    cache_manifest_path = _copy_stage_manifest_with_bytes(
        current_stage_dir,
        cache_stage_dir,
        current_manifest_path,
    )
    cached_candidate = _candidate_from_stage(cache_stage_dir, "cached_candidate")
    cache_report_path = _write_segmentation_cache_report(
        tmp_path,
        cache_manifest_path,
        cached_candidate,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "improve_raw_job.py",
            "--upload-dir",
            str(upload_dir),
            "--loop-dir",
            str(loop_dir),
            "--projects-dir",
            str(projects_dir),
            "--min-selected-frames",
            "2",
            "--max-selected-frames",
            "3",
            "--segmentation-cache-report",
            str(cache_report_path),
            "--preflight-only",
            "--force",
        ],
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_write_disk_preflight_report",
        lambda *args, **kwargs: {
            "passed": True,
            "free_gib": 10.0,
            "min_free_gib": 5.0,
            "report_path": str(loop_dir / "disk_preflight.json"),
        },
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_stage_upload_media_frames",
        lambda *args, **kwargs: improve_raw_job.UploadMediaStagingResult(
            frame_paths=current_frame_paths,
            report_path=current_manifest_path,
            input_photo_count=2,
            input_video_count=1,
            extracted_video_frame_count=1,
        ),
    )

    def fail_sweep(*args, **kwargs):
        raise AssertionError("matching cache should skip segmentation sweep")

    def fail_run_raw_pipeline(*args, **kwargs):
        raise AssertionError("preflight-only must not run reconstruction")

    monkeypatch.setattr(improve_raw_job, "_sweep_segmenters", fail_sweep)
    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fail_run_raw_pipeline)

    assert improve_raw_job.main() == 0

    report = json.loads((loop_dir / "raw_improvement_loop_report.json").read_text())
    assert report["segmentation_cache"]["used"] is True
    assert report["segmentation_cache"]["reason"] == "cache_hit"
    assert report["segmentation_cache"]["candidate_count"] == 1
    selected_frame = report["segmentation_candidates"][0]["selected_frames"][0]
    assert selected_frame["source_path"] == str(current_stage_dir / "frame_0000.jpg")


def test_main_reuses_segmentation_cache_with_stale_preflight_source_paths(
    tmp_path,
    monkeypatch,
) -> None:
    upload_dir = tmp_path / "upload"
    loop_dir = tmp_path / "loop"
    projects_dir = tmp_path / "projects"
    upload_dir.mkdir()
    current_stage_dir, current_frame_paths, current_manifest_path = _write_stage_manifest(
        tmp_path,
        "current_stage",
    )
    cache_stage_dir = tmp_path / "cache_stage"
    cache_manifest_path = _copy_stage_manifest_with_bytes(
        current_stage_dir,
        cache_stage_dir,
        current_manifest_path,
    )
    stale_preflight_stage = tmp_path / "deleted_seg" / "raw" / "images"
    stale_preflight_stage.mkdir(parents=True)
    for frame_path in current_frame_paths:
        target = stale_preflight_stage / frame_path.name
        target.write_bytes(frame_path.read_bytes())
    cached_candidate = _candidate_from_stage(stale_preflight_stage, "cached_candidate")
    cache_report_path = _write_segmentation_cache_report(
        tmp_path,
        cache_manifest_path,
        cached_candidate,
    )
    for stale_frame in stale_preflight_stage.glob("*.jpg"):
        stale_frame.unlink()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "improve_raw_job.py",
            "--upload-dir",
            str(upload_dir),
            "--loop-dir",
            str(loop_dir),
            "--projects-dir",
            str(projects_dir),
            "--min-selected-frames",
            "2",
            "--max-selected-frames",
            "3",
            "--segmentation-cache-report",
            str(cache_report_path),
            "--preflight-only",
            "--force",
        ],
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_write_disk_preflight_report",
        lambda *args, **kwargs: {
            "passed": True,
            "free_gib": 10.0,
            "min_free_gib": 5.0,
            "report_path": str(loop_dir / "disk_preflight.json"),
        },
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_stage_upload_media_frames",
        lambda *args, **kwargs: improve_raw_job.UploadMediaStagingResult(
            frame_paths=current_frame_paths,
            report_path=current_manifest_path,
            input_photo_count=2,
            input_video_count=1,
            extracted_video_frame_count=1,
        ),
    )

    def fail_sweep(*args, **kwargs):
        raise AssertionError("matching cache should skip segmentation sweep")

    def fail_run_raw_pipeline(*args, **kwargs):
        raise AssertionError("preflight-only must not run reconstruction")

    monkeypatch.setattr(improve_raw_job, "_sweep_segmenters", fail_sweep)
    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fail_run_raw_pipeline)

    assert improve_raw_job.main() == 0

    report = json.loads((loop_dir / "raw_improvement_loop_report.json").read_text())
    assert report["segmentation_cache"]["used"] is True
    selected_frame = report["segmentation_candidates"][0]["selected_frames"][0]
    assert selected_frame["source_path"] == str(current_stage_dir / "frame_0000.jpg")
    texture_frame = report["segmentation_candidates"][0]["texture_source_frames"][0]
    assert texture_frame["source_path"] == str(current_stage_dir / "frame_0000.jpg")


def test_main_falls_back_when_segmentation_cache_media_mismatches(
    tmp_path,
    monkeypatch,
) -> None:
    upload_dir = tmp_path / "upload"
    loop_dir = tmp_path / "loop"
    projects_dir = tmp_path / "projects"
    upload_dir.mkdir()
    _, current_frame_paths, current_manifest_path = _write_stage_manifest(
        tmp_path,
        "current_stage",
        variant=8,
    )
    cache_stage_dir, _, cache_manifest_path = _write_stage_manifest(tmp_path, "cache_stage")
    cached_candidate = _candidate_from_stage(cache_stage_dir, "cached_candidate")
    cache_report_path = _write_segmentation_cache_report(
        tmp_path,
        cache_manifest_path,
        cached_candidate,
    )
    sweep_candidate = _candidate_from_stage(tmp_path / "current_stage", "sweep_candidate")
    sweep_calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "improve_raw_job.py",
            "--upload-dir",
            str(upload_dir),
            "--loop-dir",
            str(loop_dir),
            "--projects-dir",
            str(projects_dir),
            "--min-selected-frames",
            "2",
            "--max-selected-frames",
            "3",
            "--segmentation-cache-report",
            str(cache_report_path),
            "--preflight-only",
            "--force",
        ],
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_write_disk_preflight_report",
        lambda *args, **kwargs: {
            "passed": True,
            "free_gib": 10.0,
            "min_free_gib": 5.0,
            "report_path": str(loop_dir / "disk_preflight.json"),
        },
    )
    monkeypatch.setattr(
        improve_raw_job,
        "_stage_upload_media_frames",
        lambda *args, **kwargs: improve_raw_job.UploadMediaStagingResult(
            frame_paths=current_frame_paths,
            report_path=current_manifest_path,
            input_photo_count=2,
            input_video_count=1,
            extracted_video_frame_count=1,
        ),
    )

    def fake_sweep(*args, **kwargs):
        sweep_calls.append((args, kwargs))
        return [sweep_candidate]

    monkeypatch.setattr(improve_raw_job, "_sweep_segmenters", fake_sweep)

    assert improve_raw_job.main() == 0

    report = json.loads((loop_dir / "raw_improvement_loop_report.json").read_text())
    assert len(sweep_calls) == 1
    assert report["segmentation_cache"]["used"] is False
    assert report["segmentation_cache"]["reason"] == "media_fingerprint_mismatch"
    assert report["segmentation_cache"]["fallback_sweep_candidate_count"] == 1
    assert report["segmentation_candidates"][0]["candidate_id"] == "sweep_candidate"


def test_geometry_safe_frames_keep_photo_video_balance(tmp_path) -> None:
    frames = (
        improve_raw_job.FrameSelection(
            frame_id="photo_body_a",
            source_path=tmp_path / "photo_a.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="photo_a.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.90,
            score=0.8,
            mask_area_ratio=0.16,
            bbox_area_ratio=0.32,
            full_frame_bbox=False,
        ),
        improve_raw_job.FrameSelection(
            frame_id="photo_body_b",
            source_path=tmp_path / "photo_b.jpg",
            media_type="photo",
            source_type="image",
            source_media_id="photo_b.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.80,
            score=0.7,
            mask_area_ratio=0.14,
            bbox_area_ratio=0.28,
            full_frame_bbox=False,
        ),
        improve_raw_job.FrameSelection(
            frame_id="video_body_a",
            source_path=tmp_path / "video_a.jpg",
            media_type="video",
            source_type="video_frame",
            source_media_id="v1.mp4",
            source_frame_index=20,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.30,
            score=0.6,
            mask_area_ratio=0.20,
            bbox_area_ratio=0.42,
            full_frame_bbox=False,
        ),
        improve_raw_job.FrameSelection(
            frame_id="video_body_b",
            source_path=tmp_path / "video_b.jpg",
            media_type="video",
            source_type="video_frame",
            source_media_id="v1.mp4",
            source_frame_index=40,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.20,
            score=0.5,
            mask_area_ratio=0.22,
            bbox_area_ratio=0.46,
            full_frame_bbox=False,
        ),
    )

    selected = improve_raw_job._geometry_safe_frames(frames, min_frames=4)

    assert [frame.frame_id for frame in selected] == [
        "photo_body_a",
        "photo_body_b",
        "video_body_a",
        "video_body_b",
    ]


def test_geometry_variants_keep_task227_closeup_and_partial_frames_texture_only(
    tmp_path,
) -> None:
    frame_specs = (
        ("video_side", "video", "video_side_body", "side_body", 0.74, ()),
        ("closeup", "photo", "closeup_face", "frontal_face", 0.74, ()),
        (
            "border_partial",
            "photo",
            "vertical_body",
            "vertical_body",
            0.19,
            ("border_partial_body",),
        ),
        ("photo_oblique", "photo", "oblique_body", "oblique_body", 0.76, ()),
        ("video_oblique", "video", "video_oblique_body", "oblique_body", 0.49, ()),
        ("video_side_b", "video", "video_side_body", "oblique_body", 0.48, ()),
        ("sparse", "photo", "oblique_body", "vertical_body", 0.25, ("sparse_geometry_mask",)),
    )
    frames = tuple(
        improve_raw_job.FrameSelection(
            frame_id=frame_id,
            source_path=tmp_path / f"{frame_id}.jpg",
            media_type=media_type,
            source_type="video_frame" if media_type == "video" else "image",
            source_media_id="v1.mp4" if media_type == "video" else f"{frame_id}.jpg",
            source_frame_index=index if media_type == "video" else None,
            geometry_reject_reasons=reject_reasons,
            geometry_body_fit_score=body_fit,
            score=0.8,
            mask_area_ratio=0.16,
            bbox_area_ratio=0.32,
            full_frame_bbox=False,
            raw_view_group=view_group,
            raw_pose_label=pose_label,
            raw_frontal_face_confidence=0.75 if pose_label == "closeup_face" else 0.55,
            raw_view_weight=0.80,
        )
        for index, (
            frame_id,
            media_type,
            pose_label,
            view_group,
            body_fit,
            reject_reasons,
        ) in enumerate(frame_specs)
    )

    variants = improve_raw_job._geometry_variants(
        frames,
        min_frames=4,
        max_variants=1,
    )

    assert variants[0].variant_id == "pose_gated_bodyfit"
    assert [frame.frame_id for frame in variants[0].geometry_frames] == [
        "photo_oblique",
        "video_side",
        "video_oblique",
        "video_side_b",
    ]
    assert {
        frame.frame_id for frame in frames if frame not in variants[0].geometry_frames
    } == {"closeup", "border_partial", "sparse"}


def test_run_full_candidate_variants_sweep_geometry_with_fixed_texture(
    tmp_path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "segmentation_raw_images"
    source_dir.mkdir()
    frame_specs = (
        ("photo_body", "photo", 0.90, 0.60, "side_body", (240, 240, 240)),
        ("photo_score", "photo", 0.20, 0.95, "frontal_face", (230, 230, 230)),
        ("video_body", "video", 0.30, 0.50, "oblique_body", (40, 80, 120)),
        ("video_score", "video", 0.10, 0.90, "frontal_face", (60, 90, 130)),
    )
    frames = []
    for index, (frame_id, media_type, body_score, score, view_group, color) in enumerate(
        frame_specs
    ):
        source_path = source_dir / f"{frame_id}.jpg"
        Image.new("RGB", (8, 6), color=color).save(source_path)
        frames.append(
            improve_raw_job.FrameSelection(
                frame_id=frame_id,
                source_path=source_path,
                media_type=media_type,
                source_type="video_frame" if media_type == "video" else "image",
                source_media_id="v1.mp4" if media_type == "video" else f"{frame_id}.jpg",
                source_frame_index=index if media_type == "video" else None,
                geometry_reject_reasons=(),
                geometry_body_fit_score=body_score,
                score=score,
                mask_area_ratio=0.16,
                bbox_area_ratio=0.34,
                full_frame_bbox=False,
                raw_view_label=f"{media_type}_{view_group}",
                raw_view_group=view_group,
                raw_pose_label=view_group,
                raw_frontal_face_confidence=0.9 if view_group == "frontal_face" else 0.2,
                raw_view_weight=0.9 if view_group == "frontal_face" else 0.5,
            )
        )
    candidate = improve_raw_job.SegmentationCandidate(
        candidate_id="candidate_video",
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=tmp_path / "seg_candidate",
        selected_frames=tuple(frames),
        frame_count=len(frames),
        selected_frame_count=len(frames),
        mean_mask_area_ratio=0.16,
        mean_bbox_area_ratio=0.34,
        full_frame_bbox_fraction=0.0,
        score=1.0,
    )
    calls: list[dict[str, object]] = []

    def fake_run_raw_pipeline(
        project_dir,
        *,
        input_paths,
        min_frames,
        color_delta_threshold,
        segmentation_backend,
        camera_root_steps,
        max_selected_frames,
        selected_frame_ids,
        texture_frame_ids,
        frame_selection_label,
        force,
    ):
        del (
            input_paths,
            min_frames,
            color_delta_threshold,
            segmentation_backend,
            camera_root_steps,
            max_selected_frames,
            frame_selection_label,
            force,
        )
        calls.append(
            {
                "project_name": project_dir.name,
                "selected_frame_ids": tuple(selected_frame_ids),
                "texture_frame_ids": tuple(texture_frame_ids),
            }
        )
        summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "face_count": 20,
                        "keypoint_loss_reduction": 0.1,
                        "overlay_mean_silhouette_iou": 0.3,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 100,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_video_frame_count": len(
                            [
                                frame_id
                                for frame_id in selected_frame_ids
                                if frame_id in {"frame_0002", "frame_0003"}
                            ]
                        ),
                        "raw_texture_foreground_fallback_luma": 0.8,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 2,
                        "silhouette_loss_reduction": 0.2,
                        "silhouette_residual_mean_union_error": 0.7,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            )
        )
        return SimpleNamespace(success=True, summary_path=summary_path)

    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fake_run_raw_pipeline)

    reports = improve_raw_job._run_full_candidate_variants(
        candidate,
        projects_dir=tmp_path / "projects",
        rank=1,
        min_selected_frames=2,
        camera_root_steps=2,
        max_geometry_variants=3,
        project_id_prefix="task229 geom sweep",
        force=True,
    )

    assert [report["geometry_variant_id"] for report in reports] == [
        "pose_gated_bodyfit",
        "view_balanced_bodyfit",
        "media_balanced_score",
    ]
    assert [call["selected_frame_ids"] for call in calls] == [
        ("frame_0000", "frame_0002"),
        ("frame_0000", "frame_0001"),
        ("frame_0001", "frame_0003"),
    ]
    assert all(
        call["texture_frame_ids"]
        == ("frame_0000", "frame_0001", "frame_0002", "frame_0003")
        for call in calls
    )
    assert all("candidate_video_" in report["candidate_id"] for report in reports)
    assert all(
        str(call["project_name"]).startswith("task229_geom_sweep_loop_01_candidate_video_")
        for call in calls
    )
    assert all(
        str(report["project_dir"]).split("/")[-1].startswith(
            "task229_geom_sweep_loop_01_candidate_video_"
        )
        for report in reports
    )
    assert reports[0]["geometry_pose_gate"]["gated_texture_only_frame_ids"] == []
    assert reports[0]["geometry_view_counts"] == {"oblique_body": 1, "side_body": 1}


def test_run_full_candidate_variants_uses_max_selected_frames_for_rich_geometry(
    tmp_path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "segmentation_raw_images"
    source_dir.mkdir()
    frame_specs = (
        ("photo_side", "photo", 0.90, "side_body", (240, 240, 240)),
        ("video_side", "video", 0.80, "side_body", (60, 90, 130)),
        ("photo_oblique", "photo", 0.70, "oblique_body", (230, 230, 230)),
        ("video_oblique", "video", 0.60, "oblique_body", (70, 100, 140)),
    )
    frames = []
    for index, (frame_id, media_type, body_score, view_group, color) in enumerate(
        frame_specs
    ):
        source_path = source_dir / f"{frame_id}.jpg"
        Image.new("RGB", (8, 6), color=color).save(source_path)
        frames.append(
            improve_raw_job.FrameSelection(
                frame_id=frame_id,
                source_path=source_path,
                media_type=media_type,
                source_type="video_frame" if media_type == "video" else "image",
                source_media_id="v1.mp4" if media_type == "video" else f"{frame_id}.jpg",
                source_frame_index=index if media_type == "video" else None,
                geometry_reject_reasons=(),
                geometry_body_fit_score=body_score,
                score=0.80,
                mask_area_ratio=0.16,
                bbox_area_ratio=0.34,
                full_frame_bbox=False,
                raw_view_label=f"{media_type}_{view_group}",
                raw_view_group=view_group,
                raw_pose_label=view_group,
                raw_frontal_face_confidence=0.20,
                raw_view_weight=0.60,
            )
        )
    candidate = improve_raw_job.SegmentationCandidate(
        candidate_id="candidate_rich_geometry",
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=tmp_path / "seg_candidate",
        selected_frames=tuple(frames),
        frame_count=len(frames),
        selected_frame_count=len(frames),
        mean_mask_area_ratio=0.16,
        mean_bbox_area_ratio=0.34,
        full_frame_bbox_fraction=0.0,
        score=1.0,
    )
    calls: list[dict[str, object]] = []

    def fake_run_raw_pipeline(
        project_dir,
        *,
        input_paths,
        min_frames,
        color_delta_threshold,
        segmentation_backend,
        camera_root_steps,
        max_selected_frames,
        selected_frame_ids,
        texture_frame_ids,
        frame_selection_label,
        force,
    ):
        del (
            input_paths,
            color_delta_threshold,
            segmentation_backend,
            camera_root_steps,
            max_selected_frames,
            frame_selection_label,
            force,
        )
        calls.append(
            {
                "project_name": project_dir.name,
                "min_frames": min_frames,
                "selected_frame_ids": tuple(selected_frame_ids),
                "texture_frame_ids": tuple(texture_frame_ids),
            }
        )
        summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "face_count": 20,
                        "keypoint_loss_reduction": 0.1,
                        "overlay_mean_silhouette_iou": 0.3,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 100,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_video_frame_count": 2,
                        "raw_texture_foreground_fallback_luma": 0.8,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 2,
                        "silhouette_loss_reduction": 0.2,
                        "silhouette_residual_mean_union_error": 0.7,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            )
        )
        return SimpleNamespace(success=True, summary_path=summary_path)

    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fake_run_raw_pipeline)

    reports = improve_raw_job._run_full_candidate_variants(
        candidate,
        projects_dir=tmp_path / "projects",
        rank=1,
        min_selected_frames=2,
        max_selected_frames=4,
        camera_root_steps=2,
        max_geometry_variants=1,
        project_id_prefix="rich geometry",
        force=True,
    )

    assert reports[0]["geometry_variant_id"] == "pose_gated_bodyfit"
    assert reports[0]["geometry_frame_count"] == 4
    assert reports[0]["geometry_frames"] == [
        "photo_side",
        "video_side",
        "photo_oblique",
        "video_oblique",
    ]
    assert calls[0]["selected_frame_ids"] == (
        "frame_0000",
        "frame_0001",
        "frame_0002",
        "frame_0003",
    )


def test_run_full_candidate_variants_geometry_frame_sweep(
    tmp_path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "segmentation_raw_images"
    source_dir.mkdir()
    frames = []
    for index in range(8):
        media_type = "photo" if index % 2 == 0 else "video"
        frame_id = f"{media_type}_{index:02d}"
        source_path = source_dir / f"{frame_id}.jpg"
        Image.new("RGB", (8, 6), color=(220 - index, 220 - index, 220 - index)).save(
            source_path
        )
        frames.append(
            improve_raw_job.FrameSelection(
                frame_id=frame_id,
                source_path=source_path,
                media_type=media_type,
                source_type="video_frame" if media_type == "video" else "image",
                source_media_id="v1.mp4" if media_type == "video" else f"{frame_id}.jpg",
                source_frame_index=index if media_type == "video" else None,
                geometry_reject_reasons=(),
                geometry_body_fit_score=0.90 - index * 0.04,
                score=0.80 - index * 0.01,
                mask_area_ratio=0.16,
                bbox_area_ratio=0.34,
                full_frame_bbox=False,
                raw_view_label="side_body",
                raw_view_group="side_body" if index < 4 else "oblique_body",
                raw_pose_label="side_body" if index < 4 else "oblique_body",
                raw_frontal_face_confidence=0.20,
                raw_view_weight=0.60,
            )
        )
    candidate = improve_raw_job.SegmentationCandidate(
        candidate_id="candidate_sweep",
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=tmp_path / "seg_candidate",
        selected_frames=tuple(frames),
        frame_count=len(frames),
        selected_frame_count=len(frames),
        mean_mask_area_ratio=0.16,
        mean_bbox_area_ratio=0.34,
        full_frame_bbox_fraction=0.0,
        score=1.0,
    )
    calls: list[dict[str, object]] = []

    def fake_run_raw_pipeline(
        project_dir,
        *,
        input_paths,
        min_frames,
        color_delta_threshold,
        segmentation_backend,
        camera_root_steps,
        max_selected_frames,
        selected_frame_ids,
        texture_frame_ids,
        frame_selection_label,
        force,
    ):
        del (
            project_dir,
            input_paths,
            min_frames,
            color_delta_threshold,
            segmentation_backend,
            camera_root_steps,
            max_selected_frames,
            frame_selection_label,
            force,
        )
        calls.append(
            {
                "selected_frame_ids": tuple(selected_frame_ids),
                "texture_frame_ids": tuple(texture_frame_ids),
            }
        )
        summary_path = tmp_path / f"summary_{len(calls)}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(tmp_path / "cat_avatar.glb")},
                    "metrics": {
                        "face_count": 20,
                        "keypoint_loss_reduction": 0.1,
                        "overlay_mean_silhouette_iou": 0.3,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 100,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_video_frame_count": len(
                            [frame_id for frame_id in selected_frame_ids if "video" in frame_id]
                        ),
                        "raw_texture_foreground_fallback_luma": 0.8,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 4,
                        "silhouette_loss_reduction": 0.2,
                        "silhouette_residual_mean_union_error": 0.7,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(success=True, summary_path=summary_path)

    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fake_run_raw_pipeline)

    reports = improve_raw_job._run_full_candidate_variants(
        candidate,
        projects_dir=tmp_path / "projects",
        rank=1,
        min_selected_frames=2,
        max_selected_frames=8,
        camera_root_steps=2,
        max_geometry_variants=4,
        geometry_frame_sweep=True,
        force=True,
    )

    assert [report["geometry_frame_count"] for report in reports] == [2, 4, 6, 8]
    assert [report["geometry_variant_id"] for report in reports] == [
        "frames_02_pose_gated_bodyfit",
        "frames_04_pose_gated_bodyfit",
        "frames_06_pose_gated_bodyfit",
        "frames_08_pose_gated_bodyfit",
    ]
    assert [len(call["selected_frame_ids"]) for call in calls] == [2, 4, 6, 8]
    assert all(len(call["texture_frame_ids"]) == 8 for call in calls)


def test_pose_gated_geometry_keeps_unsafe_vertical_frames_texture_only(
    tmp_path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "segmentation_raw_images"
    source_dir.mkdir()
    frame_specs = (
        ("frontal_closeup_safe", "photo", 0.74, 0.93, "frontal_face", "closeup_face"),
        ("oblique_photo_safe", "photo", 0.76, 0.86, "oblique_body", "oblique_body"),
        ("oblique_video_safe", "video", 0.46, 0.78, "oblique_body", "video_oblique_body"),
        ("vertical_texture_only", "video", 0.21, 0.65, "vertical_body", "video_vertical_body"),
    )
    frames = []
    for index, (frame_id, media_type, body_score, score, view_group, pose_label) in enumerate(
        frame_specs
    ):
        source_path = source_dir / f"{frame_id}.jpg"
        Image.new("RGB", (8, 6), color=(220 - index * 20, 220, 220)).save(source_path)
        frames.append(
            improve_raw_job.FrameSelection(
                frame_id=frame_id,
                source_path=source_path,
                media_type=media_type,
                source_type="video_frame" if media_type == "video" else "image",
                source_media_id="v1.mp4" if media_type == "video" else f"{frame_id}.jpg",
                source_frame_index=index if media_type == "video" else None,
                geometry_reject_reasons=(),
                geometry_body_fit_score=body_score,
                score=score,
                mask_area_ratio=0.16,
                bbox_area_ratio=0.30,
                full_frame_bbox=False,
                raw_view_label=f"{media_type}_{view_group}",
                raw_view_group=view_group,
                raw_pose_label=pose_label,
                raw_frontal_face_confidence=0.80 if "closeup" in pose_label else 0.55,
                raw_view_weight=0.86 if "closeup" in pose_label else 0.65,
            )
        )
    candidate = improve_raw_job.SegmentationCandidate(
        candidate_id="candidate_pose",
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=tmp_path / "seg_candidate",
        selected_frames=tuple(frames),
        frame_count=len(frames),
        selected_frame_count=len(frames),
        mean_mask_area_ratio=0.16,
        mean_bbox_area_ratio=0.30,
        full_frame_bbox_fraction=0.0,
        score=1.0,
    )
    calls: list[dict[str, object]] = []

    def fake_run_raw_pipeline(
        project_dir,
        *,
        input_paths,
        min_frames,
        color_delta_threshold,
        segmentation_backend,
        camera_root_steps,
        max_selected_frames,
        selected_frame_ids,
        texture_frame_ids,
        frame_selection_label,
        force,
    ):
        del (
            input_paths,
            color_delta_threshold,
            segmentation_backend,
            camera_root_steps,
            max_selected_frames,
            frame_selection_label,
            force,
        )
        calls.append(
            {
                "project_name": project_dir.name,
                "min_frames": min_frames,
                "selected_frame_ids": tuple(selected_frame_ids),
                "texture_frame_ids": tuple(texture_frame_ids),
            }
        )
        summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "face_count": 20,
                        "keypoint_loss_reduction": 0.1,
                        "overlay_mean_silhouette_iou": 0.3,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 100,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_video_frame_count": 1,
                        "raw_texture_foreground_fallback_luma": 0.8,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 2,
                        "silhouette_loss_reduction": 0.2,
                        "silhouette_residual_mean_union_error": 0.7,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            )
        )
        return SimpleNamespace(success=True, summary_path=summary_path)

    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fake_run_raw_pipeline)

    reports = improve_raw_job._run_full_candidate_variants(
        candidate,
        projects_dir=tmp_path / "projects",
        rank=1,
        min_selected_frames=4,
        camera_root_steps=2,
        max_geometry_variants=1,
        force=True,
    )

    assert reports[0]["geometry_variant_id"] == "pose_gated_bodyfit"
    assert calls[0]["min_frames"] == 2
    assert calls[0]["selected_frame_ids"] == ("frame_0001", "frame_0002")
    assert calls[0]["texture_frame_ids"] == (
        "frame_0000",
        "frame_0001",
        "frame_0002",
        "frame_0003",
    )
    assert reports[0]["geometry_pose_gate"]["gated_texture_only_frame_ids"] == [
        "frontal_closeup_safe",
        "vertical_texture_only"
    ]
    assert reports[0]["geometry_pose_gate"]["passed_risky_geometry_frame_ids"] == []


def test_run_full_candidate_adds_face_detail_texture_reserve(
    tmp_path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "segmentation_raw_images"
    source_dir.mkdir()
    body_a = source_dir / "body_a.jpg"
    body_b = source_dir / "body_b.jpg"
    face_reserve = source_dir / "face_reserve.jpg"
    Image.new("RGB", (8, 6), color=(230, 230, 230)).save(body_a)
    Image.new("RGB", (8, 6), color=(210, 210, 210)).save(body_b)
    Image.new("RGB", (8, 6), color=(190, 180, 170)).save(face_reserve)
    selected_frames = (
        improve_raw_job.FrameSelection(
            frame_id="body_a",
            source_path=body_a,
            media_type="photo",
            source_type="image",
            source_media_id="body_a.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.9,
            score=0.9,
            mask_area_ratio=0.12,
            bbox_area_ratio=0.24,
            full_frame_bbox=False,
            raw_view_group="side_body",
            raw_pose_label="side_body",
            raw_frontal_face_confidence=0.2,
            raw_view_weight=0.4,
        ),
        improve_raw_job.FrameSelection(
            frame_id="body_b",
            source_path=body_b,
            media_type="photo",
            source_type="image",
            source_media_id="body_b.jpg",
            source_frame_index=None,
            geometry_reject_reasons=(),
            geometry_body_fit_score=0.8,
            score=0.8,
            mask_area_ratio=0.13,
            bbox_area_ratio=0.26,
            full_frame_bbox=False,
            raw_view_group="oblique_body",
            raw_pose_label="oblique_body",
            raw_frontal_face_confidence=0.3,
            raw_view_weight=0.5,
        ),
    )
    texture_frames = selected_frames + (
        improve_raw_job.FrameSelection(
            frame_id="face_reserve",
            source_path=face_reserve,
            media_type="photo",
            source_type="image",
            source_media_id="face_reserve.jpg",
            source_frame_index=None,
            geometry_reject_reasons=("border_partial_body",),
            geometry_body_fit_score=0.2,
            score=0.7,
            mask_area_ratio=0.14,
            bbox_area_ratio=0.42,
            full_frame_bbox=False,
            raw_view_group="frontal_face",
            raw_pose_label="closeup_face",
            raw_frontal_face_confidence=0.82,
            raw_view_weight=0.88,
        ),
    )
    candidate = improve_raw_job.SegmentationCandidate(
        candidate_id="candidate_texture_reserve",
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=tmp_path / "seg_candidate",
        selected_frames=selected_frames,
        texture_frames=texture_frames,
        frame_count=3,
        selected_frame_count=2,
        mean_mask_area_ratio=0.13,
        mean_bbox_area_ratio=0.30,
        full_frame_bbox_fraction=0.0,
        score=1.0,
    )
    captured: dict[str, object] = {}

    def fake_run_raw_pipeline(
        project_dir,
        *,
        input_paths,
        min_frames,
        color_delta_threshold,
        segmentation_backend,
        camera_root_steps,
        max_selected_frames,
        selected_frame_ids,
        texture_frame_ids,
        frame_selection_label,
        force,
    ):
        del (
            input_paths,
            min_frames,
            color_delta_threshold,
            segmentation_backend,
            camera_root_steps,
            max_selected_frames,
            frame_selection_label,
            force,
        )
        captured["selected_frame_ids"] = tuple(selected_frame_ids)
        captured["texture_frame_ids"] = tuple(texture_frame_ids)
        summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "face_count": 20,
                        "keypoint_loss_reduction": 0.1,
                        "overlay_mean_silhouette_iou": 0.3,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 100,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_video_frame_count": 0,
                        "raw_texture_foreground_fallback_luma": 0.8,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 0,
                        "silhouette_loss_reduction": 0.2,
                        "silhouette_residual_mean_union_error": 0.7,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            )
        )
        return SimpleNamespace(success=True, summary_path=summary_path)

    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fake_run_raw_pipeline)

    report = improve_raw_job._run_full_candidate(
        candidate,
        projects_dir=tmp_path / "projects",
        rank=1,
        min_selected_frames=2,
        camera_root_steps=2,
        force=True,
    )

    assert captured["selected_frame_ids"] == ("frame_0000", "frame_0001")
    assert captured["texture_frame_ids"] == ("frame_0000", "frame_0001", "frame_0002")
    assert report["texture_source_frame_count"] == 3
    assert report["texture_source_added_reserve_frames"] == ["face_reserve"]
    assert report["geometry_pose_gate"]["gated_texture_only_frame_ids"] == [
        "face_reserve"
    ]


def test_run_baseline_geometry_texture_candidate_uses_source_as_texture_only(
    tmp_path,
    monkeypatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    source_dir = tmp_path / "source"
    projects_dir = tmp_path / "projects"
    baseline_dir.mkdir()
    source_dir.mkdir()
    blocklist_path = tmp_path / "texture_source_ablation_blocklist.json"
    prior_guard_dir = (
        projects_dir
        / "task229_texture_loop_texture_source_candidate_baseline_geometry_texture"
        / "outputs"
        / "quality"
    )
    prior_guard_dir.mkdir(parents=True)
    (prior_guard_dir / "texture_only_promotion_guard.json").write_text(
        json.dumps(
            {
                "supplemental_source_rejections": [
                    {
                        "rejected": True,
                        "reason": "strict_face_detail_metric_regression",
                        "sha256": "bad-face-hash",
                        "source_frame_id": "bad_face",
                    },
                    {
                        "rejected": True,
                        "reason": "other_reason",
                        "sha256": "ignored-hash",
                        "source_frame_id": "ignored_face",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_run_baseline_geometry_texture_bake(
        project_dir,
        *,
        baseline_project_dir,
        texture_source_project_dir,
        rejected_texture_source_hashes,
        rejected_texture_source_frame_ids,
        force,
        probe_texture_source_frame_ids=(),
        probe_texture_source_ablation_overrides=None,
    ):
        calls.append(
            {
                "project_dir": project_dir,
                "baseline_project_dir": baseline_project_dir,
                "texture_source_project_dir": texture_source_project_dir,
                "rejected_texture_source_hashes": rejected_texture_source_hashes,
                "rejected_texture_source_frame_ids": rejected_texture_source_frame_ids,
                "probe_texture_source_frame_ids": probe_texture_source_frame_ids,
                "probe_texture_source_ablation_overrides": (
                    probe_texture_source_ablation_overrides or {}
                ),
                "force": force,
            }
        )
        quality_dir = project_dir / "outputs" / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        summary_path = quality_dir / "raw_pipeline_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "basecolor_observed_texel_fraction": 0.5,
                        "face_count": 20,
                        "joint_projection_mean_keypoint_loss": 0.1,
                        "overlay_mean_silhouette_iou": 0.31,
                        "raw_baseline_geometry_texture_bake_used": 1.0,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 120,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_glb_face_feature_eye_detail_luma_std": 0.2,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_frame_count": 2,
                        "raw_texture_frame_count": 4,
                        "raw_texture_only_added_unique_frame_count": 2,
                        "raw_texture_only_promotion_guard_passed": 1.0,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 1,
                        "silhouette_residual_mean_union_error": 0.6,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            )
        )
        bake_path = quality_dir / "baseline_geometry_texture_bake.json"
        bake_path.write_text(
            json.dumps(
                {
                    "selected_frame_ids": ["body_a", "body_b"],
                    "texture_frame_ids": [
                        "body_a",
                        "body_b",
                        "source_face",
                        "source_video",
                    ],
                    "added_unique_texture_frame_count": 2,
                    "rejected_ablation_texture_frame_count": 1,
                    "rejected_ablation_texture_frames": [
                        {"source_frame_id": "bad_face", "sha256": "bad-face-hash"}
                    ],
                    "skipped_duplicate_texture_frame_count": 0,
                }
            )
        )
        return SimpleNamespace(
            added_unique_texture_frame_count=2,
            promotion_guard_passed=False,
            promotion_guard_path=quality_dir / "texture_only_promotion_guard.json",
            success=False,
            summary_path=summary_path,
        )

    monkeypatch.setattr(
        improve_raw_job,
        "run_baseline_geometry_texture_bake",
        fake_run_baseline_geometry_texture_bake,
    )

    report = improve_raw_job._run_baseline_geometry_texture_candidate(
        {
            "candidate_id": "source_candidate",
            "backend": "white_subject_v1",
            "color_delta_threshold": 0.08,
            "project_dir": str(source_dir),
        },
        baseline_project_dir=baseline_dir,
        projects_dir=projects_dir,
        texture_source_blocklist_path=blocklist_path,
        project_id_prefix="task229 texture",
        force=True,
    )

    assert calls == [
        {
            "project_dir": projects_dir
            / "task229_texture_loop_texture_source_candidate_baseline_geometry_texture",
            "baseline_project_dir": baseline_dir,
            "texture_source_project_dir": source_dir,
            "rejected_texture_source_hashes": ("bad-face-hash",),
            "rejected_texture_source_frame_ids": ("bad_face",),
            "probe_texture_source_frame_ids": (),
            "probe_texture_source_ablation_overrides": {},
            "force": True,
        }
    ]
    assert report["geometry_variant_id"] == "baseline_geometry_texture_only"
    assert report["geometry_frames"] == ["body_a", "body_b"]
    assert report["texture_frames"] == ["body_a", "body_b", "source_face", "source_video"]
    assert report["geometry_pose_gate"]["gated_texture_only_frame_ids"] == [
        "source_face",
        "source_video",
    ]
    assert report["success"] is True
    assert report["texture_only_guarded_result_success"] is False
    assert report["texture_only_promotion_guard_passed"] is False
    assert report["texture_only_added_unique_frame_count"] == 2
    assert report["texture_only_rejected_ablation_frame_count"] == 1
    assert report["texture_only_rejected_ablation_frames"] == [
        {"source_frame_id": "bad_face", "sha256": "bad-face-hash"}
    ]
    assert report["texture_only_prior_rejection_hash_count"] == 1
    assert report["texture_only_prior_rejection_source_frame_count"] == 1
    assert report["texture_only_ablation_blocklist"] == str(blocklist_path)
    assert report["texture_only_ablation_blocklist_entry_count"] == 1
    assert report["metrics"]["raw_baseline_geometry_texture_bake_used"] == 1.0
    blocklist = json.loads(blocklist_path.read_text(encoding="utf-8"))
    assert blocklist["entry_count"] == 1
    assert blocklist["sha256"] == ["bad-face-hash"]
    assert blocklist["source_frame_ids"] == ["bad_face"]


def test_run_baseline_geometry_texture_candidate_uses_stable_blocklist(
    tmp_path,
    monkeypatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    source_dir = tmp_path / "source"
    projects_dir = tmp_path / "projects"
    baseline_dir.mkdir()
    source_dir.mkdir()
    blocklist_path = tmp_path / "texture_source_ablation_blocklist.json"
    improve_raw_job._write_texture_source_ablation_blocklist(
        blocklist_path,
        [
            {
                "rejected": True,
                "reason": "strict_face_detail_metric_regression",
                "sha256": "persisted-bad-hash",
                "source_frame_id": "persisted_bad_face",
                "regressed_metric_names": ["raw_glb_face_feature_eye_detail_luma_std"],
            }
        ],
        source_paths=(),
    )
    calls: list[dict[str, object]] = []

    def fake_run_baseline_geometry_texture_bake(
        project_dir,
        *,
        baseline_project_dir,
        texture_source_project_dir,
        rejected_texture_source_hashes,
        rejected_texture_source_frame_ids,
        force,
        probe_texture_source_frame_ids=(),
        probe_texture_source_ablation_overrides=None,
    ):
        calls.append(
            {
                "rejected_texture_source_hashes": rejected_texture_source_hashes,
                "rejected_texture_source_frame_ids": rejected_texture_source_frame_ids,
                "probe_texture_source_frame_ids": probe_texture_source_frame_ids,
                "probe_texture_source_ablation_overrides": (
                    probe_texture_source_ablation_overrides or {}
                ),
            }
        )
        quality_dir = project_dir / "outputs" / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        summary_path = quality_dir / "raw_pipeline_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "basecolor_observed_texel_fraction": 0.5,
                        "face_count": 20,
                        "joint_projection_mean_keypoint_loss": 0.1,
                        "overlay_mean_silhouette_iou": 0.31,
                        "raw_baseline_geometry_texture_bake_used": 1.0,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 120,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_frame_count": 2,
                        "raw_texture_frame_count": 2,
                        "raw_texture_only_added_unique_frame_count": 0,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 1,
                        "silhouette_residual_mean_union_error": 0.6,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            )
        )
        (quality_dir / "baseline_geometry_texture_bake.json").write_text(
            json.dumps(
                {
                    "selected_frame_ids": ["body_a", "body_b"],
                    "texture_frame_ids": ["body_a", "body_b"],
                    "added_unique_texture_frame_count": 0,
                    "rejected_ablation_texture_frame_count": 1,
                    "rejected_ablation_texture_frames": [
                        {
                            "source_frame_id": "persisted_bad_face",
                            "sha256": "persisted-bad-hash",
                        }
                    ],
                    "skipped_duplicate_texture_frame_count": 0,
                }
            )
        )
        return SimpleNamespace(
            added_unique_texture_frame_count=0,
            promotion_guard_passed=False,
            promotion_guard_path=quality_dir / "texture_only_promotion_guard.json",
            success=False,
            summary_path=summary_path,
        )

    monkeypatch.setattr(
        improve_raw_job,
        "run_baseline_geometry_texture_bake",
        fake_run_baseline_geometry_texture_bake,
    )

    report = improve_raw_job._run_baseline_geometry_texture_candidate(
        {
            "candidate_id": "source_candidate",
            "project_dir": str(source_dir),
        },
        baseline_project_dir=baseline_dir,
        projects_dir=projects_dir,
        texture_source_blocklist_path=blocklist_path,
        force=True,
    )

    assert calls == [
        {
            "rejected_texture_source_hashes": ("persisted-bad-hash",),
            "rejected_texture_source_frame_ids": ("persisted_bad_face",),
            "probe_texture_source_frame_ids": (),
            "probe_texture_source_ablation_overrides": {},
        }
    ]
    assert report["texture_only_prior_rejection_hash_count"] == 1
    assert report["texture_only_rejected_ablation_frame_count"] == 1


def test_run_baseline_geometry_texture_candidates_probes_ranked_rejected_sources(
    tmp_path,
    monkeypatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    source_dir = tmp_path / "source"
    projects_dir = tmp_path / "projects"
    baseline_dir.mkdir()
    source_dir.mkdir()
    blocklist_path = tmp_path / "texture_source_ablation_blocklist.json"
    improve_raw_job._write_texture_source_ablation_blocklist(
        blocklist_path,
        [
            {
                "rejected": True,
                "reason": "strict_face_detail_metric_regression",
                "sha256": "blocked-hash",
                "source_frame_id": "blocked_frame",
            }
        ],
        source_paths=(),
    )
    calls: list[dict[str, object]] = []

    def fake_run_baseline_geometry_texture_bake(
        project_dir,
        *,
        baseline_project_dir,
        texture_source_project_dir,
        rejected_texture_source_hashes,
        rejected_texture_source_frame_ids,
        force,
        probe_texture_source_frame_ids=(),
        probe_texture_source_ablation_overrides=None,
    ):
        del baseline_project_dir, texture_source_project_dir, force
        del probe_texture_source_ablation_overrides
        calls.append(
            {
                "project_dir": project_dir,
                "rejected_texture_source_hashes": rejected_texture_source_hashes,
                "rejected_texture_source_frame_ids": rejected_texture_source_frame_ids,
                "probe_texture_source_frame_ids": probe_texture_source_frame_ids,
            }
        )
        quality_dir = project_dir / "outputs" / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        summary_path = quality_dir / "raw_pipeline_summary.json"
        is_probe = bool(probe_texture_source_frame_ids)
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "basecolor_observed_texel_fraction": 0.5,
                        "face_count": 20,
                        "joint_projection_mean_keypoint_loss": 0.1,
                        "overlay_mean_silhouette_iou": 0.31,
                        "raw_baseline_geometry_texture_bake_used": 1.0,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 120,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_frame_count": 2,
                        "raw_texture_frame_count": 3 if is_probe else 2,
                        "raw_texture_only_added_unique_frame_count": 1 if is_probe else 0,
                        "raw_texture_only_probe_added_frame_count": 1 if is_probe else 0,
                        "raw_texture_only_probe_frame_count": 1 if is_probe else 0,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 1,
                        "silhouette_residual_mean_union_error": 0.6,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            )
        )
        preflight_path = quality_dir / "texture_source_preflight.json"
        preflight_path.write_text(
            json.dumps(
                {
                    "ranked_frames": [
                        {
                            "accepted_for_bake": False,
                            "duplicate": False,
                            "sha256": "blocked-hash",
                            "source_frame_id": "blocked_frame",
                        },
                        {
                            "accepted_for_bake": False,
                            "duplicate": False,
                            "quality_gate": {
                                "severe_reject_reasons": ["border_partial_body"],
                            },
                            "rank": 2,
                            "rank_score": 0.4,
                            "sha256": "severe-hash",
                            "source_frame_id": "severe_frame",
                        },
                        {
                            "accepted_for_bake": False,
                            "duplicate": False,
                            "quality_gate": {"severe_reject_reasons": []},
                            "sha256": "probe-a-hash",
                            "source_frame_id": "probe_a",
                        },
                        {
                            "accepted_for_bake": False,
                            "duplicate": True,
                            "sha256": "dup-hash",
                            "source_frame_id": "duplicate",
                        },
                        {
                            "accepted_for_bake": True,
                            "duplicate": False,
                            "sha256": "accepted-hash",
                            "source_frame_id": "accepted",
                        },
                        {
                            "accepted_for_bake": False,
                            "duplicate": False,
                            "sha256": "probe-b-hash",
                            "source_frame_id": "probe_b",
                        },
                    ]
                }
            )
        )
        bake_path = quality_dir / "baseline_geometry_texture_bake.json"
        bake_path.write_text(
            json.dumps(
                {
                    "selected_frame_ids": ["body_a", "body_b"],
                    "texture_frame_ids": (
                        ["body_a", "body_b", "texture_probe_a"]
                        if is_probe
                        else ["body_a", "body_b"]
                    ),
                    "added_unique_texture_frame_count": 1 if is_probe else 0,
                    "probe_added_texture_frame_count": 1 if is_probe else 0,
                    "probe_added_texture_frames": (
                        [{"source_frame_id": "probe_a"}] if is_probe else []
                    ),
                    "rejected_ablation_texture_frame_count": 1,
                    "rejected_ablation_texture_frames": [
                        {"source_frame_id": "blocked_frame", "sha256": "blocked-hash"}
                    ],
                    "skipped_duplicate_texture_frame_count": 0,
                    "texture_source_preflight_path": str(preflight_path),
                }
            )
        )
        return SimpleNamespace(
            added_unique_texture_frame_count=1 if is_probe else 0,
            promotion_guard_passed=is_probe,
            promotion_guard_path=quality_dir / "texture_only_promotion_guard.json",
            success=is_probe,
            summary_path=summary_path,
        )

    monkeypatch.setattr(
        improve_raw_job,
        "run_baseline_geometry_texture_bake",
        fake_run_baseline_geometry_texture_bake,
    )

    reports = improve_raw_job._run_baseline_geometry_texture_candidates(
        {
            "candidate_id": "source_candidate",
            "project_dir": str(source_dir),
        },
        baseline_project_dir=baseline_dir,
        projects_dir=projects_dir,
        texture_source_blocklist_path=blocklist_path,
        max_texture_source_probes=1,
        force=True,
    )

    assert len(reports) == 2
    assert calls[0]["probe_texture_source_frame_ids"] == ()
    assert calls[1]["probe_texture_source_frame_ids"] == ("probe_a",)
    assert "probe_01_probe_a" in str(calls[1]["project_dir"])
    assert reports[0]["texture_only_added_unique_frame_count"] == 0
    assert reports[0]["texture_only_probe_search"]["probe_source_frame_ids"] == ["probe_a"]
    assert reports[0]["texture_only_probe_search"]["skipped_probe_frames"] == [
        {
            "matched_rejected_sha256": True,
            "matched_rejected_source_frame_id": True,
            "preflight_rank": None,
            "preflight_score": None,
            "reason": "prior_ablation_rejection",
            "sha256": "blocked-hash",
            "source_frame_id": "blocked_frame",
        },
        {
            "preflight_rank": 2,
            "preflight_score": 0.4,
            "reason": "severe_quality_reject",
            "severe_reject_reasons": ["border_partial_body"],
            "sha256": "severe-hash",
            "source_frame_id": "severe_frame",
        }
    ]
    assert reports[1]["texture_only_probe_source_frame_ids"] == ["probe_a"]
    assert reports[1]["texture_only_probe_added_frame_count"] == 1


def test_run_baseline_geometry_texture_candidates_skips_failed_admission(
    tmp_path,
    monkeypatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    source_dir = tmp_path / "source"
    projects_dir = tmp_path / "projects"
    baseline_dir.mkdir()
    source_dir.mkdir()

    def fake_admission(
        project_dir,
        *,
        baseline_project_dir,
        texture_source_project_dir,
        rejected_texture_source_hashes,
        rejected_texture_source_frame_ids,
        force,
    ):
        del baseline_project_dir, texture_source_project_dir
        del rejected_texture_source_hashes, rejected_texture_source_frame_ids, force
        output_path = project_dir / "outputs" / "quality" / "texture_source_admission_audit.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "accepted_preflight_frame_count": 2,
            "duplicate_frame_count": 4,
            "passed": False,
            "protected_face_source_patch_frame_count": 0,
            "rejected_ablation_frame_count": 2,
            "rejected_low_detail_frame_count": 1,
            "rejection_reasons": [
                "has_unique_accepted_texture_frames",
                "has_unique_protected_face_source_patch_frames",
            ],
            "report_path": str(output_path),
            "top_score": 1.7,
            "unique_accepted_frame_count": 0,
        }
        output_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def fail_bake(*args, **kwargs):
        raise AssertionError("texture bake should be skipped")

    monkeypatch.setattr(
        improve_raw_job,
        "_write_texture_source_admission_for_candidate",
        fake_admission,
    )
    monkeypatch.setattr(
        improve_raw_job,
        "run_baseline_geometry_texture_bake",
        fail_bake,
    )

    reports = improve_raw_job._run_baseline_geometry_texture_candidates(
        {
            "candidate_id": "source_candidate",
            "project_dir": str(source_dir),
        },
        baseline_project_dir=baseline_dir,
        projects_dir=projects_dir,
        texture_source_blocklist_path=None,
        max_texture_source_probes=0,
        force=True,
    )

    assert len(reports) == 1
    report = reports[0]
    assert report["success"] is False
    assert report["texture_only_bake_report"] is None
    assert report["texture_source_admission_audit_passed"] is False
    assert report["texture_source_admission_unique_accepted_frame_count"] == 0
    assert report["texture_source_admission_protected_face_frame_count"] == 0
    assert report["texture_source_admission_rejected_ablation_frame_count"] == 2
    assert report["texture_source_admission_rejection_reasons"] == [
        "has_unique_accepted_texture_frames",
        "has_unique_protected_face_source_patch_frames",
    ]
    summary_path = (
        projects_dir
        / "loop_texture_source_candidate_baseline_geometry_texture"
        / "outputs"
        / "quality"
        / "raw_pipeline_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["success"] is False
    assert summary["metrics"]["raw_texture_source_admission_audit_passed"] == 0.0


def test_run_baseline_geometry_texture_candidates_uses_explicit_probe_ids(
    tmp_path,
    monkeypatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    source_dir = tmp_path / "source"
    projects_dir = tmp_path / "projects"
    baseline_dir.mkdir()
    source_dir.mkdir()
    calls: list[tuple[object, tuple[str, ...], dict[str, tuple[str, ...]]]] = []

    def fake_admission(
        project_dir,
        *,
        baseline_project_dir,
        texture_source_project_dir,
        rejected_texture_source_hashes,
        rejected_texture_source_frame_ids,
        force,
    ):
        del baseline_project_dir, texture_source_project_dir
        del rejected_texture_source_hashes, rejected_texture_source_frame_ids, force
        output_path = project_dir / "outputs" / "quality" / "texture_source_admission_audit.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "passed": True,
            "protected_face_source_patch_frame_count": 2,
            "report_path": str(output_path),
            "rejected_ablation_frame_count": 0,
            "top_score": 1.5,
            "unique_accepted_frame_count": 2,
        }
        output_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_bake(
        project_dir,
        *,
        baseline_project_dir,
        texture_source_project_dir,
        rejected_texture_source_hashes,
        rejected_texture_source_frame_ids,
        force,
        probe_texture_source_frame_ids=(),
        probe_texture_source_ablation_overrides=None,
    ):
        del baseline_project_dir, texture_source_project_dir
        del rejected_texture_source_hashes, rejected_texture_source_frame_ids, force
        probe_overrides = probe_texture_source_ablation_overrides or {}
        probe_ids = tuple(probe_texture_source_frame_ids)
        calls.append((project_dir, probe_ids, dict(probe_overrides)))
        quality_dir = project_dir / "outputs" / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        summary_path = quality_dir / "raw_pipeline_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "joint_projection_mean_keypoint_loss": 0.1,
                        "overlay_mean_silhouette_iou": 0.31,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 1,
                        "silhouette_residual_mean_union_error": 0.6,
                    },
                    "success": True,
                }
            ),
            encoding="utf-8",
        )
        bake_path = quality_dir / "baseline_geometry_texture_bake.json"
        bake_path.write_text(
            json.dumps(
                {
                    "selected_frame_ids": ["body_a", "body_b"],
                    "texture_frame_ids": ["body_a", "body_b", f"texture_{probe_ids[0]}"],
                    "probe_added_texture_frame_count": 1,
                    "probe_added_texture_frames": [{"source_frame_id": probe_ids[0]}],
                    "rejected_ablation_texture_frame_count": 0,
                    "rejected_ablation_texture_frames": [],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            added_unique_texture_frame_count=1,
            promotion_guard_passed=False,
            promotion_guard_path=quality_dir / "texture_only_promotion_guard.json",
            success=True,
            summary_path=summary_path,
        )

    monkeypatch.setattr(
        improve_raw_job,
        "_write_texture_source_admission_for_candidate",
        fake_admission,
    )
    monkeypatch.setattr(improve_raw_job, "run_baseline_geometry_texture_bake", fake_bake)

    reports = improve_raw_job._run_baseline_geometry_texture_candidates(
        {
            "candidate_id": "source_candidate",
            "project_dir": str(source_dir),
        },
        baseline_project_dir=baseline_dir,
        projects_dir=projects_dir,
        texture_source_blocklist_path=None,
        max_texture_source_probes=99,
        explicit_texture_source_probe_frame_ids=("frame_a", "frame_b"),
        explicit_texture_source_ablation_overrides={"frame_a": ("a" * 64,)},
        force=True,
    )

    assert [probe_ids for _, probe_ids, _ in calls] == [("frame_a",), ("frame_b",)]
    assert all(probe_ids for _, probe_ids, _ in calls)
    assert calls[0][2] == {"frame_a": ("a" * 64,)}
    assert calls[1][2] == {"frame_a": ("a" * 64,)}
    assert "explicit_probe_01_frame_a" in str(calls[0][0])
    assert "explicit_probe_02_frame_b" in str(calls[1][0])
    assert [report["texture_only_probe_source_frame_ids"] for report in reports] == [
        ["frame_a"],
        ["frame_b"],
    ]
    assert reports[0]["texture_only_probe_ablation_overrides"] == {
        "frame_a": ["a" * 64]
    }


def test_texture_source_probe_search_allows_source_id_reprobe_when_hash_differs(
    tmp_path,
) -> None:
    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    preflight_path = quality_dir / "texture_source_preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "ranked_frames": [
                    {
                        "accepted_for_bake": True,
                        "duplicate": False,
                        "quality_gate": {"severe_reject_reasons": []},
                        "rank": 1,
                        "rank_score": 2.2,
                        "sha256": "new-face-hash",
                        "source_frame_id": "blocked_face",
                    },
                    {
                        "accepted_for_bake": False,
                        "duplicate": False,
                        "quality_gate": {"severe_reject_reasons": []},
                        "rank": 2,
                        "rank_score": 0.5,
                        "sha256": "fallback-face-hash",
                        "source_frame_id": "fallback_face",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    bake_path = quality_dir / "baseline_geometry_texture_bake.json"
    bake_path.write_text(
        json.dumps({"texture_source_preflight_path": str(preflight_path)}),
        encoding="utf-8",
    )

    search = improve_raw_job._texture_source_probe_search_from_bake_report(
        bake_path,
        rejected_texture_source_hashes=("old-face-hash",),
        rejected_texture_source_frame_ids=("blocked_face",),
        max_probe_count=1,
        allow_source_id_reprobe_when_hash_differs=True,
    )

    assert search["probe_source_frame_ids"] == ("blocked_face",)
    assert search["skipped_probe_frames"] == []


def test_run_baseline_geometry_texture_candidates_probe_after_guard_rejects_added_frame(
    tmp_path,
    monkeypatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    source_dir = tmp_path / "source"
    projects_dir = tmp_path / "projects"
    baseline_dir.mkdir()
    source_dir.mkdir()
    blocklist_path = tmp_path / "texture_source_ablation_blocklist.json"
    calls: list[dict[str, object]] = []

    def fake_run_baseline_geometry_texture_bake(
        project_dir,
        *,
        baseline_project_dir,
        texture_source_project_dir,
        rejected_texture_source_hashes,
        rejected_texture_source_frame_ids,
        force,
        probe_texture_source_frame_ids=(),
        probe_texture_source_ablation_overrides=None,
    ):
        del baseline_project_dir, texture_source_project_dir, force
        del probe_texture_source_ablation_overrides
        is_probe = bool(probe_texture_source_frame_ids)
        calls.append(
            {
                "project_dir": project_dir,
                "rejected_texture_source_hashes": rejected_texture_source_hashes,
                "rejected_texture_source_frame_ids": rejected_texture_source_frame_ids,
                "probe_texture_source_frame_ids": probe_texture_source_frame_ids,
            }
        )
        quality_dir = project_dir / "outputs" / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        summary_path = quality_dir / "raw_pipeline_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "basecolor_observed_texel_fraction": 0.5,
                        "face_count": 20,
                        "joint_projection_mean_keypoint_loss": 0.1,
                        "overlay_mean_silhouette_iou": 0.31,
                        "raw_baseline_geometry_texture_bake_used": 1.0,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 120,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_frame_count": 2,
                        "raw_texture_frame_count": 3,
                        "raw_texture_only_added_unique_frame_count": 1,
                        "raw_texture_only_probe_added_frame_count": 1 if is_probe else 0,
                        "raw_texture_only_probe_frame_count": 1 if is_probe else 0,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 1,
                        "silhouette_residual_mean_union_error": 0.6,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            ),
            encoding="utf-8",
        )
        preflight_path = quality_dir / "texture_source_preflight.json"
        preflight_path.write_text(
            json.dumps(
                {
                    "ranked_frames": [
                        {
                            "accepted_for_bake": True,
                            "duplicate": False,
                            "quality_gate": {"severe_reject_reasons": []},
                            "rank": 1,
                            "rank_score": 1.8,
                            "sha256": "bad-face-hash",
                            "source_frame_id": "bad_face",
                        },
                        {
                            "accepted_for_bake": False,
                            "duplicate": False,
                            "quality_gate": {"severe_reject_reasons": []},
                            "rank": 2,
                            "rank_score": 0.45,
                            "sha256": "probe-a-hash",
                            "source_frame_id": "probe_a",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        bake_path = quality_dir / "baseline_geometry_texture_bake.json"
        bake_path.write_text(
            json.dumps(
                {
                    "selected_frame_ids": ["body_a", "body_b"],
                    "texture_frame_ids": (
                        ["body_a", "body_b", "texture_probe_a"]
                        if is_probe
                        else ["body_a", "body_b", "texture_bad_face"]
                    ),
                    "added_unique_texture_frame_count": 1,
                    "probe_added_texture_frame_count": 1 if is_probe else 0,
                    "probe_added_texture_frames": (
                        [{"source_frame_id": "probe_a"}] if is_probe else []
                    ),
                    "rejected_ablation_texture_frame_count": 0,
                    "rejected_ablation_texture_frames": [],
                    "skipped_duplicate_texture_frame_count": 0,
                    "texture_source_preflight_path": str(preflight_path),
                }
            ),
            encoding="utf-8",
        )
        guard_path = quality_dir / "texture_only_promotion_guard.json"
        guard_payload = {"passed": is_probe}
        if not is_probe:
            guard_payload["supplemental_source_rejections"] = [
                {
                    "rejected": True,
                    "reason": "strict_face_detail_metric_regression",
                    "sha256": "bad-face-hash",
                    "source_frame_id": "bad_face",
                }
            ]
        guard_path.write_text(json.dumps(guard_payload), encoding="utf-8")
        return SimpleNamespace(
            added_unique_texture_frame_count=1,
            promotion_guard_passed=is_probe,
            promotion_guard_path=guard_path,
            success=is_probe,
            summary_path=summary_path,
        )

    monkeypatch.setattr(
        improve_raw_job,
        "run_baseline_geometry_texture_bake",
        fake_run_baseline_geometry_texture_bake,
    )

    reports = improve_raw_job._run_baseline_geometry_texture_candidates(
        {
            "candidate_id": "source_candidate",
            "project_dir": str(source_dir),
        },
        baseline_project_dir=baseline_dir,
        projects_dir=projects_dir,
        texture_source_blocklist_path=blocklist_path,
        max_texture_source_probes=1,
        force=True,
    )

    assert len(reports) == 2
    assert calls[0]["probe_texture_source_frame_ids"] == ()
    assert calls[1]["probe_texture_source_frame_ids"] == ("probe_a",)
    assert calls[1]["rejected_texture_source_hashes"] == ("bad-face-hash",)
    assert calls[1]["rejected_texture_source_frame_ids"] == ("bad_face",)
    assert reports[0]["texture_only_added_unique_frame_count"] == 1
    assert reports[0]["texture_only_probe_retry_reason"] == "guard_rejected_added_frames"
    assert reports[0]["texture_only_probe_search"]["probe_source_frame_ids"] == ["probe_a"]
    assert reports[1]["texture_only_probe_added_frame_count"] == 1
    assert reports[1]["texture_only_promotion_guard_passed"] is True


def test_run_baseline_geometry_texture_fallback_candidates_try_next_source(
    tmp_path,
    monkeypatch,
) -> None:
    baseline_dir = tmp_path / "baseline"
    source_a_dir = tmp_path / "source_a"
    source_b_dir = tmp_path / "source_b"
    projects_dir = tmp_path / "projects"
    for path in (baseline_dir, source_a_dir, source_b_dir):
        path.mkdir()
    calls: list[dict[str, object]] = []

    def fake_run_baseline_geometry_texture_bake(
        project_dir,
        *,
        baseline_project_dir,
        texture_source_project_dir,
        rejected_texture_source_hashes,
        rejected_texture_source_frame_ids,
        force,
        probe_texture_source_frame_ids=(),
        probe_texture_source_ablation_overrides=None,
    ):
        del baseline_project_dir, rejected_texture_source_hashes
        del rejected_texture_source_frame_ids, force, probe_texture_source_frame_ids
        del probe_texture_source_ablation_overrides
        source_name = texture_source_project_dir.name
        calls.append({"source_name": source_name, "project_dir": project_dir})
        quality_dir = project_dir / "outputs" / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        summary_path = quality_dir / "raw_pipeline_summary.json"
        added_count = 1 if source_name == "source_b" else 0
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "basecolor_observed_texel_fraction": 0.5,
                        "face_count": 20,
                        "joint_projection_mean_keypoint_loss": 0.1,
                        "overlay_mean_silhouette_iou": 0.31,
                        "raw_baseline_geometry_texture_bake_used": 1.0,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 120,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_frame_count": 2,
                        "raw_texture_frame_count": 2 + added_count,
                        "raw_texture_only_added_unique_frame_count": added_count,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 1,
                        "silhouette_residual_mean_union_error": 0.6,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            ),
            encoding="utf-8",
        )
        preflight_path = quality_dir / "texture_source_preflight.json"
        preflight_path.write_text(
            json.dumps(
                {
                    "ranked_frames": [
                        {
                            "accepted_for_bake": False,
                            "duplicate": False,
                            "quality_gate": {
                                "severe_reject_reasons": ["border_partial_body"],
                            },
                            "rank": 1,
                            "rank_score": 0.1,
                            "sha256": "severe-a",
                            "source_frame_id": "bad_a",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        texture_frame_ids = ["body_a", "body_b"]
        if added_count:
            texture_frame_ids.append("texture_source_b")
        (quality_dir / "baseline_geometry_texture_bake.json").write_text(
            json.dumps(
                {
                    "selected_frame_ids": ["body_a", "body_b"],
                    "texture_frame_ids": texture_frame_ids,
                    "added_unique_texture_frame_count": added_count,
                    "probe_added_texture_frame_count": 0,
                    "probe_added_texture_frames": [],
                    "rejected_ablation_texture_frame_count": 0,
                    "rejected_ablation_texture_frames": [],
                    "skipped_duplicate_texture_frame_count": 0,
                    "texture_source_preflight_path": str(preflight_path),
                }
            ),
            encoding="utf-8",
        )
        guard_path = quality_dir / "texture_only_promotion_guard.json"
        guard_path.write_text(json.dumps({"passed": bool(added_count)}), encoding="utf-8")
        return SimpleNamespace(
            added_unique_texture_frame_count=added_count,
            promotion_guard_passed=bool(added_count),
            promotion_guard_path=guard_path,
            success=bool(added_count),
            summary_path=summary_path,
        )

    monkeypatch.setattr(
        improve_raw_job,
        "run_baseline_geometry_texture_bake",
        fake_run_baseline_geometry_texture_bake,
    )

    reports = improve_raw_job._run_baseline_geometry_texture_fallback_candidates(
        [
            {"candidate_id": "source_a", "project_dir": str(source_a_dir)},
            {"candidate_id": "source_b", "project_dir": str(source_b_dir)},
            {"candidate_id": "source_c", "project_dir": str(tmp_path / "source_c")},
        ],
        baseline_project_dir=baseline_dir,
        projects_dir=projects_dir,
        texture_source_blocklist_path=None,
        max_texture_source_probes=1,
        force=True,
    )

    assert [call["source_name"] for call in calls] == ["source_a", "source_b"]
    assert [report["texture_source_fallback_rank"] for report in reports] == [1, 2]
    assert all(report["texture_source_fallback_count"] == 3 for report in reports)
    assert reports[0]["texture_only_probe_search"]["skipped_probe_frames"][0][
        "source_frame_id"
    ] == "bad_a"
    assert reports[1]["texture_only_added_unique_frame_count"] == 1
    assert reports[1]["texture_only_guarded_result_success"] is True


def test_texture_source_project_report_uses_stable_project_identity(tmp_path) -> None:
    project_dir = tmp_path / "source project:01"
    project_dir.mkdir()

    report = improve_raw_job._texture_source_project_report(project_dir)

    assert report["candidate_id"] == "source_project_01"
    assert report["base_candidate_id"] == "source_project_01"
    assert report["geometry_variant_id"] == "texture_only_source_project"
    assert report["backend"] == "texture_only_source_project"
    assert report["project_dir"] == str(project_dir.resolve())
    assert report["success"] is True
    assert report["solid"] is True
    assert report["metrics"] == {}


def test_should_run_baseline_texture_fallback_skips_texture_only_source_mode(
    tmp_path,
) -> None:
    pipeline_reports = [{"candidate_id": "candidate_a"}]

    assert improve_raw_job._should_run_baseline_texture_fallback(
        baseline_project_dir=tmp_path / "baseline",
        pipeline_reports=pipeline_reports,
        texture_only_source_project=None,
    )
    assert not improve_raw_job._should_run_baseline_texture_fallback(
        baseline_project_dir=tmp_path / "baseline",
        pipeline_reports=pipeline_reports,
        texture_only_source_project={"candidate_id": "source_project"},
    )
    assert not improve_raw_job._should_run_baseline_texture_fallback(
        baseline_project_dir=None,
        pipeline_reports=pipeline_reports,
        texture_only_source_project=None,
    )
    assert not improve_raw_job._should_run_baseline_texture_fallback(
        baseline_project_dir=tmp_path / "baseline",
        pipeline_reports=[],
        texture_only_source_project=None,
    )


def test_texture_source_reports_for_baseline_bake_skips_missing_texture_manifest(
    tmp_path,
) -> None:
    valid_project = tmp_path / "valid_project"
    invalid_project = tmp_path / "invalid_project"
    (valid_project / "observations" / "raw").mkdir(parents=True)
    invalid_project.mkdir()
    (valid_project / "observations" / "raw" / "texture_manifest.json").write_text(
        json.dumps({"frames": [{"frame_id": "frame_0000"}]}),
        encoding="utf-8",
    )

    reports = improve_raw_job._texture_source_reports_for_baseline_bake(
        [
            {
                "candidate_id": "invalid",
                "project_dir": str(invalid_project),
                "success": True,
                "solid": True,
                "metrics": {"raw_quality_passed_check_count": 10},
            },
            {
                "candidate_id": "valid",
                "project_dir": str(valid_project),
                "success": False,
                "solid": False,
                "metrics": {},
            },
        ],
        max_reports=2,
    )

    assert [report["candidate_id"] for report in reports] == ["valid"]


def test_ordered_texture_source_reports_prefers_uploaded_source(tmp_path) -> None:
    uploaded_project = tmp_path / "uploaded_project"
    older_project = tmp_path / "older_project"
    (uploaded_project / "observations" / "raw").mkdir(parents=True)
    (older_project / "observations" / "raw").mkdir(parents=True)
    for project in (uploaded_project, older_project):
        (project / "observations" / "raw" / "texture_manifest.json").write_text(
            json.dumps({"frames": [{"frame_id": "frame_0000"}]}),
            encoding="utf-8",
        )

    reports = improve_raw_job._ordered_texture_source_reports(
        [
            {
                "candidate_id": "older",
                "project_dir": str(older_project),
                "success": True,
                "solid": True,
                "metrics": {"raw_quality_passed_check_count": 10},
            }
        ],
        uploaded_texture_source_report={
            "candidate_id": "uploaded",
            "project_dir": str(uploaded_project),
            "success": True,
            "solid": False,
            "metrics": {},
        },
        max_reports=2,
    )

    assert [report["candidate_id"] for report in reports] == ["uploaded", "older"]


def test_build_uploaded_texture_source_report_prioritizes_closeup_texture(
    tmp_path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_frame_paths = []
    for index in range(4):
        path = raw_dir / f"input_{index}.jpg"
        Image.new("RGB", (12, 8), color=(230 - index, 230, 230)).save(path)
        raw_frame_paths.append(path)

    def fake_ingest_image_sources(project_dir, *, input_paths, min_frames, force):
        assert input_paths == raw_frame_paths
        assert min_frames == 2
        assert force is True
        (project_dir / "raw").mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(project_dir=project_dir)

    def fake_segment_raw_image_observations(
        project_dir,
        *,
        color_delta_threshold,
        segmentation_backend,
        force,
    ):
        assert color_delta_threshold == 0.08
        assert segmentation_backend == "white_subject_v1"
        assert force is True
        observations_dir = project_dir / "observations" / "raw"
        observations_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = observations_dir / "manifest.json"
        frames = [
            _test_frame_manifest(
                "frame_0000",
                raw_frame_paths[0],
                bbox=[10, 30, 90, 70],
                media_type="photo",
                source_media_id="body_a.jpg",
            ),
            _test_frame_manifest(
                "frame_0001",
                raw_frame_paths[1],
                bbox=[8, 28, 92, 72],
                media_type="video",
                source_type="video_frame",
                source_media_id="v1.mp4",
                source_frame_index=7,
            ),
            _test_frame_manifest(
                "frame_0002",
                raw_frame_paths[2],
                bbox=[15, 5, 85, 95],
                media_type="photo",
                source_media_id="face_closeup.jpg",
            ),
            _test_frame_manifest(
                "frame_0003",
                raw_frame_paths[3],
                bbox=[22, 12, 78, 88],
                media_type="photo",
                source_media_id="face_oblique.jpg",
            ),
        ]
        manifest_path.write_text(
            json.dumps({"frames": frames, "project_id": project_dir.name}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            project_dir=project_dir,
            manifest_path=manifest_path,
            frame_count=len(frames),
            mean_mask_area_ratio=0.18,
            mean_bbox_area_ratio=0.32,
            full_frame_bbox_fraction=0.0,
        )

    def fake_bootstrap_raw_keypoints_and_pose(
        project_dir,
        *,
        keypoint_confidence,
        force,
    ):
        assert keypoint_confidence == 0.5
        assert force is True
        return SimpleNamespace(
            observed_face_anchor_count=8,
            observed_face_anchor_frame_count=2,
            mean_face_anchor_confidence=0.42,
        )

    captured_selection: dict[str, object] = {}

    def fake_select_raw_observation_frames(
        project_dir,
        *,
        min_frames,
        max_selected_frames,
        selected_frame_ids,
        texture_frame_ids,
        selection_label,
        force,
    ):
        captured_selection.update(
            {
                "min_frames": min_frames,
                "max_selected_frames": max_selected_frames,
                "selected_frame_ids": tuple(selected_frame_ids),
                "texture_frame_ids": tuple(texture_frame_ids),
                "selection_label": selection_label,
                "force": force,
            }
        )
        observations_dir = project_dir / "observations" / "raw"
        report_path = observations_dir / "frame_selection.json"
        texture_manifest_path = observations_dir / "texture_manifest.json"
        report_path.write_text(json.dumps({"frames": []}), encoding="utf-8")
        texture_manifest_path.write_text(
            json.dumps({"frames": [{"frame_id": frame_id} for frame_id in texture_frame_ids]}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            report_path=report_path,
            texture_manifest_path=texture_manifest_path,
            detail_metrics={},
        )

    monkeypatch.setattr(improve_raw_job, "ingest_image_sources", fake_ingest_image_sources)
    monkeypatch.setattr(
        improve_raw_job,
        "segment_raw_image_observations",
        fake_segment_raw_image_observations,
    )
    monkeypatch.setattr(
        improve_raw_job,
        "bootstrap_raw_keypoints_and_pose",
        fake_bootstrap_raw_keypoints_and_pose,
    )
    monkeypatch.setattr(
        improve_raw_job,
        "select_raw_observation_frames",
        fake_select_raw_observation_frames,
    )

    report = improve_raw_job._build_uploaded_texture_source_report(
        raw_frame_paths=raw_frame_paths,
        source_candidate=improve_raw_job.SegmentationCandidate(
            candidate_id="source_candidate",
            backend="white_subject_v1",
            color_delta_threshold=0.08,
            project_dir=tmp_path / "seg_source",
            selected_frames=(),
            frame_count=4,
            selected_frame_count=2,
            mean_mask_area_ratio=0.18,
            mean_bbox_area_ratio=0.32,
            full_frame_bbox_fraction=0.0,
            score=1.0,
        ),
        loop_dir=tmp_path / "loop",
        min_geometry_frames=2,
        max_texture_frames=4,
        force=True,
    )

    assert report is not None
    assert set(captured_selection["selected_frame_ids"]) == {"frame_0000", "frame_0001"}
    assert "frame_0002" in captured_selection["texture_frame_ids"]
    assert "frame_0003" in captured_selection["texture_frame_ids"]
    assert captured_selection["selection_label"] == "uploaded_closeup_texture_source"
    assert report["candidate_id"] == "uploaded_texture_source_source_candidate"
    assert report["closeup_texture_frame_count"] >= 1
    assert report["metrics"]["raw_face_priority_texture_used"] == 1.0
    assert report["metrics"]["raw_uploaded_source_observed_face_anchor_count"] == 8
    assert report["metrics"]["raw_uploaded_source_observed_face_anchor_frame_count"] == 2
    assert report["metrics"]["raw_uploaded_source_mean_face_anchor_confidence"] == 0.42
    assert (tmp_path / "loop" / "texture_source_upload_source_candidate").exists()


def test_run_full_candidate_preserves_selected_video_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "segmentation_raw_images"
    source_dir.mkdir()
    photo_path = source_dir / "frame_0000.jpg"
    video_frame_path = source_dir / "frame_0008.jpg"
    rejected_texture_path = source_dir / "frame_0010.jpg"
    Image.new("RGB", (8, 6), color=(240, 240, 240)).save(photo_path)
    Image.new("RGB", (8, 6), color=(40, 60, 80)).save(video_frame_path)
    Image.new("RGB", (8, 6), color=(250, 220, 220)).save(rejected_texture_path)

    candidate = improve_raw_job.SegmentationCandidate(
        candidate_id="candidate_video",
        backend="white_subject_v1",
        color_delta_threshold=0.08,
        project_dir=tmp_path / "seg_candidate",
        selected_frames=(
            improve_raw_job.FrameSelection(
                frame_id="frame_0000",
                source_path=photo_path,
                media_type="photo",
                source_type="image",
                source_media_id="11.jpg",
                source_frame_index=None,
                geometry_reject_reasons=(),
                geometry_body_fit_score=0.9,
                score=0.9,
                mask_area_ratio=0.1,
                bbox_area_ratio=0.2,
                full_frame_bbox=False,
            ),
            improve_raw_job.FrameSelection(
                frame_id="frame_0008",
                source_path=video_frame_path,
                media_type="video",
                source_type="video_frame",
                source_media_id="v1.mp4",
                source_frame_index=265,
                geometry_reject_reasons=(),
                geometry_body_fit_score=0.8,
                score=0.8,
                mask_area_ratio=0.1,
                bbox_area_ratio=0.2,
                full_frame_bbox=False,
            ),
            improve_raw_job.FrameSelection(
                frame_id="frame_0010",
                source_path=rejected_texture_path,
                media_type="photo",
                source_type="image",
                source_media_id="closeup.jpg",
                source_frame_index=None,
                geometry_reject_reasons=("sparse_geometry_mask",),
                geometry_body_fit_score=0.1,
                score=0.7,
                mask_area_ratio=0.03,
                bbox_area_ratio=0.12,
                full_frame_bbox=False,
            ),
        ),
        frame_count=3,
        selected_frame_count=3,
        mean_mask_area_ratio=0.1,
        mean_bbox_area_ratio=0.2,
        full_frame_bbox_fraction=0.0,
        score=1.0,
    )
    captured: dict[str, object] = {}

    def fake_run_raw_pipeline(
        project_dir,
        *,
        input_paths,
        min_frames,
        color_delta_threshold,
        segmentation_backend,
        camera_root_steps,
        max_selected_frames,
        selected_frame_ids,
        texture_frame_ids,
        frame_selection_label,
        force,
    ):
        del min_frames, color_delta_threshold, segmentation_backend, camera_root_steps, force
        selected_input_dir = input_paths[0].parent
        captured["input_names"] = [path.name for path in input_paths]
        captured["max_selected_frames"] = max_selected_frames
        captured["selected_frame_ids"] = selected_frame_ids
        captured["texture_frame_ids"] = texture_frame_ids
        captured["frame_selection_label"] = frame_selection_label
        captured["manifest"] = json.loads(
            (selected_input_dir / "media_manifest.json").read_text()
        )
        summary_path = project_dir / "outputs" / "quality" / "raw_pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {"glb": str(project_dir / "outputs/avatar/cat_avatar.glb")},
                    "metrics": {
                        "face_count": 20,
                        "keypoint_loss_reduction": 0.1,
                        "overlay_mean_silhouette_iou": 0.3,
                        "raw_face_priority_texture_used": 1.0,
                        "raw_facial_feature_count": 14,
                        "raw_facial_feature_painted_texel_count": 100,
                        "raw_full_frame_bbox_fraction": 0.0,
                        "raw_mean_bbox_area_ratio": 0.2,
                        "raw_mean_mask_area_ratio": 0.1,
                        "raw_quality_passed_check_count": 10,
                        "raw_quality_total_check_count": 10,
                        "raw_selected_video_frame_count": 1,
                        "raw_texture_foreground_fallback_luma": 0.8,
                        "raw_texture_unknown_luma": 0.7,
                        "raw_texture_unknown_luma_fraction_of_foreground": 0.9,
                        "raw_texture_video_frame_count": 1,
                        "silhouette_loss_reduction": 0.2,
                        "silhouette_residual_mean_union_error": 0.7,
                        "vertex_count": 10,
                    },
                    "success": True,
                }
            )
        )
        return SimpleNamespace(success=True, summary_path=summary_path)

    monkeypatch.setattr(improve_raw_job, "run_raw_pipeline", fake_run_raw_pipeline)

    report = improve_raw_job._run_full_candidate(
        candidate,
        projects_dir=tmp_path / "projects",
        rank=1,
        min_selected_frames=2,
        camera_root_steps=2,
        force=True,
    )

    manifest = captured["manifest"]
    assert captured["input_names"] == ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]
    assert captured["max_selected_frames"] == 2
    assert captured["selected_frame_ids"] == ("frame_0000", "frame_0001")
    assert captured["texture_frame_ids"] == ("frame_0000", "frame_0001", "frame_0002")
    assert captured["frame_selection_label"] == "improve_raw_job_geometry_texture_split"
    assert manifest["frames"]["frame_0000.jpg"]["media_type"] == "photo"
    assert manifest["frames"]["frame_0000.jpg"]["source_media_id"] == "11.jpg"
    assert manifest["frames"]["frame_0001.jpg"]["media_type"] == "video"
    assert manifest["frames"]["frame_0001.jpg"]["source_type"] == "video_frame"
    assert manifest["frames"]["frame_0001.jpg"]["source_media_id"] == "v1.mp4"
    assert manifest["frames"]["frame_0001.jpg"]["source_frame_index"] == 265
    assert manifest["frames"]["frame_0002.jpg"]["media_type"] == "photo"
    assert manifest["frames"]["frame_0002.jpg"]["source_media_id"] == "closeup.jpg"
    assert report["metrics"]["raw_texture_video_frame_count"] == 1


def _test_frame_manifest(
    frame_id,
    raw_source_path,
    *,
    bbox,
    media_type="photo",
    source_type="image",
    source_media_id="source.jpg",
    source_frame_index=None,
):
    image_width = 100
    image_height = 100
    bbox_area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (
        image_width * image_height
    )
    frame = {
        "frame_id": frame_id,
        "raw_source_path": str(raw_source_path),
        "source_type": source_type,
        "media_type": media_type,
        "source_media_id": source_media_id,
        "image_width": image_width,
        "image_height": image_height,
        "bbox_xyxy": bbox,
        "quality": {
            "mask_area_ratio": min(0.24, bbox_area_ratio * 0.55),
            "bbox_area_ratio": bbox_area_ratio,
            "touches_image_border": 0.0,
            "full_frame_bbox": 0.0,
        },
    }
    if source_frame_index is not None:
        frame["source_frame_index"] = source_frame_index
    return frame
