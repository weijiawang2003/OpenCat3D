import json
from pathlib import Path

from opencat3d.preprocess.frame_selection import (
    deduplicate_raw_texture_manifest_by_image_hash,
    map_raw_frame_selection_by_image_hash,
    select_raw_observation_frames,
)


def test_select_raw_observation_frames_filters_and_rewrites_manifest(tmp_path) -> None:
    project_dir = tmp_path / "project"
    manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0",
        "project_id": "demo",
        "frame_count": 7,
        "mean_mask_area_ratio": 0.0,
        "mean_bbox_area_ratio": 0.0,
        "full_frame_bbox_fraction": 0.0,
        "frames": [
            _frame("frame_0000", mask=0.16, bbox_area=0.22, bbox=(10, 20, 171, 76)),
            _frame("frame_0001", mask=0.31, bbox_area=0.54, bbox=(0, 0, 108, 200), touch=1.0),
            _frame("frame_0002", mask=0.01, bbox_area=0.02, bbox=(80, 50, 100, 82)),
            _frame("frame_0003", mask=0.26, bbox_area=0.39, bbox=(15, 45, 177, 141)),
            _frame("frame_0004", mask=0.16, bbox_area=0.40, bbox=(60, 20, 185, 149)),
            _frame("frame_0005", mask=0.27, bbox_area=0.63, bbox=(0, 10, 191, 142)),
            _frame("frame_0006", mask=0.20, bbox_area=0.47, bbox=(0, 35, 114, 200)),
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = select_raw_observation_frames(
        project_dir,
        min_frames=2,
        max_selected_frames=3,
        force=True,
    )

    assert result.input_frame_count == 7
    assert result.selected_frame_count == 3
    assert result.rejected_frame_count == 4
    assert result.selected_frame_ids == ("frame_0000", "frame_0003", "frame_0004")
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["frame_count"] == 3
    assert updated["input_frame_count"] == 7
    assert updated["frame_selection"]["applied"] is True
    assert updated["frame_selection"]["texture_frame_count"] == 6
    assert [frame["frame_id"] for frame in updated["frames"]] == [
        "frame_0000",
        "frame_0003",
        "frame_0004",
    ]
    texture_manifest = json.loads(result.texture_manifest_path.read_text(encoding="utf-8"))
    assert texture_manifest["texture_frame_ids"] == [
        "frame_0000",
        "frame_0001",
        "frame_0003",
        "frame_0004",
        "frame_0005",
        "frame_0006",
    ]
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["selected_frame_ids"] == ["frame_0000", "frame_0003", "frame_0004"]
    rejected = {
        frame["frame_id"]: frame["reject_reasons"]
        for frame in report["frames"]
        if frame["rejected_by_quality"]
    }
    assert {"mask_too_small", "bbox_too_small"}.issubset(rejected["frame_0002"])
    assert rejected["frame_0005"] == ["bbox_too_large"]


def test_select_raw_observation_frames_balances_photo_and_video(tmp_path) -> None:
    project_dir = tmp_path / "project"
    manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    frames = []
    for index in range(8):
        frames.append(
            {
                **_frame(
                    f"frame_{index:04d}",
                    mask=0.18,
                    bbox_area=0.34,
                    bbox=(20, 40, 180, 125),
                ),
                "source_type": "video_frame",
                "media_type": "video",
            }
        )
    for index in range(8, 12):
        frames.append(
            {
                **_frame(
                    f"frame_{index:04d}",
                    mask=0.26,
                    bbox_area=0.42,
                    bbox=(18, 35, 176, 135),
                ),
                "source_type": "image",
                "media_type": "photo",
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_id": "demo",
                "frame_count": len(frames),
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )

    result = select_raw_observation_frames(
        project_dir,
        min_frames=2,
        max_selected_frames=6,
        force=True,
    )

    assert result.selected_frame_count == 6
    assert result.texture_frame_count == 12
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["media_strategy"] == "balanced_photo_video"
    assert report["selected_media_counts"] == {"photo": 2, "video": 4}
    assert report["texture_media_counts"] == {"photo": 4, "video": 8}
    assert any(
        frame_id in report["selected_frame_ids"] for frame_id in ["frame_0008", "frame_0009"]
    )
    texture_manifest = json.loads(result.texture_manifest_path.read_text(encoding="utf-8"))
    assert texture_manifest["texture_frame_count"] == 12
    assert texture_manifest["texture_policy"] == "selected_frames_plus_quality_media"


def test_select_raw_observation_frames_keeps_face_closeups_texture_only(tmp_path) -> None:
    project_dir = tmp_path / "project"
    manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    frames = [
        {
            **_frame(
                "photo_close_face",
                mask=0.31,
                bbox_area=0.66,
                bbox=(14, 18, 184, 178),
            ),
            "source_type": "image",
            "media_type": "photo",
        },
        {
            **_frame(
                "photo_body_side",
                mask=0.17,
                bbox_area=0.33,
                bbox=(18, 48, 178, 130),
            ),
            "source_type": "image",
            "media_type": "photo",
        },
        {
            **_frame(
                "video_body_walk",
                mask=0.19,
                bbox_area=0.36,
                bbox=(20, 42, 182, 134),
            ),
            "source_type": "video_frame",
            "media_type": "video",
        },
        {
            **_frame(
                "video_body_turn",
                mask=0.20,
                bbox_area=0.38,
                bbox=(24, 40, 180, 136),
            ),
            "source_type": "video_frame",
            "media_type": "video",
        },
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_id": "demo",
                "frame_count": len(frames),
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )

    result = select_raw_observation_frames(
        project_dir,
        min_frames=2,
        max_selected_frames=2,
        force=True,
    )

    assert "photo_close_face" not in result.selected_frame_ids
    assert set(result.selected_frame_ids) == {"photo_body_side", "video_body_walk"}
    texture_manifest = json.loads(result.texture_manifest_path.read_text(encoding="utf-8"))
    assert "photo_close_face" in texture_manifest["texture_frame_ids"]
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    close_face = next(
        frame for frame in report["frames"] if frame["frame_id"] == "photo_close_face"
    )
    assert close_face["rejected_by_quality"] is True
    assert close_face["texture_detail_candidate"] is True
    assert close_face["selected"] is False
    assert close_face["texture_enabled"] is True
    assert report["metrics"]["texture_only_detail_frame_count"] == 1.0
    assert report["metrics"]["geometry_selected_body_priority_frame_count"] == 2.0


def test_select_raw_observation_frames_rejects_alignment_risk_geometry(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    frames = [
        {
            **_frame(
                "safe_body_photo",
                mask=0.18,
                bbox_area=0.34,
                bbox=(18, 45, 180, 129),
            ),
            "source_type": "image",
            "media_type": "photo",
        },
        {
            **_frame(
                "safe_body_video",
                mask=0.22,
                bbox_area=0.40,
                bbox=(25, 20, 184, 142),
            ),
            "source_type": "video_frame",
            "media_type": "video",
        },
        {
            **_frame(
                "border_partial_body",
                mask=0.12,
                bbox_area=0.34,
                bbox=(0, 0, 68, 200),
                touch=1.0,
            ),
            "source_type": "image",
            "media_type": "photo",
        },
        {
            **_frame(
                "sparse_body_mask",
                mask=0.052,
                bbox_area=0.16,
                bbox=(112, 12, 170, 124),
            ),
            "source_type": "image",
            "media_type": "photo",
        },
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_id": "demo",
                "frame_count": len(frames),
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )

    result = select_raw_observation_frames(
        project_dir,
        min_frames=2,
        max_selected_frames=4,
        force=True,
    )

    assert set(result.selected_frame_ids) == {"safe_body_photo", "safe_body_video"}
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    rejected = {
        frame["frame_id"]: frame["reject_reasons"]
        for frame in report["frames"]
        if frame["rejected_by_quality"]
    }
    assert "border_partial_body" in rejected["border_partial_body"]
    assert "sparse_geometry_mask" in rejected["sparse_body_mask"]
    assert report["selected_frame_count"] == 2


def test_select_raw_observation_frames_applies_frame_set_override(tmp_path) -> None:
    project_dir = tmp_path / "project"
    manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    frames = [
        {
            **_frame("frame_0000", mask=0.16, bbox_area=0.30, bbox=(20, 30, 170, 130)),
            "source_type": "image",
            "media_type": "photo",
        },
        {
            **_frame("frame_0001", mask=0.18, bbox_area=0.34, bbox=(20, 30, 180, 128)),
            "source_type": "video_frame",
            "media_type": "video",
        },
        {
            **_frame("frame_0002", mask=0.20, bbox_area=0.36, bbox=(22, 32, 178, 130)),
            "source_type": "video_frame",
            "media_type": "video",
        },
        {
            **_frame("frame_0003", mask=0.32, bbox_area=0.62, bbox=(0, 10, 190, 150)),
            "source_type": "image",
            "media_type": "photo",
        },
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_id": "demo",
                "frame_count": len(frames),
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )

    result = select_raw_observation_frames(
        project_dir,
        min_frames=2,
        max_selected_frames=2,
        selected_frame_ids=["frame_0001", "frame_0002"],
        texture_frame_ids=["frame_0000"],
        selection_label="video_dense_body_geometry",
        force=True,
    )

    assert result.selected_frame_ids == ("frame_0001", "frame_0002")
    assert result.texture_frame_count == 3
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["override_applied"] is True
    assert report["selection_label"] == "video_dense_body_geometry"
    assert report["media_strategy"] == "override:video_dense_body_geometry"
    assert report["selected_media_counts"] == {"photo": 0, "video": 2}
    assert report["texture_policy"] == "override_plus_selected_geometry"
    assert report["texture_frame_ids"] == ["frame_0000", "frame_0001", "frame_0002"]
    texture_manifest = json.loads(result.texture_manifest_path.read_text(encoding="utf-8"))
    assert texture_manifest["override_applied"] is True
    assert texture_manifest["texture_policy"] == "override_plus_selected_geometry"


def test_select_raw_observation_frames_rejects_unknown_override_ids(tmp_path) -> None:
    project_dir = tmp_path / "project"
    manifest_path = project_dir / "observations" / "raw" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_id": "demo",
                "frame_count": 2,
                "frames": [
                    _frame("frame_0000", mask=0.16, bbox_area=0.30, bbox=(20, 30, 170, 130)),
                    _frame("frame_0001", mask=0.18, bbox_area=0.34, bbox=(20, 30, 180, 128)),
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        select_raw_observation_frames(
            project_dir,
            min_frames=2,
            max_selected_frames=2,
            selected_frame_ids=["frame_0000", "missing_frame"],
            force=True,
        )
    except ValueError as exc:
        assert "unknown frame ids" in str(exc)
    else:
        raise AssertionError("unknown override frame id should fail")


def test_map_raw_frame_selection_by_image_hash_handles_id_shift(tmp_path) -> None:
    reference_dir = tmp_path / "reference"
    target_dir = tmp_path / "target"
    _write_observation_image(reference_dir, "frame_0001", b"body pixels")
    _write_observation_image(reference_dir, "frame_0002", b"face pixels")
    _write_observation_image(target_dir, "frame_0000", b"unrelated pixels")
    _write_observation_image(target_dir, "frame_0042", b"face pixels")
    _write_observation_image(target_dir, "frame_0047", b"body pixels")
    selection_path = reference_dir / "observations" / "raw" / "frame_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selected_frame_ids": ["frame_0001"],
                "texture_frame_ids": ["frame_0002", "frame_0001"],
            }
        ),
        encoding="utf-8",
    )

    result = map_raw_frame_selection_by_image_hash(reference_dir, target_dir)

    assert result.selected_frame_ids == ("frame_0047",)
    assert result.texture_frame_ids == ("frame_0042", "frame_0047")


def test_deduplicate_raw_texture_manifest_by_image_hash_preserves_geometry_frames(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    observations_dir = project_dir / "observations" / "raw"
    raw_dir = project_dir / "raw"
    observations_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    frames = [
        _texture_frame("frame_0000", source_index=0, media_type="photo"),
        _texture_frame("frame_0001", source_index=1, media_type="video"),
        _texture_frame("frame_0002", source_index=2, media_type="photo"),
        _texture_frame("frame_0003", source_index=3, media_type="photo"),
        _texture_frame("frame_0004", source_index=4, media_type="video"),
    ]
    (raw_dir / "frames.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frames": [
                    _raw_frame("frame_0000", sha256="same_photo", source_media_id="a.jpg"),
                    _raw_frame("frame_0001", sha256="same_geometry", source_media_id="a.mp4"),
                    _raw_frame("frame_0002", sha256="same_photo", source_media_id="a.jpg"),
                    _raw_frame("frame_0003", sha256="unique_photo", source_media_id="b.jpg"),
                    _raw_frame("frame_0004", sha256="same_geometry", source_media_id="a.mp4"),
                ],
            }
        ),
        encoding="utf-8",
    )
    (observations_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_id": "demo",
                "frame_count": 1,
                "frame_selection": {
                    "selected_frame_ids": ["frame_0001"],
                    "texture_frame_ids": [frame["frame_id"] for frame in frames],
                    "texture_frame_count": len(frames),
                    "texture_media_counts": {"photo": 3, "video": 2},
                },
                "frames": [frames[1]],
            }
        ),
        encoding="utf-8",
    )
    (observations_dir / "frame_selection.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "selected_frame_ids": ["frame_0001"],
                "texture_frame_ids": [frame["frame_id"] for frame in frames],
                "texture_frame_count": len(frames),
                "texture_media_counts": {"photo": 3, "video": 2},
                "frames": [{**frame, "texture_enabled": True} for frame in frames],
            }
        ),
        encoding="utf-8",
    )
    (observations_dir / "texture_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_id": "demo",
                "geometry_selected_frame_ids": ["frame_0001"],
                "texture_frame_ids": [frame["frame_id"] for frame in frames],
                "texture_frame_count": len(frames),
                "texture_media_counts": {"photo": 3, "video": 2},
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )

    result = deduplicate_raw_texture_manifest_by_image_hash(project_dir, force=True)

    assert result.input_texture_frame_count == 5
    assert result.deduplicated_texture_frame_count == 3
    assert result.removed_duplicate_frame_ids == ("frame_0002", "frame_0004")
    texture_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert texture_manifest["texture_frame_ids"] == [
        "frame_0000",
        "frame_0001",
        "frame_0003",
    ]
    assert texture_manifest["texture_media_counts"] == {"photo": 2, "video": 1}
    assert texture_manifest["deduplication"]["removed_duplicate_frame_count"] == 2
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    geometry_group = next(
        group
        for group in report["duplicate_groups"]
        if group["kept_frame_id"] == "frame_0001"
    )
    assert geometry_group["geometry_frame_preserved"] is True
    frame_selection = json.loads(
        (observations_dir / "frame_selection.json").read_text(encoding="utf-8")
    )
    assert frame_selection["texture_frame_count"] == 3
    texture_enabled = {
        frame["frame_id"]: frame["texture_enabled"] for frame in frame_selection["frames"]
    }
    assert texture_enabled["frame_0002"] is False
    assert texture_enabled["frame_0004"] is False
    geometry_manifest = json.loads(
        (observations_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert geometry_manifest["frame_selection"]["texture_frame_ids"] == [
        "frame_0000",
        "frame_0001",
        "frame_0003",
    ]


def _frame(
    frame_id: str,
    *,
    mask: float,
    bbox_area: float,
    bbox: tuple[int, int, int, int],
    touch: float = 0.0,
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "image_width": 200,
        "image_height": 200,
        "bbox_xyxy": list(bbox),
        "observation_path": f"/tmp/{frame_id}/observation.json",
        "quality": {
            "mask_area_ratio": mask,
            "bbox_area_ratio": bbox_area,
            "touches_image_border": touch,
            "full_frame_bbox": 0.0,
        },
    }


def _texture_frame(frame_id: str, *, source_index: int, media_type: str) -> dict[str, object]:
    return {
        **_frame(frame_id, mask=0.16, bbox_area=0.30, bbox=(20, 30, 170, 130)),
        "source_index": source_index,
        "source_type": "video_frame" if media_type == "video" else "image",
        "media_type": media_type,
    }


def _raw_frame(frame_id: str, *, sha256: str, source_media_id: str) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "sha256": sha256,
        "source_media_id": source_media_id,
    }


def _write_observation_image(project_dir: Path, frame_id: str, content: bytes) -> None:
    frame_dir = project_dir / "observations" / "raw" / frame_id
    frame_dir.mkdir(parents=True)
    image_path = frame_dir / "image_srgb.npz"
    image_path.write_bytes(content)
    (frame_dir / "observation.json").write_text(
        json.dumps(
            {
                "frame_id": frame_id,
                "image_srgb_path": str(image_path),
            }
        ),
        encoding="utf-8",
    )
