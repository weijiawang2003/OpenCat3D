from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from opencat3d.schemas import (
    CameraPrior,
    CaptureManifest,
    CaptureMode,
    FitReport,
    FrameObservationMeta,
    FrameRecord,
    SourceType,
    TensorRef,
    TensorStorageFormat,
    read_json_model,
    write_json_model,
)

HASH = "0" * 64


def tensor_ref(path: str, shape: list[int], dtype: str = "float32") -> TensorRef:
    return TensorRef(
        path=path,
        shape=shape,
        dtype=dtype,
        sha256=HASH,
        storage=TensorStorageFormat.NPZ,
    )


def test_capture_manifest_round_trip(tmp_path) -> None:
    manifest = CaptureManifest(
        project_id="demo_cat",
        mode=CaptureMode.ALBUM,
        source_paths=["a.jpg"],
    )
    path = tmp_path / "manifest.lock.json"

    write_json_model(manifest, path)
    loaded = read_json_model(CaptureManifest, path)

    assert loaded == manifest


def test_frame_record_validates_hash_and_orientation() -> None:
    frame = FrameRecord(
        frame_id="f0001",
        source_path="a.jpg",
        source_type=SourceType.IMAGE,
        exif_orientation=1,
        width=640,
        height=480,
        sha256=HASH,
    )

    assert frame.width == 640
    with pytest.raises(ValidationError):
        FrameRecord(
            frame_id="f0002",
            source_path="a.jpg",
            source_type=SourceType.IMAGE,
            exif_orientation=9,
            width=640,
            height=480,
            sha256=HASH,
        )


def test_observation_meta_rejects_bad_bbox() -> None:
    with pytest.raises(ValidationError):
        FrameObservationMeta(
            frame_id="f0001",
            image_srgb_path="image.png",
            hard_mask=tensor_ref("hard.npz", [1, 16, 16]),
            soft_mask=tensor_ref("soft.npz", [1, 16, 16]),
            bbox_xyxy=[10.0, 10.0, 2.0, 20.0],
            keypoints_xyc=tensor_ref("kps.npz", [26, 3]),
            camera_prior=CameraPrior(),
        )


def test_tensor_ref_rejects_inline_or_unhashed_payload() -> None:
    with pytest.raises(ValidationError):
        TensorRef(
            path="",
            shape=[26, 3],
            dtype="float32",
            sha256="not-a-hash",
            storage=TensorStorageFormat.NPZ,
        )


def test_fit_report_accepts_loss_history() -> None:
    report = FitReport(
        project_id="demo_cat",
        success=True,
        num_frames=6,
        started_at=datetime.now(UTC),
        loss_history={"keypoint": [2.0, 1.0]},
    )

    assert report.loss_history["keypoint"][-1] == 1.0

