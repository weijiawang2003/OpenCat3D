import json
import os
from struct import pack

import pytest
from typer.testing import CliRunner

from opencat3d.cli import app
from opencat3d.ingest.images import ingest_image_sources, read_image_size
from opencat3d.schemas import SourceType


def test_ingest_image_sources_writes_sorted_raw_frame_manifest(tmp_path) -> None:
    input_dir = tmp_path / "inputs"
    project_dir = tmp_path / "project"
    input_dir.mkdir()
    _write_png_header(input_dir / "b.png", width=5, height=7)
    _write_png_header(input_dir / "a.png", width=3, height=4)
    (input_dir / "notes.txt").write_text("ignored", encoding="utf-8")

    result = ingest_image_sources(project_dir, [input_dir], min_frames=2)

    assert result.frame_count == 2
    assert result.manifest_path == project_dir / "raw" / "frames.json"
    assert [frame.frame_id for frame in result.frames] == ["frame_0000", "frame_0001"]
    assert [(frame.width, frame.height) for frame in result.frames] == [(3, 4), (5, 7)]
    assert [frame.source_path for frame in result.frames] == [
        str(project_dir / "raw" / "images" / "frame_0000.png"),
        str(project_dir / "raw" / "images" / "frame_0001.png"),
    ]
    report = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert report["frame_count"] == 2
    assert report["frames"][0]["sha256"] == result.frames[0].sha256
    assert report["supported_extensions"] == [".jpeg", ".jpg", ".png"]


def test_ingest_image_sources_marks_duplicate_images(tmp_path) -> None:
    input_dir = tmp_path / "inputs"
    project_dir = tmp_path / "project"
    input_dir.mkdir()
    payload = _png_header(width=2, height=2)
    (input_dir / "a.png").write_bytes(payload)
    (input_dir / "b.png").write_bytes(payload)

    result = ingest_image_sources(project_dir, [input_dir], min_frames=2)

    assert result.frames[0].duplicate_group == "duplicate_0000"
    assert result.frames[1].duplicate_group == "duplicate_0000"


def test_ingest_image_sources_hardlinks_staged_images_when_possible(tmp_path) -> None:
    input_dir = tmp_path / "inputs"
    project_dir = tmp_path / "project"
    input_dir.mkdir()
    first = input_dir / "a.png"
    second = input_dir / "b.png"
    _write_png_header(first, width=2, height=2)
    _write_png_header(second, width=3, height=3)

    ingest_image_sources(project_dir, [input_dir], min_frames=2)

    staged = project_dir / "raw" / "images" / "frame_0000.png"
    assert staged.exists()
    assert os.stat(first).st_ino == os.stat(staged).st_ino


def test_ingest_image_sources_preserves_upload_media_metadata(tmp_path) -> None:
    input_dir = tmp_path / "uploads"
    project_dir = tmp_path / "project"
    input_dir.mkdir()
    _write_png_header(input_dir / "frame_0000.png", width=4, height=4)
    _write_png_header(input_dir / "frame_0001.png", width=5, height=5)
    (input_dir / "media_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frames": {
                    "frame_0000.png": {
                        "source_type": "image",
                        "source_media_id": "cat_photo.png",
                    },
                    "frame_0001.png": {
                        "source_type": "video_frame",
                        "source_media_id": "cat_video.mp4",
                        "source_frame_index": 42,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = ingest_image_sources(project_dir, [input_dir], min_frames=2)

    assert result.frames[0].source_type == SourceType.IMAGE
    assert result.frames[0].source_media_id == "cat_photo.png"
    assert result.frames[1].source_type == SourceType.VIDEO_FRAME
    assert result.frames[1].source_media_id == "cat_video.mp4"
    assert result.frames[1].source_frame_index == 42


def test_ingest_image_sources_requires_multiple_images(tmp_path) -> None:
    image_path = tmp_path / "cat.png"
    _write_png_header(image_path, width=2, height=2)

    with pytest.raises(ValueError, match="at least 2 image frames"):
        ingest_image_sources(tmp_path / "project", [image_path], min_frames=2)


def test_read_image_size_rejects_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "cat.gif"
    path.write_bytes(b"GIF89a")

    with pytest.raises(ValueError, match="unsupported image extension"):
        read_image_size(path)


def test_ingest_cli_uses_project_manifest_inputs(tmp_path) -> None:
    runner = CliRunner()
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    project_dir = tmp_path / "project"
    _write_png_header(image_a, width=4, height=4)
    _write_png_header(image_b, width=6, height=5)

    init_result = runner.invoke(
        app,
        [
            "project",
            "init",
            str(project_dir),
            "--mode",
            "album",
            "--input",
            str(image_b),
            "--input",
            str(image_a),
        ],
    )
    ingest_result = runner.invoke(app, ["ingest", str(project_dir)])

    assert init_result.exit_code == 0, init_result.output
    assert ingest_result.exit_code == 0, ingest_result.output
    assert "frames=2" in ingest_result.output
    report_path = project_dir / "raw" / "frames.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["frame_count"] == 2
    assert [frame["width"] for frame in report["frames"]] == [4, 6]


def _write_png_header(path, *, width: int, height: int) -> None:
    path.write_bytes(_png_header(width=width, height=height))


def _png_header(*, width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + pack(">I", 13) + b"IHDR" + pack(">II", width, height)
