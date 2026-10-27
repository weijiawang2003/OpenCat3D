"""Deterministic raw image ingestion."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from struct import unpack
from typing import Any

from opencat3d.io.cache import sha256_file
from opencat3d.schemas import CaptureManifest, FrameRecord, SourceType, read_json_model

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class ImageIngestResult:
    project_dir: Path
    manifest_path: Path
    frame_count: int
    frames: tuple[FrameRecord, ...]


def ingest_image_sources(
    project_dir: Path,
    input_paths: list[Path] | tuple[Path, ...] | None = None,
    *,
    min_frames: int = 2,
    force: bool = False,
) -> ImageIngestResult:
    """Stage raw image inputs into a project-local deterministic frame manifest."""

    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    project_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_dir / "raw" / "frames.json"
    output_dir = project_dir / "raw" / "images"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} already exists; pass force=True to overwrite")

    sources = _resolve_input_paths(project_dir, input_paths)
    image_paths = _expand_image_paths(sources)
    if len(image_paths) < min_frames:
        raise ValueError(f"at least {min_frames} image frames are required")
    media_metadata = _load_media_metadata(sources)

    output_dir.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, Any]] = []
    for index, source_path in enumerate(image_paths):
        suffix = source_path.suffix.lower()
        target_path = output_dir / f"frame_{index:04d}{suffix}"
        if target_path.exists() and not force:
            raise FileExistsError(f"{target_path} already exists; pass force=True to overwrite")
        if source_path.resolve() != target_path.resolve():
            _stage_image_file(source_path, target_path)
        width, height = read_image_size(target_path)
        media = media_metadata.get(source_path.resolve(), {})
        frame_record = {
            "frame_id": f"frame_{index:04d}",
            "source_path": str(target_path),
            "source_type": _source_type_from_media(media),
            "width": width,
            "height": height,
            "sha256": sha256_file(target_path),
        }
        source_media_id = _optional_str(media.get("source_media_id"))
        if source_media_id is not None:
            frame_record["source_media_id"] = source_media_id
        source_frame_index = media.get("source_frame_index")
        if source_frame_index is not None:
            frame_record["source_frame_index"] = int(source_frame_index)
        staged.append(frame_record)

    duplicate_groups = _duplicate_groups([str(item["sha256"]) for item in staged])
    frames = tuple(
        FrameRecord(
            **item,
            duplicate_group=duplicate_groups.get(str(item["sha256"])),
        )
        for item in staged
    )
    capture_manifest_path = project_dir / "manifest.lock.json"
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": _project_id(project_dir, capture_manifest_path),
        "source_manifest": str(capture_manifest_path) if capture_manifest_path.exists() else None,
        "frame_count": len(frames),
        "min_frames": min_frames,
        "supported_extensions": sorted(SUPPORTED_IMAGE_EXTENSIONS),
        "frames": [frame.model_dump(mode="json") for frame in frames],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ImageIngestResult(
        project_dir=project_dir,
        manifest_path=manifest_path,
        frame_count=len(frames),
        frames=frames,
    )


def read_image_size(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _read_png_size(path)
    if suffix in {".jpg", ".jpeg"}:
        return _read_jpeg_size(path)
    raise ValueError(f"unsupported image extension: {path.suffix}")


def _resolve_input_paths(
    project_dir: Path,
    input_paths: list[Path] | tuple[Path, ...] | None,
) -> tuple[Path, ...]:
    if input_paths:
        return tuple(Path(path) for path in input_paths)
    manifest_path = project_dir / "manifest.lock.json"
    if not manifest_path.exists():
        raise FileNotFoundError("input paths or manifest.lock.json are required")
    manifest = read_json_model(CaptureManifest, manifest_path)
    if not manifest.source_paths:
        raise ValueError("capture manifest contains no source_paths")
    return tuple(Path(path) for path in manifest.source_paths)


def _expand_image_paths(input_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for input_path in input_paths:
        path = input_path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        if path.is_dir():
            expanded.extend(
                child
                for child in sorted(path.iterdir(), key=lambda item: item.name)
                if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {path.suffix}")
        expanded.append(path)

    unique = {path.resolve(): path for path in expanded}
    return tuple(unique[key] for key in sorted(unique))


def _load_media_metadata(input_paths: tuple[Path, ...]) -> dict[Path, dict[str, Any]]:
    metadata: dict[Path, dict[str, Any]] = {}
    manifest_dirs: set[Path] = set()
    for input_path in input_paths:
        path = input_path.expanduser()
        manifest_dirs.add(path if path.is_dir() else path.parent)

    for manifest_dir in sorted(manifest_dirs):
        manifest_path = manifest_dir / "media_manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = payload.get("frames", {})
        if not isinstance(frames, dict):
            continue
        for filename, value in frames.items():
            if not isinstance(filename, str) or not isinstance(value, dict):
                continue
            metadata[(manifest_dir / filename).resolve()] = dict(value)
    return metadata


def _stage_image_file(source_path: Path, target_path: Path) -> None:
    """Stage immutable source pixels without duplicating bytes when possible."""

    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copyfile(source_path, target_path)


def _source_type_from_media(media: dict[str, Any]) -> SourceType:
    source_type = media.get("source_type")
    media_type = media.get("media_type")
    if source_type == SourceType.VIDEO_FRAME.value or media_type == "video":
        return SourceType.VIDEO_FRAME
    return SourceType.IMAGE


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _duplicate_groups(hashes: list[str]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for digest in hashes:
        counts[digest] = counts.get(digest, 0) + 1
    duplicate_hashes = sorted(digest for digest, count in counts.items() if count > 1)
    return {digest: f"duplicate_{index:04d}" for index, digest in enumerate(duplicate_hashes)}


def _project_id(project_dir: Path, manifest_path: Path) -> str:
    if not manifest_path.exists():
        return project_dir.name
    manifest = read_json_model(CaptureManifest, manifest_path)
    return manifest.project_id


def _read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"{path} is not a valid PNG image")
    width, height = unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"{path} has invalid image dimensions")
    return int(width), int(height)


def _read_jpeg_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError(f"{path} is not a valid JPEG image")
        while True:
            marker_prefix = handle.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\x00", b"\xd8", b"\xd9"}:
                continue
            length_raw = handle.read(2)
            if len(length_raw) != 2:
                break
            segment_length = unpack(">H", length_raw)[0]
            if segment_length < 2:
                raise ValueError(f"{path} contains an invalid JPEG segment")
            if marker and 0xC0 <= marker[0] <= 0xCF and marker[0] not in {0xC4, 0xC8, 0xCC}:
                segment = handle.read(5)
                if len(segment) != 5:
                    break
                height, width = unpack(">HH", segment[1:5])
                if width <= 0 or height <= 0:
                    raise ValueError(f"{path} has invalid image dimensions")
                return int(width), int(height)
            handle.seek(segment_length - 2, 1)
    raise ValueError(f"{path} is missing JPEG dimensions")
