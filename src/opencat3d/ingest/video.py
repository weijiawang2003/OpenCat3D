"""Deterministic video frame extraction for raw reconstruction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".avi"})


@dataclass(frozen=True)
class VideoFrameExtractionResult:
    video_path: Path
    output_dir: Path
    report_path: Path
    frame_paths: tuple[Path, ...]
    decoded_frame_count: int
    extracted_frame_count: int
    selected_indices: tuple[int, ...]
    fps: float | None
    duration_s: float | None
    width: int
    height: int


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    min_frames: int = 2,
    max_frames: int = 24,
    force: bool = False,
) -> VideoFrameExtractionResult:
    """Decode a video and write evenly sampled RGB JPEG frames."""

    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    if max_frames < min_frames:
        raise ValueError("max_frames must be >= min_frames")
    video_path = Path(video_path)
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video extension: {video_path.suffix}")
    if not video_path.is_file():
        raise FileNotFoundError(f"{video_path} does not exist")

    report_path = output_dir / "video_frames.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; pass force=True to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, decoded_frames = _decode_video_rgb(video_path)
    if len(decoded_frames) < min_frames:
        raise ValueError(f"video must decode to at least {min_frames} frames")
    size = meta.get("size") or meta.get("source_size")
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("video decoder did not report frame size")
    width, height = int(size[0]), int(size[1])
    indices = _even_sample_indices(len(decoded_frames), max_frames=max_frames)

    frame_paths: list[Path] = []
    for output_index, frame_index in enumerate(indices):
        frame = decoded_frames[frame_index]
        frame_path = output_dir / f"video_frame_{output_index:04d}.jpg"
        if frame_path.exists() and not force:
            raise FileExistsError(f"{frame_path} already exists; pass force=True to overwrite")
        Image.fromarray(frame).save(frame_path, format="JPEG", quality=92)
        frame_paths.append(frame_path)

    fps = _optional_float(meta.get("fps"))
    duration_s = _optional_float(meta.get("duration"))
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source": "opencat3d.video_frame_extraction",
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "decoded_frame_count": len(decoded_frames),
        "extracted_frame_count": len(frame_paths),
        "selected_indices": list(indices),
        "fps": fps,
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "frames": [str(path) for path in frame_paths],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return VideoFrameExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        report_path=report_path,
        frame_paths=tuple(frame_paths),
        decoded_frame_count=len(decoded_frames),
        extracted_frame_count=len(frame_paths),
        selected_indices=tuple(indices),
        fps=fps,
        duration_s=duration_s,
        width=width,
        height=height,
    )


def _decode_video_rgb(video_path: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
    try:
        meta = next(reader)
        if not isinstance(meta, dict):
            raise ValueError("video decoder did not yield metadata")
        size = meta.get("size") or meta.get("source_size")
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError("video decoder did not report frame size")
        width, height = int(size[0]), int(size[1])
        frames: list[np.ndarray] = []
        for frame_bytes in reader:
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frames.append(frame.reshape((height, width, 3)).copy())
    finally:
        close = getattr(reader, "close", None)
        if close is not None:
            close()
    return meta, frames


def _even_sample_indices(frame_count: int, *, max_frames: int) -> tuple[int, ...]:
    sample_count = min(frame_count, max_frames)
    indices = np.linspace(0, frame_count - 1, num=sample_count)
    rounded = np.rint(indices).astype(np.int64)
    return tuple(int(index) for index in np.unique(rounded))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
