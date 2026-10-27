import json

import imageio_ffmpeg
import numpy as np
from PIL import Image

from opencat3d.ingest.video import extract_video_frames


def test_extract_video_frames_evenly_samples_to_jpegs(tmp_path) -> None:
    video_path = tmp_path / "cat.mp4"
    _write_test_video(video_path, frame_count=6, size=(16, 12))

    result = extract_video_frames(
        video_path,
        tmp_path / "frames",
        min_frames=2,
        max_frames=3,
        force=True,
    )

    assert result.decoded_frame_count == 6
    assert result.extracted_frame_count == 3
    assert result.selected_indices == (0, 2, 5)
    assert result.width == 16
    assert result.height == 12
    assert all(path.exists() for path in result.frame_paths)
    assert [Image.open(path).size for path in result.frame_paths] == [(16, 12)] * 3
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["selected_indices"] == [0, 2, 5]
    assert report["extracted_frame_count"] == 3


def _write_test_video(path, *, frame_count: int, size: tuple[int, int]) -> None:
    width, height = size
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=size,
        fps=6,
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        for index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = 20 + index * 20
            frame[:, :, 1] = 60
            frame[:, :, 2] = 120
            writer.send(frame.tobytes())
    finally:
        writer.close()
