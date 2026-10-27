import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from catnet.benchmark import run_catnet_benchmark
from catnet.reid.data import ReIDSample, write_manifest


def test_benchmark_runs_synthetic_manifest_and_compares_backbones(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, _samples(tmp_path))

    summary = run_catnet_benchmark(
        manifest_path=manifest_path,
        backbones=["deterministic", "hybrid:deterministic+deterministic"],
        output_dir=tmp_path / "benchmark",
        embedding_dim=8,
        unknown_identity_fraction=0.25,
    )

    summary_path = tmp_path / "benchmark" / "benchmark_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary.status == "complete"
    assert summary.completed_count == 2
    assert len(payload["comparison"]) == 2
    assert all(report["model_path"] for report in payload["reports"])
    assert all(
        report["report_path"].endswith("benchmark_report.json")
        for report in payload["reports"]
    )
    assert all("training_report_path" in report["metadata"] for report in payload["reports"])


def test_benchmark_missing_manifest_writes_blocked_report(tmp_path) -> None:
    summary = run_catnet_benchmark(
        manifest_path=tmp_path / "missing.json",
        backbones=["deterministic", "megadescriptor-t-224"],
        output_dir=tmp_path / "benchmark",
    )

    payload = json.loads((tmp_path / "benchmark" / "benchmark_summary.json").read_text())

    assert summary.status == "blocked"
    assert summary.reason == "missing_manifest"
    assert payload["reports"][0]["status"] == "blocked"


def test_benchmark_corrupt_images_block_backbone_without_aborting(tmp_path) -> None:
    bad_path = tmp_path / "bad.png"
    bad_path.write_bytes(b"not an image")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        [
            ReIDSample(image_path=str(bad_path), cat_id="cat_a"),
            ReIDSample(image_path=str(bad_path), cat_id="cat_a"),
            ReIDSample(image_path=str(bad_path), cat_id="cat_b"),
            ReIDSample(image_path=str(bad_path), cat_id="cat_b"),
        ],
    )

    summary = run_catnet_benchmark(
        manifest_path=manifest_path,
        backbones=["deterministic"],
        output_dir=tmp_path / "benchmark",
    )
    payload = json.loads((tmp_path / "benchmark" / "benchmark_summary.json").read_text())

    assert summary.status == "blocked"
    assert payload["reports"][0]["status"] == "blocked"
    assert payload["reports"][0]["reason"]


def _samples(tmp_path: Path) -> list[ReIDSample]:
    colors = {
        "cat_a": (170, 60, 60),
        "cat_b": (45, 130, 190),
        "cat_c": (80, 160, 75),
        "cat_unknown": (210, 175, 45),
    }
    start = datetime(2026, 1, 1, tzinfo=UTC)
    samples: list[ReIDSample] = []
    for cat_index, (cat_id, color) in enumerate(colors.items()):
        for month in range(3):
            path = tmp_path / f"{cat_id}_{month}.png"
            _write_image(path, color=color, stripe_offset=cat_index + month)
            samples.append(
                ReIDSample(
                    image_path=str(path),
                    cat_id=cat_id,
                    captured_at=start + timedelta(days=cat_index + month * 90),
                    site_id="yard",
                    camera_id=f"cam_{month % 2}",
                    age_bin=["kitten", "juvenile", "adult"][month],
                )
            )
    return samples


def _write_image(path: Path, *, color: tuple[int, int, int], stripe_offset: int) -> None:
    image = np.full((40, 40, 3), 245, dtype=np.uint8)
    image[8:32, 8:32, :] = np.asarray(color, dtype=np.uint8)
    for x in range(8 + stripe_offset, 32, 6):
        image[8:32, x : min(x + 2, 32), :] = np.clip(
            np.asarray(color, dtype=np.int16) - 35,
            0,
            255,
        )
    Image.fromarray(image).save(path)
