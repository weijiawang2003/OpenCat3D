import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from catnet.reid.adapters import (
    folder_layout_to_samples,
    petface_csv_to_samples,
    wildlife_csv_to_samples,
    wildlife_dataframe_to_samples,
)
from catnet.reid.data import load_manifest


def test_folder_layout_to_samples_writes_sorted_manifest(tmp_path) -> None:
    root = tmp_path / "folder_dataset"
    _write_image(root / "cat_b" / "second.JPG", color=(40, 120, 200))
    _write_image(root / "cat_a" / "first.png", color=(210, 90, 60))
    (root / "cat_a" / "notes.txt").write_text("not an image", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    samples = folder_layout_to_samples(root, manifest_path=manifest_path)

    assert [(sample.cat_id, Path(sample.image_path).name) for sample in samples] == [
        ("cat_a", "first.png"),
        ("cat_b", "second.JPG"),
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert [sample["cat_id"] for sample in manifest["samples"]] == ["cat_a", "cat_b"]


def test_petface_csv_to_samples_uses_configured_columns_and_image_root(tmp_path) -> None:
    image_root = tmp_path / "petface"
    _write_image(image_root / "images" / "tabby.png", color=(180, 75, 60))
    csv_path = tmp_path / "petface_records.csv"
    _write_csv(
        csv_path,
        fieldnames=["pet_id", "file_name", "split"],
        rows=[
            {
                "pet_id": "tabby_01",
                "file_name": "images/tabby.png",
                "split": "train",
            }
        ],
    )
    manifest_path = tmp_path / "petface_manifest.jsonl"

    samples = petface_csv_to_samples(
        csv_path,
        image_root=image_root,
        identity_column="pet_id",
        image_path_column="file_name",
        metadata_columns=["split"],
        manifest_path=manifest_path,
    )

    assert len(samples) == 1
    assert samples[0].cat_id == "tabby_01"
    assert Path(samples[0].image_path) == image_root / "images" / "tabby.png"
    assert samples[0].metadata == {"split": "train"}
    assert load_manifest(manifest_path)[0].metadata == {"split": "train"}


def test_wildlife_csv_to_samples_maps_configured_date_site_camera_columns(tmp_path) -> None:
    dataset_root = tmp_path / "wildlife"
    _write_image(dataset_root / "images" / "cat.png", color=(90, 160, 90))
    csv_path = dataset_root / "wildlife_records.csv"
    _write_csv(
        csv_path,
        fieldnames=["individual", "filepath", "timestamp", "location", "device"],
        rows=[
            {
                "individual": "cat-7",
                "filepath": "images/cat.png",
                "timestamp": "2026-05-03T18:30:00+08:00",
                "location": "north_yard",
                "device": "camera_2",
            }
        ],
    )

    samples = wildlife_csv_to_samples(
        csv_path,
        image_root=dataset_root,
        identity_column="individual",
        image_column="filepath",
        date_column="timestamp",
        site_column="location",
        camera_column="device",
    )

    assert samples[0].cat_id == "cat-7"
    assert samples[0].captured_at == datetime(2026, 5, 3, 10, 30, tzinfo=UTC)
    assert samples[0].site_id == "north_yard"
    assert samples[0].camera_id == "camera_2"


def test_wildlife_dataframe_to_samples_accepts_dataframe_like_records(tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    _write_image(image_path, color=(50, 80, 180))
    manifest_path = tmp_path / "wildlife_manifest.json"
    dataframe = _RecordsFrame(
        [
            {
                "id": "cat_df",
                "image_file": str(image_path),
                "when": datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
                "site": "",
                "camera": "trap_9",
                "season": "summer",
            }
        ]
    )

    samples = wildlife_dataframe_to_samples(
        dataframe,
        identity_column="id",
        image_column="image_file",
        date_column="when",
        site_column="site",
        camera_column="camera",
        metadata_columns=["season"],
        manifest_path=manifest_path,
    )

    assert samples[0].cat_id == "cat_df"
    assert samples[0].captured_at == datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    assert samples[0].site_id is None
    assert samples[0].camera_id == "trap_9"
    assert samples[0].metadata == {"season": "summer"}
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["samples"][0][
        "camera_id"
    ] == "trap_9"


class _RecordsFrame:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self._records = records

    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        if orient != "records":
            raise ValueError("test frame only supports records orientation")
        return list(self._records)


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 12, 3), color, dtype=np.uint8)
    Image.fromarray(image).save(path)


def _write_csv(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
