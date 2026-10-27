from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from catnet.reid.data import ReIDSample, load_manifest, time_aware_split, write_manifest
from catnet.reid.model import CatReIDModel, CatReIDModelConfig
from catnet.reid.training import train_reid_model
from opencat3d.cli import app
from opencat3d.web import UploadedFile, run_identity_image_job


def test_trainable_reid_model_recognizes_same_cat_over_time_and_rejects_unknown(tmp_path) -> None:
    samples = _write_longitudinal_dataset(tmp_path)
    report = train_reid_model(
        samples,
        artifact_dir=tmp_path / "artifacts",
        config=CatReIDModelConfig(embedding_dim=16),
        validation_fraction=0.34,
        unknown_identity_fraction=0.25,
    )
    model = CatReIDModel.load(report.model_path)

    assert report.metrics["overall"]["rank1"] >= 0.99
    assert report.metrics["overall"]["BAKS"] >= 0.99
    assert report.metrics["overall"]["BAUS"] >= 0.99
    assert report.metrics["overall"]["AUROC"] >= 0.99
    assert model.config.threshold > 0.0

    known_query = report.split.validation_known[0]
    known_prediction = model.predict_sample(known_query)
    unknown_prediction = model.predict_sample(report.split.validation_unknown[0])

    assert known_prediction.is_known
    assert known_prediction.cat_id == known_query.cat_id
    assert not unknown_prediction.is_known
    assert (tmp_path / "artifacts" / "catnet_reid_report.json").exists()


def test_manifest_loader_and_time_split_are_time_aware(tmp_path) -> None:
    samples = _write_longitudinal_dataset(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, samples)

    loaded = load_manifest(manifest)
    split = time_aware_split(
        loaded,
        validation_fraction=0.34,
        unknown_identity_fraction=0.25,
    )

    assert len(loaded) == len(samples)
    assert {sample.cat_id for sample in split.validation_unknown} == {"cat_z_unknown"}
    for cat_id in {"cat_a", "cat_b", "cat_c"}:
        train_times = [
            sample.captured_at for sample in split.train if sample.cat_id == cat_id
        ]
        valid_times = [
            sample.captured_at for sample in split.validation_known if sample.cat_id == cat_id
        ]
        assert max(train_times) < min(valid_times)


def test_catnet_train_cli_writes_model_and_report(tmp_path) -> None:
    samples = _write_longitudinal_dataset(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, samples)
    artifact_dir = tmp_path / "artifacts"

    result = CliRunner().invoke(
        app,
        [
            "catnet",
            "train",
            str(manifest),
            "--artifact-dir",
            str(artifact_dir),
            "--embedding-dim",
            "16",
            "--unknown-identity-fraction",
            "0.25",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "catnet train complete" in result.output
    assert (artifact_dir / "catnet_reid_model.json").exists()
    assert (artifact_dir / "catnet_reid_report.json").exists()


def test_catweb_identity_job_can_use_trained_reid_model(tmp_path) -> None:
    samples = _write_longitudinal_dataset(tmp_path)
    report = train_reid_model(
        samples,
        artifact_dir=tmp_path / "artifacts",
        config=CatReIDModelConfig(embedding_dim=16),
        validation_fraction=0.34,
        unknown_identity_fraction=0.25,
    )
    known = report.split.validation_known[0]
    unknown = report.split.validation_unknown[0]

    known_path = Path(known.image_path)
    known_result = run_identity_image_job(
        [UploadedFile(known_path.name, known_path.read_bytes())],
        work_dir=tmp_path / "web",
        captured_at=known.captured_at,
        site_id=known.site_id,
        camera_id=known.camera_id,
        model_path=report.model_path,
    )

    unknown_path = Path(unknown.image_path)
    unknown_result = run_identity_image_job(
        [UploadedFile(unknown_path.name, unknown_path.read_bytes())],
        work_dir=tmp_path / "web",
        captured_at=unknown.captured_at,
        site_id=unknown.site_id,
        camera_id=unknown.camera_id,
        model_path=report.model_path,
    )

    assert known_result.results[0].decision == "match_existing"
    assert known_result.results[0].cat_id == known.cat_id
    assert unknown_result.results[0].decision == "create_new_identity"
    assert unknown_result.results[0].cat_id != unknown.cat_id


def _write_longitudinal_dataset(tmp_path: Path) -> list[ReIDSample]:
    identities = {
        "cat_a": (170, 60, 60),
        "cat_b": (45, 130, 190),
        "cat_c": (80, 160, 75),
        "cat_z_unknown": (210, 175, 45),
    }
    start = datetime(2026, 1, 1, 10, tzinfo=UTC)
    samples: list[ReIDSample] = []
    for cat_index, (cat_id, color) in enumerate(identities.items()):
        for month in range(3):
            path = tmp_path / f"{cat_id}_{month}.png"
            _write_cat_image(
                path,
                color=color,
                pattern_offset=cat_index * 3 + month,
                growth_scale=1.0 + month * 0.08,
            )
            samples.append(
                ReIDSample(
                    image_path=str(path),
                    cat_id=cat_id,
                    captured_at=start + timedelta(days=month * 90 + cat_index),
                    site_id="yard",
                    camera_id=f"camera_{month % 2}",
                    age_bin=["kitten", "juvenile", "adult"][month],
                    face_view="front" if month != 1 else "left",
                )
            )
    return samples


def _write_cat_image(
    path: Path,
    *,
    color: tuple[int, int, int],
    pattern_offset: int,
    growth_scale: float,
) -> None:
    image = np.full((48, 48, 3), 245, dtype=np.uint8)
    body_size = int(24 * growth_scale)
    x0 = max(4, 24 - body_size // 2)
    y0 = max(6, 24 - body_size // 2)
    x1 = min(44, x0 + body_size)
    y1 = min(44, y0 + body_size)
    image[y0:y1, x0:x1, :] = np.asarray(color, dtype=np.uint8)
    for stripe in range(x0 + pattern_offset % 5, x1, 6):
        image[y0:y1, stripe : min(stripe + 2, x1), :] = np.clip(
            np.asarray(color, dtype=np.int16) - 35,
            0,
            255,
        )
    image[y0 : y0 + 5, x0 : x0 + 5, :] = np.clip(
        np.asarray(color, dtype=np.int16) + 30,
        0,
        255,
    )
    Image.fromarray(image).save(path)
