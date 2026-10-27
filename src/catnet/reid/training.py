"""Training and evaluation loop for CatNet Re-ID."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catnet.eval.growth_aware import GrowthAwareEvaluator, GrowthEvalCase
from catnet.reid.backbones import build_backbone
from catnet.reid.data import ReIDSample, ReIDSplit, time_aware_split
from catnet.reid.model import CatReIDModel, CatReIDModelConfig


@dataclass(frozen=True)
class TrainReport:
    model_path: Path
    report_path: Path
    metrics: dict[str, Any]
    split: ReIDSplit


def train_reid_model(
    samples: list[ReIDSample],
    *,
    artifact_dir: Path,
    config: CatReIDModelConfig | None = None,
    validation_fraction: float = 0.30,
    unknown_identity_fraction: float = 0.20,
) -> TrainReport:
    split = time_aware_split(
        samples,
        validation_fraction=validation_fraction,
        unknown_identity_fraction=unknown_identity_fraction,
    )
    config = config or CatReIDModelConfig()
    backbone = build_backbone(config.backbone)
    model = CatReIDModel.fit(split.train, config=config, backbone=backbone)
    model.calibrate_threshold(
        split.validation_known,
        split.validation_unknown,
        backbone=backbone,
    )
    metrics = evaluate_reid_model(
        model,
        split.validation_known,
        split.validation_unknown,
        backbone_name=config.backbone,
    )
    model_path = model.save(artifact_dir)
    report_path = artifact_dir / "catnet_reid_report.json"
    report = {
        "schema_version": "1.0",
        "model_path": str(model_path),
        "train_count": len(split.train),
        "validation_known_count": len(split.validation_known),
        "validation_unknown_count": len(split.validation_unknown),
        "threshold": model.config.threshold,
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TrainReport(
        model_path=model_path,
        report_path=report_path,
        metrics=metrics,
        split=split,
    )


def evaluate_reid_model(
    model: CatReIDModel,
    known_samples: list[ReIDSample],
    unknown_samples: list[ReIDSample],
    *,
    backbone_name: str = "deterministic",
) -> dict[str, Any]:
    backbone = build_backbone(backbone_name)
    cases: list[GrowthEvalCase] = []
    for sample in known_samples:
        prediction = model.predict_sample(sample, backbone=backbone, top_k=5)
        cases.append(
            GrowthEvalCase(
                query_obs_id=sample.resolved_obs_id(),
                true_cat_id=sample.cat_id,
                candidates=prediction.candidates,
                is_known=True,
                time_gap_days=_gap_to_identity_start(sample, known_samples),
                age_bin=sample.age_bin,
                camera_id=sample.camera_id,
                site_id=sample.site_id,
                view=sample.face_view or sample.body_view,
            )
        )
    for sample in unknown_samples:
        prediction = model.predict_sample(sample, backbone=backbone, top_k=5)
        cases.append(
            GrowthEvalCase(
                query_obs_id=sample.resolved_obs_id(),
                true_cat_id=None,
                candidates=prediction.candidates,
                is_known=False,
                time_gap_days=None,
                age_bin=sample.age_bin,
                camera_id=sample.camera_id,
                site_id=sample.site_id,
                view=sample.face_view or sample.body_view,
            )
        )
    return GrowthAwareEvaluator(threshold=model.config.threshold).evaluate(cases)


def _gap_to_identity_start(sample: ReIDSample, known_samples: list[ReIDSample]) -> float | None:
    if sample.captured_at is None:
        return None
    same_identity_times = [
        other.captured_at
        for other in known_samples
        if other.cat_id == sample.cat_id and other.captured_at is not None
    ]
    if not same_identity_times:
        return None
    return abs((sample.captured_at - min(same_identity_times)).total_seconds()) / 86400.0
