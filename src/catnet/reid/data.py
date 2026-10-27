"""Dataset manifests and time-aware splits for cat Re-ID training."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from catnet.schema import Observation, StrictModel, stable_id


class ReIDSample(StrictModel):
    """One labeled cat image observation used for Re-ID training or evaluation."""

    image_path: str
    cat_id: str
    obs_id: str | None = None
    captured_at: datetime | None = None
    site_id: str | None = None
    camera_id: str | None = None
    age_bin: str | None = None
    size_bin: str | None = None
    face_view: str | None = None
    body_view: str | None = None
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("captured_at", mode="before")
    @classmethod
    def captured_at_to_utc(cls, value: datetime | str | None) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            raw = value[:-1] + "+00:00" if value.endswith("Z") else value
            value = datetime.fromisoformat(raw)
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("image_path", "cat_id")
    @classmethod
    def required_strings_are_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("image_path and cat_id must not be empty")
        return value

    def resolved_obs_id(self) -> str:
        if self.obs_id:
            return self.obs_id
        timestamp = self.captured_at.isoformat() if self.captured_at else ""
        return stable_id("obs", self.image_path, self.cat_id, timestamp)

    def to_observation(self) -> Observation:
        return Observation(
            obs_id=self.resolved_obs_id(),
            observation_version=stable_id("obv", self.resolved_obs_id(), self.image_path),
            media_id=stable_id("media", self.image_path),
            cat_id=self.cat_id,
            captured_at=self.captured_at,
            site_id=self.site_id,
            camera_id=self.camera_id,
            age_bin=self.age_bin,
            size_bin=self.size_bin,
            face_view=self.face_view,
            body_view=self.body_view,
            quality_score=self.quality_score,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class ReIDSplit:
    train: list[ReIDSample]
    validation_known: list[ReIDSample]
    validation_unknown: list[ReIDSample]

    @property
    def validation(self) -> list[ReIDSample]:
        return [*self.validation_known, *self.validation_unknown]


def load_manifest(path: Path) -> list[ReIDSample]:
    """Load a Re-ID manifest from JSONL, JSON, or CSV."""

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            ReIDSample.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = (
            payload["samples"]
            if isinstance(payload, dict) and "samples" in payload
            else payload
        )
        if not isinstance(records, list):
            raise ValueError("JSON Re-ID manifest must be a list or contain a samples list")
        return [ReIDSample.model_validate(record) for record in records]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [ReIDSample.model_validate(row) for row in csv.DictReader(handle)]
    raise ValueError("manifest must be .jsonl, .json, or .csv")


def write_manifest(path: Path, samples: list[ReIDSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        payload = {
            "schema_version": "1.0",
            "samples": [sample.model_dump(mode="json") for sample in samples],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    lines = [json.dumps(sample.model_dump(mode="json"), sort_keys=True) for sample in samples]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def time_aware_split(
    samples: list[ReIDSample],
    *,
    validation_fraction: float = 0.30,
    unknown_identity_fraction: float = 0.20,
    min_train_per_identity: int = 1,
) -> ReIDSplit:
    """Split by identity and time, keeping later observations for validation.

    A small identity holdout simulates open-set unknown cats. Remaining identities
    keep their latest observations as known validation examples, which prevents
    random same-session leakage from inflating performance.
    """

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0.0 <= unknown_identity_fraction < 1.0:
        raise ValueError("unknown_identity_fraction must be in [0, 1)")

    by_cat: dict[str, list[ReIDSample]] = defaultdict(list)
    for sample in samples:
        by_cat[sample.cat_id].append(sample)
    if len(by_cat) < 2:
        raise ValueError("at least two identities are required for Re-ID training")

    identities = sorted(by_cat)
    unknown_count = int(round(len(identities) * unknown_identity_fraction))
    if unknown_identity_fraction > 0 and unknown_count == 0 and len(identities) >= 3:
        unknown_count = 1
    unknown_count = min(unknown_count, max(0, len(identities) - 2))
    unknown_ids = set(identities[-unknown_count:]) if unknown_count else set()

    train: list[ReIDSample] = []
    validation_known: list[ReIDSample] = []
    validation_unknown: list[ReIDSample] = []
    for cat_id in identities:
        ordered = sorted(by_cat[cat_id], key=_sample_sort_key)
        if cat_id in unknown_ids:
            validation_unknown.extend(ordered)
            continue

        if len(ordered) <= min_train_per_identity:
            train.extend(ordered)
            continue
        validation_count = max(1, int(round(len(ordered) * validation_fraction)))
        validation_count = min(validation_count, len(ordered) - min_train_per_identity)
        train.extend(ordered[:-validation_count])
        validation_known.extend(ordered[-validation_count:])

    if not train:
        raise ValueError("time-aware split produced no training samples")
    return ReIDSplit(
        train=train,
        validation_known=validation_known,
        validation_unknown=validation_unknown,
    )


def _sample_sort_key(sample: ReIDSample) -> tuple[str, str]:
    timestamp = sample.captured_at.isoformat() if sample.captured_at else ""
    return timestamp, sample.image_path
