"""Trainable NumPy metric model for cat identity recognition."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from catnet.reid.backbones import ReIDBackbone, build_backbone
from catnet.reid.data import ReIDSample
from catnet.schema import StrictModel, stable_id


class CatReIDModelConfig(StrictModel):
    model_version: str = "catnet-numpy-reid-v1"
    backbone: str = "deterministic"
    embedding_dim: int = Field(default=32, gt=0)
    regularization: float = Field(default=1e-3, gt=0)
    threshold: float = Field(default=0.80, ge=-1.0, le=1.0)
    calibration_version: str = "catnet-calibrated-threshold-v1"


@dataclass(frozen=True)
class ReIDPrediction:
    cat_id: str | None
    candidates: list[tuple[str, float]]
    confidence: float
    is_known: bool
    threshold: float


class CatReIDModel:
    """Supervised projection + identity prototype Re-ID model."""

    def __init__(
        self,
        *,
        config: CatReIDModelConfig,
        center: np.ndarray,
        projection: np.ndarray,
        prototypes: dict[str, np.ndarray],
        backbone_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.center = center.astype(np.float64)
        self.projection = projection.astype(np.float64)
        self.prototypes = {cat_id: _l2(vector) for cat_id, vector in prototypes.items()}
        self.backbone_name = backbone_name
        self.metadata = metadata or {}

    @classmethod
    def fit(
        cls,
        samples: list[ReIDSample],
        *,
        config: CatReIDModelConfig | None = None,
        backbone: ReIDBackbone | None = None,
    ) -> CatReIDModel:
        config = config or CatReIDModelConfig()
        backbone = backbone or build_backbone(config.backbone)
        if len({sample.cat_id for sample in samples}) < 2:
            raise ValueError("training requires at least two identities")

        features = np.vstack([backbone.embed_sample(sample) for sample in samples])
        labels = [sample.cat_id for sample in samples]
        center = features.mean(axis=0)
        centered = features - center
        projection = _fit_fisher_projection(
            centered,
            labels,
            output_dim=min(config.embedding_dim, features.shape[1]),
            regularization=config.regularization,
        )
        embeddings = _normalize_rows(centered @ projection)
        prototypes: dict[str, np.ndarray] = {}
        for cat_id in sorted(set(labels)):
            rows = embeddings[[index for index, label in enumerate(labels) if label == cat_id]]
            prototypes[cat_id] = _l2(rows.mean(axis=0))
        metadata = {
            "trained_at": datetime.now(UTC).isoformat(),
            "sample_count": len(samples),
            "identity_count": len(prototypes),
            "backbone": backbone.name,
            "base_feature_dim": int(features.shape[1]),
        }
        return cls(
            config=config,
            center=center,
            projection=projection,
            prototypes=prototypes,
            backbone_name=backbone.name,
            metadata=metadata,
        )

    def embed_feature(self, base_feature: np.ndarray) -> np.ndarray:
        return _l2((base_feature.astype(np.float64) - self.center) @ self.projection)

    def embed_sample(
        self,
        sample: ReIDSample,
        *,
        backbone: ReIDBackbone | None = None,
    ) -> np.ndarray:
        backbone = backbone or build_backbone(self.config.backbone)
        return self.embed_feature(backbone.embed_sample(sample))

    def predict_embedding(self, embedding: np.ndarray, *, top_k: int = 5) -> ReIDPrediction:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = [
            (cat_id, float(np.dot(_l2(embedding), prototype)))
            for cat_id, prototype in self.prototypes.items()
        ]
        candidates = sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
        top_cat_id, top_score = candidates[0] if candidates else (None, -1.0)
        is_known = bool(top_cat_id and top_score >= self.config.threshold)
        confidence = _confidence(top_score, self.config.threshold)
        return ReIDPrediction(
            cat_id=top_cat_id if is_known else None,
            candidates=candidates,
            confidence=confidence,
            is_known=is_known,
            threshold=self.config.threshold,
        )

    def predict_sample(
        self,
        sample: ReIDSample,
        *,
        backbone: ReIDBackbone | None = None,
        top_k: int = 5,
    ) -> ReIDPrediction:
        return self.predict_embedding(self.embed_sample(sample, backbone=backbone), top_k=top_k)

    def calibrate_threshold(
        self,
        known_samples: list[ReIDSample],
        unknown_samples: list[ReIDSample],
        *,
        backbone: ReIDBackbone | None = None,
    ) -> float:
        backbone = backbone or build_backbone(self.config.backbone)
        positive_scores = [
            _score_for_cat(self, sample, sample.cat_id, backbone)
            for sample in known_samples
            if sample.cat_id in self.prototypes
        ]
        negative_scores = [
            max(
                self.predict_sample(sample, backbone=backbone).candidates,
                key=lambda item: item[1],
            )[1]
            for sample in unknown_samples
        ]
        if not positive_scores or not negative_scores:
            return self.config.threshold
        candidates = sorted(set([*positive_scores, *negative_scores]))
        best_threshold = self.config.threshold
        best_score = -1.0
        for threshold in candidates:
            known_accept = sum(score >= threshold for score in positive_scores) / len(
                positive_scores
            )
            unknown_reject = sum(score < threshold for score in negative_scores) / len(
                negative_scores
            )
            balanced = (known_accept * unknown_reject) ** 0.5
            if balanced > best_score:
                best_score = balanced
                best_threshold = float(threshold)
        self.config = self.config.model_copy(update={"threshold": best_threshold})
        return best_threshold

    def save(self, artifact_dir: Path) -> Path:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "catnet_reid_model.json"
        payload = {
            "schema_version": "1.0",
            "config": self.config.model_dump(mode="json"),
            "center": self.center.tolist(),
            "projection": self.projection.tolist(),
            "prototypes": {cat_id: vector.tolist() for cat_id, vector in self.prototypes.items()},
            "backbone_name": self.backbone_name,
            "metadata": self.metadata,
            "artifact_id": stable_id(
                "reid",
                self.config.model_version,
                self.metadata.get("trained_at", ""),
                len(self.prototypes),
            ),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> CatReIDModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            config=CatReIDModelConfig.model_validate(payload["config"]),
            center=np.asarray(payload["center"], dtype=np.float64),
            projection=np.asarray(payload["projection"], dtype=np.float64),
            prototypes={
                cat_id: np.asarray(vector, dtype=np.float64)
                for cat_id, vector in payload["prototypes"].items()
            },
            backbone_name=str(payload.get("backbone_name", "deterministic")),
            metadata=dict(payload.get("metadata", {})),
        )


def _fit_fisher_projection(
    centered: np.ndarray,
    labels: list[str],
    *,
    output_dim: int,
    regularization: float,
) -> np.ndarray:
    dim = centered.shape[1]
    overall = centered.mean(axis=0)
    within = np.zeros((dim, dim), dtype=np.float64)
    between = np.zeros((dim, dim), dtype=np.float64)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    for indices in by_label.values():
        rows = centered[indices]
        mean = rows.mean(axis=0)
        residuals = rows - mean
        within += residuals.T @ residuals
        diff = (mean - overall).reshape(-1, 1)
        between += len(indices) * (diff @ diff.T)
    within += np.eye(dim) * regularization
    matrix = np.linalg.pinv(within) @ between
    values, vectors = np.linalg.eig(matrix)
    order = np.argsort(values.real)[::-1]
    projection = vectors[:, order[:output_dim]].real
    if projection.shape[1] < output_dim:
        padding = np.eye(dim, output_dim - projection.shape[1])
        projection = np.concatenate([projection, padding], axis=1)
    return projection[:, :output_dim]


def _score_for_cat(
    model: CatReIDModel,
    sample: ReIDSample,
    cat_id: str,
    backbone: ReIDBackbone,
) -> float:
    embedding = model.embed_sample(sample, backbone=backbone)
    return float(np.dot(embedding, model.prototypes[cat_id]))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float64)
    return vector.astype(np.float64) / norm


def _confidence(score: float, threshold: float) -> float:
    if score >= threshold:
        return min(1.0, 0.55 + 0.45 * ((score - threshold) / max(1e-6, 1.0 - threshold)))
    return max(0.0, 0.55 * (score + 1.0) / max(1e-6, threshold + 1.0))
