"""Deterministic CatNet embedding service.

This is a production-shaped baseline: it emits versioned face/body/metadata/fused
embeddings with stable dimensions and metadata. Real backbones can replace the
feature extractor behind the same contract.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from catnet.schema import BranchType, EmbeddingVector, ModelVersion, Observation, stable_id

DEFAULT_MODEL_VERSION = ModelVersion(
    model_version="catnet-deterministic-baseline-v1",
    backbone="color_texture_metadata_projection",
    train_dataset_snapshot="none-local-baseline",
    calibration_version="catnet-open-set-calibration-v1",
    embedding_dim=64,
)


class EmbeddingService:
    """Extract deterministic multi-branch embeddings from a cat image."""

    def __init__(self, model_version: ModelVersion = DEFAULT_MODEL_VERSION) -> None:
        self.model_version = model_version
        self.dim = model_version.embedding_dim

    def extract(
        self,
        image: bytes | Path | Image.Image,
        *,
        observation: Observation | None = None,
        obs_id: str | None = None,
        branches: Iterable[BranchType] | None = None,
    ) -> list[EmbeddingVector]:
        requested = list(
            branches
            or [
                BranchType.FACE,
                BranchType.BODY,
                BranchType.METADATA,
                BranchType.FUSED,
            ]
        )
        resolved_obs_id = obs_id or (
            observation.obs_id if observation else stable_id("obs", _image_digest(image))
        )
        array = _load_rgb_array(image)

        branch_vectors = {
            BranchType.FACE: self._face_embedding(array),
            BranchType.BODY: self._body_embedding(array),
            BranchType.METADATA: self._metadata_embedding(observation),
        }
        branch_vectors[BranchType.FUSED] = _l2_normalize(
            branch_vectors[BranchType.FACE] * 0.44
            + branch_vectors[BranchType.BODY] * 0.44
            + branch_vectors[BranchType.METADATA] * 0.12
        )

        embeddings: list[EmbeddingVector] = []
        for branch_type in requested:
            vector = branch_vectors[branch_type]
            norm = float(np.linalg.norm(vector))
            embeddings.append(
                EmbeddingVector(
                    emb_id=stable_id(
                        "emb",
                        resolved_obs_id,
                        branch_type.value,
                        self.model_version.model_version,
                    ),
                    obs_id=resolved_obs_id,
                    model_version=self.model_version.model_version,
                    branch_type=branch_type,
                    dim=int(vector.shape[0]),
                    vector=[float(value) for value in vector],
                    norm=norm,
                    metadata={
                        "backbone": self.model_version.backbone,
                        "calibration_version": self.model_version.calibration_version,
                    },
                )
            )
        return embeddings

    def extract_fused(
        self,
        image: bytes | Path | Image.Image,
        *,
        observation: Observation | None = None,
        obs_id: str | None = None,
    ) -> EmbeddingVector:
        return self.extract(
            image,
            observation=observation,
            obs_id=obs_id,
            branches=[BranchType.FUSED],
        )[0]

    def _face_embedding(self, array: np.ndarray) -> np.ndarray:
        height, width, _ = array.shape
        y0 = height // 5
        y1 = max(y0 + 1, height * 4 // 5)
        x0 = width // 5
        x1 = max(x0 + 1, width * 4 // 5)
        crop = array[y0:y1, x0:x1, :]
        features = np.concatenate(
            [
                _color_histogram(crop, bins=8),
                _channel_moments(crop),
                _grid_luma(crop, cells=4),
                _texture_features(crop),
            ]
        )
        return self._project(features, BranchType.FACE)

    def _body_embedding(self, array: np.ndarray) -> np.ndarray:
        features = np.concatenate(
            [
                _color_histogram(array, bins=8),
                _channel_moments(array),
                _grid_luma(array, cells=5),
                _texture_features(array),
            ]
        )
        return self._project(features, BranchType.BODY)

    def _metadata_embedding(self, observation: Observation | None) -> np.ndarray:
        values: list[float] = []
        if observation and observation.captured_at:
            captured_at = observation.captured_at
            values.extend(
                [
                    np.sin(2 * np.pi * captured_at.month / 12),
                    np.cos(2 * np.pi * captured_at.month / 12),
                    np.sin(2 * np.pi * captured_at.hour / 24),
                    np.cos(2 * np.pi * captured_at.hour / 24),
                ]
            )
        else:
            values.extend([0.0, 0.0, 0.0, 0.0])

        if observation is None:
            tokens = ["", "", "", "", "", ""]
            quality = 0.5
        else:
            tokens = [
                observation.site_id or "",
                observation.camera_id or "",
                observation.age_bin or "",
                observation.size_bin or "",
                observation.face_view or "",
                observation.body_view or "",
            ]
            quality = observation.quality_score
        values.append(float(quality))
        values.extend(_hash_token(token) for token in tokens)
        return self._project(np.asarray(values, dtype=np.float64), BranchType.METADATA)

    def _project(self, features: np.ndarray, branch_type: BranchType) -> np.ndarray:
        normalized = _l2_normalize(features.astype(np.float64))
        seed_material = f"{self.model_version.model_version}:{branch_type.value}:{len(normalized)}"
        seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        matrix = rng.normal(
            0.0,
            1.0 / np.sqrt(max(1, len(normalized))),
            (len(normalized), self.dim),
        )
        return _l2_normalize(normalized @ matrix)


def _load_rgb_array(image: bytes | Path | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, bytes):
        pil_image = Image.open(BytesIO(image)).convert("RGB")
    else:
        pil_image = Image.open(image).convert("RGB")
    pil_image = pil_image.resize((64, 64), Image.Resampling.BILINEAR)
    return np.asarray(pil_image, dtype=np.float64) / 255.0


def _image_digest(image: bytes | Path | Image.Image) -> str:
    if isinstance(image, bytes):
        return hashlib.sha256(image).hexdigest()
    if isinstance(image, Path):
        return hashlib.sha256(image.read_bytes()).hexdigest()
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _color_histogram(array: np.ndarray, *, bins: int) -> np.ndarray:
    parts = []
    for channel in range(3):
        hist, _ = np.histogram(array[:, :, channel], bins=bins, range=(0.0, 1.0))
        parts.append(hist.astype(np.float64))
    return np.concatenate(parts)


def _channel_moments(array: np.ndarray) -> np.ndarray:
    flat = array.reshape(-1, 3)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    p10 = np.percentile(flat, 10, axis=0)
    p90 = np.percentile(flat, 90, axis=0)
    return np.concatenate([mean, std, p10, p90])


def _grid_luma(array: np.ndarray, *, cells: int) -> np.ndarray:
    luma = array @ np.asarray([0.2126, 0.7152, 0.0722])
    height, width = luma.shape
    values = []
    for y in range(cells):
        for x in range(cells):
            y0 = y * height // cells
            y1 = max(y0 + 1, (y + 1) * height // cells)
            x0 = x * width // cells
            x1 = max(x0 + 1, (x + 1) * width // cells)
            values.append(float(luma[y0:y1, x0:x1].mean()))
    return np.asarray(values, dtype=np.float64)


def _texture_features(array: np.ndarray) -> np.ndarray:
    luma = array @ np.asarray([0.2126, 0.7152, 0.0722])
    grad_y, grad_x = np.gradient(luma)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.asarray(
        [
            float(magnitude.mean()),
            float(magnitude.std()),
            float(np.percentile(magnitude, 90)),
            float(np.mean(np.abs(grad_x))),
            float(np.mean(np.abs(grad_y))),
        ],
        dtype=np.float64,
    )


def _hash_token(token: str) -> float:
    if not token:
        return 0.0
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return value * 2.0 - 1.0


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float64)
    return vector.astype(np.float64) / norm
