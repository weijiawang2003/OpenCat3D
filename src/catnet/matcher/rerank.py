"""Local texture reranking for hard CatNet identity candidates.

This is a lightweight WildFusion/RAPID-inspired slot: coarse global retrieval
finds candidates, then local patch evidence reorders the top-k list and is
recorded for audit. It intentionally uses only Pillow/NumPy so the local
decision path remains available without heavyweight feature matchers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from catnet.matcher.faiss_index import SearchCandidate


@dataclass(frozen=True)
class LocalRerankConfig:
    image_size: int = 96
    grid_size: int = 4
    color_bins: int = 8
    global_weight: float = 0.70
    local_weight: float = 0.30
    spatial_penalty: float = 0.12
    top_fraction: float = 0.50
    version: str = "catnet-local-texture-rerank-v1"


@dataclass(frozen=True)
class LocalRerankResult:
    candidates: list[SearchCandidate]
    evidence_by_obs_id: dict[str, dict[str, object]]


@dataclass(frozen=True)
class _ImageSignature:
    descriptors: np.ndarray
    positions: np.ndarray


class LocalTextureReranker:
    """Rerank candidates using deterministic grid-local color/texture descriptors."""

    def __init__(self, config: LocalRerankConfig | None = None) -> None:
        self.config = config or LocalRerankConfig()
        if self.config.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.config.image_size < self.config.grid_size:
            raise ValueError("image_size must be >= grid_size")
        if self.config.color_bins <= 1:
            raise ValueError("color_bins must be greater than 1")
        if self.config.global_weight < 0 or self.config.local_weight < 0:
            raise ValueError("rerank weights must be non-negative")

    def rerank(
        self,
        *,
        query_image_path: Path,
        candidates: list[SearchCandidate],
        candidate_image_paths: dict[str, Path],
        top_k: int | None = None,
    ) -> LocalRerankResult:
        if not candidates:
            return LocalRerankResult(candidates=[], evidence_by_obs_id={})

        query_signature = _signature(query_image_path, self.config)
        reranked: list[SearchCandidate] = []
        evidence_by_obs_id: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            image_path = candidate_image_paths.get(candidate.obs_id)
            if image_path is None or not image_path.exists():
                evidence = {
                    "version": self.config.version,
                    "status": "missing_candidate_image",
                    "global_score": candidate.score,
                    "local_score": None,
                    "visual_fused_score": candidate.score,
                }
                reranked.append(_with_metadata(candidate, score=candidate.score, evidence=evidence))
                evidence_by_obs_id[candidate.obs_id] = evidence
                continue

            candidate_signature = _signature(image_path, self.config)
            local_score, evidence = _local_similarity(
                query_signature,
                candidate_signature,
                config=self.config,
            )
            fused = _weighted_visual_score(
                global_score=candidate.score,
                local_score=local_score,
                config=self.config,
            )
            evidence.update(
                {
                    "version": self.config.version,
                    "status": "ok",
                    "global_score": candidate.score,
                    "local_score": local_score,
                    "visual_fused_score": fused,
                }
            )
            reranked.append(_with_metadata(candidate, score=fused, evidence=evidence))
            evidence_by_obs_id[candidate.obs_id] = evidence

        ordered = sorted(reranked, key=lambda item: item.score, reverse=True)
        if top_k is not None:
            ordered = ordered[:top_k]
        return LocalRerankResult(candidates=ordered, evidence_by_obs_id=evidence_by_obs_id)


def _with_metadata(
    candidate: SearchCandidate,
    *,
    score: float,
    evidence: dict[str, object],
) -> SearchCandidate:
    metadata = dict(candidate.metadata)
    metadata["global_score"] = float(evidence["global_score"])
    metadata["local_score"] = evidence.get("local_score")
    metadata["visual_fused_score"] = float(evidence["visual_fused_score"])
    metadata["local_rerank"] = evidence
    return SearchCandidate(
        record_id=candidate.record_id,
        obs_id=candidate.obs_id,
        embedding_id=candidate.embedding_id,
        cat_id=candidate.cat_id,
        score=max(-1.0, min(1.0, float(score))),
        metadata=metadata,
    )


def _signature(path: Path, config: LocalRerankConfig) -> _ImageSignature:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(
            (config.image_size, config.image_size),
            Image.Resampling.BILINEAR,
        )
    array = np.asarray(rgb, dtype=np.float64) / 255.0
    grid = config.grid_size
    patch_size = config.image_size // grid
    descriptors: list[np.ndarray] = []
    positions: list[tuple[float, float]] = []
    for y_index in range(grid):
        for x_index in range(grid):
            y0 = y_index * patch_size
            x0 = x_index * patch_size
            y1 = config.image_size if y_index == grid - 1 else y0 + patch_size
            x1 = config.image_size if x_index == grid - 1 else x0 + patch_size
            patch = array[y0:y1, x0:x1]
            descriptors.append(_patch_descriptor(patch, config))
            denom = max(grid - 1, 1)
            positions.append((x_index / denom, y_index / denom))
    return _ImageSignature(
        descriptors=np.vstack(descriptors),
        positions=np.asarray(positions, dtype=np.float64),
    )


def _patch_descriptor(patch: np.ndarray, config: LocalRerankConfig) -> np.ndarray:
    channels = [patch[:, :, channel] for channel in range(3)]
    histograms = [
        np.histogram(channel, bins=config.color_bins, range=(0.0, 1.0))[0].astype(np.float64)
        for channel in channels
    ]
    hist = np.concatenate(histograms)
    hist = hist / max(float(hist.sum()), 1.0)

    means = np.asarray([float(channel.mean()) for channel in channels], dtype=np.float64)
    stds = np.asarray([float(channel.std()) for channel in channels], dtype=np.float64)
    gray = patch.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    gradients = np.asarray(
        [
            float(dx.mean()) if dx.size else 0.0,
            float(dy.mean()) if dy.size else 0.0,
            float(np.percentile(gray, 25)),
            float(np.percentile(gray, 75)),
        ],
        dtype=np.float64,
    )
    small = np.asarray(
        Image.fromarray(np.uint8(np.clip(gray * 255.0, 0, 255))).resize(
            (4, 4),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float64,
    ).reshape(-1)
    small = (small / 255.0) - float(gray.mean())
    descriptor = np.concatenate([hist, means, stds, gradients, small])
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1e-12:
        return np.zeros_like(descriptor)
    return descriptor / norm


def _local_similarity(
    query: _ImageSignature,
    candidate: _ImageSignature,
    *,
    config: LocalRerankConfig,
) -> tuple[float, dict[str, object]]:
    similarities = query.descriptors @ candidate.descriptors.T
    q_to_c = np.argmax(similarities, axis=1)
    c_to_q = np.argmax(similarities, axis=0)
    scores: list[float] = []
    mutual_count = 0
    for q_index, c_index in enumerate(q_to_c):
        is_mutual = int(c_to_q[c_index]) == q_index
        if is_mutual:
            mutual_count += 1
        spatial_distance = float(
            np.linalg.norm(query.positions[q_index] - candidate.positions[c_index])
            / np.sqrt(2.0)
        )
        score = float(similarities[q_index, c_index]) - config.spatial_penalty * spatial_distance
        if is_mutual:
            scores.append(max(0.0, min(1.0, score)))

    if not scores:
        best = similarities[np.arange(similarities.shape[0]), q_to_c]
        scores = [max(0.0, min(1.0, float(value))) for value in best]

    scores_array = np.asarray(scores, dtype=np.float64)
    keep_count = max(1, int(np.ceil(scores_array.shape[0] * config.top_fraction)))
    kept = np.sort(scores_array)[-keep_count:]
    mean_score = float(np.mean(scores_array))
    top_score = float(np.mean(kept))
    mutual_coverage = float(mutual_count / max(query.descriptors.shape[0], 1))
    coverage_factor = 0.75 + 0.25 * mutual_coverage
    local_score = (0.85 * mean_score + 0.15 * top_score) * coverage_factor
    evidence = {
        "query_patch_count": int(query.descriptors.shape[0]),
        "candidate_patch_count": int(candidate.descriptors.shape[0]),
        "mutual_match_count": int(mutual_count),
        "mutual_coverage": mutual_coverage,
        "kept_patch_count": int(keep_count),
        "mean_patch_similarity": mean_score,
        "top_patch_similarity": float(np.max(scores_array)),
    }
    return max(0.0, min(1.0, local_score)), evidence


def _weighted_visual_score(
    *,
    global_score: float,
    local_score: float,
    config: LocalRerankConfig,
) -> float:
    total = config.global_weight + config.local_weight
    if total <= 1e-12:
        return global_score
    fused = (
        config.global_weight * float(global_score)
        + config.local_weight * float(local_score)
    ) / total
    return max(-1.0, min(1.0, fused))
