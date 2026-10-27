"""Protocols for external perception and initialization providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from opencat3d.schemas import FrameObservation, InitPrediction


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    version: str
    license_id: str


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: str


@dataclass(frozen=True)
class Segmentation:
    hard_mask: Any
    soft_mask: Any
    confidence: float


@dataclass(frozen=True)
class KeypointPrediction:
    keypoints_xyc: Any
    provider_info: ProviderInfo


class Detector(Protocol):
    def info(self) -> ProviderInfo: ...

    def detect(self, image: Any, prompt: str = "cat.") -> list[Detection]: ...


class Segmenter(Protocol):
    def info(self) -> ProviderInfo: ...

    def segment_image(self, image: Any, box: Any = None, points: Any = None) -> Segmentation: ...

    def propagate_video(self, frames: Any, seed_prompt: Any) -> list[Segmentation]: ...


class PoseEstimator(Protocol):
    def info(self) -> ProviderInfo: ...

    def predict(self, image: Any, mask: Any) -> KeypointPrediction: ...


class PointTracker(Protocol):
    def info(self) -> ProviderInfo: ...

    def track(self, frames: Any, query_points: Any, masks: Any = None) -> Any: ...


class CameraEstimator(Protocol):
    def info(self) -> ProviderInfo: ...

    def estimate(self, frames: Any, exclusion_masks: Any) -> Any: ...


class AMRInitializer(Protocol):
    def info(self) -> ProviderInfo: ...

    def predict(self, observation: FrameObservation) -> list[InitPrediction]: ...

