"""Trainable CatNet Re-ID model components."""

from catnet.reid.data import ReIDSample, load_manifest, time_aware_split
from catnet.reid.model import CatReIDModel, CatReIDModelConfig, ReIDPrediction
from catnet.reid.training import TrainReport, train_reid_model

__all__ = [
    "CatReIDModel",
    "CatReIDModelConfig",
    "ReIDPrediction",
    "ReIDSample",
    "TrainReport",
    "load_manifest",
    "time_aware_split",
    "train_reid_model",
]
