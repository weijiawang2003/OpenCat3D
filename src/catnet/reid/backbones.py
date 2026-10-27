"""Backbone adapter contracts for CatNet Re-ID."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from catnet.embedding.service import EmbeddingService
from catnet.reid.data import ReIDSample

MEGADESCRIPTOR_T_224 = "hf-hub:BVRA/MegaDescriptor-T-224"
MEGADESCRIPTOR_L_384 = "hf-hub:BVRA/MegaDescriptor-L-384"
MIEWID_MSV3 = "conservationxlabs/miewid-msv3"


class ReIDBackbone(Protocol):
    name: str
    output_dim: int

    def embed_sample(self, sample: ReIDSample) -> np.ndarray:
        """Return one L2-normalized feature vector for a sample."""


class ExternalBackboneError(RuntimeError):
    """Raised when an optional open-source backbone dependency is unavailable."""


class DeterministicBackbone:
    """Local contract-test backbone used until trained external encoders are installed."""

    name = "deterministic_color_texture_v1"

    def __init__(self) -> None:
        self.service = EmbeddingService()
        self.output_dim = self.service.model_version.embedding_dim

    def embed_sample(self, sample: ReIDSample) -> np.ndarray:
        embedding = self.service.extract_fused(
            Path(sample.image_path),
            observation=sample.to_observation(),
            obs_id=sample.resolved_obs_id(),
        )
        return np.asarray(embedding.vector, dtype=np.float64)


class TimmImageBackbone:
    """Adapter for timm/Hugging Face animal Re-ID feature models."""

    def __init__(
        self,
        *,
        model_id: str,
        image_size: int,
        name: str,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        torch = _import_optional("torch", install_hint="uv sync --extra reid")
        timm = _import_optional("timm", install_hint="uv sync --extra reid")
        transforms = _import_optional("torchvision.transforms", install_hint="uv sync --extra reid")

        self.name = name
        self.model_id = model_id
        self.image_size = image_size
        self.torch = torch
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.output_dim = int(self.embed_image(_blank_image(image_size)).shape[0])

    def embed_sample(self, sample: ReIDSample) -> np.ndarray:
        with Image.open(sample.image_path) as image:
            return self.embed_image(image.convert("RGB"))

    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0)
        with self.torch.no_grad():
            output = self.model(tensor)
        return _tensor_to_l2_numpy(output)


class MiewIDBackbone:
    """Adapter for conservationxlabs/miewid-msv3 Hugging Face feature extractor."""

    name = "miewid-msv3"
    image_size = 440

    def __init__(self, *, model_tag: str = MIEWID_MSV3) -> None:
        torch = _import_optional("torch", install_hint="uv sync --extra reid")
        transforms = _import_optional("torchvision.transforms", install_hint="uv sync --extra reid")
        transformers = _import_optional("transformers", install_hint="uv sync --extra reid")

        self.model_tag = model_tag
        self.torch = torch
        self.model = transformers.AutoModel.from_pretrained(model_tag, trust_remote_code=True)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.output_dim = int(self.embed_image(_blank_image(self.image_size)).shape[0])

    def embed_sample(self, sample: ReIDSample) -> np.ndarray:
        with Image.open(sample.image_path) as image:
            return self.embed_image(image.convert("RGB"))

    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0)
        with self.torch.no_grad():
            output = self.model(tensor)
        return _tensor_to_l2_numpy(output)


class HybridBackbone:
    """Concatenate multiple normalized backbones into one normalized feature."""

    def __init__(self, backbones: list[ReIDBackbone]) -> None:
        if len(backbones) < 2:
            raise ValueError("hybrid backbone requires at least two child backbones")
        self.backbones = backbones
        self.name = "hybrid:" + "+".join(backbone.name for backbone in backbones)
        self.output_dim = sum(backbone.output_dim for backbone in backbones)

    def embed_sample(self, sample: ReIDSample) -> np.ndarray:
        return _l2(np.concatenate([backbone.embed_sample(sample) for backbone in self.backbones]))


def build_backbone(name: str = "deterministic") -> ReIDBackbone:
    normalized = name.lower().strip()
    if normalized in {"deterministic", "deterministic_color_texture_v1"}:
        return DeterministicBackbone()
    if normalized in {"megadescriptor", "megadescriptor-t", "megadescriptor-t-224"}:
        return TimmImageBackbone(
            model_id=MEGADESCRIPTOR_T_224,
            image_size=224,
            name="megadescriptor-t-224",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
    if normalized in {"megadescriptor-l", "megadescriptor-l-384"}:
        return TimmImageBackbone(
            model_id=MEGADESCRIPTOR_L_384,
            image_size=384,
            name="megadescriptor-l-384",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
    if normalized in {"miewid", "miewid-msv3"}:
        return MiewIDBackbone()
    if normalized.startswith("hybrid:"):
        children = [
            build_backbone(part.strip())
            for part in normalized.removeprefix("hybrid:").split("+")
            if part.strip()
        ]
        return HybridBackbone(children)
    if normalized in {"openanimals", "arbase"}:
        raise ExternalBackboneError(
            "OpenAnimals/ARBase is a training framework, not a packaged feature "
            "extractor in this repo yet. Export a checkpoint behind ReIDBackbone "
            "or use megadescriptor/miewid adapters."
        )
    raise ValueError(f"unknown CatNet backbone: {name}")


def _import_optional(module_name: str, *, install_hint: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ExternalBackboneError(
            f"Optional module '{module_name}' is required for this CatNet backbone. "
            f"Install with `{install_hint}`."
        ) from exc


def _tensor_to_l2_numpy(output: object) -> np.ndarray:
    if isinstance(output, tuple | list):
        output = output[0]
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        output = output.pooler_output
    if hasattr(output, "detach"):
        array = output.detach().cpu().numpy()
    else:
        array = np.asarray(output, dtype=np.float64)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 4:
        array = array.mean(axis=(2, 3))
    if array.ndim == 3:
        array = array.mean(axis=1)
    if array.ndim == 2:
        array = array[0]
    return _l2(array.reshape(-1))


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float64)
    return vector.astype(np.float64) / norm


def _blank_image(image_size: int) -> Image.Image:
    return Image.fromarray(np.full((image_size, image_size, 3), 127, dtype=np.uint8))
