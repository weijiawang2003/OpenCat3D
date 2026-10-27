from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image

from catnet.reid.backbones import (
    MEGADESCRIPTOR_T_224,
    ExternalBackboneError,
    HybridBackbone,
    build_backbone,
)
from catnet.reid.data import ReIDSample


def test_hybrid_backbone_combines_normalized_child_features(tmp_path) -> None:
    image_path = tmp_path / "cat.png"
    Image.fromarray(np.full((16, 16, 3), 96, dtype=np.uint8)).save(image_path)
    sample = ReIDSample(
        image_path=str(image_path),
        cat_id="cat_a",
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    backbone = build_backbone("hybrid:deterministic+deterministic")
    embedding = backbone.embed_sample(sample)

    assert isinstance(backbone, HybridBackbone)
    assert backbone.output_dim == 128
    assert embedding.shape == (128,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_external_backbone_reports_optional_dependency_install_hint(monkeypatch) -> None:
    def fake_import_module(name: str):
        if name == "torch":
            raise ImportError("no torch in this test")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("catnet.reid.backbones.importlib.import_module", fake_import_module)

    with pytest.raises(ExternalBackboneError, match="uv sync --extra reid"):
        build_backbone("megadescriptor-t-224")


def test_megadescriptor_backbone_uses_model_card_normalization(monkeypatch) -> None:
    captured = {}

    class FakeTimmBackbone:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.name = kwargs["name"]
            self.output_dim = 8

    monkeypatch.setattr("catnet.reid.backbones.TimmImageBackbone", FakeTimmBackbone)

    backbone = build_backbone("megadescriptor-t-224")

    assert backbone.name == "megadescriptor-t-224"
    assert captured["model_id"] == MEGADESCRIPTOR_T_224
    assert captured["image_size"] == 224
    assert captured["mean"] == (0.5, 0.5, 0.5)
    assert captured["std"] == (0.5, 0.5, 0.5)


def test_unknown_backbone_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="unknown CatNet backbone"):
        build_backbone("not-a-real-backbone")
