"""Hydra/OmegaConf configuration helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(config_path: Path | str | None = None, overrides: Iterable[str] = ()) -> DictConfig:
    """Load the default config and apply Hydra-style dotlist overrides."""

    if config_path is not None:
        cfg = OmegaConf.load(Path(config_path))
    elif DEFAULT_CONFIG_PATH.exists():
        cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    else:
        with files("opencat3d").joinpath("configs/default.yaml").open(
            "r",
            encoding="utf-8",
        ) as handle:
            cfg = OmegaConf.load(handle)
    override_cfg = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(cfg, override_cfg)


def config_to_container(config: DictConfig) -> object:
    return OmegaConf.to_container(config, resolve=True)


def config_hash(config: DictConfig) -> str:
    payload = json.dumps(config_to_container(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
