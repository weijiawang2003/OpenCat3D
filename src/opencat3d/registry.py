"""External model registry and license profile gate."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from opencat3d.errors import LicenseGateError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "models" / "registry.yaml"


class ModelRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    code_license: str
    weight_license: str
    redistribution: bool
    allowed_profiles: list[str]
    sha256: str | None = None
    adapter: str | None = None
    notes: str | None = None


def load_model_registry(path: Path | str | None = None) -> dict[str, ModelRegistryEntry]:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if path is not None or registry_path.exists():
        config = OmegaConf.load(registry_path)
    else:
        with files("opencat3d").joinpath("models/registry.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            config = OmegaConf.load(handle)
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("model registry must be a mapping")
    return {name: ModelRegistryEntry.model_validate(entry) for name, entry in raw.items()}


def assert_profile_allowed(
    model_name: str,
    profile: str,
    registry: dict[str, ModelRegistryEntry] | None = None,
) -> ModelRegistryEntry:
    entries = registry if registry is not None else load_model_registry()
    try:
        entry = entries[model_name]
    except KeyError as exc:
        raise LicenseGateError(f"model '{model_name}' is not registered") from exc
    if profile not in entry.allowed_profiles:
        raise LicenseGateError(
            f"model '{model_name}' is not allowed for profile '{profile}'; "
            f"allowed profiles: {entry.allowed_profiles}"
        )
    return entry
