"""Deterministic stage cache primitives."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from opencat3d.schemas import StageManifest, StageStatus, write_json_model


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_dict(value: Any) -> str:
    return sha256_bytes(stable_json_dumps(value).encode("utf-8"))


def compute_stage_key(stage_name: str, input_hash: str, config_hash: str, code_version: str) -> str:
    return hash_dict(
        {
            "stage_name": stage_name,
            "input_hash": input_hash,
            "config_hash": config_hash,
            "code_version": code_version,
        }
    )


class StageCache:
    """Manifest-backed cache for idempotent pipeline stages."""

    def __init__(self, root: Path):
        self.root = root

    def stage_dir(self, stage_name: str, cache_key: str) -> Path:
        return self.root / stage_name / cache_key

    def manifest_path(self, stage_name: str, cache_key: str) -> Path:
        return self.stage_dir(stage_name, cache_key) / "stage_manifest.json"

    def read_manifest(self, stage_name: str, cache_key: str) -> StageManifest | None:
        path = self.manifest_path(stage_name, cache_key)
        if not path.exists():
            return None
        return StageManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def is_hit(
        self,
        stage_name: str,
        input_hash: str,
        config_hash: str,
        code_version: str,
    ) -> bool:
        cache_key = compute_stage_key(stage_name, input_hash, config_hash, code_version)
        manifest = self.read_manifest(stage_name, cache_key)
        return (
            manifest is not None
            and manifest.status == StageStatus.COMPLETE
            and manifest.input_hash == input_hash
            and manifest.config_hash == config_hash
            and manifest.code_version == code_version
        )

    def write_manifest(self, manifest: StageManifest) -> Path:
        cache_key = manifest.cache_key or compute_stage_key(
            manifest.stage_name,
            manifest.input_hash,
            manifest.config_hash,
            manifest.code_version,
        )
        manifest.cache_key = cache_key
        if manifest.status in {StageStatus.COMPLETE, StageStatus.FAILED, StageStatus.SKIPPED}:
            manifest.completed_at = manifest.completed_at or datetime.now(UTC)
        path = self.manifest_path(manifest.stage_name, cache_key)
        write_json_model(manifest, path)
        return path

