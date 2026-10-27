"""Deterministic texture-space provenance images."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from opencat3d.io.cache import sha256_file

PROVENANCE_COLORS: dict[str, tuple[int, int, int]] = {
    "unknown": (24, 24, 24),
    "observed": (36, 184, 96),
    "inferred": (240, 176, 32),
    "generated": (216, 48, 160),
}


@dataclass(frozen=True)
class TextureProvenanceResult:
    texture_dir: Path
    manifest_path: Path
    provenance_path: Path
    confidence_path: Path
    width: int
    height: int


def write_texture_provenance_images(
    output_root: Path,
    *,
    face_provenance: NDArray[np.integer],
    provenance_labels: dict[str, int],
    cell_size: int = 8,
    force: bool = False,
) -> TextureProvenanceResult:
    """Write inspectable face-grid provenance and confidence PNGs."""

    labels = np.asarray(face_provenance, dtype=np.int16)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("face_provenance must be a non-empty 1D array")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    texture_dir = output_root / "outputs" / "textures"
    manifest_path = texture_dir / "texture_manifest.json"
    provenance_path = texture_dir / "provenance.png"
    confidence_path = texture_dir / "confidence.png"
    for path in (provenance_path, confidence_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass force=True to overwrite")
    texture_dir.mkdir(parents=True, exist_ok=True)

    columns = int(math.ceil(math.sqrt(float(labels.size))))
    rows = int(math.ceil(labels.size / columns))
    width = columns * cell_size
    height = rows * cell_size
    provenance_image = np.full((height, width, 3), PROVENANCE_COLORS["unknown"], dtype=np.uint8)
    confidence_image = np.zeros((height, width), dtype=np.uint8)
    reverse_labels = {value: key for key, value in provenance_labels.items()}

    for face_index, label in enumerate(labels):
        row = face_index // columns
        column = face_index % columns
        y0 = row * cell_size
        x0 = column * cell_size
        label_name = reverse_labels.get(int(label), "unknown")
        color = PROVENANCE_COLORS.get(label_name, PROVENANCE_COLORS["unknown"])
        provenance_image[y0 : y0 + cell_size, x0 : x0 + cell_size, :] = color
        confidence_image[y0 : y0 + cell_size, x0 : x0 + cell_size] = (
            255 if label_name == "observed" else 0
        )

    Image.fromarray(provenance_image).save(provenance_path)
    Image.fromarray(confidence_image).save(confidence_path)
    label_counts = {
        name: int(np.count_nonzero(labels == value))
        for name, value in sorted(provenance_labels.items())
    }
    manifest = _read_manifest(manifest_path)
    manifest.update(
        {
            "schema_version": "1.0",
            "source": "opencat3d.texture_provenance_grid",
            "face_count": int(labels.size),
            "cell_size": int(cell_size),
            "grid_columns": columns,
            "grid_rows": rows,
            "width": width,
            "height": height,
            "provenance_labels": provenance_labels,
            "provenance_colors": {key: list(value) for key, value in PROVENANCE_COLORS.items()},
            "label_counts": label_counts,
        }
    )
    artifacts = manifest.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        manifest["artifacts"] = artifacts
    artifacts.update(
        {
            "provenance_png": _manifest_artifact_path(provenance_path, manifest_path),
            "confidence_png": _manifest_artifact_path(confidence_path, manifest_path),
        }
    )
    sha = manifest.setdefault("sha256", {})
    if not isinstance(sha, dict):
        sha = {}
        manifest["sha256"] = sha
    sha.update(
        {
            "provenance_png": sha256_file(provenance_path),
            "confidence_png": sha256_file(confidence_path),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TextureProvenanceResult(
        texture_dir=texture_dir,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        confidence_path=confidence_path,
        width=width,
        height=height,
    )


def _manifest_artifact_path(path: Path, manifest_path: Path) -> str:
    project_root = manifest_path.parent.parent.parent
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
