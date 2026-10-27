import json

import numpy as np
from PIL import Image

from opencat3d.textures.provenance import (
    PROVENANCE_COLORS,
    write_texture_provenance_images,
)


def test_write_texture_provenance_images_writes_grid_pngs(tmp_path) -> None:
    labels = {"unknown": 0, "observed": 1, "inferred": 2, "generated": 3}
    face_provenance = np.asarray([0, 1, 1, 0], dtype=np.int8)

    result = write_texture_provenance_images(
        tmp_path,
        face_provenance=face_provenance,
        provenance_labels=labels,
        cell_size=4,
    )

    assert result.manifest_path.exists()
    assert result.provenance_path.exists()
    assert result.confidence_path.exists()
    assert result.width == 8
    assert result.height == 8
    provenance = np.asarray(Image.open(result.provenance_path).convert("RGB"))
    confidence = np.asarray(Image.open(result.confidence_path).convert("L"))
    assert tuple(provenance[1, 1]) == PROVENANCE_COLORS["unknown"]
    assert tuple(provenance[1, 5]) == PROVENANCE_COLORS["observed"]
    assert int(confidence[1, 1]) == 0
    assert int(confidence[1, 5]) == 255
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["label_counts"] == {
        "generated": 0,
        "inferred": 0,
        "observed": 2,
        "unknown": 2,
    }
    assert manifest["artifacts"]["provenance_png"].endswith("provenance.png")
    assert manifest["artifacts"]["confidence_png"].endswith("confidence.png")
