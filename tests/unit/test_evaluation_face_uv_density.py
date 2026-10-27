import json

import numpy as np
from PIL import Image

from opencat3d.evaluation.face_uv_density import (
    REQUIRED_FACE_UV_REGIONS,
    evaluate_face_uv_density_report,
)
from opencat3d.template.cat_template import create_placeholder_cat_template


def test_evaluate_face_uv_density_report_writes_overlay_and_region_texel_metrics(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    texture_dir = project_dir / "outputs" / "textures"
    texture_dir.mkdir(parents=True)
    basecolor = np.full((512, 512, 3), 224, dtype=np.uint8)
    Image.fromarray(basecolor).save(texture_dir / "basecolor.png")
    template = create_placeholder_cat_template()
    manifest = {
        "schema_version": "1.0",
        "basecolor": {
            "width": 512,
            "height": 512,
        },
        "template_face_uv_regions": template.face_uv_region_manifest(),
    }
    (texture_dir / "texture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_mesh_validation(project_dir, template.region_manifest_v2())

    result = evaluate_face_uv_density_report(project_dir, mode="raw")

    assert result.success
    assert all(result.checks.values())
    assert result.report_path.name == "raw_face_uv_density.json"
    assert result.overlay_path.name == "raw_face_uv_density.png"
    assert result.overlay_path.exists()
    assert result.metrics["face_uv_region_count"] == len(REQUIRED_FACE_UV_REGIONS)
    assert result.metrics["face_uv_region_allocated_texel_count"] > 0
    assert result.metrics["face_uv_eye_min_allocated_texels"] > 0
    assert result.metrics["face_uv_nose_allocated_texels"] > 0
    assert result.metrics["face_uv_ear_inner_min_allocated_texels"] > 0
    assert result.metrics["template_region_v2_uv_region_count"] == 10
    assert result.metrics["template_region_v2_uv_required_present_count"] == 10
    assert result.metrics["template_region_v2_face_uv_island_count"] == 8
    assert result.metrics["template_region_v2_facial_uv_island_count"] == 8
    assert result.metrics["template_region_v2_body_region_uv_island_count"] == 5
    assert result.metrics["template_region_v2_eye_min_allocated_texels"] == (
        result.metrics["face_uv_eye_min_allocated_texels"]
    )
    assert result.metrics["template_region_v2_nose_allocated_texels"] == (
        result.metrics["face_uv_nose_allocated_texels"]
    )
    assert result.metrics["template_region_v2_ear_inner_min_allocated_texels"] == (
        result.metrics["face_uv_ear_inner_min_allocated_texels"]
    )
    assert result.metrics["template_region_v2_uv_island_density_weighted_texel_count"] > (
        result.metrics["template_region_v2_uv_island_allocated_texel_count"]
    )
    assert result.metrics["template_region_v2_facial_density_weighted_texel_count"] > (
        result.metrics["template_region_v2_facial_uv_island_allocated_texel_count"]
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["source"] == "opencat3d.face_uv_texture_density_v1"
    assert report["required_regions"] == list(REQUIRED_FACE_UV_REGIONS)
    assert report["required_template_region_v2_regions"] == [
        "face",
        "muzzle",
        "eyes",
        "nose",
        "ear_interiors",
        "paws",
        "tail",
        "torso",
        "belly",
        "limbs",
    ]
    assert {region["name"] for region in report["regions"]} == set(
        REQUIRED_FACE_UV_REGIONS
    )
    by_template_region = {
        region["name"]: region for region in report["template_regions_v2"]
    }
    assert by_template_region["face"]["uv_island_count"] == 8
    assert by_template_region["eyes"]["uv_island_names"] == ["left_eye", "right_eye"]
    assert by_template_region["ear_interiors"]["uv_island_names"] == [
        "left_ear_inner",
        "right_ear_inner",
    ]
    assert by_template_region["belly"]["uv_island_names"] == ["belly_region"]
    nose = next(region for region in report["regions"] if region["name"] == "nose")
    assert nose["allocated_texels"] == result.metrics["face_uv_nose_allocated_texels"]
    unique_islands = report["unique_template_region_v2_uv_islands"]
    assert {island["name"] for island in unique_islands}.issuperset(
        {"left_eye", "right_eye", "nose", "muzzle", "left_ear_inner", "right_ear_inner"}
    )
    overlay = Image.open(result.overlay_path)
    assert overlay.mode == "RGB"
    assert overlay.size == (512, 512)


def _write_mesh_validation(project_dir, manifest: dict[str, object]) -> None:
    avatar_dir = project_dir / "outputs" / "avatar"
    avatar_dir.mkdir(parents=True)
    (avatar_dir / "mesh_validation.json").write_text(
        json.dumps(
            {
                "template_regions_v2": manifest,
                "template_region_v2_contract_valid": manifest["validation"]["valid"],
                "glb_template_regions_v2_valid": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
