import json

import numpy as np
from PIL import Image

from opencat3d.evaluation.face_texture_targets import (
    evaluate_face_texture_targets_report,
)
from opencat3d.evaluation.face_uv_density import evaluate_face_uv_density_report
from opencat3d.template.cat_template import create_placeholder_cat_template


def test_evaluate_face_texture_targets_report_writes_future_atlas_targets(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    texture_dir = project_dir / "outputs" / "textures"
    texture_dir.mkdir(parents=True)
    Image.fromarray(np.full((512, 512, 3), 224, dtype=np.uint8)).save(
        texture_dir / "basecolor.png"
    )
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
    density = evaluate_face_uv_density_report(project_dir, mode="raw")

    result = evaluate_face_texture_targets_report(project_dir, mode="raw")

    assert density.success
    assert result.success
    assert all(result.checks.values())
    assert result.report_path.name == "raw_face_texture_targets.json"
    assert result.metrics["target_texture_size"] == 1024
    assert result.metrics["target_texture_scale"] == 2.0
    assert result.metrics["face_texture_target_requested_texel_count"] > 0
    assert result.metrics["face_texture_target_allocated_texel_count"] >= (
        result.metrics["face_texture_target_requested_texel_count"]
    )
    assert result.metrics["face_texture_target_eye_min_texels"] > 0
    assert result.metrics["face_texture_target_nose_texels"] > 0
    assert result.metrics["face_texture_target_ear_inner_min_texels"] > 0
    assert result.metrics["template_region_v2_texture_target_size"] == 2048
    assert result.metrics["template_region_v2_texture_target_scale"] == 4.0
    assert result.metrics["template_region_v2_texture_target_island_count"] == 8
    assert result.metrics["template_region_v2_texture_target_facial_island_count"] == 8
    assert result.metrics[
        "template_region_v2_texture_target_allocated_texel_count"
    ] >= result.metrics["template_region_v2_texture_target_requested_texel_count"]
    assert result.metrics["template_region_v2_texture_target_eye_min_texels"] > (
        result.metrics["face_texture_target_eye_min_texels"]
    )
    assert result.metrics["template_region_v2_texture_target_nose_texels"] > (
        result.metrics["face_texture_target_nose_texels"]
    )
    assert result.metrics["template_region_v2_texture_target_muzzle_texels"] > 0
    assert result.metrics["template_region_v2_texture_target_ear_inner_min_texels"] > (
        result.metrics["face_texture_target_ear_inner_min_texels"]
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["source"] == "opencat3d.face_texture_upsampling_targets_v1"
    assert report["output_contract"] == {
        "target_representation": "metadata_only_future_texture_bake",
        "texture_resampled": False,
        "topology_changed": False,
    }
    assert report["artifacts"]["density_report"].endswith("raw_face_uv_density.json")
    assert report["template_region_v2_target_contract"] == {
        "downstream_bake_required": True,
        "observed_data_only": True,
        "region_scope": "facial_detail_uv_islands",
        "source": "opencat3d.cat_template_region_v2_texture_targets_v1",
        "target_texture_scale": 4.0,
        "target_texture_size": 2048,
        "texture_resampled": False,
        "topology_changed": False,
    }
    nose = next(region for region in report["regions"] if region["name"] == "nose")
    assert nose["density_scale"] == 3.0
    assert nose["target_texture_size"] == 1024
    assert nose["target_allocated_texels"] >= nose["requested_texels"]
    v2_nose = next(
        region
        for region in report["template_region_v2_texture_targets"]
        if region["name"] == "nose"
    )
    assert v2_nose["allocation_role"] == "facial_detail"
    assert v2_nose["feature_names"] == ["nose"]
    assert v2_nose["target_texture_size"] == 2048
    assert v2_nose["target_allocated_texels"] >= v2_nose["requested_texels"]


def test_evaluate_face_texture_targets_report_does_not_escalate_past_stage_target(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    texture_dir = project_dir / "outputs" / "textures"
    texture_dir.mkdir(parents=True)
    Image.fromarray(np.full((1024, 1024, 3), 224, dtype=np.uint8)).save(
        texture_dir / "basecolor.png"
    )
    template = create_placeholder_cat_template()
    manifest = {
        "schema_version": "1.0",
        "basecolor": {
            "width": 1024,
            "height": 1024,
        },
        "template_face_uv_regions": template.face_uv_region_manifest(),
    }
    (texture_dir / "texture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_mesh_validation(project_dir, template.region_manifest_v2())
    evaluate_face_uv_density_report(project_dir, mode="raw")

    result = evaluate_face_texture_targets_report(project_dir, mode="raw")

    assert result.success
    assert result.metrics["target_texture_size"] == 1024
    assert result.metrics["target_texture_scale"] == 1.0
    assert result.metrics["template_region_v2_texture_target_size"] == 2048
    assert result.metrics["template_region_v2_texture_target_scale"] == 2.0


def test_evaluate_face_texture_targets_report_holds_v2_target_once_baked(
    tmp_path,
) -> None:
    project_dir = tmp_path / "project"
    texture_dir = project_dir / "outputs" / "textures"
    texture_dir.mkdir(parents=True)
    Image.fromarray(np.full((2048, 2048, 3), 224, dtype=np.uint8)).save(
        texture_dir / "basecolor.png"
    )
    template = create_placeholder_cat_template()
    manifest = {
        "schema_version": "1.0",
        "basecolor": {
            "width": 2048,
            "height": 2048,
        },
        "template_face_uv_regions": template.face_uv_region_manifest(),
    }
    (texture_dir / "texture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_mesh_validation(project_dir, template.region_manifest_v2())
    evaluate_face_uv_density_report(project_dir, mode="raw")

    result = evaluate_face_texture_targets_report(project_dir, mode="raw")

    assert result.success
    assert result.metrics["template_region_v2_texture_target_size"] == 2048
    assert result.metrics["template_region_v2_texture_target_scale"] == 1.0
    assert result.metrics[
        "template_region_v2_texture_target_allocated_texel_count"
    ] == result.metrics["template_region_v2_texture_target_requested_texel_count"]


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
