import json
from pathlib import Path

import numpy as np
from PIL import Image

import opencat3d.evaluation.coverage as coverage_module
from opencat3d.evaluation.coverage import PROVENANCE_LABELS, evaluate_mesh_coverage
from opencat3d.fitting.optimizer import OptimizerConfig, run_synthetic_fit
from opencat3d.synthetic.generator import create_synthetic_project
from opencat3d.template.cat_template import CAT_TEMPLATE_FACE_COUNT, CAT_TEMPLATE_VERTEX_COUNT


def test_evaluate_mesh_coverage_writes_conservative_provenance(tmp_path) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=64, seed=999)
    run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=2, shape_steps=0),
        initial_root_offset=np.asarray([0.18, -0.06, 0.20]),
    )

    result = evaluate_mesh_coverage(project_dir, mode="synthetic")

    assert result.success
    assert result.report_path.exists()
    assert result.provenance_path.exists()
    assert result.texture_manifest_path.exists()
    assert result.metrics["face_count"] == CAT_TEMPLATE_FACE_COUNT
    assert result.metrics["vertex_count"] == CAT_TEMPLATE_VERTEX_COUNT
    assert result.metrics["observed_face_count"] > 0
    assert result.metrics["observed_face_fraction"] > 0.0
    assert result.metrics["basecolor_observed_face_count"] > 0
    assert result.metrics["basecolor_observed_texel_fraction"] > 0.0
    assert result.metrics["basecolor_generated_texel_fraction"] == 0.0
    assert result.metrics["pbr_observed_texel_fraction"] > 0.0
    assert result.metrics["pbr_generated_texel_fraction"] == 0.0
    assert result.metrics["generated_face_count"] == 0
    assert result.checks["no_generated_regions_claimed"]
    assert result.checks["texture_basecolor_written"]
    assert result.checks["pbr_textures_written"]
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["provenance_labels"] == PROVENANCE_LABELS
    assert report["artifacts"]["fit_state"].endswith("outputs/fitting/fit_state.safetensors")
    assert report["artifacts"]["fit_state_source_format"] == "safetensors"
    assert report["artifacts"]["provenance_arrays"].endswith("coverage_provenance.npz")
    assert report["artifacts"]["texture_manifest"].endswith(
        "outputs/textures/texture_manifest.json"
    )
    provenance_png = report["artifacts"]["provenance_png"]
    confidence_png = report["artifacts"]["confidence_png"]
    basecolor_png = report["artifacts"]["basecolor_png"]
    basecolor_confidence_png = report["artifacts"]["basecolor_confidence_png"]
    source_view_id_png = report["artifacts"]["source_view_id_png"]
    normal_png = report["artifacts"]["normal_png"]
    roughness_png = report["artifacts"]["roughness_png"]
    ao_png = report["artifacts"]["ao_png"]
    assert provenance_png.endswith("outputs/textures/provenance.png")
    assert confidence_png.endswith("outputs/textures/confidence.png")
    assert basecolor_png.endswith("outputs/textures/basecolor.png")
    assert basecolor_confidence_png.endswith("outputs/textures/basecolor_confidence.png")
    assert source_view_id_png.endswith("outputs/textures/source_view_id.png")
    assert normal_png.endswith("outputs/textures/normal.png")
    assert roughness_png.endswith("outputs/textures/roughness.png")
    assert ao_png.endswith("outputs/textures/ao.png")
    assert Image.open(provenance_png).mode == "RGB"
    assert Image.open(confidence_png).mode == "L"
    assert Image.open(basecolor_png).mode == "RGB"
    assert Image.open(basecolor_confidence_png).mode == "L"
    assert Image.open(source_view_id_png).mode == "L"
    normal = np.asarray(Image.open(normal_png).convert("RGB"))
    roughness = np.asarray(Image.open(roughness_png).convert("L"))
    ao = np.asarray(Image.open(ao_png).convert("L"))
    assert tuple(normal[0, 0]) == (128, 128, 255)
    assert set(np.unique(roughness)).issubset({204, 255})
    assert np.all(ao == 255)
    manifest = json.loads(result.texture_manifest_path.read_text(encoding="utf-8"))
    assert manifest["basecolor"]["observed_texel_fraction"] > 0.0
    assert manifest["basecolor"]["generated_texel_fraction"] == 0.0
    assert manifest["basecolor"]["width"] == 256
    assert manifest["basecolor"]["height"] == 256
    assert manifest["basecolor"]["texture_resolution_source"] == "default_256"
    assert len(manifest["basecolor"]["unknown_rgb"]) == 3
    assert manifest["basecolor"]["unknown_rgb_source"] == "observed_foreground_high_luma_median"
    assert manifest["pbr_source"] == "opencat3d.neutral_pbr_priors"
    assert manifest["pbr"]["width"] == 256
    assert manifest["pbr"]["height"] == 256
    assert manifest["pbr"]["normal_rgb_flat"] == [128, 128, 255]
    assert manifest["pbr"]["face_material_detail_applied"] is False
    assert manifest["pbr"]["face_material_detail_source"] == "none"
    assert manifest["pbr"]["face_material_detail_texel_count"] == 0
    assert manifest["pbr"]["generated_texel_fraction"] == 0.0
    assert manifest["artifacts"]["basecolor_png"].endswith("basecolor.png")
    assert manifest["artifacts"]["normal_png"].endswith("normal.png")
    for key, raw_path in manifest["artifacts"].items():
        artifact_path = Path(str(raw_path))
        assert not artifact_path.is_absolute()
        assert not str(artifact_path).startswith(str(project_dir))
        assert (project_dir / artifact_path).exists(), key
        assert key in manifest["sha256"]
    with np.load(result.provenance_path) as data:
        face_provenance = data["face_provenance"]
        vertex_provenance = data["vertex_provenance"]
    assert face_provenance.shape == (CAT_TEMPLATE_FACE_COUNT,)
    assert vertex_provenance.shape == (CAT_TEMPLATE_VERTEX_COUNT,)
    assert set(np.unique(face_provenance)).issubset(
        {PROVENANCE_LABELS["unknown"], PROVENANCE_LABELS["observed"]}
    )


def test_evaluate_mesh_coverage_can_reuse_existing_basecolor_on_force(
    tmp_path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "demo"
    create_synthetic_project(project_dir, num_views=3, image_size=48, seed=1001)
    run_synthetic_fit(
        project_dir,
        config=OptimizerConfig(camera_root_steps=1, shape_steps=0),
        initial_root_offset=np.asarray([0.10, -0.02, 0.15]),
    )
    evaluate_mesh_coverage(project_dir, mode="synthetic")

    original_writer = coverage_module.write_observed_basecolor_texture
    force_values: list[bool] = []

    def wrapped_writer(*args, **kwargs):
        force_values.append(bool(kwargs.get("force", False)))
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(
        coverage_module,
        "write_observed_basecolor_texture",
        wrapped_writer,
    )

    result = coverage_module.evaluate_mesh_coverage(
        project_dir,
        mode="synthetic",
        reuse_existing_basecolor=True,
        force=True,
    )

    assert result.success
    assert force_values == [False]
