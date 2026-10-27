"""Command line interface."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, cast

import typer
from omegaconf import OmegaConf

from catnet.active_learning import write_ambiguity_queue
from catnet.assets3d import attach_3d_asset, list_identity_assets, write_identity_assets
from catnet.benchmark import run_catnet_benchmark
from catnet.consolidation import run_consolidation, write_consolidation_report
from catnet.feedback import (
    add_feedback_constraint,
    apply_feedback_constraints,
    import_feedback_constraints,
    write_feedback_apply_report,
)
from catnet.index_registry import IndexRegistry
from catnet.reid.backbones import ExternalBackboneError
from catnet.reid.data import load_manifest, time_aware_split
from catnet.reid.model import CatReIDModel, CatReIDModelConfig
from catnet.reid.training import evaluate_reid_model, train_reid_model
from catnet.schema import FeedbackConstraintType
from catnet.storage import CatNetStore
from opencat3d.config import load_config
from opencat3d.evaluation.raw import evaluate_raw_reconstruction
from opencat3d.evaluation.synthetic import evaluate_synthetic_reconstruction
from opencat3d.export.mesh import export_fitted_canonical_mesh
from opencat3d.fitting.optimizer import OptimizerConfig, run_raw_fit, run_synthetic_fit
from opencat3d.ingest.images import ingest_image_sources
from opencat3d.logging import configure_logging
from opencat3d.pipeline import run_raw_pipeline, run_synthetic_pipeline
from opencat3d.preprocess.keypoints import bootstrap_raw_keypoints_and_pose
from opencat3d.preprocess.segmentation import (
    RawSegmentationBackend,
    segment_raw_image_observations,
)
from opencat3d.schemas import CaptureManifest, CaptureMode, write_json_model
from opencat3d.synthetic.generator import create_synthetic_project
from opencat3d.versioning import __version__
from opencat3d.web import UploadedFile, run_identity_image_job, serve_web

app = typer.Typer(no_args_is_help=True, invoke_without_command=True)
project_app = typer.Typer(no_args_is_help=True)
preprocess_app = typer.Typer(no_args_is_help=True)
synthetic_app = typer.Typer(no_args_is_help=True)
catnet_app = typer.Typer(no_args_is_help=True)
catnet_index_app = typer.Typer(no_args_is_help=True)
catnet_feedback_app = typer.Typer(no_args_is_help=True)
app.add_typer(project_app, name="project")
app.add_typer(preprocess_app, name="preprocess")
app.add_typer(synthetic_app, name="synthetic")
app.add_typer(catnet_app, name="catnet")
catnet_app.add_typer(catnet_index_app, name="index")
catnet_app.add_typer(catnet_feedback_app, name="feedback")


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Print version and exit.")] = False,
    log_level: Annotated[str, typer.Option("--log-level", help="Python logging level.")] = "INFO",
    json_logs: Annotated[bool, typer.Option("--json-logs", help="Emit JSON logs.")] = False,
) -> None:
    configure_logging(log_level, json_logs=json_logs)
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@project_app.command("init")
def project_init(
    project_dir: Annotated[Path, typer.Argument(help="Project directory to create.")],
    input_paths: Annotated[
        list[Path] | None,
        typer.Option(
            "--input",
            "-i",
            help="Input image/video paths. Can be supplied multiple times.",
        ),
    ] = None,
    mode: Annotated[CaptureMode, typer.Option("--mode", help="Capture mode.")] = CaptureMode.AUTO,
    project_id: Annotated[
        str | None,
        typer.Option("--project-id", help="Stable project id."),
    ] = None,
    subject_id: Annotated[
        str | None,
        typer.Option("--subject-id", help="Optional subject id."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic random seed.")] = 20260625,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Config YAML to copy."),
    ] = None,
) -> None:
    """Create an idempotent project scaffold and capture manifest."""

    project_dir.mkdir(parents=True, exist_ok=True)
    for child in ["raw", "observations", "init", "cache", "outputs", "logs"]:
        (project_dir / child).mkdir(exist_ok=True)

    cfg = load_config(config_path)
    OmegaConf.save(config=cfg, f=project_dir / "config.yaml")

    manifest = CaptureManifest(
        project_id=project_id or project_dir.name,
        subject_id=subject_id,
        mode=mode,
        source_paths=[str(path) for path in (input_paths or [])],
        random_seed=seed,
    )
    write_json_model(manifest, project_dir / "manifest.lock.json")
    typer.echo(f"initialized project {manifest.project_id} at {project_dir}")


@app.command("ingest")
def ingest(
    project_dir: Annotated[Path, typer.Argument(help="Project directory.")],
    input_paths: Annotated[
        list[Path] | None,
        typer.Option(
            "--input",
            "-i",
            help="Input image file or directory. Can be supplied multiple times.",
        ),
    ] = None,
    min_frames: Annotated[
        int,
        typer.Option("--min-frames", help="Minimum number of image frames required."),
    ] = 2,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing raw image manifest."),
    ] = False,
) -> None:
    """Ingest deterministic raw image frames into a project."""

    try:
        result = ingest_image_sources(
            project_dir,
            input_paths=input_paths,
            min_frames=min_frames,
            force=force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"ingested image frames frames={result.frame_count} manifest={result.manifest_path}")


@preprocess_app.command("segment-raw")
def preprocess_segment_raw(
    project_dir: Annotated[Path, typer.Argument(help="Project directory.")],
    alpha_threshold: Annotated[
        float,
        typer.Option("--alpha-threshold", help="Foreground alpha threshold."),
    ] = 0.05,
    color_delta_threshold: Annotated[
        float,
        typer.Option(
            "--color-delta-threshold",
            help="Foreground RGB distance from border background.",
        ),
    ] = 0.08,
    segmentation_backend: Annotated[
        str,
        typer.Option(
            "--segmentation-backend",
            help=(
                "Raw segmenter backend: auto_subject_v1, white_subject_v1, "
                "or alpha_or_border_color_v1."
            ),
        ),
    ] = "auto_subject_v1",
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing raw segmentation outputs."),
    ] = False,
) -> None:
    """Segment ingested raw image frames deterministically."""

    try:
        result = segment_raw_image_observations(
            project_dir,
            alpha_threshold=alpha_threshold,
            color_delta_threshold=color_delta_threshold,
            segmentation_backend=cast(RawSegmentationBackend, segmentation_backend),
            force=force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "raw segmentation complete "
        f"frames={result.frame_count} "
        f"mean_mask_area_ratio={result.mean_mask_area_ratio:.6f} "
        f"manifest={result.manifest_path}"
    )


@preprocess_app.command("bootstrap-raw")
def preprocess_bootstrap_raw(
    project_dir: Annotated[Path, typer.Argument(help="Project directory.")],
    keypoint_confidence: Annotated[
        float,
        typer.Option(
            "--keypoint-confidence",
            help="Confidence assigned to bbox-derived raw keypoints.",
        ),
    ] = 0.5,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing raw init predictions."),
    ] = False,
) -> None:
    """Bootstrap deterministic raw keypoints and pose seeds."""

    try:
        result = bootstrap_raw_keypoints_and_pose(
            project_dir,
            keypoint_confidence=keypoint_confidence,
            force=force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "raw keypoint bootstrap complete "
        f"frames={result.frame_count} "
        f"mean_keypoint_confidence={result.mean_keypoint_confidence:.6f} "
        f"manifest={result.manifest_path}"
    )


@synthetic_app.command("create")
def synthetic_create(
    project_dir: Annotated[Path, typer.Argument(help="Project directory.")],
    num_views: Annotated[int, typer.Option("--num-views", help="Number of camera views.")] = 6,
    image_size: Annotated[
        int,
        typer.Option("--image-size", help="Square image size in pixels."),
    ] = 128,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic random seed.")] = 20260625,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing synthetic outputs."),
    ] = False,
) -> None:
    """Create deterministic synthetic multi-view observations."""

    try:
        result = create_synthetic_project(
            project_dir,
            num_views=num_views,
            image_size=image_size,
            seed=seed,
            force=force,
        )
    except (FileExistsError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "created synthetic dataset "
        f"frames={result.frame_count} manifest={result.observation_manifest_path}"
    )


@app.command("fit")
def fit(
    project_dir: Annotated[Path, typer.Argument(help="Project directory.")],
    mode: Annotated[str, typer.Option("--mode", help="Fit mode: synthetic or raw.")] = "synthetic",
    camera_root_steps: Annotated[
        int,
        typer.Option("--camera-root-steps", help="Stage A root/camera optimization steps."),
    ] = 30,
    initial_root_offset: Annotated[
        str,
        typer.Option("--initial-root-offset", help="Comma-separated xyz root offset."),
    ] = "0.18,-0.06,0.20",
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing fitting outputs."),
    ] = False,
) -> None:
    """Run the first deterministic fitting pass."""

    if mode not in {"synthetic", "raw"}:
        typer.secho(
            "only --mode synthetic or --mode raw is implemented",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    try:
        offset = _parse_vector3(initial_root_offset)
        fit_fn = run_raw_fit if mode == "raw" else run_synthetic_fit
        report = fit_fn(
            project_dir,
            config=OptimizerConfig(camera_root_steps=camera_root_steps),
            initial_root_offset=offset,
            force=force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "fit complete "
        f"frames={report.num_frames} "
        f"initial_loss={report.metrics['initial_loss']:.6f} "
        f"final_loss={report.metrics['final_loss']:.6f} "
        f"report={report.output_paths['fit_report']}"
    )


@app.command("export")
def export(
    project_dir: Annotated[Path, typer.Argument(help="Project directory.")],
    mode: Annotated[
        str,
        typer.Option("--mode", help="Export mode: synthetic or raw."),
    ] = "synthetic",
    formats: Annotated[
        str,
        typer.Option("--formats", help="Comma-separated mesh formats: obj,ply,glb."),
    ] = "obj,ply,glb",
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing avatar exports."),
    ] = False,
) -> None:
    """Export fitted canonical mesh artifacts."""

    if mode not in {"synthetic", "raw"}:
        typer.secho(
            "only --mode synthetic or --mode raw is implemented",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    try:
        result = export_fitted_canonical_mesh(
            project_dir,
            mode=mode,
            formats=tuple(_parse_csv(formats)),
            force=force,
        )
    except (FileExistsError, FileNotFoundError, ValueError, AssertionError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "export complete "
        f"vertices={result.vertex_count} faces={result.face_count} "
        f"obj={result.obj_path} ply={result.ply_path} glb={result.glb_path} "
        f"skeleton={result.skeleton_path} "
        f"validation={result.validation_path}"
    )


@app.command("evaluate")
def evaluate(
    project_dir: Annotated[Path, typer.Argument(help="Project directory.")],
    mode: Annotated[
        str,
        typer.Option("--mode", help="Evaluation mode: synthetic or raw."),
    ] = "synthetic",
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing evaluation report."),
    ] = False,
) -> None:
    """Evaluate reconstruction artifacts."""

    if mode not in {"synthetic", "raw"}:
        typer.secho(
            "only --mode synthetic or --mode raw is implemented",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    try:
        if mode == "raw":
            result = evaluate_raw_reconstruction(project_dir, force=force)
        else:
            result = evaluate_synthetic_reconstruction(project_dir, force=force)
    except (FileExistsError, FileNotFoundError, ValueError, AssertionError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    loss_reduction_fraction = result.metrics.get("loss_reduction_fraction")
    if loss_reduction_fraction is None:
        loss_reduction_fraction = result.metrics["loss_reduction"] / max(
            result.metrics["initial_loss"],
            1e-12,
        )
    typer.echo(
        "evaluation complete "
        f"success={result.success} "
        f"loss_reduction_fraction={float(loss_reduction_fraction):.6f} "
        f"report={result.report_path}"
    )


@app.command("run")
def run(
    project_dir: Annotated[Path, typer.Argument(help="Project directory.")],
    mode: Annotated[
        str,
        typer.Option("--mode", help="Pipeline mode: synthetic or raw."),
    ] = "synthetic",
    input_paths: Annotated[
        list[Path] | None,
        typer.Option(
            "--input",
            "-i",
            help="Raw input image file or directory. Can be supplied multiple times.",
        ),
    ] = None,
    min_frames: Annotated[
        int,
        typer.Option("--min-frames", help="Minimum raw image frames required."),
    ] = 2,
    num_views: Annotated[int, typer.Option("--num-views", help="Number of synthetic views.")] = 3,
    image_size: Annotated[
        int,
        typer.Option("--image-size", help="Square synthetic image size in pixels."),
    ] = 64,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic random seed.")] = 20260625,
    alpha_threshold: Annotated[
        float,
        typer.Option("--alpha-threshold", help="Raw foreground alpha threshold."),
    ] = 0.05,
    color_delta_threshold: Annotated[
        float,
        typer.Option(
            "--color-delta-threshold",
            help="Raw foreground RGB distance from border background.",
        ),
    ] = 0.08,
    segmentation_backend: Annotated[
        str,
        typer.Option(
            "--segmentation-backend",
            help=(
                "Raw segmenter backend: auto_subject_v1, white_subject_v1, "
                "or alpha_or_border_color_v1."
            ),
        ),
    ] = "auto_subject_v1",
    keypoint_confidence: Annotated[
        float,
        typer.Option(
            "--keypoint-confidence",
            help="Confidence assigned to bbox-derived raw keypoints.",
        ),
    ] = 0.5,
    camera_root_steps: Annotated[
        int,
        typer.Option("--camera-root-steps", help="Raw root/camera optimization steps."),
    ] = 2,
    raw_feature_anchor_weight: Annotated[
        float,
        typer.Option(
            "--raw-feature-anchor-weight",
            help="Optional raw feature-anchor fitting loss weight.",
        ),
    ] = 0.0,
    max_selected_frames: Annotated[
        int,
        typer.Option(
            "--max-selected-frames",
            help="Maximum raw frames retained after deterministic frame selection.",
        ),
    ] = 5,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing pipeline outputs."),
    ] = False,
) -> None:
    """Run an end-to-end deterministic pipeline."""

    if mode not in {"synthetic", "raw"}:
        typer.secho(
            "only --mode synthetic or --mode raw is implemented",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    try:
        if mode == "raw":
            result = run_raw_pipeline(
                project_dir,
                input_paths=input_paths,
                min_frames=min_frames,
                alpha_threshold=alpha_threshold,
                color_delta_threshold=color_delta_threshold,
                segmentation_backend=cast(RawSegmentationBackend, segmentation_backend),
                keypoint_confidence=keypoint_confidence,
                camera_root_steps=camera_root_steps,
                raw_feature_anchor_weight=raw_feature_anchor_weight,
                max_selected_frames=max_selected_frames,
                force=force,
            )
        else:
            result = run_synthetic_pipeline(
                project_dir,
                num_views=num_views,
                image_size=image_size,
                seed=seed,
                force=force,
            )
    except (FileExistsError, FileNotFoundError, ValueError, AssertionError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "pipeline complete "
        f"success={result.success} "
        f"summary={result.summary_path} glb={result.glb_path}"
    )


@app.command("serve")
def serve(
    host: Annotated[
        str,
        typer.Option("--host", help="Host interface for the local upload website."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="HTTP port for the local upload website."),
    ] = 8000,
    work_dir: Annotated[
        Path,
        typer.Option("--work-dir", help="Directory for uploaded jobs and outputs."),
    ] = Path("web_runs"),
) -> None:
    """Run the local upload website for raw image reconstruction jobs."""

    typer.echo(f"OpenCat3D web UI listening on http://{host}:{port}")
    serve_web(host=host, port=port, work_dir=work_dir)


@catnet_app.command("identify")
def catnet_identify(
    image_paths: Annotated[
        list[Path],
        typer.Argument(help="One or more JPG/PNG images to identify."),
    ],
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory for CatNet identity state."),
    ] = Path("web_runs"),
    model_path: Annotated[
        Path | None,
        typer.Option("--model-path", help="Optional trained catnet_reid_model.json."),
    ] = None,
    captured_at: Annotated[
        str | None,
        typer.Option("--captured-at", help="ISO timestamp or datetime-local value."),
    ] = None,
    site_id: Annotated[str | None, typer.Option("--site-id", help="Site filter/metadata.")] = None,
    camera_id: Annotated[
        str | None,
        typer.Option("--camera-id", help="Camera/source metadata."),
    ] = None,
    uploader: Annotated[str | None, typer.Option("--uploader", help="Uploader id.")] = None,
) -> None:
    """Run the local CatNet identity path for image observations."""

    try:
        uploads = [UploadedFile(path.name, path.read_bytes()) for path in image_paths]
        result = run_identity_image_job(
            uploads,
            work_dir=store_dir,
            captured_at=_parse_cli_datetime(captured_at),
            site_id=site_id,
            camera_id=camera_id,
            uploader=uploader,
            model_path=model_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"identity complete job={result.job_id} store={result.store_dir}")
    for item in result.results:
        typer.echo(
            f"{item.filename}: cat_id={item.cat_id} decision={item.decision} "
            f"confidence={item.confidence:.3f} score={item.fused_score:.3f}"
        )


@catnet_app.command("train")
def catnet_train(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="JSONL/JSON/CSV manifest with image_path, cat_id, and metadata."),
    ],
    artifact_dir: Annotated[
        Path,
        typer.Option("--artifact-dir", help="Directory for trained model and report."),
    ] = Path("catnet_artifacts"),
    embedding_dim: Annotated[
        int,
        typer.Option("--embedding-dim", help="Output embedding dimension."),
    ] = 32,
    validation_fraction: Annotated[
        float,
        typer.Option(
            "--validation-fraction",
            help="Latest per-identity observations for validation.",
        ),
    ] = 0.30,
    unknown_identity_fraction: Annotated[
        float,
        typer.Option(
            "--unknown-identity-fraction",
            help="Identity holdout fraction for open-set eval.",
        ),
    ] = 0.20,
    backbone: Annotated[
        str,
        typer.Option("--backbone", help="Backbone adapter: deterministic, megadescriptor, miewid."),
    ] = "deterministic",
) -> None:
    """Train a CatNet Re-ID projection/prototype model from labeled cat images."""

    try:
        samples = load_manifest(manifest_path)
        report = train_reid_model(
            samples,
            artifact_dir=artifact_dir,
            config=CatReIDModelConfig(embedding_dim=embedding_dim, backbone=backbone),
            validation_fraction=validation_fraction,
            unknown_identity_fraction=unknown_identity_fraction,
        )
    except (FileNotFoundError, ValueError, NotImplementedError, ExternalBackboneError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    overall = report.metrics["overall"]
    typer.echo(
        "catnet train complete "
        f"model={report.model_path} report={report.report_path} "
        f"rank1={overall['rank1']:.3f} "
        f"BAKS={overall['BAKS']:.3f} BAUS={overall['BAUS']:.3f}"
    )


@catnet_app.command("evaluate")
def catnet_evaluate(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="JSONL/JSON/CSV manifest to evaluate."),
    ],
    model_path: Annotated[
        Path,
        typer.Option("--model-path", help="Path to catnet_reid_model.json."),
    ],
    unknown_identity_fraction: Annotated[
        float,
        typer.Option(
            "--unknown-identity-fraction",
            help="Identity holdout fraction for open-set eval.",
        ),
    ] = 0.20,
) -> None:
    """Evaluate a trained CatNet Re-ID model with time-aware and open-set metrics."""

    try:
        samples = load_manifest(manifest_path)
        split = time_aware_split(
            samples,
            unknown_identity_fraction=unknown_identity_fraction,
        )
        model = CatReIDModel.load(model_path)
        metrics = evaluate_reid_model(
            model,
            split.validation_known,
            split.validation_unknown,
            backbone_name=model.config.backbone,
        )
    except (FileNotFoundError, ValueError, NotImplementedError, ExternalBackboneError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    overall = metrics["overall"]
    typer.echo(
        "catnet evaluate complete "
        f"rank1={overall['rank1']:.3f} rank5={overall['rank5']:.3f} "
        f"mAP={overall['mAP']:.3f} AUROC={overall['AUROC']:.3f} "
        f"TNR@95TPR={overall['TNR@95TPR']:.3f}"
    )


@catnet_app.command("review-queue")
def catnet_review_queue(
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing CatNet catalog.json."),
    ] = Path("web_runs/catnet"),
    output_path: Annotated[
        Path,
        typer.Option("--output-path", help="Path for ambiguity review queue JSON."),
    ] = Path("catnet_artifacts/ambiguity_queue.json"),
    max_tasks: Annotated[
        int,
        typer.Option("--max-tasks", help="Maximum review tasks to write."),
    ] = 100,
) -> None:
    """Write an ambiguity-aware active-learning review queue from MatchEdges."""

    try:
        path = write_ambiguity_queue(
            store=CatNetStore(store_dir),
            output_path=output_path,
            max_tasks=max_tasks,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"catnet review queue complete output={path}")


@catnet_index_app.command("build")
def catnet_index_build(
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing CatNet catalog.json."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for CatNet index registry files."),
    ],
    alias: Annotated[
        str,
        typer.Option("--alias", help="Alias to point at the built index."),
    ] = "candidate",
    model_version: Annotated[
        str | None,
        typer.Option("--model-version", help="Optional embedding model version filter."),
    ] = None,
) -> None:
    """Build an immutable CatNet vector index and publish an alias."""

    try:
        result = IndexRegistry(output_dir).build_from_store(
            CatNetStore(store_dir),
            alias=alias,
            model_version=model_version,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "catnet index build complete "
        f"version={result.index_version} alias={result.alias} "
        f"vectors={result.vector_count} registry={result.registry_path}"
    )


@catnet_index_app.command("publish")
def catnet_index_publish(
    registry_dir: Annotated[
        Path,
        typer.Option("--registry-dir", help="Directory containing registry.json."),
    ],
    alias: Annotated[str, typer.Option("--alias", help="Alias to publish.")],
    index_version: Annotated[
        str,
        typer.Option("--index-version", help="Immutable index version to publish."),
    ],
) -> None:
    """Publish an existing CatNet index version to an alias."""

    try:
        record = IndexRegistry(registry_dir).publish(alias=alias, index_version=index_version)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        f"catnet index publish complete alias={record.alias} version={record.index_version}"
    )


@catnet_index_app.command("rollback")
def catnet_index_rollback(
    registry_dir: Annotated[
        Path,
        typer.Option("--registry-dir", help="Directory containing registry.json."),
    ],
    alias: Annotated[str, typer.Option("--alias", help="Alias to roll back.")],
) -> None:
    """Roll back a CatNet index alias to its previous version."""

    try:
        record = IndexRegistry(registry_dir).rollback(alias=alias)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        f"catnet index rollback complete alias={record.alias} version={record.index_version}"
    )


@catnet_index_app.command("status")
def catnet_index_status(
    registry_dir: Annotated[
        Path,
        typer.Option("--registry-dir", help="Directory containing registry.json."),
    ],
) -> None:
    """Print CatNet index registry aliases, manifests, and events."""

    typer.echo(json.dumps(IndexRegistry(registry_dir).status(), indent=2, sort_keys=True))


@catnet_feedback_app.command("add")
def catnet_feedback_add(
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing CatNet catalog.json."),
    ] = Path("web_runs/catnet"),
    constraint_type: Annotated[
        str,
        typer.Option(
            "--type",
            "--constraint-type",
            help="Constraint type: must_link, cannot_link, same_identity, different_identity.",
        ),
    ] = FeedbackConstraintType.MUST_LINK.value,
    left_obs_id: Annotated[str | None, typer.Option("--left-obs-id")] = None,
    right_obs_id: Annotated[str | None, typer.Option("--right-obs-id")] = None,
    left_cat_id: Annotated[str | None, typer.Option("--left-cat-id")] = None,
    right_cat_id: Annotated[str | None, typer.Option("--right-cat-id")] = None,
    review_task_id: Annotated[str | None, typer.Option("--review-task-id")] = None,
    reviewer: Annotated[str | None, typer.Option("--reviewer")] = None,
    reason: Annotated[str | None, typer.Option("--reason")] = None,
) -> None:
    """Add one durable reviewer feedback constraint."""

    try:
        store = CatNetStore(store_dir)
        constraint = add_feedback_constraint(
            store,
            constraint_type=constraint_type,
            left_obs_id=left_obs_id,
            right_obs_id=right_obs_id,
            left_cat_id=left_cat_id,
            right_cat_id=right_cat_id,
            review_task_id=review_task_id,
            reviewer=reviewer,
            actor="cli",
            source="catnet_feedback_add",
            reason=reason,
        )
        store.save()
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"catnet feedback add complete constraint={constraint.constraint_id}")


@catnet_feedback_app.command("import")
def catnet_feedback_import(
    labels_path: Annotated[Path, typer.Argument(help="JSON/JSONL labels to import.")],
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing CatNet catalog.json."),
    ] = Path("web_runs/catnet"),
) -> None:
    """Import durable feedback constraints from labels JSON/JSONL."""

    try:
        store = CatNetStore(store_dir)
        constraints = import_feedback_constraints(store, labels_path)
        store.save()
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"catnet feedback import complete constraints={len(constraints)}")


@catnet_feedback_app.command("apply")
def catnet_feedback_apply(
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing CatNet catalog.json."),
    ] = Path("web_runs/catnet"),
    output_path: Annotated[
        Path,
        typer.Option("--output-path", help="Path for feedback apply report."),
    ] = Path("catnet_artifacts/feedback_apply_report.json"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--apply", help="Audit only unless --apply is selected."),
    ] = True,
) -> None:
    """Apply feedback as a proposal/audit report by default."""

    try:
        store = CatNetStore(store_dir)
        report = apply_feedback_constraints(store, dry_run=dry_run)
        write_feedback_apply_report(report, output_path)
        if not dry_run:
            store.save()
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "catnet feedback apply complete "
        f"dry_run={report.dry_run} actions={report.action_count} output={output_path}"
    )


@catnet_app.command("benchmark")
def catnet_benchmark(
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="Re-ID benchmark manifest path."),
    ],
    backbones: Annotated[
        str,
        typer.Option("--backbones", help="Comma-separated backbone names."),
    ] = "deterministic,hybrid:deterministic+deterministic",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for benchmark reports."),
    ] = Path("catnet_artifacts/benchmark"),
    embedding_dim: Annotated[
        int,
        typer.Option("--embedding-dim", help="Benchmark model embedding dimension."),
    ] = 32,
) -> None:
    """Run a repeatable CatNet Re-ID benchmark harness."""

    try:
        summary = run_catnet_benchmark(
            manifest_path=manifest_path,
            backbones=_parse_csv(backbones),
            output_dir=output_dir,
            embedding_dim=embedding_dim,
        )
    except (FileNotFoundError, ValueError, NotImplementedError, ExternalBackboneError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "catnet benchmark complete "
        f"status={summary.status} completed={summary.completed_count} "
        f"blocked={summary.blocked_count} summary={output_dir / 'benchmark_summary.json'}"
    )


@catnet_app.command("consolidate")
def catnet_consolidate(
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing CatNet catalog.json."),
    ] = Path("web_runs/catnet"),
    output_path: Annotated[
        Path,
        typer.Option("--output-path", help="Path for consolidation report."),
    ] = Path("catnet_artifacts/consolidation_report.json"),
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Record proposal events; default is report only."),
    ] = False,
) -> None:
    """Run proposal-first identity graph consolidation."""

    try:
        store = CatNetStore(store_dir)
        report = run_consolidation(store, apply=apply)
        write_consolidation_report(report, output_path)
        if apply:
            store.save()
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        "catnet consolidate complete "
        f"merge={len(report.merge_proposals)} split={len(report.split_proposals)} "
        f"blocked={len(report.blocked_proposals)} output={output_path}"
    )


@catnet_app.command("attach-3d")
def catnet_attach_3d(
    cat_id: Annotated[str, typer.Option("--cat-id", help="CatNet identity id.")],
    project_dir: Annotated[
        Path,
        typer.Option("--project-dir", help="OpenCat3D project directory."),
    ],
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing CatNet catalog.json."),
    ] = Path("web_runs/catnet"),
    source_obs_ids: Annotated[
        str | None,
        typer.Option("--source-obs-ids", help="Comma-separated source observation ids."),
    ] = None,
    reconstruction_version: Annotated[
        str,
        typer.Option("--reconstruction-version", help="OpenCat3D reconstruction version label."),
    ] = "opencat3d-v1",
) -> None:
    """Attach OpenCat3D reconstruction artifacts to a CatNet identity."""

    try:
        store = CatNetStore(store_dir)
        asset = attach_3d_asset(
            store,
            cat_id=cat_id,
            project_dir=project_dir,
            source_obs_ids=_parse_csv(source_obs_ids) if source_obs_ids else None,
            reconstruction_version=reconstruction_version,
        )
        store.save()
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"catnet attach-3d complete cat_id={asset.cat_id} asset={asset.asset_id}")


@catnet_app.command("identity-assets")
def catnet_identity_assets(
    cat_id: Annotated[str, typer.Option("--cat-id", help="CatNet identity id.")],
    store_dir: Annotated[
        Path,
        typer.Option("--store-dir", help="Directory containing CatNet catalog.json."),
    ] = Path("web_runs/catnet"),
    output_path: Annotated[
        Path | None,
        typer.Option("--output-path", help="Optional JSON output path."),
    ] = None,
) -> None:
    """List OpenCat3D reconstruction artifacts attached to one identity."""

    try:
        assets = list_identity_assets(CatNetStore(store_dir), cat_id=cat_id)
        payload = {
            "schema_version": "1.0",
            "asset_count": len(assets),
            "assets": [asset.model_dump(mode="json") for asset in assets],
        }
        if output_path is not None:
            write_identity_assets(assets, output_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    if output_path is None:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"catnet identity-assets complete count={len(assets)} output={output_path}")


def _not_implemented(command_name: str, task_name: str) -> None:
    typer.secho(
        f"'{command_name}' is part of {task_name} and is not implemented in Task 1.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(2)


def _parse_vector3(raw: str) -> list[float]:
    values = [part.strip() for part in raw.split(",")]
    if len(values) != 3:
        raise ValueError("expected three comma-separated values")
    return [float(value) for value in values]


def _parse_csv(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def _parse_cli_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    app()
