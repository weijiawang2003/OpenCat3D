"""Repeatable CatNet Re-ID benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from catnet.reid.backbones import ExternalBackboneError
from catnet.reid.data import ReIDSample, load_manifest
from catnet.reid.model import CatReIDModelConfig
from catnet.reid.training import train_reid_model
from catnet.schema import StrictModel, stable_id, utc_now

BenchmarkStatus = Literal["complete", "blocked"]


class BackboneBenchmarkReport(StrictModel):
    schema_version: str = "1.0"
    backbone: str
    status: BenchmarkStatus
    reason: str | None = None
    model_path: str | None = None
    report_path: str | None = None
    threshold: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CatNetBenchmarkSummary(StrictModel):
    schema_version: str = "1.0"
    benchmark_version: str = "catnet-benchmark-v1"
    status: BenchmarkStatus
    manifest_path: str
    output_dir: str
    created_at: str
    completed_count: int
    blocked_count: int
    reports: list[BackboneBenchmarkReport]
    comparison: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None


def run_catnet_benchmark(
    *,
    manifest_path: Path,
    backbones: list[str],
    output_dir: Path,
    embedding_dim: int = 32,
    validation_fraction: float = 0.30,
    unknown_identity_fraction: float = 0.20,
) -> CatNetBenchmarkSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        summary = CatNetBenchmarkSummary(
            status="blocked",
            manifest_path=str(manifest_path),
            output_dir=str(output_dir),
            created_at=utc_now().isoformat(),
            completed_count=0,
            blocked_count=len(backbones),
            reports=[
                BackboneBenchmarkReport(
                    backbone=backbone,
                    status="blocked",
                    reason="missing_manifest",
                )
                for backbone in backbones
            ],
            reason="missing_manifest",
        )
        _write_summary(summary, output_dir)
        return summary

    samples = load_manifest(manifest_path)
    reports: list[BackboneBenchmarkReport] = []
    for backbone in backbones:
        report = _run_backbone(
            samples=samples,
            backbone=backbone,
            output_dir=output_dir,
            embedding_dim=embedding_dim,
            validation_fraction=validation_fraction,
            unknown_identity_fraction=unknown_identity_fraction,
        )
        reports.append(report)

    completed = [report for report in reports if report.status == "complete"]
    summary = CatNetBenchmarkSummary(
        status="complete" if completed else "blocked",
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        created_at=utc_now().isoformat(),
        completed_count=len(completed),
        blocked_count=len(reports) - len(completed),
        reports=reports,
        comparison=_comparison(completed),
        reason=None if completed else "all_backbones_blocked",
    )
    _write_summary(summary, output_dir)
    return summary


def _run_backbone(
    *,
    samples: list[ReIDSample],
    backbone: str,
    output_dir: Path,
    embedding_dim: int,
    validation_fraction: float,
    unknown_identity_fraction: float,
) -> BackboneBenchmarkReport:
    artifact_dir = output_dir / _safe_backbone_dir(backbone)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "benchmark_report.json"
    try:
        train_report = train_reid_model(
            samples,
            artifact_dir=artifact_dir,
            config=CatReIDModelConfig(
                model_version=f"catnet-benchmark-{stable_id('bb', backbone)}",
                embedding_dim=embedding_dim,
                backbone=backbone,
            ),
            validation_fraction=validation_fraction,
            unknown_identity_fraction=unknown_identity_fraction,
        )
        training_payload = json.loads(train_report.report_path.read_text(encoding="utf-8"))
        payload = BackboneBenchmarkReport(
            backbone=backbone,
            status="complete",
            model_path=str(train_report.model_path),
            report_path=str(report_path),
            threshold=float(training_payload["threshold"]),
            metrics=train_report.metrics,
            metadata={"training_report_path": str(train_report.report_path)},
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        NotImplementedError,
        ExternalBackboneError,
    ) as exc:
        payload = BackboneBenchmarkReport(
            backbone=backbone,
            status="blocked",
            reason=str(exc),
        )
    report_path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _comparison(reports: list[BackboneBenchmarkReport]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        overall = report.metrics.get("overall", {})
        if not isinstance(overall, dict):
            overall = {}
        rows.append(
            {
                "backbone": report.backbone,
                "rank1": float(overall.get("rank1", 0.0)),
                "rank5": float(overall.get("rank5", 0.0)),
                "mAP": float(overall.get("mAP", 0.0)),
                "AUROC": float(overall.get("AUROC", 0.0)),
                "BAKS": float(overall.get("BAKS", 0.0)),
                "BAUS": float(overall.get("BAUS", 0.0)),
                "model_path": report.model_path,
            }
        )
    return sorted(rows, key=lambda item: (item["BAKS"], item["BAUS"], item["rank1"]), reverse=True)


def _write_summary(summary: CatNetBenchmarkSummary, output_dir: Path) -> Path:
    path = output_dir / "benchmark_summary.json"
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _safe_backbone_dir(backbone: str) -> str:
    chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in backbone]
    value = "".join(chars).strip("_")
    return value or "backbone"
