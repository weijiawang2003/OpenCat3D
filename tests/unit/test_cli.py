from typer.testing import CliRunner

from opencat3d.cli import app
from opencat3d.schemas import CaptureManifest, read_json_model


def test_project_init_creates_manifest_and_dirs(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "demo_cat"

    result = runner.invoke(app, ["project", "init", str(project_dir), "--mode", "album"])

    assert result.exit_code == 0, result.output
    assert (project_dir / "config.yaml").exists()
    assert (project_dir / "cache").is_dir()
    manifest = read_json_model(CaptureManifest, project_dir / "manifest.lock.json")
    assert manifest.project_id == "demo_cat"
    assert manifest.mode.value == "album"


def test_evaluate_requires_existing_artifacts(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["evaluate", str(tmp_path)])

    assert result.exit_code == 2
    assert "is required" in result.stderr


def test_fit_cli_runs_synthetic_project(tmp_path) -> None:
    runner = CliRunner()
    project_dir = tmp_path / "demo"

    create_result = runner.invoke(
        app,
        ["synthetic", "create", str(project_dir), "--num-views", "3", "--image-size", "64"],
    )
    fit_result = runner.invoke(app, ["fit", str(project_dir), "--mode", "synthetic"])

    assert create_result.exit_code == 0, create_result.output
    assert fit_result.exit_code == 0, fit_result.output
    assert (project_dir / "outputs" / "fitting" / "fit_report.json").exists()
    assert "final_loss=" in fit_result.output


def test_catnet_m5_m8_command_groups_are_available() -> None:
    runner = CliRunner()
    commands = [
        ["catnet", "index", "build", "--help"],
        ["catnet", "feedback", "add", "--help"],
        ["catnet", "benchmark", "--help"],
        ["catnet", "consolidate", "--help"],
        ["catnet", "attach-3d", "--help"],
        ["catnet", "identity-assets", "--help"],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
