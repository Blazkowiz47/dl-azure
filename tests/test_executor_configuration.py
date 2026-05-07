"""Tests for Azure executor public configuration behavior."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from dl_azure.executors.azure_compute import AzureComputeExecutor


def test_update_amlignore_preserves_user_content(tmp_path: Path) -> None:
    """The managed Azure block should not overwrite user `.amlignore` content."""
    amlignore_path = tmp_path / ".amlignore"
    amlignore_path.write_text("custom-entry/\n", encoding="utf-8")

    executor = AzureComputeExecutor(
        sweep_config={"executor": {}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        executor.update_amlignore("experiments/lr_sweep.yaml")
    finally:
        os.chdir(previous_cwd)

    rendered = amlignore_path.read_text(encoding="utf-8")
    assert "custom-entry/" in rendered
    assert "# BEGIN dl-azure managed block" in rendered
    assert "# END dl-azure managed block" in rendered
    assert "lab/users/" not in rendered
    assert "lab/template/" not in rendered
    assert "outputs/" in rendered


def test_build_command_uses_remote_python_for_azure_jobs() -> None:
    """Azure executor commands should not capture the local virtualenv path."""

    executor = AzureComputeExecutor(
        sweep_config={"executor": {}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )

    command = executor.build_command(
        "experiments/lr_sweep/run_001.yaml",
        {
            "accelerator": {"type": "cpu"},
            "runtime": {"log_level": "INFO"},
        },
    )

    assert command[:3] == ["python", "-m", "dl_core.worker"]


def test_resolve_submission_command_uses_executor_command_override() -> None:
    """Azure executor should honor a configured custom command string."""

    executor = AzureComputeExecutor(
        sweep_config={
            "executor": {
                "command": (
                    "python scripts/new_azure_script.py "
                    "--config {config_path} "
                    "--name {run_name} "
                    "--job {run_number} "
                    "--parent {tracking_context} "
                    "--mount ${{inputs.dataset_path}}"
                )
            }
        },
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
        tracking_context="azure-parent-job",
    )

    command = executor._resolve_submission_command(
        Path("experiments/lr_sweep/run_001.yaml"),
        {
            "accelerator": {"type": "cpu"},
            "runtime": {"log_level": "INFO"},
        },
        run_index=0,
        run_name="demo-run",
    )

    assert command == (
        "python scripts/new_azure_script.py "
        "--config experiments/lr_sweep/run_001.yaml "
        "--name demo-run "
        "--job 1 "
        "--parent azure-parent-job "
        "--mount ${{inputs.dataset_path}}"
    )


def test_resolve_submission_command_rejects_unknown_placeholders() -> None:
    """Azure executor should fail fast on unsupported custom command placeholders."""

    executor = AzureComputeExecutor(
        sweep_config={"executor": {"command": "python script.py --value {unknown}"}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )

    with pytest.raises(ValueError, match=r"Unsupported executor\.command placeholder"):
        executor._resolve_submission_command(
            Path("experiments/lr_sweep/run_001.yaml"),
            {
                "accelerator": {"type": "cpu"},
                "runtime": {"log_level": "INFO"},
            },
            run_index=0,
            run_name="demo-run",
        )


def test_execute_run_promotes_default_output_dir_to_azure_outputs(
    tmp_path: Path,
) -> None:
    """Azure runs should write default artifacts into the managed outputs tree."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "accelerator": {"type": "cpu"},
                "runtime": {"log_level": "INFO", "output_dir": "artifacts"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    executor = AzureComputeExecutor(
        sweep_config={"executor": {}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
        dry_run=True,
    )

    executor.execute_run(0, config_path)

    saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved_config["runtime"]["output_dir"] == "outputs/artifacts"
