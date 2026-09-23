"""Tests for Azure executor public configuration behavior."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    assert ".env" in rendered
    assert ".env.*" in rendered


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


def test_configured_parent_job_name_takes_precedence_over_resume_context() -> None:
    """Explicit Azure parent job config should win over resume tracking context."""
    executor = AzureComputeExecutor(
        sweep_config={
            "executor": {
                "parent_job_name": "configured-parent-job",
            }
        },
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
        dry_run=True,
        tracking_context="resume-parent-job",
        resume=True,
    )

    executor.setup(total_runs=2)

    assert executor.parent_job_name == "configured-parent-job"
    assert executor.tracking_context == "configured-parent-job"


def test_resume_context_is_used_as_parent_job_without_configured_parent() -> None:
    """Resume behavior should still reuse the existing parent job context."""
    executor = AzureComputeExecutor(
        sweep_config={"executor": {}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
        dry_run=True,
        tracking_context="resume-parent-job",
        resume=True,
    )

    executor.setup(total_runs=2)

    assert executor.parent_job_name == "resume-parent-job"
    assert executor.tracking_context == "resume-parent-job"


def test_configured_parent_job_name_requires_string() -> None:
    """Azure parent job config should fail fast on invalid types."""
    with pytest.raises(TypeError, match=r"executor\.parent_job_name"):
        AzureComputeExecutor(
            sweep_config={"executor": {"parent_job_name": 123}},
            experiment_name="demo",
            sweep_id="sweep-1",
            compute_target="gpu-cluster",
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


@pytest.mark.parametrize(
    ("dont_wait", "interrupt", "status", "expected"),
    [
        (True, False, "Running", "submitted"),
        (False, True, "Running", "submitted"),
        (False, False, "Mystery", "unknown"),
        (False, False, "Failed", "failed"),
        (False, False, "Completed", "success"),
    ],
)
def test_execute_run_classifies_azure_status_without_false_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dont_wait: bool,
    interrupt: bool,
    status: str,
    expected: str,
) -> None:
    """Submission and interrupted log streaming are not completed runs."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump({"accelerator": {"type": "cpu"}}), encoding="utf-8"
    )
    executor = AzureComputeExecutor(
        sweep_config={"executor": {"dont_wait_for_completion": dont_wait}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )
    executor.env_vars = {"AZURE_SAS_TOKEN": "secret"}
    executor.generate_run_name = lambda config, index: "demo-run"
    executor._resolve_submission_command = lambda *args, **kwargs: "python train.py"
    submitted_kwargs: dict[str, Any] = {}
    status_checks: list[str] = []

    def fake_command(**kwargs: Any) -> dict[str, Any]:
        submitted_kwargs.update(kwargs)
        return kwargs

    def stream(job_name: str) -> None:
        assert job_name == "job-1"
        if interrupt:
            raise KeyboardInterrupt()

    def get(job_name: str) -> SimpleNamespace:
        status_checks.append(job_name)
        return SimpleNamespace(status=status)

    monkeypatch.setattr("dl_azure.executors.azure_compute.command", fake_command)
    executor.ml_client = SimpleNamespace(
        jobs=SimpleNamespace(
            create_or_update=lambda job: SimpleNamespace(name="job-1", id="id-1"),
            stream=stream,
            get=get,
        )
    )

    result = executor.execute_run(0, config_path)

    assert submitted_kwargs["environment_variables"] == executor.env_vars
    assert result[expected] is True
    assert sum(bool(result[key]) for key in ("success", "failed", "unknown", "submitted")) == 1
    assert status_checks == ([] if dont_wait else ["job-1"])


def test_sequential_submission_keeps_running_tracker_status(tmp_path: Path) -> None:
    """The default sequential path must not record accepted jobs as complete."""
    config_path = tmp_path / "run.yaml"
    executor = AzureComputeExecutor(
        sweep_config={
            "executor": {"dont_wait_for_completion": True, "retry_limit": 2}
        },
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )
    submissions: list[int] = []

    def submit(index: int, path: Path) -> dict[str, Any]:
        submissions.append(index)
        return {"submitted": True, "tracking_run_id": "job-1"}

    executor.execute_run = submit
    statuses: list[str] = []
    executor._update_tracker = (
        lambda index, status, path, result=None: statuses.append(status)
    )

    executor.execute_runs_parallel([(0, config_path)], max_workers=1)

    assert executor.submitted_runs == [0]
    assert executor.completed_runs == []
    assert executor.failed_runs == []
    assert statuses == ["running"]
    assert submissions == [0]


def test_child_job_receives_sas_but_never_storage_account_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local account key should only be used to mint a scoped job token."""
    monkeypatch.setenv("AZURE_ACCESS_KEY", "storage-secret")
    executor = AzureComputeExecutor(
        sweep_config={"executor": {}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )
    executor.azure_config = {"account_name": "demoaccount"}
    executor.generate_sas_token = lambda expiry_hours=168: "sig=job-token"

    environment = executor.get_job_environment_variables()

    assert environment["AZURE_SAS_TOKEN"] == "sig=job-token"
    assert environment["AZURE_STORAGE_ACCOUNT"] == "demoaccount"
    assert "AZURE_ACCESS_KEY" not in environment

    executor.generate_sas_token = lambda expiry_hours=168: None
    with pytest.raises(RuntimeError, match="refusing to send"):
        executor.get_job_environment_variables()
