"""Tests for Azure executor public configuration behavior."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from dl_core.core import BaseExecutor

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


def test_setup_rejects_old_core_before_creating_azure_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported core must fail before Azure setup has side effects."""
    monkeypatch.delattr(BaseExecutor, "_after_run_execution")
    executor = AzureComputeExecutor(
        sweep_config={"executor": {}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )

    with pytest.raises(RuntimeError, match="requires matching dl-core sweep hooks"):
        executor.setup(total_runs=1)

    assert executor.parent_job_name is None


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
    """The default sweep entry point must keep submitted jobs running."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: run\n", encoding="utf-8")
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    executor = AzureComputeExecutor(
        sweep_config={
            "sweep_file": str(sweep_path),
            "tracking": {"backend": "local"},
            "executor": {"dont_wait_for_completion": True, "retry_limit": 2},
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
    executor.setup = lambda total_runs: None
    executor.teardown = lambda: None

    progress = executor.run_sweep([(0, config_path)], max_workers=1)

    assert executor.submitted_runs == [0]
    assert executor.completed_runs == []
    assert executor.failed_runs == []
    assert all(progress[key] == expected for key, expected in {
        "completed": 0, "failed": 0, "skipped": 0, "total": 1,
    }.items())
    assert executor.tracker.get_sweep_data()["runs"]["0"]["status"] == "running"
    assert submissions == [0]

    resumed = AzureComputeExecutor(
        sweep_config=executor.sweep_config,
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
        resume=True,
    )
    resumed.setup = lambda total_runs: None
    resumed.teardown = lambda: None
    resumed.execute_run = lambda index, path: pytest.fail("running job resubmitted")
    resumed.run_sweep([(0, config_path)], max_workers=1)
    assert resumed.skipped_runs == [0]


def test_default_azure_sweep_keeps_unknown_job_out_of_retry_queue(
    tmp_path: Path,
) -> None:
    """An indeterminate Azure submission must not become a failed run."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: run\n", encoding="utf-8")
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    executor = AzureComputeExecutor(
        sweep_config={"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )
    executor.setup = lambda total_runs: None
    executor.teardown = lambda: None
    executor.execute_run = lambda index, path: {"unknown": True, "tracking_run_id": "job-1"}

    executor.run_sweep([(0, config_path)], max_workers=1)

    assert executor.unknown_runs == [0]
    assert executor.failed_runs == []
    assert executor.tracker.get_sweep_data()["runs"]["0"]["status"] == "unknown"


@pytest.mark.parametrize("max_workers", [1, 2])
def test_azure_sweep_retries_raised_submission_once(
    tmp_path: Path,
    max_workers: int,
) -> None:
    """Failed submissions should retry exactly once in either execution mode."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    configs = [(index, tmp_path / f"run-{index}.yaml") for index in range(2)]
    for _, config_path in configs:
        config_path.write_text("runtime:\n  name: run\n", encoding="utf-8")
    executor = AzureComputeExecutor(
        sweep_config={
            "sweep_file": str(sweep_path),
            "tracking": {"backend": "local"},
            "executor": {"retry_limit": 1},
        },
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )
    executor.setup = lambda total_runs: None
    executor.teardown = lambda: None
    attempts: list[int] = []

    def submit(index: int, path: Path) -> dict[str, Any]:
        attempts.append(index)
        if index == 0 and attempts.count(0) == 1:
            raise RuntimeError("submission rejected")
        return {"submitted": True, "tracking_run_id": f"job-{index}"}

    executor.execute_run = submit

    progress = executor.run_sweep(configs, max_workers=max_workers)

    assert attempts.count(0) == 2
    assert attempts.count(1) == 1
    if max_workers == 1:
        assert attempts == [0, 1, 0]
    assert executor.retry_attempts == {0: 1}
    assert executor.failed_runs == []
    assert sorted(executor.submitted_runs) == [0, 1]
    assert all(progress[key] == expected for key, expected in {
        "completed": 0, "failed": 0, "skipped": 0, "total": 2,
    }.items())
    statuses = executor.tracker.get_sweep_data()["runs"]
    assert statuses["0"]["status"] == "running"
    assert statuses["1"]["status"] == "running"


def test_parallel_azure_sweep_uses_status_hook(tmp_path: Path) -> None:
    """Azure's parallel path must use its result classifier as well."""
    configs = [(index, tmp_path / f"run-{index}.yaml") for index in range(2)]
    for _, config_path in configs:
        config_path.write_text("runtime:\n  name: run\n", encoding="utf-8")
    executor = AzureComputeExecutor(
        sweep_config={"tracking": {"backend": "local"}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )
    executor.setup = lambda total_runs: None
    executor.teardown = lambda: None
    executor.execute_run = lambda index, path: {"success": True}
    executor._classify_run_result = lambda result: "unknown"

    progress = executor.run_sweep(configs, max_workers=2)

    assert sorted(executor.unknown_runs) == [0, 1]
    assert executor.completed_runs == []
    assert all(progress[key] == expected for key, expected in {
        "completed": 0, "failed": 0, "skipped": 0, "total": 2,
    }.items())


def test_parallel_azure_claim_blocks_overlapping_resume(tmp_path: Path) -> None:
    """A second process cannot submit a claimed Azure run."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    configs = [(index, tmp_path / f"run-{index}.yaml") for index in range(2)]
    for _, path in configs:
        path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    settings = {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}}
    first = AzureComputeExecutor(settings, "demo", "sweep-1", compute_target="cpu")
    second = AzureComputeExecutor(
        settings, "demo", "sweep-1", compute_target="cpu", resume=True
    )
    first.setup = lambda total_runs: None
    first.teardown = lambda: None
    calls: list[int] = []

    def submit(index: int, path: Path) -> dict[str, Any]:
        calls.append(index)
        assert second._execute_single_run_wrapper(index, path)["skipped"] is True
        return {"submitted": True, "tracking_run_id": f"job-{index}"}

    first.execute_run = submit
    second.execute_run = lambda index, path: pytest.fail("duplicate Azure submission")

    first.run_sweep(configs, max_workers=2)

    assert sorted(calls) == [0, 1]
    assert sorted(first.submitted_runs) == [0, 1]


def test_azure_retry_claim_blocks_overlapping_resume(tmp_path: Path) -> None:
    """Retrying a failed job must atomically re-claim it first."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    settings = {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}}
    first = AzureComputeExecutor(settings, "demo", "sweep-1", compute_target="cpu")
    second = AzureComputeExecutor(
        settings, "demo", "sweep-1", compute_target="cpu", resume=True
    )
    first.setup = lambda total_runs: None
    first.teardown = lambda: None
    calls = 0

    def submit(index: int, path: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        assert second._execute_single_run_wrapper(index, path)["skipped"] is True
        return {"submitted": True, "tracking_run_id": "job-retry"}

    first.execute_run = submit
    second.execute_run = lambda index, path: pytest.fail("duplicate Azure retry")
    first.retry_limit = 1

    first.run_sweep([(0, config_path)], max_workers=1)

    assert calls == 2
    assert first.tracker.get_sweep_data()["runs"]["0"]["status"] == "running"


def test_azure_retry_skips_run_claimed_by_other_process(tmp_path: Path) -> None:
    """A stale failed list cannot resubmit a job another process claimed."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    settings = {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}}
    executor = AzureComputeExecutor(settings, "demo", "sweep-1", compute_target="cpu")
    executor.setup = lambda total_runs: None
    executor.teardown = lambda: None
    executor.execute_run = lambda index, path: (_ for _ in ()).throw(
        RuntimeError("submission failed")
    )
    executor.run_sweep([(0, config_path)], max_workers=1)
    assert executor.tracker.try_claim_run(0, config_path=str(config_path))
    executor.execute_run = lambda index, path: pytest.fail("duplicate Azure job")
    executor.retry_limit = 1

    executor._retry_failed_runs({0: config_path}, 1)

    assert executor.skipped_runs == [0]
    assert executor.failed_runs == []
    assert executor.tracker.get_sweep_data()["runs"]["0"]["status"] == "running"


def test_azure_tracker_write_error_never_retries_accepted_job(tmp_path: Path) -> None:
    """Tracker failure after Azure accepts a job must abort the sweep."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    configs = [(index, tmp_path / f"run-{index}.yaml") for index in range(2)]
    for _, path in configs:
        path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    executor = AzureComputeExecutor(
        {"sweep_file": str(sweep_path), "tracking": {"backend": "local"}},
        "demo", "sweep-1", compute_target="cpu",
    )
    executor.setup = lambda total_runs: None
    executor.teardown = lambda: None
    executor.retry_limit = 2
    calls: list[int] = []
    executor.execute_run = lambda index, path: (
        calls.append(index) or {"submitted": True, "tracking_run_id": f"job-{index}"}
    )
    original_update = executor._update_tracker

    def write(index: int, status: str, path: Path, **kwargs: Any) -> None:
        if status == "running":
            raise OSError("tracker disk failure")
        original_update(index, status, path, **kwargs)

    executor._update_tracker = write

    with pytest.raises(OSError, match="tracker disk failure"):
        executor.run_sweep(configs, max_workers=2)

    assert sorted(calls) == [0, 1]
    assert executor.failed_runs == []


def test_azure_retry_tracker_write_error_aborts_without_resubmit(tmp_path: Path) -> None:
    """A retry accepted by Azure must not enter a second retry on disk error."""
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("base_config: run.yaml\n", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text("runtime:\n  name: demo\n", encoding="utf-8")
    executor = AzureComputeExecutor(
        {
            "sweep_file": str(sweep_path),
            "tracking": {"backend": "local"},
            "executor": {"retry_limit": 2},
        },
        "demo", "sweep-1", compute_target="cpu",
    )
    executor.setup = lambda total_runs: None
    executor.teardown = lambda: None
    attempts = 0

    def submit(index: int, path: Path) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return {"submitted": True, "tracking_run_id": "accepted-job"}

    executor.execute_run = submit
    original_update = executor._update_tracker

    def write(index: int, status: str, path: Path, **kwargs: Any) -> None:
        if status == "running" and attempts == 2:
            raise OSError("tracker disk failure")
        original_update(index, status, path, **kwargs)

    executor._update_tracker = write

    with pytest.raises(OSError, match="tracker disk failure"):
        executor.run_sweep([(0, config_path)], max_workers=1)

    assert attempts == 2
    assert executor.retry_attempts == {0: 1}


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
