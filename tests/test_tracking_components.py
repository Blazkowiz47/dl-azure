"""Tests for Azure tracker and metrics source registration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

import dl_azure
from dl_azure.callbacks.mlflow import AzureMlflowCallback
from dl_azure.trackers.azure_mlflow import AzureMlflowTracker
from dl_core.core import CALLBACK_REGISTRY, METRICS_SOURCE_REGISTRY, TRACKER_REGISTRY


class _DummyAccelerator:
    """Small accelerator test double."""

    def is_main_process(self) -> bool:
        """Return that this is the main process."""
        return True


class _DummyTrainer:
    """Small trainer test double."""

    def __init__(self) -> None:
        self.accelerator = _DummyAccelerator()
        self.config = {
            "runtime": {"name": "demo-run"},
            "experiment": {"name": "demo-experiment"},
            "tracking": {
                "backend": "azure_mlflow",
                "uri": "azureml://tracking",
                "context": "parent-run-789",
                "run_name": "demo-run",
            },
        }
        self.artifact_manager = SimpleNamespace(run_dir=".")


def test_azure_tracker_and_metrics_source_are_registered() -> None:
    """Importing dl-azure should register tracker, metrics source, and callback."""
    assert dl_azure.__version__ == "0.0.1"
    assert TRACKER_REGISTRY.is_registered("azure_mlflow")
    assert METRICS_SOURCE_REGISTRY.is_registered("azure_mlflow")
    assert CALLBACK_REGISTRY.is_registered("azure_mlflow")


def test_azure_mlflow_callback_uses_tracking_config(
    monkeypatch: MonkeyPatch,
) -> None:
    """The Azure MLflow callback should resolve tracking URI and parent run."""
    events: list[tuple[str, str | None]] = []

    def fake_set_tracking_uri(uri: str) -> None:
        events.append(("uri", uri))

    def fake_set_experiment(name: str) -> None:
        events.append(("experiment", name))

    def fake_start_run(
        run_id: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
    ):
        del nested
        events.append(("start", run_id or run_name))
        return SimpleNamespace(info=SimpleNamespace(run_id="child-run-999"))

    monkeypatch.setattr(
        "dl_azure.callbacks.mlflow.mlflow",
        SimpleNamespace(
            set_tracking_uri=fake_set_tracking_uri,
            set_experiment=fake_set_experiment,
            start_run=fake_start_run,
            log_params=lambda *_args, **_kwargs: None,
            log_artifact=lambda *_args, **_kwargs: None,
            end_run=lambda: None,
        ),
    )

    callback = AzureMlflowCallback()
    callback.set_trainer(_DummyTrainer())
    callback.on_training_start()

    assert ("uri", "azureml://tracking") in events
    assert ("experiment", "demo-experiment") in events
    assert ("start", "parent-run-789") in events
    assert ("start", "demo-run") in events


def test_azure_mlflow_tracker_setup_sweep_creates_parent_run(
    monkeypatch: MonkeyPatch,
) -> None:
    """The Azure MLflow tracker should create one parent run for the sweep."""
    events: list[tuple[str, str]] = []

    def fake_set_tracking_uri(uri: str) -> None:
        events.append(("uri", uri))

    def fake_set_experiment(name: str) -> None:
        events.append(("experiment", name))

    def fake_start_run(run_name: str | None = None):
        events.append(("start", run_name or ""))
        return SimpleNamespace(info=SimpleNamespace(run_id="parent-run-azure"))

    monkeypatch.setattr(
        "dl_azure.trackers.azure_mlflow.mlflow",
        SimpleNamespace(
            set_tracking_uri=fake_set_tracking_uri,
            set_experiment=fake_set_experiment,
            start_run=fake_start_run,
            end_run=lambda: events.append(("end", "parent")),
        ),
    )

    tracker = AzureMlflowTracker({"tracking_uri": "azureml://tracking"})
    tracker_state = tracker.setup_sweep(
        experiment_name="demo-experiment",
        sweep_id="sweep-001",
        sweep_config={"tracking": {}},
        total_runs=2,
    )
    tracker.teardown_sweep()

    assert tracker_state == {
        "tracking_context": "parent-run-azure",
        "tracking_uri": "azureml://tracking",
    }
    assert ("uri", "azureml://tracking") in events
    assert ("experiment", "demo-experiment") in events
    assert ("start", "demo-experiment-sweep-001") in events
    assert ("end", "parent") in events


def test_azure_mlflow_metrics_source_prefers_remote_metrics(
    monkeypatch: MonkeyPatch,
) -> None:
    """The Azure MLflow metrics source should use remote metrics when present."""
    source = METRICS_SOURCE_REGISTRY.get("azure_mlflow")

    monkeypatch.setattr(
        "dl_azure.metrics_sources.azure_mlflow.mlflow",
        SimpleNamespace(
            tracking=SimpleNamespace(
                MlflowClient=lambda tracking_uri: SimpleNamespace(
                    get_run=lambda run_id: SimpleNamespace(
                        info=SimpleNamespace(status="FINISHED"),
                        data=SimpleNamespace(
                            metrics={"validation/accuracy": 0.95},
                            tags={"mlflow.runName": "demo-run"},
                        ),
                    )
                )
            )
        ),
    )

    run_record = source.collect_run(
        run_index=0,
        run_data={
            "tracking_run_id": "azure-job-123",
            "tracking_run_name": "demo-run",
            "tracking_backend": "azure_mlflow",
            "metrics_source_backend": "azure_mlflow",
            "tracking_run_ref": {
                "backend": "azure_mlflow",
                "run_id": "azure-job-123",
                "tracking_uri": "azureml://tracking",
            },
            "status": "running",
            "config_path": str(Path("config.yaml")),
        },
        sweep_data={"tracking_backend": "azure_mlflow"},
    )

    assert run_record["remote_summary_available"] is True
    assert run_record["final_metrics"]["validation/accuracy"] == 0.95
    assert run_record["status"] == "completed"
