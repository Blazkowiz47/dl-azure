"""Tests for Azure tracker and metrics source registration."""

from __future__ import annotations

from types import SimpleNamespace

from pytest import MonkeyPatch

import dl_azure
from dl_azure.callbacks.mlflow import AzureMlflowCallback
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
