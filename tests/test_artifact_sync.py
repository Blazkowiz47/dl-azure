"""Tests for Azure artifact sync helpers."""

from __future__ import annotations

import json
from pathlib import Path

from dl_azure.metrics_sources.azure_mlflow import AzureMlflowMetricsSource


class _MockJobs:
    """Minimal Azure job operations stub."""

    def __init__(self, source_dir: Path) -> None:
        self.source_dir = source_dir

    def download(
        self,
        name: str,
        *,
        download_path: str = ".",
        output_name: str | None = None,
        all: bool = False,
    ) -> None:
        del name, output_name, all
        target = Path(download_path) / "bundle" / "nested" / "run_artifacts"
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.yaml").write_text("{}", encoding="utf-8")
        final_metrics = target / "final" / "metrics"
        final_metrics.mkdir(parents=True, exist_ok=True)
        (final_metrics / "summary.json").write_text("{}", encoding="utf-8")
        (final_metrics / "history.json").write_text("{}", encoding="utf-8")


class _MockMlClient:
    """Minimal ML client stub with job downloads."""

    def __init__(self) -> None:
        self.jobs = _MockJobs(Path("."))


def test_azure_sync_downloads_into_expected_artifact_dir(tmp_path: Path) -> None:
    """Azure sync should copy the downloaded run bundle into the local run dir."""
    source = AzureMlflowMetricsSource()
    sweep_path = tmp_path / "experiments" / "demo.yaml"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text("base_config: configs/base.yaml\n", encoding="utf-8")
    (tmp_path / "azure-config.json").write_text(
        json.dumps(
            {
                "subscription_id": "sub",
                "resource_group": "rg",
                "workspace_name": "ws",
            }
        ),
        encoding="utf-8",
    )
    tracking_dir = sweep_path.parent / sweep_path.stem
    tracking_dir.mkdir(parents=True, exist_ok=True)
    config_path = tracking_dir / "run_0.yaml"
    config_path.write_text(
        "runtime:\n"
        "  output_dir: outputs/artifacts\n"
        "  name: demo_run\n"
        "sweep_file: experiments/demo.yaml\n",
        encoding="utf-8",
    )

    sweep_data = {
        "_sweep_path": str(sweep_path),
        "_tracking_dir": str(tracking_dir),
    }
    run_data = {
        "tracking_run_id": "azure-job-1",
        "tracking_run_ref": {"run_id": "azure-job-1"},
        "config_path": str(config_path),
    }

    source._get_sync_ml_client = lambda sweep_data: _MockMlClient()  # type: ignore[method-assign]
    synced = source.sync_run_artifacts(0, run_data, sweep_data)

    artifact_dir = Path(str(synced["artifact_dir"]))
    assert artifact_dir.exists()
    assert (artifact_dir / "config.yaml").exists()
    assert (artifact_dir / "final" / "metrics" / "summary.json").exists()
    assert (artifact_dir / "final" / "metrics" / "history.json").exists()
