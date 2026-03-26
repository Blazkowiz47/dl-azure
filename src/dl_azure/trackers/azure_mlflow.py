"""Azure MLflow tracker implementation for dl-core sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from azureml.core import Workspace

from dl_core.core import BaseTracker, register_tracker


@register_tracker("azure_mlflow")
class AzureMlflowTracker(BaseTracker):
    """Tracker metadata adapter for Azure MLflow-backed runs."""

    def __init__(self, tracking_config: dict[str, Any] | None = None, **kwargs: Any):
        """Initialize tracker state for Azure MLflow-backed sweeps."""
        super().__init__(tracking_config, **kwargs)
        self.parent_run: Any | None = None

    def get_backend_name(self) -> str:
        """Return the tracker backend name."""
        return "azure_mlflow"

    def _resolve_tracking_uri(self, tracking_uri: str | None = None) -> str | None:
        """Resolve the Azure MLflow tracking URI for this sweep."""
        resolved_tracking_uri = (
            tracking_uri
            or self.tracking_config.get("tracking_uri")
            or self.tracking_config.get("uri")
        )
        if isinstance(resolved_tracking_uri, str) and resolved_tracking_uri:
            return resolved_tracking_uri

        azure_config_path = Path(
            str(self.tracking_config.get("azure_config_path", "azure-config.json"))
        ).expanduser()
        if not azure_config_path.exists():
            return None

        workspace = Workspace.from_config(path=str(azure_config_path))
        return workspace.get_mlflow_tracking_uri()

    def setup_sweep(
        self,
        *,
        experiment_name: str,
        sweep_id: str,
        sweep_config: dict[str, Any],
        total_runs: int,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Reuse or create the Azure MLflow parent context for the sweep."""
        del total_runs

        resolved_tracking_uri = self._resolve_tracking_uri(tracking_uri)
        if resume and tracking_context:
            return {
                "tracking_context": tracking_context,
                "tracking_uri": resolved_tracking_uri,
            }

        if tracking_context:
            return {
                "tracking_context": tracking_context,
                "tracking_uri": resolved_tracking_uri,
            }

        if not resolved_tracking_uri:
            return {
                "tracking_context": tracking_context,
                "tracking_uri": tracking_uri,
            }

        sweep_file = sweep_config.get("sweep_file")
        sweep_name = Path(str(sweep_file)).stem if sweep_file else ""
        group_name = (
            self.tracking_config.get("group")
            or sweep_name
            or f"{experiment_name}-{sweep_id}"
        )

        mlflow.set_tracking_uri(resolved_tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.parent_run = mlflow.start_run(run_name=str(group_name))
        return {
            "tracking_context": self.parent_run.info.run_id,
            "tracking_uri": resolved_tracking_uri,
        }

    def teardown_sweep(self) -> None:
        """Close the locally-created Azure MLflow parent run when needed."""
        if self.parent_run is None:
            return
        try:
            mlflow.end_run()
        finally:
            self.parent_run = None

    def inject_tracking_config(
        self,
        config: dict[str, Any],
        *,
        run_name: str | None = None,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """Inject Azure MLflow tracking metadata into a run configuration."""
        super().inject_tracking_config(
            config,
            run_name=run_name,
            tracking_context=tracking_context,
            tracking_uri=tracking_uri,
        )
        tracking = config.setdefault("tracking", {})
        if tracking_context:
            tracking["parent_run_id"] = tracking_context
        if tracking_uri:
            tracking["tracking_uri"] = tracking_uri

    def build_run_reference(
        self,
        *,
        result: dict[str, Any] | None = None,
        run_name: str | None = None,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
    ) -> dict[str, Any] | None:
        """Build an Azure MLflow-specific run reference for sweep tracking."""
        reference = super().build_run_reference(
            result=result,
            run_name=run_name,
            tracking_context=tracking_context,
            tracking_uri=tracking_uri,
        )
        if reference is None:
            return None

        reference["backend"] = "azure_mlflow"
        if tracking_context:
            reference.setdefault("parent_run_id", tracking_context)
        if tracking_uri:
            reference.setdefault("tracking_uri", tracking_uri)

        execution_run_id = (
            result.get("tracking_run_id") if isinstance(result, dict) else None
        )
        if isinstance(execution_run_id, str) and execution_run_id:
            reference.setdefault("azure_job_name", execution_run_id)
            reference.setdefault("run_id", execution_run_id)
        return reference
