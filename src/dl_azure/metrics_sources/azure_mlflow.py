"""Azure MLflow metrics source with remote fetch and local artifact fallback."""

from __future__ import annotations

from typing import Any

import mlflow

from dl_core.core import register_metrics_source
from dl_core.metrics_sources.local import LocalMetricsSource


@register_metrics_source("azure_mlflow")
class AzureMlflowMetricsSource(LocalMetricsSource):
    """Read Azure MLflow-backed sweep results with local artifact fallback."""

    def collect_run(
        self,
        run_index: int,
        run_data: dict[str, Any],
        sweep_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Collect one analyzer record, preferring remote Azure MLflow metrics."""
        local_record = super().collect_run(run_index, run_data, sweep_data)
        tracking_ref = run_data.get("tracking_run_ref") or {}
        if not isinstance(tracking_ref, dict):
            return local_record

        run_id = tracking_ref.get("run_id") or run_data.get("tracking_run_id")
        tracking_uri = (
            tracking_ref.get("tracking_uri")
            or run_data.get("tracking_uri")
            or sweep_data.get("tracking_uri")
        )
        if not isinstance(run_id, str) or not run_id:
            return local_record
        if not isinstance(tracking_uri, str) or not tracking_uri:
            return local_record

        try:
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(run_id)
        except Exception as exc:
            local_record["metrics_source_warning"] = str(exc)
            return local_record

        remote_final = dict(run.data.metrics)
        merged_final = dict(remote_final)
        merged_final.update(local_record.get("final_metrics", {}))

        local_record["tracking_run_ref"] = tracking_ref
        local_record["remote_summary_available"] = True
        local_record["final_metrics"] = merged_final
        local_record["run_name"] = (
            local_record.get("run_name")
            or tracking_ref.get("run_name")
            or run.data.tags.get("mlflow.runName")
            or local_record["run_name"]
        )

        if local_record.get("status") in {"unknown", "running"}:
            local_record["status"] = self._map_run_status(run.info.status)

        selection_metric = local_record.get("selection_metric")
        if (
            not isinstance(local_record.get("selection_value"), (int, float))
            and isinstance(selection_metric, str)
            and selection_metric
        ):
            local_record["selection_value"] = self._resolve_remote_metric(
                remote_final,
                selection_metric,
            )

        return local_record

    def _map_run_status(self, status: str | None) -> str:
        """Map MLflow run statuses into analyzer statuses."""
        if status == "FINISHED":
            return "completed"
        if status in {"FAILED", "KILLED"}:
            return "failed"
        return "running"

    def _resolve_remote_metric(
        self,
        metrics: dict[str, Any],
        selection_metric: str,
    ) -> Any:
        """Resolve one metric value from a remote Azure MLflow metric mapping."""
        if selection_metric in metrics:
            return metrics[selection_metric]

        normalized_selection = "".join(
            char for char in selection_metric.casefold() if char.isalnum()
        )
        for metric_name, metric_value in metrics.items():
            normalized_metric = "".join(
                char for char in metric_name.casefold() if char.isalnum()
            )
            if normalized_metric == normalized_selection:
                return metric_value
        return None
