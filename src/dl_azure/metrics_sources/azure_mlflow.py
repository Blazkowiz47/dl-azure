"""Azure MLflow metrics source with remote fetch and local artifact fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from dl_core.core import register_metrics_source
from dl_core.metrics_sources.local import LocalMetricsSource, _normalize_metric_key


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

        config_path = local_record.get("config_path")
        resolved_config_path = Path(config_path) if isinstance(config_path, str) else None
        selection_metric, selection_mode = self._resolve_selection_config(
            resolved_config_path
        )
        if not local_record.get("selection_metric"):
            local_record["selection_metric"] = selection_metric
        if not local_record.get("selection_mode"):
            local_record["selection_mode"] = selection_mode

        remote_final = dict(run.data.metrics)
        merged_final = dict(remote_final)
        merged_final.update(local_record.get("final_metrics", {}))
        local_record["final_metrics"] = merged_final

        remote_selection_metric = self._resolve_remote_metric_name(
            remote_final,
            local_record.get("selection_metric"),
        )
        remote_history = self._fetch_metric_history(
            client,
            run_id,
            remote_selection_metric,
        )

        best_epoch, best_value = self._resolve_best_epoch(
            remote_history,
            local_record.get("selection_mode"),
        )
        if best_epoch is not None:
            local_record["best_epoch"] = best_epoch

        if best_value is not None:
            local_record["selection_value"] = best_value
        elif not isinstance(local_record.get("selection_value"), (int, float)):
            selection_value = self._resolve_remote_metric(
                remote_final,
                local_record.get("selection_metric"),
            )
            local_record["selection_value"] = selection_value

        if best_epoch is not None:
            best_metrics = self._collect_best_metrics(
                client,
                run_id,
                remote_final,
                best_epoch,
            )
            merged_best = dict(best_metrics)
            merged_best.update(local_record.get("best_metrics", {}))
            local_record["best_metrics"] = merged_best

        local_record["tracking_run_ref"] = tracking_ref
        local_record["remote_summary_available"] = True
        local_record["run_name"] = (
            local_record.get("run_name")
            or tracking_ref.get("run_name")
            or run.data.tags.get("mlflow.runName")
            or local_record["run_name"]
        )

        if local_record.get("status") in {"unknown", "running"}:
            local_record["status"] = self._map_run_status(run.info.status)

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
        selection_metric: Any,
    ) -> Any:
        """Resolve one metric value from a remote Azure MLflow metric mapping."""
        if not isinstance(selection_metric, str) or not selection_metric:
            return None
        if selection_metric in metrics:
            return metrics[selection_metric]

        normalized_selection = _normalize_metric_key(selection_metric)
        for metric_name, metric_value in metrics.items():
            normalized_metric = _normalize_metric_key(metric_name)
            if normalized_metric == normalized_selection:
                return metric_value
        return None

    def _resolve_selection_config(
        self,
        config_path: Path | None,
    ) -> tuple[str | None, str | None]:
        """Resolve the ranking metric and mode from one local run config."""
        if config_path is None or not config_path.exists():
            return None, None

        config = self.load_yaml(config_path)
        callbacks_config = config.get("callbacks", {})
        if not isinstance(callbacks_config, dict):
            return None, None

        checkpoint_config = callbacks_config.get("checkpoint")
        if not isinstance(checkpoint_config, dict):
            return None, None

        monitor = checkpoint_config.get("monitor")
        mode = checkpoint_config.get("mode", "min")
        if not isinstance(monitor, str) or not monitor:
            return None, None
        if mode not in {"min", "max"}:
            mode = "min"
        return monitor, mode

    def _resolve_remote_metric_name(
        self,
        metrics: dict[str, Any],
        selection_metric: Any,
    ) -> str | None:
        """Resolve the concrete remote metric key that matches one selection key."""
        if not isinstance(selection_metric, str) or not selection_metric:
            return None
        if selection_metric in metrics:
            return selection_metric

        normalized_selection = _normalize_metric_key(selection_metric)
        for metric_name in metrics:
            if _normalize_metric_key(metric_name) == normalized_selection:
                return metric_name
        return selection_metric

    def _fetch_metric_history(
        self,
        client: mlflow.tracking.MlflowClient,
        run_id: str,
        metric_name: str | None,
    ) -> list[dict[str, int | float]]:
        """Fetch one remote MLflow metric history in normalized form."""
        if not metric_name:
            return []

        try:
            history = client.get_metric_history(run_id, metric_name)
        except Exception:
            return []

        step_to_value: dict[int, float] = {}
        for point in history:
            step = getattr(point, "step", None)
            value = getattr(point, "value", None)
            if not isinstance(step, int):
                continue
            if not isinstance(value, (int, float)):
                continue
            step_to_value[step] = float(value)

        return [
            {"step": step, "value": step_to_value[step]}
            for step in sorted(step_to_value)
        ]

    def _resolve_best_epoch(
        self,
        history: list[dict[str, int | float]],
        selection_mode: Any,
    ) -> tuple[int | None, float | None]:
        """Resolve best epoch and metric value from one normalized history."""
        if not history:
            return None, None

        if selection_mode not in {"min", "max"}:
            last_point = history[-1]
            return int(last_point["step"]), float(last_point["value"])

        best_epoch: int | None = None
        best_value: float | None = None
        for point in history:
            epoch = int(point["step"])
            metric_value = float(point["value"])
            if best_value is None:
                best_epoch = epoch
                best_value = metric_value
                continue

            is_better = (
                metric_value < best_value
                if selection_mode == "min"
                else metric_value > best_value
            )
            if is_better:
                best_epoch = epoch
                best_value = metric_value

        return best_epoch, best_value

    def _collect_best_metrics(
        self,
        client: mlflow.tracking.MlflowClient,
        run_id: str,
        remote_final: dict[str, Any],
        best_epoch: int,
    ) -> dict[str, float]:
        """Collect remote metric values recorded at the selected best epoch."""
        best_metrics: dict[str, float] = {}
        for metric_name in remote_final:
            history = self._fetch_metric_history(client, run_id, metric_name)
            value_at_epoch = next(
                (
                    float(point["value"])
                    for point in history
                    if int(point["step"]) == best_epoch
                ),
                None,
            )
            if value_at_epoch is not None:
                best_metrics[metric_name] = value_at_epoch

        return best_metrics
