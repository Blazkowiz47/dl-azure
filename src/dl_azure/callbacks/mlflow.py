"""Azure MLflow callback for dl-core training loops."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import torch

from dl_core.core.base_callback import Callback
from dl_core.core.registry import register_callback


def _to_json_safe(value: Any) -> Any:
    """Convert nested config values into an MLflow-safe payload."""
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_scalars(logs: dict[str, Any] | None) -> dict[str, float]:
    """Extract scalar metrics from a callback log payload."""
    if not logs:
        return {}

    scalars: dict[str, float] = {}
    for key, value in logs.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            scalars[key] = float(value)
            continue
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            scalars[key] = float(value.item())
            continue
        if hasattr(value, "item") and callable(value.item):
            try:
                scalars[key] = float(value.item())
            except Exception:
                continue
    return scalars


def _flatten_dict(
    value: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten a nested configuration dictionary for MLflow params."""
    items: list[tuple[str, Any]] = []
    for key, item in value.items():
        flat_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(item, dict):
            items.extend(_flatten_dict(item, flat_key, sep=sep).items())
            continue
        items.append((flat_key, _to_json_safe(item)))
    return dict(items)


def _qualify_phase_metrics(
    scalars: dict[str, float],
    phase: str | None,
) -> dict[str, float]:
    """Return phase-qualified scalars for phase hooks."""
    if phase is None:
        return scalars
    return {f"{phase}/{key}": value for key, value in scalars.items()}


@register_callback("azure_mlflow")
class AzureMlflowCallback(Callback):
    """Log training metadata and metrics to Azure MLflow."""

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        parent_run_id: str | None = None,
        run_id_file: str | None = None,
        log_config: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            parent_run_id=parent_run_id,
            run_id_file=run_id_file,
            log_config=log_config,
            **kwargs,
        )
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.parent_run_id = parent_run_id
        self.run_id_file = run_id_file
        self.log_config = log_config
        self.parent_run: Any | None = None
        self.run: Any | None = None

    def _resolve_experiment_name(self) -> str:
        """Return the MLflow experiment name for the current training run."""
        trainer_config = getattr(self.trainer, "config", {})
        tracking = trainer_config.get("tracking", {})
        experiment = trainer_config.get("experiment", {})
        return (
            tracking.get("experiment_name")
            or experiment.get("name")
            or self.experiment_name
            or "default"
        )

    def _resolve_tracking_uri(self) -> str | None:
        """Return the MLflow tracking URI for the current training run."""
        trainer_config = getattr(self.trainer, "config", {})
        tracking = trainer_config.get("tracking", {})
        return self.tracking_uri or tracking.get("tracking_uri") or tracking.get("uri")

    def _resolve_parent_run_id(self) -> str | None:
        """Return the parent MLflow run id when sweep nesting is enabled."""
        trainer_config = getattr(self.trainer, "config", {})
        tracking = trainer_config.get("tracking", {})
        return (
            self.parent_run_id
            or tracking.get("parent_run_id")
            or tracking.get("context")
        )

    def _resolve_existing_run_id(self) -> str | None:
        """Return the Azure-provided MLflow run id when running inside a job."""
        return os.environ.get("MLFLOW_RUN_ID") or os.environ.get("AZUREML_RUN_ID")

    def _resolve_run_name(self) -> str | None:
        """Return the MLflow run name for the current training run."""
        trainer_config = getattr(self.trainer, "config", {})
        runtime = trainer_config.get("runtime", {})
        tracking = trainer_config.get("tracking", {})
        return tracking.get("run_name") or runtime.get("name") or self.run_name

    def _log_params(self) -> None:
        """Log flattened trainer config into MLflow parameters."""
        trainer_config = getattr(self.trainer, "config", {})
        try:
            mlflow.log_params(_flatten_dict(trainer_config))
        except Exception as exc:
            self.logger.warning(f"Failed to log Azure MLflow parameters: {exc}")

    def _log_artifact_if_exists(self, path: Path, artifact_path: str | None) -> None:
        """Upload one artifact if it exists on disk."""
        if not path.exists():
            return
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as exc:
            self.logger.warning(f"Failed to upload artifact {path.name}: {exc}")

    def _log_default_artifacts(self) -> None:
        """Upload the standard dl-core artifact files to Azure MLflow."""
        artifact_manager = getattr(self.trainer, "artifact_manager", None)
        if artifact_manager is None:
            return

        run_dir = Path(artifact_manager.run_dir)
        self._log_artifact_if_exists(run_dir / "config.yaml", None)
        self._log_artifact_if_exists(run_dir / "run_info.json", None)
        self._log_artifact_if_exists(run_dir / "metrics" / "summary.json", "metrics")
        self._log_artifact_if_exists(run_dir / "metrics" / "history.json", "metrics")

    def _write_run_id_file(self) -> None:
        """Persist the active run id for sweep bookkeeping when requested."""
        if not self.run_id_file or self.run is None:
            return
        try:
            Path(self.run_id_file).write_text(
                self.run.info.run_id,
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning(f"Failed to write Azure MLflow run id file: {exc}")

    def _write_tracking_session(self, tracking_uri: str) -> None:
        """Persist Azure MLflow session metadata inside the artifact directory."""
        artifact_manager = getattr(self.trainer, "artifact_manager", None)
        if artifact_manager is None:
            return
        if not hasattr(artifact_manager, "save_tracking_session"):
            return
        if self.run is None:
            return

        parent_run_id = self._resolve_parent_run_id()
        session_data = {
            "backend": "azure_mlflow",
            "run_id": self.run.info.run_id,
            "run_name": self._resolve_run_name(),
            "tracking_uri": tracking_uri,
            "experiment_name": self._resolve_experiment_name(),
            "parent_run_id": parent_run_id,
        }
        artifact_manager.save_tracking_session(session_data)

    def on_training_start(self, logs: dict[str, Any] | None = None) -> None:
        """Initialize the Azure MLflow run at the beginning of training."""
        super().on_training_start(logs)
        if not self.is_main_process():
            return
        if self.run is not None:
            return

        tracking_uri = self._resolve_tracking_uri()
        if not tracking_uri:
            raise RuntimeError("Azure MLflow callback requires tracking.uri")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self._resolve_experiment_name())

        existing_run_id = self._resolve_existing_run_id()
        parent_run_id = self._resolve_parent_run_id()
        if existing_run_id:
            self.run = mlflow.start_run(run_id=existing_run_id)
        elif parent_run_id:
            self.parent_run = mlflow.start_run(run_id=parent_run_id)
            self.run = mlflow.start_run(
                run_name=self._resolve_run_name(),
                nested=True,
            )
        else:
            self.run = mlflow.start_run(run_name=self._resolve_run_name())

        self._write_run_id_file()
        self._write_tracking_session(tracking_uri)
        if self.log_config:
            self._log_params()
        self._log_default_artifacts()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Log non-phase epoch metrics to Azure MLflow."""
        super().on_epoch_end(epoch, logs)
        self._log_metrics(epoch, logs, phase=None)

    def on_train_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Log train metrics to Azure MLflow."""
        super().on_train_end(epoch, logs)
        self._log_metrics(epoch, logs, phase="train")

    def on_validation_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """Log validation metrics to Azure MLflow."""
        super().on_validation_end(epoch, logs)
        self._log_metrics(epoch, logs, phase="validation")

    def on_test_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Log test metrics to Azure MLflow."""
        super().on_test_end(epoch, logs)
        self._log_metrics(epoch, logs, phase="test")

    def _log_metrics(
        self,
        epoch: int,
        logs: dict[str, Any] | None,
        *,
        phase: str | None = None,
    ) -> None:
        """Log scalar metrics to Azure MLflow with consistent step numbering."""
        if not self.is_main_process():
            return
        if self.run is None:
            return

        scalars = _extract_scalars(logs)
        scalars = _qualify_phase_metrics(scalars, phase)
        if phase is None:
            scalars = {
                key: value
                for key, value in scalars.items()
                if not key.startswith(("train/", "validation/", "test/"))
            }
        if not scalars:
            return

        try:
            mlflow.log_metrics(scalars, step=epoch)
        except Exception as exc:
            self.logger.warning(f"Failed to log Azure MLflow metrics: {exc}")

    def on_training_end(self, logs: dict[str, Any] | None = None) -> None:
        """Finalize the active Azure MLflow run at the end of training."""
        super().on_training_end(logs)
        if not self.is_main_process():
            return
        if self.run is None:
            return

        self._log_default_artifacts()
        try:
            mlflow.end_run()
        finally:
            self.run = None
            if self.parent_run is not None:
                try:
                    mlflow.end_run()
                finally:
                    self.parent_run = None
