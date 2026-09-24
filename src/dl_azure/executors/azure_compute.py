"""Azure ML Compute executor for distributed sweeps."""

import fcntl
import json
import os
import re
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    generate_account_sas,
    ResourceTypes,
    AccountSasPermissions,
)

from dl_core.core import BaseExecutor, config_field, register_executor

_AMLIGNORE_BEGIN = "# BEGIN dl-azure managed block"
_AMLIGNORE_END = "# END dl-azure managed block"
_COMMAND_PLACEHOLDER_PATTERN = re.compile(
    r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})"
)


@register_executor("azure")
class AzureComputeExecutor(BaseExecutor):
    """
    Azure ML executor.

    - Submits each run as independent Azure ML job
    - No parent process context
    - MLflow logs to Azure workspace
    - Jobs run on compute cluster
    """

    CONFIG_FIELDS = [
        config_field(
            "compute_target",
            "str",
            "Azure ML compute cluster name used for submitted jobs.",
            required=True,
        ),
        config_field(
            "environment_name",
            "str",
            "Azure ML environment name used when submitting jobs.",
            default="dl_lab",
        ),
        config_field(
            "environment_version",
            "str",
            "Azure ML environment version to resolve.",
            default="latest",
        ),
        config_field(
            "datastore_name",
            "str | None",
            "Optional datastore mounted into each Azure job.",
            default=None,
        ),
        config_field(
            "process_count_per_node",
            "int",
            "Number of distributed worker processes to launch per node.",
            default=1,
        ),
        config_field(
            "dont_wait_for_completion",
            "bool",
            "Submit jobs asynchronously instead of blocking until completion.",
            default=False,
        ),
        config_field(
            "retry_limit",
            "int",
            "Maximum number of retry submissions for failed runs.",
            default=0,
        ),
        config_field(
            "azure_config_path",
            "str",
            "Path to the Azure workspace config JSON file.",
            default="azure-config.json",
        ),
        config_field(
            "command",
            "str | None",
            "Optional Azure job command override. Supports placeholders such as "
            "{config_path}, {run_name}, {run_index}, {run_number}, "
            "{tracking_context}, and {tracking_uri}.",
            default=None,
        ),
        config_field(
            "parent_job_name",
            "str | None",
            "Existing Azure ML parent job name to nest child jobs under. "
            "Takes precedence over resume-derived tracking context.",
            default=None,
        ),
    ]

    def __init__(
        self,
        sweep_config: Dict[str, Any],
        experiment_name: str,
        sweep_id: str,
        dry_run: bool = False,
        tracking_context: Optional[str] = None,
        resume: bool = False,
        **kwargs,
    ):
        super().__init__(
            sweep_config,
            experiment_name,
            sweep_id,
            dry_run=dry_run,
            tracking_context=tracking_context,
            resume=resume,
        )
        self.compute_target = kwargs.get("compute_target") or self.executor_config.get(
            "compute_target"
        )
        if not self.compute_target:
            raise ValueError("executor.compute_target is required for Azure sweeps")

        # Get environment name and version from executor config
        self.environment_name = kwargs.get("environment_name") or self.executor_config.get(
            "environment_name", "dl_lab"
        )
        self.environment_version = self.executor_config.get(
            "environment_version", "latest"
        )

        # Get datastore name from executor config (nullable - if not present, don't mount)
        self.datastore_name = self.executor_config.get("datastore_name")

        # Get process count per node for multi-GPU training
        self.process_count_per_node = self.executor_config.get(
            "process_count_per_node", 1
        )

        # Get dont_wait_for_completion flag (default: False - wait for each job by default)
        self.dont_wait_for_completion = self.executor_config.get(
            "dont_wait_for_completion", False
        )

        # Get retry_limit (default: 0 - no retries)
        self.retry_limit = self.executor_config.get("retry_limit", 0)
        self.azure_config_path = Path(
            self.executor_config.get("azure_config_path", "azure-config.json")
        ).expanduser()

        self.ml_client: MLClient
        self.configured_parent_job_name = self._resolve_configured_parent_job_name()
        self.parent_job_name = None  # Azure ML job name for parent
        self.tracking_uri = None
        self.env_vars: Dict[str, str] = {}  # Environment variables for jobs
        self.azure_config: Dict[str, Any] = {}  # Azure config (loaded in setup)
        self.retry_attempts: Dict[int, int] = {}  # Track retry attempts per run index

    def _resolve_configured_parent_job_name(self) -> Optional[str]:
        """Return the explicit Azure parent job configured for this executor."""
        parent_job_name = self.executor_config.get("parent_job_name")
        if parent_job_name is None:
            return None
        if not isinstance(parent_job_name, str):
            raise TypeError("executor.parent_job_name must be a string when provided.")

        parent_job_name = parent_job_name.strip()
        return parent_job_name or None

    def _use_existing_parent_job(self) -> tuple[bool, str]:
        """
        Reuse an existing Azure ML parent job when configured or resuming.

        Returns:
            Tuple of ``(used_existing_parent, source)``.
        """
        if self.configured_parent_job_name:
            self.parent_job_name = self.configured_parent_job_name
            self.tracking_context = self.parent_job_name
            return True, "configured"

        if self.resume and self.tracking_context:
            self.parent_job_name = self.tracking_context
            return True, "resume"

        return False, ""

    def _resolve_custom_command(
        self,
        config_path: Path,
        *,
        run_index: int,
        run_name: str,
    ) -> Optional[str]:
        """Resolve an optional custom Azure submission command."""
        command_template = self.executor_config.get("command")
        if command_template is None:
            return None
        if not isinstance(command_template, str):
            raise TypeError("executor.command must be a string when provided.")

        resolved = command_template.strip()
        if not resolved:
            return None

        placeholders = {
            "config_path": str(config_path),
            "run_name": run_name,
            "run_index": str(run_index),
            "run_number": str(run_index + 1),
            "tracking_context": self.tracking_context or "",
            "tracking_uri": self.tracking_uri or "",
        }
        for name, value in placeholders.items():
            resolved = resolved.replace(f"{{{name}}}", value)

        unsupported = _COMMAND_PLACEHOLDER_PATTERN.search(resolved)
        if unsupported is not None:
            supported = ", ".join(
                f"{{{name}}}"
                for name in (
                    "config_path",
                    "run_name",
                    "run_index",
                    "run_number",
                    "tracking_context",
                    "tracking_uri",
                )
            )
            raise ValueError(
                "Unsupported executor.command placeholder "
                f"{{{unsupported.group(1)}}}. Supported placeholders: {supported}."
            )

        return resolved

    def _resolve_submission_command(
        self,
        config_path: Path,
        run_config: Dict[str, Any],
        *,
        run_index: int,
        run_name: str,
    ) -> str:
        """Resolve the concrete Azure ML command string for one run."""
        custom_command = self._resolve_custom_command(
            config_path,
            run_index=run_index,
            run_name=run_name,
        )
        if custom_command is not None:
            return custom_command

        cmd_list = self.build_command(str(config_path), run_config)
        return " ".join(cmd_list)

    def _build_datastore_uri(self, datastore_name: str, path: str = "") -> str:
        """
        Build full Azure ML datastore URI.

        Format: azureml://subscriptions/{sub}/resourcegroups/{rg}/workspaces/{ws}/datastores/{ds}/paths/{path}

        Args:
            datastore_name: Name of the datastore
            path: Path within the datastore (default: "data/")

        Returns:
            Full Azure ML datastore URI
        """
        subscription_id = self.azure_config["subscription_id"]
        resource_group = self.azure_config["resource_group"]
        workspace_name = self.azure_config["workspace_name"]

        uri = (
            f"azureml://subscriptions/{subscription_id}/"
            f"resourcegroups/{resource_group}/"
            f"workspaces/{workspace_name}/"
            f"datastores/{datastore_name}/"
            f"paths/{path}"
        )

        return uri

    def generate_sas_token(self, expiry_hours: int = 168) -> Optional[str]:
        """
        Generate a time-limited SAS token for Azure Storage using config from azure-config.json.

        Args:
            expiry_hours: Number of hours until token expires (default: 168)

        Returns:
            SAS token string or None if generation fails
        """
        # Get storage account name from config
        storage_account_name = self.azure_config.get("account_name")
        if not storage_account_name:
            self.logger.error(
                "Storage account name not found in azure-config.json. "
                "Add 'account_name' field to enable SAS token generation."
            )
            return None

        # Get access key from environment
        access_key = os.environ.get("AZURE_ACCESS_KEY")
        if not access_key:
            self.logger.error(
                "AZURE_ACCESS_KEY not set in environment. "
                "Set this to enable SAS token generation."
            )
            return None

        try:
            # Calculate expiry time
            start_time = datetime.utcnow()
            expiry_time = start_time + timedelta(hours=expiry_hours)

            # Generate account-level SAS token with read-only permissions
            sas_token = generate_account_sas(
                account_name=storage_account_name,
                account_key=access_key,
                resource_types=ResourceTypes(service=True, container=True, object=True),
                permission=AccountSasPermissions(read=True, list=True),
                start=start_time,
                expiry=expiry_time,
            )

            self.logger.info(
                f"Generated SAS token for storage account '{storage_account_name}' "
                f"(read-only, valid for {expiry_hours} hours, expires: {expiry_time.isoformat()})"
            )
            return sas_token

        except Exception as exc:
            self.logger.error(
                "Failed to generate SAS token: %s", type(exc).__name__
            )
            return None

    def get_job_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for Azure ML jobs.

        Generates a time-limited SAS token for secure blob storage access using
        credentials from azure-config.json and environment variables.

        Returns:
            Dictionary of environment variables for job submission
        """
        env_vars = {
            "PYTHONUNBUFFERED": "1",
        }

        # Get storage account name from config
        storage_account_name = self.azure_config.get("account_name")

        # Create a read-only token locally; never send the account key to jobs.
        if storage_account_name and os.environ.get("AZURE_ACCESS_KEY"):
            # Match the default streaming-shard SAS lifetime.
            sas_token = self.generate_sas_token()

            if sas_token:
                # Pass SAS token instead of access key (more secure)
                env_vars["AZURE_STORAGE_ACCOUNT"] = storage_account_name
                env_vars["AZURE_SAS_TOKEN"] = sas_token
                self.logger.info(
                    f"Configured time-limited SAS token for storage account '{storage_account_name}'"
                )
                self.logger.info(
                    "Jobs will use SAS token for blob storage access "
                    "(read-only, expires in 168 hours)"
                )
            else:
                raise RuntimeError(
                    "SAS token generation failed; refusing to send the storage "
                    "account key to an Azure ML job"
                )
        elif os.environ.get("AZURE_ACCESS_KEY"):
            raise ValueError(
                "azure-config.json needs account_name to issue a job SAS "
                "token from AZURE_ACCESS_KEY"
            )
        else:
            self.logger.info(
                "No storage key configured; child jobs will use Azure "
                "managed identity for blob access"
            )

        return env_vars

    def setup(self, total_runs: int) -> None:
        """Setup Azure ML client."""
        if not all(
            callable(getattr(BaseExecutor, name, None))
            for name in ("_classify_run_result", "_after_run_execution")
        ):
            raise RuntimeError(
                "This dl-azure checkout requires matching dl-core sweep hooks. "
                "Install the corresponding dl-core source revision before "
                "submitting Azure jobs."
            )

        # Determine sweep name and parent job name upfront
        sweep_file = self.sweep_config.get("sweep_file", "")
        if sweep_file:
            sweep_name = Path(sweep_file).stem
        else:
            sweep_name = self.sweep_id[:8]

        try:
            # Load Azure config (even in dry-run, for building URIs and showing what would happen)
            if not self.azure_config_path.exists():
                if self.dry_run:
                    # In dry-run without config file, use dummy values
                    self.logger.warning(
                        "[DRY RUN] azure-config.json not found, using dummy values"
                    )
                    self.azure_config = {
                        "subscription_id": "00000000-0000-0000-0000-000000000000",
                        "resource_group": "dry-run-rg",
                        "workspace_name": "dry-run-workspace",
                    }
                else:
                    raise RuntimeError(
                        "Azure config not found. Create azure-config.json with "
                        "subscription_id, resource_group, and workspace_name"
                    )
            else:
                with open(self.azure_config_path, "r", encoding="utf-8") as f:
                    self.azure_config = json.load(f)

            if self.dry_run:
                self.logger.info("[DRY RUN] Would setup Azure ML executor:")
                self.logger.info(
                    f"[DRY RUN]   Workspace: {self.azure_config.get('workspace_name', '<unknown>')}"
                )
                self.logger.info(f"[DRY RUN]   Compute: {self.compute_target}")
                self.logger.info(
                    f"[DRY RUN]   Environment: {self.environment_name}@{self.environment_version}"
                )
                self.logger.info(f"[DRY RUN]   Total jobs: {total_runs}")
                if self.datastore_name:
                    # Build datastore URI to show what would be used
                    datastore_uri = self._build_datastore_uri(self.datastore_name)
                    self.logger.info(
                        f"[DRY RUN]   Datastore: {self.datastore_name} ({datastore_uri})"
                    )
                used_parent_job, parent_source = self._use_existing_parent_job()
                if used_parent_job:
                    source_text = (
                        "configured parent job"
                        if parent_source == "configured"
                        else "existing tracking context"
                    )
                    self.logger.info(
                        f"[DRY RUN]   Would use {source_text}: "
                        f"{self.parent_job_name}"
                    )
                else:
                    self.logger.info(
                        f"[DRY RUN]   Would create parent job: {sweep_name}"
                    )
                    self.parent_job_name = "dry_run_parent_job"
                    self.tracking_context = self.parent_job_name
                # Set dummy values
                self.tracking_uri = "dry_run_tracking_uri"
                self.env_vars = {"PYTHONUNBUFFERED": "1"}
                return

            self.ml_client = MLClient(
                credential=DefaultAzureCredential(),
                subscription_id=self.azure_config["subscription_id"],
                resource_group_name=self.azure_config["resource_group"],
                workspace_name=self.azure_config["workspace_name"],
            )

            self.logger.info(
                f"Connected to Azure ML workspace: {self.azure_config['workspace_name']}"
            )
            self.logger.info(f"Compute: {self.compute_target}")
            self.logger.info(
                f"Environment: {self.environment_name}@{self.environment_version}"
            )
            self.logger.info(f"Total jobs: {total_runs}")

            # Log execution mode
            if self.dont_wait_for_completion:
                self.logger.info("Execution mode: Parallel (submit all jobs at once)")
            else:
                self.logger.info(
                    "Execution mode: Sequential (wait for each job to complete)"
            )

            # Setup Azure MLflow tracking URI from the v2 workspace entity.
            workspace = self.ml_client.workspaces.get(
                self.azure_config["workspace_name"]
            )
            self.tracking_uri = (
                workspace.mlflow_tracking_uri if workspace is not None else None
            )

            # Get environment variables for jobs (includes SAS token generation if available)
            self.env_vars = self.get_job_environment_variables()

            # Log datastore configuration (datastore mounting happens per-job in execute_run)
            if self.datastore_name:
                datastore_uri = self._build_datastore_uri(self.datastore_name)
                self.logger.info(f"Configured datastore: {self.datastore_name}")
                self.logger.info(f"Datastore URI: {datastore_uri}")
            else:
                self.logger.info(
                    "No datastore configured - jobs will not have datastore input mounted"
                )

            # Update .amlignore once for the entire sweep
            sweep_file_path = self.sweep_config.get("sweep_file", "")
            if sweep_file_path:
                self.update_amlignore(sweep_file_path)
                self.logger.info("Updated .amlignore managed block for Azure sweep")

            used_parent_job, parent_source = self._use_existing_parent_job()
            if used_parent_job:
                source_text = (
                    "configured parent job"
                    if parent_source == "configured"
                    else "existing parent job"
                )
                self.logger.info(
                    f"Using {source_text}: {self.parent_job_name}"
                )
                return

            # Submit parent job (simple placeholder that child jobs will nest under)
            # Parent job completes immediately but child jobs remain nested
            parent_job = command(
                code=".",
                command=f"echo 'Sweep parent job - {total_runs} child jobs will nest under this'",
                environment=f"{self.environment_name}@{self.environment_version}",
                compute=self.compute_target,
                experiment_name=self.experiment_name,
                display_name=sweep_name,
                description=f"Parent job for sweep {self.sweep_id} with {total_runs} runs",
            )

            # Submit parent job
            submitted_parent = self.ml_client.jobs.create_or_update(parent_job)
            self.parent_job_name = submitted_parent.name
            self.tracking_context = self.parent_job_name

            self.logger.info(f"Created parent job: {self.parent_job_name}")
            self.logger.info(f"Display name: {sweep_name}")

        except ImportError as exc:
            raise RuntimeError(
                "Azure mode requires azure-ai-ml. "
                "Install: pip install azure-ai-ml azure-identity"
            ) from exc

    def update_amlignore(self, sweep_file: str) -> None:
        """
        Update the managed Azure block in ``.amlignore``.

        Args:
            sweep_file: Path to the sweep file. Used for logging only.
        """
        del sweep_file
        amlignore_path = Path(".amlignore")
        ignore_content = self._render_amlignore_block()

        # Write .amlignore with file locking to prevent race conditions
        # when multiple threads/processes try to update simultaneously
        # Open in 'a+' mode first to avoid truncating before lock acquisition
        with open(amlignore_path, "a+", encoding="utf-8") as f:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                existing_content = f.read()
                updated_content = self._upsert_amlignore_block(
                    existing_content,
                    ignore_content,
                )
                f.seek(0)
                f.truncate()
                f.write(updated_content)
            finally:
                # Release lock (also auto-released when file closes)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self.logger.info("Updated the managed dl-azure block in .amlignore")

    def _render_amlignore_block(self) -> str:
        """Render the managed `.amlignore` content for Azure submissions."""
        patterns = [
            ".git/",
            ".github/",
            ".vscode/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            ".coverage",
            "htmlcov/",
            ".ruff_cache/",
            ".mypy_cache/",
            ".venv/",
            ".env",
            ".env.*",
            ".tox/",
            "dist/",
            "build/",
            "outputs/",
            "artifacts/",
            "scores/",
            "mlruns/",
            "wandb/",
            ".cache/",
            "*.pt",
            "*.pth",
            "*.onnx",
            "*.safetensors",
            "*.bin",
            "*.ckpt",
            "*.log",
            "*.db",
        ]
        return "\n".join(
            [
                _AMLIGNORE_BEGIN,
                "# Managed by dl-azure. Keep this block if Azure submission is used.",
                *patterns,
                _AMLIGNORE_END,
            ]
        )

    def _upsert_amlignore_block(
        self,
        existing_content: str,
        managed_block: str,
    ) -> str:
        """Insert or replace the managed Azure `.amlignore` block."""
        if not existing_content.strip():
            return f"{managed_block}\n"

        begin_index = existing_content.find(_AMLIGNORE_BEGIN)
        end_index = existing_content.find(_AMLIGNORE_END)
        if begin_index != -1 and end_index != -1 and end_index >= begin_index:
            end_index += len(_AMLIGNORE_END)
            updated = (
                existing_content[:begin_index].rstrip()
                + "\n\n"
                + managed_block
                + existing_content[end_index:]
            )
            return updated.rstrip() + "\n"

        return existing_content.rstrip() + "\n\n" + managed_block + "\n"

    @staticmethod
    def _should_use_managed_output_dir(output_dir: Any) -> bool:
        """Return whether Azure should promote the run output directory."""
        if not isinstance(output_dir, str):
            return True

        normalized = output_dir.strip().rstrip("/")
        if not normalized:
            return True

        return normalized in {"artifacts", "./artifacts"}

    def _apply_azure_output_dir(self, run_config: Dict[str, Any]) -> None:
        """Route default artifact output into Azure's managed outputs tree."""
        runtime_config = run_config.get("runtime")
        if not isinstance(runtime_config, dict):
            runtime_config = {}
            run_config["runtime"] = runtime_config

        configured_output_dir = runtime_config.get("output_dir")
        if not self._should_use_managed_output_dir(configured_output_dir):
            return

        runtime_config["output_dir"] = "outputs/artifacts"

    def _classify_run_result(self, result: Dict[str, Any]) -> str:
        """Preserve Azure submission states at any sweep worker count."""
        if result.get("success", False):
            return "completed"
        if result.get("submitted", False):
            return "running"
        if result.get("unknown", False):
            return "unknown"
        return "failed"

    def _after_run_execution(
        self, run_descriptors: List[Tuple[int, Path]]
    ) -> None:
        """Retry failed Azure submissions before sweep teardown."""
        if not run_descriptors:
            return
        self._config_dir = run_descriptors[0][1].parent
        if self.retry_limit > 0 and self.failed_runs:
            self._retry_failed_runs(dict(run_descriptors), len(run_descriptors))

    def execute_runs_parallel(
        self, run_descriptors: List[Tuple[int, Path]], max_workers: int
    ) -> None:
        """Submit Azure jobs through the shared atomic run-claim wrapper."""
        total_runs = len(run_descriptors)
        if not run_descriptors:
            return
        self._config_dir = run_descriptors[0][1].parent
        self.logger.info(
            f"Submitting {total_runs} Azure ML jobs with {max_workers} workers"
        )
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = {}
        pending = set()
        aborted = False
        try:
            for index, path in run_descriptors:
                future = pool.submit(self._execute_single_run_wrapper, index, path)
                futures[future] = (index, path)
                pending.add(future)
            for future in as_completed(futures):
                run_index, config_path = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    self._update_tracker(
                        run_index, "failed", config_path, error_message=str(error)
                    )
                    self.failed_runs.append(run_index)
                    self.logger.error(
                        f"Job {run_index + 1}/{total_runs} failed: {error}"
                    )
                    pending.discard(future)
                    continue
                if result.get("skipped"):
                    self.skipped_runs.append(run_index)
                    pending.discard(future)
                    continue

                status = self._classify_run_result(result)
                if status not in {"completed", "running", "unknown"}:
                    status = "failed"
                try:
                    self._update_tracker(run_index, status, config_path, result=result)
                except Exception:
                    self.logger.exception(
                        f"Could not record accepted Azure job "
                        f"{result.get('tracking_run_id')} for run {run_index}; "
                        "aborting without retry"
                    )
                    raise
                if status == "completed":
                    self.completed_runs.append(run_index)
                elif status == "running":
                    self.submitted_runs.append(run_index)
                elif status == "unknown":
                    self.unknown_runs.append(run_index)
                    self.logger.warning(
                        f"Job {run_index + 1}/{total_runs} has unknown status "
                        f"(tracking ID: {result.get('tracking_run_id')})"
                    )
                else:
                    self.failed_runs.append(run_index)
                pending.discard(future)
        except KeyboardInterrupt:
            aborted = True
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            for future in pending:
                if not future.cancelled():
                    run_index, config_path = futures[future]
                    self._update_tracker(
                        run_index,
                        "unknown",
                        config_path,
                        error_message="Interrupted while Azure job may be active",
                    )
            raise
        except Exception:
            aborted = True
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not aborted:
                pool.shutdown(wait=True)

    def _retry_failed_runs(
        self,
        run_lookup: Dict[int, Path],
        original_total: int,
    ) -> None:
        """
        Retry failed runs up to retry_limit times.

        Args:
            run_lookup: Mapping from run index to config path
            original_total: Original total number of runs
        """
        for retry_attempt in range(1, self.retry_limit + 1):
            if not self.failed_runs:
                break

            # Copy current failed runs list (will be modified during retry)
            runs_to_retry = self.failed_runs.copy()
            self.logger.info(
                f"\n{'=' * 60}\nRetry attempt {retry_attempt}/{self.retry_limit}: "
                f"Retrying {len(runs_to_retry)} failed jobs\n{'=' * 60}"
            )

            # Clear failed_runs - will be repopulated if retries fail
            self.failed_runs = []

            # Retry each failed run
            for run_index in runs_to_retry:
                config_path = run_lookup[run_index]
                run_name = Path(config_path).stem

                self.logger.info(
                    f"[RETRY {retry_attempt}] Job {run_index + 1}/{original_total}: {run_name}"
                )

                try:
                    result = self._execute_single_run_wrapper(run_index, config_path)
                except KeyboardInterrupt:
                    self._update_tracker(
                        run_index,
                        "unknown",
                        config_path,
                        error_message="Interrupted while Azure job may be active",
                    )
                    self.unknown_runs.append(run_index)
                    raise
                except Exception as error:
                    self._update_tracker(
                        run_index,
                        "failed",
                        config_path,
                        error_message=str(error),
                    )
                    self.failed_runs.append(run_index)
                    self.logger.error(
                        f"[RETRY {retry_attempt}] Job {run_index + 1} failed: {error}"
                    )
                    continue
                if result.get("skipped"):
                    self.skipped_runs.append(run_index)
                    continue

                self.retry_attempts[run_index] = (
                    self.retry_attempts.get(run_index, 0) + 1
                )
                status = self._classify_run_result(result)
                if status not in {"completed", "running", "unknown"}:
                    status = "failed"
                try:
                    self._update_tracker(run_index, status, config_path, result=result)
                except Exception:
                    self.logger.exception(
                        f"Could not record accepted Azure job "
                        f"{result.get('tracking_run_id')} for retry {run_index}; "
                        "aborting without another retry"
                    )
                    raise
                if status == "completed":
                    self.completed_runs.append(run_index)
                elif status == "running":
                    self.submitted_runs.append(run_index)
                elif status == "unknown":
                    self.unknown_runs.append(run_index)
                    self.logger.warning(
                        f"[RETRY {retry_attempt}] Job {run_index + 1} status unknown"
                    )
                else:
                    self.failed_runs.append(run_index)

        # Final summary
        if self.failed_runs:
            self.logger.error(
                f"\n{len(self.failed_runs)} jobs failed after {self.retry_limit} retry attempts"
            )
        if self.unknown_runs:
            self.logger.warning(
                f"\n{len(self.unknown_runs)} jobs have unknown status - "
                "verify manually in Azure ML Studio"
            )
        if (
            not self.failed_runs
            and not self.unknown_runs
            and not self.submitted_runs
            and not self.skipped_runs
        ):
            self.logger.info(
                f"\nAll failed jobs succeeded after retry (total attempts: {retry_attempt})"
            )

    def _check_job_status_with_retries(
        self, job_name: str, max_retries: int = 5, initial_wait: int = 2
    ) -> Optional[str]:
        """
        Try to get job status with exponential backoff retries.

        Args:
            job_name: Azure ML job name
            max_retries: Maximum number of retry attempts
            initial_wait: Initial wait time in seconds (doubles each retry)

        Returns:
            Job status string if successful, None if all retries failed
        """
        wait_time = initial_wait
        for attempt in range(max_retries):
            try:
                job = self.ml_client.jobs.get(job_name)
                return job.status
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} to check job status failed: {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff
                else:
                    self.logger.error(
                        f"All {max_retries} attempts to check job status failed"
                    )
                    return None
        return None

    def execute_run(
        self,
        run_index: int,
        config_path: Path,
    ) -> Dict[str, Any]:
        """
        Submit run to Azure ML as child job under parent.

        Args:
            run_index: Run index
            config_path: Path to the saved config file

        Returns:
            Dictionary with execution results:
            - "success" (bool): True if job completed successfully
            - "failed" (bool): True if job failed
            - "unknown" (bool): True if job status could not be determined
            - "submitted" (bool): True if Azure accepted a job still in progress
            - "tracking_run_id" (Optional[str]): The external tracking run ID
            - "tracking_run_name" (str): The descriptive run name
        """
        # Read config from disk
        with open(config_path, "r") as f:
            run_config = yaml.safe_load(f)

        self._apply_azure_output_dir(run_config)

        # Generate descriptive run name from grid parameters
        run_name = self.generate_run_name(run_config, run_index)

        # Inject tracking metadata for downstream adapters and logs
        self.inject_tracking_params(
            run_config,
            tracking_context=self.tracking_context,
            tracking_uri=self.tracking_uri,
            run_name=run_name,
        )

        # Save modified config back to disk
        with open(config_path, "w") as f:
            yaml.dump(run_config, f, sort_keys=False)

        # Setup datastore input (if configured)
        inputs = None
        if self.datastore_name:
            datastore_uri = self._build_datastore_uri(self.datastore_name)
            if not self.dry_run:
                inputs = {
                    "dataset_path": Input(
                        type=AssetTypes.URI_FOLDER,
                        path=datastore_uri,
                        mode=InputOutputModes.RW_MOUNT,
                    )
                }
                self.logger.debug(
                    f"Child run {run_index + 1}: Mounting datastore at {datastore_uri}"
                )
            else:
                # In dry-run, just mark that inputs would be present
                inputs = {"dataset_path": f"<would mount: {datastore_uri}>"}

        command_str = self._resolve_submission_command(
            config_path,
            run_config,
            run_index=run_index,
            run_name=run_name,
        )

        self.logger.info(f"[Job {run_index + 1}] Command: {command_str}")

        if self.dry_run:
            self.logger.info("[DRY RUN] Would submit Azure ML job:")
            self.logger.info(f"[DRY RUN]   Display name: {run_name}")
            self.logger.info(f"[DRY RUN]   Parent job: {self.parent_job_name}")
            self.logger.info(f"[DRY RUN]   Compute: {self.compute_target}")
            if inputs:
                self.logger.info(
                    f"[DRY RUN]   Datastore: {self.datastore_name} mounted"
                )
            return {
                "success": True,
                "tracking_run_name": run_name,
            }  # Dry run - no actual job submitted

        # Submit Azure ML job with parent_job_name for nesting
        job = command(
            code=".",
            command=command_str,
            environment=f"{self.environment_name}@{self.environment_version}",
            compute=self.compute_target,
            experiment_name=self.experiment_name,
            display_name=run_name,  # Descriptive name from grid params
            description=f"Child run {run_index + 1} from sweep {self.sweep_id}",
            inputs=inputs,  # Mount datastore (None if not configured)
            parent_job_name=self.parent_job_name,  # Nest under parent job
            environment_variables=self.env_vars,
        )

        # Submit job
        submitted_job = self.ml_client.jobs.create_or_update(job)

        tracking_run_id = submitted_job.name

        self.logger.info(
            f"Submitted job {run_index + 1}: {submitted_job.name} (display: {run_name}, id: {submitted_job.id})"
        )

        # A submitted job is not a completed run until Azure confirms it.
        job_succeeded: bool | None = None
        submitted_only = bool(self.dont_wait_for_completion and submitted_job.name)
        if not self.dont_wait_for_completion and submitted_job.name:
            self.logger.info(f"Waiting for job {submitted_job.name} to complete...")
            self.logger.info("Streaming logs (Ctrl+C to skip waiting and continue):")

            try:
                self.ml_client.jobs.stream(submitted_job.name)
            except KeyboardInterrupt:
                self.logger.warning(
                    f"Skipped waiting for job {submitted_job.name}. "
                    "Job will continue running in Azure."
                )
            except Exception as e:
                self.logger.warning(
                    f"Log streaming failed for job {submitted_job.name}: "
                    f"{type(e).__name__}. Checking the job status."
                )

            job_status = self._check_job_status_with_retries(submitted_job.name)
            if job_status == "Completed":
                job_succeeded = True
                self.logger.info(f"Job {submitted_job.name} completed successfully")
            elif job_status in {"Failed", "Canceled", "Cancelled"}:
                job_succeeded = False
                self.logger.error(
                    f"Job {submitted_job.name} ended with status: {job_status}"
                )
            elif job_status in {
                "Running", "Preparing", "Starting", "Provisioning", "Queued"
            }:
                submitted_only = True
                self.logger.info(
                    f"Job {submitted_job.name} remains active ({job_status})"
                )
            else:
                self.logger.warning(
                    f"Could not determine a terminal status for job "
                    f"{submitted_job.name}: {job_status}"
                )

        # Preserve accepted-but-unfinished jobs as running in the sweep tracker.
        return {
            "success": job_succeeded is True,
            "failed": job_succeeded is False,
            "unknown": job_succeeded is None and not submitted_only,
            "submitted": submitted_only,
            "tracking_run_id": tracking_run_id,
            "tracking_run_name": run_name,
        }

    def build_command(
        self, config_path: str, run_config: Optional[Dict[str, Any]] = None
    ) -> list[str]:
        """
        Build an Azure-safe worker command.

        Azure jobs must not inherit the caller's local virtualenv interpreter
        path. Use the remote environment's ``python`` executable instead.
        """
        cmd = super().build_command(config_path, run_config)
        if cmd:
            cmd[0] = "python"
        return cmd

    def teardown(self) -> None:
        """Print summary after all jobs are submitted/completed."""
        total_jobs = (
            len(self.completed_runs)
            + len(self.failed_runs)
            + len(self.unknown_runs)
            + len(self.submitted_runs)
        )

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("Azure ML Sweep Summary")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Total jobs: {total_jobs}")
        self.logger.info(f"Completed successfully: {len(self.completed_runs)}")
        self.logger.info(f"Submitted and still active: {len(self.submitted_runs)}")
        self.logger.info(f"Failed: {len(self.failed_runs)}")

        if self.unknown_runs:
            self.logger.warning(
                f"Unknown status (verify manually): {len(self.unknown_runs)}"
            )
            self.logger.warning(
                "Jobs with unknown status need manual verification in Azure ML Studio:"
            )
            for run_index in self.unknown_runs:
                self.logger.warning(f"  - Job {run_index + 1}")

        self.logger.info(f"\nParent job: {self.parent_job_name}")
        self.logger.info("Monitor jobs in Azure ML Studio")
        self.logger.info("Child jobs will appear nested under parent")

        # Keep generated configs for reproducibility
        if hasattr(self, "_config_dir"):
            self.logger.info(f"\nGenerated configs saved in: {self._config_dir}")
