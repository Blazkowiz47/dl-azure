"""Azure ML Compute executor for distributed sweeps."""

import fcntl
import json
import os
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
from azureml.core import Workspace

from dl_core.core import BaseExecutor, register_executor


@register_executor("azure")
class AzureComputeExecutor(BaseExecutor):
    """
    Azure ML executor.

    - Submits each run as independent Azure ML job
    - No parent process context
    - MLflow logs to Azure workspace
    - Jobs run on compute cluster
    """

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
        self.compute_target = kwargs["compute_target"]
        environment_name = kwargs.get("environment_name", "dl_lab")

        # Get environment name and version from executor config
        self.environment_name = self.executor_config.get(
            "environment_name", environment_name
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

        self.ml_client: MLClient
        self.parent_job_name = None  # Azure ML job name for parent
        self.tracking_uri = None
        self.env_vars: Dict[str, str] = {}  # Environment variables for jobs
        self.azure_config: Dict[str, Any] = {}  # Azure config (loaded in setup)
        self.retry_attempts: Dict[int, int] = {}  # Track retry attempts per run index

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

    def generate_sas_token(self, expiry_hours: int = 72) -> Optional[str]:
        """
        Generate a time-limited SAS token for Azure Storage using config from azure-config.json.

        Args:
            expiry_hours: Number of hours until token expires (default: 72)

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

        except Exception as e:
            self.logger.error(f"Failed to generate SAS token: {e}")
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

        # Try to generate SAS token for more secure access
        if storage_account_name and os.environ.get("AZURE_ACCESS_KEY"):
            # Generate time-limited SAS token (default: 72 hours)
            sas_token = self.generate_sas_token()

            if sas_token:
                # Pass SAS token instead of access key (more secure)
                env_vars["AZURE_STORAGE_ACCOUNT"] = storage_account_name
                env_vars["AZURE_SAS_TOKEN"] = sas_token
                self.logger.info(
                    f"Configured time-limited SAS token for storage account '{storage_account_name}'"
                )
                self.logger.info(
                    "Jobs will use SAS token for blob storage access (read-only, expires in 72 hours)"
                )
            else:
                # Fall back to access key if SAS generation fails
                access_key = os.environ.get("AZURE_ACCESS_KEY")
                if access_key:
                    self.logger.warning(
                        "SAS token generation failed, falling back to access key"
                    )
                    env_vars["AZURE_STORAGE_ACCOUNT"] = storage_account_name
                    env_vars["AZURE_ACCESS_KEY"] = access_key
        else:
            # Legacy mode: only access key without storage account name
            access_key = os.environ.get("AZURE_ACCESS_KEY")
            if access_key:
                env_vars["AZURE_ACCESS_KEY"] = access_key
                self.logger.warning(
                    "Using AZURE_ACCESS_KEY without storage account name in config. "
                    "Add 'account_name' to azure-config.json to enable SAS token generation."
                )
            else:
                self.logger.warning(
                    "AZURE_ACCESS_KEY not set. Blob storage access in jobs may fail. "
                    "Set AZURE_ACCESS_KEY environment variable for authentication."
                )

        return env_vars

    def setup(self, total_runs: int) -> None:
        """Setup Azure ML client."""
        # Determine sweep name and parent job name upfront
        sweep_file = self.sweep_config.get("sweep_file", "")
        if sweep_file:
            sweep_name = Path(sweep_file).stem
        else:
            sweep_name = self.sweep_id[:8]

        try:
            # Load Azure config (even in dry-run, for building URIs and showing what would happen)
            azure_config_file = Path("./azure-config.json")
            if not azure_config_file.exists():
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
                with open(azure_config_file, "r") as f:
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
                if self.resume and self.tracking_context:
                    self.logger.info(
                        "[DRY RUN]   Would resume under existing tracking context: "
                        f"{self.tracking_context}"
                    )
                    self.parent_job_name = self.tracking_context
                else:
                    self.logger.info(
                        f"[DRY RUN]   Would create parent job: {sweep_name}"
                    )
                    self.parent_job_name = "dry_run_parent_job"
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

            # Setup Azure MLflow tracking URI
            ws = Workspace.from_config("./azure-config.json")
            self.tracking_uri = ws.get_mlflow_tracking_uri()

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
                self.logger.info("Updated .amlignore for sweep")

            # If resuming and tracking_context provided, skip parent job creation
            if self.resume and self.tracking_context:
                self.parent_job_name = self.tracking_context
                self.logger.info(
                    f"Resuming sweep under existing parent job: {self.parent_job_name}"
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
                environment_variables=self.env_vars,
            )

            # Submit parent job
            submitted_parent = self.ml_client.jobs.create_or_update(parent_job)
            self.parent_job_name = submitted_parent.name

            self.logger.info(f"Created parent job: {self.parent_job_name}")
            self.logger.info(f"Display name: {sweep_name}")

        except ImportError:
            raise RuntimeError(
                "Azure mode requires azure-ai-ml. "
                "Install: pip install azure-ai-ml azure-identity"
            )

    def update_amlignore(self, sweep_file: str) -> None:
        """
        Update .amlignore to exclude other user directories and unnecessary files.

        Args:
            sweep_file: Path to the sweep file (e.g., lab/users/sushrut/resnet18/sweeps/resnet18_lr_sweep.yaml)
        """
        amlignore_path = Path(".amlignore")

        # Default ignore patterns
        default_ignores = [
            "# Azure ML ignore patterns",
            ".git/",
            ".vscode/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            ".coverage",
            "htmlcov/",
            ".ruff_cache/",
            ".venv",
            "*.tox",
            "*.toml","uv.lock","artifacts/",
            ".cache/",
            "scores/",
            "mlruns/",
            "*.pyc",
            ".gitignore",
            "wandb/",
            "tests/",
            "docs/",
            "mlruns/",
            "*.db",
            "*reports*",
            "artifacts/",
            "*.md",
            "*.pt",
            "*.pth",
            "*.onnx",
            "*.safetensors",
            "*.log",
            "*.csv",
            "*.json",
            "!azure-config.json",  # Exception: allow Azure config file
            "*.bin",
            "*.ckpt",
            ".dockerignore",
            "*.sh",
            "readme/",
            "phases/",
            ".claude/",
            "pytest.ini",
            "Dockerfile.*",
            "Dockerfile",
            "preprocessing/",
            "scripts/",
            ".gitattributes",
            ".geminiignore",
            "_old/",
            "",
            "# Exclude lab directories except current sweep",
        ]

        # Determine which user directory and sweep to keep
        current_user_dir = None
        current_sweep_name = None
        if sweep_file:
            sweep_path = Path(sweep_file)
            # Extract user directory: lab/users/sushrut/resnet18/sweeps/...
            parts = sweep_path.parts
            if len(parts) >= 4 and parts[0] == "lab" and parts[1] == "users":
                current_user = parts[2]  # e.g., "sushrut"
                current_experiment = parts[3]  # e.g., "resnet18"
                current_user_dir = f"lab/users/{current_user}/{current_experiment}"

                # Extract sweep name (e.g., "resnet_initial_sweep" from "resnet_initial_sweep.yaml")
                if len(parts) >= 6 and parts[4] == "sweeps":
                    current_sweep_name = (
                        sweep_path.stem
                    )  # filename without .yaml extension

        # Get all user directories to exclude (except current)
        lab_users_dir = Path("lab/users")
        exclude_patterns = []

        if lab_users_dir.exists():
            for user_dir in lab_users_dir.iterdir():
                if user_dir.is_dir():
                    for exp_dir in user_dir.iterdir():
                        if exp_dir.is_dir():
                            user_exp_path = f"lab/users/{user_dir.name}/{exp_dir.name}"
                            # Exclude if it's not the current user/experiment
                            if user_exp_path != current_user_dir:
                                exclude_patterns.append(f"{user_exp_path}/")

        # Exclude other sweep's generated config directories within the current experiment
        if current_user_dir and current_sweep_name:
            sweeps_dir = Path(current_user_dir) / "sweeps"
            if sweeps_dir.exists():
                for sweep_subdir in sweeps_dir.iterdir():
                    # Exclude directories (generated configs) that don't match current sweep
                    if (
                        sweep_subdir.is_dir()
                        and sweep_subdir.name != current_sweep_name
                    ):
                        exclude_patterns.append(
                            f"{current_user_dir}/sweeps/{sweep_subdir.name}/"
                        )

        # Add template directory (not needed on Azure)
        exclude_patterns.append("lab/template/")

        # Combine all patterns
        ignore_content = "\n".join(default_ignores + exclude_patterns)

        # Write .amlignore with file locking to prevent race conditions
        # when multiple threads/processes try to update simultaneously
        # Open in 'a+' mode first to avoid truncating before lock acquisition
        with open(amlignore_path, "a+") as f:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Now that we have the lock, truncate and write
                f.seek(0)
                f.truncate()
                f.write(ignore_content)
            finally:
                # Release lock (also auto-released when file closes)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self.logger.info(
            f"Updated .amlignore (keeping only {current_user_dir or 'current experiment'})"
        )

    def execute_runs_parallel(
        self, run_descriptors: List[Tuple[int, Path]], max_workers: int
    ) -> None:
        """
        Execute multiple runs in parallel using ThreadPoolExecutor.

        This overrides the base class to use ThreadPoolExecutor (instead of ProcessPool)
        for better Azure SDK compatibility.

        Args:
            run_descriptors: List of tuples (run_index, config_path)
            max_workers: Number of parallel workers for job submission
        """
        total_runs = len(run_descriptors)

        config_dir = Path(run_descriptors[0][1]).parent
        self._config_dir = config_dir
        self.logger.info(f"Using config directory: {config_dir}")

        run_lookup: Dict[int, Path] = {
            run_index: config_path for run_index, config_path in run_descriptors
        }

        if max_workers > 1:
            self.logger.info(
                f"Submitting {total_runs} Azure ML jobs with {max_workers} parallel workers"
            )
            self.logger.info(
                "Each job will wait for completion (jobs run in parallel using threading)"
            )

            # Use ThreadPoolExecutor for parallel submission
            # This allows multiple jobs to wait for completion concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all runs
                future_to_index = {}
                for run_index, config_path in run_descriptors:
                    run_name = Path(config_path).stem
                    future = executor.submit(self.execute_run, run_index, config_path)
                    future_to_index[future] = run_index

                # Process completed runs
                for future in as_completed(future_to_index):
                    run_index = future_to_index[future]
                    try:
                        result = future.result()
                        success = result.get("success", False)
                        unknown = result.get("unknown", False)
                        tracking_run_id = result.get("tracking_run_id")
                        run_name = result.get("tracking_run_name")

                        if success:
                            self.completed_runs.append(run_index)
                            # Update sweep tracker with completion status
                            if self.tracker and tracking_run_id:
                                self.tracker.update_run_status(
                                    run_index,
                                    status="completed",
                                    tracking_run_id=tracking_run_id,
                                    tracking_run_name=run_name,
                                )
                            self.logger.info(
                                f"Job {run_index + 1}/{total_runs} completed successfully "
                                f"(tracking ID: {tracking_run_id})"
                            )
                        elif unknown:
                            self.unknown_runs.append(run_index)
                            # Update sweep tracker with unknown status
                            if self.tracker:
                                self.tracker.update_run_status(
                                    run_index,
                                    status="unknown",
                                    tracking_run_id=tracking_run_id,
                                    tracking_run_name=run_name,
                                )
                            self.logger.warning(
                                f"Job {run_index + 1}/{total_runs} status unknown - "
                                "verify manually in Azure ML Studio "
                                f"(tracking ID: {tracking_run_id})"
                            )
                        else:
                            self.failed_runs.append(run_index)
                            # Update sweep tracker with failure status
                            if self.tracker:
                                self.tracker.update_run_status(
                                    run_index,
                                    status="failed",
                                    tracking_run_id=tracking_run_id,
                                    tracking_run_name=run_name,
                                )
                            self.logger.error(
                                f"Job {run_index + 1}/{total_runs} failed"
                            )
                    except Exception as e:
                        self.failed_runs.append(run_index)
                        # Update sweep tracker with failure status
                        if self.tracker:
                            self.tracker.update_run_status(
                                run_index,
                                status="failed",
                            )
                        self.logger.error(
                            f"Job {run_index + 1}/{total_runs} failed with exception: {e}"
                        )
        else:
            self.logger.info(f"Submitting {total_runs} Azure ML jobs sequentially")
            self.logger.info(
                "Will wait for each job to complete before submitting next"
            )

            # Sequential execution
            for run_index, config_path in run_descriptors:
                run_name = Path(config_path).stem

                try:
                    result = self.execute_run(run_index, config_path)
                    success = result.get("success", False)
                    unknown = result.get("unknown", False)
                    tracking_run_id = result.get("tracking_run_id")
                    run_name = result.get("tracking_run_name")

                    if success:
                        self.completed_runs.append(run_index)
                        # Update sweep tracker with completion status
                        if self.tracker and tracking_run_id:
                            self.tracker.update_run_status(
                                run_index,
                                status="completed",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=run_name,
                            )
                        self.logger.info(
                            f"Job {run_index + 1}/{total_runs} completed "
                            f"(tracking ID: {tracking_run_id})"
                        )
                    elif unknown:
                        self.unknown_runs.append(run_index)
                        # Update sweep tracker with unknown status
                        if self.tracker:
                            self.tracker.update_run_status(
                                run_index,
                                status="unknown",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=run_name,
                            )
                        self.logger.warning(
                            f"Job {run_index + 1}/{total_runs} status unknown - "
                            "verify manually in Azure ML Studio "
                            f"(tracking ID: {tracking_run_id})"
                        )
                    else:
                        self.failed_runs.append(run_index)
                        # Update sweep tracker with failure status
                        if self.tracker:
                            self.tracker.update_run_status(
                                run_index,
                                status="failed",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=run_name,
                            )
                        self.logger.error(
                            f"Job {run_index + 1}/{total_runs} submission failed"
                        )
                except Exception as e:
                    self.failed_runs.append(run_index)
                    # Update sweep tracker with failure status
                    if self.tracker:
                        self.tracker.update_run_status(
                            run_index,
                            status="failed",
                        )
                    self.logger.error(f"Job {run_index + 1}/{total_runs} failed: {e}")

        # Retry failed runs if retry_limit > 0
        if self.retry_limit > 0 and self.failed_runs:
            self._retry_failed_runs(run_lookup, config_dir, total_runs)

    def _retry_failed_runs(
        self,
        run_lookup: Dict[int, Path],
        config_dir: Path,
        original_total: int,
    ) -> None:
        """
        Retry failed runs up to retry_limit times.

        Args:
            run_lookup: Mapping from run index to config path
            config_dir: Directory containing config files
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
                # Track retry attempts
                if run_index not in self.retry_attempts:
                    self.retry_attempts[run_index] = 0
                self.retry_attempts[run_index] += 1

                config_path = run_lookup[run_index]
                run_name = Path(config_path).stem

                self.logger.info(
                    f"[RETRY {retry_attempt}] Job {run_index + 1}/{original_total}: {run_name}"
                )

                try:
                    result = self.execute_run(run_index, config_path)
                    success = result.get("success", False)
                    unknown = result.get("unknown", False)
                    tracking_run_id = result.get("tracking_run_id")
                    run_name = result.get("tracking_run_name")

                    if success:
                        self.completed_runs.append(run_index)
                        # Update sweep tracker with completion status
                        if self.tracker and tracking_run_id:
                            self.tracker.update_run_status(
                                run_index,
                                status="completed",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=run_name,
                            )
                        self.logger.info(
                            f"[RETRY {retry_attempt}] ✓ Job {run_index + 1} succeeded"
                        )
                    elif unknown:
                        self.unknown_runs.append(run_index)
                        # Update sweep tracker with unknown status
                        if self.tracker:
                            self.tracker.update_run_status(
                                run_index,
                                status="unknown",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=run_name,
                            )
                        self.logger.warning(
                            f"[RETRY {retry_attempt}] Job {run_index + 1} status unknown - "
                            "verify manually in Azure ML Studio"
                        )
                    else:
                        self.failed_runs.append(run_index)
                        # Update sweep tracker with failure status
                        if self.tracker:
                            self.tracker.update_run_status(
                                run_index,
                                status="failed",
                                tracking_run_id=tracking_run_id,
                                tracking_run_name=run_name,
                            )
                        self.logger.error(
                            f"[RETRY {retry_attempt}] Job {run_index + 1} failed again"
                        )
                except Exception as e:
                    self.failed_runs.append(run_index)
                    # Update sweep tracker with failure status
                    if self.tracker:
                        self.tracker.update_run_status(
                            run_index,
                            status="failed",
                        )
                    self.logger.error(
                        f"[RETRY {retry_attempt}] Job {run_index + 1} failed with exception: {e}"
                    )

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
        if not self.failed_runs and not self.unknown_runs:
            self.logger.info(
                f"\nAll failed jobs succeeded after retry (total attempts: {retry_attempt})"
            )

    def _is_connection_error(self, exception: Exception) -> bool:
        """
        Check if an exception is a network/connection error.

        Args:
            exception: The exception to check

        Returns:
            True if the exception appears to be a connection/network error
        """
        error_str = str(exception).lower()
        connection_indicators = [
            "failed to resolve",
            "name or service not known",
            "connection",
            "timeout",
            "network",
            "urllib3",
            "ssl",
            "certificate",
            "connectionerror",
            "httperror",
        ]
        return any(indicator in error_str for indicator in connection_indicators)

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
            - "tracking_run_id" (Optional[str]): The external tracking run ID
            - "tracking_run_name" (str): The descriptive run name
        """
        # Read config from disk
        with open(config_path, "r") as f:
            run_config = yaml.safe_load(f)

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

        # Build command using base class method and convert list to string (Azure ML requires string)
        cmd_list = self.build_command(str(config_path), run_config)
        command_str = " ".join(cmd_list)

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
        )

        # Submit job
        submitted_job = self.ml_client.jobs.create_or_update(job)

        tracking_run_id = submitted_job.name

        self.logger.info(
            f"Submitted job {run_index + 1}: {submitted_job.name} (display: {run_name}, id: {submitted_job.id})"
        )

        # Wait for job completion unless explicitly disabled
        job_succeeded = True
        if not self.dont_wait_for_completion and submitted_job.name:
            self.logger.info(f"Waiting for job {submitted_job.name} to complete...")
            self.logger.info("Streaming logs (Ctrl+C to skip waiting and continue):")

            streaming_failed = False

            try:
                # Stream logs and wait for completion
                self.ml_client.jobs.stream(submitted_job.name)

            except KeyboardInterrupt:
                self.logger.warning(
                    f"Skipped waiting for job {submitted_job.name}. "
                    "Job will continue running in Azure."
                )
                # Check status before returning
                status = self._check_job_status_with_retries(
                    submitted_job.name, max_retries=3
                )
                if status:
                    self.logger.info(f"Job status at interruption: {status}")
                job_succeeded = True  # Don't fail on user interruption

            except Exception as e:
                # Check if this is a connection/network error
                if self._is_connection_error(e):
                    self.logger.warning(
                        f"Connection error while streaming logs for job {submitted_job.name}: {e}"
                    )
                    self.logger.info(
                        "Log streaming failed, but job may still be running. Checking job status..."
                    )
                    streaming_failed = True
                else:
                    # Non-connection error - likely a real problem
                    self.logger.error(
                        f"Error while streaming logs for job {submitted_job.name}: {e}"
                    )
                    streaming_failed = True

            # Always try to check final job status
            try:
                if streaming_failed:
                    # Use retry logic for connection issues
                    job_status = self._check_job_status_with_retries(submitted_job.name)

                    if job_status is None:
                        # Could not determine status after retries
                        self.logger.error(
                            f"Unable to determine status for job {submitted_job.name} after connection failure. "
                            "Job may still be running - check Azure ML Studio manually."
                        )
                        # Return unknown status instead of assuming success
                        job_succeeded = None  # None indicates unknown status
                    else:
                        # Successfully got status
                        if job_status == "Completed":
                            self.logger.info(
                                f"✓ Job {submitted_job.name} completed successfully"
                            )
                            job_succeeded = True
                        elif job_status in [
                            "Running",
                            "Preparing",
                            "Starting",
                            "Provisioning",
                            "Queued",
                        ]:
                            self.logger.warning(
                                f"Job {submitted_job.name} is still running (status: {job_status}). "
                                "Status is indeterminate - check Azure ML Studio manually."
                            )
                            job_succeeded = None  # Unknown status - job still running
                        else:
                            self.logger.error(
                                f"✗ Job {submitted_job.name} failed with status: {job_status}"
                            )
                            job_succeeded = False
                else:
                    # Normal flow - streaming completed without error
                    final_job = self.ml_client.jobs.get(submitted_job.name)
                    job_status = final_job.status

                    if job_status == "Completed":
                        self.logger.info(
                            f"✓ Job {submitted_job.name} completed successfully"
                        )
                        job_succeeded = True
                    else:
                        self.logger.error(
                            f"✗ Job {submitted_job.name} failed with status: {job_status}"
                        )
                        job_succeeded = False

            except Exception as e:
                self.logger.error(
                    f"Failed to check final status for job {submitted_job.name}: {e}"
                )
                # Can't determine status - mark as unknown
                job_succeeded = None

        # Return success/failed/unknown status and tracking identifiers
        return {
            "success": job_succeeded is True,
            "failed": job_succeeded is False,
            "unknown": job_succeeded is None,
            "tracking_run_id": tracking_run_id,
            "tracking_run_name": run_name,
        }

    def teardown(self) -> None:
        """Print summary after all jobs are submitted/completed."""
        total_jobs = (
            len(self.completed_runs) + len(self.failed_runs) + len(self.unknown_runs)
        )

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("Azure ML Sweep Summary")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Total jobs: {total_jobs}")
        self.logger.info(f"Completed successfully: {len(self.completed_runs)}")
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
