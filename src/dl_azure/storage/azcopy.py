"""AzCopy-based helpers for Azure Blob Storage transfers."""

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)
DEFAULT_AZCOPY_MAX_RETRIES = 5


class AzCopyTransferBase:
    """Shared AzCopy transfer helpers."""

    def __init__(self, account_name: str, container_name: str) -> None:
        """
        Initialize the transfer helper.

        Args:
            account_name: Azure Storage account name.
            container_name: Azure Blob Storage container name.
        """
        if not account_name:
            raise ValueError("AzCopy uploader requires a storage account name")
        if not container_name:
            raise ValueError("AzCopy uploader requires a container name")

        self.account_name = account_name
        self.container_name = container_name
        self.max_retries = DEFAULT_AZCOPY_MAX_RETRIES

    def build_blob_url(self, blob_path: str) -> str:
        """
        Build a plain blob URL for an object in the configured container.

        Args:
            blob_path: Blob path relative to the container root.

        Returns:
            Blob URL suitable for AzCopy destinations.
        """
        normalized_blob_path = blob_path.lstrip("/")
        encoded_blob_path = quote(normalized_blob_path, safe="/")
        return (
            f"https://{self.account_name}.blob.core.windows.net/"
            f"{self.container_name}/{encoded_blob_path}"
        )

    def _run_azcopy_command(
        self,
        command: list[str],
        source_path: Path,
    ) -> bool:
        """
        Run an AzCopy command with retry and auth-error handling.

        Args:
            command: Full AzCopy command to execute.
            source_path: Logical source path for logging purposes.

        Returns:
            True on success, False otherwise.
        """
        if shutil.which("azcopy") is None:
            logger.error(
                "AzCopy is not installed or not on PATH. Install AzCopy and run "
                "`azcopy login` before preprocessing transfers."
            )
            return False

        concurrency_plan = self._build_concurrency_plan()
        last_error = "Unknown AzCopy error"

        for attempt, concurrency_value in enumerate(concurrency_plan, start=1):
            result = self._invoke_azcopy(
                command=command,
                concurrency_value=concurrency_value,
                source_path=source_path,
            )
            if result is None:
                return False

            if result.returncode == 0:
                logger.debug(
                    f"AzCopy transfer succeeded for {source_path} on attempt "
                    f"{attempt}"
                )
                return True

            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            combined_output = stderr or stdout or "Unknown AzCopy error"
            last_error = combined_output

            if self._looks_like_auth_error(combined_output):
                logger.error(
                    "AzCopy transfer failed due to authentication for %s: %s",
                    source_path,
                    combined_output,
                )
                return False

            if attempt < len(concurrency_plan):
                logger.warning(
                    "AzCopy transfer failed for %s on attempt %s/%s with "
                    "AZCOPY_CONCURRENCY_VALUE=%s. Retrying.",
                    source_path,
                    attempt,
                    len(concurrency_plan),
                    concurrency_value if concurrency_value is not None else "default",
                )

        logger.error(f"AzCopy transfer failed for {source_path}: {last_error}")
        return False

    @contextmanager
    def _build_safe_local_source(self, source_path: Path):
        """
        Yield a local source path that avoids AzCopy parsing bugs for '%' paths.

        Args:
            source_path: Original local source path.

        Yields:
            Path to use as the AzCopy local source.
        """
        if "%" not in str(source_path):
            yield source_path
            return

        with tempfile.TemporaryDirectory(prefix="azcopy_src_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            safe_name = source_path.name.replace("%", "_pct_")
            safe_source = temp_dir / safe_name
            os.symlink(source_path, safe_source, target_is_directory=source_path.is_dir())
            yield safe_source

    def upload_file(self, local_path: Path, blob_path: str) -> bool:
        """
        Upload a single file with AzCopy.

        Args:
            local_path: Local source file path.
            blob_path: Destination blob path.

        Returns:
            True on success, False otherwise.
        """
        destination_url = self.build_blob_url(blob_path)
        with self._build_safe_local_source(local_path) as safe_local_path:
            return self._run_azcopy(
                source_path=safe_local_path,
                destination_url=destination_url,
                recursive=False,
            )

    def upload_directory(self, local_dir: Path, blob_prefix: str) -> bool:
        """
        Upload a directory recursively with AzCopy.

        Args:
            local_dir: Local source directory.
            blob_prefix: Destination blob prefix.

        Returns:
            True on success, False otherwise.
        """
        destination_url = self.build_blob_url(blob_prefix.rstrip("/") + "/")
        with self._build_safe_local_source(local_dir) as safe_local_dir:
            return self._run_azcopy(
                source_path=safe_local_dir,
                destination_url=destination_url,
                recursive=True,
            )

    def _run_azcopy(
        self,
        source_path: Path,
        destination_url: str,
        recursive: bool,
    ) -> bool:
        """
        Run an AzCopy upload command.

        Args:
            source_path: Local source path to upload.
            destination_url: Blob URL destination.
            recursive: Whether to enable recursive upload.

        Returns:
            True on success, False otherwise.
        """
        command = [
            "azcopy",
            "copy",
            str(source_path),
            destination_url,
            "--overwrite=true",
            "--from-to=LocalBlob",
        ]
        if recursive:
            command.append("--recursive=true")

        return self._run_azcopy_command(command=command, source_path=source_path)

    def _invoke_azcopy(
        self,
        command: list[str],
        concurrency_value: int | None,
        source_path: Path,
    ) -> subprocess.CompletedProcess[str] | None:
        """
        Invoke AzCopy with a specific concurrency setting.

        Args:
            command: Base AzCopy command.
            concurrency_value: Value for AZCOPY_CONCURRENCY_VALUE, or None to use
                the default AzCopy behavior.
            source_path: Local source path being uploaded.

        Returns:
            Completed process on success, or None if the command could not start.
        """
        if concurrency_value is None:
            full_command = command
            use_shell = False
        else:
            quoted_command = " ".join(shlex.quote(part) for part in command)
            full_command = (
                f"AZCOPY_CONCURRENCY_VALUE={concurrency_value} {quoted_command}"
            )
            use_shell = True

        try:
            return subprocess.run(
                full_command,
                check=False,
                capture_output=True,
                text=True,
                shell=use_shell,
            )
        except OSError as error:
            logger.error(f"Failed to start AzCopy for {source_path}: {error}")
            return None

    def _build_concurrency_plan(self) -> list[int | None]:
        """
        Build the retry concurrency plan for AzCopy uploads.

        Returns:
            Ordered list of concurrency values to try, where None means use
            AzCopy defaults with no override.
        """
        plan: list[int | None] = [None, 2, 2, 1, 1]
        return plan[: self.max_retries]

    @staticmethod
    def _looks_like_auth_error(output: str) -> bool:
        """
        Detect likely AzCopy authentication failures from command output.

        Args:
            output: Combined stdout/stderr text from AzCopy.

        Returns:
            True if the output looks like an auth error.
        """
        lower_output = output.lower()
        auth_markers = (
            "authenticationfailed",
            "authorizationfailure",
            "no cached token found",
            "please run 'azcopy login'",
            "please run `azcopy login`",
            "login credentials missing",
            "failed to authenticate",
            "401",
            "403",
            "unauthorized",
            "forbidden",
        )
        return any(marker in lower_output for marker in auth_markers)


class AzCopyUploader(AzCopyTransferBase):
    """Upload files and directories to Azure Blob Storage via AzCopy."""


class AzCopyDownloader(AzCopyTransferBase):
    """Download directories from Azure Blob Storage via AzCopy."""

    def download_directory(
        self,
        blob_prefix: str,
        local_dir: Path,
        overwrite: bool = False,
    ) -> bool:
        """
        Download a blob prefix recursively into a local directory.

        Args:
            blob_prefix: Source blob prefix in the configured container.
            local_dir: Local destination directory.
            overwrite: Whether to overwrite existing local files.

        Returns:
            True on success, False otherwise.
        """
        source_url = self.build_blob_url(blob_prefix.rstrip("/") + "/")
        local_dir.mkdir(parents=True, exist_ok=True)
        overwrite_mode = "true" if overwrite else "ifSourceNewer"
        command = [
            "azcopy",
            "copy",
            source_url,
            str(local_dir),
            "--recursive=true",
            f"--overwrite={overwrite_mode}",
            "--from-to=BlobLocal",
        ]
        return self._run_azcopy_command(command=command, source_path=Path(blob_prefix))
