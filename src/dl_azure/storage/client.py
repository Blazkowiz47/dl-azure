"""Azure client service for blob storage operations using DefaultAzureCredential."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient, ContainerClient

logger = logging.getLogger(__name__)

# Suppress Azure SDK HTTP logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.ERROR
)
logging.getLogger("azure.identity._internal.decorators").setLevel(logging.ERROR)
logging.getLogger("azure.core").setLevel(logging.ERROR)
logging.getLogger("azure.identity").setLevel(logging.ERROR)


class AzureClientService:
    """Centralized Azure client service using DefaultAzureCredential."""

    # Class-level cache for container clients (shared across instances)
    _container_clients: Dict[str, ContainerClient] = {}

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Azure client service.

        Args:
            config: Azure configuration containing 'account_name'
        """
        self.config = config
        self.account_name = config.get("account_name")

        if not self.account_name:
            raise ValueError("Azure config must contain 'account_name'")

        # Use DefaultAzureCredential for authentication
        self.credential = DefaultAzureCredential()

        logger.info(f"Initialized Azure client for account: {self.account_name}")
        logger.info(f"Using credential type: {type(self.credential).__name__}")

    def get_credential(self):
        """Get the Azure credential."""
        return self.credential

    def get_container_url(self, container_name: str) -> str:
        """
        Get container URL for blob storage.

        Args:
            container_name: Name of the container

        Returns:
            Container URL
        """
        return f"https://{self.account_name}.blob.core.windows.net/{container_name}"

    def create_blob_client(self, container_name: str, blob_path: str) -> BlobClient:
        """
        Create a BlobClient for a specific blob.

        Args:
            container_name: Name of the container
            blob_path: Path to the blob

        Returns:
            BlobClient instance
        """
        container_url = self.get_container_url(container_name)
        blob_url = f"{container_url}/{blob_path}"

        logger.debug(f"Creating BlobClient for: {blob_url}")
        try:
            client = BlobClient.from_blob_url(blob_url, credential=self.credential)
            logger.debug(f"Successfully created BlobClient for {blob_path}")
            return client
        except Exception as e:
            logger.error(f"Failed to create BlobClient for {blob_path}: {e}")
            raise ConnectionError(f"Could not create blob client: {e}") from e

    def create_container_client(self, container_name: str) -> ContainerClient:
        """
        Create a ContainerClient for a specific container.

        Args:
            container_name: Name of the container

        Returns:
            ContainerClient instance
        """
        container_url = self.get_container_url(container_name)

        logger.debug(f"Creating ContainerClient for: {container_url}")
        try:
            client = ContainerClient.from_container_url(
                container_url, credential=self.credential
            )
            logger.debug(f"Successfully created ContainerClient for {container_name}")
            return client
        except Exception as e:
            logger.error(f"Failed to create ContainerClient for {container_name}: {e}")
            raise ConnectionError(f"Could not create container client: {e}") from e

    def get_container_client(self, container_name: str) -> ContainerClient:
        """
        Get a cached ContainerClient for a specific container.

        Uses class-level caching to reuse ContainerClient instances,
        which maintains HTTP connection pools for better performance.

        Args:
            container_name: Name of the container

        Returns:
            Cached ContainerClient instance
        """
        cache_key = f"{self.account_name}/{container_name}"

        if cache_key not in self._container_clients:
            logger.debug(f"Creating new cached ContainerClient for: {container_name}")
            self._container_clients[cache_key] = self.create_container_client(
                container_name
            )

        return self._container_clients[cache_key]

    def get_blob_client_pooled(self, container_name: str, blob_path: str) -> BlobClient:
        """
        Get a BlobClient using a pooled ContainerClient connection.

        This method reuses the cached ContainerClient's HTTP connection pool,
        avoiding the overhead of creating new connections for each blob.

        Args:
            container_name: Name of the container
            blob_path: Path to the blob

        Returns:
            BlobClient instance using pooled connection
        """
        container_client = self.get_container_client(container_name)
        return container_client.get_blob_client(blob_path)

    @staticmethod
    def create_blob_client_static(
        config: Dict[str, Any], container_name: str, blob_path: str
    ) -> BlobClient:
        """
        Static method to create a BlobClient (for backward compatibility).

        Args:
            config: Azure configuration
            container_name: Name of the container
            blob_path: Path to the blob

        Returns:
            BlobClient instance
        """
        service = AzureClientService(config)
        return service.create_blob_client(container_name, blob_path)

    @staticmethod
    def create_container_client_static(
        config: Dict[str, Any], container_name: str
    ) -> ContainerClient:
        """
        Static method to create a ContainerClient (for backward compatibility).

        Args:
            config: Azure configuration
            container_name: Name of the container

        Returns:
            ContainerClient instance
        """
        service = AzureClientService(config)
        return service.create_container_client(container_name)

    def upload_file(
        self, local_path: Path, container_name: str, blob_path: str
    ) -> bool:
        """
        Upload a single file to blob storage.

        Args:
            local_path: Path to local file
            container_name: Name of the container
            blob_path: Destination blob path

        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.get_blob_client_pooled(container_name, blob_path)

            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            logger.debug(f"Uploaded {local_path} to {blob_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {blob_path}: {e}")
            return False

    def upload_directory(
        self,
        local_dir: Path,
        container_name: str,
        blob_prefix: str,
        max_workers: int = 4,
    ) -> bool:
        """
        Upload all files in a directory to blob storage with parallel uploads.

        Args:
            local_dir: Path to local directory
            container_name: Name of the container
            blob_prefix: Blob prefix (directory path in blob storage)
            max_workers: Number of parallel upload threads

        Returns:
            True if all uploads successful, False if any failed
        """
        files = list(local_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        if not files:
            logger.warning(f"No files found in {local_dir}")
            return False

        success_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for file_path in files:
                # Get relative path from local_dir
                rel_path = file_path.relative_to(local_dir)
                blob_path = f"{blob_prefix}/{rel_path}".replace("\\", "/")

                future = executor.submit(
                    self.upload_file, file_path, container_name, blob_path
                )
                futures[future] = file_path

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        failed_count += 1
                        logger.error(f"Failed to upload: {file_path}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Exception uploading {file_path}: {e}")

        logger.debug(
            f"Upload complete: {success_count} succeeded, {failed_count} failed"
        )
        return failed_count == 0

    def download_blob(
        self, container_name: str, blob_path: str, local_path: Path
    ) -> bool:
        """
        Download a blob to local file.

        Args:
            container_name: Name of the container
            blob_path: Path to blob
            local_path: Destination local file path

        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.get_blob_client_pooled(container_name, blob_path)

            # Create parent directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())

            logger.debug(f"Downloaded {blob_path} to {local_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {blob_path} to {local_path}: {e}")
            return False

    def blob_exists(self, container_name: str, blob_path: str) -> bool:
        """
        Check if a blob exists.

        Args:
            container_name: Name of the container
            blob_path: Path to blob

        Returns:
            True if blob exists, False otherwise
        """
        try:
            blob_client = self.get_blob_client_pooled(container_name, blob_path)
            return blob_client.exists()
        except Exception as e:
            logger.debug(f"Error checking existence of {blob_path}: {e}")
            return False

    def list_blobs(
        self, container_name: str, prefix: str = "", max_results: Optional[int] = None
    ) -> List[str]:
        """
        List blobs with given prefix.

        Args:
            container_name: Name of the container
            prefix: Blob path prefix to filter
            max_results: Maximum number of results (None for all)

        Returns:
            List of blob paths
        """
        try:
            container_client = self.get_container_client(container_name)
            blobs = container_client.list_blobs(name_starts_with=prefix)

            blob_names = []
            for i, blob in enumerate(blobs):
                if max_results and i >= max_results:
                    break
                blob_names.append(blob.name)

            logger.debug(f"Listed {len(blob_names)} blobs with prefix: {prefix}")
            return blob_names

        except Exception as e:
            logger.error(f"Failed to list blobs with prefix {prefix}: {e}")
            return []

    def get_blob_url(self, container_name: str, blob_path: str) -> str:
        """
        Get blob URL (without SAS token).

        Args:
            container_name: Name of the container
            blob_path: Path to blob

        Returns:
            Blob URL
        """
        return f"https://{self.account_name}.blob.core.windows.net/{container_name}/{blob_path}"

    def get_blob_sas_url(
        self,
        container_name: str,
        blob_path: str,
        expiry_hours: int = 24,
        permissions: str = "r",
    ) -> str:
        """
        Get blob URL with SAS token for streaming access.

        Note: SAS token generation requires account key. When using DefaultAzureCredential,
        this method returns the blob URL without SAS. Ensure blobs are publicly accessible
        or Azure credentials are properly configured.

        Args:
            container_name: Name of the container
            blob_path: Path to blob
            expiry_hours: Hours until SAS token expires (unused with DefaultAzureCredential)
            permissions: Permissions string ('r' for read, 'rw' for read/write)

        Returns:
            Blob URL (without SAS token when using DefaultAzureCredential)
        """
        try:
            blob_client = self.get_blob_client_pooled(container_name, blob_path)

            # Generate SAS token (requires account key, not supported with DefaultAzureCredential)
            # For now, return URL without SAS - requires blob to be publicly accessible
            # or caller must have proper Azure credentials configured
            logger.warning(
                f"SAS token generation requires account key. "
                f"Returning URL without SAS for {blob_path}. "
                f"Ensure blob is publicly accessible or Azure credentials are configured."
            )
            return self.get_blob_url(container_name, blob_path)

        except Exception as e:
            logger.warning(
                f"Failed to get blob URL for {blob_path}: {e}"
            )
            # Fallback to regular URL
            return self.get_blob_url(container_name, blob_path)


