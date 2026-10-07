# Technical: 4. Azure Blob Downloads

## Configuration

Keep Azure connection, download, cache, and prefetch settings under `azure`.
An experiment's `dataset.azure` overrides the defaults in `azure-config.json`.
Mappings merge recursively, so changing SDK concurrency leaves other project
defaults intact.

```json
{
  "azure": {
    "subscription_id": "<subscription-id>",
    "resource_group": "<resource-group>",
    "workspace_name": "<workspace-name>",
    "account_name": "<storage-account-name>",
    "download": {
      "backend": "azcopy",
      "fallback_to_sdk": true
    }
  }
}
```

```yaml
dataset:
  azure:
    config_path: azure-config.json
    container_name: datasets
    download:
      backend: azcopy
      fallback_to_sdk: true
      max_concurrent_files: 4
      azcopy:
        executable: azcopy
        concurrency: null
        buffer_gb: null
        cap_mbps: 0
        timeout_seconds: null
      sdk:
        max_concurrency: 32
        connection_pool_size: 32
        chunk_size_mib: 4
        initial_request_size_mib: 32
        read_buffer_kib: 64
        connection_timeout_seconds: 20
        read_timeout_seconds: 120
    cache:
      cache_dir: /mnt/localssd/shards
      cache_size_gb: 3000
    prefetch:
      enabled: true
      trigger_fraction: 0.5
      max_concurrent_downloads: 2
```

All shown download values are defaults. Prefetch defaults to disabled and still
requires the trainer to supply plans and progress. Projects choose shard paths
through `get_shard_cache_path()`; download settings do not prescribe a layout.

Legacy flat project files and dataset keys remain accepted, including
`azure_config_path`, `account_name`, `container_name`, `sas_expiry_hours`,
`download`, `cache`, and `prefetch`. Precedence is project defaults, then legacy
dataset fields, then explicit `dataset.azure` settings. The Azure executor
accepts both project-file formats; its sweep settings remain under `executor`.
Scaffolding keeps the format of existing project files.

## Backend Selection

`AzureClientService.download_blob()` and `AzureShardCache.ensure()` use the same
transfer routine for any blob downloaded to a local file. The default backend
is `azcopy`; set `backend: sdk` to always use the Python SDK.

AzCopy must be installed separately. Executable lookup is cached once per
configured executable per process, with synchronized access from download
threads. If it is absent or cannot be launched, later calls use the SDK
directly. Setting `fallback_to_sdk: false` makes those calls fail instead.

The client reuses a supplied SAS token or signs a URL with its storage key or
user-delegation credentials. It prepares the URL only after finding AzCopy.
If signing or an AzCopy transfer fails, the SDK retries that file using the
existing client credentials. Such failures leave AzCopy enabled for later
files. Child processes are invoked with argument lists, without a shell.
Signed URLs and raw CLI output are excluded from package logs; temporary
AzCopy job plans and logs are removed after the command exits.

`concurrency: null` and `buffer_gb: null` preserve the corresponding environment
settings or AzCopy defaults. Concurrency accepts a positive integer or `AUTO`.
`buffer_gb` is a positive memory-buffer setting. `cap_mbps` is a per-command
transfer cap in megabits per second; `0` leaves it uncapped.
`timeout_seconds` limits the whole command; `null` has no command timeout.
On timeout, the child is killed and reaped before SDK fallback opens the file.
See Microsoft's [AzCopy settings](https://learn.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-configuration-settings).

Image and JSON loaders that return data in memory keep using the SDK, including
when the file backend is AzCopy. They share the SDK client and request settings
without starting a CLI process for every sample.

## SDK Settings and Validation

SDK file downloads use `readinto()` so `max_concurrency` can parallelize range
requests. `chunk_size_mib` controls those ranges; `initial_request_size_mib`
controls the initial GET when transactional MD5 is disabled. `read_buffer_kib`
configures the actual Requests transport, and `connection_pool_size` controls
how many reusable connections its nonblocking pool retains. The request worker
count is bounded by `max_concurrency`; pool size is not a hard active-request
limit. Size and concurrency fields take positive integers, timeouts take
positive seconds. See the [Python SDK tuning guide](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-tune-upload-download-python).

`sdk.validate_content` controls SDK transactional MD5 checks. If omitted, it is
`true` for the shard cache and `false` for generic file and in-memory reads.
Azure permits these checks only for ranges up to 4 MiB, so larger configured
chunks require `validate_content: false`. When checks are enabled, the SDK
uses the chunk size for its first GET too. See [Get Blob range validation](https://learn.microsoft.com/en-us/rest/api/storageservices/get-blob).

Both backends check final size and a stored blob MD5 when one exists. The SDK
uses an ETag condition during the download. AzCopy checks the source size and
ETag again after completion. AzCopy's stored-MD5 checks differ from the SDK's
per-range validation; a blob without stored MD5 has no whole-file checksum to
compare. The shard cache also validates tar format before atomic promotion.
Neither backend exposes a partial file as the final destination.

The existing `cache.connection_timeout_seconds` and
`cache.read_timeout_seconds` keys remain fallbacks for shard-cache requests.
Explicit values in `download.sdk` take precedence. Cache retry and backoff
settings still apply to whole-file attempts; this path does not use the
preprocessing AzCopy wrapper's separate retry loop.

`download.max_concurrent_files` limits initial indexed staging. It defaults to
`4` and accepts positive integers; `1` stages sequentially. OS file locks in the
cache state directory share this budget across staging threads and processes,
including GPU ranks. All callers sharing cache state must configure the same
limit. Different cache states have independent staging budgets.

`prefetch.max_concurrent_downloads` separately limits background files in flight
per prefetch controller. SDK and AzCopy concurrency limits requests within each
transfer. Memory and request load grow with both file and request concurrency;
AzCopy buffer settings and transfer caps apply per command. Direct single-file
downloads and streaming reads keep their existing scheduling.

## Direct File Downloads

```python
import json
from pathlib import Path
from dl_azure.storage import AzureClientService

config = json.loads(Path("azure-config.json").read_text())
client = AzureClientService(config)
ok = client.download_blob("datasets", "metadata/classes.json", Path("data/classes.json"))
```

The service writes beside the chosen destination and replaces it only after
validation. It returns `False` on failure and preserves an existing file.
Direct cache consumers can pass the same download block as
`AzureShardCache(..., download_config={...})`; locks, reservations, path
resolvers, and eviction remain owned by the cache.
