# Technical: 3. Dataset Mounts and Runtime Notes

## Compute Dataset Roots

The generic compute dataset wrappers support three root resolution paths:

- explicit `dataset.root_dir`
- Azure ML input mounts via `AZURE_ML_INPUT_<input_name>`
- optional local fallback when the wrapper config allows it

That means project-specific datasets should pass either a concrete `root_dir`
or an `input_name` instead of hardcoding a single mounted directory name.

If `dataset.root_dir` is relative and the Azure ML input mount exists, the
wrapper resolves it under that mount. If no Azure ML mount is present and local
fallback is enabled, the wrapper uses `dataset.local_fallback_root`.

## Streaming Dataset Configuration

The generic streaming wrappers read directly from Azure blob storage instead of
the mounted filesystem.

Required settings:

- `dataset.container_name`
- Azure storage config with `account_name`

Azure storage config can come from:

- `dataset.azure_config_path`, which defaults to `azure-config.json`
- inline dataset config keys such as `account_name`, `subscription_id`,
  `resource_group`, `workspace_name`, and `tenant_id`

The wrapper lists blob paths under the configured prefix and downloads images
or metadata on demand through the shared Azure client service.

When callers request a shareable blob URL, the client generates a
user-delegation SAS through `DefaultAzureCredential`. The authenticated identity
must be allowed to request a user delegation key and must have the appropriate
Blob Data role. Generation failures raise an error instead of returning an
unsigned URL that may fail later.

## Cache Behavior

The blob cache is only used by the streaming wrappers. Compute wrappers read
directly from the resolved local or mounted filesystem path and do not use the
Azure blob cache.

Streaming cache settings live under `dataset.cache`:

- `enabled`
- `cache_dir`
- `cache_splits`

`cache_splits` defaults to `train`, `validation`, and `test`, so callers can
still disable caching for selected splits without changing wrapper code.

Frame wrappers also support `face_detected_and_resized_cache`. When that flag
is enabled and a cache backend exists, the wrapper stores resized frames or
face-cropped frames in the cache as a second-level optimization. Processed
cache entries are separated by output size and, for face crops, margin.

Blob cache paths are encoded and kept beneath the configured cache directory,
including blob names with absolute or parent-like path segments. Cache
statistics and cleanup include files in the full hierarchical layout. Azure
container-client pooling is scoped to one authenticated client service so
credentials are never mixed through a process-wide cache.

## WebDataset Tar Shards

`AzureComputeTarShardWrapper` resolves relative `.tar` paths under the normal
compute root. `AzureStreamingTarShardWrapper` lists or accepts blob paths and
converts them to read-only signed URLs. WebDataset then splits the
shard stream by distributed rank and DataLoader worker before opening archives.

The on-demand local shard cache is enabled by default and is required for this
wrapper. It removes SAS query strings before shard URLs enter sample metadata;
`dataset.cache.enabled: false` is rejected. The cache is lazy: WebDataset still
splits shards by rank and worker before the selected shard is downloaded. SAS
expiry defaults to seven days and can be changed with
`dataset.sas_expiry_hours` when a new token is generated in the worker.

```yaml
dataset:
  cache:
    enabled: true
    cache_dir: /mnt/localssd/dl-azure
    cache_size_gb: 3000
    download_retries: 5
    retry_backoff_seconds: 1
    retry_backoff_max_seconds: 30
    retry_jitter: true
    connection_timeout_seconds: 20
    read_timeout_seconds: 120
    lock_timeout_seconds: 3600
```

`cache_size_gb` defaults to 3000 and is converted internally using 1024^3 bytes
per GB. The old byte-based `cache_size` key is rejected to avoid unit mistakes.
`download_retries` is the number of retries after the initial attempt.

Each retry creates a fresh Azure client and downloader. Failed partial files
are removed, and a completed file is promoted atomically only after its ETag
condition, blob size, Azure content validation, and tar format have passed. A
per-shard file lock prevents workers and ranks sharing a host cache from
downloading the same URL concurrently. SAS query strings are excluded from
cache identities, so renewed tokens reuse the same shard file.

Timeouts, connection failures, changed ETags, HTTP 408/429, and server errors
are retried with capped exponential backoff. Authentication, permission, and
missing-blob responses fail immediately. After the final attempt, a
`RuntimeError` with the public shard URL and exception type reaches the
DataLoader; signed URLs are excluded from the error text.

WebDataset supports uncompressed and compressed tar streams. Its shuffle is
buffered rather than a perfect global permutation, and resampled training may
repeat samples. Keep validation and test finite and deterministic.

Project wrappers may override `build_shard_sources(split)` to return dynamically
discovered sources with `name`, `weight`, and `shards`. Azure compute resolves
relative mounted paths after this hook; Azure streaming converts returned blob
paths to SAS URLs after the hook. This keeps project discovery and weighting
separate from backend access.

### Custom Shard Destinations

Override `get_shard_cache_path(blob_path)` to choose a shard's destination.
The input is container-relative and decoded once. Return an absolute path for
any data root, a relative path beneath `cache.cache_dir`, or `None` for the
default hashed filename. Keep the mapping stable and distinguish blobs sharing
cache state; concrete classes can incorporate account/container names as needed.

`create_shard_cache()` serves streaming, prefetching, and indexed paths. Direct
`AzureShardCache` consumers can supply `path_resolver(public_url)`; it receives
the encoded URL path with query and fragment removed. `cache.state_dir` selects
locks, pins, destination records, and size reservations independently of the
data root. When omitted, state remains beside `cache.cache_dir`. Processes
sharing destinations must share state, the size limit, and the mapping.

Locks and pins use resolved full destinations, so equal basenames in different
directories do not collide. Destination records bind files to SAS-free blob
identities; mapping another blob there fails before downloading. Eviction
accounts for registered files, including nested files and files outside
`cache_dir`. Default hashed files from older versions are adopted. Unrelated
files are not evicted unless explicitly registered. Finish older cache
processes before sharing their destinations with the new version.

Temporary payloads are written beside their destination for atomic promotion.
Disk admission checks that filesystem; the shared budget includes registered
shards and in-flight reservations. Abandoned partials are reclaimed once their
download lock is available. Cache hits update access metadata rather than tar
modification time, keeping core member indexes reusable.

### Indexed Tar Reading

Supply the selected weighted sources from `get_shard_sources(split)` once to
`cached_shard_sources(data)`. This training-process context downloads misses
and yields local paths with public shard URLs and logical source metadata.
Every selected file stays reserved until the context exits.

```python
from torch.utils.data import DataLoader

sources = wrapper.get_shard_sources("train")
with wrapper.cached_shard_sources(sources) as local_sources:
    dataset = wrapper.build_indexed_dataset(local_sources, "train")
    sampler = wrapper.build_batch_sampler(
        dataset, "train", batch_size=32, shuffle=True, drop_last=False
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=8,
        collate_fn=wrapper.collate_fn,
    )
    wrapper.reset_shard_progress({
        shard: count for shard, count in dataset.shard_totals.items() if count
    })
    with wrapper.create_shard_prefetcher() as prefetch:
        for current, upcoming in replacement_pairs:
            prefetch.plan(current, current=[current], upcoming=[upcoming])
        for batch in loader:
            if not batch:
                continue
            train_step(batch)
            wrapper.record_shard_consumption(batch["shard_id"])
            for shard, progress in wrapper.get_shard_progress().items():
                prefetch.advance(
                    shard, consumed=progress["consumed"], total=progress["total"]
                )
        # Use wait(shard) at the project's replacement boundary.
    dataset.close()
```

Enable `dataset.track_shard_progress` and `dataset.prefetch`. The trainer owns
`replacement_pairs`, training, and iterator replacement. Plan keys here match
shard IDs, which default to container-relative source paths. Eight workers
are shared across all selected shards. Supply eligible counts or finite budgets
for filtering and repeated draws; unknown totals cannot trigger a fraction.

The example fully consumes a finite loader without persistent workers. On an
early stop or with persistent workers, stop those workers before releasing the
reservation context. Reserve new active shards before releasing old plans.
Create workers before the first `advance()` or `wait()` starts download threads.
Worker datasets carry paths and metadata, never lease contexts or controllers.

### Queued Shard Prefetch

`AzureStreamingTarShardWrapper.create_shard_prefetcher()` creates a controller
owned by the trainer. It uses the same cache as normal WebDataset reads.
Configure it under `dataset.prefetch`:

```yaml
dataset:
  prefetch:
    enabled: true                 # Default: false
    trigger_fraction: 0.5         # Inclusive range: 0 to 1
    max_concurrent_downloads: 2   # Positive integer; per controller
    max_pending_shards: 32        # Positive integer; distinct upcoming shards
```

Configuration is checked when the controller is created. A fraction of `0`
starts downloads on the first `advance()` call; `1` waits until the reported
budget is consumed. Enabling this block alone does not choose replacements or
track training progress. The project wrapper keeps its existing selection
method, and the trainer supplies the resulting plan and progress.

Select each slot's next shard before creating its plan. Paths are relative to
the blob container, as in `build_shard_sources()`. In this example, `slot_plans`
contains `current`, `upcoming`, and `sample_budget`; `consumed_by_slot` counts
trained samples for each slot.

```python
with wrapper.create_shard_prefetcher() as prefetch:
    for slot, plan in slot_plans.items():
        prefetch.plan(
            slot,
            current=[plan["current"]],
            upcoming=[plan["upcoming"]],
        )

    # Call after each training batch, using cumulative counts for this plan.
    for slot, consumed in consumed_by_slot.items():
        prefetch.advance(
            slot,
            consumed=consumed,
            total=slot_plans[slot]["sample_budget"],
        )

    # At a replacement boundary, before rebuilding the relevant data iterator:
    prefetch.wait(finished_slot, timeout=120)
    # Install exactly slot_plans[finished_slot]["upcoming"] using the
    # project's existing replacement method. Before releasing the old plan,
    # reserve the new active shard in its next plan, or open its reader.
    prefetch.release(finished_slot)
```

For whole-cycle replacement, put both shard sets in one plan and report
completed batches against the cycle's batch budget. Individual shard plans
cross the threshold independently and share a download queue. Requests for
the same shard share one download, including requests with renewed SAS tokens.
`plan()` reserves files without downloading them. Release a plan before reusing
its key.

Choose the upcoming shards once and use that selection at replacement time.
Prefetch leaves dataset membership, shuffle order, sampling weights, and
DataLoader iterators unchanged. Preserve shard identity in batches to report
per-shard consumption. For resampled streams, define a finite sample
budget or use cycle progress. DataLoader workers can read ahead of the trainer,
so lower the threshold if they reach the next shard before its download finishes.

`status()` returns public shard URLs mapped to `planned`, `queued`,
`downloading`, `waiting_for_space`, `ready`, `failed`, or `cancelled`.
`wait()` starts any remaining downloads for that plan immediately and returns
local paths after all complete. It surfaces a failed download; a timeout leaves
background work running. Progress is reported per shard, without byte-level
transfer percentages. A normal cache read also joins an in-flight download
through the shared file lock.

Keep the controller in the trainer and call its API from the thread that
created it. Start DataLoader worker processes before the first `advance()` or
`wait()` call starts the download threads. Do not store the controller on the
dataset wrapper, pass it to workers, or include it in checkpoints. Recreate
plans from the training state when resuming. Use the context manager or call
`close()` in `finally`; shutdown cancels queued work and waits for in-flight
requests to finish under the configured download retries and timeouts.

The concurrency limit is per controller. Ranks and workers sharing a local
cache coordinate downloads, disk reservations, and eviction with file locks;
different hosts have separate caches and limits. Supply only the shards that
the local training process needs. SAS URLs are generated when a plan is made,
so their configured lifetime must cover that plan.

### Cache Capacity and Reservations

`AzureShardCache` protects open WebDataset streams. Prefetch plans additionally
protect all their current and upcoming shards, even before download starts.
Release a plan only when its reservations are no longer needed; create the
next plan before releasing the previous one to preserve protection during
handoff. Opening a reader acquires an independent reservation. Overlapping
plans and readers keep the file protected until the last reservation exits.

The cache budget includes completed files and the full expected size of
in-flight downloads. Eviction removes the oldest unreserved files. A download
that cannot fit raises `CacheCapacityError` on the normal read path. Background
prefetch pauses that request in `waiting_for_space` and retries as space becomes
available; it still services other queued requests. `wait()` raises
`CacheCapacityError` if a required download is waiting for space, so an
undersized cache does not silently hang a cycle boundary.

Size the cache for the active set plus its upcoming replacements. A plan
protecting both sets cannot make progress if they do not fit together. Releasing
other finished plans can free space; otherwise increase `cache_size_gb` or plan
fewer simultaneous replacements. The queue admission limit includes planned,
queued, downloading, and completed upcoming shards until their plans release
them. Cancelled jobs that have not left the queue still occupy admission slots.

Reservations use operating-system file locks. Eviction reclaims stale pins and
download-size records left by an exited process. Use the same cache size for
all processes sharing a directory. Older package versions do not respect these
reservations, so use separate cache directories when running mixed versions.

For direct storage integration, import `AzureShardCache`, `ShardPrefetcher`, and
`CacheCapacityError` from `dl_azure.storage`. A direct `ShardPrefetcher(cache)`
is enabled by default and accepts authenticated URLs. `cache.ensure(url)`
downloads and validates a remote shard, then returns its local path. Hold
`cache.reserve(url)` while retaining or reading that path to prevent eviction.

## Frame Dataset Notes

The generic frame wrappers:

- return image tensors shaped by `height` and `width`
- optionally resize frames first with `resize_height` and `resize_width`
- optionally crop faces using metadata when `use_face_detection` is enabled
- accept `margin` as an int, a two-item list or tuple, or a
  `{height, width}` mapping

Frame metadata is resolved from the image path by replacing `Raw_Frames` or
`data/frames` with `data/metadata` and swapping the file extension for
`.json`. If the metadata file is missing or does not contain `bboxes`, the
wrapper falls back to the full frame.

## Multiframe Sampling Rules

The multiframe wrappers keep grouped frame paths sorted and then build one or
more multiframe samples per video.

Relevant config lives under `dataset.multiframe`:

- `mode`
- `num_frames`
- `frame_stride`

Sampling behavior:

- `mode: random` draws `num_frames` unique frames per sample and emits
  `len(video_frames) // num_frames` samples
- `mode: consecutive` walks the sorted frames in fixed windows of
  `num_frames`, using `num_frames + frame_stride` as the step size
- videos shorter than `num_frames` are skipped

Each generated sample keeps `paths` as the selected frame tuple and uses the
first selected frame as the representative `path` field for downstream
metadata-building logic.

## Executor Runtime Notes

The Azure executor:

- reads `azure-config.json` by default, or the configured
  `executor.azure_config_path`
- updates only a managed block in `.amlignore`
- preserves existing user-defined `.amlignore` content outside that block
- excludes `.env` files from the Azure submission context
- is intended for sweep submission rather than the local-only `dl-run` path

## Recommended Operational Pattern

- use `--dry-run` first
- keep Azure config files at the experiment repo root
- set `dataset.container_name` explicitly for streaming datasets
- prefer sweep submission over trying to force Azure through the local-only
  single-run CLI
