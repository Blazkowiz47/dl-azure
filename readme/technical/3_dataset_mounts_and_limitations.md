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
face-cropped frames in the cache as a second-level optimization.

Blob cache paths are encoded and kept beneath the configured cache directory,
including blob names with absolute or parent-like path segments. Cache
statistics and cleanup include files in the full hierarchical layout. Azure
container-client pooling is scoped to one authenticated client service so
credentials are never mixed through a process-wide cache.

## Indexed Tar Shards

`AzureComputeTarShardWrapper` resolves relative `.tar` paths under the normal
compute root. `AzureStreamingTarShardWrapper` lists or accepts blob paths and
materializes each selected tar plus its optional `.idx.json` sidecar beneath
`dataset.cache.cache_dir/shards`.

Streaming shard downloads are chunked into a temporary file and atomically
renamed. A per-blob lock prevents DataLoader ranks or processes sharing one
cache directory from publishing partial files. Cache metadata records the
remote ETag, content length, and version ID when present.

Only uncompressed `.tar` shards support indexed random access. Each DataLoader
worker owns its tar handles; ranks can share the same cached file on disk but do
not share Python file objects. Exact rank-local group balance is configured by
the `dl-core` tar batch sampler rather than by Azure blob discovery.

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
