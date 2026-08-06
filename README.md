# deep-learning-azure

Public Azure integration layer for `deep-learning-core`.

`deep-learning-azure` adds Azure ML execution, Azure storage helpers, and
Azure-oriented dataset wrappers on top of `deep-learning-core`.

Current release: `deep-learning-azure==0.0.21`.
Requires `deep-learning-core>=0.1.4,<0.2`.

## What's New in Development?

- Azure tar wrappers now provide mounted paths or authenticated blob URLs to
  the optional WebDataset integration in `deep-learning-core`
- project Azure wrappers can override `build_shard_sources(split)` to construct
  blob paths and source weights dynamically before backend resolution
- WebDataset splits shards between ranks and workers before opening the stream;
  sidecar indexes and the custom Azure tar cache are no longer required
- Azure streaming tar support can use WebDataset's on-demand cache when
  `dataset.cache.cache_dir` is configured

## What's New in 0.0.21?

- mounted and streaming tar-shard wrappers extend the indexed grouped-sample
  dataset contract from `deep-learning-core`
- streaming shards and sidecar indexes use chunked atomic caching with
  process-safe locks and Azure blob identity validation
- shared Azure authentication and blob discovery remain separate from the
  vendor-neutral tar indexing, reading, and sampling implementation
- the core compatibility floor is now `deep-learning-core>=0.1.4,<0.2`

Previous versions are recorded in the [release history](RELEASES.md).

Install it directly or through the `deep-learning-core[azure]` extra. The
package is kept separate so Azure-specific dependencies and scaffold wiring do
not leak into plain `deep-learning-core` installations.

## Install

Install from PyPI through the core extra:

```bash
pip install "deep-learning-core[azure]"
```

Install the package directly:

```bash
pip install "deep-learning-azure[webdataset]"
```

Install in a `uv` project:

```bash
uv add "deep-learning-core[azure]"
```

## Scope

- Azure ML executor
- Azure storage helpers and AzCopy wrappers
- Azure dataset wrappers
- Azure experiment scaffold integration through `dl-init --with-azure`

## Out Of Scope

- Generic trainer, dataset, and metric abstractions
- Public framework defaults
- Concrete experiment repositories

## Quick Start

Install it into an experiment repository through the Azure extra:

```bash
uv add "deep-learning-core[azure]"
```

If the repository was scaffolded with `dl-init --with-azure`, the
experiment package will import `dl_azure` automatically so its executor
and generic dataset wrappers register at runtime, and the scaffold will also
create `azure-config.json`.

The Azure executor is sweep-oriented. Use
`uv run dl-sweep experiments/lr_sweep.yaml --dry-run` before the first real
submission in a new repository.

If you need Azure ML to run a custom script instead of the default
`python -m dl_core.worker ...` command, set `executor.command` in the sweep
config. Prefer plain `python ...` commands because the Azure ML environment
already controls the runtime. The command string also supports placeholders
such as `{config_path}` and `{run_name}`.

Concrete experiment flow:

```bash
uv init
uv add deep-learning-azure
uv run dl-init --root-dir . --with-azure
uv run dl-core add dataset AzureSeq --base azure_compute_multiframe
uv run dl-sweep experiments/lr_sweep.yaml --dry-run
```

Example custom Azure submission:

```yaml
fixed:
  executor:
    name: azure
    compute_target: gpu-cluster
    environment_name: dl_lab
    environment_version: latest
    # parent_job_name: existing-azure-parent-job
    command: python scripts/preprocessing/fix_nested_frame_dirs.py --config {config_path}
```

Tracker naming defaults to the repository root name. If you want Azure job
submission and Azure MLflow to use a different destination name, set
`tracking.experiment_name` in your sweep config.
Use `executor.parent_job_name` when child Azure jobs should nest under an
existing Azure ML parent job; keep `tracking.parent_run_id` for MLflow nesting.

Azure submissions automatically rewrite the default local `runtime.output_dir`
from `artifacts` to `outputs/artifacts` inside the remote job. That keeps
checkpoints, plots, metrics, and other run files under Azure ML's managed
output directory without changing the local default artifact layout.

When you analyze an Azure-backed sweep with `dl-analyze`, the Azure metrics
source fetches only the metric histories requested on the CLI, for example:

```bash
uv run dl-analyze --sweep experiments/lr_sweep.yaml \
  --metric test/eer --mode min \
  --metric test/accuracy --mode max \
  --rank-method rank-sum
```

Those fetched metric histories are cached in `analysis_cache.json` next to
`sweep_tracking.json`. Use `--force` to refresh them.

If you want the tracked Azure job outputs locally after the sweep finishes, run:

```bash
uv run dl-sync --sweep experiments/lr_sweep.yaml --artifacts
```

That downloads the Azure job bundle for each tracked run and patches
`sweep_tracking.json` with the resolved local artifact paths.

Concrete dataset scaffold examples:

```bash
uv run dl-core add dataset AzureImages --base azure_compute
uv run dl-core add dataset AzureFrames --base azure_compute_frame
uv run dl-core add dataset AzureSeq --base azure_compute_multiframe
uv run dl-core add dataset AzureStream --base azure_streaming
uv run dl-core add dataset AzureStreamSeq --base azure_streaming_multiframe
uv run dl-core add dataset AzureTar --base azure_streaming_tar
```

## Dataset Wrapper Notes

Use the compute wrappers when the dataset is already mounted into the Azure ML
job or available locally through a compatible directory layout:

- `AzureComputeWrapper`
- `AzureComputeFrameWrapper`
- `AzureComputeMultiFrameWrapper`
- `AzureComputeTarShardWrapper`

Compute wrappers resolve the dataset root in this order:

- `dataset.root_dir`
- `AZURE_ML_INPUT_<input_name>`
- `dataset.local_fallback_root` when `dataset.allow_local_fallback` is `true`

Use the streaming wrappers when you want to read directly from blob storage
instead of relying on an Azure ML input mount:

- `AzureStreamingWrapper`
- `AzureStreamingFrameWrapper`
- `AzureStreamingMultiFrameWrapper`
- `AzureStreamingTarShardWrapper`

Streaming wrappers require `dataset.container_name` and an Azure storage config
that provides `account_name`, either in `azure-config.json` or inline in the
dataset config.

`AzureClientService.get_blob_sas_url()` issues a user-delegation SAS through
`DefaultAzureCredential`; it never silently returns an unsigned URL. The active
identity therefore needs permission to request a user delegation key and the
required Blob Data role for the requested operation.

Frame wrappers share a few image-specific settings:

- `height` / `width` for the output tensor shape
- `resize_height` / `resize_width` for pre-augmentation resizing
- `use_face_detection` to enable metadata-driven face crops
- `margin` as an int, two-item sequence, or `{height, width}` mapping

If you enable `face_detected_and_resized_cache`, processed frame images are
stored in the wrapper cache when a cache backend is available. That is most
useful for the streaming frame wrappers, where blob reads can be cached locally.

Tar shard wrappers use the optional WebDataset integration from `dl-core`.
Compute wrappers provide mounted shard paths. Streaming wrappers provide
user-delegation SAS URLs, so WebDataset can divide shards between ranks and
workers before opening them. Project transforms receive grouped member bytes
through `file_dict["members"]`.

For project-specific discovery and weighting, override
`build_shard_sources(split)`. Return logical mounted paths in compute wrappers
or logical container-relative blob paths in streaming wrappers; the parent
wrapper still performs mount resolution or SAS authentication.

```yaml
dataset:
  name: my_azure_tar_dataset
  container_name: datasets
  auto_split: false
  shards:
    train:
      - path: train/attack-000.tar
        group: attack
      - path: train/real-000.tar
        group: real
  required_extensions: [png, json]
  persistent_workers: true
  cache:
    cache_dir: /mnt/localssd/dl-azure
    cache_size: 500000000000
  webdataset:
    shard_shuffle: 100
    sample_shuffle: 10000
    resampled:
      train: true
      validation: false
      test: false
```

Multiframe wrappers add one `multiframe` block:

```yaml
dataset:
  name: AzureSeq
  input_name: dataset_path
  allow_local_fallback: true
  local_fallback_root: data/my_dataset
  height: 224
  width: 224
  use_face_detection: true
  face_detected_and_resized_cache: true
  multiframe:
    mode: consecutive
    num_frames: 5
    frame_stride: 2
```

`multiframe.mode: random` draws `num_frames` unique frames per sample.
`multiframe.mode: consecutive` walks each video in fixed windows and uses
`frame_stride` to skip frames between windows. Videos with fewer than
`num_frames` frames are skipped.

## What You Get

- the `azure` executor
- Azure storage helpers and AzCopy wrappers
- generic Azure dataset foundations:
  `AzureComputeWrapper`, `AzureStreamingWrapper`,
  `AzureComputeFrameWrapper`, `AzureStreamingFrameWrapper`,
  `AzureComputeMultiFrameWrapper`, `AzureStreamingMultiFrameWrapper`,
  `AzureComputeTarShardWrapper`, and `AzureStreamingTarShardWrapper`
- `dl-init --with-azure` scaffold integration
- a managed `.amlignore` block that preserves user content while excluding
  common local-only outputs and environment files from Azure submissions
- Azure job output routing to `outputs/artifacts` for automatic artifact
  persistence in Azure ML

## Companion Packages

- [`dl-core`](https://github.com/Blazkowiz47/dl-core)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)

## Documentation

- [Documentation Index](https://github.com/Blazkowiz47/dl-azure/tree/main/readme)
- [GitHub Repository](https://github.com/Blazkowiz47/dl-azure)

## License

MIT. See [LICENSE](LICENSE).
