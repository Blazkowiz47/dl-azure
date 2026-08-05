# Guide: 1. Wiring Azure Into an Experiment Repo

The Azure package is not a standalone training framework. It is an adapter
layer for an existing `deep-learning-core` experiment repository.

## Step 1: Start From a `deep-learning-core` Experiment Repo

Install the Azure extra and scaffold the repository:

```bash
uv add "deep-learning-core[azure]"
uv run dl-init --name my-exp --root-dir . --with-azure
```

Using `--with-azure` is recommended because the scaffold adds the direct
`deep-learning-azure` dependency, imports `dl_azure` from the experiment
package root, creates `azure-config.json`, and adds Azure output ignore rules.

## Step 2: Install the Packages

Inside the experiment repository:

```bash
uv sync
```

For sibling local development:

```bash
uv add --editable ../dl-core
uv add --editable ../dl-azure
```

## Step 3: Add Azure Sweep Executor Config

Put the Azure executor into the sweep config or a preset that the sweep uses:

```yaml
fixed:
  executor:
    name: azure
    compute_target: gpu-cluster
    environment_name: dl_lab
    environment_version: latest
    datastore_name: my-datastore
```

If you want Azure to run a custom script instead of the default
`dl_core.worker` entrypoint, add `executor.command`:

```yaml
fixed:
  executor:
    name: azure
    compute_target: gpu-cluster
    environment_name: dl_lab
    command: python scripts/preprocessing/fix_nested_frame_dirs.py --config {config_path}
```

Use plain `python ...` here rather than `uv run python ...`. The Azure ML
environment already provides the interpreter and installed packages.

## Step 4: Fill In `azure-config.json`

The scaffold creates `azure-config.json` in the repository root. Replace the
placeholder values before submission. If you need to keep that file elsewhere,
set `executor.azure_config_path` to the alternate location.

## Step 5: Choose the Dataset Wrapper and Path

Use one of the generic Azure dataset base wrappers as the parent for your
project-specific dataset wrapper:

- `AzureComputeWrapper` for mounted image datasets
- `AzureComputeFrameWrapper` for mounted frame datasets
- `AzureComputeMultiFrameWrapper` for mounted multiframe datasets
- `AzureStreamingWrapper` for direct blob-backed image datasets
- `AzureStreamingFrameWrapper` for direct blob-backed frame datasets
- `AzureStreamingMultiFrameWrapper` for direct blob-backed multiframe datasets
- `AzureComputeTarShardWrapper` for indexed tar shards on mounted inputs
- `AzureStreamingTarShardWrapper` for locally cached Azure tar blobs

For mounted Azure ML inputs, pass either:

- `dataset.root_dir` for an explicit local or mounted path
- `dataset.input_name` to resolve `AZURE_ML_INPUT_<input_name>`

If you also want the same config to work outside Azure ML, keep
`dataset.allow_local_fallback: true` and point `dataset.local_fallback_root`
at a compatible local dataset root.

For streaming datasets, set `dataset.container_name` and provide Azure storage
credentials through `azure-config.json` or inline dataset config fields such as
`account_name`.

A typical multiframe dataset block looks like this:

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
    mode: random
    num_frames: 5
    frame_stride: 0
```

Use `multiframe.mode: random` to sample unique frames per video, or
`multiframe.mode: consecutive` to walk through sorted frame windows. Frame
wrappers also support `resize_height`, `resize_width`, and `margin` when you
need control over resizing or face crop margins.

## Step 6: Dry-Run Before Submission

```bash
uv run dl-sweep experiments/lr_sweep.yaml --dry-run
```

This is especially important because the Azure executor updates a managed block
in `.amlignore` for submission hygiene. The dry-run gives you a safe place to
verify the generated job config first.

## Step 7: Submit

Once the dry-run output looks correct:

```bash
uv run dl-sweep experiments/lr_sweep.yaml
```

Today the Azure path is sweep-oriented. The local-only `dl-run` path is still
the normal single-run entrypoint. `executor.command` is the escape hatch when a
sweep-managed Azure submission needs to run something other than training.
