# Guide: 1. Wiring Azure Into an Experiment Repo

The Azure package is not a standalone training framework. It is an adapter
layer for an existing `dl-core` experiment repository.

## Step 1: Start From a `dl-core` Experiment Repo

Create it from `dl-core`:

```bash
uv add "dl-core[azure]"
uv run dl-init-experiment --name my-exp --root-dir . --with-azure
```

Using `--with-azure` is recommended because the scaffold switches the generated
dependency to `dl-core[azure]`, imports `dl_azure` from the experiment package
root, and creates `azure-config.json`.

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

## Step 4: Fill In `azure-config.json`

The scaffold creates `azure-config.json` in the repository root. Replace the
placeholder values before submission.

## Step 5: Choose the Dataset Path

Use one of the generic Azure dataset base wrappers as the parent for your
project-specific dataset wrapper. For mounted Azure ML inputs, pass either:

- `dataset.root_dir` for an explicit local or mounted path
- `dataset.input_name` to resolve `AZURE_ML_INPUT_<input_name>`

The compute wrappers read from that resolved root directory directly.

## Step 6: Dry-Run Before Submission

```bash
uv run dl-sweep --sweep experiments/lr_sweep.yaml --dry-run
```

This is especially important because the current executor still contains some
legacy `.amlignore` logic tied to the old repository layout.

## Step 7: Submit

Once the dry-run output looks correct:

```bash
uv run dl-sweep --sweep experiments/lr_sweep.yaml
```

Today the Azure path is sweep-oriented. The local-only `dl-run` path is still
the normal single-run entrypoint.
