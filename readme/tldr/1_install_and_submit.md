# TLDR: Install and Submit

`dl-azure` extends `dl-core`. The normal path is:

1. install `dl-core[azure]` into the experiment repository
2. make sure `dl_azure` is imported so registrations happen
3. put Azure executor config into the sweep
4. fill in `azure-config.json`
5. set `dataset.container_name` for streaming datasets
6. run a dry-run first

## Install

```bash
uv add "dl-core[azure]"
```

For local development against sibling checkouts:

```bash
uv add --editable ../dl-core
uv add --editable ../dl-azure
```

## Ensure Registrations Load

If the experiment repository was scaffolded with `--with-azure`, this is
already handled. Otherwise import `dl_azure` from the experiment package
root.

## Add Azure Executor Config

Example sweep config block:

```yaml
fixed:
  executor:
    name: azure
    compute_target: gpu-cluster
    environment_name: dl_lab
    environment_version: latest
    datastore_name: my-datastore
    process_count_per_node: 1
    dont_wait_for_completion: false
    retry_limit: 0
```

## Dry-Run First

```bash
uv run dl-sweep --sweep experiments/lr_sweep.yaml --dry-run
```

Optional CLI overrides:

```bash
uv run dl-sweep --sweep experiments/lr_sweep.yaml \
  --compute gpu-cluster \
  --environment dl_lab \
  --dry-run
```
