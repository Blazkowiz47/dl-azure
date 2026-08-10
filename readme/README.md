# Welcome to the `dl-azure` Documentation

This documentation covers the Azure adapter layer that sits on top of
`dl-core`.

Current public release: `deep-learning-azure==0.0.22`, requiring
`deep-learning-core>=0.1.5,<0.2`.

## What's New in Development?

- Azure streaming tar downloads retry transient whole-shard failures with
  exponential backoff and clean partial files before each attempt
- process-safe shard locks prevent DataLoader workers and distributed ranks on
  the same host from downloading the same shard concurrently
- tar cache capacity uses `cache_size_gb`, defaulting to 3000 GB when enabled

## What's New in 0.0.22?

- Azure tar wrappers provide mounted paths or authenticated blob URLs to the
  optional WebDataset integration in `deep-learning-core`
- project wrappers can dynamically build weighted shard sources before Azure
  mount resolution or SAS authentication
- WebDataset handles grouped samples, buffered shuffling, on-demand caching,
  and distributed rank/worker shard splitting

- [Release History](../RELEASES.md)

## Related Packages

- [`dl-core`](https://github.com/Blazkowiz47/dl-core)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)

## Structure

### 1. [`tldr/`](./tldr/1_install_and_submit.md)

Go here if you need the shortest path to a dry-run or submission.

- [Install and Submit](./tldr/1_install_and_submit.md)

### 2. [`guide/`](./guide/1_wiring_azure_into_an_experiment_repo.md)

Go here if you want the package wired into a real experiment repository step by
step.

- [Wiring Azure Into an Experiment Repo](
  ./guide/1_wiring_azure_into_an_experiment_repo.md
  )

### 3. [`technical/`](./technical/1_package_scope.md)

Go here if you need the current implementation details, config fields, or
runtime notes.

- [Package Scope](./technical/1_package_scope.md)
- [Executor Configuration](./technical/2_executor_configuration.md)
- [Dataset Mounts and Limitations](./technical/3_dataset_mounts_and_limitations.md)
