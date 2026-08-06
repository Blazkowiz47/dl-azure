# Welcome to the `dl-azure` Documentation

This documentation covers the Azure adapter layer that sits on top of
`dl-core`.

Current public release: `deep-learning-azure==0.0.21`, requiring
`deep-learning-core>=0.1.4,<0.2`.

## What's New in Development?

- Azure tar wrappers provide mounted paths or authenticated blob URLs to the
  optional WebDataset integration in `deep-learning-core`
- WebDataset handles grouped samples, buffered shuffling, on-demand caching,
  and distributed rank/worker shard splitting

## What's New in 0.0.21?

- mounted and streaming tar-shard wrappers provide Azure-backed access to the
  indexed grouped-sample contract in `deep-learning-core`
- streaming tar and index caches use atomic chunked downloads, process-safe
  locking, and Azure blob identity checks
- Azure storage concerns remain isolated from core tar reading and sampling

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
