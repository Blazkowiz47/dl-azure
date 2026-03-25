# Welcome to the `dl-azure` Documentation

This documentation covers the Azure adapter layer that sits on top of
`dl-core`.

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
operational limitations.

- [Package Scope](./technical/1_package_scope.md)
- [Executor Configuration](./technical/2_executor_configuration.md)
- [Dataset Mounts and Limitations](./technical/3_dataset_mounts_and_limitations.md)
