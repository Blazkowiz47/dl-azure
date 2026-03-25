# dl-azure

Public Azure integration layer for `dl-core`.

`dl-azure` adds Azure ML execution, Azure storage helpers, and Azure-oriented
dataset wrappers on top of `dl-core`.

Install it directly or through the `dl-core[azure]` extra. The package is kept
separate so Azure-specific dependencies and scaffold wiring do not leak into
plain `dl-core` installations.

## Install

Install from PyPI:

```bash
pip install "dl-core[azure]"
```

Install the package directly:

```bash
pip install dl-azure
```

## Scope

- Azure ML executor
- Azure storage helpers and AzCopy wrappers
- Azure dataset wrappers
- Azure experiment scaffold integration through `dl-init-experiment`

## Out Of Scope

- Generic trainer, dataset, and metric abstractions
- Public framework defaults
- Concrete experiment repositories

## Current State

This package is usable, but it still carries some legacy assumptions that are
important to know:

- it expects `azure-config.json` in the current working directory
- `.amlignore` handling still assumes the older `lab/users/...` style layout
- Azure submission is currently sweep-oriented; `dl-run` remains local-only

Use `--dry-run` first when wiring a new experiment repository to Azure.

## Quick Start

Install it into an experiment repository through the Azure extra:

```bash
uv add "dl-core[azure]"
```

If the repository was scaffolded with `dl-init-experiment --with-azure`, the
experiment package will import `dl_azure` automatically so its executor
and generic dataset wrappers register at runtime, and the scaffold will also
create `azure-config.json`.

## What You Get

- the `azure` executor
- Azure storage helpers and AzCopy wrappers
- generic Azure dataset foundations:
  `AzureComputeWrapper`, `AzureStreamingWrapper`,
  `AzureComputeFrameWrapper`, `AzureStreamingFrameWrapper`,
  `AzureComputeMultiFrameWrapper`, and `AzureStreamingMultiFrameWrapper`
- `dl-init-experiment --with-azure` scaffold integration

## Companion Packages

- [`dl-core`](https://github.com/Blazkowiz47/dl-core)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)

## Documentation

- [Documentation Index](https://github.com/Blazkowiz47/dl-azure/tree/main/readme)
- [GitHub Repository](https://github.com/Blazkowiz47/dl-azure)
