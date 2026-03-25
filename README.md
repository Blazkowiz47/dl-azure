# dl-azure

Public Azure integration layer for `dl-core`.

`dl-azure` adds Azure ML execution, Azure storage helpers, and Azure-oriented
dataset wrappers on top of `dl-core`.

Install it directly or through the `dl-core[azure]` extra. The package is kept
separate so Azure-specific dependencies and scaffold wiring do not leak into
plain `dl-core` installations.

## Install

Current public validation releases are published on TestPyPI. Once the package
is promoted to PyPI, the same install forms below will work against the main
index.

PyPI install target:

```bash
pip install "dl-core[azure]"
```

Install the package directly:

```bash
pip install dl-azure
```

Current TestPyPI + `uv` projects should add both direct dependencies:

```bash
uv add "dl-core[azure]" dl-azure
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

## Quick Start

Install it into an experiment repository through the Azure extra:

```bash
uv add "dl-core[azure]" dl-azure
```

If the repository was scaffolded with `dl-init-experiment --with-azure`, the
experiment package will import `dl_azure` automatically so its executor
and generic dataset wrappers register at runtime, and the scaffold will also
create `azure-config.json`.

The Azure executor is sweep-oriented. Use `dl-sweep --dry-run` before the first
real submission in a new repository.

## What You Get

- the `azure` executor
- Azure storage helpers and AzCopy wrappers
- generic Azure dataset foundations:
  `AzureComputeWrapper`, `AzureStreamingWrapper`,
  `AzureComputeFrameWrapper`, `AzureStreamingFrameWrapper`,
  `AzureComputeMultiFrameWrapper`, and `AzureStreamingMultiFrameWrapper`
- `dl-init-experiment --with-azure` scaffold integration
- a managed `.amlignore` block that preserves user content while excluding
  common local-only outputs from Azure submissions

## Companion Packages

- [`dl-core`](https://github.com/Blazkowiz47/dl-core)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)
- [`dl-wandb`](https://github.com/Blazkowiz47/dl-wandb)

## Documentation

- [Documentation Index](https://github.com/Blazkowiz47/dl-azure/tree/main/readme)
- [GitHub Repository](https://github.com/Blazkowiz47/dl-azure)

## License

MIT. See [LICENSE](LICENSE).
