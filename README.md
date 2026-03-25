# deep-learning-azure

Public Azure integration layer for `deep-learning-core`.

`deep-learning-azure` adds Azure ML execution, Azure storage helpers, and
Azure-oriented dataset wrappers on top of `deep-learning-core`.

Install it directly or through the `deep-learning-core[azure]` extra. The
package is kept separate so Azure-specific dependencies and scaffold wiring do
not leak into plain `deep-learning-core` installations.

## Install

The package is now available on PyPI under the `deep-learning-azure` name.
TestPyPI remains available for validation flows.

PyPI install target:

```bash
pip install "deep-learning-core[azure]"
```

Install the package directly:

```bash
pip install deep-learning-azure
```

Current TestPyPI + `uv` projects should add both direct dependencies:

```bash
uv add "deep-learning-core[azure]" deep-learning-azure
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
uv add "deep-learning-core[azure]" deep-learning-azure
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
