# dl-azure

Public Azure integration layer for `dl-core`.

`dl-azure` adds Azure ML execution, Azure storage helpers, and Azure-oriented
dataset wrappers on top of `dl-core`.

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

## Documentation

- [Documentation Index](./readme/README.md)
- [TLDR: Install and Submit](./readme/tldr/1_install_and_submit.md)
- [Guide: Wiring Azure Into an Experiment Repo](./readme/guide/1_wiring_azure_into_an_experiment_repo.md)
- [Technical: Executor Configuration](./readme/technical/2_executor_configuration.md)
