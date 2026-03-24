# Technical: 1. Package Scope

`dl-azure` is the public Azure adapter package for `dl-core`.

## What It Adds

- `AzureComputeExecutor`
- Azure storage helpers
- AzCopy helpers
- Azure dataset wrappers such as `azure_compute_pad`

## What It Assumes

- `dl-core` is already installed
- the experiment repo imports `dl_azure` so registration happens
- Azure configuration is available locally at submission time

## What It Does Not Replace

- it does not replace `dl-core` registries or base abstractions
- it does not replace the experiment repository
- it does not currently provide a separate CLI

The intended call path remains `dl-core` entrypoints such as `dl-sweep`.
