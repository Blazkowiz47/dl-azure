# Technical: 1. Package Scope

`deep-learning-azure` is the public Azure adapter package for
`deep-learning-core`.

## What It Adds

- `AzureComputeExecutor`
- Azure storage helpers
- AzCopy helpers
- Generic Azure dataset base wrappers for compute, streaming, frame, and
  multiframe use cases

AzCopy commands are always invoked as argument lists without a shell. Retry
concurrency is supplied through the child-process environment so local paths
and blob names remain literal command arguments.

## What It Assumes

- `deep-learning-core` is already installed
- the experiment repo imports `dl_azure` so registration happens
- Azure configuration is available locally at submission time

## What It Does Not Replace

- it does not replace `deep-learning-core` registries or base abstractions
- it does not replace the experiment repository
- it does not currently provide a separate CLI

The intended call path remains the framework entrypoints such as `dl-sweep`.
