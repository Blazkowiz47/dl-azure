# Technical: 2. Executor Configuration

The Azure executor registers under the name `azure`.

## Example Config

```yaml
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

## Fields

- `name`
  - must be `azure`
- `compute_target`
  - Azure ML compute target name
- `environment_name`
  - Azure ML environment name
- `environment_version`
  - Azure ML environment version, defaults to `latest`
- `datastore_name`
  - optional datastore to mount into the job
- `process_count_per_node`
  - number of processes per node for distributed execution
- `dont_wait_for_completion`
  - if `true`, submit and return without waiting for each child job
- `retry_limit`
  - number of retry rounds for failed runs
- `azure_config_path`
  - optional path to the Azure workspace config file
  - defaults to `azure-config.json`

## Additional Inputs

The executor also reads:

- `azure-config.json` by default, or the configured `executor.azure_config_path`
- `AZURE_ACCESS_KEY` when generating SAS tokens for storage access

## Submission Model

The executor submits each generated sweep run as an Azure ML command job. The
parent process is a sweep orchestrator, while each child config becomes its own
Azure job.
