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
  # parent_job_name: existing-azure-parent-job
  # command: python scripts/custom_job.py --config {config_path}
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
  - accepted jobs remain `running` in sweep tracking until completion is known
- `retry_limit`
  - number of retry rounds for failed runs, including raised submissions
  - applies to both one-worker and parallel sweeps; running and unknown jobs
    are never retried automatically
  - retries atomically claim failed runs so overlapping `--resume` commands do
    not submit the same child job twice
- `azure_config_path`
  - optional path to the Azure workspace config file
  - defaults to `azure-config.json`
- `parent_job_name`
  - optional existing Azure ML parent job name for child-job nesting
  - takes precedence over the parent job inferred from `--resume`
  - distinct from `tracking.parent_run_id`, which controls MLflow nesting
- `command`
  - optional Azure ML command override for each submitted child job
  - when omitted, the executor submits the default `python -m dl_core.worker ...`
  - supports `{config_path}`, `{run_name}`, `{run_index}`, `{run_number}`,
    `{tracking_context}`, and `{tracking_uri}`
  - prefer plain `python ...` commands over `uv run python ...` because the
    Azure ML environment already defines the runtime

## Additional Inputs

The executor also reads:

- `azure-config.json` by default, or the configured `executor.azure_config_path`
- `AZURE_ACCESS_KEY` on the submitting machine when generating a read-only SAS
  token for child jobs; the account key itself is not sent to Azure ML jobs

## Submission Model

The executor submits each generated sweep run as an Azure ML command job. The
parent process is a sweep orchestrator, while each child config becomes its own
Azure job. Child jobs receive the scoped storage token when one is configured.
The executor records accepted jobs as `running` and indeterminate jobs as
`unknown`; `--resume` skips both until their status is reconciled.
Ctrl+C leaves only ambiguous in-flight Azure submissions `unknown`; finished
submissions retain their recorded result, and parallel sweeps cancel jobs that
have not started.
If tracking fails after Azure accepts a job, submission stops and logs the job
ID for manual reconciliation instead of retrying it.

When `executor.command` is set, the executor still uses the same Azure ML job
submission flow, but swaps the child job command to the configured string.
