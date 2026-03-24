# Technical: 3. Dataset Mounts and Limitations

The current Azure dataset and executor implementation still carries a few
important assumptions from the legacy repository.

## `azure_compute_pad`

The `azure_compute_pad` dataset wrapper expects an Azure ML input named
`dataset_path`. At runtime it reads the mount from:

```text
AZURE_ML_INPUT_dataset_path
```

and then uses:

```text
${AZURE_ML_INPUT_dataset_path}/data
```

as the base directory.

If that environment variable is missing, it falls back to local paths such as
`./data`.

## Executor Limitations

The current Azure executor still:

- expects `azure-config.json` in the working directory
- updates `.amlignore`
- assumes legacy `lab/users/...` paths when trimming `.amlignore`

That means the package is already decoupled physically, but not yet fully
generalized operationally.

## Recommended Operational Pattern

- use `--dry-run` first
- keep Azure config files at the experiment repo root
- treat `.amlignore` behavior as something to verify on each new repository
- prefer sweep submission over trying to force Azure through the local-only
  single-run CLI
