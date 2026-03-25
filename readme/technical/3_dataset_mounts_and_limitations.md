# Technical: 3. Dataset Mounts and Limitations

The current Azure dataset and executor implementation still carries a few
important assumptions from the legacy repository.

## Compute Dataset Roots

The generic compute dataset wrappers support three root resolution paths:

- explicit `dataset.root_dir`
- Azure ML input mounts via `AZURE_ML_INPUT_<input_name>`
- optional local fallback when the wrapper config allows it

That means project-specific datasets should pass either a concrete `root_dir`
or an `input_name` instead of hardcoding a single mounted directory name.

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
