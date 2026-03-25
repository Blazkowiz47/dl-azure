# Technical: 3. Dataset Mounts and Runtime Notes

## Compute Dataset Roots

The generic compute dataset wrappers support three root resolution paths:

- explicit `dataset.root_dir`
- Azure ML input mounts via `AZURE_ML_INPUT_<input_name>`
- optional local fallback when the wrapper config allows it

That means project-specific datasets should pass either a concrete `root_dir`
or an `input_name` instead of hardcoding a single mounted directory name.

## Executor Runtime Notes

The Azure executor:

- reads `azure-config.json` by default, or the configured
  `executor.azure_config_path`
- updates only a managed block in `.amlignore`
- preserves existing user-defined `.amlignore` content outside that block
- is intended for sweep submission rather than the local-only `dl-run` path

## Recommended Operational Pattern

- use `--dry-run` first
- keep Azure config files at the experiment repo root
- set `dataset.container_name` explicitly for streaming datasets
- prefer sweep submission over trying to force Azure through the local-only
  single-run CLI
