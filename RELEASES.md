# deep-learning-azure Release History

The main README shows only the latest release. This page preserves the
release-by-release changes.

## 0.0.21

- mounted and streaming tar-shard wrappers extend the indexed grouped-sample
  dataset contract from `deep-learning-core`
- streaming shards and sidecar indexes use chunked atomic caching with
  process-safe locks and Azure blob identity validation
- shared Azure authentication and blob discovery remain separate from the
  vendor-neutral tar indexing, reading, and sampling implementation
- the core compatibility floor moved to `deep-learning-core>=0.1.4,<0.2`

## 0.0.20

- the supported core range includes the architecture-free
  `deep-learning-core==0.1.0` trainer and registry boundary
- Azure execution, storage, datasets, callbacks, and scaffold behavior remain
  unchanged

## 0.0.19

- blob URLs support user-delegation SAS tokens with explicit validation and
  signing failures
- blob caches remain within their configured roots and authenticated container
  clients no longer share unsafe pooled state
- AzCopy runs without a shell and receives retry concurrency through its child
  process environment
- Azure MLflow workspace discovery uses the Azure ML v2 client without the
  legacy `azureml-core` dependency
- generated repositories ignore Azure output/log directories and submissions
  exclude local environment files
- the core compatibility floor moved to `deep-learning-core>=0.0.26,<0.1`

## 0.0.18

- the core compatibility floor moved to `deep-learning-core>=0.0.25,<0.1`
- Azure execution, storage helpers, dataset wrappers, and scaffold integration
  remained in the companion package rather than the core runtime

Structured release notes begin with 0.0.18. Earlier package history remains
available through the repository's Git history.
