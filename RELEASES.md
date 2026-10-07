# deep-learning-azure Release History

The main README shows only the latest release. This page preserves the
release-by-release changes.

## 0.0.29

- initial indexed tar staging uses bounded concurrent downloads, configured
  with `azure.download.max_concurrent_files` (default `4`; `1` is sequential)
- OS-backed slots share the staging limit across processes using the same
  cache state and configuration; source order and full-selection pins remain
  intact, including custom destinations
- failed or interrupted batches stop scheduling new work, wait for active
  transfers, and release reservations and slots during cleanup
- training prefetch and per-file SDK/AzCopy concurrency retain their own
  settings; Mobai eligibility and checksum preparation are outside this change
- generated Azure defaults expose the staging limit; requires
  `deep-learning-core>=0.1.12,<0.2`

## 0.0.28

- generic blob-to-file downloads and the shard cache share an AzCopy-first
  downloader with SDK fallback; missing executables are remembered per process,
  while transfer and SAS-signing failures affect only the current file
- project defaults and `dataset.azure` settings merge recursively, covering
  storage, downloads, cache, and prefetch; flat configs remain supported
- SDK range concurrency, connection pools, chunk sizes, transport read buffers,
  content validation, and timeouts are configurable; file downloads use
  `readinto()` for parallel ranges
- transfers verify source consistency, size, and stored MD5 when available;
  the cache keeps its tar validation, atomic writes, path hooks, and reservations
- scaffolding writes nested Azure defaults and preserves existing config formats
- requires `deep-learning-core>=0.1.12,<0.2`; no core version bump is needed

## 0.0.27

- concrete tar wrappers choose destinations through a public path hook;
  a shared cache factory serves streaming reads, prefetching, and indexed paths
- destination records support nested paths and arbitrary data roots, with
  process-safe pins, duplicate-destination checks, and capacity-aware eviction
- temporary downloads use their destination filesystem; access tracking leaves
  tar modification time intact so core indexes remain reusable
- `cached_shard_sources()` reserves local files for the opt-in core indexed
  reader; trainers can feed per-shard progress to the existing queue
- requires `deep-learning-core>=0.1.12,<0.2`

## 0.0.26

- background downloads support per-shard and whole-cycle replacement plans,
  with a configurable consumption threshold that defaults to 0.5
- bounded concurrency, duplicate-request handling, progress status, and explicit
  shutdown are available through the trainer-owned shard prefetch controller
- cache reservations protect active and upcoming shards across processes and
  account for in-flight download sizes; stale reservations are recovered
- project trainers supply replacement plans and progress; prefetch is disabled
  by default
- requires `deep-learning-core>=0.1.9,<0.2`; no core version bump is needed

## 0.0.25

- Azure job and tracking names reuse the generated run config's `runtime.name`
- requires `deep-learning-core>=0.1.9,<0.2`

## 0.0.24

- Azure sweep submissions are claimed consistently, retries preserve
  indeterminate jobs, and interrupted parallel sweeps keep completed results
- child jobs use scoped storage credentials; in-place scaffolding preserves
  existing dataset files and face-crop caches include crop settings
- requires released `deep-learning-core>=0.1.8,<0.2` for sweep status hooks
- development PyTorch requirement is `torch>2.3` without an upper cap

## 0.0.23

- streaming tar downloads retry transient whole-shard failures with backoff
- shard caches use atomic writes, process-safe locks, and content checks
- tar cache capacity uses `cache_size_gb` with a 3000 GB default

## 0.0.22

- Azure tar wrappers provide mounted paths or authenticated blob URLs to the
  optional WebDataset integration in `deep-learning-core`
- project wrappers can dynamically build weighted shard sources before Azure
  mount resolution or SAS authentication
- WebDataset handles grouped samples, buffered shuffling, on-demand caching,
  and distributed rank/worker shard splitting

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
