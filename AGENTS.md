<agent_spec>
  <title>dl-core Repository Guidelines</title>

  <project_structure>
    <rule>Repository root includes <code>.github/</code>, <code>src/</code>, <code>tests/</code>, <code>readme/</code>, <code>README.md</code>, <code>pyproject.toml</code>, and <code>uv.lock</code>.</rule>
    <rule><code>src/dl_core/</code> stores the reusable framework package.</rule>
    <rule><code>src/dl_core/core/</code> stores base abstractions and registries.</rule>
    <rule><code>src/dl_core/templates/</code> stores bundled scaffold files copied into generated experiment repositories.</rule>
    <rule><code>tests/</code> stores repository-level <code>pytest</code> coverage for scaffolding, CLI behavior, and local components.</rule>
    <rule><code>readme/</code> stores extended user and technical documentation.</rule>
    <rule><code>.github/workflows/</code> stores CI and publishing workflows.</rule>
    <rule><code>dist/</code> contains generated build artifacts; treat it as output, not source.</rule>
  </project_structure>

  <command_policy>
    <rule>Run commands from the repository root unless a narrower directory is clearly better for the task.</rule>
    <rule>Prefer fast search commands: <code>rg</code>, <code>rg --files</code>.</rule>
    <rule>Use non-destructive commands by default.</rule>
    <rule>Prefer repository-standard <code>uv</code> commands for Python execution, validation, and build steps.</rule>
    <rule>Standard validation commands are <code>uv run --extra dev pytest</code>, <code>uv run python -m compileall src/dl_core</code>, and <code>uv build --no-sources</code>.</rule>
    <rule>Do not invent alternate environment-management flows when <code>uv</code> already covers the task.</rule>
  </command_policy>

  <development_rules>
    <rule>Type hints are required for Python code.</rule>
    <rule>Public modules, public APIs, and CLI entry points should include concise docstrings.</rule>
    <rule>Match the existing style: 4-space indentation, <code>snake_case</code> functions and modules, <code>PascalCase</code> classes, and short module docstrings.</rule>
    <rule>Use f-strings for string formatting.</rule>
    <rule>Keep functions focused and avoid unnecessary nesting.</rule>
    <rule>No formatter or linter configuration is defined in <code>pyproject.toml</code>; mirror nearby files instead of re-styling code arbitrarily.</rule>
  </development_rules>

  <architecture_rules>
    <rule>Keep <code>dl-core</code> reusable and vendor-neutral; company-specific integrations belong outside the core package.</rule>
    <rule>Preserve the registry-driven extension model in <code>src/dl_core/core/registry.py</code> and related package exports.</rule>
    <rule>Keep scaffold logic in <code>src/dl_core/init_experiment.py</code> aligned with the bundled files in <code>src/dl_core/templates/</code>.</rule>
    <rule>Keep generated config conventions consistent: top-level <code>accelerator</code>, <code>dataset</code>, singular <code>trainer</code>, and plural component sections such as <code>models</code>, <code>optimizers</code>, <code>criterions</code>, <code>metric_managers</code>, and <code>callbacks</code>.</rule>
    <rule>Use <code>dataset.classes</code> for class ordering, and keep sweep dotted paths aligned with the generated config structure.</rule>
    <rule>Do not introduce experiment-specific assumptions into the library when the behavior belongs in a generated consumer repository.</rule>
  </architecture_rules>

  <execution_policy>
    <rule>Never run training jobs, sweeps, worker processes, or publish workflows unless explicitly requested.</rule>
    <rule>Prefer targeted validation for the files you changed; full test runs are appropriate for code, packaging, or release work.</rule>
    <rule>If validation was not run, state it explicitly in the final response.</rule>
    <rule>Do not push commits or trigger manual GitHub Actions unless the user asked for that outcome.</rule>
  </execution_policy>

  <editing_policy>
    <rule>Keep edits minimal and task-scoped.</rule>
    <rule>Do not refactor unrelated code.</rule>
    <rule>Prefer <code>apply_patch</code> for focused, reviewable edits.</rule>
    <rule>Do not overwrite user changes or revert unrelated work in a dirty tree.</rule>
    <rule>When changing scaffolded files or config conventions, update both the generator and the bundled templates together.</rule>
  </editing_policy>

  <parallel_agent_policy>
    <rule>Assume multiple agents may edit in parallel.</rule>
    <rule>If a file changes while being edited, re-read the latest content and integrate safely.</rule>
    <rule>Only stage or commit the specific changes created for the current task.</rule>
    <rule>Before any commit, verify that staged changes match the intended scope.</rule>
  </parallel_agent_policy>

  <versioning_policy>
    <rule>When bumping the package version, update <code>pyproject.toml</code>, <code>src/dl_core/__init__.py</code>, and the scaffold dependency floor in <code>src/dl_core/init_experiment.py</code>.</rule>
    <rule>Unless the user explicitly requests otherwise, keep pre-release version bumps on the same alpha line by incrementing the alpha suffix, for example <code>0.0.1a0 -&gt; 0.0.1a1</code>; only move to the next stable version such as <code>0.0.2</code> after the release line is considered stable and tested.</rule>
    <rule>If the version bump affects package metadata or the local project entry in <code>uv.lock</code>, refresh the lockfile rather than editing it by hand.</rule>
    <rule>Before any publish action, make sure version references are internally consistent and validation has passed.</rule>
    <rule>Use concise release commits such as <code>release: bump dl-core to 0.1.8</code>.</rule>
  </versioning_policy>

  <github_actions_policy>
    <rule>Use <code>gh</code> to inspect workflows and runs when GitHub Actions work is requested.</rule>
    <rule>The repository workflows are <code>CI</code>, <code>Publish TestPyPI</code>, and <code>Publish</code>.</rule>
    <rule>Prefer checking existing runs with <code>gh run list</code>, <code>gh run view</code>, and <code>gh run watch</code> before dispatching anything new.</rule>
    <rule>When the user says <code>publish</code>, push the current commits and trigger <code>Publish TestPyPI</code> by default.</rule>
    <rule>Only trigger the <code>Publish</code> workflow when the user explicitly says <code>publish on pypi</code> or clearly requests a PyPI release.</rule>
    <rule>After dispatching a workflow with <code>gh workflow run</code>, monitor it and report the final status and run identifier or URL.</rule>
    <rule>Do not cancel, rerun, or approve workflows unless the user explicitly asks for that action.</rule>
  </github_actions_policy>

  <jira_policy>
    <rule>Use <code>acli</code> for Jira issue lookup, creation, and status tracking when Jira work is requested.</rule>
    <rule>Before starting substantial new implementation work, first create appropriately nested Jira work items that match the repository task hierarchy.</rule>
    <rule>When work spans multiple steps or components, create parent and child Jira tasks instead of tracking everything in a single flat issue.</rule>
    <rule>Do not create Jira tasks for very small edits, narrow documentation changes, or other low-overhead work unless the user explicitly asks for Jira tracking.</rule>
    <rule>When creating a Jira task, include a concise summary, repository context, affected paths, and clear acceptance criteria.</rule>
    <rule>When tracking work against Jira, report the issue key and the exact status change or comment that was added.</rule>
    <rule>After completing a task, transition the corresponding Jira issue to <code>Done</code>.</rule>
    <rule>If the Jira project, issue type, assignee, or workflow step is unclear, inspect existing issues first instead of guessing.</rule>
  </jira_policy>

  <git_policy>
    <rule>After every completed repository change, stage the task-scoped files and create a concise local git commit.</rule>
    <rule>Do not push commits by default; pushing is reserved for an explicit <code>publish</code> request.</rule>
    <rule>Use clear, imperative, prefix-based commit messages such as <code>docs: update contributor guide</code>, <code>test: add scaffold smoke coverage</code>, or <code>release: bump dl-core to 0.1.8</code>.</rule>
    <rule>Keep pull requests focused and include scope summary, validation evidence, and any config or release implications.</rule>
    <rule>When CLI behavior, scaffold output, or release flow changes, include sample commands or output in the PR description.</rule>
    <rule>Never mention tool-generated authorship metadata such as <code>co-authored-by</code> unless the user explicitly asks for it.</rule>
  </git_policy>

  <agent_behavior>
    <rule>Before substantial tool use, restate the goal and provide a short plan.</rule>
    <rule>For multi-step tasks, provide concise progress updates.</rule>
    <rule>Complete end-to-end resolution in one turn when feasible.</rule>
    <rule>If uncertain, gather evidence with tools rather than guessing.</rule>
    <rule>When a task involves release, packaging, GitHub Actions, or Jira, summarize the intended commands before executing them.</rule>
    <rule>When helping with release work, prefer validating locally first, then using <code>gh</code> for workflow dispatch or inspection only if the user asked for it.</rule>
  </agent_behavior>
</agent_spec>
