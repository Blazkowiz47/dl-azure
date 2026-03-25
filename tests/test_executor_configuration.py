"""Tests for Azure executor public configuration behavior."""

from __future__ import annotations

import os
from pathlib import Path

from dl_azure.executors.azure_compute import AzureComputeExecutor


def test_update_amlignore_preserves_user_content(tmp_path: Path) -> None:
    """The managed Azure block should not overwrite user `.amlignore` content."""
    amlignore_path = tmp_path / ".amlignore"
    amlignore_path.write_text("custom-entry/\n", encoding="utf-8")

    executor = AzureComputeExecutor(
        sweep_config={"executor": {}},
        experiment_name="demo",
        sweep_id="sweep-1",
        compute_target="gpu-cluster",
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        executor.update_amlignore("experiments/lr_sweep.yaml")
    finally:
        os.chdir(previous_cwd)

    rendered = amlignore_path.read_text(encoding="utf-8")
    assert "custom-entry/" in rendered
    assert "# BEGIN dl-azure managed block" in rendered
    assert "# END dl-azure managed block" in rendered
    assert "lab/users/" not in rendered
    assert "lab/template/" not in rendered
