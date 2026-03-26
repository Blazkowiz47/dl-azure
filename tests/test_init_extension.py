"""Tests for the Azure init extension plugin."""

from __future__ import annotations

from pathlib import Path

from dl_core.init_extensions import ProjectNames, ScaffoldContext

from dl_azure.init_extension import AzureInitExtension


def test_azure_init_extension_updates_scaffold_files(tmp_path: Path) -> None:
    """The Azure init extension should patch the scaffold for Azure usage."""
    context = ScaffoldContext(
        target_dir=tmp_path,
        templates_dir=tmp_path,
        project=ProjectNames(
            project_name="demo",
            project_slug="demo",
            component_name="demo",
            dataset_name="demo",
            dataset_class_name="DemoDataset",
            model_name="resnet_example",
            model_class_name="ResNetExample",
            trainer_name="demo",
            trainer_class_name="DemoTrainer",
        ),
        files={
            Path("pyproject.toml"): (
                "[project]\n"
                "dependencies = [\n"
                '    "deep-learning-core",\n'
                "]\n"
            ),
            Path("README.md"): "# demo\n",
            Path("src") / "bootstrap.py": (
                '"""Project bootstrap hooks for local component loading."""\n'
            ),
            Path("configs") / "base.yaml": (
                "callbacks:\n"
                "  metric_logger:\n"
                "    log_frequency: 1\n"
            ),
            Path("configs") / "base_sweep.yaml": (
                "tracking:\n"
                "  group: demo\n"
                '  run_name_template: "lr_{optimizers.lr}"\n'
                "fixed:\n"
                "  executor: preset:executors.local\n"
            ),
            Path("configs") / "presets.yaml": "# presets\n",
        },
        enabled_extensions={"azure"},
    )

    AzureInitExtension().apply(context)

    assert '"deep-learning-core[azure]"' in context.get_file("pyproject.toml")
    assert "import dl_azure" in context.get_file(Path("src") / "bootstrap.py")
    assert "preset:executors.azure" in context.get_file(
        Path("configs") / "base_sweep.yaml"
    )
    assert "backend: azure_mlflow" in context.get_file(
        Path("configs") / "base_sweep.yaml"
    )
    assert "group: demo" in context.get_file(Path("configs") / "base_sweep.yaml")
    assert "azure_mlflow:" in context.get_file(Path("configs") / "base.yaml")
    assert "executors:" in context.get_file(Path("configs") / "presets.yaml")
    presets_text = context.get_file(Path("configs") / "presets.yaml")
    assert 'executor.compute_target: "<compute-target>"' in presets_text
    assert 'executor.environment_name: "<environment-name>"' in presets_text
    assert 'executor.datastore_name: "<datastore-name-or-null>"' in presets_text
    assert "executor.retry_limit: 0" in presets_text
    assert "executor.dont_wait_for_completion: false" in presets_text
    assert 'executor.azure_config_path: "azure-config.json"' in presets_text
    assert '"subscription_id": "<subscription-id>"' in context.get_file(
        "azure-config.json"
    )
    assert '"deep-learning-azure"' in context.get_file("pyproject.toml")
    dataset_file = context.get_file(Path("src") / "datasets" / "demo.py")
    assert "pad-datasets" not in dataset_file
    assert "dataset.container_name" in dataset_file


def test_azure_init_extension_merges_existing_azure_config(tmp_path: Path) -> None:
    """Existing Azure config values should be preserved and missing keys added."""
    (tmp_path / "azure-config.json").write_text(
        '{\n  "workspace_name": "existing-workspace"\n}\n',
        encoding="utf-8",
    )
    context = ScaffoldContext(
        target_dir=tmp_path,
        templates_dir=tmp_path,
        project=ProjectNames(
            project_name="demo",
            project_slug="demo",
            component_name="demo",
            dataset_name="demo",
            dataset_class_name="DemoDataset",
            model_name="resnet_example",
            model_class_name="ResNetExample",
            trainer_name="demo",
            trainer_class_name="DemoTrainer",
        ),
        files={
            Path("pyproject.toml"): (
                "[project]\n"
                "dependencies = [\n"
                '    "deep-learning-core",\n'
                "]\n"
            ),
            Path("README.md"): "# demo\n",
            Path("src") / "bootstrap.py": (
                '"""Project bootstrap hooks for local component loading."""\n'
            ),
            Path("configs") / "base.yaml": (
                "callbacks:\n"
                "  metric_logger:\n"
                "    log_frequency: 1\n"
            ),
            Path("configs") / "base_sweep.yaml": (
                "tracking:\n"
                "  group: demo\n"
                '  run_name_template: "lr_{optimizers.lr}"\n'
                "fixed:\n"
                "  executor: preset:executors.local\n"
            ),
            Path("configs") / "presets.yaml": "# presets\n",
        },
        enabled_extensions={"azure"},
    )

    AzureInitExtension().apply(context)

    rendered = context.get_file("azure-config.json")
    assert '"workspace_name": "existing-workspace"' in rendered
    assert '"subscription_id": "<subscription-id>"' in rendered
    assert '"account_name": "<storage-account-name>"' in rendered
