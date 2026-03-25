"""Azure scaffold extension for dl-init-experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dl_core.init_extensions import InitExtension, ScaffoldContext


def _azure_config_template() -> str:
    """Render the placeholder Azure workspace config."""
    return """{
  "subscription_id": "<subscription-id>",
  "resource_group": "<resource-group>",
  "workspace_name": "<workspace-name>",
  "account_name": "<storage-account-name>"
}
"""


def _azure_executor_preset() -> str:
    """Render a minimal Azure executor preset block."""
    return """

executors:
  azure:
    executor.name: azure
    executor.compute_target: "<compute-target>"
    executor.environment_name: "<environment-name>"
    executor.environment_version: "latest"
    executor.datastore_name: null
    executor.process_count_per_node: 1
"""


def _azure_mlflow_callback_block() -> str:
    """Render the scaffold callback block for Azure MLflow logging."""
    return """
  azure_mlflow:
    experiment_name: my_experiment
    run_name: my_run
    log_config: true
"""


def _azure_tracking_fields() -> str:
    """Render Azure MLflow additions to the scaffold tracking block."""
    return """tracking:
  backend: azure_mlflow
  group: my_experiment
"""


def _merged_azure_config(target_dir: Path) -> str:
    """Render Azure config while preserving any existing user-provided values."""
    default_config = json.loads(_azure_config_template())
    existing_path = target_dir / "azure-config.json"
    if not existing_path.exists():
        return json.dumps(default_config, indent=2) + "\n"

    try:
        existing_config = json.loads(existing_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return json.dumps(default_config, indent=2) + "\n"

    if not isinstance(existing_config, dict):
        return json.dumps(default_config, indent=2) + "\n"

    merged_config = {**default_config, **existing_config}
    return json.dumps(merged_config, indent=2) + "\n"


def _azure_dataset_init(
    module_name: str,
    compute_class_name: str,
    streaming_class_name: str,
) -> str:
    """Render the Azure dataset package ``__init__`` file."""
    return (
        '"""Local Azure dataset extensions."""\n\n'
        f"from .{module_name} import {compute_class_name}, {streaming_class_name}\n\n"
        f'__all__ = ["{compute_class_name}", "{streaming_class_name}"]\n'
    )


def _azure_dataset_template(
    dataset_name: str,
    class_stem: str,
) -> str:
    """Render the scaffolded Azure dataset wrappers."""
    compute_registration_names = [dataset_name, f"{dataset_name}_compute_azure"]
    streaming_registration_name = f"{dataset_name}_streaming_azure"
    compute_class_name = f"{class_stem}AzureComputeWrapper"
    streaming_class_name = f"{class_stem}StreamingAzureWrapper"
    return f'''"""Project Azure dataset wrappers.

Available generic dl-azure foundations:
- AzureComputeWrapper / AzureStreamingWrapper
- AzureComputeFrameWrapper / AzureStreamingFrameWrapper
- AzureComputeMultiFrameWrapper / AzureStreamingMultiFrameWrapper

Use the compute wrapper for mounted Azure ML inputs and the streaming wrapper
for direct blob reads from Azure storage. Streaming wrappers require an
explicit `dataset.container_name` in config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dl_azure.datasets import AzureComputeWrapper, AzureStreamingWrapper
from dl_core.core import register_dataset


def _select_label(classes: list[str], requested_label: str) -> str:
    """Return a valid label name for generated Azure wrappers."""
    if requested_label in classes:
        return requested_label
    if classes:
        return classes[-1]
    return requested_label


def _scan_image_records(
    wrapper: AzureComputeWrapper | AzureStreamingWrapper,
    split: str,
) -> list[dict[str, str]]:
    """Collect image records using the configured split or scan prefix."""
    split_prefixes = wrapper.config.get("scan_prefixes", {{}})
    prefix = split_prefixes.get(split, wrapper.config.get("scan_prefix"))
    if not prefix:
        raise NotImplementedError(
            "TODO: set dataset.scan_prefix or dataset.scan_prefixes in the "
            "config, or override get_file_list()."
        )

    allowed_suffixes = {{
        suffix.replace("*", "").lower() for suffix in wrapper.file_extensions
    }}
    label = _select_label(wrapper.classes, wrapper.config.get("default_label", "class1"))
    paths = [
        blob_path
        for blob_path in wrapper.scan_paths(prefix)
        if Path(blob_path).suffix.lower() in allowed_suffixes
    ]
    return [{{"path": blob_path, "label": label}} for blob_path in paths]


def _transform_image_record(
    wrapper: AzureComputeWrapper | AzureStreamingWrapper,
    file_dict: dict[str, str],
    split: str,
) -> dict[str, Any]:
    """Load one image record through the wrapper transport helpers."""
    path = file_dict["path"]
    label = file_dict["label"]
    class_index = wrapper.classes.index(label) if label in wrapper.classes else 0
    image = wrapper.load_image_data(path, use_cache=wrapper.should_use_cache(split))
    if image is None:
        return {{
            "image": torch.zeros(3, wrapper.height, wrapper.width),
            "label": class_index,
            "class": label,
            "path": path,
        }}

    if wrapper.augmentation:
        image_tensor = wrapper.augmentation.apply(image, split)
    else:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    return {{
        "image": image_tensor.float(),
        "label": class_index,
        "class": label,
        "path": path,
    }}


@register_dataset({compute_registration_names!r})
class {compute_class_name}(AzureComputeWrapper):
    """Project-specific Azure ML mounted dataset wrapper."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        config.setdefault("input_name", "dataset_path")
        config.setdefault("scan_prefix", "data/")
        config.setdefault("default_label", "class1")
        super().__init__(config, **kwargs)
        self.height = self.config.get("height", 64)
        self.width = self.config.get("width", 64)

    def get_file_list(self, split: str) -> list[dict[str, str]]:
        """Return image records from the configured Azure ML mount."""
        return _scan_image_records(self, split)

    def transform(self, file_dict: dict[str, str], split: str) -> dict[str, Any]:
        """Load one mounted Azure ML image sample."""
        return _transform_image_record(self, file_dict, split)


@register_dataset("{streaming_registration_name}")
class {streaming_class_name}(AzureStreamingWrapper):
    """Project-specific Azure blob streaming dataset wrapper."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        config.setdefault("scan_prefix", "data/")
        config.setdefault("default_label", "class1")
        super().__init__(config, **kwargs)
        self.height = self.config.get("height", 64)
        self.width = self.config.get("width", 64)

    def get_file_list(self, split: str) -> list[dict[str, str]]:
        """Return blob-backed image records from the configured prefix."""
        return _scan_image_records(self, split)

    def transform(self, file_dict: dict[str, str], split: str) -> dict[str, Any]:
        """Load one blob-backed Azure image sample."""
        return _transform_image_record(self, file_dict, split)
'''


class AzureInitExtension(InitExtension):
    """Expose Azure scaffold wiring when dl-azure is installed."""

    name = "azure"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register the Azure scaffold flag."""
        parser.add_argument(
            "--with-azure",
            action="store_true",
            help="Include Azure executor wiring and azure-config.json.",
        )

    def is_enabled(
        self,
        args: argparse.Namespace,
        discovered_extensions: dict[str, InitExtension],
    ) -> bool:
        """Enable Azure wiring when explicitly requested."""
        return bool(getattr(args, "with_azure", False))

    def apply(self, context: ScaffoldContext) -> None:
        """Apply Azure-specific scaffold mutations."""
        context.replace_in_file(
            "pyproject.toml",
            '"deep-learning-core"',
            '"deep-learning-core[azure]"',
        )
        context.add_dependency("deep-learning-azure")
        context.append_bootstrap_import("import dl_azure  # noqa: F401")
        context.append_readme_note(
            "Azure support is enabled. Fill in `azure-config.json` and the "
            "`executors.azure` preset placeholders before submitting runs."
        )
        context.replace_in_file(
            Path("configs") / "base.yaml",
            "  metric_logger:\n    log_frequency: 1\n",
            "  metric_logger:\n    log_frequency: 1\n"
            f"{_azure_mlflow_callback_block()}",
        )
        context.replace_in_file(
            Path("configs") / "base_sweep.yaml",
            "tracking:\n  group: my_experiment\n",
            _azure_tracking_fields(),
        )
        context.replace_in_file(
            Path("configs") / "base_sweep.yaml",
            "executor: preset:executors.local",
            "executor: preset:executors.azure",
        )
        presets_path = Path("configs") / "presets.yaml"
        context.set_file(
            presets_path,
            f"{context.get_file(presets_path).rstrip()}{_azure_executor_preset()}",
        )
        context.set_file(
            "azure-config.json",
            _merged_azure_config(context.target_dir),
        )
        dataset_module = context.project.dataset_name
        class_stem = context.project.dataset_class_name.removesuffix("Dataset")
        context.set_file(
            Path("src") / "datasets" / "__init__.py",
            _azure_dataset_init(
                dataset_module,
                f"{class_stem}AzureComputeWrapper",
                f"{class_stem}StreamingAzureWrapper",
            ),
        )
        context.set_file(
            Path("src") / "datasets" / f"{dataset_module}.py",
            _azure_dataset_template(
                context.project.dataset_name,
                class_stem,
            ),
        )
