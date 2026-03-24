"""Azure scaffold extension for dl-init-experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_core import __version__ as dl_core_version
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
            f'"dl-core>={dl_core_version}"',
            f'"dl-core[azure]>={dl_core_version}"',
        )
        context.append_bootstrap_import("import dl_azure  # noqa: F401")
        context.append_readme_note(
            "Azure support is enabled. Fill in `azure-config.json` and the "
            "`executors.azure` preset placeholders before submitting runs."
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
        context.set_file("azure-config.json", _azure_config_template())

