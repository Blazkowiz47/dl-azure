"""Basic tests for the Azure extension package."""

from __future__ import annotations

from importlib.metadata import distribution

import dl_azure


def test_package_import_exposes_version() -> None:
    """The package root should import successfully and expose a version."""
    assert dl_azure.__version__ == "0.0.24"


def test_package_exposes_runtime_entry_point() -> None:
    """Installed Azure integrations should register without scaffold imports."""
    runtime_points = distribution("deep-learning-azure").entry_points
    assert any(
        point.group == "dl_core.runtime_extensions"
        and point.name == "azure"
        and point.value == "dl_azure"
        for point in runtime_points
    )
