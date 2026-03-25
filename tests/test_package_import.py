"""Basic tests for the Azure extension package."""

from __future__ import annotations

import sys

import dl_azure


def test_package_import_exposes_version() -> None:
    """The package root should import successfully and expose a version."""
    assert dl_azure.__version__ == "1.0.0"
    assert "azureml.core" not in sys.modules
