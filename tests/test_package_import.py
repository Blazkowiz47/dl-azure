"""Basic tests for the Azure extension package."""

from __future__ import annotations

import dl_azure


def test_package_import_exposes_version() -> None:
    """The package root should import successfully and expose a version."""
    assert dl_azure.__version__ == "0.0.3"
