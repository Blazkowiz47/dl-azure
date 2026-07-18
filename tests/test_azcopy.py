"""Tests for AzCopy command invocation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pytest import MonkeyPatch

from dl_azure.storage.azcopy import AzCopyTransferBase


def test_azcopy_retry_concurrency_does_not_use_a_shell(
    monkeypatch: MonkeyPatch,
) -> None:
    """AzCopy arguments should remain literal when retry concurrency is set."""
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("AZCOPY_TEST_SENTINEL", "present")
    monkeypatch.setattr("dl_azure.storage.azcopy.subprocess.run", fake_run)
    transfer = AzCopyTransferBase("demoaccount", "images")
    command = [
        "azcopy",
        "copy",
        "/tmp/path with spaces;still-literal",
        "https://demoaccount.blob.core.windows.net/images/example.jpg",
    ]

    result = transfer._invoke_azcopy(
        command,
        concurrency_value=2,
        source_path=Path("/tmp/path with spaces;still-literal"),
    )

    assert result is not None
    invoked_command, invoked_options = calls[0]
    assert invoked_command == command
    assert invoked_options["shell"] is False
    process_env = invoked_options["env"]
    assert isinstance(process_env, dict)
    assert process_env["AZCOPY_CONCURRENCY_VALUE"] == "2"
    assert process_env["AZCOPY_TEST_SENTINEL"] == "present"
