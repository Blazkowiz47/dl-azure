"""Verify backend selection, transfer integrity, and SDK transport settings."""

from __future__ import annotations

import errno
import hashlib
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from azure.core import MatchConditions
from azure.core.exceptions import ResourceModifiedError

from dl_azure.storage import download
from dl_azure.storage.download import AzureDownloader


@pytest.fixture
def transfer(monkeypatch: pytest.MonkeyPatch) -> Any:
    payload = b"example Azure blob"
    state = SimpleNamespace(
        payload=payload,
        properties=SimpleNamespace(
            size=len(payload), etag='"v1"',
            content_settings=SimpleNamespace(content_md5=hashlib.md5(payload).digest()),
        ),
        lookups=[], commands=[], sdk_calls=[], property_calls=[],
        real_run=subprocess.run,
    )

    def which(executable: str) -> str:
        state.lookups.append(executable)
        return f"/installed/{executable}"

    def run(command: list[str], **kwargs: Any) -> Any:
        state.commands.append((command, kwargs))
        Path(command[3]).write_bytes(state.payload)
        return SimpleNamespace(returncode=0)

    def properties(**kwargs: Any) -> Any:
        state.property_calls.append(kwargs)
        return state.properties

    def sdk_download(**kwargs: Any) -> Any:
        state.sdk_calls.append(kwargs)
        return SimpleNamespace(readinto=lambda handle: handle.write(state.payload))

    state.client = SimpleNamespace(
        url="https://demo.blob.core.windows.net/data/file.bin?sig=secret",
        get_blob_properties=properties,
        download_blob=sdk_download,
    )
    download._AZCOPY_PATHS.clear()
    monkeypatch.setattr(download.shutil, "which", which)
    monkeypatch.setattr(download.subprocess, "run", run)
    yield state
    download._AZCOPY_PATHS.clear()


def test_azcopy_is_default_and_uses_private_state(
    tmp_path: Path, transfer: Any,
) -> None:
    downloader = AzureDownloader({"azcopy": {
        "executable": "custom-azcopy", "concurrency": "AUTO",
        "buffer_gb": 2.5, "cap_mbps": 250, "timeout_seconds": 300,
    }})
    destination = tmp_path / "file with spaces%25.part"
    downloader.download(transfer.client, destination)
    assert destination.read_bytes() == transfer.payload
    assert not transfer.sdk_calls
    command, kwargs = transfer.commands[0]
    assert command[:4] == [
        "/installed/custom-azcopy", "copy", transfer.client.url, str(destination),
    ]
    assert "--cap-mbps=250" in command
    assert "--check-md5=FailIfDifferent" in command
    assert kwargs["shell"] is False
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["timeout"] == 300
    environment = kwargs["env"]
    assert environment["AZCOPY_CONCURRENCY_VALUE"] == "AUTO"
    assert environment["AZCOPY_BUFFER_GB"] == "2.5"
    for variable in ("AZCOPY_LOG_LOCATION", "AZCOPY_JOB_PLAN_LOCATION"):
        assert not Path(environment[variable]).exists()
    assert len(transfer.property_calls) == 2


def test_sdk_uses_parallel_readinto_and_source_condition(
    tmp_path: Path, transfer: Any,
) -> None:
    downloader = AzureDownloader({"backend": "sdk", "sdk": {
        "max_concurrency": 8, "connection_timeout_seconds": 7,
        "read_timeout_seconds": 11, "validate_content": True,
    }})
    destination = tmp_path / "file.part"
    downloader.download(transfer.client, destination)
    assert destination.read_bytes() == transfer.payload
    assert not transfer.lookups and not transfer.commands
    assert transfer.sdk_calls == [{
        "max_concurrency": 8, "connection_timeout": 7, "read_timeout": 11,
        "validate_content": True, "etag": '"v1"',
        "match_condition": MatchConditions.IfNotModified,
    }]


def test_missing_azcopy_is_detected_once_across_threads_and_instances(
    tmp_path: Path, transfer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(executable: str) -> None:
        transfer.lookups.append(executable)
        return None

    monkeypatch.setattr(download.shutil, "which", missing)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(
            AzureDownloader().download, transfer.client, tmp_path / f"{i}.part",
        ) for i in range(8)]
        for future in futures:
            future.result()
    assert transfer.lookups == ["azcopy"]
    assert len(transfer.sdk_calls) == 8
    assert not transfer.commands
    AzureDownloader({"azcopy": {"executable": "alternate"}}).download(
        transfer.client, tmp_path / "alternate.part",
    )
    assert transfer.lookups == ["azcopy", "alternate"]


def test_transfer_failure_falls_back_without_disabling_azcopy(
    tmp_path: Path, transfer: Any, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = []

    def run(command: list[str], **kwargs: Any) -> Any:
        attempts.append(command)
        # SDK must truncate this longer, incomplete file before writing.
        Path(command[3]).write_bytes(transfer.payload * 2)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(download.subprocess, "run", run)
    for name in ("first", "second"):
        destination = tmp_path / f"{name}.part"
        AzureDownloader().download(transfer.client, destination)
        assert destination.read_bytes() == transfer.payload
    assert len(attempts) == 2 and len(transfer.sdk_calls) == 2
    assert transfer.lookups == ["azcopy"]
    assert "sig=secret" not in caplog.text


def test_signing_failure_uses_sdk_for_only_this_file(
    tmp_path: Path, transfer: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    def denied() -> str:
        raise PermissionError("Cannot sign sig=secret")

    downloader = AzureDownloader()
    downloader.download(transfer.client, tmp_path / "unsigned.part", azcopy_url=denied)
    downloader.download(transfer.client, tmp_path / "signed.part")
    assert len(transfer.sdk_calls) == 1
    assert len(transfer.commands) == 1
    assert "sig=secret" not in caplog.text


def test_unusable_executable_is_remembered(
    tmp_path: Path, transfer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = []

    def run(command: list[str], **kwargs: Any) -> None:
        attempts.append(command)
        raise OSError(errno.ENOEXEC, "Invalid executable")

    monkeypatch.setattr(download.subprocess, "run", run)
    for name in ("first", "second"):
        AzureDownloader().download(transfer.client, tmp_path / f"{name}.part")
    assert len(attempts) == 1 and len(transfer.sdk_calls) == 2


@pytest.mark.parametrize("available", [True, False])
def test_disabled_fallback_never_invokes_sdk(
    tmp_path: Path, transfer: Any, monkeypatch: pytest.MonkeyPatch,
    available: bool,
) -> None:
    if available:
        monkeypatch.setattr(download.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    else:
        monkeypatch.setattr(download.shutil, "which", lambda executable: None)
    with pytest.raises((OSError, RuntimeError)):
        AzureDownloader({"fallback_to_sdk": False}).download(
            transfer.client, tmp_path / "file.part",
        )
    assert not transfer.sdk_calls


def test_timeout_waits_for_child_before_sdk_fallback(
    tmp_path: Path, transfer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "slow-azcopy"
    executable.write_text(
        f"#!{sys.executable}\nimport sys, time\n"
        "from pathlib import Path\nPath(sys.argv[3]).write_bytes(b'partial')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    real_popen = subprocess.Popen
    children = []

    def popen(*args: Any, **kwargs: Any) -> Any:
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(download.shutil, "which", lambda name: str(executable))
    monkeypatch.setattr(download.subprocess, "run", transfer.real_run)
    monkeypatch.setattr(download.subprocess, "Popen", popen)
    destination = tmp_path / "file.part"
    AzureDownloader({"azcopy": {"timeout_seconds": 0.2}}).download(
        transfer.client, destination,
    )
    assert len(children) == 1 and children[0].poll() is not None
    assert destination.read_bytes() == transfer.payload
    assert len(transfer.sdk_calls) == 1


def test_azcopy_source_change_is_rejected_before_accepting_file(
    tmp_path: Path, transfer: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = transfer.properties
    monkeypatch.setattr(transfer.client, "get_blob_properties", lambda **kwargs: SimpleNamespace(
        size=properties.size, etag='"v2"',
    ))
    with pytest.raises(ResourceModifiedError, match="changed"):
        AzureDownloader({"fallback_to_sdk": False}).download(
            transfer.client, tmp_path / "file.part", properties=properties,
        )
    assert not transfer.sdk_calls


@pytest.mark.parametrize("backend", ["sdk", "azcopy"])
@pytest.mark.parametrize("corruption", ["size", "md5"])
def test_corrupt_download_is_rejected(
    tmp_path: Path, transfer: Any, backend: str, corruption: str,
) -> None:
    transfer.payload = (
        b"short" if corruption == "size" else b"x" * len(transfer.payload)
    )
    with pytest.raises(OSError, match=f"{corruption.upper() if corruption == 'md5' else 'size'} mismatch"):
        AzureDownloader({"backend": backend, "fallback_to_sdk": False}).download(
            transfer.client, tmp_path / "file.part",
        )


def test_transport_settings_apply_to_actual_blob_client() -> None:
    from azure.storage.blob import BlobClient

    downloader = AzureDownloader({"sdk": {
        "connection_pool_size": 64, "chunk_size_mib": 16,
        "initial_request_size_mib": 64, "read_buffer_kib": 256,
    }})
    with BlobClient.from_blob_url(
        "https://demo.blob.core.windows.net/data/file.bin",
        **downloader.client_options(),
    ) as client:
        assert client._config.max_chunk_get_size == 16 * 1024**2
        assert client._config.max_single_get_size == 64 * 1024**2
        transport = client._pipeline._transport
        assert transport.connection_config.data_block_size == 256 * 1024
        adapter = transport.session.get_adapter(client.url)
        assert adapter._pool_maxsize == 64
        assert adapter._pool_block is False
    # The controller contains only settings, so caches remain serializable.
    assert pickle.loads(pickle.dumps(downloader)).sdk == downloader.sdk


@pytest.mark.parametrize("config", [
    {"backend": "other"}, {"fallback_to_sdk": "true"}, {"sdk": []},
    {"sdk": {"max_concurrency": 0}}, {"sdk": {"max_concurrency": True}},
    {"sdk": {"connection_pool_size": 1.5}},
    {"sdk": {"read_timeout_seconds": float("nan")}},
    {"sdk": {"chunk_size_mib": 16, "validate_content": True}},
    {"azcopy": {"executable": ""}}, {"azcopy": {"concurrency": -1}},
    {"azcopy": {"buffer_gb": float("inf")}}, {"azcopy": {"cap_mbps": -1}},
    {"azcopy": {"timeout_seconds": 0}}, {"sdk": {"unknown": 1}},
])
def test_invalid_settings_fail_before_a_transfer(config: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        AzureDownloader(config)
