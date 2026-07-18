"""Tests for Azure blob storage client behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from dl_azure.storage.client import AzureClientService


def _client(monkeypatch: MonkeyPatch) -> AzureClientService:
    """Create a client without consulting the host credential chain."""
    monkeypatch.setattr(
        "dl_azure.storage.client.DefaultAzureCredential",
        lambda: "credential",
    )
    return AzureClientService({"account_name": "demoaccount"})


def test_get_blob_sas_url_generates_user_delegation_signature(
    monkeypatch: MonkeyPatch,
) -> None:
    """SAS URLs should be signed with a user delegation key."""
    calls: dict[str, object] = {}

    class _FakeBlobServiceClient:
        def __init__(self, **kwargs: object) -> None:
            calls["service"] = kwargs

        def get_user_delegation_key(self, **kwargs: object) -> object:
            calls["delegation"] = kwargs
            return "delegation-key"

    def fake_generate_blob_sas(**kwargs: object) -> str:
        calls["sas"] = kwargs
        return "sig=test-token"

    monkeypatch.setattr(
        "dl_azure.storage.client.BlobServiceClient",
        _FakeBlobServiceClient,
    )
    monkeypatch.setattr(
        "dl_azure.storage.client.generate_blob_sas",
        fake_generate_blob_sas,
    )

    url = _client(monkeypatch).get_blob_sas_url(
        "images",
        "train/example.jpg",
        expiry_hours=2,
        permissions="rw",
    )

    assert url == (
        "https://demoaccount.blob.core.windows.net/images/train/example.jpg"
        "?sig=test-token"
    )
    assert calls["service"] == {
        "account_url": "https://demoaccount.blob.core.windows.net",
        "credential": "credential",
    }
    sas_arguments = calls["sas"]
    assert isinstance(sas_arguments, dict)
    assert sas_arguments["account_name"] == "demoaccount"
    assert sas_arguments["container_name"] == "images"
    assert sas_arguments["blob_name"] == "train/example.jpg"
    assert sas_arguments["user_delegation_key"] == "delegation-key"
    permission = sas_arguments["permission"]
    assert permission.read is True
    assert permission.write is True


@pytest.mark.parametrize(
    ("expiry_hours", "permissions"),
    [(0, "r"), (-1, "r"), (1, ""), (1, "z")],
)
def test_get_blob_sas_url_rejects_invalid_options(
    monkeypatch: MonkeyPatch,
    expiry_hours: int,
    permissions: str,
) -> None:
    """Invalid SAS options should fail before contacting Azure."""
    client = _client(monkeypatch)

    with pytest.raises(ValueError):
        client.get_blob_sas_url(
            "images",
            "example.jpg",
            expiry_hours=expiry_hours,
            permissions=permissions,
        )


def test_get_blob_sas_url_does_not_return_unsigned_fallback(
    monkeypatch: MonkeyPatch,
) -> None:
    """Credential failures must not silently downgrade to an unsigned URL."""
    monkeypatch.setattr(
        "dl_azure.storage.client.BlobServiceClient",
        lambda **_kwargs: SimpleNamespace(
            get_user_delegation_key=lambda **_key_kwargs: (_ for _ in ()).throw(
                PermissionError("delegation denied")
            )
        ),
    )

    with pytest.raises(RuntimeError, match="delegation denied"):
        _client(monkeypatch).get_blob_sas_url("images", "example.jpg")


def test_container_client_cache_is_scoped_to_one_credential(
    monkeypatch: MonkeyPatch,
) -> None:
    """Container clients must not leak between authenticated service instances."""
    credentials = iter(["credential-one", "credential-two"])
    monkeypatch.setattr(
        "dl_azure.storage.client.DefaultAzureCredential",
        lambda: next(credentials),
    )
    first_service = AzureClientService({"account_name": "demoaccount"})
    second_service = AzureClientService({"account_name": "demoaccount"})
    first_container = object()
    second_container = object()
    first_calls: list[str] = []
    second_calls: list[str] = []

    def create_first(container_name: str) -> object:
        first_calls.append(container_name)
        return first_container

    def create_second(container_name: str) -> object:
        second_calls.append(container_name)
        return second_container

    monkeypatch.setattr(first_service, "create_container_client", create_first)
    monkeypatch.setattr(second_service, "create_container_client", create_second)

    assert first_service.get_container_client("images") is first_container
    assert first_service.get_container_client("images") is first_container
    assert second_service.get_container_client("images") is second_container
    assert first_calls == ["images"]
    assert second_calls == ["images"]
