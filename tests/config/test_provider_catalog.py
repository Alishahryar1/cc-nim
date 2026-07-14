from dataclasses import FrozenInstanceError

import pytest

from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderDescriptor,
)


def test_provider_descriptors_are_immutable_values() -> None:
    descriptor = ProviderDescriptor(
        provider_id="local",
        display_name="Local",
        local=True,
    )

    assert descriptor.local is True
    assert not hasattr(descriptor, "__dict__")
    with pytest.raises(FrozenInstanceError):
        descriptor.__setattr__("local", False)


def test_catalog_has_no_transport_metadata() -> None:
    assert "transport_type" not in ProviderDescriptor.__slots__
    assert "capabilities" not in ProviderDescriptor.__slots__


def test_catalog_local_assignments_are_exact() -> None:
    assert {
        provider_id
        for provider_id, descriptor in PROVIDER_CATALOG.items()
        if descriptor.local
    } == {"lmstudio", "llamacpp", "ollama"}


def test_ollama_cloud_is_remote_and_distinct_from_local_ollama() -> None:
    cloud = PROVIDER_CATALOG["ollama_cloud"]
    local = PROVIDER_CATALOG["ollama"]

    assert cloud.local is False
    assert cloud.credential_env == "OLLAMA_API_KEY"
    assert local.local is True
    assert local.credential_env is None
