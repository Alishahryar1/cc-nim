from dataclasses import FrozenInstanceError, fields

import pytest

from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderCapabilities,
    ProviderDescriptor,
)


def test_provider_capabilities_are_an_exact_immutable_value_shape() -> None:
    capabilities = ProviderCapabilities(local=True, server_tool_passthrough=True)

    assert tuple(field.name for field in fields(ProviderCapabilities)) == (
        "local",
        "server_tool_passthrough",
    )
    assert ProviderCapabilities.__slots__ == (
        "local",
        "server_tool_passthrough",
    )
    assert not hasattr(capabilities, "__dict__")
    assert capabilities == ProviderCapabilities(
        local=True,
        server_tool_passthrough=True,
    )
    assert ProviderCapabilities() == ProviderCapabilities(
        local=False,
        server_tool_passthrough=False,
    )
    with pytest.raises(FrozenInstanceError):
        capabilities.__setattr__("local", False)


def test_catalog_has_no_transport_or_string_capability_metadata() -> None:
    assert "transport_type" not in ProviderDescriptor.__slots__
    assert all(
        not hasattr(descriptor, "transport_type")
        for descriptor in PROVIDER_CATALOG.values()
    )
    assert all(
        type(descriptor.capabilities) is ProviderCapabilities
        for descriptor in PROVIDER_CATALOG.values()
    )


def test_catalog_capability_assignments_are_exact() -> None:
    assert {
        provider_id
        for provider_id, descriptor in PROVIDER_CATALOG.items()
        if descriptor.capabilities.local
    } == {"lmstudio", "llamacpp", "ollama"}
    assert {
        provider_id
        for provider_id, descriptor in PROVIDER_CATALOG.items()
        if descriptor.capabilities.server_tool_passthrough
    } == {"llamacpp", "ollama"}
