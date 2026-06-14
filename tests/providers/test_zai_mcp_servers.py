"""Tests for z.ai native MCP server injection helpers."""

from typing import Any

from providers.zai.mcp_servers import (
    ZAI_MCP_HTTP_SERVERS,
    build_zai_mcp_servers,
    merge_zai_mcp_servers,
)


def test_build_zai_mcp_servers_structure():
    servers = build_zai_mcp_servers("secret-key")

    assert len(servers) == len(ZAI_MCP_HTTP_SERVERS)
    for server in servers:
        assert server["type"] == "url"
        assert server["name"]
        assert server["url"].startswith("https://api.z.ai/api/mcp/")
        assert server["authorization_token"] == "secret-key"


def test_build_zai_mcp_servers_names_match_docs():
    names = {server["name"] for server in build_zai_mcp_servers("k")}
    assert names == {"web-reader", "web-search-prime", "zread"}


def test_merge_injects_into_empty_body():
    body: dict[str, Any] = {}

    merge_zai_mcp_servers(body, "key")

    assert len(body["mcp_servers"]) == 3
    assert body["mcp_servers"][0]["authorization_token"] == "key"


def test_merge_preserves_client_servers():
    client_server = {"type": "url", "url": "https://x/mcp", "name": "custom"}
    body: dict[str, Any] = {"mcp_servers": [client_server]}

    merge_zai_mcp_servers(body, "key")

    injected = body["mcp_servers"]
    assert injected[0] is client_server
    names = [server["name"] for server in injected]
    assert {"web-reader", "web-search-prime", "zread", "custom"} == set(names)
    assert len(injected) == 4


def test_merge_skips_shadowed_names():
    body: dict[str, Any] = {
        "mcp_servers": [{"type": "url", "url": "https://x", "name": "web-reader"}]
    }

    merge_zai_mcp_servers(body, "key")

    names = [server["name"] for server in body["mcp_servers"]]
    assert names.count("web-reader") == 1
    assert {"web-search-prime", "zread"}.issubset(set(names))


def test_merge_handles_non_list_existing():
    body: dict[str, Any] = {"mcp_servers": "not-a-list"}

    merge_zai_mcp_servers(body, "key")

    assert isinstance(body["mcp_servers"], list)
    assert len(body["mcp_servers"]) == 3


def test_merge_does_not_mutate_specs_constant():
    before = [dict(spec) for spec in ZAI_MCP_HTTP_SERVERS]

    merge_zai_mcp_servers({}, "key")

    assert [dict(spec) for spec in ZAI_MCP_HTTP_SERVERS] == before
