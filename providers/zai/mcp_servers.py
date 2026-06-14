"""Z.ai native MCP server injection for the Anthropic-compatible Messages API.

Z.ai publishes four MCP servers (https://docs.z.ai/devpack/mcp).  Three are
HTTP remote endpoints and are injected here as native Anthropic
``mcp_servers`` (``type: url``) so the model can call them server-side when
routing through this provider (e.g. the ``glm-5.2`` alias).  The fourth
(Vision, ``@z_ai/mcp-server``) is a stdio ``npx`` server and cannot be injected
server-side; see the README for its client-side install command.

``authorization_token`` carries the raw Z.ai API key: Anthropic forwards it to
the upstream endpoint as ``Authorization: Bearer <key>``.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# The three HTTP remote z.ai MCP servers (no auth here; auth is per-request).
ZAI_MCP_HTTP_SERVERS: tuple[dict[str, str], ...] = (
    {"name": "web-reader", "url": "https://api.z.ai/api/mcp/web_reader/mcp"},
    {
        "name": "web-search-prime",
        "url": "https://api.z.ai/api/mcp/web_search_prime/mcp",
    },
    {"name": "zread", "url": "https://api.z.ai/api/mcp/zread/mcp"},
)


def build_zai_mcp_servers(api_key: str) -> list[dict[str, str]]:
    """Return native Anthropic ``mcp_servers`` entries for the z.ai HTTP MCPs."""
    return [
        {
            "type": "url",
            "url": spec["url"],
            "name": spec["name"],
            "authorization_token": api_key,
        }
        for spec in ZAI_MCP_HTTP_SERVERS
    ]


def merge_zai_mcp_servers(body: dict[str, Any], api_key: str) -> None:
    """Inject the z.ai HTTP MCP servers into a native Anthropic request body.

    Client-provided ``mcp_servers`` are preserved and take precedence: a z.ai
    server whose ``name`` collides with an existing entry is skipped so callers
    can override or disable individual servers.
    """
    existing = body.get("mcp_servers")
    base_servers: list[Any] = list(existing) if isinstance(existing, list) else []
    existing_names = {
        server.get("name") for server in base_servers if isinstance(server, dict)
    }

    zai_servers = build_zai_mcp_servers(api_key)
    merged = list(base_servers)
    appended = 0
    for server in zai_servers:
        if server["name"] in existing_names:
            logger.debug(
                "ZAI_MCP: skipping injected server '{}' "
                "(shadowed by client mcp_servers)",
                server["name"],
            )
            continue
        merged.append(server)
        appended += 1

    body["mcp_servers"] = merged
    logger.debug(
        "ZAI_MCP: injected {}/{} servers (total mcp_servers={})",
        appended,
        len(zai_servers),
        len(merged),
    )
