"""Tool-approval policy for MCP tool calls emitted by the upstream LLM.

Each concrete policy is stateless: ``evaluate`` returns either a ``Decision`` or
a ``DenyReason``. ``build_policy`` constructs the active policy from ``Settings``.
"""

from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from free_claude_code.config.settings import Settings

if TYPE_CHECKING:
    from .mcp_client import MCPServerConfig


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class DenyReason:
    code: str
    message: str


@dataclass(frozen=True)
class McpServerDescriptor:
    name: str
    transport: str
    host: str | None

    @classmethod
    def from_server_config(
        cls, server: MCPServerConfig, *, name: str = ""
    ) -> McpServerDescriptor:
        # Best-effort host extraction; SSE hosts are URI-derived, stdio is local.
        host: str | None = None
        if server.transport == "sse" and server.url:
            host = _extract_host(server.url)
        return cls(name=name, transport=server.transport, host=host)


def _extract_host(url: str) -> str | None:
    """Pull host out of an SSE URL without dragging urllib into this module."""
    # Strip scheme:// prefix if present, then take up to the next '/' or ':'.
    stripped = url
    for prefix in ("https://", "http://"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break
    host_end = len(stripped)
    for sep in ("/", "?", "#"):
        idx = stripped.find(sep)
        if idx != -1 and idx < host_end:
            host_end = idx
    host_part = stripped[:host_end]
    if "@" in host_part:
        host_part = host_part.split("@", 1)[1]
    # IPv6 bracket form: [::1]:port
    if host_part.startswith("[") and "]" in host_part:
        return host_part[1 : host_part.index("]")]
    # Strip :port
    if ":" in host_part:
        host_part = host_part.split(":", 1)[0]
    return host_part or None


class ToolApprovalPolicy(Protocol):
    def evaluate(
        self,
        *,
        tool_name: str,
        tool_input: dict,
        server: McpServerDescriptor | None,
    ) -> Decision | DenyReason: ...


def _glob_match(name: str, pattern: str) -> bool:
    # fnmatch handles plain substrings (no '*' -> exact match) and '*' / '?' globs.
    return fnmatch(name, pattern)


class OffPolicy:
    """Pass-through: allow every tool call."""

    def evaluate(
        self,
        *,
        tool_name: str,
        tool_input: dict,
        server: McpServerDescriptor | None,
    ) -> Decision | DenyReason:
        return Decision.ALLOW


class DenyCoworkPolicy:
    """Deny any ``mcp__cowork__*`` tool unless explicitly allowlisted."""

    _COWORK_PREFIX = "mcp__cowork__"

    def __init__(
        self,
        allowlist: tuple[str, ...] = (),
        denylist: tuple[str, ...] = (),
    ) -> None:
        self._allowlist = allowlist
        self._denylist = denylist

    def evaluate(
        self,
        *,
        tool_name: str,
        tool_input: dict,
        server: McpServerDescriptor | None,
    ) -> Decision | DenyReason:
        if tool_name.startswith(self._COWORK_PREFIX) and not self._matches_any(
            tool_name, self._allowlist
        ):
            bare = tool_name[len(self._COWORK_PREFIX) :]
            return DenyReason(
                code="permission_denied",
                message=(
                    "Tool 'mcp__cowork__" + bare + "' is denied by FCC "
                    "tool_approval_policy=deny_cowork."
                ),
            )
        if self._matches_any(tool_name, self._denylist):
            return DenyReason(
                code="permission_denied",
                message=(
                    "Tool '" + tool_name + "' is denied by FCC mcp_tool_denylist."
                ),
            )
        return Decision.ALLOW

    @staticmethod
    def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
        return any(_glob_match(name, p) for p in patterns)


class LocalhostOnlyPolicy:
    """Allow only stdio MCP servers or SSE servers bound to loopback hosts."""

    _LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

    def __init__(
        self,
        allowlist: tuple[str, ...] = (),
        denylist: tuple[str, ...] = (),
    ) -> None:
        self._allowlist = allowlist
        self._denylist = denylist

    def evaluate(
        self,
        *,
        tool_name: str,
        tool_input: dict,
        server: McpServerDescriptor | None,
    ) -> Decision | DenyReason:
        if server is None:
            return DenyReason(
                code="transport_untrusted",
                message="Server descriptor unavailable",
            )
        if server.transport == "stdio":
            host_check_ok = True
        elif server.transport == "sse":
            host = server.host
            host_check_ok = host is None or host in self._LOOPBACK_HOSTS
        else:
            host_check_ok = False

        if not host_check_ok:
            host_repr = server.host if server is not None else None
            return DenyReason(
                code="transport_untrusted",
                message=(
                    "MCP server at '" + str(host_repr) + "' not in localhost allowlist"
                ),
            )

        if self._matches_any(tool_name, self._denylist):
            return DenyReason(
                code="permission_denied",
                message=(
                    "Tool '" + tool_name + "' is denied by FCC mcp_tool_denylist."
                ),
            )
        return Decision.ALLOW

    @staticmethod
    def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
        return any(_glob_match(name, p) for p in patterns)


def build_policy(settings: Settings) -> ToolApprovalPolicy:
    """Construct the active tool-approval policy from ``Settings``."""
    name = settings.tool_approval_policy
    allow = tuple(settings.mcp_tool_allowlist)
    deny = tuple(settings.mcp_tool_denylist)

    if name == "off":
        return OffPolicy()
    if name == "deny_cowork":
        return DenyCoworkPolicy(allowlist=allow, denylist=deny)
    if name == "localhost_only":
        return LocalhostOnlyPolicy(allowlist=allow, denylist=deny)

    logger.warning("Unknown tool_approval_policy={!r}; falling back to OffPolicy", name)
    return OffPolicy()
