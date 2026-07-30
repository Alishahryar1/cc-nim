"""MCP client manager for per-request MCP server lifecycle."""

from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client


@dataclass
class MCPServerConfig:
    transport: Literal["stdio", "sse"]
    command: str | None = None
    args: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]  # MCP JSON Schema


@dataclass
class ToolResult:
    content: Any
    is_error: bool = False


class MCPClientManager:
    """Manages MCP client connections for a single request."""

    def __init__(self) -> None:
        self._sessions: list[ClientSession] = []
        self._streams: list[tuple] = []

    async def connect(
        self,
        servers: list[MCPServerConfig],
        client_host: str | None = None,
    ) -> list[MCPTool]:
        """Connect to MCP servers and return available tools.

        Security guardrail: Only allow localhost connections.
        """
        # Security: Only allow localhost/127.0.0.1/::1
        if client_host and client_host not in ("127.0.0.1", "localhost", "::1"):
            return []

        all_tools: list[MCPTool] = []

        for config in servers:
            if config.transport == "stdio":
                if not config.command:
                    continue
                params = StdioServerParameters(
                    command=config.command,
                    args=config.args or [],
                )
                read_stream, write_stream = await stdio_client(params).__aenter__()
                self._streams.append((read_stream, write_stream))
                session = ClientSession(read_stream, write_stream)
            else:  # sse
                if not config.url:
                    continue
                read_stream, write_stream = await sse_client(config.url).__aenter__()
                self._streams.append((read_stream, write_stream))
                session = ClientSession(read_stream, write_stream)

            await session.__aenter__()
            await session.initialize()
            self._sessions.append(session)

            tools_result = await session.list_tools()
            all_tools.extend(
                MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                )
                for tool in tools_result.tools
            )

        return all_tools

    async def execute_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a tool on the connected MCP servers."""
        for session in self._sessions:
            try:
                result = await session.call_tool(tool_name, arguments)
                return ToolResult(content=result.content, is_error=result.isError)
            except Exception:
                continue  # Try next session

        # Tool not found on any server
        return ToolResult(content=[], is_error=True)

    async def close(self) -> None:
        """Close all MCP connections."""
        for session in self._sessions:
            with suppress(Exception):
                await session.__aexit__(None, None, None)
        for read_stream, write_stream in self._streams:
            with suppress(Exception):
                await read_stream.__aexit__(None, None, None)
                await write_stream.__aexit__(None, None, None)
        self._sessions.clear()
        self._streams.clear()


def mcp_tool_to_openai_function(mcp_tool: MCPTool) -> dict[str, Any]:
    """Convert MCP tool to OpenAI function declaration format."""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.input_schema,
        },
    }
