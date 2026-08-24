"""Tests for MCPClientManager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMCPClientManager:
    """Test MCP client manager."""

    @pytest.fixture
    def mock_mcp_server(self):
        """Mock MCP server for testing."""
        # This will be used to mock stdio_client and ClientSession
        pass

    def _create_mock_tool(self, name: str, description: str, input_schema: dict):
        """Create a proper mock tool object with real attributes."""
        mock_tool = MagicMock()
        mock_tool.name = name
        mock_tool.description = description
        mock_tool.inputSchema = input_schema
        return mock_tool

    @pytest.mark.asyncio
    async def test_mcp_client_manager_connects_and_lists_tools(self, mock_mcp_server):
        from free_claude_code.providers.openai_chat.mcp_client import (
            MCPClientManager,
            MCPServerConfig,
        )

        manager = MCPClientManager()
        servers = [
            MCPServerConfig(transport="stdio", command="uvx", args=["mcp-server-git"])
        ]

        with patch(
            "free_claude_code.providers.openai_chat.mcp_client.stdio_client"
        ) as mock_stdio_client:
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_stdio_client.return_value.__aenter__.return_value = (
                mock_read_stream,
                mock_write_stream,
            )

            with patch(
                "free_claude_code.providers.openai_chat.mcp_client.ClientSession"
            ) as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value = mock_session
                mock_session.__aenter__.return_value = mock_session
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(
                    return_value=MagicMock(
                        tools=[
                            self._create_mock_tool(
                                "git_status",
                                "Get git status",
                                {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"],
                                },
                            )
                        ]
                    )
                )

                tools = await manager.connect(servers)

                assert len(tools) == 1
                assert tools[0].name == "git_status"
                assert tools[0].description == "Get git status"
                assert tools[0].input_schema["type"] == "object"
                assert "path" in tools[0].input_schema["properties"]
                await manager.close()

    @pytest.mark.asyncio
    async def test_mcp_client_manager_execute_tool(self):
        from free_claude_code.providers.openai_chat.mcp_client import (
            MCPClientManager,
            MCPServerConfig,
            ToolResult,
        )

        manager = MCPClientManager()

        with patch(
            "free_claude_code.providers.openai_chat.mcp_client.stdio_client"
        ) as mock_stdio_client:
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_stdio_client.return_value.__aenter__.return_value = (
                mock_read_stream,
                mock_write_stream,
            )

            with patch(
                "free_claude_code.providers.openai_chat.mcp_client.ClientSession"
            ) as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value = mock_session
                mock_session.__aenter__.return_value = mock_session
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(
                    return_value=MagicMock(
                        tools=[
                            self._create_mock_tool(
                                "test_tool",
                                "Test tool",
                                {"type": "object", "properties": {}},
                            )
                        ]
                    )
                )
                mock_session.call_tool = AsyncMock(
                    return_value=MagicMock(
                        content=[MagicMock(text="tool result")], isError=False
                    )
                )

                await manager.connect(
                    [MCPServerConfig(transport="stdio", command="test", args=[])]
                )

                result = await manager.execute_tool("test_tool", {"arg": "value"})

                assert isinstance(result, ToolResult)
                assert isinstance(result.content, list)
                assert result.is_error is False

                await manager.close()

    def test_mcp_tool_to_openai_function_declaration(self):
        from free_claude_code.providers.openai_chat.mcp_client import (
            MCPTool,
            mcp_tool_to_openai_function,
        )

        mcp_tool = MCPTool(
            name="git_status",
            description="Get git status",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

        openai_func = mcp_tool_to_openai_function(mcp_tool)

        assert openai_func["type"] == "function"
        assert openai_func["function"]["name"] == "git_status"
        assert openai_func["function"]["description"] == "Get git status"
        assert openai_func["function"]["parameters"] == mcp_tool.input_schema

    @pytest.mark.asyncio
    async def test_mcp_client_manager_security_guardrail_localhost_only(self):
        """Test that MCP is skipped for non-localhost requests."""
        from free_claude_code.providers.openai_chat.mcp_client import (
            MCPClientManager,
            MCPServerConfig,
        )

        manager = MCPClientManager()

        # Non-localhost client should return empty tools
        tools = await manager.connect(
            [MCPServerConfig(transport="stdio", command="test", args=[])],
            client_host="192.168.1.1",
        )
        assert tools == []

        # Localhost should allow connection (though it won't actually connect without mock)
        # We'll test this in a more complete test
        await manager.close()

    @pytest.mark.asyncio
    async def test_mcp_client_manager_with_sse_transport(self):
        """Test SSE transport connection."""
        from free_claude_code.providers.openai_chat.mcp_client import (
            MCPClientManager,
            MCPServerConfig,
        )

        manager = MCPClientManager()
        servers = [MCPServerConfig(transport="sse", url="http://localhost:3000/sse")]

        with patch(
            "free_claude_code.providers.openai_chat.mcp_client.sse_client"
        ) as mock_sse_client:
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_sse_client.return_value.__aenter__.return_value = (
                mock_read_stream,
                mock_write_stream,
            )

            with patch(
                "free_claude_code.providers.openai_chat.mcp_client.ClientSession"
            ) as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value = mock_session
                mock_session.__aenter__.return_value = mock_session
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(
                    return_value=MagicMock(
                        tools=[
                            self._create_mock_tool(
                                "sse_tool",
                                "SSE tool",
                                {"type": "object", "properties": {}},
                            )
                        ]
                    )
                )

                tools = await manager.connect(servers)

                assert len(tools) == 1
                assert tools[0].name == "sse_tool"
                await manager.close()
