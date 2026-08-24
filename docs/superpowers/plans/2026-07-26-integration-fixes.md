# FCC Integration Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three priority integration fixes: native tool approvals flow, MCP SDK integration, Files/Voice proxy endpoints

**Architecture:** Three independent phases. Phase 1 disables auto-intercepts so Desktop drives tool approval loop natively. Phase 2 adds MCP client to OpenAIChatProvider with security guardrails. Phase 3 adds `/v1/files` and `/v1/audio/transcriptions` handlers with text extraction fallback for documents.

**Tech Stack:** Python 3.14+, FastAPI, mcp>=1.0.0, pypdf>=5.0, existing FCC codebase patterns

## Global Constraints

- All features opt-in via `Settings` (fcc.toml) — feature flags default to safe values
- No breaking changes to existing API — backward compatible
- Follow existing code style: `from __future__` removed, `contextlib.suppress`, type hints
- CI must pass: suppressions, ruff-format, ruff-check, ty, pytest
- Trace events for all new operations with `request_id` correlation

---

## Phase 1: Native Tool Approvals (Issue 1)

### Task 1.1: Add `enable_native_tool_approvals` Setting

**Files:**
- Create: (none)
- Modify: `src/free_claude_code/config/settings.py`
- Test: `tests/config/test_settings.py`

**Interfaces:**
- Produces: `Settings.enable_native_tool_approvals: bool` (default `True`)

```markdown
- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_settings.py
def test_settings_enable_native_tool_approvals_default_true():
    from free_claude_code.config.settings import Settings
    s = Settings()
    assert s.enable_native_tool_approvals is True

def test_settings_enable_native_tool_approvals_configurable():
    from free_claude_code.config.settings import Settings
    s = Settings(enable_native_tool_approvals=False)
    assert s.enable_native_tool_approvals is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_settings.py::test_settings_enable_native_tool_approvals_default_true -v`
Expected: FAIL — attribute missing

- [ ] **Step 3: Write minimal implementation**

```python
# src/free_claude_code/config/settings.py
# Add to Settings class:
enable_native_tool_approvals: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_settings.py::test_settings_enable_native_tool_approvals_default_true -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/free_claude_code/config/settings.py tests/config/test_settings.py
git commit -m "feat: add enable_native_tool_approvals setting"
```

### Task 1.2: Disable Web Server Tool Intercept When Tools Present

**Files:**
- Modify: `src/free_claude_code/api/handlers/messages.py`
- Test: `tests/api/test_messages_handler.py`

**Interfaces:**
- Consumes: `Settings.enable_native_tool_approvals` from Task 1.1
- Produces: `_intercept_web_server_tool` returns `None` when request has tools AND setting enabled

```markdown
- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_messages_handler.py
@pytest.mark.asyncio
async def test_intercept_web_server_tool_skipped_when_tools_present():
    from free_claude_code.api.handlers.messages import MessagesHandler
    from free_claude_code.config.settings import Settings
    from free_claude_code.core.anthropic.models import MessagesRequest, Tool
    
    settings = Settings(enable_native_tool_approvals=True)
    handler = MessagesHandler(settings=settings, provider_resolver=lambda x: None)
    
    request = MessagesRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        tools=[Tool(name="web_search", description="", input_schema={})]
    )
    
    routed = MagicMock()
    routed.request = request
    
    result = handler._intercept_web_server_tool(routed)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_messages_handler.py::test_intercept_web_server_tool_skipped_when_tools_present -v`
Expected: FAIL — intercept returns result

- [ ] **Step 3: Write minimal implementation**

```python
# src/free_claude_code/api/handlers/messages.py
def _intercept_web_server_tool(
    self, routed: RoutedMessagesRequest
) -> _MessagesResult | None:
    if not self._settings.enable_web_server_tools:
        return None
    if not is_web_server_tool_request(routed.request):
        return None
    # NEW: skip if native tool approvals enabled and request has tools
    if self._settings.enable_native_tool_approvals and routed.request.tools:
        return None
    # ... rest unchanged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_messages_handler.py::test_intercept_web_server_tool_skipped_when_tools_present -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/free_claude_code/api/handlers/messages.py tests/api/test_messages_handler.py
git commit -m "feat: disable web tool intercept when native approvals enabled and tools present"
```

### Task 1.3: Disable Local Optimization Intercept When Tools Present

**Files:**
- Modify: `src/free_claude_code/api/handlers/messages.py`
- Test: `tests/api/test_messages_handler.py`

```markdown
- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_messages_handler.py
@pytest.mark.asyncio
async def test_intercept_local_optimization_skipped_when_tools_present():
    from free_claude_code.api.handlers.messages import MessagesHandler
    from free_claude_code.config.settings import Settings
    from free_claude_code.core.anthropic.models import MessagesRequest, Tool
    
    settings = Settings(enable_native_tool_approvals=True)
    handler = MessagesHandler(settings=settings, provider_resolver=lambda x: None)
    
    request = MessagesRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        tools=[Tool(name="some=[Tool(name="any_tool", description="", input_schema={})]
    )
    
    routed = MagicMock()
    routed.request = request
    
    result = handler._intercept_local_optimization(routed)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_messages_handler.py::test_intercept_local_optimization_skipped_when_tools_present -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/free_claude_code/api/handlers/messages.py
def _intercept_local_optimization(
    self, routed: RoutedMessagesRequest
) -> _MessagesResult | None:
    if self._settings.enable_native_tool_approvals and routed.request.tools:
        return None
    # ... rest unchanged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_messages_handler.py::test_intercept_local_optimization_skipped_when_tools_present -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/free_claude_code/api/handlers/messages.py tests/api/test_messages_handler.py
git commit -m "feat: disable local optimization intercept when native approvals enabled and tools present"
```

### Task 1.4: Integration Test — Tool Use → Stop Reason → Tool Result Loop

**Files:**
- Test: `tests/api/test_tool_approval_flow.py` (new)

```markdown
- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_tool_approval_flow.py
@pytest.mark.asyncio
async def test_tool_use_emits_stop_reason_tool_use():
    """First request returns tool_use + stop_reason: tool_use; second with tool_result executes."""
    # Uses test client against /v1/messages endpoint
    # Request contains tools + user message triggering tool
    # Verify SSE stream ends with message_stop, stop_reason: tool_use
    pass

@pytest.mark.asyncio
async def test_tool_result_in_next_request_executes():
    """Follow-up request with tool_result processes correctly."""
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_tool_approval_flow.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation** (verify existing provider behavior handles this)
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

## Phase 2: MCP SDK Integration (Issue 2)

### Task 2.1: Add MCP Optional Dependency

**Files:**
- Modify: `pyproject.toml`

```markdown
- [ ] **Step 1: Add dependency**

```toml
[project.optional-dependencies]
mcp = ["mcp>=1.0.0"]
```

- [ ] **Step 2: Verify install**

Run: `uv sync --extra mcp`
Expected: mcp installed

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add mcp optional dependency"
```

### Task 2.2: Add `enable_mcp` Setting

**Files:**
- Modify: `src/free_claude_code/config/settings.py`
- Test: `tests/config/test_settings.py`

```markdown
- [ ] **Step 1: Write failing test**

```python
def test_settings_enable_mcp_default_false():
    from free_claude_code.config.settings import Settings
    s = Settings()
    assert s.enable_mcp is False
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# src/free_claude_code/config/settings.py
enable_mcp: bool = False
```

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

### Task 2.3: Create MCPClientManager

**Files:**
- Create: `src/free_claude_code/providers/openai_chat/mcp_client.py`
- Test: `tests/providers/test_mcp_client.py`

**Interfaces:**
- Produces: `MCPClientManager` class with methods:
  - `connect(servers: list[MCPServerConfig]) -> list[MCPTool]`
  - `execute_tool(tool_name: str, arguments: dict) -> ToolResult`
  - `close() -> None`
- Consumes: `mcp` SDK, `Settings.enable_mcp`

```markdown
- [ ] **Step 1: Write failing test**

```python
# tests/providers/test_mcp_client.py
@pytest.fixture
def mock_mcp_server():
    # stdio mock that responds to initialize, list_tools, call_tool
    pass

@pytest.mark.asyncio
async def test_mcp_client_manager_connects_and_lists_tools(mock_mcp_server):
    from free_claude_code.providers.openai_chat.mcp_client import MCPClientManager, MCPServerConfig
    
    manager = MCPClientManager()
    servers = [MCPServerConfig(transport="stdio", command="uvx", args=["mcp-server-git"])]
    
    tools = await manager.connect(servers)
    assert len(tools) > 0
    assert all("name" in t and "inputSchema" in t for t in tools)
    await manager.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_mcp_client.py::test_mcp_client_manager_connects_and_lists_tools -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# src/free_claude_code/providers/openai_chat/mcp_client.py
"""MCP client manager for per-request MCP server lifecycle."""

import asyncio
from dataclasses import dataclass
from typing import Any, Literal
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
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
        self._streams: list[Any] = []

    async def connect(self, servers: list[MCPServerConfig]) -> list[MCPTool]:
        all_tools: list[MCPTool] = []
        for config in servers:
            if config.transport == "stdio":
                params = StdioServerParameters(
                    command=config.command or "",
                    args=config.args or [],
                )
                read_stream, write_stream = await stdio_client(params).__aenter__()
                self._streams.append((read_stream, write_stream))
                session = ClientSession(read_stream, write_stream)
            else:  # sse
                # SSE transport not in stdio_client — use sse_client
                from mcp.client.sse import sse_client

                read_stream, write_stream = await sse_client(config.url).__aenter__()
                self._streams.append((read_stream, write_stream))
                session = ClientSession(read_stream, write_stream)

            await session.__aenter__()
            await session.initialize()
            self._sessions.append(session)

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                all_tools.append(
                    MCPTool(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema,
                    )
                )
        return all_tools

    async def execute_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        for session in self._sessions:
            try:
                result = await session.call_tool(tool_name, arguments)
                return ToolResult(
                    content=result.content,
                    is_error=result.isError,
                )
            except Exception:
                continue
        raise ValueError(f"Tool {tool_name} not found in any MCP server")

    async def close(self) -> None:
        for session in self._sessions:
            await session.__aexit__(None, None, None)
        for read_stream, write_stream in self._streams:
            await read_stream.__aexit__(None, None, None)
            await write_stream.__aexit__(None, None, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_mcp_client.py::test_mcp_client_manager_connects_and_lists_tools -v`
Expected: PASS (with mock)

- [ ] **Step 5: Commit**

```bash
git add src/free_claude_code/providers/openai_chat/mcp_client.py tests/providers/test_mcp_client.py
git commit -m "feat: add MCPClientManager for per-request MCP lifecycle"
```

### Task 2.4: Add Schema Mapping (MCP JSON Schema → OpenAI Function Declaration)

**Files:**
- Modify: `src/free_claude_code/providers/openai_chat/mcp_client.py`
- Test: `tests/providers/test_mcp_client.py`

```markdown
- [ ] **Step 1: Write failing test**

```python
# tests/providers/test_mcp_client.py
def test_mcp_tool_to_openai_function_declaration():
    from free_claude_code.providers.openai_chat.mcp_client import mcp_tool_to_openai_function
    
    mcp_tool = MCPTool(
        name="git_status",
        description="Get git status",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    )
    
    openai_func = mcp_tool_to_openai_function(mcp_tool)
    
    assert openai_func["type"] == "function"
    assert openai_func["function"]["name"] == "git_status"
    assert openai_func["function"]["description"] == "Get git status"
    assert openai_func["function"]["parameters"] == mcp_tool.input_schema
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# In mcp_client.py, add:
def mcp_tool_to_openai_function(tool: MCPTool) -> dict[str, Any]:
    """Convert MCP tool to OpenAI function declaration format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

### Task 2.5: Security Guardrail — Localhost Only

**Files:**
- Modify: `src/free_claude_code/providers/openai_chat/mcp_client.py`
- Modify: `src/free_claude_code/providers/openai_chat/provider.py`
- Test: `tests/providers/test_mcp_client.py`

```markdown
- [ ] **Step 1: Write failing test**

```python
# tests/providers/test_mcp_client.py
@pytest.mark.asyncio
async def test_mcp_skipped_for_non_localhost_request():
    # When request.client.host != localhost, MCPClientManager.connect should return []
    pass
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Modify connect to accept request context**

```python
# In mcp_client.py
async def connect(
    self, servers: list[MCPServerConfig], client_host: str | None = None
) -> list[MCPTool]:
    if client_host not in ("127.0.0.1", "::1", "localhost", None):
        logger.warning("MCP skipped for non-localhost request: {}", client_host)
        return []
    # ... rest unchanged
```

- [ ] **Step 4: In provider.py, pass client host**

```python
# In stream_response, before calling manager.connect():
client_host = getattr(request, "client", None)
if client_host:
    client_host = client_host.host
```

- [ ] **Step 5: Run test to verify it passes**
- [ ] **Step 6: Commit**

### Task 2.6: Integrate MCP in OpenAIChatProvider.stream_response

**Files:**
- Modify: `src/free_claude_code/providers/openai_chat/provider.py`
- Test: `tests/providers/test_provider_mcp.py` (new)

```markdown
- [ ] **Step 1: Write failing test**

```python
# tests/providers/test_provider_mcp.py
@pytest.mark.asyncio
async def test_provider_includes_mcp_tools_in_upstream_request():
    # Mock provider with mcp_servers in request
    # Verify tools sent upstream include MCP-discovered tools
    pass

@pytest.mark.asyncio
async def test_provider_executes_mcp_tool_via_client():
    # Mock MCPClientManager.execute_tool
    # Verify tool_result emitted in stream
    pass
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write implementation in provider.py**

```python
# In stream_response, after preflight_stream:
if request.mcp_servers and self._config.enable_mcp:
    from .mcp_client import (
        MCPClientManager,
        MCPServerConfig,
        mcp_tool_to_openai_function,
    )

    manager = MCPClientManager()
    try:
        mcp_configs = [
            MCPServerConfig(
                transport=s.get("transport", "stdio"),
                command=s.get("command"),
                args=s.get("args"),
                url=s.get("url"),
                headers=s.get("headers"),
            )
            for s in request.mcp_servers
        ]

        client_host = request.client.host if getattr(request, "client", None) else None
        mcp_tools = await manager.connect(mcp_configs, client_host=client_host)

        # Convert to OpenAI format and merge with existing tools
        mcp_openai_tools = [mcp_tool_to_openai_function(t) for t in mcp_tools]
        if routed_request.request.tools:
            all_tools = list(routed_request.request.tools) + mcp_openai_tools
        else:
            all_tools = mcp_openai_tools

        # Store manager for tool execution during streaming
        self._mcp_manager = manager

    except Exception as e:
        logger.warning("MCP initialization failed: {}", e)
        await manager.close()

# In streaming loop, when tool_use detected for MCP tool:
if hasattr(self, "_mcp_manager") and tool_name in mcp_tool_names:
    result = await self._mcp_manager.execute_tool(tool_name, arguments)
    # Emit tool_result block
    yield format_sse_event(...)

# In finally block:
if hasattr(self, "_mcp_manager"):
    await self._mcp_manager.close()
```

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

## Phase 3: Files/Voice Proxy Endpoints (Issues 4, 8)

### Task 3.1: Add pypdf Optional Dependency

**Files:**
- Modify: `pyproject.toml`

```markdown
- [ ] **Step 1: Add dependency**

```toml
[project.optional-dependencies]
voice = ["grpcio>=1.82.1", "grpcio-tools>=1.81.1", "nvidia-riva-client>=2.26.0", "pypdf>=5.0.0"]
```

- [ ] **Step 2: Verify install**

Run: `uv sync --extra voice`
Expected: pypdf installed

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add pypdf for document text extraction"
```

### Task 3.2: Add File Storage Handler

**Files:**
- Create: `src/free_claude_code/api/handlers/files.py`
- Create: `src/free_claude_code/api/handlers/__init__.py` (if missing)
- Test: `tests/api/test_files_handler.py`

**Interfaces:**
- Produces: `POST /v1/files`, `GET /v1/files/{file_id}/content`
- Storage: local temp dir with TTL cleanup

```markdown
- [ ] **Step 1: Write failing test**

```python
# tests/api/test_files_handler.py
@pytest.mark.asyncio
async def test_upload_file_returns_file_object():
    from fastapi.testclient import TestClient
    from free_claude_code.api.main import app
    
    client = TestClient(app)
    response = client.post("/v1/files", files={"file": ("test.txt", b"hello", "text/plain")})
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["bytes"] == 5
    assert data["filename"] == "test.txt"
    assert data["purpose"] == "assistants"

@pytest.mark.asyncio
async def test_download_file_content():
    # Upload then download
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_files_handler.py -v`
Expected: FAIL — 404

- [ ] **Step 3: Write implementation**

```python
# src/free_claude_code/api/handlers/files.py
"""Files API handler for document upload/download."""

import uuid
import time
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["files"])


class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


UPLOAD_DIR = Path("/tmp/fcc_files")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
FILE_TTL_SECONDS = 3600  # 1 hour

_file_metadata: dict[str, dict[str, Any]] = {}


@router.post("/files")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = "assistants",
) -> FileObject:
    file_id = f"file-{uuid.uuid4().hex}"
    content = await file.read()

    file_path = UPLOAD_DIR / file_id
    file_path.write_bytes(content)

    metadata = {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": int(time.time()),
        "filename": file.filename or "unknown",
        "purpose": purpose,
        "path": str(file_path),
    }
    _file_metadata[file_id] = metadata

    # Cleanup old files (simple approach)
    _cleanup_expired_files()

    return FileObject(**metadata)


@router.get("/files/{file_id}/content")
async def download_file_content(file_id: str) -> StreamingResponse:
    metadata = _file_metadata.get(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = Path(metadata["path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File content missing")

    def iter_file():
        with file_path.open("rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{metadata["filename"]}"'
        },
    )


def _cleanup_expired_files() -> None:
    now = time.time()
    expired = [
        fid
        for fid, meta in _file_metadata.items()
        if now - meta["created_at"] > FILE_TTL_SECONDS
    ]
    for fid in expired:
        meta = _file_metadata.pop(fid)
        Path(meta["path"]).unlink(missing_ok=True)
```

- [ ] **Step 4: Register router in main app**

```python
# src/free_claude_code/api/main.py
from free_claude_code.api.handlers.files import router as files_router

app.include_router(files_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/api/test_files_handler.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/free_claude_code/api/handlers/files.py src/free_claude_code/api/main.py tests/api/test_files_handler.py
git commit -m "feat: add /v1/files upload and download endpoints"
```

### Task 3.3: Document Text Extraction Fallback in Conversion

**Files:**
- Modify: `src/free_claude_code/core/anthropic/conversion.py`
- Test: `tests/core/test_conversion_document.py`

```markdown
- [ ] **Step 1: Write failing test**

```python
# tests/core/test_conversion_document.py
def test_openai_user_document_part_extracts_text_when_provider_lacks_support():
    from free_claude_code.core.anthropic.conversion import _openai_user_document_part
    from free_claude_code.providers.openai_chat.profiles import OpenAIChatProfile
    
    # Profile with supports_documents=False
    profile = OpenAIChatProfile(
        provider_name="test",
        base_url="http://localhost",
        supports_documents=False,
    )
    
    # Upload a test PDF first, get file_id
    # Then call with document block referencing file_id
    # Verify returns text block with extracted content
    pass
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# In conversion.py
def _openai_user_document_part(
    block: ContentBlockDocument,
    request: MessagesRequest,
    profile: OpenAIChatProfile,
) -> dict[str, Any] | None:
    source = block.source

    # If provider supports documents natively, try to pass through
    if profile.supports_documents:
        # ... existing logic for native document support
        return _openai_document_native(block)

    # FALLBACK: Extract text locally
    if source.get("type") == "file_id":
        file_id = source["file_id"]
        # Read from file storage
        from free_claude_code.api.handlers.files import _file_metadata

        meta = _file_metadata.get(file_id)
        if not meta:
            return None

        file_path = Path(meta["path"])
        if file_path.suffix.lower() == ".pdf":
            import pypdf

            reader = pypdf.PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            extracted = "\n\n".join(text_parts)
        else:
            # Plain text, markdown, etc.
            extracted = file_path.read_text()

        return {
            "type": "text",
            "text": f"<document name='{meta['filename']}'>\n{extracted}\n</document>",
        }

    return None
```

- [ ] **Step 4: Add supports_documents to profile**

```python
# src/free_claude_code/providers/openai_chat/profiles.py
@dataclass
class OpenAIChatProfile:
    # ... existing fields
    supports_documents: bool = False
```

- [ ] **Step 5: Run test to verify it passes**
- [ ] **Step 6: Commit**

### Task 3.4: Voice Transcription Handler

**Files:**
- Create: `src/free_claude_code/api/handlers/audio.py`
- Test: `tests/api/test_audio_handler.py`

```markdown
- [ ] **Step 1: Write failing test**

```python
# tests/api/test_audio_handler.py
@pytest.mark.asyncio
async def test_transcribe_audio_returns_text():
    from fastapi.testclient import TestClient
    from free_claude_code.api.main import app
    
    client = TestClient(app)
    # Use a small test audio file
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", SAMPLE_WAV_BYTES, "audio/wav")},
        data={"model": "parakeet", "language": "en"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert data["language"] == "en"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_audio_handler.py -v`
Expected: FAIL — 404

- [ ] **Step 3: Write implementation**

```python
# src/free_claude_code/api/handlers/audio.py
"""Audio transcription endpoint proxying to upstream providers."""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["audio"])


class TranscriptionResponse(BaseModel):
    text: str
    language: str


@router.post("/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("parakeet"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
) -> TranscriptionResponse:
    # Read audio content
    audio_content = await file.read()

    # Determine provider from model or settings
    # For now, proxy to NVIDIA NIM parakeet or OpenAI-compatible
    from free_claude_code.config.settings import Settings

    settings = Settings()
    provider_name = settings.voice_provider or "nvidia_nim/parakeet"

    # Get provider from resolver
    from free_claude_code.application.ports import ProviderResolver
    # ... resolve provider and call audio endpoint

    # Simplified: return placeholder
    # Real impl: call provider with audio bytes

    return TranscriptionResponse(
        text="[Transcription would be here]",
        language=language or "en",
    )
```

- [ ] **Step 4: Register router in main app**
- [ ] **Step 5: Implement actual provider proxy** (connect to NVIDIA NIM or OpenAI Whisper)
- [ ] **Step 6: Run test to verify it passes**
- [ ] **Step 7: Commit**

### Task 3.5: Add `voice_provider` Setting

**Files:**
- Modify: `src/free_claude_code/config/settings.py`
- Test: `tests/config/test_settings.py`

```markdown
- [ ] **Step 1: Write failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write implementation**
```python
voice_provider: str | None = None
```
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

## Final Validation

### Task 4.1: Run Full CI Suite

```markdown
- [ ] **Step 1: Run all CI checks**

Run: `./scripts/ci.sh`
Expected: All checks PASS (suppressions, ruff-format, ruff-check, ty, pytest)

- [ ] **Step 2: Fix any failures**
- [ ] **Step 3: Commit final fixes**
```

### Task 4.2: Manual Integration Test Checklist

```markdown
- [ ] Tool Approvals: Desktop picker → chat → tool_use → approval → tool_result → response
- [ ] MCP: Desktop config with mcp_servers → FCC → MCP server → tool execution
- [ ] Files: Upload PDF → document block in chat → text extraction fallback works
- [ ] Voice: POST audio → transcription returned
```

---

## Plan Complete

**Execution options:**

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks
2. **Inline Execution** - Execute tasks in this session using executing-plans

**Which approach?**