# FCC Integration Fixes — Unified Design Spec

**Date**: 2026-07-26  
**Scope**: Three priority integration fixes for Claude Desktop ↔ FCC proxy

---

## 1. Tool Approvals Callback Mechanism (Issue 1)

### Problem
Claude Desktop's Cowork/Code tabs need native tool approval dialogs. Currently FCC auto-executes intercepts without user consent.

### Root Cause
`MessagesHandler.create()` → `_intercept_web_server_tool()` / `_intercept_local_optimization()` execute immediately. Desktop expects standard Anthropic protocol: model emits `tool_use` block → turn ends with `stop_reason: "tool_use"` → Desktop shows approval UI → user clicks → Desktop sends new request with `tool_result`.

### Correct Design: Pass-Through Tool Use Flow

**No custom SSE events. No polling. Standard protocol only.**

**Changes**:

1. **Settings**: `enable_native_tool_approvals: bool = True` (on by default)
2. **Remove auto-execution in intercepts**: 
   - `_intercept_web_server_tool()` → return `None` if request has tools that could be called
   - `_intercept_local_optimization()` → return `None` if tools present
3. **Let provider streaming handle tool_use**: OpenAIChatProvider already emits proper `tool_use` blocks via `OpenAIToolCallAssembler`
4. **Desktop drives the loop**:
   - Turn 1: Model returns `tool_use` + `stop_reason: "tool_use"` → Desktop shows approval
   - User clicks "Allow" → Desktop POSTs `/v1/messages` with `tool_result`
   - Turn 2: FCC routes `tool_result` to provider → execution happens

**For intercepts that MUST execute locally** (web_search/web_fetch):
- Only execute if `enable_web_server_tools` AND no upstream tool approval needed
- OR: expose as regular `tools` in request so Desktop approves them too
- Simplest: disable intercepts when `enable_native_tool_approvals=True` and tools exist

**CLI (`fcc-claude`)**: No change — already works via turn-based tool loop.

### Testing
- Unit: Verify `_run_message_intercepts` returns `None` when tools in request
- Integration: Emit `tool_use` + `stop_reason: tool_use` → Desktop approval → `tool_result` in next request → execution

---

## 2. Full MCP SDK Integration (Issue 2)

### Problem
`mcp_servers` in Desktop config ignored. No provider implements MCP.

### Design: MCP Client in OpenAI-Chat Provider (with guardrails)

**Dependency**: `mcp>=1.0.0` in `[project.optional-dependencies] mcp`

**Architecture**:
```
Claude Desktop → FCC /v1/messages (with mcp_servers list)
               → OpenAIChatProvider.stream_response()
               → MCPClientManager (per-request lifecycle)
               → stdio/SSE transport per configured server
               → Tool discovery: list_tools() on connect
               → Schema mapping: MCP JSON Schema → OpenAI function declaration
               → Tool execution: call_tool() with timeout
               → Results returned as tool_result blocks
```

**Components**:

1. **`free_claude_code/providers/openai_chat/mcp_client.py`**
   - `MCPClientManager`: lifecycle per request
   - `MCPServerConfig`: parsed from `mcp_servers` request field
   - Transport: stdio (local) or SSE (remote)
   - **Security guardrail**: Only启动 MCP if `request.client.host in {"127.0.0.1", "::1", "localhost"}` (via Starlette request)
   - Tool discovery: `list_tools()` on connect
   - **Schema mapping guardrail**: Convert MCP JSON Schema → OpenAI `{"type": "function", "function": {...}}` for generic upstream
   - Tool execution: `call_tool()` with 30s timeout
   - Cleanup on request end (success or error)

2. **Integration in `provider.py` (`stream_response`)**:
   - If `request.mcp_servers` and `settings.enable_mcp`:
     - Start MCP clients, discover tools
     - Merge MCP tools into `tools` sent to upstream
     - Stream response; on `tool_use` for MCP tool:
       - Execute via MCP client
       - Emit `tool_result` block to stream
   - Error handling: MCP connect failure → log, continue without MCP tools

### Testing
- Unit: Mock MCP server (stdio); verify schema mapping + tool execution
- Integration: End-to-end with `uvx mcp-server-git` or similar
- Security: Verify non-localhost requests skip MCP

---

## 3. Files/Voice Proxy to Upstream Providers (Issues 4, 8)

### Problem
- No `/v1/files` endpoint; document blocks dropped
- No `/v1/voice/transcribe`; voice input broken
- Standard LLMs (NIM llama-3.3-70b, DeepSeek) don't natively parse PDFs

### Design: Proxy Endpoints + Text Extraction Fallback

**Files API** (`/v1/files` + `/v1/files/{id}/content`):

1. **New handler** (`free_claude_code/api/handlers/files.py`):
   - `POST /v1/files` (multipart) → upload to local temp storage (TTL 1h)
   - Return `FileObject`: `{id, bytes, filename, purpose, created_at}`
   - `GET /v1/files/{id}/content` → stream file bytes
   - Cleanup: background task sweeps expired files

2. **Document block conversion** in `conversion.py`:
   - `_openai_user_document_part()` → if provider supports `document` natively: pass through
   - **Fallback for non-document providers**: Extract text locally (`pypdf`/`pdfplumber`)
     - Inject as text block: `{"type": "text", "text": "<document name='...'>[Extracted Text]</document>"}`
     - Requires new optional dep: `pypdf>=5.0` in `voice` or new `documents` extra

3. **Provider capability flag**: `supports_documents: bool` in `OpenAIChatProfile`

**Voice API** (`/v1/audio/transcriptions`):

1. **New handler** (`free_claude_code/api/handlers/audio.py`):
   - `POST /v1/audio/transcriptions` (multipart: file, model, language, prompt)
   - Proxy to configured upstream with audio support
   - Current targets: NVIDIA NIM `parakeet` or OpenAI-compatible Whisper
   - Return standard: `{text: "...", language: "..."}`

2. **Settings**: `voice_provider: str | None` — which provider handles voice

### Testing
- Files: Upload → download round-trip; document block → text fallback injection
- Voice: POST audio → get transcription; ensure multilingual

---

## Cross-Cutting Concerns

### Configuration
All features opt-in via `Settings` (fcc.toml):
```toml
[features]
enable_native_tool_approvals = true
enable_mcp = false
voice_provider = "nvidia_nim/parakeet"
```

### Dependencies
```toml
[project.optional-dependencies]
mcp = ["mcp>=1.0.0"]
voice = ["grpcio", "nvidia-riva-client", "pypdf>=5.0"]
```

### Observability
- Trace events: `mcp.client.connected`, `mcp.tool.executed`, `files.uploaded`, `voice.transcribed`
- Structured logs with `request_id` correlation

---

## File Map (New/Modified)

| Path | Role |
|------|------|
| `src/free_claude_code/api/handlers/messages.py` | Disable intercepts when tools present |
| `src/free_claude_code/providers/openai_chat/mcp_client.py` | MCP client manager (NEW) |
| `src/free_claude_code/providers/openai_chat/provider.py` | MCP integration + security guard |
| `src/free_claude_code/api/handlers/files.py` | Files API handler (NEW) |
| `src/free_claude_code/api/handlers/audio.py` | Voice transcription handler (NEW) |
| `src/free_claude_code/core/anthropic/conversion.py` | Document text extraction fallback |
| `pyproject.toml` | New optional deps (mcp, pypdf) |
| `src/free_claude_code/config/settings.py` | Feature flags |

---

## Rollout Order

1. **Phase 1**: Tool Approvals — disable intercepts when tools present (lowest risk)
2. **Phase 2**: MCP SDK — new dep, provider-side only, security guardrails
3. **Phase 3**: Files/Voice — new endpoints, upstream proxy + text extraction

Each phase independently testable; no cross-dependencies.

---

*Design ready for review. Approve to proceed to implementation plan.*