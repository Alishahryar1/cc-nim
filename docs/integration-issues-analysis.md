# FCC / Claude Desktop Integration Issues — Root Cause Analysis & Fix Strategies

**Scope**: 10 integration issues identified when routing Claude Desktop through FCC local proxy

**Date**: 2026-07-26  
**Context**: Free Claude Code (FCC) v4.13.0 with picker aliasing workaround for NIM models

---

## Issue 1: Tool Approvals in Cowork/Code Tabs

**Symptom**: Tool approval dialogs don't appear or auto-approve incorrectly in Cowork/Code tabs when routed through FCC.

### Root Cause (FCC Codebase)
- **File**: `/free_claude_code/api/handlers/messages.py` (lines 104-106)
- `_reject_unsupported_server_tools()` rejects Anthropic server-side tools (`web_search`, `web_fetch`) unconditionally
- `_run_message_intercepts()` applies local tool intercepts without user consent flow
- No approval callback mechanism exposed to Claude Desktop UI

**Location**: `MessagesHandler.create()` → model resolution → intercepts (no approval gate)

### Feasible Fixes
| Approach | Effort | Risk | Notes |
|----------|--------|------|-------|
| Add `approval_callback` hook to `ModelRouter` | Medium | Low | Requires FCC UI integration; upstream may prefer CLI-only |
| Return `tool_use` with `stop_reason: "tool_use"` + approval metadata | Low | Medium | Desktop polls for approval; works without FCC UI changes |
| Inject approval middleware in SSE stream | Medium | High | Modifies streaming protocol; brittle across versions |

### Workaround
- Use CLI (`fcc-claude`) for Cowork/Code workflows where tool approval matters
- Disable intercepts: set `enable_web_tools=false` in FCC config

### Upstream Dependency
None — entirely within FCC. Could implement approval SSE events as optional extension.

---

## Issue 2: MCP Server Integration

**Symptom**: `mcp_servers` config in `claude_desktop_config.json` is ignored; no tools appear.

### Root Cause (FCC Codebase)
- **Model field exists**: `/free_claude_code/core/anthropic/models.py` lines 134, 152
- **Serialization passes through**: `/free_claude_code/core/anthropic/request_serialization.py` line 24
- **NO provider implements MCP protocol**: DeepSeek explicitly rejects (compat.py:254-257)
- OpenAI-compatible providers ignore the field silently
- No MCP client library in FCC dependencies

### Feasible Fixes
| Approach | Effort | Risk | Notes |
|----------|--------|------|-------|
| Add MCP client to OpenAI-compatible provider | High | Medium | Requires MCP SDK; upstream may not accept |
| Local tool shim registry (MCP server → local function) | Medium | Low | Map MCP servers to FCC local tools at startup |
| Pass-through to Anthropic API (if API key provided) | Low | Low | Only works with real Anthropic key |

### Workaround
- Define tools locally in FCC config (`tools.yaml`) as `type: function` entries
- Use `fcc-claude` CLI for MCP-dependent workflows

### Upstream Dependency
**High** — Requires MCP SDK integration; likely needs RFC in upstream.

---

## Issue 3: Streaming Completion Handling

**Symptom**: Streaming cuts off, duplicate chunks, or reasoning blocks missing in Desktop.

### Root Cause (FCC Codebase)
- **SSE implementation is solid**: `streaming/ledger.py` emits proper event sequence
- **Potential issue**: First-chunk pre-fetching in `response_streams.py:215-251` (`_first_chunk_streaming_response`)
  - If first chunk errors, non-streaming fallback may lose reasoning context
- **Thinking block handling**: `ThinkTagParser` (thinking.py) expects `<? ... ?>` tags
  - Some providers emit different reasoning formats (native, `reasoning_content`, tags)
  - `ReasoningReplayMode` (conversion.py) handles this but must match provider profile

### Feasible Fixes
| Approach | Effort | Risk |
|----------|--------|------|
| Add provider profile validation at startup | Low | Low |
| Emit `reasoning` signature for all thinking blocks | Low | Low |
| Test all provider profiles with reasoning enabled | Medium | Low |

### Workaround
- Use `fcc-claude` CLI (streaming works reliably in terminal)
- Disable thinking: `"thinking": {"enabled": false}` in request

### Upstream Dependency
None — FCC handles this internally.

---

## Issue 4: File Uploads (Images, Documents)

**Symptom**: Images don't render; PDF uploads fail silently.

### Root Cause (FCC Codebase)
- **Images**: ✅ Supported — `ContentBlockImage` → OpenAI `image_url` conversion (conversion.py:195-219)
  - Base64 and URL sources both handled
- **Documents**: ❌ Not forwarded
  - `ContentBlockDocument` accepted in model (models.py:24-29)
  - No conversion path in any provider
  - DeepSeek strips explicitly (compat.py:32-44)
- **No Files API endpoint**: FCC has no `/v1/files` implementation

### Feasible Fixes
| Approach | Effort | Risk |
|----------|--------|------|
| Add document → base64 conversion for OpenAI-compatible | Medium | Medium |
| Implement minimal Files API (local storage + signed URLs) | High | Medium |
| Add provider-specific document handling | Medium | High |

### Workaround
- Images: Use base64 encoding (works with OpenAI-compatible)
- Documents: Extract text first, send as text block

### Upstream Dependency
**Medium** — Files API would be new feature; document support depends on provider capabilities.

---

## Issue 5: Conversation / Message History Sync

**Symptom**: History not shared between Desktop tabs; Cowork/Code tabs show different context.

### Root Cause (Architectural)
- **FCC is stateless proxy**: No conversation storage; each request independent
- **Claude Desktop manages history**: Stored locally in `~/.config/Claude/conversations/`
- **Routing through FCC**: Desktop sends full history each request; FCC forwards to provider
- **No divergence in FCC** — but provider context limits may truncate differently per tab

### Feasible Fixes
| Approach | Effort | Risk |
|----------|--------|------|
| N/A — not an FCC issue | — | — |
| Provider context window awareness | Medium | Low |

### Workaround
- Ensure all tabs use same provider/model (consistent context limits)
- Monitor `usage.input_tokens` to detect truncation

### Upstream Dependency
**None** — this is expected behavior; Desktop manages history.

---

## Issue 6: Model Features (Thinking, Tool Use, Vision)

**Symptom**: Some models lack thinking, tool use, or vision when selected in picker.

### Root Cause (FCC Codebase)
- **Thinking**: ✅ Fully supported — `ThinkingConfig`, stream ledger, `ReasoningReplayMode`
- **Tool Use**: ✅ Fully supported — streaming `tool_use` blocks, heuristic parser fallback
- **Vision**: ⚠️ Partial — images work, but *model capability advertisement* may be incomplete
  - Model catalog (`model_catalog.py`) emits capabilities from provider metadata
  - NIM models may not advertise vision in `/v1/models` payload

### Feasible Fixes
| Approach | Effort | Risk |
|----------|--------|------|
| Enhance model catalog capability detection | Medium | Low |
| Add `vision: true` to NIM model metadata if supported | Low | Low |
| Validate per-model feature matrix at startup | Medium | Low |

### Workaround
- Use picker aliases (`claude-sonnet-nim-XXXX`) — they inherit full Anthropic capability profile
- For direct gateway IDs, check provider docs

### Upstream Dependency
**Low** — FCC handles feature translation; depends on upstream provider metadata accuracy.

---

## Issue 7: Auth / Billing Headers

**Symptom**: "Invalid API key" or billing errors in Desktop when using `ANTHROPIC_API_KEY=freecc`.

### Root Cause (Architectural)
- **FCC ignores `ANTHROPIC_API_KEY`**: Hardcoded to accept any key (or `freecc`)
- **No billing integration**: FCC is free proxy; no usage reporting to Anthropic
- **Desktop expects Anthropic headers**: `anthropic-beta`, `anthropic-version`, usage in response

### Current Behavior (FCC)
- `x-anthropic-billing-header: cc_version=2.1.209.e2b; cc_entrypoint=cli` sent in responses (verify in code)
- Usage tracking: `Usage` model includes `cache_creation_input_tokens`, `cache_read_input_tokens`

### Feasible Fixes
| Approach | Effort | Risk |
|----------|--------|------|
| Pass through Anthropic headers unchanged | Low | Low |
| Add usage tracking endpoint for local analytics | Medium | Low |
| Mock billing header for Desktop compatibility | Low | Low |

### Workaround
- Use `fcc-configure-claude-desktop` which sets correct config
- Accept that billing shows $0 (FCC is free)

### Upstream Dependency
**None** — FCC controls response headers.

---

## Issue 8: Extensions (Voice, Prompt Library, etc.)

**Symptom**: Voice input, prompt library, or extensions don't work through FCC.

### Root Cause
- **Voice**: Requires WebRTC/gRPC to NVIDIA Riva or local Whisper
  - FCC has `voice` optional dependency (grpcio, nvidia-riva-client)
  - No voice endpoint in FCC API — only CLI `fcc-pi` has voice support
- **Prompt Library**: Desktop feature; stores locally; not proxied
- **Extensions**: Desktop loads from Electron app; FCC unaware

### Feasible Fixes
| Approach | Effort | Risk |
|----------|--------|------|
| Add `/v1/voice/transcribe` endpoint | High | Medium |
| Delegate voice to local whisper.cpp via FCC CLI | Medium | Low |
| Document: extensions work natively (Desktop manages) | Zero | Zero |

### Workaround
- Voice: Use `fcc-pi` CLI for voice→text, then paste into Desktop
- Extensions: Native Desktop behavior unaffected by proxy

### Upstream Dependency
**Low** — Voice endpoint would be new FCC feature; extensions are Desktop-local.

---

## Issue 9: Telemetry / Analytics

**Symptom**: Desktop telemetry fails or shows errors in console.

### Root Cause
- **Desktop sends telemetry to Anthropic**: Requires valid API key + billing
- **FCC returns 200 but no real backend**: Telemetry calls may 404 or return mock data
- **No telemetry endpoint in FCC**: `/v1/telemetry` not implemented

### Feasible Fixes
| Approach | Effort | Risk |
|----------|--------|------|
| Implement `/v1/telemetry` no-op endpoint | Low | Zero |
| Proxy telemetry to local file (opt-in) | Medium | Low |
| Document: telemetry disabled in FCC mode | Zero | Zero |

### Workaround
- Disable telemetry in Desktop: `preferences.telemetryEnabled = false`
- Accept console warnings (harmless)

### Upstream Dependency
**None** — No-op endpoint trivial to add.

---

## Issue 10: Voice / Language Support

**Symptom**: Voice input doesn't work; non-English languages have issues.

### Root Cause
- **Voice**: Covered in Issue 8 — no API endpoint
- **Languages**: FCC passes through all text unchanged (UTF-8)
  - No language detection or translation layer
  - Provider-dependent: NIM models support multilingual; OpenAI-compatible vary

### Feasible Fixes
| Approach | Effort | Risk |
|----------|--------|------|
| Document language support per provider | Low | Zero |
| Add language hint in system prompt | Low | Low |

### Workaround
- Use providers with strong multilingual support (Nemotron, Llama-3.3)
- Set system prompt language explicitly

### Upstream Dependency
**None** — Pass-through behavior is correct.

---

## Summary Matrix

| # | Issue | FCC Root Cause | Fix Effort | Workaround Exists? | Upstream Needed? |
|---|-------|----------------|------------|-------------------|------------------|
| 1 | Tool Approvals | No approval callback in intercepts | Medium | ✅ CLI | No |
| 2 | MCP Servers | No provider implements MCP | High | ✅ Local tools | Yes (RFC) |
| 3 | Streaming | First-chunk pre-fetch edge case | Low | ✅ CLI | No |
| 4 | File Uploads | No Files API; docs dropped | Medium/High | ⚠️ Partial | Yes (Files API) |
| 5 | History Sync | Not an FCC issue | — | ✅ Native | No |
| 6 | Model Features | Capability metadata incomplete | Medium | ✅ Picker aliases | No |
| 7 | Auth/Billing | FCC ignores key; no billing | Low | ✅ `freecc` key | No |
| 8 | Extensions/Voice | No voice endpoint; extensions local | Medium/High | ⚠️ CLI voice | Maybe (voice) |
| 9 | Telemetry | No endpoint | Low | ✅ Disable | No |
| 10 | Voice/Languages | No voice API; pass-through text | Low | ✅ Provider choice | No |

---

## Recommended Prioritization for Upstream PR

**Phase 1 (Low effort, high impact)**:
1. Add `/v1/telemetry` no-op endpoint
2. Mock billing header for Desktop compatibility
3. Enhance model catalog with vision/tool-use flags per provider

**Phase 2 (Medium effort)**:
4. Add document → base64 conversion for OpenAI-compatible providers
5. Implement approval SSE events (optional, gated by config)
6. Add voice transcription endpoint (delegate to local whisper.cpp)

**Phase 3 (RFC required)**:
7. MCP server execution (local shim or MCP SDK integration)
8. Files API implementation

---

## Files to Reference for PR

| File | Purpose |
|------|---------|
| `docs/fcc-integration-analysis.md` | Full codebase analysis (7 integration points) |
| `docs/claude-desktop-picker-aliasing.md` | Picker aliasing workaround (v4.13.0) |
| `docs/integration-issues-analysis.md` | This document (10 issues) |
| `src/free_claude_code/api/handlers/messages.py` | Tool approval / intercept logic |
| `src/free_claude_code/providers/openai_chat/provider.py` | Streaming, thinking, reasoning |
| `src/free_claude_code/core/anthropic/models.py` | All Anthropic protocol models |
| `src/free_claude_code/core/anthropic/conversion.py` | Format translation |
| `pyproject.toml` | Dependencies (voice optional) |

---

*End of analysis. Ready for upstream contribution preparation.*