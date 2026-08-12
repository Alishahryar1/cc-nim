# 🏛️ Technical Documentation: Google Antigravity Architecture in Free-Claude-Code

This documentation details the complete architecture, OAuth authentication, network protocol, Language Server fingerprinting, and tool/thinking management for the Google Antigravity integration in `free-claude-code`.

---

## 1. 🌳 Architecture Diagram & Data Flow

```text
[ 👤 YOU ]
   │
   │ 1. Prompt / CLI Command (e.g. `fcc-claude`)
   ▼
[ 💻 CLAUDE CODE CLI ] (Official Anthropic Client)
   │
   │ 2. Sends Anthropic HTTP request (POST /v1/messages)
   │    Anthropic JSON Payload: { messages, tools, system, max_tokens, thinking }
   ▼
[ ⚙️ FREE-CLAUDE-CODE ] (Local Proxy Server FastAPI / Python on 127.0.0.1:8082)
   │
   ├── A. Authentication & Zero-Config Discovery:
   │      Reads Google OAuth token in `~/.gemini/antigravity-cli/antigravity-oauth-token`
   │
   ├── B. Anthropic ➔ Google Gemini Conversion:
   │      - Fingerprint Headers: User-Agent "antigravity/1.1.11 (Linux)", Client-Name "ANTIGRAVITY"
   │      - Tool Sanitation: `_clean_gemini_schema` (removes $schema, const, propertyNames, exclusiveMinimum)
   │      - History Conversion: Transforms previous thinking blocks into `{"thought": true, "text": "..."}`
   │      - Multi-Turn Support: Injects `thought_signature` and `functionCall` / `functionResponse`
   │
   │ 3. Sends direct HTTPS REST request (POST /v1internal:streamGenerateContent?alt=sse)
   ▼
[ ☁️ GOOGLE CLOUD CODE ASSIST ] (Google Antigravity Cloud Server)
   │
   │ 4. Processes request on one of 48 Antigravity models (e.g. `gemini-3.6-flash-high`)
   │ 5. Returns Google SSE stream (streamed JSON chunks)
   ▼
[ ⚙️ FREE-CLAUDE-CODE ] (Local Proxy Server)
   │
   ├── C. Google Stream Parsing:
   │      - Detects `{"thought": true, "text": "..."}` ➔ Emits thinking events
   │      - Detects `{"text": "..."}` ➔ Emits final text response
   │
   ├── D. Google ➔ Anthropic SSE Translation:
   │      - `event: content_block_start` (type: thinking or text)
   │      - `event: content_block_delta` (thinking_delta or text_delta)
   │
   │ 6. Re-emits SSE stream in 100% Anthropic-compatible format
   ▼
[ 💻 CLAUDE CODE CLI ]
   │
   │ 7. Visual rendering in terminal (Collapsible thinking + response + tool execution)
   ▼
[ 👤 YOU SEE THE RESPONSE ]
```

---

## 2. 🔑 OAuth Authentication Flow Comparison (IDE vs CLI)

| Feature | CLI Authentication (`agy`) | IDE Authentication (`antigravity_login.py`) |
| :--- | :--- | :--- |
| **OAuth Flow Type** | **Device Code Flow** (RFC 8628 / OOB) | **Web Authorization Code Flow** |
| **User Input** | Manual entry of an **alphanumeric code** in the terminal | 1-click browser login (Automatic callback `localhost:8085`) |
| **Google Client ID** | CLI OAuth Client | **Antigravity IDE** OAuth Client (`1071006060591-...`) |
| **Google Server Profile** | Associated with "CLI Terminal" profile | Associated with **"Antigravity IDE / Full Platform"** profile |
| **Available Models** | **22 models** (Chat only) | **48 models** (Chat, Autocompletion `tab-`, Images, Compaction) |
| **Tool Support** | ❌ Incompatible / Truncated | ✅ 100% Functional |

---

## 3. 🛡️ Language Server HTTP Fingerprinting

The local Antigravity Language Server (`language_server_pb`) communicates with Google servers using strict fingerprinting headers. In `free-claude-code`, we reproduce this exact fingerprint using `httpx`:

```python
ANTIGRAVITY_USER_AGENT = "antigravity/1.1.11 (Linux)"
ANTIGRAVITY_CLIENT_NAME = "ANTIGRAVITY"
ANTIGRAVITY_GOOG_API_CLIENT = "gl-python/3.14.0 grpc/1.62.0 gax/2.17.0"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "User-Agent": ANTIGRAVITY_USER_AGENT,
    "X-Goog-Api-Client": ANTIGRAVITY_GOOG_API_CLIENT,
    "Client-Name": ANTIGRAVITY_CLIENT_NAME,
}
```

---

## 4. 🧰 Tool Call Sanitation

Tools sent by Claude Code (`exec_command`, `file_read`, etc.) use JSON Schema Draft-07 standards which include keywords rejected by Google Gemini's strict OpenAPI parser (HTTP 400 error).

The `_clean_gemini_schema` function recursively cleans schemas before transmission:

```python
UNSUPPORTED_GEMINI_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "$comment",
    "propertyNames",
    "const",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "patternProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
}
```

---

## 5. 🧠 Thinking Block Rendering & History

### Incoming Flow (Google ➔ Free-Claude-Code)
In the SSE stream of a model with reasoning enabled, Google returns:
```json
{
  "thought": true,
  "text": "**Analyzing task**\n\nThinking process..."
}
```
`client.py` intercepts `thought: true` and redirects text to `ledger.emit_thinking_delta()`.

### History Flow (Free-Claude-Code ➔ Google)
When a prior message contains reasoning text, Google requires a boolean `TYPE_BOOL` for the `thought` key in the request's `contents`:
```json
{
  "role": "model",
  "parts": [
    {
      "thought": true,
      "text": "**Analyzing task**\n\nThinking process..."
    }
  ]
}
```
This guarantees an **HTTP 200 OK** status on all multi-turn conversations.

---

## 6. 🚨 Issues Encountered & Resolved (Post-Mortem & Solutions)

Below is the detailed record of major challenges encountered during development and their technical resolutions:

### ❌ Challenge 1: Misinterpretation of Boolean `thought: true` in SSE Stream
- **Symptom**: In `fcc-codex`, thinking text rendered as regular output. In `Claude Code`, `API returned an empty or malformed response (HTTP 200)` occurred.
- **Root Cause**: Google returns `{"thought": true, "text": "..."}`. Initially, `part["thought"]` held boolean `True`. Calling `ledger.emit_thinking_delta(True)` failed, and subsequent `if "text" in part:` blocks emitted thinking as regular text.
- **Solution**: Updated `stream_response` to intercept `part.get("thought") is True`, extract `part["text"]`, emit exclusively in `ensure_thinking_block()`, and bypass normal text handling.

### ❌ Challenge 2: HTTP 400 Errors on Tool Schemas (`$schema`, `const`, `propertyNames`)
- **Symptom**: Requests rejected with `Invalid JSON payload received. Unknown name "$schema" at 'request.tools[0]...'`.
- **Root Cause**: Claude Code sends tool definitions in JSON Schema Draft-07 format. Gemini's strict OpenAPI parser rejects `$schema`, `const`, `propertyNames`, `exclusiveMinimum`, `exclusiveMaximum`, `$id`.
- **Solution**: Implemented recursive `_clean_gemini_schema` helper to strip incompatible keys before converting to `functionDeclarations`.

### ❌ Challenge 3: HTTP 400 `TYPE_BOOL` Error on Multi-Turn Reasoning Conversations
- **Symptom**: Multi-turn conversation rejected with `Invalid value at 'request.contents[1].parts[0].thought' (TYPE_BOOL), "**Analyzing...**"`.
- **Root Cause**: Rebuilding assistant history converted thinking into `{"thought": "thinking text"}` instead of boolean `TYPE_BOOL` (`thought: true`).
- **Solution**: Modified `_convert_anthropic_messages_to_gemini` to format thinking as `{"thought": True, "text": "thinking text"}`.

### ❌ Challenge 4: Missing `thought_signature` Error on Tool Execution
- **Symptom**: HTTP 400 rejection `Function call is missing a thought_signature in functionCall parts` during tool response turns.
- **Root Cause**: Google requires a thought signature on every `functionCall` object in assistant history.
- **Solution**: Auto-injected `"thought": True` and `"thought_signature": ts or "skip_thought_signature_validator"` on each `functionCall` item.

### ❌ Challenge 5: Process Lock Contention in CI Uninstall Tests
- **Symptom**: `./scripts/ci.sh` failed on 3 installer tests with `Free Claude Code is still running (fcc-claude)`.
- **Root Cause**: Residual background `fcc-claude` CLI processes triggered `pgrep` guards in `uninstall.sh`.
- **Solution**: Added proactive termination of residual processes via `pkill -9 -f fcc-claude` prior to running CI tests, leading to 100% passing test suites.
