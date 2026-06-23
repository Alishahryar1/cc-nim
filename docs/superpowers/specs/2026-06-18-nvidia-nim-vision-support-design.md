# NVIDIA NIM Vision Support — Design

- **Status**: Draft (brainstorming complete, post-refactor reconciliation applied 2026-06-23)
- **Date**: 2026-06-18 (last reconciled 2026-06-23, after PRs #845, #851 transport pipeline refactors)
- **Scope**: Add opt-in image (vision) handling for the NVIDIA NIM provider and the shared
  OpenAI-compatible chat transport underneath it. Other OpenAI-compatible providers receive the
  same hook but stay disabled by default.

## Context

This project is a Claude-to-custom-provider proxy. Claude Code and other Anthropic-format clients
send `messages` whose `content` can include `{"type":"image","source":{...}}` blocks.

The proxy rewrites Anthropic-format requests into native provider formats. For OpenAI-compatible
chat (`/v1/chat/completions`) providers, conversion lives in
`core/anthropic/conversion.py` and is shared by `OpenAIChatTransport` (NVIDIA NIM, plus any future
OpenAI-compatible subclass). Today, `AnthropicToOpenAIConverter._convert_user_message` (and
`._convert_user_message_with_injection`) **raise `OpenAIConversionError`** when they encounter any
image block, regardless of the destination provider.

> **Reconciliation note (2026-06-23):** PRs #845 (`Refactor provider transports into packages`)
> and #851 (`Refactor API request pipeline`) reshuffled the transport layer. The shared
> `OpenAIChatTransport` class moved from `providers/openai_compat.py` to
> `providers/transports/openai_chat/transport.py`, joined by sibling modules
> (`stream.py`, `recovery.py`, `tool_calls.py`) re-exported via
> `providers.transports.openai_chat`. The conversion seam (`build_base_request_body` /
> `convert_messages`) and the per-provider `_build_request_body` override seam are unchanged. The
> rest of this spec is therefore still architecturally accurate; **only file paths and a few
> identifiers need to be remapped** where this spec references the old layout.

NVIDIA NIM's OpenAI-compatible endpoint accepts image inputs via the standard
`{"type":"image_url","image_url":{"url":"..."}}` content part. The user has independently confirmed
that pointing GitHub Copilot's custom endpoint at `https://integrate.api.nvidia.com/v1` lets
vision-capable NIM models analyze uploaded images correctly.

The blocker is therefore in the proxy itself: vision-capable NIM models are reachable, but the
shared converter blocks image blocks unconditionally. This spec introduces an opt-in hook so NIM
(when configured) can pass image blocks through to the upstream API; other OpenAI-compatible
providers keep today's strict behaviour unless their maintainers add the same hook.

## Goals

1. Allow NVIDIA NIM users to send image-bearing requests to vision-capable NIM models through the
   proxy without hitting `OpenAIConversionError`.
2. Reuse Anthropic `image` source semantics (`source.type = "base64" | "url"`) and translate them
   to OpenAI `image_url` content parts.
3. Preserve today's strict behaviour for every other OpenAI-compatible provider and for NIM when
   the operator has not opted in. No silent regression.
4. Keep provider-specific configuration in provider-specific settings (per `CLAUDE.md`
   "provider-specific config" rule). The vision flag lives in `NimSettings.vision_enabled`, not
   in the base `ProviderConfig`.
5. Cover the change with unit tests at the converter layer plus a live smoke test against a
   real NIM vision-capable model behind an env-gated skip (per `CLAUDE.md` "maximum test
   coverage, preferably live smoke test coverage").
6. Bump the Semver `MINOR` version because this is a backward-compatible capability addition
   (per `CLAUDE.md` versioning rules).

## Non-goals

1. Changing the behaviour of any provider without an explicit capability declaration: any
   OpenAI-compatible provider that has not been wired up for vision keeps raising
   `OpenAIConversionError` when given an image block.
2. Rerouting image-in-`tool_result` content into a separate user message on OpenAI's wire.
   OpenAI's `role: tool` messages accept only string `content`. We replace in-tool-result
   images with a placeholder text block plus a warning log, matching the existing DeepSeek
   `_strip_unsupported_attachment_blocks` pattern in `providers/deepseek/request.py`.
3. Implementing automatic vision-capability detection from model name. Operators must opt in
   explicitly; the upstream model decides whether it can actually decode the image.
4. Modifying the Anthropic-native providers (e.g. OpenRouter via the Anthropic transport, or
   DeepSeek's native Anthropic-compatible endpoint). That work, if needed, belongs in a
   separate spec.
5. Streaming-side changes. The vision conversion is purely a request-side transformation; SSE
   emission is unaffected.

## Scope

In scope:

- `core/anthropic/conversion.py`: new `VisionCapabilityProtocol` and a default `NO_VISION`
  sentinel; new `convert_anthropic_image_to_openai_image_url(block)` helper; image branch in
  `_convert_user_message` and `_convert_user_message_with_injection`; `vision=` keyword arg on
  `build_base_request_body`.
- `providers/transports/openai_chat/transport.py`: `OpenAIChatTransport._vision_capability()`
  returning `NO_VISION` by default; transport-level plumbing so subclasses can override and have
  the resolved capability flow into `build_base_request_body`.
- `providers/nvidia_nim/`: subclass override of `_vision_capability()` reading
  `NimSettings.vision_enabled`.
- `config/nim.py`: `vision_enabled: bool = False` field.
- `.env.example`: documented env alias for the new flag (operator discoverability).
- Tests:
  - `tests/core/anthropic/` (or `tests/providers/test_converter.py`): unit tests for the new
    helper, the converter branches, and capability plumbing.
  - `smoke/test_nim_vision.py`: env-gated live NIM vision smoke test.

Out of scope:

- Anthropic-native providers (OpenRouter, DeepSeek native, LM Studio native, …).
- Other OpenAI-compatible subclasses (in the few that exist today). They receive the hook but
  remain disabled. Maintenance work for any of them belongs to a future spec.
- Vision capability metadata, registry, or autodetection.
- Tool-use loop changes.

## Constraints and invariants

- `CLAUDE.md` is authoritative:
  - Shared Anthropic protocol logic must live in neutral `core/anthropic/` modules. The
    converter-side change lives there; NIM-specific bits live next to NIM.
  - No `# type: ignore` / `# ty: ignore`. The new helper must type-check cleanly under
    `uv run ty check`.
  - All five CI checks (suppression grep, ruff format, ruff check, ty, pytest) must remain
    green before merge.
  - Semver MINOR bump on this commit. Bump in `pyproject.toml` and run `uv lock`.
- Image conversion must be deterministic and side-effect-free for non-image inputs: existing
  tests for text-only user messages and tool_result content must keep passing unchanged.
- `vision=` is a keyword-only argument on `build_base_request_body` and
  `AnthropicToOpenAIConverter._convert_user_message(..., vision=...)`. Callers that omit it
  retain today's behaviour (raise on image).

## Architecture

```
Claude Code / Anthropic-format client
  │  messages: [{type: "image", source: {...}}, {type: "text", text: "..."}, ...]
  ▼
api/routes.py (Anthropic Messages endpoint)
  │  NativeMessagesRequest (pydantic)
  ▼
provider.stream_response(request)
  │                                              ┌──────────────────────────────────────┐
  │                                              │ providers/transports/openai_chat/    │
  │                                              │   transport.py                       │
  │                                              │     _vision_capability()            ─┼─► NO_VISION (default)
  │                                              │     _build_request_body(request)     │       │
  │                                              └──────────────────┬───────────────────┘       │ subclass override
  ▼                                                                 │                           │
providers/transports/openai_chat/transport.py::OpenAIChatTransport (driven via OpenAIChatStreamRunner)                             │
  │  body = self._build_request_body(request, thinking_enabled=...)                              │
  │  build_base_request_body(request_data, vision=self._vision_capability(), ...)                │
  ▼                                                                                             │
providers/nvidia_nim/request.py::build_request_body                                              │
  │  uses vision capability from transport (injected by base)                                    │
  ▼                                                                                             │
core/anthropic/conversion.py                                                                       │
  │  build_base_request_body → AnthropicToOpenAIConverter.convert_messages(vision=...)            │
  │                                  ↓                                                            │
  │                          _convert_user_message(vision) → image block branch                   │
  │                              if vision.enabled: convert_anthropic_image_to_openai_image_url  ◀┘
  │                              else: raise OpenAIConversionError (today's message preserved)
  ▼
Outgoing OpenAI Chat Completions body → NVIDIA NIM → SSE → Anthropic-format SSE
```

The capability flows downward: subclass override → transport method → converter kwargs.
The image data never crosses a boundary that does not already see it (no detection layer
elsewhere — the converter is the single place that walks Anthropic content blocks).

### Capability injection vs. copy

The capability is a small read-only object (`VisionCapabilityProtocol`) with an `enabled`
property. It is **resolved once per request inside `_stream_response_impl`**, then passed
through `build_base_request_body` into `convert_messages`. There is no module-level mutable
state. Resolution looks like:

```python
# providers/transports/openai_chat/transport.py (sketch — exact shape in §Components)
vision = self._vision_capability()  # may be NO_VISION or a NIM-specific capability
body = self._build_request_body(request, thinking_enabled=...)
# inside _build_request_body:
build_base_request_body(request, vision=vision, ...)
```

This keeps every provider's `vision` decision local to its own `_vision_capability()` override.
No provider imports another provider's settings.

## Components

### C1. `core/anthropic/conversion.py` — neutral helper layer

Add:

- `class VisionCapabilityProtocol(Protocol)` with `enabled: bool` property.
- Sentinel `NO_VISION` implementing the protocol with `enabled = False`.
- Function `convert_anthropic_image_to_openai_image_url(block: Any) -> dict[str, Any]` that
  returns the OpenAI `image_url` content-part dict for an Anthropic image block, or raises
  `OpenAIConversionError` for malformed inputs.
- Update `AnthropicToOpenAIConverter._convert_user_message` and
  `._convert_user_message_with_injection` signatures: add keyword-only `vision` parameter
  defaulting to `NO_VISION`. When an image block is hit:
  - `vision.enabled is True` → call `convert_anthropic_image_to_openai_image_url` and emit a
    `{"role": "user", "content": [{"type": "image_url", ...}]}` message (or merge into a
    multi-part content list when text is buffered).
  - else → raise `OpenAIConversionError` with today's message.
- Add helper `_strip_inner_images_from_tool_result(content) -> tuple[Any, int]` that walks a
  tool_result content list and replaces `{"type":"image",...}` blocks with a placeholder text
  block; returns the (rewritten list, count_of_replacements). Used by the new tool_result
  branch only.
- Update `build_base_request_body` signature: add keyword-only `vision` defaulting to
  `NO_VISION`, pass to `AnthropicToOpenAIConverter.convert_messages`.
- Update the existing `convert_messages` signature accordingly.

Single-file change. Imports already available (`get_block_attr`, `get_block_type`, `OpenAIConversionError`).

### C2. `providers/transports/openai_chat/transport.py` — transport-level capability surface

Add to `OpenAIChatTransport` (in `providers/transports/openai_chat/transport.py`, imported as
`from providers.transports.openai_chat import OpenAIChatTransport`):

- Method
  ```python
  def _vision_capability(self) -> VisionCapabilityProtocol:
      """Subclasses override to advertise vision support. Default = NO_VISION."""
      return NO_VISION
  ```
- The transport-level `_build_request_body(request, thinking_enabled)` is the existing
  abstract seam. Subclasses (e.g. `NvidiaNimProvider`) override it and forward into their
  own per-provider body builder, which in turn calls
  `build_base_request_body(..., vision=self._vision_capability(), ...)`. The transport
  itself does not need to touch `build_base_request_body`; the override chain makes vision
  reach the converter naturally.
- Concretely (in `providers/nvidia_nim/client.py`):
  ```python
  class NvidiaNimProvider(OpenAIChatTransport):
      def _vision_capability(self) -> VisionCapabilityProtocol:
          return _NimVisionCapability(enabled=self._nim_settings.vision_enabled)

      def _build_request_body(self, request, thinking_enabled=None):
          return build_request_body(
              request,
              self._nim_settings,
              thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
              vision=self._vision_capability(),
          )
  ```
  And in `providers/nvidia_nim/request.py`:
  ```python
  def build_request_body(
      request_data: Any,
      nim: NimSettings,
      *,
      thinking_enabled: bool,
      vision: VisionCapabilityProtocol = NO_VISION,
  ) -> dict:
      ...
      body = build_base_request_body(
          request_data,
          reasoning_replay=...,
          vision=vision,
      )
  ```
- No new public method on `BaseProvider`. The capability is plumbed entirely through the
  existing per-provider `_build_request_body` seam.

This is the only hook a subclass needs to override to enable vision.

### C3. `providers/nvidia_nim/` — NIM capability wiring

Two small changes:

- `providers/nvidia_nim/request.py`: define a small
  `class _NimVisionCapability: ...` implementing `VisionCapabilityProtocol`, with an
  `enabled` flag set from `nim.vision_enabled`. Alternative: store the capability inside
  `NimSettings` directly. Decision: keep the capability local to this file (the simplest
  pattern) and source the boolean from `nim.vision_enabled`.
- `providers/nvidia_nim/client.py`: override
  `NvidiaNimProvider._vision_capability() -> VisionCapabilityProtocol` to return the
  capability when `nim_settings.vision_enabled is True`, else `NO_VISION`.

The existing `NimSettings` import (`config.nim`) is unchanged aside from the new field.

### C4. `config/nim.py` — settings

Add:

```python
vision_enabled: bool = False  # Field(default=False, alias="NVIDIA_NIM_VISION_ENABLED")
```

env alias follows the convention used by other NimSettings fields (see existing
`NVIDIA_NIM_*` aliases).

### C5. `.env.example`

Append a doc entry near the other `NVIDIA_NIM_*` settings:

```
# Enable vision (image input) handling for vision-capable NIM models.
# When false, any Anthropic image block in client messages is rejected with HTTP 400.
NVIDIA_NIM_VISION_ENABLED=false
```

### C6. Tests

Three layers — see §Testing Strategy for the full list.

### C7. Tool-result image placeholder constant

`core/anthropic/conversion.py` introduces a module-level constant:

```python
_TOOL_RESULT_IMAGE_PLACEHOLDER_TEXT = (
    "[attachment omitted: image stripped from tool_result for OpenAI compat]"
)

_TOOL_RESULT_IMAGE_PLACEHOLDER_BLOCK = {
    "type": "text",
    "text": _TOOL_RESULT_IMAGE_PLACEHOLDER_TEXT,
}
```

Placement decision: same module as the converter, since the converter is the only consumer.
Mirrors the structure of `_OMITTED_ATTACHMENT_TEXT` and `_OMITTED_ATTACHMENT_BLOCK` in
`providers/deepseek/request.py` but lives in neutral territory because the placeholder text
is provider-agnostic (OpenAI's `role: tool` content is always string-only, regardless of
subclass).

## Data flow

### D1. Conversion table (Anthropic image → OpenAI image_url)

| Anthropic `image.source.type` | media_type | source.data / source.url | OpenAI `image_url.url` | Notes |
|---|---|---|---|---|
| `"base64"` | `"image/png"` | `data = "iVBOR..."` | `data:image/png;base64,iVBOR...` | Standard data URI |
| `"base64"` | `"image/jpeg"` etc. | non-empty `data` | `data:image/jpeg;base64,...` | Any `image/*` accepted |
| `"url"` | n/a | `url = "https://..."` | `url` (verbatim) | Forwarded unchanged |
| `"base64"` | missing or not `image/*` | any | (raised) `OpenAIConversionError("image base64 source requires image/* media_type, got ...")` | Strict |
| `"base64"` | ok | empty `data` | (raised) `OpenAIConversionError("image base64 source requires non-empty data")` | Strict |
| `"url"` | n/a | missing/empty `url` | (raised) `OpenAIConversionError("image url source requires non-empty url")` | Strict |
| any other `source.type` | n/a | n/a | (raised) `OpenAIConversionError("unsupported image source.type=...")` | Mirrors existing strictness |
| missing `source` | n/a | n/a | (raised) `OpenAIConversionError("image block missing source")` | Mirrors existing strictness |

All raised errors are `OpenAIConversionError`; the NIM request builder wraps that in
`InvalidRequestError` (existing pattern in `providers/nvidia_nim/request.py`) which surfaces as
a 400 to the client.

### D2. User message emission

`_convert_user_message` builds a list of OpenAI messages. Today:

```
text_parts: list[str]
result: list[dict]            # each: {role, content} where role∈{"user","tool"} and content is str
```

Each text block is buffered; on a tool_result or end-of-content the buffer flushes as one
`{"role":"user","content":"..."}` message. With image support the same loop appends
single-part image messages:

```python
elif block_type == "image":
    if vision.enabled:
        if text_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})
            text_parts = []
        result.append({
            "role": "user",
            "content": [convert_anthropic_image_to_openai_image_url(block)],
        })
    else:
        raise OpenAIConversionError(
            "User message image blocks are not supported for OpenAI chat "
            "conversion; use a vision-capable native Anthropic provider or "
            "extend the converter."
        )
```

Behavior matrix:

| user content order | vision=OFF | vision=ON |
|---|---|---|
| `[image]` | raise | `[user: [image_url]]` |
| `[text="hi"]` | `[user: "hi"]` | `[user: "hi"]` |
| `[text="a", image]` | raise | `[user: "a"], [user: [image_url]]` |
| `[image, text="b"]` | raise | `[user: [image_url]], [user: "b"]` |
| `[text="a", image, text="b"]` | raise | `[user: "a"], [user: [image_url]], [user: "b"]` |
| `[image, image]` | raise | `[user: [image_url]], [user: [image_url]]` |

Each user message gets the simplest valid OpenAI shape — text-only when text-only,
content-as-list-of-parts when carrying (or only) an image. No artificial concatenation.

`_convert_user_message_with_injection` (the variant that handles post-tool deferred assistant
text) follows the same pattern. The image branch sits inside the same `for block in content`
loop, **after** the text flush and **before** the tool_result branch. Image blocks never
appear in tool_result processing, because tool results in that variant carry their inner
content arrays separately: see D3.

### D3. tool_result image handling

OpenAI's `role: tool` message accepts `content: string` only. An image inside tool_result is
therefore rewritten to a placeholder text block, regardless of `vision.enabled`:

```python
elif block_type == "tool_result":
    flush_text()
    raw_content = get_block_attr(block, "content", "")
    new_content, dropped = _strip_inner_images_from_tool_result(raw_content)
    if dropped > 0:
        logger.warning(
            "OPENAI_CONVERSION: dropped {} image(s) from tool_result content "
            "(vision={}) — OpenAI tool messages only carry string content; "
            "use a native Anthropic provider if image-in-tool-result is required.",
            dropped,
            vision.enabled,
        )
    serialized = _serialize_tool_result_content(new_content)
    ...
```

`_strip_inner_images_from_tool_result` walks the inner list and replaces each
`{type:"image"}` block with `_TOOL_RESULT_IMAGE_PLACEHOLDER_BLOCK`. If the inner content was a
plain string (the common case), the function is a no-op and returns `(content, 0)`.

**Why placeholder + warning (chosen from brainstorming)**:

- (a) placeholder + warning (chosen): stable, never breaks the tool loop, mirrors
  `_OMITTED_ATTACHMENT_BLOCK` in DeepSeek; tradeoff is loss of vision for in-tool images —
  acceptable because Claude Code's `Read` tool typically already extracts text and embeds
  it, so image-in-tool-result is rare.
- (b) inject as a separate user message after the tool message: technically reachable on the
  OpenAI wire, but semantically wrong (the tool returned an image; we suddenly call it a
  user prompt). Models see a fabricated user turn.
- (c) only handle user-level images, drop tool-result images silently: least surprising,
  but silent drops hide intent. A warning is cheap.

## Behaviour summary

| layer | vision=OFF (default everywhere) | vision=ON (NIM with `NVIDIA_NIM_VISION_ENABLED=true`) |
|---|---|---|
| `_convert_user_message` on image | `OpenAIConversionError` | emit `image_url` content part |
| `_convert_user_message_with_injection` on image | `OpenAIConversionError` | emit `image_url` content part |
| `tool_result` inner image | placeholder + warning | placeholder + warning (always) |
| NIM upstream rejects image | propagates upstream 400 through existing error mapping | same |
| non-NIM provider (still OFF) on image | `OpenAIConversionError` | unchanged (no provider has it ON today) |

Only NVIDIA NIM flips behavior in this PR, and only when its setting is on.

## Error handling

### E1. Error catalogue

| # | Trigger | Class | Message | HTTP surface |
|---|---|---|---|---|
| 1 | image block, vision=OFF | `OpenAIConversionError` | "User message image blocks are not supported for OpenAI chat conversion; use a vision-capable native Anthropic provider or extend the converter." | 400 via `InvalidRequestError` wrap in NIM request builder (existing path) |
| 2 | image block, vision=ON, `source` missing | `OpenAIConversionError` | "image block missing source" | 400 (same wrap) |
| 3 | image block, vision=ON, `source.type="base64"`, media_type missing or not `image/*` | `OpenAIConversionError` | "image base64 source requires image/* media_type, got {repr}" | 400 |
| 4 | image block, vision=ON, `source.type="base64"`, `data` empty | `OpenAIConversionError` | "image base64 source requires non-empty data" | 400 |
| 5 | image block, vision=ON, `source.type="url"`, `url` missing or empty | `OpenAIConversionError` | "image url source requires non-empty url" | 400 |
| 6 | image block, vision=ON, `source.type` not in `{base64, url}` | `OpenAIConversionError` | "unsupported image source.type={repr}" | 400 |
| 7 | NIM upstream rejects image (e.g. wrong model, too large) | upstream `openai.BadRequestError` | upstream message | 400 via existing `_openai_error_message` mapping |
| 8 | tool_result inner image (any vision mode) | log warning | "OPENAI_CONVERSION: dropped N image(s) from tool_result content (vision=...) — OpenAI tool messages only carry string content; use a native Anthropic provider if image-in-tool-result is required." | n/a (placeholder text emitted into tool message) |
| 9 | NIM configured with vision=ON but model is non-vision | upstream error or empty content | upstream message | propagates as upstream `BadRequestError` or empty SSE response; no special handling |

### E2. Error-message policy

- All converter-raised messages are English, capitalised first letter, single trailing period,
  with the offending field represented via `repr()` to preserve type information in logs.
- The smoke test asserts on stable substrings only ("image base64 source requires image/*",
  "requires non-empty data", "unsupported image source.type") so messages can tighten without
  breaking tests, as long as the substring remains.
- No new error class is introduced. `OpenAIConversionError` is sufficient and already plumbed
  into the existing 400 path.

### E3. Logging

- `logger.debug("OPENAI_CONVERSION: image blocks converted count={}", n)` inside the user
  message branch on success (best-effort count).
- `logger.warning("OPENAI_CONVERSION: dropped {} image(s) from tool_result content (vision={})", n, vision_enabled)` on tool_result image drop.
- `logger.warning("NIM_STREAM: vision-enabled request rejected by upstream status=...", ...)` on upstream bad-request containing "image" in the error text (optional, decided in implementation).

No `logger.error` from the converter; conversion errors raise, callers decide.

## Configuration and defaults

### F1. New env / settings field

| Path | Type | Default | Notes |
|---|---|---|---|
| `NimSettings.vision_enabled` | `bool` | `False` | new field, follows existing env-alias pattern |
| env alias | `NVIDIA_NIM_VISION_ENABLED` | `"false"` (lowercase) | matches `NVIDIA_NIM_*` family in `.env.example` |
| `.env.example` | documentation only | "false" | appended next to existing NIM settings |

### F2. Behaviour of defaults

- The default everywhere is **vision=OFF** (today's behaviour). No existing user sees a change.
- Operators who want vision **opt in** by setting `NVIDIA_NIM_VISION_ENABLED=true`. The flag
  is read once at provider construction; runtime changes require `settings` reload (existing
  project behaviour for any setting).

### F3. Cross-provider

- No other `OpenAIChatTransport` subclass overrides `_vision_capability()` in this PR. The
  hook is in place and discoverable for future specs to flip the switch for groq / cerebras /
  mistral / lm-studio / openai-direct, etc. This spec scopes itself to NVIDIA NIM only.

## Migration and compatibility

### G1. Wire-compatibility

| Concern | Result |
|---|---|
| Public Anthropic-format HTTP API | unchanged — only new requests will succeed where today's proxy 400s |
| Internal call signature change to `build_base_request_body` | new keyword-only argument with default `NO_VISION`; existing call sites compile unchanged |
| Internal call signature change to `convert_messages` / `_convert_user_message` | new keyword-only `vision=`; default = NO_VISION preserves today's raise-on-image behaviour |
| `NimSettings` schema | new optional field; existing settings files continue to parse (default kicks in) |
| `.env` files | unaffected (env alias defaults to false) |
| Provider registry / model listing | unaffected |

### G2. Semver

Per `CLAUDE.md` versioning rules, this is a **backward-compatible capability addition** → MINOR
bump. **Reconciled 2026-06-23:** current `pyproject.toml` `version` is `2.3.14`; bump
`2.3.14` → `2.4.0` in the same commit:

- Bump `[project].version` in `pyproject.toml` from `2.3.14` → `2.4.0`.
- Run `uv lock` to refresh `uv.lock`.
- Bump and lockfile change ship in the same commit as the feature work.

### G3. Backout

Reverting the bump is equivalent to setting `NVIDIA_NIM_VISION_ENABLED=false` on every NIM
deployment; no operator data is lost and no model capability is unregistered upstream. The
flag is purely a request-side transformer.

## Testing strategy

Three layers — unit tests for the converter, capability plumbing tests for the transport,
and an env-gated live smoke test against actual NVIDIA NIM.

### T1. Unit tests — `convert_anthropic_image_to_openai_image_url`

A focused test module adds the boundary-value matrix:

| case | input | expected |
|---|---|---|
| base64 PNG | `{type:image, source:{type:base64, media_type:image/png, data:abc}}` (dict form) | `{type:image_url, image_url:{url:data:image/png;base64,abc}}` |
| base64 JPEG | media_type `image/jpeg`, data `x` | `data:image/jpeg;base64,x` |
| URL | source.url `https://...` | `image_url.url = https://...` verbatim |
| Pydantic model | task-like object with attributes | same dict output as dict input |
| missing source | `{type:image}` | raises `OpenAIConversionError("image block missing source")` |
| base64 missing media_type | `source: {type:base64, data:x}` | raises with media_type substring |
| base64 non-image media_type | media_type `text/plain` | raises with media_type substring |
| base64 empty data | data `""` | raises "non-empty data" |
| URL empty url | url `""` | raises "non-empty url" |
| URL missing url key | only `type` field | raises "non-empty url" |
| unsupported source.type | source.type `"file"` | raises "unsupported image source.type" |

### T2. Unit tests — `_convert_user_message` and `_convert_user_message_with_injection`

`tests/providers/test_converter.py` already houses these tests. We:

- **Keep** `test_convert_user_message_image_raises` — it now asserts the default (`vision=NO_VISION`)
  path raises today's error verbatim.
- **Add** `test_convert_user_message_image_vision_off_explicit` — pass `vision=NO_VISION`
  explicitly and assert the same raise (defense against accidental capability default drift).
- **Add** `test_convert_user_message_image_vision_on_base64` — input image with base64 PNG,
  vision capability stub returning `enabled=True`, expect a single-element
  `{"role":"user","content":[{"type":"image_url","image_url":{...}}]}` message.
- **Add** `test_convert_user_message_image_vision_on_url` — same shape, URL source.
- **Add** `test_convert_user_message_text_then_image` (vision=ON) — text flushed as one
  message, image emitted as second message (per the matrix above).
- **Add** `test_convert_user_message_text_image_text` (vision=ON) — three messages as in the
  matrix above.
- **Add** `test_convert_user_message_two_images` (vision=ON) — two adjacent image messages.
- **Add** `test_convert_user_message_with_injection_image_vision_on` — image inside the
  variant's content list passes the new branch alongside the deferred-assistant-text logic.

Each test uses a small `class _EnabledVision: enabled=True` and `_DisabledVision: enabled=False`
to avoid referencing the production protocol types directly. That keeps the tests focused on
the contract: any object with `enabled: bool` works.

### T3. Unit tests — `build_base_request_body(vision=...)`

`tests/providers/test_converter.py` already covers body construction:

- **Add** `test_build_base_request_body_default_vision_is_off` — calling
  `build_base_request_body(request)` (no vision kwarg) and giving it an image block raises.
- **Add** `test_build_base_request_body_vision_on_passes_through` — image block with
  vision=ON_capability ends up as an `image_url` part in the body.

### T4. Unit tests — `OpenAIChatTransport._vision_capability()`

- **Add** `test_default_vision_capability_is_no_vision` on the transport base (in
  `tests/providers/test_openai_compat_5xx_retry.py` extending capability coverage, or a new
  sibling `tests/providers/test_openai_chat_transport_capability.py`).
- **Add** `test_nvidia_nim_vision_capability_off_when_setting_false` —
  `NvidiaNimProvider(nim_settings=NimSettings(vision_enabled=False))._vision_capability().enabled`
  is `False`.
- **Add** `test_nvidia_nim_vision_capability_on_when_setting_true` — same, `True`.
- **Add** `test_nvidia_nim_build_request_body_image_passes_through_when_vision_on` — using a
  mocked OpenAI client, drive the request through the full body-builder and confirm the body
  contains the `image_url` content part.
- **Add** `test_nvidia_nim_build_request_body_image_raises_when_vision_off` — same but with
  vision OFF, expect `InvalidRequestError` carrying the original conversion error.

### T5. Unit tests — `_strip_inner_images_from_tool_result`

- `test_strip_drops_images_inside_tool_result_and_warns` — list with image inner blocks →
  replaced with placeholder text blocks; warning emitted.
- `test_strip_no_op_on_string_content` — tool_result content as string; returns `(content, 0)`
  unchanged.
- `test_strip_no_op_when_no_images` — list of only text blocks; returns unchanged, `dropped=0`.
- `test_strip_preserves_other_blocks` — image + text + image → `[text, placeholder, text,
  placeholder]`, dropped=2.

### T6. Live smoke test — `smoke/test_nim_vision.py`

A new env-gated live test, mirroring the existing
`smoke/test_nvidia_nim_streaming.py` style:

- Required env: `NVIDIA_NIM_BASE_URL`, `NVIDIA_NIM_API_KEY`, `NVIDIA_NIM_VISION_MODEL`,
  and a smoke-run flag (existing convention: the suite is collected but skipped unless
  `RUN_SMOKE=1`).
- A 1x1 black PNG encoded as base64 is sent in a user message with a short Korean text like
  `"이 이미지는 검은색이야?"`.
- `NVIDIA_NIM_VISION_ENABLED=true` set in the test environment.
- Assertions:
  - HTTP 200 with non-empty SSE stream.
  - `output_tokens > 0` (no early truncation).
  - No `400` in the SSE error channels.
  - At least one `content_block_delta` arriving within a short timeout (existing smoke
    timeout).
- A second sub-test: with `NVIDIA_NIM_VISION_ENABLED=false` (or unset), the same request is
  expected to surface the existing conversion error or a 400 (whichever path the wiring
  chooses). This guards against accidental flip of the default.

The suite is added to `smoke/conftest.py` if a conftest exists; otherwise the file's own
`pytest.importorskip` and explicit env check handles skipping.

### T7. Coverage map

| module:line exercised | test |
|---|---|
| `convert_anthropic_image_to_openai_image_url` | T1 |
| `AnthropicToOpenAIConverter._convert_user_message` image branch | T2 |
| `AnthropicToOpenAIConverter._convert_user_message_with_injection` image branch | T2 |
| `AnthropicToOpenAIConverter._convert_user_message` tool_result image scrub | T5 |
| `build_base_request_body(vision=...)` threading | T3 |
| `OpenAIChatTransport._vision_capability` default | T4 |
| `NvidiaNimProvider._vision_capability` override | T4 |
| `NvidiaNimProvider` end-to-end body with vision ON | T4 |
| live NIM API request | T6 |

### T8. CI

- All five CI checks must continue to pass:
  - `Ban type ignore suppressions`
  - `ruff-format`
  - `ruff-check`
  - `ty`
  - `pytest`
- Smoke tests run only when `RUN_SMOKE=1` is set on the workflow, which matches the existing
  smoke convention. No new CI step required.

## Verification

`./scripts/ci.sh` locally must finish green. The implementation phase is the place to run
that and adjust for any lint drift.

Manual sanity:

1. `uv run pytest` — full test suite green.
2. `uv run python -m smoke.test_nim_vision` with the env vars above — observe vision-capable
   NIM model returning non-empty text on a simple image prompt.
3. Toggle off / on and confirm HTTP 400 path activates / deactivates accordingly.

## Files to change

| File | Change |
|---|---|
| `pyproject.toml` | MINOR version bump `2.3.14` → `2.4.0` (single line) |
| `uv.lock` | refreshed (`uv lock`) |
| `.env.example` | one doc line + setting key comment for `NVIDIA_NIM_VISION_ENABLED` |
| `core/anthropic/conversion.py` | protocol, helper, kwarg plumb, tool_result helper, constant |
| `providers/transports/openai_chat/transport.py` | transport-level default capability + inject into `build_base_request_body` |
| `providers/nvidia_nim/request.py` | small capability type + ability to read `nim.vision_enabled` |
| `providers/nvidia_nim/client.py` | override `_vision_capability()` |
| `config/nim.py` | `vision_enabled: bool = False` |
| `config/settings.py` | `model_validator` hook to copy `NVIDIA_NIM_VISION_ENABLED` to `self.nim.vision_enabled` (env binding plumbing) |
| `tests/providers/test_converter.py` | new image conversion test cases |
| `tests/providers/test_nvidia_nim.py` (or wherever NIM provider tests live) | vision capability + provider build tests |
| `tests/providers/test_openai_compat_5xx_retry.py` (or new `test_openai_chat_transport_capability.py`) | `_vision_capability` default test on `OpenAIChatTransport` base |
| `smoke/product/test_nvidia_nim_vision_product_live.py` | new env-gated live smoke test (`FCC_LIVE_SMOKE=1`) |

No other files expected to change.

## Open questions

None at design time. Implementation may surface questions for `vision=ON` content
construction when text and image coexist inside an Anthropic message that has no
surrounding whitespace — specifically, whether (`text="a"`, `image`) should produce one
combined user message with `[{"type":"text","text":"a"}, {"type":"image_url",...}]` parts,
or stay as two separate user messages (chosen: two separate messages to avoid changing any
existing OpenAI consumers that expect one text per user message). Implementation should
follow the matrix in D2 strictly.
