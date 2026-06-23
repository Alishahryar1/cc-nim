# NVIDIA NIM Vision Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in image (vision) handling for the NVIDIA NIM provider by threading a `VisionCapabilityProtocol` through the shared OpenAI-compatible chat transport, so vision-capable NIM models can receive Anthropic `image` blocks (translated to OpenAI `image_url` content parts) while every other OpenAI-compatible provider keeps today's strict `OpenAIConversionError` behavior.

**Architecture:** Single-pass capability injection. `OpenAIChatTransport._vision_capability()` returns `NO_VISION` by default; `NvidiaNimProvider` overrides it with a small `_NimVisionCapability` sourced from `NimSettings.vision_enabled`. Subclass `_build_request_body` threads the resolved capability into `build_base_request_body(vision=...)`, which threads it into `convert_messages(vision=...)`, which threads it into `_convert_user_message(vision=...)` and `_convert_user_message_with_injection(vision=...)`. The single image branch in each converter emits `{"type":"image_url","image_url":{"url":"data:..." or "https://..."}}` when capability is enabled, else raises today's error. tool_result inner images are always replaced with a placeholder text block because OpenAI `role: tool` content is string-only.

**Tech Stack:** Python 3.14, pydantic v2, pytest, Ruff, ty, loguru (existing). OpenAI Python SDK (`openai.AsyncOpenAI`) for upstream calls (existing).

## Global Constraints

> Every task inherits these; do not weaken them per-task.

- **`CLAUDE.md` is authoritative.** Shared Anthropic protocol logic lives in `core/anthropic/`. No cross-provider utils imports. No `# type: ignore` / `# ty: ignore`. All 5 CI checks must pass: `Ban type ignore suppressions`, `ruff-format`, `ruff-check`, `ty`, `pytest`.
- **Backward-compatible MINOR Semver bump** in this PR. Set `[project].version` in `pyproject.toml` from `2.3.14` → `2.4.0` (reconciled 2026-06-23 — current `pyproject.toml` reports `2.3.14`). Run `uv lock` so `uv.lock` reflects the bump. Both ship in the same commit as the feature work.
- **Default = vision OFF everywhere.** Every existing user's behavior is preserved unless they explicitly set `NVIDIA_NIM_VISION_ENABLED=true`.
- **`vision=` is a keyword-only argument** on `build_base_request_body`, `AnthropicToOpenAIConverter.convert_messages`, `_convert_user_message`, and `_convert_user_message_with_injection`. Default = `NO_VISION`. Callers omitting it retain today's raise-on-image behavior.
- **Single testability point** for the capability: `VisionCapabilityProtocol` is a `Protocol` with `enabled: bool`; `NO_VISION` is a sentinel module-level instance. Tests use small stub classes `_EnabledVision`/`_DisabledVision` to assert on the contract, not the production types.
- **Anthropic `source.type` ∈ `{"base64","url"}`** is the only supported surface. Anything else → `OpenAIConversionError("unsupported image source.type=...")`.
- **tool_result inner images are always stripped** with placeholder text + warning log, regardless of `vision.enabled`. OpenAI `role: tool` accepts only string content.
- **Live smoke is env-gated** behind `FCC_LIVE_SMOKE=1`; the new smoke test lives in `smoke/product/` and uses the existing `SmokeServerDriver`/model-from-config plumbing.
- **No new public error class.** `OpenAIConversionError` is sufficient; existing `InvalidRequestError` wrap at `providers/nvidia_nim/request.py:357-365` keeps the 400 path.

---

## File Structure (locked-in decomposition)

| File | Role | Touched |
|---|---|---|
| `pyproject.toml` | version bump | yes (`2.3.14` → `2.4.0`) |
| `uv.lock` | lockfile refresh | yes (`uv lock`) |
| `.env.example` | discoverability for `NVIDIA_NIM_VISION_ENABLED` | yes (+4 lines) |
| `config/nim.py` | `vision_enabled: bool = False` field on `NimSettings` | yes (+1 field) |
| `config/settings.py` | env binding + `model_validator` to copy to `self.nim.vision_enabled` | yes (+1 field, +1 validator) |
| `core/anthropic/conversion.py` | protocol, sentinel, image → `image_url` helper, `_strip_inner_images_from_tool_result`, `vision=` kwarg on `_convert_user_message` / `_convert_user_message_with_injection` / `convert_messages` / `build_base_request_body`, image placeholder constants | yes |
| `core/anthropic/__init__.py` | re-export `VisionCapabilityProtocol`, `NO_VISION`, `convert_anthropic_image_to_openai_image_url` | yes (+3 lines) |
| `providers/transports/openai_chat/transport.py` | transport-level `_vision_capability()` returning `NO_VISION` | yes (+1 method) |
| `providers/nvidia_nim/request.py` | `_NimVisionCapability` class + `vision:` kwarg threaded through `build_request_body` → `build_base_request_body` | yes (+1 class, +1 kwarg, +1 arg) |
| `providers/nvidia_nim/client.py` | override `_vision_capability()`; pass `vision=self._vision_capability()` into `build_request_body` | yes (+1 method override, +1 kwarg) |
| `tests/providers/test_converter.py` | image-aware unit tests (T1, T2, T3, T5 from spec) | yes (+~10 tests, count it matters) |
| `tests/providers/test_nvidia_nim.py` | NIM vision capability + end-to-end body-build tests (T4) | yes (+~4 tests) |
| `smoke/product/test_nvidia_nim_vision_product_live.py` | env-gated live NIM vision smoke (T6) | yes (new file) |

No other files expected to change. Decomposition follows existing seams (`build_base_request_body` is the single converter-side chokepoint, and per-provider `_build_request_body` is the single provider-side chokepoint).

---

### Task 1: VisionCapabilityProtocol, NO_VISION sentinel, and convert_anthropic_image_to_openai_image_url

**Files:**
- Modify: `core/anthropic/conversion.py` (add protocol, sentinel, helper after `OpenAIConversionError` class)
- Test: `tests/providers/test_converter.py` (add T1 test cases at end of file)

**Interfaces:**
- Consumes: `get_block_attr`, `get_block_type` from `core/anthropic/content` (existing imports)
- Produces: `VisionCapabilityProtocol` (Protocol, `enabled: bool`), `NO_VISION` (sentinel instance), `convert_anthropic_image_to_openai_image_url(block) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/providers/test_converter.py`:

```python
# --- Vision Conversion Tests (T1: convert_anthropic_image_to_openai_image_url) ---

from core.anthropic.conversion import (
    NO_VISION,
    VisionCapabilityProtocol,
    convert_anthropic_image_to_openai_image_url,
)


class _EnabledVision:
    """Stub for vision=ON in converter tests."""

    enabled = True


class _DisabledVision:
    """Stub for vision=OFF in converter tests."""

    enabled = False


def test_vision_capability_protocol_no_vision_is_disabled():
    assert NO_VISION.enabled is False


def test_convert_image_base64_png():
    block = MockBlock(
        type="image",
        source={"type": "base64", "media_type": "image/png", "data": "abc"},
    )
    result = convert_anthropic_image_to_openai_image_url(block)
    assert result == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc"},
    }


def test_convert_image_base64_jpeg():
    block = MockBlock(
        type="image",
        source={"type": "base64", "media_type": "image/jpeg", "data": "x"},
    )
    result = convert_anthropic_image_to_openai_image_url(block)
    assert result == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,x"},
    }


def test_convert_image_url_source():
    block = MockBlock(
        type="image",
        source={"type": "url", "url": "https://example.com/img.png"},
    )
    result = convert_anthropic_image_to_openai_image_url(block)
    assert result == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/img.png"},
    }


def test_convert_image_pydantic_model_source():
    """Pydantic-style object with attributes (not dict) also works."""

    class _Source:
        type = "url"
        url = "https://example.com/a.png"

    block = MockBlock(type="image", source=_Source())
    result = convert_anthropic_image_to_openai_image_url(block)
    assert result == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png"},
    }


def test_convert_image_missing_source_raises():
    block = MockBlock(type="image")
    with pytest.raises(OpenAIConversionError, match="image block missing source"):
        convert_anthropic_image_to_openai_image_url(block)


def test_convert_image_base64_missing_media_type_raises():
    block = MockBlock(
        type="image",
        source={"type": "base64", "data": "abc"},
    )
    with pytest.raises(OpenAIConversionError, match="image/* media_type"):
        convert_anthropic_image_to_openai_image_url(block)


def test_convert_image_base64_non_image_media_type_raises():
    block = MockBlock(
        type="image",
        source={"type": "base64", "media_type": "text/plain", "data": "abc"},
    )
    with pytest.raises(OpenAIConversionError, match="image.* media_type"):
        convert_anthropic_image_to_openai_image_url(block)


def test_convert_image_base64_empty_data_raises():
    block = MockBlock(
        type="image",
        source={"type": "base64", "media_type": "image/png", "data": ""},
    )
    with pytest.raises(OpenAIConversionError, match="non-empty data"):
        convert_anthropic_image_to_openai_image_url(block)


def test_convert_image_url_empty_url_raises():
    block = MockBlock(
        type="image",
        source={"type": "url", "url": ""},
    )
    with pytest.raises(OpenAIConversionError, match="non-empty url"):
        convert_anthropic_image_to_openai_image_url(block)


def test_convert_image_url_missing_url_raises():
    block = MockBlock(type="image", source={"type": "url"})
    with pytest.raises(OpenAIConversionError, match="non-empty url"):
        convert_anthropic_image_to_openai_image_url(block)


def test_convert_image_unsupported_source_type_raises():
    block = MockBlock(type="image", source={"type": "file"})
    with pytest.raises(OpenAIConversionError, match="unsupported image source.type"):
        convert_anthropic_image_to_openai_image_url(block)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_converter.py -k "vision_or_convert_image" -v`
Expected: FAIL — `ImportError: cannot import name 'VisionCapabilityProtocol' / 'NO_VISION' / 'convert_anthropic_image_to_openai_image_url'`

- [ ] **Step 3: Write the implementation**

Add to `core/anthropic/conversion.py`, after the `OpenAIConversionError` class (around line 17):

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionCapabilityProtocol(Protocol):
    """Protocol for vision capability injection into the converter."""

    @property
    def enabled(self) -> bool: ...


class _NoVision:
    """Sentinel: vision is disabled (default for all OpenAI-compatible transports)."""

    @property
    def enabled(self) -> bool:
        return False


NO_VISION: VisionCapabilityProtocol = _NoVision()


def convert_anthropic_image_to_openai_image_url(block: Any) -> dict[str, Any]:
    """Convert an Anthropic image block to an OpenAI ``image_url`` content-part dict.

    Raises ``OpenAIConversionError`` for malformed or unsupported image blocks.
    """
    source = get_block_attr(block, "source", None)
    if source is None:
        raise OpenAIConversionError("image block missing source")

    source_type = get_block_attr(source, "type", None) or (
        source.get("type") if isinstance(source, dict) else None
    )
    if source_type == "base64":
        media_type = get_block_attr(source, "media_type", None) or (
            source.get("media_type") if isinstance(source, dict) else None
        )
        if not isinstance(media_type, str) or not media_type.startswith("image/"):
            raise OpenAIConversionError(
                f"image base64 source requires image/* media_type, got {media_type!r}"
            )
        data = get_block_attr(source, "data", None) or (
            source.get("data") if isinstance(source, dict) else None
        )
        if not data:
            raise OpenAIConversionError(
                "image base64 source requires non-empty data"
            )
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    if source_type == "url":
        url = get_block_attr(source, "url", None) or (
            source.get("url") if isinstance(source, dict) else None
        )
        if not url:
            raise OpenAIConversionError(
                "image url source requires non-empty url"
            )
        return {
            "type": "image_url",
            "image_url": {"url": url},
        }
    raise OpenAIConversionError(
        f"unsupported image source.type={source_type!r}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_converter.py -k "vision_or_convert_image" -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add core/anthropic/conversion.py tests/providers/test_converter.py
git commit -m "feat(vision): add VisionCapabilityProtocol, NO_VISION sentinel, and image→image_url helper"
```

---

### Task 2: Image branch in _convert_user_message and _convert_user_message_with_injection

**Files:**
- Modify: `core/anthropic/conversion.py:465-498` (`_convert_user_message`) and `:406-462` (`_convert_user_message_with_injection`)
- Test: `tests/providers/test_converter.py` (add T2 test cases)

**Interfaces:**
- Consumes: `VisionCapabilityProtocol`, `convert_anthropic_image_to_openai_image_url` from Task 1
- Produces: `_convert_user_message(*, vision=...)` and `_convert_user_message_with_injection(*, vision=...)` with image branch; same return types as before

- [ ] **Step 1: Write the failing tests**

Append to `tests/providers/test_converter.py`:

```python
# --- T2: _convert_user_message image branch ---


def test_convert_user_message_image_vision_off_explicit():
    """Passing vision=OFF explicitly still raises."""
    content = [
        MockBlock(
            type="image",
            source={"type": "url", "url": "https://example.com/i.png"},
        )
    ]
    messages = [MockMessage("user", content)]
    with pytest.raises(OpenAIConversionError):
        AnthropicToOpenAIConverter.convert_messages(
            messages, vision=_DisabledVision()
        )


def test_convert_user_message_image_vision_on_base64():
    """vision=ON: single base64 image emits an image_url user message."""
    content = [
        MockBlock(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "abc",
            },
        )
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, vision=_EnabledVision()
    )
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc"},
        }
    ]


def test_convert_user_message_image_vision_on_url():
    """vision=ON: URL image emits verbatim."""
    content = [
        MockBlock(
            type="image",
            source={"type": "url", "url": "https://example.com/x.png"},
        )
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, vision=_EnabledVision()
    )
    assert len(result) == 1
    assert result[0]["content"][0]["type"] == "image_url"
    assert (
        result[0]["content"][0]["image_url"]["url"]
        == "https://example.com/x.png"
    )


def test_convert_user_message_text_then_image():
    """vision=ON: text flushed first, then image as second user message."""
    content = [
        MockBlock(type="text", text="describe this"),
        MockBlock(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "abc",
            },
        ),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, vision=_EnabledVision()
    )
    assert len(result) == 2
    assert result[0] == {"role": "user", "content": "describe this"}
    assert result[1]["content"][0]["type"] == "image_url"


def test_convert_user_message_text_image_text():
    """vision=ON: text, image, text → three separate user messages."""
    content = [
        MockBlock(type="text", text="first"),
        MockBlock(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "Z",
            },
        ),
        MockBlock(type="text", text="second"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, vision=_EnabledVision()
    )
    assert len(result) == 3
    assert result[0] == {"role": "user", "content": "first"}
    assert result[1]["content"][0]["type"] == "image_url"
    assert result[2] == {"role": "user", "content": "second"}


def test_convert_user_message_two_images():
    """vision=ON: two adjacent images → two separate image_url user messages."""
    content = [
        MockBlock(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "A",
            },
        ),
        MockBlock(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "B",
            },
        ),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, vision=_EnabledVision()
    )
    assert len(result) == 2
    assert result[0]["content"][0]["type"] == "image_url"
    assert result[1]["content"][0]["type"] == "image_url"


def test_convert_user_message_with_injection_image_vision_on():
    """vision=ON inside _convert_user_message_with_injection (deferred-assistant path)."""
    pending = _PendingAfterTools(
        remaining_tool_ids={"call_z"},
        deferred_blocks=[MockBlock(type="text", text="after tool")],
        reasoning_replay=ReasoningReplayMode.THINK_TAGS,
    )
    content = [
        MockBlock(
            type="tool_result",
            tool_use_id="call_z",
            content="ok",
        ),
        MockBlock(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "X",
            },
        ),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, vision=_EnabledVision()
    )
    # tool_result + deferred-assistant + image user message
    tool_msgs = [m for m in result if m["role"] == "tool"]
    image_msgs = [
        m
        for m in result
        if m["role"] == "user"
        and isinstance(m.get("content"), list)
        and any(
            p.get("type") == "image_url" for p in m["content"] if isinstance(p, dict)
        )
    ]
    assert len(tool_msgs) == 1
    assert len(image_msgs) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_converter.py -k "vision_on_or_vision_off_explicit_or_text_then_image_or_text_image_text_or_two_images_or_injection_image" -v`
Expected: FAIL — `convert_messages() got an unexpected keyword argument 'vision'`

- [ ] **Step 3: Implement — add `vision=` kwarg to both `_convert_user_message` methods and `convert_messages`**

In `core/anthropic/conversion.py`, update three methods:

**(a)** `convert_messages` signature — add `*, vision: VisionCapabilityProtocol = NO_VISION` and pass it through:

```python
@staticmethod
def convert_messages(
    messages: list[Any],
    *,
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
    vision: VisionCapabilityProtocol = NO_VISION,
) -> list[dict[str, Any]]:
```

Then pass `vision=vision` to every `_convert_user_message` and `_convert_user_message_with_injection` call inside `convert_messages`. There are 4 call sites inside `convert_messages`:

- Line ~255: `AnthropicToOpenAIConverter._convert_user_message(content)` → add `vision=vision`
- Line ~260: `AnthropicToOpenAIConverter._convert_user_message_with_injection(content, pending)` → add `vision=vision`
- Line ~267: `AnthropicToOpenAIConverter._convert_user_message(content)` → add `vision=vision`
- Line ~269: `AnthropicToOpenAIConverter._convert_user_message(content)` (the else-branch) → add `vision=vision`

**(b)** `_convert_user_message` — add `*, vision: VisionCapabilityProtocol = NO_VISION` and replace the `raise` with a branch:

```python
@staticmethod
def _convert_user_message(
    content: list[Any],
    *,
    vision: VisionCapabilityProtocol = NO_VISION,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        if text_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})
            text_parts.clear()

    for block in content:
        block_type = get_block_type(block)

        if block_type == "text":
            text_parts.append(get_block_attr(block, "text", ""))
        elif block_type == "image":
            if vision.enabled:
                flush_text()
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
        elif block_type == "tool_result":
            flush_text()
            tool_content = get_block_attr(block, "content", "")
            serialized = _serialize_tool_result_content(tool_content)
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": get_block_attr(block, "tool_use_id"),
                    "content": serialized if serialized else "",
                }
            )

    flush_text()
    return result
```

**(c)** `_convert_user_message_with_injection` — add `*, vision: VisionCapabilityProtocol = NO_VISION` and replace the `raise` with a branch (mirrors the same pattern):

```python
@staticmethod
def _convert_user_message_with_injection(
    content: list[Any],
    pending: _PendingAfterTools,
    *,
    vision: VisionCapabilityProtocol = NO_VISION,
) -> dict[str, Any]:
    if not pending.needs_deferred() or not pending.remaining_tool_ids:
        return {
            "messages": AnthropicToOpenAIConverter._convert_user_message(
                content, vision=vision
            ),
            "cleared_pending": False,
        }

    result: list[dict[str, Any]] = []
    text_parts: list[str] = []
    cleared = False

    def flush_text() -> None:
        if text_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})
            text_parts.clear()

    for block in content:
        block_type = get_block_type(block)
        if block_type == "text":
            text_parts.append(get_block_attr(block, "text", ""))
        elif block_type == "image":
            if vision.enabled:
                flush_text()
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
        elif block_type == "tool_result":
            flush_text()
            tool_content = get_block_attr(block, "content", "")
            serialized = _serialize_tool_result_content(tool_content)
            tuid = get_block_attr(block, "tool_use_id")
            tuid_s = str(tuid) if tuid is not None else ""
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": tuid,
                    "content": serialized if serialized else "",
                }
            )
            if tuid_s in pending.remaining_tool_ids:
                pending.remaining_tool_ids.discard(tuid_s)
            if not pending.remaining_tool_ids:
                result.extend(
                    AnthropicToOpenAIConverter._deferred_post_tool_to_messages(
                        pending
                    )
                )
                pending.deferred_emitted = True
                cleared = True
        else:
            pass

    flush_text()
    return {"messages": result, "cleared_pending": cleared}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_converter.py -v`
Expected: ALL PASS (both old and new tests)

- [ ] **Step 5: Commit**

```bash
git add core/anthropic/conversion.py tests/providers/test_converter.py
git commit -m "feat(vision): add vision= kwarg to converter user-message methods with image branch"
```

---

### Task 3: _strip_inner_images_from_tool_result and tool_result image placeholder

**Files:**
- Modify: `core/anthropic/conversion.py` (add helper + constants + integrate into both converter methods)
- Test: `tests/providers/test_converter.py` (add T5 test cases)

**Interfaces:**
- Consumes: `get_block_type`, `get_block_attr` (existing), `VisionCapabilityProtocol` (Task 1)
- Produces: `_strip_inner_images_from_tool_result(content) -> tuple[Any, int]`, module constants `_TOOL_RESULT_IMAGE_PLACEHOLDER_TEXT` and `_TOOL_RESULT_IMAGE_PLACEHOLDER_BLOCK`

- [ ] **Step 1: Write the failing tests**

Append to `tests/providers/test_converter.py`:

```python
# --- T5: _strip_inner_images_from_tool_result ---


from core.anthropic.conversion import _strip_inner_images_from_tool_result


def test_strip_drops_images_inside_tool_result_and_warns():
    inner = [
        {"type": "text", "text": "screenshot below:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
        {"type": "text", "text": "end"},
    ]
    result, dropped = _strip_inner_images_from_tool_result(inner)
    assert dropped == 1
    assert result[0] == {"type": "text", "text": "screenshot below:"}
    assert result[1]["type"] == "text"
    assert "omitted" in result[1]["text"]
    assert result[2] == {"type": "text", "text": "end"}


def test_strip_no_op_on_string_content():
    result, dropped = _strip_inner_images_from_tool_result("just a string")
    assert result == "just a string"
    assert dropped == 0


def test_strip_no_op_when_no_images():
    inner = [
        {"type": "text", "text": "all text"},
    ]
    result, dropped = _strip_inner_images_from_tool_result(inner)
    assert result == inner
    assert dropped == 0


def test_strip_preserves_other_blocks():
    inner = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A"}},
        {"type": "text", "text": "between"},
        {"type": "image", "source": {"type": "url", "url": "https://x.com/i.png"}},
    ]
    result, dropped = _strip_inner_images_from_tool_result(inner)
    assert dropped == 2
    assert len(result) == 3
    assert result[0]["type"] == "text"
    assert result[1] == {"type": "text", "text": "between"}
    assert result[2]["type"] == "text"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_converter.py -k "strip_" -v`
Expected: FAIL — `ImportError: cannot import name '_strip_inner_images_from_tool_result'`

- [ ] **Step 3: Write the implementation**

Add these constants and helper to `core/anthropic/conversion.py`, just after `convert_anthropic_image_to_openai_image_url`:

```python
_TOOL_RESULT_IMAGE_PLACEHOLDER_TEXT = (
    "[attachment omitted: image stripped from tool_result for OpenAI compat]"
)

_TOOL_RESULT_IMAGE_PLACEHOLDER_BLOCK: dict[str, Any] = {
    "type": "text",
    "text": _TOOL_RESULT_IMAGE_PLACEHOLDER_TEXT,
}


def _strip_inner_images_from_tool_result(
    raw_content: Any,
) -> tuple[Any, int]:
    """Replace ``image`` blocks inside a tool_result content list with a placeholder.

    Returns ``(rewritten_content, count_of_replacements)``.  When ``raw_content``
    is a plain string (the common case), this is a no-op returning ``(content, 0)``.
    """
    if not isinstance(raw_content, list):
        return raw_content, 0

    dropped = 0
    rewritten: list[Any] = []
    for item in raw_content:
        if isinstance(item, dict) and item.get("type") == "image":
            rewritten.append(_TOOL_RESULT_IMAGE_PLACEHOLDER_BLOCK)
            dropped += 1
        else:
            rewritten.append(item)
    return rewritten, dropped
```

Then integrate into both `_convert_user_message` and `_convert_user_message_with_injection` — replace the `elif block_type == "tool_result":` branch in each method:

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
            # ... rest same as before (tool_call_id, result.append ...)
```

Add `from loguru import logger` at the top of `core/anthropic/conversion.py` if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_converter.py -v`
Expected: ALL PASS (including existing tests and all new T5 tests)

- [ ] **Step 5: Commit**

```bash
git add core/anthropic/conversion.py tests/providers/test_converter.py
git commit -m "feat(vision): add _strip_inner_images_from_tool_result and wire into converter tool_result branch"
```

---

### Task 4: Thread vision= through build_base_request_body and re-export from core/anthropic/__init__.py

**Files:**
- Modify: `core/anthropic/conversion.py:548-587` (`build_base_request_body` signature)
- Modify: `core/anthropic/__init__.py` (add re-exports)
- Test: `tests/providers/test_converter.py` (add T3 test cases)

**Interfaces:**
- Consumes: `VisionCapabilityProtocol`, `NO_VISION` from Task 1; `convert_messages(*, vision=...)` from Task 2
- Produces: `build_base_request_body(*, vision=...)` (keyword-only, default `NO_VISION`); public re-exports `VisionCapabilityProtocol`, `NO_VISION`, `convert_anthropic_image_to_openai_image_url` from `core.anthropic`

- [ ] **Step 1: Write the failing tests**

Append to `tests/providers/test_converter.py`:

```python
# --- T3: build_base_request_body vision= kwarg ---


def test_build_base_request_body_default_vision_is_off():
    """Calling build_base_request_body without vision= and giving an image block raises."""
    req = MessagesRequest(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://x.com/i.png"}},
                    {"type": "text", "text": "what is this?"},
                ],
            }
        ],
        max_tokens=100,
    )
    with pytest.raises(OpenAIConversionError):
        build_base_request_body(req)


def test_build_base_request_body_vision_on_passes_through():
    """vision=ON: image block ends up as image_url content part in the body."""
    req = MessagesRequest(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    },
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
        max_tokens=100,
    )
    body = build_base_request_body(req, vision=_EnabledVision())
    user_msgs = [m for m in body["messages"] if m["role"] == "user"]
    image_parts = [
        m for m in user_msgs
        if isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"] if isinstance(p, dict))
    ]
    assert len(image_parts) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_converter.py -k "build_base_request_body_default_vision_or_build_base_request_body_vision_on" -v`
Expected: FAIL — `build_base_request_body() got an unexpected keyword argument 'vision'`

- [ ] **Step 3: Implement — add `vision=` kwarg to `build_base_request_body`**

In `core/anthropic/conversion.py`, update `build_base_request_body`:

```python
def build_base_request_body(
    request_data: Any,
    *,
    default_max_tokens: int | None = None,
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
    vision: VisionCapabilityProtocol = NO_VISION,
) -> dict[str, Any]:
    """Build the common parts of an OpenAI-format request body."""
    _openai_reject_native_only_top_level_fields(request_data)
    messages = AnthropicToOpenAIConverter.convert_messages(
        request_data.messages,
        reasoning_replay=reasoning_replay,
        vision=vision,
    )
    # ... rest unchanged
```

- [ ] **Step 4: Re-export from `core/anthropic/__init__.py`**

Add to the import block and `__all__`:

```python
from .conversion import (
    # ... existing imports ...
    NO_VISION,
    VisionCapabilityProtocol,
    convert_anthropic_image_to_openai_image_url,
)
```

And add to `__all__`:

```python
    "NO_VISION",
    "VisionCapabilityProtocol",
    "convert_anthropic_image_to_openai_image_url",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_converter.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add core/anthropic/conversion.py core/anthropic/__init__.py tests/providers/test_converter.py
git commit -m "feat(vision): thread vision= through build_base_request_body and re-export from core.anthropic"
```

---

### Task 5: Transport-level _vision_capability() hook on OpenAIChatTransport

**Files:**
- Modify: `providers/transports/openai_chat/transport.py` (add `_vision_capability` method)
- Modify: `providers/transports/openai_chat/stream.py` (thread `vision=self._vision_capability()` into `_build_request_body`)
- Test: `tests/providers/test_openai_compat_5xx_retry.py` (add T4-a: default-capability test)

**Interfaces:**
- Consumes: `VisionCapabilityProtocol`, `NO_VISION` from `core.anthropic` (Task 4)
- Produces: `OpenAIChatTransport._vision_capability() -> VisionCapabilityProtocol` (returns `NO_VISION` by default; subclasses override)

- [ ] **Step 1: Write the failing test**

In `tests/providers/test_openai_compat_5xx_retry.py`, add at the bottom:

```python
from core.anthropic import NO_VISION, VisionCapabilityProtocol


def test_default_vision_capability_is_no_vision():
    """Base OpenAIChatTransport._vision_capability() returns NO_VISION by default."""
    # Use a minimal concrete subclass since OpenAIChatTransport is abstract
    from providers.transports.openai_chat import OpenAIChatTransport
    from providers.base import ProviderConfig

    class _StubTransport(OpenAIChatTransport):
        def _build_request_body(self, request, thinking_enabled=None):
            return {}

    config = ProviderConfig(
        base_url="http://localhost",
        api_key="test",
    )
    transport = _StubTransport(
        config,
        provider_name="stub",
        base_url="http://localhost",
        api_key="test",
    )
    capability = transport._vision_capability()
    assert isinstance(capability, VisionCapabilityProtocol)
    assert capability.enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/providers/test_openai_compat_5xx_retry.py -k "default_vision_capability" -v`
Expected: FAIL — `AttributeError: 'StubTransport' object has no attribute '_vision_capability'`

- [ ] **Step 3: Write the implementation**

Add to `providers/transports/openai_chat/transport.py`, after `_prepare_create_body`:

```python
from core.anthropic import NO_VISION, VisionCapabilityProtocol

    def _vision_capability(self) -> VisionCapabilityProtocol:
        """Subclasses override to advertise vision support. Default = NO_VISION."""
        return NO_VISION
```

Then in `providers/transports/openai_chat/stream.py`, inside `OpenAIChatStreamRunner`, locate where `_build_request_body` is called and thread `vision=self._vision_capability()`:

Read the file first to find the exact call site. The runner holds a reference to the transport (`self._transport`). Change the `_build_request_body` call to also pass `vision=self._transport._vision_capability()`, which then gets forwarded through to `build_base_request_body` via the per-provider override chain (Task 7 will wire the NIM side).

Note: The runner does not call `build_base_request_body` directly — it calls `self._transport._build_request_body(request, ...)`, and each provider's `_build_request_body` override calls its own body builder (e.g. `providers/nvidia_nim/request.py::build_request_body`), which calls `build_base_request_body`. So the threading is:

```
stream runner → transport._build_request_body(request, thinking_enabled)
    → provider's build_request_body(request, nim, thinking_enabled, vision=...)
        → build_base_request_body(request, ..., vision=vision)
```

The transport-level method is used by each provider's `_build_request_body` override to source the capability. No change to `stream.py` is needed now — the transport exposes `_vision_capability()`, and each provider override reads it. The actual threading is in Task 7.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/providers/test_openai_compat_5xx_retry.py -k "default_vision_capability" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/transports/openai_chat/transport.py tests/providers/test_openai_compat_5xx_retry.py
git commit -m "feat(vision): add _vision_capability() hook on OpenAIChatTransport (default NO_VISION)"
```

---

### Task 6: NimSettings.vision_enabled field, env binding in config/settings.py, and .env.example

**Files:**
- Modify: `config/nim.py` (add `vision_enabled: bool = False` field)
- Modify: `config/settings.py` (add env binding to pipe `NVIDIA_NIM_VISION_ENABLED` to `self.nim.vision_enabled`)
- Modify: `.env.example` (add documented setting)
- Test: `tests/providers/test_nvidia_nim_request.py` (add settings-field test)

**Interfaces:**
- Consumes: nothing new
- Produces: `NimSettings.vision_enabled` (bool, default False); env alias `NVIDIA_NIM_VISION_ENABLED`

- [ ] **Step 1: Write the failing test**

Append to `tests/providers/test_nvidia_nim_request.py`:

```python
def test_nim_settings_vision_enabled_defaults_false():
    """vision_enabled defaults to False on NimSettings."""
    nim = NimSettings()
    assert nim.vision_enabled is False


def test_nim_settings_vision_enabled_can_be_set_true():
    nim = NimSettings(vision_enabled=True)
    assert nim.vision_enabled is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_nvidia_nim_request.py -k "vision_enabled" -v`
Expected: FAIL — `ValidationError: Extra inputs are not permitted` (because `NimSettings` has `extra="forbid"` and the field doesn't exist yet)

- [ ] **Step 3: Add `vision_enabled` to `NimSettings`**

In `config/nim.py`, add to the `NimSettings` class body (after `request_id`):

```python
    vision_enabled: bool = Field(
        False,
        alias="NVIDIA_NIM_VISION_ENABLED",
        description="Enable vision (image input) handling for vision-capable NIM models.",
    )
```

Because `NimSettings` has `model_config = ConfigDict(extra="forbid")`, the alias allows the env var name to map into this field.

- [ ] **Step 4: Add env binding in `config/settings.py`**

Inspect `config/settings.py` to see how other `NVIDIA_NIM_*` env vars are piped into `NimSettings`. The existing pattern uses a `model_validator(mode="after")`. Add `NVIDIA_NIM_VISION_ENABLED` to the same validator:

```python
# Inside the existing NIM env-binding validator, add:
    if env_nim_vision := os.environ.get("NVIDIA_NIM_VISION_ENABLED"):
        self.nim.vision_enabled = env_nim_vision.lower() in ("true", "1", "yes")
```

Check the exact validator structure first — the implementor should read `config/settings.py` and mirror the exact pattern used for other `NVIDIA_NIM_*` fields.

- [ ] **Step 5: Add to `.env.example`**

After the existing `NVIDIA_NIM_*` block:

```
# Enable vision (image input) handling for vision-capable NIM models.
# When false, any Anthropic image block in client messages is rejected with HTTP 400.
NVIDIA_NIM_VISION_ENABLED=false
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_nvidia_nim_request.py -k "vision_enabled" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add config/nim.py config/settings.py .env.example tests/providers/test_nvidia_nim_request.py
git commit -m "feat(vision): add NimSettings.vision_enabled field, env binding, and .env.example entry"
```

---

### Task 7: NIM vision wiring — _NimVisionCapability, _vision_capability() override, and vision= threading through build_request_body

**Files:**
- Modify: `providers/nvidia_nim/request.py:348-430` (`build_request_body` — add `vision:` kwarg, thread to `build_base_request_body`)
- Modify: `providers/nvidia_nim/client.py:24-91` (`NvidiaNimProvider` — override `_vision_capability()`, pass `vision=` into `build_request_body`)
- Test: `tests/providers/test_nvidia_nim.py` (add T4-b: NIM capability + body-build tests)

**Interfaces:**
- Consumes: `NimSettings.vision_enabled` from Task 6; `VisionCapabilityProtocol`, `NO_VISION` from Task 4; `_vision_capability()` from Task 5
- Produces: `NvidiaNimProvider._vision_capability() -> VisionCapabilityProtocol`; `build_request_body(*, vision=...)` which threads vision into `build_base_request_body(vision=...)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/providers/test_nvidia_nim.py`:

```python
from core.anthropic import NO_VISION, VisionCapabilityProtocol


def test_nvidia_nim_vision_capability_off_when_setting_false():
    """NvidiaNimProvider with vision_enabled=False returns NO_VISION."""
    nim = NimSettings(vision_enabled=False)
    config = ProviderConfig(
        base_url="http://localhost:8000",
        api_key="test",
    )
    provider = NvidiaNimProvider(config, nim_settings=nim)
    capability = provider._vision_capability()
    assert capability.enabled is False


def test_nvidia_nim_vision_capability_on_when_setting_true():
    """NvidiaNimProvider with vision_enabled=True returns an enabled capability."""
    nim = NimSettings(vision_enabled=True)
    config = ProviderConfig(
        base_url="http://localhost:8000",
        api_key="test",
    )
    provider = NvidiaNimProvider(config, nim_settings=nim)
    capability = provider._vision_capability()
    assert isinstance(capability, VisionCapabilityProtocol)
    assert capability.enabled is True


def test_nvidia_nim_build_request_body_image_passes_through_when_vision_on():
    """End-to-end: image block in request body survives when vision=ON."""
    from api.models.anthropic import MessagesRequest

    nim = NimSettings(vision_enabled=True, max_tokens=100)
    req = MessagesRequest(
        model="nvidia/nemotron",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    },
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
        max_tokens=100,
    )
    body = build_request_body(req, nim, thinking_enabled=False, vision=nim.vision_enabled and _NimVisionCap() or NO_VISION)
    user_msgs = [m for m in body["messages"] if m["role"] == "user"]
    image_parts = [
        m for m in user_msgs
        if isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"] if isinstance(p, dict))
    ]
    assert len(image_parts) >= 1


def test_nvidia_nim_build_request_body_image_raises_when_vision_off():
    """End-to-end: image block raises InvalidRequestError when vision=OFF."""
    from api.models.anthropic import MessagesRequest
    from providers.exceptions import InvalidRequestError

    nim = NimSettings(vision_enabled=False)
    req = MessagesRequest(
        model="nvidia/nemotron",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://x.com/i.png"},
                    },
                ],
            }
        ],
        max_tokens=100,
    )
    with pytest.raises(InvalidRequestError, match="image blocks"):
        build_request_body(req, nim, thinking_enabled=False)
```

Note: The `test_nvidia_nim_build_request_body_image_passes_through_when_vision_on` test will need to use the `_NimVisionCapability` class that gets defined in this task, or use a simple stub. The implementor should adjust the test to match the actual class name.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_nvidia_nim.py -k "vision_capability" -v`
Expected: FAIL — `AttributeError: 'NvidiaNimProvider' object has no attribute '_vision_capability'`

- [ ] **Step 3: Add `_NimVisionCapability` to `providers/nvidia_nim/request.py`**

Add after the existing imports, before `build_request_body`:

```python
from core.anthropic import NO_VISION, VisionCapabilityProtocol


class _NimVisionCapability:
    """Vision capability sourced from NimSettings.vision_enabled."""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled
```

- [ ] **Step 4: Add `vision=` kwarg to `build_request_body` and thread into `build_base_request_body`**

Update the signature:

```python
def build_request_body(
    request_data: Any,
    nim: NimSettings,
    *,
    thinking_enabled: bool,
    vision: VisionCapabilityProtocol = NO_VISION,
) -> dict:
```

And thread `vision=vision` into the `build_base_request_body` call (around line 358):

```python
        body = build_base_request_body(
            request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
            if thinking_enabled
            else ReasoningReplayMode.DISABLED,
            vision=vision,
        )
```

- [ ] **Step 5: Override `_vision_capability()` in `NvidiaNimProvider` and thread `vision=` into `build_request_body`**

In `providers/nvidia_nim/client.py`:

```python
from core.anthropic import NO_VISION, VisionCapabilityProtocol

    def _vision_capability(self) -> VisionCapabilityProtocol:
        """NIM vision capability sourced from NimSettings."""
        if self._nim_settings.vision_enabled:
            from .request import _NimVisionCapability
            return _NimVisionCapability(enabled=True)
        return NO_VISION
```

And update `_build_request_body` to pass `vision`:

```python
    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
            request,
            self._nim_settings,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            vision=self._vision_capability(),
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_nvidia_nim.py -v`
Expected: ALL PASS (including existing and new tests)

- [ ] **Step 7: Commit**

```bash
git add providers/nvidia_nim/request.py providers/nvidia_nim/client.py tests/providers/test_nvidia_nim.py
git commit -m "feat(vision): wire NIM vision capability through provider override and request builder"
```

---

### Task 8: Semver MINOR bump and uv.lock refresh

**Files:**
- Modify: `pyproject.toml:7` (version `2.3.14` → `2.4.0`)
- Modify: `uv.lock` (auto-refreshed by `uv lock`)

**Interfaces:**
- Consumes: nothing new
- Produces: bumped version and refreshed lockfile

- [ ] **Step 1: Bump version in pyproject.toml**

Change line 7 from `version = "2.3.14"` to `version = "2.4.0"`.

- [ ] **Step 2: Run uv lock**

Run: `uv lock`
Expected: `uv.lock` updated with new package version. Exit 0.

- [ ] **Step 3: Verify CI passes locally**

Run: `./scripts/ci.sh`
Expected: All 5 checks green (suppression grep, ruff-format, ruff-check, ty, pytest)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump version 2.3.14 → 2.4.0 (MINOR — vision capability addition)"
```

---

### Task 9: Live smoke test — test_nvidia_nim_vision_product_live.py

**Files:**
- Create: `smoke/product/test_nvidia_nim_vision_product_live.py`

**Interfaces:**
- Consumes: `SmokeConfig`, `SmokeServerDriver` from `smoke.lib` (existing smoke infrastructure); `FCC_LIVE_SMOKE=1` env gate from `smoke/conftest.py`
- Produces: env-gated live NIM vision smoke test (T6 from spec)

- [ ] **Step 1: Create the smoke test file**

Create `smoke/product/test_nvidia_nim_vision_product_live.py`:

```python
"""Live smoke test: NVIDIA NIM vision (image input) handling.

Gated behind FCC_LIVE_SMOKE=1 and smoke_target("nvidia_nim_vision").
Requires NVIDIA_NIM_API_KEY and a vision-capable model configured in smoke config.
"""

from __future__ import annotations

import base64

import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import SmokeServerDriver

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("nvidia_nim_vision")]

# Minimal 1x1 black PNG (base64-encoded, standard)
_BLACK_1X1_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_nim_vision_base64_image_succeeds(
    smoke_config: SmokeConfig,
) -> None:
    """Send a base64 PNG image to a vision-capable NIM model and expect a 200 response."""
    if not smoke_config.has_provider_configuration("nvidia_nim"):
        pytest.skip("missing_env: NVIDIA_NIM_API_KEY is not configured")

    provider_models = smoke_config.nvidia_nim_cli_models()
    if not provider_models:
        pytest.skip("missing_env: no NVIDIA NIM vision smoke models configured")

    # Use the first configured model
    provider_model = provider_models[0]

    with SmokeServerDriver(
        smoke_config,
        name="product-nvidia-nim-vision-base64",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
            "NVIDIA_NIM_VISION_ENABLED": "true",
        },
    ).run() as server:
        # Build an Anthropic-format request with image content
        import httpx

        response = httpx.post(
            f"http://localhost:{server.port}/v1/messages",
            headers={
                "x-api-key": server.api_key,
                "content-type": "application/json",
            },
            json={
                "model": provider_model.full_model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _BLACK_1X1_PNG_B64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "이 이미지는 검은색이야?",
                            },
                        ],
                    }
                ],
            },
            timeout=60.0,
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:500]}"
        )
        # Verify non-empty SSE content
        body = response.text
        assert "content_block_delta" in body or "content_block_start" in body, (
            f"Expected SSE content blocks in response: {body[:500]}"
        )


def test_nim_vision_disabled_rejects_image(
    smoke_config: SmokeConfig,
) -> None:
    """When NVIDIA_NIM_VISION_ENABLED=false, image blocks should be rejected with 400."""
    if not smoke_config.has_provider_configuration("nvidia_nim"):
        pytest.skip("missing_env: NVIDIA_NIM_API_KEY is not configured")

    provider_models = smoke_config.nvidia_nim_cli_models()
    if not provider_models:
        pytest.skip("missing_env: no NVIDIA NIM vision smoke models configured")

    provider_model = provider_models[0]

    with SmokeServerDriver(
        smoke_config,
        name="product-nvidia-nim-vision-disabled",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
            "NVIDIA_NIM_VISION_ENABLED": "false",
        },
    ).run() as server:
        import httpx

        response = httpx.post(
            f"http://localhost:{server.port}/v1/messages",
            headers={
                "x-api-key": server.api_key,
                "content-type": "application/json",
            },
            json={
                "model": provider_model.full_model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _BLACK_1X1_PNG_B64,
                                },
                            },
                        ],
                    }
                ],
            },
            timeout=30.0,
        )
        assert response.status_code == 400, (
            f"Expected 400 (vision off), got {response.status_code}: {response.text[:500]}"
        )
        assert "image" in response.text.lower(), (
            f"Expected 'image' in error message: {response.text[:500]}"
        )
```

- [ ] **Step 2: Verify smoke test is collected but skipped without FCC_LIVE_SMOKE=1**

Run: `uv run pytest smoke/product/test_nvidia_nim_vision_product_live.py --co`
Expected: Tests collected but marked `SKIPPED` (no `FCC_LIVE_SMOKE=1`)

- [ ] **Step 3: Commit**

```bash
git add smoke/product/test_nvidia_nim_vision_product_live.py
git commit -m "test(vision): add env-gated live NIM vision smoke test"
```
