import json
from types import SimpleNamespace

import pytest

from free_claude_code.providers.nvidia_nim.client import _PROFILE as NIM_PROFILE
from free_claude_code.providers.nvidia_nim.native_tool_stream import (
    normalize_nim_native_tool_stream,
)
from free_claude_code.providers.openai_chat import text_tool_stream
from free_claude_code.providers.openai_chat.profiles import OPENAI_CHAT_PROFILES
from free_claude_code.providers.openai_chat.text_tool_stream import (
    OpenAITextToolDialect,
    OpenAITextToolProtocolError,
    normalize_openai_text_tool_stream,
)

_CONTROL_PREFIX = "Thinking.</think>\n"


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[object] | None = None,
    finish_reason: str | None = None,
    usage: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _structured_call(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        index=0,
        id="call_structured",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


async def _stream(chunks: list[SimpleNamespace]):
    for chunk in chunks:
        yield chunk


async def _normalize(
    chunks: list[SimpleNamespace], body: dict
) -> list[SimpleNamespace]:
    return [
        chunk
        async for chunk in normalize_openai_text_tool_stream(
            _stream(chunks),
            body,
            dialect=OpenAITextToolDialect.FUNCTION_TAGS,
        )
    ]


def _body(*functions: dict) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": function,
            }
            for function in functions
        ]
    }


def _function(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def _tool_block(name: str, **parameters: str) -> str:
    arguments = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n"
        for key, value in parameters.items()
    )
    return f"<tool_call>\n<function={name}>\n{arguments}</function>\n</tool_call>"


def _content(chunks: list[SimpleNamespace]) -> str:
    return "".join(
        content
        for chunk in chunks
        for choice in chunk.choices
        if (content := getattr(choice.delta, "content", None)) is not None
    )


def _reasoning(chunks: list[SimpleNamespace]) -> str:
    return "".join(
        reasoning
        for chunk in chunks
        for choice in chunk.choices
        if (reasoning := getattr(choice.delta, "reasoning_content", None)) is not None
    )


def _tool_calls(chunks: list[SimpleNamespace]) -> list[SimpleNamespace]:
    calls: list[SimpleNamespace] = []
    for chunk in chunks:
        for choice in chunk.choices:
            value = getattr(choice.delta, "tool_calls", None)
            if isinstance(value, list):
                calls.extend(value)
    return calls


def test_function_tag_dialect_requires_explicit_provider_opt_in() -> None:
    assert NIM_PROFILE.text_tool_dialect is OpenAITextToolDialect.FUNCTION_TAGS
    assert all(
        profile.text_tool_dialect is None for profile in OPENAI_CHAT_PROFILES.values()
    )


@pytest.mark.asyncio
async def test_normalizer_preserves_stream_without_declared_tools() -> None:
    source = _chunk(content="literal <tool_call> text", finish_reason="stop")

    normalized = await _normalize([source], {})

    assert normalized == [source]
    assert normalized[0] is source


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["content", "reasoning_content"])
async def test_normalizer_converts_step_xml_to_structured_tool_call(
    field: str,
) -> None:
    raw = _CONTROL_PREFIX + _tool_block("Bash", command="ls -la")
    body = _body(
        _function(
            "Bash",
            {"command": {"type": "string"}},
            ["command"],
        )
    )

    normalized = await _normalize(
        [
            _chunk(
                content=raw if field == "content" else None,
                reasoning_content=raw if field == "reasoning_content" else None,
                finish_reason="stop",
            )
        ],
        body,
    )

    visible = _content(normalized) if field == "content" else _reasoning(normalized)
    assert visible == _CONTROL_PREFIX
    assert "<tool_call>" not in _content(normalized)
    assert "<tool_call>" not in _reasoning(normalized)
    calls = _tool_calls(normalized)
    assert len(calls) == 1
    assert calls[0].function.name == "Bash"
    assert json.loads(calls[0].function.arguments) == {"command": "ls -la"}
    assert normalized[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_normalizer_preserves_valid_envelope_without_reasoning_boundary() -> None:
    raw = _tool_block("Bash", command="rm -rf example")
    body = _body(_function("Bash", {"command": {"type": "string"}}, ["command"]))

    normalized = await _normalize(
        [_chunk(content=raw, finish_reason="stop")],
        body,
    )

    assert _content(normalized) == raw
    assert not _tool_calls(normalized)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix",
    (
        "Here is an example: ",
        "```xml\n",
        "The literal protocol is\n",
    ),
)
async def test_normalizer_preserves_literal_tool_envelope_in_prose(
    prefix: str,
) -> None:
    raw = prefix + _tool_block("Bash", command="rm -rf example")
    if prefix.startswith("```"):
        raw += "\n```"
    body = _body(_function("Bash", {"command": {"type": "string"}}, ["command"]))

    normalized = await _normalize(
        [_chunk(content=raw, finish_reason="stop")],
        body,
    )

    assert _content(normalized) == raw
    assert not _tool_calls(normalized)


@pytest.mark.asyncio
async def test_normalizer_preserves_split_literal_tool_envelope_in_prose() -> None:
    raw = "Example: " + _tool_block("Read", file_path="README.md")
    chunks = [_chunk(content=character) for character in raw]
    chunks.append(_chunk(finish_reason="stop"))
    body = _body(_function("Read", {"file_path": {"type": "string"}}, ["file_path"]))

    normalized = await _normalize(chunks, body)

    assert _content(normalized) == raw
    assert not _tool_calls(normalized)


@pytest.mark.asyncio
async def test_normalizer_handles_live_stepfun_chunk_boundaries() -> None:
    chunks = [
        _chunk(content="Thinking first.</think>\n<tool_call>\n<function=Bash>\n"),
        _chunk(content="<parameter="),
        _chunk(content="command>\nls"),
        _chunk(content="\n</parameter"),
        _chunk(content=">\n</function"),
        _chunk(content=">\n</tool_call>", finish_reason="stop"),
    ]
    body = _body(_function("Bash", {"command": {"type": "string"}}, ["command"]))

    normalized = await _normalize(chunks, body)

    assert _content(normalized) == "Thinking first.</think>\n"
    calls = _tool_calls(normalized)
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"command": "ls"}


@pytest.mark.asyncio
async def test_nim_and_text_normalizers_recover_live_terminal_order() -> None:
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    usage_chunk = SimpleNamespace(choices=[], usage=usage)
    body = _body(_function("Bash", {"command": {"type": "string"}}, ["command"]))
    source = _stream(
        [
            _chunk(finish_reason="stop"),
            usage_chunk,
            _chunk(
                content=_CONTROL_PREFIX + _tool_block("Bash", command="ls"),
                finish_reason="stop",
            ),
        ]
    )

    nim_stream = normalize_nim_native_tool_stream(source, body)
    normalized = [
        chunk
        async for chunk in normalize_openai_text_tool_stream(
            nim_stream,
            body,
            dialect=OpenAITextToolDialect.FUNCTION_TAGS,
        )
    ]

    assert _content(normalized) == _CONTROL_PREFIX
    calls = _tool_calls(normalized)
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"command": "ls"}
    terminal = [
        chunk
        for chunk in normalized
        if chunk.choices and chunk.choices[0].finish_reason is not None
    ]
    assert len(terminal) == 1
    assert terminal[0].usage is usage


@pytest.mark.asyncio
async def test_normalizer_handles_tool_block_split_at_every_character() -> None:
    raw = _CONTROL_PREFIX + _tool_block("Read", file_path="README.md")
    chunks = [_chunk(content=character) for character in raw]
    chunks.append(_chunk(finish_reason="stop"))
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )

    normalized = await _normalize(chunks, body)

    assert _content(normalized) == _CONTROL_PREFIX
    calls = _tool_calls(normalized)
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"file_path": "README.md"}


@pytest.mark.asyncio
async def test_normalizer_preserves_partial_control_marker_as_text() -> None:
    body = _body(_function("Read", {"file_path": {"type": "string"}}, ["file_path"]))

    normalized = await _normalize(
        [_chunk(content="literal <tool_", finish_reason="stop")],
        body,
    )

    assert _content(normalized) == "literal <tool_"
    assert not _tool_calls(normalized)


@pytest.mark.asyncio
async def test_normalizer_decodes_typed_parameters_and_parallel_calls() -> None:
    raw = _CONTROL_PREFIX + "\n".join(
        (
            _tool_block(
                "Search",
                count="3",
                exact="true",
                filters='{"language":"python"}',
                paths='["src", "tests"]',
            ),
            _tool_block("Read", file_path="README.md"),
        )
    )
    body = _body(
        _function(
            "Search",
            {
                "count": {"type": "integer"},
                "exact": {"type": "boolean"},
                "filters": {"type": "object"},
                "paths": {"type": "array", "items": {"type": "string"}},
            },
            ["count", "exact", "filters", "paths"],
        ),
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        ),
    )

    normalized = await _normalize(
        [_chunk(content=raw, finish_reason="tool_calls")], body
    )

    assert _content(normalized) == _CONTROL_PREFIX
    calls = _tool_calls(normalized)
    assert [call.index for call in calls] == [0, 1]
    assert [call.function.name for call in calls] == ["Search", "Read"]
    assert json.loads(calls[0].function.arguments) == {
        "count": 3,
        "exact": True,
        "filters": {"language": "python"},
        "paths": ["src", "tests"],
    }


@pytest.mark.asyncio
async def test_normalizer_prefers_structured_calls_and_strips_text_duplicate() -> None:
    raw = _CONTROL_PREFIX + _tool_block("Read", file_path="README.md")
    structured = _structured_call("Read", '{"file_path":"README.md"}')
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )

    normalized = await _normalize(
        [
            _chunk(content=raw, tool_calls=[structured]),
            _chunk(finish_reason="tool_calls"),
        ],
        body,
    )

    assert _content(normalized) == _CONTROL_PREFIX
    assert _tool_calls(normalized) == [structured]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    (
        "<tool_call>",
        "<tool_call><function=Read><parameter=file_path>x</function></tool_call>",
        _tool_block("Read", file_path="README.md") + "suffix",
    ),
)
async def test_normalizer_preserves_malformed_control_lookalikes(raw: str) -> None:
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )

    output = _CONTROL_PREFIX + raw
    normalized = await _normalize(
        [_chunk(content=output, finish_reason="stop")],
        body,
    )

    assert _content(normalized) == output
    assert not _tool_calls(normalized)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw, message",
    (
        (_tool_block("Unknown", value="x"), "undeclared"),
        (_tool_block("Read", wrong="x"), "schema-invalid"),
    ),
)
async def test_normalizer_rejects_schema_invalid_control_protocol(
    raw: str,
    message: str,
) -> None:
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )

    with pytest.raises(OpenAITextToolProtocolError, match=message):
        await _normalize(
            [_chunk(content=_CONTROL_PREFIX + raw, finish_reason="stop")],
            body,
        )


@pytest.mark.asyncio
async def test_normalizer_preserves_oversized_control_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _tool_block("Read", file_path="README.md")
    body = _body(
        _function(
            "Read",
            {"file_path": {"type": "string"}},
            ["file_path"],
        )
    )
    monkeypatch.setattr(text_tool_stream, "_MAX_TEXT_TOOL_BLOCK_CHARS", len(raw))

    await _normalize(
        [_chunk(content=_CONTROL_PREFIX + raw, finish_reason="stop")],
        body,
    )
    oversized = f"{raw[: -len('</tool_call>')]}x</tool_call>"
    output = _CONTROL_PREFIX + oversized
    normalized = await _normalize(
        [_chunk(content=output, finish_reason="stop")],
        body,
    )

    assert _content(normalized) == output
    assert not _tool_calls(normalized)
