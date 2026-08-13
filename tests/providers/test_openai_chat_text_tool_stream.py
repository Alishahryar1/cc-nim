import json
from types import SimpleNamespace

import pytest

from free_claude_code.providers.nvidia_nim.native_tool_stream import (
    normalize_nim_native_tool_stream,
)
from free_claude_code.providers.openai_chat import text_tool_stream
from free_claude_code.providers.openai_chat.text_tool_stream import (
    OpenAITextToolProtocolError,
    normalize_openai_text_tool_stream,
)


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
        async for chunk in normalize_openai_text_tool_stream(_stream(chunks), body)
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
    raw = "Preparing. " + _tool_block("Bash", command="ls -la")
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
    assert visible == "Preparing. "
    assert "<tool_call>" not in _content(normalized)
    assert "<tool_call>" not in _reasoning(normalized)
    calls = _tool_calls(normalized)
    assert len(calls) == 1
    assert calls[0].function.name == "Bash"
    assert json.loads(calls[0].function.arguments) == {"command": "ls -la"}
    assert normalized[-1].choices[0].finish_reason == "stop"


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
            _chunk(content=_tool_block("Bash", command="ls"), finish_reason="stop"),
        ]
    )

    nim_stream = normalize_nim_native_tool_stream(source, body)
    normalized = [
        chunk async for chunk in normalize_openai_text_tool_stream(nim_stream, body)
    ]

    assert _content(normalized) == ""
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
    raw = "Before. " + _tool_block("Read", file_path="README.md")
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

    assert _content(normalized) == "Before. "
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
    raw = "\n".join(
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
    raw = "<tool_call><function=Broken>"
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

    assert _content(normalized) == ""
    assert _tool_calls(normalized) == [structured]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw, message",
    (
        ("<tool_call>", "incomplete"),
        (_tool_block("Unknown", value="x"), "undeclared"),
        (
            "<tool_call><function=Read><parameter=file_path>x</function></tool_call>",
            "incomplete parameter",
        ),
        (
            _tool_block("Read", wrong="x"),
            "schema-invalid",
        ),
        (
            _tool_block("Read", file_path="README.md") + "suffix",
            "content after",
        ),
    ),
)
async def test_normalizer_rejects_invalid_native_tool_protocol(
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
        await _normalize([_chunk(content=raw, finish_reason="stop")], body)


@pytest.mark.asyncio
async def test_normalizer_enforces_native_tool_block_size(
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

    await _normalize([_chunk(content=raw, finish_reason="stop")], body)
    with pytest.raises(OpenAITextToolProtocolError, match="oversized"):
        await _normalize(
            [_chunk(content=f"{raw[: -len('</tool_call>')]}x</tool_call>")], body
        )
