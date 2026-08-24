import pytest

from free_claude_code.core.inference import (
    CustomTool,
    CustomToolFormatType,
    FunctionTool,
    InstructionItem,
    InstructionOrigin,
    InstructionPlacement,
    MessageItem,
    MessageRole,
    ReasoningItem,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    ReplayCompatibilityScope,
    TextContent,
    ToolCallItem,
    ToolCallKind,
    ToolChoiceMode,
    ToolResultItem,
)
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    responses_to_inference_request,
)
from free_claude_code.core.reasoning import ReasoningControl, ReasoningEffort
from free_claude_code.core.replay_envelope import encode_replay_envelope


def _ingest(payload: dict[str, object]):
    return responses_to_inference_request(
        OpenAIResponsesRequest.model_validate(payload)
    ).request


def test_responses_string_input_becomes_canonical_instruction_and_message() -> None:
    request = _ingest(
        {
            "model": "nvidia_nim/test-model",
            "instructions": "System instructions",
            "input": "Hello",
            "max_output_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9,
            "metadata": {"trace": "abc"},
        }
    )

    assert request.model == "nvidia_nim/test-model"
    assert request.items == (
        InstructionItem(
            text="System instructions",
            origin=InstructionOrigin.SYSTEM,
            placement=InstructionPlacement.TOP_LEVEL,
        ),
        MessageItem(
            turn_id="turn_0",
            role=MessageRole.USER,
            content=(TextContent("Hello"),),
        ),
    )
    assert request.max_output_tokens == 64
    assert request.temperature == 0.2
    assert request.top_p == 0.9
    assert request.metadata == {"trace": "abc"}


@pytest.mark.parametrize(
    ("effort", "control", "expected_effort"),
    [
        ("none", ReasoningControl.OFF, None),
        ("low", ReasoningControl.ON, ReasoningEffort.LOW),
        ("medium", ReasoningControl.ON, ReasoningEffort.MEDIUM),
        ("high", ReasoningControl.ON, ReasoningEffort.HIGH),
        ("xhigh", ReasoningControl.ON, ReasoningEffort.XHIGH),
    ],
)
def test_responses_reasoning_effort_becomes_canonical_intent(
    effort: str,
    control: ReasoningControl,
    expected_effort: ReasoningEffort | None,
) -> None:
    request = _ingest(
        {
            "model": "nvidia_nim/test-model",
            "input": "Hello",
            "reasoning": {"effort": effort},
        }
    )

    assert request.reasoning.control is control
    assert request.reasoning.effort is expected_effort


def test_responses_messages_tools_and_results_preserve_semantics() -> None:
    request = _ingest(
        {
            "model": "deepseek/deepseek-chat",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Developer rules"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Use the tool"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "echo",
                    "arguments": '{"value":"FCC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "FCC",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "echo",
                    "description": "Echo a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": "echo"},
        }
    )

    assert request.items == (
        InstructionItem(
            text="Developer rules",
            origin=InstructionOrigin.DEVELOPER,
            placement=InstructionPlacement.TRANSCRIPT,
            turn_id="turn_0",
        ),
        MessageItem(
            turn_id="turn_1",
            role=MessageRole.USER,
            content=(TextContent("Use the tool"),),
        ),
        ToolCallItem(
            turn_id="turn_2",
            call_id="call_1",
            kind=ToolCallKind.FUNCTION,
            name="echo",
            input={"value": "FCC"},
        ),
        ToolResultItem(
            turn_id="turn_3",
            call_id="call_1",
            content="FCC",
        ),
    )
    assert request.tools == (
        FunctionTool(
            name="echo",
            description="Echo a value",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            strict=True,
        ),
    )
    assert request.tool_choice is not None
    assert request.tool_choice.mode is ToolChoiceMode.SPECIFIC
    assert request.tool_choice.name == "echo"


def test_none_choice_retains_definitions_but_disables_tool_calls() -> None:
    request = _ingest(
        {
            "model": "model",
            "input": "Reply without tools",
            "tools": [
                {
                    "type": "function",
                    "name": "echo",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": "none",
        }
    )

    assert [tool.name for tool in request.tools] == ["echo"]
    assert request.tool_choice is not None
    assert request.tool_choice.mode is ToolChoiceMode.NONE


def test_passive_responses_controls_are_accepted_only_at_the_settled_values() -> None:
    request = _ingest(
        {
            "model": "model",
            "input": "Hello",
            "stream": True,
            "store": False,
            "previous_response_id": None,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "session-1",
            "reasoning": {"summary": "auto"},
        }
    )

    assert request.model == "model"
    assert request.reasoning.control is ReasoningControl.DEFAULT


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"stream": False}, "stream=false"),
        ({"store": True}, "store=true"),
        ({"previous_response_id": "resp_1"}, "previous_response_id"),
        ({"include": []}, "include must be exactly"),
        (
            {
                "include": [
                    "reasoning.encrypted_content",
                    "reasoning.encrypted_content",
                ]
            },
            "include must be exactly",
        ),
        (
            {"include": ["reasoning.encrypted_content", "message.output_text"]},
            "include must be exactly",
        ),
        ({"include": ["reasoning.encrypted_contents"]}, "include must be exactly"),
        ({"prompt_cache_key": ""}, "prompt_cache_key"),
        ({"prompt_cache_key": "   "}, "prompt_cache_key"),
        ({"prompt_cache_key": 42}, "prompt_cache_key"),
        ({"reasoning": {"summary": "concise"}}, "reasoning.summary"),
        ({"reasoning": {"summary": []}}, "reasoning.summary"),
        ({"provider_extension": True}, "request.provider_extension"),
    ],
)
def test_responses_passive_field_near_misses_are_rejected(
    override: dict[str, object],
    match: str,
) -> None:
    payload: dict[str, object] = {"model": "model", "input": "Hello"}
    payload.update(override)

    with pytest.raises(ResponsesConversionError, match=match):
        _ingest(payload)


def test_namespace_and_custom_tool_identity_remain_structured() -> None:
    request = _ingest(
        {
            "model": "model",
            "input": "Use shell",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__shell",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "exec",
                            "description": "Run shell text",
                            "format": {"type": "text"},
                        }
                    ],
                }
            ],
            "tool_choice": {
                "type": "custom",
                "namespace": "mcp__shell",
                "custom": {"name": "exec"},
            },
        }
    )

    assert request.tools == (
        CustomTool(
            name="exec",
            description="Run shell text",
            format=request.tools[0].format,
            namespace="mcp__shell",
        ),
    )
    assert isinstance(request.tools[0], CustomTool)
    assert request.tools[0].format.type is CustomToolFormatType.TEXT
    assert request.tool_choice is not None
    assert request.tool_choice.kind is ToolCallKind.CUSTOM
    assert request.tool_choice.name == "exec"
    assert request.tool_choice.namespace == "mcp__shell"


def test_passive_hosted_tools_are_removed_but_executable_tools_remain() -> None:
    request = _ingest(
        {
            "model": "model",
            "input": "Hello",
            "tools": [
                {"type": "web_search", "external_web_access": True},
                {"type": "image_generation", "output_format": "png"},
                {"type": "tool_search"},
                {
                    "type": "function",
                    "name": "echo",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        }
    )

    assert [tool.name for tool in request.tools] == ["echo"]


@pytest.mark.parametrize(
    ("raw_choice", "expected_mode"),
    [
        (None, None),
        ("auto", ToolChoiceMode.AUTO),
        ("none", ToolChoiceMode.NONE),
        ("required", ToolChoiceMode.REQUIRED),
        ({"type": "function", "name": "echo"}, ToolChoiceMode.SPECIFIC),
    ],
)
def test_ambient_and_executable_tool_combinations_preserve_client_choice(
    raw_choice: object,
    expected_mode: ToolChoiceMode | None,
) -> None:
    payload: dict[str, object] = {
        "model": "model",
        "input": "Hello",
        "tools": [
            {"type": "web_search"},
            {"type": "image_generation"},
            {"type": "tool_search"},
            {
                "type": "function",
                "name": "echo",
                "parameters": {"type": "object"},
            },
        ],
    }
    if raw_choice is not None:
        payload["tool_choice"] = raw_choice

    request = _ingest(payload)

    assert [tool.name for tool in request.tools] == ["echo"]
    assert (
        request.tool_choice.mode if request.tool_choice is not None else None
    ) is expected_mode


@pytest.mark.parametrize(
    ("raw_choice", "expected_mode"),
    [
        (None, None),
        ("auto", ToolChoiceMode.AUTO),
        ("none", ToolChoiceMode.NONE),
    ],
)
def test_ambient_only_tools_are_safe_no_ops_when_not_required(
    raw_choice: object,
    expected_mode: ToolChoiceMode | None,
) -> None:
    payload: dict[str, object] = {
        "model": "model",
        "input": "Hello",
        "tools": [{"type": "web_search"}, {"type": "tool_search"}],
    }
    if raw_choice is not None:
        payload["tool_choice"] = raw_choice

    request = _ingest(payload)

    assert request.tools == ()
    assert (
        request.tool_choice.mode if request.tool_choice is not None else None
    ) is expected_mode


def test_prior_custom_call_preserves_kind_namespace_and_raw_input() -> None:
    request = _ingest(
        {
            "model": "model",
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "namespace": "mcp__shell",
                    "name": "exec",
                    "input": "printf FCC",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "FCC",
                },
            ],
        }
    )

    call, result = request.items
    assert isinstance(call, ToolCallItem)
    assert call.kind is ToolCallKind.CUSTOM
    assert call.namespace == "mcp__shell"
    assert call.input == "printf FCC"
    assert isinstance(result, ToolResultItem)
    assert result.call_id == call.call_id


def test_reasoning_stays_attached_to_the_following_tool_turn() -> None:
    request = _ingest(
        {
            "model": "model",
            "input": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Need the result."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "echo",
                    "arguments": '{"value":"FCC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "FCC",
                },
            ],
        }
    )

    reasoning, call, result = request.items
    assert isinstance(reasoning, ReasoningItem)
    assert isinstance(call, ToolCallItem)
    assert reasoning.turn_id == call.turn_id
    assert reasoning.reasoning == "Need the result."
    assert isinstance(result, ToolResultItem)


def test_responses_encrypted_reasoning_carrier_restores_scoped_artifacts() -> None:
    artifact = ReplayArtifact(
        origin=ReplayArtifactOrigin.OPENAI_COMPATIBLE,
        kind=ReplayArtifactKind.REASONING_DETAILS,
        attachment=ReplayAttachment.REASONING,
        scope=ReplayCompatibilityScope("provider:model"),
        payload=[{"type": "reasoning.text", "text": "opaque"}],
    )
    request = _ingest(
        {
            "model": "model",
            "input": [
                {
                    "type": "reasoning",
                    "encrypted_content": encode_replay_envelope((artifact,)),
                },
                {"role": "assistant", "content": "Done"},
                {"role": "user", "content": "Continue"},
            ],
        }
    )

    reasoning = next(item for item in request.items if isinstance(item, ReasoningItem))
    assert reasoning.artifacts == (artifact,)


def test_malformed_prior_call_quarantines_only_it_and_matching_output() -> None:
    request = _ingest(
        {
            "model": "model",
            "input": [
                {"role": "user", "content": "hello"},
                {
                    "type": "function_call",
                    "call_id": "call_bad",
                    "name": "echo",
                    "arguments": "{",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_bad",
                    "output": "stale output",
                },
                {
                    "type": "function_call",
                    "call_id": "call_good",
                    "name": "echo",
                    "arguments": '{"value":"ok"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_good",
                    "output": "ok",
                },
                {"role": "user", "content": "continue"},
            ],
        }
    )

    assert all(
        not isinstance(item, ToolCallItem | ToolResultItem)
        or item.call_id != "call_bad"
        for item in request.items
    )
    assert any(
        isinstance(item, ToolCallItem) and item.call_id == "call_good"
        for item in request.items
    )
    assert any(
        isinstance(item, ToolResultItem) and item.call_id == "call_good"
        for item in request.items
    )


def test_malformed_only_call_has_no_routable_message() -> None:
    with pytest.raises(ResponsesConversionError, match="must contain a message"):
        _ingest(
            {
                "model": "model",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_bad",
                        "name": "echo",
                        "arguments": "{",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_bad",
                        "output": "stale output",
                    },
                ],
            }
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                "model": "model",
                "input": "Hello",
                "tools": [{"type": "web_search_preview"}],
            },
            "tools\\[0\\]\\.type",
        ),
        (
            {
                "model": "model",
                "input": "Hello",
                "tools": [{"type": "web_search"}],
                "tool_choice": "required",
            },
            "needs at least one executable",
        ),
        (
            {
                "model": "model",
                "input": "Hello",
                "tool_choice": {"type": "web_search"},
            },
            "cannot be selected explicitly",
        ),
    ],
)
def test_unsupported_or_active_hosted_tool_semantics_are_rejected(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ResponsesConversionError, match=match):
        _ingest(payload)


@pytest.mark.parametrize("part_type", ["input_image", "input_file", "computer_call"])
def test_unsupported_responses_input_types_are_rejected_explicitly(
    part_type: str,
) -> None:
    with pytest.raises(ResponsesConversionError, match=part_type):
        _ingest(
            {
                "model": "model",
                "input": [{"type": part_type, "image_url": "https://example.invalid"}],
            }
        )
