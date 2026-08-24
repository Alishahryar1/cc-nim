"""Contracts for canonical-to-OpenAI Chat request encoding."""

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic import messages_to_inference_request
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.inference import (
    CacheControl,
    CustomTool,
    CustomToolFormat,
    CustomToolFormatType,
    DocumentContent,
    FunctionTool,
    InferenceRequest,
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
    ToolChoice,
    ToolChoiceMode,
    ToolResultItem,
    UrlMediaSource,
)
from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_chat import (
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)
from free_claude_code.providers.openai_chat.request_codec import (
    OpenAIConversionError,
    ReasoningReplayMode,
    build_base_request_body,
    is_synthetic_openai_tool_turn_boundary,
    serialize_tool_result_content,
)
from free_claude_code.providers.openai_compat import OpenAIToolNameCodec


def _canonical(**overrides: object) -> InferenceRequest:
    payload: dict[str, object] = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return messages_to_inference_request(MessagesRequest.model_validate(payload))


def _body(
    request: InferenceRequest,
    *,
    provider_model: str = "upstream-model",
    replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
    replay_scope: ReplayCompatibilityScope | None = None,
    default_max_tokens: int | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {}
    body.update(
        build_base_request_body(
            request,
            provider_model=provider_model,
            tool_names=OpenAIToolNameCodec.from_request(request),
            replay_scope=replay_scope,
            default_max_tokens=default_max_tokens,
            reasoning_replay=replay,
        )
    )
    return body


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _objects(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return value


def test_top_level_and_inline_instructions_preserve_order() -> None:
    request = _canonical(
        system="Conversation-wide instructions",
        messages=[
            {"role": "user", "content": "First question"},
            {"role": "system", "content": "Instructions from this point"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ],
    )

    assert _body(request)["messages"] == [
        {"role": "system", "content": "Conversation-wide instructions"},
        {
            "role": "user",
            "content": "First question\n\nInstructions from this point",
        },
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]


def test_provider_model_is_explicit_and_request_is_immutable() -> None:
    request = _canonical(max_tokens=321, temperature=0.4, top_p=0.8)

    body = _body(request, provider_model="provider/selected")

    assert body["model"] == "provider/selected"
    assert body["max_tokens"] == 321
    assert body["temperature"] == 0.4
    assert body["top_p"] == 0.8
    assert request.model == "client-model"


def test_default_output_cap_only_applies_when_client_omits_it() -> None:
    assert (
        _body(_canonical(max_tokens=None), default_max_tokens=777)["max_tokens"] == 777
    )
    assert (
        _body(_canonical(max_tokens=123), default_max_tokens=777)["max_tokens"] == 123
    )


def test_chat_policy_must_explicitly_consume_or_reject_caller_extra_body() -> None:
    request = _canonical(extra_body={"provider_option": True})
    policy = OpenAIChatRequestPolicy(
        provider_name="TEST",
        reasoning_replay=ReasoningReplayMode.THINK_TAGS,
    )

    with pytest.raises(InvalidRequestError, match="does not support caller extra_body"):
        build_openai_chat_request_body(
            request,
            provider_model="upstream-model",
            reasoning=ReasoningPolicy.provider_default(),
            policy=policy,
            tool_names=OpenAIToolNameCodec.from_request(request),
            replay_scope=None,
        )


def test_user_text_and_image_parts_are_encoded_without_reordering() -> None:
    request = _canonical(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.invalid/image.png",
                        },
                    },
                    {"type": "text", "text": "carefully"},
                ],
            }
        ]
    )

    assert _body(request)["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.invalid/image.png"},
                },
                {"type": "text", "text": "carefully"},
            ],
        }
    ]


def test_document_content_is_rejected_instead_of_dropped() -> None:
    request = InferenceRequest(
        model="client-model",
        items=(
            MessageItem(
                "turn_0",
                MessageRole.USER,
                (DocumentContent(UrlMediaSource("https://example.invalid/a.pdf")),),
            ),
        ),
    )

    with pytest.raises(OpenAIConversionError, match="document content"):
        _body(request)


def test_assistant_media_is_rejected_at_ingress() -> None:
    wire = MessagesRequest.model_validate(
        {
            "model": "client-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.invalid/image.png",
                            },
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="assistant images"):
        messages_to_inference_request(wire)


@pytest.mark.parametrize(
    ("mode", "reasoning_field", "visible"),
    [
        (ReasoningReplayMode.THINK_TAGS, None, "<think>\nWork\n</think>\n\nDone"),
        (ReasoningReplayMode.REASONING_CONTENT, "reasoning_content", "Done"),
        (ReasoningReplayMode.REASONING, "reasoning", "Done"),
        (ReasoningReplayMode.DISABLED, None, "Done"),
    ],
)
def test_reasoning_replay_modes(
    mode: ReasoningReplayMode,
    reasoning_field: str | None,
    visible: str,
) -> None:
    request = _canonical(
        messages=[
            {"role": "user", "content": "Solve"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Work"},
                    {"type": "text", "text": "Done"},
                ],
            },
        ]
    )

    assistant = _objects(_body(request, replay=mode)["messages"])[1]

    assert assistant["content"] == visible
    if reasoning_field is None:
        assert "reasoning_content" not in assistant
        assert "reasoning" not in assistant
    else:
        assert assistant[reasoning_field] == "Work"


def test_reasoning_replay_artifacts_require_an_exact_route_scope() -> None:
    matching = ReplayCompatibilityScope("route-a")
    request = InferenceRequest(
        model="client-model",
        items=(
            ReasoningItem(
                turn_id="turn_0",
                reasoning="Work",
                artifacts=(
                    ReplayArtifact(
                        origin=ReplayArtifactOrigin.OPENAI_COMPATIBLE,
                        kind=ReplayArtifactKind.REASONING_DETAILS,
                        attachment=ReplayAttachment.REASONING,
                        payload=[{"type": "reasoning.text", "text": "opaque"}],
                        scope=matching,
                    ),
                ),
            ),
            MessageItem("turn_0", MessageRole.ASSISTANT, (TextContent("Done"),)),
        ),
    )

    matched = _objects(_body(request, replay_scope=matching)["messages"])[0]
    mismatched = _objects(
        _body(
            request,
            replay_scope=ReplayCompatibilityScope("route-b"),
        )["messages"]
    )[0]

    assert matched["reasoning_details"] == [
        {"type": "reasoning.text", "text": "opaque"}
    ]
    assert "reasoning_details" not in mismatched


def test_function_tools_preserve_schema_strictness_and_choice() -> None:
    request = _canonical(
        tools=[
            {
                "name": "lookup",
                "description": "Look up a value",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
                "strict": True,
            }
        ],
        tool_choice={"type": "tool", "name": "lookup"},
    )

    body = _body(request)

    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
                "strict": True,
            },
        }
    ]
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ToolChoiceMode.AUTO, "auto"),
        (ToolChoiceMode.NONE, "none"),
        (ToolChoiceMode.REQUIRED, "required"),
    ],
)
def test_nonspecific_tool_choice_modes(
    mode: ToolChoiceMode,
    expected: str,
) -> None:
    request = InferenceRequest(
        model="client-model",
        items=(MessageItem("turn_0", MessageRole.USER, (TextContent("Hi"),)),),
        tools=(FunctionTool("lookup", None, {"type": "object"}),),
        tool_choice=ToolChoice(mode),
    )

    assert _body(request)["tool_choice"] == expected


def test_custom_and_namespaced_tools_share_a_reversible_name_map() -> None:
    request = InferenceRequest(
        model="client-model",
        items=(MessageItem("turn_0", MessageRole.USER, (TextContent("Run"),)),),
        tools=(
            CustomTool(
                name="shell",
                description="Execute text",
                format=CustomToolFormat(CustomToolFormatType.TEXT),
                namespace="workspace",
            ),
        ),
        tool_choice=ToolChoice(
            ToolChoiceMode.SPECIFIC,
            kind=ToolCallKind.CUSTOM,
            name="shell",
            namespace="workspace",
        ),
    )

    codec = OpenAIToolNameCodec.from_request(request)
    body = build_base_request_body(
        request,
        provider_model="upstream-model",
        tool_names=codec,
        replay_scope=None,
    )
    wire_tool = _objects(body["tools"])[0]
    wire_function = _object(wire_tool["function"])
    wire_name = wire_function["name"]
    assert isinstance(wire_name, str)

    choice = _object(body["tool_choice"])
    assert _object(choice["function"])["name"] == wire_name
    identity = codec.decode_identity(wire_name)
    assert identity.kind is ToolCallKind.CUSTOM
    assert identity.name == "shell"
    assert identity.namespace == "workspace"


def test_tool_history_keeps_calls_before_results_and_preserves_error_content() -> None:
    request = InferenceRequest(
        model="client-model",
        items=(
            MessageItem("turn_0", MessageRole.USER, (TextContent("Use it"),)),
            ToolCallItem(
                turn_id="turn_1",
                call_id="call_1",
                kind=ToolCallKind.FUNCTION,
                name="lookup",
                input={"value": "x"},
            ),
            ToolResultItem(
                turn_id="turn_2",
                call_id="call_1",
                content={"error": "not found"},
                is_error=True,
            ),
            MessageItem("turn_3", MessageRole.USER, (TextContent("Continue"),)),
        ),
        tools=(FunctionTool("lookup", None, {"type": "object"}),),
    )

    messages = _objects(_body(request)["messages"])

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert _objects(messages[1]["tool_calls"])[0]["id"] == "call_1"
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"error": "not found"}',
    }
    assert is_synthetic_openai_tool_turn_boundary(messages[3])


def test_incomplete_tool_history_is_rejected() -> None:
    request = InferenceRequest(
        model="client-model",
        items=(
            ToolCallItem(
                turn_id="turn_0",
                call_id="missing",
                kind=ToolCallKind.FUNCTION,
                name="lookup",
                input={},
            ),
        ),
        tools=(FunctionTool("lookup", None, {"type": "object"}),),
    )

    with pytest.raises(OpenAIConversionError, match="missing tool results"):
        _body(request)


def test_tool_replay_artifacts_require_an_exact_route_scope() -> None:
    matching = ReplayCompatibilityScope("route-a")
    request = InferenceRequest(
        model="client-model",
        items=(
            ToolCallItem(
                turn_id="turn_0",
                call_id="call_1",
                kind=ToolCallKind.FUNCTION,
                name="lookup",
                input={},
                artifacts=(
                    ReplayArtifact(
                        origin=ReplayArtifactOrigin.GOOGLE,
                        kind=ReplayArtifactKind.THOUGHT_SIGNATURE,
                        attachment=ReplayAttachment.TOOL_CALL,
                        payload="signature",
                        scope=matching,
                    ),
                ),
            ),
            ToolResultItem("turn_1", "call_1", "done"),
        ),
        tools=(FunctionTool("lookup", None, {"type": "object"}),),
    )

    matched_message = _objects(_body(request, replay_scope=matching)["messages"])[0]
    matched = _objects(matched_message["tool_calls"])[0]
    mismatched_message = _objects(
        _body(
            request,
            replay_scope=ReplayCompatibilityScope("route-b"),
        )["messages"]
    )[0]
    mismatched = _objects(mismatched_message["tool_calls"])[0]

    assert matched["extra_content"] == {"google": {"thought_signature": "signature"}}
    assert "extra_content" not in mismatched


def test_cache_control_is_preserved_only_at_supported_locations() -> None:
    request = InferenceRequest(
        model="client-model",
        items=(
            InstructionItem(
                "System",
                InstructionOrigin.SYSTEM,
                InstructionPlacement.TOP_LEVEL,
                cache_control=CacheControl(),
            ),
            MessageItem(
                "turn_0",
                MessageRole.USER,
                (TextContent("Hello", cache_control=CacheControl()),),
            ),
        ),
        tools=(
            FunctionTool(
                "lookup",
                None,
                {"type": "object"},
                cache_control=CacheControl(),
            ),
        ),
    )

    body = _body(request)

    messages = _objects(body["messages"])
    first_content = _objects(messages[0]["content"])
    second_content = _objects(messages[1]["content"])
    tools = _objects(body["tools"])
    assert first_content[0]["cache_control"] == {"type": "ephemeral"}
    assert second_content[0]["cache_control"] == {"type": "ephemeral"}
    assert tools[0]["cache_control"] == {"type": "ephemeral"}


def test_inline_instruction_after_tool_result_gets_a_synthetic_boundary() -> None:
    request = _canonical(
        messages=[
            {"role": "user", "content": "Use it"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "lookup",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "done",
                    }
                ],
            },
            {"role": "system", "content": "New instructions"},
        ],
        tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
    )

    messages = _objects(_body(request)["messages"])

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert is_synthetic_openai_tool_turn_boundary(messages[3])


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, ""),
        ("plain", "plain"),
        ({"value": 1}, '{"value": 1}'),
        (
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "a\nb",
        ),
    ],
)
def test_tool_result_serialization(content: JsonValue, expected: str) -> None:
    assert serialize_tool_result_content(content) == expected
