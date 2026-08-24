import json

import pytest

from free_claude_code.core.inference import (
    Base64MediaSource,
    CacheControl,
    CustomTool,
    CustomToolFormat,
    CustomToolFormatType,
    DocumentContent,
    FileMediaSource,
    FunctionTool,
    ImageContent,
    InferenceRequest,
    InstructionItem,
    InstructionOrigin,
    InstructionPlacement,
    MessageItem,
    MessageRole,
    OpenAIChatExtension,
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
)
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.openai_compat import OpenAIToolNameCodec
from free_claude_code.providers.openai_responses.request_codec import (
    ResponsesRequestEncodingError,
    build_responses_request_body,
)

_SCOPE = ReplayCompatibilityScope("openai_responses:test-model")


def _body(
    request: InferenceRequest,
    *,
    scope: ReplayCompatibilityScope = _SCOPE,
    reasoning: ReasoningPolicy = ReasoningPolicy.provider_default(),
) -> dict[str, object]:
    return build_responses_request_body(
        request,
        provider_model="test-model",
        reasoning=reasoning,
        tool_names=OpenAIToolNameCodec.from_request(request),
        replay_scope=scope,
    )


def test_responses_request_codec_preserves_multiturn_canonical_semantics() -> None:
    request = InferenceRequest(
        model="gateway-model",
        max_output_tokens=4_096,
        items=(
            InstructionItem(
                text="System instructions",
                origin=InstructionOrigin.SYSTEM,
                placement=InstructionPlacement.TOP_LEVEL,
            ),
            ReasoningItem(
                turn_id="turn_0",
                reasoning="summary",
                artifacts=(
                    ReplayArtifact(
                        origin=ReplayArtifactOrigin.OPENAI,
                        kind=ReplayArtifactKind.ENCRYPTED_REASONING,
                        attachment=ReplayAttachment.REASONING,
                        payload="opaque",
                        scope=_SCOPE,
                    ),
                ),
            ),
            MessageItem(
                turn_id="turn_0",
                role=MessageRole.ASSISTANT,
                content=(TextContent("Calling a tool"),),
            ),
            ToolCallItem(
                turn_id="turn_0",
                call_id="call_1",
                kind=ToolCallKind.FUNCTION,
                name="lookup",
                input={"q": "value"},
            ),
            ToolResultItem(
                turn_id="turn_1",
                call_id="call_1",
                content={"answer": 42},
            ),
            MessageItem(
                turn_id="turn_1",
                role=MessageRole.USER,
                content=(
                    TextContent("Continue"),
                    ImageContent(
                        Base64MediaSource(
                            media_type="image/png",
                            data="aGVsbG8=",
                        )
                    ),
                ),
            ),
        ),
        tools=(
            FunctionTool(
                name="lookup",
                description="Look up a value",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            ),
        ),
        tool_choice=ToolChoice(
            ToolChoiceMode.SPECIFIC,
            kind=ToolCallKind.FUNCTION,
            name="lookup",
        ),
    )

    body = _body(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.XHIGH),
    )

    assert body["model"] == "test-model"
    assert body["instructions"] == "System instructions"
    assert body["max_output_tokens"] == 4_096
    assert body["stream"] is True
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}
    input_items = body["input"]
    assert isinstance(input_items, list)
    assert input_items[0] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "summary"}],
        "encrypted_content": "opaque",
    }
    assert input_items[1]["role"] == "assistant"
    assert input_items[2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": json.dumps(
            {"q": "value"}, ensure_ascii=False, separators=(",", ":")
        ),
    }
    assert input_items[3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"answer": 42}',
    }
    assert input_items[4]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,aGVsbG8=",
    }


def test_only_scope_matched_reasoning_artifact_is_replayed() -> None:
    request = InferenceRequest(
        model="gateway-model",
        items=(
            ReasoningItem(
                turn_id="turn_0",
                reasoning="summary",
                artifacts=(
                    ReplayArtifact(
                        origin=ReplayArtifactOrigin.OPENAI,
                        kind=ReplayArtifactKind.ENCRYPTED_REASONING,
                        attachment=ReplayAttachment.REASONING,
                        payload="opaque",
                        scope=_SCOPE,
                    ),
                ),
            ),
            MessageItem(
                turn_id="turn_1",
                role=MessageRole.USER,
                content=(TextContent("Continue"),),
            ),
        ),
    )

    matching = _body(request)
    incompatible = _body(
        request,
        scope=ReplayCompatibilityScope("openai_responses:other-model"),
    )

    matching_input = matching["input"]
    incompatible_input = incompatible["input"]
    assert isinstance(matching_input, list)
    assert isinstance(incompatible_input, list)
    assert matching_input[0]["encrypted_content"] == "opaque"
    assert "encrypted_content" not in incompatible_input[0]


def test_namespaced_and_custom_tool_identity_is_encoded_once() -> None:
    request = InferenceRequest(
        model="gateway-model",
        items=(
            MessageItem(
                turn_id="turn_0",
                role=MessageRole.USER,
                content=(TextContent("Use tools"),),
            ),
        ),
        tools=(
            FunctionTool(
                name="lookup",
                namespace="mcp__db",
                description=None,
                input_schema={"type": "object"},
            ),
            CustomTool(
                name="patch",
                namespace="repo",
                description="Apply a patch",
                format=CustomToolFormat(
                    CustomToolFormatType.GRAMMAR,
                    syntax="lark",
                    definition="start: /.+/",
                ),
            ),
        ),
        tool_choice=ToolChoice(
            ToolChoiceMode.SPECIFIC,
            kind=ToolCallKind.CUSTOM,
            name="patch",
            namespace="repo",
        ),
    )

    body = _body(request)

    tools = body["tools"]
    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools] == ["mcp__db__lookup", "repo__patch"]
    assert body["tool_choice"] == {"type": "custom", "name": "repo__patch"}


def test_auto_choice_is_materialized_only_when_tools_exist() -> None:
    message = MessageItem(
        turn_id="turn_0",
        role=MessageRole.USER,
        content=(TextContent("Hello"),),
    )
    without_tools = _body(InferenceRequest(model="model", items=(message,)))
    with_tools = _body(
        InferenceRequest(
            model="model",
            items=(message,),
            tools=(
                FunctionTool(
                    name="echo",
                    description=None,
                    input_schema={"type": "object"},
                ),
            ),
        )
    )

    assert "tool_choice" not in without_tools
    assert with_tools["tool_choice"] == "auto"


def test_resolved_reasoning_policy_is_the_only_upstream_reasoning_owner() -> None:
    request = InferenceRequest(
        model="model",
        items=(
            MessageItem(
                turn_id="turn_0",
                role=MessageRole.USER,
                content=(TextContent("Hello"),),
            ),
        ),
    )

    assert _body(request, reasoning=ReasoningPolicy.off())["reasoning"] == {
        "effort": "none"
    }
    assert _body(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
    )["reasoning"] == {"effort": "high", "summary": "auto"}


@pytest.mark.parametrize(
    ("canonical", "match"),
    [
        (
            InferenceRequest(
                model="model",
                items=(
                    MessageItem(
                        turn_id="turn_0",
                        role=MessageRole.USER,
                        content=(TextContent("Hello"),),
                    ),
                ),
                stop_sequences=("stop",),
            ),
            "stop_sequences",
        ),
        (
            InferenceRequest(
                model="model",
                items=(
                    MessageItem(
                        turn_id="turn_0",
                        role=MessageRole.USER,
                        content=(DocumentContent(FileMediaSource("file_1")),),
                    ),
                ),
            ),
            "document",
        ),
        (
            InferenceRequest(
                model="model",
                items=(
                    MessageItem(
                        turn_id="turn_0",
                        role=MessageRole.USER,
                        content=(TextContent("Hello", CacheControl()),),
                    ),
                ),
            ),
            "cache_control",
        ),
        (
            InferenceRequest(
                model="model",
                items=(
                    MessageItem(
                        turn_id="turn_0",
                        role=MessageRole.USER,
                        content=(TextContent("Hello"),),
                    ),
                ),
                extensions=(OpenAIChatExtension({"extension": True}),),
            ),
            "extensions",
        ),
    ],
)
def test_unrepresentable_canonical_semantics_are_rejected_before_io(
    canonical: InferenceRequest,
    match: str,
) -> None:
    with pytest.raises(ResponsesRequestEncodingError, match=match):
        _body(canonical)


def test_codec_requires_a_conversational_input_item() -> None:
    request = InferenceRequest(
        model="model",
        items=(
            InstructionItem(
                text="System only",
                origin=InstructionOrigin.SYSTEM,
                placement=InstructionPlacement.TOP_LEVEL,
            ),
        ),
    )

    with pytest.raises(ResponsesRequestEncodingError, match="conversational input"):
        _body(request)
