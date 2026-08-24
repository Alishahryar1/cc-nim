"""Protocol models live with the protocol logic that consumes them."""

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from free_claude_code.core.anthropic import (
    MessagesRequest as PublicMessagesRequest,
)
from free_claude_code.core.anthropic import (
    MessagesResponse,
    TokenCountResponse,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.inference import (
    FunctionTool,
    InferenceRequest,
    MessageItem,
    MessageRole,
    OpenAIChatExtension,
    ReasoningItem,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    TextContent,
    ToolCallItem,
    ToolCallKind,
    ToolResultItem,
    inference_request_snapshot,
)
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest as PublicOpenAIResponsesRequest,
)
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest


def test_anthropic_request_model_is_core_owned_and_permissive() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "provider-model",
            "messages": [{"role": "user", "content": "hello"}],
            "provider_extension": {"enabled": True},
        }
    )

    assert MessagesRequest.__module__ == "free_claude_code.core.anthropic.models"
    assert PublicMessagesRequest is MessagesRequest
    assert request.model_extra == {"provider_extension": {"enabled": True}}


def test_responses_request_model_is_core_owned_and_permissive() -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "provider-model",
            "input": "hello",
            "provider_extension": {"enabled": True},
        }
    )

    assert (
        OpenAIResponsesRequest.__module__
        == "free_claude_code.core.openai_responses.models"
    )
    assert PublicOpenAIResponsesRequest is OpenAIResponsesRequest
    assert request.model_extra == {"provider_extension": {"enabled": True}}


def test_anthropic_response_models_are_protocol_owned() -> None:
    assert MessagesResponse.__module__ == "free_claude_code.core.anthropic.models"
    assert TokenCountResponse.__module__ == "free_claude_code.core.anthropic.models"


def test_wire_request_models_do_not_carry_internal_route_state() -> None:
    internal_fields = {
        "original_model",
        "provider_model",
        "resolved_provider_model",
        "resolved_reasoning",
    }

    assert internal_fields.isdisjoint(MessagesRequest.model_fields)
    assert internal_fields.isdisjoint(OpenAIResponsesRequest.model_fields)


def test_canonical_request_owns_recursive_copies_and_is_frozen() -> None:
    arguments = {"nested": {"value": 1}}
    result = {"items": ["first"]}
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    metadata = {"trace": {"enabled": True}}
    extension = {"service": {"tier": "free"}}
    request = InferenceRequest(
        model="client-model",
        items=(
            MessageItem("turn_0", MessageRole.USER, (TextContent("Hello"),)),
            ToolCallItem(
                "turn_1",
                "call_1",
                ToolCallKind.FUNCTION,
                "lookup",
                arguments,
            ),
            ToolResultItem("turn_2", "call_1", result),
        ),
        tools=(FunctionTool("lookup", None, schema),),
        metadata=metadata,
        extensions=(OpenAIChatExtension(extension),),
    )

    arguments["nested"] = {"value": 2}
    result["items"] = ["changed"]
    schema["properties"] = {}
    metadata["trace"] = {"enabled": False}
    extension["service"] = {"tier": "paid"}

    call = request.items[1]
    tool_result = request.items[2]
    assert isinstance(call, ToolCallItem)
    assert call.input == {"nested": {"value": 1}}
    assert isinstance(tool_result, ToolResultItem)
    assert tool_result.content == {"items": ("first",)}
    tool = request.tools[0]
    assert isinstance(tool, FunctionTool)
    assert tool.input_schema["properties"] == {"value": {"type": "integer"}}
    assert request.metadata == {"trace": {"enabled": True}}
    assert request.openai_chat_extension is not None
    assert request.openai_chat_extension.extra_body == {"service": {"tier": "free"}}
    field_name = "model"
    with pytest.raises(FrozenInstanceError):
        setattr(request, field_name, "other-model")


def test_canonical_request_snapshot_contains_structure_but_no_client_payloads() -> None:
    request = InferenceRequest(
        model="public-model",
        items=(
            MessageItem(
                "turn_0",
                MessageRole.USER,
                (TextContent("private-prompt"),),
            ),
            ReasoningItem(
                "turn_1",
                "private-reasoning",
                artifacts=(
                    ReplayArtifact(
                        origin=ReplayArtifactOrigin.OPENAI,
                        kind=ReplayArtifactKind.ENCRYPTED_REASONING,
                        attachment=ReplayAttachment.REASONING,
                        payload="private-replay",
                    ),
                ),
            ),
            ToolCallItem(
                "turn_1",
                "call_1",
                ToolCallKind.FUNCTION,
                "lookup",
                {"secret": "private-tool-arguments"},
            ),
        ),
        tools=(FunctionTool("lookup", None, {"type": "object"}),),
        metadata={"secret": "private-metadata"},
        extensions=(OpenAIChatExtension({"secret": "private-extra-body"}),),
    )

    snapshot = inference_request_snapshot(request)
    rendered = repr(snapshot)

    assert snapshot["item_count"] == 3
    assert snapshot["message_count"] == 2
    assert snapshot["tool_count"] == 1
    for secret in (
        "private-prompt",
        "private-reasoning",
        "private-replay",
        "private-tool-arguments",
        "private-metadata",
        "private-extra-body",
    ):
        assert secret not in rendered


def test_protocol_facades_are_import_order_independent() -> None:
    import_orders = (
        (
            "free_claude_code.core.anthropic",
            "free_claude_code.core.openai_responses",
        ),
        (
            "free_claude_code.core.openai_responses",
            "free_claude_code.core.anthropic",
        ),
    )

    for modules in import_orders:
        script = "; ".join(f"import {module}" for module in modules)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
