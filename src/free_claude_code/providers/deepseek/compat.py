"""DeepSeek-owned OpenAI Chat compatibility policy."""

from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.inference import (
    DocumentContent,
    ImageContent,
    InferenceRequest,
    MessageItem,
    ReasoningItem,
    ToolCallItem,
    ToolChoiceMode,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.openai_chat import (
    OpenAIChatRequestPolicy,
    ReasoningReplayMode,
    build_openai_chat_request_body,
)
from free_claude_code.providers.openai_compat import (
    OpenAIToolNameCodec,
    openai_replay_scope,
)

DEEPSEEK_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="DEEPSEEK",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
)


def build_deepseek_request_body(
    request: InferenceRequest,
    *,
    provider_model: str,
    reasoning: ReasoningPolicy,
) -> JsonObject:
    """Build one DeepSeek request without rewriting canonical history."""

    _validate_media(request, provider_model=provider_model)
    effective_reasoning = reasoning
    if (
        reasoning.control is not ReasoningControl.OFF
        and _has_tool_history(request)
        and not _all_tool_calls_have_replayable_reasoning(request)
    ):
        logger.debug(
            "DEEPSEEK_REQUEST: disabling thinking for a tool follow-up without "
            "replayable reasoning model={} items={}",
            provider_model,
            len(request.items),
        )
        effective_reasoning = ReasoningPolicy.off()

    body = build_openai_chat_request_body(
        request,
        provider_model=provider_model,
        reasoning=effective_reasoning,
        policy=DEEPSEEK_REQUEST_POLICY,
        tool_names=OpenAIToolNameCodec.from_request(request),
        replay_scope=openai_replay_scope(
            "DEEPSEEK",
            provider_model,
            replay_format="chat-completions",
        ),
        postprocessors=(_apply_deepseek_chat_extras,),
    )
    if (
        request.tool_choice is not None
        and request.tool_choice.mode is ToolChoiceMode.SPECIFIC
    ):
        logger.debug(
            "DEEPSEEK_REQUEST: downgrading unsupported specific tool choice to auto"
        )
        body["tool_choice"] = "auto"
    body.setdefault("max_tokens", ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS)
    return body


def _validate_media(request: InferenceRequest, *, provider_model: str) -> None:
    vision_capable = "vision" in provider_model.rsplit("/", 1)[-1].lower()
    for item in request.items:
        if not isinstance(item, MessageItem):
            continue
        for content in item.content:
            if isinstance(content, DocumentContent):
                raise InvalidRequestError(
                    "DeepSeek Chat does not support document content."
                )
            if isinstance(content, ImageContent) and not vision_capable:
                raise InvalidRequestError(
                    "The selected DeepSeek model does not support image content."
                )


def _has_tool_history(request: InferenceRequest) -> bool:
    return any(isinstance(item, ToolCallItem) for item in request.items)


def _all_tool_calls_have_replayable_reasoning(request: InferenceRequest) -> bool:
    reasoning_turns = {
        item.turn_id for item in request.items if isinstance(item, ReasoningItem)
    }
    tool_turns = {
        item.turn_id for item in request.items if isinstance(item, ToolCallItem)
    }
    return bool(tool_turns) and tool_turns <= reasoning_turns


def _apply_deepseek_chat_extras(
    body: JsonObject,
    _request: InferenceRequest,
    policy: ReasoningPolicy,
) -> None:
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and not message.get("tool_calls")
            ):
                message.pop("reasoning_content", None)

    raw_extra = body.setdefault("extra_body", {})
    if not isinstance(raw_extra, dict):
        raise InvalidRequestError("DeepSeek extra_body must be an object.")
    if policy.control is ReasoningControl.OFF:
        raw_extra["thinking"] = {"type": "disabled"}
        return
    if policy.effort in {ReasoningEffort.XHIGH, ReasoningEffort.MAX}:
        body["reasoning_effort"] = "max"
    elif policy.effort is not None:
        body["reasoning_effort"] = "high"
    elif policy.requests_reasoning:
        raw_extra["thinking"] = {"type": "enabled"}
