from free_claude_code.application.chat import (
    ChatCompaction,
    ChatGeneration,
    ChatReasoning,
    ChatSegment,
    ChatSession,
    ChatTranscript,
    ChatTurn,
    ChatValidationError,
    GenerationStatus,
    SegmentKind,
)
from free_claude_code.application.chat.context import ChatContextBuilder
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import RequestRuntimeLease
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import ContentBlockText, ContentBlockThinking
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningControl, ReasoningEffort


class FakeRuntime:
    def __init__(
        self,
        *,
        configured: ProviderModelInfo,
        discovered: tuple[ProviderModelInfo, ...] = (),
    ) -> None:
        self.settings = Settings().model_copy(
            update={
                "model": "groq/model",
                "model_fallbacks": None,
            }
        )
        self.configured = configured
        self.discovered = discovered

    async def acquire(
        self, *, include_model_infos: bool = False
    ) -> RequestRuntimeLease:
        del include_model_infos
        raise AssertionError("Context construction must not acquire a runtime lease")

    def current_settings(self) -> Settings:
        return self.settings

    def cached_model_info(
        self, provider_id: str, model_id: str
    ) -> ProviderModelInfo | None:
        if provider_id == "groq" and model_id == "model":
            return self.configured
        return None

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        return self.discovered


def _transcript(
    *,
    reasoning: ChatReasoning = ChatReasoning.HIGH,
    compaction: ChatCompaction | None = None,
) -> ChatTranscript:
    session = ChatSession(
        id="session",
        title="Chat",
        model="groq/model",
        reasoning=reasoning,
        revision=1,
        created_at=1,
        updated_at=1,
    )
    generation = ChatGeneration(
        id="generation",
        status=GenerationStatus.COMPLETED,
        requested_model=session.model,
        actual_model=session.model,
        reasoning=reasoning,
        effective_output_limit=4096,
        stop_reason="end_turn",
        error_code=None,
        error_message=None,
        started_at=1,
        finished_at=2,
        segments=(
            ChatSegment(0, SegmentKind.THINKING, "private-looking but exposed"),
            ChatSegment(1, SegmentKind.TEXT, "answer"),
        ),
    )
    return ChatTranscript(
        session=session,
        turns=(
            ChatTurn(
                id="turn",
                session_id=session.id,
                operation_id="operation",
                sequence=1,
                user_text="question",
                created_at=1,
                generation=generation,
            ),
        ),
        compaction=compaction,
    )


def test_models_merge_configured_and_discovered_capabilities():
    runtime = FakeRuntime(
        configured=ProviderModelInfo(
            "model",
            supports_thinking=True,
            input_modalities=frozenset(
                {ModelInputModality.TEXT, ModelInputModality.IMAGE}
            ),
            context_window_tokens=100_000,
            max_output_tokens=20_000,
        ),
        discovered=(
            ProviderModelInfo(
                "groq/discovered",
                supports_thinking=False,
                context_window_tokens=32_000,
            ),
        ),
    )

    options = ChatContextBuilder(runtime).models()

    assert [option.model_ref for option in options] == [
        "groq/discovered",
        "groq/model",
    ]
    assert options[1].input_modalities == frozenset(
        {ModelInputModality.TEXT, ModelInputModality.IMAGE}
    )


def test_explicit_model_metadata_snapshot_does_not_follow_live_cache():
    runtime = FakeRuntime(
        configured=ProviderModelInfo(
            "model",
            context_window_tokens=100_000,
            max_output_tokens=20_000,
        )
    )
    builder = ChatContextBuilder(
        runtime,
        model_infos=(
            ProviderModelInfo(
                "groq/model",
                context_window_tokens=32_000,
                max_output_tokens=4_096,
            ),
        ),
    )

    prepared = builder.prepare(
        _transcript(reasoning=ChatReasoning.OFF),
        system_prompt="",
        draft="hello",
    )

    assert prepared.model.context_window_tokens == 32_000
    assert prepared.routed.request.max_tokens == 4_096


def test_prepare_builds_ordered_messages_and_authoritative_reasoning():
    runtime = FakeRuntime(
        configured=ProviderModelInfo(
            "model",
            supports_thinking=True,
            context_window_tokens=100_000,
            max_output_tokens=20_000,
        )
    )
    prepared = ChatContextBuilder(runtime).prepare(
        _transcript(),
        system_prompt="custom",
        draft="follow up",
    )

    request = prepared.routed.request
    assert request.system == "custom"
    assert [message.role for message in request.messages] == [
        "user",
        "assistant",
        "user",
    ]
    assistant = request.messages[1].content
    assert isinstance(assistant, list)
    assert isinstance(assistant[0], ContentBlockThinking)
    assert isinstance(assistant[1], ContentBlockText)
    assert prepared.routed.reasoning.control is ReasoningControl.ON
    assert prepared.routed.reasoning.effort is ReasoningEffort.HIGH
    assert request.max_tokens == 18_432


def test_known_unsupported_reasoning_is_rejected_but_unknown_is_allowed():
    unsupported = ChatContextBuilder(
        FakeRuntime(configured=ProviderModelInfo("model", supports_thinking=False))
    )
    try:
        unsupported.prepare(_transcript(), system_prompt="")
    except ChatValidationError as exc:
        assert "Thinking" in str(exc) or "thinking" in str(exc)
    else:
        raise AssertionError("Known unsupported reasoning was accepted")

    unknown = ChatContextBuilder(
        FakeRuntime(configured=ProviderModelInfo("model", supports_thinking=None))
    )
    assert unknown.prepare(
        _transcript(), system_prompt=""
    ).routed.reasoning.requests_reasoning


def test_compaction_disables_reasoning_for_known_non_reasoning_model():
    routed = ChatContextBuilder(
        FakeRuntime(
            configured=ProviderModelInfo(
                "model",
                supports_thinking=False,
                max_output_tokens=20_000,
            )
        )
    ).prepare_summary(
        model_ref="groq/model",
        source="Summarize this conversation.",
        output_tokens=4_096,
    )

    assert routed.reasoning.control is ReasoningControl.OFF
    assert routed.request.max_tokens == 4_096


def test_unknown_context_disables_auto_compaction_without_inventing_a_limit():
    builder = ChatContextBuilder(
        FakeRuntime(configured=ProviderModelInfo("model", context_window_tokens=None))
    )
    estimate = builder.prepare(_transcript(), system_prompt="").estimate

    assert estimate.context_window_tokens is None
    assert estimate.usage_ratio is None
    assert estimate.should_auto_compact is False


def test_known_context_reserves_input_when_output_cap_is_unknown():
    builder = ChatContextBuilder(
        FakeRuntime(
            configured=ProviderModelInfo(
                "model",
                context_window_tokens=16_384,
                max_output_tokens=None,
            )
        )
    )

    prepared = builder.prepare(
        _transcript(reasoning=ChatReasoning.OFF),
        system_prompt="",
        draft="hello",
    )

    assert prepared.routed.request.max_tokens == 15_360
    assert prepared.estimate.usable_input_tokens == 1_024
    assert prepared.estimate.usage_ratio == (
        prepared.estimate.estimated_input_tokens / 16_384
    )


def test_auto_compaction_uses_visible_context_ratio_with_fit_safety():
    transcript = _transcript(reasoning=ChatReasoning.OFF)
    transcript = ChatTranscript(
        session=transcript.session,
        turns=transcript.turns * 2,
        compaction=None,
    )
    builder = ChatContextBuilder(
        FakeRuntime(
            configured=ProviderModelInfo(
                "model",
                context_window_tokens=100_000,
                max_output_tokens=4_096,
            )
        )
    )

    below = builder.prepare(
        transcript,
        system_prompt="",
        draft="token " * 83_000,
    ).estimate
    above = builder.prepare(
        transcript,
        system_prompt="",
        draft="token " * 86_000,
    ).estimate

    assert below.usable_input_tokens is not None
    assert below.usage_ratio is not None and below.usage_ratio < 0.85
    assert below.estimated_input_tokens / below.usable_input_tokens > 0.85
    assert below.should_auto_compact is False
    assert above.usage_ratio is not None and above.usage_ratio > 0.85
    assert above.should_auto_compact is True

    tight = (
        ChatContextBuilder(
            FakeRuntime(
                configured=ProviderModelInfo(
                    "model",
                    context_window_tokens=40_000,
                    max_output_tokens=20_000,
                )
            )
        )
        .prepare(
            transcript,
            system_prompt="",
            draft="token " * 24_000,
        )
        .estimate
    )
    assert tight.usable_input_tokens is not None
    assert tight.usage_ratio is not None and tight.usage_ratio < 0.85
    assert tight.estimated_input_tokens > tight.usable_input_tokens
    assert tight.should_auto_compact is True


def test_existing_compaction_replaces_only_covered_context():
    runtime = FakeRuntime(
        configured=ProviderModelInfo("model", context_window_tokens=100_000)
    )
    transcript = _transcript(
        reasoning=ChatReasoning.OFF,
        compaction=ChatCompaction(
            session_id="session",
            covered_through_sequence=1,
            summary="durable summary",
            estimated_tokens=10,
            requested_model="groq/model",
            actual_model="groq/model",
            updated_at=2,
        ),
    )
    request = (
        ChatContextBuilder(runtime)
        .prepare(
            transcript,
            system_prompt="",
            draft="new message",
        )
        .routed.request
    )

    assert len(request.messages) == 2
    assert "durable summary" in str(request.messages[0].content)
    assert request.messages[1].content == "new message"
