"""Portable request construction and context accounting for Chat Sessions."""

from dataclasses import dataclass

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import RequestRuntimePort
from free_claude_code.application.routing import ModelRouter, RoutedMessagesRequest
from free_claude_code.config.model_refs import (
    configured_chat_model_refs,
    split_provider_model_ref,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    ContentBlockText,
    ContentBlockThinking,
    Message,
    MessagesRequest,
    get_token_count,
)
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy

from .models import (
    ChatContextEstimate,
    ChatModelOption,
    ChatReasoning,
    ChatTranscript,
    ChatTurn,
    ChatValidationError,
    SegmentKind,
)

_VISIBLE_ANSWER_TOKENS = 16_384
_MINIMUM_VISIBLE_ANSWER_TOKENS = 1_024
_MINIMUM_INPUT_TOKENS = 1_024
_AUTO_COMPACT_RATIO = 0.85
_COMPACT_TARGET_RATIO = 0.60
_SUMMARY_SYSTEM_PROMPT = """Summarize the earlier conversation for another assistant.
Preserve user facts and preferences, decisions, requirements, unresolved work,
important explanations, and source titles or URLs. Preserve the meaning of any
described documents or images. Do not include hidden reasoning, chain-of-thought,
provider errors, or conversational filler. Be concise, factual, and structured.
"""


@dataclass(frozen=True, slots=True)
class PreparedChatRequest:
    routed: RoutedMessagesRequest
    estimate: ChatContextEstimate
    model: ChatModelOption


class ChatContextBuilder:
    """Build exactly one provider-portable request from durable Chat state."""

    def __init__(
        self,
        runtime: RequestRuntimePort,
        *,
        settings: Settings | None = None,
        model_infos: tuple[ProviderModelInfo, ...] | None = None,
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._model_infos = model_infos

    def _current_settings(self) -> Settings:
        return self._settings or self._runtime.current_settings()

    def models(self) -> tuple[ChatModelOption, ...]:
        """Return configured and discovered direct models with rich metadata."""

        settings = self._current_settings()
        snapshot = self._model_infos
        discovered = (
            self._runtime.cached_prefixed_model_infos()
            if snapshot is None
            else snapshot
        )
        snapshot_by_ref = (
            {info.model_id: info for info in snapshot} if snapshot is not None else None
        )
        options: dict[str, ChatModelOption] = {}
        for ref in configured_chat_model_refs(settings):
            info = (
                self._runtime.cached_model_info(ref.provider_id, ref.model_id)
                if snapshot_by_ref is None
                else snapshot_by_ref.get(ref.model_ref)
            )
            options[ref.model_ref] = ChatModelOption(
                model_ref=ref.model_ref,
                provider_id=ref.provider_id,
                model_id=ref.model_id,
                supports_reasoning=(
                    info.supports_thinking if info is not None else None
                ),
                input_modalities=(info.input_modalities if info is not None else None),
                context_window_tokens=(
                    info.context_window_tokens if info is not None else None
                ),
                max_output_tokens=(
                    info.max_output_tokens if info is not None else None
                ),
            )

        for info in discovered:
            try:
                provider_id, model_id = split_provider_model_ref(info.model_id)
            except ValueError:
                continue
            options[info.model_id] = ChatModelOption(
                model_ref=info.model_id,
                provider_id=provider_id,
                model_id=model_id,
                supports_reasoning=info.supports_thinking,
                input_modalities=info.input_modalities,
                context_window_tokens=info.context_window_tokens,
                max_output_tokens=info.max_output_tokens,
            )
        return tuple(
            sorted(options.values(), key=lambda item: item.model_ref.casefold())
        )

    def model(self, model_ref: str) -> ChatModelOption:
        option = next(
            (
                candidate
                for candidate in self.models()
                if candidate.model_ref == model_ref
            ),
            None,
        )
        if option is None:
            raise ChatValidationError(
                "The selected model is unavailable. Choose an available model."
            )
        return option

    def prepare(
        self,
        transcript: ChatTranscript,
        *,
        system_prompt: str,
        draft: str | None = None,
        exclude_generation_id: str | None = None,
    ) -> PreparedChatRequest:
        option = self.model(transcript.session.model)
        reasoning = transcript.session.reasoning.policy()
        self._validate_reasoning(option, transcript.session.reasoning)
        output_limit = self._completion_limit(option, reasoning)
        request = MessagesRequest(
            model=transcript.session.model,
            max_tokens=output_limit,
            messages=self._messages(
                transcript,
                draft=draft,
                exclude_generation_id=exclude_generation_id,
            ),
            system=system_prompt or None,
            stream=True,
        )
        estimate = self._estimate(
            request,
            option=option,
            output_limit=output_limit,
            can_compact=self.can_compact(transcript),
        )
        routed = ModelRouter(
            self._current_settings()
        ).resolve_messages_request_with_policy(request, reasoning=reasoning)
        return PreparedChatRequest(routed=routed, estimate=estimate, model=option)

    def prepare_summary(
        self,
        *,
        model_ref: str,
        source: str,
        output_tokens: int,
    ) -> RoutedMessagesRequest:
        option = self.model(model_ref)
        reasoning = (
            ReasoningPolicy.off()
            if option.supports_reasoning is False
            else ReasoningPolicy.on(effort=ReasoningEffort.LOW)
        )
        reasoning_tokens = reasoning.numeric_budget_tokens or 0
        total_output = output_tokens + reasoning_tokens
        if option.max_output_tokens is not None and option.max_output_tokens > 0:
            total_output = min(total_output, option.max_output_tokens)
        if total_output <= reasoning_tokens:
            raise ChatValidationError(
                "The selected model cannot reserve enough output for compaction."
            )
        request = MessagesRequest(
            model=model_ref,
            max_tokens=total_output,
            messages=[Message(role="user", content=source)],
            system=_SUMMARY_SYSTEM_PROMPT,
            stream=True,
        )
        return ModelRouter(
            self._current_settings()
        ).resolve_messages_request_with_policy(
            request,
            reasoning=reasoning,
        )

    def summary_output_tokens(self, option: ChatModelOption) -> int:
        if option.context_window_tokens is None:
            return 4_096
        return max(1_024, min(8_192, option.context_window_tokens // 20))

    def summary_source_fits(
        self,
        *,
        model_ref: str,
        source: str,
        output_tokens: int,
    ) -> bool:
        option = self.model(model_ref)
        if option.context_window_tokens is None:
            return True
        routed = self.prepare_summary(
            model_ref=model_ref,
            source=source,
            output_tokens=output_tokens,
        )
        estimated = get_token_count(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )
        return (
            estimated + (routed.request.max_tokens or 0) <= option.context_window_tokens
        )

    @staticmethod
    def can_compact(transcript: ChatTranscript) -> bool:
        covered = (
            transcript.compaction.covered_through_sequence
            if transcript.compaction is not None
            else 0
        )
        return sum(turn.sequence > covered for turn in transcript.turns) > 1

    @staticmethod
    def compaction_source(
        existing_summary: str | None,
        turns: tuple[ChatTurn, ...],
    ) -> str:
        sections: list[str] = []
        if existing_summary:
            sections.append(f"CURRENT SUMMARY\n{existing_summary}")
        for turn in turns:
            answer = "\n".join(
                segment.text
                for segment in turn.generation.segments
                if segment.kind is SegmentKind.TEXT and segment.text
            )
            section = f"USER\n{turn.user_text}"
            if answer:
                section = f"{section}\n\nASSISTANT\n{answer}"
            sections.append(section)
        return "\n\n--- EXCHANGE ---\n\n".join(sections)

    @staticmethod
    def _messages(
        transcript: ChatTranscript,
        *,
        draft: str | None,
        exclude_generation_id: str | None,
    ) -> list[Message]:
        messages: list[Message] = []
        covered = 0
        if transcript.compaction is not None:
            covered = transcript.compaction.covered_through_sequence
            messages.append(
                Message(
                    role="user",
                    content=(
                        "[Earlier conversation summary]\n"
                        f"{transcript.compaction.summary}\n"
                        "[End earlier conversation summary]"
                    ),
                )
            )

        for turn in transcript.turns:
            if turn.sequence <= covered:
                continue
            messages.append(Message(role="user", content=turn.user_text))
            generation = turn.generation
            if generation.id == exclude_generation_id:
                continue
            blocks: list[ContentBlockText | ContentBlockThinking] = []
            for segment in generation.segments:
                if not segment.text:
                    continue
                if segment.kind is SegmentKind.THINKING:
                    blocks.append(
                        ContentBlockThinking(
                            type="thinking",
                            thinking=segment.text,
                            signature=None,
                        )
                    )
                else:
                    blocks.append(ContentBlockText(type="text", text=segment.text))
            if blocks:
                messages.append(Message(role="assistant", content=blocks))

        if draft is not None:
            messages.append(Message(role="user", content=draft))
        return messages

    @staticmethod
    def _completion_limit(option: ChatModelOption, reasoning: ReasoningPolicy) -> int:
        reasoning_tokens = reasoning.numeric_budget_tokens or 0
        desired = _VISIBLE_ANSWER_TOKENS + reasoning_tokens
        if option.max_output_tokens is not None and option.max_output_tokens > 0:
            desired = min(desired, option.max_output_tokens)
        if (
            option.context_window_tokens is not None
            and option.context_window_tokens > 0
        ):
            desired = min(
                desired,
                option.context_window_tokens - _MINIMUM_INPUT_TOKENS,
            )
        if desired - reasoning_tokens < _MINIMUM_VISIBLE_ANSWER_TOKENS:
            raise ChatValidationError(
                "The selected model cannot fit this thinking level. Lower thinking."
            )
        return desired

    @staticmethod
    def _validate_reasoning(option: ChatModelOption, reasoning: ChatReasoning) -> None:
        if option.supports_reasoning is False and reasoning is not ChatReasoning.OFF:
            raise ChatValidationError(
                "This model does not support reasoning. Set thinking to Off."
            )

    @staticmethod
    def _estimate(
        request: MessagesRequest,
        *,
        option: ChatModelOption,
        output_limit: int,
        can_compact: bool,
    ) -> ChatContextEstimate:
        input_tokens = get_token_count(request.messages, request.system, request.tools)
        context_window = option.context_window_tokens
        if context_window is None or context_window <= 0:
            return ChatContextEstimate(
                estimated_input_tokens=input_tokens,
                completion_tokens=output_limit,
                context_window_tokens=None,
                usable_input_tokens=None,
                usage_ratio=None,
                should_auto_compact=False,
                can_compact=can_compact,
            )
        usable = context_window - output_limit
        if usable <= 0:
            raise ChatValidationError(
                "The selected model's output reserve leaves no input context."
            )
        ratio = input_tokens / context_window
        return ChatContextEstimate(
            estimated_input_tokens=input_tokens,
            completion_tokens=output_limit,
            context_window_tokens=context_window,
            usable_input_tokens=usable,
            usage_ratio=ratio,
            should_auto_compact=(
                can_compact and (ratio > _AUTO_COMPACT_RATIO or input_tokens > usable)
            ),
            can_compact=can_compact,
        )


def compaction_target_tokens(estimate: ChatContextEstimate) -> int | None:
    if estimate.usable_input_tokens is None:
        return None
    return int(estimate.usable_input_tokens * _COMPACT_TARGET_RATIO)
