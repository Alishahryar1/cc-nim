"""Model routing for Claude-compatible requests."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from config.provider_ids import SUPPORTED_PROVIDER_IDS
from config.settings import Settings

from .gateway_model_ids import decode_gateway_model_id
from .models.anthropic import MessagesRequest, TokenCountRequest
from .performance import performance_tracker


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    original_model: str
    provider_id: str
    provider_model: str
    provider_model_ref: str
    thinking_enabled: bool


@dataclass(frozen=True, slots=True)
class RoutedMessagesRequest:
    request: MessagesRequest
    resolved: ResolvedModel


@dataclass(frozen=True, slots=True)
class RoutedTokenCountRequest:
    request: TokenCountRequest
    resolved: ResolvedModel


class ModelRouter:
    """Resolve incoming Claude model names to configured provider/model pairs."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def resolve_candidates(self, claude_model_name: str) -> list[ResolvedModel]:
        """Resolve a Claude model name to one or more prioritized candidate models."""
        (
            direct_provider_id,
            direct_provider_model,
            force_thinking_enabled,
        ) = self._direct_provider_model(claude_model_name)

        if direct_provider_id is not None and direct_provider_model is not None:
            thinking_enabled = (
                force_thinking_enabled
                if force_thinking_enabled is not None
                else self._settings.resolve_thinking(direct_provider_model)
            )
            logger.debug(
                "MODEL DIRECT: '{}' -> provider='{}' model='{}' thinking={}",
                claude_model_name,
                direct_provider_id,
                direct_provider_model,
                thinking_enabled,
            )
            return [
                ResolvedModel(
                    original_model=claude_model_name,
                    provider_id=direct_provider_id,
                    provider_model=direct_provider_model,
                    provider_model_ref=claude_model_name,
                    thinking_enabled=thinking_enabled,
                )
            ]

        candidates: list[ResolvedModel] = []
        thinking_enabled = self._settings.resolve_thinking(claude_model_name)

        provider_refs = list(self._settings.resolve_models(claude_model_name))

        # Dynamic Ranking: Sort candidates by score (reliability / (latency + ttft))
        if len(provider_refs) > 1:
            def _score(ref: str) -> float:
                p_id = Settings.parse_provider_type(ref)
                m = performance_tracker.get_metrics(p_id)
                if m.success_count == 0 and m.failure_count == 0:
                    return 1000.0 # Default high score for unused providers

                # Weigh TTFT heavily for "snappiness"
                latency_weight = max(m.avg_latency + (m.avg_ttft * 2.0), 0.001)
                reliability = 1.0 - m.error_rate
                return reliability / latency_weight

            provider_refs.sort(key=_score, reverse=True)

        for provider_model_ref in provider_refs:
            provider_id = Settings.parse_provider_type(provider_model_ref)
            provider_model = Settings.parse_model_name(provider_model_ref)
            if provider_model != claude_model_name:
                logger.debug(
                    "MODEL MAPPING: '{}' -> '{}'", claude_model_name, provider_model
                )
            candidates.append(
                ResolvedModel(
                    original_model=claude_model_name,
                    provider_id=provider_id,
                    provider_model=provider_model,
                    provider_model_ref=provider_model_ref,
                    thinking_enabled=thinking_enabled,
                )
            )

        if not candidates:
            # Fallback to single resolve if split somehow failed
            return [self.resolve(claude_model_name)]

        return candidates

    def resolve(self, claude_model_name: str) -> ResolvedModel:
        return self.resolve_candidates(claude_model_name)[0]

    def _direct_provider_model(
        self, model_name: str
    ) -> tuple[str | None, str | None, bool | None]:
        decoded = decode_gateway_model_id(model_name)
        if decoded is not None:
            if decoded.provider_id not in SUPPORTED_PROVIDER_IDS:
                return None, None, None
            return (
                decoded.provider_id,
                decoded.provider_model,
                decoded.force_thinking_enabled,
            )

        provider_id, separator, provider_model = model_name.partition("/")
        if not separator:
            return None, None, None
        if provider_id not in SUPPORTED_PROVIDER_IDS:
            return None, None, None
        if not provider_model:
            return None, None, None
        return provider_id, provider_model, None

    def resolve_messages_request(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        """Return an internal routed request context."""
        resolved = self.resolve(request.model)
        routed = request.model_copy(deep=True)
        routed.model = resolved.provider_model
        return RoutedMessagesRequest(request=routed, resolved=resolved)

    def resolve_token_count_request(
        self, request: TokenCountRequest
    ) -> RoutedTokenCountRequest:
        """Return an internal token-count request context."""
        resolved = self.resolve(request.model)
        routed = request.model_copy(
            update={"model": resolved.provider_model}, deep=True
        )
        return RoutedTokenCountRequest(request=routed, resolved=resolved)
