"""Model routing for Claude-compatible requests."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from config.provider_ids import SUPPORTED_PROVIDER_IDS
from config.settings import Settings

from .content_scanner import has_vision_content
from .gateway_model_ids import decode_gateway_model_id
from .models.anthropic import MessagesRequest, TokenCountRequest


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
    fallback_request: MessagesRequest | None = None
    fallback_resolved: ResolvedModel | None = None


@dataclass(frozen=True, slots=True)
class RoutedTokenCountRequest:
    request: TokenCountRequest
    resolved: ResolvedModel


class ModelRouter:
    """Resolve incoming Claude model names to configured provider/model pairs."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def resolve(self, claude_model_name: str) -> ResolvedModel:
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
            return ResolvedModel(
                original_model=claude_model_name,
                provider_id=direct_provider_id,
                provider_model=direct_provider_model,
                provider_model_ref=claude_model_name,
                thinking_enabled=thinking_enabled,
            )

        provider_model_ref = self._settings.resolve_model(claude_model_name)
        thinking_enabled = self._settings.resolve_thinking(claude_model_name)
        provider_id = Settings.parse_provider_type(provider_model_ref)
        provider_model = Settings.parse_model_name(provider_model_ref)
        if provider_model != claude_model_name:
            logger.debug(
                "MODEL MAPPING: '{}' -> '{}'", claude_model_name, provider_model
            )
        return ResolvedModel(
            original_model=claude_model_name,
            provider_id=provider_id,
            provider_model=provider_model,
            provider_model_ref=provider_model_ref,
            thinking_enabled=thinking_enabled,
        )

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

    def resolve_vision_aware(
        self, request: MessagesRequest
    ) -> ResolvedModel:
        """Resolve model with vision-aware routing.

        When the request contains image/document blocks AND a ``VISION_MODEL``
        is configured, route to the vision-capable provider so DeepSeek
        (which lacks vision support) is bypassed for these requests.

        Falls back to normal ``resolve()`` when no vision content is detected
        or no ``VISION_MODEL`` is configured.
        """
        vision_model_ref = self._settings.model_vision

        if vision_model_ref is not None:
            # Serialize messages to dicts for scanning.  MessagesRequest
            # stores messages as Pydantic models; model_dump() gives us
            # plain dicts for the content scanner.
            raw_messages = [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in request.messages
            ]
            if has_vision_content(raw_messages):
                vision_thinking = (
                    self._settings.enable_vision_thinking
                    if self._settings.enable_vision_thinking is not None
                    else False
                )
                provider_id = Settings.parse_provider_type(vision_model_ref)
                provider_model = Settings.parse_model_name(vision_model_ref)
                logger.info(
                    "VISION_ROUTING: {} image/document blocks detected → "
                    "provider={} model={} thinking={}",
                    sum(
                        1
                        for m in raw_messages
                        if isinstance(m.get("content"), list)
                        and any(
                            b.get("type") in ("image", "document")
                            for b in m["content"]
                            if isinstance(b, dict)
                        )
                    ),
                    provider_id,
                    provider_model,
                    vision_thinking,
                )
                return ResolvedModel(
                    original_model=request.model,
                    provider_id=provider_id,
                    provider_model=provider_model,
                    provider_model_ref=vision_model_ref,
                    thinking_enabled=vision_thinking,
                )

        # No vision content or no VISION_MODEL configured → normal routing.
        return self.resolve(request.model)

    def _resolve_fallback(
        self, original_model: str, primary: ResolvedModel
    ) -> ResolvedModel | None:
        """Resolve the optional failover route for an incoming Claude model name."""
        ref = self._settings.resolve_model_fallback(original_model)
        if ref is None:
            return None
        provider_id = Settings.parse_provider_type(ref)
        provider_model = Settings.parse_model_name(ref)
        # Failing over to the same provider+model would be pointless.
        if (
            provider_id == primary.provider_id
            and provider_model == primary.provider_model
        ):
            return None
        return ResolvedModel(
            original_model=original_model,
            provider_id=provider_id,
            provider_model=provider_model,
            provider_model_ref=ref,
            thinking_enabled=self._settings.resolve_thinking(original_model),
        )

    def _resolve_vision_fallback(
        self, primary: ResolvedModel
    ) -> ResolvedModel | None:
        """Resolve the optional failover route for the vision path.

        Vision requests carry image/document blocks, so the fallback MUST itself
        be a vision-capable model. Configured solely via ``VISION_MODEL_FALLBACK``
        (the vision path has a dedicated target and does not use the tier
        ``MODEL_*_FALLBACK`` chain). Returns ``None`` when unset or when it would
        resolve to the same provider+model as the primary vision target (a no-op).
        """
        ref = self._settings.model_vision_fallback
        if ref is None:
            return None
        provider_id = Settings.parse_provider_type(ref)
        provider_model = Settings.parse_model_name(ref)
        if (
            provider_id == primary.provider_id
            and provider_model == primary.provider_model
        ):
            return None
        vision_thinking = (
            self._settings.enable_vision_thinking
            if self._settings.enable_vision_thinking is not None
            else False
        )
        return ResolvedModel(
            original_model=primary.original_model,
            provider_id=provider_id,
            provider_model=provider_model,
            provider_model_ref=ref,
            thinking_enabled=vision_thinking,
        )

    def resolve_messages_request(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        """Return an internal routed request context (vision-aware), with an
        optional failover route. Vision routes fail over to
        ``VISION_MODEL_FALLBACK`` (must be vision-capable); all other routes use
        the tier ``MODEL_*_FALLBACK`` overrides."""
        resolved = self.resolve_vision_aware(request)
        routed = request.model_copy(deep=True)
        routed.model = resolved.provider_model

        # Vision routing has its own dedicated target with a dedicated fallback
        # (VISION_MODEL_FALLBACK); it does not use the tier MODEL_*_FALLBACK chain.
        is_vision_route = (
            self._settings.model_vision is not None
            and resolved.provider_model_ref == self._settings.model_vision
        )
        fallback_resolved = (
            self._resolve_vision_fallback(resolved)
            if is_vision_route
            else self._resolve_fallback(request.model, resolved)
        )
        fallback_request: MessagesRequest | None = None
        if fallback_resolved is not None:
            fallback_request = request.model_copy(deep=True)
            fallback_request.model = fallback_resolved.provider_model

        return RoutedMessagesRequest(
            request=routed,
            resolved=resolved,
            fallback_request=fallback_request,
            fallback_resolved=fallback_resolved,
        )

    def resolve_token_count_request(
        self, request: TokenCountRequest
    ) -> RoutedTokenCountRequest:
        """Return an internal token-count request context."""
        resolved = self.resolve(request.model)
        routed = request.model_copy(
            update={"model": resolved.provider_model}, deep=True
        )
        return RoutedTokenCountRequest(request=routed, resolved=resolved)
