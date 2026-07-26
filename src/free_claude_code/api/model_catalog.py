"""Model-list response construction for Claude-compatible clients."""

from typing import Literal

from pydantic import BaseModel

from free_claude_code.application.ports import RequestRuntimePort
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import (
    decode_gateway_model_id,
    gateway_model_id,
    no_thinking_gateway_model_id,
)

DISCOVERED_MODEL_CREATED_AT = "1970-01-01T00:00:00Z"


def _normalize_model_ref(model_ref: str) -> str:
    """Normalize a model reference by stripping gateway ID prefixes.
    Returns the underlying provider/model format for gateway IDs, or the original string otherwise.
    """
    decoded = decode_gateway_model_id(model_ref)
    if decoded:
        return f"{decoded.provider_id}/{decoded.provider_model}"
    return model_ref


def _is_free_model(model_ref: str, settings: Settings) -> bool:
    """Return True if the model is considered free.
    Free if:
    - model_ref ends with ":free", "-free", or "/free"
    - model_ref is in the user-provided free models list. (with gateway ID normalization)
    """
    if (
        model_ref.endswith(":free")
        or model_ref.endswith("-free")
        or model_ref.endswith("/free")
    ):
        return True
    normalized_model_ref = _normalize_model_ref(model_ref)
    for free_model in settings.free_models_list:
        if normalized_model_ref == _normalize_model_ref(free_model):
            return True
    return False


class ModelResponse(BaseModel):
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "free-claude-code"
    created_at: str
    display_name: str
    id: str
    type: Literal["model"] = "model"


class ModelsListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelResponse]
    first_id: str | None
    has_more: bool
    last_id: str | None


SUPPORTED_CLAUDE_MODELS = [
    ModelResponse(
        id="claude-fable-5",
        display_name="Claude Fable 5",
        created_at="2026-06-09T00:00:00Z",
    ),
    ModelResponse(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        created_at="2024-02-29T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        created_at="2024-10-22T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        created_at="2024-03-07T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        created_at="2024-10-22T00:00:00Z",
    ),
]


def build_models_list_response(
    settings: Settings, runtime: RequestRuntimePort
) -> ModelsListResponse:
    """Return configured, cached, and compatibility model ids."""
    models: list[ModelResponse] = []
    free_models: list[ModelResponse] = []
    seen: set[str] = set()

    # Determine whether to show free models first
    show_free_models_first = settings.show_free_models_first
    # Determine whether to only show free models
    only_show_free_models = settings.only_show_free_models

    for ref in configured_chat_model_refs(settings):
        supports_thinking = runtime.cached_model_supports_thinking(
            ref.provider_id, ref.model_id
        )
        is_free = _is_free_model(ref.model_ref, settings)
        if only_show_free_models and not is_free:
            continue
        _append_provider_model_variants(
            free_models if show_free_models_first and is_free else models,
            seen,
            ref.model_ref,
            supports_thinking=supports_thinking,
            is_free=is_free,
        )

    for model_info in runtime.cached_prefixed_model_infos():
        is_free = _is_free_model(model_info.model_id, settings)
        if only_show_free_models and not is_free:
            continue
        _append_provider_model_variants(
            free_models if show_free_models_first and is_free else models,
            seen,
            model_info.model_id,
            supports_thinking=model_info.supports_thinking,
            is_free=is_free,
        )

    for model in SUPPORTED_CLAUDE_MODELS:
        is_free = _is_free_model(model.id, settings)
        if only_show_free_models and not is_free:
            continue
        claude_model = _discovered_model_response(
            model_id=model.id,
            display_name=model.display_name,
            is_free=is_free,
            supports_thinking=True,
        )
        claude_model.created_at = model.created_at
        _append_unique_model(
            free_models if show_free_models_first and is_free else models,
            seen,
            claude_model,
        )

    combined_models = free_models + models

    return ModelsListResponse(
        data=combined_models,
        first_id=combined_models[0].id if combined_models else None,
        has_more=False,
        last_id=combined_models[-1].id if combined_models else None,
    )


def _discovered_model_response(
    model_id: str,
    *,
    display_name: str,
    is_free: bool = False,
    supports_thinking: bool | None = None,
) -> ModelResponse:
    if supports_thinking is False:
        display_name = f"{display_name} (no thinking)"
    if is_free:
        display_name = f"{display_name} (free)"
    return ModelResponse(
        id=model_id,
        display_name=display_name,
        created_at=DISCOVERED_MODEL_CREATED_AT,
    )


def _append_unique_model(
    models: list[ModelResponse], seen: set[str], model: ModelResponse
) -> None:
    if model.id in seen:
        return
    seen.add(model.id)
    models.append(model)


def _append_provider_model_variants(
    models: list[ModelResponse],
    seen: set[str],
    provider_model_ref: str,
    *,
    supports_thinking: bool | None = None,
    is_free: bool = False,
) -> None:
    if supports_thinking is not False:
        _append_unique_model(
            models,
            seen,
            _discovered_model_response(
                gateway_model_id(provider_model_ref),
                display_name=provider_model_ref,
                is_free=is_free,
                supports_thinking=True,
            ),
        )
    _append_unique_model(
        models,
        seen,
        _discovered_model_response(
            no_thinking_gateway_model_id(provider_model_ref),
            display_name=provider_model_ref,
            is_free=is_free,
            supports_thinking=False,
        ),
    )
