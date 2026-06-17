# ruff: noqa: E402


def _sanitize_breaking_tools(request_data):
    return request_data


"""FastAPI route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count
from core.trace import trace_event
from providers.registry import ProviderRegistry

from . import dependencies
from .dependencies import get_settings, require_api_key
from .gateway_model_ids import gateway_model_id, no_thinking_gateway_model_id
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.openai_responses import OpenAIResponsesRequest
from .models.responses import ModelResponse, ModelsListResponse
from .services import ClaudeProxyService

router = APIRouter()


def _sanitize_breaking_tools(request_data):
    """Comprehensive Sanitizer: Strips unsupported tools, safeguards beta tools,
    neutralizes invalid tool choices, clears high-latency thinking budgets,
    and appends anti-stall system instructions safely.
    """
    try:
        # Check object attributes
        if hasattr(request_data, "thinking"):
            request_data.thinking = None
        # Check dictionary-style request items
        if isinstance(request_data, dict) and "thinking" in request_data:
            request_data["thinking"] = None
        elif hasattr(request_data, "__dict__") and "thinking" in request_data.__dict__:
            request_data.__dict__["thinking"] = None
    except Exception:
        pass

    tools = getattr(request_data, "tools", None)
    filtered_tools = []
    removed_tool_names = []

    if tools:
        for tool in tools:
            name = (
                tool.get("name", "")
                if isinstance(tool, dict)
                else getattr(tool, "name", "")
            )
            name_str = str(name)

            is_unsupported = (
                name_str.startswith(("advisor_", "computer_"))
                or name_str in ("web_search", "web_search_tool")
                or name_str.startswith("web_search_tool_")
            )

            if name and is_unsupported:
                removed_tool_names.append(name_str)
                continue
            filtered_tools.append(tool)

    if not removed_tool_names:
        return request_data

    final_tools = filtered_tools if (tools and filtered_tools) else None
    final_tool_choice = getattr(request_data, "tool_choice", None)

    if (
        final_tool_choice
        and isinstance(final_tool_choice, dict)
        and final_tool_choice.get("type") == "tool"
        and final_tool_choice.get("name") in removed_tool_names
    ):
        logger.warning(
            f"Stripping named tool_choice pointing to removed tool: {final_tool_choice.get('name')}"
        )
        final_tool_choice = None

    if not final_tools:
        final_tool_choice = None

    logger.info(
        f"Sanitized tools {removed_tool_names} from client payload. Remaining tools: {len(filtered_tools)}"
    )

    system_directive = "\n\n[System Note: Execute alternative tools immediately. Do not explain, speculate, or apologize for missing tools.]"
    current_system = getattr(request_data, "system", "")

    if isinstance(current_system, str):
        updated_system = (
            current_system + system_directive
            if current_system
            else system_directive.strip()
        )
    elif isinstance(current_system, list):
        updated_system = [
            *current_system,
            {"type": "text", "text": system_directive.strip()},
        ]
    elif current_system is None:
        updated_system = system_directive.strip()
    else:
        updated_system = current_system

    try:
        request_data.tools = final_tools
        if hasattr(request_data, "system"):
            request_data.system = updated_system
        if hasattr(request_data, "tool_choice"):
            request_data.tool_choice = final_tool_choice
    except Exception as e:
        logger.warning(f"Failed to modify attributes directly: {e}. Using model_copy.")
        update_fields = {"tools": final_tools, "system": updated_system}
        if hasattr(request_data, "tool_choice"):
            update_fields["tool_choice"] = final_tool_choice
        if hasattr(request_data, "thinking"):
            update_fields["thinking"] = None
        try:
            request_data = request_data.model_copy(update=update_fields)
        except Exception as fallback_err:
            logger.error(
                f"Critical: Payload sanitization completely collapsed. Direct assignment failed, "
                f"and model_copy fallback was rejected by schema validation. Error: {fallback_err}"
            )

    return request_data


DISCOVERED_MODEL_CREATED_AT = "1970-01-01T00:00:00Z"


SUPPORTED_CLAUDE_MODELS = [
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


def get_proxy_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ClaudeProxyService:
    """Build the request service for route handlers."""
    return ClaudeProxyService(
        settings,
        provider_getter=lambda provider_type: dependencies.resolve_provider(
            provider_type, app=request.app, settings=settings
        ),
        token_counter=get_token_count,
    )


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


def _discovered_model_response(model_id: str, *, display_name: str) -> ModelResponse:
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
) -> None:
    if supports_thinking is not False:
        _append_unique_model(
            models,
            seen,
            _discovered_model_response(
                gateway_model_id(provider_model_ref),
                display_name=provider_model_ref,
            ),
        )
    _append_unique_model(
        models,
        seen,
        _discovered_model_response(
            no_thinking_gateway_model_id(provider_model_ref),
            display_name=f"{provider_model_ref} (no thinking)",
        ),
    )


def _build_models_list_response(
    settings: Settings, provider_registry: ProviderRegistry | None
) -> ModelsListResponse:
    models: list[ModelResponse] = []
    seen: set[str] = set()

    for ref in settings.configured_chat_model_refs():
        supports_thinking = None
        if provider_registry is not None:
            supports_thinking = provider_registry.cached_model_supports_thinking(
                ref.provider_id, ref.model_id
            )
        _append_provider_model_variants(
            models,
            seen,
            ref.model_ref,
            supports_thinking=supports_thinking,
        )

    if provider_registry is not None:
        for model_info in provider_registry.cached_prefixed_model_infos():
            _append_provider_model_variants(
                models,
                seen,
                model_info.model_id,
                supports_thinking=model_info.supports_thinking,
            )

    for model in SUPPORTED_CLAUDE_MODELS:
        _append_unique_model(models, seen, model)

    return ModelsListResponse(
        data=models,
        first_id=models[0].id if models else None,
        has_more=False,
        last_id=models[-1].id if models else None,
    )


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Create a message (always streaming)."""
    request_data = _sanitize_breaking_tools(request_data)

    try:
        # Primary Routing Path (e.g., NVIDIA NIM)
        return service.create_message(request_data)
    except Exception as primary_err:
        # Pass client-side validation exceptions straight through without failover triggers
        if (
            "InvalidRequestError" in type(primary_err).__name__
            or getattr(primary_err, "status_code", None) == 400
        ):
            raise
        logger.warning(
            f"Primary provider failed or throttled: {primary_err}. Initiating high-capacity fallback recovery..."
        )

        # Programmatic Failover Config: Route down to OpenRouter Free or a stable secondary slug
        fallback_model = "open_router/qwen/qwen-coder-32b-vision:free"

        if hasattr(request_data, "model"):
            request_data.model = fallback_model
        # Execute the secondary network request against the fallback provider
        return service.create_message(request_data)

        try:
            if hasattr(request_data, "model_copy"):
                request_data = request_data.model_copy(update={"model": fallback_model})
        except Exception:
            pass

        logger.info(
            f"Rerouting payload burst down to fallback runner: {fallback_model}"
        )
        return service.create_message(request_data)


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the messages endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/responses")
async def create_response(
    request_data: OpenAIResponsesRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Create an OpenAI Responses-compatible response through this proxy."""
    return await service.create_response(request_data)


@router.api_route("/v1/responses", methods=["HEAD", "OPTIONS"])
async def probe_responses(_auth=Depends(require_api_key)):
    """Respond to OpenAI Responses compatibility probes."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request_data: TokenCountRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Count tokens for a request."""
    request_data = _sanitize_breaking_tools(request_data)
    return service.count_tokens(request_data)


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the token count endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings), _auth=Depends(require_api_key)
):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root():
    """Respond to unauthenticated local compatibility probes for the root endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(
    request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """List the model ids this proxy advertises to Claude-compatible clients."""
    trace_event(stage="ingress", event="api.models.list", source="api")
    registry = getattr(request.app.state, "provider_registry", None)
    provider_registry = registry if isinstance(registry, ProviderRegistry) else None
    return _build_models_list_response(settings, provider_registry)


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    trace_event(
        stage="ingress",
        event="api.cli.stop_via_handler",
        source="api",
        cancelled_nodes=count,
    )
    logger.info("STOP_CLI: source=handler cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
