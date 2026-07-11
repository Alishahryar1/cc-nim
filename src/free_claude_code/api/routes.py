"""FastAPI route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from loguru import logger

from free_claude_code.application.ports import ProviderResolver, RequestRuntimeLease
from free_claude_code.config.model_refs import parse_provider_type
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    MessagesRequest,
    TokenCountRequest,
    get_token_count,
)
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.trace import trace_event

from .dependencies import (
    get_services,
    get_settings,
    require_api_key,
    resolve_provider,
)
from .handlers import MessagesHandler, ResponsesHandler, TokenCountHandler
from .model_catalog import ModelsListResponse, build_models_list_response
from .ports import ApiServices
from .request_ids import get_request_id
from .response_streams import bind_response_lifetime

router = APIRouter()


def _provider_resolver(lease: RequestRuntimeLease) -> ProviderResolver:
    return lambda provider_type: resolve_provider(provider_type, lease=lease)


async def _create_messages_response(
    services: ApiServices,
    request_data: MessagesRequest,
    *,
    request_id: str,
) -> object:
    lease = await services.requests.acquire()
    try:
        handler = MessagesHandler(
            lease.settings,
            provider_resolver=_provider_resolver(lease),
            token_counter=get_token_count,
            generation_id=lease.generation_id,
        )
        response = await handler.create(request_data, request_id=request_id)
    except BaseException:
        await lease.release()
        raise
    return await bind_response_lifetime(response, lease.release)


async def _create_responses_response(
    services: ApiServices,
    request_data: OpenAIResponsesRequest,
    *,
    request_id: str,
) -> object:
    lease = await services.requests.acquire()
    try:
        handler = ResponsesHandler(
            lease.settings,
            provider_resolver=_provider_resolver(lease),
            generation_id=lease.generation_id,
        )
        response = await handler.create(request_data, request_id=request_id)
    except BaseException:
        await lease.release()
        raise
    return await bind_response_lifetime(response, lease.release)


def _probe_response(allow: str) -> Response:
    return Response(status_code=204, headers={"Allow": allow})


@router.post("/v1/messages")
async def create_message(
    request: Request,
    request_data: MessagesRequest,
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_api_key),
):
    """Create a message (streaming by default; stream=false gets aggregated JSON)."""
    return await _create_messages_response(
        services,
        request_data,
        request_id=get_request_id(request),
    )


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/responses")
async def create_response(
    request: Request,
    request_data: OpenAIResponsesRequest,
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_api_key),
):
    """Create an OpenAI Responses-compatible response through this proxy."""
    return await _create_responses_response(
        services,
        request_data,
        request_id=get_request_id(request),
    )


@router.api_route("/v1/responses", methods=["HEAD", "OPTIONS"])
async def probe_responses(_auth=Depends(require_api_key)):
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    request_data: TokenCountRequest,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """Count tokens for a request."""
    handler = TokenCountHandler(settings, token_counter=get_token_count)
    return handler.count(request_data, request_id=get_request_id(request))


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    return {
        "status": "ok",
        "provider": parse_provider_type(settings.model),
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root():
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(
    services: ApiServices = Depends(get_services),
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """List the model ids this proxy advertises to compatible clients."""
    trace_event(stage="ingress", event="free_claude_code.api.models.list", source="api")
    return build_models_list_response(settings, services.requests)


@router.post("/stop")
async def stop_cli(
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_api_key),
):
    """Stop all CLI sessions and pending tasks."""
    result = await services.tasks.stop_all()
    if result is None:
        raise HTTPException(status_code=503, detail="Messaging system not initialized")
    if result.source is not None:
        logger.info("STOP_CLI: source={} cancelled_count=N/A", result.source)
        return {"status": "stopped", "source": result.source}

    count = result.cancelled_count or 0
    trace_event(
        stage="ingress",
        event="free_claude_code.api.cli.stop_via_messaging_workflow",
        source="api",
        cancelled_nodes=count,
    )
    logger.info("STOP_CLI: source=messaging_workflow cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
