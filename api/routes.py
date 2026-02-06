"""FastAPI route handlers."""

import logging
import uuid

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse

from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import MessagesResponse, TokenCountResponse, Usage
from .dependencies import get_provider, get_settings
from .request_utils import (
    is_quota_check_request,
    is_title_generation_request,
    is_prefix_detection_request,
    is_suggestion_mode_request,
    is_filepath_extraction_request,
    extract_command_prefix,
    extract_filepaths_from_command,
    get_token_count,
)
from config.settings import Settings
from providers.base import BaseProvider
from providers.exceptions import ProviderError
from providers.logging_utils import log_request_compact

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Routes
# =============================================================================


@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    provider: BaseProvider = Depends(get_provider),
    settings: Settings = Depends(get_settings),
):
    """Create a message (streaming or non-streaming)."""

    try:
        if settings.fast_prefix_detection:
            is_prefix_req, command = is_prefix_detection_request(request_data)
            if is_prefix_req:
                return MessagesResponse(
                    id=f"msg_{uuid.uuid4()}",
                    model=request_data.model,
                    content=[{"type": "text", "text": extract_command_prefix(command)}],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=100, output_tokens=5),
                )

        # Optimization: Mock network probe/quota requests
        if settings.enable_network_probe_mock and is_quota_check_request(request_data):
            logger.info("Optimization: Intercepted and mocked quota probe")
            return MessagesResponse(
                id=f"msg_{uuid.uuid4()}",
                model=request_data.model,
                role="assistant",
                content=[{"type": "text", "text": "Quota check passed."}],
                stop_reason="end_turn",
                usage=Usage(input_tokens=10, output_tokens=5),
            )

        # Optimization: Skip title generation requests
        if settings.enable_title_generation_skip and is_title_generation_request(
            request_data
        ):
            logger.info("Optimization: Skipped title generation request")
            return MessagesResponse(
                id=f"msg_{uuid.uuid4()}",
                model=request_data.model,
                role="assistant",
                content=[{"type": "text", "text": "Conversation"}],
                stop_reason="end_turn",
                usage=Usage(input_tokens=100, output_tokens=5),
            )

        # Optimization: Skip suggestion mode requests
        if settings.enable_suggestion_mode_skip and is_suggestion_mode_request(
            request_data
        ):
            logger.info("Optimization: Skipped suggestion mode request")
            return MessagesResponse(
                id=f"msg_{uuid.uuid4()}",
                model=request_data.model,
                role="assistant",
                content=[{"type": "text", "text": ""}],
                stop_reason="end_turn",
                usage=Usage(input_tokens=100, output_tokens=1),
            )

        # Optimization: Mock filepath extraction requests
        if settings.enable_filepath_extraction_mock:
            is_fp, cmd, output = is_filepath_extraction_request(request_data)
            if is_fp:
                filepaths = extract_filepaths_from_command(cmd, output)
                logger.info("Optimization: Mocked filepath extraction")
                return MessagesResponse(
                    id=f"msg_{uuid.uuid4()}",
                    model=request_data.model,
                    role="assistant",
                    content=[{"type": "text", "text": filepaths}],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=100, output_tokens=10),
                )

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        log_request_compact(logger, request_id, request_data)

        if request_data.stream:
            input_tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            return StreamingResponse(
                provider.stream_response(request_data, input_tokens=input_tokens),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            response_json = await provider.complete(request_data)
            return provider.convert_response(response_json, request_data)

    except ProviderError:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=str(e))


@router.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest):
    """Count tokens for a request."""
    try:
        return TokenCountResponse(
            input_tokens=get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root(settings: Settings = Depends(get_settings)):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": "nvidia_nim",
        "model": settings.model,
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/stop")
async def stop_cli(request: Request):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    return {"status": "stopped", "cancelled_count": count}
