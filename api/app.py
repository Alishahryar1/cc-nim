"""FastAPI application factory and configuration."""

import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from starlette.types import Receive, Scope, Send

from config.logging_config import configure_logging
from config.settings import get_settings
from core.rate_limit import StrictSlidingWindowLimiter
from core.trace import extract_claude_session_id_from_headers, trace_event
from providers.exceptions import ProviderError

from .admin_routes import STATIC_DIR
from .admin_routes import router as admin_router
from .routes import router
from .runtime import AppRuntime, startup_failure_message
from .services import _log_unexpected_service_exception
from .validation_log import summarize_request_validation_body


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    runtime = AppRuntime.for_app(app, settings=get_settings())
    await runtime.startup()

    yield

    await runtime.shutdown()


class GracefulLifespanApp:
    """ASGI wrapper that reports startup failures without Starlette tracebacks."""

    def __init__(self, app: FastAPI):
        self.app = app

    def __getattr__(self, name: str) -> Any:
        return getattr(self.app, name)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return
        await self._lifespan(receive, send)

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        settings = get_settings()
        runtime = AppRuntime.for_app(self.app, settings=settings)
        startup_complete = False
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await runtime.startup()
                except Exception as exc:
                    await send(
                        {
                            "type": "lifespan.startup.failed",
                            "message": startup_failure_message(settings, exc),
                        }
                    )
                    return
                startup_complete = True
                await send({"type": "lifespan.startup.complete"})
                continue

            if message["type"] == "lifespan.shutdown":
                if startup_complete:
                    try:
                        await runtime.shutdown()
                    except Exception as exc:
                        logger.error("Shutdown failed: exc_type={}", type(exc).__name__)
                        await send({"type": "lifespan.shutdown.failed", "message": ""})
                        return
                await send({"type": "lifespan.shutdown.complete"})
                return


def create_app(*, lifespan_enabled: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(
        settings.log_file, verbose_third_party=settings.log_raw_api_payloads
    )

    app_kwargs: dict[str, Any] = {
        "title": "Claude Code Proxy",
        "version": "2.0.0",
    }
    if lifespan_enabled:
        app_kwargs["lifespan"] = lifespan
    app = FastAPI(**app_kwargs)

    cors_origins_list = [
        origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins_list,
        allow_credentials=cors_origins_list != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    trusted_hosts_list = [
        host.strip() for host in settings.trusted_hosts.split(",") if host.strip()
    ]
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=trusted_hosts_list,
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Pre-calculated security headers (Optimization: ⚡ 1-10)
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "X-Download-Options": "noopen",
        "X-Permitted-Cross-Domain-Policies": "none",
        "X-DNS-Prefetch-Control": "off",
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Permissions-Policy": "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self';"
        ),
    }

    # Global rate limiters (per-instance)

    # Global rate limiters (per-instance)
    # Admin UI: 60 requests per minute
    admin_limiter = StrictSlidingWindowLimiter(60, 60.0)
    # Main API: 120 requests per minute
    api_limiter = StrictSlidingWindowLimiter(120, 60.0)

    @app.middleware("http")
    async def unified_middleware(request: Request, call_next):
        """Unified middleware for rate limiting, security, and tracing (⚡ Bolt Optimization).

        Consolidating multiple middlewares into one reduces TaskGroup overhead and avoids
        Starlette's 'No response returned' RuntimeError in complex middleware stacks.
        """
        claude_sid = extract_claude_session_id_from_headers(request.headers)

        async def handle_request():
            # 1. Rate Limiting
            if request.url.path.startswith("/admin/api"):
                async with admin_limiter:
                    return await call_next(request)
            if request.url.path.startswith("/v1"):
                async with api_limiter:
                    return await call_next(request)
            return await call_next(request)

        with logger.contextualize(
            http_method=request.method,
            http_path=request.url.path,
            claude_session_id=claude_sid,
        ):
            try:
                response = await handle_request()
            except Exception as e:
                # Ensure we always return a response even if inner handlers crash
                _log_unexpected_service_exception(
                    settings, e, context="MIDDLEWARE_ERROR"
                )
                response = JSONResponse(
                    status_code=500,
                    content={
                        "type": "error",
                        "error": {
                            "message": "Internal Server Error",
                            "type": "api_error",
                        },
                    },
                )

            # 2. Security Headers
            response.headers.update(SECURITY_HEADERS)
            return response

    # Register routes
    app.include_router(admin_router)
    app.include_router(router)

    @app.get("/.well-known/security.txt", include_in_schema=False)
    @app.get("/security.txt", include_in_schema=False)
    async def security_txt():
        """Serve RFC 9116 security.txt."""
        path = STATIC_DIR / "security.txt"
        return FileResponse(path, media_type="text/plain")

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Log request shape for 422 debugging without content values."""
        body: Any
        try:
            body = await request.json()
        except Exception as e:
            body = {"_json_error": type(e).__name__}

        message_summary, tool_names = summarize_request_validation_body(body)

        trace_event(
            stage="ingress",
            event="server.request.validation_failed",
            source="api",
            path=request.url.path,
            query=dict(request.query_params),
            error_locs=[list(error.get("loc", ())) for error in exc.errors()],
            error_types=[str(error.get("type", "")) for error in exc.errors()],
            message_summary=message_summary,
            tool_names=tool_names,
        )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        """Handle provider-specific errors and return Anthropic format."""
        err_settings = get_settings()
        if err_settings.log_api_error_tracebacks:
            logger.error(
                "Provider Error: error_type={} status_code={} message={}",
                exc.error_type,
                exc.status_code,
                exc.message,
            )
        else:
            logger.error(
                "Provider Error: error_type={} status_code={}",
                exc.error_type,
                exc.status_code,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_anthropic_format(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle general errors and return Anthropic format."""
        settings = get_settings()
        if settings.log_api_error_tracebacks:
            logger.error("General Error: {}", exc)
            logger.error(traceback.format_exc())
        else:
            logger.error(
                "General Error: path={} method={} exc_type={}",
                request.url.path,
                request.method,
                type(exc).__name__,
            )
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "An unexpected error occurred.",
                },
            },
        )

    return app


def create_asgi_app() -> GracefulLifespanApp:
    """Create the server ASGI app with graceful lifespan failure reporting."""
    return GracefulLifespanApp(create_app(lifespan_enabled=False))
