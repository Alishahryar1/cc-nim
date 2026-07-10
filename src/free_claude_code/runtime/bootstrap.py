"""Single production composition root for the FCC server."""

import os
from pathlib import Path

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.config.logging_config import configure_logging
from free_claude_code.config.paths import server_log_path
from free_claude_code.config.settings import Settings

from .application import ApplicationRuntime, RestartCallback
from .asgi import RuntimeASGIApp
from .provider_manager import ProviderRuntimeManager


def build_asgi_app(
    settings: Settings,
    restart_callback: RestartCallback | None = None,
) -> RuntimeASGIApp:
    """Construct the complete server application and its resource owner."""
    log_path = Path(os.getenv("LOG_FILE", server_log_path()))
    configure_logging(log_path, verbose_third_party=settings.log_raw_api_payloads)
    provider_manager = ProviderRuntimeManager(settings)
    runtime = ApplicationRuntime(
        provider_manager,
        restart_callback=restart_callback,
    )
    services = ApiServices(
        requests=provider_manager,
        admin=runtime,
        sessions=runtime,
    )
    return RuntimeASGIApp(create_app(services), runtime)
