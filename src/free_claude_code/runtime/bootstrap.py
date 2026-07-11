"""Single production composition root for the FCC server."""

import os
from pathlib import Path

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.config.logging_config import configure_logging
from free_claude_code.config.paths import server_log_path
from free_claude_code.config.settings import Settings
from free_claude_code.messaging.transcription import TranscriptionService
from free_claude_code.messaging.voice import Transcriber
from free_claude_code.providers.nvidia_nim.voice import NvidiaNimTranscriber

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
        transcriber=_create_transcriber(settings),
        restart_callback=restart_callback,
    )
    services = ApiServices(
        requests=provider_manager,
        admin=runtime,
        tasks=runtime,
    )
    return RuntimeASGIApp(create_app(services), runtime)


def _create_transcriber(settings: Settings) -> Transcriber | None:
    if not settings.voice_note_enabled:
        return None
    if settings.whisper_device == "nvidia_nim":
        return NvidiaNimTranscriber(
            model=settings.whisper_model,
            api_key=settings.nvidia_nim_api_key,
        )
    return TranscriptionService(
        model=settings.whisper_model,
        device=settings.whisper_device,
        huggingface_api_key=settings.huggingface_api_key,
    )
