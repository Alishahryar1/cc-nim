"""Application runtime composition and lifecycle ownership."""

import asyncio
import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from loguru import logger

import cli.managed as cli_managed
import messaging.limiter as messaging_limiter
import messaging.session as messaging_session
import messaging.trees as messaging_trees
import messaging.workflow as messaging_workflow_module
from api.admin_urls import local_admin_url
from config.env_files import ANTHROPIC_AUTH_TOKEN_ENV, process_env_key_is_effective
from config.paths import default_claude_workspace_path
from config.settings import Settings, get_settings
from messaging.platforms import factory as messaging_platform_factory
from messaging.platforms.factory import MessagingPlatformOptions
from messaging.platforms.ports import MessagingPlatformComponents, MessagingRuntime
from providers.exceptions import ServiceUnavailableError
from providers.runtime import ProviderRuntime

_SHUTDOWN_TIMEOUT_S = 5.0


async def best_effort(
    name: str,
    awaitable: Any,
    timeout_s: float = _SHUTDOWN_TIMEOUT_S,
    *,
    log_verbose_errors: bool = False,
) -> None:
    """Run a shutdown step with timeout; never raise to callers."""
    try:
        await asyncio.wait_for(awaitable, timeout=timeout_s)
    except TimeoutError:
        logger.warning("Shutdown step timed out: {} ({}s)", name, timeout_s)
    except Exception as e:
        if log_verbose_errors:
            logger.warning(
                "Shutdown step failed: {}: {}: {}",
                name,
                type(e).__name__,
                e,
            )
        else:
            logger.warning(
                "Shutdown step failed: {}: exc_type={}",
                name,
                type(e).__name__,
            )


def warn_if_process_auth_token(settings: Settings) -> None:
    """Warn when server auth was implicitly inherited from the shell."""
    model_config = getattr(settings, "model_config", Settings.model_config)
    if process_env_key_is_effective(model_config, ANTHROPIC_AUTH_TOKEN_ENV):
        logger.warning(
            "ANTHROPIC_AUTH_TOKEN is set in the process environment but not in "
            "a configured .env file. The proxy will require that token. Add "
            "ANTHROPIC_AUTH_TOKEN= to .env to disable proxy auth, or set the "
            "same token in .env to make server auth explicit."
        )


def log_startup_failure(settings: Settings, exc: Exception) -> None:
    """Log startup failures without traceback noise unless verbose diagnostics are enabled."""
    message = startup_failure_message(settings, exc)
    logger.error("Startup failed:\n{}", message)


def startup_failure_message(settings: Settings, exc: Exception) -> str:
    """Return a concise startup failure message for logs and ASGI lifespan failure."""
    if isinstance(exc, ServiceUnavailableError):
        return exc.message.strip() or "Server startup failed."

    if settings.log_api_error_tracebacks:
        return f"{type(exc).__name__}: {exc}"

    return f"Server startup failed: exc_type={type(exc).__name__}"


@dataclass(slots=True)
class AppRuntime:
    """Own optional messaging, CLI, session, and provider runtime resources."""

    app: FastAPI
    settings: Settings
    _provider_runtime: ProviderRuntime | None = field(default=None, init=False)
    messaging_runtime: MessagingRuntime | None = None
    messaging_workflow: messaging_workflow_module.MessagingWorkflow | None = None
    cli_manager: cli_managed.ManagedClaudeSessionManager | None = None

    @classmethod
    def for_app(
        cls,
        app: FastAPI,
        settings: Settings | None = None,
    ) -> AppRuntime:
        return cls(app=app, settings=settings or get_settings())

    async def startup(self) -> None:
        logger.info("Starting Claude Code Proxy...")
        admin_url = local_admin_url(self.settings)
        self._provider_runtime = ProviderRuntime(self.settings)
        self.app.state.provider_runtime = self._provider_runtime
        try:
            warn_if_process_auth_token(self.settings)
            await self._validate_configured_models_best_effort()
            self._provider_runtime.start_model_list_refresh()
            await self._start_messaging_if_configured()
            self._publish_state()
            logging.getLogger("uvicorn.error").info(
                "Admin UI: %s (local-only)", admin_url
            )
        except Exception as exc:
            log_startup_failure(self.settings, exc)
            await best_effort(
                "provider_runtime.cleanup",
                self._provider_runtime.cleanup(),
                log_verbose_errors=self.settings.log_api_error_tracebacks,
            )
            raise

    async def _validate_configured_models_best_effort(self) -> None:
        """Warm validation status without blocking first-run/admin access."""
        if self._provider_runtime is None:
            return
        try:
            await self._provider_runtime.validate_configured_models()
        except ServiceUnavailableError as exc:
            self.app.state.startup_validation_error = exc.message
            logger.warning(
                "Configured provider model validation failed during startup; "
                "server will continue and requests will fail at provider resolution "
                "when config is incomplete. {}",
                exc.message,
            )

    async def shutdown(self) -> None:
        verbose = self.settings.log_api_error_tracebacks
        if self.messaging_workflow is not None:
            try:
                self.messaging_workflow.session_store.flush_pending_save()
            except Exception as e:
                if verbose:
                    logger.warning("Session store flush on shutdown: {}", e)
                else:
                    logger.warning(
                        "Session store flush on shutdown: exc_type={}",
                        type(e).__name__,
                    )

        logger.info("Shutdown requested, cleaning up...")
        if self.messaging_runtime:
            await best_effort(
                "messaging_runtime.stop",
                self.messaging_runtime.stop(),
                log_verbose_errors=verbose,
            )
        if self.cli_manager:
            await best_effort(
                "cli_manager.stop_all",
                self.cli_manager.stop_all(),
                log_verbose_errors=verbose,
            )
        if self._provider_runtime is not None:
            await best_effort(
                "provider_runtime.cleanup",
                self._provider_runtime.cleanup(),
                log_verbose_errors=verbose,
            )
        await self._shutdown_limiter()
        logger.info("Server shut down cleanly")

    async def _start_messaging_if_configured(self) -> None:
        try:
            components = messaging_platform_factory.create_messaging_components(
                self.settings.messaging_platform,
                MessagingPlatformOptions(
                    telegram_bot_token=self.settings.telegram_bot_token,
                    allowed_telegram_user_id=self.settings.allowed_telegram_user_id,
                    discord_bot_token=self.settings.discord_bot_token,
                    allowed_discord_channels=self.settings.allowed_discord_channels,
                    voice_note_enabled=self.settings.voice_note_enabled,
                    whisper_model=self.settings.whisper_model,
                    whisper_device=self.settings.whisper_device,
                    huggingface_api_key=self.settings.huggingface_api_key,
                    nvidia_nim_api_key=self.settings.nvidia_nim_api_key,
                    messaging_rate_limit=self.settings.messaging_rate_limit,
                    messaging_rate_window=self.settings.messaging_rate_window,
                    log_raw_messaging_content=self.settings.log_raw_messaging_content,
                    log_api_error_tracebacks=self.settings.log_api_error_tracebacks,
                ),
            )

            if components:
                await self._start_messaging_workflow(components)

        except ImportError as e:
            if self.settings.log_api_error_tracebacks:
                logger.warning("Messaging module import error: {}", e)
            else:
                logger.warning(
                    "Messaging module import error: exc_type={}",
                    type(e).__name__,
                )
        except Exception as e:
            if self.settings.log_api_error_tracebacks:
                logger.error("Failed to start messaging platform: {}", e)
                logger.error(traceback.format_exc())
            else:
                logger.error(
                    "Failed to start messaging platform: exc_type={}",
                    type(e).__name__,
                )

    async def _start_messaging_workflow(
        self, components: MessagingPlatformComponents
    ) -> None:
        workspace = (
            os.path.abspath(self.settings.allowed_dir)
            if self.settings.allowed_dir
            else os.getcwd()
        )
        os.makedirs(workspace, exist_ok=True)

        data_path = os.path.abspath(default_claude_workspace_path())
        os.makedirs(data_path, exist_ok=True)

        api_url = f"http://{self.settings.host}:{self.settings.port}/v1"
        allowed_dirs = [workspace] if self.settings.allowed_dir else []
        plans_dir_abs = os.path.abspath(os.path.join(data_path, "plans"))
        plans_directory = os.path.relpath(plans_dir_abs, workspace)
        self.cli_manager = cli_managed.ManagedClaudeSessionManager(
            workspace_path=workspace,
            api_url=api_url,
            allowed_dirs=allowed_dirs,
            plans_directory=plans_directory,
            auth_token=getattr(self.settings, "anthropic_auth_token", ""),
            log_raw_cli_diagnostics=self.settings.log_raw_cli_diagnostics,
            log_messaging_error_details=self.settings.log_messaging_error_details,
        )

        session_store = messaging_session.SessionStore(
            storage_path=os.path.join(data_path, "sessions.json"),
            message_log_cap=self.settings.max_message_log_entries_per_chat,
        )
        self.messaging_runtime = components.runtime
        self.messaging_workflow = messaging_workflow_module.MessagingWorkflow(
            platform_name=components.name,
            outbound=components.outbound,
            voice_cancellation=components.voice_cancellation,
            cli_manager=self.cli_manager,
            session_store=session_store,
            debug_platform_edits=self.settings.debug_platform_edits,
            debug_subagent_stack=self.settings.debug_subagent_stack,
            log_raw_messaging_content=self.settings.log_raw_messaging_content,
            log_raw_cli_diagnostics=self.settings.log_raw_cli_diagnostics,
            log_messaging_error_details=self.settings.log_messaging_error_details,
        )
        self._restore_tree_state(session_store)

        components.runtime.on_message(self.messaging_workflow.handle_message)
        await components.runtime.start()
        logger.info("{} platform started with messaging workflow", components.name)

    def _restore_tree_state(
        self, session_store: messaging_session.SessionStore
    ) -> None:
        conversation_snapshot = session_store.load_conversation_snapshot()
        if conversation_snapshot.is_empty:
            return
        if self.messaging_workflow is None:
            return

        logger.info(
            "Restoring {} conversation trees...",
            len(conversation_snapshot.trees),
        )

        self.messaging_workflow.replace_tree_queue(
            messaging_trees.TreeQueueManager.from_snapshot(
                conversation_snapshot,
                queue_update_callback=self.messaging_workflow.update_queue_positions,
                node_started_callback=self.messaging_workflow.mark_node_processing,
            )
        )
        if self.messaging_workflow.tree_queue.cleanup_stale_nodes() > 0:
            session_store.save_conversation_snapshot(
                self.messaging_workflow.tree_queue.snapshot()
            )

    def _publish_state(self) -> None:
        self.app.state.messaging_runtime = self.messaging_runtime
        self.app.state.messaging_workflow = self.messaging_workflow
        self.app.state.cli_manager = self.cli_manager

    async def _shutdown_limiter(self) -> None:
        verbose = self.settings.log_api_error_tracebacks
        try:
            await best_effort(
                "MessagingRateLimiter.shutdown_instance",
                messaging_limiter.MessagingRateLimiter.shutdown_instance(),
                timeout_s=2.0,
                log_verbose_errors=verbose,
            )
        except Exception as e:
            if verbose:
                logger.debug(
                    "Rate limiter shutdown skipped: {}: {}", type(e).__name__, e
                )
            else:
                logger.debug(
                    "Rate limiter shutdown skipped: exc_type={}", type(e).__name__
                )
