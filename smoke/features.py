"""Public feature inventory for contract, prerequisite, and product smoke tests.

The inventory is intentionally explicit. README-advertised behavior and exposed
public surface area must have deterministic pytest contract coverage plus a
product E2E scenario when that behavior is a user-facing product path. Liveness
and route probes live in ``smoke/prereq`` and do not count as product coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FeatureSource = Literal["readme", "public_surface"]


@dataclass(frozen=True, slots=True)
class FeatureCoverage:
    feature_id: str
    title: str
    source: FeatureSource
    pytest_contract_tests: tuple[str, ...]
    live_prereq_tests: tuple[str, ...]
    product_e2e_tests: tuple[str, ...]
    smoke_targets: tuple[str, ...]
    required_env: tuple[str, ...]
    skip_policy: str
    product_e2e_reason: str = ""

    @property
    def has_pytest_coverage(self) -> bool:
        return bool(self.pytest_contract_tests)


README_FEATURES: tuple[str, ...] = (
    "zero_cost_provider_access",
    "drop_in_claude_code_replacement",
    "drop_in_codex_replacement",
    "provider_matrix",
    "per_model_mapping",
    "thinking_token_support",
    "heuristic_tool_parser",
    "request_optimization",
    "smart_rate_limiting",
    "discord_telegram_bot",
    "subagent_control",
    "extensible_provider_platform_abcs",
    "optional_authentication",
    "vscode_extension",
    "intellij_extension",
    "voice_notes",
)


FEATURE_INVENTORY: tuple[FeatureCoverage, ...] = (
    FeatureCoverage(
        "zero_cost_provider_access",
        "Configured provider accepts real conversation turns",
        "readme",
        ("tests/api/test_dependencies.py", "tests/providers/"),
        ("test_configured_provider_models_stream_successfully",),
        ("test_provider_text_multiturn_e2e",),
        ("providers",),
        ("configured provider credentials or local provider endpoint",),
        "missing providers are missing_env unless FCC_ALLOW_NO_PROVIDER_SMOKE=1",
    ),
    FeatureCoverage(
        "drop_in_claude_code_replacement",
        "Claude-compatible API, CLI, and editor protocol flows work",
        "readme",
        ("tests/api/test_api.py", "tests/cli/test_cli.py"),
        ("test_probe_and_models_routes", "test_claude_cli_prompt_when_available"),
        (
            "test_api_basic_conversation_e2e",
            "test_claude_cli_adaptive_thinking_e2e",
            "test_nvidia_nim_cli_matrix_e2e",
            "test_openrouter_free_cli_matrix_e2e",
            "test_vscode_protocol_e2e",
            "test_jetbrains_protocol_e2e",
        ),
        ("api", "cli", "clients", "nvidia_nim_cli", "openrouter_free_cli"),
        (
            "configured provider",
            "FCC_SMOKE_CLAUDE_BIN for real Claude CLI",
            "NVIDIA_NIM_API_KEY",
            "OPENROUTER_API_KEY",
        ),
        "skip real CLI when binary is absent; configured providers must pass",
    ),
    FeatureCoverage(
        "drop_in_codex_replacement",
        "OpenAI Responses API and Codex CLI adapter route through the proxy",
        "readme",
        (
            "tests/api/test_openai_responses.py",
            "tests/cli/test_entrypoints.py",
            "tests/cli/test_codex_model_catalog.py",
            "tests/core/openai_responses/test_sse.py",
        ),
        ("test_probe_and_models_routes",),
        ("test_provider_codex_responses_text_e2e",),
        ("api", "providers"),
        ("configured provider credentials or local provider endpoint",),
        "missing providers are missing_env unless FCC_ALLOW_NO_PROVIDER_SMOKE=1",
    ),
    FeatureCoverage(
        "provider_matrix",
        "Every configured provider prefix can satisfy conversation scenarios",
        "readme",
        ("tests/api/test_dependencies.py", "tests/providers/"),
        ("test_configured_provider_models_stream_successfully",),
        ("test_provider_matrix_presence_e2e", "test_provider_text_multiturn_e2e"),
        ("providers",),
        ("configured provider credentials/endpoints", "optional FCC_SMOKE_MODEL_*"),
        "selected providers missing credentials are failing missing_env",
    ),
    FeatureCoverage(
        "per_model_mapping",
        "Opus, Sonnet, Haiku, and fallback mappings route explicitly",
        "readme",
        ("tests/api/test_model_router.py", "tests/config/test_config.py"),
        ("test_model_mapping_configuration_is_consistent",),
        ("test_model_mapping_matrix_e2e",),
        ("providers",),
        ("MODEL", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"),
        "skip only when live smoke is intentionally allowed to run with no provider",
    ),
    FeatureCoverage(
        "mixed_provider_mapping",
        "Model-specific overrides can route to different providers",
        "public_surface",
        ("tests/api/test_model_router.py", "tests/config/test_config.py"),
        ("test_mixed_provider_model_mapping_when_configured",),
        ("test_model_mapping_matrix_e2e",),
        ("providers",),
        ("MODEL", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"),
        "configured mixed-provider mappings must resolve consistently",
    ),
    FeatureCoverage(
        "thinking_token_support",
        "Thinking history, adaptive thinking, and redacted blocks are accepted",
        "readme",
        (
            "tests/contracts/test_stream_contracts.py",
            "tests/providers/test_open_router.py",
        ),
        (),
        (
            "test_provider_adaptive_thinking_history_e2e",
            "test_provider_reasoning_tool_continuation_e2e",
            "test_gemini_thought_signature_tool_continuation_e2e",
            "test_claude_cli_adaptive_thinking_e2e",
            "test_per_model_thinking_config_e2e",
        ),
        ("providers", "cli", "config"),
        ("configured provider",),
        "configured providers must not reject adaptive thinking payloads",
    ),
    FeatureCoverage(
        "heuristic_tool_parser",
        "Tool use and tool result continuation survive provider/client paths",
        "readme",
        ("tests/providers/test_parsers.py", "tests/contracts/test_stream_contracts.py"),
        ("test_live_tool_use_when_configured_model_supports_tools",),
        (
            "test_provider_interleaved_thinking_tool_e2e",
            "test_provider_tool_result_continuation_e2e",
            "test_gemini_thought_signature_tool_continuation_e2e",
            "test_provider_reasoning_tool_continuation_e2e",
        ),
        ("tools", "providers"),
        ("configured tool-capable provider",),
        "tool-capable configured providers must emit or continue tool results",
    ),
    FeatureCoverage(
        "request_optimization",
        "Local request optimizations return product responses without providers",
        "readme",
        (
            "tests/api/test_optimization_handlers.py",
            "tests/api/test_routes_optimizations.py",
        ),
        ("test_optimization_fast_paths_do_not_need_provider",),
        ("test_api_request_optimizations_e2e",),
        ("api",),
        (),
        "always runnable once the local smoke server starts",
    ),
    FeatureCoverage(
        "smart_rate_limiting",
        "Disconnect and limiter cleanup preserve follow-up requests",
        "readme",
        ("tests/providers/test_provider_rate_limit.py",),
        ("test_client_disconnect_mid_stream_does_not_crash_server",),
        ("test_provider_disconnect_e2e",),
        ("rate_limit", "providers"),
        ("configured provider",),
        "upstream disconnects are skips only when classified upstream_unavailable",
    ),
    FeatureCoverage(
        "discord_telegram_bot",
        "Discord and Telegram product flows render progress and transcripts",
        "readme",
        (
            "tests/messaging/test_discord_platform.py",
            "tests/messaging/test_telegram.py",
        ),
        (
            "test_telegram_bot_api_permissions",
            "test_discord_bot_api_permissions",
        ),
        (
            "test_messaging_fake_full_flow_e2e",
            "test_telegram_live_permissions_e2e",
            "test_discord_live_permissions_e2e",
        ),
        ("messaging", "telegram", "discord"),
        ("bot tokens/channels only for side-effectful live platform tests",),
        "fake platform is required; real platforms skip without explicit env",
    ),
    FeatureCoverage(
        "subagent_control",
        "Task-like tool output is rendered and controlled as foreground work",
        "readme",
        ("tests/providers/test_subagent_interception.py",),
        (),
        ("test_messaging_subagent_control_e2e",),
        ("messaging",),
        (),
        "fake platform exercises tool transcript behavior without spawning agents",
    ),
    FeatureCoverage(
        "extensible_provider_platform_abcs",
        "Provider and platform factories expose built-in extension points",
        "readme",
        (
            "tests/contracts/test_feature_manifest.py",
            "tests/providers/test_registry.py",
        ),
        (),
        ("test_provider_registry_e2e", "test_platform_factory_e2e"),
        ("extensibility",),
        (),
        "always runnable with isolated settings",
    ),
    FeatureCoverage(
        "optional_authentication",
        "Anthropic-style auth headers are enforced and accepted",
        "readme",
        ("tests/api/test_auth.py",),
        ("test_auth_token_is_enforced_for_all_supported_header_shapes",),
        ("test_api_auth_header_variants_e2e",),
        ("auth",),
        ("ANTHROPIC_AUTH_TOKEN",),
        "product test starts an isolated token-protected server",
    ),
    FeatureCoverage(
        "vscode_extension",
        "VS Code protocol-shaped requests work against the proxy",
        "readme",
        (
            "tests/api/test_models_validators.py::test_messages_request_accepts_adaptive_thinking_type",
        ),
        ("test_vscode_and_jetbrains_shaped_requests",),
        ("test_vscode_protocol_e2e",),
        ("clients",),
        ("configured provider",),
        "extension source is external; protocol payload is covered here",
    ),
    FeatureCoverage(
        "intellij_extension",
        "JetBrains/ACP protocol-shaped requests work against the proxy",
        "readme",
        (
            "tests/api/test_models_validators.py::test_messages_request_accepts_adaptive_thinking_type",
        ),
        ("test_vscode_and_jetbrains_shaped_requests",),
        ("test_jetbrains_protocol_e2e",),
        ("clients",),
        ("configured provider",),
        "extension source is external; protocol payload is covered here",
    ),
    FeatureCoverage(
        "voice_notes",
        "Voice note intake, cancellation, and transcription backends work",
        "readme",
        (
            "tests/messaging/test_voice_handlers.py",
            "tests/messaging/test_transcription.py",
        ),
        ("test_voice_transcription_backend_when_explicitly_enabled",),
        (
            "test_voice_platform_fake_e2e",
            "test_voice_local_backend_e2e",
            "test_voice_nim_backend_e2e",
        ),
        ("messaging", "voice"),
        ("VOICE_NOTE_ENABLED", "FCC_SMOKE_RUN_VOICE", "WHISPER_DEVICE"),
        "fake cancellation is required; backend transcription is opt-in",
    ),
    FeatureCoverage(
        "anthropic_api_routes",
        "Messages, count_tokens, errors, and stop use Anthropic-compatible shapes",
        "public_surface",
        ("tests/api/test_api.py",),
        ("test_probe_and_models_routes", "test_stop_endpoint_reports_no_messaging"),
        (
            "test_api_basic_conversation_e2e",
            "test_api_count_tokens_full_payload_e2e",
            "test_api_error_shape_e2e",
            "test_api_stop_e2e",
        ),
        ("api",),
        ("configured provider for streaming messages",),
        "route pings are prereqs; product route behavior is covered by E2E cases",
    ),
    FeatureCoverage(
        "probe_routes",
        "HEAD and OPTIONS compatibility probes are accepted",
        "public_surface",
        ("tests/api/test_api.py::test_probe_endpoints_return_204_with_allow_headers",),
        ("test_probe_and_models_routes",),
        (),
        ("api",),
        (),
        "always runnable once the local smoke server starts",
        "probe routes are compatibility prerequisites, not product behavior",
    ),
    FeatureCoverage(
        "count_tokens_contract",
        "Token counting accepts full Claude content payloads",
        "public_surface",
        ("tests/api/test_request_utils.py",),
        ("test_count_tokens_accepts_thinking_tools_and_results",),
        ("test_api_count_tokens_full_payload_e2e",),
        ("api",),
        (),
        "always runnable once the local smoke server starts",
    ),
    FeatureCoverage(
        "provider_proxy_timeout_config",
        "Provider proxies and HTTP timeout settings reach provider config",
        "public_surface",
        ("tests/api/test_dependencies.py", "tests/providers/test_registry.py"),
        (),
        ("test_proxy_timeout_config_e2e",),
        ("config",),
        (),
        "always runnable with isolated env files",
    ),
    FeatureCoverage(
        "lmstudio_endpoint",
        "LM Studio native messages and local no-key operation work when running",
        "public_surface",
        (
            "tests/providers/test_lmstudio.py",
            "tests/providers/test_anthropic_messages.py",
        ),
        ("test_lmstudio_models_endpoint_when_available",),
        ("test_lmstudio_native_messages_e2e",),
        ("lmstudio",),
        ("LM_STUDIO_BASE_URL with running LM Studio server",),
        "skip when local upstream is unavailable",
    ),
    FeatureCoverage(
        "llamacpp_endpoint",
        "llama.cpp native messages and local no-key operation work when running",
        "public_surface",
        (
            "tests/providers/test_llamacpp.py",
            "tests/providers/test_anthropic_messages.py",
        ),
        ("test_llamacpp_models_endpoint_when_available",),
        ("test_llamacpp_native_messages_e2e",),
        ("llamacpp",),
        ("LLAMACPP_BASE_URL with running llama-server",),
        "skip when local upstream is unavailable",
    ),
    FeatureCoverage(
        "ollama_endpoint",
        "Ollama native Anthropic messages and local no-key operation work when running",
        "public_surface",
        ("tests/providers/test_ollama.py",),
        ("test_ollama_models_endpoint_when_available",),
        ("test_ollama_native_messages_e2e",),
        ("ollama",),
        ("OLLAMA_BASE_URL with running Ollama server",),
        "skip when local upstream is unavailable",
    ),
    FeatureCoverage(
        "package_cli_entrypoints",
        "Installed package scripts scaffold config and start the server",
        "public_surface",
        ("tests/cli/test_entrypoints.py",),
        (
            "test_fcc_init_scaffolds_user_config",
            "test_free_claude_code_entrypoint_starts_server",
        ),
        ("test_entrypoint_init_e2e", "test_entrypoint_server_e2e"),
        ("cli",),
        (),
        "always runnable once uv project dependencies are available",
    ),
    FeatureCoverage(
        "claude_cli_drop_in",
        "Claude CLI can send adaptive thinking and tool-shaped history",
        "public_surface",
        ("tests/cli/test_cli.py",),
        ("test_claude_cli_prompt_when_available",),
        (
            "test_claude_cli_adaptive_thinking_e2e",
            "test_claude_cli_multiturn_tool_protocol_e2e",
            "test_nvidia_nim_cli_matrix_e2e",
            "test_openrouter_free_cli_matrix_e2e",
        ),
        ("cli", "nvidia_nim_cli", "openrouter_free_cli"),
        (
            "FCC_SMOKE_CLAUDE_BIN",
            "configured provider",
            "NVIDIA_NIM_API_KEY",
            "OPENROUTER_API_KEY",
        ),
        "skip only when Claude CLI binary is absent",
    ),
    FeatureCoverage(
        "messaging_commands",
        "Messaging /stop, /clear, and /stats operate on product state",
        "public_surface",
        (
            "tests/messaging/test_handler.py",
            "tests/messaging/test_handler_integration.py",
        ),
        (),
        ("test_messaging_commands_stop_clear_stats_e2e",),
        ("messaging",),
        (),
        "required fake-platform product flow",
    ),
    FeatureCoverage(
        "tree_threading",
        "Reply-based branches fork sessions and stay scoped",
        "public_surface",
        (
            "tests/messaging/test_tree_queue.py",
            "tests/messaging/test_tree_concurrency.py",
        ),
        (),
        ("test_tree_threading_e2e",),
        ("messaging",),
        (),
        "required fake-platform product flow",
    ),
    FeatureCoverage(
        "restart_restore",
        "Persisted tree state restores reply routing after restart",
        "public_surface",
        ("tests/messaging/test_restart_reply_restore.py",),
        (),
        ("test_restart_restore_and_session_persistence_e2e",),
        ("messaging",),
        (),
        "required fake-platform persistence flow",
    ),
    FeatureCoverage(
        "session_persistence",
        "Session JSON preserves trees, node mapping, and message logs",
        "public_surface",
        ("tests/messaging/test_session_store_edge_cases.py",),
        (),
        ("test_restart_restore_and_session_persistence_e2e",),
        ("messaging",),
        (),
        "required fake-platform persistence flow",
    ),
    FeatureCoverage(
        "config_env_precedence",
        "FCC_ENV_FILE, dotenv, and process env precedence are deterministic",
        "public_surface",
        ("tests/config/test_config.py",),
        (),
        ("test_env_precedence_e2e",),
        ("config",),
        (),
        "always runnable with isolated env files",
    ),
    FeatureCoverage(
        "removed_env_migration",
        "Removed thinking env vars are ignored without changing defaults",
        "public_surface",
        ("tests/config/test_config.py",),
        (),
        ("test_removed_env_migration_e2e",),
        ("config",),
        (),
        "always runnable with isolated env files",
    ),
    FeatureCoverage(
        "streaming_error_mapping",
        "Provider and validation errors map to Anthropic-compatible payloads",
        "public_surface",
        (
            "tests/providers/test_streaming_errors.py",
            "tests/providers/test_error_mapping.py",
        ),
        (),
        ("test_api_error_shape_e2e", "test_provider_error_e2e"),
        ("api", "providers"),
        ("configured provider for provider error scenario",),
        "invalid request path is required; provider error path requires provider",
    ),
)


def feature_ids(*, source: FeatureSource | None = None) -> set[str]:
    """Return feature IDs covered by the inventory."""
    return {
        feature.feature_id
        for feature in FEATURE_INVENTORY
        if source is None or feature.source == source
    }
