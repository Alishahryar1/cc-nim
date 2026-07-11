"""Package import contract tests (static AST; dynamic ``importlib`` loads are not scanned)."""

import ast
from pathlib import Path

_PACKAGE_ROOT = Path("src") / "free_claude_code"


def test_python314_native_annotations_do_not_use_legacy_future_import() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in repo_root.rglob("*.py"):
        if ".git" in path.parts or ".venv" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "__future__":
                continue
            if any(alias.name == "annotations" for alias in node.names):
                offenders.append(path.relative_to(repo_root).as_posix())

    assert sorted(offenders) == []


def test_server_startup_is_owned_by_cli_entrypoint() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert not (repo_root / "server.py").exists()
    assert _text_occurrences(repo_root, "server" + ":app") == []

    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'fcc-server = "free_claude_code.cli.entrypoints:serve"' in pyproject_text
    assert (
        'free-claude-code = "free_claude_code.cli.entrypoints:serve"' in pyproject_text
    )


def test_runtime_packages_live_only_under_src_namespace() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src" / "free_claude_code"

    assert (package_root / "__init__.py").exists()
    for package_name in {
        "application",
        "api",
        "cli",
        "config",
        "core",
        "messaging",
        "providers",
        "runtime",
    }:
        assert (package_root / package_name).is_dir()
        assert not (repo_root / package_name).exists()


def test_no_old_top_level_first_party_imports_remain() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = {
        "application",
        "api",
        "cli",
        "config",
        "core",
        "messaging",
        "providers",
        "runtime",
    }
    offenders: list[str] = []

    for path in repo_root.rglob("*.py"):
        if ".git" in path.parts or ".venv" in path.parts:
            continue
        for imported in _imports_from(path, repo_root):
            if imported is None:
                continue
            root = imported.split(".", 1)[0]
            if root in forbidden:
                offenders.append(f"{path.relative_to(repo_root)}: {imported}")

    assert sorted(offenders) == []


def test_api_and_messaging_do_not_import_provider_common() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert not (
        repo_root / "src" / "free_claude_code" / "providers" / "common"
    ).exists()
    offenders = _imports_matching(
        [
            repo_root / "src" / "free_claude_code" / "api",
            repo_root / "src" / "free_claude_code" / "messaging",
        ],
        forbidden_prefixes=("free_claude_code.providers.common",),
    )

    assert offenders == []


def test_provider_adapters_do_not_import_runtime_layers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _imports_matching(
        [repo_root / "src" / "free_claude_code" / "providers"],
        forbidden_prefixes=(
            "free_claude_code.api.",
            "free_claude_code.messaging.",
            "free_claude_code.cli.",
            "free_claude_code.runtime.",
        ),
    )

    assert offenders == []


def test_anthropic_request_boundaries_use_the_protocol_model() -> None:
    """Known Messages fields must not cross core/provider boundaries by duck typing."""
    repo_root = Path(__file__).resolve().parents[2]
    roots = [
        repo_root / "src" / "free_claude_code" / "core" / "anthropic",
        repo_root / "src" / "free_claude_code" / "providers",
    ]
    request_names = {"request", "request_data"}
    protocol_fields = {
        "extra_body",
        "max_tokens",
        "messages",
        "model",
        "stop_sequences",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    }
    offenders: list[str] = []

    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(repo_root).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    arguments = [
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    ]
                    for argument in arguments:
                        if argument.arg.lstrip("_") not in request_names:
                            continue
                        if isinstance(argument.annotation, ast.Name) and (
                            argument.annotation.id == "Any"
                        ):
                            offenders.append(
                                f"{relative}:{argument.lineno}: "
                                f"{node.name}({argument.arg}: Any)"
                            )
                if not isinstance(node, ast.Call) or not (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id.lstrip("_") in request_names
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in protocol_fields
                ):
                    continue
                offenders.append(
                    f"{relative}:{node.lineno}: "
                    f"getattr({node.args[0].id}, {node.args[1].value!r})"
                )

    assert sorted(offenders) == []


def test_core_does_not_import_product_packages() -> None:
    """Neutral ``core`` must stay independent of API, workers, and providers."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _imports_matching(
        [repo_root / "src" / "free_claude_code" / "core"],
        forbidden_prefixes=(
            "free_claude_code.application.",
            "free_claude_code.api.",
            "free_claude_code.messaging.",
            "free_claude_code.cli.",
            "smoke.",
            "free_claude_code.providers.",
            "free_claude_code.config.",
        ),
    )
    assert offenders == []


def test_core_does_not_import_provider_transport_sdks() -> None:
    """Provider SDK and HTTP failure policy belongs under ``providers``."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _imports_matching(
        [repo_root / "src" / "free_claude_code" / "core"],
        forbidden_prefixes=("httpx", "openai"),
    )
    assert offenders == []


def test_providers_do_not_own_wire_error_type_literals() -> None:
    """Provider failures carry semantic kinds, never protocol error names."""
    repo_root = Path(__file__).resolve().parents[2]
    providers_root = repo_root / "src" / "free_claude_code" / "providers"
    wire_types = {
        "api_error",
        "authentication_error",
        "billing_error",
        "invalid_request_error",
        "not_found_error",
        "overloaded_error",
        "permission_error",
        "rate_limit_error",
        "request_too_large",
        "timeout_error",
    }
    offenders: list[str] = []
    for path in providers_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(
            f"{path.relative_to(repo_root).as_posix()}:{node.lineno}: {node.value}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value in wire_types
        )
    assert offenders == []


def test_application_owns_routing_execution_and_consumer_ports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src" / "free_claude_code"
    application_root = package_root / "application"
    api_root = package_root / "api"

    for filename in {
        "__init__.py",
        "execution.py",
        "model_metadata.py",
        "ports.py",
        "routing.py",
    }:
        assert (application_root / filename).exists()

    assert not (api_root / "model_router.py").exists()
    assert not (api_root / "provider_execution.py").exists()
    assert (
        _imports_matching(
            [application_root],
            forbidden_prefixes=(
                "free_claude_code.api.",
                "free_claude_code.cli.",
                "free_claude_code.messaging.",
                "free_claude_code.providers.",
                "free_claude_code.runtime.",
            ),
        )
        == []
    )

    provider_imports = _imports_matching(
        [api_root],
        forbidden_prefixes=(
            "free_claude_code.providers.base",
            "free_claude_code.providers.model_listing",
            "free_claude_code.providers.runtime",
        ),
    )
    assert provider_imports == []

    unexpected_provider_application_imports: list[str] = []
    providers_root = package_root / "providers"
    for path in providers_root.rglob("*.py"):
        for imported in _imports_from(path, repo_root):
            if imported is None or not imported.startswith(
                "free_claude_code.application."
            ):
                continue
            if imported in {
                "free_claude_code.application.errors",
                "free_claude_code.application.model_metadata",
            }:
                continue
            unexpected_provider_application_imports.append(
                f"{path.relative_to(repo_root)}: {imported}"
            )
    assert unexpected_provider_application_imports == []


def test_provider_catalog_is_single_source_for_supported_ids() -> None:
    from free_claude_code.config.provider_catalog import (
        PROVIDER_CATALOG,
        SUPPORTED_PROVIDER_IDS,
    )
    from free_claude_code.providers.runtime import PROVIDER_FACTORIES

    assert tuple(PROVIDER_CATALOG.keys()) == SUPPORTED_PROVIDER_IDS
    assert set(SUPPORTED_PROVIDER_IDS) == set(PROVIDER_FACTORIES)


def test_provider_runtime_replaces_old_registry_module() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert not (
        repo_root / "src" / "free_claude_code" / "providers" / "registry.py"
    ).exists()
    assert (
        repo_root / "src" / "free_claude_code" / "providers" / "runtime" / "runtime.py"
    ).exists()
    assert (
        repo_root / "src" / "free_claude_code" / "providers" / "runtime" / "factory.py"
    ).exists()
    assert (
        repo_root
        / "src"
        / "free_claude_code"
        / "providers"
        / "runtime"
        / "discovery.py"
    ).exists()

    offenders = _imports_matching(
        [
            repo_root / "src" / "free_claude_code" / "api",
            repo_root / "tests",
            repo_root / "smoke",
        ],
        forbidden_prefixes=("free_claude_code.providers.registry",),
    )
    assert offenders == []


def test_config_does_not_import_non_config_packages() -> None:
    """Settings and env handling must not depend on transport or protocol layers."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _imports_matching(
        [repo_root / "src" / "free_claude_code" / "config"],
        forbidden_prefixes=(
            "free_claude_code.api.",
            "free_claude_code.messaging.",
            "free_claude_code.cli.",
            "smoke.",
            "free_claude_code.providers.",
            "free_claude_code.core.",
        ),
    )
    assert offenders == []


def test_settings_stays_schema_only() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_root = repo_root / "src" / "free_claude_code" / "config"

    assert (config_root / "env_files.py").exists()
    assert (config_root / "model_refs.py").exists()

    settings_text = (config_root / "settings.py").read_text(encoding="utf-8")
    for removed_api in {
        "def resolve_model",
        "def resolve_thinking",
        "def configured_chat_model_refs",
        "def web_fetch_allowed_scheme_set",
        "def parse_provider_type",
        "def parse_model_name",
        "def uses_process_anthropic_auth_token",
        "def claude_workspace",
        "def claude_cli_bin",
        "def codex_cli_bin",
        "def provider_type",
        "def model_name",
    }:
        assert removed_api not in settings_text


def test_messaging_does_not_import_disallowed_modules() -> None:
    """Runtime composition keeps messaging independent of concrete products."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in (repo_root / "src" / "free_claude_code" / "messaging").rglob("*.py"):
        for imported in _imports_from(path, repo_root):
            if imported is None:
                continue
            if (
                imported == "free_claude_code.api"
                or imported.startswith("free_claude_code.api.")
                or imported == "free_claude_code.cli"
                or imported.startswith("free_claude_code.cli.")
                or imported == "smoke"
                or imported.startswith("smoke.")
            ) or imported.startswith("free_claude_code.providers."):
                rel = path.relative_to(repo_root)
                offenders.append(f"{rel}: {imported}")

    assert sorted(offenders) == []


def test_single_owner_runtime_dependency_direction() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src" / "free_claude_code"
    api_root = package_root / "api"
    cli_root = package_root / "cli"
    runtime_root = package_root / "runtime"

    assert runtime_root.is_dir()
    for removed in {
        api_root / "runtime.py",
        api_root / "admin_urls.py",
        api_root / "gateway_model_ids.py",
        api_root / "admin_config",
    }:
        assert not removed.exists()

    assert (
        _imports_matching(
            [api_root],
            forbidden_prefixes=(
                "free_claude_code.cli",
                "free_claude_code.messaging",
                "free_claude_code.runtime",
            ),
        )
        == []
    )
    assert (
        _imports_matching(
            [cli_root],
            forbidden_prefixes=("free_claude_code.api",),
        )
        == []
    )

    api_text = "\n".join(
        path.read_text(encoding="utf-8") for path in api_root.rglob("*.py")
    )
    for removed_state in {
        "app.state.provider_runtime",
        "app.state.messaging_runtime",
        "app.state.messaging_workflow",
        "app.state.cli_manager",
        "app.state.admin_restart_callback",
        "app.state.admin_pending_fields",
    }:
        assert removed_state not in api_text
    assert "app.state.services" in api_text

    for marker in {
        package_root / "application" / "__init__.py",
        api_root / "__init__.py",
        runtime_root / "__init__.py",
    }:
        marker_text = marker.read_text(encoding="utf-8")
        assert "from " not in marker_text
        assert "import " not in marker_text
        assert "__all__" not in marker_text


def test_neutral_moved_helpers_keep_their_dependency_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src" / "free_claude_code"
    admin_root = package_root / "config" / "admin"
    gateway_ids = package_root / "core" / "gateway_model_ids.py"

    admin_offenders: list[str] = []
    for path in admin_root.rglob("*.py"):
        for imported in _imports_from(path, repo_root):
            if imported is None or not imported.startswith("free_claude_code."):
                continue
            if not imported.startswith("free_claude_code.config"):
                admin_offenders.append(f"{path.relative_to(repo_root)}: {imported}")
    assert admin_offenders == []

    assert all(
        not imported.startswith("free_claude_code.")
        for imported in _imports_from(gateway_ids, repo_root)
    )


def test_api_does_not_import_provider_implementation_packages() -> None:
    """HTTP adapters depend on application contracts, never provider internals."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _imports_matching(
        [repo_root / "src" / "free_claude_code" / "api"],
        forbidden_prefixes=("free_claude_code.providers",),
    )
    assert offenders == []


def test_removed_openrouter_rollback_transport_stays_removed() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert not (
        repo_root
        / "src"
        / "free_claude_code"
        / "providers"
        / "open_router"
        / "chat_request.py"
    ).exists()
    assert _text_occurrences(repo_root, "OpenRouter" + "ChatProvider") == []
    assert _text_occurrences(repo_root, "OPENROUTER" + "_TRANSPORT") == []


def test_provider_transports_live_under_transport_family_packages() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    providers_root = repo_root / "src" / "free_claude_code" / "providers"

    assert not (providers_root / "openai_compat.py").exists()
    assert not (providers_root / "anthropic_messages.py").exists()
    assert (providers_root / "transports" / "openai_chat" / "transport.py").exists()
    assert (
        providers_root / "transports" / "anthropic_messages" / "transport.py"
    ).exists()

    offenders = _imports_matching(
        [providers_root, repo_root / "tests"],
        forbidden_prefixes=(
            "free_claude_code.providers.openai_compat",
            "free_claude_code.providers.anthropic_messages",
        ),
    )
    assert offenders == []


def test_cloud_providers_do_not_import_native_anthropic_transport() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    for provider_dir in ("open_router", "wafer", "kimi", "minimax", "fireworks", "zai"):
        provider_root = (
            repo_root / "src" / "free_claude_code" / "providers" / provider_dir
        )
        occurrences = [
            path.relative_to(repo_root).as_posix()
            for path in provider_root.rglob("*.py")
            if "free_claude_code.providers.transports.anthropic_messages"
            in path.read_text(encoding="utf-8")
        ]
        assert occurrences == []


def test_provider_request_policy_lives_with_transport_families() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    providers_root = repo_root / "src" / "free_claude_code" / "providers"

    deleted_request_modules = (
        "free_claude_code.providers.cerebras.request",
        "free_claude_code.providers.deepseek.request",
        "free_claude_code.providers.fireworks.request",
        "free_claude_code.providers.gemini.request",
        "free_claude_code.providers.groq.request",
        "free_claude_code.providers.kimi.request",
        "free_claude_code.providers.mistral.request",
        "free_claude_code.providers.nvidia_nim.request",
        "free_claude_code.providers.opencode.request",
        "free_claude_code.providers.open_router.request",
        "free_claude_code.providers.zai.request",
    )

    assert (
        providers_root / "transports" / "openai_chat" / "request_policy.py"
    ).exists()
    assert (
        providers_root / "transports" / "anthropic_messages" / "request_policy.py"
    ).exists()
    assert not sorted(
        path.relative_to(repo_root).as_posix()
        for path in providers_root.glob("*/request.py")
    )

    offenders = _imports_matching(
        [providers_root, repo_root / "tests"],
        forbidden_prefixes=deleted_request_modules,
    )
    assert offenders == []


def test_anthropic_core_has_no_cloud_provider_native_policy() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    anthropic_core = repo_root / "src" / "free_claude_code" / "core" / "anthropic"

    occurrences: list[str] = []
    for path in anthropic_core.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "OpenRouter" in text or "openrouter" in text:
            occurrences.append(path.relative_to(repo_root).as_posix())

    assert occurrences == []


def test_protocol_stream_state_and_provider_recovery_have_separate_owners() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src" / "free_claude_code"
    streaming_root = package_root / "core" / "anthropic" / "streaming"
    providers_root = package_root / "providers"
    stream_recovery = providers_root / "stream_recovery.py"

    assert (
        _imports_matching(
            [streaming_root],
            forbidden_prefixes=("free_claude_code.providers",),
        )
        == []
    )
    assert "free_claude_code.providers.failure_policy" in set(
        _imports_from(stream_recovery, repo_root)
    )

    streaming_exports = (streaming_root / "__init__.py").read_text(encoding="utf-8")
    for provider_owned_name in {
        "RecoveryController",
        "RecoveryFailureAction",
        "RecoveryHoldbackBuffer",
        "is_retryable_stream_error",
    }:
        assert provider_owned_name not in streaming_exports

    for transport in {"openai_chat", "anthropic_messages"}:
        imports = set(
            _imports_from(
                providers_root / "transports" / transport / "stream.py",
                repo_root,
            )
        )
        assert "free_claude_code.providers.stream_recovery" in imports


def test_openai_responses_uses_adapter_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    responses_root = (
        repo_root / "src" / "free_claude_code" / "core" / "openai_responses"
    )
    responses_streaming_root = responses_root / "streaming"
    api_root = repo_root / "src" / "free_claude_code" / "api"
    handlers_root = api_root / "handlers"

    assert not (repo_root / "src" / "free_claude_code" / "api" / "services.py").exists()
    assert not (api_root / "request_pipeline.py").exists()
    assert not (responses_root / "conversion.py").exists()
    assert not (responses_root / "sse.py").exists()
    assert not (responses_root / "output.py").exists()
    assert not (responses_root / "stream_state.py").exists()
    for filename in {
        "adapter.py",
        "anthropic_sse.py",
        "errors.py",
        "events.py",
        "ids.py",
        "input.py",
        "items.py",
        "models.py",
        "reasoning.py",
        "stream.py",
        "tools.py",
    }:
        assert (responses_root / filename).exists()
    for filename in {
        "__init__.py",
        "assembler.py",
        "blocks.py",
        "completion.py",
        "error_mapping.py",
        "event_builders.py",
        "ledger.py",
    }:
        assert (responses_streaming_root / filename).exists()

    stream_text = (responses_root / "stream.py").read_text(encoding="utf-8")
    assert "from .streaming import ResponsesStreamAssembler" in stream_text

    responses_handler = handlers_root / "responses.py"
    responses_handler_text = responses_handler.read_text(encoding="utf-8")
    assert "free_claude_code.core.openai_responses" in set(
        _imports_from(responses_handler, repo_root)
    )
    assert "OpenAIResponsesAdapter" in responses_handler_text
    assert "OpenAIResponsesRequest" in responses_handler_text
    assert not (
        repo_root
        / "src"
        / "free_claude_code"
        / "api"
        / "models"
        / "openai_responses.py"
    ).exists()
    routes_text = (
        repo_root / "src" / "free_claude_code" / "api" / "routes.py"
    ).read_text(encoding="utf-8")
    assert "ApiRequestPipeline" not in routes_text
    assert "request_pipeline" not in routes_text
    assert "from .handlers import" in routes_text
    assert "free_claude_code.api.services" not in routes_text
    for old_helper in {
        "responses_request_to_anthropic_payload",
        "anthropic_message_response_to_openai_response",
        "iter_anthropic_sse_as_openai_responses",
        "collect_openai_response_from_anthropic_sse",
        "iter_message_response_as_openai_responses",
    }:
        assert old_helper not in responses_handler_text

    offenders: list[str] = []
    for path in (repo_root / "src" / "free_claude_code" / "api").rglob("*.py"):
        for imported in _imports_from(path, repo_root):
            if imported is not None and imported.startswith(
                "free_claude_code.core.openai_responses."
            ):
                rel = path.relative_to(repo_root)
                offenders.append(f"{rel}: {imported}")
    assert sorted(offenders) == []

    responses_importers: list[str] = []
    for path in (repo_root / "src" / "free_claude_code" / "api").rglob("*.py"):
        imports = set(_imports_from(path, repo_root))
        if "free_claude_code.core.openai_responses" in imports:
            responses_importers.append(path.relative_to(repo_root).as_posix())
    assert sorted(responses_importers) == [
        "src/free_claude_code/api/app.py",
        "src/free_claude_code/api/handlers/responses.py",
        "src/free_claude_code/api/request_errors.py",
        "src/free_claude_code/api/routes.py",
    ]

    response_handler_imports = set(_imports_from(responses_handler, repo_root))
    for forbidden in {
        "free_claude_code.api.optimization_handlers",
        "free_claude_code.api.detection",
        "free_claude_code.api.web_tools",
    }:
        assert forbidden not in response_handler_imports

    provider_execution_text = (
        repo_root / "src" / "free_claude_code" / "application" / "execution.py"
    ).read_text(encoding="utf-8")
    assert "StreamingResponse" not in provider_execution_text
    assert "OpenAIResponsesAdapter" not in provider_execution_text

    adapter_text = (responses_root / "adapter.py").read_text(encoding="utf-8")
    for deleted_api in {
        "from_anthropic_message",
        "collect_from_anthropic_sse",
        "iter_sse_from_anthropic_message",
    }:
        assert deleted_api not in adapter_text


def test_admin_config_uses_package_owners_and_catalog_manifest() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api_root = repo_root / "src" / "free_claude_code" / "api"
    config_root = repo_root / "src" / "free_claude_code" / "config"
    admin_config_root = config_root / "admin"

    assert not (api_root / "admin_config.py").exists()
    assert not (api_root / "admin_config").exists()
    for filename in {
        "__init__.py",
        "manifest.py",
        "provider_manifest.py",
        "sources.py",
        "values.py",
        "validation.py",
        "persistence.py",
        "status.py",
    }:
        assert (admin_config_root / filename).exists()

    init_text = (admin_config_root / "__init__.py").read_text(encoding="utf-8")
    assert "from " not in init_text
    assert "__all__" not in init_text

    routes_imports = set(_imports_from(api_root / "admin_routes.py", repo_root))
    assert "free_claude_code.api.admin_config" not in routes_imports
    for expected in {
        "free_claude_code.config.admin.manifest",
        "free_claude_code.config.admin.persistence",
        "free_claude_code.config.admin.values",
    }:
        assert expected in routes_imports

    provider_manifest_text = (admin_config_root / "provider_manifest.py").read_text(
        encoding="utf-8"
    )
    assert "PROVIDER_CATALOG" in provider_manifest_text
    admin_js = (api_root / "admin_static" / "admin.js").read_text(encoding="utf-8")
    assert "function providerName" not in admin_js
    assert "display_name || provider.provider_id" in admin_js

    entrypoints_imports = set(
        _imports_from(
            repo_root / "src" / "free_claude_code" / "cli" / "entrypoints.py", repo_root
        )
    )
    assert "free_claude_code.config.env_template" in entrypoints_imports
    assert "_load_env_template" not in (
        repo_root / "src" / "free_claude_code" / "cli" / "entrypoints.py"
    ).read_text(encoding="utf-8")


def test_messaging_transcript_uses_package_owners() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    messaging_root = repo_root / "src" / "free_claude_code" / "messaging"
    transcript_root = messaging_root / "transcript"

    assert not (messaging_root / "transcript.py").exists()
    for filename in {
        "__init__.py",
        "buffer.py",
        "context.py",
        "renderer.py",
        "segments.py",
        "subagents.py",
    }:
        assert (transcript_root / filename).exists()

    init_text = (transcript_root / "__init__.py").read_text(encoding="utf-8")
    assert "TranscriptBuffer" in init_text
    assert "RenderCtx" in init_text


def test_messaging_conversation_state_uses_package_owners() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    messaging_root = repo_root / "src" / "free_claude_code" / "messaging"
    trees_root = messaging_root / "trees"
    session_root = messaging_root / "session"

    assert not (messaging_root / "session.py").exists()
    assert not (trees_root / "data.py").exists()
    for filename in {
        "__init__.py",
        "graph.py",
        "manager.py",
        "node.py",
        "processor.py",
        "queue.py",
        "repository.py",
        "runtime.py",
        "snapshot.py",
    }:
        assert (trees_root / filename).exists()
    for filename in {
        "__init__.py",
        "message_log.py",
        "persistence.py",
        "store.py",
    }:
        assert (session_root / filename).exists()

    offenders = _imports_matching(
        [
            messaging_root,
            repo_root / "src" / "free_claude_code" / "api",
            repo_root / "tests",
        ],
        forbidden_prefixes=("free_claude_code.messaging.trees.data",),
    )
    assert offenders == []

    runtime_text = (
        repo_root / "src" / "free_claude_code" / "runtime" / "application.py"
    ).read_text(encoding="utf-8")
    workflow_text = (messaging_root / "workflow.py").read_text(encoding="utf-8")
    for removed_api in {
        "get_all_trees",
        "get_node_mapping",
        "sync_from_tree_data",
        "TreeQueueManager.from_dict",
    }:
        assert removed_api not in runtime_text
        assert removed_api not in workflow_text


def test_messaging_workflow_uses_split_runtime_owners() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    messaging_root = repo_root / "src" / "free_claude_code" / "messaging"
    trees_root = messaging_root / "trees"

    assert not (messaging_root / "handler.py").exists()
    assert not (trees_root / "queue_manager.py").exists()

    for path in {
        messaging_root / "workflow.py",
        messaging_root / "turn_intake.py",
        messaging_root / "node_runner.py",
        messaging_root / "command_context.py",
        trees_root / "manager.py",
        trees_root / "processor.py",
        trees_root / "repository.py",
    }:
        assert path.exists()

    offenders = _imports_matching(
        [
            messaging_root,
            repo_root / "src" / "free_claude_code" / "api",
            repo_root / "smoke",
            repo_root / "tests",
        ],
        forbidden_prefixes=(
            "free_claude_code.messaging.handler",
            "free_claude_code.messaging.trees.queue_manager",
        ),
    )
    assert offenders == []


def test_messaging_platforms_use_shared_outbox_and_voice_flow() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    platforms_root = repo_root / "src" / "free_claude_code" / "messaging" / "platforms"

    assert not (platforms_root / "base.py").exists()
    assert (platforms_root / "ports.py").exists()
    assert (platforms_root / "outbox.py").exists()
    assert (platforms_root / "voice_flow.py").exists()
    assert "def queue_delete_message(" not in (platforms_root / "ports.py").read_text(
        encoding="utf-8"
    )
    assert "def queue_delete_message(" not in (platforms_root / "outbox.py").read_text(
        encoding="utf-8"
    )

    for runtime in {
        platforms_root / "telegram.py",
        platforms_root / "discord.py",
    }:
        text = runtime.read_text(encoding="utf-8")
        assert "PlatformOutbox" not in text
        assert "VoiceNoteFlow" in text
        assert "PendingVoiceRegistry" not in text
        assert "NamedTemporaryFile" not in text

    for messenger in {
        platforms_root / "telegram_io.py",
        platforms_root / "discord_io.py",
    }:
        text = messenger.read_text(encoding="utf-8")
        assert "PlatformOutbox" in text
        assert "def queue_delete_message(" not in text


def test_cli_surfaces_are_explicit_launchers_and_managed_claude() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cli_root = repo_root / "src" / "free_claude_code" / "cli"

    assert not (cli_root / "adapters" / "__init__.py").exists()
    assert not any((cli_root / "adapters").glob("*.py"))
    assert not (cli_root / "session.py").exists()
    assert not (cli_root / "manager.py").exists()
    assert not (cli_root / "codex_model_catalog.py").exists()

    for path in {
        cli_root / "claude_env.py",
        cli_root / "launchers" / "claude.py",
        cli_root / "launchers" / "codex.py",
        cli_root / "launchers" / "codex_model_catalog.py",
        cli_root / "managed" / "claude.py",
        cli_root / "managed" / "session.py",
        cli_root / "managed" / "manager.py",
    }:
        assert path.exists()

    entrypoints_text = (cli_root / "entrypoints.py").read_text(encoding="utf-8")
    assert "launch_claude" not in entrypoints_text
    assert "launch_codex" not in entrypoints_text
    assert "codex_model_catalog" not in entrypoints_text
    assert "_preflight" + "_proxy" not in entrypoints_text
    assert _text_occurrences(repo_root, "_preflight" + "_proxy") == []

    claude_env_text = (cli_root / "claude_env.py").read_text(encoding="utf-8")
    assert 'CLAUDE_CODE_AUTO_COMPACT_WINDOW = "190000"' in claude_env_text
    assert 'CLAUDE_NO_AUTH_SENTINEL = "fcc-no-auth"' in claude_env_text
    managed_claude_text = (cli_root / "managed" / "claude.py").read_text(
        encoding="utf-8"
    )
    assert 'MANAGED_CLAUDE_MODEL_TIER = "opus"' in managed_claude_text
    assert '"managed_model_tier": MANAGED_CLAUDE_MODEL_TIER' in managed_claude_text
    for path in {
        cli_root / "launchers" / "claude.py",
        cli_root / "managed" / "claude.py",
    }:
        text = path.read_text(encoding="utf-8")
        assert '"190000"' not in text
        assert '"fcc-no-auth"' not in text

    messaging_protocols_text = (
        repo_root / "src" / "free_claude_code" / "messaging" / "managed_protocols.py"
    ).read_text(encoding="utf-8")
    assert "class ManagedClaudeSessionProtocol(Protocol)" in messaging_protocols_text
    assert "class ManagedClaudeSession(Protocol)" not in messaging_protocols_text
    assert (
        "class ManagedClaudeSessionManagerProtocol(Protocol)"
        in messaging_protocols_text
    )
    assert "class SessionManagerInterface(Protocol)" not in messaging_protocols_text
    for path in {
        repo_root / "src" / "free_claude_code" / "messaging" / "__init__.py",
        repo_root
        / "src"
        / "free_claude_code"
        / "messaging"
        / "platforms"
        / "__init__.py",
    }:
        text = path.read_text(encoding="utf-8")
        assert '"ManagedClaudeSession"' not in text
        assert "SessionManagerInterface" not in text

    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'fcc-claude = "free_claude_code.cli.launchers.claude:launch"' in pyproject_text
    )
    assert 'fcc-codex = "free_claude_code.cli.launchers.codex:launch"' in pyproject_text


def _imports_matching(
    roots: list[Path], *, forbidden_prefixes: tuple[str, ...]
) -> list[str]:
    offenders: list[str] = []
    repo_root = Path(__file__).resolve().parents[2]
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo_root)
            offenders.extend(
                f"{rel}: {imported}"
                for imported in _imports_from(path, repo_root)
                if imported is not None and _is_forbidden(imported, forbidden_prefixes)
            )
    return sorted(offenders)


def _is_forbidden(name: str, forbidden: tuple[str, ...]) -> bool:
    """Match root modules (``import api``) and submodules (``import free_claude_code.api.x``)."""
    for token in forbidden:
        if not token:
            continue
        root = token.rstrip(".")
        if name == root or name.startswith(f"{root}."):
            return True
    return False


def _module_fqn_from_path(repo_root: Path, path: Path) -> str:
    rel = path.relative_to(repo_root)
    if rel.parts[:2] == _PACKAGE_ROOT.parts:
        rel = Path("free_claude_code", *rel.parts[2:])
    if rel.name == "__init__.py":
        return ".".join(rel.parent.parts) if rel.parent != Path() else rel.parent.name
    return ".".join(rel.with_suffix("").parts)


def _importing_package_parts(repo_root: Path, path: Path) -> list[str]:
    """Package in which this file's module lives (for relative imports)."""
    rel = path.relative_to(repo_root)
    if rel.name == "__init__.py":
        return list(rel.parent.parts)
    fqn = _module_fqn_from_path(repo_root, path)
    parts = fqn.split(".")
    if len(parts) <= 1:
        return []
    return parts[:-1]


def _resolve_relative_import(
    repo_root: Path, path: Path, node: ast.ImportFrom
) -> str | None:
    """Best-effort absolute name for ``from .x`` / ``from ..y`` (level >= 1)."""
    if node.level == 0 and node.module:
        return node.module
    base = _importing_package_parts(repo_root, path)
    for _ in range(node.level - 1):
        if not base:
            return None
        base.pop()
    if not node.module:
        return ".".join(base) if base else None
    return ".".join(base + node.module.split("."))


def _imports_from(path: Path, repo_root: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    imports.append(node.module)
                continue
            if node.module is not None:
                resolved = _resolve_relative_import(repo_root, path, node)
                if resolved:
                    imports.append(resolved)
            else:
                base = _importing_package_parts(repo_root, path).copy()
                for _ in range(node.level - 1):
                    if base:
                        base.pop()
                for alias in node.names:
                    if base:
                        imports.append(".".join([*base, alias.name]))
                    else:
                        imports.append(alias.name)
    return imports


def _text_occurrences(repo_root: Path, needle: str) -> list[str]:
    searchable_paths = [
        repo_root / "src" / "free_claude_code" / "api",
        repo_root / "src" / "free_claude_code" / "cli",
        repo_root / "src" / "free_claude_code" / "config",
        repo_root / "src" / "free_claude_code" / "core",
        repo_root / "src" / "free_claude_code" / "messaging",
        repo_root / "src" / "free_claude_code" / "providers",
        repo_root / "src" / "free_claude_code" / "runtime",
        repo_root / "smoke",
        repo_root / "tests",
        repo_root / ".env.example",
        repo_root / "AGENTS.md",
        repo_root / "README.md",
        repo_root / "pyproject.toml",
    ]
    occurrences: list[str] = []
    for root in searchable_paths:
        paths = root.rglob("*") if root.is_dir() else (root,)
        for path in paths:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if needle in text:
                occurrences.append(str(path.relative_to(repo_root)))
    return sorted(occurrences)
