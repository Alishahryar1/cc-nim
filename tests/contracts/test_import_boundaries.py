"""Package import contract tests (static AST; dynamic ``importlib`` loads are not scanned)."""

import ast
from pathlib import Path

# `api` may only import this narrow ``providers`` surface (see AGENTS.md).
_API_ALLOWED_PROVIDER_MODULES = frozenset(
    {
        "providers",
        "providers.base",
        "providers.exceptions",
        "providers.runtime",
    }
)


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


def test_api_and_messaging_do_not_import_provider_common() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "providers" / "common").exists()
    offenders = _imports_matching(
        [repo_root / "api", repo_root / "messaging"],
        forbidden_prefixes=("providers.common",),
    )

    assert offenders == []


def test_provider_adapters_do_not_import_runtime_layers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _imports_matching(
        [repo_root / "providers"],
        forbidden_prefixes=("api.", "messaging.", "cli."),
    )

    assert offenders == []


def test_core_does_not_import_product_packages() -> None:
    """Neutral ``core`` must stay independent of API, workers, and providers."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _imports_matching(
        [repo_root / "core"],
        forbidden_prefixes=(
            "api.",
            "messaging.",
            "cli.",
            "smoke.",
            "providers.",
            "config.",
        ),
    )
    assert offenders == []


def test_provider_catalog_is_single_source_for_supported_ids() -> None:
    from config.provider_catalog import PROVIDER_CATALOG, SUPPORTED_PROVIDER_IDS
    from providers.runtime import PROVIDER_FACTORIES

    assert tuple(PROVIDER_CATALOG.keys()) == SUPPORTED_PROVIDER_IDS
    assert set(SUPPORTED_PROVIDER_IDS) == set(PROVIDER_FACTORIES)


def test_provider_runtime_replaces_old_registry_module() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert not (repo_root / "providers" / "registry.py").exists()
    assert (repo_root / "providers" / "runtime" / "runtime.py").exists()
    assert (repo_root / "providers" / "runtime" / "factory.py").exists()
    assert (repo_root / "providers" / "runtime" / "discovery.py").exists()

    offenders = _imports_matching(
        [repo_root / "api", repo_root / "tests", repo_root / "smoke"],
        forbidden_prefixes=("providers.registry",),
    )
    assert offenders == []


def test_config_does_not_import_non_config_packages() -> None:
    """Settings and env handling must not depend on transport or protocol layers."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders = _imports_matching(
        [repo_root / "config"],
        forbidden_prefixes=(
            "api.",
            "messaging.",
            "cli.",
            "smoke.",
            "providers.",
            "core.",
        ),
    )
    assert offenders == []


def test_settings_stays_schema_only() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_root = repo_root / "config"

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


_MESSAGING_ALLOWED_PROVIDER_MODULES = frozenset({"providers.nvidia_nim.voice"})


def test_messaging_does_not_import_disallowed_modules() -> None:
    """Messaging is wired by ``api.runtime``; narrow provider imports only for NIM voice ASR."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in (repo_root / "messaging").rglob("*.py"):
        for imported in _imports_from(path, repo_root):
            if imported is None:
                continue
            if (
                imported == "api"
                or imported.startswith("api.")
                or imported == "cli"
                or imported.startswith("cli.")
                or imported == "smoke"
                or imported.startswith("smoke.")
            ):
                rel = path.relative_to(repo_root)
                offenders.append(f"{rel}: {imported}")
            elif imported.startswith("providers."):
                if imported in _MESSAGING_ALLOWED_PROVIDER_MODULES:
                    continue
                rel = path.relative_to(repo_root)
                offenders.append(f"{rel}: {imported}")

    assert sorted(offenders) == []


def test_api_may_only_import_narrow_provider_facade() -> None:
    """HTTP layer must not depend on per-adapter provider subpackages."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in (repo_root / "api").rglob("*.py"):
        for imported in _imports_from(path, repo_root):
            if imported is None or not imported.startswith("providers"):
                continue
            if imported in _API_ALLOWED_PROVIDER_MODULES:
                continue
            if imported.startswith("providers."):
                rel = path.relative_to(repo_root)
                offenders.append(f"{rel}: {imported}")
    assert sorted(offenders) == []


def test_removed_openrouter_rollback_transport_stays_removed() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert not (repo_root / "providers" / "open_router" / "chat_request.py").exists()
    assert _text_occurrences(repo_root, "OpenRouter" + "ChatProvider") == []
    assert _text_occurrences(repo_root, "OPENROUTER" + "_TRANSPORT") == []


def test_provider_transports_live_under_transport_family_packages() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    providers_root = repo_root / "providers"

    assert not (providers_root / "openai_compat.py").exists()
    assert not (providers_root / "anthropic_messages.py").exists()
    assert (providers_root / "transports" / "openai_chat" / "transport.py").exists()
    assert (
        providers_root / "transports" / "anthropic_messages" / "transport.py"
    ).exists()

    offenders = _imports_matching(
        [providers_root, repo_root / "tests"],
        forbidden_prefixes=(
            "providers.openai_compat",
            "providers.anthropic_messages",
        ),
    )
    assert offenders == []


def test_provider_request_policy_lives_with_transport_families() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    providers_root = repo_root / "providers"

    deleted_request_modules = (
        "providers.cerebras.request",
        "providers.deepseek.request",
        "providers.fireworks.request",
        "providers.gemini.request",
        "providers.groq.request",
        "providers.kimi.request",
        "providers.mistral.request",
        "providers.nvidia_nim.request",
        "providers.opencode.request",
        "providers.open_router.request",
        "providers.zai.request",
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


def test_anthropic_stream_engine_owns_provider_stream_state() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    anthropic_root = repo_root / "core" / "anthropic"
    streaming_root = anthropic_root / "streaming"

    for removed in {
        "sse.py",
        "emitted_sse_tracker.py",
        "stream_recovery.py",
        "stream_recovery_session.py",
    }:
        assert not (anthropic_root / removed).exists()

    for filename in {
        "__init__.py",
        "emitter.py",
        "ledger.py",
        "lifecycle.py",
        "recovery.py",
    }:
        assert (streaming_root / filename).exists()

    forbidden = (
        "SSEBuilder",
        "ContentBlockManager",
        "ToolCallState",
        "EmittedNativeSseTracker",
        "StreamRecoverySession",
        "OpenAIChatStreamRunner",
        "AnthropicMessagesStreamRunner",
    )
    offenders: list[str] = []
    for path in [
        *anthropic_root.rglob("*.py"),
        *(repo_root / "providers" / "transports").rglob("*.py"),
    ]:
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(repo_root)}: {name}"
            for name in forbidden
            if name in text
        )
    assert sorted(offenders) == []


def test_openai_responses_uses_adapter_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    responses_root = repo_root / "core" / "openai_responses"
    responses_streaming_root = responses_root / "streaming"
    api_root = repo_root / "api"
    handlers_root = api_root / "handlers"

    assert not (repo_root / "api" / "services.py").exists()
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
    assert (
        "from core.openai_responses import OpenAIResponsesAdapter"
        in responses_handler_text
    )
    routes_text = (repo_root / "api" / "routes.py").read_text(encoding="utf-8")
    assert "ApiRequestPipeline" not in routes_text
    assert "request_pipeline" not in routes_text
    assert "from .handlers import" in routes_text
    assert "api.services" not in routes_text
    for old_helper in {
        "responses_request_to_anthropic_payload",
        "anthropic_message_response_to_openai_response",
        "iter_anthropic_sse_as_openai_responses",
        "collect_openai_response_from_anthropic_sse",
        "iter_message_response_as_openai_responses",
    }:
        assert old_helper not in responses_handler_text

    offenders: list[str] = []
    for path in (repo_root / "api").rglob("*.py"):
        for imported in _imports_from(path, repo_root):
            if imported is not None and imported.startswith("core.openai_responses."):
                rel = path.relative_to(repo_root)
                offenders.append(f"{rel}: {imported}")
    assert sorted(offenders) == []

    adapter_importers: list[str] = []
    for path in (repo_root / "api").rglob("*.py"):
        imports = set(_imports_from(path, repo_root))
        if "core.openai_responses" in imports:
            adapter_importers.append(path.relative_to(repo_root).as_posix())
    assert sorted(adapter_importers) == ["api/handlers/responses.py"]

    response_handler_imports = set(_imports_from(responses_handler, repo_root))
    for forbidden in {
        "api.optimization_handlers",
        "api.detection",
        "api.web_tools",
    }:
        assert forbidden not in response_handler_imports

    provider_execution_text = (api_root / "provider_execution.py").read_text(
        encoding="utf-8"
    )
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
    api_root = repo_root / "api"
    admin_config_root = api_root / "admin_config"

    assert not (api_root / "admin_config.py").exists()
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
    assert "api.admin_config" not in routes_imports
    for expected in {
        "api.admin_config.manifest",
        "api.admin_config.persistence",
        "api.admin_config.status",
        "api.admin_config.values",
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
        _imports_from(repo_root / "cli" / "entrypoints.py", repo_root)
    )
    assert "config.env_template" in entrypoints_imports
    assert "_load_env_template" not in (repo_root / "cli" / "entrypoints.py").read_text(
        encoding="utf-8"
    )


def test_messaging_transcript_uses_package_owners() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    messaging_root = repo_root / "messaging"
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
    messaging_root = repo_root / "messaging"
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
        [messaging_root, repo_root / "api", repo_root / "tests"],
        forbidden_prefixes=("messaging.trees.data",),
    )
    assert offenders == []

    runtime_text = (repo_root / "api" / "runtime.py").read_text(encoding="utf-8")
    for removed_api in {
        "get_all_trees",
        "get_node_mapping",
        "sync_from_tree_data",
        "TreeQueueManager.from_dict",
    }:
        assert removed_api not in runtime_text


def test_messaging_workflow_uses_split_runtime_owners() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    messaging_root = repo_root / "messaging"
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
        [messaging_root, repo_root / "api", repo_root / "smoke", repo_root / "tests"],
        forbidden_prefixes=(
            "messaging.handler",
            "messaging.trees.queue_manager",
        ),
    )
    assert offenders == []


def test_messaging_platforms_use_shared_outbox_and_voice_flow() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    platforms_root = repo_root / "messaging" / "platforms"

    assert not (platforms_root / "base.py").exists()
    assert (platforms_root / "ports.py").exists()
    assert (platforms_root / "outbox.py").exists()
    assert (platforms_root / "voice_flow.py").exists()

    for runtime in {
        platforms_root / "telegram.py",
        platforms_root / "discord.py",
    }:
        text = runtime.read_text(encoding="utf-8")
        assert "PlatformOutbox" not in text
        assert "VoiceNoteFlow" in text
        assert "from ..voice" not in text
        assert "NamedTemporaryFile" not in text

    for messenger in {
        platforms_root / "telegram_io.py",
        platforms_root / "discord_io.py",
    }:
        text = messenger.read_text(encoding="utf-8")
        assert "PlatformOutbox" in text


def test_cli_surfaces_are_explicit_launchers_and_managed_claude() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cli_root = repo_root / "cli"

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
    for path in {
        cli_root / "launchers" / "claude.py",
        cli_root / "managed" / "claude.py",
    }:
        text = path.read_text(encoding="utf-8")
        assert '"190000"' not in text
        assert '"fcc-no-auth"' not in text

    messaging_protocols_text = (
        repo_root / "messaging" / "managed_protocols.py"
    ).read_text(encoding="utf-8")
    assert "class ManagedClaudeSessionProtocol(Protocol)" in messaging_protocols_text
    assert "class ManagedClaudeSession(Protocol)" not in messaging_protocols_text
    assert (
        "class ManagedClaudeSessionManagerProtocol(Protocol)"
        in messaging_protocols_text
    )
    assert "class SessionManagerInterface(Protocol)" not in messaging_protocols_text
    for path in {
        repo_root / "messaging" / "__init__.py",
        repo_root / "messaging" / "platforms" / "__init__.py",
    }:
        text = path.read_text(encoding="utf-8")
        assert '"ManagedClaudeSession"' not in text
        assert "SessionManagerInterface" not in text

    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'fcc-claude = "cli.launchers.claude:launch"' in pyproject_text
    assert 'fcc-codex = "cli.launchers.codex:launch"' in pyproject_text


def _imports_matching(
    roots: list[Path], *, forbidden_prefixes: tuple[str, ...]
) -> list[str]:
    offenders: list[str] = []
    repo_root = roots[0].parent
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(root.parent)
            offenders.extend(
                f"{rel}: {imported}"
                for imported in _imports_from(path, repo_root)
                if imported is not None and _is_forbidden(imported, forbidden_prefixes)
            )
    return sorted(offenders)


def _is_forbidden(name: str, forbidden: tuple[str, ...]) -> bool:
    """Match root modules (``import api``) and submodules (``import api.x``)."""
    for token in forbidden:
        if not token:
            continue
        root = token.rstrip(".")
        if name == root or name.startswith(f"{root}."):
            return True
    return False


def _module_fqn_from_path(repo_root: Path, path: Path) -> str:
    rel = path.relative_to(repo_root)
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
        repo_root / "api",
        repo_root / "cli",
        repo_root / "config",
        repo_root / "core",
        repo_root / "messaging",
        repo_root / "providers",
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
