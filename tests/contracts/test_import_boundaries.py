"""Package import contract tests (static AST; dynamic ``importlib`` loads are not scanned)."""

from __future__ import annotations

import ast
from pathlib import Path

# `api` may only import this narrow ``providers`` surface (see AGENTS.md).
_API_ALLOWED_PROVIDER_MODULES = frozenset(
    {
        "providers",
        "providers.base",
        "providers.exceptions",
        "providers.registry",
    }
)


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
    from providers.registry import PROVIDER_DESCRIPTORS, PROVIDER_FACTORIES

    assert tuple(PROVIDER_CATALOG.keys()) == SUPPORTED_PROVIDER_IDS
    assert PROVIDER_DESCRIPTORS is PROVIDER_CATALOG
    assert set(SUPPORTED_PROVIDER_IDS) == set(PROVIDER_FACTORIES)


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
