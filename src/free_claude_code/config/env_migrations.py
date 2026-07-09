"""One-time dotenv key migrations for FCC-owned config files."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .env_files import explicit_env_path, repo_env_path
from .paths import managed_env_path

LEGACY_HUGGINGFACE_TOKEN_ENV = "HF_TOKEN"
HUGGINGFACE_API_KEY_ENV = "HUGGINGFACE_API_KEY"

_DOTENV_ASSIGNMENT_RE = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P<suffix>\s*(?:=|$))"
)


@dataclass(frozen=True, slots=True)
class EnvKeyMigration:
    """A dotenv key rename migration."""

    old_key: str
    new_key: str


HUGGINGFACE_TOKEN_MIGRATION = EnvKeyMigration(
    old_key=LEGACY_HUGGINGFACE_TOKEN_ENV,
    new_key=HUGGINGFACE_API_KEY_ENV,
)


def migrate_owned_env_files() -> tuple[Path, ...]:
    """Apply key migrations to repo and managed dotenv files."""

    return tuple(
        path.resolve()
        for path in _unique_paths((repo_env_path(), managed_env_path()))
        if migrate_env_key_in_file(path, HUGGINGFACE_TOKEN_MIGRATION)
    )


def explicit_env_file_huggingface_warning(
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return a warning when an explicit env file still uses ``HF_TOKEN``."""

    path = explicit_env_path(env)
    if path is None or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not env_text_needs_migration(text, HUGGINGFACE_TOKEN_MIGRATION):
        return None
    return (
        f"{LEGACY_HUGGINGFACE_TOKEN_ENV} is set in explicit FCC_ENV_FILE {path}. "
        f"Rename it to {HUGGINGFACE_API_KEY_ENV}; explicit env files are not "
        "rewritten automatically."
    )


def migrate_env_key_in_file(path: Path, migration: EnvKeyMigration) -> bool:
    """Rename a dotenv key in ``path`` when the new key is absent."""

    if not path.is_file():
        return False
    original = path.read_text(encoding="utf-8")
    migrated, changed = migrate_env_key_in_text(original, migration)
    if not changed:
        return False
    path.write_text(migrated, encoding="utf-8")
    return True


def migrate_env_key_in_text(
    text: str,
    migration: EnvKeyMigration,
) -> tuple[str, bool]:
    """Return text with ``old_key`` renamed to ``new_key`` when safe."""

    if _defines_key(text, migration.new_key):
        return text, False

    lines = text.splitlines(keepends=True)
    changed = False
    for index, line in enumerate(lines):
        match = _DOTENV_ASSIGNMENT_RE.match(line)
        if match is None or match.group("key") != migration.old_key:
            continue
        lines[index] = (
            f"{match.group('prefix')}{migration.new_key}{match.group('suffix')}"
            f"{line[match.end() :]}"
        )
        changed = True
    if not changed:
        return text, False
    return "".join(lines), True


def env_text_needs_migration(text: str, migration: EnvKeyMigration) -> bool:
    """Return whether text defines old key without new key."""

    return _defines_key(text, migration.old_key) and not _defines_key(
        text, migration.new_key
    )


def _defines_key(text: str, key: str) -> bool:
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = _DOTENV_ASSIGNMENT_RE.match(line)
        if match is not None and match.group("key") == key:
            return True
    return False


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return tuple(unique)
