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
    value_map: tuple[tuple[str, str], ...] = ()


HUGGINGFACE_TOKEN_MIGRATION = EnvKeyMigration(
    old_key=LEGACY_HUGGINGFACE_TOKEN_ENV,
    new_key=HUGGINGFACE_API_KEY_ENV,
)

_LEGACY_TRUE_VALUES = ("1", "true", "t", "on", "yes", "y")
_LEGACY_FALSE_VALUES = ("0", "false", "f", "off", "no", "n")
_LEGACY_REASONING_BOOLEAN_MAP = (
    *((value, "client") for value in _LEGACY_TRUE_VALUES),
    *((value, "off") for value in _LEGACY_FALSE_VALUES),
)

REASONING_MIGRATIONS = (
    EnvKeyMigration(
        "ENABLE_MODEL_THINKING",
        "REASONING_POLICY",
        _LEGACY_REASONING_BOOLEAN_MAP,
    ),
    *(
        EnvKeyMigration(
            f"ENABLE_{route}_THINKING",
            f"REASONING_{route}",
            (("", "inherit"), *_LEGACY_REASONING_BOOLEAN_MAP),
        )
        for route in ("FABLE", "OPUS", "SONNET", "HAIKU")
    ),
)

ENV_MIGRATIONS = (HUGGINGFACE_TOKEN_MIGRATION, *REASONING_MIGRATIONS)


def migrate_owned_env_files() -> tuple[Path, ...]:
    """Apply key migrations to repo and managed dotenv files."""

    changed_paths: list[Path] = []
    for path in _unique_paths((repo_env_path(), managed_env_path())):
        changed = False
        for migration in ENV_MIGRATIONS:
            changed = migrate_env_key_in_file(path, migration) or changed
        if changed:
            changed_paths.append(path.resolve())
    return tuple(changed_paths)


def explicit_env_file_migration_warning(
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return a warning when an explicit env file uses a retired setting."""

    path = explicit_env_path(env)
    if path is None or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    pending = tuple(
        migration
        for migration in ENV_MIGRATIONS
        if env_text_needs_migration(text, migration)
    )
    if not pending:
        return None
    renames = ", ".join(
        f"{migration.old_key} to {migration.new_key}" for migration in pending
    )
    return (
        f"Explicit FCC_ENV_FILE {path} uses retired settings. Rename {renames}; "
        "explicit env files are not rewritten automatically."
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
        remainder = line[match.end() :]
        if migration.value_map:
            remainder = _mapped_value(remainder, migration.value_map)
        lines[index] = (
            f"{match.group('prefix')}{migration.new_key}{match.group('suffix')}"
            f"{remainder}"
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


def _mapped_value(value: str, mapping: tuple[tuple[str, str], ...]) -> str:
    """Map a simple dotenv value while preserving comments and line endings."""

    line = value.rstrip("\r\n")
    newline = value[len(line) :]
    raw_value, separator, comment = line.partition("#")
    normalized = raw_value.strip().strip("'\"").lower()
    replacement = dict(mapping).get(normalized)
    if replacement is None:
        return value
    suffix = f" #{comment}" if separator else ""
    return f"{replacement}{suffix}{newline}"


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
