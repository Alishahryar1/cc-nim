"""One-time dotenv migrations for FCC-owned config files."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .env_files import explicit_env_path, repo_env_path
from .paths import managed_env_path

LEGACY_HUGGINGFACE_TOKEN_ENV = "HF_TOKEN"
HUGGINGFACE_API_KEY_ENV = "HUGGINGFACE_API_KEY"
LEGACY_OPENCODE_PROXY_ENV = "OPENCODE_PROXY"
OPENCODE_ZEN_PROXY_ENV = "OPENCODE_ZEN_PROXY"
LEGACY_OPENCODE_SMOKE_MODEL_ENV = "FCC_SMOKE_MODEL_OPENCODE"
OPENCODE_ZEN_SMOKE_MODEL_ENV = "FCC_SMOKE_MODEL_OPENCODE_ZEN"

_DOTENV_ASSIGNMENT_RE = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P<suffix>\s*(?:=|$))"
)


@dataclass(frozen=True, slots=True)
class EnvMigration:
    """A migration for one dotenv setting."""

    old_key: str
    new_key: str
    value_map: tuple[tuple[str, str], ...] = ()
    value_prefix_map: tuple[tuple[str, str], ...] = ()


HUGGINGFACE_TOKEN_MIGRATION = EnvMigration(
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
    EnvMigration(
        "ENABLE_MODEL_THINKING",
        "REASONING_POLICY",
        _LEGACY_REASONING_BOOLEAN_MAP,
    ),
    *(
        EnvMigration(
            f"ENABLE_{route}_THINKING",
            f"REASONING_{route}",
            (("", "inherit"), *_LEGACY_REASONING_BOOLEAN_MAP),
        )
        for route in ("FABLE", "OPUS", "SONNET", "HAIKU")
    ),
)

OPENCODE_ZEN_KEY_MIGRATIONS = (
    EnvMigration(LEGACY_OPENCODE_PROXY_ENV, OPENCODE_ZEN_PROXY_ENV),
    EnvMigration(
        LEGACY_OPENCODE_SMOKE_MODEL_ENV,
        OPENCODE_ZEN_SMOKE_MODEL_ENV,
        value_prefix_map=(("opencode/", "opencode_zen/"),),
    ),
)

OPENCODE_ZEN_MODEL_REF_MIGRATIONS = tuple(
    EnvMigration(
        key,
        key,
        value_prefix_map=(("opencode/", "opencode_zen/"),),
    )
    for key in (
        "MODEL",
        "MODEL_FABLE",
        "MODEL_OPUS",
        "MODEL_SONNET",
        "MODEL_HAIKU",
        OPENCODE_ZEN_SMOKE_MODEL_ENV,
    )
)

ENV_MIGRATIONS = (
    HUGGINGFACE_TOKEN_MIGRATION,
    *REASONING_MIGRATIONS,
    *OPENCODE_ZEN_KEY_MIGRATIONS,
    *OPENCODE_ZEN_MODEL_REF_MIGRATIONS,
)


def migrate_owned_env_files() -> tuple[Path, ...]:
    """Apply config migrations to repo and managed dotenv files."""

    changed_paths: list[Path] = []
    for path in _unique_paths((repo_env_path(), managed_env_path())):
        changed = False
        for migration in ENV_MIGRATIONS:
            changed = migrate_env_setting_in_file(path, migration) or changed
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
    actions: list[str] = []
    for migration in pending:
        if migration.old_key != migration.new_key:
            actions.append(f"rename {migration.old_key} to {migration.new_key}")
        actions.extend(
            f"replace {old_prefix} with {new_prefix} in {migration.new_key}"
            for old_prefix, new_prefix in migration.value_prefix_map
        )
    return (
        f"Explicit FCC_ENV_FILE {path} uses retired settings. "
        f"{'; '.join(actions)}; "
        "explicit env files are not rewritten automatically."
    )


def migrate_env_setting_in_file(path: Path, migration: EnvMigration) -> bool:
    """Apply one setting migration to ``path``."""

    if not path.is_file():
        return False
    original = path.read_text(encoding="utf-8")
    migrated, changed = migrate_env_setting_in_text(original, migration)
    if not changed:
        return False
    path.write_text(migrated, encoding="utf-8")
    return True


def migrate_env_setting_in_text(
    text: str,
    migration: EnvMigration,
) -> tuple[str, bool]:
    """Return text with one setting migration applied when safe."""

    if migration.old_key != migration.new_key and _defines_key(text, migration.new_key):
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
        for old_prefix, new_prefix in migration.value_prefix_map:
            remainder = _mapped_prefix_value(remainder, old_prefix, new_prefix)
        migrated_line = (
            f"{match.group('prefix')}{migration.new_key}{match.group('suffix')}"
            f"{remainder}"
        )
        if migrated_line == line:
            continue
        lines[index] = migrated_line
        changed = True
    if not changed:
        return text, False
    return "".join(lines), True


def env_text_needs_migration(text: str, migration: EnvMigration) -> bool:
    """Return whether one setting migration would change text."""

    return migrate_env_setting_in_text(text, migration)[1]


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


def _mapped_prefix_value(value: str, old_prefix: str, new_prefix: str) -> str:
    """Rewrite one simple dotenv value while preserving its original syntax."""

    pattern = rf"^(?P<leading>\s*['\"]?){re.escape(old_prefix)}"
    return re.sub(
        pattern,
        lambda match: f"{match.group('leading')}{new_prefix}",
        value,
        count=1,
    )


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
