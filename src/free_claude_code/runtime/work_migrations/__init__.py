"""Ordered immutable schema migrations for the Work registry."""

from collections.abc import Callable
from dataclasses import dataclass
from sqlite3 import Connection

from . import v001_initial


@dataclass(frozen=True, slots=True)
class WorkMigration:
    version: int
    apply: Callable[[Connection], None]


MIGRATIONS = (WorkMigration(version=1, apply=v001_initial.apply),)
