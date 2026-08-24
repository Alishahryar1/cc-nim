"""Request-scoped reversible tool identity for OpenAI transports."""

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from free_claude_code.core.inference import (
    CustomTool,
    InferenceRequest,
    ToolCallItem,
    ToolCallKind,
    ToolChoiceMode,
)

OPENAI_TOOL_NAME_MAX_LENGTH = 64
_ALIAS_DIGEST_LENGTH = 16
_PORTABLE_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_INVALID_TOOL_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class OpenAIToolIdentity:
    """Canonical tool identity represented by one upstream wire name."""

    kind: ToolCallKind
    name: str
    namespace: str | None = None


@dataclass(frozen=True, slots=True)
class OpenAIToolNameCodec:
    """Map canonical identities to deterministic OpenAI-compatible names."""

    _identity_to_wire: dict[OpenAIToolIdentity, str]
    _wire_to_identity: dict[str, OpenAIToolIdentity]
    _aliased_names: frozenset[str]

    @classmethod
    def from_request(cls, request: InferenceRequest) -> OpenAIToolNameCodec:
        """Build the codec from definitions, choice, and prior calls."""

        identities: list[OpenAIToolIdentity] = [
            OpenAIToolIdentity(
                ToolCallKind.CUSTOM
                if isinstance(tool, CustomTool)
                else ToolCallKind.FUNCTION,
                tool.name,
                tool.namespace,
            )
            for tool in request.tools
        ]
        choice = request.tool_choice
        if choice is not None and choice.mode is ToolChoiceMode.SPECIFIC:
            if choice.kind is None or choice.name is None:
                raise ValueError("specific tool choice is missing its identity")
            identities.append(
                OpenAIToolIdentity(choice.kind, choice.name, choice.namespace)
            )
        identities.extend(
            OpenAIToolIdentity(item.kind, item.name, item.namespace)
            for item in request.items
            if isinstance(item, ToolCallItem)
        )
        return cls.from_identities(identities)

    @classmethod
    def from_names(cls, names: Iterable[str]) -> OpenAIToolNameCodec:
        """Build function-tool identities for compatibility-focused unit tests."""

        return cls.from_identities(
            OpenAIToolIdentity(ToolCallKind.FUNCTION, name) for name in names
        )

    @classmethod
    def from_identities(
        cls,
        identities: Iterable[OpenAIToolIdentity],
    ) -> OpenAIToolNameCodec:
        unique = sorted(
            set(identities),
            key=lambda identity: (
                identity.namespace or "",
                identity.name,
                identity.kind.value,
            ),
        )
        candidates = [_readable_identity(identity) for identity in unique]
        counts = Counter(candidates)
        reserved: set[str] = set()
        identity_to_wire: dict[OpenAIToolIdentity, str] = {}
        wire_to_identity: dict[str, OpenAIToolIdentity] = {}
        aliased: set[str] = set()
        for identity, candidate in zip(unique, candidates, strict=True):
            if (
                counts[candidate] == 1
                and _PORTABLE_TOOL_NAME.fullmatch(candidate)
                and candidate not in reserved
            ):
                wire_name = candidate
            else:
                wire_name = _unique_alias(identity, candidate, reserved)
                aliased.add(wire_name)
            reserved.add(wire_name)
            identity_to_wire[identity] = wire_name
            wire_to_identity[wire_name] = identity
        return cls(identity_to_wire, wire_to_identity, frozenset(aliased))

    @property
    def has_aliases(self) -> bool:
        return bool(self._aliased_names)

    def encode(
        self,
        name: str,
        *,
        kind: ToolCallKind = ToolCallKind.FUNCTION,
        namespace: str | None = None,
    ) -> str:
        identity = OpenAIToolIdentity(kind, name, namespace)
        return self._identity_to_wire.get(identity, _readable_identity(identity))

    def decode_identity(self, wire_name: str) -> OpenAIToolIdentity:
        return self._wire_to_identity.get(
            wire_name,
            OpenAIToolIdentity(ToolCallKind.FUNCTION, wire_name),
        )

    def decode(self, wire_name: str) -> str:
        """Return the canonical name for one known wire identity."""

        return self.decode_identity(wire_name).name

    def is_alias(self, value: str) -> bool:
        return value in self._aliased_names

    def is_unchanged_name(self, value: str) -> bool:
        return value in self._wire_to_identity and value not in self._aliased_names

    def is_alias_prefix(self, value: str) -> bool:
        return bool(value) and any(
            alias != value and alias.startswith(value) for alias in self._aliased_names
        )


def _readable_identity(identity: OpenAIToolIdentity) -> str:
    if identity.namespace:
        return f"{identity.namespace}__{identity.name}"
    return identity.name


def _unique_alias(
    identity: OpenAIToolIdentity,
    candidate: str,
    reserved: set[str],
) -> str:
    readable = _INVALID_TOOL_NAME_CHARACTERS.sub("_", candidate).strip("_-") or "tool"
    max_readable_length = OPENAI_TOOL_NAME_MAX_LENGTH - _ALIAS_DIGEST_LENGTH - 1
    readable = readable[:max_readable_length].rstrip("_-") or "tool"
    identity_key = "\0".join(
        (identity.kind.value, identity.namespace or "", identity.name)
    )
    attempt = 0
    while True:
        digest_input = identity_key if attempt == 0 else f"{identity_key}\0{attempt}"
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[
            :_ALIAS_DIGEST_LENGTH
        ]
        alias = f"{readable}_{digest}"
        if alias not in reserved:
            return alias
        attempt += 1
