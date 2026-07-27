from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ReasoningEncoder(Protocol):
    def encode(self, body: dict[str, Any], policy: Any) -> None: ...
