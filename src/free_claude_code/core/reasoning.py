"""Provider-neutral reasoning intent."""

from dataclasses import dataclass
from enum import StrEnum


class ReasoningControl(StrEnum):
    """Whether a request explicitly controls reasoning computation."""

    DEFAULT = "default"
    OFF = "off"
    ON = "on"


class ReasoningEffort(StrEnum):
    """Named reasoning effort understood at the FCC application boundary."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class ReasoningPolicy:
    """Resolved client and configuration intent passed to one provider.

    ``control`` and ``effort`` remain independent because clients may set an
    overall effort while separately disabling extended thinking. Providers
    translate the representable subset without changing the original intent.
    """

    control: ReasoningControl = ReasoningControl.DEFAULT
    effort: ReasoningEffort | None = None
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.budget_tokens is not None and (
            not isinstance(self.budget_tokens, int)
            or isinstance(self.budget_tokens, bool)
            or self.budget_tokens <= 0
        ):
            raise ValueError("Reasoning budget must be a positive integer.")
        if self.budget_tokens is not None and self.control is not ReasoningControl.ON:
            raise ValueError("A reasoning budget requires reasoning control to be on.")

    @classmethod
    def provider_default(cls) -> ReasoningPolicy:
        """Leave reasoning computation to the provider."""

        return cls()

    @classmethod
    def off(cls) -> ReasoningPolicy:
        """Explicitly disable reasoning computation and output."""

        return cls(control=ReasoningControl.OFF)

    @classmethod
    def on(
        cls,
        *,
        effort: ReasoningEffort | None = None,
        budget_tokens: int | None = None,
    ) -> ReasoningPolicy:
        """Explicitly enable reasoning with optional client controls."""

        return cls(
            control=ReasoningControl.ON,
            effort=effort,
            budget_tokens=budget_tokens,
        )

    @property
    def output_enabled(self) -> bool:
        """Return whether provider reasoning may be exposed to the client."""

        return self.control is not ReasoningControl.OFF

    @property
    def requests_reasoning(self) -> bool:
        """Return whether the request explicitly asks the provider to reason."""

        return self.control is not ReasoningControl.OFF and (
            self.control is ReasoningControl.ON
            or self.effort is not None
            or self.budget_tokens is not None
        )


DEFAULT_REASONING_POLICY = ReasoningPolicy.provider_default()
