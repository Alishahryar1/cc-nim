"""Wire and presentation models for OpenAI Responses-compatible ingress."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


class OpenAIResponsesRequest(BaseModel):
    """Permissive subset of the OpenAI Responses API request shape."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: object = None
    instructions: str | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: object = None
    parallel_tool_calls: bool | None = None
    stream: bool | None = True
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, object] | None = None
    reasoning: dict[str, object] | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    include: object = None
    prompt_cache_key: object = None


@dataclass(frozen=True, slots=True)
class ResponsesPresentationSnapshot:
    """Validated fields echoed in public Responses envelopes."""

    model: str
    parallel_tool_calls: bool
    tool_choice: object
    temperature: float | None
    top_p: float | None
    max_output_tokens: int | None
