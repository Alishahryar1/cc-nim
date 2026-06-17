"""Agent Governor — runtime safety for weak-RLHF model agentic loops.

The governor inspects each ``/v1/messages`` request, looks at conversation
history and the tool list, and either:

* lets the request through unchanged,
* augments it (cull tools, prepend system hints, inject reflection turn), or
* aborts with a forced "task budget exhausted" SSE response.

It exists because open-weight 80B–120B models (Qwen, Nemotron, DeepSeek, Kimi,
GLM, GPT-OSS, Mistral, Llama) have weaker termination judgment than RLHF-tuned
proprietary models (Claude, GPT-5). Without intervention they tool-loop forever
and never emit ``stop_reason=end_turn``.

Design goal: completely transparent for strong models (auto-bypass via model id
pattern). For weak models, structurally constrain the agent loop so it cannot
run away overnight.
"""

from __future__ import annotations

from .config import GovernorConfig, load_governor_config
from .governor import AgentGovernor, GovernorAction, Verdict

__all__ = [
    "AgentGovernor",
    "GovernorAction",
    "GovernorConfig",
    "Verdict",
    "load_governor_config",
]
