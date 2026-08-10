"""Dynamic model fallback chain for Free Claude Code.

When a requested model (default, opus/sonnet/haiku role, or a direct
``provider/model``) fails by capacity (quota / rate-limit / overload), retry
other ``provider/model`` refs discovered from the API keys that are
configured, ordered by heuristic potency (most -> least potent). This is a
best-effort, local-only behavior that only switches before the first streamed
chunk is delivered to the client.

NOTE: this module patches the editable Free Claude Code install at
``/private/tmp/free-claude-code``. Changes are lost if fcc is reinstalled or
updated; re-apply after any upgrade.
"""

import re
from collections.abc import Iterable

from loguru import logger

from free_claude_code.application.model_metadata import ProviderModelInfo

# Potency is name-based only: ProviderModelInfo carries no capability metadata.
# A model's size CLASS dominates the score; version and parameter count only
# break ties within a class. Markers are matched against whole letter-tokens
# (not substrings) so "flash-lite" reads as small and "gemini" is never
# mistaken for "mini".
# "gemma" is intentionally NOT a small marker: Gemma 4 26B/31B are mid-size
# models whose parameter count places them correctly. Their class is decided by
# the "26b"/"31b" in the name, not the family word.
# "oss" (open-source) is likewise NOT a size marker: gpt-oss-120b is a large
# model and must be classed by its "120b", not sunk as if it were small.
_SMALL_TOKENS = frozenset(
    {
        "lite",
        "mini",
        "nano",
        "micro",
        "tiny",
        "small",
        "haiku",
        "xs",
        "phi",
    }
)
_LARGE_TOKENS = frozenset({"opus", "ultra", "max", "pro", "large", "nemotron"})
_MEDIUM_TOKENS = frozenset(
    {"flash", "sonnet", "plus", "medium", "coder", "command", "mistral"}
)
# Capability variant within one model family, used to break ties inside a
# curated tier: a "pro/max/ultra" build outranks the base build, which outranks
# a reduced "flash/lite/mini/air" build (e.g. deepseek-v4-pro > deepseek-v4 >
# deepseek-v4-flash). These are family-suffixes, not size classes.
_HIGH_VARIANT_TOKENS = frozenset({"pro", "max", "ultra", "plus"})
_LOW_VARIANT_TOKENS = frozenset(
    {"flash", "lite", "mini", "nano", "air", "tiny", "micro", "small"}
)

_CLASS_SMALL = 1
_CLASS_MEDIUM = 2
_CLASS_LARGE = 3

_WORD_RE = re.compile(r"[a-z]+")
# Explicit parameter count in the name, e.g. "550b", "120b", "8b".
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z0-9])")
# Family version, e.g. gemini-3.1, gpt-5, claude-opus-4.8, qwen3.
_VERSION_RE = re.compile(
    r"(?:gpt-|gemini-|claude-[a-z]+-|grok-|llama-|qwen-?|deepseek-|"
    r"mistral-|hy|nemotron-|command-r-?|glm-|kimi-k|minimax-m|-v|/v)(\d+(?:\.\d+)?)"
)


def _size_class(ref: str, params: float) -> int:
    """Coarse capability class from name tokens and any explicit param count."""
    words = set(_WORD_RE.findall(ref))
    if words & _SMALL_TOKENS:
        return _CLASS_SMALL
    if params >= 100 or (words & _LARGE_TOKENS):
        return _CLASS_LARGE
    if params and params <= 9:
        return _CLASS_SMALL
    if words & _MEDIUM_TOKENS:
        return _CLASS_MEDIUM
    return _CLASS_MEDIUM


# Model kinds that are NOT chat-completion models; exclude from the chain.
_NON_CHAT_MARKERS = (
    "embedding",
    "audio",
    "tts",
    "whisper",
    "transcribe",
    "rerank",
    "reranker",
    "moderation",
    "moderate",
    "image",
    "/img",
    "dall",
    "vision",
    "guard",
    "classify",
    "detect",
    "speech",
    "voxtral",
    "orpheus",
    "lyria",
    "veo",
    # Agent/specialized Gemini models that only serve the Interactions or Live
    # API, not chat completions - unusable as derivation candidates.
    "antigravity",
    "deep-research",
    "computer-use",
    "robotics",
    "-live",
    "live-",
)


def is_chat_model(model_ref: str) -> bool:
    """Return whether a ``provider/model`` ref looks like a chat model."""

    ref = model_ref.lower()
    return not any(marker in ref for marker in _NON_CHAT_MARKERS)


def is_free_candidate(model_ref: str) -> bool:
    """Return whether a ref is a good automatic derivation candidate.

    Automatic derivation must never spend money nor waste attempts on models
    that reliably reject real requests:

    - OpenRouter mixes free and paid models with no price field; only ``:free``
      ids are safe.
    - Gemini serves Flash/Flash-Lite free even with billing enabled, but its
      ``pro`` models are paid-only (excluded), and its ``gemma`` models cap
      input at 16K tokens/minute - far below a real coding request - so they
      reject almost everything and are excluded too.

    Other providers are single-tier (free, or bring-your-own-key at a rate the
    user already accepted), so they pass through.
    """

    if model_ref.startswith("open_router/"):
        return model_ref.endswith(":free")
    if model_ref.startswith("gemini/"):
        ref = model_ref.lower()
        return "pro" not in ref and "gemma" not in ref
    return True


# Curated order by SIMILARITY to the Claude frontier experience (Opus 4.8 /
# Fable 5) inside a coding agent - NOT raw coding-benchmark score. Weighted by:
# native "thinking", agentic/tool-use (the point of Claude Code), real repo
# coding, broad generality, and large context. Narrow coding specialists
# (codestral/devstral, qwen-coder) rank BELOW comparable general frontier models
# because Opus/Fable are generalist reasoners, not narrow coders; models without
# native thinking (gpt-4.1/4o, mistral) rank below thinkers. Grounded in 2026
# agentic/coding benchmarks (SWE-bench Verified/Pro, tau-bench/tool-use, Aider
# Polyglot). Opinionated and time-sensitive - update as models change.
# Parameter count, version, and pro/flash variant only break ties within a tier;
# the name/size heuristic ranks unknown families below the curated band.
_CODING_ORDER = (
    # Anthropic frontier - the experience to match (the user runs Opus and Fable).
    r"claude-opus",
    r"claude-fable",
    r"claude-sonnet",
    # Proprietary frontier with native thinking - most Claude-like non-Claude.
    r"gpt-5",
    # Closest free models: native thinking + agentic tool-use + generalist +
    # large context. DeepSeek-V4 (~1M ctx, top generality) edges GLM, which
    # matches Opus on tool-use benchmarks; both lead the open-weight frontier.
    r"deepseek-v\d",
    r"glm-\d|zai-glm",
    # Elite agentic/tool-use with thinking, one notch down on context or breadth.
    r"kimi",
    r"minimax",
    # General Qwen3 (reasoning + tool-use) - NOT the narrow coder build far below.
    r"qwen-?3(?![\w.-]*coder)",
    # Frontier-adjacent thinkers with a Claude-style or availability gap.
    r"grok-[4-9]",
    r"deepseek-r\d",
    r"(?:^|[/-])o[1-9](?:[/-]|$)",
    # Capable generalists WITHOUT native thinking, or narrow specialists: these
    # diverge from Claude's reason-then-act loop.
    r"gpt-4\.1",
    r"qwen[\d.]*-?coder",
    r"magistral",
    r"gpt-4o",
    r"command-a|command-r",
    r"llama-4",
    # Most different from the Claude experience: narrow coders, older or
    # non-thinking generalists.
    r"codestral|devstral",
    r"nemotron",
    r"llama-3\.3",
    r"llama-3\.1",
    r"mistral-(?:large|medium)",
    r"gpt-oss",
    r"gemma-4",
    # Gemini: capable with 1M context (kept as the safety net for giant /compact
    # requests) but deliberately low per the user's feedback that it fits their
    # coding style less well.
    r"gemini-3(?:\.\d+)?-pro",
    r"gemini-3(?:\.\d+)?-flash",
    r"gemini-2\.5-pro",
    r"gemini-2\.5-flash",
    r"gemini-2\.0-flash",
    r"gemini[\w.-]*flash",
)
_CODING_PATTERNS = tuple(re.compile(pattern) for pattern in _CODING_ORDER)


def _size_score(ref: str) -> int:
    """Name/size heuristic used to rank models the curated table does not know."""

    sizes = [float(match) for match in _SIZE_RE.findall(ref)]
    params = max(sizes) if sizes else 0.0
    size_class = _size_class(ref, params)
    version_match = _VERSION_RE.search(ref)
    version = float(version_match.group(1)) if version_match else 0.0
    return int(size_class * 10000 + min(version, 99) * 100 + min(params, 999))


def _coding_index(ref: str) -> int | None:
    """Index of the first curated family a ref matches, or None if unknown."""

    for index, pattern in enumerate(_CODING_PATTERNS):
        if pattern.search(ref):
            return index
    return None


def rank_potency(model_ref: str) -> int:
    """Score for ordering candidates strongest -> weakest for coding.

    The curated `_CODING_ORDER` table decides the order of known frontier
    families so a big open-weight model never outranks a frontier one. A
    `-mini/-nano/-lite` variant of a known family sits below its full-size
    siblings, and families the table does not know fall back to the pure size
    heuristic, always below the curated band.

    Flash/lite/mini builds of a top family are demoted one curated tier so a
    missing ``*-pro`` does not make derivation prefer that family's flash over
    the next family's full/pro model (e.g. deepseek-v4-flash must not beat
    glm-4.7). Gemini is exempt: its flash models are the free coding path
    (pro is paid) and already sit at the bottom of the curated band.
    """

    ref = model_ref.lower()
    index = _coding_index(ref)
    if index is None:
        return _size_score(ref)
    # Within a curated tier, order by newer version, then stronger capability
    # variant (pro/max/ultra > base > flash/lite/mini), then raw parameter count
    # (nemotron-ultra-550b > nemotron-super-120b). The tier always dominates the
    # tiebreak. Version+variant fix the old alphabetical tiebreak that ranked
    # deepseek-v4-flash above the stronger deepseek-v4-pro.
    words = set(_WORD_RE.findall(ref))
    sizes = [float(match) for match in _SIZE_RE.findall(ref)]
    params = int(max(sizes)) if sizes else 0
    version_match = _VERSION_RE.search(ref)
    version = float(version_match.group(1)) if version_match else 0.0
    if words & _HIGH_VARIANT_TOKENS:
        variant = 2
    elif words & _LOW_VARIANT_TOKENS:
        variant = 0
    else:
        variant = 1
    # Drop flash/lite/mini one tier so the next family's full/pro is tried
    # before a reduced build of the current family. Skip Gemini (see docstring).
    effective_index = index
    if variant == 0 and "gemini" not in words:
        effective_index = min(index + 1, len(_CODING_PATTERNS) - 1)
    curated = (
        (len(_CODING_PATTERNS) - effective_index) * 100_000
        + int(min(version, 9) * 10_000)
        + variant * 2_000
        + min(params, 999)
    )
    is_small_variant = bool(words & _SMALL_TOKENS)
    return (10_000_000 if is_small_variant else 20_000_000) + curated


def eligible_candidate_refs(
    model_infos: Iterable[ProviderModelInfo],
) -> list[str]:
    """Discovered chat models the fallback may auto-substitute, most potent first.

    Excludes non-chat models (embeddings, image, TTS, ...) and paid OpenRouter
    models - an automatic substitution the user never requested must never
    reach a model that could cost money. This is the single source of truth
    for "models the derivation system can use", shared by the fallback chain
    and the admin Usage tab so they never drift apart.
    """

    seen: set[str] = set()
    refs: list[str] = []
    for info in model_infos:
        ref = info.model_id
        if ref in seen:
            continue
        if not is_chat_model(ref) or not is_free_candidate(ref):
            continue
        seen.add(ref)
        refs.append(ref)

    refs.sort(key=lambda ref: (-rank_potency(ref), ref))
    return refs


# A request at/over the common 128K window (compaction sends ~190K) only fits
# models with a bigger window. Below this it stays on the normal coding order.
LARGE_REQUEST_TOKENS = 130_000
# Known large-window families, ordered by fitness for a giant request (window +
# speed on huge inputs). Gemini leads: 1M window and fast at large context - the
# opposite of deepseek-v4-pro, which the normal order leads with but is slow on
# huge inputs. deepseek-v4 (256K) is verified; the rest are likely-large
# candidates - if one turns out smaller it just overflows and is parked, so
# listing it here only changes ORDER, never correctness.
_LARGE_CONTEXT_ORDER = (
    r"gemini[\w.-]*flash",
    r"gemini",
    r"deepseek-v4",
    r"kimi",
    r"qwen-?3\.5",
    r"minimax",
    r"llama-4",
)
_LARGE_CONTEXT_PATTERNS = tuple(re.compile(pattern) for pattern in _LARGE_CONTEXT_ORDER)


def _large_context_rank(ref: str) -> int | None:
    """Index of the first large-window family a ref matches, or None if unknown."""

    lowered = ref.lower()
    for index, pattern in enumerate(_LARGE_CONTEXT_PATTERNS):
        if pattern.search(lowered):
            return index
    return None


def prioritize_large_context(refs: list[str], input_tokens: int) -> list[str]:
    """Front-load known large-window models when the request is giant.

    A ~190K compaction request overflows every 128K model, so leading the chain
    with big-window models (gemini first for speed, then deepseek-v4, ...) keeps
    /compact from stalling on a slow model or burning turns on small ones. Below
    the threshold the order is unchanged. Stable: within each group the incoming
    potency order is preserved.
    """

    if input_tokens < LARGE_REQUEST_TOKENS:
        return refs
    ranked = [(i, r, _large_context_rank(r)) for i, r in enumerate(refs)]
    large = sorted((t for t in ranked if t[2] is not None), key=lambda t: (t[2], t[0]))
    rest = [t for t in ranked if t[2] is None]
    return [r for _, r, _ in large] + [r for _, r, _ in rest]


def build_fallback_chain(
    primary_ref: str,
    model_infos: Iterable[ProviderModelInfo],
) -> list[str]:
    """Return ``provider/model`` refs to try when ``primary_ref`` fails.

    ``primary_ref`` (whatever model the request originally resolved to - the
    default, opus/sonnet/haiku role, or a direct ``provider/model`` request) is
    always first; the eligible discovered candidates follow in descending
    heuristic potency.
    """

    rest = [ref for ref in eligible_candidate_refs(model_infos) if ref != primary_ref]
    chain = [primary_ref, *rest]
    logger.debug("MODEL FALLBACK CHAIN ({} candidates): {}", len(chain), chain)
    return chain
