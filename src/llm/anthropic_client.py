"""
Native Anthropic Messages API client behind the CaptainClient interface.

Captains, admirals, the draft and the notebooks talk to an LLM through the
OpenRouter-shaped surface of `client.CaptainClient`: OpenAI-style messages and
function tools in, `ToolCall`s out, plus `last_call_meta()`,
`last_finish_reason()`, `stats`, `budget` and `audit_scope()`. This module
serves the same surface from the official `anthropic` SDK for bare Anthropic
model ids (`claude-sonnet-5-5`, ...), translating at the boundary:

    OpenAI side                          Anthropic side
    -----------                          --------------
    leading {"role": "system"} messages  top-level `system` text blocks; the
                                         last one carries the cache breakpoint
    {"type": "function", "function":     {"name", "description", "input_schema"}
      {name, description, parameters}}
    image_url data-URL parts             image blocks with a base64 source
    assistant `tool_calls`               tool_use blocks - or, when this client
                                         produced that turn, its full stored
                                         `response.content` (thinking blocks
                                         replayed unchanged)
    consecutive `tool` role messages     tool_result blocks, all in ONE user turn
    stop_reason tool_use/end_turn/       finish_reason tool_calls/stop/
      max_tokens/refusal/pause_turn        length/refusal (pause_turn resumed)
    usage tokens                         OpenRouter-style usage block with a
                                         `cost` from ANTHROPIC_MODELS prices

The request rules (no sampling parameters, adaptive thinking, an explicit
`output_config.effort`, tool_choice "auto", the non-streaming max_tokens
ceiling) come from the claude-api skill's model tables; see ANTHROPIC_MODELS.
"""

import json
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import anthropic

from .client import CallStats, LLMCallError, LLMResponse, ToolCall, RETRYABLE_STATUS
from .spending import RESERVE_FACTOR, _strip_images, image_prompt_tokens

# -----------------------------------------------------------------------------
# Model table: prices and request rules, ONE place.
# -----------------------------------------------------------------------------
# Prices in USD per million tokens, from the claude-api skill's "Current
# Models" table (cached 2026-11-03) and shared/prompt-caching.md: cache writes
# cost 1.25x base input for the 5-minute TTL and 2x for the 1-hour TTL; cache
# reads 0.1x, except Sonnet 5.5 ($0.10) and Opus 5.5 ($0.20) at 0.05x and
# Fable 5.2 ($0.20) at 0.025x. Haiku 5.5 has a second rate card for prompts
# over 100K tokens ($0.50 / $2.50, same multipliers).
ANTHROPIC_PRICES_AS_OF = "2026-11-03"

ANTHROPIC_MODELS: Dict[str, Dict[str, Any]] = {
    "claude-fable-5-2": {
        "input": 8.00, "output": 40.00, "cache_read": 0.20,
        "cache_write_5m": 10.00, "cache_write_1h": 16.00,
        "default_effort": "high", "forced_tool_choice": False, "cache_min_tokens": 512,
    },
    "claude-opus-5-5": {
        "input": 4.00, "output": 20.00, "cache_read": 0.20,
        "cache_write_5m": 5.00, "cache_write_1h": 8.00,
        "default_effort": "medium", "forced_tool_choice": False, "cache_min_tokens": 512,
    },
    "claude-sonnet-5-5": {
        "input": 2.00, "output": 10.00, "cache_read": 0.10,
        "cache_write_5m": 2.50, "cache_write_1h": 4.00,
        "default_effort": "high", "forced_tool_choice": False, "cache_min_tokens": 512,
    },
    "claude-sonnet-5": {
        "input": 2.00, "output": 10.00, "cache_read": 0.20,
        "cache_write_5m": 2.50, "cache_write_1h": 4.00,
        "default_effort": "high", "forced_tool_choice": True, "cache_min_tokens": 1024,
    },
    "claude-haiku-5-5": {
        "input": 0.10, "output": 0.50, "cache_read": 0.01,
        "cache_write_5m": 0.125, "cache_write_1h": 0.20,
        "long_prompt_tokens": 100_000,
        "long": {"input": 0.50, "output": 2.50, "cache_read": 0.05,
                 "cache_write_5m": 0.625, "cache_write_1h": 1.00},
        "default_effort": "medium", "forced_tool_choice": True, "cache_min_tokens": 512,
        # No server-side fallback exists for Haiku 5.5 (a fallbacks list is a 400).
        "no_fallbacks": True,
    },
}

# Effort levels the API takes in output_config.effort. The project's
# reasoning_effort setting (OpenRouter vocabulary) maps onto them; "none" /
# "minimal" have no Anthropic equivalent and map to the lowest level.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
_EFFORT_ALIASES = {"none": "low", "minimal": "low"}
# Used when the battle sets no reasoning_effort: stated explicitly on every
# request so both sides of a battle run at the same effort (the API defaults
# differ: medium on Haiku 5.5 / Opus 5.5, high on the Sonnets).
DEFAULT_ANTHROPIC_EFFORT = "medium"

# Non-streaming requests stay under this output ceiling (the SDK guidance for
# requests without streaming; larger values risk HTTP timeouts).
MAX_NONSTREAMING_TOKENS = 16000

# The prompt cache TTL. A checkpoint of a real battle can take longer than 5
# minutes of wall clock (admiral directive -> per-ship orders -> captains, each
# a model call at high effort), and a 5-minute entry would then expire between
# checkpoints - the OpenRouter path measured exactly that (client.py,
# CACHE_TTL_BY_PREFIX). The 1-hour entry costs 2x input to write instead of
# 1.25x and pays for itself on its first re-read.
DEFAULT_CACHE_TTL = "1h"

# Stored assistant turns (for tool-round continuations), per client.
_NATIVE_TURN_LIMIT = 256


def is_anthropic_model(model: Optional[str]) -> bool:
    """True for a bare Anthropic id (`claude-...`, no vendor prefix)."""
    return isinstance(model, str) and "/" not in model and model.startswith("claude-")


def model_spec(model: str) -> Dict[str, Any]:
    """The table entry for an Anthropic model id; unknown ids raise (never priced at $0)."""
    try:
        return ANTHROPIC_MODELS[model]
    except KeyError:
        known = ", ".join(sorted(ANTHROPIC_MODELS))
        raise ValueError(
            f"Unknown Anthropic model id {model!r}: not in the price table "
            f"(src/llm/anthropic_client.py ANTHROPIC_MODELS, as of {ANTHROPIC_PRICES_AS_OF}). "
            f"Known: {known}. Use the exact id without a date suffix, or a "
            f"vendor/model OpenRouter id.") from None


def anthropic_effort(reasoning_effort: Optional[str]) -> str:
    """Map the project's reasoning_effort onto output_config.effort."""
    if reasoning_effort is None:
        return DEFAULT_ANTHROPIC_EFFORT
    effort = _EFFORT_ALIASES.get(str(reasoning_effort).lower(), str(reasoning_effort).lower())
    if effort not in EFFORT_LEVELS:
        raise ValueError(f"Unsupported reasoning effort {reasoning_effort!r} for the Anthropic "
                         f"API (one of {', '.join(EFFORT_LEVELS)})")
    return effort


def _rates(model: str, prompt_tokens: int) -> Dict[str, float]:
    spec = model_spec(model)
    threshold = spec.get("long_prompt_tokens")
    if threshold and prompt_tokens > threshold:
        return spec["long"]
    return spec


def _get(obj: Any, key: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    return default if value is None else value


def cost_usd(model: str, usage: Any, requested_ttl: Optional[str] = DEFAULT_CACHE_TTL) -> float:
    """
    Dollar cost of one response from its usage block (SDK object or dict).

    Cache writes are split by TTL from usage.cache_creation where the API
    reports it; otherwise every written token is billed at the TTL the request
    asked for.
    """
    uncached = int(_get(usage, "input_tokens"))
    read = int(_get(usage, "cache_read_input_tokens"))
    written = int(_get(usage, "cache_creation_input_tokens"))
    output = int(_get(usage, "output_tokens"))
    split = _get(usage, "cache_creation", None)
    w5 = int(_get(split, "ephemeral_5m_input_tokens"))
    w1 = int(_get(split, "ephemeral_1h_input_tokens"))
    if w5 + w1 < written:
        # Unsplit (or partially split) writes: bill the remainder at the TTL asked for.
        rest = written - w5 - w1
        if requested_ttl == "1h":
            w1 += rest
        else:
            w5 += rest
    rates = _rates(model, uncached + read + written)
    return (uncached * rates["input"] + read * rates["cache_read"]
            + w5 * rates["cache_write_5m"] + w1 * rates["cache_write_1h"]
            + output * rates["output"]) / 1e6


def reservation_usd(model: str, openai_payload: Dict[str, Any], max_tokens: int) -> Tuple[float, int]:
    """
    Upper bound on one request's cost, for RequestBudget.reserve_usd: every
    prompt byte priced as a token at the dearest input rate (a 1-hour cache
    write), images by pixel size, and a full max_tokens of output. Returns
    (dollars, images). A prompt whose byte count passes Haiku 5.5's 100K-token
    line is priced on its long-prompt card.
    """
    images: List[Any] = []
    encoded = json.dumps(_strip_images(openai_payload, images), ensure_ascii=True, default=str)
    image_tokens = sum(image_prompt_tokens(url, "anthropic/" + model) for url in images)
    prompt_bound = len(encoded.encode()) + 4096 + image_tokens
    rates = _rates(model, prompt_bound)
    input_rate = max(rates["input"], rates["cache_write_5m"], rates["cache_write_1h"])
    dollars = (prompt_bound * input_rate + max_tokens * rates["output"]) / 1e6
    return dollars * RESERVE_FACTOR, len(images)


# -----------------------------------------------------------------------------
# Translation: OpenAI-style request -> Anthropic request
# -----------------------------------------------------------------------------

def translate_tools(tools: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    OpenAI function tools -> Anthropic tools, order preserved (the tool list is
    the head of the cache prefix, so it must stay byte-stable).

    `strict` is NOT set: it requires every object schema to carry
    additionalProperties: false with all properties required, and the
    captain/admiral tools have optional fields (the project validates
    arguments itself).
    """
    out = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            raise ValueError(f"Tool without a name: {tool!r}")
        entry: Dict[str, Any] = {"name": name}
        if fn.get("description"):
            entry["description"] = fn["description"]
        entry["input_schema"] = fn.get("parameters") or fn.get("input_schema") or {
            "type": "object", "properties": {}}
        out.append(entry)
    return out


def _image_block(part: Dict[str, Any]) -> Dict[str, Any]:
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    if not isinstance(url, str):
        raise ValueError("image_url part without a url")
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        if ";base64" not in header:
            raise ValueError("Only base64 data URLs are supported for images")
        media_type = header[len("data:"):].split(";")[0] or "image/png"
        return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                            "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _content_blocks(content: Any) -> List[Dict[str, Any]]:
    """OpenAI message content (string or parts) -> Anthropic content blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for part in content:
        if isinstance(part, str):
            if part:
                blocks.append({"type": "text", "text": part})
            continue
        kind = part.get("type")
        if kind == "text":
            if part.get("text"):
                blocks.append({"type": "text", "text": part["text"]})
        elif kind == "image_url" or (kind is None and "image_url" in part):
            blocks.append(_image_block(part))
        elif kind in ("image", "tool_use", "tool_result", "thinking", "redacted_thinking"):
            blocks.append(dict(part))   # already Anthropic-shaped
        else:
            raise ValueError(f"Unsupported content part for the Anthropic API: {kind!r}")
    return blocks


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or []
                     if isinstance(p, dict) and p.get("type") == "text")


def translate_messages(
    messages: List[Dict[str, Any]],
    native_turn: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    OpenAI-style messages -> (system blocks, Anthropic messages).

    native_turn(tool_call_ids) returns the stored `response.content` of the
    assistant turn that made those calls (thinking + tool_use blocks), or None;
    a turn found there is replayed unchanged, as the API requires within a
    tool-use turn. Consecutive `tool` results become ONE user message, and a
    user message right after them joins that same message.
    """
    system: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        system.extend(_content_blocks(messages[i].get("content")))
        i += 1

    def append(role: str, blocks: List[Any]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"] = list(out[-1]["content"]) + list(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for msg in messages[i:]:
        role = msg.get("role")
        if role == "user":
            append("user", _content_blocks(msg.get("content")))
        elif role == "system":
            # A later system message (none of the callers send one): keep it
            # as clearly labelled user text rather than editing the cached
            # top-level system prompt.
            text = _text_of(msg.get("content"))
            if text:
                append("user", [{"type": "text", "text": f"[system] {text}"}])
        elif role == "assistant":
            calls = msg.get("tool_calls") or []
            ids = tuple(c.get("id", "") for c in calls)
            stored = native_turn(ids) if (calls and native_turn) else None
            if stored is not None:
                append("assistant", list(stored))
                continue
            blocks = _content_blocks(msg.get("content"))
            for call in calls:
                fn = call.get("function") or {}
                raw = fn.get("arguments")
                args = raw if isinstance(raw, dict) else (json.loads(raw) if raw else {})
                blocks.append({"type": "tool_use", "id": call.get("id", ""),
                               "name": fn.get("name", ""), "input": args})
            append("assistant", blocks)
        elif role == "tool":
            content = msg.get("content")
            result: Dict[str, Any] = {"type": "tool_result",
                                      "tool_use_id": msg.get("tool_call_id", "")}
            if isinstance(content, str):
                result["content"] = content
            elif content:
                result["content"] = _content_blocks(content)
            if msg.get("is_error"):
                result["is_error"] = True
            append("user", [result])
        else:
            raise ValueError(f"Unsupported message role for the Anthropic API: {role!r}")
    if out and out[-1]["role"] == "assistant":
        # Current models reject a trailing assistant turn (prefill).
        raise ValueError("The last message must be a user turn (assistant prefill is rejected)")
    return system, out


def mark_cache_breakpoint(system: List[Dict[str, Any]], ttl: Optional[str]) -> List[Dict[str, Any]]:
    """
    Cache breakpoint on the last system text block. The API renders
    tools -> system -> messages, so this one marker caches the tool list and
    the doctrine together; the volatile per-checkpoint user turn comes after it.
    """
    if not system:
        return system
    blocks = [dict(b) for b in system]
    control: Dict[str, str] = {"type": "ephemeral"}
    if ttl and ttl != "5m":
        control["ttl"] = ttl
    for block in reversed(blocks):
        if block.get("type") == "text":
            block["cache_control"] = control
            break
    return blocks


# -----------------------------------------------------------------------------
# Response -> OpenAI-style view
# -----------------------------------------------------------------------------

FINISH_REASONS = {
    "tool_use": "tool_calls",
    "end_turn": "stop",
    "stop_sequence": "stop",
    # The callers' truncation handling (captain retry nudge, admiral effort
    # ladder) keys on OpenRouter's "length".
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "refusal",
    "pause_turn": "pause_turn",
}


def finish_reason_for(stop_reason: Optional[str]) -> Optional[str]:
    if stop_reason is None:
        return None
    return FINISH_REASONS.get(stop_reason, stop_reason)


def usage_dict(usage: Any) -> Dict[str, Any]:
    """The API's usage fields as a plain dict (for recordings)."""
    split = _get(usage, "cache_creation", None)
    details = _get(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(_get(usage, "input_tokens")),
        "output_tokens": int(_get(usage, "output_tokens")),
        "cache_creation_input_tokens": int(_get(usage, "cache_creation_input_tokens")),
        "cache_read_input_tokens": int(_get(usage, "cache_read_input_tokens")),
        "cache_creation": {
            "ephemeral_5m_input_tokens": int(_get(split, "ephemeral_5m_input_tokens")),
            "ephemeral_1h_input_tokens": int(_get(split, "ephemeral_1h_input_tokens")),
        },
        "thinking_tokens": _get(details, "thinking_tokens", None),
    }


def openai_usage(native: Dict[str, Any], cost: float) -> Dict[str, Any]:
    """OpenRouter-shaped usage block, so CallStats and the recordings read it unchanged."""
    prompt = (native["input_tokens"] + native["cache_creation_input_tokens"]
              + native["cache_read_input_tokens"])
    return {
        "prompt_tokens": prompt,
        "completion_tokens": native["output_tokens"],
        "total_tokens": prompt + native["output_tokens"],
        "prompt_tokens_details": {"cached_tokens": native["cache_read_input_tokens"],
                                  "cache_write_tokens": native["cache_creation_input_tokens"]},
        "completion_tokens_details": {"reasoning_tokens": native.get("thinking_tokens")},
        "cost": cost,
        "anthropic": native,
    }


class AnthropicClient:
    """
    CaptainClient-compatible client for the Anthropic Messages API.

    Thread-safe for the battle's concurrency (parallel_llm runs calls on a
    ThreadPoolExecutor): the sync SDK client is shared, and per-call state
    (last_call_meta, last_finish_reason, the audit observer) is thread-local.
    """

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = MAX_NONSTREAMING_TOKENS,
        timeout: float = 120.0,
        session_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        cache_ttl: Optional[str] = DEFAULT_CACHE_TTL,
        enable_fallbacks: bool = False,
        max_retries: int = 2,
        http_client: Any = None,
        stats: Optional[CallStats] = None,
    ):
        """
        Args:
            model: bare Anthropic id from ANTHROPIC_MODELS (no date suffix).
            api_key: defaults to ANTHROPIC_API_KEY (loaded from .env by
                src/llm/client.py's load_dotenv, like OPENROUTER_API_KEY).
            temperature: accepted for interface parity and ignored - current
                models reject temperature/top_p/top_k.
            max_tokens: per-request output ceiling, clamped to
                MAX_NONSTREAMING_TOKENS (thinking counts toward it).
            session_id: OpenRouter sticky-routing key; unused here (kept for
                interface parity).
            reasoning_effort: project setting, mapped to output_config.effort
                and sent on every request (anthropic_effort()).
            cache_ttl: "1h" (default, see DEFAULT_CACHE_TTL) or "5m".
            enable_fallbacks: server-side refusal fallbacks (beta). OFF by
                default and deliberately so: these battles compare one model
                against another, and a refused turn silently re-run on a
                different model would put that model's decision in this
                model's column. Refusals are recorded and counted instead.
            max_retries: the SDK's own retries (429/5xx/408/409/connection,
                exponential backoff). This client adds no retry loop on top.
            http_client: an `anthropic.DefaultHttpxClient` (tests inject one
                with a mock transport).
        """
        model_spec(model)   # unknown ids fail here, at construction
        if enable_fallbacks and ANTHROPIC_MODELS[model].get("no_fallbacks"):
            raise ValueError(f"{model} has no server-side refusal fallback")
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                f"ANTHROPIC_API_KEY is not set, and model {model!r} is served by the "
                "Anthropic API directly. Add ANTHROPIC_API_KEY to .env (or the "
                "environment), or use the OpenRouter id 'anthropic/<model>' instead.")
        self.model = model
        self.temperature = temperature
        self.max_tokens = min(int(max_tokens), MAX_NONSTREAMING_TOKENS)
        self.session_id = session_id
        self.reasoning_effort = reasoning_effort
        self.effort = anthropic_effort(reasoning_effort)
        self.cache_ttl = cache_ttl
        self.enable_fallbacks = enable_fallbacks
        self.max_retries = max_retries
        self.stats = stats if stats is not None else CallStats()
        self.budget = None
        self.log_cache = os.getenv("AIC_LOG_CACHE", "") not in ("", "0")
        self._audit_local = threading.local()
        self._call_meta = threading.local()
        self._native_turns: "OrderedDict[Tuple[str, ...], List[Any]]" = OrderedDict()
        self._native_lock = threading.Lock()
        options: Dict[str, Any] = {"api_key": key, "timeout": timeout, "max_retries": max_retries}
        if http_client is not None:
            options["http_client"] = http_client
        # The key lives only inside the SDK client; nothing here serializes it.
        self._sdk = anthropic.Anthropic(**options)

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self.model!r}, effort={self.effort!r})"

    # -- interface parity --------------------------------------------------

    @contextmanager
    def audit_scope(self, observer):
        previous = getattr(self._audit_local, "observer", None)
        self._audit_local.observer = observer
        try:
            yield
        finally:
            self._audit_local.observer = previous

    def last_call_meta(self) -> Dict[str, Any]:
        """Same keys as CaptainClient.last_call_meta(), plus stop_reason / stop_details."""
        return dict(getattr(self._call_meta, "meta", None) or {})

    def last_finish_reason(self) -> Optional[str]:
        return getattr(self._audit_local, "last_finish_reason", None)

    def provenance(self, model: Optional[str] = None) -> Dict[str, Any]:
        """What served a side: recorded in the battle provenance."""
        model = model or self.model
        return {
            "provider": "anthropic",
            "model": model,
            "effort": self.effort,
            "reasoning_effort_setting": self.reasoning_effort,
            "thinking": "adaptive",
            "max_tokens": self.max_tokens,
            "tool_choice": "auto",
            "cache_ttl": self.cache_ttl,
            "fallbacks": self.enable_fallbacks,
            "sdk": f"anthropic-python {anthropic.__version__}",
            "api_version": self._api_version(),
            "prices_as_of": ANTHROPIC_PRICES_AS_OF,
        }

    def _api_version(self) -> Optional[str]:
        try:
            return self._sdk.default_headers.get("anthropic-version")
        except Exception:  # noqa: BLE001 - provenance only
            return None

    # -- native history (tool-round continuations) -------------------------

    def _store_turn(self, response: Any) -> None:
        ids = tuple(b.id for b in response.content if getattr(b, "type", None) == "tool_use")
        if not ids:
            return
        with self._native_lock:
            self._native_turns[ids] = list(response.content)
            self._native_turns.move_to_end(ids)
            while len(self._native_turns) > _NATIVE_TURN_LIMIT:
                self._native_turns.popitem(last=False)

    def _native_turn(self, ids: Tuple[str, ...]) -> Optional[List[Any]]:
        with self._native_lock:
            turn = self._native_turns.get(ids)
            return list(turn) if turn is not None else None

    # -- request building --------------------------------------------------

    def build_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        tool_choice: Optional[str] = "auto",
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The keyword arguments for messages.create (no network)."""
        request_model = model or self.model
        model_spec(request_model)
        system, converted = translate_messages(messages, self._native_turn)
        limit = min(int(max_tokens or self.max_tokens), MAX_NONSTREAMING_TOKENS)
        effort = anthropic_effort(reasoning_effort) if reasoning_effort else self.effort
        request: Dict[str, Any] = {
            "model": request_model,
            "max_tokens": limit,
            "messages": converted,
            # Adaptive thinking is accepted by every model in the table (Sonnet
            # 5.5 / Opus 5.5 / Fable 5.2 reject "disabled"); depth is effort.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if system:
            request["system"] = mark_cache_breakpoint(system, self.cache_ttl)
        if tools:
            request["tools"] = translate_tools(tools)
            # "required" (forced) would be a 400 on Sonnet 5.5 / Opus 5.5 /
            # Fable 5.2 and skips thinking where accepted, so every model is
            # asked with "auto" (both sides of a battle alike); the prompts
            # already say "answer with a tool call" and the callers re-ask
            # when none arrives - the same downgrade forced_tool_use_conflicts
            # makes on the OpenRouter path.
            request["tool_choice"] = {"type": "none"} if tool_choice == "none" else {"type": "auto"}
        return request

    # -- sending -----------------------------------------------------------

    def _create(self, request: Dict[str, Any]) -> Any:
        if self.enable_fallbacks:
            return self._sdk.beta.messages.create(
                **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return self._sdk.messages.create(**request)

    def _send(self, request: Dict[str, Any], what: str,
              openai_payload: Dict[str, Any]) -> Tuple[Any, float]:
        """
        One API call (the SDK retries transient errors itself) -> (response, cost).

        Raises:
            LLMCallError: translated from the SDK's typed exceptions, or when
                the spend cap refuses the reservation.
        """
        model = request["model"]
        reserved = None
        if self.budget is not None:
            amount, images = reservation_usd(model, openai_payload, request["max_tokens"])
            try:
                reserved = self.budget.reserve_usd(amount, images=images)
            except RuntimeError as e:
                msg = f"{what} failed: {e}"
                self.stats.record_error(msg)
                raise LLMCallError(msg, retryable=False) from e
        self.stats.record_call()
        total_cost = 0.0
        try:
            response = self._create(request)
            total_cost += cost_usd(model, response.usage, self.cache_ttl)
            # pause_turn: only server tools pause a turn and this client sends
            # none, but resume it (bounded) rather than lose the turn.
            for _ in range(2):
                if response.stop_reason != "pause_turn":
                    break
                request = {**request, "messages": list(request["messages"]) + [
                    {"role": "assistant", "content": response.content}]}
                response = self._create(request)
                total_cost += cost_usd(model, response.usage, self.cache_ttl)
        except anthropic.AuthenticationError as e:
            raise self._fail(what, e, retryable=False, hint="check ANTHROPIC_API_KEY") from e
        except anthropic.PermissionDeniedError as e:
            raise self._fail(what, e, retryable=False) from e
        except anthropic.NotFoundError as e:
            raise self._fail(what, e, retryable=False, hint="unknown or unavailable model id") from e
        except anthropic.RateLimitError as e:
            raise self._fail(what, e, retryable=True) from e
        except anthropic.InternalServerError as e:   # 5xx incl. 529 overloaded
            raise self._fail(what, e, retryable=True) from e
        except anthropic.BadRequestError as e:
            raise self._fail(what, e, retryable=False) from e
        except anthropic.APIStatusError as e:
            raise self._fail(what, e, retryable=e.status_code in RETRYABLE_STATUS) from e
        except anthropic.APITimeoutError as e:
            raise self._fail(what, e, retryable=True) from e
        except anthropic.APIConnectionError as e:
            raise self._fail(what, e, retryable=True) from e
        except anthropic.AnthropicError as e:
            raise self._fail(what, e, retryable=False) from e
        if self.budget is not None:
            self.budget.settle(reserved, total_cost)
        return response, total_cost

    def _fail(self, what: str, error: Exception, retryable: bool,
              hint: Optional[str] = None) -> LLMCallError:
        status = getattr(error, "status_code", None)
        # e.message is the API's error text (no request headers, so no key).
        detail = str(getattr(error, "message", None) or error)[:400]
        where = f"HTTP {status}" if status else type(error).__name__
        tries = f" after {self.max_retries} SDK retries" if retryable else ""
        msg = f"{what} failed{tries} ({where}: {detail})" + (f" - {hint}" if hint else "")
        self.stats.record_error(msg)
        return LLMCallError(msg, status_code=status, retryable=retryable)

    def _finish(self, request: Dict[str, Any], response: Any, cost: float,
                openai_payload: Dict[str, Any]) -> Tuple[List[ToolCall], str, Dict[str, Any]]:
        """Bookkeeping shared by both entry points -> (tool calls, text, OpenAI-shaped data)."""
        model = request["model"]
        stop = response.stop_reason
        finish = finish_reason_for(stop)
        native = usage_dict(response.usage)
        usage = openai_usage(native, cost)
        self.stats.record_usage(usage)

        blocks = list(response.content or [])
        tool_blocks = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
        text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text")
        stop_details = None
        if stop == "refusal":
            # A refusal can cut a tool_use off mid-input: never act on that turn.
            details = getattr(response, "stop_details", None)
            stop_details = {"category": _get(details, "category", None),
                            "explanation": _get(details, "explanation", None)}
            self.stats.record_refusal(model)
            tool_blocks = []
            print(f"[LLM REFUSAL] {model}: stop_reason=refusal "
                  f"(category={stop_details['category']}) - no orders from this call", flush=True)
        elif stop == "max_tokens":
            self.stats.record_truncated()
            print(f"[LLM WARN] Response truncated at max_tokens={request['max_tokens']} "
                  f"for {model}; some orders may be missing.", flush=True)
            if tool_blocks and blocks and blocks[-1] is tool_blocks[-1]:
                # The last block was being written when the limit hit: its
                # input may be a silently partial object. Earlier calls are whole.
                self.stats.record_malformed()
                print(f"[LLM WARN] Discarding tool call '{tool_blocks[-1].name}' cut off "
                      f"at max_tokens", flush=True)
                tool_blocks = tool_blocks[:-1]

        calls = [ToolCall(id=b.id, name=b.name,
                          arguments=dict(b.input) if isinstance(b.input, dict) else {})
                 for b in tool_blocks]
        if tool_blocks:
            self._store_turn(response)

        self._audit_local.last_finish_reason = finish
        self._call_meta.meta = {
            "model": model,
            "served_by": getattr(response, "model", model),
            "provider": "anthropic",
            "finish_reason": finish,
            "stop_reason": stop,
            "stop_details": stop_details,
            "usage": usage,
            "reasoning_tokens": native.get("thinking_tokens"),
            "completion_tokens": native["output_tokens"],
            "max_tokens": request["max_tokens"],
            "content_chars": len(text),
            "effort": request["output_config"]["effort"],
        }
        if self.log_cache:
            read, written = native["cache_read_input_tokens"], native["cache_creation_input_tokens"]
            outcome = "HIT" if read else ("WRITE" if written else "cold")
            print(f"[LLM CACHE] {model}: {outcome} read {read} / write {written} of "
                  f"{usage['prompt_tokens']} prompt tokens (${cost:.5f})", flush=True)

        data = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", model),
            "provider": "anthropic",
            "choices": [{
                "message": {
                    "content": text or None,
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name,
                                                 "arguments": json.dumps(c.arguments)}}
                                   for c in calls],
                },
                "finish_reason": finish,
            }],
            "stop_reason": stop,
            "stop_details": stop_details,
            "usage": usage,
        }
        observer = getattr(self._audit_local, "observer", None)
        if observer:
            observer(self._audit_payload(request, openai_payload), data)
        return calls, text, data

    def _audit_payload(self, request: Dict[str, Any], openai_payload: Dict[str, Any]) -> Dict[str, Any]:
        """What the replay audit records: the caller's messages/tools, plus the
        request settings that shaped the answer (never headers or the key)."""
        payload = {k: v for k, v in request.items() if k not in ("messages", "system", "tools")}
        payload["provider"] = "anthropic"
        payload["cache_ttl"] = self.cache_ttl
        payload["api_version"] = self._api_version()
        payload["messages"] = openai_payload.get("messages", [])
        payload["tools"] = openai_payload.get("tools", [])
        return payload

    # -- public API (CaptainClient parity) ---------------------------------

    def decide_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tool_choice: str = "auto",
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> List[ToolCall]:
        """
        Same contract as CaptainClient.decide_with_tools. `temperature` is
        ignored (rejected by current models). An empty list means no tool call;
        last_finish_reason() then says whether that was a choice ("stop"), a
        truncation ("length") or a refusal ("refusal").
        """
        self._call_meta.meta = {}
        self._audit_local.last_finish_reason = None
        request = self.build_request(messages, tools, model=model, tool_choice=tool_choice,
                                     max_tokens=max_tokens, reasoning_effort=reasoning_effort)
        openai_payload = {"messages": messages, "tools": tools}
        response, cost = self._send(request, f"decide_with_tools[{request['model']}]", openai_payload)
        calls, _, _ = self._finish(request, response, cost, openai_payload)
        return calls

    def complete(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Same contract as CaptainClient.complete (no tools)."""
        self._call_meta.meta = {}
        self._audit_local.last_finish_reason = None
        request = self.build_request(messages, None, model=model)
        openai_payload = {"messages": messages, "tools": []}
        response, cost = self._send(request, f"complete[{request['model']}]", openai_payload)
        _, text, data = self._finish(request, response, cost, openai_payload)
        return LLMResponse(
            content=text,
            tool_calls=[],
            model=data["model"],
            usage=data["usage"],
            finish_reason=data["choices"][0]["finish_reason"],
            raw_response=data,
        )
