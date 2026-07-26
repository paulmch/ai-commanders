"""
OpenRouter client for captain decision-making.

Uses the OpenRouter API directly with httpx for tool/function calling. OpenRouter
(rather than a single vendor SDK) is deliberate: this project pits models from
different providers against each other in the same battle.
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# -----------------------------------------------------------------------------
# Model defaults
# -----------------------------------------------------------------------------
# Single source of truth. Every module that needs a fallback model imports from
# here rather than re-hardcoding an id, so the catalog only has to be updated in
# one place when models are retired.
DEFAULT_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_ADMIRAL_MODEL = "anthropic/claude-opus-5"

# Retry policy for transient failures.
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 20.0

# Status codes that are worth retrying. Everything else is a caller/config error
# and will fail identically on the next attempt.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class LLMCallError(RuntimeError):
    """
    Raised when an API call fails and could not be recovered by retrying.

    This exists so that a failed call is never mistaken for a deliberate
    "the captain issued no orders" decision.
    """

    def __init__(self, message: str, status_code: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class LLMResponse:
    """Response from LLM API call."""
    content: str
    tool_calls: List[Any]
    model: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    raw_response: Any = None


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class CallStats:
    """Per-client counters so degraded runs are visible instead of silent."""
    calls: int = 0
    failures: int = 0
    retries: int = 0
    truncated: int = 0
    malformed_arguments: int = 0
    cached_tokens: int = 0
    prompt_tokens: int = 0
    cache_discount: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from cache across this client."""
        return (self.cached_tokens / self.prompt_tokens) if self.prompt_tokens else 0.0

    def record_usage(self, usage: Dict[str, Any]) -> None:
        """Record cache effectiveness from an OpenRouter usage block."""
        if not usage:
            return
        self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
        details = usage.get("prompt_tokens_details") or {}
        self.cached_tokens += details.get("cached_tokens", 0) or 0
        discount = usage.get("cache_discount")
        if discount:
            self.cache_discount += discount

    def record_error(self, msg: str) -> None:
        self.failures += 1
        # Keep the log bounded; a wedged battle should not accumulate forever.
        if len(self.errors) < 50:
            self.errors.append(msg)


def _strip_prefix(model: str) -> str:
    """Strip the optional ``openrouter/`` routing prefix."""
    prefix = "openrouter/"
    return model[len(prefix):] if model.startswith(prefix) else model


# Providers that cache automatically on a prefix match (OpenAI, Grok, DeepSeek,
# Groq, Moonshot, Z.AI) need no markup. Anthropic, Qwen and Gemini require an
# explicit cache_control breakpoint or nothing is cached at all.
_EXPLICIT_CACHE_PREFIXES = ("anthropic/", "qwen/", "alibaba/", "google/")

# Approximate minimum cacheable prefix per model family (tokens). A prefix
# shorter than this silently will not cache.
_CACHE_MIN_TOKENS = {
    "anthropic/claude-opus-5": 512,
    "anthropic/claude-fable-5": 512,
    "anthropic/claude-sonnet-5": 1024,
    "anthropic/claude-sonnet-4.6": 1024,
    "anthropic/claude-opus-4.8": 1024,
    "anthropic/claude-haiku-4.5": 4096,
    "anthropic/claude-opus-4.6": 4096,
}


def needs_explicit_cache_breakpoint(model: str) -> bool:
    """True if this provider only caches when given an explicit breakpoint."""
    return _strip_prefix(model).startswith(_EXPLICIT_CACHE_PREFIXES)


def cache_minimum_tokens(model: str) -> int:
    """Approximate minimum cacheable prefix length for a model, in tokens."""
    return _CACHE_MIN_TOKENS.get(_strip_prefix(model), 1024)


def apply_cache_breakpoint(
    messages: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    """
    Mark the system prompt as a cache breakpoint for providers that need one.

    Anthropic renders the prefix as tools -> system -> messages, so a single
    breakpoint on the system message also covers the (stable) tool definitions.
    Providers that cache automatically are left untouched, since the markup is
    unnecessary there and can be rejected.
    """
    if not messages or not needs_explicit_cache_breakpoint(model):
        return messages

    patched = []
    marked = False
    for msg in messages:
        if not marked and msg.get("role") == "system" and isinstance(msg.get("content"), str):
            patched.append({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": msg["content"],
                    "cache_control": {"type": "ephemeral"},
                }],
            })
            marked = True
        else:
            patched.append(msg)
    return patched


class CaptainClient:
    """
    LLM client for captain decision-making using OpenRouter directly.

    Uses tool/function calling for structured command output.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the captain client.

        Args:
            model: Model ID (e.g., "anthropic/claude-sonnet-5")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            temperature: Sampling temperature, or None to omit the parameter
                entirely (reasoning models reject an explicit temperature)
            max_tokens: Maximum response tokens. Needs to cover reasoning plus a
                full set of tool calls - 1024 truncated multi-tool turns.
            timeout: Per-request timeout in seconds. Reasoning models routinely
                take longer than the old 60s ceiling on a hard tactical turn.
            session_id: Sticky-routing key. OpenRouter otherwise derives one by
                hashing the first system and first non-system message; since our
                per-turn message changes every checkpoint, an explicit stable id
                keeps every checkpoint of a battle pinned to the same provider
                endpoint, which is what makes a warm cache reachable at all.
        """
        self.model = _strip_prefix(model)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_id = session_id
        self.stats = CallStats()

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key."
            )

        self._client = httpx.Client(timeout=timeout)

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-commanders",
            "X-Title": "AI Commanders",
        }

    def _post(self, payload: Dict[str, Any], what: str) -> Dict[str, Any]:
        """
        POST with bounded retries on transient failures.

        Raises:
            LLMCallError: on a non-retryable error, or after exhausting retries.
        """
        last_error: Optional[str] = None
        last_status: Optional[int] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                self.stats.calls += 1
                response = self._client.post(self.BASE_URL, headers=self._headers(), json=payload)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_status = e.response.status_code
                body = (e.response.text or "")[:400]
                last_error = f"HTTP {last_status}: {body}"

                if last_status not in RETRYABLE_STATUS:
                    # Config errors (bad/retired model id, bad key) fail identically
                    # forever - surface immediately instead of burning the battle.
                    msg = f"{what} failed permanently ({last_error})"
                    self.stats.record_error(msg)
                    raise LLMCallError(msg, status_code=last_status, retryable=False) from e

            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = f"{type(e).__name__}: {e}"

            except Exception as e:  # noqa: BLE001 - defensive, still surfaced below
                msg = f"{what} failed: {type(e).__name__}: {e}"
                self.stats.record_error(msg)
                raise LLMCallError(msg, retryable=False) from e

            if attempt < MAX_ATTEMPTS:
                self.stats.retries += 1
                delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)
                delay += random.uniform(0, delay * 0.25)  # jitter
                print(f"[LLM RETRY] {what}: {last_error} - retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{MAX_ATTEMPTS})")
                time.sleep(delay)

        msg = f"{what} failed after {MAX_ATTEMPTS} attempts ({last_error})"
        self.stats.record_error(msg)
        raise LLMCallError(msg, status_code=last_status, retryable=True)

    def _parse_tool_calls(self, message: Dict[str, Any], finish_reason: Optional[str]) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            raw_args = tc.get("function", {}).get("arguments", "")
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                # Truncated or malformed JSON. Do NOT silently substitute {} - an
                # empty argument dict is a valid-looking command that means
                # something entirely different from what the model intended.
                self.stats.malformed_arguments += 1
                print(f"[LLM WARN] Discarding tool call '{tc.get('function', {}).get('name')}' "
                      f"with unparseable arguments (finish_reason={finish_reason})")
                continue

            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("function", {}).get("name", ""),
                arguments=args,
            ))
        return tool_calls

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def decide_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> List[ToolCall]:
        """
        Make a decision using tool/function calling.

        Args:
            messages: Conversation messages (system, user, assistant)
            tools: Available tools in OpenAI function calling format
            model: Optional model to use (defaults to client's model)
            temperature: Optional per-call sampling temperature

        Returns:
            List of ToolCall objects representing the LLM's decisions. An empty
            list means the model genuinely called no tools.

        Raises:
            LLMCallError: if the API call itself failed.
        """
        request_model = _strip_prefix(model or self.model)

        payload = {
            "model": request_model,
            "messages": apply_cache_breakpoint(messages, request_model),
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            # Ask OpenRouter to report cache effectiveness so a silently cold
            # cache shows up as a number instead of an unnoticed cost.
            "usage": {"include": True},
        }
        # Omit the key entirely when unset: several reasoning models reject an
        # explicit temperature with a 400 rather than ignoring it.
        effective_temp = self.temperature if temperature is None else temperature
        if effective_temp is not None:
            payload["temperature"] = effective_temp
        if self.session_id:
            payload["session_id"] = self.session_id

        data = self._post(payload, f"decide_with_tools[{request_model}]")

        self.stats.record_usage(data.get("usage") or {})

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        finish_reason = choice.get("finish_reason")

        if finish_reason == "length":
            self.stats.truncated += 1
            print(f"[LLM WARN] Response truncated at max_tokens={self.max_tokens} "
                  f"for {request_model}; some orders may be missing.")

        return self._parse_tool_calls(message, finish_reason)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Make a simple completion without tools.

        Args:
            messages: Conversation messages
            model: Optional model to use (defaults to client's model). Without
                this, a shared client answers as whichever model constructed it.
            temperature: Optional per-call sampling temperature

        Returns:
            LLMResponse with content

        Raises:
            LLMCallError: if the API call itself failed.
        """
        request_model = _strip_prefix(model or self.model)

        payload = {
            "model": request_model,
            "messages": apply_cache_breakpoint(messages, request_model),
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
        }
        # See decide_with_tools: omitted rather than sent as None.
        effective_temp = self.temperature if temperature is None else temperature
        if effective_temp is not None:
            payload["temperature"] = effective_temp
        if self.session_id:
            payload["session_id"] = self.session_id

        data = self._post(payload, f"complete[{request_model}]")

        self.stats.record_usage(data.get("usage") or {})

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}

        return LLMResponse(
            content=message.get("content", "") or "",
            tool_calls=[],
            model=data.get("model", request_model),
            usage=data.get("usage", {}) or {},
            finish_reason=choice.get("finish_reason"),
            raw_response=data,
        )
