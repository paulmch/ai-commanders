"""
Pick the LLM provider from the model id.

    claude-sonnet-5-5, claude-haiku-5-5, ...   bare Anthropic id -> Anthropic API
                                                (src/llm/anthropic_client.py)
    anthropic/claude-sonnet-5, deepseek/...    vendor/model id   -> OpenRouter
                                                (src/llm/client.py, unchanged)
    heuristic, mcp, dummy                      no LLM

`create_llm_client()` is used wherever a battle script used to construct a
CaptainClient. A battle whose models are all OpenRouter ids gets a plain
CaptainClient exactly as before; one that names an Anthropic id gets an
`LLMRouter`, which holds one provider client per Anthropic model (and one
OpenRouter client for every vendor/model id) and sends each call to the client
of the model the caller names, so the two sides of a battle can use different
providers and never share a model's client by accident.
"""

import threading
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterable, List, Optional

from .anthropic_client import AnthropicClient, is_anthropic_model, model_spec
from .client import CallStats, CaptainClient, _strip_prefix

# Model ids that never reach an LLM provider.
NON_LLM_MODELS = frozenset({"heuristic", "mcp", "dummy", "", None})


def provider_for_model(model: Optional[str]) -> Optional[str]:
    """'anthropic', 'openrouter', or None for a non-LLM controller."""
    if model in NON_LLM_MODELS:
        return None
    name = _strip_prefix(str(model))
    if is_anthropic_model(name):
        model_spec(name)   # unknown bare Claude ids raise: never priced at $0
        return "anthropic"
    return "openrouter"


def llm_models(models: Iterable[Optional[str]]) -> List[str]:
    """The distinct LLM model ids among `models`, in first-seen order."""
    seen: List[str] = []
    for m in models:
        if provider_for_model(m) is not None and m not in seen:
            seen.append(m)
    return seen


class LLMRouter:
    """
    CaptainClient-shaped facade over several provider clients.

    Every caller passes `model=` per call (one client serves a whole battle);
    the router forwards the call to that model's client. Stats are one shared
    CallStats (scripts and the ticker read client.stats), and per-thread
    last_call_meta / last_finish_reason come from the client this thread
    called last.
    """

    def __init__(
        self,
        model: str,
        models: Iterable[Optional[str]] = (),
        *,
        temperature: Optional[float] = 0.7,
        max_tokens: int = 32768,
        timeout: float = 120.0,
        session_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        anthropic_options: Optional[Dict[str, Any]] = None,
        openrouter_options: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session_id = session_id
        self.reasoning_effort = reasoning_effort
        self.anthropic_options = dict(anthropic_options or {})
        self.openrouter_options = dict(openrouter_options or {})
        self.stats = CallStats()
        self._budget = None
        self._clients: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._local = threading.local()
        # Construct every client now: a missing key or an unknown model id
        # fails at start-up, not at checkpoint 1.
        for m in llm_models([model, *models]):
            self.client_for(m)

    # -- clients -------------------------------------------------------------

    @staticmethod
    def _key(model: str) -> str:
        name = _strip_prefix(model)
        return name if provider_for_model(name) == "anthropic" else "openrouter"

    def client_for(self, model: Optional[str]) -> Any:
        """The provider client serving `model` (created on first use)."""
        model = model or self.model
        provider = provider_for_model(model)
        if provider is None:
            raise ValueError(f"{model!r} is not an LLM model id")
        key = self._key(model)
        with self._lock:
            client = self._clients.get(key)
            if client is None:
                if provider == "anthropic":
                    client = AnthropicClient(
                        model=_strip_prefix(model), temperature=self.temperature,
                        max_tokens=self.max_tokens, timeout=self.timeout,
                        session_id=self.session_id, reasoning_effort=self.reasoning_effort,
                        stats=self.stats, **self.anthropic_options)
                else:
                    client = CaptainClient(
                        model=model, temperature=self.temperature, max_tokens=self.max_tokens,
                        timeout=self.timeout, session_id=self.session_id,
                        reasoning_effort=self.reasoning_effort, **self.openrouter_options)
                    client.stats = self.stats
                client.budget = self._budget
                self._clients[key] = client
            return client

    @property
    def clients(self) -> Dict[str, Any]:
        """{'claude-sonnet-5-5': AnthropicClient, 'openrouter': CaptainClient, ...}"""
        with self._lock:
            return dict(self._clients)

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, value) -> None:
        self._budget = value
        for client in self.clients.values():
            client.budget = value

    # -- CaptainClient interface ---------------------------------------------

    @contextmanager
    def audit_scope(self, observer):
        previous = getattr(self._local, "observer", None)
        self._local.observer = observer
        try:
            yield
        finally:
            self._local.observer = previous

    def _dispatch(self, method: str, model: Optional[str], *args, **kwargs):
        client = self.client_for(model)
        self._local.last = client
        observer = getattr(self._local, "observer", None)
        with ExitStack() as stack:
            if observer is not None:
                stack.enter_context(client.audit_scope(observer))
            return getattr(client, method)(*args, model=model or self.model, **kwargs)

    def decide_with_tools(self, messages, tools, model: Optional[str] = None, **kwargs):
        return self._dispatch("decide_with_tools", model, messages, tools, **kwargs)

    def complete(self, messages, model: Optional[str] = None, **kwargs):
        return self._dispatch("complete", model, messages, **kwargs)

    def last_call_meta(self) -> Dict[str, Any]:
        client = getattr(self._local, "last", None)
        return client.last_call_meta() if client is not None else {}

    def last_finish_reason(self) -> Optional[str]:
        client = getattr(self._local, "last", None)
        return client.last_finish_reason() if client is not None else None

    def provenance(self, model: Optional[str] = None) -> Dict[str, Any]:
        return self.client_for(model).provenance(model or self.model)


def fleet_config_models(fleet_config: Any) -> List[str]:
    """Every LLM model id a fleet config names (ships, admirals, draft captains)."""
    models: List[Optional[str]] = []
    for fleet in (fleet_config.alpha_fleet, fleet_config.beta_fleet):
        models.extend(getattr(s, "model", None) for s in fleet.ships)
        admiral = getattr(fleet, "admiral", None)
        if admiral is not None and getattr(admiral, "enabled", True):
            models.append(admiral.model)
    draft = getattr(fleet_config, "draft", None)
    if draft is not None and getattr(draft, "enabled", False):
        models.append(getattr(draft, "captain_model", None))
    return llm_models(models)


def create_llm_client(
    model: str,
    models: Iterable[Optional[str]] = (),
    **kwargs: Any,
) -> Any:
    """
    The battle's LLM client for `model` (the default) and every id in `models`.

    All OpenRouter ids: a CaptainClient built exactly as the scripts built it
    before (anthropic_options is ignored). Any bare Anthropic id: an LLMRouter
    - which raises here if ANTHROPIC_API_KEY is missing or an id is unknown.
    """
    wanted = llm_models([model, *models])
    if not any(provider_for_model(m) == "anthropic" for m in wanted):
        kwargs.pop("anthropic_options", None)
        openrouter_options = kwargs.pop("openrouter_options", None) or {}
        return CaptainClient(model=model, **kwargs, **openrouter_options)
    return LLMRouter(model, wanted, **kwargs)


def llm_provenance(client: Any, side_models: Dict[str, Iterable[Optional[str]]]) -> Dict[str, Any]:
    """
    Which provider and exact model id served each side, at which effort.

    side_models: {'alpha': [admiral model, captain models...], 'beta': [...]}.
    """
    describe = getattr(client, "provenance", None)
    out: Dict[str, Any] = {}
    for side, models in side_models.items():
        entries = []
        for m in llm_models(models):
            try:
                entries.append(describe(m) if callable(describe) else {"model": m})
            except Exception as exc:  # noqa: BLE001 - provenance must not stop a battle
                entries.append({"model": m, "error": f"{type(exc).__name__}"})
        out[side] = entries
    return out
