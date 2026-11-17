"""
Per-call policy for LLM commanders: which ROLE a call plays, and (for
experiments) which effort it is sent at.

Roles. A battle's model calls play five roles, told apart from what the
recording or the request already carries (old recordings have only `actor`,
the draft `phase` and the tool list):

    draft       the pre-battle fleet purchase and formation (select_fleet,
                set_formation)
    directive   the admiral's Phase-1 call (set_fleet_directive offered)
    order       the admiral's per-ship Phase-2 call (only issue_order offered)
    captain     a captain's decision call
    other       anything else: personality selection, an admiral's free-text
                answer to discuss_with_admiral, notebook distillation

`call_role()` is the one classifier; the replay harness, the cost report and
the effort policy all use it.
"""
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROLES = ("draft", "directive", "order", "captain", "other")


def tool_names(tools: Optional[Iterable[Any]]) -> list:
    """Names of OpenAI-style function tools (or Anthropic-shaped tools)."""
    names = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.append(name)
    return names


def call_role(actor: Optional[str], phase: Optional[str] = None,
              names: Optional[Iterable[str]] = None) -> str:
    """
    The role of one model call (see the module docstring).

    actor: 'captain' / 'admiral' (model_call.actor), phase: model_call.phase
    ('draft' for calls merged from the draft recorders), names: the tool names
    offered to the call.
    """
    names = set(names or ())
    if phase == "draft" or names & {"select_fleet", "set_formation"}:
        return "draft"
    if actor == "captain":
        return "captain" if names else "other"
    if actor == "admiral":
        if "set_fleet_directive" in names:
            return "directive"
        if names == {"issue_order"}:
            return "order"
        return "other"
    return "other"


# -----------------------------------------------------------------------------
# Effort policy: per side, per role, plus a captain trigger effort
# -----------------------------------------------------------------------------
#
# Off by default. With no setting for a role the call carries no effort
# override at all, so the request is exactly what the client would send
# without this module (the client's battle-wide reasoning effort).

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Roles an effort can be set for. "trigger" = captain turns with a trigger
# (captain_triggers); "captain" is then the base effort of the other turns.
EFFORT_ROLES = ("draft", "directive", "order", "captain", "trigger")
_ROLE_ALIASES = {"admiral-directive": "directive", "admiral_directive": "directive",
                 "admiral-order": "order", "admiral_order": "order",
                 "captain-trigger": "trigger", "captain_trigger": "trigger"}
SIDES = ("alpha", "beta")


def parse_effort_specs(specs: Optional[Iterable[str]]) -> Dict[str, Dict[str, str]]:
    """
    `[SIDE.]ROLE=LEVEL` strings -> {side: {role: level}}. SIDE is alpha, beta
    or both (omitted = both); ROLE one of EFFORT_ROLES; later specs win.

        ["directive=high", "alpha.captain=low", "trigger=high"]
    """
    out: Dict[str, Dict[str, str]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Effort spec {spec!r}: expected [SIDE.]ROLE=LEVEL")
        key, level = (s.strip().lower() for s in spec.split("=", 1))
        side, _, role = key.rpartition(".")
        side = side or "both"
        role = _ROLE_ALIASES.get(role, role)
        if side not in SIDES + ("both",):
            raise ValueError(f"Effort spec {spec!r}: side must be alpha, beta or both")
        if role not in EFFORT_ROLES:
            raise ValueError(f"Effort spec {spec!r}: role must be one of {', '.join(EFFORT_ROLES)}")
        if level not in EFFORT_LEVELS:
            raise ValueError(f"Effort spec {spec!r}: level must be one of {', '.join(EFFORT_LEVELS)}")
        for s in (SIDES if side == "both" else (side,)):
            out.setdefault(s, {})[role] = level
    return out


def validate_effort_policy(policy: Any) -> Dict[str, Dict[str, str]]:
    """A {side: {role: level}} dict from config/JSON, checked; raises on junk."""
    if not policy:
        return {}
    if not isinstance(policy, dict):
        raise ValueError("effort_policy must be a {side: {role: level}} mapping")
    specs = []
    for side, roles in policy.items():
        if not isinstance(roles, dict):
            raise ValueError(f"effort_policy[{side!r}] must be a {{role: level}} mapping")
        specs.extend(f"{side}.{role}={level}" for role, level in roles.items())
    return parse_effort_specs(specs)


@dataclass(frozen=True)
class TurnFacts:
    """What the runner knows about a captain's turn before the call (all cheap)."""
    first_turn: bool = False
    torpedoes_inbound: int = 0          # guided rounds homing on this ship
    merge_warning: bool = False         # RANGE CONTROL merge warning on any group
    damage_taken: bool = False          # hits on this ship since its last turn
    order_moves_ship: bool = False      # an admiral heading / maneuver is applied
    order_kind: Optional[str] = None    # this checkpoint's applied order kind
    previous_order_kind: Optional[str] = None
    order_changed: bool = False         # applied order differs from last checkpoint's
    new_messages: int = 0               # FLEET NET traffic from peers / admiral


def captain_triggers(facts: TurnFacts) -> List[str]:
    """
    Why this captain turn deserves the trigger effort (empty = a quiet turn).

    torpedo   guided torpedoes inbound on this ship
    merge     a merge warning from range control
    conflict  an applied admiral maneuver while a merge is warned (the
              captain's "order and RANGE CONTROL conflict" line needs both)
    damage    hit since the last turn
    new_order the applied order's kind changed (HEADING -> PADLOCK, a salvo...)
    first     the battle's first decision
    """
    reasons = []
    if facts.first_turn:
        reasons.append("first")
    if facts.torpedoes_inbound > 0:
        reasons.append("torpedo")
    if facts.merge_warning:
        reasons.append("merge")
        if facts.order_moves_ship:
            reasons.append("conflict")
    if facts.damage_taken:
        reasons.append("damage")
    if (facts.order_kind or facts.previous_order_kind) and facts.order_kind != facts.previous_order_kind:
        reasons.append("new_order")
    return reasons


def quiet_turn_reason(facts: TurnFacts) -> Optional[str]:
    """
    The quiet-turn policy (BattleConfig.quiet_turns): the reason to SKIP this
    captain call, or None to call it. Skipped only with no trigger, no new
    traffic and an applied order identical to last checkpoint's.
    """
    if captain_triggers(facts) or facts.new_messages or facts.order_changed:
        return None
    return "quiet: no trigger, no new traffic, applied order unchanged"


def _round(value: Any, step: float = 0.05) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(round(float(value) / step) * step, 3)
    if isinstance(value, dict):
        return {k: _round(v, step) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_round(v, step) for v in value]
    return value


def order_kind(orders: Optional[Iterable[Any]]) -> Optional[str]:
    """Kind of this checkpoint's admiral order(s): maneuver type / HEADING / TEXT (+SALVO)."""
    kinds = []
    for order in orders or []:
        spec = getattr(order, "maneuver", None)
        if isinstance(spec, dict) and spec and not getattr(order, "maneuver_error", None):
            kind = str(spec.get("maneuver_type") or spec.get("type") or "MANEUVER").upper()
        elif isinstance(getattr(order, "heading", None), dict):
            kind = "HEADING"
        else:
            kind = "TEXT"
        if getattr(order, "torpedo_salvo", None):
            kind += "+SALVO"
        kinds.append(kind)
    return "/".join(kinds) if kinds else None


def order_signature(orders: Optional[Iterable[Any]]) -> Optional[str]:
    """What the engine applies from the order(s), prose excluded (quiet-turn equality)."""
    parts = []
    for order in orders or []:
        parts.append({
            "maneuver": _round(getattr(order, "maneuver", None)),
            "maneuver_error": bool(getattr(order, "maneuver_error", None)),
            "heading": _round(getattr(order, "heading", None)),
            "throttle": _round(getattr(order, "throttle", None)),
            "target": getattr(order, "suggested_target", None),
            "salvo": getattr(order, "torpedo_salvo", None),
            "salvo_target": getattr(order, "torpedo_target", None),
        })
    return json.dumps(parts, sort_keys=True, default=str) if parts else None


class EffortPolicy:
    """Resolves the effort of a call from {side: {role: level}}."""

    def __init__(self, policy: Optional[Dict[str, Dict[str, str]]] = None):
        self.policy = validate_effort_policy(policy or {})

    def __bool__(self) -> bool:
        return any(self.policy.values())

    def effort(self, side: Optional[str], role: str) -> Optional[str]:
        return (self.policy.get(side or "") or {}).get(role)

    def captain_turn(self, side: Optional[str], triggers: List[str]) -> Tuple[Optional[str], str]:
        """(effort or None = client default, reason) for one captain turn."""
        trigger = self.effort(side, "trigger")
        if trigger and triggers:
            return trigger, "trigger:" + ",".join(triggers)
        base = self.effort(side, "captain")
        return base, ("role" if base else "default")


class PolicyClient:
    """
    Per-agent wrapper over the battle's LLM client that sends each call at its
    role's effort. Everything else is delegated unchanged. A role with no
    setting passes no reasoning_effort (the request is byte-identical to the
    unwrapped client's); an explicit reasoning_effort from the caller (the
    admiral's retry ladder) always wins.

    Captains: the runner sets `turn_effort` before each turn (trigger effort).
    """

    def __init__(self, inner: Any, side: Optional[str], actor: str, policy: EffortPolicy):
        self.inner = inner
        self.side = side
        self.actor = actor
        self.policy = policy
        self.turn_effort: Optional[Tuple[Optional[str], str]] = None
        self._local = threading.local()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def reasoning_effort(self) -> Optional[str]:
        """The effort the agent's own logic sees (the admiral's ladder starts here)."""
        role = "captain" if self.actor == "captain" else "directive"
        return self.policy.effort(self.side, role) or getattr(self.inner, "reasoning_effort", None)

    def _effort(self, role: str) -> Tuple[Optional[str], str]:
        if role == "captain" and self.turn_effort is not None:
            return self.turn_effort
        if role == "captain":
            return self.policy.captain_turn(self.side, [])
        effort = self.policy.effort(self.side, role) if role in EFFORT_ROLES else None
        return effort, ("role" if effort else "default")

    def call_annotation(self) -> Dict[str, Any]:
        """Role and effort reason of this thread's current call (for model_call)."""
        return dict(getattr(self._local, "annotation", None) or {})

    def decide_with_tools(self, messages, tools, model=None, reasoning_effort=None, **kwargs):
        role = call_role(self.actor, None, tool_names(tools))
        if reasoning_effort is not None:
            effort, reason = reasoning_effort, "explicit"
        else:
            effort, reason = self._effort(role)
        self._local.annotation = {"role": role, "effort_reason": reason}
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        return self.inner.decide_with_tools(messages, tools, model=model, **kwargs)

    def complete(self, messages, model=None, **kwargs):
        self._local.annotation = {"role": "other", "effort_reason": "default"}
        return self.inner.complete(messages, model=model, **kwargs)
