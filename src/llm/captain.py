"""
LLM-powered captain for space combat simulation.

Makes strategic decisions via tool/function calling.
"""

import json
import re
from src.replay_evidence import tool_evidence
import math
from dataclasses import dataclass, field, fields, replace
from typing import List, Dict, Any, Optional, Tuple

from .client import DEFAULT_MODEL, CaptainClient, LLMCallError, ToolCall
from .tools import get_captain_tools, get_captain_tools_for_ship, get_weapon_groups_for_ship, PERSONALITY_SELECTION_TOOLS, RESPOND_TO_ORDERS_TOOL, radiator_command_from_args
from .prompts import (
    build_captain_prompt,
    build_captain_messages,
    build_personality_selection_prompt,
    format_admiral_orders_for_captain,
    ship_has_guns,
    ship_ref,
    pd_doctrine_numbers,
    CaptainPersonality,
    PERSONALITY_PRESETS,
)
from .communication import (
    CaptainMessage, MessageType, MAX_DISCUSSION_EXCHANGES, MAX_ENEMY_MESSAGES_PER_DECISION,
    MAX_FLEET_SIGNALS_PER_DECISION, SIGNAL_TEXT_CHARS, clip, normalize_report, normalize_signal,
)
from .tools import FLEET_SIGNAL_TOOL, REPORT_TO_ADMIRAL_TOOL
from .admiral_tools import DISCUSS_WITH_ADMIRAL_TOOL


# Tools that make the ship DO something (move, fire, aim). A decision that
# produces none of their commands leaves the ship coasting with its guns on
# whatever policy they had - measured across three prompt-carrying battles,
# 31% of captain decisions were exactly that (an acknowledgment and nothing
# else), and two whole sides never moved or fired.
ACTION_TOOLS = frozenset({"set_maneuver", "set_heading", "launch_torpedo",
                          "set_weapons_order", "set_primary_target"})
# Follow-up rounds per checkpoint when a reply issued no action (all decide()
# calls of one checkpoint - discussion re-decides included - share it).
MAX_FOLLOWUP_ROUNDS = 2
# finish_reason values that mean the reply was cut off or broke, not that the
# model chose to say nothing (CP-09: 'length' at max_tokens spent entirely on
# reasoning, 'error' mid-generation - both used to be recorded as "none").
FAILED_FINISH_REASONS = frozenset({"error", "length"})
# Torpedo fire solutions per decision: the nearest contacts get a flown
# solution (each ~5-50 ms of shadow engine), the rest are out of reach anyway.
MAX_TORPEDO_SOLUTIONS = 6
MAX_THREAT_HULL_SOLUTIONS = 4
# Distances quoted in an order that disagree with this ship's sensors by more
# than this (and by NUMBER_CHECK_MIN_KM) get a NUMBER CHECK line.
NUMBER_CHECK_REL = 0.3
NUMBER_CHECK_MIN_KM = 30.0
_TRIPLE_RE = re.compile(
    r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)")
_XYZ_RE = re.compile(
    r"X\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,?\s*Y\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,?\s*Z\s*=\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE)
# A distance, not a speed: "86 km" but never the "3" of "3 km/s".
_KM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*km\b(?!\s*/\s*s)", re.IGNORECASE)
_THROTTLE_RE = re.compile(r"(\d{1,3})\s*%")
_INTERCEPT_RE = re.compile(r"\bINTERCEPT\b", re.IGNORECASE)
# Order prose is not a parser: these keep the text heuristics from printing
# an engine "check" of something the order does not say (a CHECK showing an
# order failing is a reason to DEVIATE, so a false one does real harm).
_NEGATION_RE = re.compile(r"\b(?:not|no|never|avoid|don'?t|cancel|stop|without)\b",
                          re.IGNORECASE)
_SPEED_AFTER_RE = re.compile(r"\s*(?:k?m\s*/\s*s|kps)\b", re.IGNORECASE)
_VELOCITY_BEFORE_RE = re.compile(r"\b(?:vel(?:ocity)?|moving|speed|drift(?:ing)?)\b[^()]{0,16}$",
                                 re.IGNORECASE)
_NOT_THROTTLE_RE = re.compile(r"\b(?:hull|armou?r|heat|fuel|integrity|health|delta|dv|PD)\b",
                              re.IGNORECASE)


def _ordered_intercept(text: str) -> Optional[float]:
    """
    Throttle (0-1) of an INTERCEPT the order text actually orders, or None.

    Skips a negated mention ("do NOT intercept"), and reads the throttle only
    from a percentage next to the word - in the same clause right after it
    ("INTERCEPT Lancer-1 at 80%", "INTERCEPT@80%") or just before it ("80%
    INTERCEPT") - never from an unrelated "hull is at 40%". 1.0 when unstated.
    """
    for m in _INTERCEPT_RE.finditer(text):
        clause = re.split(r"[.;:!?,\n]", text[max(0, m.start() - 40):m.start()])[-1]
        if _NEGATION_RE.search(" ".join(clause.split()[-3:])):
            continue
        after = re.split(r"[.;!?\n]", text[m.end():m.end() + 40], maxsplit=1)[0]
        pct = _THROTTLE_RE.search(after)
        if pct is not None and _NOT_THROTTLE_RE.search(after[:pct.start()][-14:]):
            pct = None
        if pct is None:
            pct = re.search(r"(\d{1,3})\s*%\s*(?:throttle\s*|thrust\s*)?$",
                            text[max(0, m.start() - 18):m.start()], re.IGNORECASE)
        return min(100, int(pct.group(1))) / 100.0 if pct else 1.0
    return None


def _quoted_triples(text: str) -> List[Tuple[float, float, float]]:
    """
    Coordinate triples quoted in order text ("(90,55,10)", "X=90, Y=55,
    Z=10"), minus velocities: a triple followed by km/s / m/s, or introduced
    by "velocity" / "moving at", is a speed, not a position.
    """
    out = []
    for m in list(_TRIPLE_RE.finditer(text)) + list(_XYZ_RE.finditer(text)):
        if _SPEED_AFTER_RE.match(text, m.end()):
            continue
        if _VELOCITY_BEFORE_RE.search(text[max(0, m.start() - 30):m.start()]):
            continue
        out.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def _has_launchers(ship: Any) -> bool:
    """True when the ship mounts torpedo launchers (tolerates test doubles)."""
    launchers = getattr(ship, "ready_torpedo_launchers", None)
    if isinstance(launchers, (list, tuple)):
        return bool(launchers)
    single = getattr(ship, "torpedo_launcher", None)
    return single is not None and hasattr(single, "current_magazine")


def _bare_name(name: str) -> str:
    """'OCS Lancer-4' -> 'Lancer-4' (admiral prose drops the fleet prefix)."""
    parts = str(name).split(" ", 1)
    return parts[1] if len(parts) == 2 and parts[0].isupper() and len(parts[0]) <= 4 else str(name)


@dataclass
class LLMCaptainConfig:
    """Configuration for an LLM captain."""
    name: str
    ship_name: str
    model: str = DEFAULT_MODEL
    personality: CaptainPersonality = CaptainPersonality.BALANCED
    personality_text: Optional[str] = None  # Custom personality description
    temperature: float = 0.7
    has_torpedoes: bool = False
    ship_type: str = "destroyer"
    fleet_data: Optional[Dict[str, Any]] = None
    # Accepted commander-notebook lessons for this model (cross-battle memory,
    # src/llm/notebook.py). Stable for the whole battle -> lives in the cached
    # doctrine prefix. None when notebooks are disabled.
    notebook_text: Optional[str] = None


def ship_has_torpedoes(ship_type: str, fleet_data: Dict[str, Any]) -> bool:
    """True if this ship class mounts a torpedo launcher, per the fleet data."""
    spec = (fleet_data or {}).get("ships", {}).get(ship_type, {})
    return any(
        (w.get("type") or "").startswith("torpedo")
        for w in spec.get("weapons", [])
    )


class LLMCaptain:
    """
    LLM-powered captain that makes strategic decisions via tools.

    Uses OpenRouter/LiteLLM for model flexibility.
    Decisions are made via structured tool calls, not text parsing.
    """

    def __init__(
        self,
        config: LLMCaptainConfig,
        client: CaptainClient,
    ):
        """
        Initialize the LLM captain.

        Args:
            config: Captain configuration
            client: LLM client for API calls
        """
        self.config = config
        self.client = client

        # Derive torpedo availability from the fleet data. has_torpedoes defaults
        # to False and nothing ever set it, so launch_torpedo was never offered to
        # any captain. For a corvette - whose only weapon IS the torpedo launcher
        # - that left the captain with no usable weapon at all.
        if not config.has_torpedoes and config.fleet_data and config.ship_type:
            config.has_torpedoes = ship_has_torpedoes(config.ship_type, config.fleet_data)

        self.tools = self._build_tools()

        # State tracking
        self.decision_count = 0
        self.last_call_failed = False
        self.enemy_ship_class: Optional[str] = None
        self.decision_history: List[Dict[str, Any]] = []
        self.pending_message: Optional[str] = None
        self.received_messages: List[CaptainMessage] = []
        self.message_history: List[Dict[str, Any]] = []  # Full conversation log
        self.has_surrendered = False
        self.has_proposed_draw = False
        self.has_retracted_draw = False

        # Personality (can be updated by select_personality)
        self.personality_text: Optional[str] = config.personality_text
        self.chosen_name: Optional[str] = None

        # Battle state tracking for context
        self.initial_distance_km: Optional[float] = None
        self.min_distance_km: float = float('inf')
        self.max_distance_km: float = 0
        self.passes_count: int = 0  # Number of close passes
        self.last_distance_km: float = 0

        # Shot history for learning from engagement data
        # Each entry: {distance_km, rel_velocity_kps, weapon, result: "HIT"/"MISS", damage_gj}
        self.shot_history: List[Dict[str, Any]] = []

        # Multi-ship targeting support
        self.primary_target_id: Optional[str] = None  # Current target ship ID
        self.targeting_me: List[str] = []  # Ship IDs that have us as primary target

        # Recent hits tracking (cleared each checkpoint)
        # Each entry: {time, weapon, location, damage_cm, remaining_cm, source}
        self.recent_hits: List[Dict[str, Any]] = []

        # Tool errors from the previous decision. There is no assistant/tool-result
        # round trip in this design, so without replaying them into the next turn
        # the model never learns that a call was rejected and repeats it forever.
        self.pending_tool_errors: List[str] = []
        self.last_tool_errors: List[str] = []
        # Accepted-with-a-warning results (e.g. a launch the engine check says
        # is ESCAPABLE or OVERKILL): the tool's answer, replayed next turn.
        self.pending_tool_notes: List[str] = []
        self.last_tool_notes: List[str] = []

        # Failed / truncated model calls (CP-09). lost_turn is the notice for
        # the NEXT prompt; last_failure_reason is for the battle record.
        self.lost_turn: Optional[Dict[str, Any]] = None
        self.last_failure_reason: Optional[str] = None
        # Follow-up rounds already spent per checkpoint time (R1).
        self._followups_used: Dict[float, int] = {}
        self.last_followup_rounds = 0
        # Shadow-engine fire solutions, memoised per decision.
        self._solution_cache: Dict[Any, Any] = {}
        # Standing orders as last seen (directive text/time, orders/time), so
        # a checkpoint without new orders shows the old ones WITH their age.
        self._last_directive_seen: Optional[Tuple[str, float]] = None
        self._last_orders_seen: Optional[Tuple[List[Any], float]] = None
        # Tool names offered so far (sticky: keeps the cached doctrine stable).
        self._known_tool_names: set = set()
        self._recorded_msg_ids: set = set()

        # Captain's log: self-authored notes (log_note tool) echoed back into
        # every subsequent turn. The rest of the prompt is telemetry the harness
        # compressed for the captain; this is the only channel where the captain
        # decides what its future self needs to remember.
        self.captain_log: List[Dict[str, Any]] = []

        # Weapon groups - will be populated when we know the ship type
        self.weapon_groups: Dict[str, List[str]] = {}

        # Current weapon configuration for display
        self.current_weapon_orders: Dict[str, str] = {
            "spinal": "HOLD_FIRE",
            "turret": "HOLD_FIRE",
        }

        # Admiral interaction
        self.has_admiral: bool = False  # Set by battle runner if Admiral exists
        self.admiral_orders: List[Any] = []  # Orders from Admiral
        self.fleet_directive: str = ""  # Overall fleet strategy
        self.discussion_exchanges: int = 0  # Track discussion rounds
        self.max_discussion_exchanges: int = MAX_DISCUSSION_EXCHANGES
        # This checkpoint's report to the Admiral (report_to_admiral, or the
        # legacy respond_to_orders alias): {status, reason, intent, contact,
        # request, tool}. The runner files it after the decision; the Admiral
        # reads it at the next checkpoint (admiral.captain_reports).
        self.order_response: Optional[Dict[str, Any]] = None

        # Fleet comms. outbox: messages composed this checkpoint (fleet_signal,
        # send_message), drained and routed by the runner after the decision.
        # inbox: comms_message dicts delivered to this ship this checkpoint
        # (fleet net, direct, engine datalink, enemy channel) - replaced by
        # the runner every checkpoint. fleet_peer_names: living friendly
        # ships, set with the inbox (no peers: no fleet_signal tool).
        # discussions: this checkpoint's discuss_with_admiral Q&A.
        self.outbox: List[Dict[str, Any]] = []
        self.inbox: List[Dict[str, Any]] = []
        self.inbox_delivered_t: Optional[float] = None
        self.fleet_peer_names: List[str] = []
        self.discussions: List[Dict[str, Any]] = []

    def setup_weapon_groups(self, ship_type: str, fleet_data: Dict[str, Any]) -> None:
        """Set up weapon groups based on ship type."""
        self.weapon_groups = get_weapon_groups_for_ship(ship_type, fleet_data)

    @property
    def name(self) -> str:
        """Get captain's name (from config or chosen during personality selection)."""
        return self.chosen_name or self.config.name

    @property
    def ship_name(self) -> str:
        """Get ship's name from config."""
        return self.config.ship_name

    def select_personality(self, distance_km: float, verbose: bool = False) -> Dict[str, Any]:
        """
        Let the LLM choose its personality before battle starts.

        Args:
            distance_km: Starting distance for scenario context
            verbose: Whether to print selection info

        Returns:
            Dict with chosen personality info
        """
        # Extract a clean model name for personalization
        # e.g., "anthropic/claude-3.5-sonnet" -> "Claude-3.5-Sonnet"
        model_path = self.config.model.replace("openrouter/", "")
        model_name = model_path.split("/")[-1]  # Get last part after provider
        # Capitalize nicely: claude-3.5-sonnet -> Claude-3.5-Sonnet
        model_name = "-".join(part.capitalize() for part in model_name.split("-"))

        # Pass the ACTUAL ship class. build_personality_selection_prompt defaults
        # to "Destroyer", and this call never overrode it - so every captain in
        # every battle formed its doctrine believing it commanded a destroyer.
        # In a corvette duel that is badly wrong: a corvette is a torpedo boat
        # with no guns at all, and the captains reasoned about gunnery.
        ship_class = (self.config.ship_type or "destroyer").replace("_", " ").title()
        prompt = build_personality_selection_prompt(
            distance_km,
            model_name=model_name,
            ship_class=ship_class,
            enemy_ship_class=(self.enemy_ship_class or ship_class),
        )

        messages = [{"role": "user", "content": prompt}]

        # Call LLM with personality selection tool (use captain's configured model)
        try:
            tool_calls = self.client.decide_with_tools(
                messages=messages,
                tools=PERSONALITY_SELECTION_TOOLS,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        except LLMCallError as e:
            print(f"[CAPTAIN {self.config.name}] personality selection failed: {e}")
            tool_calls = []

        result = {
            "personality_description": None,
        }

        for tc in tool_calls:
            if tc.name == "choose_personality":
                personality_desc = tc.arguments.get("personality_description", "")

                result["personality_description"] = personality_desc

                # Update personality
                if personality_desc:
                    self.config.personality = CaptainPersonality.CUSTOM
                    self.personality_text = personality_desc

                if verbose:
                    print(f"  [{self.config.ship_name}] Defined personality")
                    if personality_desc:
                        print(f"    {personality_desc[:80]}...")

        return result

    def receive_messages(self, messages: List[CaptainMessage]) -> None:
        """
        Receive messages from enemy captain.

        Args:
            messages: List of messages to receive
        """
        self.received_messages.extend(messages)

    def receive_admiral_orders(
        self,
        orders: List[Any],
        fleet_directive: str = "",
    ) -> None:
        """
        Receive orders from Admiral before making decisions.

        Args:
            orders: List of AdmiralOrder objects for this ship
            fleet_directive: Overall fleet strategy
        """
        self.admiral_orders = orders
        self.fleet_directive = fleet_directive
        # Orders reached this ship this checkpoint (an empty directive then
        # means "no NEW directive", not "nothing was delivered").
        self.orders_delivered = True
        # Reset discussion counter for new checkpoint
        self.discussion_exchanges = 0

    def get_tools_for_context(self) -> List[Dict[str, Any]]:
        """
        Get tools appropriate for current context.

        If Admiral exists:
        - Remove propose_draw and retract_draw (only Admiral can)
        - Add discuss_with_admiral and report_to_admiral (the captain->admiral
          channel; respond_to_orders survives only as an execution alias)
        With living friendly ships to talk to: add fleet_signal.
        """
        # Get base tools
        tools = self._build_tools()

        if self.has_admiral:
            # Remove draw tools - only Admiral can propose draws
            tools = [
                t for t in tools
                if t["function"]["name"] not in ("propose_draw", "retract_draw")
            ]
            tools.append(DISCUSS_WITH_ADMIRAL_TOOL)
            tools.append(REPORT_TO_ADMIRAL_TOOL)
        if self.fleet_peer_names:
            tools.append(FLEET_SIGNAL_TOOL)

        return tools

    def clear_admiral_context(self) -> None:
        """Clear Admiral context at end of checkpoint."""
        self.admiral_orders = []
        self.fleet_directive = ""
        self.orders_delivered = False
        self.discussion_exchanges = 0
        self.order_response = None
        self.discussions = []

    def decide(
        self,
        ship_id: str,
        simulation: Any,
    ) -> List[Any]:
        """
        Make strategic decisions using LLM tool calls.

        One request per checkpoint normally. Two harness rules on top:
        - a reply that issues NO action (only a report, a note, prose) gets a
          follow-up request in the same checkpoint saying so, up to
          MAX_FOLLOWUP_ROUNDS (R1: ack-only turns left ships drifting);
        - a call that fails or is cut off before any tool call is retried
          once, and if it still fails the previous maneuver is renewed and the
          next prompt says the orders were lost (CP-09).

        Args:
            ship_id: ID of this captain's ship
            simulation: CombatSimulation instance

        Returns:
            List of commands to execute (Maneuvers, fire orders, etc.)
        """
        self.last_tool_calls = []
        self.last_requested_tool_calls = []
        self.last_followup_rounds = 0
        self.last_failure_reason = None
        if self.has_surrendered:
            return []

        # Get ship state
        ship = simulation.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return []
        self.ship_id = ship_id
        now = float(getattr(simulation, "current_time", 0.0))
        interval = float(getattr(simulation, "decision_interval", 30.0) or 30.0)
        self._decision_interval_s = interval

        # Set up weapon groups if not done yet
        if not self.weapon_groups and self.config.fleet_data and self.config.ship_type:
            self.setup_weapon_groups(self.config.ship_type, self.config.fleet_data)

        # Get enemy for tactical info
        enemies = simulation.get_enemy_ships(ship_id)
        enemy = enemies[0] if enemies else None

        # Build status dicts
        self._solution_cache = {}
        ship_status = self._build_ship_status(ship)
        self._add_launch_status(ship_status, ship, simulation)
        tactical_status = self._build_tactical_status(ship, enemy, simulation)

        # Update battle tracking
        distance_km = tactical_status.get("distance_km", 1000)
        self._update_battle_tracking(distance_km)

        # FLEET NET / ENEMY TRANSMISSIONS for this checkpoint (inbox + legacy).
        comms = self._collect_comms(simulation, ship)

        # Roll last turn's rejected tool calls into this turn's feedback, then
        # start a fresh error bucket for the calls we are about to execute.
        self.last_tool_errors = self.pending_tool_errors
        self.pending_tool_errors = []
        self.last_tool_notes = self.pending_tool_notes
        self.pending_tool_notes = []

        # Get context-appropriate tools (may exclude draw tools if Admiral exists)
        tools = self.get_tools_for_context()
        tool_names = self._tool_names(tools)

        # Range control: closest approach, stopping distance, merge warning and
        # ready-made world-frame headings (cancel closure / open range / pass
        # wide / flank / coast & shoot) - the vector math done for the captain.
        range_infos, range_text = self._range_control(simulation, ship)
        merge_warnings = [f"{info.label}: {info.warning}" for info in range_infos if info.warning]

        orders_text = None
        if self.has_admiral:
            orders_text = self._format_orders(simulation, ship, tactical_status, range_infos,
                                              tool_names)

        from .execution_feedback import execution_feedback
        feedback = execution_feedback(simulation, [ship_id])

        prompt_kwargs = dict(
            captain_name=self.config.name,
            ship_name=self.config.ship_name,
            ship_status=ship_status,
            tactical_status=tactical_status,
            personality=self.config.personality,
            personality_text=self.personality_text,
            decision_history=self._format_decision_history(last_n=5, simulation=simulation),
            message_history=self._format_message_history(last_n=6, before=now),
            battle_summary=self._format_battle_summary(distance_km),
            shot_history=self._format_shot_history(last_n=10),
            recent_hits=self._format_recent_hits() or None,
            ship_type=self.config.ship_type,
            fleet_data=self.config.fleet_data,
            notebook_text=self.config.notebook_text,
            decision_interval_s=interval,
            decision_number=self.decision_count + 1,
            lost_turn=self._format_lost_turn(),
            range_control=range_text,
            merge_warnings=merge_warnings,
            orders=orders_text,
            comms=comms,
            own_faction=getattr(ship, "faction", None),
            captain_log=self._format_captain_log() or None,
            execution_feedback=feedback or None,
            tool_errors=self._format_tool_errors() or None,
            tool_names=tool_names,
            has_admiral=self.has_admiral,
        )
        # Split into a stable system prompt and a volatile user turn so the
        # provider can serve the doctrine from cache across checkpoints.
        messages = build_captain_messages(**prompt_kwargs)
        self.lost_turn = None  # delivered

        # Call LLM with tools (use captain's configured model)
        tool_calls, failure = self._call_model(messages, tools)
        if failure is not None:
            # One retry, with a nudge: a truncated reply usually spent its
            # whole budget reasoning, an errored one may simply recover.
            retry = self._with_notice(messages, (
                f"=== RETRY (same checkpoint, T+{now:.0f}) ===\nYour previous reply failed before "
                f"any tool call ({failure}). Keep your reasoning short and issue your orders as "
                "tool calls now."))
            tool_calls, failure = self._call_model(retry, tools)
            if failure is not None:
                return self._lose_turn(ship, simulation, failure)
        self.last_call_failed = False

        commands: List[Any] = []
        executed: List[ToolCall] = []
        requested: List[ToolCall] = list(tool_calls)
        state = {"maneuver_issued": False}
        self._execute_round(tool_calls, simulation, ship_id, commands, executed, state)

        # R1: a reply with no action gets told so, in the same checkpoint.
        self._followups_used = {t: n for t, n in self._followups_used.items() if t >= now}
        while enemies and self._needs_followup(executed, commands, ship, now):
            self._followups_used[now] = self._followups_used.get(now, 0) + 1
            self.last_followup_rounds += 1
            notice = self._followup_notice(executed, ship, simulation, tool_names)
            calls, failure = self._call_model(self._with_notice(messages, notice), tools)
            if failure is not None:
                print(f"[CAPTAIN {self.config.name}] follow-up failed: {failure}")
                break
            requested.extend(calls)
            self._execute_round(calls, simulation, ship_id, commands, executed, state)

        # Track decision
        self.decision_count += 1
        self.last_tool_calls = executed  # Store only executed calls for verbose output
        self.last_requested_tool_calls = requested
        self.decision_history.append({
            "checkpoint": self.decision_count,
            "time": now,
            "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in executed],
            "commands_count": len(commands),
            "followups": self.last_followup_rounds,
            "acted": any(self._is_action_command(c) for c in commands),
        })

        return commands

    # ------------------------------------------------------------------
    # decide() helpers
    # ------------------------------------------------------------------

    def _call_model(self, messages: List[Dict[str, Any]],
                    tools: List[Dict[str, Any]]) -> Tuple[List[ToolCall], Optional[str]]:
        """
        One model request -> (tool calls, failure reason or None).

        A reply that ends with finish_reason 'error' or 'length' and carries no
        tool call is a FAILURE, not a decision to do nothing.
        """
        try:
            # Per-captain temperature was parsed from fleet config and stored, but
            # never sent - every captain silently sampled at the shared client's
            # default regardless of its configured value.
            calls = self.client.decide_with_tools(
                messages,
                tools,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        except LLMCallError as e:
            print(f"[CAPTAIN {self.config.name}] decision call failed: {e}")
            return [], f"model call failed: {e}"
        calls = list(calls or [])
        finish = getattr(self.client, "last_finish_reason", None)
        finish = finish() if callable(finish) else None
        if not calls and isinstance(finish, str) and finish in FAILED_FINISH_REASONS:
            what = ("cut off at the output limit" if finish == "length"
                    else "the provider returned an error")
            print(f"[CAPTAIN {self.config.name}] reply {what} with no tool call")
            return [], f"reply {what} (finish_reason={finish}) before any tool call"
        return calls, None

    @staticmethod
    def _with_notice(messages: List[Dict[str, Any]], notice: str) -> List[Dict[str, Any]]:
        """
        Same request with a notice appended to the user turn.

        A fresh request rather than an assistant/tool-result continuation:
        reasoning models reject a continuation whose assistant turn lacks the
        (unavailable) thinking blocks, and the system prefix stays cached.
        """
        out = [dict(m) for m in messages]
        out[-1]["content"] = f"{out[-1]['content']}\n\n{notice}"
        return out

    def _execute_round(self, tool_calls: List[ToolCall], simulation: Any, ship_id: str,
                       commands: List[Any], executed: List[ToolCall],
                       state: Dict[str, Any]) -> None:
        """Execute one round of tool calls into ``commands`` / ``executed``."""
        # Track maneuver commands - only one maneuver per decision allowed
        maneuver_tools = {"set_maneuver", "set_heading"}
        seen = {(tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))
                for tc in executed}
        for tc in tool_calls:
            key = (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))
            if key in seen and tc.name not in ACTION_TOOLS:
                continue  # a follow-up repeating the report / note verbatim
            seen.add(key)
            # Skip duplicate maneuver commands (only first one takes effect)
            if tc.name in maneuver_tools:
                if state.get("maneuver_issued"):
                    self.pending_tool_errors.append(
                        f'{tc.name}: ignored because only the first maneuver is applied in a decision')
                    continue
                state["maneuver_issued"] = True

            cmd = self._execute_tool(tc, simulation, ship_id)
            executed.append(tc)
            if cmd is not None:
                # A tool may emit a salvo (e.g. launch_torpedo count=2).
                if isinstance(cmd, list):
                    commands.extend(c for c in cmd if c is not None)
                else:
                    commands.append(cmd)

    @staticmethod
    def _is_action_command(cmd: Any) -> bool:
        """A command that moves, aims or fires the ship (not radiators/notes)."""
        if isinstance(cmd, dict):
            return cmd.get("type") in ("set_target", "weapons_orders", "weapons_order",
                                       "launch_torpedo")
        return cmd is not None  # Maneuver objects (and any other engine command)

    def _admiral_heading_applied(self) -> bool:
        """
        True when an admiral order this checkpoint carries motion the runner
        already applied to the ship: a HEADING (``heading``) or a structured
        maneuver (``maneuver``: INTERCEPT with standoff, HOLD_RANGE, ...).
        """
        return any(isinstance(getattr(o, "heading", None), dict) or getattr(o, "maneuver", None)
                   for o in self.admiral_orders or [])

    def _needs_followup(self, executed: List[ToolCall], commands: List[Any],
                        ship: Any, now: float) -> bool:
        """
        True when this checkpoint's replies issued no action and a follow-up
        round is still allowed. Not when the captain surrendered, asked its
        admiral (the runner re-decides with the answer), or is already flying
        an admiral HEADING / MANEUVER order (acknowledging it is the
        prescribed reply - a follow-up would push a set_maneuver over it).
        """
        if self.has_surrendered or any(self._is_action_command(c) for c in commands):
            return False
        names = {tc.name for tc in executed}
        if "discuss_with_admiral" in names or "surrender" in names:
            return False
        if self._admiral_heading_applied():
            return False
        return self._followups_used.get(now, 0) < MAX_FOLLOWUP_ROUNDS

    def _followup_notice(self, executed: List[ToolCall], ship: Any, simulation: Any,
                         tool_names: List[str]) -> str:
        """The 'NO COMMANDS ISSUED' follow-up for an action-less reply."""
        now = simulation.current_time
        if executed:
            called = ", ".join(
                tc.name + (f"({tc.arguments.get('response_type') or tc.arguments.get('status')})"
                           if tc.arguments.get("response_type") or tc.arguments.get("status") else "")
                for tc in executed)
            what = f"Your reply called only: {called} - recorded."
        else:
            what = "Your reply contained no tool calls."
        speed = ship.velocity.magnitude / 1000.0
        m = getattr(ship, "current_maneuver", None)
        if m is not None and not m.is_complete(now + 1e-6):
            motion = (f"Your {m.maneuver_type.name} maneuver keeps running, but nothing was "
                      "aimed, launched or renewed.")
        else:
            motion = (f"No maneuver is active, so the ship COASTS at {speed:.2f} km/s for the next "
                      f"{simulation.decision_interval:g} s.")
        actions = ["set_maneuver or set_heading (MAINTAIN = a deliberate coast)"]
        if "set_weapons_order" in tool_names:
            actions.append("set_weapons_order")
        if "launch_torpedo" in tool_names:
            actions.append("launch_torpedo")
        actions.append("set_primary_target")
        lines = [f"=== FOLLOW-UP (same checkpoint, T+{now:.0f}) ===",
                 f"{what} NO COMMANDS ISSUED: nothing will move, aim or fire. {motion}"]
        if self.pending_tool_errors:
            lines.append("Rejected: " + "; ".join(self.pending_tool_errors[-3:]))
        lines.append(f"Issue your orders now as tool calls: {', '.join(actions)}. "
                     "Do not repeat the report or the note.")
        return "\n".join(lines)

    def _renewed_maneuver(self, ship: Any, now: float, interval: float) -> Optional[Any]:
        """The ship's current maneuver re-issued for the next window, if any."""
        m = getattr(ship, "current_maneuver", None)
        if m is None or not hasattr(m, "start_time"):
            return None
        if m.start_time >= now - 1e-6:
            return None  # already fresh (e.g. an admiral heading applied now)
        try:
            names = {f.name for f in fields(m)}
            changes = {"start_time": now, "duration": interval}
            # Runtime latches and formation sync belong to the old window.
            for name, value in (("burn_start_time", None), ("hold_velocity", None),
                                ("hold_velocity_t", -1.0)):
                if name in names:
                    changes[name] = value
            return replace(m, **changes)
        except TypeError:
            return None

    def _lose_turn(self, ship: Any, simulation: Any, reason: str) -> List[Any]:
        """
        CP-09: the model failed twice. Keep flying the previous maneuver for
        another window instead of letting it expire mid-merge, record the turn
        as LOST (not as a deliberate "none") and tell the next prompt.
        """
        now = simulation.current_time
        interval = getattr(simulation, "decision_interval", 30.0)
        self.last_call_failed = True
        self.last_failure_reason = reason
        renewed = self._renewed_maneuver(ship, now, interval)
        kept = fresh = None
        until = now + interval
        if renewed is not None:
            kept = f"{renewed.maneuver_type.name}@{renewed.throttle * 100:.0f}%"
        else:
            # A maneuver set THIS checkpoint (the runner applied an admiral
            # HEADING / MANEUVER order before asking) needs no renewal - it
            # flies on. The ship did not coast, and the record must not say so.
            m = getattr(ship, "current_maneuver", None)
            if (m is not None and getattr(m, "start_time", None) is not None
                    and m.start_time >= now - 1e-6 and not m.is_complete(now + 1e-6)):
                whose = "your Admiral's " if self._admiral_heading_applied() else ""
                fresh = f"{whose}{m.maneuver_type.name}@{m.throttle * 100:.0f}%"
                if 0 < m.duration < 1e6:  # duration <= 0: open-ended
                    until = m.start_time + m.duration
        self.lost_turn = {"time": now, "reason": reason, "renewed": kept, "kept": fresh,
                          "until": until}
        self.decision_count += 1
        self.decision_history.append({
            "checkpoint": self.decision_count, "time": now, "tool_calls": [],
            "commands_count": 1 if renewed is not None else 0,
            "lost": reason, "renewed": kept, "kept": fresh,
        })
        return [renewed] if renewed is not None else []

    def _format_lost_turn(self) -> Optional[str]:
        lost = self.lost_turn
        if not lost:
            return None
        if lost.get("renewed"):
            what = f"the engine renewed your previous {lost['renewed']} until T+{lost['until']:.0f}"
        elif lost.get("kept"):
            what = (f"{lost['kept']} (applied at T+{lost['time']:.0f}) kept flying until "
                    f"T+{lost['until']:.0f}")
        else:
            what = (f"no maneuver was active, so the ship COASTED "
                    f"T+{lost['time']:.0f}-{lost['until']:.0f}")
        return (f"!! YOUR LAST ORDERS WERE LOST (T+{lost['time']:.0f}: {lost['reason']}): {what}. "
                "Nothing else was aimed or launched. Re-issue your orders now.")

    def _tool_names(self, tools: List[Dict[str, Any]]) -> List[str]:
        """Names of the offered tools, sticky across the battle (cache-stable doctrine)."""
        self._known_tool_names.update(t["function"]["name"] for t in tools)
        return sorted(self._known_tool_names)

    def _range_control(self, simulation: Any, ship: Any) -> Tuple[List[Any], str]:
        """Range-control analysis (merge warnings feed THREATS TO ME) and its block."""
        try:
            from .range_control import analyze_range_control, format_range_control
            infos = analyze_range_control(simulation, [ship], detail=True)
            return infos, format_range_control(infos, who="Your")
        except Exception:  # noqa: BLE001 - advisory only: never let it kill a decision
            return [], ""

    def _collect_comms(self, simulation: Any, ship: Any) -> List[Dict[str, Any]]:
        """
        This checkpoint's traffic as comms_message dicts: ``self.inbox``
        (filled by the runner, comms schema) plus legacy CaptainMessage
        deliveries, each classified by the SENDER's side - nothing is
        labelled enemy traffic just for having arrived.

        This is the ONE place received traffic enters ``message_history``
        (the comms runner's ``receive_inbox`` only stores the list), once per
        message: a same-checkpoint re-decide sees the same inbox again and is
        de-duplicated by msg_id. Engine datalink items (sender_role 'engine':
        launches, kills, PD losses) are telemetry - shown in FLEET NET the
        turn they arrive, kept out of the 6-entry log they would flush.
        """
        from .prompts import is_friendly_comms
        now = simulation.current_time
        own_faction = getattr(ship, "faction", None)
        items: List[Dict[str, Any]] = []
        # A discussion re-decide in the same checkpoint must still see the
        # legacy deliveries the first decide() consumed.
        cached_t, cached = getattr(self, "_legacy_comms", (None, []))
        legacy: List[Dict[str, Any]] = list(cached) if cached_t == now else []
        for item in getattr(self, "inbox", None) or []:
            if not isinstance(item, dict):
                continue
            items.append(item)
            if item.get("sender_role") == "engine":
                continue
            key = item.get("msg_id") or (item.get("sender_id"), item.get("sent_t"),
                                         str(item.get("text", "")))
            if key in self._recorded_msg_ids:
                continue
            self._recorded_msg_ids.add(key)
            self._record_received_message(
                str(item.get("text", "")), float(item.get("sent_t", now) or now),
                sender=ship_ref(item.get("sender_name"), item.get("sender_id")),
                side="friendly" if is_friendly_comms(item, own_faction) else "enemy",
                received_at=now)
        for msg in self.received_messages:
            sender_ship = (simulation.get_ship(msg.sender_id)
                           if getattr(msg, "sender_id", None) and hasattr(simulation, "get_ship")
                           else None)
            friendly = bool(sender_ship is not None and own_faction is not None
                            and sender_ship.faction == own_faction)
            if msg.message_type == MessageType.NORMAL:
                text = msg.content
            else:
                text = msg.format_for_llm()
            legacy.append({
                "channel": "fleet_net" if friendly else "enemy_broadcast",
                "sender_role": "admiral" if msg.message_type in (
                    MessageType.ADMIRAL_ORDER, MessageType.ADMIRAL_TO_ADMIRAL) else "captain",
                "sender_name": msg.ship_name or msg.sender_name,
                "sender_id": msg.sender_id,
                "faction": sender_ship.faction if sender_ship is not None else None,
                "text": text,
                "sent_t": msg.timestamp,
            })
            self._record_received_message(
                msg.content, msg.timestamp,
                sender=ship_ref(msg.ship_name or msg.sender_name, msg.sender_id),
                side="friendly" if friendly else "enemy", received_at=now)
        self.received_messages.clear()
        self._legacy_comms = (now, legacy)
        return items + legacy

    def _add_launch_status(self, status: Dict[str, Any], ship: Any, simulation: Any) -> None:
        """Tube readiness and launches possible before the next decision."""
        if status.get("torpedoes_remaining") is None:
            return
        now = getattr(simulation, "current_time", 0.0)
        tubes = []
        for tube in getattr(ship, "ready_torpedo_launchers", None) or []:
            try:
                wait = max(0.0, tube.last_launch_time + tube.cooldown_seconds - now)
            except (AttributeError, TypeError):
                continue
            tubes.append({"ready": wait <= 0 and tube.current_magazine > 0, "cooldown": wait})
        if tubes:
            status["torpedo_tubes"] = tubes
        try:
            status["launch_capacity"] = self._max_torpedo_salvo(ship.ship_id, simulation)
        except Exception:  # noqa: BLE001 - display only
            pass

    # ------------------------------------------------------------------
    # Ship resolution, fire solutions, order checks
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ship(ref: Any, ships: List[Any]) -> Tuple[Optional[Any], List[Any]]:
        """
        Find a ship by id, exact name, bare name ("Lancer-4" for "OCS
        Lancer-4") or a UNIQUE partial match. Returns (ship, candidates):
        ship is None when nothing or several ships match (candidates then
        lists the ambiguous matches). Names repeat across fleets ("TIS
        Lancer-1" / "OCS Lancer-1"), so a first-substring-wins match picked
        whichever ship happened to be listed first.
        """
        text = str(ref or "").strip()
        if not text:
            return None, []
        low = text.lower()
        exact = [s for s in ships
                 if s.ship_id == text or str(getattr(s, "name", s.ship_id)).lower() == low]
        if exact:
            return exact[0], []
        bare = [s for s in ships
                if _bare_name(getattr(s, "name", s.ship_id)).lower() == _bare_name(text).lower()]
        if len(bare) == 1:
            return bare[0], []
        partial = [s for s in ships if low in str(getattr(s, "name", s.ship_id)).lower()]
        if len(partial) == 1:
            return partial[0], []
        return None, bare or partial

    def _torpedo_solution(self, simulation: Any, ship: Any, target: Any):
        """Flown torpedo fire solution (memoised per decision); None if n/a."""
        if target is None or not self.config.fleet_data or not self.config.has_torpedoes:
            return None
        from .fire_solutions import torpedo_solution
        if len(self._solution_cache) > 256:  # callers that never reset it per decision
            self._solution_cache.clear()
        try:
            return torpedo_solution(simulation, ship, target, self.config.fleet_data,
                                    cache=self._solution_cache)
        except Exception as e:  # noqa: BLE001 - advisory only, never fail a decision
            print(f"[CAPTAIN {self.config.name}] torpedo solution failed: {e}")
            return None

    @staticmethod
    def _salvo_check_text(sol: Any, ordered: int, capacity: Optional[int]) -> str:
        if sol is None:
            return ""
        if sol.verdict == "UNREACHABLE":
            text = f"{sol.summary()}. DEVIATE is the correct response - say why."
        elif sol.verdict == "ESCAPABLE":
            text = f"{sol.summary()}. Launch only if he is not evading; otherwise DEVIATE."
        elif sol.verdict in ("TOO CLOSE", "OVERKILL"):
            text = f"{sol.summary()}. DEVIATE (say why) unless the admiral knows better."
        else:
            text = sol.summary()
        if capacity is not None and ordered > capacity:
            text += f" You can launch only {capacity} before your next decision."
        return text

    def _cpa_under_burn(self, simulation: Any, ship: Any, enemies: List[Any],
                        direction: Any, throttle: float, burn_s: float,
                        horizon_s: float = 600.0) -> Optional[Tuple[float, float, str, float]]:
        """
        Closest approach to any enemy if this ship turns onto ``direction``,
        burns ``burn_s`` at ``throttle`` and then coasts (enemies coast).
        Returns (cpa_km, t_s, enemy name, relative speed km/s at the pass).
        """
        from ..physics import Vector3D
        try:
            accel = ship.kinematic_state.max_acceleration_ms2() * ship.get_effective_thrust_fraction()
        except (AttributeError, TypeError):
            return None
        d = direction.normalized() if direction.magnitude > 0 else direction
        turn_s = 0.0
        if hasattr(simulation, "_estimate_turn_time"):
            try:
                turn_s = simulation._estimate_turn_time(ship, ship.forward.angle_to(d), engines_on=True)
            except (AttributeError, TypeError, ValueError):
                turn_s = 0.0
        a = d * (accel * max(0.0, min(1.0, throttle)))
        best = None
        dt = 1.0
        for e in enemies:
            p = ship.position - e.position
            v = ship.velocity - e.velocity
            t = 0.0
            while t < horizon_s:
                acc = a if turn_s <= t < turn_s + burn_s else Vector3D(0.0, 0.0, 0.0)
                p = p + v * dt + acc * (0.5 * dt * dt)
                v = v + acc * dt
                t += dt
                rng = p.magnitude
                if best is None or rng < best[0]:
                    best = (rng, t, getattr(e, "name", e.ship_id), v.magnitude, e.ship_id)
                # Past the closest approach and opening after the burn: done.
                if t > turn_s + burn_s and p.dot(v) > 0:
                    break
        if best is None:
            return None
        return best[0] / 1000.0, best[1], ship_ref(best[2], best[4]), best[3] / 1000.0

    def _spec_motion(self, spec: Dict[str, Any], thr: float, target: Any, ship: Any,
                     enemies: List[Any], friendlies: List[Any],
                     interval: float) -> Optional[Tuple[str, Any, float, float]]:
        """
        The straight-line motion of an admiral's structured maneuver, for the
        ENGINE ORDER CHECK - (label, direction, throttle, burn s) - or None.

        Only what flies straight is modelled: a plain INTERCEPT (it never
        brakes) and a HEADING toward a point or ship. An INTERCEPT with a
        standoff or arrival speed, HOLD_RANGE and the rest brake or steer on
        their own, so a straight-burn closest approach would be a false MERGE.
        """
        from ..physics import Vector3D
        kind = str(spec.get("maneuver_type") or "").upper()
        if thr <= 0:
            return None
        if kind == "INTERCEPT":
            if spec.get("standoff_km") is not None or spec.get("arrival_closing_kps") is not None:
                return None
            aim = self._resolve_ship(spec["target"], enemies)[0] if spec.get("target") else None
            aim = aim or target
            if aim is None and self.primary_target_id:
                aim = next((e for e in enemies if e.ship_id == self.primary_target_id), None)
            if aim is None and enemies:
                aim = min(enemies, key=lambda e: (e.position - ship.position).magnitude)
            if aim is None:
                return None
            return (f"INTERCEPT@{thr * 100:.0f}% held for 2 checkpoints",
                    aim.position - ship.position, thr, 2 * interval)
        if kind == "HEADING":
            vec = None
            point = spec.get("toward_point")
            if isinstance(point, dict):
                vec = Vector3D(float(point.get("x", 0)) * 1000.0, float(point.get("y", 0)) * 1000.0,
                               float(point.get("z", 0)) * 1000.0) - ship.position
            elif spec.get("toward_ship"):
                other, _ = self._resolve_ship(spec["toward_ship"], list(enemies) + list(friendlies))
                vec = other.position - ship.position if other is not None else None
            if vec is not None and vec.magnitude > 0:
                return ("this heading for one checkpoint, then coasting", vec, thr, interval)
        return None

    @staticmethod
    def _enemy_mention(text: str, enemy: Any, own_side: List[Any]) -> Optional[re.Match]:
        """
        Where ``text`` names ``enemy``: by id, full name, or bare name - the
        bare name only when no ship of ours shares it and no other fleet's
        prefix stands in front of it ("Form up on TIS Lancer-2" is not about
        OCS Lancer-2; names repeat across fleets).
        """
        name = str(getattr(enemy, "name", enemy.ship_id))
        for candidate in (enemy.ship_id, name):
            m = re.search(r"(?<![\w-])" + re.escape(candidate) + r"(?![\w-])", text, re.IGNORECASE)
            if m:
                return m
        bare = _bare_name(name)
        ours = [str(getattr(s, "name", s.ship_id)) for s in own_side]
        if bare == name or any(_bare_name(n).lower() == bare.lower() for n in ours):
            return None
        # Our fleet prefixes ("TIS"), when they differ from the enemy's.
        foreign = {n[:-len(_bare_name(n))].strip().upper() for n in ours if _bare_name(n) != n}
        foreign.discard(name[:-len(bare)].strip().upper())
        for m in re.finditer(r"(?<![\w-])" + re.escape(bare) + r"(?![\w-])", text, re.IGNORECASE):
            prefix = re.search(r"(\w+)\s+$", text[max(0, m.start() - 8):m.start()])
            if prefix and prefix.group(1).upper() in foreign:
                continue
            return m
        return None

    def _order_check(self, order: Any, simulation: Any, ship: Any,
                     range_infos: List[Any]) -> Dict[str, Any]:
        """Engine-resolved lines for one order (see format_admiral_orders_for_captain)."""
        from ..physics import Vector3D
        from .range_control import MERGE_KM
        check: Dict[str, Any] = {}
        enemies = simulation.get_enemy_ships(ship.ship_id)
        friendlies = (simulation.get_friendly_ships(ship.ship_id)
                      if hasattr(simulation, "get_friendly_ships") else [])
        text = str(getattr(order, "order_text", "") or "")
        interval = float(getattr(simulation, "decision_interval", 30.0) or 30.0)

        target = None
        tname = getattr(order, "suggested_target", None) or getattr(order, "torpedo_target", None)
        if tname:
            target, candidates = self._resolve_ship(tname, enemies)
            if target is not None:
                rel = target.position - ship.position
                rng = rel.magnitude
                closing = -(rel.normalized()).dot(target.velocity - ship.velocity) / 1000.0 if rng else 0.0
                closing = 0.0 if abs(closing) < 0.005 else closing
                rate = (f"closing {closing:.2f} km/s" if closing >= 0
                        else f"OPENING {-closing:.2f} km/s")
                line = (f"{ship_ref(getattr(target, 'name', target.ship_id), target.ship_id)} - "
                        f"your sensors: {rng / 1000.0:.0f} km, {rate}")
                twin, _ = self._resolve_ship(_bare_name(str(tname)), list(friendlies) + [ship])
                if twin is not None and _bare_name(str(tname)) == _bare_name(getattr(twin, "name", "")):
                    line += (f" (the name also matches your own "
                             f"{ship_ref(getattr(twin, 'name', twin.ship_id), twin.ship_id)})")
                check["target"] = line
            elif candidates:
                check["target"] = (f"{tname} is ambiguous: " + ", ".join(
                    ship_ref(getattr(c, "name", c.ship_id), c.ship_id) for c in candidates[:4])
                    + " - name the id")
            else:
                check["target"] = f"{tname} - no such enemy on your sensors (destroyed or misnamed?)"

        salvo = getattr(order, "torpedo_salvo", None)
        if salvo and self.config.has_torpedoes:
            tgt = target
            if getattr(order, "torpedo_target", None):
                tgt, _ = self._resolve_ship(order.torpedo_target, enemies)
            if tgt is None and self.primary_target_id:
                tgt = next((e for e in enemies if e.ship_id == self.primary_target_id), None)
            if tgt is None and enemies:
                tgt = min(enemies, key=lambda e: (e.position - ship.position).magnitude)
            sol = self._torpedo_solution(simulation, ship, tgt)
            try:
                capacity = self._max_torpedo_salvo(ship.ship_id, simulation)
            except Exception:  # noqa: BLE001 - display only
                capacity = None
            check["salvo"] = self._salvo_check_text(sol, int(salvo), capacity)

        # ENGINE ORDER CHECK: what the ordered motion does to the closest approach.
        motion = None
        heading = getattr(order, "heading", None)
        spec = getattr(order, "maneuver", None)
        thr = getattr(order, "applied_throttle", None)
        thr = getattr(order, "throttle", None) if thr is None else thr
        if isinstance(heading, dict):
            thr = 1.0 if thr is None else float(thr)
            vec = Vector3D(float(heading.get("x", 0)), float(heading.get("y", 0)),
                           float(heading.get("z", 0)))
            if vec.magnitude > 0 and thr > 0:
                motion = ("this heading for one checkpoint, then coasting", vec, thr, interval)
        elif isinstance(spec, dict):
            # A structured admiral maneuver (AdmiralOrder.maneuver), applied by
            # the runner: it, not the prose, is what the ship flies.
            if thr is None:
                thr = spec.get("throttle")
            motion = self._spec_motion(spec, 1.0 if thr is None else float(thr), target,
                                       ship, enemies, friendlies, interval)
        else:
            thr = _ordered_intercept(text)
            aim = target
            if aim is None and self.primary_target_id:
                aim = next((e for e in enemies if e.ship_id == self.primary_target_id), None)
            if aim is None and enemies:
                aim = min(enemies, key=lambda e: (e.position - ship.position).magnitude)
            if thr is not None and aim is not None and thr > 0:
                motion = (f"INTERCEPT@{thr * 100:.0f}% held for 2 checkpoints",
                          aim.position - ship.position, thr, 2 * interval)
        if motion and enemies:
            label, vec, thr, burn = motion
            res = self._cpa_under_burn(simulation, ship, enemies, vec, thr, burn)
            if res is not None:
                cpa_km, t_s, who, rel_kps = res
                engine = (f"{label} -> closest approach {cpa_km:.1f} km to {who} in {t_s:.0f} s, "
                          f"passing at {rel_kps:.1f} km/s")
                if cpa_km < MERGE_KM:
                    engine += " = a MERGE (seconds of fire, then flying apart)"
                    if any(getattr(i, "warning", None) for i in range_infos):
                        check["conflict"] = ("Your order and RANGE CONTROL conflict (merge): choose "
                                             "and explain - DEVIATE is allowed.")
                check["engine"] = engine

        # Absolute positions quoted in the text: resolve to a direction from here.
        own_km = ship.position * (1.0 / 1000.0)
        points = []
        for x, y, z in _quoted_triples(text):
            p = Vector3D(x, y, z)
            if p.magnitude <= 2.0:
                continue  # a unit-ish direction, not a position
            off = p - own_km
            if off.magnitude <= 0:
                continue
            u = off.normalized()
            points.append(f"({p.x:.0f},{p.y:.0f},{p.z:.0f}) km is a POSITION: from you dir_to "
                          f"({u.x:+.2f},{u.y:+.2f},{u.z:+.2f}), {off.magnitude:.0f} km - "
                          f"set_heading takes that direction, not the position")
            if len(points) >= 2:
                break
        if points:
            check["points"] = points

        # Distances quoted for named enemies vs this ship's own sensors.
        numbers = []
        own_side = list(friendlies) + [ship]
        for e in enemies:
            m = self._enemy_mention(text, e, own_side)
            if not m:
                continue
            # The distance quoted right after the name, inside the same clause.
            snippet = re.split(r"[.;]\s", text[m.end():m.end() + 40], maxsplit=1)[0]
            km = _KM_RE.search(snippet)
            if not km:
                continue
            claimed = float(km.group(1))
            actual = (e.position - ship.position).magnitude / 1000.0
            if abs(claimed - actual) > max(NUMBER_CHECK_MIN_KM, NUMBER_CHECK_REL * actual):
                numbers.append(f"order quotes {claimed:.0f} km next to "
                               f"{ship_ref(getattr(e, 'name', e.ship_id), e.ship_id)}; "
                               f"your sensors: {actual:.0f} km from you")
            if len(numbers) >= 2:
                break
        if numbers:
            check["numbers"] = numbers
        return check

    def _format_orders(self, simulation: Any, ship: Any, tactical_status: Dict[str, Any],
                       range_infos: List[Any], tool_names: List[str]) -> str:
        """The ORDERS block: orders as fields, with ages and engine checks."""
        now = simulation.current_time
        delivered = getattr(self, "orders_delivered", bool(self.admiral_orders or self.fleet_directive))
        directive = self.fleet_directive if delivered else None
        previous_directive = self._last_directive_seen
        if directive:
            self._last_directive_seen = (directive, now)
            previous_directive = None
        orders = list(self.admiral_orders or [])
        previous_orders = self._last_orders_seen
        standing = [o for o in orders if not str(o.order_text).lstrip().startswith("[CLARIFICATION]")]
        if standing:
            self._last_orders_seen = (standing, now)
        checks = []
        for order in orders:
            try:
                checks.append(self._order_check(order, simulation, ship, range_infos))
            except Exception as e:  # noqa: BLE001 - checks are advisory
                print(f"[CAPTAIN {self.config.name}] order check failed: {e}")
                checks.append({})
        reply = "report_to_admiral" if "report_to_admiral" in tool_names else "respond_to_orders"
        return format_admiral_orders_for_captain(
            orders, directive, now=now, checks=checks, reply_tool=reply,
            previous_directive=previous_directive,
            previous_orders=None if orders else previous_orders,
        )

    def revert_last_decision(self) -> None:
        """
        Undo the bookkeeping of the most recent ``decide()`` call.

        The runner may call ``decide()`` several times for a single checkpoint
        (Admiral discussion, then a forced retry) and keep only one of the
        resulting command lists. Without this, one wall-clock checkpoint produced
        up to three history entries all stamped with the same simulation time -
        flooding the 5-entry window the captain is shown - and pushed
        ``decision_count`` (rendered as "DECISION POINT n") out of step with the
        actual checkpoint number.
        """
        if self.decision_history:
            self.decision_history.pop()
        if self.decision_count > 0:
            self.decision_count -= 1

    def get_last_decision_summary(self) -> str:
        """Get a human-readable summary of the last decision."""
        if not hasattr(self, 'last_tool_calls') or not self.last_tool_calls:
            return "No actions"

        actions = []
        had_discussion_or_response = False

        for tc in self.last_tool_calls:
            if tc.name == "set_maneuver":
                maneuver = tc.arguments.get("maneuver_type", "?")
                throttle = tc.arguments.get("throttle", 1.0)
                actions.append(f"{maneuver} @ {throttle*100:.0f}%")
            elif tc.name == "set_primary_target":
                target = tc.arguments.get("target_name", "?")
                actions.append(f"TARGET: {target}")
            elif tc.name == "set_heading":
                direction = tc.arguments.get("direction") or {}
                throttle = tc.arguments.get("throttle", 1.0)
                point = tc.arguments.get("toward_point")
                if tc.arguments.get("toward_ship"):
                    where = f"to ship {tc.arguments['toward_ship']}"
                elif isinstance(point, dict):
                    where = (f"to point ({point.get('x', 0)},{point.get('y', 0)},"
                             f"{point.get('z', 0)}) km")
                elif isinstance(direction, dict):
                    where = (f"({direction.get('x', 0):.1f},{direction.get('y', 0):.1f},"
                             f"{direction.get('z', 0):.1f})")
                else:
                    where = "(?)"
                actions.append(f"HEADING {where} @ {throttle*100:.0f}%")
            elif tc.name == "set_weapons_order":
                spinal = tc.arguments.get("spinal_mode")
                turret = tc.arguments.get("turret_mode")
                if spinal and turret:
                    actions.append(f"WEAPONS spinal:{spinal}, turret:{turret}")
                elif spinal:
                    actions.append(f"WEAPONS spinal:{spinal}")
                elif turret:
                    actions.append(f"WEAPONS turret:{turret}")
                else:
                    actions.append("WEAPONS all: FIRE_WHEN_OPTIMAL")
            elif tc.name == "launch_torpedo":
                actions.append("LAUNCH TORPEDO")
            elif tc.name == "set_radiators":
                cmd = radiator_command_from_args(tc.arguments)
                if cmd.get("mode") == "auto":
                    actions.append(f"AUTO radiators (extend at {cmd['auto_extend_pct']:.0f}%)")
                else:
                    actions.append("EXTEND radiators" if cmd["extend"] else "RETRACT radiators")
            elif tc.name == "send_message":
                msg = tc.arguments.get("message", "")[:30]
                recipient = tc.arguments.get("recipient", "ALL_ENEMIES")
                if recipient != "ALL_ENEMIES":
                    actions.append(f"MSG ({recipient}): \"{msg}...\"" if len(msg) >= 30 else f"MSG ({recipient}): \"{msg}\"")
                else:
                    actions.append(f"MSG: \"{msg}...\"" if len(msg) >= 30 else f"MSG: \"{msg}\"")
            elif tc.name == "log_note":
                note = tc.arguments.get("note", "")[:40]
                actions.append(f"LOG: \"{note}\"")
            elif tc.name == "surrender":
                actions.append("SURRENDER")
            elif tc.name == "propose_draw":
                actions.append("PROPOSE DRAW")
            elif tc.name == "retract_draw":
                actions.append("RETRACT DRAW")
            elif tc.name in ("respond_to_orders", "discuss_with_admiral"):
                had_discussion_or_response = True

        if actions:
            return ", ".join(actions)
        elif had_discussion_or_response:
            return "(maintaining current orders)"
        else:
            return "No actions"

    def _launch_outcomes(self, simulation: Any) -> Dict[str, Dict[str, Any]]:
        """
        torpedo_id -> {t, target_id, outcome, gj} for every round THIS ship
        launched, from the simulation's own event log (impact > PD-destroyed
        > blinded > fuel out > retargeted > in flight).
        """
        ship_id = getattr(self, "ship_id", None)
        events = getattr(simulation, "events", None) if simulation is not None else None
        if not ship_id or not events:
            return {}
        rank = {"in flight": 0, "retargeted": 1, "fuel out": 2, "blinded": 3,
                "destroyed by PD": 4, "hit": 5}
        out: Dict[str, Dict[str, Any]] = {}
        for ev in events:
            kind = getattr(getattr(ev, "event_type", None), "name", "")
            data = getattr(ev, "data", None) or {}
            tid = data.get("torpedo_id")
            if not tid:
                continue
            if kind == "TORPEDO_LAUNCHED" and ev.ship_id == ship_id:
                out[tid] = {"t": ev.timestamp, "target_id": ev.target_id,
                            "outcome": "in flight", "gj": 0.0}
                continue
            rec = out.get(tid)
            if rec is None:
                continue
            outcome = {"TORPEDO_IMPACT": "hit", "PD_TORPEDO_DESTROYED": "destroyed by PD",
                       "PD_TORPEDO_DISABLED": "blinded", "TORPEDO_FUEL_EXHAUSTED": "fuel out",
                       "TORPEDO_RETARGETED": "retargeted"}.get(kind)
            if outcome is None:
                continue
            if outcome == "hit":
                rec["gj"] = float(data.get("total_damage_gj", 0.0) or 0.0)
            if outcome == "retargeted":
                rec["target_id"] = data.get("new_target_id", rec["target_id"])
            if rank[outcome] > rank[rec["outcome"]]:
                rec["outcome"] = outcome
        return out

    @staticmethod
    def _history_part(name: str, args: Dict[str, Any], resolve) -> Optional[str]:
        """Compact text for one executed tool call (None: shown elsewhere)."""
        args = args or {}
        if name == "set_maneuver":
            thr = args.get("throttle", 1.0)
            try:
                thr = float(thr)
            except (TypeError, ValueError):
                thr = 1.0
            return f"{args.get('maneuver_type', '?')}@{thr * 100:.0f}%"
        if name == "set_heading":
            thr = args.get("throttle", 1.0)
            try:
                thr = float(thr)
            except (TypeError, ValueError):
                thr = 1.0
            d = args.get("direction")
            if isinstance(d, dict):
                where = (f"({float(d.get('x', 0)):+.2f},{float(d.get('y', 0)):+.2f},"
                         f"{float(d.get('z', 0)):+.2f})")
            elif isinstance(args.get("toward_point"), dict):
                p = args["toward_point"]
                where = (f"toward point ({float(p.get('x', 0)):+.0f},{float(p.get('y', 0)):+.0f},"
                         f"{float(p.get('z', 0)):+.0f}) km")
            elif args.get("toward_ship"):
                where = f"toward {resolve(args['toward_ship'])}"
            else:
                where = "?"
            return f"HEADING {where}@{thr * 100:.0f}%"
        if name == "set_weapons_order":
            modes = [f"{k[:-5]} {v}" for k, v in args.items() if k.endswith("_mode") and v]
            return "guns " + (", ".join(modes) if modes else "FIRE_WHEN_OPTIMAL")
        if name == "set_primary_target":
            return f"tgt {resolve(args.get('target_id') or args.get('target_name', '?'))}"
        if name == "set_radiators":
            cmd = radiator_command_from_args(args)
            return "radiators " + ("auto" if cmd.get("mode") == "auto"
                                   else "extend" if cmd["extend"] else "retract")
        if name in ("respond_to_orders", "report_to_admiral"):
            kind = args.get("response_type") or args.get("status") or "ACK"
            kind = "ACK" if str(kind).upper().startswith("ACK") else str(kind).upper()
            reason = args.get("deviation_reason") or args.get("reason") or ""
            reason = " ".join(str(reason).split())
            return f"report {kind}" + (f": {reason[:60]}{'...' if len(reason) > 60 else ''}"
                                       if reason and kind != "ACK" else "")
        if name in ("send_message", "fleet_signal"):
            return "signal fleet" if name == "fleet_signal" else \
                f"msg {str(args.get('recipient', 'ALL_ENEMIES')).lower()}"
        if name == "log_note":
            return "log"
        if name == "discuss_with_admiral":
            return "asked admiral"
        if name in ("surrender", "propose_draw", "retract_draw"):
            return name.upper()
        if name == "launch_torpedo":
            return None  # rendered from the launch events, with outcomes
        return name

    def _format_decision_history(self, last_n: int = 5, simulation: Any = None) -> str:
        """
        Recent decisions as issued, newest last, with what the launches did.

        Every executed tool appears (torpedo launches, headings, targets,
        reports); the old renderer knew maneuvers and spinal/turret orders
        only, so a torpedo boat's history read "none" on the turn it fired.
        """
        if not self.decision_history:
            return ""
        names: Dict[str, str] = {}
        ships = getattr(simulation, "ships", None) if simulation is not None else None
        by_name = ({str(getattr(s, "name", sid)).lower(): sid for sid, s in ships.items()}
                   if isinstance(ships, dict) else {})

        def resolve(ref: Any) -> str:
            ref = str(ref)
            if ref not in names:
                sid = ref if isinstance(ships, dict) and ref in ships else by_name.get(ref.lower())
                ship = ships.get(sid) if sid else None
                names[ref] = ship_ref(getattr(ship, "name", sid), sid) if ship is not None else ref
            return names[ref]

        outcomes = self._launch_outcomes(simulation)
        interval = float(getattr(simulation, "decision_interval", 30.0) or 30.0) \
            if simulation is not None else 30.0
        lines = ["YOUR RECENT DECISIONS (newest last; launches with what they did):"]
        for d in self.decision_history[-last_n:]:
            t = d["time"]
            if d.get("lost"):
                kept = (f"engine renewed {d['renewed']}" if d.get("renewed")
                        else f"{d['kept']} kept flying" if d.get("kept")
                        else "the ship coasted")
                lines.append(f"  T+{t:.0f} LOST ({d['lost']}) - {kept}")
                continue
            parts = []
            requested = 0
            for tc in d["tool_calls"]:
                if tc["name"] == "launch_torpedo":
                    try:
                        requested += int(tc["args"].get("count", 1))
                    except (TypeError, ValueError):
                        requested += 1
                part = self._history_part(tc["name"], tc["args"], resolve)
                if part:
                    parts.append(part)
            launched = [r for r in outcomes.values() if t - 1e-6 <= r["t"] < t + interval - 1e-6]
            if launched:
                by_target: Dict[str, List[Dict[str, Any]]] = {}
                for r in launched:
                    by_target.setdefault(r["target_id"], []).append(r)
                for target_id, rounds in by_target.items():
                    hits = [r for r in rounds if r["outcome"] == "hit"]
                    tally = []
                    if hits:
                        tally.append(f"{len(hits)} hit {sum(r['gj'] for r in hits):.0f} GJ")
                    for outcome in ("destroyed by PD", "blinded", "fuel out", "retargeted",
                                    "in flight"):
                        n = sum(1 for r in rounds if r["outcome"] == outcome)
                        if n:
                            tally.append(f"{n} {outcome}")
                    parts.append(f"torp x{len(rounds)} -> {resolve(target_id)}: {', '.join(tally)}")
            elif requested:
                parts.append(f"torp x{requested} requested - none left the tube")
            text = " | ".join(parts) if parts else "none"
            if d.get("followups"):
                text += f" (after {d['followups']} follow-up{'s' if d['followups'] > 1 else ''})"
            if d.get("acted") is False:
                text += " - NO COMMANDS: the ship coasted"
            lines.append(f"  T+{t:.0f} {text}")
        return "\n".join(lines)

    def _format_message_history(self, last_n: int = 6, before: Optional[float] = None) -> str:
        """
        Earlier traffic, labelled by side. Messages that arrived THIS
        checkpoint (received_at == before) are in FLEET NET / ENEMY
        TRANSMISSIONS already and are not repeated here.
        """
        msgs = [m for m in self.message_history
                if before is None or m.get("received_at") is None or m["received_at"] < before]
        if not msgs:
            return ""
        lines = ["COMMUNICATION LOG (earlier checkpoints):"]
        for msg in msgs[-last_n:]:
            text = msg["text"]
            # Truncate long messages
            if len(text) > 100:
                text = text[:100] + "..."
            if msg["sender"] == "self":
                who = "You"
            elif msg.get("side") == "friendly":
                who = f"{msg.get('from') or 'friendly'} (friendly)"
            else:
                who = f"ENEMY {msg.get('from') or ''}".rstrip()
            lines.append(f"  T+{msg['time']:.0f}s [{who}]: \"{text}\"")
        return "\n".join(lines)

    def _format_battle_summary(self, distance_km: float) -> str:
        """Format battle progression summary."""
        if self.initial_distance_km is None:
            return ""

        line = (f"BATTLE PROGRESSION: started {self.initial_distance_km:.0f} km apart, closest "
                f"approach so far {self.min_distance_km:.0f} km")
        if self.passes_count > 0:
            line += f", {self.passes_count} close pass{'es' if self.passes_count > 1 else ''} (<100 km)"
        return line

    def _update_battle_tracking(self, distance_km: float) -> None:
        """Update battle state tracking for history context."""
        # Track initial distance
        if self.initial_distance_km is None:
            self.initial_distance_km = distance_km

        # Track min/max distance
        self.min_distance_km = min(self.min_distance_km, distance_km)
        self.max_distance_km = max(self.max_distance_km, distance_km)

        # Detect close passes (crossed under 100km threshold)
        if self.last_distance_km > 100 and distance_km <= 100:
            self.passes_count += 1

        self.last_distance_km = distance_km

    def record_shot(
        self,
        weapon: str,
        distance_km: float,
        rel_velocity_kps: float,
        result: str,
        damage_gj: float = 0.0
    ) -> None:
        """Record a shot fired by this captain for learning."""
        self.shot_history.append({
            "weapon": weapon,
            "distance_km": distance_km,
            "rel_velocity_kps": rel_velocity_kps,
            "result": result,  # "HIT" or "MISS"
            "damage_gj": damage_gj,
        })

    def record_hit_received(
        self,
        time: float,
        weapon: str,
        location: str,
        damage_cm: float,
        remaining_cm: float,
        source_ship: str,
    ) -> None:
        """Record a hit received by this ship."""
        self.recent_hits.append({
            "time": time,
            "weapon": weapon,
            "location": location,
            "damage_cm": damage_cm,
            "remaining_cm": remaining_cm,
            "source": source_ship,
        })

    def clear_recent_hits(self) -> None:
        """Clear recent hits at start of each checkpoint."""
        self.recent_hits = []

    def _build_tools(self) -> List[Dict[str, Any]]:
        """
        Tool surface matched to this ship's actual armament.

        Falls back to the generic set when fleet data is unavailable (test
        doubles, legacy callers).
        """
        if self.config.fleet_data and self.config.ship_type:
            return get_captain_tools_for_ship(
                self.config.ship_type,
                self.config.fleet_data,
                has_torpedoes=self.config.has_torpedoes,
                decision_interval_s=getattr(self, "_decision_interval_s", 30.0),
            )
        return get_captain_tools(has_torpedoes=self.config.has_torpedoes)

    def _max_torpedo_salvo(self, ship_id: str, simulation: Any) -> int:
        """
        How many torpedoes can physically be launched before the next decision.

        Bounded by the reload interval over the decision window and by rounds
        remaining in the magazine.
        """
        if hasattr(simulation, "torpedo_launch_capacity"):
            capacity = simulation.torpedo_launch_capacity(ship_id)
            if isinstance(capacity, int):
                return capacity
        # Tolerate simulation doubles that do not implement get_ship: fall back to
        # the reload-derived ceiling rather than refusing to launch.
        ship = None
        if simulation is not None and hasattr(simulation, "get_ship"):
            try:
                ship = simulation.get_ship(ship_id)
            except Exception:
                ship = None
        launcher = getattr(ship, "torpedo_launcher", None) if ship else None
        if launcher is None:
            interval = getattr(simulation, "decision_interval", 30.0) or 30.0
            return max(1, int(interval // 12.0))

        interval = getattr(simulation, "decision_interval", 30.0) or 30.0
        cooldown = getattr(launcher, "cooldown_seconds", 12.0) or 12.0
        # Scale with the number of launchers the hull mounts. ShipCombatState
        # currently exposes a single `torpedo_launcher`, so this is 1 today - but
        # the ceiling is derived rather than hardcoded so multi-launcher hulls
        # get their full salvo automatically once the model supports them.
        launcher_count = getattr(ship, "torpedo_launcher_count", None) or 1
        by_cooldown = max(1, int(interval // cooldown)) * int(launcher_count)
        by_magazine = getattr(launcher, "current_magazine", None)
        if by_magazine is None:
            return by_cooldown
        return max(0, min(by_cooldown, int(by_magazine)))

    def _record_tool_error(self, message: str) -> None:
        """
        Record a rejected/ignored tool call so the model can be told about it.

        Errors used to go to stdout only, which the model never sees; the same
        invalid call would then be reissued at every checkpoint for the rest of
        the battle.
        """
        print(f"[CAPTAIN {self.name}] {message}")
        self.pending_tool_errors.append(message)

    def _record_tool_note(self, message: str) -> None:
        """
        An EXECUTED call's answer worth seeing next turn (a launch warning, an
        ignored parameter). Its own channel, replayed as "ENGINE NOTES ON YOUR
        LAST ORDERS (accepted ...)": on the tool-error channel it would sit
        under "REJECTED, reissue correctly" and invite a second launch.
        """
        print(f"[CAPTAIN {self.name}] note: {message}")
        self.pending_tool_notes.append(message)

    def _torpedo_feasible(self, simulation: Any, ship_id: str, target_id: str,
                          count: int, args: Dict[str, Any]) -> bool:
        """
        Launch feasibility from a flown shadow round (see fire_solutions).

        Refused: UNREACHABLE while the target is opening (it misses even if he
        holds course) - unless override_check=true. Accepted with a warning
        note: UNREACHABLE while closing (rare), ESCAPABLE, TOO CLOSE, OVERKILL,
        and a lone round trickled into point defense that blinds it.
        """
        get_ship = getattr(simulation, "get_ship", None)
        if get_ship is None:
            return True
        ship, target = get_ship(ship_id), get_ship(target_id)
        if ship is None or target is None:
            return True
        sol = self._torpedo_solution(simulation, ship, target)
        if sol is None:
            return True
        ref = ship_ref(getattr(target, 'name', target_id), target_id)
        override = str(args.get("override_check", "")).lower() in ("true", "1", "yes")
        if sol.verdict == "UNREACHABLE" and sol.closing_kps < 0 and not override:
            self._record_tool_error(
                f"launch_torpedo x{count} -> {ref}: REFUSED, {sol.summary()}. "
                f"Pass override_check=true to launch anyway.")
            return False
        if sol.verdict != "REACHABLE":
            self._record_tool_note(f"launch_torpedo x{count} -> {ref} launched: {sol.summary()}")
        elif count + sol.friendly_inbound < sol.salvo_to_saturate:
            self._record_tool_note(
                f"launch_torpedo x{count} -> {ref} launched: TRICKLE - {sol.pd_cover} PD turrets "
                f"cover him; rounds arriving alone get blinded (salvo >= "
                f"{sol.salvo_to_saturate} together needed, {sol.friendly_inbound} already inbound)")
        return True

    def _format_tool_errors(self) -> str:
        """Format last turn's rejected tool calls (and launch warnings) for the next prompt."""
        lines = []
        if self.last_tool_errors:
            lines.append("ERRORS FROM YOUR LAST ORDERS (these calls were REJECTED, reissue correctly):")
            lines.extend(f"  - {err}" for err in self.last_tool_errors[-5:])
        notes = getattr(self, "last_tool_notes", None) or []
        if notes:
            lines.append("ENGINE NOTES ON YOUR LAST ORDERS (accepted, but check these):")
            lines.extend(f"  - {note}" for note in notes[-4:])
        return "\n".join(lines)

    # Keep the echoed log small: it rides in the volatile user turn every
    # checkpoint, so a runaway log would tax every remaining call of the battle.
    MAX_LOG_ENTRIES = 3
    MAX_LOG_NOTE_CHARS = 300

    def _format_captain_log(self) -> str:
        """Render the captain's own notes back into the next turn."""
        if not self.captain_log:
            return ""
        lines = ["YOUR CAPTAIN'S LOG (notes you wrote to your future self - execute or supersede them):"]
        for entry in self.captain_log[-self.MAX_LOG_ENTRIES:]:
            lines.append(f"  T+{entry['time']:.0f}s: {entry['note']}")
        return "\n".join(lines)

    def _format_torpedo_section(
        self,
        ship_status: Dict[str, Any],
        tactical_status: Dict[str, Any],
    ) -> str:
        """
        Render this ship's own torpedo magazine.

        Inbound threats are rendered once, in the turn state, by
        prompts.format_torpedo_threats - this section used to duplicate them.
        Returns "" for hulls without launchers so gun ships carry no dead text.
        """
        remaining = ship_status.get("torpedoes_remaining")
        if remaining is None:
            return ""
        capacity = ship_status.get("torpedo_capacity")
        cap_txt = f"/{capacity}" if capacity else ""
        return f"YOUR TORPEDOES REMAINING: {remaining}{cap_txt}"

    def set_primary_target(self, target_id: Optional[str]) -> None:
        """Set the primary target for this captain."""
        self.primary_target_id = target_id

    def update_targeting_me(self, ship_ids: List[str]) -> None:
        """Update list of ships that have us as their primary target."""
        self.targeting_me = ship_ids

    def get_primary_target_id(self) -> Optional[str]:
        """Get current primary target ID."""
        return self.primary_target_id

    def _format_shot_history(self, last_n: int = 10) -> str:
        """Format recent shot history for prompt."""
        if not self.shot_history:
            return ""

        recent = self.shot_history[-last_n:]

        # Calculate stats by range bracket
        hits_by_range = {"<100km": [0, 0], "100-300km": [0, 0], ">300km": [0, 0]}
        for shot in self.shot_history:
            d = shot["distance_km"]
            if d < 100:
                bracket = "<100km"
            elif d < 300:
                bracket = "100-300km"
            else:
                bracket = ">300km"
            hits_by_range[bracket][1] += 1  # total
            if shot["result"] == "HIT":
                hits_by_range[bracket][0] += 1  # hits

        lines = ["YOUR SHOTS FIRED (at enemy):"]

        # Stats by range
        for bracket, (hits, total) in hits_by_range.items():
            if total > 0:
                pct = (hits / total) * 100
                lines.append(f"  {bracket}: {hits}/{total} hits ({pct:.0f}%)")

        # Recent shots detail
        lines.append("  Recent:")
        for shot in recent[-5:]:  # Last 5 only for detail
            result_str = f"HIT enemy {shot['damage_gj']:.1f}GJ" if shot["result"] == "HIT" else "MISS"
            # rel_velocity_kps follows the repo-wide convention -r_hat . v_rel,
            # i.e. POSITIVE means closing (see battle_runner._record_captain_shot).
            # This comparison used to be inverted, so every shot the captain was
            # shown had its geometry described backwards.
            closing = "closing" if shot["rel_velocity_kps"] > 0 else "separating"
            lines.append(
                f"    You fired {shot['weapon']}: {shot['distance_km']:.0f}km, "
                f"{abs(shot['rel_velocity_kps']):.1f}km/s {closing} -> {result_str}"
            )

        return "\n".join(lines)

    def _format_recent_hits(self) -> str:
        """Format recent hits received for the prompt."""
        if not self.recent_hits:
            return ""

        lines = ["DAMAGE TAKEN (last 30s):"]
        for hit in self.recent_hits[-5:]:  # Last 5 hits
            time = hit["time"]
            weapon = hit["weapon"].capitalize()
            location = hit["location"].upper()
            damage = hit["damage_cm"]
            remaining = hit["remaining_cm"]
            source = hit.get("source", "Unknown")
            lines.append(
                f"  T+{time:.0f}s: {weapon} from {source} → {location} armor "
                f"(-{damage:.1f} cm, {remaining:.1f} cm remaining)"
            )

        return "\n".join(lines)

    def _record_sent_message(self, message: str, time: float) -> None:
        """Record a message sent by this captain."""
        self.message_history.append({
            "sender": "self",
            "time": time,
            "text": message,
        })

    def _record_received_message(self, message: str, time: float,
                                 sender: Optional[str] = None, side: str = "enemy",
                                 received_at: Optional[float] = None) -> None:
        """Record a received message with the sender's side (friendly/enemy)."""
        self.message_history.append({
            "sender": "enemy" if side != "friendly" else "friendly",
            "side": side,
            "from": sender,
            "time": time,
            "received_at": received_at,
            "text": message,
        })

    def _build_ship_status(self, ship: Any) -> Dict[str, Any]:
        """Build ship status dict from ship state."""
        status = {
            "hull_integrity": ship.hull_integrity,
            "delta_v_remaining": ship.remaining_delta_v_kps,
        }

        # Thermal
        if ship.thermal_system:
            status["heat_percent"] = ship.thermal_system.heat_percent
            status["heatsink_capacity"] = ship.thermal_system.heatsink.capacity_gj
            # Damaged-but-deployed panels count as extended (they still reject heat)
            extended = getattr(ship.thermal_system, "radiators_extended", False)
            status["radiators_extended"] = extended if isinstance(extended, bool) else False
            # Net rate, time to 100% / meltdown retracted vs extended, panels
            outlook = getattr(ship.thermal_system, "outlook", None)
            outlook = outlook() if callable(outlook) else None
            if isinstance(outlook, dict):
                status["thermal"] = outlook
        else:
            status["heat_percent"] = 0
            status["heatsink_capacity"] = 320
            status["radiators_extended"] = False

        # Armor
        if ship.armor:
            from ..combat import HitLocation

            nose = ship.armor.get_section(HitLocation.NOSE)
            lateral = ship.armor.get_section(HitLocation.LATERAL)
            tail = ship.armor.get_section(HitLocation.TAIL)
            # 0.0 means "no armor left on this facing" - never invent a
            # plausible-looking thickness, the captain acts on these numbers.
            status["nose_armor"] = nose.current_thickness_cm if nose else 0.0
            status["lateral_armor"] = lateral.current_thickness_cm if lateral else 0.0
            status["tail_armor"] = tail.current_thickness_cm if tail else 0.0
        else:
            status["nose_armor"] = 0.0
            status["lateral_armor"] = 0.0
            status["tail_armor"] = 0.0

        # Weapon status - use weapon name instead of slot
        weapon_status = {}
        if hasattr(ship, 'weapons'):
            for slot, weapon_state in ship.weapons.items():
                # Get weapon name from weapon spec if available
                weapon_name = slot
                if hasattr(weapon_state, 'weapon') and hasattr(weapon_state.weapon, 'name'):
                    weapon_name = weapon_state.weapon.name
                weapon_status[weapon_name] = {
                    "operational": weapon_state.is_operational,
                    "ready": weapon_state.is_ready if hasattr(weapon_state, 'is_ready') else True,
                    "cooldown": weapon_state.cooldown_remaining if hasattr(weapon_state, 'cooldown_remaining') else 0,
                }
        status["weapons"] = weapon_status

        # Gun groups as the captain commands them (one entry per weapon group:
        # mode, how many mounts are ready / reloading / destroyed). Keyed by
        # slot - the name-keyed dict above merges identical turrets.
        gun_groups = []
        weapons = getattr(ship, 'weapons', None) or {}
        for group, slots in (self.weapon_groups or {}).items():
            states = [weapons[s] for s in slots if s in weapons]
            if not states:
                continue
            ready = reloading_n = destroyed = 0
            reloading = []
            for ws in states:
                if not getattr(ws, 'is_operational', True):
                    destroyed += 1
                elif getattr(ws, 'cooldown_remaining', 0) > 0:
                    reloading.append(float(ws.cooldown_remaining))
                else:
                    ready += 1
            weapon = getattr(states[0], 'weapon', None)
            label = getattr(weapon, 'name', None) or group.replace('_', ' ').title()
            mode = (self.current_weapon_orders.get(group)
                    or (self.current_weapon_orders.get('turret') if group != 'spinal' else None)
                    or 'HOLD_FIRE')
            gun_groups.append({'group': group, 'label': label, 'mode': mode, 'count': len(states),
                               'ready': ready, 'reloading': reloading, 'destroyed': destroyed})
        if gun_groups:
            status["gun_groups"] = gun_groups

        # Torpedo magazine. The launch_torpedo tool warns "Limited ammunition!"
        # but the remaining count was never disclosed anywhere, so the captain had
        # no way to know when the magazine was empty.
        # NOTE: sum across ALL launchers - reading only ship.torpedo_launcher
        # under-reported a 4-launcher torpedo cruiser's magazine by 4x.
        launchers = getattr(ship, 'ready_torpedo_launchers', None)
        if not isinstance(launchers, (list, tuple)) or not launchers:
            single = getattr(ship, 'torpedo_launcher', None)
            launchers = [single] if single is not None else []
        def _launcher_count(launcher, *attrs) -> Optional[int]:
            for attr in attrs:
                value = getattr(launcher, attr, None)
                if isinstance(value, (int, float)):
                    return int(value)
            return None

        if launchers:
            per_launcher = [
                _launcher_count(l, 'current_magazine', 'torpedoes_remaining')
                for l in launchers
            ]
            capacities = [_launcher_count(l, 'magazine_capacity')
                          for l in launchers]
            if any(v is not None for v in per_launcher):
                status["torpedoes_remaining"] = sum(v or 0 for v in per_launcher)
                capacity = sum(v or 0 for v in capacities)
                status["torpedo_capacity"] = capacity or None

        # Point defense turret state for the status display. PD is automatic,
        # but the captain must know how much blinding capacity is left.
        pd_list = getattr(ship, 'point_defense', None)
        if isinstance(pd_list, (list, tuple)):
            status["pd_turrets_total"] = len(pd_list)
            status["pd_turrets_operational"] = sum(
                1 for pd in pd_list if getattr(pd, 'is_operational', True)
            )

        # Module damage status
        module_status = {}
        if ship.module_layout:
            for module in ship.module_layout.get_all_modules():
                if module.health_percent < 100:
                    module_status[module.name] = {
                        "health": module.health_percent,
                        "operational": module.is_functional,
                        "destroyed": module.is_destroyed,
                        "type": module.module_type.value,
                    }
        status["damaged_modules"] = module_status

        return status

    def _calculate_impact_bearing(self, ship: Any, proj_pos: Any, proj_vel: Any) -> str:
        """
        Calculate which armor section a projectile is likely to hit.

        Returns bearing like 'NOSE', 'TAIL', 'PORT', 'STARBOARD', 'PORT-AFT', etc.
        """
        import math

        # Direction from projectile to ship (approach vector)
        to_ship = (ship.position - proj_pos).normalized()

        # Get ship orientation vectors
        forward = ship.forward
        up = ship.up if hasattr(ship, 'up') else type(forward)(0, 0, 1)
        right = forward.cross(up).normalized() if hasattr(forward, 'cross') else type(forward)(0, 1, 0)

        # Calculate angles
        forward_dot = forward.dot(to_ship)  # positive = coming from ahead
        right_dot = right.dot(to_ship)  # positive = coming from starboard
        up_dot = up.dot(to_ship)  # positive = coming from above

        # Determine primary bearing
        bearings = []

        # Forward/Aft component
        if forward_dot > 0.5:
            bearings.append("NOSE")
        elif forward_dot < -0.5:
            bearings.append("TAIL")

        # Port/Starboard component
        if right_dot > 0.3:
            bearings.append("STARBOARD")
        elif right_dot < -0.3:
            bearings.append("PORT")

        # If no strong direction, it's a flank shot
        if not bearings:
            if abs(right_dot) > abs(forward_dot):
                bearings.append("PORT" if right_dot < 0 else "STARBOARD")
            else:
                bearings.append("NOSE" if forward_dot > 0 else "TAIL")

        bearing = "-".join(bearings)

        # Add armor zone hint
        if "NOSE" in bearing:
            bearing += " (frontal)"
        elif "TAIL" in bearing:
            bearing += " (rear)"
        else:
            bearing += " (flank)"

        return bearing

    def _identify_projectile(self, proj: Any) -> tuple:
        """
        Identify an inbound round from its launch parameters.

        Returns:
            (label, kinetic_energy_gj) - label is the weapon name when it can be
            matched against fleet data, otherwise a class based on muzzle velocity.

        The previous implementation classified by the projectile's WORLD-frame
        speed (>8 km/s = "Spinal"). Projectiles inherit shooter velocity, so that
        threshold mislabels every round fired from a moving ship and permanently
        mislabelled the 4.7 km/s siege coiler round (the heaviest in the game) as
        a light turret round. Muzzle velocity and mass are frame-independent and
        are recorded on the projectile at launch.
        """
        muzzle_kps = getattr(proj, 'muzzle_velocity_kps', None)
        mass_kg = getattr(proj, 'mass_kg', None)

        weapon_types = (self.config.fleet_data or {}).get("weapon_types", {})
        if muzzle_kps is not None and mass_kg is not None:
            for wname, spec in weapon_types.items():
                spec_muzzle = spec.get("muzzle_velocity_kps")
                spec_mass = spec.get("warhead_mass_kg")
                if spec_muzzle is None or spec_mass is None:
                    continue
                if abs(spec_muzzle - muzzle_kps) < 1e-6 and abs(spec_mass - mass_kg) < 1e-6:
                    # Report the weapon's NOMINAL (muzzle-frame) energy, which is
                    # what the weapon tables in the prompt list. The projectile's
                    # own kinetic_energy_gj is world-frame and therefore depends
                    # on the shooter's velocity, so quoting it here would
                    # contradict those tables for the same round.
                    return wname, spec.get("kinetic_energy_gj", getattr(proj, 'kinetic_energy_gj', 0.0))

        energy_gj = getattr(proj, 'kinetic_energy_gj', 0.0)
        if muzzle_kps is None:
            return "Unknown round", energy_gj
        # Frame-independent fallback: spinal mounts are the high-muzzle-velocity
        # weapons, turrets the low ones.
        return ("Spinal" if muzzle_kps > 8 else "Turret"), energy_gj

    def _build_enemy_info(self, ship: Any, enemy: Any, simulation: Any) -> Dict[str, Any]:
        """Build tactical info for a single enemy ship."""
        import math

        info = {
            "ship_id": enemy.ship_id,
            "name": getattr(enemy, 'name', enemy.ship_id),
            "ship_class": getattr(enemy, 'ship_class', 'unknown'),
            # ShipCombatState carries ship_type, not ship_class. Without this the
            # prompt formatter's fallback never fires and every enemy renders as a
            # generic "(ship)" - captains could not tell a corvette from a
            # dreadnought, which drives target priority.
            "ship_type": getattr(enemy, 'ship_type', None),
        }

        # Calculate relative position
        rel_pos = enemy.position - ship.position
        distance_m = rel_pos.magnitude
        distance_km = distance_m / 1000
        info["distance_km"] = distance_km

        info["relative_position"] = {
            "x": rel_pos.x / 1000,
            "y": rel_pos.y / 1000,
            "z": rel_pos.z / 1000,
        }

        # Body-frame bearing. The world-frame vector above must never be labelled
        # "ahead/starboard/above" - those words are only meaningful relative to
        # where this ship is actually pointing. Beta-fleet ships start facing
        # (-1,0,0), so world +x is BEHIND them and the raw labels were inverted
        # from the first checkpoint of every battle.
        try:
            fwd = ship.forward.normalized()
            up = ship.up.normalized()
            starboard = up.cross(fwd).normalized()
            info["relative_bearing_km"] = {
                "forward": rel_pos.dot(fwd) / 1000,
                "starboard": rel_pos.dot(starboard) / 1000,
                "up": rel_pos.dot(up) / 1000,
            }
        except (AttributeError, TypeError):
            # Ship without a usable orientation (e.g. a test double). The prompt
            # formatter falls back to explicitly world-labelled coordinates rather
            # than printing misleading body-frame words.
            pass

        # Calculate relative velocity
        rel_vel = enemy.velocity - ship.velocity
        info["relative_velocity"] = {
            "x": rel_vel.x / 1000,
            "y": rel_vel.y / 1000,
            "z": rel_vel.z / 1000,
        }

        # Closing rate
        if distance_m > 0:
            info["closing_rate"] = -rel_pos.normalized().dot(rel_vel) / 1000
        else:
            info["closing_rate"] = 0

        # Angle to enemy
        if distance_m > 0:
            direction_to_enemy = rel_pos.normalized()
            dot = ship.forward.dot(direction_to_enemy)
            dot = max(-1.0, min(1.0, dot))
            info["angle_deg"] = math.degrees(math.acos(dot))
        else:
            info["angle_deg"] = 0

        # Show the same P(hit) that gates each gun shot, plus the number of
        # opportunities in this decision window (including a partial salvo).
        # simulation.gun_firing_solution is the gate's own call: it prices in
        # where the target can be by the time the round arrives (dodges,
        # unforeseen burns), not just the round's dispersion.
        window_s = getattr(simulation, "decision_interval", 30.0)
        forecasts = []
        for slot, ws in getattr(ship, "weapons", {}).items():
            if ws.weapon.is_missile or ws.weapon.muzzle_velocity_kps <= 0:
                continue
            solution = simulation.gun_firing_solution(ship, enemy, ws)
            probability = solution.hit_probability if solution.can_fire else 0.0
            in_arc = ws.is_target_in_arc(ship.forward, rel_pos.normalized())
            shots = ws.shots_in_window(window_s) if in_arc and solution.can_fire else 0
            forecasts.append({
                "weapon_slot": slot, "weapon_name": ws.weapon.name,
                "hit_probability": probability, "shots": shots,
                "window_s": window_s, "expected_hits": probability * shots,
                # Expected damage per shot (P(hit) x round energy): the number
                # that ranks targets, not closeness (R10).
                "expected_gj_per_shot": probability * float(
                    getattr(ws.weapon, "kinetic_energy_gj", 0.0) or 0.0),
            })
        info["weapon_hit_forecasts"] = forecasts

        # World-frame unit vector to him (paste-able into set_heading), his
        # live point defense and magazine, and his remaining gun output.
        if distance_m > 0:
            u = rel_pos.normalized()
            info["dir_to"] = {"x": u.x, "y": u.y, "z": u.z}
        pd_list = getattr(enemy, "point_defense", None)
        if isinstance(pd_list, (list, tuple)):
            info["pd_operational"] = sum(1 for pd in pd_list if getattr(pd, "is_operational", True))
        if _has_launchers(enemy):
            remaining = getattr(enemy, "torpedoes_remaining", None)
            if isinstance(remaining, (int, float)):
                info["torpedoes_remaining"] = int(remaining)
        gun_rate = 0.0
        enemy_weapons = getattr(enemy, "weapons", None)
        for ews in (enemy_weapons.values() if isinstance(enemy_weapons, dict) else ()):
            w = getattr(ews, "weapon", None)
            if (w is None or getattr(w, "is_missile", False)
                    or (getattr(w, "muzzle_velocity_kps", 0) or 0) <= 0
                    or not getattr(ews, "is_operational", True)):
                continue
            salvo = max(1, int(getattr(w, "salvo_size", 1) or 1))
            cycle = (float(getattr(w, "cooldown_s", 0) or 0)
                     + (salvo - 1) * float(getattr(w, "intra_salvo_cooldown_s", 0) or 0))
            if cycle > 0:
                gun_rate += float(getattr(w, "kinetic_energy_gj", 0) or 0) * salvo * 60.0 / cycle
        if gun_rate > 0:
            info["gun_gj_per_min"] = gun_rate
        info["hit_chance"] = 100 * max(
            (f["hit_probability"] for f in forecasts), default=0.0)

        # Enemy condition
        info["hull_percent"] = enemy.hull_integrity

        # Armor status
        if enemy.armor:
            enemy_nose = enemy.armor.get_section("nose")
            enemy_lateral = enemy.armor.get_section("lateral")
            enemy_tail = enemy.armor.get_section("tail")
            info["armor"] = {
                "nose_damage_pct": enemy_nose.damage_percent if enemy_nose else 0,
                "lateral_damage_pct": enemy_lateral.damage_percent if enemy_lateral else 0,
                "tail_damage_pct": enemy_tail.damage_percent if enemy_tail else 0,
            }
        else:
            info["armor"] = {}

        # Combat stats
        info["shots_fired"] = enemy.shots_fired
        info["hits_scored"] = enemy.hits_scored
        info["damage_dealt_gj"] = enemy.damage_dealt_gj
        info["damage_taken_gj"] = enemy.damage_taken_gj

        return info

    def _build_tactical_status(
        self,
        ship: Any,
        enemy: Optional[Any],
        simulation: Any,
    ) -> Dict[str, Any]:
        """Build tactical status dict with multi-ship support."""
        from ..physics import Vector3D
        import math

        status = {
            "sim_time": simulation.current_time,
            "ship_forward": {
                "x": ship.forward.x,
                "y": ship.forward.y,
                "z": ship.forward.z,
            },
            # Own ship combat stats
            "our_shots": ship.shots_fired,
            "our_hits": ship.hits_scored,
            "our_damage_dealt": ship.damage_dealt_gj,
            "our_damage_taken": ship.damage_taken_gj,
            # Multi-ship data
            "enemies": [],
            "friendlies": [],
            "primary_target_id": self.primary_target_id,
            "targeting_me": self.targeting_me,
            "incoming_projectiles": [],
        }

        # Get all ships
        all_enemies = simulation.get_enemy_ships(ship.ship_id)
        all_friendlies = simulation.get_friendly_ships(ship.ship_id) if hasattr(simulation, 'get_friendly_ships') else []

        # Build enemy info list with primary target first
        enemies_info = []
        for e in all_enemies:
            info = self._build_enemy_info(ship, e, simulation)
            info["is_primary_target"] = (e.ship_id == self.primary_target_id)
            info["has_us_targeted"] = (e.ship_id in self.targeting_me)
            enemies_info.append(info)

        # Sort: primary target first, then by distance
        enemies_info.sort(key=lambda x: (not x["is_primary_target"], x["distance_km"]))
        status["enemies"] = enemies_info

        # MY SHIP in the world frame: position and velocity are what
        # set_heading directions and admiral coordinates are expressed in.
        # Captains used to get neither, and copied absolute positions from
        # orders into set_heading (one corvette burned 144 deg off target).
        try:
            pos, vel = ship.position, ship.velocity
            own = {
                "ship_id": ship.ship_id,
                "position_km": {"x": pos.x / 1000, "y": pos.y / 1000, "z": pos.z / 1000},
                "velocity_kps": {"x": vel.x / 1000, "y": vel.y / 1000, "z": vel.z / 1000},
                "speed_kps": vel.magnitude / 1000,
            }
            try:
                own["accel_g"] = (ship.kinematic_state.max_acceleration_ms2()
                                  * ship.get_effective_thrust_fraction() / 9.80665)
            except (AttributeError, TypeError):
                pass
            if hasattr(simulation, "_estimate_turn_time"):
                try:
                    own["turn_90_s"] = simulation._estimate_turn_time(
                        ship, math.pi / 2, engines_on=True)
                except (AttributeError, TypeError, ValueError):
                    pass
            status["own"] = own
        except AttributeError:
            pass

        # Friendlies: where, doing what, at whom, with how much PD - formation
        # defense is real only if the captain can see who covers whom.
        pd_range_km = None
        if self.config.fleet_data:
            pd_range_km = pd_doctrine_numbers(self.config.fleet_data)["range_km"]
        umbrella = []
        for f in all_friendlies:
            if f.ship_id == ship.ship_id:
                continue  # Skip self
            status["friendlies"].append(self._friendly_info(ship, f, simulation, all_enemies))
            info = status["friendlies"][-1]
            if (pd_range_km is not None and info.get("pd_operational")
                    and info["distance_km"] <= pd_range_km):
                umbrella.append((info["name"], info["pd_operational"]))
        own_pd_list = getattr(ship, 'point_defense', None)
        own_pd = [pd for pd in (own_pd_list if isinstance(own_pd_list, (list, tuple)) else [])
                  if getattr(pd, 'is_operational', True)]
        if pd_range_km is not None:
            status["pd_umbrella"] = {"own": len(own_pd), "allies": umbrella,
                                     "range_km": pd_range_km}

        # Legacy fields for backward compatibility (from primary target or first enemy)
        primary_enemy = None
        if self.primary_target_id:
            for e in all_enemies:
                if e.ship_id == self.primary_target_id:
                    primary_enemy = e
                    break
        if not primary_enemy and all_enemies:
            primary_enemy = all_enemies[0]

        if primary_enemy:
            rel_pos = primary_enemy.position - ship.position
            distance_m = rel_pos.magnitude
            distance_km = distance_m / 1000
            status["distance_km"] = distance_km
            status["relative_position"] = {
                "x": rel_pos.x / 1000,
                "y": rel_pos.y / 1000,
                "z": rel_pos.z / 1000,
            }
            rel_vel = primary_enemy.velocity - ship.velocity
            status["relative_velocity"] = {
                "x": rel_vel.x / 1000,
                "y": rel_vel.y / 1000,
                "z": rel_vel.z / 1000,
            }
            if distance_m > 0:
                status["closing_rate"] = -rel_pos.normalized().dot(rel_vel) / 1000
                direction_to_enemy = rel_pos.normalized()
                dot = ship.forward.dot(direction_to_enemy)
                dot = max(-1.0, min(1.0, dot))
                status["angle_to_enemy_deg"] = math.degrees(math.acos(dot))
            else:
                status["closing_rate"] = 0
                status["angle_to_enemy_deg"] = 0

            if distance_km <= 500:
                base_hit = max(0.05, 0.9 - (distance_km / 500) * 0.85)
            else:
                base_hit = 0.05
            status["our_hit_chance"] = base_hit * 100

            status["enemy_shots"] = primary_enemy.shots_fired
            status["enemy_hits"] = primary_enemy.hits_scored
            status["enemy_hull_percent"] = primary_enemy.hull_integrity

            if primary_enemy.armor:
                enemy_nose = primary_enemy.armor.get_section("nose")
                enemy_lateral = primary_enemy.armor.get_section("lateral")
                enemy_tail = primary_enemy.armor.get_section("tail")
                status["enemy_armor"] = {
                    "nose_damage_pct": enemy_nose.damage_percent if enemy_nose else 0,
                    "lateral_damage_pct": enemy_lateral.damage_percent if enemy_lateral else 0,
                    "tail_damage_pct": enemy_tail.damage_percent if enemy_tail else 0,
                }
            else:
                status["enemy_armor"] = {}
        else:
            # No enemies - set defaults
            status["distance_km"] = 1000
            status["closing_rate"] = 0
            status["relative_position"] = {"x": 0, "y": 0, "z": 0}
            status["relative_velocity"] = {"x": 0, "y": 0, "z": 0}
            status["angle_to_enemy_deg"] = 0
            status["our_hit_chance"] = 0
            status["enemy_shots"] = 0
            status["enemy_hits"] = 0
            status["enemy_hull_percent"] = 100
            status["enemy_armor"] = {}

        # Build incoming projectiles with source and bearing
        incoming_projectiles = []
        if hasattr(simulation, 'projectiles') and simulation.projectiles:
            for proj_flight in simulation.projectiles:
                # simulation.projectiles holds ProjectileInFlight wrappers, whose
                # target field is `target_ship_id` and whose kinematics live on
                # `.projectile`. Reading `.target_id` / `.position` off the wrapper
                # silently matched nothing, so INCOMING FIRE was empty in every
                # battle. Accept either shape so test doubles keep working.
                target_id = getattr(proj_flight, 'target_ship_id', None)
                if target_id is None:
                    target_id = getattr(proj_flight, 'target_id', None)
                if target_id != ship.ship_id:
                    continue

                proj = getattr(proj_flight, 'projectile', proj_flight)
                proj_pos = getattr(proj, 'position', None) or Vector3D(0, 0, 0)
                proj_vel = getattr(proj, 'velocity', None) or Vector3D(0, 0, 0)

                offset = ship.position - proj_pos
                dist_to_ship = offset.magnitude
                dist_km = dist_to_ship / 1000

                if dist_to_ship > 0:
                    to_ship = offset.normalized()
                    # The target is moving too - what matters is the CLOSING rate
                    # of the projectile relative to this ship, not the projectile's
                    # speed in the world frame. Ignoring own velocity made the ETA
                    # wrong by up to a factor of two in either direction.
                    approach_speed = (proj_vel - ship.velocity).dot(to_ship)
                else:
                    to_ship = Vector3D(1, 0, 0)
                    approach_speed = 0.0

                eta_s = dist_to_ship / approach_speed if approach_speed > 0 else 999

                weapon_type, energy_gj = self._identify_projectile(proj)

                # Get source ship name
                source_name = "Unknown"
                source_id = getattr(proj_flight, 'source_ship_id', None)
                if source_id:
                    source_ship = simulation.get_ship(source_id)
                    if source_ship:
                        source_name = getattr(source_ship, 'name', source_id)

                # Calculate bearing
                bearing = self._calculate_impact_bearing(ship, proj_pos, proj_vel)

                incoming_projectiles.append({
                    "weapon_type": weapon_type,
                    "energy_gj": energy_gj,
                    "source": source_name,
                    "distance_km": dist_km,
                    "eta_seconds": eta_s,
                    "closing_kps": approach_speed / 1000,
                    "bearing": bearing,
                })

        # Sort by ETA
        incoming_projectiles.sort(key=lambda p: p["eta_seconds"])
        status["incoming_projectiles"] = incoming_projectiles[:5]  # Limit to 5

        # Check for incoming torpedoes. GUIDED threats get the engine's own
        # threat evaluation (NEZ, projected impact energy, terminal flag) via
        # _gather_guided_torpedo_threats (read-only: latch_commit=False), plus
        # a PD triage: how many turrets a seeker kill needs before impact vs
        # how many this ship has. BLINDED torpedoes coasting at us are listed
        # too - they are ballistic and still hit a non-maneuvering ship.
        from ..torpedo import MIN_CLOSING_SPEED_KPS

        torpedo_threats = []
        _pd_list = getattr(ship, 'point_defense', None)
        if not isinstance(_pd_list, (list, tuple)):
            _pd_list = []
        own_pd = [pd for pd in _pd_list if getattr(pd, 'is_operational', True)]
        rep_laser = own_pd[0].laser if own_pd else None

        def _pd_triage(torp_flight, dist_km, closing_ms):
            """
            Turrets needed to seeker-kill this torpedo before impact.

            The closed form assumes a CONSTANT closing speed, but a torpedo that
            still has delta-v is accelerating: 30 s after launch it is only doing
            ~4 km/s and will cross the PD envelope at the guidance closure floor
            of MIN_CLOSING_SPEED_KPS. Feeding it the instantaneous speed made the
            verdict optimistic exactly at the checkpoint where the captain
            decides - measured over 31 checkpoints of live engagements, the
            instantaneous form was right 24/31 times and EVERY error was a false
            "your PD can blind it" at the first checkpoint after launch. Flooring
            a fuelled round at the guidance floor scores 30/31, and its one error
            is conservative.
            """
            if rep_laser is None or closing_ms <= 0:
                return None
            try:
                torp = getattr(torp_flight, 'torpedo', None)
                fuelled = not getattr(torp, 'fuel_exhausted', True)
                effective_ms = (max(closing_ms, MIN_CLOSING_SPEED_KPS * 1000.0)
                                if fuelled else closing_ms)
                e = rep_laser.energy_before_impact_j(dist_km, effective_ms)
                heat_to_kill = max(
                    0.0,
                    torp_flight.ELECTRONICS_THRESHOLD_J
                    - torp_flight.heat_absorbed_j,
                )
                if e <= 0:
                    return 99
                return max(1, math.ceil(heat_to_kill / e))
            except (AttributeError, TypeError):
                return None

        gathered = None
        if hasattr(simulation, '_gather_guided_torpedo_threats'):
            try:
                gathered = simulation._gather_guided_torpedo_threats(
                    ship, latch_commit=False)
            except (TypeError, AttributeError):
                gathered = None

        if gathered is not None:
            for t in gathered:
                tf = t['torp_flight']
                closing_ms = (t['distance_m'] / t['t_go']
                              if t['t_go'] not in (0, float('inf')) else 0.0)
                source_id = getattr(tf, 'source_ship_id', 'Unknown')
                source_ship = simulation.get_ship(source_id) if source_id else None
                source_name = (ship_ref(getattr(source_ship, 'name', source_id), source_id)
                               if source_ship else source_id)
                torpedo_threats.append({
                    "distance_km": t['distance_m'] / 1000,
                    "closing_kps": closing_ms / 1000,
                    "eta_seconds": t['t_go'] if t['t_go'] != float('inf') else 999,
                    "source": source_name,
                    "est_impact_gj": t['ke_gj'],
                    "nez_inside": t['nez'].inside,
                    "terminal": t['terminal'],
                    "blinded": False,
                    "pd_turrets_needed": _pd_triage(
                        tf, t['distance_m'] / 1000, closing_ms),
                    "own_pd_turrets": len(own_pd),
                    # Allied turrets whose envelope the round's path to us
                    # crosses: their dwell stacks with ours (CP-13).
                    "allied_pd_turrets": self._allied_pd_on_path(
                        ship, tf.torpedo.position, all_friendlies, pd_range_km),
                })

        # Blinded wrecks: only those on a COLLISION course with this ship, from
        # whichever side - the engine's own closest-approach filter, the same
        # one behind the threat count, so the count and the list agree.
        if hasattr(simulation, 'blinded_wreck_threats'):
            for w in simulation.blinded_wreck_threats(ship):
                tf = w['torp_flight']
                source_id = getattr(tf, 'source_ship_id', None)
                source_ship = simulation.get_ship(source_id) if source_id else None
                torpedo_threats.append({
                    "distance_km": w['distance_m'] / 1000,
                    "closing_kps": w['closing_ms'] / 1000,
                    "eta_seconds": w['tca_s'],
                    "source": (ship_ref(getattr(source_ship, 'name', source_id), source_id)
                               if source_ship else source_id),
                    "blinded": True,
                    "own_side": bool(source_ship is not None
                                     and source_ship.faction == ship.faction),
                    "miss_km": w['miss_m'] / 1000,
                    "est_impact_gj": w['ke_gj'],
                })
        elif hasattr(simulation, 'torpedoes') and simulation.torpedoes:
            for torp_flight in simulation.torpedoes:
                torp = torp_flight.torpedo
                is_blinded = getattr(torp_flight, 'is_disabled', False)
                if gathered is not None and not is_blinded:
                    continue  # live threats already covered above
                if not is_blinded and torp.target_id != ship.ship_id:
                    continue
                offset = ship.position - torp.position
                dist_m = offset.magnitude
                if dist_m > 0:
                    closing_ms = (torp.velocity - ship.velocity).dot(offset.normalized())
                else:
                    closing_ms = 0.0
                if is_blinded and (closing_ms <= 0 or dist_m > 500_000):
                    continue  # a receding or distant wreck is not a threat
                eta_s = dist_m / closing_ms if closing_ms > 0 else 999

                source_id = getattr(torp_flight, 'source_ship_id', None) or getattr(
                    torp, 'source_ship_id', 'Unknown')
                source_ship = simulation.get_ship(source_id) if source_id else None
                source_name = getattr(source_ship, 'name', source_id) if source_ship else source_id

                threat = {
                    "distance_km": dist_m / 1000,
                    "closing_kps": closing_ms / 1000,
                    "eta_seconds": eta_s,
                    "source": source_name,
                    "blinded": is_blinded,
                }
                if is_blinded:
                    specs = getattr(torp, 'specs', None)
                    if specs is not None and closing_ms > 0:
                        threat["est_impact_gj"] = (
                            0.5 * specs.penetrator_mass_kg * closing_ms ** 2 / 1e9
                        )
                torpedo_threats.append(threat)
        torpedo_threats.sort(key=lambda t: t["eta_seconds"])
        status["torpedo_threats"] = torpedo_threats

        # Add current configuration to status
        current_maneuver_info = None
        if ship.current_maneuver:
            maneuver = ship.current_maneuver
            maneuver_type = maneuver.maneuver_type.name if hasattr(maneuver, 'maneuver_type') else "UNKNOWN"
            throttle = maneuver.throttle if hasattr(maneuver, 'throttle') else 1.0
            current_maneuver_info = {
                "type": maneuver_type,
                "throttle": throttle,
                # When it was issued and how long it runs: a maneuver that
                # ends at this checkpoint must be renewed or the ship coasts.
                "start_time": getattr(maneuver, 'start_time', None),
                "duration": getattr(maneuver, 'duration', None),
            }
            # Add heading direction if it's a heading maneuver
            if hasattr(maneuver, 'heading_direction') and maneuver.heading_direction:
                current_maneuver_info["heading"] = maneuver.heading_direction

        status["current_config"] = {
            "primary_target": self.primary_target_id,
            "weapon_orders": self.current_weapon_orders.copy(),
            "current_maneuver": current_maneuver_info,
        }

        # Always include threat assessment so LLM can decide whether to evade
        if hasattr(simulation, '_get_evasion_status'):
            evasion_status = simulation._get_evasion_status(ship)
            status["evasion_status"] = evasion_status

        # Fire solutions this hull can act on, and who can hit it next.
        self._add_torpedo_solutions(status, ship, all_enemies, simulation)
        self._add_threats_to_me(status, ship, all_enemies, simulation)
        return status

    @staticmethod
    def _allied_pd_on_path(ship: Any, torp_pos: Any, friendlies: List[Any],
                           pd_range_km: Optional[float]) -> int:
        """Allied live PD turrets within PD range of the segment torpedo -> ship."""
        if pd_range_km is None:
            return 0
        a, b = torp_pos, ship.position
        ab = b - a
        L2 = ab.dot(ab)
        total = 0
        for f in friendlies:
            if f.ship_id == ship.ship_id:
                continue
            n = sum(1 for pd in (getattr(f, 'point_defense', None) or [])
                    if getattr(pd, 'is_operational', True))
            if not n:
                continue
            t = 0.0 if L2 <= 0 else max(0.0, min(1.0, (f.position - a).dot(ab) / L2))
            if (f.position - (a + ab * t)).magnitude <= pd_range_km * 1000.0:
                total += n
        return total

    def _friendly_info(self, ship: Any, f: Any, simulation: Any,
                       enemies: List[Any]) -> Dict[str, Any]:
        """One friendly as the FRIENDLIES block shows it."""
        rel_pos = f.position - ship.position
        distance_m = rel_pos.magnitude
        info: Dict[str, Any] = {
            "ship_id": f.ship_id,
            "name": getattr(f, 'name', f.ship_id),
            "ship_type": getattr(f, 'ship_type', None),
            "distance_km": distance_m / 1000,
            "hull_percent": f.hull_integrity,
            "relative_position": {"x": rel_pos.x / 1000, "y": rel_pos.y / 1000,
                                  "z": rel_pos.z / 1000},
        }
        if distance_m > 0:
            u = rel_pos.normalized()
            info["dir_to"] = {"x": u.x, "y": u.y, "z": u.z}
        try:
            info["rel_speed_kps"] = (f.velocity - ship.velocity).magnitude / 1000
        except AttributeError:
            pass
        m = getattr(f, 'current_maneuver', None)
        if m is not None and hasattr(m, 'maneuver_type'):
            info["maneuver"] = {"type": m.maneuver_type.name,
                                "throttle": getattr(m, 'throttle', 1.0),
                                "heading": getattr(m, 'heading_direction', None)}
        elif hasattr(f, 'current_maneuver'):
            info["maneuver"] = None
        tid = getattr(f, 'primary_target_id', None)
        if tid:
            tgt = simulation.get_ship(tid) if hasattr(simulation, 'get_ship') else None
            info["target_id"] = tid
            info["target_name"] = getattr(tgt, 'name', tid) if tgt else tid
        pd_list = getattr(f, 'point_defense', None)
        if isinstance(pd_list, (list, tuple)) and pd_list:
            info["pd_total"] = len(pd_list)
            info["pd_operational"] = sum(1 for pd in pd_list if getattr(pd, 'is_operational', True))
        if _has_launchers(f):
            remaining = getattr(f, 'torpedoes_remaining', None)
            if isinstance(remaining, (int, float)):
                info["torpedoes_remaining"] = int(remaining)
            in_flight = []
            for tf in getattr(simulation, 'torpedoes', None) or []:
                if tf.source_ship_id != f.ship_id or getattr(tf, 'is_disabled', False):
                    continue
                tgt = simulation.get_ship(tf.torpedo.target_id)
                if tgt is None:
                    continue
                off = tgt.position - tf.torpedo.position
                closing = -(off.normalized()).dot(tgt.velocity - tf.torpedo.velocity) \
                    if off.magnitude > 0 else 0.0
                in_flight.append(off.magnitude / closing if closing > 1.0 else None)
            if in_flight:
                info["rounds_in_flight"] = len(in_flight)
                etas = [e for e in in_flight if e is not None]
                if etas:
                    info["first_impact_s"] = min(etas)
        return info

    def _add_torpedo_solutions(self, status: Dict[str, Any], ship: Any,
                               enemies: List[Any], simulation: Any) -> None:
        """
        Flown torpedo solutions for the nearest contacts (torpedo hulls only):
        TOF, impact energy, PD cover, friendly rounds inbound and a verdict.
        """
        if not self.config.has_torpedoes or not self.config.fleet_data or not enemies:
            return
        remaining = getattr(ship, "torpedoes_remaining", None)
        if isinstance(remaining, (int, float)) and remaining <= 0:
            return  # magazine empty: no launch to plan
        infos = {e["ship_id"]: e for e in status.get("enemies", [])}
        nearest = sorted(enemies, key=lambda e: (e.position - ship.position).magnitude)
        for e in nearest[:MAX_TORPEDO_SOLUTIONS]:
            sol = self._torpedo_solution(simulation, ship, e)
            if sol is not None and e.ship_id in infos:
                infos[e.ship_id]["torpedo_solution"] = sol.summary()
                infos[e.ship_id]["torpedo_verdict"] = sol.verdict

    def _add_threats_to_me(self, status: Dict[str, Any], ship: Any,
                           enemies: List[Any], simulation: Any) -> None:
        """
        Enemy torpedo hulls that have this ship targeted (and when a launch
        made now would land: a flown shadow round), and enemy guns on it.
        """
        targeting = set(self.targeting_me or [])
        fleet = self.config.fleet_data
        hulls, guns = [], []
        from .fire_solutions import fly_torpedo
        evading = bool(ship.current_maneuver is not None
                       and getattr(ship.current_maneuver.maneuver_type, 'name', '') == 'EVASIVE')
        window_s = getattr(simulation, "decision_interval", 30.0)
        for e in sorted(enemies, key=lambda x: (x.position - ship.position).magnitude):
            if e.ship_id not in targeting:
                continue
            rng_km = (e.position - ship.position).magnitude / 1000
            remaining = getattr(e, 'torpedoes_remaining', 0)
            remaining = remaining if isinstance(remaining, (int, float)) else 0
            launchers = getattr(e, 'ready_torpedo_launchers', None)
            launchers = launchers if isinstance(launchers, (list, tuple)) else []
            if launchers and remaining > 0 and len(hulls) < MAX_THREAT_HULL_SOLUTIONS:
                key = ("threat", round(float(simulation.current_time), 3), e.ship_id, ship.ship_id)
                shot = self._solution_cache.get(key)
                if key not in self._solution_cache and fleet:
                    try:
                        shot = fly_torpedo(simulation, e, ship, fleet, target_evades=evading)
                    except Exception:  # noqa: BLE001 - advisory only
                        shot = None
                    self._solution_cache[key] = shot
                hulls.append({"ship_id": e.ship_id, "name": getattr(e, 'name', e.ship_id),
                              "tubes": len(launchers), "rounds": int(remaining),
                              "range_km": rng_km,
                              "launch_tof_s": shot.tof_s if shot is not None and shot.hits else None})
            best, shots = 0.0, 0
            to_me = ship.position - e.position
            if to_me.magnitude <= 0:
                continue
            e_weapons = getattr(e, 'weapons', None)
            for ws in (e_weapons.values() if isinstance(e_weapons, dict) else ()):
                w = getattr(ws, 'weapon', None)
                if (w is None or w.is_missile or w.muzzle_velocity_kps <= 0
                        or not getattr(ws, 'is_operational', True)):
                    continue
                try:
                    # His gate's own number against us - our dodge included.
                    sol = simulation.gun_firing_solution(e, ship, ws)
                except (AttributeError, TypeError):
                    continue
                if not sol.can_fire or not ws.is_target_in_arc(e.forward, to_me.normalized()):
                    continue
                best = max(best, sol.hit_probability)
                shots += ws.shots_in_window(window_s)
            if shots:
                guns.append({"ship_id": e.ship_id, "name": getattr(e, 'name', e.ship_id),
                             "range_km": rng_km, "p_hit": best, "shots": shots})
        if hulls:
            status["torpedo_hulls_on_me"] = hulls
        if guns:
            status["guns_on_me"] = guns

    @tool_evidence
    def _execute_tool(
        self,
        tool_call: ToolCall,
        simulation: Any,
        ship_id: str,
    ) -> Optional[Any]:
        """
        Execute a tool call and return the resulting command.

        Args:
            tool_call: Tool call from LLM
            simulation: Combat simulation
            ship_id: This ship's ID

        Returns:
            Command object or None
        """
        name = tool_call.name
        args = tool_call.arguments

        if name == "set_maneuver":
            from ..simulation import Maneuver, ManeuverType
            from .maneuver_orders import (
                ManeuverArgError, maneuver_extras, resolve_ship, ship_choices)
            try:
                # Map LLM-friendly names to actual enum values
                maneuver_name = args["maneuver_type"]
                if maneuver_name == "EVADE":
                    maneuver_name = "EVASIVE"  # LLM uses EVADE, enum uses EVASIVE
                maneuver_type = ManeuverType[maneuver_name]
                throttle = args.get("throttle", 1.0)
            except (KeyError, ValueError) as e:
                self._record_tool_error(
                    f"set_maneuver: invalid maneuver_type {args.get('maneuver_type')!r} ({e}). "
                    f"Valid values: {', '.join(m.name for m in ManeuverType)}"
                )
                return None

            # INTERCEPT / PADLOCK / HOLD_RANGE fly relative to a ship: the
            # one named in target_id (one call instead of set_primary_target
            # first - and it leaves the gun target alone), else the primary
            # target, else the first enemy.
            target_id = None
            if maneuver_type in (ManeuverType.INTERCEPT, ManeuverType.PADLOCK,
                                 ManeuverType.HOLD_RANGE):
                ref = args.get("target_id")
                if ref is not None and str(ref).strip():
                    target = resolve_ship(simulation, ship_id, ref)
                    if target is None:
                        self._record_tool_error(
                            f"set_maneuver: no live ship named {ref!r} for target_id. "
                            f"Ships: {ship_choices(simulation, ship_id)}")
                        return None
                    target_id = target.ship_id
                elif self.primary_target_id:
                    target_id = self.primary_target_id
                else:
                    enemies = simulation.get_enemy_ships(ship_id)
                    target_id = enemies[0].ship_id if enemies else None

            # Optional parameters (standoff, arrival speed, hold range, evade):
            # none given -> exactly the maneuver this tool always built.
            extras: Dict[str, Any] = {}
            if maneuver_type == ManeuverType.HOLD_RANGE or any(
                    args.get(k) not in (None, "") for k in
                    ("standoff_km", "arrival_closing_kps", "range_km", "evade")):
                try:
                    extras, notes = maneuver_extras(simulation, ship_id, maneuver_type,
                                                    args, target_id)
                except ManeuverArgError as e:
                    self._record_tool_error(f"set_maneuver: {e}")
                    return None
                for note in notes:
                    self._record_tool_note(f"set_maneuver {maneuver_type.name}: {note}")

            return Maneuver(
                maneuver_type=maneuver_type,
                target_id=target_id,
                start_time=simulation.current_time,
                duration=getattr(simulation, "decision_interval", 30.0),
                throttle=throttle,
                **extras,
            )

        elif name == "set_primary_target":
            # Set the primary target for this captain: by id, exact name, bare
            # name or a UNIQUE partial match (names repeat across fleets, so
            # "first substring wins" could silently pick the wrong ship).
            target_ref = args.get("target_id") or args.get("target_name", "")
            enemies = simulation.get_enemy_ships(ship_id)
            # The captain-side field alone is not enough: the Admiral's per-ship
            # snapshot (current_target / targeted_by) reads ship.primary_target_id,
            # which is only written by a 'set_target' command. Returning the command
            # as well as setting the local field keeps the two in sync.
            enemy, candidates = self._resolve_ship(target_ref, enemies)
            if enemy is not None:
                self.primary_target_id = enemy.ship_id
                return {"type": "set_target", "target_id": enemy.ship_id}
            if candidates:
                self._record_tool_error(
                    f"set_primary_target: '{target_ref}' is ambiguous - "
                    + ", ".join(ship_ref(getattr(c, 'name', c.ship_id), c.ship_id) for c in candidates)
                    + ". Pass target_id."
                )
                return None
            valid = ", ".join(ship_ref(getattr(e, 'name', e.ship_id), e.ship_id)
                              for e in enemies) or "(none)"
            self._record_tool_error(
                f"set_primary_target: no enemy named '{target_ref}'. Valid targets: {valid}"
            )
            return None

        elif name == "set_heading":
            # Set a course in a specific 3D direction - or toward a position /
            # a ship's current bearing (toward_point / toward_ship).
            from ..simulation import Maneuver, ManeuverType
            from .maneuver_orders import (
                ManeuverArgError, heading_target, parse_evade, position_like_direction_note,
                usable_direction)
            throttle = args.get("throttle", 1.0)
            extras: Dict[str, Any] = {}
            try:
                live, notes = heading_target(simulation, ship_id, args)
                if args.get("evade") not in (None, ""):
                    evade = parse_evade(args.get("evade"))
                    if evade is not None:
                        extras["evade"] = evade
            except ManeuverArgError as e:
                self._record_tool_error(f"set_heading: {e}")
                return None
            for note in notes:
                self._record_tool_note(f"set_heading: {note}")
            if live is not None:
                extras.update(live)
                direction = extras.pop("heading_direction", None)
            else:
                direction = args.get("direction")
                if not usable_direction(direction):
                    # No silent default: a call without a usable direction
                    # used to burn along world +X.
                    self._record_tool_error(
                        f"set_heading: needs direction {{x, y, z}} (a world-frame direction, "
                        f"not all zero), toward_point {{x, y, z}} km or toward_ship - got "
                        f"direction={direction!r}")
                    return None
                # A position pasted from the orders is still flown as the
                # direction it literally is - but say so (CP-03).
                if isinstance(direction, dict) and (self.admiral_orders or self.fleet_directive):
                    texts = [getattr(o, "order_text", "") or "" for o in self.admiral_orders or []]
                    texts.append(self.fleet_directive or "")
                    note = position_like_direction_note(
                        direction, simulation.get_ship(ship_id), texts)
                    if note:
                        self._record_tool_note(note)

            return Maneuver(
                maneuver_type=ManeuverType.HEADING,
                target_id=None,
                start_time=simulation.current_time,
                duration=getattr(simulation, "decision_interval", 30.0),
                throttle=throttle,
                heading_direction=direction,
                **extras,
            )

        elif name == "set_weapons_order":
            from ..firecontrol import WeaponsCommand, WeaponsOrder

            # Persistent gun orders follow the ship's live primary target.
            # Capturing an id here made later target changes ineffective.
            target_id = None

            orders = []

            # The tool schema advertises "spinal_mode" / "turret_mode" (tools.py),
            # but weapon groups are named spinal / coilguns / heavy_coilguns. Without
            # this aliasing every turret order the model issues matches no group and
            # is silently dropped.
            def _arg_for(group: str, suffix: str):
                direct = args.get(f"{group}_{suffix}")
                if direct is not None:
                    return direct
                alias = "spinal" if group == "spinal" else "turret"
                return args.get(f"{alias}_{suffix}")

            # Process each weapon group dynamically
            for group_name, slots in self.weapon_groups.items():
                mode = _arg_for(group_name, "mode")
                if mode:
                    try:
                        command = WeaponsCommand[mode]
                    except KeyError:
                        command = WeaponsCommand.FIRE_WHEN_OPTIMAL

                    min_prob = _arg_for(group_name, "min_probability")
                    min_prob = 0.3 if min_prob is None else min_prob
                    max_range = _arg_for(group_name, "max_range_km")
                    max_range = 500.0 if max_range is None else max_range

                    # Create order for each weapon in this group
                    for slot in slots:
                        orders.append(WeaponsOrder(
                            command=command,
                            weapon_slot=slot,
                            target_id=target_id,
                            min_hit_probability=min_prob,
                            max_range_km=max_range,
                        ))

                    # Track current weapon orders (use group name for display)
                    self.current_weapon_orders[group_name] = mode

            # If no specific modes set, default all groups to FIRE_WHEN_OPTIMAL
            if not orders and self.weapon_groups:
                for group_name, slots in self.weapon_groups.items():
                    for slot in slots:
                        orders.append(WeaponsOrder(
                            command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
                            weapon_slot=slot,
                            target_id=target_id,
                            min_hit_probability=0.3,
                        ))
                    self.current_weapon_orders[group_name] = "FIRE_WHEN_OPTIMAL"

            return {
                "type": "weapons_orders",
                "orders": orders,
                "default_target_id": self.primary_target_id or next(
                    (e.ship_id for e in simulation.get_enemy_ships(ship_id)), None),
            }

        elif name == "launch_torpedo":
            enemies = simulation.get_enemy_ships(ship_id)
            if not enemies:
                self._record_tool_error("launch_torpedo: no enemy ships to target")
                return None
            # Honour the designated primary target. Firing at enemies[0] meant a
            # torpedo - the scarcest munition on the ship - could be spent on a
            # ship the captain had explicitly declined to engage.
            target_id = None
            explicit = args.get("target_name")
            if explicit:
                enemy, candidates = self._resolve_ship(explicit, enemies)
                if enemy is not None:
                    target_id = enemy.ship_id
                else:
                    which = (" is ambiguous (" + ", ".join(
                        ship_ref(getattr(c, 'name', c.ship_id), c.ship_id) for c in candidates)
                        + ")") if candidates else " matches no enemy"
                    self._record_tool_error(
                        f"launch_torpedo: '{explicit}'{which}; using primary target"
                    )
            if target_id is None and self.primary_target_id:
                if any(e.ship_id == self.primary_target_id for e in enemies):
                    target_id = self.primary_target_id
            # Explicit target_id from the schema wins over the legacy target_name.
            requested_id = args.get("target_id")
            if requested_id:
                if any(e.ship_id == requested_id for e in enemies):
                    target_id = requested_id
                else:
                    valid = ", ".join(e.ship_id for e in enemies)
                    self._record_tool_error(
                        f"launch_torpedo: unknown target_id '{requested_id}'. Valid: {valid}"
                    )
            if target_id is None and self.primary_target_id:
                if any(e.ship_id == self.primary_target_id for e in enemies):
                    target_id = self.primary_target_id
            if target_id is None:
                # Nearest contact (the enemy list's order is arbitrary).
                try:
                    me = simulation.get_ship(ship_id)
                    target_id = min(enemies, key=lambda e: (e.position - me.position).magnitude).ship_id
                except (AttributeError, TypeError):
                    target_id = enemies[0].ship_id

            # Reserve only rounds that can leave a tube before the next decision.
            try:
                count = int(args.get("count", 1))
            except (TypeError, ValueError):
                count = 1
            max_salvo = self._max_torpedo_salvo(ship_id, simulation)
            if count > max_salvo:
                self._record_tool_error(
                    f"launch_torpedo: requested {count}, launcher can fire {max_salvo} "
                    f"this decision (12s reload, magazine limits)"
                )
            count = max(1, min(count, max_salvo)) if max_salvo > 0 else 0
            if count == 0:
                self._record_tool_error(
                    "launch_torpedo: magazine empty or launcher still reloading"
                )
                return None

            # Feasibility: fly the round on the engine shadow first. A launch
            # that misses even if the target holds course while it OPENS range
            # is refused (19 of 49 rounds in one battle went at receding
            # targets for 0 GJ); every other doubtful verdict is accepted with
            # the warning as the tool's answer (shown next turn).
            if not self._torpedo_feasible(simulation, ship_id, target_id, count, args):
                return None

            return [
                {"type": "launch_torpedo", "target_id": target_id, "queue_if_reloading": True}
                for _ in range(count)
            ]

        elif name == "set_radiators":
            return radiator_command_from_args(args)

        elif name == "send_message":
            # Enemy/open channel. Queued in the outbox; the runner routes it
            # (faction-checked) after the decision. Several per decision are
            # kept - the old single slot silently dropped all but the last.
            text = clip(args.get("message"), SIGNAL_TEXT_CHARS)
            if not text:
                return None
            sent = sum(1 for m in self.outbox if m["tool"] == "send_message")
            if sent >= MAX_ENEMY_MESSAGES_PER_DECISION:
                self._record_tool_error(
                    f"send_message: only {MAX_ENEMY_MESSAGES_PER_DECISION} transmissions per "
                    "decision - this one was not sent")
                return None
            recipient = str(args.get("recipient") or "ALL_ENEMIES").strip().upper()
            target_ship = args.get("target_ship")
            self.outbox.append({
                "tool": "send_message",
                "to": target_ship if recipient == "SPECIFIC" and target_ship else recipient,
                "recipient": recipient,
                "target_ship": target_ship,
                "kind": "FREE",
                "text": text,
            })
            return None

        elif name == "fleet_signal":
            signal, errors = normalize_signal(args)
            for error in errors:
                self._record_tool_error(error)
            if signal is None:
                return None
            sent = sum(1 for m in self.outbox if m["tool"] == "fleet_signal")
            if sent >= MAX_FLEET_SIGNALS_PER_DECISION:
                self._record_tool_error(
                    f"fleet_signal: only {MAX_FLEET_SIGNALS_PER_DECISION} signals per decision - "
                    f"this {signal['kind']} was not sent")
                return None
            self.outbox.append({"tool": "fleet_signal", **signal})
            return None

        elif name == "log_note":
            note = str(args.get("note", "")).strip()
            if not note:
                return None
            # Bound both dimensions: entry length and entry count. The tool
            # advertises ~250 chars; hard-cut a little above that so a verbose
            # model degrades gracefully instead of flooding its own prompt.
            if len(note) > self.MAX_LOG_NOTE_CHARS:
                note = note[:self.MAX_LOG_NOTE_CHARS] + "..."
            self.captain_log.append({
                "time": simulation.current_time,
                "note": note,
            })
            del self.captain_log[:-self.MAX_LOG_ENTRIES]
            return None

        elif name == "surrender":
            self.has_surrendered = True
            return None

        elif name == "propose_draw":
            self.has_proposed_draw = True
            self.has_retracted_draw = False  # Clear retraction if re-proposing
            return None

        elif name == "retract_draw":
            if self.has_proposed_draw:
                self.has_retracted_draw = True
                self.has_proposed_draw = False
            return None

        elif name == "discuss_with_admiral":
            # Captain wants to discuss with Admiral
            if not self.has_admiral:
                self._record_tool_error(
                    "discuss_with_admiral: you have no Admiral - you are an independent command."
                )
                return None

            if self.discussion_exchanges >= self.max_discussion_exchanges:
                self._record_tool_error(
                    f"discuss_with_admiral: you have used all "
                    f"{self.max_discussion_exchanges} discussion exchanges this checkpoint. "
                    "Issue tactical orders instead."
                )
                return {
                    "type": "discussion_limit_reached",
                    "message": f"You have already used your {self.max_discussion_exchanges} discussion exchanges with the Admiral this checkpoint."
                }

            question = args.get("question", "")
            if not question:
                return None

            self.discussion_exchanges += 1
            # Return a marker for battle_runner to handle
            # The battle_runner will call Admiral.respond_to_captain() and inject the response
            return {
                "type": "discuss_with_admiral",
                "question": question,
                "exchange_number": self.discussion_exchanges,
            }

        elif name in ("report_to_admiral", "respond_to_orders"):
            # Captain -> Admiral. respond_to_orders is the legacy alias (old
            # prompts, recorded tool calls); its args map onto the report.
            if not self.has_admiral:
                self._record_tool_error(
                    f"{name}: you have no Admiral - you are an independent command.")
                return None
            report, errors = normalize_report(args, legacy=(name == "respond_to_orders"))
            for error in errors:
                self._record_tool_error(error)
            report["tool"] = name
            # Legacy keys, for anything still reading the old response shape.
            report["type"] = "ACKNOWLEDGE" if report["status"] == "ACK" else report["status"]
            report["deviation_reason"] = report["reason"]
            report["acknowledgment_note"] = report["intent"]
            self.order_response = report
            text = report["reason"] or report["intent"] or "Orders received, executing."
            print(f"  [{report['status']}] {self.name}: {text}")
            return None  # No command: the runner files it for the Admiral

        else:
            self._record_tool_error(f"unknown tool {name!r} - it was ignored.")
            return None

    def get_pending_message(self) -> Optional[Dict[str, Any]]:
        """
        Pop the next outgoing enemy-channel message (1v1 duel path).

        Returns:
            Dict with 'content', 'recipient', 'target_ship' keys, or None
        """
        msg = self.pending_message
        self.pending_message = None
        if isinstance(msg, str):  # legacy string format
            return {"content": msg, "recipient": "ALL_ENEMIES", "target_ship": None}
        if msg:
            return msg
        for i, entry in enumerate(self.outbox):
            if entry.get("tool") == "send_message":
                del self.outbox[i]
                self._record_sent_message(entry["text"], self._last_decision_time())
                return {"content": entry["text"], "recipient": entry.get("recipient", "ALL_ENEMIES"),
                        "target_ship": entry.get("target_ship")}
        return None

    def _last_decision_time(self) -> float:
        return self.decision_history[-1]["time"] if self.decision_history else 0.0

    def note_sent(self, text: str, time: float, to: str) -> None:
        """
        Log a message the runner actually transmitted (COMMUNICATION LOG).

        `to` (recipient and signal kind, e.g. "ALL [TARGET_CLAIM]") leads the
        logged text: the log entry has no recipient field, and "I claimed
        Lancer to my fleet" must not read like a taunt sent to the enemy.
        """
        self._record_sent_message(f"-> {to}: {text}", time)

    def drain_outbox(self) -> List[Dict[str, Any]]:
        """
        Take every message composed this checkpoint (fleet battles).

        Entries: {tool: 'fleet_signal'|'send_message', to, kind, text, ...}.
        A legacy `pending_message` dict (set directly by older callers) is
        drained too, as a send_message.
        """
        out = list(self.outbox)
        self.outbox = []
        legacy = self.pending_message
        self.pending_message = None
        if isinstance(legacy, str):
            legacy = {"content": legacy, "recipient": "ALL_ENEMIES", "target_ship": None}
        if isinstance(legacy, dict) and legacy.get("content"):
            recipient = str(legacy.get("recipient") or "ALL_ENEMIES").upper()
            target = legacy.get("target_ship")
            out.append({"tool": "send_message", "kind": "FREE", "text": str(legacy["content"]),
                        "recipient": recipient, "target_ship": target,
                        "to": target if recipient == "SPECIFIC" and target else recipient})
        return out

    def take_comms_pass(self) -> Dict[str, Any]:
        """
        Detach this decision pass's outbox and report (discussion re-decide).

        The runner may re-run decide() in one checkpoint (after an Admiral
        discussion); the re-run executes the comms tools again. The runner
        detaches the superseded pass's comms here and hands them back to
        `restore_comms_pass`, which keeps whatever the re-run did not replace.
        """
        taken = {"outbox": self.outbox, "report": self.order_response}
        self.outbox = []
        self.order_response = None
        return taken

    def restore_comms_pass(self, earlier: Dict[str, Any]) -> None:
        """Fall back to an earlier pass's outbox/report where the re-run sent none."""
        if not self.outbox:
            self.outbox = list(earlier.get("outbox") or [])
        if self.order_response is None:
            self.order_response = earlier.get("report")

    def receive_inbox(self, items: List[Dict[str, Any]], delivered_t: float) -> None:
        """
        This checkpoint's fleet comms (replaces the previous checkpoint's).

        Data only: the captain prompt renders the inbox (FLEET NET / ENEMY
        TRANSMISSIONS) and copies it into the COMMUNICATION LOG history,
        de-duplicated by msg_id - logging it here as well would log every
        message twice.
        """
        self.inbox = list(items)
        self.inbox_delivered_t = delivered_t
