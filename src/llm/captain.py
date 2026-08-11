"""
LLM-powered captain for space combat simulation.

Makes strategic decisions via tool/function calling.
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .client import DEFAULT_MODEL, CaptainClient, LLMCallError, ToolCall
from .tools import get_captain_tools, get_captain_tools_for_ship, get_weapon_groups_for_ship, PERSONALITY_SELECTION_TOOLS, RESPOND_TO_ORDERS_TOOL
from .prompts import (
    build_captain_prompt,
    build_captain_messages,
    build_personality_selection_prompt,
    format_admiral_orders_for_captain,
    CaptainPersonality,
    PERSONALITY_PRESETS,
)
from .communication import CaptainMessage, MessageType
from .admiral_tools import DISCUSS_WITH_ADMIRAL_TOOL


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
        self.max_discussion_exchanges: int = 2
        self.order_response: Optional[Dict[str, Any]] = None  # Response to orders

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
        # Reset discussion counter for new checkpoint
        self.discussion_exchanges = 0

    def get_tools_for_context(self) -> List[Dict[str, Any]]:
        """
        Get tools appropriate for current context.

        If Admiral exists:
        - Remove propose_draw and retract_draw (only Admiral can)
        - Add discuss_with_admiral tool
        - Add respond_to_orders tool if orders were received
        """
        # Get base tools
        tools = self._build_tools()

        if self.has_admiral:
            # Remove draw tools - only Admiral can propose draws
            tools = [
                t for t in tools
                if t["function"]["name"] not in ("propose_draw", "retract_draw")
            ]
            # Add discuss_with_admiral tool
            tools.append(DISCUSS_WITH_ADMIRAL_TOOL)
            # Add respond_to_orders tool if we have orders
            if self.admiral_orders or self.fleet_directive:
                tools.append(RESPOND_TO_ORDERS_TOOL)

        return tools

    def clear_admiral_context(self) -> None:
        """Clear Admiral context at end of checkpoint."""
        self.admiral_orders = []
        self.fleet_directive = ""
        self.discussion_exchanges = 0
        self.order_response = None

    def decide(
        self,
        ship_id: str,
        simulation: Any,
    ) -> List[Any]:
        """
        Make strategic decisions using LLM tool calls.

        Args:
            ship_id: ID of this captain's ship
            simulation: CombatSimulation instance

        Returns:
            List of commands to execute (Maneuvers, fire orders, etc.)
        """
        if self.has_surrendered:
            return []

        # Get ship state
        ship = simulation.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return []

        # Set up weapon groups if not done yet
        if not self.weapon_groups and self.config.fleet_data and self.config.ship_type:
            self.setup_weapon_groups(self.config.ship_type, self.config.fleet_data)

        # Get enemy for tactical info
        enemies = simulation.get_enemy_ships(ship_id)
        enemy = enemies[0] if enemies else None

        # Build status dicts
        ship_status = self._build_ship_status(ship)
        tactical_status = self._build_tactical_status(ship, enemy, simulation)

        # Update battle tracking
        distance_km = tactical_status.get("distance_km", 1000)
        self._update_battle_tracking(distance_km)

        # Record received messages to history and format for prompt
        messages_text = ""
        if self.received_messages:
            for msg in self.received_messages:
                self._record_received_message(msg.content, msg.timestamp)
            messages_text = "\n".join(
                msg.format_for_llm() for msg in self.received_messages
            )
            self.received_messages.clear()

        # Roll last turn's rejected tool calls into this turn's feedback, then
        # start a fresh error bucket for the calls we are about to execute.
        self.last_tool_errors = self.pending_tool_errors
        self.pending_tool_errors = []

        # Build history context
        decision_history = self._format_decision_history(last_n=5)
        message_history = self._format_message_history(last_n=6)
        battle_summary = self._format_battle_summary(distance_km)
        shot_history = self._format_shot_history(last_n=10)

        # Build prompt
        recent_hits_text = self._format_recent_hits()
        prompt_kwargs = dict(
            captain_name=self.config.name,
            ship_name=self.config.ship_name,
            ship_status=ship_status,
            tactical_status=tactical_status,
            personality=self.config.personality,
            personality_text=self.personality_text,
            received_messages=messages_text if messages_text else None,
            decision_history=decision_history,
            message_history=message_history,
            battle_summary=battle_summary,
            shot_history=shot_history,
            recent_hits=recent_hits_text if recent_hits_text else None,
            ship_type=self.config.ship_type,
            fleet_data=self.config.fleet_data,
            notebook_text=self.config.notebook_text,
        )

        # Split into a stable system prompt and a volatile user turn so the
        # provider can serve the doctrine from cache across checkpoints.
        prompt_messages = build_captain_messages(**prompt_kwargs)

        turn_parts = []
        if len(prompt_messages) > 1:
            turn_parts.append(prompt_messages[1]["content"])

        # Torpedo threats / magazine. _build_tactical_status has always computed
        # `torpedo_threats`, but nothing ever rendered it, so a captain was never
        # told a torpedo was inbound, and the remaining torpedo count (the ship's
        # scarcest munition) was never disclosed either.
        torpedo_text = self._format_torpedo_section(ship_status, tactical_status)
        if torpedo_text:
            turn_parts.append(torpedo_text)

        log_text = self._format_captain_log()
        if log_text:
            turn_parts.append(log_text)

        # Admiral orders change per checkpoint - they belong in the volatile turn.
        if self.has_admiral and (self.admiral_orders or self.fleet_directive):
            turn_parts.append(format_admiral_orders_for_captain(
                self.admiral_orders,
                self.fleet_directive,
            ))

        errors_text = self._format_tool_errors()
        if errors_text:
            turn_parts.append(errors_text)

        turn_parts.append(
            f"DECISION POINT {self.decision_count + 1}. What are your orders, Captain?"
        )

        messages = [prompt_messages[0], {"role": "user", "content": "\n\n".join(turn_parts)}]

        # Get context-appropriate tools (may exclude draw tools if Admiral exists)
        tools = self.get_tools_for_context()

        # Call LLM with tools (use captain's configured model)
        try:
            # Per-captain temperature was parsed from fleet config and stored, but
            # never sent - every captain silently sampled at the shared client's
            # default regardless of its configured value.
            tool_calls = self.client.decide_with_tools(
                messages,
                tools,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        except LLMCallError as e:
            # Distinguish "call failed" from "captain chose to do nothing": do not
            # advance decision history, and mark the turn as degraded.
            print(f"[CAPTAIN {self.config.name}] decision call failed: {e}")
            self.last_call_failed = True
            return []
        self.last_call_failed = False

        # Execute tool calls
        # Track maneuver commands - only one maneuver per decision allowed
        maneuver_tools = {"set_maneuver", "set_heading"}
        maneuver_issued = False

        commands = []
        executed_tool_calls = []  # Track actually executed tool calls
        for tc in tool_calls:
            # Skip duplicate maneuver commands (only first one takes effect)
            if tc.name in maneuver_tools:
                if maneuver_issued:
                    continue  # Skip this maneuver, one already issued
                maneuver_issued = True

            cmd = self._execute_tool(tc, simulation, ship_id)
            executed_tool_calls.append(tc)
            if cmd is not None:
                # A tool may emit a salvo (e.g. launch_torpedo count=2).
                if isinstance(cmd, list):
                    commands.extend(c for c in cmd if c is not None)
                else:
                    commands.append(cmd)

        # Track decision
        self.decision_count += 1
        self.last_tool_calls = executed_tool_calls  # Store only executed calls for verbose output
        self.decision_history.append({
            "checkpoint": self.decision_count,
            "time": simulation.current_time,
            "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in tool_calls],
            "commands_count": len(commands),
        })

        return commands

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
                direction = tc.arguments.get("direction", {})
                throttle = tc.arguments.get("throttle", 1.0)
                actions.append(f"HEADING ({direction.get('x', 0):.1f},{direction.get('y', 0):.1f},{direction.get('z', 0):.1f}) @ {throttle*100:.0f}%")
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
                extend = tc.arguments.get("extend", False)
                actions.append("EXTEND radiators" if extend else "RETRACT radiators")
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

    def _format_decision_history(self, last_n: int = 5) -> str:
        """Format recent decision history for the prompt."""
        if not self.decision_history:
            return ""

        recent = self.decision_history[-last_n:]
        lines = ["YOUR RECENT DECISIONS:"]

        for decision in recent:
            checkpoint = decision["checkpoint"]
            time = decision["time"]
            actions = []

            for tc in decision["tool_calls"]:
                name = tc["name"]
                args = tc["args"]

                if name == "set_maneuver":
                    maneuver = args.get("maneuver_type", "?")
                    throttle = args.get("throttle", 1.0)
                    actions.append(f"{maneuver}@{throttle*100:.0f}%")
                elif name == "set_weapons_order":
                    spinal = args.get("spinal_mode", "")
                    turret = args.get("turret_mode", "")
                    if spinal and turret:
                        actions.append(f"weapons:{spinal}/{turret}")
                    elif spinal:
                        actions.append(f"spinal:{spinal}")
                    elif turret:
                        actions.append(f"turret:{turret}")
                elif name == "set_radiators":
                    actions.append("radiators:" + ("extend" if args.get("extend") else "retract"))
                elif name == "send_message":
                    actions.append("sent_msg")
                elif name == "log_note":
                    actions.append("logged_note")
                elif name == "surrender":
                    actions.append("SURRENDER")
                elif name == "propose_draw":
                    actions.append("PROPOSE_DRAW")
                elif name == "retract_draw":
                    actions.append("RETRACT_DRAW")

            action_str = ", ".join(actions) if actions else "none"
            lines.append(f"  T+{time:.0f}s: {action_str}")

        return "\n".join(lines)

    def _format_message_history(self, last_n: int = 6) -> str:
        """Format recent message history for the prompt."""
        if not self.message_history:
            return ""

        recent = self.message_history[-last_n:]
        lines = ["COMMUNICATION LOG:"]

        for msg in recent:
            sender = msg["sender"]
            time = msg["time"]
            text = msg["text"]
            # Truncate long messages
            if len(text) > 100:
                text = text[:100] + "..."
            who = "You" if sender == "self" else "Enemy"
            lines.append(f"  T+{time:.0f}s [{who}]: \"{text}\"")

        return "\n".join(lines)

    def _format_battle_summary(self, distance_km: float) -> str:
        """Format battle progression summary."""
        if self.initial_distance_km is None:
            return ""

        lines = ["BATTLE PROGRESSION:"]
        lines.append(f"  Started: {self.initial_distance_km:.0f} km apart")
        lines.append(f"  Closest approach: {self.min_distance_km:.0f} km")
        if self.passes_count > 0:
            lines.append(f"  Close passes (<100km): {self.passes_count}")
        lines.append(f"  Checkpoints elapsed: {self.decision_count}")

        return "\n".join(lines)

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
            )
        return get_captain_tools(has_torpedoes=self.config.has_torpedoes)

    def _max_torpedo_salvo(self, ship_id: str, simulation: Any) -> int:
        """
        How many torpedoes can physically be launched before the next decision.

        Bounded by the reload interval over the decision window and by rounds
        remaining in the magazine.
        """
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

    def _format_tool_errors(self) -> str:
        """Format last turn's rejected tool calls for the next prompt."""
        if not self.last_tool_errors:
            return ""
        lines = ["ERRORS FROM YOUR LAST ORDERS (these calls were REJECTED, reissue correctly):"]
        for err in self.last_tool_errors[-5:]:
            lines.append(f"  - {err}")
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

    def _record_received_message(self, message: str, time: float) -> None:
        """Record a message received from enemy."""
        self.message_history.append({
            "sender": "enemy",
            "time": time,
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
            # Check if radiators extended
            if ship.thermal_system.radiators:
                from ..thermal import RadiatorState
                extended = any(
                    rad.state == RadiatorState.EXTENDED
                    for rad in ship.thermal_system.radiators.radiators.values()
                )
                status["radiators_extended"] = extended
            else:
                status["radiators_extended"] = False
        else:
            status["heat_percent"] = 0
            status["heatsink_capacity"] = 525
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

        # Hit probability
        if distance_km <= 500:
            base_hit = max(0.05, 0.9 - (distance_km / 500) * 0.85)
        else:
            base_hit = 0.05
        info["hit_chance"] = base_hit * 100

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

        # Build friendly info list (simpler, less detail needed)
        for f in all_friendlies:
            if f.ship_id == ship.ship_id:
                continue  # Skip self
            rel_pos = f.position - ship.position
            distance_km = rel_pos.magnitude / 1000
            status["friendlies"].append({
                "ship_id": f.ship_id,
                "name": getattr(f, 'name', f.ship_id),
                "distance_km": distance_km,
                "hull_percent": f.hull_integrity,
                "relative_position": {
                    "x": rel_pos.x / 1000,
                    "y": rel_pos.y / 1000,
                    "z": rel_pos.z / 1000,
                },
            })

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
                source_name = (getattr(source_ship, 'name', source_id)
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
                })

        if hasattr(simulation, 'torpedoes') and simulation.torpedoes:
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

        return status

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
            try:
                # Map LLM-friendly names to actual enum values
                maneuver_name = args["maneuver_type"]
                if maneuver_name == "EVADE":
                    maneuver_name = "EVASIVE"  # LLM uses EVADE, enum uses EVASIVE
                maneuver_type = ManeuverType[maneuver_name]
                throttle = args.get("throttle", 1.0)

                # For INTERCEPT and PADLOCK, use primary target or first enemy
                target_id = None
                if maneuver_type in (ManeuverType.INTERCEPT, ManeuverType.PADLOCK):
                    if self.primary_target_id:
                        target_id = self.primary_target_id
                    else:
                        enemies = simulation.get_enemy_ships(ship_id)
                        target_id = enemies[0].ship_id if enemies else None

                return Maneuver(
                    maneuver_type=maneuver_type,
                    target_id=target_id,
                    start_time=simulation.current_time,
                    duration=30.0,
                    throttle=throttle,
                )
            except (KeyError, ValueError) as e:
                self._record_tool_error(
                    f"set_maneuver: invalid maneuver_type {args.get('maneuver_type')!r} ({e}). "
                    f"Valid values: {', '.join(m.name for m in ManeuverType)}"
                )
                return None

        elif name == "set_primary_target":
            # Set the primary target for this captain
            target_name = args.get("target_name", "")
            # Find enemy ship by name
            enemies = simulation.get_enemy_ships(ship_id)
            # The captain-side field alone is not enough: the Admiral's per-ship
            # snapshot (current_target / targeted_by) reads ship.primary_target_id,
            # which is only written by a 'set_target' command. Returning the command
            # as well as setting the local field keeps the two in sync.
            for enemy in enemies:
                enemy_name = getattr(enemy, 'name', enemy.ship_id)
                if enemy_name.lower() == target_name.lower() or enemy.ship_id == target_name:
                    self.primary_target_id = enemy.ship_id
                    return {"type": "set_target", "target_id": enemy.ship_id}
            # If not found, try partial match
            for enemy in enemies:
                enemy_name = getattr(enemy, 'name', enemy.ship_id)
                if target_name.lower() in enemy_name.lower():
                    self.primary_target_id = enemy.ship_id
                    return {"type": "set_target", "target_id": enemy.ship_id}
            valid = ", ".join(getattr(e, 'name', e.ship_id) for e in enemies) or "(none)"
            self._record_tool_error(
                f"set_primary_target: no enemy named '{target_name}'. Valid targets: {valid}"
            )
            return None

        elif name == "set_heading":
            # Set a course in a specific 3D direction
            from ..simulation import Maneuver, ManeuverType
            direction = args.get("direction", {"x": 1, "y": 0, "z": 0})
            throttle = args.get("throttle", 1.0)

            return Maneuver(
                maneuver_type=ManeuverType.HEADING,
                target_id=None,
                start_time=simulation.current_time,
                duration=30.0,
                throttle=throttle,
                heading_direction=direction,
            )

        elif name == "set_weapons_order":
            from ..firecontrol import WeaponsCommand, WeaponsOrder

            # Use primary target or first enemy
            if self.primary_target_id:
                target_id = self.primary_target_id
            else:
                enemies = simulation.get_enemy_ships(ship_id)
                target_id = enemies[0].ship_id if enemies else None

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
                for enemy in enemies:
                    enemy_name = getattr(enemy, 'name', enemy.ship_id)
                    if (enemy_name.lower() == explicit.lower()
                            or enemy.ship_id == explicit
                            or explicit.lower() in enemy_name.lower()):
                        target_id = enemy.ship_id
                        break
                if target_id is None:
                    self._record_tool_error(
                        f"launch_torpedo: no enemy named '{explicit}'; using primary target"
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
                target_id = enemies[0].ship_id

            # Salvo size. The launcher reloads every 12s and a decision covers
            # 30s, so 2 is the physical ceiling; the simulation still enforces
            # cooldown and magazine, this just stops the captain asking for more
            # than can possibly be fired.
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

            return [
                {"type": "launch_torpedo", "target_id": target_id}
                for _ in range(count)
            ]

        elif name == "set_radiators":
            extend = args.get("extend", False)
            return {
                "type": "set_radiators",
                "extend": extend,
            }

        elif name == "send_message":
            # Queue message for delivery and record in history
            message = args.get("message", "")
            recipient = args.get("recipient", "ALL_ENEMIES")
            target_ship = args.get("target_ship", None)

            # Store message with recipient info
            self.pending_message = {
                "content": message,
                "recipient": recipient,
                "target_ship": target_ship,
            }
            self._record_sent_message(message, simulation.current_time)
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

        elif name == "respond_to_orders":
            # Captain responding to Admiral orders
            response_type = args.get("response_type", "ACKNOWLEDGE")
            deviation_reason = args.get("deviation_reason", "")
            acknowledgment_note = args.get("acknowledgment_note", "")

            # Store the response for logging/display
            self.order_response = {
                "type": response_type,
                "deviation_reason": deviation_reason,
                "acknowledgment_note": acknowledgment_note,
            }

            if response_type == "DEVIATE":
                print(f"  [DEVIATION] {self.name}:")
                for line in deviation_reason.split('\n'):
                    print(f"    {line}")
            elif acknowledgment_note:
                print(f"  [ACKNOWLEDGE] {self.name}:")
                for line in acknowledgment_note.split('\n'):
                    print(f"    {line}")
            else:
                print(f"  [ACKNOWLEDGE] {self.name}: Orders received, executing.")

            return None  # No command, just tracking

        else:
            self._record_tool_error(f"unknown tool {name!r} - it was ignored.")
            return None

    def get_pending_message(self) -> Optional[Dict[str, Any]]:
        """
        Get and clear pending outgoing message.

        Returns:
            Dict with 'content', 'recipient', 'target_ship' keys, or None
        """
        msg = self.pending_message
        self.pending_message = None

        # Handle legacy string format
        if isinstance(msg, str):
            return {
                "content": msg,
                "recipient": "ALL_ENEMIES",
                "target_ship": None,
            }
        return msg
