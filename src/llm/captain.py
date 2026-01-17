"""
LLM-powered captain for space combat simulation.

Makes strategic decisions via tool/function calling.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .client import CaptainClient, ToolCall
from .tools import get_captain_tools, get_weapon_groups_for_ship, PERSONALITY_SELECTION_TOOLS, RESPOND_TO_ORDERS_TOOL
from .prompts import (
    build_captain_prompt,
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
    model: str = "openrouter/anthropic/claude-3.5-sonnet"
    personality: CaptainPersonality = CaptainPersonality.BALANCED
    personality_text: Optional[str] = None  # Custom personality description
    temperature: float = 0.7
    has_torpedoes: bool = False
    ship_type: str = "destroyer"
    fleet_data: Optional[Dict[str, Any]] = None


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
        self.tools = get_captain_tools(has_torpedoes=config.has_torpedoes)

        # State tracking
        self.decision_count = 0
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

        prompt = build_personality_selection_prompt(distance_km, model_name=model_name)

        messages = [{"role": "user", "content": prompt}]

        # Call LLM with personality selection tool (use captain's configured model)
        tool_calls = self.client.decide_with_tools(
            messages=messages,
            tools=PERSONALITY_SELECTION_TOOLS,
            model=self.config.model,
        )

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
        tools = get_captain_tools(has_torpedoes=self.config.has_torpedoes)

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

        # Build history context
        decision_history = self._format_decision_history(last_n=5)
        message_history = self._format_message_history(last_n=6)
        battle_summary = self._format_battle_summary(distance_km)
        shot_history = self._format_shot_history(last_n=10)

        # Build prompt
        recent_hits_text = self._format_recent_hits()
        system_prompt = build_captain_prompt(
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
        )

        # Add Admiral orders to prompt if present
        if self.has_admiral and (self.admiral_orders or self.fleet_directive):
            admiral_orders_text = format_admiral_orders_for_captain(
                self.admiral_orders,
                self.fleet_directive,
            )
            system_prompt = system_prompt + "\n\n" + admiral_orders_text

        # Build messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"DECISION POINT {self.decision_count + 1}. What are your orders, Captain?"},
        ]

        # Get context-appropriate tools (may exclude draw tools if Admiral exists)
        tools = self.get_tools_for_context()

        # Call LLM with tools (use captain's configured model)
        tool_calls = self.client.decide_with_tools(messages, tools, model=self.config.model)

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
            closing = "closing" if shot["rel_velocity_kps"] < 0 else "separating"
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
            nose = ship.armor.get_section("nose")
            lateral = ship.armor.get_section("lateral")
            tail = ship.armor.get_section("tail")
            status["nose_armor"] = nose.current_thickness_cm if nose else 10
            status["lateral_armor"] = lateral.current_thickness_cm if lateral else 5
            status["tail_armor"] = tail.current_thickness_cm if tail else 3
        else:
            status["nose_armor"] = 10
            status["lateral_armor"] = 5
            status["tail_armor"] = 3

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

    def _build_enemy_info(self, ship: Any, enemy: Any, simulation: Any) -> Dict[str, Any]:
        """Build tactical info for a single enemy ship."""
        import math

        info = {
            "ship_id": enemy.ship_id,
            "name": getattr(enemy, 'name', enemy.ship_id),
            "ship_class": getattr(enemy, 'ship_class', 'unknown'),
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
            for proj in simulation.projectiles:
                if hasattr(proj, 'target_id') and proj.target_id == ship.ship_id:
                    proj_pos = proj.position if hasattr(proj, 'position') else Vector3D(0, 0, 0)
                    proj_vel = proj.velocity if hasattr(proj, 'velocity') else Vector3D(0, 0, 0)

                    dist_to_ship = (ship.position - proj_pos).magnitude
                    dist_km = dist_to_ship / 1000

                    to_ship = (ship.position - proj_pos).normalized()
                    approach_speed = proj_vel.dot(to_ship)

                    if approach_speed > 0:
                        eta_s = dist_to_ship / approach_speed
                    else:
                        eta_s = 999

                    proj_speed_kps = proj_vel.magnitude / 1000
                    weapon_type = "Spinal" if proj_speed_kps > 8 else "Turret"

                    # Get source ship name
                    source_name = "Unknown"
                    if hasattr(proj, 'source_ship_id'):
                        source_ship = simulation.get_ship(proj.source_ship_id)
                        if source_ship:
                            source_name = getattr(source_ship, 'name', proj.source_ship_id)

                    # Calculate bearing
                    bearing = self._calculate_impact_bearing(ship, proj_pos, proj_vel)

                    incoming_projectiles.append({
                        "weapon_type": weapon_type,
                        "source": source_name,
                        "distance_km": dist_km,
                        "eta_seconds": eta_s,
                        "bearing": bearing,
                    })

        # Sort by ETA
        incoming_projectiles.sort(key=lambda p: p["eta_seconds"])
        status["incoming_projectiles"] = incoming_projectiles[:5]  # Limit to 5

        # Check for incoming torpedoes
        torpedo_threats = []
        if hasattr(simulation, 'torpedoes') and simulation.torpedoes:
            for torp_flight in simulation.torpedoes:
                torp = torp_flight.torpedo
                if torp.target_id == ship.ship_id and not torp_flight.is_disabled:
                    dist = (torp.position - ship.position).magnitude / 1000
                    torpedo_threats.append({
                        "distance_km": dist,
                        "source": getattr(torp, 'source_ship_id', 'Unknown'),
                    })
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
                print(f"[CAPTAIN] Invalid maneuver: {e}")
                return None

        elif name == "set_primary_target":
            # Set the primary target for this captain
            target_name = args.get("target_name", "")
            # Find enemy ship by name
            enemies = simulation.get_enemy_ships(ship_id)
            for enemy in enemies:
                enemy_name = getattr(enemy, 'name', enemy.ship_id)
                if enemy_name.lower() == target_name.lower() or enemy.ship_id == target_name:
                    self.primary_target_id = enemy.ship_id
                    return None
            # If not found, try partial match
            for enemy in enemies:
                enemy_name = getattr(enemy, 'name', enemy.ship_id)
                if target_name.lower() in enemy_name.lower():
                    self.primary_target_id = enemy.ship_id
                    return None
            print(f"[CAPTAIN] Target not found: {target_name}")
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

            # Process each weapon group dynamically
            for group_name, slots in self.weapon_groups.items():
                mode_key = f"{group_name}_mode"
                prob_key = f"{group_name}_min_probability"
                range_key = f"{group_name}_max_range_km"

                mode = args.get(mode_key)
                if mode:
                    try:
                        command = WeaponsCommand[mode]
                    except KeyError:
                        command = WeaponsCommand.FIRE_WHEN_OPTIMAL

                    min_prob = args.get(prob_key, 0.3)
                    max_range = args.get(range_key, 500.0)

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
                return None
            target = enemies[0]
            return {
                "type": "launch_torpedo",
                "target_id": target.ship_id,
            }

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
                print(f"[CAPTAIN] {self.name} tried to discuss with Admiral but has no Admiral")
                return None

            if self.discussion_exchanges >= self.max_discussion_exchanges:
                print(f"[CAPTAIN] {self.name} has used all {self.max_discussion_exchanges} discussion exchanges")
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
            print(f"[CAPTAIN] Unknown tool: {name}")
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
