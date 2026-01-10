"""
LLM-powered captain for space combat simulation.

Makes strategic decisions via tool/function calling.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .client import CaptainClient, ToolCall
from .tools import get_captain_tools
from .prompts import build_captain_prompt, CaptainPersonality
from .communication import CaptainMessage, MessageType


@dataclass
class LLMCaptainConfig:
    """Configuration for an LLM captain."""
    name: str
    ship_name: str
    model: str = "openrouter/anthropic/claude-3.5-sonnet"
    personality: CaptainPersonality = CaptainPersonality.BALANCED
    temperature: float = 0.7
    has_torpedoes: bool = False


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
        self.has_surrendered = False
        self.has_proposed_draw = False

    def receive_messages(self, messages: List[CaptainMessage]) -> None:
        """
        Receive messages from enemy captain.

        Args:
            messages: List of messages to receive
        """
        self.received_messages.extend(messages)

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

        # Get enemy for tactical info
        enemies = simulation.get_enemy_ships(ship_id)
        enemy = enemies[0] if enemies else None

        # Build status dicts
        ship_status = self._build_ship_status(ship)
        tactical_status = self._build_tactical_status(ship, enemy, simulation)

        # Format received messages
        messages_text = ""
        if self.received_messages:
            from .communication import CommunicationChannel
            messages_text = "\n".join(
                msg.format_for_llm() for msg in self.received_messages
            )
            self.received_messages.clear()

        # Build prompt
        system_prompt = build_captain_prompt(
            captain_name=self.config.name,
            ship_name=self.config.ship_name,
            ship_status=ship_status,
            tactical_status=tactical_status,
            personality=self.config.personality,
            received_messages=messages_text if messages_text else None,
        )

        # Build messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"DECISION POINT {self.decision_count + 1}. What are your orders, Captain?"},
        ]

        # Call LLM with tools
        tool_calls = self.client.decide_with_tools(messages, self.tools)

        # Execute tool calls
        commands = []
        for tc in tool_calls:
            cmd = self._execute_tool(tc, simulation, ship_id)
            if cmd is not None:
                commands.append(cmd)

        # Track decision
        self.decision_count += 1
        self.last_tool_calls = tool_calls  # Store for verbose output
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
        for tc in self.last_tool_calls:
            if tc.name == "set_maneuver":
                maneuver = tc.arguments.get("maneuver_type", "?")
                throttle = tc.arguments.get("throttle", 1.0)
                actions.append(f"{maneuver} @ {throttle*100:.0f}%")
            elif tc.name == "set_weapons_order":
                mode = tc.arguments.get("firing_mode", "?")
                slot = tc.arguments.get("weapon_slot", "all")
                actions.append(f"WEAPONS {slot}: {mode}")
            elif tc.name == "launch_torpedo":
                actions.append("LAUNCH TORPEDO")
            elif tc.name == "set_radiators":
                extend = tc.arguments.get("extend", False)
                actions.append("EXTEND radiators" if extend else "RETRACT radiators")
            elif tc.name == "send_message":
                msg = tc.arguments.get("message", "")[:30]
                actions.append(f"MSG: \"{msg}...\"" if len(msg) >= 30 else f"MSG: \"{msg}\"")
            elif tc.name == "surrender":
                actions.append("SURRENDER")
            elif tc.name == "propose_draw":
                actions.append("PROPOSE DRAW")

        return ", ".join(actions) if actions else "No actions"

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

        return status

    def _build_tactical_status(
        self,
        ship: Any,
        enemy: Optional[Any],
        simulation: Any,
    ) -> Dict[str, Any]:
        """Build tactical status dict."""
        status = {
            "distance_km": 1000,
            "closing_rate": 0,
            "enemy_bearing": "unknown",
            "threats": [],
            "enemy_hit_chance": 0,
            "our_hit_chance": 0,
            "our_shots": 0,
            "our_hits": 0,
            "enemy_shots": 0,
            "enemy_hits": 0,
            "incoming_rounds": 0,
            "our_damage_dealt": 0,
            "our_damage_taken": 0,
        }

        if enemy:
            # Calculate distance
            from ..physics import Vector3D
            rel_pos = enemy.position - ship.position
            distance_m = rel_pos.magnitude
            distance_km = distance_m / 1000
            status["distance_km"] = distance_km

            # Calculate closing rate
            rel_vel = enemy.velocity - ship.velocity
            closing_rate = -rel_pos.normalized().dot(rel_vel) / 1000
            status["closing_rate"] = closing_rate

            # Simple bearing
            if abs(rel_pos.x) > abs(rel_pos.y) and abs(rel_pos.x) > abs(rel_pos.z):
                status["enemy_bearing"] = "ahead" if rel_pos.x > 0 else "behind"
            else:
                status["enemy_bearing"] = "lateral"

            # Calculate hit probabilities using simple model
            # Coilgun accuracy degrades with distance
            # Base: 90% at 0km, drops off with distance
            if distance_km <= 500:
                base_hit = max(0.05, 0.9 - (distance_km / 500) * 0.85)
            else:
                base_hit = 0.05

            status["enemy_hit_chance"] = base_hit * 100  # percentage
            status["our_hit_chance"] = base_hit * 100

            # Combat stats
            status["our_shots"] = ship.shots_fired
            status["our_hits"] = ship.hits_scored
            status["enemy_shots"] = enemy.shots_fired
            status["enemy_hits"] = enemy.hits_scored
            status["our_damage_dealt"] = ship.damage_dealt_gj
            status["our_damage_taken"] = ship.damage_taken_gj

        # Count incoming projectiles (coilgun rounds)
        if hasattr(simulation, 'projectiles') and simulation.projectiles:
            for proj in simulation.projectiles:
                if hasattr(proj, 'target_id') and proj.target_id == ship.ship_id:
                    status["incoming_rounds"] += 1

        # Check for incoming torpedoes
        if hasattr(simulation, 'torpedoes') and simulation.torpedoes:
            for torp_flight in simulation.torpedoes:
                # TorpedoInFlight has .torpedo which is the actual Torpedo object
                torp = torp_flight.torpedo
                if torp.target_id == ship.ship_id and not torp_flight.is_disabled:
                    dist = (torp.position - ship.position).magnitude / 1000
                    status["threats"].append(f"Torpedo {dist:.0f}km away")

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

                # For INTERCEPT, we need target_id
                target_id = None
                if maneuver_type == ManeuverType.INTERCEPT:
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

        elif name == "set_weapons_order":
            from ..firecontrol import WeaponsCommand, WeaponsOrder
            enemies = simulation.get_enemy_ships(ship_id)
            target_id = enemies[0].ship_id if enemies else None

            firing_mode = args.get("firing_mode", "FIRE_WHEN_OPTIMAL")
            weapon_slot = args.get("weapon_slot", "all")
            min_prob = args.get("min_hit_probability", 0.3)
            max_range = args.get("max_range_km", 500.0)
            conserve = args.get("conserve_ammo", False)

            # Map weapon slot names
            slot_map = {
                "all": "all",
                "spinal": "weapon_0",
                "turret": "weapon_1",
            }

            try:
                command = WeaponsCommand[firing_mode]
            except KeyError:
                command = WeaponsCommand.FIRE_WHEN_OPTIMAL

            return {
                "type": "weapons_order",
                "order": WeaponsOrder(
                    command=command,
                    weapon_slot=slot_map.get(weapon_slot, weapon_slot),
                    target_id=target_id,
                    min_hit_probability=min_prob,
                    max_range_km=max_range,
                    conserve_ammo=conserve,
                )
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
            # Queue message for delivery - don't return command
            self.pending_message = args.get("message", "")
            return None

        elif name == "surrender":
            self.has_surrendered = True
            return None

        elif name == "propose_draw":
            self.has_proposed_draw = True
            return None

        else:
            print(f"[CAPTAIN] Unknown tool: {name}")
            return None

    def get_pending_message(self) -> Optional[str]:
        """Get and clear pending outgoing message."""
        msg = self.pending_message
        self.pending_message = None
        return msg
