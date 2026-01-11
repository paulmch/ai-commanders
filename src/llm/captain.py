"""
LLM-powered captain for space combat simulation.

Makes strategic decisions via tool/function calling.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .client import CaptainClient, ToolCall
from .tools import get_captain_tools, PERSONALITY_SELECTION_TOOLS
from .prompts import (
    build_captain_prompt,
    build_personality_selection_prompt,
    CaptainPersonality,
    PERSONALITY_PRESETS,
)
from .communication import CaptainMessage, MessageType


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

    def select_personality(self, distance_km: float, verbose: bool = False) -> Dict[str, Any]:
        """
        Let the LLM choose its personality before battle starts.

        Args:
            distance_km: Starting distance for scenario context
            verbose: Whether to print selection info

        Returns:
            Dict with chosen personality info
        """
        prompt = build_personality_selection_prompt(distance_km)

        messages = [{"role": "user", "content": prompt}]

        # Call LLM with personality selection tool
        tool_calls = self.client.decide_with_tools(
            messages=messages,
            tools=PERSONALITY_SELECTION_TOOLS,
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
                actions.append(f"MSG: \"{msg}...\"" if len(msg) >= 30 else f"MSG: \"{msg}\"")
            elif tc.name == "surrender":
                actions.append("SURRENDER")
            elif tc.name == "propose_draw":
                actions.append("PROPOSE DRAW")

        return ", ".join(actions) if actions else "No actions"

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

        lines = ["YOUR SHOT HISTORY:"]

        # Stats by range
        for bracket, (hits, total) in hits_by_range.items():
            if total > 0:
                pct = (hits / total) * 100
                lines.append(f"  {bracket}: {hits}/{total} hits ({pct:.0f}%)")

        # Recent shots detail
        lines.append("  Recent shots:")
        for shot in recent[-5:]:  # Last 5 only for detail
            result_str = f"HIT {shot['damage_gj']:.1f}GJ" if shot["result"] == "HIT" else "MISS"
            closing = "closing" if shot["rel_velocity_kps"] < 0 else "separating"
            lines.append(
                f"    {shot['weapon']}: {shot['distance_km']:.0f}km, "
                f"{abs(shot['rel_velocity_kps']):.1f}km/s {closing} -> {result_str}"
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

    def _build_tactical_status(
        self,
        ship: Any,
        enemy: Optional[Any],
        simulation: Any,
    ) -> Dict[str, Any]:
        """Build tactical status dict with raw data for LLM decision making."""
        from ..physics import Vector3D

        status = {
            "sim_time": simulation.current_time,
            "distance_km": 1000,
            "closing_rate": 0,
            "relative_position": {"x": 0, "y": 0, "z": 0},
            "relative_velocity": {"x": 0, "y": 0, "z": 0},
            "ship_forward": {"x": 1, "y": 0, "z": 0},
            "angle_to_enemy_deg": 0,
            "our_hit_chance": 0,
            "our_shots": 0,
            "our_hits": 0,
            "enemy_shots": 0,
            "enemy_hits": 0,
            "our_damage_dealt": 0,
            "our_damage_taken": 0,
            "incoming_projectiles": [],
        }

        # Ship forward direction
        status["ship_forward"] = {
            "x": ship.forward.x,
            "y": ship.forward.y,
            "z": ship.forward.z,
        }

        if enemy:
            # Calculate relative position (enemy relative to us)
            rel_pos = enemy.position - ship.position
            distance_m = rel_pos.magnitude
            distance_km = distance_m / 1000
            status["distance_km"] = distance_km

            # Relative position in km (X=forward, Y=right, Z=up in ship frame)
            # For simplicity, we use world coordinates but label them
            status["relative_position"] = {
                "x": rel_pos.x / 1000,  # km
                "y": rel_pos.y / 1000,
                "z": rel_pos.z / 1000,
            }

            # Calculate relative velocity (how enemy moves relative to us)
            rel_vel = enemy.velocity - ship.velocity
            status["relative_velocity"] = {
                "x": rel_vel.x / 1000,  # km/s
                "y": rel_vel.y / 1000,
                "z": rel_vel.z / 1000,
            }

            # Closing rate (positive = getting closer)
            if distance_m > 0:
                closing_rate = -rel_pos.normalized().dot(rel_vel) / 1000
            else:
                closing_rate = 0
            status["closing_rate"] = closing_rate

            # Calculate angle from ship nose to enemy (0° = pointing directly at enemy)
            import math
            if distance_m > 0:
                direction_to_enemy = rel_pos.normalized()
                dot = ship.forward.dot(direction_to_enemy)
                dot = max(-1.0, min(1.0, dot))  # Clamp for acos
                angle_deg = math.degrees(math.acos(dot))
                status["angle_to_enemy_deg"] = angle_deg

            # Calculate hit probability using simple model
            if distance_km <= 500:
                base_hit = max(0.05, 0.9 - (distance_km / 500) * 0.85)
            else:
                base_hit = 0.05
            status["our_hit_chance"] = base_hit * 100  # percentage

            # Combat stats
            status["our_shots"] = ship.shots_fired
            status["our_hits"] = ship.hits_scored
            status["enemy_shots"] = enemy.shots_fired
            status["enemy_hits"] = enemy.hits_scored
            status["our_damage_dealt"] = ship.damage_dealt_gj
            status["our_damage_taken"] = ship.damage_taken_gj

            # Enemy armor status (for visual damage assessment)
            if enemy.armor:
                enemy_nose = enemy.armor.get_section("nose")
                enemy_lateral = enemy.armor.get_section("lateral")
                enemy_tail = enemy.armor.get_section("tail")
                status["enemy_armor"] = {
                    "nose_damage_pct": enemy_nose.damage_percent if enemy_nose else 0,
                    "lateral_damage_pct": enemy_lateral.damage_percent if enemy_lateral else 0,
                    "tail_damage_pct": enemy_tail.damage_percent if enemy_tail else 0,
                }
            else:
                status["enemy_armor"] = {}

            # Enemy hull damage (estimated from visible damage)
            # In real combat you'd estimate from secondary explosions, debris, power fluctuations
            status["enemy_hull_percent"] = enemy.hull_integrity

        # Build incoming projectiles list with ETAs
        incoming_projectiles = []
        if hasattr(simulation, 'projectiles') and simulation.projectiles:
            for proj in simulation.projectiles:
                if hasattr(proj, 'target_id') and proj.target_id == ship.ship_id:
                    # Calculate distance and ETA
                    proj_pos = proj.position if hasattr(proj, 'position') else Vector3D(0, 0, 0)
                    proj_vel = proj.velocity if hasattr(proj, 'velocity') else Vector3D(0, 0, 0)

                    dist_to_ship = (ship.position - proj_pos).magnitude
                    dist_km = dist_to_ship / 1000

                    # Calculate relative velocity toward ship
                    to_ship = (ship.position - proj_pos).normalized()
                    approach_speed = proj_vel.dot(to_ship)  # m/s toward ship

                    # Estimate ETA
                    if approach_speed > 0:
                        eta_s = dist_to_ship / approach_speed
                    else:
                        eta_s = 999  # Not approaching

                    # Determine weapon type by velocity
                    proj_speed_kps = proj_vel.magnitude / 1000
                    if proj_speed_kps > 8:
                        weapon_type = "spinal"
                    else:
                        weapon_type = "turret"

                    incoming_projectiles.append({
                        "weapon_type": weapon_type,
                        "distance_km": dist_km,
                        "eta_seconds": eta_s,
                    })

        # Sort by ETA
        incoming_projectiles.sort(key=lambda p: p["eta_seconds"])
        status["incoming_projectiles"] = incoming_projectiles[:5]  # Limit to 5

        # Check for incoming torpedoes (add to projectiles list)
        if hasattr(simulation, 'torpedoes') and simulation.torpedoes:
            for torp_flight in simulation.torpedoes:
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

            # Handle new independent spinal/turret modes
            orders = []

            # Spinal weapon order
            spinal_mode = args.get("spinal_mode")
            if spinal_mode:
                try:
                    spinal_command = WeaponsCommand[spinal_mode]
                except KeyError:
                    spinal_command = WeaponsCommand.FIRE_WHEN_OPTIMAL

                orders.append(WeaponsOrder(
                    command=spinal_command,
                    weapon_slot="weapon_0",
                    target_id=target_id,
                    min_hit_probability=args.get("spinal_min_probability", 0.3),
                    max_range_km=args.get("spinal_max_range_km", 500.0),
                ))

            # Turret weapon order
            turret_mode = args.get("turret_mode")
            if turret_mode:
                try:
                    turret_command = WeaponsCommand[turret_mode]
                except KeyError:
                    turret_command = WeaponsCommand.FIRE_WHEN_OPTIMAL

                orders.append(WeaponsOrder(
                    command=turret_command,
                    weapon_slot="weapon_1",
                    target_id=target_id,
                    min_hit_probability=args.get("turret_min_probability", 0.3),
                    max_range_km=args.get("turret_max_range_km", 300.0),
                ))

            # If no specific modes set, default both to FIRE_WHEN_OPTIMAL
            if not orders:
                orders.append(WeaponsOrder(
                    command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
                    weapon_slot="all",
                    target_id=target_id,
                    min_hit_probability=0.3,
                ))

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
            self.pending_message = message
            self._record_sent_message(message, simulation.current_time)
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
