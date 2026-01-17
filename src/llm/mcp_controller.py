"""
MCP Controller - Replaces LLMAdmiral for MCP-controlled fleets.

Key design: MCP receives admiral-level visibility (full friendly ship data,
observable enemy data) but issues captain-level commands directly to ships,
bypassing AI captains entirely.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from .mcp_state import (
    get_mcp_state,
    MCPBattleState,
    MCPCommand,
    MCPCommandType,
)
from .mcp_chat import AdmiralChat, ChatMessage
from .admiral import (
    AdmiralSnapshot,
    FriendlyShipSnapshot,
    EnemyShipSnapshot,
    ProjectileSnapshot,
)

if TYPE_CHECKING:
    from .captain import LLMCaptain


@dataclass
class MCPControllerConfig:
    """Configuration for an MCP-controlled fleet."""
    faction: str
    name: str = "MCP Commander"
    command_timeout: float = 60.0  # Seconds to wait for MCP client


class MCPController:
    """
    Controller for MCP-controlled fleets.

    Replaces LLMAdmiral when a fleet is controlled via MCP.
    Provides admiral-level visibility but accepts captain-level commands.
    """

    def __init__(
        self,
        config: MCPControllerConfig,
        fleet_data: Dict[str, Any],
        chat: Optional[AdmiralChat] = None,
    ):
        """
        Initialize MCP controller.

        Args:
            config: Controller configuration
            fleet_data: Ship specifications from fleet_ships.json
            chat: Shared chat system (optional)
        """
        self.config = config
        self.faction = config.faction
        self.fleet_data = fleet_data
        self.chat = chat or AdmiralChat()

        # Get shared state singleton
        self._state = get_mcp_state()
        self._state.register_faction(self.faction)

        # State tracking
        self.decision_count = 0
        self.has_proposed_draw = False
        self.has_accepted_draw = False
        self.has_surrendered = False

        # Pending message to enemy (via chat system)
        self._pending_enemy_message: Optional[str] = None

        # Ship mappings
        self._ship_id_to_name: Dict[str, str] = {}
        self._ship_name_to_id: Dict[str, str] = {}

    @property
    def name(self) -> str:
        """Get controller name."""
        return self.config.name

    def set_ship_mapping(self, name_to_id: Dict[str, str]) -> None:
        """Set the mapping from ship names to IDs."""
        self._ship_name_to_id = name_to_id
        self._ship_id_to_name = {v: k for k, v in name_to_id.items()}

    def build_state_for_mcp(
        self,
        simulation: Any,
        captains: List['LLMCaptain'],
    ) -> MCPBattleState:
        """
        Build battle state for MCP consumption.

        Converts simulation state to MCPBattleState format.

        Args:
            simulation: Combat simulation instance
            captains: List of friendly captains (for reference, not control)

        Returns:
            MCPBattleState ready for MCP client
        """
        # Build friendly ship data
        friendly_ships = []
        for captain in captains:
            ship_id = getattr(captain, 'ship_id', None)
            if not ship_id:
                continue
            ship = simulation.get_ship(ship_id)
            if ship and not ship.is_destroyed:
                ship_data = self._build_friendly_ship_data(ship, captain, simulation)
                friendly_ships.append(ship_data)

        # Build enemy ship data
        enemy_ships = []
        if captains:
            first_ship_id = getattr(captains[0], 'ship_id', None)
            if first_ship_id:
                for enemy in simulation.get_enemy_ships(first_ship_id):
                    if not enemy.is_destroyed:
                        enemy_data = self._build_enemy_ship_data(enemy, friendly_ships, simulation)
                        enemy_ships.append(enemy_data)

        # Build projectile data
        projectiles = self._build_projectile_data(simulation)

        # Build torpedo data
        torpedoes = self._build_torpedo_data(simulation, friendly_ships)

        # Get chat history
        chat_history = self.chat.get_recent_history(self.faction)

        # Build fleet summary
        ship_types = [s.get("ship_type", "unknown") for s in friendly_ships]
        type_counts = {}
        for t in ship_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        summary_parts = [f"{count} {stype.title()}" for stype, count in type_counts.items()]
        fleet_summary = f"{len(friendly_ships)} ships: {', '.join(summary_parts)}"

        return MCPBattleState(
            timestamp=simulation.current_time,
            faction=self.faction,
            friendly_ships=friendly_ships,
            enemy_ships=enemy_ships,
            projectiles=projectiles,
            torpedoes=torpedoes,
            chat_history=chat_history,
            fleet_summary=fleet_summary,
            is_battle_active=True,
            checkpoint_number=self.decision_count,
            enemy_proposed_draw=False,  # TODO: Get from enemy controller
        )

    def _build_friendly_ship_data(
        self,
        ship: Any,
        captain: 'LLMCaptain',
        simulation: Any,
    ) -> Dict[str, Any]:
        """Build detailed data for a friendly ship (same info captains see)."""
        # Get ship capabilities from fleet data
        ship_spec = self.fleet_data.get("ships", {}).get(captain.config.ship_type, {})
        propulsion = ship_spec.get("propulsion", {})

        # Position in km
        pos = ship.position
        pos_km = {"x": pos.x / 1000, "y": pos.y / 1000, "z": pos.z / 1000}

        # Velocity
        vel = ship.velocity
        vel_vector = {"x": vel.x / 1000, "y": vel.y / 1000, "z": vel.z / 1000}
        vel_kps = vel.magnitude / 1000

        # Ship forward vector (for firing arc calculations)
        forward = {"x": ship.forward.x, "y": ship.forward.y, "z": ship.forward.z}

        # Weapons status - detailed like captains see
        weapons_ready = []
        weapons_cooling = []
        weapons_destroyed = []
        weapons_detailed = {}
        if hasattr(ship, 'weapons'):
            for slot, weapon in ship.weapons.items():
                if slot.startswith("pd_"):
                    continue
                weapon_info = {
                    "slot": slot,
                    "is_turreted": weapon.weapon.is_turreted if hasattr(weapon, 'weapon') else False,
                    "operational": weapon.is_operational if hasattr(weapon, 'is_operational') else True,
                    "ready": weapon.cooldown_remaining <= 0,
                    "cooldown_remaining": weapon.cooldown_remaining,
                }
                weapons_detailed[slot] = weapon_info

                if hasattr(weapon, 'is_operational') and not weapon.is_operational:
                    weapons_destroyed.append(slot)
                elif weapon.cooldown_remaining <= 0:
                    weapons_ready.append(slot)
                else:
                    weapons_cooling.append({
                        "slot": slot,
                        "cooldown_remaining": weapon.cooldown_remaining,
                    })

        # Armor status per section (like captains see)
        from ..combat import HitLocation
        armor_status = {
            "nose": {"thickness_cm": 0, "damage_percent": 0},
            "lateral": {"thickness_cm": 0, "damage_percent": 0},
            "tail": {"thickness_cm": 0, "damage_percent": 0},
        }
        if hasattr(ship, 'armor') and ship.armor:
            for section_name, location in [("nose", HitLocation.NOSE), ("lateral", HitLocation.LATERAL), ("tail", HitLocation.TAIL)]:
                section = ship.armor.get_section(location)
                if section:
                    armor_status[section_name] = {
                        "thickness_cm": section.thickness_cm,
                        "damage_percent": section.damage_percent,
                    }

        # Module damage status (only damaged/destroyed modules, like captains see)
        damaged_modules = {}
        if hasattr(ship, 'module_layout') and ship.module_layout:
            for module in ship.module_layout.get_all_modules():
                if module.health_percent < 100:
                    damaged_modules[module.name] = {
                        "health_percent": module.health_percent,
                        "operational": module.is_functional,
                        "destroyed": module.is_destroyed,
                        "type": module.module_type.value,
                    }

        # Combat statistics (like captains see)
        combat_stats = {
            "shots_fired": ship.shots_fired if hasattr(ship, 'shots_fired') else 0,
            "hits_scored": ship.hits_scored if hasattr(ship, 'hits_scored') else 0,
            "damage_dealt_gj": ship.damage_dealt_gj if hasattr(ship, 'damage_dealt_gj') else 0,
            "damage_taken_gj": ship.damage_taken_gj if hasattr(ship, 'damage_taken_gj') else 0,
        }

        # Get targeting info
        targeted_by = []
        for enemy in simulation.get_enemy_ships(ship.ship_id):
            if hasattr(enemy, 'primary_target_id') and enemy.primary_target_id == ship.ship_id:
                targeted_by.append(enemy.ship_name if hasattr(enemy, 'ship_name') else enemy.ship_id)

        # Current maneuver
        maneuver_str = "MAINTAIN"
        if hasattr(ship, 'current_maneuver') and ship.current_maneuver:
            maneuver_str = ship.current_maneuver.maneuver_type.name

        # Heatsink capacity
        heatsink_capacity = 0
        if hasattr(ship, 'heatsink_capacity_gj'):
            heatsink_capacity = ship.heatsink_capacity_gj
        elif hasattr(ship, 'thermal') and ship.thermal:
            heatsink_capacity = ship.thermal.heatsink_capacity_gj if hasattr(ship.thermal, 'heatsink_capacity_gj') else 0

        return {
            "ship_id": ship.ship_id,
            "ship_name": ship.ship_name if hasattr(ship, 'ship_name') else ship.ship_id,
            "ship_type": captain.config.ship_type,
            "captain_name": captain.config.name,
            "position_km": pos_km,
            "velocity_kps": vel_kps,
            "velocity_vector": vel_vector,
            "forward_vector": forward,
            "hull_integrity": ship.hull_integrity,  # Already a percentage (0-100)
            "delta_v_remaining": ship.remaining_delta_v_kps,
            "heat_percent": ship.heat_percent if hasattr(ship, 'heat_percent') else 0,
            "heatsink_capacity_gj": heatsink_capacity,
            "max_acceleration_g": propulsion.get("combat_acceleration_g", 2.0),
            "max_delta_v": propulsion.get("delta_v_kps", 500),
            # Armor (per section)
            "armor": armor_status,
            # Weapons (detailed)
            "weapons": weapons_detailed,
            "weapons_ready": weapons_ready,
            "weapons_cooling": weapons_cooling,
            "weapons_destroyed": weapons_destroyed,
            # Modules
            "damaged_modules": damaged_modules,
            # Combat stats
            "combat_stats": combat_stats,
            # Current state
            "current_maneuver": maneuver_str,
            "current_target": ship.primary_target_id if hasattr(ship, 'primary_target_id') else None,
            "radiators_extended": ship.radiators_extended if hasattr(ship, 'radiators_extended') else False,
            "targeted_by": targeted_by,
        }

    def _build_enemy_ship_data(
        self,
        ship: Any,
        friendly_ships: List[Dict[str, Any]],
        simulation: Any,
    ) -> Dict[str, Any]:
        """Build observable data for an enemy ship (same info captains see about enemies)."""
        pos = ship.position
        pos_km = {"x": pos.x / 1000, "y": pos.y / 1000, "z": pos.z / 1000}

        vel = ship.velocity
        vel_vector = {"x": vel.x / 1000, "y": vel.y / 1000, "z": vel.z / 1000}
        vel_kps = vel.magnitude / 1000

        # Find closest friendly and calculate relative info
        min_dist = float('inf')
        closing_rate = 0.0
        closest_friendly_id = None
        closest_friendly_forward = None
        relative_position = {"x": 0, "y": 0, "z": 0}
        relative_velocity = {"x": 0, "y": 0, "z": 0}

        for friendly in friendly_ships:
            f_pos = friendly["position_km"]
            dx = pos_km["x"] - f_pos["x"]
            dy = pos_km["y"] - f_pos["y"]
            dz = pos_km["z"] - f_pos["z"]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5

            if dist < min_dist:
                min_dist = dist
                closest_friendly_id = friendly["ship_id"]
                closest_friendly_forward = friendly.get("forward_vector")
                relative_position = {"x": dx, "y": dy, "z": dz}
                f_vel = friendly["velocity_vector"]
                dvx = vel_vector["x"] - f_vel["x"]
                dvy = vel_vector["y"] - f_vel["y"]
                dvz = vel_vector["z"] - f_vel["z"]
                relative_velocity = {"x": dvx, "y": dvy, "z": dvz}
                if dist > 0:
                    closing_rate = -(dx*dvx + dy*dvy + dz*dvz) / dist

        # Hull integrity (actual - like captains see, already 0-100%)
        hull_percent = ship.hull_integrity if hasattr(ship, 'hull_integrity') else 100

        # Armor damage per section (actual - like captains see)
        from ..combat import HitLocation
        armor_damage = {
            "nose_damage_pct": 0,
            "lateral_damage_pct": 0,
            "tail_damage_pct": 0,
        }
        if hasattr(ship, 'armor') and ship.armor:
            nose = ship.armor.get_section(HitLocation.NOSE)
            lateral = ship.armor.get_section(HitLocation.LATERAL)
            tail = ship.armor.get_section(HitLocation.TAIL)
            if nose:
                armor_damage["nose_damage_pct"] = nose.damage_percent
            if lateral:
                armor_damage["lateral_damage_pct"] = lateral.damage_percent
            if tail:
                armor_damage["tail_damage_pct"] = tail.damage_percent

        # Calculate angle from closest friendly's nose to this enemy
        angle_deg = 0.0
        if closest_friendly_forward and min_dist > 0:
            # Normalize relative position
            import math
            rel_mag = math.sqrt(relative_position["x"]**2 + relative_position["y"]**2 + relative_position["z"]**2)
            if rel_mag > 0:
                rel_norm = {
                    "x": relative_position["x"] / rel_mag,
                    "y": relative_position["y"] / rel_mag,
                    "z": relative_position["z"] / rel_mag,
                }
                # Dot product with forward vector
                dot = (closest_friendly_forward["x"] * rel_norm["x"] +
                       closest_friendly_forward["y"] * rel_norm["y"] +
                       closest_friendly_forward["z"] * rel_norm["z"])
                # Clamp to [-1, 1] to avoid math domain error
                dot = max(-1.0, min(1.0, dot))
                angle_deg = math.degrees(math.acos(dot))

        # Estimate hit chance based on range (simplified - captains use more complex calculation)
        # This is a rough approximation based on typical weapon accuracy curves
        hit_chance = 0.0
        if min_dist > 0:
            # Base hit chance decreases with distance
            # ~80% at 100km, ~40% at 500km, ~10% at 1000km
            hit_chance = max(0, min(100, 100 * (1.0 - (min_dist / 1500))))

        # Combat statistics (like captains see)
        combat_stats = {
            "shots_fired": ship.shots_fired if hasattr(ship, 'shots_fired') else 0,
            "hits_scored": ship.hits_scored if hasattr(ship, 'hits_scored') else 0,
            "damage_dealt_gj": ship.damage_dealt_gj if hasattr(ship, 'damage_dealt_gj') else 0,
            "damage_taken_gj": ship.damage_taken_gj if hasattr(ship, 'damage_taken_gj') else 0,
        }

        # Check if enemy has us targeted
        has_friendly_targeted = False
        targeting_friendly_id = None
        if hasattr(ship, 'primary_target_id') and ship.primary_target_id:
            for friendly in friendly_ships:
                if ship.primary_target_id == friendly["ship_id"]:
                    has_friendly_targeted = True
                    targeting_friendly_id = friendly["ship_id"]
                    break

        # Get enemy ship capabilities from fleet data (if available)
        ship_type = ship.ship_type if hasattr(ship, 'ship_type') else "unknown"
        ship_spec = self.fleet_data.get("ships", {}).get(ship_type, {})
        propulsion = ship_spec.get("propulsion", {})

        return {
            "ship_id": ship.ship_id,
            "ship_name": ship.ship_name if hasattr(ship, 'ship_name') else ship.ship_id,
            "ship_type": ship_type,
            "ship_class": ship_type.title() if ship_type != "unknown" else "Unknown",
            "position_km": pos_km,
            "velocity_kps": vel_kps,
            "velocity_vector": vel_vector,
            # Distance and closing info
            "distance_from_closest_friendly_km": min_dist,
            "distance_km": min_dist,  # Alias for compatibility
            "closing_rate_kps": closing_rate,
            "relative_position": relative_position,
            "relative_velocity": relative_velocity,
            # Firing solution
            "angle_deg": angle_deg,
            "hit_chance": hit_chance,
            # Condition (actual values - like captains see)
            "hull_percent": hull_percent,
            "armor": armor_damage,
            # Combat stats
            "combat_stats": combat_stats,
            # Targeting
            "has_friendly_targeted": has_friendly_targeted,
            "targeting_friendly_id": targeting_friendly_id,
            # Capabilities (from ship class data)
            "max_acceleration_g": propulsion.get("combat_acceleration_g"),
            "max_delta_v": propulsion.get("delta_v_kps"),
        }

    def _build_projectile_data(self, simulation: Any) -> List[Dict[str, Any]]:
        """Build data for projectiles in flight (enhanced like captains see)."""
        import math
        projectiles = []

        for proj in simulation.projectiles:
            if hasattr(proj, 'target_ship_id') and proj.target_ship_id:
                target = simulation.get_ship(proj.target_ship_id)
                if target:
                    # Position and distance
                    proj_pos = proj.projectile.position if hasattr(proj.projectile, 'position') else proj.position
                    proj_vel = proj.projectile.velocity if hasattr(proj.projectile, 'velocity') else proj.velocity

                    dist_m = (target.position - proj_pos).magnitude
                    dist_km = dist_m / 1000

                    # Calculate ETA based on approach speed
                    to_target = (target.position - proj_pos).normalized()
                    approach_speed = proj_vel.dot(to_target) if hasattr(proj_vel, 'dot') else proj_vel.magnitude
                    eta = dist_m / approach_speed if approach_speed > 0 else 999

                    # Infer weapon type from speed (like captains do)
                    proj_speed_kps = proj_vel.magnitude / 1000
                    weapon_type = "Spinal" if proj_speed_kps > 8 else "Turret"

                    # Get source ship name
                    source_name = proj.source_ship_id
                    source_ship = simulation.get_ship(proj.source_ship_id)
                    if source_ship:
                        source_name = getattr(source_ship, 'ship_name', proj.source_ship_id)

                    # Calculate bearing to target (what direction it's coming from)
                    bearing = self._calculate_bearing(target, proj_pos, proj_vel)

                    # Damage
                    damage_gj = proj.projectile.kinetic_energy_gj if hasattr(proj.projectile, 'kinetic_energy_gj') else 0

                    projectiles.append({
                        "source_ship": proj.source_ship_id,
                        "source_name": source_name,
                        "target_ship": proj.target_ship_id,
                        "weapon_type": weapon_type,
                        "distance_km": dist_km,
                        "eta_seconds": eta,
                        "damage_gj": damage_gj,
                        "bearing": bearing,
                        "speed_kps": proj_speed_kps,
                    })

        # Sort by ETA
        projectiles.sort(key=lambda p: p["eta_seconds"])
        return projectiles

    def _calculate_bearing(self, ship: Any, proj_pos: Any, proj_vel: Any) -> str:
        """Calculate bearing from ship's perspective (like captains do)."""
        import math

        # Vector from ship to projectile
        to_proj = proj_pos - ship.position
        if to_proj.magnitude < 1:
            return "IMPACT"

        to_proj_norm = to_proj.normalized()

        # Ship's orientation vectors
        forward = ship.forward
        # Assume +Y is "up" for the ship, calculate right vector
        up = getattr(ship, 'up', None)
        if up is None:
            # Default up vector
            from ..physics import Vector3D
            up = Vector3D(0, 0, 1)
        right = forward.cross(up).normalized()

        # Project onto ship's reference frame
        forward_comp = to_proj_norm.dot(forward)  # +1 = nose, -1 = tail
        right_comp = to_proj_norm.dot(right)      # +1 = starboard, -1 = port
        up_comp = to_proj_norm.dot(up)            # +1 = dorsal, -1 = ventral

        # Determine primary direction
        if abs(forward_comp) > 0.7:
            if forward_comp > 0:
                base = "NOSE"
            else:
                base = "TAIL"
        else:
            if abs(right_comp) > abs(up_comp):
                base = "STARBOARD" if right_comp > 0 else "PORT"
            else:
                base = "DORSAL" if up_comp > 0 else "VENTRAL"

        # Add qualifier for oblique angles
        if abs(forward_comp) < 0.7 and abs(forward_comp) > 0.3:
            if forward_comp > 0:
                base = base + "-FWD"
            else:
                base = base + "-AFT"

        # Add threat level based on angle
        if abs(forward_comp) > 0.8:
            qualifier = "(frontal)" if forward_comp > 0 else "(rear)"
        elif abs(forward_comp) < 0.3:
            qualifier = "(flank)"
        else:
            qualifier = "(oblique)"

        return f"{base} {qualifier}"

    def _build_torpedo_data(
        self,
        simulation: Any,
        friendly_ships: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build data for torpedo threats targeting our ships."""
        torpedoes = []
        friendly_ids = {s["ship_id"] for s in friendly_ships}

        if hasattr(simulation, 'torpedoes') and simulation.torpedoes:
            for torp_flight in simulation.torpedoes:
                torp = torp_flight.torpedo
                # Only include torpedoes targeting our ships
                if torp.target_id in friendly_ids and not torp_flight.is_disabled:
                    # Get target ship info
                    target_ship = simulation.get_ship(torp.target_id)
                    if target_ship:
                        dist_km = (torp.position - target_ship.position).magnitude / 1000

                        # Calculate ETA
                        to_target = (target_ship.position - torp.position).normalized()
                        approach_speed = torp.velocity.dot(to_target) if hasattr(torp.velocity, 'dot') else torp.velocity.magnitude
                        eta = (target_ship.position - torp.position).magnitude / approach_speed if approach_speed > 0 else 999

                        # Get source info
                        source_name = getattr(torp, 'source_ship_id', 'Unknown')
                        source_ship = simulation.get_ship(source_name) if source_name != 'Unknown' else None
                        if source_ship:
                            source_name = getattr(source_ship, 'ship_name', source_name)

                        torpedoes.append({
                            "target_ship": torp.target_id,
                            "source_ship": getattr(torp, 'source_ship_id', 'Unknown'),
                            "source_name": source_name,
                            "distance_km": dist_km,
                            "eta_seconds": eta,
                            "speed_kps": torp.velocity.magnitude / 1000 if hasattr(torp, 'velocity') else 0,
                        })

        # Sort by ETA
        torpedoes.sort(key=lambda t: t["eta_seconds"])
        return torpedoes

    def update_battle_state(
        self,
        simulation: Any,
        captains: List['LLMCaptain'],
    ) -> None:
        """
        Update the shared state with current battle snapshot.

        Called by battle runner before waiting for MCP commands.

        Args:
            simulation: Combat simulation instance
            captains: List of friendly captains
        """
        state = self.build_state_for_mcp(simulation, captains)
        self._state.update_state(self.faction, state)

    async def get_commands(
        self,
        simulation: Any,
        captains: List['LLMCaptain'],
        timeout: Optional[float] = None,
    ) -> List[MCPCommand]:
        """
        Wait for and retrieve commands from MCP client.

        Args:
            simulation: Combat simulation instance
            captains: List of friendly captains
            timeout: Override default timeout

        Returns:
            List of commands from MCP client
        """
        # Update state for MCP client to read
        self.update_battle_state(simulation, captains)

        # Clear ready flag for this turn
        self._state.clear_ready(self.faction)

        # Wait for MCP client to signal ready
        timeout = timeout or self.config.command_timeout
        ready = await self._state.wait_for_ready(self.faction, timeout)

        if not ready:
            print(f"[MCPController] Warning: {self.faction} timed out waiting for commands")

        # Get pending commands
        commands = self._state.get_pending_commands(self.faction)

        # Process special commands
        for cmd in commands:
            if cmd.command_type == MCPCommandType.SEND_MESSAGE:
                content = cmd.parameters.get("content", "")
                if content:
                    self.chat.send_message(
                        self.faction,
                        content,
                        simulation.current_time,
                    )
            elif cmd.command_type == MCPCommandType.PROPOSE_DRAW:
                if cmd.parameters.get("accept"):
                    self.has_accepted_draw = True
                else:
                    self.has_proposed_draw = True
            elif cmd.command_type == MCPCommandType.SURRENDER:
                self.has_surrendered = True

        self.decision_count += 1
        return commands

    def receive_enemy_message(self, message: str) -> None:
        """
        Receive a message from enemy admiral.

        Messages are stored in chat history and will be visible in next state update.

        Args:
            message: Message content
        """
        # Enemy messages are already added to chat by the sender
        # This method exists for API compatibility with LLMAdmiral
        pass

    def get_pending_message(self) -> Optional[str]:
        """
        Get and clear pending message to enemy.

        Returns:
            Message content or None
        """
        msg = self._pending_enemy_message
        self._pending_enemy_message = None
        return msg


def apply_mcp_commands_to_simulation(
    commands: List[MCPCommand],
    simulation: Any,
    faction: str,
) -> Dict[str, Any]:
    """
    Apply MCP commands to the simulation.

    Converts MCP commands to simulation-level actions.

    Args:
        commands: List of MCP commands
        simulation: Combat simulation instance
        faction: Faction issuing commands

    Returns:
        Dict with results/errors for each command
    """
    from ..simulation import ManeuverType, Maneuver

    results = {"applied": [], "errors": []}

    for cmd in commands:
        try:
            if cmd.command_type == MCPCommandType.SET_MANEUVER:
                ship = simulation.get_ship(cmd.ship_id)
                if ship:
                    # Convert maneuver type string to enum
                    maneuver_type_str = cmd.parameters.get("maneuver_type", "MAINTAIN")
                    maneuver_type = ManeuverType[maneuver_type_str]

                    # Get heading direction if provided (for HEADING maneuver)
                    heading_direction = cmd.parameters.get("heading_direction")

                    # Create maneuver (same format as LLM captains)
                    maneuver = Maneuver(
                        maneuver_type=maneuver_type,
                        start_time=simulation.current_time,
                        duration=30.0,  # Match LLM captain duration
                        throttle=cmd.parameters.get("throttle", 1.0),
                        target_id=cmd.parameters.get("target_id"),
                        heading_direction=heading_direction,
                    )
                    # Use inject_command for consistency with LLM captain flow
                    simulation.inject_command(cmd.ship_id, maneuver)
                    results["applied"].append({
                        "command": "set_maneuver",
                        "ship_id": cmd.ship_id,
                        "maneuver": maneuver_type_str,
                        "heading_direction": heading_direction,
                    })
                else:
                    results["errors"].append({
                        "command": "set_maneuver",
                        "ship_id": cmd.ship_id,
                        "error": "Ship not found",
                    })

            elif cmd.command_type == MCPCommandType.SET_WEAPONS_ORDER:
                ship = simulation.get_ship(cmd.ship_id)
                if ship:
                    from ..firecontrol import WeaponsCommand, WeaponsOrder

                    spinal_mode_str = cmd.parameters.get("spinal_mode")
                    turret_mode_str = cmd.parameters.get("turret_mode")
                    target_id = ship.primary_target_id  # Use ship's current primary target

                    applied_modes = []
                    orders = []

                    # Build spinal weapons orders (non-turreted weapons)
                    if spinal_mode_str and hasattr(ship, 'weapons'):
                        try:
                            weapons_cmd = WeaponsCommand[spinal_mode_str]
                            # Find spinal (non-turreted) weapon slots
                            for slot, weapon_state in ship.weapons.items():
                                if slot.startswith("pd_"):
                                    continue  # Skip point defense
                                # Check weapon's is_turreted property
                                if hasattr(weapon_state, 'weapon') and not weapon_state.weapon.is_turreted:
                                    orders.append(WeaponsOrder(
                                        command=weapons_cmd,
                                        weapon_slot=slot,
                                        target_id=target_id,
                                    ))
                            applied_modes.append(f"spinal={spinal_mode_str}")
                        except KeyError:
                            results["errors"].append({
                                "command": "set_weapons_order",
                                "ship_id": cmd.ship_id,
                                "error": f"Invalid spinal_mode: {spinal_mode_str}",
                            })

                    # Build turret weapons orders (turreted weapons)
                    if turret_mode_str and hasattr(ship, 'weapons'):
                        try:
                            weapons_cmd = WeaponsCommand[turret_mode_str]
                            # Find turreted weapon slots
                            for slot, weapon_state in ship.weapons.items():
                                if slot.startswith("pd_"):
                                    continue  # Skip point defense
                                # Check weapon's is_turreted property
                                if hasattr(weapon_state, 'weapon') and weapon_state.weapon.is_turreted:
                                    orders.append(WeaponsOrder(
                                        command=weapons_cmd,
                                        weapon_slot=slot,
                                        target_id=target_id,
                                    ))
                            applied_modes.append(f"turret={turret_mode_str}")
                        except KeyError:
                            results["errors"].append({
                                "command": "set_weapons_order",
                                "ship_id": cmd.ship_id,
                                "error": f"Invalid turret_mode: {turret_mode_str}",
                            })

                    # Use inject_command with same format as LLM captains
                    if orders:
                        simulation.inject_command(cmd.ship_id, {
                            "type": "weapons_orders",
                            "orders": orders,
                        })
                        results["applied"].append({
                            "command": "set_weapons_order",
                            "ship_id": cmd.ship_id,
                            "modes": applied_modes,
                            "orders_count": len(orders),
                        })
                else:
                    results["errors"].append({
                        "command": "set_weapons_order",
                        "ship_id": cmd.ship_id,
                        "error": "Ship not found",
                    })

            elif cmd.command_type == MCPCommandType.SET_PRIMARY_TARGET:
                ship = simulation.get_ship(cmd.ship_id)
                if ship:
                    target_id = cmd.parameters.get("target_id")
                    if target_id == "NONE":
                        ship.primary_target_id = None
                    else:
                        ship.primary_target_id = target_id
                    results["applied"].append({
                        "command": "set_primary_target",
                        "ship_id": cmd.ship_id,
                        "target_id": target_id,
                    })
                else:
                    results["errors"].append({
                        "command": "set_primary_target",
                        "ship_id": cmd.ship_id,
                        "error": "Ship not found",
                    })

            elif cmd.command_type == MCPCommandType.SET_RADIATORS:
                ship = simulation.get_ship(cmd.ship_id)
                if ship:
                    extend = cmd.parameters.get("extend", True)
                    # Actually extend/retract radiators in thermal system
                    if ship.thermal_system and ship.thermal_system.radiators:
                        if extend:
                            ship.thermal_system.radiators.extend_all()
                        else:
                            ship.thermal_system.radiators.retract_all()
                    # Also set the flag for status reporting
                    ship.radiators_extended = extend
                    results["applied"].append({
                        "command": "set_radiators",
                        "ship_id": cmd.ship_id,
                        "extend": extend,
                    })
                else:
                    results["errors"].append({
                        "command": "set_radiators",
                        "ship_id": cmd.ship_id,
                        "error": "Ship not found",
                    })

            elif cmd.command_type == MCPCommandType.LAUNCH_TORPEDO:
                ship = simulation.get_ship(cmd.ship_id)
                if ship:
                    target_id = cmd.parameters.get("target_id")
                    # TODO: Implement torpedo launch
                    results["applied"].append({
                        "command": "launch_torpedo",
                        "ship_id": cmd.ship_id,
                        "target_id": target_id,
                        "note": "Torpedo launch not yet implemented",
                    })
                else:
                    results["errors"].append({
                        "command": "launch_torpedo",
                        "ship_id": cmd.ship_id,
                        "error": "Ship not found",
                    })

            # SEND_MESSAGE, PROPOSE_DRAW, SURRENDER, READY are handled in get_commands

        except Exception as e:
            results["errors"].append({
                "command": cmd.command_type.value,
                "ship_id": cmd.ship_id,
                "error": str(e),
            })

    return results
