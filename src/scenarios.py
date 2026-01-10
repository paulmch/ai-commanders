#!/usr/bin/env python3
"""
Combat Scenario Definitions for AI Commanders Space Battle Simulator.

This module provides a comprehensive set of combat scenarios that test different
tactical situations. Each scenario defines:
- Ships involved (types, initial positions, velocities, factions)
- Initial engagement geometry
- Decision interval (20-60 seconds)
- Victory conditions
- Time limits
- Special rules (ammo limits, no retreat, etc.)

Scenarios are designed to test AI captains across a variety of tactical challenges,
from head-on passes to long-range standoffs, from torpedo exchanges to damaged ship fights.

Usage:
    runner = ScenarioRunner()
    result = runner.run_scenario("head_on_pass", captain_a, captain_b)
"""

from __future__ import annotations

import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

# =============================================================================
# IMPORT HANDLING
# =============================================================================
# This module uses try/except imports to work both as a package and standalone.
# Some modules in the codebase use relative imports only, so we handle gracefully.

_IMPORT_ERRORS: List[str] = []

# Physics module - required
try:
    from physics import Vector3D, ShipState, create_ship_state_from_specs
except ImportError:
    try:
        from .physics import Vector3D, ShipState, create_ship_state_from_specs
    except ImportError as e:
        _IMPORT_ERRORS.append(f"physics: {e}")
        # Provide minimal fallback
        from dataclasses import dataclass as _dc
        @_dc
        class Vector3D:
            x: float = 0.0
            y: float = 0.0
            z: float = 0.0
            @property
            def magnitude(self) -> float:
                return (self.x**2 + self.y**2 + self.z**2)**0.5
            def normalized(self) -> 'Vector3D':
                m = self.magnitude
                if m < 1e-10:
                    return Vector3D(1, 0, 0)
                return Vector3D(self.x/m, self.y/m, self.z/m)
            def dot(self, other: 'Vector3D') -> float:
                return self.x*other.x + self.y*other.y + self.z*other.z
            def cross(self, other: 'Vector3D') -> 'Vector3D':
                return Vector3D(
                    self.y*other.z - self.z*other.y,
                    self.z*other.x - self.x*other.z,
                    self.x*other.y - self.y*other.x
                )
            def distance_to(self, other: 'Vector3D') -> float:
                return ((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2)**0.5
            def __add__(self, other: 'Vector3D') -> 'Vector3D':
                return Vector3D(self.x+other.x, self.y+other.y, self.z+other.z)
            def __sub__(self, other: 'Vector3D') -> 'Vector3D':
                return Vector3D(self.x-other.x, self.y-other.y, self.z-other.z)
            def __mul__(self, scalar: float) -> 'Vector3D':
                return Vector3D(self.x*scalar, self.y*scalar, self.z*scalar)
            def __neg__(self) -> 'Vector3D':
                return Vector3D(-self.x, -self.y, -self.z)
            @classmethod
            def zero(cls) -> 'Vector3D':
                return cls(0, 0, 0)
            @classmethod
            def unit_x(cls) -> 'Vector3D':
                return cls(1, 0, 0)
        ShipState = None
        create_ship_state_from_specs = None

# Simulation module - core functionality
_SIMULATION_AVAILABLE = False
try:
    from simulation import (
        CombatSimulation, ShipCombatState, Maneuver, ManeuverType,
        WeaponState, SimulationEvent, SimulationEventType
    )
    _SIMULATION_AVAILABLE = True
except ImportError:
    try:
        from .simulation import (
            CombatSimulation, ShipCombatState, Maneuver, ManeuverType,
            WeaponState, SimulationEvent, SimulationEventType
        )
        _SIMULATION_AVAILABLE = True
    except ImportError as e:
        _IMPORT_ERRORS.append(f"simulation: {e}")
        # Provide stubs for scenarios to be defined even without simulation
        CombatSimulation = None
        ShipCombatState = None
        class ManeuverType(Enum):
            BURN = auto()
            ROTATE = auto()
            EVASIVE = auto()
            INTERCEPT = auto()
            BRAKE = auto()
        @dataclass
        class Maneuver:
            maneuver_type: ManeuverType
            start_time: float = 0.0
            duration: float = 0.0
            throttle: float = 1.0
            direction: Optional[Vector3D] = None
            target_id: Optional[str] = None
        WeaponState = None
        @dataclass
        class SimulationEvent:
            event_type: Any
            timestamp: float
            ship_id: Optional[str] = None
            target_id: Optional[str] = None
            data: dict = field(default_factory=dict)
        class SimulationEventType(Enum):
            PROJECTILE_LAUNCHED = auto()
            PROJECTILE_IMPACT = auto()
            SHIP_DESTROYED = auto()
            SIMULATION_STARTED = auto()
            SIMULATION_ENDED = auto()

# Combat module - optional
try:
    from combat import Weapon, ShipArmor, create_weapon_from_fleet_data, create_ship_armor_from_fleet_data
except ImportError:
    try:
        from .combat import Weapon, ShipArmor, create_weapon_from_fleet_data, create_ship_armor_from_fleet_data
    except ImportError as e:
        _IMPORT_ERRORS.append(f"combat: {e}")
        Weapon = None
        ShipArmor = None
        create_weapon_from_fleet_data = None
        create_ship_armor_from_fleet_data = None

# Thermal module - optional
try:
    from thermal import ThermalSystem
except ImportError:
    try:
        from .thermal import ThermalSystem
    except ImportError as e:
        _IMPORT_ERRORS.append(f"thermal: {e}")
        ThermalSystem = None

# Torpedo module - optional
try:
    from torpedo import TorpedoLauncher, TorpedoSpecs
except ImportError:
    try:
        from .torpedo import TorpedoLauncher, TorpedoSpecs
    except ImportError as e:
        _IMPORT_ERRORS.append(f"torpedo: {e}")
        TorpedoLauncher = None
        TorpedoSpecs = None

# Modules module - optional
try:
    from modules import ModuleLayout
except ImportError:
    try:
        from .modules import ModuleLayout
    except ImportError as e:
        _IMPORT_ERRORS.append(f"modules: {e}")
        ModuleLayout = None


# =============================================================================
# ENUMERATIONS
# =============================================================================

class VictoryCondition(Enum):
    """Types of victory conditions for scenarios."""
    DESTROY_ENEMY = auto()          # All enemy ships destroyed
    DISABLE_ENEMY = auto()          # Enemy ships disabled (critical modules)
    SURVIVE_TIME_LIMIT = auto()     # Survive until time limit
    ESCAPE_ENGAGEMENT = auto()      # Retreat beyond engagement range
    CONTROL_OBJECTIVE = auto()      # Maintain position near objective


class ScenarioOutcome(Enum):
    """Possible outcomes of a scenario."""
    ALPHA_VICTORY = "alpha_victory"
    BETA_VICTORY = "beta_victory"
    DRAW = "draw"
    MUTUAL_DESTRUCTION = "mutual_destruction"
    TIME_LIMIT = "time_limit"


# =============================================================================
# SHIP CONFIGURATION
# =============================================================================

@dataclass
class ShipConfiguration:
    """
    Configuration for a ship in a scenario.

    Attributes:
        ship_id: Unique identifier for this ship.
        ship_type: Ship class (e.g., 'destroyer', 'cruiser').
        faction: Faction identifier.
        position_km: Initial position in kilometers [x, y, z].
        velocity_kps: Initial velocity in km/s [vx, vy, vz].
        forward: Initial forward direction [x, y, z] (optional).
        damage_preset: Pre-applied damage configuration (optional).
        ammo_multiplier: Multiplier for default ammo counts.
        fuel_fraction: Starting propellant fraction (0.0 to 1.0).
    """
    ship_id: str
    ship_type: str
    faction: str
    position_km: Tuple[float, float, float]
    velocity_kps: Tuple[float, float, float]
    forward: Optional[Tuple[float, float, float]] = None
    damage_preset: Optional[Dict[str, Any]] = None
    ammo_multiplier: float = 1.0
    fuel_fraction: float = 1.0

    def to_vectors(self) -> Tuple[Vector3D, Vector3D, Optional[Vector3D]]:
        """Convert configuration to Vector3D objects."""
        position = Vector3D(
            self.position_km[0] * 1000,  # Convert km to m
            self.position_km[1] * 1000,
            self.position_km[2] * 1000
        )
        velocity = Vector3D(
            self.velocity_kps[0] * 1000,  # Convert km/s to m/s
            self.velocity_kps[1] * 1000,
            self.velocity_kps[2] * 1000
        )
        forward = None
        if self.forward:
            forward = Vector3D(*self.forward).normalized()
        return position, velocity, forward


# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

@dataclass
class ScenarioConfig:
    """
    Complete configuration for a combat scenario.

    Attributes:
        name: Unique scenario identifier.
        display_name: Human-readable scenario name.
        description: Detailed scenario description.
        ships: List of ship configurations.
        decision_interval: Seconds between AI decision points (20-60).
        time_limit_s: Maximum scenario duration in seconds.
        victory_conditions: List of victory conditions.
        special_rules: Dictionary of special rules for this scenario.
        expected_outcome: Expected winner and win rate range for testing.
    """
    name: str
    display_name: str
    description: str
    ships: List[ShipConfiguration]
    decision_interval: float = 30.0
    time_limit_s: float = 600.0  # 10 minutes default
    victory_conditions: List[VictoryCondition] = field(default_factory=lambda: [VictoryCondition.DESTROY_ENEMY])
    special_rules: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate scenario configuration."""
        # Clamp decision interval to valid range
        self.decision_interval = max(20.0, min(60.0, self.decision_interval))

        # Ensure we have at least 2 ships from different factions
        if len(self.ships) < 2:
            raise ValueError("Scenario must have at least 2 ships")

        factions = set(ship.faction for ship in self.ships)
        if len(factions) < 2:
            raise ValueError("Scenario must have ships from at least 2 factions")

    def get_ships_by_faction(self, faction: str) -> List[ShipConfiguration]:
        """Get all ship configurations for a faction."""
        return [ship for ship in self.ships if ship.faction == faction]


# =============================================================================
# SCENARIO RESULT
# =============================================================================

@dataclass
class ScenarioResult:
    """
    Result of a completed scenario.

    Attributes:
        scenario_name: Name of the scenario that was run.
        outcome: Final outcome of the scenario.
        winning_faction: Faction that won (if any).
        duration_s: Total simulation duration in seconds.
        alpha_ships_remaining: Number of alpha faction ships surviving.
        beta_ships_remaining: Number of beta faction ships surviving.
        total_damage_dealt: Total damage dealt in GJ.
        events: List of significant events during the battle.
        metrics: Detailed engagement metrics.
    """
    scenario_name: str
    outcome: ScenarioOutcome
    winning_faction: Optional[str]
    duration_s: float
    alpha_ships_remaining: int
    beta_ships_remaining: int
    total_damage_dealt: float
    events: List[SimulationEvent]
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CAPTAIN BEHAVIOR BASE CLASS
# =============================================================================

class CaptainBehavior(ABC):
    """
    Base class for programmed AI captain behaviors.

    Captains receive battle snapshots and return lists of commands.
    Different behavior implementations test various tactical approaches.
    """

    def __init__(self, name: str = "BaseCaptain") -> None:
        """
        Initialize the captain.

        Args:
            name: Display name for this captain.
        """
        self.name = name

    @abstractmethod
    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        """
        Make a decision given the current battle state.

        Args:
            ship_id: ID of the ship this captain controls.
            simulation: The current combat simulation state.

        Returns:
            List of commands to execute.
        """
        pass

    def _get_nearest_enemy(
        self,
        ship: ShipCombatState,
        enemies: List[ShipCombatState]
    ) -> Optional[ShipCombatState]:
        """Get the nearest enemy ship."""
        if not enemies:
            return None
        return min(enemies, key=lambda e: ship.distance_to(e))

    def _create_attack_maneuver(
        self,
        target_id: str,
        throttle: float = 1.0,
        duration: float = 30.0
    ) -> Maneuver:
        """Create an intercept maneuver toward a target."""
        return Maneuver(
            maneuver_type=ManeuverType.INTERCEPT,
            start_time=0.0,  # Will be set by simulation
            duration=duration,
            throttle=throttle,
            target_id=target_id
        )

    def _create_evasive_maneuver(
        self,
        throttle: float = 1.0,
        duration: float = 10.0
    ) -> Maneuver:
        """Create an evasive maneuver."""
        return Maneuver(
            maneuver_type=ManeuverType.EVASIVE,
            start_time=0.0,
            duration=duration,
            throttle=throttle
        )

    def _create_fire_command(
        self,
        weapon_slot: str,
        target_id: str
    ) -> Dict[str, Any]:
        """Create a fire weapon command."""
        return {
            'type': 'fire_at',
            'weapon_slot': weapon_slot,
            'target_id': target_id
        }

    def _create_torpedo_command(self, target_id: str) -> Dict[str, Any]:
        """Create a torpedo launch command."""
        return {
            'type': 'launch_torpedo',
            'target_id': target_id
        }


# =============================================================================
# SPECIFIC CAPTAIN BEHAVIORS
# =============================================================================

class AggressiveCaptain(CaptainBehavior):
    """
    Aggressive captain: Close range engagement, maximum firepower.

    Tactics:
    - Always accelerate toward the nearest enemy
    - Fire all weapons as soon as in range
    - Launch torpedoes at close range
    - Never retreat, even when damaged
    """

    def __init__(self) -> None:
        super().__init__("Aggressive Captain")

    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        commands = []
        ship = simulation.get_ship(ship_id)
        if not ship:
            return commands

        enemies = simulation.get_enemy_ships(ship_id)
        if not enemies:
            return commands

        # Target nearest enemy
        target = self._get_nearest_enemy(ship, enemies)
        if not target:
            return commands

        distance_km = ship.distance_to(target) / 1000

        # Always close in on target
        commands.append(self._create_attack_maneuver(
            target.ship_id, throttle=1.0, duration=30.0
        ))

        # Fire all ready weapons
        for slot, weapon_state in ship.weapons.items():
            if weapon_state.can_fire():
                # Check if weapon is in range
                if weapon_state.weapon.range_km >= distance_km:
                    commands.append(self._create_fire_command(slot, target.ship_id))

        # Launch torpedoes at medium range
        if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
            if distance_km < 500:  # Close enough for torpedo attack
                commands.append(self._create_torpedo_command(target.ship_id))

        return commands


class CautiousCaptain(CaptainBehavior):
    """
    Cautious captain: Long range engagement, preserve delta-v.

    Tactics:
    - Maintain distance from enemy
    - Fire only when hit probability is high
    - Conserve fuel for evasive maneuvers
    - Retreat when significantly damaged
    """

    def __init__(self) -> None:
        super().__init__("Cautious Captain")
        self.preferred_range_km = 600.0
        self.retreat_threshold = 40.0  # Retreat at 40% hull

    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        commands = []
        ship = simulation.get_ship(ship_id)
        if not ship:
            return commands

        enemies = simulation.get_enemy_ships(ship_id)
        if not enemies:
            return commands

        target = self._get_nearest_enemy(ship, enemies)
        if not target:
            return commands

        distance_km = ship.distance_to(target) / 1000

        # Check if we should retreat
        if ship.hull_integrity < self.retreat_threshold:
            # Retreat - burn away from enemy
            commands.append(Maneuver(
                maneuver_type=ManeuverType.BRAKE,
                start_time=0.0,
                duration=30.0,
                throttle=0.5,
                direction=Vector3D(
                    ship.position.x - target.position.x,
                    ship.position.y - target.position.y,
                    ship.position.z - target.position.z
                ).normalized()
            ))
            return commands

        # Manage range - use minimal throttle
        throttle = 0.3
        if distance_km < self.preferred_range_km * 0.8:
            # Too close - back off
            commands.append(Maneuver(
                maneuver_type=ManeuverType.BRAKE,
                start_time=0.0,
                duration=20.0,
                throttle=throttle
            ))
        elif distance_km > self.preferred_range_km * 1.2:
            # Too far - close in slowly
            commands.append(self._create_attack_maneuver(
                target.ship_id, throttle=throttle, duration=20.0
            ))
        # Else maintain current trajectory

        # Fire only at optimal range
        for slot, weapon_state in ship.weapons.items():
            if weapon_state.can_fire():
                # Only fire at 70-90% of max range for accuracy
                optimal_min = weapon_state.weapon.range_km * 0.5
                optimal_max = weapon_state.weapon.range_km * 0.9
                if optimal_min <= distance_km <= optimal_max:
                    commands.append(self._create_fire_command(slot, target.ship_id))

        # Launch torpedoes at medium-long range
        if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
            if 300 < distance_km < 800:
                commands.append(self._create_torpedo_command(target.ship_id))

        return commands


class EvasiveCaptain(CaptainBehavior):
    """
    Evasive captain: Focus on dodging, opportunistic fire.

    Tactics:
    - Constant evasive maneuvering
    - Fire only when enemy is distracted or close
    - Use ECM and point defense heavily
    - Survive at all costs
    """

    def __init__(self) -> None:
        super().__init__("Evasive Captain")
        self.evasion_cycles = 0

    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        commands = []
        ship = simulation.get_ship(ship_id)
        if not ship:
            return commands

        enemies = simulation.get_enemy_ships(ship_id)
        if not enemies:
            return commands

        target = self._get_nearest_enemy(ship, enemies)
        if not target:
            return commands

        distance_km = ship.distance_to(target) / 1000

        # Check for incoming threats
        incoming_torpedoes = [
            t for t in simulation.torpedoes
            if t.torpedo.target_id == ship_id
        ]

        # Evasive maneuvers priority
        if incoming_torpedoes:
            # High intensity evasion when torpedoes incoming
            commands.append(self._create_evasive_maneuver(throttle=1.0, duration=15.0))
        else:
            # Standard evasion pattern
            commands.append(self._create_evasive_maneuver(throttle=0.6, duration=10.0))

        self.evasion_cycles += 1

        # Opportunistic fire - only every other decision cycle
        if self.evasion_cycles % 2 == 0:
            for slot, weapon_state in ship.weapons.items():
                if weapon_state.can_fire():
                    if weapon_state.weapon.range_km >= distance_km:
                        commands.append(self._create_fire_command(slot, target.ship_id))

        # Torpedoes only at close range
        if ship.torpedo_launcher and ship.torpedo_launcher.can_launch(simulation.current_time):
            if distance_km < 200:  # Very close
                commands.append(self._create_torpedo_command(target.ship_id))

        return commands


class SnipeCaptain(CaptainBehavior):
    """
    Sniper captain: Maximum range engagement, precise shots.

    Tactics:
    - Stay at maximum weapon range
    - Orient ship to use spinal weapons
    - Fire only when aligned
    - Avoid close combat
    """

    def __init__(self) -> None:
        super().__init__("Snipe Captain")
        self.max_range_km = 800.0

    def decide(self, ship_id: str, simulation: CombatSimulation) -> List[Any]:
        commands = []
        ship = simulation.get_ship(ship_id)
        if not ship:
            return commands

        enemies = simulation.get_enemy_ships(ship_id)
        if not enemies:
            return commands

        target = self._get_nearest_enemy(ship, enemies)
        if not target:
            return commands

        distance_km = ship.distance_to(target) / 1000

        # Calculate bearing to target
        rel_pos = target.position - ship.position
        if rel_pos.magnitude > 0:
            target_dir = rel_pos.normalized()
            # Check alignment with forward direction
            alignment = ship.forward.dot(target_dir)
            is_aligned = alignment > 0.98  # Within ~11 degrees
        else:
            is_aligned = False

        # Stay at max range
        if distance_km < self.max_range_km * 0.8:
            # Too close - back off
            commands.append(Maneuver(
                maneuver_type=ManeuverType.BRAKE,
                start_time=0.0,
                duration=30.0,
                throttle=0.8,
                direction=(-target_dir) if rel_pos.magnitude > 0 else Vector3D.unit_x()
            ))
        elif distance_km > self.max_range_km * 1.1:
            # Too far - close in while maintaining orientation
            commands.append(Maneuver(
                maneuver_type=ManeuverType.ROTATE,
                start_time=0.0,
                duration=10.0,
                throttle=0.2,
                direction=target_dir if rel_pos.magnitude > 0 else Vector3D.unit_x()
            ))
        else:
            # At optimal range - orient to target
            commands.append(Maneuver(
                maneuver_type=ManeuverType.ROTATE,
                start_time=0.0,
                duration=15.0,
                throttle=0.0,
                direction=target_dir if rel_pos.magnitude > 0 else Vector3D.unit_x()
            ))

        # Fire spinal weapons when aligned
        if is_aligned:
            for slot, weapon_state in ship.weapons.items():
                if weapon_state.can_fire():
                    # Prioritize spinal (nose-only) weapons
                    if weapon_state.weapon.mount == "nose_only":
                        if weapon_state.weapon.range_km >= distance_km:
                            commands.append(self._create_fire_command(slot, target.ship_id))
                    elif distance_km < weapon_state.weapon.range_km * 0.7:
                        # Fire turrets at closer range
                        commands.append(self._create_fire_command(slot, target.ship_id))

        # No torpedoes - sniper relies on kinetic weapons
        return commands


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

def create_head_on_pass() -> ScenarioConfig:
    """
    Head-On Pass scenario: Two destroyers approach head-on at high speed.

    Tests: Firing solutions, damage exchange, pass-through maneuvers.
    Expected outcome: Roughly even, slight advantage to first-strike.
    """
    return ScenarioConfig(
        name="head_on_pass",
        display_name="Head-On Pass",
        description=(
            "Two destroyers approach head-on at high speed (10 km/s each). "
            "Start 1000 km apart with 20 km/s combined closing rate. "
            "Tests firing solutions during rapid approach and pass-through."
        ),
        ships=[
            ShipConfiguration(
                ship_id="alpha_1",
                ship_type="destroyer",
                faction="alpha",
                position_km=(-500, 0, 0),
                velocity_kps=(10, 0, 0),
                forward=(1, 0, 0)
            ),
            ShipConfiguration(
                ship_id="beta_1",
                ship_type="destroyer",
                faction="beta",
                position_km=(500, 0, 0),
                velocity_kps=(-10, 0, 0),
                forward=(-1, 0, 0)
            )
        ],
        decision_interval=20.0,  # Fast decisions needed
        time_limit_s=300.0,  # 5 minutes max
        victory_conditions=[VictoryCondition.DESTROY_ENEMY],
        special_rules={
            'no_retreat': True,
            'scenario_type': 'pass_through'
        },
        expected_outcome={
            'winner': None,  # Could go either way
            'win_rate_alpha': (0.40, 0.60),
            'draw_rate': (0.10, 0.30)
        }
    )


def create_tail_chase() -> ScenarioConfig:
    """
    Tail Chase scenario: Pursuer vs retreating target.

    Tests: Torpedo pursuit, delta-v management, sustained engagement.
    Expected outcome: Pursuer should win 60-80% of time.
    """
    return ScenarioConfig(
        name="tail_chase",
        display_name="Tail Chase",
        description=(
            "Target ship is 500 km ahead, both moving same direction. "
            "Pursuer is faster (+3 km/s) and must catch up. "
            "Tests torpedo pursuit and delta-v management over extended engagement."
        ),
        ships=[
            ShipConfiguration(
                ship_id="alpha_1",
                ship_type="destroyer",
                faction="alpha",
                position_km=(0, 0, 0),
                velocity_kps=(8, 0, 0),  # Faster pursuer
                forward=(1, 0, 0)
            ),
            ShipConfiguration(
                ship_id="beta_1",
                ship_type="destroyer",
                faction="beta",
                position_km=(500, 0, 0),  # 500 km ahead
                velocity_kps=(5, 0, 0),  # Slower target
                forward=(1, 0, 0)
            )
        ],
        decision_interval=30.0,
        time_limit_s=900.0,  # 15 minutes - longer chase
        victory_conditions=[VictoryCondition.DESTROY_ENEMY, VictoryCondition.ESCAPE_ENGAGEMENT],
        special_rules={
            'escape_distance_km': 2000,  # Target wins if opens to 2000 km
            'scenario_type': 'chase'
        },
        expected_outcome={
            'winner': 'alpha',
            'win_rate_alpha': (0.60, 0.80),
            'escape_rate_beta': (0.10, 0.25)
        }
    )


def create_flanking_attack() -> ScenarioConfig:
    """
    Flanking Attack scenario: Attacker approaches from the side.

    Tests: Lead calculation, lateral armor hits, crossing-T engagement.
    Expected outcome: Flanker has moderate advantage (55-70%).
    """
    return ScenarioConfig(
        name="flanking_attack",
        display_name="Flanking Attack",
        description=(
            "Target is moving perpendicular to attacker's approach. "
            "Attacker must calculate lead and engage lateral armor. "
            "Classic crossing-T naval engagement geometry."
        ),
        ships=[
            ShipConfiguration(
                ship_id="alpha_1",
                ship_type="destroyer",
                faction="alpha",
                position_km=(0, -400, 0),  # South of target
                velocity_kps=(0, 6, 0),  # Moving north
                forward=(0, 1, 0)
            ),
            ShipConfiguration(
                ship_id="beta_1",
                ship_type="destroyer",
                faction="beta",
                position_km=(0, 0, 0),  # At center
                velocity_kps=(5, 0, 0),  # Moving east
                forward=(1, 0, 0)
            )
        ],
        decision_interval=25.0,
        time_limit_s=480.0,  # 8 minutes
        victory_conditions=[VictoryCondition.DESTROY_ENEMY],
        special_rules={
            'scenario_type': 'flanking'
        },
        expected_outcome={
            'winner': 'alpha',
            'win_rate_alpha': (0.55, 0.70)
        }
    )


def create_ambush_scenario() -> ScenarioConfig:
    """
    Ambush Scenario: Stationary target, fast attacker.

    Tests: Alpha strike tactics, initial engagement advantage.
    Expected outcome: Attacker should win decisively (70-90%).
    """
    return ScenarioConfig(
        name="ambush",
        display_name="Ambush Strike",
        description=(
            "Target is stationary (engine trouble, docking, etc). "
            "Attacker approaches at 15 km/s for alpha strike. "
            "Tests maximum firepower on first pass."
        ),
        ships=[
            ShipConfiguration(
                ship_id="alpha_1",
                ship_type="destroyer",
                faction="alpha",
                position_km=(-600, 0, 0),
                velocity_kps=(15, 0, 0),  # Fast approach
                forward=(1, 0, 0)
            ),
            ShipConfiguration(
                ship_id="beta_1",
                ship_type="destroyer",
                faction="beta",
                position_km=(0, 0, 0),
                velocity_kps=(0, 0, 0),  # Stationary
                forward=(1, 0, 0)  # Facing wrong way initially
            )
        ],
        decision_interval=20.0,  # Fast decisions
        time_limit_s=240.0,  # 4 minutes
        victory_conditions=[VictoryCondition.DESTROY_ENEMY],
        special_rules={
            'beta_delayed_start': 10.0,  # Beta takes 10s to react
            'scenario_type': 'ambush'
        },
        expected_outcome={
            'winner': 'alpha',
            'win_rate_alpha': (0.70, 0.90)
        }
    )


def create_dogfight_duel() -> ScenarioConfig:
    """
    Dogfight Duel: Close range maneuvering combat.

    Tests: Rotation, weapon arcs, heat management.
    Expected outcome: Even matchup, skill determines winner.
    """
    return ScenarioConfig(
        name="dogfight_duel",
        display_name="Dogfight Duel",
        description=(
            "Start at 50 km with low relative velocity. "
            "Close-quarters maneuvering combat. "
            "Tests rotation speed, weapon arcs, and heat management."
        ),
        ships=[
            ShipConfiguration(
                ship_id="alpha_1",
                ship_type="destroyer",
                faction="alpha",
                position_km=(25, 0, 0),
                velocity_kps=(1, 0.5, 0),  # Slow drift
                forward=(0, -1, 0)  # Facing each other
            ),
            ShipConfiguration(
                ship_id="beta_1",
                ship_type="destroyer",
                faction="beta",
                position_km=(-25, 0, 0),
                velocity_kps=(-0.5, 0.5, 0),  # Slow drift
                forward=(0, 1, 0)  # Facing each other
            )
        ],
        decision_interval=20.0,  # Quick reactions needed
        time_limit_s=420.0,  # 7 minutes
        victory_conditions=[VictoryCondition.DESTROY_ENEMY],
        special_rules={
            'radiators_vulnerable': True,  # Heat management critical
            'scenario_type': 'dogfight'
        },
        expected_outcome={
            'winner': None,  # Skill dependent
            'win_rate_alpha': (0.45, 0.55),
            'draw_rate': (0.10, 0.20)
        }
    )


def create_long_range_standoff() -> ScenarioConfig:
    """
    Long Range Standoff: Maximum range engagement.

    Tests: Accuracy at range, evasion time, patience.
    Expected outcome: Drawn out engagement, moderate damage.
    """
    return ScenarioConfig(
        name="long_range_standoff",
        display_name="Long Range Standoff",
        description=(
            "Start at weapon range limit (850 km). "
            "Both ships have time to evade incoming fire. "
            "Tests accuracy at extreme range and evasion tactics."
        ),
        ships=[
            ShipConfiguration(
                ship_id="alpha_1",
                ship_type="destroyer",
                faction="alpha",
                position_km=(-425, 0, 0),
                velocity_kps=(0, 0, 0),
                forward=(1, 0, 0)
            ),
            ShipConfiguration(
                ship_id="beta_1",
                ship_type="destroyer",
                faction="beta",
                position_km=(425, 0, 0),
                velocity_kps=(0, 0, 0),
                forward=(-1, 0, 0)
            )
        ],
        decision_interval=40.0,  # More time for deliberation
        time_limit_s=600.0,  # 10 minutes
        victory_conditions=[VictoryCondition.DESTROY_ENEMY, VictoryCondition.SURVIVE_TIME_LIMIT],
        special_rules={
            'engagement_range_km': 850,
            'scenario_type': 'standoff'
        },
        expected_outcome={
            'winner': None,  # Often ends in time limit
            'win_rate_alpha': (0.35, 0.50),
            'time_limit_rate': (0.20, 0.40)
        }
    )


def create_missile_exchange() -> ScenarioConfig:
    """
    Missile Exchange: Torpedo-focused combat.

    Tests: Torpedo tracking, point defense, evasive maneuvering.
    Expected outcome: High damage to both sides.
    """
    return ScenarioConfig(
        name="missile_exchange",
        display_name="Torpedo Exchange",
        description=(
            "Both ships have full torpedo magazines. "
            "Initial exchange of torpedoes, then evasion. "
            "Tests torpedo tracking and point defense systems."
        ),
        ships=[
            ShipConfiguration(
                ship_id="alpha_1",
                ship_type="destroyer",
                faction="alpha",
                position_km=(-350, 0, 0),
                velocity_kps=(2, 0, 0),
                forward=(1, 0, 0)
            ),
            ShipConfiguration(
                ship_id="beta_1",
                ship_type="destroyer",
                faction="beta",
                position_km=(350, 0, 0),
                velocity_kps=(-2, 0, 0),
                forward=(-1, 0, 0)
            )
        ],
        decision_interval=25.0,
        time_limit_s=480.0,  # 8 minutes
        victory_conditions=[VictoryCondition.DESTROY_ENEMY],
        special_rules={
            'torpedo_priority': True,
            'ammo_limited': False,  # Full magazines
            'scenario_type': 'missile_exchange'
        },
        expected_outcome={
            'winner': None,  # High mutual damage
            'mutual_destruction_rate': (0.20, 0.40),
            'survivor_damaged_rate': (0.70, 0.90)
        }
    )


def create_damaged_ship_fight() -> ScenarioConfig:
    """
    Damaged Ship Fight: One ship starts significantly damaged.

    Tests: Fighting while damaged, tactical disengagement decisions.
    Expected outcome: Healthy ship wins 65-80%.
    """
    return ScenarioConfig(
        name="damaged_ship_fight",
        display_name="Damaged Ship Fight",
        description=(
            "One ship starts at 50% armor with damaged modules. "
            "Damaged ship must decide: fight or disengage? "
            "Tests combat effectiveness under damage and disengagement tactics."
        ),
        ships=[
            ShipConfiguration(
                ship_id="alpha_1",
                ship_type="destroyer",
                faction="alpha",
                position_km=(-200, 0, 0),
                velocity_kps=(4, 0, 0),
                forward=(1, 0, 0)
                # Healthy ship
            ),
            ShipConfiguration(
                ship_id="beta_1",
                ship_type="destroyer",
                faction="beta",
                position_km=(200, 0, 0),
                velocity_kps=(-2, 0, 0),
                forward=(-1, 0, 0),
                damage_preset={
                    'armor_damage_percent': 50,
                    'modules_damaged': ['Primary Sensor Array', 'Coilgun Battery A'],
                    'fuel_used_percent': 20
                }
            )
        ],
        decision_interval=25.0,
        time_limit_s=480.0,  # 8 minutes
        victory_conditions=[VictoryCondition.DESTROY_ENEMY, VictoryCondition.ESCAPE_ENGAGEMENT],
        special_rules={
            'escape_distance_km': 1500,
            'scenario_type': 'damaged_fight'
        },
        expected_outcome={
            'winner': 'alpha',
            'win_rate_alpha': (0.65, 0.80),
            'escape_rate_beta': (0.10, 0.20)
        }
    )


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

SCENARIO_REGISTRY: Dict[str, Callable[[], ScenarioConfig]] = {
    'head_on_pass': create_head_on_pass,
    'tail_chase': create_tail_chase,
    'flanking_attack': create_flanking_attack,
    'ambush': create_ambush_scenario,
    'dogfight_duel': create_dogfight_duel,
    'long_range_standoff': create_long_range_standoff,
    'missile_exchange': create_missile_exchange,
    'damaged_ship_fight': create_damaged_ship_fight,
}


# =============================================================================
# SCENARIO RUNNER
# =============================================================================

class ScenarioRunner:
    """
    Runner for executing combat scenarios.

    Handles scenario setup, execution, and result collection.

    Usage:
        runner = ScenarioRunner()
        result = runner.run_scenario("head_on_pass", captain_alpha, captain_beta)
    """

    def __init__(self, seed: Optional[int] = None, fleet_data: Optional[Dict] = None) -> None:
        """
        Initialize the scenario runner.

        Args:
            seed: Random seed for reproducibility.
            fleet_data: Fleet data dictionary (loads default if None).
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.fleet_data = fleet_data or self._load_default_fleet_data()

    def _load_default_fleet_data(self) -> Dict:
        """Load default fleet data from file."""
        import json
        from pathlib import Path

        data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
        if data_path.exists():
            with open(data_path, 'r') as f:
                return json.load(f)
        else:
            # Return minimal default data
            return {
                'ships': {
                    'destroyer': {
                        'hull': {'length_m': 100},
                        'mass': {'wet_mass_tons': 2500, 'dry_mass_tons': 2375},
                        'propulsion': {
                            'drive': {'thrust_mn': 58.56, 'exhaust_velocity_kps': 10256}
                        },
                        'thermal': {
                            'heatsink': {'capacity_gj': 525},
                            'radiator': {'mass_tons': 10, 'dissipation_kw_per_kg': 13}
                        },
                        'armor': {
                            'type': 'Titanium',
                            'properties': {'baryonic_half_cm': 10.5, 'chip_resist': 0.0},
                            'sections': {
                                'nose': {'thickness_cm': 30},
                                'lateral': {'thickness_cm': 20},
                                'tail': {'thickness_cm': 15}
                            }
                        }
                    }
                },
                'weapon_types': {
                    'spinal_coiler_mk3': {
                        'name': 'Spinal Coiler Mk3',
                        'kinetic_energy_gj': 15,
                        'cooldown_s': 10,
                        'range_km': 900,
                        'flat_chipping': 0.8,
                        'magazine': 50,
                        'muzzle_velocity_kps': 10,
                        'warhead_mass_kg': 25
                    }
                }
            }

    def list_scenarios(self) -> List[str]:
        """Get list of all available scenario names."""
        return list(SCENARIO_REGISTRY.keys())

    def get_scenario_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific scenario."""
        if name not in SCENARIO_REGISTRY:
            return None

        config = SCENARIO_REGISTRY[name]()
        return {
            'name': config.name,
            'display_name': config.display_name,
            'description': config.description,
            'decision_interval': config.decision_interval,
            'time_limit_s': config.time_limit_s,
            'ship_count': len(config.ships),
            'expected_outcome': config.expected_outcome
        }

    def create_scenario(self, name: str) -> Optional[ScenarioConfig]:
        """
        Create a scenario configuration by name.

        Args:
            name: Scenario name from registry.

        Returns:
            ScenarioConfig or None if not found.
        """
        if name not in SCENARIO_REGISTRY:
            return None
        return SCENARIO_REGISTRY[name]()

    def run_scenario(
        self,
        name: str,
        captain_a: CaptainBehavior,
        captain_b: CaptainBehavior,
        verbose: bool = False
    ) -> Optional[ScenarioResult]:
        """
        Execute a scenario with the given AI captains.

        Args:
            name: Scenario name to run.
            captain_a: Captain controlling alpha faction ships.
            captain_b: Captain controlling beta faction ships.
            verbose: If True, print progress during simulation.

        Returns:
            ScenarioResult with battle outcome, or None if scenario not found.
        """
        config = self.create_scenario(name)
        if not config:
            return None

        # Create simulation
        sim = CombatSimulation(
            time_step=1.0,
            decision_interval=config.decision_interval,
            seed=self.seed
        )

        # Add ships
        self._setup_ships(sim, config)

        # Set up decision callback
        def decision_callback(ship_id: str, simulation: CombatSimulation) -> List[Any]:
            ship = simulation.get_ship(ship_id)
            if not ship:
                return []

            if ship.faction == "alpha":
                return captain_a.decide(ship_id, simulation)
            else:
                return captain_b.decide(ship_id, simulation)

        sim.set_decision_callback(decision_callback)

        # Run simulation
        if verbose:
            print(f"Running scenario: {config.display_name}")
            print(f"  Alpha captain: {captain_a.name}")
            print(f"  Beta captain: {captain_b.name}")
            print(f"  Time limit: {config.time_limit_s}s")

        sim.run(config.time_limit_s)

        # Determine outcome
        result = self._determine_outcome(sim, config)

        if verbose:
            print(f"  Outcome: {result.outcome.value}")
            print(f"  Duration: {result.duration_s:.1f}s")
            if result.winning_faction:
                print(f"  Winner: {result.winning_faction}")

        return result

    def _setup_ships(self, sim: CombatSimulation, config: ScenarioConfig) -> None:
        """Set up ships in the simulation from scenario config."""
        for ship_config in config.ships:
            position, velocity, forward = ship_config.to_vectors()

            # Create kinematic state
            ship_data = self.fleet_data.get('ships', {}).get(ship_config.ship_type, {})
            mass_data = ship_data.get('mass', {})
            propulsion = ship_data.get('propulsion', {})
            drive = propulsion.get('drive', {})
            hull = ship_data.get('hull', {})

            kinematic_state = create_ship_state_from_specs(
                wet_mass_tons=mass_data.get('wet_mass_tons', 2500),
                dry_mass_tons=mass_data.get('dry_mass_tons', 2375),
                length_m=hull.get('length_m', 100),
                thrust_mn=drive.get('thrust_mn', 58.56),
                exhaust_velocity_kps=drive.get('exhaust_velocity_kps', 10256),
                position=position,
                velocity=velocity,
                forward=forward
            )

            # Adjust propellant for fuel_fraction
            if ship_config.fuel_fraction < 1.0:
                full_propellant = kinematic_state.propellant_kg
                kinematic_state.propellant_kg = full_propellant * ship_config.fuel_fraction
                kinematic_state.mass_kg = kinematic_state.dry_mass_kg + kinematic_state.propellant_kg

            # Create combat state
            ship = ShipCombatState(
                ship_id=ship_config.ship_id,
                ship_type=ship_config.ship_type,
                faction=ship_config.faction,
                kinematic_state=kinematic_state
            )

            # Add thermal system
            try:
                ship.thermal_system = ThermalSystem.from_ship_data(ship_data)
            except Exception:
                pass

            # Add armor
            try:
                ship.armor = create_ship_armor_from_fleet_data(self.fleet_data, ship_config.ship_type)
            except Exception:
                pass

            # Add modules
            try:
                ship.module_layout = ModuleLayout.from_ship_type(ship_config.ship_type, self.fleet_data)
            except Exception:
                pass

            # Add weapons
            weapon_types = self.fleet_data.get('weapon_types', {})
            for wtype, wdata in weapon_types.items():
                try:
                    weapon = create_weapon_from_fleet_data(self.fleet_data, wtype)
                    ammo = int(weapon.magazine * ship_config.ammo_multiplier)
                    ship.weapons[wtype] = WeaponState(weapon=weapon, ammo_remaining=ammo)
                except Exception:
                    pass

            # Add torpedo launcher
            torpedo_data = ship_data.get('torpedo', {})
            if torpedo_data:
                try:
                    specs = TorpedoSpecs.from_fleet_data(
                        warhead_yield_gj=torpedo_data.get('warhead_yield_gj', 0),  # Pure kinetic penetrator
                        penetrator_mass_kg=torpedo_data.get('penetrator_mass_kg', 100),  # 100 kg penetrator
                        ammo_mass_kg=torpedo_data.get('ammo_mass_kg', 1600)
                    )
                    ship.torpedo_launcher = TorpedoLauncher(
                        specs=specs,
                        magazine_capacity=torpedo_data.get('magazine', 16),
                        current_magazine=torpedo_data.get('magazine', 16),
                        cooldown_seconds=torpedo_data.get('cooldown_s', 30)
                    )
                except Exception:
                    pass

            # Apply damage preset if specified
            if ship_config.damage_preset:
                self._apply_damage_preset(ship, ship_config.damage_preset)

            sim.add_ship(ship)

    def _apply_damage_preset(self, ship: ShipCombatState, damage: Dict[str, Any]) -> None:
        """Apply pre-configured damage to a ship."""
        # Armor damage
        armor_damage_pct = damage.get('armor_damage_percent', 0)
        if armor_damage_pct > 0 and ship.armor:
            for section in ship.armor.sections.values():
                section.thickness_cm *= (1 - armor_damage_pct / 100)

        # Module damage
        damaged_modules = damage.get('modules_damaged', [])
        if damaged_modules and ship.module_layout:
            for module in ship.module_layout.get_all_modules():
                if module.name in damaged_modules:
                    module.health_percent = 25.0  # Damaged but functional

        # Fuel used
        fuel_used_pct = damage.get('fuel_used_percent', 0)
        if fuel_used_pct > 0:
            current_prop = ship.kinematic_state.propellant_kg
            ship.kinematic_state.propellant_kg = current_prop * (1 - fuel_used_pct / 100)
            ship.kinematic_state.mass_kg = (
                ship.kinematic_state.dry_mass_kg + ship.kinematic_state.propellant_kg
            )

    def _determine_outcome(
        self,
        sim: CombatSimulation,
        config: ScenarioConfig
    ) -> ScenarioResult:
        """Determine the outcome of a completed simulation."""
        alpha_alive = [s for s in sim.ships.values() if s.faction == 'alpha' and not s.is_destroyed]
        beta_alive = [s for s in sim.ships.values() if s.faction == 'beta' and not s.is_destroyed]

        alpha_count = len(alpha_alive)
        beta_count = len(beta_alive)

        # Determine outcome
        if alpha_count > 0 and beta_count == 0:
            outcome = ScenarioOutcome.ALPHA_VICTORY
            winner = 'alpha'
        elif beta_count > 0 and alpha_count == 0:
            outcome = ScenarioOutcome.BETA_VICTORY
            winner = 'beta'
        elif alpha_count == 0 and beta_count == 0:
            outcome = ScenarioOutcome.MUTUAL_DESTRUCTION
            winner = None
        elif sim.current_time >= config.time_limit_s:
            # Time limit - check for partial victory conditions
            if alpha_count > beta_count:
                outcome = ScenarioOutcome.ALPHA_VICTORY
                winner = 'alpha'
            elif beta_count > alpha_count:
                outcome = ScenarioOutcome.BETA_VICTORY
                winner = 'beta'
            else:
                outcome = ScenarioOutcome.TIME_LIMIT
                winner = None
        else:
            outcome = ScenarioOutcome.DRAW
            winner = None

        return ScenarioResult(
            scenario_name=config.name,
            outcome=outcome,
            winning_faction=winner,
            duration_s=sim.current_time,
            alpha_ships_remaining=alpha_count,
            beta_ships_remaining=beta_count,
            total_damage_dealt=sim.metrics.total_damage_dealt,
            events=sim.events,
            metrics={
                'total_shots': sim.metrics.total_shots_fired,
                'total_hits': sim.metrics.total_hits,
                'hit_rate': sim.metrics.hit_rate,
                'torpedoes_launched': sim.metrics.total_torpedoes_launched,
                'torpedo_hits': sim.metrics.total_torpedo_hits
            }
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_quick_scenario(
    scenario_name: str,
    captain_a: Optional[CaptainBehavior] = None,
    captain_b: Optional[CaptainBehavior] = None,
    seed: Optional[int] = None
) -> Optional[ScenarioResult]:
    """
    Quickly run a scenario with default or provided captains.

    Args:
        scenario_name: Name of scenario to run.
        captain_a: Alpha faction captain (defaults to Aggressive).
        captain_b: Beta faction captain (defaults to Cautious).
        seed: Random seed.

    Returns:
        ScenarioResult or None if scenario not found.
    """
    if captain_a is None:
        captain_a = AggressiveCaptain()
    if captain_b is None:
        captain_b = CautiousCaptain()

    runner = ScenarioRunner(seed=seed)
    return runner.run_scenario(scenario_name, captain_a, captain_b, verbose=True)


def run_matchup_test(
    scenario_name: str,
    captain_a: CaptainBehavior,
    captain_b: CaptainBehavior,
    iterations: int = 10,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run multiple iterations of a scenario to test captain matchup.

    Args:
        scenario_name: Scenario to test.
        captain_a: Alpha faction captain.
        captain_b: Beta faction captain.
        iterations: Number of times to run the scenario.
        seed: Base random seed.

    Returns:
        Dictionary with aggregated results.
    """
    results = {
        'scenario': scenario_name,
        'captain_a': captain_a.name,
        'captain_b': captain_b.name,
        'iterations': iterations,
        'alpha_wins': 0,
        'beta_wins': 0,
        'draws': 0,
        'mutual_destruction': 0,
        'time_limits': 0,
        'avg_duration_s': 0.0
    }

    total_duration = 0.0

    for i in range(iterations):
        iter_seed = (seed + i) if seed else None
        runner = ScenarioRunner(seed=iter_seed)
        result = runner.run_scenario(scenario_name, captain_a, captain_b)

        if result:
            total_duration += result.duration_s

            if result.outcome == ScenarioOutcome.ALPHA_VICTORY:
                results['alpha_wins'] += 1
            elif result.outcome == ScenarioOutcome.BETA_VICTORY:
                results['beta_wins'] += 1
            elif result.outcome == ScenarioOutcome.MUTUAL_DESTRUCTION:
                results['mutual_destruction'] += 1
            elif result.outcome == ScenarioOutcome.TIME_LIMIT:
                results['time_limits'] += 1
            else:
                results['draws'] += 1

    results['avg_duration_s'] = total_duration / iterations if iterations > 0 else 0
    results['alpha_win_rate'] = results['alpha_wins'] / iterations if iterations > 0 else 0
    results['beta_win_rate'] = results['beta_wins'] / iterations if iterations > 0 else 0

    return results


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS SCENARIOS MODULE - SELF TEST")
    print("=" * 70)

    # Check for import errors
    if _IMPORT_ERRORS:
        print("\n--- Import Warnings ---")
        for err in _IMPORT_ERRORS:
            print(f"  Warning: {err}")

    # List all scenarios
    runner = ScenarioRunner(seed=42)
    print("\n--- Available Scenarios ---")
    for name in runner.list_scenarios():
        info = runner.get_scenario_info(name)
        if info:
            print(f"\n{info['display_name']} ({name})")
            print(f"  {info['description'][:80]}...")
            print(f"  Time limit: {info['time_limit_s']}s, Decision interval: {info['decision_interval']}s")

    # Test captain behaviors
    print("\n--- Captain Behaviors ---")
    captains = [
        AggressiveCaptain(),
        CautiousCaptain(),
        EvasiveCaptain(),
        SnipeCaptain()
    ]
    for captain in captains:
        print(f"  - {captain.name}")

    # Test scenario creation
    print("\n--- Scenario Creation Test ---")
    for scenario_name in runner.list_scenarios():
        config = runner.create_scenario(scenario_name)
        if config:
            print(f"  {scenario_name}: {len(config.ships)} ships, "
                  f"victory={[v.name for v in config.victory_conditions]}")

    # Run simulation tests only if simulation is available
    if _SIMULATION_AVAILABLE:
        print("\n--- Quick Scenario Test ---")
        result = run_quick_scenario(
            "head_on_pass",
            captain_a=AggressiveCaptain(),
            captain_b=CautiousCaptain(),
            seed=42
        )

        if result:
            print(f"\nScenario: {result.scenario_name}")
            print(f"Outcome: {result.outcome.value}")
            print(f"Duration: {result.duration_s:.1f}s")
            print(f"Alpha ships remaining: {result.alpha_ships_remaining}")
            print(f"Beta ships remaining: {result.beta_ships_remaining}")
            print(f"Total damage: {result.total_damage_dealt:.2f} GJ")
            print(f"Events logged: {len(result.events)}")

        # Run a matchup test
        print("\n--- Matchup Test (3 iterations) ---")
        matchup = run_matchup_test(
            "ambush",
            AggressiveCaptain(),
            EvasiveCaptain(),
            iterations=3,
            seed=42
        )

        print(f"Scenario: {matchup['scenario']}")
        print(f"Alpha ({matchup['captain_a']}) wins: {matchup['alpha_wins']}/{matchup['iterations']}")
        print(f"Beta ({matchup['captain_b']}) wins: {matchup['beta_wins']}/{matchup['iterations']}")
        print(f"Alpha win rate: {matchup['alpha_win_rate']*100:.0f}%")
        print(f"Average duration: {matchup['avg_duration_s']:.1f}s")
    else:
        print("\n--- Simulation Test Skipped ---")
        print("  CombatSimulation module not available.")
        print("  Scenario definitions and captain behaviors validated successfully.")
        print("  To run full simulation tests, ensure all dependencies are importable.")

    print("\n" + "=" * 70)
    print("Scenario module tests completed!")
    print("=" * 70)
