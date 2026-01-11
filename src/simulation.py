#!/usr/bin/env python3
"""
Combat Simulation Engine for AI Commanders Space Battle Simulator.

This module implements the core combat simulation loop that:
- Runs in discrete time steps (configurable, default 1 second)
- Tracks multiple ships with full state (position, velocity, orientation, heat, delta-v)
- Tracks all projectiles (coilgun slugs, torpedoes) in flight
- Updates physics, thermal, and weapon states each tick
- Supports "decision points" where an LLM would act

The simulation produces a comprehensive event log for analysis and replay.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Any

# Import from existing modules using try/except for compatibility
try:
    from .physics import Vector3D, ShipState as KinematicState, propagate_state, create_ship_state_from_specs
    from .thermal import (
        ThermalSystem, RadiatorArray, HeatSink, RadiatorState,
        RadiatorPosition, HEAT_GENERATION_RATES
    )
    from .projectile import KineticProjectile, ProjectileLauncher
    from .torpedo import (
        Torpedo, TorpedoLauncher, TorpedoSpecs, GuidanceMode,
        TorpedoGuidance, GuidanceCommand, SAFE_ARMING_DISTANCE_M
    )
    from .combat import (
        CombatResolver, Weapon, ShipArmor, HitResult, HitLocation,
        create_weapon_from_fleet_data, create_ship_armor_from_fleet_data
    )
    from .geometry import ShipGeometry, calculate_hit_probability_modifier, create_geometry_from_fleet_data
    from .modules import ModuleLayout, Module, ModuleType
    from .damage import DamagePropagator, DamageCone
    from .pointdefense import PDLaser, PDEngagement, EngagementOutcome
    from .firecontrol import (
        calculate_hit_probability, FiringSolution,
        HelmCommand, WeaponsCommand, TacticalPosture,
        HelmOrder, WeaponsOrder, TacticalOrder, WeaponsOfficer
    )
    from .power import PowerSystem, WeaponCapacitor, Battery, Reactor
except ImportError:
    from physics import Vector3D, ShipState as KinematicState, propagate_state, create_ship_state_from_specs
    from thermal import (
        ThermalSystem, RadiatorArray, HeatSink, RadiatorState,
        RadiatorPosition, HEAT_GENERATION_RATES
    )
    from projectile import KineticProjectile, ProjectileLauncher
    from torpedo import (
        Torpedo, TorpedoLauncher, TorpedoSpecs, GuidanceMode,
        TorpedoGuidance, GuidanceCommand, SAFE_ARMING_DISTANCE_M
    )
    from combat import (
        CombatResolver, Weapon, ShipArmor, HitResult, HitLocation,
        create_weapon_from_fleet_data, create_ship_armor_from_fleet_data
    )
    from geometry import ShipGeometry, calculate_hit_probability_modifier, create_geometry_from_fleet_data
    from modules import ModuleLayout, Module, ModuleType
    from damage import DamagePropagator, DamageCone
    from pointdefense import PDLaser, PDEngagement, EngagementOutcome
    from firecontrol import (
        calculate_hit_probability, FiringSolution,
        HelmCommand, WeaponsCommand, TacticalPosture,
        HelmOrder, WeaponsOrder, TacticalOrder, WeaponsOfficer
    )
    from power import PowerSystem, WeaponCapacitor, Battery, Reactor


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TIME_STEP = 1.0  # seconds
DEFAULT_DECISION_INTERVAL = 30.0  # seconds between LLM decision points
MIN_DECISION_INTERVAL = 20.0
MAX_DECISION_INTERVAL = 60.0

# Hit detection radius in meters
HIT_DETECTION_RADIUS = 50.0

# Thermal heat generation per coilgun shot (GJ)
COILGUN_HEAT_PER_SHOT_GJ = 0.5


# =============================================================================
# EVENT TYPES
# =============================================================================

class SimulationEventType(Enum):
    """Types of events that can occur during simulation."""
    # Projectile events
    PROJECTILE_LAUNCHED = auto()
    PROJECTILE_IMPACT = auto()
    PROJECTILE_MISS = auto()
    TORPEDO_LAUNCHED = auto()
    TORPEDO_IMPACT = auto()
    TORPEDO_INTERCEPTED = auto()
    TORPEDO_FUEL_EXHAUSTED = auto()

    # Maneuver events
    MANEUVER_STARTED = auto()
    MANEUVER_COMPLETED = auto()

    # Damage events
    DAMAGE_TAKEN = auto()
    MODULE_DAMAGED = auto()
    MODULE_DESTROYED = auto()
    SHIP_DESTROYED = auto()
    ARMOR_PENETRATED = auto()

    # Thermal events
    THERMAL_WARNING = auto()
    THERMAL_CRITICAL = auto()
    RADIATOR_EXTENDED = auto()
    RADIATOR_RETRACTED = auto()
    RADIATOR_DAMAGED = auto()

    # Point defense events
    PD_ENGAGED = auto()
    PD_TORPEDO_DISABLED = auto()
    PD_TORPEDO_DESTROYED = auto()
    PD_SLUG_DAMAGED = auto()
    PD_SLUG_DESTROYED = auto()

    # Control events
    DECISION_POINT_REACHED = auto()
    COMMAND_ISSUED = auto()

    # Battle flow events
    SIMULATION_STARTED = auto()
    SIMULATION_ENDED = auto()


# =============================================================================
# SIMULATION EVENT
# =============================================================================

@dataclass
class SimulationEvent:
    """
    An event that occurs during simulation.

    Events are recorded for analysis and can be used to reconstruct
    the battle timeline.

    Attributes:
        event_type: The type of event.
        timestamp: Simulation time when event occurred (seconds).
        ship_id: ID of the ship involved (if applicable).
        target_id: ID of the target (if applicable).
        data: Additional event-specific data.
    """
    event_type: SimulationEventType
    timestamp: float
    ship_id: Optional[str] = None
    target_id: Optional[str] = None
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        ship_str = f"[{self.ship_id}]" if self.ship_id else ""
        target_str = f" -> {self.target_id}" if self.target_id else ""
        return f"T+{self.timestamp:.1f}s {ship_str} {self.event_type.name}{target_str}"


# =============================================================================
# MANEUVER TYPES
# =============================================================================

class ManeuverType(Enum):
    """Types of maneuvers a ship can execute."""
    BURN = auto()           # Constant thrust burn
    ROTATE = auto()         # Rotate to new heading
    EVASIVE = auto()        # Random evasive pattern
    INTERCEPT = auto()      # Intercept course toward target
    BRAKE = auto()          # Deceleration burn
    MAINTAIN = auto()       # Hold current course (coast, no thrust)


@dataclass
class Maneuver:
    """
    A maneuver being executed by a ship.

    Attributes:
        maneuver_type: Type of maneuver.
        start_time: Simulation time when maneuver started.
        duration: Total duration in seconds (0 = indefinite).
        throttle: Throttle setting (0.0 to 1.0).
        direction: Thrust or rotation direction.
        target_id: Target ship ID for intercept maneuvers.
    """
    maneuver_type: ManeuverType
    start_time: float
    duration: float = 0.0
    throttle: float = 1.0
    direction: Optional[Vector3D] = None
    target_id: Optional[str] = None

    def is_complete(self, current_time: float) -> bool:
        """Check if maneuver has completed."""
        if self.duration <= 0:
            return False
        return current_time >= self.start_time + self.duration


# =============================================================================
# ATTITUDE CONTROL
# =============================================================================

@dataclass
class AttitudeControlSpecs:
    """
    Attitude control specifications for a ship.

    Stores angular acceleration capabilities from thrust vectoring and RCS,
    used for realistic bang-bang rotation control.

    Attributes:
        moment_of_inertia_kg_m2: Ship's moment of inertia (kg*m^2)
        tv_angular_accel_deg_s2: Angular acceleration from thrust vectoring (deg/s^2)
        tv_max_angular_vel_deg_s: Max angular velocity with thrust vectoring (deg/s)
        rcs_angular_accel_deg_s2: Angular acceleration from RCS only (deg/s^2)
        rcs_max_angular_vel_deg_s: Max angular velocity with RCS only (deg/s)
    """
    moment_of_inertia_kg_m2: float = 700_645_833  # Default: corvette
    tv_angular_accel_deg_s2: float = 2.445  # Thrust vectoring (requires engines)
    tv_max_angular_vel_deg_s: float = 14.83
    rcs_angular_accel_deg_s2: float = 0.1227  # RCS only (engines off)
    rcs_max_angular_vel_deg_s: float = 3.32

    def get_angular_accel(self, engines_on: bool) -> float:
        """Get available angular acceleration in rad/s^2."""
        if engines_on:
            # TV + RCS combined
            return math.radians(self.tv_angular_accel_deg_s2 + self.rcs_angular_accel_deg_s2)
        else:
            # RCS only
            return math.radians(self.rcs_angular_accel_deg_s2)

    def get_max_angular_vel(self, engines_on: bool) -> float:
        """Get max angular velocity in rad/s."""
        if engines_on:
            return math.radians(self.tv_max_angular_vel_deg_s)
        else:
            return math.radians(self.rcs_max_angular_vel_deg_s)


@dataclass
class RotationState:
    """
    Tracks the state of an ongoing rotation maneuver.

    Implements bang-bang control: accelerate to midpoint, then decelerate.

    Attributes:
        target_direction: The direction we're rotating toward
        initial_direction: Direction when rotation started
        total_angle_rad: Total angle to rotate
        current_angular_vel_rad_s: Current angular velocity (rad/s)
        rotation_axis: Axis of rotation (normalized)
        phase: 'accelerate', 'decelerate', or 'complete'
    """
    target_direction: Vector3D
    initial_direction: Vector3D
    total_angle_rad: float
    current_angular_vel_rad_s: float = 0.0
    rotation_axis: Vector3D = field(default_factory=Vector3D.zero)
    phase: str = "accelerate"

    def angle_remaining(self, current_forward: Vector3D) -> float:
        """Calculate remaining angle to target in radians."""
        return current_forward.angle_to(self.target_direction)

    def is_complete(self, current_forward: Vector3D, tolerance_rad: float = 0.01) -> bool:
        """Check if rotation is complete."""
        return (self.angle_remaining(current_forward) < tolerance_rad and
                abs(self.current_angular_vel_rad_s) < 0.01)


# =============================================================================
# WEAPON STATE
# =============================================================================

@dataclass
class WeaponState:
    """
    State of a weapon system on a ship.

    Attributes:
        weapon: The weapon specification.
        ammo_remaining: Current ammunition count.
        cooldown_remaining: Time until weapon can fire again (seconds).
        is_operational: Whether the weapon is functional.
        current_aim_direction: Current turret aim direction (for turreted weapons).
        mount_position: Position on hull {'x': 0-1, 'side': str}.
        slot_name: Identifier for this weapon slot.
    """
    weapon: Weapon
    ammo_remaining: int
    cooldown_remaining: float = 0.0
    is_operational: bool = True
    current_aim_direction: Optional[Vector3D] = None
    mount_position: Optional[dict] = None
    slot_name: str = ""

    def can_fire(self) -> bool:
        """Check if weapon can fire."""
        return (
            self.is_operational and
            self.ammo_remaining > 0 and
            self.cooldown_remaining <= 0
        )

    def fire(self) -> bool:
        """Attempt to fire the weapon. Returns True if successful."""
        if not self.can_fire():
            return False
        self.ammo_remaining -= 1
        self.cooldown_remaining = self.weapon.cooldown_s
        return True

    def update(self, dt: float) -> None:
        """Update cooldown timer."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining = max(0.0, self.cooldown_remaining - dt)

    def is_target_in_arc(self, ship_forward: Vector3D, target_direction: Vector3D) -> bool:
        """
        Check if target is within weapon's firing arc.

        Args:
            ship_forward: Ship's forward direction vector.
            target_direction: Direction to target (normalized).

        Returns:
            True if target is within weapon's arc.
        """
        # Calculate angle from ship forward to target
        angle_to_target = ship_forward.angle_to(target_direction)
        angle_deg = math.degrees(angle_to_target)

        # Determine the weapon's boresight direction based on facing
        if self.weapon.facing == "rear":
            # Rear-facing weapons: boresight is opposite of forward
            # Target is in arc if angle from forward is > (180 - pivot_range)
            min_angle = 180.0 - self.weapon.pivot_range_deg
            return angle_deg >= min_angle
        else:
            # Forward-facing weapons: target in arc if angle from forward < pivot_range
            return angle_deg <= self.weapon.pivot_range_deg

    def calculate_fire_direction(
        self,
        ship_forward: Vector3D,
        ship_velocity: Vector3D,
        target_position: Vector3D,
        target_velocity: Vector3D,
        shooter_position: Vector3D
    ) -> Optional[Vector3D]:
        """
        Calculate the direction to fire this weapon.

        For turreted weapons: calculates lead and checks arc.
        For fixed weapons: fires along ship forward (with gimbal adjustment).

        Returns:
            Fire direction vector, or None if target not in arc.
        """
        # Direction to target
        to_target = (target_position - shooter_position)
        distance = to_target.magnitude
        if distance < 1.0:
            return None
        target_dir = to_target.normalized()

        # Check if target is in arc
        if not self.is_target_in_arc(ship_forward, target_dir):
            return None

        if self.weapon.is_turreted:
            # Turreted weapon: calculate intercept/lead direction
            return self._calculate_lead_direction(
                shooter_position, ship_velocity,
                target_position, target_velocity
            )
        else:
            # Fixed weapon (spinal): calculate lead, but limit to gimbal range
            lead_dir = self._calculate_lead_direction(
                shooter_position, ship_velocity,
                target_position, target_velocity
            )

            # Check if lead direction is within gimbal range of ship forward
            lead_angle_deg = math.degrees(ship_forward.angle_to(lead_dir))

            if lead_angle_deg <= self.weapon.pivot_range_deg:
                # Lead is within gimbal range - use full lead
                return lead_dir
            elif lead_angle_deg <= self.weapon.pivot_range_deg * 2:
                # Lead is slightly outside gimbal - aim at gimbal limit toward lead
                # Blend ship_forward toward lead_dir, limited by gimbal
                # Calculate the direction from forward toward lead, limited by gimbal
                gimbal_fraction = self.weapon.pivot_range_deg / lead_angle_deg
                adjusted_dir = (ship_forward * (1 - gimbal_fraction) + lead_dir * gimbal_fraction).normalized()
                return adjusted_dir
            else:
                # Target too far off-axis even with lead - can't fire
                return None

    def _calculate_lead_direction(
        self,
        shooter_position: Vector3D,
        shooter_velocity: Vector3D,
        target_position: Vector3D,
        target_velocity: Vector3D
    ) -> Vector3D:
        """
        Calculate intercept direction with proper lead for turreted weapons.

        Uses quadratic solution for exact intercept time accounting for
        relative velocity between shooter and target.

        In shooter's reference frame:
        - Projectile travels at muzzle_speed in aim direction
        - Target moves at relative velocity
        - Solve: aim * muzzle_speed * T = rel_pos + rel_vel * T
        """
        # Relative position and velocity (in shooter's reference frame)
        rel_pos = target_position - shooter_position
        rel_vel = target_velocity - shooter_velocity
        distance_sq = rel_pos.dot(rel_pos)
        distance = math.sqrt(distance_sq)

        # Projectile speed (m/s)
        muzzle_mps = self.weapon.muzzle_velocity_kps * 1000
        if muzzle_mps <= 0:
            return rel_pos.normalized()

        # Solve quadratic for time-of-flight:
        # |aim * M * T|² = |D + V*T|²
        # M² * T² = D² + 2*D·V*T + V²*T²
        # (M² - V²) * T² - 2*D·V*T - D² = 0
        muzzle_sq = muzzle_mps * muzzle_mps
        rel_vel_sq = rel_vel.dot(rel_vel)
        d_dot_v = rel_pos.dot(rel_vel)

        a = muzzle_sq - rel_vel_sq
        b = -2.0 * d_dot_v
        c = -distance_sq

        # Handle edge cases
        if abs(a) < 1e-10:
            # Degenerate case: muzzle speed equals relative speed
            if abs(b) < 1e-10:
                return rel_pos.normalized()
            tof = -c / b
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0:
                # No intercept possible (target faster than projectile)
                # Fall back to direct aim
                return rel_pos.normalized()

            sqrt_disc = math.sqrt(discriminant)
            # Take positive root (future intercept)
            tof = (-b + sqrt_disc) / (2.0 * a)

            if tof < 0:
                # Try other root
                tof = (-b - sqrt_disc) / (2.0 * a)

            if tof < 0:
                # No future intercept
                return rel_pos.normalized()

        # Calculate aim direction: aim = (D + V*T) / (M*T)
        intercept_rel_pos = rel_pos + rel_vel * tof
        aim_dir = intercept_rel_pos.normalized()

        # Update aim direction
        self.current_aim_direction = aim_dir

        return aim_dir


# =============================================================================
# POINT DEFENSE STATE
# =============================================================================

@dataclass
class PDLaserState:
    """
    State of a point defense laser turret on a ship.

    Attributes:
        laser: The PDLaser specification.
        cooldown_remaining: Time until laser can fire again (seconds).
        is_operational: Whether the PD laser is functional.
        current_target_id: ID of torpedo/slug being engaged.
        heat_delivered_j: Total heat delivered to current target.
        turret_name: Identifier for this PD turret.
    """
    laser: PDLaser
    cooldown_remaining: float = 0.0
    is_operational: bool = True
    current_target_id: Optional[str] = None
    heat_delivered_j: float = 0.0
    turret_name: str = "PD-1"

    def can_fire(self) -> bool:
        """Check if PD laser can engage."""
        return self.is_operational and self.cooldown_remaining <= 0

    def engage(self) -> bool:
        """Attempt to fire the PD laser. Returns True if successful."""
        if not self.can_fire():
            return False
        self.cooldown_remaining = self.laser.cooldown_s
        return True

    def update(self, dt: float) -> None:
        """Update cooldown timer."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining = max(0.0, self.cooldown_remaining - dt)

    def reset_target(self) -> None:
        """Reset target tracking when target is destroyed or out of range."""
        self.current_target_id = None
        self.heat_delivered_j = 0.0


# =============================================================================
# SHIP COMBAT STATE
# =============================================================================

@dataclass
class ShipCombatState:
    """
    Complete combat state for a ship in the simulation.

    This class tracks all aspects of a ship during combat:
    - Kinematic state (position, velocity, orientation)
    - Propulsion state (delta-v, propellant)
    - Thermal state (heat sink, radiators)
    - Weapon states (cooldowns, ammo)
    - Damage state (armor, modules)
    - Current maneuver

    Attributes:
        ship_id: Unique identifier for this ship.
        ship_type: Ship class (e.g., 'destroyer', 'cruiser').
        faction: Faction identifier for friend/foe determination.
        kinematic_state: Physics state (position, velocity, etc.).
        thermal_system: Heat management system.
        armor: Ship armor configuration.
        module_layout: Internal module arrangement.
        geometry: Ship geometry for hit calculations.
        weapons: Dict of weapon slot to weapon state.
        torpedo_launcher: Ship's torpedo launcher system.
        point_defense: List of point defense laser turrets.
        current_maneuver: Currently executing maneuver.
        is_destroyed: Whether ship has been destroyed.
        kill_credit: ID of ship that destroyed this one.
    """
    ship_id: str
    ship_type: str
    faction: str
    kinematic_state: KinematicState
    thermal_system: Optional[ThermalSystem] = None
    armor: Optional[ShipArmor] = None
    module_layout: Optional[ModuleLayout] = None
    geometry: Optional[ShipGeometry] = None
    weapons: dict[str, WeaponState] = field(default_factory=dict)
    torpedo_launcher: Optional[TorpedoLauncher] = None
    point_defense: list[PDLaserState] = field(default_factory=list)
    current_maneuver: Optional[Maneuver] = None
    is_destroyed: bool = False
    kill_credit: Optional[str] = None

    # Attitude control (rotation)
    attitude_control: Optional[AttitudeControlSpecs] = None
    rotation_state: Optional[RotationState] = None
    angular_velocity_rad_s: float = 0.0  # Current angular velocity

    # Power system
    power_system: Optional[PowerSystem] = None

    # LLM captain weapons orders (indexed by weapon slot or 'all')
    weapons_orders: dict = field(default_factory=dict)

    # Engagement tracking
    primary_target_id: Optional[str] = None
    shots_fired: int = 0
    hits_scored: int = 0
    damage_dealt_gj: float = 0.0
    damage_taken_gj: float = 0.0
    pd_intercepts: int = 0  # Torpedoes/slugs destroyed by PD

    @property
    def position(self) -> Vector3D:
        """Current position."""
        return self.kinematic_state.position

    @property
    def velocity(self) -> Vector3D:
        """Current velocity."""
        return self.kinematic_state.velocity

    @property
    def forward(self) -> Vector3D:
        """Forward direction vector."""
        return self.kinematic_state.forward

    @property
    def up(self) -> Vector3D:
        """Up direction vector."""
        return self.kinematic_state.up

    @property
    def remaining_delta_v_kps(self) -> float:
        """Remaining delta-v in km/s."""
        return self.kinematic_state.remaining_delta_v_kps()

    @property
    def heat_percent(self) -> float:
        """Current heat level as percentage."""
        if self.thermal_system:
            return self.thermal_system.heat_percent
        return 0.0

    @property
    def hull_integrity(self) -> float:
        """Overall hull integrity percentage."""
        if self.module_layout:
            return self.module_layout.ship_integrity_percent
        return 100.0

    # -------------------------------------------------------------------------
    # Module Damage Effects
    # -------------------------------------------------------------------------

    def _get_modules_by_type(self, module_type: str) -> list:
        """Get all modules of a specific type."""
        if not self.module_layout:
            return []
        from src.modules import ModuleType
        try:
            mtype = ModuleType(module_type)
        except ValueError:
            return []
        return [m for m in self.module_layout._module_cache.values()
                if m.module_type == mtype]

    @property
    def sensor_effectiveness(self) -> float:
        """
        Calculate sensor system effectiveness based on module damage.

        Returns value from 0.0 to 1.0:
        - 1.0 = sensors fully operational
        - 0.5 = sensors at 50% effectiveness (reduced accuracy)
        - 0.0 = sensors destroyed (blind)

        Uses best available sensor if multiple exist.
        """
        sensors = self._get_modules_by_type("sensor")
        if not sensors:
            return 1.0  # No sensor modules defined = assume working

        # Use best available sensor (damaged ships can rely on backup)
        best_effectiveness = max(s.effectiveness for s in sensors)

        # If best sensor is non-functional (<25% health), major penalty
        functional_sensors = [s for s in sensors if s.is_functional]
        if not functional_sensors:
            return best_effectiveness * 0.25  # Severely degraded

        return best_effectiveness

    @property
    def engine_effectiveness(self) -> float:
        """
        Calculate engine system effectiveness based on module damage.

        Returns value from 0.0 to 1.0:
        - 1.0 = engines fully operational
        - 0.5 = engines at 50% (reduced thrust and maneuverability)
        - 0.0 = engines destroyed (no thrust)

        Below 25% health, engines are non-functional.
        """
        engines = self._get_modules_by_type("engine")
        if not engines:
            return 1.0  # No engine modules defined = assume working

        # Average effectiveness of all engines
        avg_effectiveness = sum(e.effectiveness for e in engines) / len(engines)

        # If no engines are functional, ship is adrift
        functional_engines = [e for e in engines if e.is_functional]
        if not functional_engines:
            return 0.0  # No thrust available

        return avg_effectiveness

    @property
    def reactor_effectiveness(self) -> float:
        """
        Calculate reactor system effectiveness based on module damage.

        Returns value from 0.0 to 1.0:
        - 1.0 = reactors fully operational
        - 0.5 = reduced power (slower weapon cooldowns)
        - 0.0 = reactors destroyed (no power)

        Multiple reactors provide redundancy.
        """
        reactors = self._get_modules_by_type("reactor")
        if not reactors:
            return 1.0  # No reactor modules defined = assume working

        # Use best reactor (primary + aux provide redundancy)
        best_effectiveness = max(r.effectiveness for r in reactors)

        # If primary reactor down but aux works, 75% power
        functional_reactors = [r for r in reactors if r.is_functional]
        if not functional_reactors:
            return best_effectiveness * 0.1  # Emergency power only

        # If we have redundancy, we're better off
        if len(functional_reactors) >= 2:
            return max(0.75, best_effectiveness)

        return best_effectiveness

    @property
    def bridge_effectiveness(self) -> float:
        """
        Calculate bridge/command effectiveness based on module damage.

        Returns value from 0.0 to 1.0:
        - 1.0 = bridge fully operational
        - 0.5 = reduced command capability
        - 0.0 = bridge destroyed (ship may be destroyed)
        """
        bridges = self._get_modules_by_type("bridge")
        if not bridges:
            return 1.0  # No bridge modules defined = assume working

        # Bridge is usually singular and critical
        return bridges[0].effectiveness if bridges else 1.0

    def get_effective_thrust_fraction(self) -> float:
        """
        Get the effective thrust as a fraction of maximum.

        Combines engine damage with throttle settings.
        """
        return self.engine_effectiveness

    def get_effective_turn_rate_multiplier(self) -> float:
        """
        Get the effective turn rate multiplier based on engine damage.

        Damaged engines reduce thrust vectoring capability.
        - RCS still works at 100% (no engine needed)
        - Thrust vectoring reduced proportionally to engine damage
        """
        return self.engine_effectiveness

    def get_targeting_accuracy_multiplier(self) -> float:
        """
        Get targeting accuracy multiplier based on sensor and bridge damage.

        Combines sensor effectiveness with bridge (fire control).
        """
        # Both sensors and bridge affect targeting
        sensor_factor = self.sensor_effectiveness
        bridge_factor = self.bridge_effectiveness

        # Combined effect (both need to work well)
        return sensor_factor * (0.5 + 0.5 * bridge_factor)

    def get_weapon_cooldown_multiplier(self) -> float:
        """
        Get weapon cooldown multiplier based on reactor damage.

        Damaged reactors = slower weapon cycling.
        Returns value >= 1.0 (1.0 = normal, 2.0 = double cooldown).
        """
        reactor_eff = self.reactor_effectiveness
        if reactor_eff <= 0:
            return 10.0  # Effectively disabled
        if reactor_eff >= 1.0:
            return 1.0
        # Inverse relationship: 50% reactor = 2x cooldown
        return 1.0 / reactor_eff

    @property
    def fuel_tank_effectiveness(self) -> float:
        """
        Calculate fuel tank system effectiveness based on module damage.

        Damaged fuel tanks leak propellant, reducing available delta-v.
        Returns value from 0.0 to 1.0.
        """
        fuel_tanks = self._get_modules_by_type("fuel_tank")
        if not fuel_tanks:
            return 1.0  # No fuel tank modules defined = assume working

        # Average effectiveness of all tanks (damaged tanks leak)
        total_capacity = sum(t.size_m2 for t in fuel_tanks)  # size as proxy for capacity
        effective_capacity = sum(t.size_m2 * t.effectiveness for t in fuel_tanks)

        if total_capacity <= 0:
            return 1.0
        return effective_capacity / total_capacity

    def get_effective_delta_v(self) -> float:
        """
        Get effective remaining delta-v accounting for fuel tank damage.

        Damaged fuel tanks = leaked propellant = reduced delta-v.
        """
        base_dv = self.remaining_delta_v_kps
        fuel_eff = self.fuel_tank_effectiveness
        return base_dv * fuel_eff

    def distance_to(self, other: ShipCombatState) -> float:
        """Calculate distance to another ship in meters."""
        return self.position.distance_to(other.position)

    def relative_velocity_to(self, other: ShipCombatState) -> Vector3D:
        """Calculate relative velocity to another ship."""
        return self.velocity - other.velocity

    def closing_rate_to(self, other: ShipCombatState) -> float:
        """Calculate closing rate with another ship (positive = closing)."""
        rel_pos = other.position - self.position
        rel_vel = self.velocity - other.velocity
        distance = rel_pos.magnitude
        if distance < 1.0:
            return 0.0
        # rel_vel points in same direction as rel_pos → we're closing
        return rel_vel.dot(rel_pos.normalized())


# =============================================================================
# PROJECTILE IN FLIGHT
# =============================================================================

@dataclass
class ProjectileInFlight:
    """
    A kinetic projectile being tracked in the simulation.

    Attributes:
        projectile_id: Unique identifier.
        projectile: The projectile physics object.
        source_ship_id: ID of ship that fired this.
        target_ship_id: Intended target (for tracking).
        launch_time: When projectile was launched.
        min_distance_to_target: Closest distance achieved to target (meters).
        prev_distance_to_target: Previous tick's distance to target (for miss detection).
    """
    projectile_id: str
    projectile: KineticProjectile
    source_ship_id: str
    target_ship_id: Optional[str] = None
    launch_time: float = 0.0
    min_distance_to_target: float = float('inf')
    prev_distance_to_target: float = float('inf')


@dataclass
class TorpedoInFlight:
    """
    A torpedo being tracked in the simulation.

    Attributes:
        torpedo_id: Unique identifier.
        torpedo: The torpedo physics object.
        source_ship_id: ID of ship that launched this.
        launch_time: When torpedo was launched.
        heat_absorbed_j: Heat damage from point defense lasers.
        is_disabled: Whether torpedo electronics are disabled.
        prev_distance_to_target: Distance to target last tick (for closest approach detection).
        min_distance_to_target: Minimum distance achieved to target.
        main_engine_dv_used_kps: Delta-v used by main engine.
        rcs_dv_used_kps: Delta-v used by lateral RCS corrections.
    """
    torpedo_id: str
    torpedo: Torpedo
    source_ship_id: str
    launch_time: float = 0.0
    heat_absorbed_j: float = 0.0
    is_disabled: bool = False
    prev_distance_to_target: float = float('inf')
    min_distance_to_target: float = float('inf')
    main_engine_dv_used_kps: float = 0.0
    rcs_dv_used_kps: float = 0.0

    # PD damage thresholds (from pointdefense module)
    ELECTRONICS_THRESHOLD_J: float = 10_000.0   # 10 kJ - electronics fail
    WARHEAD_THRESHOLD_J: float = 100_000.0      # 100 kJ - warhead detonates

    def absorb_pd_heat(self, heat_j: float) -> bool:
        """
        Absorb heat from point defense laser.

        Returns:
            True if torpedo was destroyed (warhead detonated).
        """
        self.heat_absorbed_j += heat_j
        if self.heat_absorbed_j >= self.ELECTRONICS_THRESHOLD_J:
            self.is_disabled = True
        return self.heat_absorbed_j >= self.WARHEAD_THRESHOLD_J


# =============================================================================
# ENGAGEMENT METRICS
# =============================================================================

@dataclass
class EngagementMetrics:
    """
    Metrics tracking overall engagement statistics.

    Attributes:
        total_shots_fired: Total projectiles launched.
        total_hits: Total hits scored.
        total_torpedoes_launched: Total torpedoes launched.
        total_torpedo_hits: Total torpedo impacts.
        total_damage_dealt: Total damage dealt (GJ).
        ships_destroyed: List of destroyed ship IDs.
        battle_duration: Total simulation time.
        torpedo_main_engine_dv_kps: Total main engine delta-v used by torpedoes.
        torpedo_rcs_dv_kps: Total RCS delta-v used by torpedoes.
    """
    total_shots_fired: int = 0
    total_hits: int = 0
    total_torpedoes_launched: int = 0
    total_torpedo_hits: int = 0
    total_torpedo_intercepted: int = 0
    total_damage_dealt: float = 0.0
    ships_destroyed: list[str] = field(default_factory=list)
    battle_duration: float = 0.0
    torpedo_main_engine_dv_kps: float = 0.0
    torpedo_rcs_dv_kps: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate overall hit rate."""
        if self.total_shots_fired == 0:
            return 0.0
        return self.total_hits / self.total_shots_fired


# =============================================================================
# COMBAT SIMULATION
# =============================================================================

class CombatSimulation:
    """
    Main combat simulation engine.

    This class manages the complete simulation loop:
    - Discrete time stepping
    - Ship state updates (physics, thermal, weapons)
    - Projectile tracking and hit detection
    - Damage resolution
    - Event logging
    - Decision point callbacks

    Usage:
        sim = CombatSimulation(time_step=1.0, decision_interval=30.0)
        sim.add_ship(ship_state)
        sim.set_decision_callback(my_callback)
        sim.run(duration=300.0)

    Attributes:
        time_step: Simulation time step in seconds.
        decision_interval: Time between LLM decision points.
        current_time: Current simulation time.
        ships: Dict of ship_id to ShipCombatState.
        projectiles: List of active projectiles.
        torpedoes: List of active torpedoes.
        events: List of simulation events.
        metrics: Engagement metrics.
    """

    def __init__(
        self,
        time_step: float = DEFAULT_TIME_STEP,
        decision_interval: float = DEFAULT_DECISION_INTERVAL,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the combat simulation.

        Args:
            time_step: Simulation time step in seconds (default 1.0).
            decision_interval: Seconds between decision points (default 30.0).
            seed: Random seed for reproducibility.
        """
        self.time_step = time_step
        self.decision_interval = max(
            MIN_DECISION_INTERVAL,
            min(MAX_DECISION_INTERVAL, decision_interval)
        )
        self.current_time: float = 0.0
        self.last_decision_time: float = 0.0

        # Ship tracking
        self.ships: dict[str, ShipCombatState] = {}

        # Projectile tracking
        self.projectiles: list[ProjectileInFlight] = []
        self.torpedoes: list[TorpedoInFlight] = []

        # Event log
        self.events: list[SimulationEvent] = []

        # Metrics
        self.metrics = EngagementMetrics()

        # Combat resolver for damage calculations
        self._initial_seed = seed if seed is not None else 42
        self.rng = random.Random(seed)
        self.combat_resolver = CombatResolver(rng=self.rng)
        self.damage_propagator = DamagePropagator()

        # Decision callback
        self._decision_callback: Optional[Callable[[str, 'CombatSimulation'], list[Any]]] = None

        # Event callbacks (for external recording/logging)
        self._event_callbacks: list[Callable[['SimulationEvent'], None]] = []

        # Simulation state
        self._running = False
        self._paused = False

    # -------------------------------------------------------------------------
    # Ship Management
    # -------------------------------------------------------------------------

    def add_ship(self, ship: ShipCombatState) -> None:
        """
        Add a ship to the simulation.

        Args:
            ship: The ship combat state to add.
        """
        self.ships[ship.ship_id] = ship

    def remove_ship(self, ship_id: str) -> Optional[ShipCombatState]:
        """
        Remove a ship from the simulation.

        Args:
            ship_id: ID of ship to remove.

        Returns:
            The removed ship, or None if not found.
        """
        return self.ships.pop(ship_id, None)

    def get_ship(self, ship_id: str) -> Optional[ShipCombatState]:
        """Get a ship by ID."""
        return self.ships.get(ship_id)

    def get_ships_by_faction(self, faction: str) -> list[ShipCombatState]:
        """Get all ships of a given faction."""
        return [s for s in self.ships.values() if s.faction == faction]

    def get_enemy_ships(self, ship_id: str) -> list[ShipCombatState]:
        """Get all enemy ships relative to a given ship."""
        ship = self.get_ship(ship_id)
        if not ship:
            return []
        return [
            s for s in self.ships.values()
            if s.faction != ship.faction and not s.is_destroyed
        ]

    def get_friendly_ships(self, ship_id: str) -> list[ShipCombatState]:
        """Get all friendly ships relative to a given ship."""
        ship = self.get_ship(ship_id)
        if not ship:
            return []
        return [
            s for s in self.ships.values()
            if s.faction == ship.faction and s.ship_id != ship_id and not s.is_destroyed
        ]

    # -------------------------------------------------------------------------
    # Decision Point Callback
    # -------------------------------------------------------------------------

    def set_decision_callback(
        self,
        callback: Callable[[str, 'CombatSimulation'], list[Any]]
    ) -> None:
        """
        Set the callback function for decision points.

        The callback is called for each ship when a decision point is reached.
        It receives the ship_id and the simulation instance, and should return
        a list of commands to execute.

        Args:
            callback: Function(ship_id, simulation) -> list[commands]
        """
        self._decision_callback = callback

    # -------------------------------------------------------------------------
    # Command Injection
    # -------------------------------------------------------------------------

    def inject_command(self, ship_id: str, command: Any) -> bool:
        """
        Inject a command for a ship to execute.

        Commands can be:
        - Maneuver objects for movement
        - 'fire_at' dict for weapon firing
        - 'launch_torpedo' dict for torpedo launch

        Args:
            ship_id: ID of ship to command.
            command: The command to execute.

        Returns:
            True if command was accepted.
        """
        ship = self.get_ship(ship_id)
        if not ship or ship.is_destroyed:
            return False

        # Handle different command types
        if isinstance(command, Maneuver):
            ship.current_maneuver = command
            self._log_event(SimulationEventType.MANEUVER_STARTED, ship_id, data={
                'maneuver_type': command.maneuver_type.name,
                'duration': command.duration
            })
            return True

        if isinstance(command, dict):
            if command.get('type') == 'fire_at':
                return self._handle_fire_command(ship, command)
            elif command.get('type') == 'launch_torpedo':
                return self._handle_torpedo_command(ship, command)
            elif command.get('type') == 'set_radiators':
                return self._handle_radiator_command(ship, command)
            elif command.get('type') == 'set_target':
                ship.primary_target_id = command.get('target_id')
                return True
            elif command.get('type') == 'weapons_order':
                return self._handle_weapons_order(ship, command)
            elif command.get('type') == 'weapons_orders':
                # Handle multiple weapon orders (new format with separate spinal/turret modes)
                orders = command.get('orders', [])
                for order in orders:
                    self._handle_weapons_order(ship, {'order': order})
                return True

        return False

    def _handle_fire_command(self, ship: ShipCombatState, command: dict) -> bool:
        """Handle a fire weapon command."""
        weapon_slot = command.get('weapon_slot')
        target_id = command.get('target_id')

        if not weapon_slot or weapon_slot not in ship.weapons:
            return False

        weapon_state = ship.weapons[weapon_slot]
        if not weapon_state.can_fire():
            return False

        target = self.get_ship(target_id)
        if not target or target.is_destroyed:
            return False

        # Check if target is in weapon arc before consuming ammo
        to_target = (target.position - ship.position).normalized()
        if not weapon_state.is_target_in_arc(ship.forward, to_target):
            # Target not in arc - don't fire
            return False

        # Fire the weapon (consumes ammo, starts cooldown)
        if weapon_state.fire():
            # Discharge weapon capacitor and generate heat
            if ship.power_system:
                heat_gj = ship.power_system.fire_weapon(weapon_slot)
                if ship.thermal_system and heat_gj > 0:
                    ship.thermal_system.add_heat("weapons", heat_gj)

            # Launch projectile with proper aiming
            self._launch_projectile(ship, target, weapon_state)
            return True

        return False

    def _handle_torpedo_command(self, ship: ShipCombatState, command: dict) -> bool:
        """
        Handle a torpedo launch command.

        Command dict fields:
            target_id: ID of target ship (required)
            guidance_mode: Optional GuidanceMode to use (e.g., GuidanceMode.SMART)
        """
        target_id = command.get('target_id')
        target = self.get_ship(target_id)

        if not target or target.is_destroyed:
            return False

        if not ship.torpedo_launcher:
            return False

        torpedo = ship.torpedo_launcher.launch(
            shooter_position=ship.position,
            shooter_velocity=ship.velocity,
            target_id=target_id,
            target_position=target.position,
            target_velocity=target.velocity,
            current_time=self.current_time
        )

        if torpedo:
            # Discharge torpedo launcher capacitor and generate heat
            if ship.power_system:
                heat_gj = ship.power_system.fire_weapon("torpedo_launcher")
                if ship.thermal_system and heat_gj > 0:
                    ship.thermal_system.add_heat("weapons", heat_gj)

            # Override guidance mode if specified in command
            guidance_mode = command.get('guidance_mode')
            if guidance_mode is not None and isinstance(guidance_mode, GuidanceMode):
                torpedo.guidance_mode = guidance_mode

            torpedo_id = f"torp_{ship.ship_id}_{uuid.uuid4().hex[:8]}"
            self.torpedoes.append(TorpedoInFlight(
                torpedo_id=torpedo_id,
                torpedo=torpedo,
                source_ship_id=ship.ship_id,
                launch_time=self.current_time
            ))
            self.metrics.total_torpedoes_launched += 1
            self._log_event(SimulationEventType.TORPEDO_LAUNCHED, ship.ship_id, target_id, {
                'torpedo_id': torpedo_id,
                'initial_velocity_kps': torpedo.velocity.magnitude / 1000.0,
                'warhead_gj': torpedo.specs.warhead_yield_gj,
                'delta_v_kps': torpedo.specs.total_delta_v_kps,
                'guidance_mode': torpedo.guidance_mode.name
            })
            return True

        return False

    def _handle_radiator_command(self, ship: ShipCombatState, command: dict) -> bool:
        """Handle a radiator extend/retract command."""
        if not ship.thermal_system:
            return False

        extend = command.get('extend', True)
        if extend:
            ship.thermal_system.radiators.extend_all()
            self._log_event(SimulationEventType.RADIATOR_EXTENDED, ship.ship_id)
        else:
            ship.thermal_system.radiators.retract_all()
            self._log_event(SimulationEventType.RADIATOR_RETRACTED, ship.ship_id)

        return True

    def _handle_weapons_order(self, ship: ShipCombatState, command: dict) -> bool:
        """Handle a weapons order command from LLM captain."""
        order = command.get('order')
        if not order:
            return False

        # Store the order - the step() method will process it
        weapon_slot = order.weapon_slot
        if weapon_slot == 'all':
            # Apply to all weapons
            for slot in ship.weapons:
                ship.weapons_orders[slot] = order
        else:
            ship.weapons_orders[weapon_slot] = order

        return True

    def _process_weapons_orders(self, ship: ShipCombatState, debug: bool = False) -> None:
        """Process weapons orders and fire weapons according to their orders."""
        from .firecontrol import WeaponsCommand, calculate_hit_probability

        if not ship.weapons_orders:
            if debug:
                print(f"  [{ship.ship_id}] No weapons orders")
            return

        for weapon_slot, order in ship.weapons_orders.items():
            if weapon_slot not in ship.weapons:
                continue

            weapon_state = ship.weapons[weapon_slot]
            if not weapon_state.can_fire():
                continue

            # Get target
            target_id = order.target_id or ship.primary_target_id
            if not target_id:
                continue

            target = self.get_ship(target_id)
            if not target or target.is_destroyed:
                continue

            # Check if target is in weapon arc
            to_target = (target.position - ship.position).normalized()
            if not weapon_state.is_target_in_arc(ship.forward, to_target):
                continue

            # For non-turreted weapons (spinal), also check if lead direction is achievable
            # This is important because target might be in arc but lead required for hit is outside gimbal
            if not weapon_state.weapon.is_turreted:
                fire_direction = weapon_state.calculate_fire_direction(
                    ship_forward=ship.forward,
                    ship_velocity=ship.velocity,
                    target_position=target.position,
                    target_velocity=target.velocity,
                    shooter_position=ship.position
                )
                if fire_direction is None:
                    # Lead direction is outside gimbal range - can't effectively fire
                    continue

            # Calculate firing solution
            is_evading = (target.current_maneuver is not None and
                         hasattr(target.current_maneuver, 'maneuver_type') and
                         target.current_maneuver.maneuver_type.name == 'EVADE')

            # Use target geometry if available, otherwise create a simple one
            target_geometry = target.geometry
            if not target_geometry:
                target_geometry = ShipGeometry(
                    length_m=100.0,
                    beam_m=20.0,
                    nose_section_length=0.2,
                    tail_section_length=0.2,
                )

            solution = calculate_hit_probability(
                shooter_position=ship.position,
                shooter_velocity=ship.velocity,
                target_position=target.position,
                target_velocity=target.velocity,
                target_geometry=target_geometry,
                target_forward=target.forward,
                muzzle_velocity_kps=weapon_state.weapon.muzzle_velocity_kps,
                target_is_evading=is_evading,
            )

            if not solution.can_fire:
                continue

            # Evaluate based on order command
            should_fire = False
            if order.command == WeaponsCommand.FIRE_IMMEDIATE:
                should_fire = True
            elif order.command == WeaponsCommand.FIRE_WHEN_OPTIMAL:
                should_fire = solution.hit_probability >= order.min_hit_probability
            elif order.command == WeaponsCommand.FIRE_AT_RANGE:
                distance_km = (target.position - ship.position).magnitude / 1000
                should_fire = distance_km <= order.max_range_km
            elif order.command == WeaponsCommand.HOLD_FIRE:
                should_fire = False
            elif order.command == WeaponsCommand.FREE_FIRE:
                should_fire = solution.hit_probability >= 0.1

            if should_fire:
                # Fire the weapon
                fired = weapon_state.fire()
                if fired:
                    # Handle heat
                    if ship.power_system:
                        heat_gj = ship.power_system.fire_weapon(weapon_slot)
                        if ship.thermal_system and heat_gj > 0:
                            ship.thermal_system.add_heat("weapons", heat_gj)

                    # Launch projectile
                    self._launch_projectile(ship, target, weapon_state)

                    # Calculate distance and time to target for display
                    distance_km = (target.position - ship.position).magnitude / 1000
                    muzzle_v = weapon_state.weapon.muzzle_velocity_kps
                    time_to_target = distance_km / muzzle_v if muzzle_v > 0 else 0
                    weapon_name = weapon_state.weapon.name

                    print(f"  [{ship.ship_id}] FIRED {weapon_name} at {target_id}")
                    print(f"       Distance: {distance_km:.0f}km, ETA: {time_to_target:.1f}s, "
                          f"v: {muzzle_v:.1f}km/s, P(hit): {solution.hit_probability:.1%}")

    # -------------------------------------------------------------------------
    # Projectile Launch
    # -------------------------------------------------------------------------

    def _launch_projectile(
        self,
        shooter: ShipCombatState,
        target: ShipCombatState,
        weapon_state: WeaponState
    ) -> bool:
        """Launch a kinetic projectile from shooter toward target.

        Handles both turreted and fixed weapons:
        - Turreted weapons (coilgun batteries): calculate lead, aim turret
        - Fixed weapons (spinal coilers): fire along ship forward with limited gimbal

        Returns:
            True if projectile was launched, False if target not in arc.
        """
        weapon = weapon_state.weapon

        # Calculate fire direction based on weapon type
        fire_direction = weapon_state.calculate_fire_direction(
            ship_forward=shooter.forward,
            ship_velocity=shooter.velocity,
            target_position=target.position,
            target_velocity=target.velocity,
            shooter_position=shooter.position
        )

        if fire_direction is None:
            # Target not in weapon arc - can't fire
            return False

        # Create projectile
        projectile = KineticProjectile.from_launch(
            shooter_position=shooter.position,
            shooter_velocity=shooter.velocity,
            target_direction=fire_direction,
            muzzle_velocity_kps=weapon.muzzle_velocity_kps,
            mass_kg=weapon.warhead_mass_kg
        )

        # Track projectile
        proj_id = f"proj_{shooter.ship_id}_{uuid.uuid4().hex[:8]}"
        self.projectiles.append(ProjectileInFlight(
            projectile_id=proj_id,
            projectile=projectile,
            source_ship_id=shooter.ship_id,
            target_ship_id=target.ship_id,
            launch_time=self.current_time
        ))

        # Update stats
        shooter.shots_fired += 1
        self.metrics.total_shots_fired += 1

        # Add heat from firing
        if shooter.thermal_system:
            shooter.thermal_system.add_heat("coilgun", COILGUN_HEAT_PER_SHOT_GJ)

        self._log_event(SimulationEventType.PROJECTILE_LAUNCHED, shooter.ship_id, target.ship_id, {
            'projectile_id': proj_id,
            'kinetic_energy_gj': projectile.kinetic_energy_gj,
            'muzzle_velocity_kps': weapon.muzzle_velocity_kps,
            'weapon_type': weapon.weapon_type,
            'is_turreted': weapon.is_turreted,
            'fire_direction': f"({fire_direction.x:.3f}, {fire_direction.y:.3f}, {fire_direction.z:.3f})"
        })

        return True

    # -------------------------------------------------------------------------
    # Simulation Loop
    # -------------------------------------------------------------------------

    def run(self, duration: float, realtime: bool = False) -> None:
        """
        Run the simulation for a specified duration.

        Args:
            duration: Total simulation time in seconds.
            realtime: If True, run at realtime speed (not implemented).
        """
        self._running = True
        self._log_event(SimulationEventType.SIMULATION_STARTED)

        end_time = self.current_time + duration

        while self._running and self.current_time < end_time:
            if not self._paused:
                self.step()

        self.metrics.battle_duration = self.current_time

        # Log projectiles still in flight at end
        in_flight_count = len(self.projectiles)

        self._log_event(SimulationEventType.SIMULATION_ENDED, data={
            'duration': self.current_time,
            'ships_destroyed': len(self.metrics.ships_destroyed),
            'projectiles_in_flight': in_flight_count
        })

    def step(self) -> list[SimulationEvent]:
        """
        Execute a single simulation step.

        Returns:
            List of events that occurred during this step.
        """
        step_events: list[SimulationEvent] = []
        dt = self.time_step

        # Check for decision point
        if self.current_time - self.last_decision_time >= self.decision_interval:
            self._trigger_decision_points()
            self.last_decision_time = self.current_time

        # Update all ships
        for ship in self.ships.values():
            if not ship.is_destroyed:
                self._update_ship(ship, dt)

        # Update projectiles and check hits
        self._update_projectiles(dt)

        # Point defense engages incoming torpedoes and projectiles
        self._update_point_defense(dt)

        # Update torpedoes and check hits
        self._update_torpedoes(dt)

        # Check for battle end conditions
        self._check_battle_end()

        # Advance time
        self.current_time += dt

        return step_events

    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False

    def pause(self) -> None:
        """Pause the simulation."""
        self._paused = True

    def resume(self) -> None:
        """Resume the simulation."""
        self._paused = False

    # -------------------------------------------------------------------------
    # Ship Update
    # -------------------------------------------------------------------------

    def _update_ship(self, ship: ShipCombatState, dt: float) -> None:
        """Update a single ship for one time step."""
        # Process current maneuver
        throttle = 0.0
        gimbal_pitch = 0.0
        gimbal_yaw = 0.0

        if ship.current_maneuver:
            maneuver = ship.current_maneuver

            if maneuver.is_complete(self.current_time):
                self._log_event(SimulationEventType.MANEUVER_COMPLETED, ship.ship_id, data={
                    'maneuver_type': maneuver.maneuver_type.name
                })
                ship.current_maneuver = None
            else:
                throttle = maneuver.throttle

                # Apply maneuver-specific logic
                if maneuver.maneuver_type == ManeuverType.BURN and maneuver.direction:
                    # Rotate to burn direction (engines on if throttle > 0)
                    self._rotate_ship_toward(ship, maneuver.direction, dt, engines_on=(throttle > 0))

                elif maneuver.maneuver_type == ManeuverType.EVASIVE:
                    # Corkscrew evasive pattern: keep nose at target, use gimbal for spiral
                    # First, find target and point at it (like INTERCEPT but with gimbal jinking)
                    target = None
                    if maneuver.target_id:
                        target = self.get_ship(maneuver.target_id)
                    else:
                        # Find nearest enemy
                        enemies = self.get_enemy_ships(ship.ship_id)
                        if enemies:
                            target = min(enemies, key=lambda e: ship.distance_to(e))

                    if target:
                        intercept_dir = self._calculate_intercept_direction(ship, target)
                        self._rotate_ship_toward(ship, intercept_dir, dt, engines_on=(throttle > 0))

                    # Apply corkscrew gimbal pattern for evasion
                    gimbal_pitch, gimbal_yaw = self._apply_evasive_pattern(ship, dt, engines_on=(throttle > 0))

                elif maneuver.maneuver_type == ManeuverType.INTERCEPT and maneuver.target_id:
                    target = self.get_ship(maneuver.target_id)
                    if target:
                        intercept_dir = self._calculate_intercept_direction(ship, target)
                        self._rotate_ship_toward(ship, intercept_dir, dt, engines_on=(throttle > 0))

                elif maneuver.maneuver_type == ManeuverType.BRAKE:
                    # Thrust retrograde (opposite to velocity) to slow down
                    velocity = ship.velocity
                    if velocity.magnitude > 0.1:  # Only brake if moving
                        retrograde_dir = velocity.normalized() * -1  # Opposite to velocity
                        self._rotate_ship_toward(ship, retrograde_dir, dt, engines_on=(throttle > 0))

                elif maneuver.maneuver_type == ManeuverType.MAINTAIN:
                    # Coast - maintain current course, no thrust
                    # Just keep nose pointed at target for tactical awareness
                    pass  # No maneuver needed, ship coasts

        # Apply engine damage to effective thrust
        engine_eff = ship.get_effective_thrust_fraction()
        effective_throttle = throttle * engine_eff

        # Update kinematic state with reduced thrust if engines damaged
        ship.kinematic_state = propagate_state(
            ship.kinematic_state, dt, effective_throttle, gimbal_pitch, gimbal_yaw
        )

        # Update thermal system
        if ship.thermal_system:
            # Activate/deactivate engine heat source based on throttle
            # (ThermalSystem.update() handles heat generation from active sources)
            if throttle > 0:
                ship.thermal_system.set_source_active("engines", True)
            else:
                ship.thermal_system.set_source_active("engines", False)

            # Update thermal system (generates heat from active sources, dissipates via radiators)
            thermal_result = ship.thermal_system.update(dt)

            # Check for thermal warnings
            if thermal_result['is_critical']:
                self._log_event(SimulationEventType.THERMAL_CRITICAL, ship.ship_id, data={
                    'heat_percent': thermal_result['heat_percent']
                })
            elif thermal_result['is_overheating']:
                self._log_event(SimulationEventType.THERMAL_WARNING, ship.ship_id, data={
                    'heat_percent': thermal_result['heat_percent']
                })

        # Update power system (charges weapon capacitors)
        if ship.power_system:
            # Set drive power consumption based on throttle
            ship.power_system.set_drive_throttle(throttle)
            # Update power distribution (charges capacitors from available power)
            ship.power_system.update(dt)

        # Update weapon cooldowns (reactor damage slows recharge)
        cooldown_multiplier = ship.get_weapon_cooldown_multiplier()
        effective_cooldown_dt = dt / cooldown_multiplier  # Slower cooldown recovery
        for weapon_state in ship.weapons.values():
            weapon_state.update(effective_cooldown_dt)

        # Process weapons orders from LLM captain
        self._process_weapons_orders(ship)

    def _rotate_ship_toward(
        self,
        ship: ShipCombatState,
        target_dir: Vector3D,
        dt: float,
        engines_on: bool = True
    ) -> None:
        """
        Rotate ship to face a direction using bang-bang control.

        Implements realistic rotation physics:
        1. Accelerate angular velocity using available torque (TV + RCS or RCS only)
        2. At the midpoint, switch to braking
        3. Decelerate to zero angular velocity exactly at target orientation

        Args:
            ship: The ship to rotate
            target_dir: Target direction to face
            dt: Time step in seconds
            engines_on: Whether main engines are firing (enables thrust vectoring)
        """
        current_forward = ship.forward
        target_normalized = target_dir.normalized()
        angle_to_target = current_forward.angle_to(target_normalized)

        # Tolerance for "close enough"
        ANGLE_TOLERANCE = 0.01  # ~0.6 degrees
        VELOCITY_TOLERANCE = 0.01  # rad/s

        # Check if rotation is complete
        if angle_to_target < ANGLE_TOLERANCE and abs(ship.angular_velocity_rad_s) < VELOCITY_TOLERANCE:
            ship.angular_velocity_rad_s = 0.0
            ship.rotation_state = None
            return

        # Get attitude control specs (use defaults if not set)
        # Engine damage reduces thrust vectoring but RCS still works
        engine_eff = ship.get_effective_turn_rate_multiplier()

        if ship.attitude_control:
            # Get base values
            tv_accel = ship.attitude_control.tv_angular_accel_deg_s2
            tv_max_vel = ship.attitude_control.tv_max_angular_vel_deg_s
            rcs_accel = ship.attitude_control.rcs_angular_accel_deg_s2
            rcs_max_vel = ship.attitude_control.rcs_max_angular_vel_deg_s

            # Apply engine damage to thrust vectoring only
            effective_tv_accel = tv_accel * engine_eff
            effective_tv_max_vel = tv_max_vel * engine_eff

            if engines_on:
                angular_accel = math.radians(effective_tv_accel + rcs_accel)
                max_angular_vel = math.radians(max(effective_tv_max_vel, rcs_max_vel))
            else:
                angular_accel = math.radians(rcs_accel)
                max_angular_vel = math.radians(rcs_max_vel)
        else:
            # Default: cruiser-like specs with engine damage applied
            if engines_on:
                tv_accel = 0.454 * engine_eff  # TV reduced by damage
                angular_accel = math.radians(tv_accel + 0.0085)  # TV + RCS
                max_angular_vel = math.radians(6.39 * engine_eff)
            else:
                angular_accel = math.radians(0.0085)  # RCS only (unaffected)
                max_angular_vel = math.radians(0.87)

        # Calculate rotation axis
        rotation_axis = current_forward.cross(target_normalized)
        if rotation_axis.magnitude < 1e-6:
            # Vectors are nearly parallel
            if angle_to_target < ANGLE_TOLERANCE:
                # Already facing target
                ship.angular_velocity_rad_s = 0.0
                return
            # Use up as rotation axis for 180-degree flip
            rotation_axis = ship.up
        rotation_axis = rotation_axis.normalized()

        # Check if we need to initialize or update rotation state
        if ship.rotation_state is None or ship.rotation_state.target_direction.distance_to(target_normalized) > 0.1:
            # New rotation - initialize state
            ship.rotation_state = RotationState(
                target_direction=target_normalized,
                initial_direction=Vector3D(current_forward.x, current_forward.y, current_forward.z),
                total_angle_rad=angle_to_target,
                current_angular_vel_rad_s=ship.angular_velocity_rad_s,
                rotation_axis=rotation_axis,
                phase="accelerate"
            )

        # Bang-bang control logic
        # To stop from velocity ω with deceleration α: need angle = ω²/(2α)
        current_omega = abs(ship.angular_velocity_rad_s)
        stopping_angle = (current_omega ** 2) / (2 * angular_accel) if angular_accel > 0 else 0

        # Determine phase: accelerate or decelerate
        if angle_to_target <= stopping_angle + ANGLE_TOLERANCE:
            # Need to brake NOW to stop at target
            phase = "decelerate"
        elif current_omega >= max_angular_vel:
            # At max velocity, coast or brake slightly
            phase = "coast"
        else:
            # Still accelerating
            phase = "accelerate"

        ship.rotation_state.phase = phase

        # Apply torque based on phase
        if phase == "accelerate":
            # Accelerate rotation
            new_omega = current_omega + angular_accel * dt
            new_omega = min(new_omega, max_angular_vel)
            ship.angular_velocity_rad_s = new_omega
        elif phase == "decelerate":
            # Decelerate rotation (braking)
            new_omega = current_omega - angular_accel * dt
            new_omega = max(0.0, new_omega)
            ship.angular_velocity_rad_s = new_omega
        else:  # coast
            # Maintain current velocity (maybe slight adjustment)
            pass

        # Apply rotation for this time step
        rotation_amount = ship.angular_velocity_rad_s * dt

        # Don't overshoot
        rotation_amount = min(rotation_amount, angle_to_target)

        if rotation_amount > 0:
            # Update forward and up vectors
            new_forward = ship.kinematic_state.forward.rotate_around_axis(
                rotation_axis, rotation_amount
            )
            new_up = ship.kinematic_state.up.rotate_around_axis(
                rotation_axis, rotation_amount
            )

            ship.kinematic_state.forward = new_forward.normalized()
            ship.kinematic_state.up = new_up.normalized()

    def _apply_evasive_pattern(
        self,
        ship: ShipCombatState,
        dt: float,
        engines_on: bool = True
    ) -> tuple[float, float]:
        """
        Apply jinking evasive maneuver pattern.

        Jinking works by:
        1. Deflecting thrust in a random perpendicular direction for a few seconds
        2. Then deflecting in the OPPOSITE direction to cancel lateral velocity
        3. Repeat with new random direction

        This makes the ship's position unpredictable while keeping net lateral
        velocity near zero (no drift). The main thrust vector toward/away from
        target stays intact.

        Returns:
            Tuple of (gimbal_pitch_deg, gimbal_yaw_deg) for the jink pattern.
        """
        # Jink parameters
        jink_period = 3.0  # Seconds per jink direction (3s one way, 3s back)
        max_deflection = 0.9  # Max gimbal deflection in degrees (limit is ~1°)

        # Determine which jink cycle we're in and the phase within it
        cycle_time = self.current_time % (jink_period * 2)  # 0 to 6 seconds
        first_half = cycle_time < jink_period  # First 3s or second 3s

        # Use ship-specific seed for this jink cycle to pick direction
        cycle_number = int(self.current_time / (jink_period * 2))
        self.rng.seed(hash(ship.ship_id) + cycle_number)

        # Random jink direction (angle in the pitch/yaw plane)
        jink_angle = self.rng.random() * 2 * math.pi

        # Deflection magnitude (full deflection for clear jinking)
        deflection = max_deflection

        # Calculate gimbal based on jink angle
        gimbal_pitch = deflection * math.sin(jink_angle)
        gimbal_yaw = deflection * math.cos(jink_angle)

        # In second half, reverse direction to cancel lateral velocity
        if not first_half:
            gimbal_pitch = -gimbal_pitch
            gimbal_yaw = -gimbal_yaw

        # Reset RNG seed to not affect other random operations
        self.rng.seed(self._initial_seed + int(self.current_time * 1000))

        return (gimbal_pitch, gimbal_yaw)

    def _calculate_intercept_direction(
        self,
        ship: ShipCombatState,
        target: ShipCombatState
    ) -> Vector3D:
        """
        Calculate optimal burn direction to intercept target.

        Uses collision course guidance (same as torpedo logic):
        1. Decompose relative velocity into closing speed + lateral drift
        2. If not closing: burn toward target
        3. If closing but have lateral drift: blend toward-target with cancel-lateral
        4. If on good course: burn toward target to increase closing speed
        """
        # Line of sight to target
        to_target = target.position - ship.position
        distance_m = to_target.magnitude

        if distance_m < 100.0:  # Essentially at target
            return ship.forward  # Maintain current heading

        los = to_target.normalized()  # Unit vector toward target

        # Relative velocity: how we move relative to target
        # Positive closing speed = we're approaching
        rel_vel = ship.velocity - target.velocity

        # Decompose relative velocity into:
        # - Closing speed (along LOS, toward target is positive)
        # - Lateral velocity (perpendicular to LOS, causes miss)
        closing_speed_mps = rel_vel.dot(los)
        lateral_vel = rel_vel - los * closing_speed_mps
        lateral_speed_mps = lateral_vel.magnitude

        # Minimum closing speed we want (km/s)
        MIN_CLOSING_SPEED_KPS = 2.0  # Want at least 2 km/s closing
        MIN_CLOSING_SPEED_MPS = MIN_CLOSING_SPEED_KPS * 1000.0

        # PHASE 1: Not closing - burn directly toward target
        if closing_speed_mps <= 0:
            return los

        # PHASE 2: Closing but too slow - prioritize building closing speed
        if closing_speed_mps < MIN_CLOSING_SPEED_MPS:
            if lateral_speed_mps > 100.0:  # Have significant lateral drift
                # Blend: mostly toward target, some lateral correction
                lateral_correction = (lateral_vel * -1.0).normalized()
                # Weight toward target more heavily when we need speed
                weight_toward = 0.8
                weight_lateral = 0.2
                burn_dir = los * weight_toward + lateral_correction * weight_lateral
                return burn_dir.normalized()
            else:
                # On course, just need speed - burn toward target
                return los

        # PHASE 3: Good closing speed - focus on lateral correction
        if lateral_speed_mps > 50.0:  # Have lateral drift to correct
            lateral_correction = (lateral_vel * -1.0).normalized()
            # More weight to lateral correction when we have enough closing speed
            # Scale based on how much lateral drift we have
            lateral_ratio = min(1.0, lateral_speed_mps / 500.0)  # Max at 500 m/s drift
            weight_toward = 1.0 - lateral_ratio * 0.6  # At most 60% to lateral
            weight_lateral = lateral_ratio * 0.6
            burn_dir = los * weight_toward + lateral_correction * weight_lateral
            return burn_dir.normalized()

        # PHASE 4: On good intercept course - burn toward target
        return los

    # -------------------------------------------------------------------------
    # Projectile Update
    # -------------------------------------------------------------------------

    def _update_projectiles(self, dt: float) -> None:
        """
        Update all projectiles using adaptive timestep with geometric hit detection.

        When a projectile is far from target (TCA > 4s), uses normal timestep.
        When close to target, switches to micro-timesteps (1ms) for precise
        geometric intersection detection against ship cylinder.
        """
        projectiles_to_remove: list[ProjectileInFlight] = []

        # Adaptive timestep constants
        TCA_THRESHOLD_S = 4.0  # Switch to micro-steps when TCA < 4 seconds
        MICRO_DT = 0.001  # 1ms micro-timestep for precise detection
        MAX_MICRO_STEPS = 5000  # Safety limit (5 seconds at 1ms)

        # Hit tolerance: extra radius added to ship for hit detection
        # Simulates weapon spread, tracking error, and fire control inaccuracy
        # A destroyer is ~15m radius, so 500m tolerance is ~33x the ship size
        # This makes hits possible even with evasion-induced miss distances
        HIT_TOLERANCE_M = 500.0

        for proj_flight in self.projectiles:
            if proj_flight in projectiles_to_remove:
                continue

            proj = proj_flight.projectile
            target_ship = self.get_ship(proj_flight.target_ship_id) if proj_flight.target_ship_id else None

            if not target_ship or target_ship.is_destroyed:
                # No valid target - just update position normally
                proj.update(dt)
                continue

            # Calculate time to closest approach
            tca, closest_dist = self._calculate_time_to_closest_approach(
                proj.position, proj.velocity,
                target_ship.position, target_ship.velocity
            )

            # Update minimum distance tracking
            current_dist = proj.distance_to(target_ship.position)
            if current_dist < proj_flight.min_distance_to_target:
                proj_flight.min_distance_to_target = current_dist

            # Decide on timestep strategy
            if tca > TCA_THRESHOLD_S:
                # Far from target - use normal timestep
                prev_position = Vector3D(proj.position.x, proj.position.y, proj.position.z)
                proj.update(dt)

                # Quick check: did we pass through target during this step?
                # (Catches cases where projectile is very fast and might skip past)
                hit, impact_point, t_param = self._check_line_cylinder_intersection(
                    prev_position, proj.position, target_ship, HIT_TOLERANCE_M
                )
                if hit:
                    # Hit detected even in coarse step
                    self._resolve_projectile_hit_geometric(
                        proj_flight, target_ship, impact_point
                    )
                    projectiles_to_remove.append(proj_flight)
                    continue

                proj_flight.prev_distance_to_target = current_dist

            else:
                # Close to target - use micro-timesteps for precision
                time_remaining = dt
                micro_steps = 0
                hit_detected = False

                while time_remaining > 0 and micro_steps < MAX_MICRO_STEPS:
                    micro_dt = min(MICRO_DT, time_remaining)

                    # Store previous position for intersection check
                    prev_position = Vector3D(proj.position.x, proj.position.y, proj.position.z)

                    # Update projectile position
                    proj.update(micro_dt)

                    # Update target position (ships move too!)
                    # Note: We can't easily move the ship here since step() handles it
                    # So we use predicted target position based on velocity
                    target_predicted_pos = target_ship.position + target_ship.velocity * (micro_steps * MICRO_DT)

                    # Check for geometric intersection with ship cylinder
                    hit, impact_point, t_param = self._check_line_cylinder_intersection(
                        prev_position, proj.position, target_ship, HIT_TOLERANCE_M
                    )

                    if hit:
                        # Geometric hit! Resolve damage
                        self._resolve_projectile_hit_geometric(
                            proj_flight, target_ship, impact_point
                        )
                        projectiles_to_remove.append(proj_flight)
                        hit_detected = True
                        break

                    # Check if we've passed closest approach (distance increasing)
                    new_dist = proj.distance_to(target_ship.position)
                    if new_dist > current_dist and current_dist < proj_flight.min_distance_to_target + 100:
                        # Past closest approach - it's a miss
                        closest_km = proj_flight.min_distance_to_target / 1000.0
                        flight_time = self.current_time - proj_flight.launch_time
                        self._log_event(SimulationEventType.PROJECTILE_MISS,
                                       proj_flight.source_ship_id, proj_flight.target_ship_id, {
                            'projectile_id': proj_flight.projectile_id,
                            'closest_approach_km': closest_km,
                            'detection': 'geometric',
                            'micro_steps': micro_steps
                        })
                        print(f"  --- [{proj_flight.source_ship_id}] MISS {proj_flight.target_ship_id} "
                              f"(closest: {closest_km:.2f}km, flight: {flight_time:.1f}s)")
                        projectiles_to_remove.append(proj_flight)
                        hit_detected = True  # Well, miss detected
                        break

                    current_dist = new_dist
                    time_remaining -= micro_dt
                    micro_steps += 1

                if not hit_detected:
                    proj_flight.prev_distance_to_target = current_dist

            # Cleanup: remove if projectile is way too far from any target
            max_distance = 5_000_000  # 5000 km
            if proj_flight not in projectiles_to_remove:
                min_dist = min(
                    (proj.distance_to(s.position)
                     for s in self.ships.values()
                     if s.ship_id != proj_flight.source_ship_id),
                    default=max_distance + 1
                )
                if min_dist > max_distance:
                    self._log_event(SimulationEventType.PROJECTILE_MISS,
                                   proj_flight.source_ship_id, proj_flight.target_ship_id, {
                        'projectile_id': proj_flight.projectile_id,
                        'reason': 'too_far'
                    })
                    projectiles_to_remove.append(proj_flight)

        # Remove finished projectiles
        for proj in projectiles_to_remove:
            if proj in self.projectiles:
                self.projectiles.remove(proj)

    def _resolve_projectile_hit_geometric(
        self,
        proj_flight: ProjectileInFlight,
        target: ShipCombatState,
        impact_point: Optional[Vector3D]
    ) -> None:
        """
        Resolve a projectile hit detected via geometric intersection.

        Delegates to the existing _resolve_projectile_hit method which handles
        all armor, modules, and combat mechanics correctly.

        Args:
            proj_flight: The projectile that hit
            target: The ship that was hit
            impact_point: The point of impact (if known, for logging)
        """
        # Use the existing projectile hit resolution which handles all combat mechanics
        self._resolve_projectile_hit(proj_flight, target)

    def _calculate_time_to_closest_approach(
        self,
        proj_pos: Vector3D,
        proj_vel: Vector3D,
        target_pos: Vector3D,
        target_vel: Vector3D
    ) -> tuple[float, float]:
        """
        Calculate time to closest approach (TCA) for a projectile vs target.

        Uses relative kinematics: models target as stationary with relative velocity.
        TCA = -dot(r, v) / dot(v, v) where r = relative position, v = relative velocity

        Args:
            proj_pos: Projectile position
            proj_vel: Projectile velocity
            target_pos: Target position
            target_vel: Target velocity

        Returns:
            Tuple of (time_to_closest_approach_seconds, closest_approach_distance_m)
            TCA is clamped to [0, inf) - negative means already past closest approach.
        """
        # Relative position and velocity
        rel_pos = proj_pos - target_pos
        rel_vel = proj_vel - target_vel

        vel_mag_sq = rel_vel.dot(rel_vel)

        if vel_mag_sq < 1e-10:
            # No relative motion - current distance is closest
            return 0.0, rel_pos.magnitude

        # TCA = -dot(r, v) / |v|^2
        tca = -rel_pos.dot(rel_vel) / vel_mag_sq

        if tca < 0:
            # Already past closest approach
            return 0.0, rel_pos.magnitude

        # Calculate closest approach distance
        closest_pos = rel_pos + rel_vel * tca
        closest_dist = closest_pos.magnitude

        return tca, closest_dist

    def _check_line_cylinder_intersection(
        self,
        line_start: Vector3D,
        line_end: Vector3D,
        ship: ShipCombatState,
        hit_tolerance_m: float = 0.0
    ) -> tuple[bool, Optional[Vector3D], Optional[float]]:
        """
        Check if a line segment intersects a ship's bounding cylinder.

        The ship is modeled as a cylinder aligned with ship.forward.
        Center of cylinder is at ship.position, extending half-length in each direction.

        Args:
            line_start: Start of line segment (previous position)
            line_end: End of line segment (current position)
            ship: The target ship
            hit_tolerance_m: Extra radius added to ship for hit detection (meters).
                           Simulates weapon spread, tracking error, etc.

        Returns:
            Tuple of (hit, impact_point, t_parameter)
            - hit: True if intersection occurred
            - impact_point: Point of intersection (if hit)
            - t_parameter: Parametric position along segment [0,1] where hit occurred
        """
        if ship.geometry is None:
            # Fallback to sphere check with default size
            radius = 50.0 + hit_tolerance_m  # 50m default + tolerance
            hit = self._check_line_sphere_intersection(
                line_start, line_end, ship.position, radius
            )
            if hit:
                return True, ship.position, 0.5
            return False, None, None

        geom = ship.geometry
        length = geom.length_m
        # Add hit tolerance to effective radius (simulates weapon spread, tracking error)
        radius = geom.radius_m + hit_tolerance_m

        # Ship coordinate system
        forward = ship.forward.normalized()
        # Create orthonormal basis (forward is X-axis in ship frame)
        if abs(forward.z) < 0.9:
            up = Vector3D(0, 0, 1)
        else:
            up = Vector3D(1, 0, 0)
        right = forward.cross(up).normalized()
        up = right.cross(forward).normalized()

        # Ship center is at ship.position
        # Cylinder extends from -length/2 to +length/2 along forward axis
        half_length = length / 2.0

        # Transform line segment to ship-local coordinates
        # In ship frame: forward = X, right = Y, up = Z
        def to_local(p: Vector3D) -> tuple[float, float, float]:
            rel = p - ship.position
            x = rel.dot(forward)  # Along ship axis
            y = rel.dot(right)
            z = rel.dot(up)
            return x, y, z

        start_local = to_local(line_start)
        end_local = to_local(line_end)

        # Line direction in local coords
        dx = end_local[0] - start_local[0]
        dy = end_local[1] - start_local[1]
        dz = end_local[2] - start_local[2]

        # Check for infinite cylinder intersection (ignoring end caps)
        # Ray: p = start + t * d
        # Cylinder surface: y^2 + z^2 = r^2
        # Substitute: (sy + t*dy)^2 + (sz + t*dz)^2 = r^2
        # Expand: (dy^2 + dz^2)*t^2 + 2*(sy*dy + sz*dz)*t + (sy^2 + sz^2 - r^2) = 0

        A = dy * dy + dz * dz
        B = 2.0 * (start_local[1] * dy + start_local[2] * dz)
        C = start_local[1] ** 2 + start_local[2] ** 2 - radius ** 2

        # Also check end caps (circles at x = -half_length and x = +half_length)
        intersection_t = None
        intersection_type = None  # 'cylinder' or 'cap'

        if A > 1e-10:
            # Quadratic in t
            discriminant = B * B - 4.0 * A * C

            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                t1 = (-B - sqrt_disc) / (2.0 * A)
                t2 = (-B + sqrt_disc) / (2.0 * A)

                # Check each solution
                for t in [t1, t2]:
                    if 0.0 <= t <= 1.0:
                        # Check if within cylinder length bounds
                        hit_x = start_local[0] + t * dx
                        if -half_length <= hit_x <= half_length:
                            if intersection_t is None or t < intersection_t:
                                intersection_t = t
                                intersection_type = 'cylinder'

        # Check end caps (front at x = half_length, back at x = -half_length)
        if abs(dx) > 1e-10:
            for cap_x in [half_length, -half_length]:
                t_cap = (cap_x - start_local[0]) / dx
                if 0.0 <= t_cap <= 1.0:
                    # Check if within radius
                    hit_y = start_local[1] + t_cap * dy
                    hit_z = start_local[2] + t_cap * dz
                    if hit_y ** 2 + hit_z ** 2 <= radius ** 2:
                        if intersection_t is None or t_cap < intersection_t:
                            intersection_t = t_cap
                            intersection_type = 'cap'

        if intersection_t is not None:
            # Calculate world-space impact point
            impact_point = Vector3D(
                line_start.x + intersection_t * (line_end.x - line_start.x),
                line_start.y + intersection_t * (line_end.y - line_start.y),
                line_start.z + intersection_t * (line_end.z - line_start.z)
            )
            return True, impact_point, intersection_t

        return False, None, None

    def _check_line_sphere_intersection(
        self,
        line_start: Vector3D,
        line_end: Vector3D,
        sphere_center: Vector3D,
        sphere_radius: float
    ) -> bool:
        """
        Check if a line segment intersects a sphere.

        Uses closest point on line segment to sphere center.
        This handles high-speed projectiles passing through targets.

        Args:
            line_start: Start of line segment (previous position)
            line_end: End of line segment (current position)
            sphere_center: Center of target sphere
            sphere_radius: Radius of collision sphere

        Returns:
            True if line segment passes within sphere_radius of center
        """
        # Vector from start to end
        line_vec = line_end - line_start
        line_len_sq = line_vec.dot(line_vec)

        if line_len_sq < 1e-10:
            # Degenerate line segment - just check point distance
            return line_start.distance_to(sphere_center) <= sphere_radius

        # Vector from start to sphere center
        start_to_center = sphere_center - line_start

        # Project sphere center onto line, clamped to segment
        t = max(0.0, min(1.0, start_to_center.dot(line_vec) / line_len_sq))

        # Closest point on segment to sphere center
        closest_point = line_start + line_vec * t

        # Check distance
        distance = closest_point.distance_to(sphere_center)
        return distance <= sphere_radius

    def _resolve_projectile_hit(
        self,
        proj_flight: ProjectileInFlight,
        target: ShipCombatState
    ) -> None:
        """Resolve a projectile hit on a ship."""
        proj = proj_flight.projectile
        source_ship = self.get_ship(proj_flight.source_ship_id)

        # Calculate impact angle for hit location
        impact_vector = proj.velocity.normalized()
        hit_location = HitLocation.LATERAL

        if target.geometry:
            hit_location = target.geometry.calculate_hit_location(
                impact_vector, target.forward
            )

        # Calculate angle for debugging and effective armor
        import math
        angle_deg = math.degrees(impact_vector.angle_to(target.forward))

        # Calculate impact angle from surface normal for armor effectiveness
        # Surface normal varies by hit location:
        # - NOSE: normal points forward (same as ship forward)
        # - TAIL: normal points backward (opposite to ship forward)
        # - LATERAL: normal is radial (perpendicular to ship axis)
        if hit_location == HitLocation.NOSE:
            # For nose hit, normal is ship forward
            # Projectile coming from ahead has velocity opposite to normal
            # Impact angle = 180 - angle_to_forward (0° = perpendicular)
            impact_angle_from_normal = 180.0 - angle_deg
        elif hit_location == HitLocation.TAIL:
            # For tail hit, normal is -ship forward
            # Impact angle = angle_to_forward
            impact_angle_from_normal = angle_deg
        else:
            # For lateral hit, assume roughly perpendicular (simplified)
            # In reality, would depend on exact hit point on cylinder
            impact_angle_from_normal = abs(90.0 - angle_deg) if angle_deg <= 90 else abs(angle_deg - 90.0)

        # Clamp to 0-89 degrees
        impact_angle_from_normal = max(0.0, min(89.0, impact_angle_from_normal))

        # Calculate kinetic energy accounting for PD ablation
        # If PD has ablated some mass, the remaining slug has less energy
        original_mass = proj.mass_kg
        pd_ablation = getattr(proj, '_pd_ablation', 0.0)
        remaining_mass = max(0.1, original_mass - pd_ablation)  # Min 0.1 kg
        mass_fraction = remaining_mass / original_mass if original_mass > 0 else 1.0

        # KE scales linearly with mass: KE = 0.5 * m * v²
        effective_ke_gj = proj.kinetic_energy_gj * mass_fraction

        # Update metrics
        self.metrics.total_hits += 1
        if source_ship:
            source_ship.hits_scored += 1
            source_ship.damage_dealt_gj += effective_ke_gj

        target.damage_taken_gj += effective_ke_gj

        self._log_event(SimulationEventType.PROJECTILE_IMPACT,
                       proj_flight.source_ship_id, target.ship_id, {
            'projectile_id': proj_flight.projectile_id,
            'kinetic_energy_gj': effective_ke_gj,
            'original_energy_gj': proj.kinetic_energy_gj,
            'pd_ablation_kg': pd_ablation,
            'mass_remaining_kg': remaining_mass,
            'hit_location': hit_location.value,
            'impact_vector': impact_vector.to_tuple(),
            'target_forward': target.forward.to_tuple(),
            'angle_deg': angle_deg,
            'impact_angle_from_normal': impact_angle_from_normal
        })

        # Calculate flight time for display
        flight_time = self.current_time - proj_flight.launch_time
        print(f"  >>> [{proj_flight.source_ship_id}] HIT {target.ship_id} "
              f"({hit_location.value}): {effective_ke_gj:.1f} GJ, flight: {flight_time:.1f}s")

        # Apply armor damage using Terra Invicta physics-based ablation
        penetrated = False
        remaining_energy_gj = 0.0
        armor_ablation_cm = 0.0
        energy_absorbed_gj = 0.0

        if target.armor:
            armor_section = target.armor.get_section(hit_location)
            if armor_section:
                # Check for critical hit through chipped armor
                critical_through_chip = armor_section.roll_critical_through_chipping(self.rng)

                # Calculate impact area based on projectile mass
                # High-velocity impacts create larger craters than the projectile size
                # Impact crater spreads energy over larger area, reducing ablation per hit
                # ~30cm diameter for light coilgun (10kg), ~50cm for spinal (88kg)
                impact_area_m2 = 0.1 + (remaining_mass / 100) * 0.1  # 0.1-0.2 m²
                impact_area_m2 = min(0.3, max(0.1, impact_area_m2))  # Clamp to 0.1-0.3 m²

                # Account for impact angle - oblique hits are less effective
                # At 45°, projectile must travel through 1.41x more armor
                # Energy transferred is reduced by cos(angle) factor
                import math
                angle_rad = math.radians(impact_angle_from_normal)
                cos_angle = max(0.1736, math.cos(angle_rad))  # Cap at 80° (5.76x armor)

                # Effective energy is reduced at oblique angles (projectile deflects/skips)
                angle_efficiency = cos_angle  # 0° = 100%, 45° = 71%, 60° = 50%
                angled_ke_gj = effective_ke_gj * angle_efficiency

                # Apply physics-based damage using Terra Invicta formula
                # flat_chipping for coilguns: focused penetrators
                flat_chipping = 0.35  # Standard coilgun chipping

                armor_ablation_cm, energy_absorbed_gj, chipping_added = armor_section.apply_energy_damage(
                    energy_gj=angled_ke_gj,
                    flat_chipping=flat_chipping,
                    impact_area_m2=impact_area_m2
                )

                # Check if armor was penetrated
                penetrated = armor_section.is_penetrated() or critical_through_chip

                if penetrated:
                    remaining_energy_gj = angled_ke_gj - energy_absorbed_gj

        # Log damage taken
        self._log_event(SimulationEventType.DAMAGE_TAKEN, target.ship_id, data={
            'damage_gj': effective_ke_gj,
            'location': hit_location.value,
            'absorbed_gj': energy_absorbed_gj,
            'armor_ablation_cm': armor_ablation_cm,
            'penetrated': penetrated
        })

        if penetrated:
            self._log_event(SimulationEventType.ARMOR_PENETRATED, target.ship_id, data={
                'location': hit_location.value,
                'remaining_damage_gj': remaining_energy_gj
            })

            # Propagate damage through modules
            # High-energy kinetic penetrators travel through the ship
            if target.module_layout and remaining_energy_gj > 0.1:
                # Narrower cone for kinetic projectile (focused penetrator)
                # Spinal coilers are narrow, coilguns slightly wider
                cone_angle = 15.0 if effective_ke_gj > 5.0 else 25.0

                all_modules = target.module_layout.get_modules_in_cone(
                    entry_point=hit_location,
                    angle_deg=cone_angle,
                    direction_vector=impact_vector.to_tuple()
                )

                # Filter out already-destroyed modules - projectile passes through wreckage
                modules = [m for m in all_modules if not m.is_destroyed]

                # Number of modules affected scales with energy
                # 1 GJ = 1-2 modules, 10 GJ = 3-4 modules, 17+ GJ = 5+ modules
                max_modules = min(len(modules), max(2, int(effective_ke_gj / 3) + 1))

                for module in modules[:max_modules]:
                    if remaining_energy_gj < 0.1:
                        break
                    # Each module absorbs some energy based on its size/armor
                    damage_fraction = 0.2 + (module.size_m2 / 50.0) * 0.1  # 20-30% per module
                    damage_to_module = remaining_energy_gj * damage_fraction
                    actual_damage = module.damage(damage_to_module)
                    was_destroyed = module.is_destroyed
                    self._log_event(SimulationEventType.MODULE_DAMAGED, target.ship_id, data={
                        'module_name': module.name,
                        'damage_gj': actual_damage,
                        'destroyed': was_destroyed
                    })
                    if was_destroyed:
                        self._log_event(SimulationEventType.MODULE_DESTROYED, target.ship_id, data={
                            'module_name': module.name
                        })
                        # Disable corresponding weapon if weapon module destroyed
                        self._disable_weapon_for_module(target, module.name)
                    # Energy loss as penetrator travels through ship
                    # Denser modules (reactors, engines) absorb more
                    absorption = 0.4 if module.is_critical else 0.3
                    remaining_energy_gj *= (1.0 - absorption)

        # Check for ship destruction
        self._check_ship_destroyed(target, proj_flight.source_ship_id)

    # -------------------------------------------------------------------------
    # Torpedo Update
    # -------------------------------------------------------------------------

    def _update_torpedoes(self, dt: float) -> None:
        """Update all torpedoes and check for hits."""
        torpedoes_to_remove: list[TorpedoInFlight] = []

        for torp_flight in self.torpedoes:
            torp = torp_flight.torpedo
            target = self.get_ship(torp.target_id)

            if not target or target.is_destroyed:
                torpedoes_to_remove.append(torp_flight)
                continue

            # Disabled torpedoes coast ballistically without guidance
            if torp_flight.is_disabled:
                # Just update position without guidance
                torp.position = torp.position + torp.velocity * dt
                continue

            # Store previous position for sweep detection
            prev_pos = Vector3D(torp.position.x, torp.position.y, torp.position.z)
            prev_target_pos = Vector3D(target.position.x, target.position.y, target.position.z)

            # Update torpedo with appropriate guidance mode
            if torp.guidance_mode == GuidanceMode.COLLISION:
                # COLLISION guidance: align relative velocity with LOS
                guidance = TorpedoGuidance()
                command = guidance.update_collision_guidance(
                    torp, target.position, target.velocity, dt
                )

                # Update time and arming
                torp.time_since_launch += dt
                if not torp.armed and torp.distance_from_launch() >= SAFE_ARMING_DISTANCE_M:
                    torp.armed = True

                # Apply thrust command if not coasting - track delta-v usage
                if command.throttle > 0 and command.direction.magnitude > 0.01:
                    dv_before = torp.remaining_delta_v_kps
                    if command.use_rcs:
                        # Use lateral RCS ONLY for small, precise corrections
                        torp.apply_lateral_thrust(command.direction, dt, command.throttle)
                        dv_used = dv_before - torp.remaining_delta_v_kps
                        torp_flight.rcs_dv_used_kps += dv_used
                    else:
                        # Use main engine
                        torp.apply_thrust(command.direction, dt, command.throttle)
                        dv_used = dv_before - torp.remaining_delta_v_kps
                        torp_flight.main_engine_dv_used_kps += dv_used

                # Apply simultaneous RCS correction if present (combined guidance)
                if command.rcs_direction is not None and command.rcs_throttle > 0:
                    dv_before = torp.remaining_delta_v_kps
                    torp.apply_lateral_thrust(command.rcs_direction, dt, command.rcs_throttle)
                    dv_used = dv_before - torp.remaining_delta_v_kps
                    torp_flight.rcs_dv_used_kps += dv_used

                # Update position (Euler integration)
                torp.position = torp.position + torp.velocity * dt

            elif torp.guidance_mode == GuidanceMode.SMART:
                # SMART guidance: pursuit cone + fuel reserve management
                # Get target's max acceleration for evasion budget calculation
                target_accel_g = target.kinematic_state.max_acceleration_g()

                # Create guidance system and get command
                guidance = TorpedoGuidance()
                command = guidance.update_smart_guidance(
                    torp, target.position, target.velocity, target_accel_g, dt
                )

                # Update time and arming
                torp.time_since_launch += dt
                if not torp.armed and torp.distance_from_launch() >= SAFE_ARMING_DISTANCE_M:
                    torp.armed = True

                # Apply thrust command if not coasting
                if command.throttle > 0 and command.direction.magnitude > 0.01:
                    torp.apply_thrust(command.direction, dt, command.throttle)

                # Update position (Euler integration)
                torp.position = torp.position + torp.velocity * dt
            else:
                # Standard guidance modes (PURSUIT, INTERCEPT, PROPORTIONAL_NAV, etc.)
                torp.update(dt, target.position, target.velocity)

            # Check fuel exhaustion
            if torp.fuel_exhausted and torp.guidance_mode == GuidanceMode.COAST:
                self._log_event(SimulationEventType.TORPEDO_FUEL_EXHAUSTED,
                               torp_flight.source_ship_id, target.ship_id, {
                    'torpedo_id': torp_flight.torpedo_id
                })

            # ================================================================
            # HIT DETECTION - Same logic as kinetic projectiles
            # Torpedoes are kinetic penetrators in terminal phase
            # ================================================================
            current_dist = torp.position.distance_to(target.position)
            prev_dist = torp_flight.prev_distance_to_target

            # Update minimum distance tracking
            if current_dist < torp_flight.min_distance_to_target:
                torp_flight.min_distance_to_target = current_dist

            # Check if at closest approach (distance starting to increase)
            at_closest_approach = (
                prev_dist < float('inf') and
                current_dist > prev_dist and
                torp_flight.min_distance_to_target < 50_000  # Within 50km at some point
            )

            if at_closest_approach and torp.armed and torp_flight not in torpedoes_to_remove:
                # Calculate hit probability at closest approach
                closest_dist_m = torp_flight.min_distance_to_target

                # Get geometry for probability calculation
                geometry = target.geometry
                if geometry is None:
                    geometry = ShipGeometry(length_m=100, beam_m=20, height_m=15)

                # Get cross-section based on approach angle
                approach_dir = (target.position - torp.position).normalized()
                angle_to_forward = math.degrees(target.forward.angle_to(approach_dir))

                if angle_to_forward < 30:
                    cross_section = geometry.nose_cross_section_m2
                elif angle_to_forward > 150:
                    cross_section = geometry.tail_cross_section_m2
                else:
                    cross_section = geometry.lateral_cross_section_m2

                # Effective target radius from cross-section
                target_radius_m = math.sqrt(cross_section / math.pi)

                # Hit probability based on angular size
                if closest_dist_m > 0:
                    angular_size = target_radius_m / closest_dist_m
                else:
                    angular_size = 1.0

                # Torpedoes are GUIDED, so higher accuracy than dumb projectiles
                # Base constant is 115, but torpedoes get 3x for guidance
                accuracy_constant = 345  # 3x better than projectiles

                # If torpedo still has fuel, it can make corrections - even higher accuracy
                if not torp.fuel_exhausted:
                    accuracy_constant *= 1.5  # 50% bonus for active guidance

                hit_probability = 1.0 - math.exp(-accuracy_constant * angular_size)
                hit_probability = max(0.05, min(0.98, hit_probability))

                # Roll for hit
                roll = random.random()
                hit = roll < hit_probability

                if hit:
                    # Hit! Resolve damage
                    self._resolve_torpedo_hit(torp_flight, target)
                    torpedoes_to_remove.append(torp_flight)
                else:
                    # Miss - log with probability info
                    self._log_event(SimulationEventType.PROJECTILE_MISS,
                                   torp_flight.source_ship_id, target.ship_id, {
                        'projectile_id': torp_flight.torpedo_id,
                        'closest_approach_km': closest_dist_m / 1000,
                        'hit_probability': hit_probability,
                        'roll': roll,
                        'type': 'torpedo'
                    })
                    torpedoes_to_remove.append(torp_flight)

            # Update previous distance for next tick
            torp_flight.prev_distance_to_target = current_dist

            # Check if torpedo is too far or timed out
            flight_time = self.current_time - torp_flight.launch_time
            if flight_time > 3600 or current_dist > 10_000_000:  # 1 hour or 10,000 km
                torpedoes_to_remove.append(torp_flight)

        # Remove finished torpedoes and accumulate delta-v stats
        for torp in torpedoes_to_remove:
            if torp in self.torpedoes:
                # Track delta-v usage before removing
                self.metrics.torpedo_main_engine_dv_kps += torp.main_engine_dv_used_kps
                self.metrics.torpedo_rcs_dv_kps += torp.rcs_dv_used_kps
                self.torpedoes.remove(torp)

    def _resolve_torpedo_hit(
        self,
        torp_flight: TorpedoInFlight,
        target: ShipCombatState
    ) -> None:
        """
        Resolve a torpedo hit on a ship.

        Torpedo damage has two phases:
        1. KINETIC - dry mass at impact velocity, applied directionally like a slug
        2. EXPLOSIVE - warhead detonation, spreads in a cone

        The kinetic impact punches through armor first, then the warhead detonates.
        """
        torp = torp_flight.torpedo
        source_ship = self.get_ship(torp_flight.source_ship_id)

        # Calculate impact velocity (relative to target)
        rel_velocity = torp.velocity - target.velocity
        impact_speed_ms = rel_velocity.magnitude
        impact_vector = rel_velocity.normalized() if impact_speed_ms > 0 else torp.velocity.normalized()

        # KINETIC PENETRATOR: Damage from penetrator mass at impact velocity
        # KE = 0.5 * m * v²  (in Joules, convert to GJ)
        # Using penetrator mass (dense material like tungsten/DU), not full dry mass
        penetrator_mass_kg = torp.specs.penetrator_mass_kg
        kinetic_energy_j = 0.5 * penetrator_mass_kg * (impact_speed_ms ** 2)
        kinetic_damage_gj = kinetic_energy_j / 1e9

        # Explosive yield (0 for pure kinetic penetrators)
        explosive_damage_gj = torp.specs.warhead_yield_gj

        # Total damage
        total_damage_gj = kinetic_damage_gj + explosive_damage_gj

        self.metrics.total_torpedo_hits += 1
        self.metrics.total_damage_dealt += total_damage_gj

        if source_ship:
            source_ship.damage_dealt_gj += total_damage_gj

        target.damage_taken_gj += total_damage_gj

        # Determine hit location
        hit_location = HitLocation.LATERAL
        if target.geometry:
            hit_location = target.geometry.calculate_hit_location(
                impact_vector, target.forward
            )

        self._log_event(SimulationEventType.TORPEDO_IMPACT,
                       torp_flight.source_ship_id, target.ship_id, {
            'torpedo_id': torp_flight.torpedo_id,
            'kinetic_damage_gj': kinetic_damage_gj,
            'explosive_damage_gj': explosive_damage_gj,
            'total_damage_gj': total_damage_gj,
            'impact_speed_kps': impact_speed_ms / 1000,
            'penetrator_mass_kg': penetrator_mass_kg,
            'hit_location': hit_location.value
        })

        # Apply armor damage using physics-based ablation
        penetrated = False
        remaining_energy_gj = 0.0

        if target.armor:
            armor_section = target.armor.get_section(hit_location)
            if armor_section:
                initial_thickness = armor_section.thickness_cm
                initial_chipping = armor_section.chipping_fraction

                # Check for critical hit through chipped armor
                critical_through_chip = armor_section.roll_critical_through_chipping(self.rng)

                # PHASE 1: Kinetic penetrator using physics-based damage
                # Impact area spreads as crater forms: ~36cm diameter = 0.1 m²
                kinetic_impact_area = 0.1
                kinetic_ablation, kinetic_absorbed, kinetic_chip = armor_section.apply_energy_damage(
                    energy_gj=kinetic_damage_gj,
                    flat_chipping=0.3,  # Penetrators are focused
                    impact_area_m2=kinetic_impact_area
                )

                self._log_event(SimulationEventType.DAMAGE_TAKEN, target.ship_id, data={
                    'damage_type': 'kinetic',
                    'damage_gj': kinetic_damage_gj,
                    'absorbed_gj': kinetic_absorbed,
                    'location': hit_location.value,
                    'armor_ablation_cm': kinetic_ablation,
                    'chipping_added': kinetic_chip
                })

                # PHASE 2: Explosive warhead (if any)
                explosive_ablation = 0.0
                explosive_absorbed = 0.0
                if explosive_damage_gj > 0.01:
                    # Explosive has larger impact area (spreads damage)
                    explosive_impact_area = 0.5
                    explosive_ablation, explosive_absorbed, explosive_chip = armor_section.apply_energy_damage(
                        energy_gj=explosive_damage_gj,
                        flat_chipping=0.5,  # Explosives chip more
                        impact_area_m2=explosive_impact_area
                    )

                    self._log_event(SimulationEventType.DAMAGE_TAKEN, target.ship_id, data={
                        'damage_type': 'explosive',
                        'damage_gj': explosive_damage_gj,
                        'absorbed_gj': explosive_absorbed,
                        'location': hit_location.value,
                        'armor_ablation_cm': explosive_ablation,
                        'chipping_added': explosive_chip
                    })

                # Check if armor was penetrated (either through ablation or chipping)
                penetrated = armor_section.is_penetrated() or critical_through_chip

                if penetrated:
                    # Calculate remaining energy that penetrates
                    total_absorbed = kinetic_absorbed + explosive_absorbed
                    remaining_energy_gj = total_damage_gj - total_absorbed

                    self._log_event(SimulationEventType.ARMOR_PENETRATED, target.ship_id, data={
                        'location': hit_location.value,
                        'remaining_energy_gj': remaining_energy_gj,
                        'critical_through_chip': critical_through_chip,
                        'chipping_fraction': armor_section.chipping_fraction
                    })

        # Propagate internal damage if penetrated
        # Torpedoes cause massive internal damage - kinetic penetrator + explosive
        if target.module_layout and penetrated and remaining_energy_gj > 0.1:
            # Wide cone for torpedo - explosive blast spreads damage
            all_modules = target.module_layout.get_modules_in_cone(
                entry_point=hit_location,
                angle_deg=45.0,  # Wide cone for torpedo explosive
                direction_vector=impact_vector.to_tuple()
            )

            # Filter out already-destroyed modules
            modules = [m for m in all_modules if not m.is_destroyed]

            # Torpedoes affect many modules due to massive energy
            # 10 GJ = 4 modules, 20 GJ = 7 modules, 35 GJ = 12 modules
            max_modules = min(len(modules), max(3, int(total_damage_gj / 3) + 1))

            for module in modules[:max_modules]:
                if remaining_energy_gj < 0.1:
                    break
                # Torpedoes deal heavy damage - explosive + kinetic
                damage_fraction = 0.25 + (module.size_m2 / 40.0) * 0.15  # 25-40% per module
                damage_to_module = remaining_energy_gj * damage_fraction
                actual_damage = module.damage(damage_to_module)
                was_destroyed = module.is_destroyed
                self._log_event(SimulationEventType.MODULE_DAMAGED, target.ship_id, data={
                    'module_name': module.name,
                    'damage_gj': actual_damage,
                    'destroyed': was_destroyed
                })
                if was_destroyed:
                    self._log_event(SimulationEventType.MODULE_DESTROYED, target.ship_id, data={
                        'module_name': module.name
                    })
                    # Disable corresponding weapon if weapon module destroyed
                    self._disable_weapon_for_module(target, module.name)
                # Energy dissipates as blast travels through ship
                # Critical modules (armored) absorb more blast
                absorption = 0.35 if module.is_critical else 0.25
                remaining_energy_gj *= (1.0 - absorption)

        self._check_ship_destroyed(target, torp_flight.source_ship_id)

    # -------------------------------------------------------------------------
    # Point Defense Update - Smart Targeting System
    # -------------------------------------------------------------------------

    # Target type priorities (lower = higher priority)
    PD_PRIORITY_SLUG_INTERCEPT = 1       # Slugs on collision course - fastest, hardest
    PD_PRIORITY_TORPEDO_COLLISION = 2    # Torpedoes on collision course
    PD_PRIORITY_TORPEDO_MANEUVERING = 3  # Torpedoes with remaining delta-v
    PD_PRIORITY_ALLIED_DEFENSE = 4       # Projectiles headed to allied ships
    PD_PRIORITY_ENEMY_SHIP = 10          # Enemy ships (lowest priority)

    def _update_point_defense(self, dt: float) -> None:
        """
        Update all point defense systems with coordinated targeting.

        PD turrets automatically engage threats with smart prioritization:
        1. Slugs on intercept/collision course (fastest, hardest to track)
        2. Torpedoes on collision course or with remaining delta-v
        3. Projectiles headed toward allied ships
        4. Closest enemy ship in range (if no other targets)

        Turrets coordinate to avoid overkill - spreading fire across targets.
        """
        for ship in self.ships.values():
            if ship.is_destroyed or not ship.point_defense:
                continue

            # Update PD cooldowns
            for pd in ship.point_defense:
                pd.update(dt)

            # Build prioritized target list for this ship
            targets = self._build_pd_target_list(ship)

            # Coordinate turret assignments to avoid overkill
            turret_assignments = self._coordinate_pd_turrets(ship, targets, dt)

            # Execute engagements
            for pd, target_info in turret_assignments:
                if target_info is None:
                    continue
                self._pd_execute_engagement(ship, pd, target_info, dt)

    def _build_pd_target_list(self, ship: ShipCombatState) -> list[dict]:
        """
        Build prioritized list of all potential PD targets.

        Returns list of dicts with:
            - target_type: 'torpedo', 'projectile', 'ship'
            - target: The actual target object
            - priority: Priority level (lower = more urgent)
            - distance_km: Distance to target
            - time_to_impact: Estimated time to impact (if applicable)
            - heat_to_kill: Estimated heat/damage needed to destroy
            - turrets_assigned: Number of turrets already assigned this tick
        """
        targets: list[dict] = []

        # Get allied ships for defense calculations
        allied_ships = self.get_friendly_ships(ship.ship_id)

        # Assess all torpedoes
        for torp_flight in self.torpedoes:
            if torp_flight.is_disabled:
                continue

            # Don't target own faction's torpedoes
            source_ship = self.get_ship(torp_flight.source_ship_id)
            if source_ship and source_ship.faction == ship.faction:
                continue

            torp = torp_flight.torpedo
            distance_m = torp.position.distance_to(ship.position)
            distance_km = distance_m / 1000

            # Check if torpedo threatens this ship or allies
            threatens_self = self._threatens_ship(torp.position, torp.velocity, ship.position)
            target_is_self = torp.target_id == ship.ship_id

            # Check if threatens allied ships
            threatens_ally = False
            allied_target = None
            for ally in allied_ships:
                if self._threatens_ship(torp.position, torp.velocity, ally.position):
                    threatens_ally = True
                    allied_target = ally
                    break

            if not (threatens_self or target_is_self or threatens_ally):
                continue

            # Determine priority based on threat level
            has_delta_v = not torp.fuel_exhausted
            on_collision = self._on_collision_course(
                torp.position, torp.velocity, ship.position, ship.velocity
            )

            if on_collision:
                priority = self.PD_PRIORITY_TORPEDO_COLLISION
            elif has_delta_v:
                priority = self.PD_PRIORITY_TORPEDO_MANEUVERING
            else:
                priority = self.PD_PRIORITY_ALLIED_DEFENSE

            # Estimate time to impact
            closing_speed = self._calculate_closing_speed(
                torp.position, torp.velocity, ship.position, ship.velocity
            )
            time_to_impact = distance_m / max(closing_speed, 1.0) if closing_speed > 0 else float('inf')

            # Estimate heat needed to destroy (100 kJ threshold minus already absorbed)
            heat_to_kill = max(0, torp_flight.WARHEAD_THRESHOLD_J - torp_flight.heat_absorbed_j)

            targets.append({
                'target_type': 'torpedo',
                'target': torp_flight,
                'target_id': torp_flight.torpedo_id,
                'priority': priority,
                'distance_km': distance_km,
                'time_to_impact': time_to_impact,
                'heat_to_kill': heat_to_kill,
                'turrets_assigned': 0,
                'allied_target': allied_target
            })

        # Assess all projectiles (slugs)
        for proj_flight in self.projectiles:
            # Don't target own faction's projectiles
            source_ship = self.get_ship(proj_flight.source_ship_id)
            if source_ship and source_ship.faction == ship.faction:
                continue

            proj = proj_flight.projectile
            distance_m = proj.position.distance_to(ship.position)
            distance_km = distance_m / 1000

            # Check collision course with self
            on_collision_self = self._on_collision_course(
                proj.position, proj.velocity, ship.position, ship.velocity
            )

            # Check collision course with allies
            threatens_ally = False
            allied_target = None
            for ally in allied_ships:
                if self._on_collision_course(proj.position, proj.velocity, ally.position, ally.velocity):
                    threatens_ally = True
                    allied_target = ally
                    break

            if not (on_collision_self or threatens_ally):
                continue

            # Slugs on intercept are highest priority (fast, hard to track)
            if on_collision_self:
                priority = self.PD_PRIORITY_SLUG_INTERCEPT
            else:
                priority = self.PD_PRIORITY_ALLIED_DEFENSE

            # Time to impact
            closing_speed = self._calculate_closing_speed(
                proj.position, proj.velocity, ship.position, ship.velocity
            )
            time_to_impact = distance_m / max(closing_speed, 1.0) if closing_speed > 0 else float('inf')

            # Estimate mass to ablate
            slug_mass = getattr(proj, 'mass_kg', 50.0)
            already_ablated = getattr(proj, '_pd_ablation', 0.0)
            mass_to_kill = max(0, slug_mass - already_ablated)

            targets.append({
                'target_type': 'projectile',
                'target': proj_flight,
                'target_id': proj_flight.projectile_id,
                'priority': priority,
                'distance_km': distance_km,
                'time_to_impact': time_to_impact,
                'mass_to_kill': mass_to_kill,
                'turrets_assigned': 0,
                'allied_target': allied_target
            })

        # Add enemy ships as lowest priority targets
        for enemy in self.get_enemy_ships(ship.ship_id):
            if enemy.is_destroyed:
                continue

            distance_km = ship.position.distance_to(enemy.position) / 1000

            targets.append({
                'target_type': 'ship',
                'target': enemy,
                'target_id': enemy.ship_id,
                'priority': self.PD_PRIORITY_ENEMY_SHIP,
                'distance_km': distance_km,
                'time_to_impact': float('inf'),
                'turrets_assigned': 0
            })

        # Sort by priority, then by time to impact (most urgent first)
        targets.sort(key=lambda t: (t['priority'], t['time_to_impact'], t['distance_km']))

        return targets

    def _coordinate_pd_turrets(
        self,
        ship: ShipCombatState,
        targets: list[dict],
        dt: float
    ) -> list[tuple[PDLaserState, Optional[dict]]]:
        """
        Coordinate PD turret assignments to avoid overkill.

        Distributes turrets across targets efficiently:
        - Estimates damage per turret per engagement
        - Assigns enough turrets to kill each target
        - Spreads remaining turrets to other targets

        Returns:
            List of (turret, target_info) tuples
        """
        assignments: list[tuple[PDLaserState, Optional[dict]]] = []
        available_turrets = [pd for pd in ship.point_defense if pd.can_fire()]
        assigned_turrets: set[int] = set()  # Track by index

        if not available_turrets or not targets:
            for pd in ship.point_defense:
                assignments.append((pd, None))
            return assignments

        # Calculate damage per turret at various ranges
        for target_info in targets:
            target_info['turrets_needed'] = self._estimate_turrets_needed(
                ship, target_info, dt
            )

        # First pass: assign turrets to highest priority targets
        for target_info in targets:
            if len(assigned_turrets) >= len(available_turrets):
                break

            distance_km = target_info['distance_km']

            # Check if any turret can reach this target
            reachable_indices = [
                i for i, pd in enumerate(available_turrets)
                if pd.laser.is_in_range(distance_km) and i not in assigned_turrets
            ]

            if not reachable_indices:
                continue

            # Determine how many turrets to assign
            turrets_needed = target_info['turrets_needed']
            already_assigned = target_info['turrets_assigned']
            turrets_to_assign = min(
                turrets_needed - already_assigned,
                len(reachable_indices)
            )

            # For low-priority targets (enemy ships), only use 1 turret
            if target_info['priority'] >= self.PD_PRIORITY_ENEMY_SHIP:
                turrets_to_assign = min(1, turrets_to_assign)

            # Assign turrets
            for i in range(turrets_to_assign):
                idx = reachable_indices[i]
                pd = available_turrets[idx]
                assignments.append((pd, target_info))
                target_info['turrets_assigned'] += 1
                assigned_turrets.add(idx)

        # Assign remaining available turrets to None (no target)
        for i, pd in enumerate(available_turrets):
            if i not in assigned_turrets:
                assignments.append((pd, None))

        # Also include turrets that couldn't fire (on cooldown)
        for pd in ship.point_defense:
            if not pd.can_fire():
                assignments.append((pd, None))

        return assignments

    def _estimate_turrets_needed(
        self,
        ship: ShipCombatState,
        target_info: dict,
        dt: float
    ) -> int:
        """Estimate how many turrets needed to kill target before impact."""
        target_type = target_info['target_type']
        distance_km = target_info['distance_km']
        time_to_impact = target_info['time_to_impact']

        if not ship.point_defense:
            return 0

        # Get representative PD laser for calculations
        pd = ship.point_defense[0]

        if target_type == 'torpedo':
            heat_to_kill = target_info.get('heat_to_kill', 100_000)
            # Heat per turret per second at this range
            engagement = PDEngagement(pd.laser)
            heat_per_shot = engagement.calculate_heat_transfer(
                pd.laser.power_w, distance_km, pd.laser.cooldown_s
            )
            shots_to_kill = heat_to_kill / max(heat_per_shot, 1)
            time_to_kill_one_turret = shots_to_kill * pd.laser.cooldown_s

            # How many turrets to kill before impact?
            if time_to_impact <= 0:
                return len(ship.point_defense)
            turrets_needed = max(1, int(time_to_kill_one_turret / time_to_impact) + 1)
            return min(turrets_needed, len(ship.point_defense))

        elif target_type == 'projectile':
            mass_to_kill = target_info.get('mass_to_kill', 50.0)
            # Ablation rate per turret
            ablation_rate = pd.laser.calculate_ablation_rate(distance_km)
            ablation_per_shot = ablation_rate * pd.laser.cooldown_s
            shots_to_kill = mass_to_kill / max(ablation_per_shot, 0.001)
            time_to_kill_one_turret = shots_to_kill * pd.laser.cooldown_s

            if time_to_impact <= 0:
                return len(ship.point_defense)
            turrets_needed = max(1, int(time_to_kill_one_turret / time_to_impact) + 1)
            return min(turrets_needed, len(ship.point_defense))

        else:  # enemy ship
            return 1  # Only use 1 turret for harassing enemy ships

    def _pd_execute_engagement(
        self,
        ship: ShipCombatState,
        pd: PDLaserState,
        target_info: dict,
        dt: float
    ) -> None:
        """Execute a PD engagement based on target info."""
        target_type = target_info['target_type']

        if target_type == 'torpedo':
            self._pd_engage_torpedo(ship, pd, target_info['target'], dt)
        elif target_type == 'projectile':
            self._pd_engage_projectile(ship, pd, target_info['target'], dt)
        elif target_type == 'ship':
            self._pd_engage_enemy_ship(ship, pd, target_info['target'], dt)

    def _threatens_ship(
        self,
        threat_pos: Vector3D,
        threat_vel: Vector3D,
        ship_pos: Vector3D,
        threat_range_m: float = 500_000
    ) -> bool:
        """Check if a threat is approaching a ship."""
        distance = threat_pos.distance_to(ship_pos)
        if distance > threat_range_m:
            return False
        return self._is_closing(threat_pos, threat_vel, ship_pos)

    def _on_collision_course(
        self,
        obj_pos: Vector3D,
        obj_vel: Vector3D,
        target_pos: Vector3D,
        target_vel: Vector3D,
        miss_threshold_m: float = 5000.0  # 5 km - matches hit detection range
    ) -> bool:
        """
        Check if object is on collision course with target.

        Uses closest point of approach (CPA) calculation.
        The 5km threshold accounts for hit probability at close range.
        """
        # Relative position and velocity
        rel_pos = obj_pos - target_pos
        rel_vel = obj_vel - target_vel

        rel_speed_sq = rel_vel.dot(rel_vel)
        if rel_speed_sq < 1e-10:
            # Not moving relative to each other
            return rel_pos.magnitude < miss_threshold_m

        # Time to closest point of approach
        t_cpa = -rel_pos.dot(rel_vel) / rel_speed_sq

        if t_cpa < 0:
            # CPA is in the past, object moving away
            return False

        # Position at CPA
        cpa_pos = rel_pos + rel_vel * t_cpa
        miss_distance = cpa_pos.magnitude

        return miss_distance < miss_threshold_m

    def _calculate_closing_speed(
        self,
        obj_pos: Vector3D,
        obj_vel: Vector3D,
        target_pos: Vector3D,
        target_vel: Vector3D
    ) -> float:
        """Calculate closing speed between object and target."""
        rel_pos = target_pos - obj_pos
        rel_vel = obj_vel - target_vel
        distance = rel_pos.magnitude

        if distance < 1.0:
            return 0.0

        # Closing speed is component of relative velocity toward target
        return rel_vel.dot(rel_pos.normalized())

    def _is_closing(self, pos: Vector3D, vel: Vector3D, target_pos: Vector3D) -> bool:
        """Check if object is closing on target."""
        to_target = target_pos - pos
        return vel.dot(to_target) > 0

    def _pd_engage_torpedo(
        self,
        ship: ShipCombatState,
        pd: PDLaserState,
        torp_flight: TorpedoInFlight,
        dt: float
    ) -> None:
        """Engage a torpedo with point defense laser."""
        torp = torp_flight.torpedo
        distance_km = torp.position.distance_to(ship.position) / 1000

        if not pd.laser.is_in_range(distance_km):
            return

        # Fire PD laser
        if not pd.engage():
            return

        # Calculate heat delivered (exposure time = cooldown period)
        exposure_time = pd.laser.cooldown_s
        engagement = PDEngagement(pd.laser)
        heat_delivered = engagement.calculate_heat_transfer(
            pd.laser.power_w, distance_km, exposure_time
        )

        # Apply damage to torpedo
        pd.current_target_id = torp_flight.torpedo_id
        pd.heat_delivered_j += heat_delivered
        destroyed = torp_flight.absorb_pd_heat(heat_delivered)

        # Log engagement
        self._log_event(SimulationEventType.PD_ENGAGED, ship.ship_id, data={
            'turret': pd.turret_name,
            'target_type': 'torpedo',
            'target_id': torp_flight.torpedo_id,
            'distance_km': distance_km,
            'heat_delivered_j': heat_delivered,
            'total_heat_j': torp_flight.heat_absorbed_j
        })

        # Check results
        if destroyed:
            self._log_event(SimulationEventType.PD_TORPEDO_DESTROYED, ship.ship_id, data={
                'torpedo_id': torp_flight.torpedo_id,
                'source_ship': torp_flight.source_ship_id,
                'total_heat_absorbed_j': torp_flight.heat_absorbed_j
            })
            self.metrics.total_torpedo_intercepted += 1
            ship.pd_intercepts += 1
            pd.reset_target()

            # Remove torpedo from simulation
            if torp_flight in self.torpedoes:
                self.torpedoes.remove(torp_flight)

        elif torp_flight.is_disabled:
            self._log_event(SimulationEventType.PD_TORPEDO_DISABLED, ship.ship_id, data={
                'torpedo_id': torp_flight.torpedo_id,
                'source_ship': torp_flight.source_ship_id
            })

    def _pd_engage_projectile(
        self,
        ship: ShipCombatState,
        pd: PDLaserState,
        proj_flight: ProjectileInFlight,
        dt: float
    ) -> None:
        """
        Engage a kinetic projectile with point defense laser.

        Note: Projectiles are harder to destroy than torpedoes - need sustained
        ablation to vaporize the slug.
        """
        proj = proj_flight.projectile
        distance_km = proj.position.distance_to(ship.position) / 1000

        if not pd.laser.is_in_range(distance_km):
            return

        # Fire PD laser
        if not pd.engage():
            return

        # Calculate ablation (slugs need sustained fire to destroy)
        exposure_time = pd.laser.cooldown_s
        ablation_rate = pd.laser.calculate_ablation_rate(distance_km)
        mass_ablated = ablation_rate * exposure_time

        # Track cumulative damage to this projectile
        if not hasattr(proj, '_pd_ablation'):
            proj._pd_ablation = 0.0
        proj._pd_ablation += mass_ablated

        # Log engagement
        self._log_event(SimulationEventType.PD_ENGAGED, ship.ship_id, data={
            'turret': pd.turret_name,
            'target_type': 'slug',
            'target_id': proj_flight.projectile_id,
            'distance_km': distance_km,
            'mass_ablated_kg': mass_ablated,
            'total_ablated_kg': proj._pd_ablation
        })

        # Check if slug is destroyed (assume 50kg slug)
        slug_mass_kg = getattr(proj, 'mass_kg', 50.0)
        if proj._pd_ablation >= slug_mass_kg:
            self._log_event(SimulationEventType.PD_SLUG_DESTROYED, ship.ship_id, data={
                'projectile_id': proj_flight.projectile_id,
                'source_ship': proj_flight.source_ship_id
            })
            ship.pd_intercepts += 1

            # Remove projectile from simulation
            if proj_flight in self.projectiles:
                self.projectiles.remove(proj_flight)
        else:
            self._log_event(SimulationEventType.PD_SLUG_DAMAGED, ship.ship_id, data={
                'projectile_id': proj_flight.projectile_id,
                'remaining_mass_kg': slug_mass_kg - proj._pd_ablation
            })

    def _pd_engage_enemy_ship(
        self,
        ship: ShipCombatState,
        pd: PDLaserState,
        target: ShipCombatState,
        dt: float
    ) -> None:
        """
        Engage an enemy ship with point defense laser.

        This is lowest priority - only happens when no projectiles or torpedoes
        to engage. PD lasers are ineffective against ship armor but can cause
        minor surface heating and sensor interference.
        """
        distance_km = ship.position.distance_to(target.position) / 1000

        if not pd.laser.is_in_range(distance_km):
            return

        # Fire PD laser
        if not pd.engage():
            return

        # Calculate energy delivered (minimal effect against armor)
        exposure_time = pd.laser.cooldown_s
        energy_delivered_j = pd.laser.power_w * exposure_time

        # Log engagement
        self._log_event(SimulationEventType.PD_ENGAGED, ship.ship_id, target.ship_id, {
            'turret': pd.turret_name,
            'target_type': 'ship',
            'target_id': target.ship_id,
            'distance_km': distance_km,
            'energy_delivered_j': energy_delivered_j
        })

        # PD lasers do minimal damage to ships - mostly harassment
        # At 5 MW and 100 km, intensity is too low to ablate armor
        # But at very close range (<10 km), could cause some heating
        if distance_km < 10.0 and target.armor:
            # Very minor armor heating effect
            minor_damage_gj = energy_delivered_j / 1e12  # Negligible
            # Not enough to actually damage - just logged for awareness

    # -------------------------------------------------------------------------
    # Damage Propagation
    # -------------------------------------------------------------------------

    def _propagate_internal_damage(
        self,
        ship: ShipCombatState,
        hit_result: HitResult,
        impact_vector: Vector3D
    ) -> None:
        """Propagate penetrating damage through ship modules."""
        if not ship.module_layout or not hit_result.location:
            return

        # Create damage cone
        entry_point = ship.position  # Simplified
        damage_cone = DamageCone.from_weapon_type(
            entry_point=entry_point,
            direction=impact_vector,
            energy_gj=hit_result.remaining_damage_gj,
            is_missile=False
        )

        # Propagate damage
        results = self.damage_propagator.propagate(damage_cone, ship.module_layout)

        for result in results:
            self._log_event(SimulationEventType.MODULE_DAMAGED, ship.ship_id, data={
                'module_name': result.module_name,
                'damage_gj': result.damage_taken_gj,
                'health_remaining': result.health_after,
                'destroyed': result.destroyed
            })

            if result.destroyed:
                self._log_event(SimulationEventType.MODULE_DESTROYED, ship.ship_id, data={
                    'module_name': result.module_name
                })
                self._disable_weapon_for_module(ship, result.module_name)

    def _disable_weapon_for_module(self, ship: ShipCombatState, module_name: str) -> None:
        """
        Disable the weapon corresponding to a destroyed weapon module.

        Maps module names to weapon slots and disables them.
        """
        # Map module names to weapon slots
        module_to_weapon = {
            "Spinal Coiler Mount": "weapon_0",
            "Dorsal Turret Mount": "weapon_1",
            "PD Laser Dorsal": "pd_0",
            "PD Laser Ventral": "pd_1",
        }

        weapon_slot = module_to_weapon.get(module_name)
        if weapon_slot and weapon_slot in ship.weapons:
            ship.weapons[weapon_slot].is_operational = False
            print(f"  !!! [{ship.ship_id}] {module_name} DESTROYED - {weapon_slot} disabled")

        # Also check PD lasers
        if weapon_slot and hasattr(ship, 'pd_lasers') and ship.pd_lasers:
            if weapon_slot in ship.pd_lasers:
                ship.pd_lasers[weapon_slot].is_operational = False

    def _check_ship_destroyed(self, ship: ShipCombatState, attacker_id: Optional[str]) -> None:
        """Check if a ship has been destroyed."""
        if ship.is_destroyed:
            return

        destroyed = False

        # Check for critical module destruction
        if ship.module_layout and ship.module_layout.has_critical_damage:
            destroyed = True

        # Check for overall hull integrity
        if ship.hull_integrity <= 0:
            destroyed = True

        if destroyed:
            ship.is_destroyed = True
            ship.kill_credit = attacker_id
            self.metrics.ships_destroyed.append(ship.ship_id)

            self._log_event(SimulationEventType.SHIP_DESTROYED, ship.ship_id, data={
                'killer_id': attacker_id,
                'hull_integrity': ship.hull_integrity
            })

    # -------------------------------------------------------------------------
    # Decision Points
    # -------------------------------------------------------------------------

    def _trigger_decision_points(self) -> None:
        """Trigger decision callbacks for all active ships."""
        self._log_event(SimulationEventType.DECISION_POINT_REACHED, data={
            'time': self.current_time
        })

        if not self._decision_callback:
            return

        for ship in self.ships.values():
            if ship.is_destroyed:
                continue

            try:
                commands = self._decision_callback(ship.ship_id, self)
                if commands:
                    for cmd in commands:
                        self.inject_command(ship.ship_id, cmd)
                        self._log_event(SimulationEventType.COMMAND_ISSUED, ship.ship_id, data={
                            'command': str(type(cmd).__name__)
                        })
            except Exception as e:
                # Log error but don't crash simulation
                self._log_event(SimulationEventType.COMMAND_ISSUED, ship.ship_id, data={
                    'error': str(e)
                })

    # -------------------------------------------------------------------------
    # Battle End Check
    # -------------------------------------------------------------------------

    def _check_battle_end(self) -> None:
        """Check if the battle should end."""
        active_factions: set[str] = set()

        for ship in self.ships.values():
            if not ship.is_destroyed:
                active_factions.add(ship.faction)

        if len(active_factions) <= 1:
            self._running = False

    # -------------------------------------------------------------------------
    # Event Logging
    # -------------------------------------------------------------------------

    def add_event_callback(self, callback: Callable[['SimulationEvent'], None]) -> None:
        """
        Register a callback to be called for each simulation event.

        Args:
            callback: Function that takes a SimulationEvent.
        """
        self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: Callable[['SimulationEvent'], None]) -> None:
        """Remove an event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    def _log_event(
        self,
        event_type: SimulationEventType,
        ship_id: Optional[str] = None,
        target_id: Optional[str] = None,
        data: Optional[dict] = None
    ) -> SimulationEvent:
        """Log a simulation event and notify callbacks."""
        event = SimulationEvent(
            event_type=event_type,
            timestamp=self.current_time,
            ship_id=ship_id,
            target_id=target_id,
            data=data or {}
        )
        self.events.append(event)

        # Notify event callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"[SIM] Event callback error: {e}")

        return event

    # -------------------------------------------------------------------------
    # State Snapshot
    # -------------------------------------------------------------------------

    def get_battle_snapshot(self, ship_id: str) -> dict:
        """
        Get a snapshot of the battle state from a ship's perspective.

        This is the data that would be passed to an LLM for decision making.

        Args:
            ship_id: ID of the ship requesting the snapshot.

        Returns:
            Dict containing battle state information.
        """
        ship = self.get_ship(ship_id)
        if not ship:
            return {}

        enemies = self.get_enemy_ships(ship_id)
        friendlies = self.get_friendly_ships(ship_id)

        # Find closest enemy
        closest_enemy = None
        closest_distance = float('inf')
        for enemy in enemies:
            dist = ship.distance_to(enemy)
            if dist < closest_distance:
                closest_distance = dist
                closest_enemy = enemy

        # Get incoming threats
        incoming_torpedoes = [
            {
                'torpedo_id': t.torpedo_id,
                'source_ship_id': t.source_ship_id,
                'position': t.torpedo.position.to_tuple(),
                'velocity': t.torpedo.velocity.magnitude / 1000,
                'distance_km': ship.position.distance_to(t.torpedo.position) / 1000,
                'armed': t.torpedo.armed
            }
            for t in self.torpedoes
            if t.torpedo.target_id == ship_id
        ]

        return {
            'timestamp': self.current_time,
            'own_ship': {
                'ship_id': ship.ship_id,
                'ship_type': ship.ship_type,
                'position_km': (ship.position / 1000).to_tuple(),
                'velocity_kps': (ship.velocity / 1000).magnitude,
                'forward': ship.forward.to_tuple(),
                'delta_v_remaining_kps': ship.remaining_delta_v_kps,
                'heat_percent': ship.heat_percent,
                'hull_integrity': ship.hull_integrity,
                'weapons': {
                    slot: {
                        'ammo': ws.ammo_remaining,
                        'cooldown': ws.cooldown_remaining,
                        'operational': ws.is_operational
                    }
                    for slot, ws in ship.weapons.items()
                }
            },
            'enemies': [
                {
                    'ship_id': e.ship_id,
                    'ship_type': e.ship_type,
                    'position_km': (e.position / 1000).to_tuple(),
                    'velocity_kps': (e.velocity / 1000).magnitude,
                    'distance_km': ship.distance_to(e) / 1000,
                    'closing_rate_kps': ship.closing_rate_to(e) / 1000,
                    'hull_integrity': e.hull_integrity
                }
                for e in enemies
            ],
            'friendlies': [
                {
                    'ship_id': f.ship_id,
                    'ship_type': f.ship_type,
                    'distance_km': ship.distance_to(f) / 1000
                }
                for f in friendlies
            ],
            'incoming_threats': incoming_torpedoes,
            'engagement_range_km': closest_distance / 1000 if closest_enemy else 0,
            'primary_target_id': ship.primary_target_id,
            'current_maneuver': ship.current_maneuver.maneuver_type.name if ship.current_maneuver else None
        }

    # -------------------------------------------------------------------------
    # Event History
    # -------------------------------------------------------------------------

    def get_events_since(self, since_time: float) -> list[SimulationEvent]:
        """Get all events since a given time."""
        return [e for e in self.events if e.timestamp >= since_time]

    def get_events_for_ship(self, ship_id: str) -> list[SimulationEvent]:
        """Get all events involving a specific ship."""
        return [
            e for e in self.events
            if e.ship_id == ship_id or e.target_id == ship_id
        ]

    def get_events_by_type(self, event_type: SimulationEventType) -> list[SimulationEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    # -------------------------------------------------------------------------
    # LLM Sensor Report
    # -------------------------------------------------------------------------

    def generate_sensor_report(self, ship_id: str) -> str:
        """
        Generate a human-readable sensor report for LLM decision making.

        This produces a clear, structured text report that an LLM can easily
        parse and use to make tactical decisions. The format is designed for
        clarity and actionability.

        Args:
            ship_id: ID of the ship requesting the report.

        Returns:
            Formatted string containing tactical situation report.
        """
        snapshot = self.get_battle_snapshot(ship_id)
        if not snapshot:
            return f"ERROR: Ship {ship_id} not found in simulation."

        ship = self.get_ship(ship_id)
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append(f"TACTICAL SENSOR REPORT - {ship_id.upper()}")
        lines.append(f"Time: T+{snapshot['timestamp']:.1f}s")
        lines.append("=" * 60)

        # Own ship status
        own = snapshot['own_ship']
        lines.append("")
        lines.append("YOUR SHIP STATUS:")
        lines.append(f"  Class: {own['ship_type'].title()}")
        lines.append(f"  Position: ({own['position_km'][0]:.1f}, {own['position_km'][1]:.1f}, {own['position_km'][2]:.1f}) km")
        lines.append(f"  Speed: {own['velocity_kps']:.2f} km/s")
        lines.append(f"  Hull Integrity: {own['hull_integrity']:.1f}%")
        lines.append(f"  Heat Level: {own['heat_percent']:.1f}%")
        lines.append(f"  Delta-V Remaining: {own['delta_v_remaining_kps']:.1f} km/s")

        # Armor status
        if ship and ship.armor:
            lines.append("")
            lines.append("  ARMOR:")
            for loc in [HitLocation.NOSE, HitLocation.LATERAL, HitLocation.TAIL]:
                section = ship.armor.get_section(loc)
                if section:
                    lines.append(f"    {loc.value.title()}: {section.thickness_cm:.1f}cm ({section.protection_percent:.0f}% protection)")

        # Weapons status
        lines.append("")
        lines.append("  WEAPONS:")
        for slot, weapon_data in own['weapons'].items():
            status = "READY" if weapon_data['operational'] and weapon_data['cooldown'] <= 0 else "RELOADING"
            if not weapon_data['operational']:
                status = "DAMAGED"
            lines.append(f"    {slot}: {status} (Ammo: {weapon_data['ammo']}, Cooldown: {weapon_data['cooldown']:.1f}s)")

        # Torpedo launcher
        if ship and ship.torpedo_launcher:
            torp_status = "READY" if ship.torpedo_launcher.can_launch(self.current_time) else "RELOADING"
            lines.append(f"    torpedo_launcher: {torp_status} (Torpedoes: {ship.torpedo_launcher.torpedoes_remaining})")

        # Point defense
        if ship and ship.point_defense:
            pd_ready = sum(1 for pd in ship.point_defense if pd.can_fire())
            lines.append(f"    point_defense: {pd_ready}/{len(ship.point_defense)} turrets ready")

        # Enemy contacts
        enemies = snapshot['enemies']
        lines.append("")
        lines.append(f"ENEMY CONTACTS ({len(enemies)}):")
        if enemies:
            for i, enemy in enumerate(sorted(enemies, key=lambda e: e['distance_km']), 1):
                closing = enemy['closing_rate_kps']
                closing_str = f"closing at {closing:.1f} km/s" if closing > 0 else f"opening at {-closing:.1f} km/s"
                bearing = self._calculate_bearing(ship_id, enemy['ship_id'])
                lines.append(f"  [{i}] {enemy['ship_id']} ({enemy['ship_type'].title()})")
                lines.append(f"      Distance: {enemy['distance_km']:.1f} km, {closing_str}")
                lines.append(f"      Bearing: {bearing}")
                lines.append(f"      Hull: {enemy['hull_integrity']:.0f}%")
        else:
            lines.append("  No enemy contacts detected.")

        # Friendly contacts
        friendlies = snapshot['friendlies']
        if friendlies:
            lines.append("")
            lines.append(f"FRIENDLY CONTACTS ({len(friendlies)}):")
            for friendly in friendlies:
                lines.append(f"  - {friendly['ship_id']} ({friendly['ship_type'].title()}) at {friendly['distance_km']:.1f} km")

        # Incoming threats
        threats = snapshot['incoming_threats']
        lines.append("")
        lines.append(f"INCOMING THREATS ({len(threats)}):")
        if threats:
            for threat in threats:
                eta = threat['distance_km'] / max(threat['velocity'], 0.001)
                status = "ARMED" if threat['armed'] else "arming"
                lines.append(f"  ! TORPEDO from {threat['source_ship_id']}")
                lines.append(f"    Distance: {threat['distance_km']:.1f} km, ETA: {eta:.1f}s, Status: {status}")
        else:
            lines.append("  No incoming threats detected.")

        # Recent events (last 30 seconds)
        recent_events = self.get_events_since(self.current_time - 30.0)
        combat_events = [e for e in recent_events if e.event_type in [
            SimulationEventType.DAMAGE_TAKEN,
            SimulationEventType.PROJECTILE_IMPACT,
            SimulationEventType.TORPEDO_IMPACT,
            SimulationEventType.PD_TORPEDO_DESTROYED,
            SimulationEventType.PD_SLUG_DESTROYED
        ] and (e.ship_id == ship_id or e.target_id == ship_id)]

        if combat_events:
            lines.append("")
            lines.append("RECENT COMBAT EVENTS (last 30s):")
            for event in combat_events[-5:]:  # Last 5 events
                if event.event_type == SimulationEventType.DAMAGE_TAKEN and event.ship_id == ship_id:
                    dmg = event.data.get('damage_gj', 0)
                    loc = event.data.get('location', 'unknown')
                    lines.append(f"  - T+{event.timestamp:.1f}s: Took {dmg:.1f} GJ damage to {loc}")
                elif event.event_type == SimulationEventType.PROJECTILE_IMPACT and event.target_id == ship_id:
                    lines.append(f"  - T+{event.timestamp:.1f}s: Hit by projectile from {event.ship_id}")
                elif event.event_type == SimulationEventType.PD_TORPEDO_DESTROYED and event.ship_id == ship_id:
                    lines.append(f"  - T+{event.timestamp:.1f}s: Point defense destroyed incoming torpedo")

        # Tactical summary
        lines.append("")
        lines.append("-" * 60)
        lines.append("TACTICAL SUMMARY:")

        if enemies:
            closest_enemy = min(enemies, key=lambda e: e['distance_km'])
            engagement_range = closest_enemy['distance_km']

            # Determine engagement phase
            if closest_enemy['closing_rate_kps'] > 1.0:
                phase = "CLOSING - Engagement imminent"
            elif closest_enemy['closing_rate_kps'] < -1.0:
                phase = "SEPARATING - Consider pursuit or disengage"
            else:
                phase = "HOLDING - Relative position stable"

            lines.append(f"  Engagement Range: {engagement_range:.1f} km")
            lines.append(f"  Phase: {phase}")

            # Weapon recommendations
            ready_weapons = [slot for slot, w in own['weapons'].items()
                           if w['operational'] and w['cooldown'] <= 0 and w['ammo'] > 0]
            if ready_weapons:
                lines.append(f"  Ready Weapons: {', '.join(ready_weapons)}")
        else:
            lines.append("  No enemies in sensor range.")

        # Current orders
        if snapshot['current_maneuver']:
            lines.append(f"  Current Maneuver: {snapshot['current_maneuver']}")
        if snapshot['primary_target_id']:
            lines.append(f"  Primary Target: {snapshot['primary_target_id']}")

        lines.append("-" * 60)
        lines.append("END SENSOR REPORT")
        lines.append("")

        return "\n".join(lines)

    def _calculate_bearing(self, from_ship_id: str, to_ship_id: str) -> str:
        """Calculate bearing from one ship to another in human-readable form."""
        from_ship = self.get_ship(from_ship_id)
        to_ship = self.get_ship(to_ship_id)

        if not from_ship or not to_ship:
            return "unknown"

        # Get direction vector
        direction = (to_ship.position - from_ship.position).normalized()
        forward = from_ship.forward

        # Calculate angle to forward
        import math
        angle = math.degrees(forward.angle_to(direction))

        # Determine rough bearing
        if angle < 30:
            return "AHEAD"
        elif angle < 60:
            return "FORWARD-QUARTER"
        elif angle < 120:
            return "BEAM (side)"
        elif angle < 150:
            return "AFT-QUARTER"
        else:
            return "ASTERN"


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_ship_from_fleet_data(
    ship_id: str,
    ship_type: str,
    faction: str,
    fleet_data: dict,
    position: Optional[Vector3D] = None,
    velocity: Optional[Vector3D] = None,
    forward: Optional[Vector3D] = None
) -> ShipCombatState:
    """
    Create a ShipCombatState from fleet data.

    Args:
        ship_id: Unique identifier for the ship.
        ship_type: Ship class (e.g., 'destroyer').
        faction: Faction identifier.
        fleet_data: Loaded fleet_ships.json data.
        position: Initial position (default: origin).
        velocity: Initial velocity (default: zero).
        forward: Initial forward direction (default: +X).

    Returns:
        Configured ShipCombatState.
    """
    ships = fleet_data.get("ships", {})
    if ship_type not in ships:
        raise KeyError(f"Ship type '{ship_type}' not found in fleet data")

    ship_data = ships[ship_type]
    hull_data = ship_data.get("hull", {})

    # Create kinematic state (using already imported function)
    propulsion = ship_data.get("propulsion", {})
    drive = propulsion.get("drive", {})
    mass_data = ship_data.get("mass", {})

    kinematic_state = create_ship_state_from_specs(
        wet_mass_tons=mass_data.get("wet_mass_tons", 2000),
        dry_mass_tons=mass_data.get("dry_mass_tons", 1900),
        length_m=hull_data.get("length_m", 100),
        thrust_mn=drive.get("thrust_mn", 58.56),
        exhaust_velocity_kps=drive.get("exhaust_velocity_kps", 10256),
        position=position,
        velocity=velocity,
        forward=forward
    )

    # Create thermal system (using already imported ThermalSystem)
    thermal_system = None
    try:
        thermal_system = ThermalSystem.from_ship_data(ship_data)
    except Exception:
        pass

    # Create armor (using already imported factory function)
    armor = None
    try:
        armor = create_ship_armor_from_fleet_data(fleet_data, ship_type)
    except Exception:
        pass

    # Create geometry (using already imported factory function)
    geometry = None
    try:
        geometry = create_geometry_from_fleet_data(ship_type, fleet_data)
    except Exception:
        pass

    # Create module layout
    module_layout = None
    try:
        module_layout = ModuleLayout.from_ship_type(ship_type, fleet_data)
    except Exception:
        pass

    # Create weapons
    weapons = {}
    weapon_list = ship_data.get("weapons", [])
    weapon_types = fleet_data.get("weapon_types", {})

    for i, weapon_data in enumerate(weapon_list):
        weapon_type = weapon_data.get("type", "unknown")
        slot_name = weapon_data.get("slot", f"weapon_{i}")

        if weapon_type in weapon_types:
            weapon = create_weapon_from_fleet_data(fleet_data, weapon_type)
            weapons[slot_name] = WeaponState(
                weapon=weapon,
                ammo_remaining=weapon.magazine
            )

    # Create torpedo launcher
    torpedo_launcher = None
    torpedo_data = ship_data.get("torpedo", {})
    if torpedo_data:
        specs = TorpedoSpecs.from_fleet_data(
            warhead_yield_gj=torpedo_data.get("warhead_yield_gj", 0),  # Pure kinetic penetrator
            penetrator_mass_kg=torpedo_data.get("penetrator_mass_kg", 100),  # 100 kg dense penetrator
            ammo_mass_kg=torpedo_data.get("ammo_mass_kg", 1600)
        )
        torpedo_launcher = TorpedoLauncher(
            specs=specs,
            magazine_capacity=torpedo_data.get("magazine", 16),
            current_magazine=torpedo_data.get("magazine", 16),
            cooldown_seconds=torpedo_data.get("cooldown_s", 30)
        )
    else:
        # Check weapons array for torpedo launcher
        for weapon_data in weapon_list:
            weapon_type = weapon_data.get("type", "")
            if weapon_type == "torpedo_launcher":
                # Get torpedo specs from weapon_types
                torp_specs = weapon_types.get("torpedo_launcher", {})
                specs = TorpedoSpecs.from_fleet_data(
                    warhead_yield_gj=torp_specs.get("warhead_yield_gj", 0),  # Pure kinetic penetrator
                    penetrator_mass_kg=torp_specs.get("penetrator_mass_kg", 100),  # 100 kg dense penetrator
                    ammo_mass_kg=torp_specs.get("ammo_mass_kg", 1600)
                )
                torpedo_launcher = TorpedoLauncher(
                    specs=specs,
                    magazine_capacity=torp_specs.get("magazine", 16),
                    current_magazine=torp_specs.get("magazine", 16),
                    cooldown_seconds=torp_specs.get("cooldown_s", 30)
                )
                break

    # Create point defense lasers
    point_defense: list[PDLaserState] = []
    pd_data = ship_data.get("point_defense", [])

    # Check in weapons array for PD lasers if not in dedicated section
    if not pd_data:
        for weapon_data in weapon_list:
            weapon_type = weapon_data.get("type", "")
            if weapon_type in ("pd_laser", "point_defense", "pdl"):
                pd_data.append(weapon_data)

    # Also check weapon_types for PD specs
    pd_specs = weapon_types.get("pd_laser", {})

    for i, pd_info in enumerate(pd_data):
        turret_name = pd_info.get("name", f"PD-{i+1}")
        pd_laser = PDLaser(
            power_mw=pd_info.get("power_mw", pd_specs.get("power_mw", 5.0)),
            aperture_m=pd_info.get("aperture_m", pd_specs.get("aperture_m", 0.5)),
            wavelength_nm=pd_info.get("wavelength_nm", pd_specs.get("wavelength_nm", 1000.0)),
            range_km=pd_info.get("range_km", pd_specs.get("range_km", 100.0)),
            cooldown_s=pd_info.get("cooldown_s", pd_specs.get("cooldown_s", 0.5)),
            name=turret_name
        )
        point_defense.append(PDLaserState(
            laser=pd_laser,
            turret_name=turret_name
        ))

    # If no PD defined, give default PD based on ship class
    if not point_defense:
        # Default PD based on ship class - larger ships get more turrets
        default_pd_counts = {
            "corvette": 1,
            "frigate": 2,
            "destroyer": 3,
            "cruiser": 4,
            "battlecruiser": 5,
            "battleship": 6,
            "dreadnought": 8
        }
        num_pd = default_pd_counts.get(ship_type.lower(), 2)
        for i in range(num_pd):
            pd_laser = PDLaser(
                power_mw=5.0,
                aperture_m=0.5,
                wavelength_nm=1000.0,
                range_km=100.0,
                cooldown_s=0.5,
                name=f"PD-{i+1}"
            )
            point_defense.append(PDLaserState(
                laser=pd_laser,
                turret_name=f"PD-{i+1}"
            ))

    # Load attitude control specs from fleet data
    attitude_control = None
    attitude_data = ship_data.get("attitude_control", {})
    if attitude_data:
        moi_data = attitude_data.get("moment_of_inertia", {})
        tv_data = attitude_data.get("thrust_vectoring", {})
        rcs_data = attitude_data.get("rcs", {})

        attitude_control = AttitudeControlSpecs(
            moment_of_inertia_kg_m2=moi_data.get("pitch_yaw_kg_m2", 700_645_833),
            tv_angular_accel_deg_s2=tv_data.get("angular_accel_deg_s2", 2.445),
            tv_max_angular_vel_deg_s=tv_data.get("max_angular_velocity_deg_s", 14.83),
            rcs_angular_accel_deg_s2=rcs_data.get("angular_accel_deg_s2", 0.1227),
            rcs_max_angular_vel_deg_s=rcs_data.get("max_angular_velocity_deg_s", 3.32)
        )

    # Create power system
    power_system = None
    try:
        power_system = PowerSystem.from_ship_data(ship_data)
        # Add capacitors for each weapon
        for slot_name, weapon_state in weapons.items():
            weapon_data = {
                "type": weapon_state.weapon.weapon_type,
                "kinetic_energy_gj": weapon_state.weapon.kinetic_energy_gj,
                "cooldown_s": weapon_state.weapon.cooldown_s,
                "power_draw_mw": getattr(weapon_state.weapon, 'power_draw_mw', 5.0)
            }
            power_system.add_weapon_capacitor(slot_name, weapon_data)
        # Add capacitors for PD lasers
        for pd_state in point_defense:
            pd_data = {
                "type": "point_defense",
                "power_draw_mw": pd_state.laser.power_mw,
                "cooldown_s": pd_state.laser.cooldown_s
            }
            power_system.add_weapon_capacitor(pd_state.turret_name, pd_data)
        # Add capacitor for torpedo launcher
        if torpedo_launcher:
            torp_data = {
                "type": "missile",
                "cooldown_s": torpedo_launcher.cooldown_seconds
            }
            power_system.add_weapon_capacitor("torpedo_launcher", torp_data)
    except Exception:
        pass

    return ShipCombatState(
        ship_id=ship_id,
        ship_type=ship_type,
        faction=faction,
        kinematic_state=kinematic_state,
        thermal_system=thermal_system,
        armor=armor,
        module_layout=module_layout,
        geometry=geometry,
        weapons=weapons,
        torpedo_launcher=torpedo_launcher,
        point_defense=point_defense,
        attitude_control=attitude_control,
        power_system=power_system
    )


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS COMBAT SIMULATION ENGINE - SELF TEST")
    print("=" * 70)

    # Create a simple test simulation
    print("\n--- Creating Simulation ---")
    sim = CombatSimulation(time_step=1.0, decision_interval=30.0, seed=42)

    # Create two test ships
    from physics import create_ship_state_from_specs

    # Ship A - at origin, facing +X
    ship_a_kinematic = create_ship_state_from_specs(
        wet_mass_tons=2000,
        dry_mass_tons=1900,
        length_m=100,
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(5000, 0, 0),  # 5 km/s
        forward=Vector3D(1, 0, 0)
    )

    ship_a = ShipCombatState(
        ship_id="alpha_1",
        ship_type="destroyer",
        faction="alpha",
        kinematic_state=ship_a_kinematic
    )

    # Add a weapon
    test_weapon = Weapon(
        name="Test Coilgun",
        weapon_type="coilgun",
        kinetic_energy_gj=15.0,
        cooldown_s=5.0,
        range_km=500,
        flat_chipping=0.8,
        magazine=100,
        muzzle_velocity_kps=10.0,
        warhead_mass_kg=25.0
    )
    ship_a.weapons["spinal"] = WeaponState(weapon=test_weapon, ammo_remaining=100)

    # Ship B - 200 km away, facing -X (toward Ship A)
    ship_b_kinematic = create_ship_state_from_specs(
        wet_mass_tons=2000,
        dry_mass_tons=1900,
        length_m=100,
        position=Vector3D(200_000, 10_000, 0),  # 200 km away
        velocity=Vector3D(-5000, 0, 0),  # 5 km/s toward ship A
        forward=Vector3D(-1, 0, 0)
    )

    ship_b = ShipCombatState(
        ship_id="beta_1",
        ship_type="destroyer",
        faction="beta",
        kinematic_state=ship_b_kinematic
    )
    ship_b.weapons["spinal"] = WeaponState(weapon=test_weapon, ammo_remaining=100)

    sim.add_ship(ship_a)
    sim.add_ship(ship_b)

    print(f"Ships added: {list(sim.ships.keys())}")
    print(f"Ship A position: {ship_a.position}")
    print(f"Ship B position: {ship_b.position}")
    print(f"Initial distance: {ship_a.distance_to(ship_b) / 1000:.1f} km")
    print(f"Closing rate: {ship_a.closing_rate_to(ship_b) / 1000:.1f} km/s")

    # Test fire command
    print("\n--- Testing Fire Command ---")
    result = sim.inject_command("alpha_1", {
        'type': 'fire_at',
        'weapon_slot': 'spinal',
        'target_id': 'beta_1'
    })
    print(f"Fire command accepted: {result}")
    print(f"Projectiles in flight: {len(sim.projectiles)}")

    # Run a few simulation steps
    print("\n--- Running 10 Steps ---")
    for i in range(10):
        sim.step()

    print(f"Current time: {sim.current_time:.1f}s")
    print(f"Projectiles in flight: {len(sim.projectiles)}")
    print(f"Ship A ammo: {sim.ships['alpha_1'].weapons['spinal'].ammo_remaining}")
    print(f"Events logged: {len(sim.events)}")

    # Test maneuver
    print("\n--- Testing Maneuver ---")
    maneuver = Maneuver(
        maneuver_type=ManeuverType.INTERCEPT,
        start_time=sim.current_time,
        duration=60.0,
        throttle=0.5,
        target_id="beta_1"
    )
    sim.inject_command("alpha_1", maneuver)

    # Run more steps
    for i in range(50):
        sim.step()

    print(f"Current time: {sim.current_time:.1f}s")
    print(f"Ship A position: {ship_a.position}")
    print(f"Distance now: {ship_a.distance_to(ship_b) / 1000:.1f} km")

    # Print battle snapshot
    print("\n--- Battle Snapshot ---")
    snapshot = sim.get_battle_snapshot("alpha_1")
    print(f"Timestamp: {snapshot['timestamp']:.1f}s")
    print(f"Enemies: {len(snapshot['enemies'])}")
    if snapshot['enemies']:
        print(f"  - {snapshot['enemies'][0]['ship_id']}: {snapshot['enemies'][0]['distance_km']:.1f} km")
    print(f"Engagement range: {snapshot['engagement_range_km']:.1f} km")

    # Print event summary
    print("\n--- Event Summary ---")
    event_counts: dict[str, int] = {}
    for event in sim.events:
        name = event.event_type.name
        event_counts[name] = event_counts.get(name, 0) + 1

    for event_type, count in sorted(event_counts.items()):
        print(f"  {event_type}: {count}")

    # Print metrics
    print("\n--- Metrics ---")
    print(f"Total shots fired: {sim.metrics.total_shots_fired}")
    print(f"Total hits: {sim.metrics.total_hits}")
    print(f"Hit rate: {sim.metrics.hit_rate * 100:.1f}%")

    print("\n" + "=" * 70)
    print("Combat simulation engine tests completed!")
    print("=" * 70)
