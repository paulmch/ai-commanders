#!/usr/bin/env python3
"""
Captain Interface Module for AI Commanders Space Battle Simulator

Provides the interface between the simulation and LLM captains.
Formats game state into decision-ready information and parses captain orders.

Key classes:
- SensorSnapshot: What the captain "sees" at a decision point
- ThreatAssessment: Analyzed threats with evadability calculations
- CaptainDecision: Orders the captain can issue
- CaptainInterface: Formats data for LLM consumption
- SituationReport: Formatted text output for human-readable display

This module bridges the detailed simulation state to the high-level
decision-making interface needed by LLM captains.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

# Try/except import pattern for relative/absolute imports
try:
    from .physics import Vector3D, ShipState
except ImportError:
    from physics import Vector3D, ShipState

try:
    from .combat import Weapon, HitLocation, ShipArmor, Armor
except ImportError:
    from combat import Weapon, HitLocation, ShipArmor, Armor

try:
    from .thermal import ThermalSystem, RadiatorState, RadiatorPosition
except ImportError:
    from thermal import ThermalSystem, RadiatorState, RadiatorPosition

try:
    from .targeting import (
        TargetingSystem, FiringSolution, ECMSystem, TargetingComputer,
        FiringArc, SpinalWeaponConstraint, LeadCalculator
    )
except ImportError:
    from targeting import (
        TargetingSystem, FiringSolution, ECMSystem, TargetingComputer,
        FiringArc, SpinalWeaponConstraint, LeadCalculator
    )

try:
    from .torpedo import Torpedo, TorpedoSpecs, GuidanceMode
except ImportError:
    from torpedo import Torpedo, TorpedoSpecs, GuidanceMode

try:
    from .projectile import Projectile, KineticProjectile
except ImportError:
    from projectile import Projectile, KineticProjectile

# damage.py uses relative imports that fail in standalone mode
# Import the specific classes we need directly
try:
    from .damage import ModuleLayout, Module
except ImportError:
    try:
        from damage import ModuleLayout, Module
    except ImportError:
        # Define minimal stubs if damage module unavailable
        from dataclasses import dataclass as _dataclass
        from typing import List as _List, Optional as _Optional

        @_dataclass
        class Module:
            """Minimal stub for Module when damage.py is unavailable."""
            name: str
            position: Vector3D
            health: float
            max_health: float
            radius_m: float = 2.0
            is_critical: bool = False
            is_destroyed: bool = False

        @_dataclass
        class ModuleLayout:
            """Minimal stub for ModuleLayout when damage.py is unavailable."""
            modules: _List[Module] = None
            ship_length_m: float = 65.0

            def __post_init__(self):
                if self.modules is None:
                    self.modules = []


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ThreatType(Enum):
    """Types of threats that can be assessed."""
    PROJECTILE = "projectile"
    TORPEDO = "torpedo"
    SHIP = "ship"
    MISSILE = "missile"


class ManeuverType(Enum):
    """Types of maneuvers the captain can order."""
    BURN_TOWARD = "burn_toward"      # Thrust toward a point/target
    BURN_AWAY = "burn_away"          # Thrust away from a point/target
    BURN_DIRECTION = "burn_direction"  # Thrust in a specific direction
    EVASIVE = "evasive"              # Random evasive maneuvers
    MAINTAIN = "maintain"            # Maintain current velocity
    FLIP_AND_BURN = "flip_and_burn"  # Flip ship and decelerate
    ORIENT_TO = "orient_to"          # Rotate to face a direction/target


class WeaponCommandType(Enum):
    """Types of weapon commands."""
    FIRE_SPINAL = "fire_spinal"           # Fire spinal mount weapon
    FIRE_TURRET = "fire_turret"           # Fire a turret weapon
    LAUNCH_TORPEDO = "launch_torpedo"     # Launch a torpedo
    HOLD_FIRE = "hold_fire"               # Do not fire weapons
    CHANGE_TARGET = "change_target"       # Switch weapon target


class DefensiveCommandType(Enum):
    """Types of defensive commands."""
    ECM_ACTIVATE = "ecm_activate"         # Turn on ECM
    ECM_DEACTIVATE = "ecm_deactivate"     # Turn off ECM
    POINT_DEFENSE_AUTO = "pd_auto"        # Point defense automatic mode
    POINT_DEFENSE_MANUAL = "pd_manual"    # Point defense manual targeting
    POINT_DEFENSE_OFF = "pd_off"          # Point defense off


class SystemCommandType(Enum):
    """Types of system commands."""
    RADIATORS_EXTEND = "radiators_extend"     # Extend radiators for cooling
    RADIATORS_RETRACT = "radiators_retract"   # Retract radiators for protection
    DAMAGE_CONTROL = "damage_control"         # Focus on damage control


# =============================================================================
# SENSOR SNAPSHOT - What the captain "sees"
# =============================================================================

@dataclass
class ContactInfo:
    """
    Information about a detected contact.

    Attributes:
        contact_id: Unique identifier for this contact
        contact_type: Type of contact (ship, torpedo, projectile)
        position: Position vector in world coordinates (meters)
        velocity: Velocity vector (m/s)
        distance_km: Distance to contact in kilometers
        bearing_deg: Bearing to contact in degrees (0=forward, 90=right)
        closing_rate_kps: Rate of closure in km/s (positive=approaching)
        time_to_intercept_s: Estimated time until intercept/collision
        is_hostile: Whether contact is identified as hostile
        classification: Ship class or weapon type if known
    """
    contact_id: str
    contact_type: ThreatType
    position: Vector3D
    velocity: Vector3D
    distance_km: float
    bearing_deg: float
    closing_rate_kps: float
    time_to_intercept_s: float
    is_hostile: bool = True
    classification: str = "Unknown"


@dataclass
class OwnShipStatus:
    """
    Status of own ship for the captain's awareness.

    Attributes:
        position: Current position (meters)
        velocity: Current velocity (m/s)
        speed_kps: Current speed in km/s
        heading: Forward direction vector
        acceleration_g: Current acceleration capability in g's
        remaining_delta_v_kps: Remaining delta-v budget (km/s)
        hull_integrity_percent: Overall hull integrity (0-100)
        propellant_percent: Remaining propellant (0-100)
    """
    position: Vector3D
    velocity: Vector3D
    speed_kps: float
    heading: Vector3D
    acceleration_g: float
    remaining_delta_v_kps: float
    hull_integrity_percent: float
    propellant_percent: float


@dataclass
class ThermalStatus:
    """
    Thermal system status.

    Attributes:
        heat_percent: Current heat level (0-100)
        is_overheating: Whether in overheating warning state
        is_critical: Whether in critical thermal state
        radiator_status: Dict of radiator positions to their status
        dissipation_kw: Current heat dissipation rate in kW
        capacity_gj: Total heat sink capacity in GJ
    """
    heat_percent: float
    is_overheating: bool
    is_critical: bool
    radiator_status: Dict[str, str]  # position -> state
    dissipation_kw: float
    capacity_gj: float


@dataclass
class WeaponStatus:
    """
    Status of a weapon system.

    Attributes:
        weapon_id: Unique identifier
        weapon_name: Display name
        weapon_type: Type (spinal, turret, torpedo, pd)
        ready: Whether weapon can fire
        cooldown_remaining_s: Seconds until weapon is ready
        ammo_remaining: Rounds remaining
        target_locked: Whether a target lock is held
        target_id: ID of locked target (if any)
        in_arc: Whether current target is in firing arc
    """
    weapon_id: str
    weapon_name: str
    weapon_type: str
    ready: bool
    cooldown_remaining_s: float
    ammo_remaining: int
    target_locked: bool
    target_id: Optional[str] = None
    in_arc: bool = False


@dataclass
class DamageStatus:
    """
    Damage status summary.

    Attributes:
        armor_sections: Dict of section name to armor thickness remaining
        critical_systems_damaged: List of damaged critical systems
        modules_destroyed: List of destroyed modules
        overall_integrity: Overall structural integrity (0-100)
    """
    armor_sections: Dict[str, float]  # section -> thickness_cm
    critical_systems_damaged: List[str]
    modules_destroyed: List[str]
    overall_integrity: float


@dataclass
class SensorSnapshot:
    """
    Complete snapshot of what the captain perceives at a decision point.

    This is the primary input for captain decision-making, containing
    all information needed to make tactical decisions.

    Attributes:
        timestamp: Simulation time in seconds
        own_ship: Own ship status
        thermal: Thermal system status
        weapons: List of weapon statuses
        damage: Damage status summary
        enemy_contacts: List of detected enemy contacts
        incoming_threats: List of incoming threats (torpedoes, projectiles)
        engagement_geometry: Summary of tactical geometry
    """
    timestamp: float
    own_ship: OwnShipStatus
    thermal: ThermalStatus
    weapons: List[WeaponStatus]
    damage: DamageStatus
    enemy_contacts: List[ContactInfo]
    incoming_threats: List[ContactInfo]
    engagement_geometry: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate engagement geometry if not provided."""
        if not self.engagement_geometry and self.enemy_contacts:
            self._calculate_engagement_geometry()

    def _calculate_engagement_geometry(self) -> None:
        """Calculate tactical geometry from contacts."""
        if not self.enemy_contacts:
            return

        # Find nearest enemy
        nearest = min(self.enemy_contacts, key=lambda c: c.distance_km)

        # Calculate if we can bring spinal weapons to bear
        spinal_arc = 5.0  # degrees
        can_engage_spinal = abs(nearest.bearing_deg) <= spinal_arc

        self.engagement_geometry = {
            'nearest_enemy_km': nearest.distance_km,
            'nearest_enemy_bearing': nearest.bearing_deg,
            'nearest_enemy_closing_kps': nearest.closing_rate_kps,
            'can_engage_spinal': can_engage_spinal,
            'time_to_spinal_engagement_s': abs(nearest.bearing_deg) / 0.5 if not can_engage_spinal else 0.0,
        }


# =============================================================================
# THREAT ASSESSMENT
# =============================================================================

@dataclass
class ThreatAssessment:
    """
    Analysis of a specific threat with tactical recommendations.

    Provides calculated threat level, evadability, and recommended response
    for each incoming threat.

    Attributes:
        threat_id: Reference to the threat contact
        threat_type: Type of threat
        threat_level: Normalized threat level (0.0=negligible, 1.0=critical)
        impact_energy_gj: Estimated impact energy in gigajoules
        time_to_impact_s: Seconds until potential impact
        evade_delta_v_kps: Delta-v required to evade this threat
        can_evade: Whether evasion is possible with current delta-v budget
        evade_direction: Recommended evasion direction
        can_intercept_pd: Whether point defense can engage
        recommended_action: Suggested response to this threat
        details: Additional threat-specific details
    """
    threat_id: str
    threat_type: ThreatType
    threat_level: float  # 0.0 to 1.0
    impact_energy_gj: float
    time_to_impact_s: float
    evade_delta_v_kps: float
    can_evade: bool
    evade_direction: Vector3D
    can_intercept_pd: bool
    recommended_action: str
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_torpedo(
        cls,
        torpedo: Torpedo,
        own_position: Vector3D,
        own_velocity: Vector3D,
        remaining_delta_v_kps: float,
        pd_range_km: float = 50.0
    ) -> ThreatAssessment:
        """
        Create threat assessment from an incoming torpedo.

        Args:
            torpedo: The incoming torpedo
            own_position: Our position
            own_velocity: Our velocity
            remaining_delta_v_kps: Our remaining delta-v
            pd_range_km: Point defense engagement range

        Returns:
            ThreatAssessment for this torpedo
        """
        # Calculate relative geometry
        rel_pos = torpedo.position - own_position
        distance_m = rel_pos.magnitude
        distance_km = distance_m / 1000.0

        rel_vel = torpedo.velocity - own_velocity
        closing_rate = -rel_vel.dot(rel_pos.normalized()) if distance_m > 0 else 0

        # Time to impact
        if closing_rate > 0:
            time_to_impact = distance_m / closing_rate
        else:
            time_to_impact = float('inf')

        # Impact energy (warhead + kinetic)
        kinetic_energy = 0.5 * torpedo.current_mass_kg * (closing_rate ** 2) / 1e9
        warhead_energy = torpedo.specs.warhead_yield_gj
        impact_energy = kinetic_energy + warhead_energy

        # Calculate evasion requirements
        # To evade, we need to move perpendicular to threat vector
        if rel_pos.magnitude > 0:
            threat_dir = rel_pos.normalized()
            # Find perpendicular direction (cross with up or forward)
            up = Vector3D(0, 0, 1)
            evade_dir = threat_dir.cross(up)
            if evade_dir.magnitude < 0.1:
                evade_dir = threat_dir.cross(Vector3D(1, 0, 0))
            evade_dir = evade_dir.normalized()
        else:
            evade_dir = Vector3D(0, 1, 0)

        # Estimate delta-v needed to evade
        # Need to move enough to miss during remaining time
        # Assuming ship radius ~50m, need to move at least 100m
        min_miss_distance = 100.0  # meters
        if time_to_impact > 0 and time_to_impact < float('inf'):
            # v = d / t, so delta_v = min_miss_distance / time_to_impact
            evade_delta_v = min_miss_distance / time_to_impact / 1000.0  # km/s
            # Account for torpedo tracking ability (it will adjust)
            evade_delta_v *= 3.0  # Need more delta-v due to tracking
        else:
            evade_delta_v = 0.0

        can_evade = remaining_delta_v_kps >= evade_delta_v

        # Point defense intercept possibility
        can_pd = distance_km <= pd_range_km and time_to_impact > 2.0

        # Threat level (0-1)
        # Based on time to impact and ability to evade
        if time_to_impact <= 5.0:
            threat_level = 1.0
        elif time_to_impact <= 30.0:
            threat_level = 0.8
        elif can_evade or can_pd:
            threat_level = 0.4
        else:
            threat_level = 0.9

        # Recommended action
        if time_to_impact <= 2.0:
            action = "BRACE_FOR_IMPACT"
        elif can_pd and time_to_impact > 5.0:
            action = "ENGAGE_POINT_DEFENSE"
        elif can_evade:
            action = "EVASIVE_MANEUVER"
        else:
            action = "MAXIMIZE_EVASION"

        return cls(
            threat_id=torpedo.target_id,
            threat_type=ThreatType.TORPEDO,
            threat_level=threat_level,
            impact_energy_gj=impact_energy,
            time_to_impact_s=time_to_impact,
            evade_delta_v_kps=evade_delta_v,
            can_evade=can_evade,
            evade_direction=evade_dir,
            can_intercept_pd=can_pd,
            recommended_action=action,
            details={
                'guidance_mode': torpedo.guidance_mode.name,
                'fuel_remaining_kps': torpedo.remaining_delta_v_kps,
                'warhead_yield_gj': warhead_energy,
                'distance_km': distance_km
            }
        )

    @classmethod
    def from_projectile(
        cls,
        projectile: Projectile,
        own_position: Vector3D,
        own_velocity: Vector3D,
        remaining_delta_v_kps: float,
        pd_range_km: float = 20.0
    ) -> ThreatAssessment:
        """
        Create threat assessment from an incoming projectile.

        Args:
            projectile: The incoming projectile
            own_position: Our position
            own_velocity: Our velocity
            remaining_delta_v_kps: Our remaining delta-v
            pd_range_km: Point defense engagement range

        Returns:
            ThreatAssessment for this projectile
        """
        # Calculate relative geometry
        rel_pos = projectile.position - own_position
        distance_m = rel_pos.magnitude
        distance_km = distance_m / 1000.0

        rel_vel = projectile.velocity - own_velocity
        closing_rate = -rel_vel.dot(rel_pos.normalized()) if distance_m > 0 else 0

        # Time to impact (projectiles are ballistic)
        if closing_rate > 0:
            time_to_impact = distance_m / closing_rate
        else:
            time_to_impact = float('inf')

        # Impact energy
        impact_energy = projectile.kinetic_energy_gj

        # Evasion calculation (easier than torpedo - no tracking)
        if rel_pos.magnitude > 0:
            threat_dir = rel_pos.normalized()
            up = Vector3D(0, 0, 1)
            evade_dir = threat_dir.cross(up)
            if evade_dir.magnitude < 0.1:
                evade_dir = threat_dir.cross(Vector3D(1, 0, 0))
            evade_dir = evade_dir.normalized()
        else:
            evade_dir = Vector3D(0, 1, 0)

        # Delta-v needed to evade unguided projectile
        min_miss_distance = 100.0
        if time_to_impact > 0 and time_to_impact < float('inf'):
            evade_delta_v = min_miss_distance / time_to_impact / 1000.0
        else:
            evade_delta_v = 0.0

        can_evade = remaining_delta_v_kps >= evade_delta_v

        # Point defense (harder against fast projectiles)
        projectile_speed_kps = projectile.speed_kps
        can_pd = distance_km <= pd_range_km and projectile_speed_kps < 50.0 and time_to_impact > 1.0

        # Threat level
        if time_to_impact <= 2.0:
            threat_level = 0.9
        elif time_to_impact <= 10.0:
            threat_level = 0.6
        elif can_evade:
            threat_level = 0.2
        else:
            threat_level = 0.5

        # Recommended action
        if time_to_impact <= 1.0:
            action = "BRACE_FOR_IMPACT"
        elif can_evade and time_to_impact > 5.0:
            action = "EVASIVE_MANEUVER"
        elif can_pd:
            action = "ENGAGE_POINT_DEFENSE"
        else:
            action = "ACCEPT_HIT"

        return cls(
            threat_id=f"projectile_{id(projectile)}",
            threat_type=ThreatType.PROJECTILE,
            threat_level=threat_level,
            impact_energy_gj=impact_energy,
            time_to_impact_s=time_to_impact,
            evade_delta_v_kps=evade_delta_v,
            can_evade=can_evade,
            evade_direction=evade_dir,
            can_intercept_pd=can_pd,
            recommended_action=action,
            details={
                'projectile_speed_kps': projectile_speed_kps,
                'distance_km': distance_km
            }
        )


# =============================================================================
# CAPTAIN DECISION - What the captain can order
# =============================================================================

@dataclass
class ManeuverCommand:
    """
    A maneuver command from the captain.

    Attributes:
        maneuver_type: Type of maneuver to execute
        target: Target point or entity ID (depending on maneuver type)
        direction: Direction vector (for BURN_DIRECTION)
        throttle: Throttle setting 0.0 to 1.0
        duration_s: Duration of maneuver in seconds (0 = continuous)
    """
    maneuver_type: ManeuverType
    target: Optional[Union[Vector3D, str]] = None
    direction: Optional[Vector3D] = None
    throttle: float = 1.0
    duration_s: float = 0.0


@dataclass
class WeaponCommand:
    """
    A weapon command from the captain.

    Attributes:
        command_type: Type of weapon command
        weapon_id: Which weapon to command (or "all")
        target_id: Target to engage
        priority: Priority level (higher = more important)
    """
    command_type: WeaponCommandType
    weapon_id: str = "all"
    target_id: Optional[str] = None
    priority: int = 1


@dataclass
class DefensiveCommand:
    """
    A defensive system command from the captain.

    Attributes:
        command_type: Type of defensive command
        target_id: Target for point defense (if manual mode)
    """
    command_type: DefensiveCommandType
    target_id: Optional[str] = None


@dataclass
class SystemCommand:
    """
    A ship system command from the captain.

    Attributes:
        command_type: Type of system command
        parameters: Additional parameters for the command
    """
    command_type: SystemCommandType
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptainDecision:
    """
    Complete decision package from the captain.

    Represents all orders the captain issues for a decision cycle.

    Attributes:
        timestamp: Simulation time of decision
        maneuver: Maneuver command (if any)
        weapon_commands: List of weapon commands
        defensive_commands: List of defensive commands
        system_commands: List of system commands
        priority_target: Primary target ID for engagement
        tactical_intent: Brief description of tactical intent
        confidence: Captain's confidence in this decision (0.0 to 1.0)
    """
    timestamp: float
    maneuver: Optional[ManeuverCommand] = None
    weapon_commands: List[WeaponCommand] = field(default_factory=list)
    defensive_commands: List[DefensiveCommand] = field(default_factory=list)
    system_commands: List[SystemCommand] = field(default_factory=list)
    priority_target: Optional[str] = None
    tactical_intent: str = ""
    confidence: float = 0.8


# =============================================================================
# CAPTAIN INTERFACE - LLM Integration
# =============================================================================

class CaptainInterface:
    """
    Interface for LLM captain integration.

    Provides methods to format simulation state for LLM consumption
    and parse LLM responses into executable commands.
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the captain interface.

        Args:
            verbose: If True, include additional detail in outputs
        """
        self.verbose = verbose

    def to_prompt(self, snapshot: SensorSnapshot) -> str:
        """
        Format sensor snapshot as a text prompt for LLM.

        Creates a human-readable situation report suitable for
        LLM captains to make decisions from.

        Args:
            snapshot: Current sensor snapshot

        Returns:
            Formatted text prompt string
        """
        return SituationReport.generate(snapshot)

    def to_json(self, snapshot: SensorSnapshot) -> str:
        """
        Format sensor snapshot as JSON for structured processing.

        Args:
            snapshot: Current sensor snapshot

        Returns:
            JSON string representation
        """
        data = self._snapshot_to_dict(snapshot)
        return json.dumps(data, indent=2, default=str)

    def _snapshot_to_dict(self, snapshot: SensorSnapshot) -> Dict[str, Any]:
        """Convert snapshot to dictionary for JSON serialization."""
        return {
            'timestamp': snapshot.timestamp,
            'own_ship': {
                'position': [snapshot.own_ship.position.x,
                            snapshot.own_ship.position.y,
                            snapshot.own_ship.position.z],
                'velocity': [snapshot.own_ship.velocity.x,
                            snapshot.own_ship.velocity.y,
                            snapshot.own_ship.velocity.z],
                'speed_kps': snapshot.own_ship.speed_kps,
                'heading': [snapshot.own_ship.heading.x,
                           snapshot.own_ship.heading.y,
                           snapshot.own_ship.heading.z],
                'acceleration_g': snapshot.own_ship.acceleration_g,
                'remaining_delta_v_kps': snapshot.own_ship.remaining_delta_v_kps,
                'hull_integrity_percent': snapshot.own_ship.hull_integrity_percent,
                'propellant_percent': snapshot.own_ship.propellant_percent
            },
            'thermal': {
                'heat_percent': snapshot.thermal.heat_percent,
                'is_overheating': snapshot.thermal.is_overheating,
                'is_critical': snapshot.thermal.is_critical,
                'radiator_status': snapshot.thermal.radiator_status,
                'dissipation_kw': snapshot.thermal.dissipation_kw
            },
            'weapons': [
                {
                    'weapon_id': w.weapon_id,
                    'weapon_name': w.weapon_name,
                    'weapon_type': w.weapon_type,
                    'ready': w.ready,
                    'cooldown_remaining_s': w.cooldown_remaining_s,
                    'ammo_remaining': w.ammo_remaining,
                    'target_locked': w.target_locked,
                    'in_arc': w.in_arc
                }
                for w in snapshot.weapons
            ],
            'damage': {
                'armor_sections': snapshot.damage.armor_sections,
                'critical_systems_damaged': snapshot.damage.critical_systems_damaged,
                'modules_destroyed': snapshot.damage.modules_destroyed,
                'overall_integrity': snapshot.damage.overall_integrity
            },
            'enemy_contacts': [
                {
                    'contact_id': c.contact_id,
                    'contact_type': c.contact_type.value,
                    'distance_km': c.distance_km,
                    'bearing_deg': c.bearing_deg,
                    'closing_rate_kps': c.closing_rate_kps,
                    'time_to_intercept_s': c.time_to_intercept_s,
                    'classification': c.classification
                }
                for c in snapshot.enemy_contacts
            ],
            'incoming_threats': [
                {
                    'contact_id': c.contact_id,
                    'contact_type': c.contact_type.value,
                    'distance_km': c.distance_km,
                    'time_to_intercept_s': c.time_to_intercept_s
                }
                for c in snapshot.incoming_threats
            ],
            'engagement_geometry': snapshot.engagement_geometry
        }

    def parse_response(self, response: str) -> CaptainDecision:
        """
        Parse LLM response into a CaptainDecision.

        Attempts to extract commands from natural language or
        structured format responses.

        Args:
            response: LLM response string

        Returns:
            Parsed CaptainDecision
        """
        # Try JSON parsing first
        try:
            return self._parse_json_response(response)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fall back to natural language parsing
        return self._parse_natural_response(response)

    def _parse_json_response(self, response: str) -> CaptainDecision:
        """Parse a JSON-formatted response."""
        # Extract JSON if embedded in text
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
        else:
            data = json.loads(response)

        decision = CaptainDecision(
            timestamp=data.get('timestamp', 0.0),
            tactical_intent=data.get('tactical_intent', ''),
            confidence=data.get('confidence', 0.8)
        )

        # Parse maneuver
        if 'maneuver' in data and data['maneuver']:
            m = data['maneuver']
            maneuver_type = ManeuverType(m.get('type', 'maintain'))
            decision.maneuver = ManeuverCommand(
                maneuver_type=maneuver_type,
                throttle=m.get('throttle', 1.0),
                duration_s=m.get('duration_s', 0.0)
            )
            if 'target' in m:
                decision.maneuver.target = m['target']
            if 'direction' in m:
                d = m['direction']
                decision.maneuver.direction = Vector3D(d[0], d[1], d[2])

        # Parse weapon commands
        for wc in data.get('weapon_commands', []):
            cmd = WeaponCommand(
                command_type=WeaponCommandType(wc.get('type', 'hold_fire')),
                weapon_id=wc.get('weapon_id', 'all'),
                target_id=wc.get('target_id'),
                priority=wc.get('priority', 1)
            )
            decision.weapon_commands.append(cmd)

        # Parse defensive commands
        for dc in data.get('defensive_commands', []):
            cmd = DefensiveCommand(
                command_type=DefensiveCommandType(dc.get('type', 'pd_auto')),
                target_id=dc.get('target_id')
            )
            decision.defensive_commands.append(cmd)

        # Parse system commands
        for sc in data.get('system_commands', []):
            cmd = SystemCommand(
                command_type=SystemCommandType(sc.get('type', 'radiators_retract')),
                parameters=sc.get('parameters', {})
            )
            decision.system_commands.append(cmd)

        decision.priority_target = data.get('priority_target')

        return decision

    def _parse_natural_response(self, response: str) -> CaptainDecision:
        """Parse a natural language response."""
        response_lower = response.lower()
        decision = CaptainDecision(timestamp=0.0)

        # Parse maneuver intent
        if 'evade' in response_lower or 'evasive' in response_lower:
            decision.maneuver = ManeuverCommand(
                maneuver_type=ManeuverType.EVASIVE,
                throttle=1.0
            )
        elif 'attack' in response_lower or 'engage' in response_lower:
            decision.maneuver = ManeuverCommand(
                maneuver_type=ManeuverType.BURN_TOWARD,
                throttle=1.0
            )
        elif 'retreat' in response_lower or 'flee' in response_lower:
            decision.maneuver = ManeuverCommand(
                maneuver_type=ManeuverType.BURN_AWAY,
                throttle=1.0
            )
        elif 'maintain' in response_lower or 'hold' in response_lower:
            decision.maneuver = ManeuverCommand(
                maneuver_type=ManeuverType.MAINTAIN
            )

        # Parse weapon intent
        if 'fire' in response_lower:
            if 'torpedo' in response_lower:
                decision.weapon_commands.append(WeaponCommand(
                    command_type=WeaponCommandType.LAUNCH_TORPEDO
                ))
            else:
                decision.weapon_commands.append(WeaponCommand(
                    command_type=WeaponCommandType.FIRE_SPINAL
                ))
        elif 'hold fire' in response_lower:
            decision.weapon_commands.append(WeaponCommand(
                command_type=WeaponCommandType.HOLD_FIRE
            ))

        # Parse defensive intent
        if 'ecm' in response_lower and 'on' in response_lower:
            decision.defensive_commands.append(DefensiveCommand(
                command_type=DefensiveCommandType.ECM_ACTIVATE
            ))
        elif 'ecm' in response_lower and 'off' in response_lower:
            decision.defensive_commands.append(DefensiveCommand(
                command_type=DefensiveCommandType.ECM_DEACTIVATE
            ))
        if 'point defense' in response_lower:
            decision.defensive_commands.append(DefensiveCommand(
                command_type=DefensiveCommandType.POINT_DEFENSE_AUTO
            ))

        # Parse system intent
        if 'extend radiator' in response_lower:
            decision.system_commands.append(SystemCommand(
                command_type=SystemCommandType.RADIATORS_EXTEND
            ))
        elif 'retract radiator' in response_lower:
            decision.system_commands.append(SystemCommand(
                command_type=SystemCommandType.RADIATORS_RETRACT
            ))

        # Extract tactical intent
        lines = response.strip().split('\n')
        if lines:
            decision.tactical_intent = lines[0][:200]  # First line, max 200 chars

        return decision

    def validate_decision(self, decision: CaptainDecision, snapshot: SensorSnapshot) -> Tuple[bool, List[str]]:
        """
        Validate a captain decision against current state.

        Checks that commands are executable given current game state.

        Args:
            decision: The decision to validate
            snapshot: Current sensor snapshot

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []

        # Validate maneuver
        if decision.maneuver:
            if decision.maneuver.throttle < 0 or decision.maneuver.throttle > 1:
                errors.append(f"Invalid throttle: {decision.maneuver.throttle}")

            if decision.maneuver.maneuver_type == ManeuverType.BURN_TOWARD:
                if decision.maneuver.target is None:
                    errors.append("BURN_TOWARD requires a target")

        # Validate weapon commands
        weapon_ids = {w.weapon_id for w in snapshot.weapons}
        for wc in decision.weapon_commands:
            if wc.weapon_id != "all" and wc.weapon_id not in weapon_ids:
                errors.append(f"Unknown weapon_id: {wc.weapon_id}")

            if wc.command_type in [WeaponCommandType.FIRE_SPINAL, WeaponCommandType.FIRE_TURRET]:
                if wc.target_id is None and not any(c.contact_id for c in snapshot.enemy_contacts):
                    errors.append("Fire command with no valid targets")

        # Validate target references
        all_contact_ids = {c.contact_id for c in snapshot.enemy_contacts}
        all_contact_ids.update(c.contact_id for c in snapshot.incoming_threats)

        if decision.priority_target and decision.priority_target not in all_contact_ids:
            errors.append(f"Unknown priority_target: {decision.priority_target}")

        is_valid = len(errors) == 0
        return is_valid, errors


# =============================================================================
# SITUATION REPORT - Formatted Text Output
# =============================================================================

class SituationReport:
    """
    Generates formatted situation reports for captain consumption.

    Provides human-readable tactical information formatted for
    LLM captains to understand and act upon.
    """

    @staticmethod
    def generate(snapshot: SensorSnapshot) -> str:
        """
        Generate a complete situation report.

        Args:
            snapshot: Current sensor snapshot

        Returns:
            Formatted situation report string
        """
        lines: List[str] = []

        # Header
        lines.append("=" * 60)
        lines.append("TACTICAL SITUATION REPORT")
        lines.append(f"Time: {snapshot.timestamp:.1f}s")
        lines.append("=" * 60)

        # Own Ship Status
        lines.append("")
        lines.append("--- OWN SHIP STATUS ---")
        ship = snapshot.own_ship
        lines.append(f"Speed: {ship.speed_kps:.1f} km/s | Accel: {ship.acceleration_g:.1f}g")
        lines.append(f"Delta-V Remaining: {ship.remaining_delta_v_kps:.1f} km/s")
        lines.append(f"Hull Integrity: {ship.hull_integrity_percent:.0f}%")
        lines.append(f"Propellant: {ship.propellant_percent:.0f}%")

        # Thermal Status
        lines.append("")
        lines.append("--- THERMAL STATUS ---")
        thermal = snapshot.thermal
        status_str = "CRITICAL" if thermal.is_critical else ("WARNING" if thermal.is_overheating else "NOMINAL")
        lines.append(f"Heat: {thermal.heat_percent:.0f}% [{status_str}]")
        lines.append(f"Dissipation: {thermal.dissipation_kw:.0f} kW")

        radiator_summary = []
        for pos, state in thermal.radiator_status.items():
            radiator_summary.append(f"{pos}: {state}")
        lines.append(f"Radiators: {', '.join(radiator_summary)}")

        # Damage Status
        if snapshot.damage.critical_systems_damaged or snapshot.damage.modules_destroyed:
            lines.append("")
            lines.append("--- DAMAGE REPORT ---")
            if snapshot.damage.critical_systems_damaged:
                lines.append(f"Critical Damage: {', '.join(snapshot.damage.critical_systems_damaged)}")
            if snapshot.damage.modules_destroyed:
                lines.append(f"Destroyed: {', '.join(snapshot.damage.modules_destroyed)}")

            lines.append("Armor Status:")
            for section, thickness in snapshot.damage.armor_sections.items():
                lines.append(f"  {section}: {thickness:.1f} cm")

        # Weapons Status
        lines.append("")
        lines.append("--- WEAPONS STATUS ---")
        for weapon in snapshot.weapons:
            ready_str = "READY" if weapon.ready else f"COOLDOWN {weapon.cooldown_remaining_s:.1f}s"
            lock_str = f"LOCKED on {weapon.target_id}" if weapon.target_locked else "NO LOCK"
            arc_str = "IN ARC" if weapon.in_arc else "OUT OF ARC"
            lines.append(f"{weapon.weapon_name} [{weapon.weapon_type}]: {ready_str} | {lock_str} | {arc_str}")
            lines.append(f"  Ammo: {weapon.ammo_remaining}")

        # Enemy Contacts
        lines.append("")
        lines.append("--- ENEMY CONTACTS ---")
        if snapshot.enemy_contacts:
            for contact in sorted(snapshot.enemy_contacts, key=lambda c: c.distance_km):
                closing_str = f"+{contact.closing_rate_kps:.1f}" if contact.closing_rate_kps > 0 else f"{contact.closing_rate_kps:.1f}"
                lines.append(f"{contact.contact_id} [{contact.classification}]:")
                lines.append(f"  Range: {contact.distance_km:.0f} km | Bearing: {contact.bearing_deg:.0f} deg")
                lines.append(f"  Closing: {closing_str} km/s | ETA: {contact.time_to_intercept_s:.0f}s")
        else:
            lines.append("No enemy contacts detected.")

        # Incoming Threats
        if snapshot.incoming_threats:
            lines.append("")
            lines.append("--- INCOMING THREATS ---")
            for threat in sorted(snapshot.incoming_threats, key=lambda t: t.time_to_intercept_s):
                lines.append(f"! {threat.contact_type.value.upper()} - {threat.distance_km:.0f} km - "
                           f"ETA {threat.time_to_intercept_s:.0f}s")

        # Engagement Geometry
        if snapshot.engagement_geometry:
            lines.append("")
            lines.append("--- ENGAGEMENT GEOMETRY ---")
            geom = snapshot.engagement_geometry
            lines.append(f"Nearest Enemy: {geom.get('nearest_enemy_km', 0):.0f} km at "
                        f"{geom.get('nearest_enemy_bearing', 0):.0f} deg")
            spinal_str = "YES" if geom.get('can_engage_spinal', False) else "NO"
            lines.append(f"Spinal Weapon Engagement: {spinal_str}")
            if not geom.get('can_engage_spinal', False):
                lines.append(f"  Time to spinal engagement: {geom.get('time_to_spinal_engagement_s', 0):.1f}s")

        lines.append("")
        lines.append("=" * 60)
        lines.append("AWAITING ORDERS, CAPTAIN")
        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_sensor_snapshot(
    timestamp: float,
    ship_state: ShipState,
    thermal_system: Optional[ThermalSystem] = None,
    weapons: Optional[List[Weapon]] = None,
    module_layout: Optional[ModuleLayout] = None,
    armor: Optional[ShipArmor] = None,
    enemy_ships: Optional[List[Dict[str, Any]]] = None,
    incoming_torpedoes: Optional[List[Torpedo]] = None,
    incoming_projectiles: Optional[List[Projectile]] = None
) -> SensorSnapshot:
    """
    Create a SensorSnapshot from simulation state.

    Factory function to build a complete sensor snapshot from
    various simulation components.

    Args:
        timestamp: Current simulation time
        ship_state: Own ship's physical state
        thermal_system: Ship's thermal management system
        weapons: List of weapon systems
        module_layout: Ship's internal module layout
        armor: Ship's armor configuration
        enemy_ships: List of enemy ship data dicts
        incoming_torpedoes: List of incoming torpedoes
        incoming_projectiles: List of incoming projectiles

    Returns:
        Complete SensorSnapshot
    """
    # Build own ship status
    own_ship = OwnShipStatus(
        position=ship_state.position,
        velocity=ship_state.velocity,
        speed_kps=ship_state.velocity.magnitude / 1000.0,
        heading=ship_state.forward,
        acceleration_g=ship_state.max_acceleration_g(),
        remaining_delta_v_kps=ship_state.remaining_delta_v_kps(),
        hull_integrity_percent=100.0,  # Default, can be calculated from modules
        propellant_percent=(ship_state.propellant_kg /
                          (ship_state.wet_mass_kg - ship_state.dry_mass_kg) * 100.0
                          if ship_state.wet_mass_kg > ship_state.dry_mass_kg else 100.0)
    )

    # Build thermal status
    if thermal_system:
        radiator_status = {}
        for pos, radiator in thermal_system.radiators.radiators.items():
            radiator_status[pos.value] = radiator.state.value

        thermal = ThermalStatus(
            heat_percent=thermal_system.heat_percent,
            is_overheating=thermal_system.is_overheating,
            is_critical=thermal_system.is_critical,
            radiator_status=radiator_status,
            dissipation_kw=thermal_system.radiators.total_dissipation_kw,
            capacity_gj=thermal_system.heatsink.capacity_gj
        )
    else:
        thermal = ThermalStatus(
            heat_percent=0.0,
            is_overheating=False,
            is_critical=False,
            radiator_status={},
            dissipation_kw=0.0,
            capacity_gj=0.0
        )

    # Build weapon statuses
    weapon_statuses: List[WeaponStatus] = []
    if weapons:
        for i, weapon in enumerate(weapons):
            weapon_type = "spinal" if weapon.mount == "nose_only" else "turret"
            if weapon.is_missile:
                weapon_type = "torpedo"

            weapon_statuses.append(WeaponStatus(
                weapon_id=f"weapon_{i}",
                weapon_name=weapon.name,
                weapon_type=weapon_type,
                ready=True,  # Would need cooldown tracking
                cooldown_remaining_s=0.0,
                ammo_remaining=weapon.magazine,
                target_locked=False,
                in_arc=True  # Would need actual arc calculation
            ))

    # Build damage status
    armor_sections: Dict[str, float] = {}
    if armor:
        for loc, armor_section in armor.sections.items():
            armor_sections[loc.value] = armor_section.thickness_cm

    damaged_modules: List[str] = []
    destroyed_modules: List[str] = []
    if module_layout:
        for module in module_layout.modules:
            if module.is_destroyed:
                destroyed_modules.append(module.name)
            elif module.health < module.max_health:
                damaged_modules.append(module.name)

    damage = DamageStatus(
        armor_sections=armor_sections,
        critical_systems_damaged=[m for m in damaged_modules if module_layout and
                                  any(mod.name == m and mod.is_critical for mod in module_layout.modules)],
        modules_destroyed=destroyed_modules,
        overall_integrity=100.0 - (len(destroyed_modules) * 10)  # Simple estimate
    )

    # Build enemy contacts
    enemy_contacts: List[ContactInfo] = []
    if enemy_ships:
        for enemy in enemy_ships:
            enemy_pos = Vector3D(*enemy.get('position', [0, 0, 0]))
            enemy_vel = Vector3D(*enemy.get('velocity', [0, 0, 0]))

            rel_pos = enemy_pos - ship_state.position
            distance = rel_pos.magnitude

            # Calculate bearing (0 = forward, 90 = right)
            if distance > 0:
                rel_dir = rel_pos.normalized()
                forward_component = ship_state.forward.dot(rel_dir)
                right_component = ship_state.right.dot(rel_dir)
                bearing = math.degrees(math.atan2(right_component, forward_component))
            else:
                bearing = 0.0

            # Calculate closing rate
            rel_vel = enemy_vel - ship_state.velocity
            closing_rate = -rel_vel.dot(rel_pos.normalized()) if distance > 0 else 0

            # Estimate time to intercept
            if closing_rate > 0:
                tti = distance / closing_rate
            else:
                tti = float('inf')

            enemy_contacts.append(ContactInfo(
                contact_id=enemy.get('id', f"enemy_{len(enemy_contacts)}"),
                contact_type=ThreatType.SHIP,
                position=enemy_pos,
                velocity=enemy_vel,
                distance_km=distance / 1000.0,
                bearing_deg=bearing,
                closing_rate_kps=closing_rate / 1000.0,
                time_to_intercept_s=tti,
                is_hostile=True,
                classification=enemy.get('class', 'Unknown')
            ))

    # Build incoming threats
    incoming_threats: List[ContactInfo] = []

    if incoming_torpedoes:
        for torpedo in incoming_torpedoes:
            rel_pos = torpedo.position - ship_state.position
            distance = rel_pos.magnitude
            rel_vel = torpedo.velocity - ship_state.velocity
            closing_rate = -rel_vel.dot(rel_pos.normalized()) if distance > 0 else 0

            if closing_rate > 0:
                tti = distance / closing_rate
            else:
                tti = float('inf')

            if distance > 0:
                rel_dir = rel_pos.normalized()
                forward_component = ship_state.forward.dot(rel_dir)
                right_component = ship_state.right.dot(rel_dir)
                bearing = math.degrees(math.atan2(right_component, forward_component))
            else:
                bearing = 0.0

            incoming_threats.append(ContactInfo(
                contact_id=torpedo.target_id,
                contact_type=ThreatType.TORPEDO,
                position=torpedo.position,
                velocity=torpedo.velocity,
                distance_km=distance / 1000.0,
                bearing_deg=bearing,
                closing_rate_kps=closing_rate / 1000.0,
                time_to_intercept_s=tti,
                is_hostile=True,
                classification="Torpedo"
            ))

    if incoming_projectiles:
        for proj in incoming_projectiles:
            rel_pos = proj.position - ship_state.position
            distance = rel_pos.magnitude
            rel_vel = proj.velocity - ship_state.velocity
            closing_rate = -rel_vel.dot(rel_pos.normalized()) if distance > 0 else 0

            if closing_rate > 0:
                tti = distance / closing_rate
            else:
                tti = float('inf')

            if distance > 0:
                rel_dir = rel_pos.normalized()
                forward_component = ship_state.forward.dot(rel_dir)
                right_component = ship_state.right.dot(rel_dir)
                bearing = math.degrees(math.atan2(right_component, forward_component))
            else:
                bearing = 0.0

            incoming_threats.append(ContactInfo(
                contact_id=f"projectile_{id(proj)}",
                contact_type=ThreatType.PROJECTILE,
                position=proj.position,
                velocity=proj.velocity,
                distance_km=distance / 1000.0,
                bearing_deg=bearing,
                closing_rate_kps=closing_rate / 1000.0,
                time_to_intercept_s=tti,
                is_hostile=True,
                classification="Kinetic Projectile"
            ))

    return SensorSnapshot(
        timestamp=timestamp,
        own_ship=own_ship,
        thermal=thermal,
        weapons=weapon_statuses,
        damage=damage,
        enemy_contacts=enemy_contacts,
        incoming_threats=incoming_threats
    )


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS CAPTAIN INTERFACE - SELF TEST")
    print("=" * 70)

    # Create test ship state
    ship_state = ShipState(
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(50000, 0, 0),  # 50 km/s
        forward=Vector3D(1, 0, 0),
        up=Vector3D(0, 0, 1),
        mass_kg=1_990_000,
        dry_mass_kg=1_895_000,
        propellant_kg=95_000,
        thrust_n=58.56e6,
        exhaust_velocity_ms=10_256_000
    )

    # Create test enemy
    enemy_ships = [
        {
            'id': 'enemy_alpha',
            'position': [500_000, 50_000, 0],  # 500 km ahead, 50 km right
            'velocity': [-30000, 0, 0],  # 30 km/s toward us
            'class': 'Destroyer'
        }
    ]

    # Create sensor snapshot
    snapshot = create_sensor_snapshot(
        timestamp=120.5,
        ship_state=ship_state,
        enemy_ships=enemy_ships
    )

    # Test SituationReport
    print("\n--- SITUATION REPORT TEST ---")
    report = SituationReport.generate(snapshot)
    print(report)

    # Test CaptainInterface
    print("\n--- CAPTAIN INTERFACE TEST ---")
    interface = CaptainInterface()

    # Test JSON output
    print("\n-- JSON Output --")
    json_output = interface.to_json(snapshot)
    print(json_output[:500] + "...")  # First 500 chars

    # Test response parsing
    print("\n-- Response Parsing Test --")

    # Test natural language parsing
    natural_response = "Execute evasive maneuvers and fire spinal weapon at the nearest enemy. ECM on."
    decision = interface.parse_response(natural_response)
    print(f"Natural language parsed:")
    print(f"  Maneuver: {decision.maneuver.maneuver_type.value if decision.maneuver else 'None'}")
    print(f"  Weapon commands: {[wc.command_type.value for wc in decision.weapon_commands]}")
    print(f"  Defensive commands: {[dc.command_type.value for dc in decision.defensive_commands]}")

    # Test JSON parsing
    json_response = '''
    {
        "timestamp": 120.5,
        "maneuver": {
            "type": "burn_toward",
            "target": "enemy_alpha",
            "throttle": 0.8
        },
        "weapon_commands": [
            {"type": "fire_spinal", "target_id": "enemy_alpha"}
        ],
        "tactical_intent": "Close and engage"
    }
    '''
    decision_json = interface.parse_response(json_response)
    print(f"\nJSON parsed:")
    print(f"  Maneuver: {decision_json.maneuver.maneuver_type.value if decision_json.maneuver else 'None'}")
    print(f"  Tactical intent: {decision_json.tactical_intent}")

    # Test validation
    print("\n-- Decision Validation Test --")
    is_valid, errors = interface.validate_decision(decision_json, snapshot)
    print(f"Decision valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")

    # Test ThreatAssessment
    print("\n--- THREAT ASSESSMENT TEST ---")

    # Create test torpedo
    test_torpedo = Torpedo(
        specs=TorpedoSpecs(),
        position=Vector3D(100_000, 10_000, 0),  # 100 km away
        velocity=Vector3D(-60000, -5000, 0),  # Closing at 60 km/s
        target_id="torpedo_001",
        guidance_mode=GuidanceMode.PROPORTIONAL_NAV
    )

    assessment = ThreatAssessment.from_torpedo(
        torpedo=test_torpedo,
        own_position=ship_state.position,
        own_velocity=ship_state.velocity,
        remaining_delta_v_kps=ship_state.remaining_delta_v_kps()
    )

    print(f"Torpedo Threat Assessment:")
    print(f"  Threat Level: {assessment.threat_level:.2f}")
    print(f"  Time to Impact: {assessment.time_to_impact_s:.1f}s")
    print(f"  Impact Energy: {assessment.impact_energy_gj:.1f} GJ")
    print(f"  Can Evade: {assessment.can_evade}")
    print(f"  Evade Delta-V Required: {assessment.evade_delta_v_kps:.2f} km/s")
    print(f"  Recommended Action: {assessment.recommended_action}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
