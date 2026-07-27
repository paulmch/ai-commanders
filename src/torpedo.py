#!/usr/bin/env python3
"""
Torpedo Simulation Module for AI Commanders Space Battle Simulator

Implements realistic torpedo mechanics for space combat:
- TorpedoSpecs: Physical parameters derived from fleet data
- Torpedo: Individual torpedo state and trajectory
- TorpedoLauncher: Launch calculations and optimal release points
- TorpedoGuidance: Multiple guidance modes (pursuit, proportional nav, terminal)

Torpedo Classes (based on Terra Invicta research):
- Trident (default): 250 kg penetrator, 16 km/s delta-v, 12g accel (fusion torch drive)
- Poseidon (NTR): 100 kg penetrator, 18.5 km/s delta-v, 4.9g accel (best human NTR)
- Athena (explosive): 600 kg warhead, 12.8 km/s delta-v, 4.9g accel

Performance comparison vs Alien Iridescent Star (256 kg, 14.2 km/s dv, 11.5g):
- Trident matches or exceeds alien performance in acceleration and delta-v

Kinetic penetrator damage: KE = 0.5 * mass * velocity^2
- 250 kg @ 5 km/s (tail chase) = 3.1 GJ
- 250 kg @ 12 km/s = 18 GJ
- 250 kg @ 18 km/s (head-on) = 40.5 GJ

Physics based on Newtonian mechanics with Tsiolkovsky rocket equation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

try:
    from .physics import Vector3D, tsiolkovsky_delta_v, mass_after_burn
except ImportError:
    from physics import Vector3D, tsiolkovsky_delta_v, mass_after_burn


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Default torpedo specifications (derived from fleet_ships.json)
DEFAULT_WARHEAD_YIELD_GJ = 0.0  # Pure kinetic penetrator - no explosive warhead
DEFAULT_TORPEDO_MASS_KG = 1600.0  # Total mass including propellant
DEFAULT_PROPELLANT_FRACTION = 0.70  # 70% of mass is propellant
DEFAULT_EXHAUST_VELOCITY_KPS = 50.0  # km/s (high-thrust chemical/plasma hybrid)

# Guidance constants
SAFE_ARMING_DISTANCE_M = 500.0  # Minimum distance before torpedo arms
TERMINAL_APPROACH_DISTANCE_M = 10_000.0  # Switch to terminal guidance
PROPORTIONAL_NAV_CONSTANT = 3.0  # N' for proportional navigation

# =============================================================================
# AUGMENTED PROPORTIONAL NAVIGATION / NO-ESCAPE-ZONE CONSTANTS
# =============================================================================

G0_MS2 = 9.81  # standard gravity (m/s^2)

# Cruise closing-speed floor (km/s). Below this the torpedo prioritises
# building closure over conserving delta-v: a slow torpedo lingers inside
# point-defense envelopes and may never complete the intercept at all.
# 12 km/s crosses a 100 km PD envelope in ~8 s.
MIN_CLOSING_SPEED_KPS = 12.0

# Navigation ratio N for (augmented) proportional navigation. PN theory
# requires N > 2 for the predicted miss to collapse against a maneuvering
# target (below that, LOS rotation regenerates lateral velocity as fast as
# the correction removes it); practical missiles use N in [3, 5]. N = 4 sits
# mid-band: firm authority without amplifying estimate noise.
APN_NAV_RATIO = 4.0

# Design assumption for the worst-case target: the most agile hull in the
# fleet (corvette) pulls 3.0 g of lateral acceleration (fleet_ships.json,
# performance.combat_acceleration_g). The no-escape zone must assume the
# target can use ALL of this capability at any moment, regardless of what it
# is currently observed doing - a commit decision based on the *observed*
# acceleration would be defeated by a target that waits to jink.
ASSUMED_TARGET_MAX_LATERAL_ACCEL_G = 3.0
ASSUMED_TARGET_MAX_LATERAL_ACCEL_MS2 = ASSUMED_TARGET_MAX_LATERAL_ACCEL_G * G0_MS2

# Guidance-overhead margin applied to both NEZ conditions (acceleration and
# delta-v). Two standard PN results motivate the value:
#   - against a constant-g target, APN's peak commanded lateral acceleration
#     is (N/2) * a_target (the feed-forward term) - N/2 times the kinematic
#     minimum needed to match the target's displacement;
#   - nulling an initial heading error under PN costs up to N/(N-2) times
#     the initial lateral rate in delta-v.
# At N = 4 both factors equal 2.0, so a single margin covers both shares.
NEZ_GUIDANCE_MARGIN = APN_NAV_RATIO / 2.0

# Delta-v the torpedo must still hold at impact so late corrections remain
# possible (replaces the old TERMINAL_MIN_RESERVE_KPS). Nulling one full
# last-second 3 g jink over the final ~8 s costs ~0.24 km/s; 0.5 km/s covers
# it with the NEZ_GUIDANCE_MARGIN applied.
NEZ_DV_RESERVE_KPS = 0.5

# Do not light the main engine for closure gains smaller than this (km/s).
TERMINAL_MIN_BURN_KPS = 0.2

# Predicted-miss deadband (m): once the zero-effort miss is below hull
# scale, stop steering. Hulls in this sim are 65-300 m long, so 50 m is
# inside every target's physical envelope.
APN_MISS_DEADBAND_M = 50.0

# Time constant (s) for the exponential filter over the finite-differenced
# target-acceleration estimate. Observations in this sim are exact, so the
# filter only suppresses single-tick spikes (impulsive maneuver switches);
# 0.5 s converges ~95% within three guidance ticks at dt = 0.5 s.
TARGET_ACCEL_FILTER_TAU_S = 0.5


# =============================================================================
# TORPEDO SPECIFICATIONS
# =============================================================================

@dataclass
class TorpedoSpecs:
    """
    Physical specifications for a torpedo type.

    Trident (default) kinetic penetrator:
    - 2000 kg total mass
    - 250 kg dense penetrator (tungsten/DU)
    - 12 km/s delta-v, 8g acceleration
    - 6 km/s exhaust velocity (advanced chemical)

    Attributes:
        warhead_yield_gj: Explosive warhead yield (0 for kinetic penetrator)
        penetrator_mass_kg: Mass of the kinetic penetrator (dense material)
        mass_kg: Total wet mass including propellant (kg)
        thrust_n: Engine thrust (Newtons)
        exhaust_velocity_kps: Exhaust velocity (km/s)
        propellant_fraction: Fraction of mass that is propellant (0.0 to 1.0)
        total_delta_v_kps: Calculated total delta-v budget (km/s)
        dry_mass_kg: Mass without propellant (kg)
        propellant_mass_kg: Initial propellant mass (kg)
        rcs_thrust_fraction: Fraction of main thrust for lateral RCS (default 5%)
    """
    warhead_yield_gj: float = DEFAULT_WARHEAD_YIELD_GJ
    penetrator_mass_kg: float = 250.0  # 250 kg dense tungsten/DU kinetic penetrator
    mass_kg: float = DEFAULT_TORPEDO_MASS_KG
    thrust_n: float = 0.0  # Calculated in __post_init__
    exhaust_velocity_kps: float = DEFAULT_EXHAUST_VELOCITY_KPS
    propellant_fraction: float = DEFAULT_PROPELLANT_FRACTION
    rcs_thrust_fraction: float = 0.05  # Lateral RCS is 5% of main thrust

    # Calculated fields
    total_delta_v_kps: float = field(init=False, default=0.0)
    dry_mass_kg: float = field(init=False, default=0.0)
    propellant_mass_kg: float = field(init=False, default=0.0)
    rcs_thrust_n: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        """Calculate derived values after initialization."""
        # Calculate dry and propellant masses
        self.propellant_mass_kg = self.mass_kg * self.propellant_fraction
        self.dry_mass_kg = self.mass_kg - self.propellant_mass_kg

        # Calculate total delta-v using Tsiolkovsky equation
        # delta_v = v_e * ln(m_wet / m_dry)
        exhaust_velocity_ms = self.exhaust_velocity_kps * 1000
        delta_v_ms = tsiolkovsky_delta_v(
            exhaust_velocity_ms=exhaust_velocity_ms,
            wet_mass_kg=self.mass_kg,
            dry_mass_kg=self.dry_mass_kg
        )
        self.total_delta_v_kps = delta_v_ms / 1000

        # Estimate thrust if not provided
        # Assume 10g acceleration at wet mass for high-thrust torpedo motor
        if self.thrust_n <= 0:
            # F = m * a, where a = 10g = 98.1 m/s^2
            self.thrust_n = self.mass_kg * 98.1

        # Calculate RCS thrust for lateral corrections
        self.rcs_thrust_n = self.thrust_n * self.rcs_thrust_fraction

    @classmethod
    def from_fleet_data(
        cls,
        warhead_yield_gj: float = DEFAULT_WARHEAD_YIELD_GJ,
        penetrator_mass_kg: float = 250.0,
        ammo_mass_kg: float = DEFAULT_TORPEDO_MASS_KG,
        range_km: float = 1500.0,
        propellant_fraction: float = DEFAULT_PROPELLANT_FRACTION,
        exhaust_velocity_kps: float = DEFAULT_EXHAUST_VELOCITY_KPS
    ) -> TorpedoSpecs:
        """
        Create torpedo specs from fleet data.

        Args:
            warhead_yield_gj: Explosive warhead yield (0 for kinetic penetrator)
            penetrator_mass_kg: Mass of kinetic penetrator (kg)
            ammo_mass_kg: Total torpedo mass (kg)
            range_km: Nominal range (km) - used for reference only
            propellant_fraction: Fraction of mass as propellant
            exhaust_velocity_kps: Exhaust velocity (km/s)

        Returns:
            Configured TorpedoSpecs
        """
        return cls(
            warhead_yield_gj=warhead_yield_gj,
            penetrator_mass_kg=penetrator_mass_kg,
            mass_kg=ammo_mass_kg,
            exhaust_velocity_kps=exhaust_velocity_kps,
            propellant_fraction=propellant_fraction
        )

    @classmethod
    def from_terra_invicta(
        cls,
        name: str,
        delta_v_kps: float,
        acceleration_g: float,
        exhaust_velocity_kps: float,
        warhead_mass_kg: float,
        ammo_mass_kg: float,
        fuel_mass_kg: float,
        system_mass_kg: float,
        warhead_class: str = "Penetrator",
        turn_rate_deg_s: float = 20.0
    ) -> 'TorpedoSpecs':
        """
        Create torpedo specs from Terra Invicta game data.

        Args:
            name: Torpedo name (for reference)
            delta_v_kps: Total delta-v budget (km/s)
            acceleration_g: Acceleration in g's
            exhaust_velocity_kps: Exhaust velocity (km/s)
            warhead_mass_kg: Warhead mass (kg)
            ammo_mass_kg: Total ammo mass per torpedo (kg)
            fuel_mass_kg: Fuel/propellant mass (kg)
            system_mass_kg: System/structure mass (kg)
            warhead_class: Warhead type (Explosive, Penetrator, etc.)
            turn_rate_deg_s: Turn rate in degrees per second

        Returns:
            Configured TorpedoSpecs
        """
        # Calculate total torpedo mass
        total_mass_kg = warhead_mass_kg + fuel_mass_kg + system_mass_kg

        # Calculate propellant fraction
        propellant_fraction = fuel_mass_kg / total_mass_kg

        # Calculate thrust from acceleration: F = m * a
        thrust_n = total_mass_kg * acceleration_g * 9.81

        # Warhead yield estimate based on warhead class and mass
        # Penetrator: kinetic energy focus, no explosive yield
        # Explosive: high yield per mass
        if warhead_class == "Penetrator":
            # Penetrators rely on kinetic energy, no explosive yield
            warhead_yield_gj = 0.0
            penetrator_mass = warhead_mass_kg  # The warhead IS the penetrator
        elif warhead_class == "Explosive":
            warhead_yield_gj = warhead_mass_kg * 0.003  # 3 MJ/kg (like Athena's 1.8 GJ / 600 kg)
            penetrator_mass = 0.0  # No kinetic penetrator
        elif warhead_class == "Fragmentation":
            warhead_yield_gj = warhead_mass_kg * 0.002  # 2 MJ/kg
            penetrator_mass = 0.0  # No kinetic penetrator
        else:
            warhead_yield_gj = warhead_mass_kg * 0.003
            penetrator_mass = 0.0

        specs = cls(
            warhead_yield_gj=warhead_yield_gj,
            penetrator_mass_kg=penetrator_mass,
            mass_kg=total_mass_kg,
            thrust_n=thrust_n,
            exhaust_velocity_kps=exhaust_velocity_kps,
            propellant_fraction=propellant_fraction
        )

        # Store additional Terra Invicta data
        specs._ti_name = name
        specs._ti_turn_rate_deg_s = turn_rate_deg_s
        specs._ti_warhead_class = warhead_class

        return specs

    @classmethod
    def poseidon(cls) -> 'TorpedoSpecs':
        """
        Create Poseidon (Hestia) torpedo specs - best human NTR torpedo.

        Poseidon specs from Terra Invicta:
        - Delta-V: 18.5 km/s (highest human conventional)
        - Acceleration: 4.89g
        - Exhaust velocity: 8.18 km/s (Nuclear Thermal Rocket)
        - Warhead: 100 kg penetrator
        - Turn rate: 20°/s
        """
        return cls.from_terra_invicta(
            name="Poseidon",
            delta_v_kps=18.5,
            acceleration_g=4.89,
            exhaust_velocity_kps=8.18,
            warhead_mass_kg=100,
            ammo_mass_kg=4800,  # Per torpedo bay
            fuel_mass_kg=4300,
            system_mass_kg=400,
            warhead_class="Penetrator",
            turn_rate_deg_s=20.0
        )

    @classmethod
    def athena(cls) -> 'TorpedoSpecs':
        """
        Create Athena torpedo specs - balanced NTR torpedo.

        Athena specs from Terra Invicta:
        - Delta-V: 12.83 km/s
        - Acceleration: 4.89g
        - Exhaust velocity: 8.18 km/s (Nuclear Thermal Rocket)
        - Warhead: 600 kg explosive (1.8 GJ)
        - Turn rate: 20°/s
        """
        return cls.from_terra_invicta(
            name="Athena",
            delta_v_kps=12.83,
            acceleration_g=4.89,
            exhaust_velocity_kps=8.18,
            warhead_mass_kg=600,
            ammo_mass_kg=4800,
            fuel_mass_kg=3800,
            system_mass_kg=400,
            warhead_class="Explosive",
            turn_rate_deg_s=20.0
        )

    @classmethod
    def trident(cls) -> 'TorpedoSpecs':
        """
        Create Trident torpedo specs - advanced kinetic penetrator.

        Trident uses a compact fusion torch drive:
        - Delta-V: 14.0 km/s (excellent range)
        - Acceleration: 12.0g (high-g intercept capability)
        - Exhaust velocity: 8.0 km/s (compact fusion torch)
        - Penetrator: 250 kg tungsten/DU (2.5x standard)
        - Turn rate: 30°/s (agile)

        Performance comparison:
        - Poseidon NTR: 4.9g, 18.5 km/s dv (long range, low accel)
        - Trident: 12g, 14 km/s dv (balanced, high accel)
        - Alien Iridescent: 11.5g, 14.2 km/s dv (matched!)

        Damage at impact:
        - 5 km/s (tail chase): 3.1 GJ
        - 12 km/s (medium): 18 GJ
        - 18 km/s (head-on): 40.5 GJ
        """
        return cls.from_terra_invicta(
            name="Trident",
            delta_v_kps=14.0,
            acceleration_g=12.0,
            exhaust_velocity_kps=8.0,
            warhead_mass_kg=250,
            ammo_mass_kg=3600,
            fuel_mass_kg=2950,
            system_mass_kg=400,
            warhead_class="Penetrator",
            turn_rate_deg_s=30.0
        )

    def acceleration_at_mass(self, current_mass_kg: float) -> float:
        """
        Calculate acceleration at a given mass.

        Args:
            current_mass_kg: Current torpedo mass (kg)

        Returns:
            Acceleration in m/s^2
        """
        if current_mass_kg <= 0:
            return 0.0
        return self.thrust_n / current_mass_kg

    def burn_time_seconds(self) -> float:
        """
        Calculate total burn time until fuel exhaustion.

        Using mass flow rate: dm/dt = F / v_e
        Total burn time = propellant_mass / mass_flow_rate

        Returns:
            Total burn time in seconds
        """
        exhaust_velocity_ms = self.exhaust_velocity_kps * 1000
        mass_flow_rate = self.thrust_n / exhaust_velocity_ms  # kg/s
        if mass_flow_rate <= 0:
            return float('inf')
        return self.propellant_mass_kg / mass_flow_rate


# =============================================================================
# GUIDANCE MODES
# =============================================================================

class GuidanceMode(Enum):
    """Torpedo guidance modes."""
    PURSUIT = auto()           # Point directly at target, full thrust
    INTERCEPT = auto()         # Cruise phase: closure-priority, burn toward intercept point
    PROPORTIONAL_NAV = auto()  # Terminal phase: precise LOS tracking
    TERMINAL = auto()          # Final approach, maximize closing velocity
    COAST = auto()             # No thrust, fuel conservation or exhausted
    SMART = auto()             # Smart guidance: pursuit cone + fuel reserve management
    COLLISION = auto()         # Collision course: align relative velocity with LOS


@dataclass
class GuidanceCommand:
    """
    Command output from guidance system.

    Supports simultaneous main engine + RCS burns for optimal guidance:
    - Main engine: Burns toward target to build closing speed
    - RCS: Corrects lateral velocity simultaneously

    Attributes:
        direction: Main engine thrust direction (normalized), or zero for coast
        throttle: Main engine throttle setting 0.0 to 1.0
        reason: Human-readable reason for this command
        use_rcs: If True, use ONLY lateral RCS (no main engine)
        rcs_direction: Simultaneous RCS correction direction (optional)
        rcs_throttle: RCS throttle for simultaneous correction (0.0 to 1.0)
    """
    direction: Vector3D
    throttle: float = 1.0
    reason: str = ""
    use_rcs: bool = False
    rcs_direction: Optional[Vector3D] = None
    rcs_throttle: float = 0.0

    @classmethod
    def coast(cls, reason: str = "coasting") -> 'GuidanceCommand':
        """Create a coast command (no thrust)."""
        return cls(direction=Vector3D.zero(), throttle=0.0, reason=reason)

    @classmethod
    def burn(cls, direction: Vector3D, throttle: float = 1.0, reason: str = "") -> 'GuidanceCommand':
        """Create a burn command using main engine only."""
        return cls(direction=direction.normalized(), throttle=throttle, reason=reason)

    @classmethod
    def rcs_correction(cls, direction: Vector3D, throttle: float = 1.0, reason: str = "") -> 'GuidanceCommand':
        """Create an RCS-only correction command for fine lateral adjustments."""
        return cls(direction=direction.normalized(), throttle=throttle, reason=reason, use_rcs=True)

    @classmethod
    def combined(
        cls,
        main_direction: Vector3D,
        main_throttle: float,
        rcs_direction: Vector3D,
        rcs_throttle: float,
        reason: str = ""
    ) -> 'GuidanceCommand':
        """
        Create a combined main engine + RCS command.

        Main engine builds closing speed while RCS corrects lateral drift.
        This is the most efficient guidance mode.
        """
        return cls(
            direction=main_direction.normalized(),
            throttle=main_throttle,
            reason=reason,
            use_rcs=False,
            rcs_direction=rcs_direction.normalized() if rcs_direction.magnitude > 0.01 else None,
            rcs_throttle=rcs_throttle
        )


# =============================================================================
# NO-ESCAPE-ZONE STATUS
# =============================================================================

@dataclass
class NEZStatus:
    """
    Result of a no-escape-zone evaluation (see TorpedoGuidance.nez_status).

    The torpedo is "inside the NEZ" when, even if the target immediately
    commits its full assumed lateral acceleration in the worst direction for
    the rest of the flight, the torpedo still has both the acceleration
    authority and the delta-v to null the resulting miss (with guidance
    margin and a terminal reserve).

    Attributes:
        inside: True if the target can no longer escape.
        reason: Which condition failed (or "inside" when none did).
        time_to_go_s: Estimated time to impact at current closing rate.
        target_escape_displacement_m: Max lateral displacement the target can
            still add before impact (0.5 * a_t * t_go^2).
        required_lateral_accel_ms2: Peak lateral acceleration the APN law
            would command in the worst case (must not saturate the engine).
        available_lateral_accel_ms2: Torpedo's current full-vector thrust
            acceleration (the engine is fully gimbaled, so all of it is
            available laterally).
        accel_margin: available / required (inf when nothing is required).
        required_delta_v_kps: Worst-case correction budget including the
            terminal reserve.
        available_delta_v_kps: Torpedo's remaining delta-v.
        dv_margin_kps: available - required (negative means outside).
    """
    inside: bool
    reason: str
    time_to_go_s: float
    target_escape_displacement_m: float
    required_lateral_accel_ms2: float
    available_lateral_accel_ms2: float
    accel_margin: float
    required_delta_v_kps: float
    available_delta_v_kps: float
    dv_margin_kps: float


# =============================================================================
# TORPEDO CLASS
# =============================================================================

@dataclass
class Torpedo:
    """
    Individual torpedo state and trajectory tracking.

    Maintains position, velocity, fuel state, and targeting information.
    Steering is accomplished via thrust vectoring only (no RCS).

    Velocity Inheritance:
        Torpedoes inherit the shooter's velocity at launch. The initial
        velocity field should already include the shooter's velocity.
        The launched_from_velocity field stores the shooter's velocity
        for reference and relative velocity calculations.

    Attributes:
        specs: Physical specifications
        position: Current position (meters)
        velocity: Current velocity (m/s) - INCLUDES shooter velocity at launch
        target_id: Identifier of target entity
        remaining_delta_v_kps: Remaining delta-v budget (km/s)
        current_mass_kg: Current mass including remaining propellant (kg)
        fuel_exhausted: True if no propellant remains
        armed: True if warhead is armed (after safe distance)
        guidance_mode: Current guidance mode
        time_since_launch: Time since launch (seconds)
        launch_position: Position at launch (meters)
        launched_from_velocity: Velocity of shooter at launch (m/s) - for reference
    """
    specs: TorpedoSpecs
    position: Vector3D
    velocity: Vector3D
    target_id: str
    remaining_delta_v_kps: float = field(init=False)
    current_mass_kg: float = field(init=False)
    fuel_exhausted: bool = False
    armed: bool = False
    guidance_mode: GuidanceMode = GuidanceMode.PURSUIT
    time_since_launch: float = 0.0
    launch_position: Vector3D = field(default_factory=Vector3D.zero)
    launched_from_velocity: Vector3D = field(default_factory=Vector3D.zero)

    # --- Guidance filter state (owned by the torpedo, NOT TorpedoGuidance) ---
    # The simulation constructs a fresh TorpedoGuidance every tick, so any
    # state that must survive between guidance calls has to live on the
    # torpedo itself. These two fields carry the target-acceleration
    # estimator used by augmented proportional navigation: the previously
    # observed target velocity, and the filtered finite-difference estimate
    # of the target's acceleration. See
    # TorpedoGuidance._update_target_accel_estimate.
    last_observed_target_velocity: Optional[Vector3D] = field(
        default=None, init=False, repr=False
    )
    estimated_target_accel: Vector3D = field(
        default_factory=Vector3D.zero, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize calculated fields."""
        self.remaining_delta_v_kps = self.specs.total_delta_v_kps
        self.current_mass_kg = self.specs.mass_kg
        self.launch_position = Vector3D(
            self.position.x, self.position.y, self.position.z
        )

    def distance_from_launch(self) -> float:
        """
        Calculate distance traveled from launch point.

        Returns:
            Distance in meters
        """
        return self.position.distance_to(self.launch_position)

    def update(
        self,
        dt_seconds: float,
        target_position: Vector3D,
        target_velocity: Vector3D
    ) -> None:
        """
        Update torpedo state for one time step.

        Updates position, velocity, fuel state, and arming status.
        Uses current guidance mode to determine thrust direction.

        Args:
            dt_seconds: Time step (seconds)
            target_position: Current target position (meters)
            target_velocity: Current target velocity (m/s)
        """
        self.time_since_launch += dt_seconds

        # Check arming status
        if not self.armed:
            if self.distance_from_launch() >= SAFE_ARMING_DISTANCE_M:
                self.armed = True

        # Get the FULL guidance command (direction + throttle + RCS channel).
        # Using only the direction here silently executed RCS-only precision trims
        # as full-throttle main-engine burns (~20x the intended delta-v), which
        # both wasted propellant and massively over-corrected.
        guidance = TorpedoGuidance()
        command = guidance.update_guidance_command(
            self, target_position, target_velocity, dt_seconds
        )

        # Apply primary thrust on the channel the guidance actually asked for
        if (not self.fuel_exhausted
                and command.throttle > 0.0
                and command.direction.magnitude > 0.01):
            if command.use_rcs:
                self.apply_lateral_thrust(command.direction, dt_seconds, command.throttle)
            else:
                self.apply_thrust(command.direction, dt_seconds, command.throttle)

        # Apply simultaneous RCS trim if the command carries one (combined mode)
        if (not self.fuel_exhausted
                and command.rcs_direction is not None
                and command.rcs_throttle > 0.0):
            self.apply_lateral_thrust(
                command.rcs_direction, dt_seconds, command.rcs_throttle
            )

        # Update position (Euler integration)
        # position += velocity * dt
        self.position = self.position + self.velocity * dt_seconds

    def calculate_intercept(
        self,
        target_pos: Vector3D,
        target_vel: Vector3D,
        target_accel: Optional[Vector3D] = None
    ) -> Tuple[Vector3D, float]:
        """
        Calculate intercept point with target.

        Uses iterative refinement to find intercept point accounting for
        torpedo acceleration capability and target motion.

        Args:
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            target_accel: Target acceleration (m/s^2), assumes constant

        Returns:
            Tuple of (intercept_position, time_to_intercept)
        """
        if target_accel is None:
            target_accel = Vector3D.zero()

        # Initial estimate: relative position and velocity
        rel_pos = target_pos - self.position
        rel_vel = target_vel - self.velocity

        # Current acceleration capability
        current_accel = self.specs.acceleration_at_mass(self.current_mass_kg)

        # Simple intercept estimate: time to close distance at closing rate
        distance = rel_pos.magnitude
        closing_rate = -rel_vel.dot(rel_pos.normalized()) if distance > 0 else 0

        # Initial time estimate
        if closing_rate > 0:
            # Already closing
            t_intercept = distance / closing_rate
        else:
            # Need to accelerate toward target
            # Use kinematic equation: d = v0*t + 0.5*a*t^2
            # Estimate assuming we accelerate directly toward target
            if current_accel > 0:
                # Quadratic solution
                discriminant = rel_vel.magnitude_squared + 2 * current_accel * distance
                if discriminant >= 0:
                    t_intercept = (-rel_vel.magnitude + math.sqrt(discriminant)) / current_accel
                    t_intercept = max(1.0, t_intercept)
                else:
                    t_intercept = distance / 1000  # Fallback estimate
            else:
                t_intercept = float('inf')

        # Cap at reasonable maximum
        t_intercept = min(t_intercept, 3600.0)  # 1 hour max

        # Iterative refinement (3 iterations)
        for _ in range(3):
            # Predict target position at intercept time
            # pos = pos0 + vel*t + 0.5*accel*t^2
            predicted_target = (
                target_pos +
                target_vel * t_intercept +
                target_accel * (0.5 * t_intercept * t_intercept)
            )

            # Update intercept time based on new distance
            new_distance = (predicted_target - self.position).magnitude
            if closing_rate > 0:
                t_intercept = new_distance / closing_rate
            elif current_accel > 0:
                discriminant = rel_vel.magnitude_squared + 2 * current_accel * new_distance
                if discriminant >= 0:
                    t_intercept = (-rel_vel.magnitude + math.sqrt(discriminant)) / current_accel
                    t_intercept = max(1.0, t_intercept)

            t_intercept = min(t_intercept, 3600.0)

        # Final intercept position
        intercept_pos = (
            target_pos +
            target_vel * t_intercept +
            target_accel * (0.5 * t_intercept * t_intercept)
        )

        return intercept_pos, t_intercept

    def apply_thrust(
        self,
        direction: Vector3D,
        dt_seconds: float,
        throttle: float = 1.0
    ) -> None:
        """
        Apply thrust in specified direction, consuming fuel.

        Updates velocity and mass based on rocket equation.

        Args:
            direction: Thrust direction (will be normalized)
            dt_seconds: Burn duration (seconds)
            throttle: Throttle setting 0.0 to 1.0
        """
        if self.fuel_exhausted or throttle <= 0:
            return

        direction = direction.normalized()
        if direction.magnitude < 0.01:
            return

        throttle = max(0.0, min(1.0, throttle))

        # Calculate thrust and mass flow
        thrust_n = self.specs.thrust_n * throttle
        exhaust_velocity_ms = self.specs.exhaust_velocity_kps * 1000
        mass_flow_rate = thrust_n / exhaust_velocity_ms  # kg/s

        # Propellant consumed this step
        propellant_consumed = mass_flow_rate * dt_seconds

        # Check available propellant
        propellant_remaining = self.current_mass_kg - self.specs.dry_mass_kg
        if propellant_consumed >= propellant_remaining:
            # Partial burn - use remaining fuel
            propellant_consumed = propellant_remaining
            actual_burn_time = propellant_remaining / mass_flow_rate
            self.fuel_exhausted = True
            self.guidance_mode = GuidanceMode.COAST
        else:
            actual_burn_time = dt_seconds

        # Calculate delta-v for this burn
        initial_mass = self.current_mass_kg
        final_mass = initial_mass - propellant_consumed
        delta_v_ms = exhaust_velocity_ms * math.log(initial_mass / final_mass)
        delta_v_kps = delta_v_ms / 1000

        # Update state
        self.current_mass_kg = final_mass
        self.remaining_delta_v_kps -= delta_v_kps
        self.remaining_delta_v_kps = max(0.0, self.remaining_delta_v_kps)

        # Apply velocity change
        self.velocity = self.velocity + direction * delta_v_ms

    def apply_lateral_thrust(
        self,
        lateral_direction: Vector3D,
        dt_seconds: float,
        throttle: float = 1.0
    ) -> None:
        """
        Apply lateral RCS thrust for course corrections without changing main engine direction.

        This uses dedicated lateral thrusters (RCS) that are 5% of main thrust,
        allowing precise corrections perpendicular to the current velocity vector.

        Args:
            lateral_direction: Direction to thrust (will be normalized)
            dt_seconds: Burn duration (seconds)
            throttle: Throttle setting 0.0 to 1.0
        """
        if self.fuel_exhausted or throttle <= 0:
            return

        lateral_direction = lateral_direction.normalized()
        if lateral_direction.magnitude < 0.01:
            return

        throttle = max(0.0, min(1.0, throttle))

        # RCS uses same propellant as main engine, but lower thrust
        rcs_thrust_n = self.specs.rcs_thrust_n * throttle
        exhaust_velocity_ms = self.specs.exhaust_velocity_kps * 1000
        mass_flow_rate = rcs_thrust_n / exhaust_velocity_ms  # kg/s

        # Propellant consumed this step
        propellant_consumed = mass_flow_rate * dt_seconds

        # Check available propellant
        propellant_remaining = self.current_mass_kg - self.specs.dry_mass_kg
        if propellant_consumed >= propellant_remaining:
            # Partial burn - use remaining fuel
            propellant_consumed = propellant_remaining
            self.fuel_exhausted = True
            self.guidance_mode = GuidanceMode.COAST

        # Calculate delta-v for this burn
        initial_mass = self.current_mass_kg
        final_mass = initial_mass - propellant_consumed

        if final_mass > 0 and initial_mass > final_mass:
            delta_v_ms = exhaust_velocity_ms * math.log(initial_mass / final_mass)
            delta_v_kps = delta_v_ms / 1000

            # Update state
            self.current_mass_kg = final_mass
            self.remaining_delta_v_kps -= delta_v_kps
            self.remaining_delta_v_kps = max(0.0, self.remaining_delta_v_kps)

            # Apply velocity change in the lateral direction
            self.velocity = self.velocity + lateral_direction * delta_v_ms

    def can_intercept(
        self,
        target_pos: Vector3D,
        target_vel: Vector3D,
        target_accel: Optional[Vector3D] = None
    ) -> Tuple[bool, float, float]:
        """
        Determine if torpedo can intercept target with remaining delta-v.

        Args:
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            target_accel: Target acceleration (m/s^2)

        Returns:
            Tuple of (can_intercept, time_to_intercept, fuel_remaining_at_intercept_kps)
        """
        intercept_pos, t_intercept = self.calculate_intercept(
            target_pos, target_vel, target_accel
        )

        # Calculate required delta-v
        rel_pos = intercept_pos - self.position
        distance = rel_pos.magnitude

        # Velocity needed to reach intercept
        rel_vel = target_vel - self.velocity
        required_velocity_change = rel_pos / t_intercept if t_intercept > 0 else Vector3D.zero()
        delta_v_needed_ms = (required_velocity_change - self.velocity).magnitude

        # Account for target's velocity at intercept
        # We need to match the target's position at intercept time
        delta_v_needed_kps = delta_v_needed_ms / 1000

        # Check if we have enough delta-v
        if delta_v_needed_kps > self.remaining_delta_v_kps:
            return False, t_intercept, 0.0

        fuel_remaining = self.remaining_delta_v_kps - delta_v_needed_kps
        return True, t_intercept, fuel_remaining


# =============================================================================
# TORPEDO LAUNCHER CLASS
# =============================================================================

@dataclass
class TorpedoLauncher:
    """
    Torpedo launch system for calculating optimal release parameters.

    Determines optimal release distance and timing so torpedo can execute
    continuous burn to intercept target efficiently.

    Attributes:
        specs: Torpedo specifications for launched torpedoes
        magazine_capacity: Maximum torpedoes in magazine
        current_magazine: Current torpedoes available
        cooldown_seconds: Time between launches
        last_launch_time: Time of last launch (for cooldown tracking)
    """
    specs: TorpedoSpecs = field(default_factory=TorpedoSpecs)
    magazine_capacity: int = 16  # From fleet data
    current_magazine: int = 16
    cooldown_seconds: float = 30.0  # From fleet data
    last_launch_time: float = -30.0  # Ready to fire immediately

    def calculate_release_distance(
        self,
        shooter_vel: Vector3D,
        target_pos: Vector3D,
        target_vel: Vector3D,
        target_accel: Optional[Vector3D] = None
    ) -> float:
        """
        Calculate optimal release distance for torpedo launch.

        Determines distance at which to launch so torpedo can burn
        continuously to target, using most of its delta-v budget.

        Args:
            shooter_vel: Launching ship's velocity (m/s)
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            target_accel: Target acceleration (m/s^2), assumes constant

        Returns:
            Optimal release distance in meters
        """
        if target_accel is None:
            target_accel = Vector3D.zero()

        # Relative velocity
        rel_vel = target_vel - shooter_vel
        closing_rate_kps = rel_vel.magnitude / 1000

        # Torpedo delta-v budget
        torpedo_dv_kps = self.specs.total_delta_v_kps

        # The torpedo needs delta-v for:
        # 1. Nullifying relative velocity
        # 2. Building closing velocity
        # 3. Terminal maneuvering reserve (~10%)

        maneuvering_reserve_fraction = 0.10
        available_dv_kps = torpedo_dv_kps * (1 - maneuvering_reserve_fraction)

        # Estimate required velocity change to intercept
        # For continuous burn, most efficient launch is when torpedo
        # can use ~90% of delta-v to reach target

        # Burn time and average acceleration
        burn_time = self.specs.burn_time_seconds()
        avg_mass = (self.specs.mass_kg + self.specs.dry_mass_kg) / 2
        avg_accel = self.specs.thrust_n / avg_mass  # m/s^2

        # Distance covered during burn (assuming constant acceleration)
        # d = v0*t + 0.5*a*t^2
        initial_speed = shooter_vel.magnitude
        burn_distance = initial_speed * burn_time + 0.5 * avg_accel * burn_time * burn_time

        # Add target motion during burn
        target_travel = target_vel.magnitude * burn_time

        # Optimal release distance accounts for both torpedo travel
        # and target motion
        optimal_distance = burn_distance

        # Adjust for target acceleration
        if target_accel.magnitude > 0:
            # Account for target evading
            target_accel_distance = 0.5 * target_accel.magnitude * burn_time * burn_time
            optimal_distance -= target_accel_distance * 0.5  # Conservative

        # Minimum is safe arming distance, maximum is torpedo range
        nominal_range_m = 2_000_000  # 2000 km from fleet data
        optimal_distance = max(SAFE_ARMING_DISTANCE_M * 10, optimal_distance)
        optimal_distance = min(nominal_range_m * 0.8, optimal_distance)

        return optimal_distance

    def can_launch(self, current_time: float) -> bool:
        """
        Check if launcher can fire (magazine and cooldown).

        Args:
            current_time: Current simulation time (seconds)

        Returns:
            True if ready to launch
        """
        if self.current_magazine <= 0:
            return False
        if current_time - self.last_launch_time < self.cooldown_seconds:
            return False
        return True

    def launch(
        self,
        shooter_position: Vector3D,
        shooter_velocity: Vector3D,
        target_id: str,
        target_position: Vector3D,
        target_velocity: Vector3D,
        current_time: float
    ) -> Optional[Torpedo]:
        """
        Launch a torpedo at specified target.

        Creates torpedo with initial velocity INHERITED from the launching ship.
        This is the critical velocity inheritance step - the torpedo starts
        with the shooter's velocity and will add to it via thrust.

        Velocity Inheritance:
            The torpedo's initial velocity equals the shooter's velocity.
            All subsequent thrust adds to this inherited velocity.

        Args:
            shooter_position: Launching ship position (meters)
            shooter_velocity: Launching ship velocity (m/s) - INHERITED by torpedo
            target_id: Target identifier
            target_position: Target position (meters)
            target_velocity: Target velocity (m/s)
            current_time: Current simulation time (seconds)

        Returns:
            Launched Torpedo or None if cannot launch
        """
        if not self.can_launch(current_time):
            return None

        # Create torpedo with shooter's velocity - VELOCITY INHERITANCE
        # The torpedo inherits the shooter's velocity at launch
        torpedo = Torpedo(
            specs=self.specs,
            position=Vector3D(
                shooter_position.x,
                shooter_position.y,
                shooter_position.z
            ),
            velocity=Vector3D(
                shooter_velocity.x,
                shooter_velocity.y,
                shooter_velocity.z
            ),
            target_id=target_id,
            guidance_mode=GuidanceMode.COLLISION,  # Use collision course guidance
            launched_from_velocity=Vector3D(
                shooter_velocity.x,
                shooter_velocity.y,
                shooter_velocity.z
            )
        )

        # SMART guidance handles all phases internally:
        # - Long range (>50km): Coast when on good intercept, burn to correct
        # - Medium range (10-50km): Active proportional navigation
        # - Terminal (<10km): Aggressive pursuit with full throttle

        # Update launcher state
        self.current_magazine -= 1
        self.last_launch_time = current_time

        return torpedo


# =============================================================================
# TORPEDO GUIDANCE CLASS
# =============================================================================

@dataclass
class TorpedoGuidance:
    """
    Guidance algorithms for torpedo homing.

    Provides multiple guidance modes:
    - PURSUIT: Direct pursuit, points at target
    - PROPORTIONAL_NAV: Lead pursuit for efficiency
    - TERMINAL: Final approach optimization
    - COAST: No guidance (fuel exhausted)

    Attributes:
        nav_constant: Proportional navigation constant (N')
    """
    nav_constant: float = PROPORTIONAL_NAV_CONSTANT

    def update_guidance(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D,
        dt: float
    ) -> Vector3D:
        """
        Calculate thrust direction based on current guidance mode.

        NOTE: this is a lossy view of the guidance solution - it discards the
        throttle setting and the RCS channel. Callers that actually apply thrust
        should use :meth:`update_guidance_command` instead.

        Args:
            torpedo: Torpedo to guide
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            dt: Time step (seconds)

        Returns:
            Thrust direction vector (normalized)
        """
        command = self.update_guidance_command(torpedo, target_pos, target_vel, dt)
        if command.throttle < 0.01:
            return Vector3D.zero()
        return command.direction

    def update_guidance_command(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D,
        dt: float
    ) -> GuidanceCommand:
        """
        Calculate the full guidance command for the current guidance mode.

        Unlike :meth:`update_guidance` this preserves the throttle setting and
        the RCS channel, so an RCS-only precision trim is not executed as a
        full-throttle main-engine burn.

        Guidance phases:
        1. INTERCEPT (cruise): Closure-priority, burn toward predicted intercept
        2. PROPORTIONAL_NAV (terminal approach): Precise LOS tracking at <50km
        3. TERMINAL (final): Pure pursuit at <10km

        Args:
            torpedo: Torpedo to guide
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            dt: Time step (seconds)

        Returns:
            GuidanceCommand with direction, throttle and (optional) RCS channel
        """
        # Check for mode transitions
        distance = torpedo.position.distance_to(target_pos)

        # Terminal range thresholds
        TERMINAL_RANGE_M = TERMINAL_APPROACH_DISTANCE_M  # 10 km
        PROPORTIONAL_NAV_RANGE_M = 50_000.0  # 50 km - switch to precise tracking

        if torpedo.guidance_mode != GuidanceMode.COAST:
            # Final approach: pure pursuit for impact
            if distance <= TERMINAL_RANGE_M:
                torpedo.guidance_mode = GuidanceMode.TERMINAL
            # Terminal phase: precise proportional nav
            elif distance <= PROPORTIONAL_NAV_RANGE_M:
                if torpedo.guidance_mode == GuidanceMode.INTERCEPT:
                    torpedo.guidance_mode = GuidanceMode.PROPORTIONAL_NAV

        # Execute guidance based on mode
        if torpedo.guidance_mode == GuidanceMode.SMART:
            return self._smart_guidance(torpedo, target_pos, target_vel, 1.0, dt)
        if torpedo.guidance_mode == GuidanceMode.COLLISION:
            # Collision course guidance: align relative velocity with LOS
            return self._collision_course_guidance(torpedo, target_pos, target_vel, dt)

        # Direction-only modes: full main-engine burn along the returned vector
        if torpedo.guidance_mode == GuidanceMode.PURSUIT:
            direction = self._pursuit_guidance(torpedo, target_pos, target_vel)
        elif torpedo.guidance_mode == GuidanceMode.INTERCEPT:
            direction = self._intercept_guidance(torpedo, target_pos, target_vel, dt)
        elif torpedo.guidance_mode == GuidanceMode.PROPORTIONAL_NAV:
            direction = self._proportional_nav_guidance(
                torpedo, target_pos, target_vel, dt
            )
        elif torpedo.guidance_mode == GuidanceMode.TERMINAL:
            direction = self._terminal_guidance(torpedo, target_pos, target_vel)
        else:  # COAST
            return GuidanceCommand.coast("coast mode")

        if direction.magnitude < 0.01:
            return GuidanceCommand.coast(f"{torpedo.guidance_mode.name}: no command")
        return GuidanceCommand.burn(
            direction, throttle=1.0, reason=torpedo.guidance_mode.name
        )

    def _pursuit_guidance(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D
    ) -> Vector3D:
        """
        Pure pursuit guidance: point directly at target.

        Simple but inefficient - always chases target's current position.

        Args:
            torpedo: Torpedo being guided
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)

        Returns:
            Thrust direction pointing at target
        """
        direction = target_pos - torpedo.position
        return direction.normalized()

    def _intercept_guidance(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D,
        dt: float
    ) -> Vector3D:
        """
        Intercept guidance: closure-priority cruise mode with velocity management.

        Prioritizes closing distance over exact LOS corrections.
        Calculates predicted intercept point and burns toward it with
        small bias corrections for target motion changes.

        This mode is fuel-efficient for long-range engagements and
        tolerates target evasive maneuvers by not over-correcting.
        Switch to PROPORTIONAL_NAV for terminal guidance when close.

        **Velocity Management**:
        When closing very fast at short range, the torpedo will overshoot
        if it doesn't brake. This guidance mode includes braking logic:
        - If closing_rate > 5 km/s AND distance < 100 km: start braking
        - Goal: arrive at terminal range with manageable relative velocity

        Strategy:
        1. Calculate intercept point based on target velocity
        2. If closing too fast at short range: BRAKE (retrograde thrust)
        3. Otherwise: burn toward intercept with small corrections

        Args:
            torpedo: Torpedo being guided
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            dt: Time step for calculations (seconds)

        Returns:
            Thrust direction for intercept approach
        """
        # Current line of sight to target
        los_to_target = target_pos - torpedo.position
        distance_to_target = los_to_target.magnitude
        if distance_to_target < 1.0:
            return Vector3D.zero()
        los_unit = los_to_target.normalized()

        # Relative velocity (target relative to torpedo)
        rel_vel = target_vel - torpedo.velocity

        # Closing rate (positive = torpedo approaching target)
        closing_velocity = -rel_vel.dot(los_unit)
        closing_velocity_kps = closing_velocity / 1000.0
        distance_km = distance_to_target / 1000.0

        # =====================================================================
        # VELOCITY MANAGEMENT: Brake if closing too fast at short range
        # =====================================================================
        # Calculate time to reach target at current closing rate
        if closing_velocity > 0:
            time_to_target = distance_to_target / closing_velocity
        else:
            time_to_target = float('inf')

        # Determine ideal closing velocity based on distance
        # We want to arrive at terminal range (50km) with ~3 km/s relative velocity
        # This gives proportional nav time to track and correct
        IDEAL_TERMINAL_VELOCITY_KPS = 3.0
        BRAKING_DISTANCE_KM = 150.0  # Start braking considerations at 150km

        if distance_km < BRAKING_DISTANCE_KM and closing_velocity_kps > IDEAL_TERMINAL_VELOCITY_KPS:
            # Calculate what velocity we need to arrive at target with ideal speed
            # Using v^2 = v0^2 + 2*a*d (need to decelerate)
            # We want final velocity of IDEAL_TERMINAL_VELOCITY_KPS at terminal range (50km)

            distance_to_terminal_km = max(0.1, distance_km - 50.0)

            # If we're going much faster than ideal, apply braking
            excess_velocity_kps = closing_velocity_kps - IDEAL_TERMINAL_VELOCITY_KPS

            # Braking must be computed against the velocity RELATIVE TO THE TARGET,
            # not the absolute inertial velocity. Using torpedo.velocity makes the
            # command depend on the observer's frame: the same engagement geometry
            # viewed from a boosted frame produces a different thrust direction.
            relative_velocity = torpedo.velocity - target_vel

            if excess_velocity_kps > 2.0 and distance_km < 100:
                # BRAKE: burn retrograde (against our closing velocity)
                torpedo_velocity_dir = relative_velocity.normalized()
                if torpedo_velocity_dir.magnitude > 0.01:
                    # Burn against our relative velocity to slow down
                    return (torpedo_velocity_dir * -1.0).normalized()

            # Moderate closure - blend braking with intercept
            if excess_velocity_kps > 0.5:
                torpedo_velocity_dir = relative_velocity.normalized()
                if torpedo_velocity_dir.magnitude > 0.01:
                    # Partial braking: blend retrograde with intercept direction
                    retrograde = torpedo_velocity_dir * -1.0
                    intercept_pos, _ = torpedo.calculate_intercept(target_pos, target_vel, None)
                    intercept_dir = (intercept_pos - torpedo.position).normalized()

                    # More braking when closer and faster
                    brake_factor = min(0.7, excess_velocity_kps / 10.0)
                    guidance_direction = retrograde * brake_factor + intercept_dir * (1.0 - brake_factor)
                    return guidance_direction.normalized()

        # =====================================================================
        # NORMAL INTERCEPT: Close distance efficiently
        # =====================================================================

        # Calculate intercept point - where target will be when we arrive
        intercept_pos, t_intercept = torpedo.calculate_intercept(
            target_pos, target_vel, None
        )

        # Primary direction: toward intercept point
        to_intercept = intercept_pos - torpedo.position
        distance_to_intercept = to_intercept.magnitude
        if distance_to_intercept < 1.0:
            return los_unit

        intercept_direction = to_intercept.normalized()

        # If we're not closing at all, pure burn toward intercept
        if closing_velocity <= 0:
            return intercept_direction

        # Calculate small correction for LOS rotation
        # This is much less aggressive than full proportional nav
        los_cross_vel = los_to_target.cross(rel_vel)
        omega_los_magnitude = los_cross_vel.magnitude / (distance_to_target * distance_to_target)

        # Only apply correction if LOS is rotating significantly
        # This prevents over-reacting to small jinks
        if omega_los_magnitude > 0.0001:  # Threshold for significant rotation
            omega_los = los_cross_vel / (distance_to_target * distance_to_target)
            # Reduced nav constant (0.5 vs 3.0) - we're not trying to track precisely yet
            # PN law is a = N * Vc * (omega x u_los). The operands must be in THIS
            # order: u_los x omega is the negation, which steers away from the lead point.
            correction = omega_los.cross(los_unit) * 0.5 * closing_velocity

            # Blend: 90% intercept direction, 10% correction (capped)
            correction_magnitude = min(correction.magnitude, 0.2)  # Cap correction influence
            if correction.magnitude > 0.01:
                correction = correction.normalized() * correction_magnitude
                guidance_direction = intercept_direction * 0.9 + correction
            else:
                guidance_direction = intercept_direction
        else:
            guidance_direction = intercept_direction

        return guidance_direction.normalized()

    def _proportional_nav_guidance(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D,
        dt: float
    ) -> Vector3D:
        """
        Proportional navigation guidance: lead pursuit.

        More efficient than pure pursuit. The torpedo acceleration is
        proportional to the line-of-sight rotation rate.

        a_commanded = N' * V_c * omega_los

        Where:
        - N' is navigation constant (typically 3-5)
        - V_c is closing velocity
        - omega_los is line-of-sight rotation rate

        Args:
            torpedo: Torpedo being guided
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            dt: Time step for LOS rate calculation (seconds)

        Returns:
            Thrust direction for proportional navigation
        """
        # Line of sight vector
        los = target_pos - torpedo.position
        distance = los.magnitude
        if distance < 1.0:
            return Vector3D.zero()

        los_unit = los.normalized()

        # Relative velocity
        rel_vel = target_vel - torpedo.velocity

        # Closing velocity (negative of range rate)
        closing_velocity = -rel_vel.dot(los_unit)

        # Line-of-sight rotation rate
        # omega_los = (los x rel_vel) / |los|^2
        los_cross_vel = los.cross(rel_vel)
        omega_los = los_cross_vel / (distance * distance)

        # Commanded acceleration direction
        # For 3D, acceleration perpendicular to LOS
        # a = N * Vc * (omega_los x u_los). Reversing the cross-product operands
        # flips the sign of the command and steers away from the lead point.
        accel_perpendicular = omega_los.cross(los_unit) * self.nav_constant * closing_velocity

        # Add component toward intercept point
        if closing_velocity > 0:
            # Bias toward target
            intercept_bias = los_unit * 0.3
            guidance_direction = accel_perpendicular + intercept_bias
        else:
            # Not closing - accelerate toward target
            guidance_direction = los_unit

        if guidance_direction.magnitude < 0.01:
            return los_unit

        return guidance_direction.normalized()

    def _terminal_guidance(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D
    ) -> Vector3D:
        """
        Terminal guidance: maximize closing velocity for impact.

        In final approach, prioritize achieving high closing speed
        for maximum kinetic damage contribution.

        Args:
            torpedo: Torpedo being guided
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)

        Returns:
            Thrust direction for terminal approach
        """
        # Vector to target
        los = target_pos - torpedo.position
        distance = los.magnitude
        if distance < 1.0:
            return Vector3D.zero()

        los_unit = los.normalized()

        # Relative velocity
        rel_vel = target_vel - torpedo.velocity

        # Current closing rate
        closing_rate = -rel_vel.dot(los_unit)

        # Time to impact estimate
        if closing_rate > 0:
            time_to_impact = distance / closing_rate
        else:
            time_to_impact = distance / 1000  # Fallback

        # In terminal phase, we want to:
        # 1. Minimize miss distance (pure pursuit component)
        # 2. Maximize closing velocity (thrust toward target)

        # For short time to impact, pure pursuit is best
        if time_to_impact < 2.0:
            return los_unit

        # Otherwise, lead slightly to intercept
        # Predict target position
        predicted_target = target_pos + target_vel * time_to_impact * 0.5
        lead_direction = (predicted_target - torpedo.position).normalized()

        # Blend between pure pursuit and lead
        blend_factor = min(1.0, time_to_impact / 10.0)
        guidance = los_unit * (1 - blend_factor * 0.3) + lead_direction * (blend_factor * 0.3)

        return guidance.normalized()

    def _smart_guidance(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D,
        target_accel_g: float,
        dt: float
    ) -> GuidanceCommand:
        """
        Smart guidance: aggressive proportional navigation with intercept prediction.

        Philosophy:
        1. ALWAYS pursue intercept - never let target escape
        2. Use proportional navigation throughout - proven missile guidance
        3. Throttle based on geometry, not arbitrary fuel reserve

        The key insight: a torpedo with higher acceleration than the target
        WILL hit if it uses proper proportional navigation. The nav_constant
        of 4 provides sufficient authority to overcome target evasion.

        Args:
            torpedo: Torpedo being guided
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            target_accel_g: Target max acceleration in g's (for evasion budget)
            dt: Time step (seconds)

        Returns:
            GuidanceCommand with direction and throttle
        """
        # =====================================================================
        # CALCULATE ENGAGEMENT GEOMETRY
        # =====================================================================
        los = target_pos - torpedo.position
        distance = los.magnitude
        if distance < 1.0:
            return GuidanceCommand.coast("at target")

        distance_km = distance / 1000.0
        los_unit = los.normalized()

        # Relative velocity (target - torpedo)
        rel_vel = target_vel - torpedo.velocity

        # Closing rate (positive = closing)
        closing_rate = -rel_vel.dot(los_unit)
        closing_rate_kps = closing_rate / 1000.0

        # Lateral velocity (perpendicular to LOS) - indicates target maneuvering
        lateral_vel = rel_vel + los_unit * closing_rate
        lateral_speed = lateral_vel.magnitude

        # Torpedo state
        dv_remaining_kps = torpedo.remaining_delta_v_kps
        torp_speed = torpedo.velocity.magnitude
        torp_accel = torpedo.specs.acceleration_at_mass(torpedo.current_mass_kg)
        torp_accel_g = torp_accel / 9.81

        # =====================================================================
        # CALCULATE LINE-OF-SIGHT ROTATION RATE
        # =====================================================================
        # This is the key to proportional navigation
        # omega_los = (LOS x relative_velocity) / distance^2
        los_cross_vel = los.cross(rel_vel)
        if distance > 0:
            omega_los = los_cross_vel / (distance * distance)
        else:
            omega_los = Vector3D.zero()

        # LOS rotation rate magnitude (rad/s)
        omega_magnitude = omega_los.magnitude

        # =====================================================================
        # PROPORTIONAL NAVIGATION COMMAND
        # =====================================================================
        # a_commanded = N * V_closing * omega_los
        # where N is the navigation constant (typically 3-5)
        # Higher N = more aggressive tracking

        # For evading targets, use higher nav constant
        effective_nav_constant = self.nav_constant
        if target_accel_g > 1.0:
            # Target can evade significantly - be more aggressive
            effective_nav_constant = max(self.nav_constant, 5.0)

        # Commanded acceleration perpendicular to LOS
        if closing_rate > 0:
            # Closing - use classic pro-nav
            accel_cmd_perpendicular = omega_los.cross(los_unit) * effective_nav_constant * abs(closing_rate)
        else:
            # Separating - need to close first.
            # Scale by the RELATIVE speed, not the absolute inertial speed, so the
            # command is Galilean-invariant (identical geometry -> identical command).
            accel_cmd_perpendicular = (
                omega_los.cross(los_unit) * effective_nav_constant * rel_vel.magnitude
            )

        # Add bias toward target (pursuit component)
        # This ensures we're always trying to close distance
        pursuit_bias = los_unit

        # =====================================================================
        # DETERMINE THROTTLE BASED ON SITUATION
        # =====================================================================

        # Check if we're on a good intercept course
        # Zero lateral velocity = collision course
        miss_distance_rate = lateral_speed  # How fast we'd miss
        time_to_intercept = distance / max(1.0, abs(closing_rate))

        # Estimated miss distance if we coast
        estimated_miss = miss_distance_rate * time_to_intercept

        if closing_rate <= 0:
            # NOT CLOSING - this is critical, must burn toward target
            guidance_direction = los_unit
            throttle = 1.0
            return GuidanceCommand.burn(
                guidance_direction,
                throttle=throttle,
                reason=f"pursuit: not closing ({closing_rate_kps:.1f} km/s)"
            )

        if estimated_miss > distance * 0.5:
            # Large miss predicted - aggressive correction needed
            guidance_direction = accel_cmd_perpendicular.normalized() + pursuit_bias * 0.5
            throttle = 1.0
            return GuidanceCommand.burn(
                guidance_direction,
                throttle=throttle,
                reason=f"correction: miss={estimated_miss/1000:.1f}km"
            )

        # PD engagement range is typically 100km - need high speed to survive
        # A torpedo at 20 km/s closing crosses 100km PD zone in 5 seconds
        # PD cooldown is 0.5s, so PD gets ~10 shots - need to be VERY FAST
        # Strategy: burn aggressively from the start, coast only when truly fast
        MIN_TERMINAL_SPEED_KPS = 20.0  # Need 20+ km/s closing to survive PD

        if distance_km > 300.0:
            # Far out - burn toward intercept, build speed early
            if closing_rate_kps < MIN_TERMINAL_SPEED_KPS:
                guidance_direction = pursuit_bias
                throttle = 1.0
            else:
                # Already fast - can ease off slightly
                if omega_magnitude > 0.001:
                    guidance_direction = accel_cmd_perpendicular.normalized() + pursuit_bias * 0.3
                    throttle = 0.7
                else:
                    return GuidanceCommand.coast(f"cruise: fast ({closing_rate_kps:.1f}km/s)")

        elif distance_km > 100.0:
            # Approaching PD range - MUST be at terminal speed
            if closing_rate_kps < MIN_TERMINAL_SPEED_KPS:
                # Not fast enough - FULL BURN
                guidance_direction = pursuit_bias
                throttle = 1.0
            else:
                # Fast enough - track with full power
                guidance_direction = accel_cmd_perpendicular + pursuit_bias * 0.7
                if guidance_direction.magnitude < 0.01:
                    guidance_direction = pursuit_bias
                throttle = 1.0

        else:
            # IN PD RANGE (<100km) - maximum aggression
            # Every fraction of a second counts
            guidance_direction = pursuit_bias + accel_cmd_perpendicular * 0.3
            if guidance_direction.magnitude < 0.01:
                guidance_direction = pursuit_bias
            throttle = 1.0

        if guidance_direction.magnitude < 0.01:
            guidance_direction = los_unit

        return GuidanceCommand.burn(
            guidance_direction,
            throttle=throttle,
            reason=f"proNav: {distance_km:.1f}km, ω={omega_magnitude:.4f}"
        )

    def _update_target_accel_estimate(
        self,
        torpedo: Torpedo,
        target_vel: Vector3D,
        dt: float
    ) -> Vector3D:
        """
        Update the torpedo's target-acceleration estimate from observed velocity.

        Finite-differences the target velocity across successive guidance calls
        and low-pass filters the result (time constant
        TARGET_ACCEL_FILTER_TAU_S) to suppress single-tick spikes such as
        impulsive maneuver switches. State lives on the torpedo because the
        simulation constructs a fresh TorpedoGuidance every tick.

        Assumes successive calls are ``dt`` seconds apart, which is how both
        the simulation loop and Torpedo.update invoke guidance.

        Args:
            torpedo: Torpedo whose estimator state to update.
            target_vel: Currently observed target velocity (m/s).
            dt: Time since the previous observation (seconds).

        Returns:
            Filtered target acceleration estimate (m/s^2).
        """
        prev = torpedo.last_observed_target_velocity
        torpedo.last_observed_target_velocity = Vector3D(
            target_vel.x, target_vel.y, target_vel.z
        )
        if prev is None or dt <= 0.0:
            return torpedo.estimated_target_accel

        raw_accel = (target_vel - prev) / dt
        alpha = dt / (TARGET_ACCEL_FILTER_TAU_S + dt)
        torpedo.estimated_target_accel = (
            torpedo.estimated_target_accel * (1.0 - alpha) + raw_accel * alpha
        )
        return torpedo.estimated_target_accel

    def nez_status(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D,
        target_max_lateral_accel_ms2: float = ASSUMED_TARGET_MAX_LATERAL_ACCEL_MS2
    ) -> NEZStatus:
        """
        Evaluate the no-escape zone: can the target still get away?

        Derivation (worst case for the torpedo, best case for the target):

        Let t_go = d / Vc be the time to impact at the current closing rate,
        a_t the target's maximum lateral acceleration, v_lat the current
        lateral (perpendicular-to-LOS) relative velocity, and a_m the
        torpedo's full-vector thrust acceleration.

        1. Displacement the target can still force. Holding max lateral
           acceleration in the direction that reinforces the existing drift,
           the target grows the ballistic miss to at most
               M_worst = |v_lat| * t_go + 0.5 * a_t * t_go^2.

        2. Acceleration condition. The kinematically minimal constant lateral
           acceleration that removes displacement M in time t_go is
           2*M/t_go^2 (from s = 0.5*a*t^2). Substituting M_worst:
               a_min = 2*|v_lat|/t_go + a_t.
           APN does not fly the minimal profile: its peak commanded lateral
           acceleration against this worst case is N*|v_lat|/t_go (the
           velocity term, = N*Vc*omega) plus (N/2)*a_t (the feed-forward
           term) - i.e. NEZ_GUIDANCE_MARGIN (= N/2) times a_min. If the
           engine cannot supply that, the law saturates and the guaranteed
           miss collapse no longer holds:
               required: a_m >= NEZ_GUIDANCE_MARGIN * a_min.

        3. Delta-v condition. Worst case the correction is held for the rest
           of the flight: the feed-forward share integrates to
           (N/2)*a_t*t_go, and nulling the existing lateral rate costs up to
           N/(N-2) * |v_lat| (standard PN heading-error result; equal to N/2
           at N = 4). With the shared margin:
               dv_req = NEZ_GUIDANCE_MARGIN * (|v_lat| + a_t * t_go)
               required: dv_remaining >= dv_req + NEZ_DV_RESERVE_KPS.

        4. A torpedo that is not closing, or whose fuel is exhausted, is
           never inside the NEZ.

        The target's capability defaults to the assumed fleet-wide worst case
        (3 g), NOT the currently observed acceleration - a commit decision
        must not be defeated by a target that simply waits before jinking.

        Args:
            torpedo: Torpedo to evaluate.
            target_pos: Target position (meters).
            target_vel: Target velocity (m/s).
            target_max_lateral_accel_ms2: Target's assumed max lateral
                acceleration (m/s^2). Defaults to 3 g.

        Returns:
            NEZStatus with the commit decision and both margins.
        """
        a_t = max(0.0, target_max_lateral_accel_ms2)

        to_target = target_pos - torpedo.position
        distance_m = to_target.magnitude
        if distance_m <= 0.0:
            return NEZStatus(
                inside=True, reason="at target", time_to_go_s=0.0,
                target_escape_displacement_m=0.0,
                required_lateral_accel_ms2=0.0,
                available_lateral_accel_ms2=0.0,
                accel_margin=float('inf'),
                required_delta_v_kps=NEZ_DV_RESERVE_KPS,
                available_delta_v_kps=torpedo.remaining_delta_v_kps,
                dv_margin_kps=torpedo.remaining_delta_v_kps - NEZ_DV_RESERVE_KPS,
            )

        los = to_target.normalized()
        rel_vel = torpedo.velocity - target_vel
        closing_speed_mps = rel_vel.dot(los)
        lateral_vel = rel_vel - los * closing_speed_mps
        lateral_speed_mps = lateral_vel.magnitude

        available_accel = torpedo.specs.acceleration_at_mass(torpedo.current_mass_kg)
        available_dv_kps = torpedo.remaining_delta_v_kps

        def _outside(reason: str, t_go: float, escape_m: float,
                     req_accel: float, req_dv: float) -> NEZStatus:
            accel_margin = (
                available_accel / req_accel if req_accel > 0 else float('inf')
            )
            return NEZStatus(
                inside=False, reason=reason, time_to_go_s=t_go,
                target_escape_displacement_m=escape_m,
                required_lateral_accel_ms2=req_accel,
                available_lateral_accel_ms2=available_accel,
                accel_margin=accel_margin,
                required_delta_v_kps=req_dv,
                available_delta_v_kps=available_dv_kps,
                dv_margin_kps=available_dv_kps - req_dv,
            )

        # Condition 4: a dead engine can never guarantee anything.
        if torpedo.fuel_exhausted or available_dv_kps <= 0.0:
            t_go = (
                distance_m / closing_speed_mps
                if closing_speed_mps > 0 else float('inf')
            )
            escape = (
                0.5 * a_t * t_go * t_go if t_go < float('inf') else float('inf')
            )
            return _outside("fuel exhausted", t_go, escape,
                            float('inf'), float('inf'))

        if closing_speed_mps <= 0.0:
            return _outside("not closing", float('inf'), float('inf'),
                            float('inf'), float('inf'))

        t_go = distance_m / closing_speed_mps
        escape_m = 0.5 * a_t * t_go * t_go

        # Condition 2: peak APN command must not saturate the engine.
        a_min = 2.0 * lateral_speed_mps / t_go + a_t
        required_accel = NEZ_GUIDANCE_MARGIN * a_min

        # Condition 3: worst-case correction budget plus terminal reserve.
        dv_req_kps = (
            NEZ_GUIDANCE_MARGIN * (lateral_speed_mps + a_t * t_go) / 1000.0
            + NEZ_DV_RESERVE_KPS
        )

        if available_accel < required_accel:
            return _outside("insufficient acceleration", t_go, escape_m,
                            required_accel, dv_req_kps)
        if available_dv_kps < dv_req_kps:
            return _outside("insufficient delta-v", t_go, escape_m,
                            required_accel, dv_req_kps)

        return NEZStatus(
            inside=True, reason="inside", time_to_go_s=t_go,
            target_escape_displacement_m=escape_m,
            required_lateral_accel_ms2=required_accel,
            available_lateral_accel_ms2=available_accel,
            accel_margin=(
                available_accel / required_accel
                if required_accel > 0 else float('inf')
            ),
            required_delta_v_kps=dv_req_kps,
            available_delta_v_kps=available_dv_kps,
            dv_margin_kps=available_dv_kps - dv_req_kps,
        )

    def _collision_course_guidance(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D,
        dt: float
    ) -> GuidanceCommand:
        """
        Augmented proportional navigation with a no-escape-zone commit logic.

        The previous law estimated the coast miss as lateral_speed *
        time_to_impact - a linear extrapolation that (a) ignores target
        acceleration entirely and (b) is self-suppressing: as the torpedo
        closes, time_to_impact collapses, so the estimate shrinks even while
        an accelerating target pumps lateral velocity in. Its correction
        phase then nulled lateral VELOCITY only, which always lags a target
        that keeps accelerating. Net effect: guidance declared itself on
        course ("miss~0.0km") right up to a kilometre-scale miss.

        This law works on the zero-effort miss (ZEM) instead:

            ZEM_perp = -v_lat * t_go + 0.5 * a_T_perp * t_go^2

        the predicted miss vector if the torpedo stops thrusting now and the
        target holds its (estimated) acceleration. The commanded lateral
        acceleration is the standard augmented-PN form

            a_cmd = N * ZEM_perp / t_go^2
                  = N * Vc * omega  +  (N/2) * a_T_perp

        (the two forms are algebraically identical: the velocity part of
        ZEM_perp/t_go^2 is v_lat/t_go = Vc*omega). Target acceleration is
        estimated by finite-differencing observed target velocity across
        guidance calls (see _update_target_accel_estimate).

        Throttle policy keys off the no-escape zone (see nez_status):
        - Inside the NEZ with delta-v to spare AND on course (predicted miss
          inside the deadband): burn the surplus into closing speed (impact
          energy scales as v^2 and a shorter run gives point defense fewer
          windows), steering with the APN command at the same time - the
          engine direction is a_cmd plus whatever axial component the
          remaining authority allows. Correct first, then spend.
        - Outside the NEZ: spend only what the APN law commands (throttle
          proportional to |a_cmd|), preserving delta-v so the torpedo stays
          able to reach the NEZ; closure is only forced while below the
          cruise floor MIN_CLOSING_SPEED_KPS.
        - The NEZ delta-v requirement always includes NEZ_DV_RESERVE_KPS, so
          a reserve for late corrections survives to impact and the torpedo
          is NOT unconditionally lethal: engaged outside its NEZ against an
          early, hard jink it runs out of budget and misses.

        Args:
            torpedo: Torpedo being guided
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            dt: Time step (seconds)

        Returns:
            GuidanceCommand with direction and throttle
        """
        # ---- Observe: update the target-acceleration estimate every call,
        # even when about to coast, so the filter never starves. -----------
        target_accel_est = self._update_target_accel_estimate(
            torpedo, target_vel, dt
        )

        # ---- Synchronize the observation. CombatSimulation updates ships
        # BEFORE torpedoes, so `target_pos` arrives already advanced to the
        # end of this tick while the torpedo has not been integrated yet.
        # Roll the target back one tick (exactly as the sim's own CPA check
        # does: target_pos_t0 = target.position - target.velocity * dt) so
        # both bodies are measured at the same instant. Without this the
        # guidance flies a perfect collision course to a phantom point
        # displaced by (torpedo lateral velocity * dt) - ~800 m against a
        # hard-evading target - while reporting a near-zero predicted miss.
        target_pos = target_pos - target_vel * dt

        to_target = target_pos - torpedo.position
        distance_m = to_target.magnitude
        if distance_m < 100.0:  # Within 100 m - essentially at target
            return GuidanceCommand.coast("at target")

        los = to_target.normalized()
        rel_vel = torpedo.velocity - target_vel
        closing_speed_mps = rel_vel.dot(los)
        closing_speed_kps = closing_speed_mps / 1000.0
        lateral_vel = rel_vel - los * closing_speed_mps

        if torpedo.fuel_exhausted:
            return GuidanceCommand.coast("fuel exhausted")

        accel_max = torpedo.specs.acceleration_at_mass(torpedo.current_mass_kg)
        if accel_max <= 0.0:
            return GuidanceCommand.coast("no thrust available")

        # ---- Not closing: build closure while killing drift with the same
        # velocity-error law as before (APN needs a finite t_go to operate).
        if closing_speed_mps <= 0.0:
            velocity_error = (
                los * (MIN_CLOSING_SPEED_KPS * 1000.0 - closing_speed_mps)
                - lateral_vel * APN_NAV_RATIO
            )
            direction = (
                velocity_error.normalized()
                if velocity_error.magnitude > 1.0 else los
            )
            return GuidanceCommand.burn(
                direction, throttle=1.0,
                reason=f"PURSUIT: closing={closing_speed_kps:.1f}km/s"
            )

        # Time-to-go, floored at one guidance tick: inside the final tick the
        # 1/t_go^2 gain would otherwise blow up on a miss the torpedo can no
        # longer influence anyway.
        t_go = max(distance_m / closing_speed_mps, max(dt, 1e-3))

        # ---- Augmented PN command (ZEM form). The perpendicular part of the
        # target's relative velocity is -lateral_vel (lateral_vel is the
        # torpedo's drift across the LOS).
        accel_perp = target_accel_est - los * target_accel_est.dot(los)
        zem_perp = lateral_vel * (-t_go) + accel_perp * (0.5 * t_go * t_go)
        predicted_miss_m = zem_perp.magnitude
        a_cmd = zem_perp * (APN_NAV_RATIO / (t_go * t_go))
        a_cmd_mag = a_cmd.magnitude

        # ---- Commit logic: the NEZ is evaluated against the ASSUMED
        # worst-case target (3 g), never the observed estimate. The terminal
        # closure burn additionally requires being on course (predicted miss
        # inside the deadband): correct first, then spend - investing in
        # closing speed while a real miss is outstanding shortens the very
        # window the correction needs.
        nez = self.nez_status(torpedo, target_pos, target_vel)
        surplus_kps = nez.dv_margin_kps  # already net of reserve
        terminal_burn = (
            nez.inside
            and surplus_kps > TERMINAL_MIN_BURN_KPS
            and predicted_miss_m < APN_MISS_DEADBAND_M
        )
        below_cruise_floor = closing_speed_kps < MIN_CLOSING_SPEED_KPS
        want_closure = terminal_burn or below_cruise_floor

        # ---- Saturated: every m/s^2 goes to the lateral correction.
        if a_cmd_mag >= accel_max:
            return GuidanceCommand.burn(
                a_cmd.normalized(), throttle=1.0,
                reason=(
                    f"APN_SAT: miss~{predicted_miss_m / 1000.0:.2f}km, "
                    f"cmd={a_cmd_mag:.0f}m/s2 > {accel_max:.0f}m/s2"
                )
            )

        # ---- Closure wanted (terminal burn inside NEZ, or cruise floor):
        # fill the authority left over by the lateral command with axial
        # thrust. |a_cmd|^2 + fill^2 = accel_max^2 keeps the lateral
        # component exactly what APN commanded.
        if want_closure:
            axial_fill = math.sqrt(accel_max * accel_max - a_cmd_mag * a_cmd_mag)
            desired_accel = a_cmd + los * axial_fill
            if terminal_burn:
                reason = (
                    f"TERMINAL(NEZ): {nez.time_to_go_s:.0f}s to impact, "
                    f"spending {surplus_kps:.1f} km/s surplus "
                    f"(reserve {nez.required_delta_v_kps:.2f}), "
                    f"miss~{predicted_miss_m:.0f}m"
                )
            else:
                reason = (
                    f"ACCEL: {closing_speed_kps:.1f} -> "
                    f"{MIN_CLOSING_SPEED_KPS:.0f}km/s, "
                    f"miss~{predicted_miss_m / 1000.0:.2f}km"
                )
            return GuidanceCommand.burn(desired_accel.normalized(),
                                        throttle=1.0, reason=reason)

        # ---- On course, no closure mandate: coast (delta-v preservation).
        if predicted_miss_m < APN_MISS_DEADBAND_M:
            return GuidanceCommand.coast(
                f"ON_COURSE: miss~{predicted_miss_m:.0f}m, "
                f"{t_go:.0f}s to impact ({nez.reason})"
            )

        # ---- Correction: spend exactly what the APN law commands and
        # nothing more (delta-v preservation - this is also the only burn
        # mode outside the NEZ). Route small commands through the lateral
        # RCS (5% of main thrust): same propellant, same commanded
        # acceleration, but the main engine stays cold for precision trims.
        rcs_accel = (
            torpedo.specs.rcs_thrust_n / torpedo.current_mass_kg
            if torpedo.current_mass_kg > 0 else 0.0
        )
        if 0.0 < a_cmd_mag <= rcs_accel:
            return GuidanceCommand.rcs_correction(
                a_cmd.normalized(), throttle=a_cmd_mag / rcs_accel,
                reason=(
                    f"RCS_TRIM: miss~{predicted_miss_m / 1000.0:.2f}km, "
                    f"cmd={a_cmd_mag:.1f}m/s2"
                )
            )
        throttle = a_cmd_mag / accel_max
        if throttle < 0.01:
            return GuidanceCommand.coast(
                f"ON_COURSE: miss~{predicted_miss_m:.0f}m (cmd negligible)"
            )
        return GuidanceCommand.burn(
            a_cmd.normalized(), throttle=throttle,
            reason=(
                f"APN: miss~{predicted_miss_m / 1000.0:.2f}km, "
                f"cmd={a_cmd_mag:.1f}m/s2, {nez.reason}"
            )
        )

    def update_collision_guidance(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D,
        dt: float
    ) -> GuidanceCommand:
        """
        Main entry point for collision course guidance.

        Args:
            torpedo: Torpedo to guide
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            dt: Time step (seconds)

        Returns:
            GuidanceCommand with thrust direction and throttle
        """
        return self._collision_course_guidance(torpedo, target_pos, target_vel, dt)

    def update_smart_guidance(
        self,
        torpedo: Torpedo,
        target_pos: Vector3D,
        target_vel: Vector3D,
        target_accel_g: float,
        dt: float
    ) -> GuidanceCommand:
        """
        Main entry point for smart guidance.

        Args:
            torpedo: Torpedo to guide
            target_pos: Target position (meters)
            target_vel: Target velocity (m/s)
            target_accel_g: Target's max acceleration in g's
            dt: Time step (seconds)

        Returns:
            GuidanceCommand with direction and throttle
        """
        if torpedo.fuel_exhausted:
            return GuidanceCommand.coast("fuel exhausted")

        return self._smart_guidance(torpedo, target_pos, target_vel, target_accel_g, dt)


# =============================================================================
# INTERCEPT ANALYSIS
# =============================================================================

def analyze_intercept(
    torpedo_specs: TorpedoSpecs,
    launch_pos: Vector3D,
    launch_vel: Vector3D,
    target_pos: Vector3D,
    target_vel: Vector3D,
    target_accel: Optional[Vector3D] = None
) -> dict:
    """
    Analyze torpedo intercept possibility.

    Given torpedo specifications and initial conditions, determine
    if intercept is possible and estimate engagement parameters.

    Args:
        torpedo_specs: Torpedo physical specifications
        launch_pos: Launch position (meters)
        launch_vel: Launch velocity (m/s)
        target_pos: Target position (meters)
        target_vel: Target velocity (m/s)
        target_accel: Target acceleration (m/s^2)

    Returns:
        Dict with intercept analysis results
    """
    if target_accel is None:
        target_accel = Vector3D.zero()

    # Create temporary torpedo for analysis
    torpedo = Torpedo(
        specs=torpedo_specs,
        position=Vector3D(launch_pos.x, launch_pos.y, launch_pos.z),
        velocity=Vector3D(launch_vel.x, launch_vel.y, launch_vel.z),
        target_id="analysis_target"
    )

    # Calculate intercept
    can_intercept, time_to_intercept, fuel_remaining = torpedo.can_intercept(
        target_pos, target_vel, target_accel
    )

    intercept_pos, _ = torpedo.calculate_intercept(
        target_pos, target_vel, target_accel
    )

    # Calculate engagement parameters
    distance = launch_pos.distance_to(target_pos)
    rel_vel = target_vel - launch_vel
    closing_rate = -rel_vel.dot((target_pos - launch_pos).normalized())

    return {
        'can_intercept': can_intercept,
        'time_to_intercept_seconds': time_to_intercept,
        'fuel_remaining_at_intercept_kps': fuel_remaining,
        'intercept_position': intercept_pos,
        'initial_distance_m': distance,
        'initial_distance_km': distance / 1000,
        'initial_closing_rate_mps': closing_rate,
        'initial_closing_rate_kps': closing_rate / 1000,
        'torpedo_delta_v_budget_kps': torpedo_specs.total_delta_v_kps,
        'torpedo_burn_time_s': torpedo_specs.burn_time_seconds()
    }


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS TORPEDO MODULE - SELF TEST")
    print("=" * 70)

    # Test TorpedoSpecs
    print("\n--- Torpedo Specifications ---")
    specs = TorpedoSpecs()
    print(f"Warhead yield: {specs.warhead_yield_gj} GJ")
    print(f"Total mass: {specs.mass_kg} kg")
    print(f"Dry mass: {specs.dry_mass_kg:.1f} kg")
    print(f"Propellant mass: {specs.propellant_mass_kg:.1f} kg")
    print(f"Propellant fraction: {specs.propellant_fraction * 100:.0f}%")
    print(f"Exhaust velocity: {specs.exhaust_velocity_kps} km/s")
    print(f"Total delta-v: {specs.total_delta_v_kps:.2f} km/s")
    print(f"Thrust: {specs.thrust_n / 1000:.1f} kN")
    print(f"Initial acceleration: {specs.acceleration_at_mass(specs.mass_kg):.1f} m/s^2 "
          f"({specs.acceleration_at_mass(specs.mass_kg) / 9.81:.1f} g)")
    print(f"Final acceleration: {specs.acceleration_at_mass(specs.dry_mass_kg):.1f} m/s^2 "
          f"({specs.acceleration_at_mass(specs.dry_mass_kg) / 9.81:.1f} g)")
    print(f"Total burn time: {specs.burn_time_seconds():.1f} s")

    # Test from fleet data
    print("\n--- Fleet Data Torpedo ---")
    fleet_specs = TorpedoSpecs.from_fleet_data(
        warhead_yield_gj=50,
        ammo_mass_kg=1600,
        range_km=2000
    )
    print(f"Delta-v budget: {fleet_specs.total_delta_v_kps:.2f} km/s")
    print(f"This matches design estimate: ~60 km/s")

    # Test Torpedo
    print("\n--- Torpedo State ---")
    torpedo = Torpedo(
        specs=specs,
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(10000, 0, 0),  # 10 km/s initial
        target_id="target_001"
    )
    print(f"Initial velocity: {torpedo.velocity.magnitude / 1000:.1f} km/s")
    print(f"Remaining delta-v: {torpedo.remaining_delta_v_kps:.2f} km/s")
    print(f"Armed: {torpedo.armed}")

    # Test intercept calculation
    print("\n--- Intercept Calculation ---")
    target_pos = Vector3D(500_000, 100_000, 0)  # 500 km away
    target_vel = Vector3D(-5000, 0, 0)  # 5 km/s toward us
    intercept_pos, t_intercept = torpedo.calculate_intercept(target_pos, target_vel)
    print(f"Target at: ({target_pos.x/1000:.0f}, {target_pos.y/1000:.0f}, {target_pos.z/1000:.0f}) km")
    print(f"Target velocity: {target_vel.magnitude/1000:.1f} km/s")
    print(f"Intercept point: ({intercept_pos.x/1000:.0f}, {intercept_pos.y/1000:.0f}, {intercept_pos.z/1000:.0f}) km")
    print(f"Time to intercept: {t_intercept:.1f} s")

    # Test can_intercept
    print("\n--- Intercept Feasibility ---")
    can_int, t_int, fuel_rem = torpedo.can_intercept(target_pos, target_vel)
    print(f"Can intercept: {can_int}")
    print(f"Time to intercept: {t_int:.1f} s")
    print(f"Fuel remaining at intercept: {fuel_rem:.2f} km/s")

    # Test thrust application
    print("\n--- Thrust Application ---")
    print(f"Before thrust:")
    print(f"  Velocity: {torpedo.velocity.magnitude / 1000:.2f} km/s")
    print(f"  Mass: {torpedo.current_mass_kg:.1f} kg")
    print(f"  Delta-v remaining: {torpedo.remaining_delta_v_kps:.2f} km/s")

    torpedo.apply_thrust(Vector3D(1, 0, 0), 10.0)  # 10 second burn

    print(f"After 10s burn:")
    print(f"  Velocity: {torpedo.velocity.magnitude / 1000:.2f} km/s")
    print(f"  Mass: {torpedo.current_mass_kg:.1f} kg")
    print(f"  Delta-v remaining: {torpedo.remaining_delta_v_kps:.2f} km/s")
    print(f"  Fuel exhausted: {torpedo.fuel_exhausted}")

    # Test launcher
    print("\n--- Torpedo Launcher ---")
    launcher = TorpedoLauncher(specs=specs)
    print(f"Magazine: {launcher.current_magazine}/{launcher.magazine_capacity}")
    print(f"Cooldown: {launcher.cooldown_seconds} s")

    release_dist = launcher.calculate_release_distance(
        shooter_vel=Vector3D(8000, 0, 0),
        target_pos=Vector3D(1_000_000, 0, 0),
        target_vel=Vector3D(-5000, 0, 0)
    )
    print(f"Optimal release distance: {release_dist / 1000:.0f} km")

    # Test launch
    launched = launcher.launch(
        shooter_position=Vector3D(0, 0, 0),
        shooter_velocity=Vector3D(8000, 0, 0),
        target_id="target_001",
        target_position=Vector3D(500_000, 0, 0),
        target_velocity=Vector3D(-5000, 0, 0),
        current_time=0.0
    )
    if launched:
        print(f"Torpedo launched!")
        print(f"  Initial velocity: {launched.velocity.magnitude / 1000:.1f} km/s")
        print(f"  Guidance mode: {launched.guidance_mode.name}")
        print(f"Magazine after launch: {launcher.current_magazine}/{launcher.magazine_capacity}")

    # Test guidance
    print("\n--- Guidance Modes ---")
    guidance = TorpedoGuidance()

    test_torpedo = Torpedo(
        specs=specs,
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(10000, 0, 0),
        target_id="test",
        guidance_mode=GuidanceMode.PURSUIT
    )

    # Pursuit guidance
    pursuit_dir = guidance._pursuit_guidance(
        test_torpedo,
        Vector3D(100_000, 20_000, 0),
        Vector3D(-5000, 0, 0)
    )
    print(f"Pursuit direction: ({pursuit_dir.x:.3f}, {pursuit_dir.y:.3f}, {pursuit_dir.z:.3f})")

    # Proportional nav
    test_torpedo.guidance_mode = GuidanceMode.PROPORTIONAL_NAV
    pn_dir = guidance._proportional_nav_guidance(
        test_torpedo,
        Vector3D(100_000, 20_000, 0),
        Vector3D(-5000, 0, 0),
        dt=1.0
    )
    print(f"Proportional nav direction: ({pn_dir.x:.3f}, {pn_dir.y:.3f}, {pn_dir.z:.3f})")

    # Terminal guidance
    test_torpedo.guidance_mode = GuidanceMode.TERMINAL
    test_torpedo.position = Vector3D(95_000, 18_000, 0)  # Close to target
    term_dir = guidance._terminal_guidance(
        test_torpedo,
        Vector3D(100_000, 20_000, 0),
        Vector3D(-5000, 0, 0)
    )
    print(f"Terminal guidance direction: ({term_dir.x:.3f}, {term_dir.y:.3f}, {term_dir.z:.3f})")

    # Test full intercept analysis
    print("\n--- Intercept Analysis ---")
    analysis = analyze_intercept(
        torpedo_specs=specs,
        launch_pos=Vector3D(0, 0, 0),
        launch_vel=Vector3D(8000, 0, 0),
        target_pos=Vector3D(500_000, 50_000, 0),
        target_vel=Vector3D(-5000, 1000, 0),
        target_accel=Vector3D(0, 50, 0)  # Target maneuvering
    )
    print(f"Can intercept: {analysis['can_intercept']}")
    print(f"Initial distance: {analysis['initial_distance_km']:.0f} km")
    print(f"Initial closing rate: {analysis['initial_closing_rate_kps']:.2f} km/s")
    print(f"Time to intercept: {analysis['time_to_intercept_seconds']:.1f} s")
    print(f"Fuel remaining: {analysis['fuel_remaining_at_intercept_kps']:.2f} km/s")

    # Test update loop
    print("\n--- Simulation Loop (10 seconds) ---")
    sim_torpedo = Torpedo(
        specs=specs,
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(8000, 0, 0),
        target_id="sim_target",
        guidance_mode=GuidanceMode.PROPORTIONAL_NAV
    )
    target_pos = Vector3D(100_000, 10_000, 0)
    target_vel = Vector3D(-3000, 500, 0)

    dt = 1.0
    for i in range(10):
        sim_torpedo.update(dt, target_pos, target_vel)
        # Move target
        target_pos = target_pos + target_vel * dt

        if i % 2 == 0:
            dist = sim_torpedo.position.distance_to(target_pos) / 1000
            vel = sim_torpedo.velocity.magnitude / 1000
            print(f"  t={i+1}s: dist={dist:.1f}km, vel={vel:.2f}km/s, "
                  f"dv_rem={sim_torpedo.remaining_delta_v_kps:.2f}km/s, "
                  f"armed={sim_torpedo.armed}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
