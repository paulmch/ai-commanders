#!/usr/bin/env python3
"""
Physics Simulation Module for AI Commanders Space Battle Simulator

Implements Newtonian mechanics for spacecraft:
- 3D vector operations
- Ship state with position, velocity, orientation, angular velocity
- Thrust application (F=ma with thrust vectoring)
- Trajectory propagation (Euler integration)
- Delta-v calculations (Tsiolkovsky rocket equation)
- Rotation dynamics (moment of inertia)

Based on Terra Invicta reference data:
- Exhaust velocity: 10,256 km/s
- Main thrust: 58.56 MN
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Standard gravity (m/s^2)
G_STANDARD = 9.81

# Fleet propulsion constants (from fleet_ships.json)
EXHAUST_VELOCITY_KPS = 10_256  # km/s
EXHAUST_VELOCITY_MS = EXHAUST_VELOCITY_KPS * 1000  # m/s
MAIN_THRUST_MN = 58.56  # Meganewtons
MAIN_THRUST_N = MAIN_THRUST_MN * 1e6  # Newtons

# Thrust vectoring limits
MAX_GIMBAL_ANGLE_DEG = 3.0  # Maximum nozzle deflection
COMBAT_GIMBAL_ANGLE_DEG = 1.0  # Typical combat deflection


# =============================================================================
# VECTOR3D CLASS
# =============================================================================

@dataclass
class Vector3D:
    """
    3D vector for positions, velocities, and directions in space.

    Uses a right-handed coordinate system where:
    - X: forward (ship nose direction)
    - Y: right (starboard)
    - Z: up (dorsal)

    All units in SI (meters, m/s, etc.) unless otherwise specified.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vector3D) -> Vector3D:
        """Vector addition."""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        """Vector subtraction."""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3D:
        """Scalar multiplication."""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3D:
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector3D:
        """Scalar division."""
        if scalar == 0:
            raise ValueError("Cannot divide vector by zero")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vector3D:
        """Negation."""
        return Vector3D(-self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        """Equality check with tolerance."""
        if not isinstance(other, Vector3D):
            return False
        eps = 1e-10
        return (abs(self.x - other.x) < eps and
                abs(self.y - other.y) < eps and
                abs(self.z - other.z) < eps)

    def dot(self, other: Vector3D) -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        """Cross product."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    @property
    def magnitude(self) -> float:
        """Vector magnitude (length)."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def magnitude_squared(self) -> float:
        """Squared magnitude (avoids sqrt for comparisons)."""
        return self.x**2 + self.y**2 + self.z**2

    def normalized(self) -> Vector3D:
        """Return unit vector in same direction."""
        mag = self.magnitude
        if mag == 0:
            return Vector3D(0, 0, 0)
        return self / mag

    def distance_to(self, other: Vector3D) -> float:
        """Distance to another point."""
        return (self - other).magnitude

    def angle_to(self, other: Vector3D) -> float:
        """Angle between vectors in radians."""
        dot = self.dot(other)
        mags = self.magnitude * other.magnitude
        if mags == 0:
            return 0.0
        # Clamp to avoid floating point errors with acos
        cos_angle = max(-1.0, min(1.0, dot / mags))
        return math.acos(cos_angle)

    def rotate_around_axis(self, axis: Vector3D, angle_rad: float) -> Vector3D:
        """Rotate vector around an axis using Rodrigues' rotation formula."""
        k = axis.normalized()
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # v_rot = v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
        return (self * cos_a +
                k.cross(self) * sin_a +
                k * k.dot(self) * (1 - cos_a))

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float]) -> Vector3D:
        """Create from tuple."""
        return cls(t[0], t[1], t[2])

    @classmethod
    def zero(cls) -> Vector3D:
        """Zero vector."""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def unit_x(cls) -> Vector3D:
        """Unit vector in X direction (forward)."""
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def unit_y(cls) -> Vector3D:
        """Unit vector in Y direction (right)."""
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def unit_z(cls) -> Vector3D:
        """Unit vector in Z direction (up)."""
        return cls(0.0, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"Vector3D({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"


# =============================================================================
# SHIP STATE CLASS
# =============================================================================

@dataclass
class ShipState:
    """
    Complete kinematic state of a spacecraft.

    Position and velocity are in world coordinates (meters, m/s).
    Orientation is represented as a forward direction vector.
    Angular velocity is in rad/s around each axis.

    Attributes:
        position: Ship position in world coordinates (meters)
        velocity: Ship velocity in world coordinates (m/s)
        forward: Unit vector pointing in ship's forward direction
        up: Unit vector pointing in ship's up direction
        angular_velocity: Angular velocity vector (rad/s)
        mass_kg: Current total mass including propellant (kg)
        dry_mass_kg: Mass without propellant (kg)
        propellant_kg: Current propellant mass (kg)
        thrust_n: Main engine thrust (Newtons)
        exhaust_velocity_ms: Engine exhaust velocity (m/s)
        moment_of_inertia_kg_m2: Moment of inertia for pitch/yaw (kg*m^2)
    """
    # Kinematic state
    position: Vector3D = field(default_factory=Vector3D.zero)
    velocity: Vector3D = field(default_factory=Vector3D.zero)
    forward: Vector3D = field(default_factory=Vector3D.unit_x)
    up: Vector3D = field(default_factory=Vector3D.unit_z)
    angular_velocity: Vector3D = field(default_factory=Vector3D.zero)

    # Mass properties
    mass_kg: float = 1_990_000  # Default: corvette wet mass (1990 tons)
    dry_mass_kg: float = 1_895_000  # Default: corvette dry mass
    propellant_kg: float = 95_000  # Default: corvette propellant

    # Propulsion properties
    thrust_n: float = MAIN_THRUST_N
    exhaust_velocity_ms: float = EXHAUST_VELOCITY_MS

    # Rotational properties
    moment_of_inertia_kg_m2: float = 700_645_833  # Default: corvette

    @property
    def right(self) -> Vector3D:
        """Unit vector pointing to ship's right (starboard)."""
        return self.forward.cross(self.up).normalized()

    @property
    def wet_mass_kg(self) -> float:
        """Total mass with propellant."""
        return self.dry_mass_kg + self.propellant_kg

    @property
    def mass_ratio(self) -> float:
        """Wet mass / dry mass ratio."""
        if self.dry_mass_kg <= 0:
            return 1.0
        return self.wet_mass_kg / self.dry_mass_kg

    def remaining_delta_v_ms(self) -> float:
        """
        Calculate remaining delta-v using Tsiolkovsky equation.

        Returns:
            Remaining delta-v in m/s
        """
        return tsiolkovsky_delta_v(
            exhaust_velocity_ms=self.exhaust_velocity_ms,
            wet_mass_kg=self.wet_mass_kg,
            dry_mass_kg=self.dry_mass_kg
        )

    def remaining_delta_v_kps(self) -> float:
        """Remaining delta-v in km/s."""
        return self.remaining_delta_v_ms() / 1000

    def max_acceleration_ms2(self) -> float:
        """Maximum acceleration at current mass (m/s^2)."""
        if self.mass_kg <= 0:
            return 0.0
        return self.thrust_n / self.mass_kg

    def max_acceleration_g(self) -> float:
        """Maximum acceleration at current mass (in g)."""
        return self.max_acceleration_ms2() / G_STANDARD

    def copy(self) -> ShipState:
        """Create a deep copy of the state."""
        return ShipState(
            position=Vector3D(self.position.x, self.position.y, self.position.z),
            velocity=Vector3D(self.velocity.x, self.velocity.y, self.velocity.z),
            forward=Vector3D(self.forward.x, self.forward.y, self.forward.z),
            up=Vector3D(self.up.x, self.up.y, self.up.z),
            angular_velocity=Vector3D(
                self.angular_velocity.x,
                self.angular_velocity.y,
                self.angular_velocity.z
            ),
            mass_kg=self.mass_kg,
            dry_mass_kg=self.dry_mass_kg,
            propellant_kg=self.propellant_kg,
            thrust_n=self.thrust_n,
            exhaust_velocity_ms=self.exhaust_velocity_ms,
            moment_of_inertia_kg_m2=self.moment_of_inertia_kg_m2
        )


# =============================================================================
# DELTA-V CALCULATIONS (TSIOLKOVSKY EQUATION)
# =============================================================================

def tsiolkovsky_delta_v(
    exhaust_velocity_ms: float,
    wet_mass_kg: float,
    dry_mass_kg: float
) -> float:
    """
    Calculate delta-v using the Tsiolkovsky rocket equation.

    delta_v = v_e * ln(m_wet / m_dry)

    Args:
        exhaust_velocity_ms: Engine exhaust velocity (m/s)
        wet_mass_kg: Initial mass with propellant (kg)
        dry_mass_kg: Final mass without propellant (kg)

    Returns:
        Delta-v in m/s
    """
    if dry_mass_kg <= 0 or wet_mass_kg < dry_mass_kg:
        return 0.0

    mass_ratio = wet_mass_kg / dry_mass_kg
    return exhaust_velocity_ms * math.log(mass_ratio)


def propellant_for_delta_v(
    delta_v_ms: float,
    exhaust_velocity_ms: float,
    dry_mass_kg: float
) -> float:
    """
    Calculate propellant needed for a given delta-v.

    From Tsiolkovsky: m_wet = m_dry * exp(delta_v / v_e)
    propellant = m_wet - m_dry

    Args:
        delta_v_ms: Desired delta-v (m/s)
        exhaust_velocity_ms: Engine exhaust velocity (m/s)
        dry_mass_kg: Ship dry mass (kg)

    Returns:
        Required propellant mass in kg
    """
    if exhaust_velocity_ms <= 0 or dry_mass_kg <= 0:
        return 0.0

    mass_ratio = math.exp(delta_v_ms / exhaust_velocity_ms)
    wet_mass_kg = dry_mass_kg * mass_ratio
    return wet_mass_kg - dry_mass_kg


def mass_after_burn(
    initial_mass_kg: float,
    delta_v_ms: float,
    exhaust_velocity_ms: float
) -> float:
    """
    Calculate mass remaining after a delta-v burn.

    m_final = m_initial / exp(delta_v / v_e)

    Args:
        initial_mass_kg: Mass before burn (kg)
        delta_v_ms: Delta-v of the burn (m/s)
        exhaust_velocity_ms: Engine exhaust velocity (m/s)

    Returns:
        Mass after burn in kg
    """
    if exhaust_velocity_ms <= 0:
        return initial_mass_kg

    return initial_mass_kg / math.exp(delta_v_ms / exhaust_velocity_ms)


# =============================================================================
# THRUST APPLICATION
# =============================================================================

def apply_thrust(
    state: ShipState,
    throttle: float = 1.0,
    gimbal_pitch_deg: float = 0.0,
    gimbal_yaw_deg: float = 0.0,
    dt: float = 1.0
) -> tuple[Vector3D, float]:
    """
    Calculate acceleration and propellant consumption from thrust.

    Implements F = ma with thrust vectoring (gimbal).

    Args:
        state: Current ship state
        throttle: Throttle setting 0.0 to 1.0
        gimbal_pitch_deg: Nozzle pitch deflection (degrees)
        gimbal_yaw_deg: Nozzle yaw deflection (degrees)
        dt: Time step (seconds)

    Returns:
        Tuple of (acceleration_vector, propellant_consumed_kg)
    """
    if throttle <= 0 or state.propellant_kg <= 0:
        return Vector3D.zero(), 0.0

    throttle = max(0.0, min(1.0, throttle))

    # Clamp gimbal angles
    gimbal_pitch_deg = max(-MAX_GIMBAL_ANGLE_DEG,
                          min(MAX_GIMBAL_ANGLE_DEG, gimbal_pitch_deg))
    gimbal_yaw_deg = max(-MAX_GIMBAL_ANGLE_DEG,
                        min(MAX_GIMBAL_ANGLE_DEG, gimbal_yaw_deg))

    # Calculate thrust direction with gimbal
    thrust_direction = state.forward

    if gimbal_pitch_deg != 0:
        pitch_rad = math.radians(gimbal_pitch_deg)
        thrust_direction = thrust_direction.rotate_around_axis(
            state.right, pitch_rad
        )

    if gimbal_yaw_deg != 0:
        yaw_rad = math.radians(gimbal_yaw_deg)
        thrust_direction = thrust_direction.rotate_around_axis(
            state.up, -yaw_rad  # Negative for right-handed
        )

    # Calculate thrust force
    thrust_force_n = state.thrust_n * throttle

    # Mass flow rate: dm/dt = F / v_e
    mass_flow_rate = thrust_force_n / state.exhaust_velocity_ms
    propellant_consumed = mass_flow_rate * dt

    # Don't consume more propellant than available
    propellant_consumed = min(propellant_consumed, state.propellant_kg)

    # Average mass during burn (for more accurate acceleration)
    avg_mass = state.mass_kg - propellant_consumed / 2
    if avg_mass <= 0:
        avg_mass = state.mass_kg

    # Acceleration: a = F / m
    acceleration_magnitude = thrust_force_n / avg_mass
    acceleration = thrust_direction * acceleration_magnitude

    return acceleration, propellant_consumed


def calculate_torque_from_thrust(
    state: ShipState,
    gimbal_pitch_deg: float = 0.0,
    gimbal_yaw_deg: float = 0.0,
    lever_arm_m: Optional[float] = None,
    throttle: float = 1.0
) -> Vector3D:
    """
    Calculate torque from thrust vectoring.

    Torque = lever_arm x Force

    Args:
        state: Current ship state
        gimbal_pitch_deg: Nozzle pitch deflection (degrees)
        gimbal_yaw_deg: Nozzle yaw deflection (degrees)
        lever_arm_m: Distance from CoM to engine (default: estimate from MOI)
        throttle: Throttle setting 0.0 to 1.0

    Returns:
        Torque vector in N*m
    """
    if throttle <= 0:
        return Vector3D.zero()

    # Estimate lever arm from moment of inertia if not provided
    # For elongated cylinder: I = (1/12) * m * L^2
    # L = sqrt(12 * I / m)
    # Lever arm (CoM to engine) approx 45% of length
    if lever_arm_m is None:
        if state.mass_kg > 0 and state.moment_of_inertia_kg_m2 > 0:
            length_m = math.sqrt(12 * state.moment_of_inertia_kg_m2 / state.mass_kg)
            lever_arm_m = length_m * 0.45
        else:
            lever_arm_m = 30.0  # Default estimate

    throttle = max(0.0, min(1.0, throttle))
    thrust_n = state.thrust_n * throttle

    # Calculate lateral force components
    pitch_rad = math.radians(gimbal_pitch_deg)
    yaw_rad = math.radians(gimbal_yaw_deg)

    lateral_pitch = thrust_n * math.sin(pitch_rad)  # Force in up direction
    lateral_yaw = thrust_n * math.sin(yaw_rad)  # Force in right direction

    # Torque = r x F
    # Pitch gimbal creates yaw torque (rotation around up axis)
    # Yaw gimbal creates pitch torque (rotation around right axis)
    torque_x = 0.0  # Roll (not from thrust vectoring)
    torque_y = lateral_pitch * lever_arm_m  # Pitch torque
    torque_z = -lateral_yaw * lever_arm_m  # Yaw torque

    return Vector3D(torque_x, torque_y, torque_z)


# =============================================================================
# ROTATION DYNAMICS
# =============================================================================

def calculate_moment_of_inertia(
    mass_kg: float,
    length_m: float
) -> dict[str, float]:
    """
    Calculate moment of inertia for a spacecraft.

    Models ship as elongated cylinder for rotation calculations.

    Args:
        mass_kg: Ship mass (kg)
        length_m: Ship length (meters)

    Returns:
        Dict with pitch_yaw and roll moments of inertia (kg*m^2)
    """
    # Pitch/Yaw - rotating perpendicular to length axis
    # I = (1/12) * m * L^2
    i_pitch_yaw = (1/12) * mass_kg * (length_m ** 2)

    # Roll - rotating about the long axis
    # Assuming width = length/4, radius = width/2 = length/8
    radius = length_m / 8
    i_roll = (1/2) * mass_kg * (radius ** 2)

    return {
        'pitch_yaw_kg_m2': i_pitch_yaw,
        'roll_kg_m2': i_roll
    }


def angular_acceleration_from_torque(
    torque_nm: float,
    moment_of_inertia_kg_m2: float
) -> float:
    """
    Calculate angular acceleration from torque.

    alpha = tau / I

    Args:
        torque_nm: Applied torque (N*m)
        moment_of_inertia_kg_m2: Moment of inertia (kg*m^2)

    Returns:
        Angular acceleration in rad/s^2
    """
    if moment_of_inertia_kg_m2 <= 0:
        return 0.0
    return torque_nm / moment_of_inertia_kg_m2


def time_to_rotate(
    angular_accel_rad_s2: float,
    angle_deg: float
) -> float:
    """
    Calculate time to rotate a given angle.

    Assumes bang-bang control: accelerate to midpoint, decelerate to stop.
    t = 2 * sqrt(theta / alpha)

    Args:
        angular_accel_rad_s2: Angular acceleration (rad/s^2)
        angle_deg: Rotation angle (degrees)

    Returns:
        Time to complete rotation in seconds
    """
    if angular_accel_rad_s2 <= 0:
        return float('inf')

    angle_rad = math.radians(angle_deg)
    return 2 * math.sqrt(angle_rad / angular_accel_rad_s2)


def max_angular_velocity(
    angular_accel_rad_s2: float,
    angle_deg: float
) -> float:
    """
    Calculate maximum angular velocity during rotation.

    Occurs at midpoint of bang-bang maneuver.
    omega_max = alpha * (t / 2)

    Args:
        angular_accel_rad_s2: Angular acceleration (rad/s^2)
        angle_deg: Total rotation angle (degrees)

    Returns:
        Maximum angular velocity in rad/s
    """
    t = time_to_rotate(angular_accel_rad_s2, angle_deg)
    if t == float('inf'):
        return 0.0
    return angular_accel_rad_s2 * (t / 2)


# =============================================================================
# TRAJECTORY PROPAGATION (EULER INTEGRATION)
# =============================================================================

def propagate_state(
    state: ShipState,
    dt: float,
    throttle: float = 0.0,
    gimbal_pitch_deg: float = 0.0,
    gimbal_yaw_deg: float = 0.0
) -> ShipState:
    """
    Propagate ship state forward in time using Euler integration.

    Updates position, velocity, orientation, angular velocity, and mass.

    Args:
        state: Current ship state
        dt: Time step in seconds
        throttle: Engine throttle 0.0 to 1.0
        gimbal_pitch_deg: Nozzle pitch deflection (degrees)
        gimbal_yaw_deg: Nozzle yaw deflection (degrees)

    Returns:
        New ship state after time step
    """
    new_state = state.copy()

    # Apply thrust if throttle > 0
    if throttle > 0 and new_state.propellant_kg > 0:
        acceleration, propellant_used = apply_thrust(
            new_state, throttle, gimbal_pitch_deg, gimbal_yaw_deg, dt
        )

        # Update velocity (Euler: v_new = v + a * dt)
        new_state.velocity = new_state.velocity + acceleration * dt

        # Update mass
        new_state.propellant_kg -= propellant_used
        new_state.propellant_kg = max(0.0, new_state.propellant_kg)
        new_state.mass_kg = new_state.dry_mass_kg + new_state.propellant_kg

        # Calculate torque from thrust vectoring
        torque = calculate_torque_from_thrust(
            new_state, gimbal_pitch_deg, gimbal_yaw_deg, throttle=throttle
        )

        # Angular acceleration (assuming torque is in body frame)
        if new_state.moment_of_inertia_kg_m2 > 0:
            alpha_pitch = torque.y / new_state.moment_of_inertia_kg_m2
            alpha_yaw = torque.z / new_state.moment_of_inertia_kg_m2

            # Update angular velocity
            new_state.angular_velocity = Vector3D(
                new_state.angular_velocity.x,
                new_state.angular_velocity.y + alpha_pitch * dt,
                new_state.angular_velocity.z + alpha_yaw * dt
            )

    # Update position (Euler: x_new = x + v * dt)
    new_state.position = new_state.position + new_state.velocity * dt

    # Update orientation from angular velocity
    if new_state.angular_velocity.magnitude > 0:
        omega = new_state.angular_velocity

        # Rotate forward vector
        # Pitch rotation (around right axis)
        if abs(omega.y) > 1e-10:
            new_state.forward = new_state.forward.rotate_around_axis(
                new_state.right, omega.y * dt
            )
            new_state.up = new_state.up.rotate_around_axis(
                new_state.right, omega.y * dt
            )

        # Yaw rotation (around up axis)
        if abs(omega.z) > 1e-10:
            new_state.forward = new_state.forward.rotate_around_axis(
                state.up, omega.z * dt  # Use original up for consistency
            )

        # Roll rotation (around forward axis)
        if abs(omega.x) > 1e-10:
            new_state.up = new_state.up.rotate_around_axis(
                new_state.forward, omega.x * dt
            )

        # Re-normalize to prevent drift
        new_state.forward = new_state.forward.normalized()
        new_state.up = new_state.up.normalized()

    return new_state


def propagate_trajectory(
    initial_state: ShipState,
    total_time: float,
    dt: float = 1.0,
    throttle: float = 0.0,
    gimbal_pitch_deg: float = 0.0,
    gimbal_yaw_deg: float = 0.0
) -> list[ShipState]:
    """
    Propagate ship trajectory over multiple time steps.

    Args:
        initial_state: Starting ship state
        total_time: Total simulation time (seconds)
        dt: Time step (seconds)
        throttle: Constant throttle setting
        gimbal_pitch_deg: Constant gimbal pitch
        gimbal_yaw_deg: Constant gimbal yaw

    Returns:
        List of ship states at each time step
    """
    states = [initial_state.copy()]
    current_state = initial_state.copy()

    t = 0.0
    while t < total_time:
        step = min(dt, total_time - t)
        current_state = propagate_state(
            current_state, step, throttle, gimbal_pitch_deg, gimbal_yaw_deg
        )
        states.append(current_state.copy())
        t += step

    return states


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_ship_state_from_specs(
    wet_mass_tons: float,
    dry_mass_tons: float,
    length_m: float,
    thrust_mn: float = MAIN_THRUST_MN,
    exhaust_velocity_kps: float = EXHAUST_VELOCITY_KPS,
    position: Optional[Vector3D] = None,
    velocity: Optional[Vector3D] = None,
    forward: Optional[Vector3D] = None
) -> ShipState:
    """
    Create a ShipState from ship specifications.

    Args:
        wet_mass_tons: Total mass with propellant (metric tons)
        dry_mass_tons: Mass without propellant (metric tons)
        length_m: Ship length (meters)
        thrust_mn: Main engine thrust (Meganewtons)
        exhaust_velocity_kps: Exhaust velocity (km/s)
        position: Initial position (default: origin)
        velocity: Initial velocity (default: zero)
        forward: Initial forward direction (default: +X)

    Returns:
        Configured ShipState
    """
    mass_kg = wet_mass_tons * 1000
    dry_mass_kg = dry_mass_tons * 1000
    propellant_kg = mass_kg - dry_mass_kg

    moi = calculate_moment_of_inertia(mass_kg, length_m)

    return ShipState(
        position=position or Vector3D.zero(),
        velocity=velocity or Vector3D.zero(),
        forward=forward.normalized() if forward else Vector3D.unit_x(),
        up=Vector3D.unit_z(),
        angular_velocity=Vector3D.zero(),
        mass_kg=mass_kg,
        dry_mass_kg=dry_mass_kg,
        propellant_kg=propellant_kg,
        thrust_n=thrust_mn * 1e6,
        exhaust_velocity_ms=exhaust_velocity_kps * 1000,
        moment_of_inertia_kg_m2=moi['pitch_yaw_kg_m2']
    )


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS PHYSICS MODULE - SELF TEST")
    print("=" * 70)

    # Test Vector3D
    print("\n--- Vector3D Tests ---")
    v1 = Vector3D(1, 0, 0)
    v2 = Vector3D(0, 1, 0)
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 dot v2 = {v1.dot(v2)}")
    print(f"v1 cross v2 = {v1.cross(v2)}")
    print(f"v1 magnitude = {v1.magnitude}")

    # Test delta-v calculations
    print("\n--- Delta-V Calculations ---")
    print(f"Exhaust velocity: {EXHAUST_VELOCITY_KPS} km/s")

    # Corvette example
    wet_mass = 1990  # tons
    dry_mass = 1895  # tons
    dv = tsiolkovsky_delta_v(
        EXHAUST_VELOCITY_MS,
        wet_mass * 1000,
        dry_mass * 1000
    )
    print(f"Corvette delta-v: {dv/1000:.1f} km/s")
    print(f"  (wet: {wet_mass}t, dry: {dry_mass}t, propellant: {wet_mass-dry_mass}t)")

    # Test propellant calculation
    target_dv = 500_000  # 500 km/s
    prop_needed = propellant_for_delta_v(
        target_dv,
        EXHAUST_VELOCITY_MS,
        dry_mass * 1000
    )
    print(f"Propellant for 500 km/s: {prop_needed/1000:.1f} tons")

    # Test ShipState
    print("\n--- ShipState Tests ---")
    corvette = create_ship_state_from_specs(
        wet_mass_tons=1990,
        dry_mass_tons=1895,
        length_m=65,
        thrust_mn=58.56,
        exhaust_velocity_kps=10256
    )
    print(f"Corvette state created:")
    print(f"  Mass: {corvette.mass_kg/1000:.0f} tons")
    print(f"  Max accel: {corvette.max_acceleration_g():.2f} g")
    print(f"  Remaining delta-v: {corvette.remaining_delta_v_kps():.1f} km/s")
    print(f"  MOI: {corvette.moment_of_inertia_kg_m2:.2e} kg*m^2")

    # Test thrust application
    print("\n--- Thrust Application ---")
    accel, prop_used = apply_thrust(corvette, throttle=1.0, dt=1.0)
    print(f"Full thrust for 1s:")
    print(f"  Acceleration: {accel.magnitude:.2f} m/s^2 ({accel.magnitude/G_STANDARD:.2f} g)")
    print(f"  Propellant used: {prop_used:.2f} kg")

    # Test rotation dynamics
    print("\n--- Rotation Dynamics ---")
    torque = calculate_torque_from_thrust(
        corvette,
        gimbal_pitch_deg=1.0,
        throttle=1.0
    )
    alpha = angular_acceleration_from_torque(
        torque.y,
        corvette.moment_of_inertia_kg_m2
    )
    t90 = time_to_rotate(alpha, 90)
    print(f"1 degree gimbal:")
    print(f"  Torque: {torque.y/1e6:.2f} MN*m")
    print(f"  Angular accel: {math.degrees(alpha):.3f} deg/s^2")
    print(f"  Time for 90 deg turn: {t90:.1f} s")

    # Test trajectory propagation
    print("\n--- Trajectory Propagation ---")
    initial = create_ship_state_from_specs(
        wet_mass_tons=1990,
        dry_mass_tons=1895,
        length_m=65
    )

    # 10 second burn at full throttle
    trajectory = propagate_trajectory(
        initial,
        total_time=10.0,
        dt=1.0,
        throttle=1.0
    )

    final = trajectory[-1]
    print(f"After 10s full throttle burn:")
    print(f"  Position: {final.position.x/1000:.1f} km")
    print(f"  Velocity: {final.velocity.x:.1f} m/s ({final.velocity.x/1000:.3f} km/s)")
    print(f"  Propellant remaining: {final.propellant_kg:.0f} kg")
    print(f"  Delta-v used: {(initial.remaining_delta_v_kps() - final.remaining_delta_v_kps()):.2f} km/s")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
