#!/usr/bin/env python3
"""
Projectile System Module for AI Commanders Space Battle Simulator

Implements velocity inheritance mechanics for space combat projectiles:
- Projectile: Base class with proper velocity inheritance from shooter
- KineticProjectile: Coilgun slugs with kinetic energy calculations
- ProjectileLauncher: System for launching kinetic projectiles

Physics principles:
- Projectiles inherit shooter's velocity at launch
- Final velocity = shooter_velocity + muzzle_direction * muzzle_velocity
- Kinetic energy is calculated from velocity in rest frame (total velocity)

Example:
- Ship moving at 50 km/s fires 10 km/s slug forward: slug = 60 km/s
- Ship moving at 50 km/s fires 10 km/s slug backward: slug = 40 km/s
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

try:
    from .physics import Vector3D, ShipState
except ImportError:
    from physics import Vector3D, ShipState


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Conversion factor: km/s to m/s
KPS_TO_MS = 1000.0

# Conversion factor: GJ to Joules
GJ_TO_J = 1e9


# =============================================================================
# PROJECTILE BASE CLASS
# =============================================================================

@dataclass
class Projectile:
    """
    Base class for all projectiles with velocity inheritance from shooter.

    All projectiles in space combat inherit the shooter's velocity at the
    moment of launch. The final projectile velocity is the vector sum of
    the shooter's velocity and the muzzle velocity.

    Attributes:
        position: Current position in world coordinates (meters)
        velocity: Current velocity in world coordinates (m/s)
                  MUST include shooter velocity - this is the velocity
                  in the rest frame of the simulation
        mass_kg: Projectile mass in kilograms
        launched_from_velocity: Velocity of the shooter at launch time (m/s)
                                Stored for reference and energy calculations
    """
    position: Vector3D
    velocity: Vector3D
    mass_kg: float
    launched_from_velocity: Vector3D = field(default_factory=Vector3D.zero)

    def __post_init__(self) -> None:
        """Validate that velocity inheritance is properly implemented."""
        # If launched_from_velocity is not set, assume it was included in velocity
        if self.launched_from_velocity == Vector3D.zero() and self.velocity != Vector3D.zero():
            # This is acceptable - velocity already contains the full velocity
            pass

    @property
    def speed_ms(self) -> float:
        """Current speed in m/s (rest frame)."""
        return self.velocity.magnitude

    @property
    def speed_kps(self) -> float:
        """Current speed in km/s (rest frame)."""
        return self.velocity.magnitude / KPS_TO_MS

    @property
    def kinetic_energy_j(self) -> float:
        """
        Kinetic energy in Joules, calculated from REST FRAME velocity.

        KE = 0.5 * m * v^2

        This uses the total velocity (including shooter velocity inheritance),
        which is the physically correct kinetic energy in the simulation frame.
        """
        v_squared = self.velocity.magnitude_squared
        return 0.5 * self.mass_kg * v_squared

    @property
    def kinetic_energy_gj(self) -> float:
        """Kinetic energy in Gigajoules, from REST FRAME velocity."""
        return self.kinetic_energy_j / GJ_TO_J

    def update(self, dt_seconds: float) -> None:
        """
        Update projectile position for a time step.

        Projectiles follow ballistic trajectories (no thrust after launch).

        Args:
            dt_seconds: Time step in seconds
        """
        self.position = self.position + self.velocity * dt_seconds

    def distance_to(self, target_position: Vector3D) -> float:
        """
        Calculate distance to a target position.

        Args:
            target_position: Target position in meters

        Returns:
            Distance in meters
        """
        return self.position.distance_to(target_position)


# =============================================================================
# KINETIC PROJECTILE CLASS (COILGUN SLUGS)
# =============================================================================

@dataclass
class KineticProjectile(Projectile):
    """
    Kinetic projectile for coilgun/railgun weapon systems.

    Coilgun slugs are accelerated electromagnetically and rely on kinetic
    energy for damage. The muzzle velocity is relative to the shooter,
    so the absolute velocity includes the shooter's motion.

    Velocity inheritance example:
    - Ship at 50 km/s fires forward at 10 km/s muzzle: slug = 60 km/s
    - Ship at 50 km/s fires backward at 10 km/s muzzle: slug = 40 km/s
    - Ship at 50 km/s fires perpendicular at 10 km/s: slug = sqrt(50^2+10^2) km/s

    Attributes:
        muzzle_velocity_kps: Muzzle velocity relative to shooter (km/s)
        muzzle_direction: Direction of fire (unit vector at launch)
    """
    muzzle_velocity_kps: float = 0.0
    muzzle_direction: Vector3D = field(default_factory=Vector3D.unit_x)

    @classmethod
    def from_launch(
        cls,
        shooter_position: Vector3D,
        shooter_velocity: Vector3D,
        target_direction: Vector3D,
        muzzle_velocity_kps: float,
        mass_kg: float
    ) -> KineticProjectile:
        """
        Create a kinetic projectile from launch parameters.

        This is the primary factory method for creating projectiles.
        It properly handles velocity inheritance from the shooter.

        Args:
            shooter_position: Position of the shooter (meters)
            shooter_velocity: Velocity of the shooter (m/s)
            target_direction: Direction to fire (will be normalized)
            muzzle_velocity_kps: Muzzle velocity relative to shooter (km/s)
            mass_kg: Projectile mass (kg)

        Returns:
            KineticProjectile with proper velocity inheritance
        """
        # Normalize the firing direction
        fire_direction = target_direction.normalized()

        # Calculate muzzle velocity vector (relative to shooter)
        muzzle_velocity_ms = muzzle_velocity_kps * KPS_TO_MS
        muzzle_vel_vector = fire_direction * muzzle_velocity_ms

        # Final velocity = shooter velocity + muzzle velocity
        # This is the CRITICAL velocity inheritance step
        final_velocity = shooter_velocity + muzzle_vel_vector

        return cls(
            position=Vector3D(
                shooter_position.x,
                shooter_position.y,
                shooter_position.z
            ),
            velocity=final_velocity,
            mass_kg=mass_kg,
            launched_from_velocity=Vector3D(
                shooter_velocity.x,
                shooter_velocity.y,
                shooter_velocity.z
            ),
            muzzle_velocity_kps=muzzle_velocity_kps,
            muzzle_direction=fire_direction
        )

    @property
    def muzzle_velocity_component(self) -> Vector3D:
        """
        The muzzle velocity component (relative to shooter).

        Returns:
            Muzzle velocity vector in m/s
        """
        return self.muzzle_direction * (self.muzzle_velocity_kps * KPS_TO_MS)

    @property
    def relative_velocity(self) -> Vector3D:
        """
        Velocity relative to the shooter at launch (muzzle velocity).

        This is useful for understanding the projectile's motion
        from the shooter's reference frame.

        Returns:
            Relative velocity in m/s
        """
        return self.velocity - self.launched_from_velocity


# =============================================================================
# PROJECTILE LAUNCHER CLASS
# =============================================================================

@dataclass
class ProjectileLauncher:
    """
    Launcher system for kinetic projectiles (coilguns/railguns).

    Handles projectile creation with proper velocity inheritance.
    Each launch explicitly adds shooter velocity to the projectile.

    Attributes:
        default_mass_kg: Default projectile mass (kg)
        default_muzzle_velocity_kps: Default muzzle velocity (km/s)
        magazine_capacity: Maximum rounds in magazine
        current_magazine: Current rounds available
        cooldown_seconds: Time between shots
        last_fire_time: Time of last shot (for cooldown)
    """
    default_mass_kg: float = 25.0  # Typical coilgun slug mass
    default_muzzle_velocity_kps: float = 10.0  # Typical coilgun muzzle velocity
    magazine_capacity: int = 100
    current_magazine: int = 100
    cooldown_seconds: float = 5.0
    last_fire_time: float = -5.0

    def can_fire(self, current_time: float) -> bool:
        """
        Check if the launcher can fire.

        Args:
            current_time: Current simulation time (seconds)

        Returns:
            True if ready to fire
        """
        if self.current_magazine <= 0:
            return False
        if current_time - self.last_fire_time < self.cooldown_seconds:
            return False
        return True

    def launch_kinetic(
        self,
        shooter_state: ShipState,
        target_direction: Vector3D,
        muzzle_velocity_kps: Optional[float] = None,
        mass_kg: Optional[float] = None,
        current_time: float = 0.0
    ) -> Optional[KineticProjectile]:
        """
        Launch a kinetic projectile from a ship.

        This method EXPLICITLY adds shooter velocity to the projectile,
        ensuring proper velocity inheritance.

        Args:
            shooter_state: Current state of the launching ship
            target_direction: Direction to fire (will be normalized)
            muzzle_velocity_kps: Override muzzle velocity (km/s)
            mass_kg: Override projectile mass (kg)
            current_time: Current simulation time (seconds)

        Returns:
            KineticProjectile with inherited velocity, or None if cannot fire
        """
        if not self.can_fire(current_time):
            return None

        # Use provided values or defaults
        muzzle_vel = muzzle_velocity_kps if muzzle_velocity_kps is not None else self.default_muzzle_velocity_kps
        proj_mass = mass_kg if mass_kg is not None else self.default_mass_kg

        # Create projectile with velocity inheritance
        projectile = KineticProjectile.from_launch(
            shooter_position=shooter_state.position,
            shooter_velocity=shooter_state.velocity,
            target_direction=target_direction,
            muzzle_velocity_kps=muzzle_vel,
            mass_kg=proj_mass
        )

        # Update launcher state
        self.current_magazine -= 1
        self.last_fire_time = current_time

        return projectile

    def calculate_intercept_direction(
        self,
        shooter_state: ShipState,
        target_position: Vector3D,
        target_velocity: Vector3D,
        muzzle_velocity_kps: Optional[float] = None
    ) -> Optional[Vector3D]:
        """
        Calculate the direction to fire for intercept.

        Accounts for relative motion and projectile travel time.

        Args:
            shooter_state: Current state of the launching ship
            target_position: Target position (meters)
            target_velocity: Target velocity (m/s)
            muzzle_velocity_kps: Muzzle velocity to use (km/s)

        Returns:
            Direction to fire for intercept, or None if no solution
        """
        muzzle_vel_kps = muzzle_velocity_kps if muzzle_velocity_kps is not None else self.default_muzzle_velocity_kps
        muzzle_vel_ms = muzzle_vel_kps * KPS_TO_MS

        # Relative position and velocity
        rel_pos = target_position - shooter_state.position
        rel_vel = target_velocity - shooter_state.velocity

        # Solve quadratic for intercept time
        # |rel_pos + rel_vel * t| = muzzle_vel * t
        # Expanding: |rel_pos|^2 + 2*(rel_pos.rel_vel)*t + |rel_vel|^2*t^2 = muzzle_vel^2 * t^2
        # (|rel_vel|^2 - muzzle_vel^2) * t^2 + 2*(rel_pos.rel_vel)*t + |rel_pos|^2 = 0

        a = rel_vel.magnitude_squared - muzzle_vel_ms * muzzle_vel_ms
        b = 2.0 * rel_pos.dot(rel_vel)
        c = rel_pos.magnitude_squared

        # Handle special case where relative velocity equals muzzle velocity
        if abs(a) < 1e-10:
            if abs(b) < 1e-10:
                return None
            t = -c / b
            if t <= 0:
                return None
        else:
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                return None

            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)

            # Pick the smallest positive time
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
            elif t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                return None

        # Calculate intercept point and direction
        intercept_point = target_position + target_velocity * t
        direction = (intercept_point - shooter_state.position).normalized()

        return direction


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_kinetic_energy_gj(mass_kg: float, velocity_kps: float) -> float:
    """
    Calculate kinetic energy in Gigajoules.

    Args:
        mass_kg: Mass in kilograms
        velocity_kps: Velocity in km/s (REST FRAME)

    Returns:
        Kinetic energy in Gigajoules
    """
    velocity_ms = velocity_kps * KPS_TO_MS
    energy_j = 0.5 * mass_kg * velocity_ms * velocity_ms
    return energy_j / GJ_TO_J


def calculate_impact_velocity(
    shooter_velocity: Vector3D,
    muzzle_velocity_kps: float,
    fire_direction: Vector3D,
    target_velocity: Vector3D
) -> float:
    """
    Calculate the relative impact velocity between projectile and target.

    Args:
        shooter_velocity: Shooter velocity (m/s)
        muzzle_velocity_kps: Muzzle velocity relative to shooter (km/s)
        fire_direction: Direction of fire (unit vector)
        target_velocity: Target velocity (m/s)

    Returns:
        Relative impact velocity in m/s
    """
    # Projectile velocity in rest frame
    muzzle_vel_ms = muzzle_velocity_kps * KPS_TO_MS
    projectile_velocity = shooter_velocity + fire_direction.normalized() * muzzle_vel_ms

    # Relative velocity
    relative_velocity = projectile_velocity - target_velocity

    return relative_velocity.magnitude


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS PROJECTILE MODULE - SELF TEST")
    print("=" * 70)

    # Test velocity inheritance - forward fire
    print("\n--- Velocity Inheritance: Forward Fire ---")
    shooter_vel = Vector3D(50000, 0, 0)  # 50 km/s forward
    fire_dir = Vector3D(1, 0, 0)  # Fire forward
    muzzle_vel = 10.0  # 10 km/s

    proj = KineticProjectile.from_launch(
        shooter_position=Vector3D.zero(),
        shooter_velocity=shooter_vel,
        target_direction=fire_dir,
        muzzle_velocity_kps=muzzle_vel,
        mass_kg=25.0
    )

    print(f"Shooter velocity: {shooter_vel.magnitude / 1000:.1f} km/s")
    print(f"Muzzle velocity: {muzzle_vel} km/s")
    print(f"Final projectile velocity: {proj.speed_kps:.1f} km/s")
    print(f"Expected: 60.0 km/s")
    assert abs(proj.speed_kps - 60.0) < 0.01, "Forward fire velocity inheritance failed!"
    print("PASS: 50 km/s ship + 10 km/s forward = 60 km/s")

    # Test velocity inheritance - backward fire
    print("\n--- Velocity Inheritance: Backward Fire ---")
    fire_dir_back = Vector3D(-1, 0, 0)  # Fire backward

    proj_back = KineticProjectile.from_launch(
        shooter_position=Vector3D.zero(),
        shooter_velocity=shooter_vel,
        target_direction=fire_dir_back,
        muzzle_velocity_kps=muzzle_vel,
        mass_kg=25.0
    )

    print(f"Shooter velocity: {shooter_vel.magnitude / 1000:.1f} km/s (forward)")
    print(f"Muzzle velocity: {muzzle_vel} km/s (backward)")
    print(f"Final projectile velocity: {proj_back.speed_kps:.1f} km/s")
    print(f"Expected: 40.0 km/s")
    assert abs(proj_back.speed_kps - 40.0) < 0.01, "Backward fire velocity inheritance failed!"
    print("PASS: 50 km/s ship - 10 km/s backward = 40 km/s")

    # Test velocity inheritance - perpendicular fire
    print("\n--- Velocity Inheritance: Perpendicular Fire ---")
    fire_dir_perp = Vector3D(0, 1, 0)  # Fire perpendicular

    proj_perp = KineticProjectile.from_launch(
        shooter_position=Vector3D.zero(),
        shooter_velocity=shooter_vel,
        target_direction=fire_dir_perp,
        muzzle_velocity_kps=muzzle_vel,
        mass_kg=25.0
    )

    expected_perp = math.sqrt(50**2 + 10**2)
    print(f"Shooter velocity: {shooter_vel.magnitude / 1000:.1f} km/s (forward)")
    print(f"Muzzle velocity: {muzzle_vel} km/s (perpendicular)")
    print(f"Final projectile velocity: {proj_perp.speed_kps:.2f} km/s")
    print(f"Expected: {expected_perp:.2f} km/s")
    assert abs(proj_perp.speed_kps - expected_perp) < 0.01, "Perpendicular fire velocity inheritance failed!"
    print(f"PASS: sqrt(50^2 + 10^2) = {expected_perp:.2f} km/s")

    # Test kinetic energy calculation from REST FRAME
    print("\n--- Kinetic Energy (REST FRAME) ---")
    # 25 kg slug at 60 km/s
    # KE = 0.5 * 25 * (60000)^2 = 0.5 * 25 * 3.6e9 = 45e9 J = 45 GJ
    print(f"Projectile mass: {proj.mass_kg} kg")
    print(f"Projectile speed (rest frame): {proj.speed_kps:.1f} km/s")
    print(f"Kinetic energy: {proj.kinetic_energy_gj:.2f} GJ")
    expected_ke = 0.5 * 25 * (60000**2) / 1e9
    print(f"Expected: {expected_ke:.2f} GJ")
    assert abs(proj.kinetic_energy_gj - expected_ke) < 0.01, "Kinetic energy calculation failed!"
    print("PASS: KE calculated from REST FRAME velocity")

    # Compare to backward-fired slug
    print("\n--- Kinetic Energy Comparison ---")
    print(f"Forward-fired slug (60 km/s): {proj.kinetic_energy_gj:.2f} GJ")
    print(f"Backward-fired slug (40 km/s): {proj_back.kinetic_energy_gj:.2f} GJ")
    print(f"Difference: {proj.kinetic_energy_gj - proj_back.kinetic_energy_gj:.2f} GJ")
    print("This demonstrates why firing direction matters for kinetic energy!")

    # Test ProjectileLauncher
    print("\n--- Projectile Launcher ---")
    launcher = ProjectileLauncher(
        default_mass_kg=25.0,
        default_muzzle_velocity_kps=10.0
    )

    ship_state = ShipState(
        position=Vector3D.zero(),
        velocity=Vector3D(50000, 0, 0)  # 50 km/s
    )

    launched = launcher.launch_kinetic(
        shooter_state=ship_state,
        target_direction=Vector3D(1, 0, 0),
        current_time=0.0
    )

    if launched:
        print(f"Launched projectile:")
        print(f"  Position: {launched.position}")
        print(f"  Velocity: {launched.speed_kps:.1f} km/s")
        print(f"  Kinetic energy: {launched.kinetic_energy_gj:.2f} GJ")
        print(f"  Magazine remaining: {launcher.current_magazine}/{launcher.magazine_capacity}")

    # Test cannot fire during cooldown
    print("\n--- Cooldown Test ---")
    cannot_fire = launcher.launch_kinetic(
        shooter_state=ship_state,
        target_direction=Vector3D(1, 0, 0),
        current_time=1.0  # Only 1 second later
    )
    print(f"Fire during cooldown: {cannot_fire}")
    assert cannot_fire is None, "Should not be able to fire during cooldown!"
    print("PASS: Correctly blocked during cooldown")

    can_fire = launcher.launch_kinetic(
        shooter_state=ship_state,
        target_direction=Vector3D(1, 0, 0),
        current_time=10.0  # After cooldown
    )
    print(f"Fire after cooldown: {can_fire is not None}")
    assert can_fire is not None, "Should be able to fire after cooldown!"
    print("PASS: Correctly allowed after cooldown")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
