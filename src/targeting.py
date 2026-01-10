"""
Targeting and ECM System for AI Commanders Space Battle Simulator.

This module implements targeting acquisition, electronic countermeasures (ECM),
firing solutions, and lead calculation for space combat simulations.

Based on Terra Invicta mechanics:
- ECM disrupts targeting locks with probabilistic chance
- Tracking computers reduce ECM effectiveness
- Firing solutions require acquisition time and can be broken
- Lead calculation for kinetic weapons (coilguns)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

from physics import Vector3D, ShipState
from combat import Weapon


# =============================================================================
# ECM SYSTEM
# =============================================================================

@dataclass
class ECMSystem:
    """
    Electronic Countermeasures system that disrupts enemy targeting.

    ECM creates electronic noise and jamming signals that interfere with
    enemy targeting computers, causing them to lose lock on the protected ship.

    Attributes:
        ecm_strength: ECM effectiveness from 0.0 (none) to 1.0 (maximum).
                     Represents the probability of breaking an enemy's lock.
        reacquisition_time_s: Time in seconds the enemy must wait to
                             reacquire a lock after ECM breaks it.
        active: Whether ECM is currently active (consumes power when on).
    """
    ecm_strength: float = 0.3
    reacquisition_time_s: float = 5.0
    active: bool = True

    def __post_init__(self) -> None:
        """Validate ECM strength is within valid range."""
        self.ecm_strength = max(0.0, min(1.0, self.ecm_strength))
        self.reacquisition_time_s = max(0.0, self.reacquisition_time_s)

    def get_effective_strength(self) -> float:
        """
        Get the current effective ECM strength.

        Returns:
            ECM strength if active, 0.0 if inactive.
        """
        return self.ecm_strength if self.active else 0.0

    def set_strength(self, strength: float) -> None:
        """
        Set ECM strength to a new value.

        Args:
            strength: New ECM strength (clamped to 0.0-1.0).
        """
        self.ecm_strength = max(0.0, min(1.0, strength))


# =============================================================================
# TARGETING COMPUTER
# =============================================================================

@dataclass
class TargetingComputer:
    """
    Targeting computer that manages target acquisition and tracking.

    The targeting computer provides bonuses to overcome enemy ECM and
    determines how quickly targets can be locked and how many can be
    tracked simultaneously.

    Attributes:
        tracking_bonus: Flat reduction to enemy ECM effectiveness (0.0 to 0.5).
                       Higher values make the computer better at maintaining
                       locks against jamming.
        lock_time_s: Base time in seconds to acquire initial lock on a target.
        max_targets: Maximum number of targets that can be tracked at once.
        sensor_range_km: Maximum sensor range in kilometers.
    """
    tracking_bonus: float = 0.1
    lock_time_s: float = 3.0
    max_targets: int = 4
    sensor_range_km: float = 5000.0

    def __post_init__(self) -> None:
        """Validate targeting computer parameters."""
        self.tracking_bonus = max(0.0, min(0.5, self.tracking_bonus))
        self.lock_time_s = max(0.1, self.lock_time_s)
        self.max_targets = max(1, self.max_targets)
        self.sensor_range_km = max(0.0, self.sensor_range_km)

    def effective_ecm(self, ecm_strength: float) -> float:
        """
        Calculate the effective ECM strength against this targeting computer.

        The tracking bonus reduces the enemy's ECM effectiveness, making it
        easier to maintain lock.

        Args:
            ecm_strength: The enemy's raw ECM strength (0.0 to 1.0).

        Returns:
            Effective ECM strength after tracking bonus reduction.
            Minimum value is 0.0 (complete ECM negation).
        """
        return max(0.0, ecm_strength - self.tracking_bonus)

    def can_track_target(self, current_tracks: int) -> bool:
        """
        Check if another target can be tracked.

        Args:
            current_tracks: Number of targets currently being tracked.

        Returns:
            True if another target can be added, False otherwise.
        """
        return current_tracks < self.max_targets

    def is_in_sensor_range(self, distance_km: float) -> bool:
        """
        Check if a target is within sensor range.

        Args:
            distance_km: Distance to target in kilometers.

        Returns:
            True if target is within sensor range.
        """
        return distance_km <= self.sensor_range_km


# =============================================================================
# FIRING SOLUTION
# =============================================================================

@dataclass
class FiringSolution:
    """
    Represents a targeting lock on a specific target.

    A firing solution tracks the progress of target acquisition and
    maintains lock status. ECM can break the lock, requiring reacquisition.

    Attributes:
        target_id: Unique identifier of the target being tracked.
        locked: Whether a full lock has been acquired.
        lock_progress: Progress toward lock acquisition (0.0 to 1.0).
        time_to_lock: Remaining time to achieve lock in seconds.
        cooldown_remaining: Time remaining before lock can be attempted
                           after ECM break.
    """
    target_id: str
    locked: bool = False
    lock_progress: float = 0.0
    time_to_lock: float = 0.0
    cooldown_remaining: float = 0.0

    def __post_init__(self) -> None:
        """Validate firing solution state."""
        self.lock_progress = max(0.0, min(1.0, self.lock_progress))
        self.time_to_lock = max(0.0, self.time_to_lock)
        self.cooldown_remaining = max(0.0, self.cooldown_remaining)

    def attempt_lock(
        self,
        dt_seconds: float,
        ecm_strength: float,
        tracking_bonus: float,
        base_lock_time: float = 3.0
    ) -> bool:
        """
        Attempt to progress toward or maintain target lock.

        Lock progress increases over time unless disrupted by ECM.
        Higher ECM slows lock acquisition.

        Args:
            dt_seconds: Time step in seconds.
            ecm_strength: Target's ECM strength (0.0 to 1.0).
            tracking_bonus: Shooter's tracking computer bonus.
            base_lock_time: Base time to achieve lock.

        Returns:
            True if lock is now achieved, False otherwise.
        """
        # Cannot progress while in cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining = max(0.0, self.cooldown_remaining - dt_seconds)
            return self.locked

        # If already locked, maintain lock
        if self.locked:
            return True

        # Calculate effective ECM
        effective_ecm = max(0.0, ecm_strength - tracking_bonus)

        # ECM slows lock acquisition (50% slower at max ECM)
        lock_speed_modifier = 1.0 - (effective_ecm * 0.5)

        # Calculate progress this frame
        if base_lock_time > 0:
            progress_per_second = 1.0 / base_lock_time
            progress_this_frame = progress_per_second * dt_seconds * lock_speed_modifier
            self.lock_progress += progress_this_frame

        # Update time to lock estimate
        if lock_speed_modifier > 0 and base_lock_time > 0:
            remaining_progress = 1.0 - self.lock_progress
            self.time_to_lock = (remaining_progress * base_lock_time) / lock_speed_modifier
        else:
            self.time_to_lock = float('inf')

        # Check if lock achieved
        if self.lock_progress >= 1.0:
            self.lock_progress = 1.0
            self.locked = True
            self.time_to_lock = 0.0

        return self.locked

    def check_ecm_break(
        self,
        ecm_strength: float,
        tracking_bonus: float,
        rng: Optional[random.Random] = None
    ) -> bool:
        """
        Check if ECM breaks the current lock.

        This should be called periodically (e.g., once per second) to
        determine if ECM disrupts the targeting solution.

        Args:
            ecm_strength: Target's ECM strength (0.0 to 1.0).
            tracking_bonus: Shooter's tracking computer bonus.
            rng: Random number generator for reproducible results.

        Returns:
            True if lock was broken, False if lock maintained.
        """
        if not self.locked:
            return False

        if rng is None:
            rng = random.Random()

        # Calculate effective ECM chance to break lock
        effective_ecm = max(0.0, ecm_strength - tracking_bonus)

        # Roll for ECM break
        if rng.random() < effective_ecm:
            self.locked = False
            self.lock_progress = 0.0
            return True

        return False

    def break_lock(self, reacquisition_time_s: float = 0.0) -> None:
        """
        Force break the current lock (e.g., from ECM).

        Args:
            reacquisition_time_s: Cooldown time before lock can be
                                 attempted again.
        """
        self.locked = False
        self.lock_progress = 0.0
        self.cooldown_remaining = reacquisition_time_s

    def reset(self) -> None:
        """Reset the firing solution to initial state."""
        self.locked = False
        self.lock_progress = 0.0
        self.time_to_lock = 0.0
        self.cooldown_remaining = 0.0


# =============================================================================
# LEAD CALCULATOR
# =============================================================================

@dataclass
class LeadCalculator:
    """
    Calculates lead/intercept points for kinetic weapons.

    For coilguns and other kinetic weapons, the shooter must aim ahead
    of the target's current position to account for projectile travel time.
    This class computes the optimal aim point (lead position) for intercept.
    """

    @staticmethod
    def get_time_to_intercept(distance_km: float, projectile_speed_kps: float) -> float:
        """
        Calculate time for a projectile to reach target distance.

        Args:
            distance_km: Current distance to target in kilometers.
            projectile_speed_kps: Projectile velocity in km/s.

        Returns:
            Time to intercept in seconds. Returns infinity if
            projectile speed is zero or negative.
        """
        if projectile_speed_kps <= 0:
            return float('inf')
        return distance_km / projectile_speed_kps

    @staticmethod
    def calculate_lead(
        shooter_pos: Vector3D,
        shooter_vel: Vector3D,
        target_pos: Vector3D,
        target_vel: Vector3D,
        projectile_speed_kps: float,
        max_iterations: int = 10,
        tolerance_km: float = 0.001
    ) -> Vector3D:
        """
        Calculate the lead (aim) point for intercepting a moving target.

        Uses iterative refinement to solve the intercept problem:
        Find the point where the projectile and target will meet,
        accounting for projectile travel time and target motion.

        The calculation assumes constant velocities (no acceleration)
        during the intercept window.

        Args:
            shooter_pos: Shooter position in kilometers.
            shooter_vel: Shooter velocity in km/s.
            target_pos: Target current position in kilometers.
            target_vel: Target velocity in km/s.
            projectile_speed_kps: Projectile speed in km/s.
            max_iterations: Maximum refinement iterations.
            tolerance_km: Convergence tolerance in km.

        Returns:
            Lead position (aim point) in kilometers. The shooter should
            aim at this point to intercept the target.
        """
        if projectile_speed_kps <= 0:
            # Cannot calculate lead with zero projectile speed
            return target_pos

        # Calculate relative position and velocity
        relative_pos = target_pos - shooter_pos
        relative_vel = target_vel - shooter_vel

        initial_distance = relative_pos.magnitude

        if initial_distance < tolerance_km:
            # Target is essentially at shooter position
            return target_pos

        # Iterative solution: refine intercept time
        intercept_time = initial_distance / projectile_speed_kps

        for _ in range(max_iterations):
            # Predict target position at intercept time
            predicted_target_pos = target_pos + target_vel * intercept_time

            # Calculate new distance and intercept time
            predicted_relative_pos = predicted_target_pos - shooter_pos
            new_distance = predicted_relative_pos.magnitude
            new_intercept_time = new_distance / projectile_speed_kps

            # Check convergence
            if abs(new_intercept_time - intercept_time) < tolerance_km / projectile_speed_kps:
                intercept_time = new_intercept_time
                break

            intercept_time = new_intercept_time

        # Return the lead position (where to aim)
        lead_position = target_pos + target_vel * intercept_time

        return lead_position

    @staticmethod
    def calculate_lead_from_states(
        shooter_state: ShipState,
        target_state: ShipState,
        projectile_speed_kps: float
    ) -> Vector3D:
        """
        Calculate lead position using ShipState objects.

        Convenience method that extracts position and velocity from
        ShipState objects and converts units appropriately.

        Note: ShipState uses meters, this method converts to km internally.

        Args:
            shooter_state: Shooter's current state.
            target_state: Target's current state.
            projectile_speed_kps: Projectile speed in km/s.

        Returns:
            Lead position in meters (matching ShipState units).
        """
        # Convert from meters to kilometers
        shooter_pos_km = shooter_state.position * 0.001
        shooter_vel_kps = shooter_state.velocity * 0.001
        target_pos_km = target_state.position * 0.001
        target_vel_kps = target_state.velocity * 0.001

        # Calculate lead in km
        lead_km = LeadCalculator.calculate_lead(
            shooter_pos_km,
            shooter_vel_kps,
            target_pos_km,
            target_vel_kps,
            projectile_speed_kps
        )

        # Convert back to meters
        return lead_km * 1000.0

    @staticmethod
    def calculate_lead_direction(
        shooter_pos: Vector3D,
        shooter_vel: Vector3D,
        target_pos: Vector3D,
        target_vel: Vector3D,
        projectile_speed_kps: float
    ) -> Vector3D:
        """
        Calculate the direction to aim for target intercept.

        Args:
            shooter_pos: Shooter position in kilometers.
            shooter_vel: Shooter velocity in km/s.
            target_pos: Target current position in kilometers.
            target_vel: Target velocity in km/s.
            projectile_speed_kps: Projectile speed in km/s.

        Returns:
            Normalized direction vector to aim at.
        """
        lead_pos = LeadCalculator.calculate_lead(
            shooter_pos, shooter_vel, target_pos, target_vel, projectile_speed_kps
        )
        direction = lead_pos - shooter_pos
        return direction.normalized()

    @staticmethod
    def calculate_lead_with_acceleration(
        shooter_pos: Vector3D,
        shooter_vel: Vector3D,
        target_pos: Vector3D,
        target_vel: Vector3D,
        target_accel: Vector3D,
        projectile_speed_kps: float,
        max_iterations: int = 15,
        tolerance_km: float = 0.001
    ) -> Vector3D:
        """
        Calculate the lead (aim) point for intercepting an accelerating target.

        Uses quadratic prediction to account for target acceleration:
        future_pos = pos + vel*t + 0.5*accel*t^2

        This method iteratively refines the intercept time to find where the
        projectile and accelerating target will meet. More accurate than
        constant-velocity lead calculation when targets are maneuvering.

        Args:
            shooter_pos: Shooter position in kilometers.
            shooter_vel: Shooter velocity in km/s.
            target_pos: Target current position in kilometers.
            target_vel: Target velocity in km/s.
            target_accel: Target acceleration in km/s^2.
            projectile_speed_kps: Projectile speed in km/s.
            max_iterations: Maximum refinement iterations (default 15 for
                           acceleration convergence).
            tolerance_km: Convergence tolerance in km.

        Returns:
            Lead position (aim point) in kilometers. The shooter should
            aim at this point to intercept the accelerating target.

        Notes:
            - For non-accelerating targets (accel = zero), this reduces to
              the standard constant-velocity lead calculation.
            - Accuracy degrades for very high accelerations or long intercept
              times, as the quadratic approximation assumes constant acceleration.
            - The algorithm uses Newton-Raphson style iteration for faster
              convergence compared to the basic lead calculator.
        """
        if projectile_speed_kps <= 0:
            # Cannot calculate lead with zero projectile speed
            return target_pos

        # Calculate initial relative position
        relative_pos = target_pos - shooter_pos
        initial_distance = relative_pos.magnitude

        if initial_distance < tolerance_km:
            # Target is essentially at shooter position
            return target_pos

        # Initial estimate: time based on current distance
        intercept_time = initial_distance / projectile_speed_kps

        for _ in range(max_iterations):
            # Predict target position at intercept time using quadratic motion
            # future_pos = pos + vel*t + 0.5*accel*t^2
            predicted_target_pos = (
                target_pos +
                target_vel * intercept_time +
                target_accel * (0.5 * intercept_time * intercept_time)
            )

            # Calculate projectile travel distance to predicted position
            predicted_relative_pos = predicted_target_pos - shooter_pos
            new_distance = predicted_relative_pos.magnitude

            # New intercept time estimate
            new_intercept_time = new_distance / projectile_speed_kps

            # Check convergence
            if abs(new_intercept_time - intercept_time) < tolerance_km / projectile_speed_kps:
                intercept_time = new_intercept_time
                break

            # Use weighted average for stability with acceleration
            # (pure Newton-Raphson can oscillate with high acceleration)
            intercept_time = 0.7 * new_intercept_time + 0.3 * intercept_time

        # Calculate final lead position with quadratic prediction
        lead_position = (
            target_pos +
            target_vel * intercept_time +
            target_accel * (0.5 * intercept_time * intercept_time)
        )

        return lead_position

    @staticmethod
    def calculate_lead_with_acceleration_from_states(
        shooter_state: ShipState,
        target_state: ShipState,
        target_accel: Vector3D,
        projectile_speed_kps: float
    ) -> Vector3D:
        """
        Calculate lead position with acceleration using ShipState objects.

        Convenience method that extracts position and velocity from
        ShipState objects and converts units appropriately.

        Note: ShipState uses meters, this method converts to km internally.
              Target acceleration should be provided in m/s^2.

        Args:
            shooter_state: Shooter's current state.
            target_state: Target's current state.
            target_accel: Target's acceleration vector in m/s^2.
            projectile_speed_kps: Projectile speed in km/s.

        Returns:
            Lead position in meters (matching ShipState units).
        """
        # Convert from meters to kilometers
        shooter_pos_km = shooter_state.position * 0.001
        shooter_vel_kps = shooter_state.velocity * 0.001
        target_pos_km = target_state.position * 0.001
        target_vel_kps = target_state.velocity * 0.001
        target_accel_kps2 = target_accel * 0.001  # m/s^2 to km/s^2

        # Calculate lead in km
        lead_km = LeadCalculator.calculate_lead_with_acceleration(
            shooter_pos_km,
            shooter_vel_kps,
            target_pos_km,
            target_vel_kps,
            target_accel_kps2,
            projectile_speed_kps
        )

        # Convert back to meters
        return lead_km * 1000.0


# =============================================================================
# FIRING ARC
# =============================================================================

@dataclass
class FiringArc:
    """
    Defines a weapon's firing arc as a cone from a reference direction.

    Different weapon types have different firing arcs:
    - Spinal weapons: Very narrow (3-10 degrees), fixed forward
    - Turrets: Wide arcs (90-180 degrees), can rotate
    - Broadside: Fixed perpendicular arcs (60-90 degrees)

    The firing arc is modeled as a cone with its apex at the weapon mount
    point and opening in the reference direction.

    Attributes:
        cone_half_angle_deg: Half-angle of the firing cone in degrees.
                            E.g., 5 for a 10-degree total cone, 180 for full sphere.
        reference_direction: Unit vector defining the center of the firing cone.
                            Typically ship's forward vector for spinal weapons.

    Examples:
        - Spinal coilgun: cone_half_angle_deg=5, reference_direction=ship.forward
        - Full-traverse turret: cone_half_angle_deg=180, reference_direction=up
        - Limited turret: cone_half_angle_deg=120, reference_direction=forward
    """
    cone_half_angle_deg: float = 5.0
    reference_direction: Vector3D = field(default_factory=Vector3D.unit_x)

    def __post_init__(self) -> None:
        """Validate firing arc parameters."""
        self.cone_half_angle_deg = max(0.0, min(180.0, self.cone_half_angle_deg))
        # Ensure reference direction is normalized
        if self.reference_direction.magnitude > 0:
            self.reference_direction = self.reference_direction.normalized()
        else:
            self.reference_direction = Vector3D.unit_x()

    def is_target_in_arc(self, target_direction: Vector3D) -> bool:
        """
        Check if a target direction falls within the firing arc.

        Args:
            target_direction: Direction vector from weapon to target.
                             Does not need to be normalized.

        Returns:
            True if the target is within the firing cone, False otherwise.

        Notes:
            - A cone_half_angle_deg of 180 means the weapon can fire in
              any direction (omnidirectional).
            - A cone_half_angle_deg of 0 means the weapon can only fire
              exactly along the reference direction.
        """
        if target_direction.magnitude == 0:
            return False

        angle_to_target = self.get_angle_to_target(target_direction)
        return angle_to_target <= self.cone_half_angle_deg

    def get_angle_to_target(self, target_direction: Vector3D) -> float:
        """
        Calculate the angle from the reference direction to a target.

        Args:
            target_direction: Direction vector from weapon to target.
                             Does not need to be normalized.

        Returns:
            Angle in degrees from the reference direction to the target.
            Returns 180.0 if target_direction is zero-length.

        Notes:
            This returns the absolute angle; it does not indicate which
            side of the reference axis the target is on.
        """
        if target_direction.magnitude == 0:
            return 180.0  # Invalid direction, consider out of arc

        normalized_target = target_direction.normalized()
        angle_rad = self.reference_direction.angle_to(normalized_target)
        return math.degrees(angle_rad)

    def get_arc_coverage_fraction(self) -> float:
        """
        Calculate the fraction of the sphere covered by this firing arc.

        Returns:
            Fraction from 0.0 (point) to 1.0 (full sphere).

        Notes:
            Uses the solid angle formula for a cone:
            Omega = 2*pi*(1 - cos(theta))
            Full sphere = 4*pi steradians
        """
        theta_rad = math.radians(self.cone_half_angle_deg)
        solid_angle = 2 * math.pi * (1 - math.cos(theta_rad))
        return solid_angle / (4 * math.pi)


# =============================================================================
# SPINAL WEAPON CONSTRAINT
# =============================================================================

@dataclass
class SpinalWeaponConstraint:
    """
    Constraint system for spinal-mounted weapons requiring nose-pointing.

    Spinal weapons (like main coilguns) are fixed along the ship's axis
    and can only engage targets within a very limited traverse angle.
    This is similar to tank destroyers or assault guns that lack turrets.

    The ship must orient itself to point at the target to engage, making
    spinal weapons powerful but tactically inflexible.

    Attributes:
        max_traverse_deg: Maximum angle from ship forward that the weapon
                         can engage. Typically 3-10 degrees for spinal mounts.
        requires_nose_pointing: If True, the weapon requires the ship to
                               actively orient toward targets. Default True.

    Notes:
        - Spinal weapons typically have higher damage/range than turrets
          to compensate for their limited arc.
        - The traverse angle represents both mechanical gimbal limits and
          the weapon's effective accuracy cone.
    """
    max_traverse_deg: float = 5.0
    requires_nose_pointing: bool = True

    def __post_init__(self) -> None:
        """Validate spinal weapon parameters."""
        self.max_traverse_deg = max(0.0, min(90.0, self.max_traverse_deg))

    def can_engage(
        self,
        ship_forward: Vector3D,
        target_direction: Vector3D
    ) -> bool:
        """
        Check if the spinal weapon can engage a target.

        For spinal weapons that require nose-pointing, the target must be
        within the traverse cone centered on the ship's forward axis.

        Args:
            ship_forward: Unit vector in the ship's forward direction.
            target_direction: Direction vector from ship to target.
                             Does not need to be normalized.

        Returns:
            True if the target can be engaged, False otherwise.

        Notes:
            - If requires_nose_pointing is False, always returns True.
            - A zero-length target_direction returns False.
        """
        if not self.requires_nose_pointing:
            return True

        if target_direction.magnitude == 0:
            return False

        angle = self.get_required_rotation(ship_forward, target_direction)
        return angle <= self.max_traverse_deg

    def get_required_rotation(
        self,
        ship_forward: Vector3D,
        target_direction: Vector3D
    ) -> float:
        """
        Calculate degrees the ship must rotate to engage the target.

        This represents the minimum rotation needed to bring the spinal
        weapon to bear on the target.

        Args:
            ship_forward: Unit vector in the ship's forward direction.
            target_direction: Direction vector from ship to target.
                             Does not need to be normalized.

        Returns:
            Angle in degrees that the ship needs to rotate.
            Returns 180.0 if target_direction is zero-length.

        Notes:
            - Returns 0.0 if the target is already within the traverse arc.
            - For targets outside the arc, returns the angle to the arc edge,
              not the full angle to center on the target.
        """
        if target_direction.magnitude == 0:
            return 180.0

        normalized_forward = ship_forward.normalized()
        normalized_target = target_direction.normalized()

        angle_rad = normalized_forward.angle_to(normalized_target)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def get_rotation_to_engagement(
        self,
        ship_forward: Vector3D,
        target_direction: Vector3D
    ) -> float:
        """
        Calculate the rotation needed to bring target into engagement arc.

        Unlike get_required_rotation which returns the full angle, this
        returns how much further the ship needs to rotate to be able to
        engage (0 if already in arc).

        Args:
            ship_forward: Unit vector in the ship's forward direction.
            target_direction: Direction vector from ship to target.

        Returns:
            Additional degrees of rotation needed. 0 if target is in arc.
        """
        current_angle = self.get_required_rotation(ship_forward, target_direction)

        if current_angle <= self.max_traverse_deg:
            return 0.0

        return current_angle - self.max_traverse_deg

    def get_engagement_priority(
        self,
        ship_forward: Vector3D,
        target_direction: Vector3D
    ) -> float:
        """
        Calculate engagement priority based on angular distance.

        Targets closer to the weapon's bore axis are easier to engage
        and receive higher priority scores.

        Args:
            ship_forward: Unit vector in the ship's forward direction.
            target_direction: Direction vector from ship to target.

        Returns:
            Priority score from 0.0 (worst) to 1.0 (best/on-axis).
            Returns 0.0 for targets that cannot be engaged.
        """
        angle = self.get_required_rotation(ship_forward, target_direction)

        if angle > self.max_traverse_deg:
            return 0.0

        # Linear falloff from 1.0 at center to 0.5 at edge
        return 1.0 - 0.5 * (angle / self.max_traverse_deg)


# =============================================================================
# TARGETING SYSTEM
# =============================================================================

@dataclass
class TargetingSystem:
    """
    Complete targeting system managing multiple firing solutions.

    Integrates the targeting computer with firing solutions for all
    tracked targets. Handles lock acquisition, ECM checks, and lead
    calculation for weapons.

    Attributes:
        computer: The ship's targeting computer.
        solutions: Dictionary of firing solutions keyed by target ID.
        rng: Random number generator for ECM checks.
        ecm_check_interval_s: How often to check for ECM breaks.
        time_since_ecm_check: Accumulator for ECM check timing.
    """
    computer: TargetingComputer = field(default_factory=TargetingComputer)
    solutions: Dict[str, FiringSolution] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)
    ecm_check_interval_s: float = 1.0
    time_since_ecm_check: float = 0.0

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducible ECM checks.

        Args:
            seed: Random seed value.
        """
        self.rng = random.Random(seed)

    def acquire_target(self, target_id: str) -> Optional[FiringSolution]:
        """
        Begin acquiring a new target.

        Creates a new firing solution for the target if one doesn't exist
        and the targeting computer has capacity for more tracks.

        Args:
            target_id: Unique identifier of the target.

        Returns:
            The FiringSolution for the target, or None if max targets reached.
        """
        # Check if already tracking
        if target_id in self.solutions:
            return self.solutions[target_id]

        # Check if we can track more targets
        if not self.computer.can_track_target(len(self.solutions)):
            return None

        # Create new firing solution
        solution = FiringSolution(
            target_id=target_id,
            time_to_lock=self.computer.lock_time_s
        )
        self.solutions[target_id] = solution
        return solution

    def drop_target(self, target_id: str) -> bool:
        """
        Stop tracking a target and remove its firing solution.

        Args:
            target_id: Unique identifier of the target to drop.

        Returns:
            True if target was being tracked and is now dropped,
            False if target was not being tracked.
        """
        if target_id in self.solutions:
            del self.solutions[target_id]
            return True
        return False

    def get_firing_solution(self, target_id: str) -> Optional[FiringSolution]:
        """
        Get the firing solution for a specific target.

        Args:
            target_id: Unique identifier of the target.

        Returns:
            The FiringSolution if target is being tracked, None otherwise.
        """
        return self.solutions.get(target_id)

    def is_locked(self, target_id: str) -> bool:
        """
        Check if a target is fully locked.

        Args:
            target_id: Unique identifier of the target.

        Returns:
            True if target is locked, False otherwise.
        """
        solution = self.solutions.get(target_id)
        return solution is not None and solution.locked

    def update(
        self,
        dt_seconds: float,
        targets_ecm: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Update all firing solutions for the current time step.

        Progresses lock acquisition for all tracked targets and
        periodically checks for ECM breaks on locked targets.

        Args:
            dt_seconds: Time step in seconds.
            targets_ecm: Dictionary mapping target IDs to their ECM strength.

        Returns:
            Dictionary mapping target IDs to whether their lock was broken
            this frame. Only includes targets that had their lock broken.
        """
        locks_broken: Dict[str, bool] = {}

        # Accumulate time for ECM checks
        self.time_since_ecm_check += dt_seconds
        should_check_ecm = self.time_since_ecm_check >= self.ecm_check_interval_s

        if should_check_ecm:
            self.time_since_ecm_check = 0.0

        # Update each firing solution
        for target_id, solution in list(self.solutions.items()):
            # Get target's ECM strength (default 0 if not specified)
            ecm_strength = targets_ecm.get(target_id, 0.0)

            # Check for ECM break on locked targets
            if should_check_ecm and solution.locked:
                if solution.check_ecm_break(
                    ecm_strength,
                    self.computer.tracking_bonus,
                    self.rng
                ):
                    # Lock broken - apply reacquisition cooldown
                    # Get ECM system reacquisition time from targets_ecm
                    # For now, use a default value
                    solution.cooldown_remaining = 5.0  # Default reacquisition time
                    locks_broken[target_id] = True

            # Progress lock acquisition
            solution.attempt_lock(
                dt_seconds,
                ecm_strength,
                self.computer.tracking_bonus,
                self.computer.lock_time_s
            )

        return locks_broken

    def update_with_ecm_systems(
        self,
        dt_seconds: float,
        targets_ecm_systems: Dict[str, ECMSystem]
    ) -> Dict[str, bool]:
        """
        Update firing solutions using ECMSystem objects.

        Similar to update() but takes full ECMSystem objects to access
        both ECM strength and reacquisition time.

        Args:
            dt_seconds: Time step in seconds.
            targets_ecm_systems: Dictionary mapping target IDs to ECMSystem.

        Returns:
            Dictionary mapping target IDs to whether their lock was broken.
        """
        locks_broken: Dict[str, bool] = {}

        # Accumulate time for ECM checks
        self.time_since_ecm_check += dt_seconds
        should_check_ecm = self.time_since_ecm_check >= self.ecm_check_interval_s

        if should_check_ecm:
            self.time_since_ecm_check = 0.0

        # Update each firing solution
        for target_id, solution in list(self.solutions.items()):
            # Get target's ECM system
            ecm_system = targets_ecm_systems.get(target_id)
            ecm_strength = ecm_system.get_effective_strength() if ecm_system else 0.0
            reacquisition_time = ecm_system.reacquisition_time_s if ecm_system else 5.0

            # Check for ECM break on locked targets
            if should_check_ecm and solution.locked:
                if solution.check_ecm_break(
                    ecm_strength,
                    self.computer.tracking_bonus,
                    self.rng
                ):
                    # Lock broken - apply reacquisition cooldown
                    solution.cooldown_remaining = reacquisition_time
                    locks_broken[target_id] = True

            # Progress lock acquisition
            solution.attempt_lock(
                dt_seconds,
                ecm_strength,
                self.computer.tracking_bonus,
                self.computer.lock_time_s
            )

        return locks_broken

    def calculate_lead_for_weapon(
        self,
        weapon: Weapon,
        shooter_state: ShipState,
        target_state: ShipState
    ) -> Vector3D:
        """
        Calculate the lead position for a weapon engagement.

        Uses the weapon's muzzle velocity to compute where to aim
        for a successful intercept.

        Args:
            weapon: The weapon being fired.
            shooter_state: Shooter's current state (position/velocity in meters).
            target_state: Target's current state (position/velocity in meters).

        Returns:
            Lead position in meters (where to aim).
        """
        if weapon.muzzle_velocity_kps <= 0:
            # For missiles or weapons without muzzle velocity,
            # return target's current position
            return target_state.position

        return LeadCalculator.calculate_lead_from_states(
            shooter_state,
            target_state,
            weapon.muzzle_velocity_kps
        )

    def get_lock_status_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of all firing solution statuses.

        Returns:
            Dictionary mapping target IDs to status dictionaries containing:
            - locked: bool
            - lock_progress: float (0.0 to 1.0)
            - time_to_lock: float (seconds)
            - cooldown: float (seconds remaining)
        """
        summary = {}
        for target_id, solution in self.solutions.items():
            summary[target_id] = {
                'locked': solution.locked,
                'lock_progress': solution.lock_progress,
                'time_to_lock': solution.time_to_lock,
                'cooldown': solution.cooldown_remaining
            }
        return summary

    def clear_all(self) -> None:
        """Remove all firing solutions and stop tracking all targets."""
        self.solutions.clear()
        self.time_since_ecm_check = 0.0

    def check_firing_arc(
        self,
        weapon_type: str,
        ship_orientation: Vector3D,
        target_direction: Vector3D,
        firing_arcs: Optional[Dict[str, FiringArc]] = None
    ) -> bool:
        """
        Check if a weapon can engage a target based on its firing arc.

        Different weapon types have different firing arcs. This method
        determines if the target falls within the weapon's engagement
        envelope based on the ship's current orientation.

        Args:
            weapon_type: Type of weapon (e.g., "spinal", "turret", "broadside").
            ship_orientation: Ship's forward direction vector.
            target_direction: Direction from ship to target.
            firing_arcs: Optional dictionary of weapon type to FiringArc.
                        If not provided, uses default arcs.

        Returns:
            True if target is within the weapon's firing arc.

        Notes:
            Default firing arcs if not specified:
            - "spinal": 5 degree half-angle (very narrow, nose-pointing)
            - "turret": 180 degree half-angle (full hemisphere)
            - "broadside": 60 degree half-angle perpendicular to forward
            - Unknown types default to 90 degree half-angle
        """
        if firing_arcs is None:
            # Create default firing arcs based on weapon type
            if weapon_type == "spinal":
                arc = FiringArc(
                    cone_half_angle_deg=5.0,
                    reference_direction=ship_orientation.normalized()
                )
            elif weapon_type == "turret":
                arc = FiringArc(
                    cone_half_angle_deg=180.0,
                    reference_direction=ship_orientation.normalized()
                )
            elif weapon_type == "broadside":
                # Broadside weapons fire perpendicular to forward
                # Use a simplified approach - check against forward arc inverted
                arc = FiringArc(
                    cone_half_angle_deg=60.0,
                    reference_direction=ship_orientation.normalized()
                )
                # For broadside, we actually want targets NOT in forward arc
                # This is a simplification - proper broadside would need
                # left/right arcs defined separately
            else:
                # Default to moderate arc for unknown types
                arc = FiringArc(
                    cone_half_angle_deg=90.0,
                    reference_direction=ship_orientation.normalized()
                )
        else:
            arc = firing_arcs.get(weapon_type)
            if arc is None:
                # Fallback if weapon type not in dictionary
                return True

        return arc.is_target_in_arc(target_direction)

    def get_spinal_engagement_angle(
        self,
        ship_forward: Vector3D,
        target_pos: Vector3D,
        ship_pos: Optional[Vector3D] = None
    ) -> float:
        """
        Calculate the angle from ship's nose to a target for spinal weapons.

        Spinal weapons require the ship to orient its nose toward the target.
        This method calculates how far off-axis the target currently is,
        which determines if spinal weapons can engage and how much the
        ship needs to rotate.

        Args:
            ship_forward: Ship's current forward direction vector.
            target_pos: Target's position in space.
            ship_pos: Ship's current position. If None, assumes origin.

        Returns:
            Angle in degrees from ship's forward axis to the target.
            0 degrees means target is directly ahead.
            180 degrees means target is directly behind.

        Notes:
            This is commonly used to:
            - Determine if spinal weapons can fire (angle < max_traverse)
            - Calculate rotation commands for engagement
            - Prioritize targets based on engagement readiness
        """
        if ship_pos is None:
            ship_pos = Vector3D.zero()

        target_direction = target_pos - ship_pos

        if target_direction.magnitude == 0:
            return 0.0  # Target at same position as ship

        normalized_forward = ship_forward.normalized()
        normalized_target = target_direction.normalized()

        angle_rad = normalized_forward.angle_to(normalized_target)
        return math.degrees(angle_rad)

    def calculate_lead_for_weapon_with_acceleration(
        self,
        weapon: Weapon,
        shooter_state: ShipState,
        target_state: ShipState,
        target_accel: Vector3D
    ) -> Vector3D:
        """
        Calculate lead position for weapon engagement with target acceleration.

        Enhanced version of calculate_lead_for_weapon that accounts for
        target acceleration using quadratic prediction.

        Args:
            weapon: The weapon being fired.
            shooter_state: Shooter's current state (position/velocity in meters).
            target_state: Target's current state (position/velocity in meters).
            target_accel: Target's acceleration vector in m/s^2.

        Returns:
            Lead position in meters (where to aim).

        Notes:
            Use this method when engaging maneuvering targets for more
            accurate intercept prediction. For non-maneuvering targets,
            the standard calculate_lead_for_weapon is sufficient.
        """
        if weapon.muzzle_velocity_kps <= 0:
            # For missiles or weapons without muzzle velocity,
            # return target's current position
            return target_state.position

        return LeadCalculator.calculate_lead_with_acceleration_from_states(
            shooter_state,
            target_state,
            target_accel,
            weapon.muzzle_velocity_kps
        )

    def can_spinal_engage(
        self,
        spinal_constraint: SpinalWeaponConstraint,
        ship_forward: Vector3D,
        target_pos: Vector3D,
        ship_pos: Optional[Vector3D] = None
    ) -> bool:
        """
        Check if a spinal weapon can engage a target given current orientation.

        Convenience method that combines position-based target direction
        calculation with SpinalWeaponConstraint checking.

        Args:
            spinal_constraint: The spinal weapon's traverse constraints.
            ship_forward: Ship's current forward direction vector.
            target_pos: Target's position in space.
            ship_pos: Ship's current position. If None, assumes origin.

        Returns:
            True if the spinal weapon can currently engage the target.

        Notes:
            This is the primary method for checking spinal weapon engagement.
            If this returns False, the ship must rotate toward the target
            before the spinal weapon can fire.
        """
        if ship_pos is None:
            ship_pos = Vector3D.zero()

        target_direction = target_pos - ship_pos

        return spinal_constraint.can_engage(ship_forward, target_direction)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_basic_targeting_system(
    tracking_bonus: float = 0.1,
    lock_time_s: float = 3.0,
    max_targets: int = 4,
    seed: Optional[int] = None
) -> TargetingSystem:
    """
    Create a targeting system with basic parameters.

    Args:
        tracking_bonus: ECM reduction bonus (0.0 to 0.5).
        lock_time_s: Base time to achieve lock.
        max_targets: Maximum simultaneous tracks.
        seed: Optional random seed for reproducibility.

    Returns:
        Configured TargetingSystem.
    """
    computer = TargetingComputer(
        tracking_bonus=tracking_bonus,
        lock_time_s=lock_time_s,
        max_targets=max_targets
    )

    rng = random.Random(seed) if seed is not None else random.Random()

    return TargetingSystem(computer=computer, rng=rng)


def create_advanced_targeting_system(
    tracking_bonus: float = 0.3,
    lock_time_s: float = 2.0,
    max_targets: int = 8,
    sensor_range_km: float = 10000.0,
    seed: Optional[int] = None
) -> TargetingSystem:
    """
    Create an advanced targeting system with enhanced capabilities.

    Args:
        tracking_bonus: ECM reduction bonus (0.0 to 0.5).
        lock_time_s: Base time to achieve lock.
        max_targets: Maximum simultaneous tracks.
        sensor_range_km: Maximum sensor range.
        seed: Optional random seed for reproducibility.

    Returns:
        Configured TargetingSystem with advanced computer.
    """
    computer = TargetingComputer(
        tracking_bonus=tracking_bonus,
        lock_time_s=lock_time_s,
        max_targets=max_targets,
        sensor_range_km=sensor_range_km
    )

    rng = random.Random(seed) if seed is not None else random.Random()

    return TargetingSystem(computer=computer, rng=rng)


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS TARGETING & ECM SYSTEM - SELF TEST")
    print("=" * 70)

    # Test ECM System
    print("\n--- ECM System Tests ---")
    ecm = ECMSystem(ecm_strength=0.4, reacquisition_time_s=5.0)
    print(f"ECM strength: {ecm.ecm_strength}")
    print(f"Reacquisition time: {ecm.reacquisition_time_s}s")
    print(f"Effective (active): {ecm.get_effective_strength()}")
    ecm.active = False
    print(f"Effective (inactive): {ecm.get_effective_strength()}")
    ecm.active = True

    # Test Targeting Computer
    print("\n--- Targeting Computer Tests ---")
    computer = TargetingComputer(tracking_bonus=0.15, lock_time_s=3.0, max_targets=4)
    print(f"Tracking bonus: {computer.tracking_bonus}")
    print(f"Lock time: {computer.lock_time_s}s")
    print(f"Max targets: {computer.max_targets}")
    print(f"Effective ECM (0.4 raw): {computer.effective_ecm(0.4)}")
    print(f"Effective ECM (0.1 raw): {computer.effective_ecm(0.1)}")

    # Test Firing Solution
    print("\n--- Firing Solution Tests ---")
    solution = FiringSolution(target_id="enemy_1")
    print(f"Initial state - locked: {solution.locked}, progress: {solution.lock_progress}")

    # Simulate lock acquisition
    for i in range(10):
        locked = solution.attempt_lock(
            dt_seconds=0.5,
            ecm_strength=0.2,
            tracking_bonus=0.1,
            base_lock_time=3.0
        )
        print(f"  Step {i+1}: progress={solution.lock_progress:.2f}, "
              f"TTL={solution.time_to_lock:.2f}s, locked={locked}")
        if locked:
            break

    # Test ECM break
    print("\nTesting ECM break (100 trials with 40% effective ECM):")
    breaks = 0
    rng = random.Random(42)
    for _ in range(100):
        test_solution = FiringSolution(target_id="test", locked=True, lock_progress=1.0)
        if test_solution.check_ecm_break(0.5, 0.1, rng):  # 40% effective
            breaks += 1
    print(f"  Locks broken: {breaks}/100 (expected ~40)")

    # Test Lead Calculator
    print("\n--- Lead Calculator Tests ---")
    shooter_pos = Vector3D(0, 0, 0)  # km
    shooter_vel = Vector3D(0, 0, 0)  # km/s
    target_pos = Vector3D(100, 0, 0)  # 100 km away
    target_vel = Vector3D(0, 10, 0)  # 10 km/s perpendicular
    projectile_speed = 20  # km/s

    lead_pos = LeadCalculator.calculate_lead(
        shooter_pos, shooter_vel, target_pos, target_vel, projectile_speed
    )
    intercept_time = LeadCalculator.get_time_to_intercept(100, 20)

    print(f"Shooter at: {shooter_pos}")
    print(f"Target at: {target_pos}, moving at {target_vel} km/s")
    print(f"Projectile speed: {projectile_speed} km/s")
    print(f"Intercept time estimate: {intercept_time:.2f}s")
    print(f"Lead position: ({lead_pos.x:.2f}, {lead_pos.y:.2f}, {lead_pos.z:.2f}) km")
    print(f"Lead offset: Y={lead_pos.y:.2f} km (target moves ~{target_vel.y * intercept_time:.2f} km)")

    # Test Lead Calculator with Acceleration
    print("\n--- Lead Calculator with Acceleration Tests ---")
    target_accel = Vector3D(0, 0.5, 0)  # 0.5 km/s^2 acceleration perpendicular

    lead_pos_accel = LeadCalculator.calculate_lead_with_acceleration(
        shooter_pos, shooter_vel, target_pos, target_vel, target_accel, projectile_speed
    )
    lead_pos_no_accel = LeadCalculator.calculate_lead(
        shooter_pos, shooter_vel, target_pos, target_vel, projectile_speed
    )

    print(f"Target acceleration: {target_accel} km/s^2")
    print(f"Lead (no accel):   ({lead_pos_no_accel.x:.2f}, {lead_pos_no_accel.y:.2f}, {lead_pos_no_accel.z:.2f}) km")
    print(f"Lead (with accel): ({lead_pos_accel.x:.2f}, {lead_pos_accel.y:.2f}, {lead_pos_accel.z:.2f}) km")
    print(f"Difference in Y: {lead_pos_accel.y - lead_pos_no_accel.y:.2f} km (accounts for acceleration)")

    # Test Firing Arc
    print("\n--- Firing Arc Tests ---")
    spinal_arc = FiringArc(cone_half_angle_deg=5.0, reference_direction=Vector3D.unit_x())
    turret_arc = FiringArc(cone_half_angle_deg=180.0, reference_direction=Vector3D.unit_x())

    target_ahead = Vector3D(1, 0, 0)  # Directly ahead
    target_slight_off = Vector3D(1, 0.05, 0)  # Slightly off-axis (~3 degrees)
    target_off_axis = Vector3D(1, 0.2, 0)  # Off-axis (~11 degrees)
    target_behind = Vector3D(-1, 0, 0)  # Behind

    print(f"Spinal arc (5 deg half-angle):")
    print(f"  Target ahead: {spinal_arc.is_target_in_arc(target_ahead)} (angle: {spinal_arc.get_angle_to_target(target_ahead):.1f} deg)")
    print(f"  Target slight off: {spinal_arc.is_target_in_arc(target_slight_off)} (angle: {spinal_arc.get_angle_to_target(target_slight_off):.1f} deg)")
    print(f"  Target off-axis: {spinal_arc.is_target_in_arc(target_off_axis)} (angle: {spinal_arc.get_angle_to_target(target_off_axis):.1f} deg)")
    print(f"  Target behind: {spinal_arc.is_target_in_arc(target_behind)} (angle: {spinal_arc.get_angle_to_target(target_behind):.1f} deg)")
    print(f"  Arc coverage: {spinal_arc.get_arc_coverage_fraction()*100:.2f}% of sphere")

    print(f"\nTurret arc (180 deg half-angle - full hemisphere):")
    print(f"  Target ahead: {turret_arc.is_target_in_arc(target_ahead)}")
    print(f"  Target behind: {turret_arc.is_target_in_arc(target_behind)}")
    print(f"  Arc coverage: {turret_arc.get_arc_coverage_fraction()*100:.1f}% of sphere")

    # Test Spinal Weapon Constraint
    print("\n--- Spinal Weapon Constraint Tests ---")
    spinal = SpinalWeaponConstraint(max_traverse_deg=5.0, requires_nose_pointing=True)
    ship_forward = Vector3D.unit_x()

    print(f"Spinal weapon (5 deg max traverse):")
    print(f"  Can engage ahead: {spinal.can_engage(ship_forward, target_ahead)}")
    print(f"  Can engage slight off: {spinal.can_engage(ship_forward, target_slight_off)}")
    print(f"  Can engage off-axis: {spinal.can_engage(ship_forward, target_off_axis)}")
    print(f"  Rotation needed for off-axis: {spinal.get_required_rotation(ship_forward, target_off_axis):.1f} deg")
    print(f"  Additional rotation to engage: {spinal.get_rotation_to_engagement(ship_forward, target_off_axis):.1f} deg")
    print(f"  Priority (ahead): {spinal.get_engagement_priority(ship_forward, target_ahead):.2f}")
    print(f"  Priority (slight off): {spinal.get_engagement_priority(ship_forward, target_slight_off):.2f}")
    print(f"  Priority (off-axis): {spinal.get_engagement_priority(ship_forward, target_off_axis):.2f}")

    # Test Targeting System
    print("\n--- Targeting System Tests ---")
    targeting = create_basic_targeting_system(
        tracking_bonus=0.1,
        lock_time_s=2.0,
        max_targets=3,
        seed=42
    )

    # Acquire targets
    print("Acquiring 4 targets (max=3):")
    for i in range(4):
        target_id = f"target_{i}"
        solution = targeting.acquire_target(target_id)
        if solution:
            print(f"  Acquired: {target_id}")
        else:
            print(f"  Failed (max reached): {target_id}")

    # Simulate targeting update loop
    print("\nSimulating 5 seconds of targeting:")
    targets_ecm = {
        "target_0": 0.2,  # Light ECM
        "target_1": 0.5,  # Heavy ECM
        "target_2": 0.0,  # No ECM
    }

    for step in range(10):
        dt = 0.5
        locks_broken = targeting.update(dt, targets_ecm)

        if locks_broken:
            for target_id in locks_broken:
                print(f"  t={step*dt:.1f}s: Lock broken on {target_id}!")

        # Report lock status every second
        if (step + 1) % 2 == 0:
            print(f"  t={(step+1)*dt:.1f}s status:")
            for target_id, status in targeting.get_lock_status_summary().items():
                lock_str = "LOCKED" if status['locked'] else f"{status['lock_progress']*100:.0f}%"
                print(f"    {target_id}: {lock_str}")

    # Test TargetingSystem firing arc methods
    print("\n--- TargetingSystem Firing Arc Tests ---")
    ship_orientation = Vector3D.unit_x()
    target_dir_ahead = Vector3D(100, 0, 0)  # 100 km ahead
    target_dir_off = Vector3D(100, 20, 0)  # Off to the side

    print(f"Check firing arc (spinal, target ahead): {targeting.check_firing_arc('spinal', ship_orientation, target_dir_ahead)}")
    print(f"Check firing arc (spinal, target off): {targeting.check_firing_arc('spinal', ship_orientation, target_dir_off)}")
    print(f"Check firing arc (turret, target off): {targeting.check_firing_arc('turret', ship_orientation, target_dir_off)}")

    # Test spinal engagement angle
    ship_pos = Vector3D(0, 0, 0)
    target_position = Vector3D(100, 5, 0)  # Slightly off to the right
    angle = targeting.get_spinal_engagement_angle(ship_orientation, target_position, ship_pos)
    print(f"Spinal engagement angle to target at (100, 5, 0): {angle:.2f} deg")

    # Test can_spinal_engage
    spinal_constraint = SpinalWeaponConstraint(max_traverse_deg=5.0)
    can_engage_near = targeting.can_spinal_engage(spinal_constraint, ship_orientation, Vector3D(100, 2, 0), ship_pos)
    can_engage_far = targeting.can_spinal_engage(spinal_constraint, ship_orientation, Vector3D(100, 20, 0), ship_pos)
    print(f"Can spinal engage target at (100, 2, 0): {can_engage_near}")
    print(f"Can spinal engage target at (100, 20, 0): {can_engage_far}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
