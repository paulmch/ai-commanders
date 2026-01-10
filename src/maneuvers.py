#!/usr/bin/env python3
"""
Maneuver Command System for AI Commanders Space Battle Simulator.

This module provides a comprehensive set of maneuver commands that an LLM captain
can order, along with execution and planning systems:

- ThrustManeuver: Main engine burns (BurnToward, BurnAway, MatchVelocity, etc.)
- RotationManeuver: Ship orientation changes (RotateToFace, RotateToBroadside, etc.)
- CombinedManeuver: Complex multi-step maneuvers (FlipAndBurn, EvasiveJink, etc.)
- ManeuverExecutor: Executes maneuvers step by step
- ManeuverPlanner: Suggests optimal maneuvers for tactical situations

Each maneuver provides:
- Estimated completion time
- Estimated delta-v cost
- Interruptibility
- Progress tracking (0-100%)
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

# Try/except import pattern for compatibility
try:
    from .physics import (
        Vector3D, ShipState, G_STANDARD,
        apply_thrust, propagate_state,
        time_to_rotate, angular_acceleration_from_torque,
        calculate_torque_from_thrust, propellant_for_delta_v,
        mass_after_burn
    )
except ImportError:
    from physics import (
        Vector3D, ShipState, G_STANDARD,
        apply_thrust, propagate_state,
        time_to_rotate, angular_acceleration_from_torque,
        calculate_torque_from_thrust, propellant_for_delta_v,
        mass_after_burn
    )


# =============================================================================
# MANEUVER STATUS
# =============================================================================

class ManeuverStatus(Enum):
    """Status of a maneuver execution."""
    PENDING = auto()      # Not yet started
    IN_PROGRESS = auto()  # Currently executing
    COMPLETED = auto()    # Successfully finished
    ABORTED = auto()      # Cancelled mid-execution
    FAILED = auto()       # Could not complete (e.g., insufficient fuel)


@dataclass
class ManeuverResult:
    """
    Result of a maneuver execution step.

    Attributes:
        status: Current maneuver status
        progress: Completion percentage (0-100)
        throttle: Recommended throttle setting (0.0-1.0)
        thrust_direction: Direction to apply thrust (unit vector)
        gimbal_pitch_deg: Gimbal pitch adjustment
        gimbal_yaw_deg: Gimbal yaw adjustment
        rotation_target: Target orientation (if rotating)
        message: Human-readable status message
    """
    status: ManeuverStatus
    progress: float  # 0-100%
    throttle: float = 0.0
    thrust_direction: Optional[Vector3D] = None
    gimbal_pitch_deg: float = 0.0
    gimbal_yaw_deg: float = 0.0
    rotation_target: Optional[Vector3D] = None
    message: str = ""


# =============================================================================
# BASE MANEUVER CLASS
# =============================================================================

@dataclass
class Maneuver(ABC):
    """
    Abstract base class for all maneuvers.

    All maneuvers must provide estimates for completion time and delta-v cost,
    and must be interruptible with progress tracking.
    """

    # Maneuver metadata
    name: str = field(default="", init=False)
    description: str = field(default="", init=False)

    # Execution state
    _status: ManeuverStatus = field(default=ManeuverStatus.PENDING, init=False)
    _progress: float = field(default=0.0, init=False)  # 0-100
    _elapsed_time: float = field(default=0.0, init=False)
    _delta_v_expended: float = field(default=0.0, init=False)

    @property
    def status(self) -> ManeuverStatus:
        """Current maneuver status."""
        return self._status

    @property
    def progress(self) -> float:
        """Completion progress (0-100%)."""
        return self._progress

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since maneuver started (seconds)."""
        return self._elapsed_time

    @property
    def delta_v_expended(self) -> float:
        """Delta-v expended so far (m/s)."""
        return self._delta_v_expended

    @property
    def is_complete(self) -> bool:
        """Check if maneuver has finished (successfully or not)."""
        return self._status in (ManeuverStatus.COMPLETED, ManeuverStatus.ABORTED, ManeuverStatus.FAILED)

    @abstractmethod
    def estimate_completion_time(self, ship: ShipState) -> float:
        """
        Estimate time to complete the maneuver.

        Args:
            ship: Current ship state

        Returns:
            Estimated completion time in seconds
        """
        ...

    @abstractmethod
    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        """
        Estimate delta-v required for the maneuver.

        Args:
            ship: Current ship state

        Returns:
            Estimated delta-v cost in m/s
        """
        ...

    @abstractmethod
    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        """
        Execute one timestep of the maneuver.

        Args:
            ship: Current ship state
            dt: Time step in seconds

        Returns:
            ManeuverResult with control outputs for this step
        """
        ...

    def abort(self) -> None:
        """Abort the maneuver mid-execution."""
        if self._status == ManeuverStatus.IN_PROGRESS:
            self._status = ManeuverStatus.ABORTED

    def reset(self) -> None:
        """Reset maneuver to initial state for re-execution."""
        self._status = ManeuverStatus.PENDING
        self._progress = 0.0
        self._elapsed_time = 0.0
        self._delta_v_expended = 0.0


# =============================================================================
# THRUST MANEUVERS
# =============================================================================

@dataclass
class BurnToward(Maneuver):
    """
    Burn toward a target position.

    The ship will orient toward the target and apply thrust to accelerate
    in that direction.
    """
    target_position: Vector3D
    throttle: float = 1.0  # Throttle level (0.0-1.0)
    max_duration: float = 60.0  # Maximum burn duration (seconds)

    def __post_init__(self):
        self.name = "BurnToward"
        self.description = f"Burn toward target at {self.throttle*100:.0f}% throttle"

    def estimate_completion_time(self, ship: ShipState) -> float:
        """Returns max_duration as this is a timed burn."""
        return self.max_duration

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        """Estimate delta-v for the full burn duration."""
        accel = ship.max_acceleration_ms2() * self.throttle
        return accel * self.max_duration

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        # Calculate direction to target
        direction = self.target_position - ship.position
        distance = direction.magnitude

        if distance < 1.0:  # Within 1 meter - close enough
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                message="Reached target position"
            )

        direction = direction.normalized()

        # Update elapsed time and progress
        self._elapsed_time += dt
        self._progress = min(100.0, (self._elapsed_time / self.max_duration) * 100)

        # Track delta-v expended
        accel = ship.max_acceleration_ms2() * self.throttle
        self._delta_v_expended += accel * dt

        # Check if max duration reached
        if self._elapsed_time >= self.max_duration:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                throttle=0.0,
                message="Burn complete - max duration reached"
            )

        # Check fuel
        if ship.propellant_kg <= 0:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Burn failed - out of propellant"
            )

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=self.throttle,
            thrust_direction=direction,
            rotation_target=direction,
            message=f"Burning toward target, distance: {distance/1000:.1f} km"
        )


@dataclass
class BurnAway(Maneuver):
    """
    Burn away from a target position (retreat/escape).

    The ship will orient away from the target and apply thrust.
    """
    target_position: Vector3D
    throttle: float = 1.0
    max_duration: float = 60.0

    def __post_init__(self):
        self.name = "BurnAway"
        self.description = f"Burn away from target at {self.throttle*100:.0f}% throttle"

    def estimate_completion_time(self, ship: ShipState) -> float:
        return self.max_duration

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        accel = ship.max_acceleration_ms2() * self.throttle
        return accel * self.max_duration

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        # Calculate direction away from target
        direction = ship.position - self.target_position
        distance = direction.magnitude

        if distance < 1.0:
            # At same position - pick arbitrary direction
            direction = Vector3D.unit_x()
        else:
            direction = direction.normalized()

        self._elapsed_time += dt
        self._progress = min(100.0, (self._elapsed_time / self.max_duration) * 100)

        accel = ship.max_acceleration_ms2() * self.throttle
        self._delta_v_expended += accel * dt

        if self._elapsed_time >= self.max_duration:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                throttle=0.0,
                message="Escape burn complete"
            )

        if ship.propellant_kg <= 0:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Burn failed - out of propellant"
            )

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=self.throttle,
            thrust_direction=direction,
            rotation_target=direction,
            message=f"Burning away from target, distance: {distance/1000:.1f} km"
        )


@dataclass
class MatchVelocity(Maneuver):
    """
    Burn to match another ship's velocity.

    Useful for formation flying or closing to boarding range.
    """
    target_velocity: Vector3D
    tolerance_ms: float = 10.0  # Velocity match tolerance (m/s)
    max_duration: float = 300.0

    def __post_init__(self):
        self.name = "MatchVelocity"
        self.description = f"Match target velocity (tolerance: {self.tolerance_ms} m/s)"

    def estimate_completion_time(self, ship: ShipState) -> float:
        delta_v = (self.target_velocity - ship.velocity).magnitude
        accel = ship.max_acceleration_ms2()
        if accel <= 0:
            return float('inf')
        return min(delta_v / accel, self.max_duration)

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        return (self.target_velocity - ship.velocity).magnitude

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        # Calculate velocity difference
        delta_v = self.target_velocity - ship.velocity
        delta_v_mag = delta_v.magnitude

        # Check if we've matched velocity
        if delta_v_mag <= self.tolerance_ms:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                throttle=0.0,
                message=f"Velocity matched within {self.tolerance_ms} m/s"
            )

        direction = delta_v.normalized()

        self._elapsed_time += dt

        # Calculate progress based on initial delta-v
        initial_delta_v = self.estimate_delta_v_cost(ship) + self._delta_v_expended
        if initial_delta_v > 0:
            self._progress = min(100.0, (self._delta_v_expended / initial_delta_v) * 100)

        # Calculate required throttle (reduce as we approach target velocity)
        accel = ship.max_acceleration_ms2()
        if accel > 0:
            # Time to stop at current decel
            time_to_match = delta_v_mag / accel
            # Use full throttle if > 1 second away, proportionally less otherwise
            throttle = min(1.0, time_to_match)
        else:
            throttle = 1.0

        accel_actual = ship.max_acceleration_ms2() * throttle
        self._delta_v_expended += accel_actual * dt

        if self._elapsed_time >= self.max_duration:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message=f"Velocity match timed out, remaining delta-v: {delta_v_mag:.1f} m/s"
            )

        if ship.propellant_kg <= 0:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Velocity match failed - out of propellant"
            )

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=throttle,
            thrust_direction=direction,
            rotation_target=direction,
            message=f"Matching velocity, delta-v remaining: {delta_v_mag:.1f} m/s"
        )


@dataclass
class BrakingBurn(Maneuver):
    """
    Reduce speed to a target value.

    Burns retrograde to current velocity until target speed is reached.
    """
    target_speed_ms: float  # Target speed in m/s
    tolerance_ms: float = 5.0

    def __post_init__(self):
        self.name = "BrakingBurn"
        self.description = f"Reduce speed to {self.target_speed_ms} m/s"

    def estimate_completion_time(self, ship: ShipState) -> float:
        current_speed = ship.velocity.magnitude
        delta_v = max(0, current_speed - self.target_speed_ms)
        accel = ship.max_acceleration_ms2()
        if accel <= 0:
            return float('inf')
        return delta_v / accel

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        current_speed = ship.velocity.magnitude
        return max(0, current_speed - self.target_speed_ms)

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        current_speed = ship.velocity.magnitude

        if current_speed <= self.target_speed_ms + self.tolerance_ms:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                throttle=0.0,
                message=f"Target speed reached: {current_speed:.1f} m/s"
            )

        # Burn retrograde (opposite to velocity)
        if current_speed > 0:
            direction = -ship.velocity.normalized()
        else:
            self._status = ManeuverStatus.COMPLETED
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                message="Already at rest"
            )

        self._elapsed_time += dt

        # Progress based on speed reduction
        initial_delta_v = self.estimate_delta_v_cost(ship) + self._delta_v_expended
        if initial_delta_v > 0:
            self._progress = min(100.0, (self._delta_v_expended / initial_delta_v) * 100)

        # Calculate throttle - reduce as approaching target speed
        speed_excess = current_speed - self.target_speed_ms
        accel = ship.max_acceleration_ms2()
        if accel > 0:
            time_to_target = speed_excess / accel
            throttle = min(1.0, max(0.1, time_to_target))
        else:
            throttle = 1.0

        self._delta_v_expended += accel * throttle * dt

        if ship.propellant_kg <= 0:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Braking burn failed - out of propellant"
            )

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=throttle,
            thrust_direction=direction,
            rotation_target=direction,
            message=f"Braking, current speed: {current_speed:.1f} m/s"
        )


@dataclass
class AccelerationBurn(Maneuver):
    """
    Burn in a specified direction for a set duration.

    Simple directed burn for tactical maneuvers.
    """
    direction: Vector3D  # Burn direction (will be normalized)
    duration: float  # Burn duration in seconds
    throttle: float = 1.0

    def __post_init__(self):
        self.name = "AccelerationBurn"
        self.description = f"Burn for {self.duration}s at {self.throttle*100:.0f}% throttle"
        # Normalize direction
        if self.direction.magnitude > 0:
            object.__setattr__(self, 'direction', self.direction.normalized())

    def estimate_completion_time(self, ship: ShipState) -> float:
        return self.duration

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        accel = ship.max_acceleration_ms2() * self.throttle
        return accel * self.duration

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        self._elapsed_time += dt
        self._progress = min(100.0, (self._elapsed_time / self.duration) * 100)

        accel = ship.max_acceleration_ms2() * self.throttle
        self._delta_v_expended += accel * dt

        if self._elapsed_time >= self.duration:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                throttle=0.0,
                message="Acceleration burn complete"
            )

        if ship.propellant_kg <= 0:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Acceleration burn failed - out of propellant"
            )

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=self.throttle,
            thrust_direction=self.direction,
            rotation_target=self.direction,
            message=f"Accelerating, {self.duration - self._elapsed_time:.1f}s remaining"
        )


# =============================================================================
# ROTATION MANEUVERS
# =============================================================================

@dataclass
class RotateToFace(Maneuver):
    """
    Rotate to point nose at a target position.

    Essential for spinal weapon targeting.
    """
    target_position: Vector3D
    tolerance_deg: float = 1.0  # Angular tolerance in degrees

    def __post_init__(self):
        self.name = "RotateToFace"
        self.description = "Point nose at target"

    def _calculate_angle_to_target(self, ship: ShipState) -> float:
        """Calculate angle between current forward and target direction."""
        direction = self.target_position - ship.position
        if direction.magnitude < 1.0:
            return 0.0
        direction = direction.normalized()
        return math.degrees(ship.forward.angle_to(direction))

    def estimate_completion_time(self, ship: ShipState) -> float:
        angle = self._calculate_angle_to_target(ship)
        # Estimate angular acceleration from thrust vectoring
        torque = calculate_torque_from_thrust(ship, gimbal_pitch_deg=1.0, throttle=0.3)
        if torque.magnitude > 0:
            alpha = torque.magnitude / ship.moment_of_inertia_kg_m2
            return time_to_rotate(alpha, angle)
        return angle / 5.0  # Fallback: assume 5 deg/s rotation rate

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        # Rotation uses RCS, minimal delta-v cost
        # Estimate based on thrust vectoring fuel usage
        return self.estimate_completion_time(ship) * 0.1  # Rough estimate

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        angle = self._calculate_angle_to_target(ship)

        if angle <= self.tolerance_deg:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                message="Now facing target"
            )

        self._elapsed_time += dt

        # Calculate target direction
        direction = self.target_position - ship.position
        if direction.magnitude > 0:
            direction = direction.normalized()
        else:
            direction = ship.forward

        # Progress based on angle remaining
        initial_angle = 180.0  # Assume worst case
        self._progress = min(100.0, ((initial_angle - angle) / initial_angle) * 100)

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=0.0,  # No main engine thrust during rotation
            rotation_target=direction,
            message=f"Rotating to face target, {angle:.1f} degrees remaining"
        )


@dataclass
class RotateToBroadside(Maneuver):
    """
    Rotate to present lateral armor to a target.

    Used when presenting maximum armor profile or using broadside weapons.
    """
    target_position: Vector3D
    prefer_port: bool = True  # Prefer port (left) broadside if True
    tolerance_deg: float = 5.0

    def __post_init__(self):
        side = "port" if self.prefer_port else "starboard"
        self.name = "RotateToBroadside"
        self.description = f"Present {side} broadside to target"

    def _calculate_broadside_direction(self, ship: ShipState) -> Vector3D:
        """Calculate the direction the nose should point for broadside."""
        to_target = self.target_position - ship.position
        if to_target.magnitude < 1.0:
            return ship.forward
        to_target = to_target.normalized()

        # Broadside direction is perpendicular to target direction
        # Cross with up to get a perpendicular vector
        up = Vector3D.unit_z()
        perpendicular = to_target.cross(up)
        if perpendicular.magnitude < 0.01:
            perpendicular = to_target.cross(Vector3D.unit_x())
        perpendicular = perpendicular.normalized()

        if not self.prefer_port:
            perpendicular = -perpendicular

        return perpendicular

    def _calculate_angle_to_broadside(self, ship: ShipState) -> float:
        """Calculate angle from current forward to broadside orientation."""
        broadside_dir = self._calculate_broadside_direction(ship)
        return math.degrees(ship.forward.angle_to(broadside_dir))

    def estimate_completion_time(self, ship: ShipState) -> float:
        angle = self._calculate_angle_to_broadside(ship)
        return angle / 5.0  # Assume 5 deg/s rotation rate

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        return self.estimate_completion_time(ship) * 0.1

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        angle = self._calculate_angle_to_broadside(ship)

        if angle <= self.tolerance_deg:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                message="Broadside orientation achieved"
            )

        self._elapsed_time += dt
        broadside_dir = self._calculate_broadside_direction(ship)

        initial_angle = 90.0  # Assume average starting angle
        self._progress = min(100.0, ((initial_angle - angle) / initial_angle) * 100)

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=0.0,
            rotation_target=broadside_dir,
            message=f"Rotating to broadside, {angle:.1f} degrees remaining"
        )


@dataclass
class RotateToRetreat(Maneuver):
    """
    Rotate to point tail at target (for escape/retreat).

    Positions ship for maximum thrust away from threat.
    """
    target_position: Vector3D
    tolerance_deg: float = 5.0

    def __post_init__(self):
        self.name = "RotateToRetreat"
        self.description = "Orient for retreat (tail to target)"

    def _calculate_retreat_direction(self, ship: ShipState) -> Vector3D:
        """Calculate the direction nose should point for retreat."""
        away_from_target = ship.position - self.target_position
        if away_from_target.magnitude < 1.0:
            return ship.forward
        return away_from_target.normalized()

    def _calculate_angle_to_retreat(self, ship: ShipState) -> float:
        retreat_dir = self._calculate_retreat_direction(ship)
        return math.degrees(ship.forward.angle_to(retreat_dir))

    def estimate_completion_time(self, ship: ShipState) -> float:
        angle = self._calculate_angle_to_retreat(ship)
        return angle / 5.0

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        return self.estimate_completion_time(ship) * 0.1

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        angle = self._calculate_angle_to_retreat(ship)

        if angle <= self.tolerance_deg:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                message="Retreat orientation achieved"
            )

        self._elapsed_time += dt
        retreat_dir = self._calculate_retreat_direction(ship)

        initial_angle = 180.0
        self._progress = min(100.0, ((initial_angle - angle) / initial_angle) * 100)

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=0.0,
            rotation_target=retreat_dir,
            message=f"Rotating for retreat, {angle:.1f} degrees remaining"
        )


@dataclass
class RotateToAngle(Maneuver):
    """
    Rotate to a specific heading and pitch.

    For precise orientation control.
    """
    heading_deg: float  # Yaw angle in degrees (0 = +X axis)
    pitch_deg: float = 0.0  # Pitch angle in degrees (0 = level)
    tolerance_deg: float = 1.0

    def __post_init__(self):
        self.name = "RotateToAngle"
        self.description = f"Rotate to heading {self.heading_deg}deg, pitch {self.pitch_deg}deg"

    def _calculate_target_direction(self) -> Vector3D:
        """Calculate target forward direction from heading/pitch."""
        heading_rad = math.radians(self.heading_deg)
        pitch_rad = math.radians(self.pitch_deg)

        x = math.cos(heading_rad) * math.cos(pitch_rad)
        y = math.sin(heading_rad) * math.cos(pitch_rad)
        z = math.sin(pitch_rad)

        return Vector3D(x, y, z).normalized()

    def _calculate_angle_to_target(self, ship: ShipState) -> float:
        target_dir = self._calculate_target_direction()
        return math.degrees(ship.forward.angle_to(target_dir))

    def estimate_completion_time(self, ship: ShipState) -> float:
        angle = self._calculate_angle_to_target(ship)
        return angle / 5.0

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        return self.estimate_completion_time(ship) * 0.1

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        angle = self._calculate_angle_to_target(ship)

        if angle <= self.tolerance_deg:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                message=f"Reached heading {self.heading_deg}deg, pitch {self.pitch_deg}deg"
            )

        self._elapsed_time += dt
        target_dir = self._calculate_target_direction()

        initial_angle = 180.0
        self._progress = min(100.0, ((initial_angle - angle) / initial_angle) * 100)

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=0.0,
            rotation_target=target_dir,
            message=f"Rotating to specified angle, {angle:.1f} degrees remaining"
        )


# =============================================================================
# COMBINED MANEUVERS
# =============================================================================

@dataclass
class FlipAndBurn(Maneuver):
    """
    Rotate 180 degrees and burn (classic deceleration maneuver).

    Rotates to point nose opposite to current velocity direction,
    then burns to decelerate.
    """
    target_speed_ms: float = 0.0  # Target speed to reach
    throttle: float = 1.0
    max_duration: float = 300.0

    # Internal phase tracking
    _phase: str = field(default="rotate", init=False)  # "rotate" or "burn"

    def __post_init__(self):
        self.name = "FlipAndBurn"
        self.description = f"Flip and decelerate to {self.target_speed_ms} m/s"

    def estimate_completion_time(self, ship: ShipState) -> float:
        # Time to rotate 180 degrees
        rotation_time = 180.0 / 5.0  # Assume 5 deg/s

        # Time to decelerate
        current_speed = ship.velocity.magnitude
        delta_v = max(0, current_speed - self.target_speed_ms)
        accel = ship.max_acceleration_ms2() * self.throttle
        burn_time = delta_v / accel if accel > 0 else 0

        return rotation_time + burn_time

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        current_speed = ship.velocity.magnitude
        return max(0, current_speed - self.target_speed_ms)

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS
            self._phase = "rotate"

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        current_speed = ship.velocity.magnitude

        if self._phase == "rotate":
            # Calculate retrograde direction
            if current_speed > 0.1:
                retrograde = -ship.velocity.normalized()
            else:
                # Already nearly stopped
                self._status = ManeuverStatus.COMPLETED
                self._progress = 100.0
                return ManeuverResult(
                    status=self._status,
                    progress=100.0,
                    message="Already at target speed"
                )

            # Check if we're pointed retrograde
            angle = math.degrees(ship.forward.angle_to(retrograde))

            if angle <= 5.0:
                # Rotation complete, start burn
                self._phase = "burn"
                self._progress = 50.0  # Halfway done
            else:
                self._elapsed_time += dt
                self._progress = min(50.0, ((180.0 - angle) / 180.0) * 50)

                return ManeuverResult(
                    status=self._status,
                    progress=self._progress,
                    throttle=0.0,
                    rotation_target=retrograde,
                    message=f"Flip phase: {angle:.1f} degrees to retrograde"
                )

        # Burn phase
        if current_speed <= self.target_speed_ms + 5.0:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                throttle=0.0,
                message=f"Flip and burn complete, speed: {current_speed:.1f} m/s"
            )

        self._elapsed_time += dt

        # Retrograde direction
        retrograde = -ship.velocity.normalized() if current_speed > 0.1 else ship.forward

        # Track delta-v
        accel = ship.max_acceleration_ms2() * self.throttle
        self._delta_v_expended += accel * dt

        # Progress during burn phase (50-100%)
        initial_delta_v = self.estimate_delta_v_cost(ship) + self._delta_v_expended
        if initial_delta_v > 0:
            burn_progress = (self._delta_v_expended / initial_delta_v) * 50
            self._progress = min(100.0, 50.0 + burn_progress)

        if self._elapsed_time >= self.max_duration:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Flip and burn timed out"
            )

        if ship.propellant_kg <= 0:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Flip and burn failed - out of propellant"
            )

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=self.throttle,
            thrust_direction=retrograde,
            rotation_target=retrograde,
            message=f"Burn phase: speed {current_speed:.1f} m/s"
        )


@dataclass
class EvasiveJink(Maneuver):
    """
    Execute random lateral thrust bursts to evade targeting.

    Unpredictable movement pattern to make the ship harder to hit.
    """
    duration: float = 30.0  # Total evasion duration
    jink_interval: float = 3.0  # Seconds between direction changes
    throttle: float = 0.8

    _current_jink_direction: Vector3D = field(default_factory=Vector3D.zero, init=False)
    _time_since_jink: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.name = "EvasiveJink"
        self.description = f"Evasive maneuvers for {self.duration}s"
        self._pick_new_jink_direction()

    def _pick_new_jink_direction(self) -> None:
        """Pick a random perpendicular direction for the next jink."""
        # Random angles
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(-math.pi/4, math.pi/4)  # Limit vertical component

        x = math.cos(theta) * math.cos(phi)
        y = math.sin(theta) * math.cos(phi)
        z = math.sin(phi)

        self._current_jink_direction = Vector3D(x, y, z).normalized()

    def estimate_completion_time(self, ship: ShipState) -> float:
        return self.duration

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        accel = ship.max_acceleration_ms2() * self.throttle
        return accel * self.duration

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS
            self._pick_new_jink_direction()

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        self._elapsed_time += dt
        self._time_since_jink += dt

        # Time to change direction?
        if self._time_since_jink >= self.jink_interval:
            self._pick_new_jink_direction()
            self._time_since_jink = 0.0

        self._progress = min(100.0, (self._elapsed_time / self.duration) * 100)

        accel = ship.max_acceleration_ms2() * self.throttle
        self._delta_v_expended += accel * dt

        if self._elapsed_time >= self.duration:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                throttle=0.0,
                message="Evasive maneuvers complete"
            )

        if ship.propellant_kg <= 0:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Evasive maneuvers failed - out of propellant"
            )

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=self.throttle,
            thrust_direction=self._current_jink_direction,
            message=f"Jinking, {self.duration - self._elapsed_time:.1f}s remaining"
        )


@dataclass
class SpiralApproach(Maneuver):
    """
    Corkscrew approach toward a target.

    Combines forward movement with lateral oscillation to make
    approach vector harder to predict.
    """
    target_position: Vector3D
    approach_throttle: float = 0.6
    spiral_rate_deg_per_s: float = 30.0  # Rate of spiral rotation
    spiral_amplitude: float = 0.3  # Lateral thrust as fraction of forward
    max_duration: float = 120.0

    _spiral_phase: float = field(default=0.0, init=False)  # Current phase in radians

    def __post_init__(self):
        self.name = "SpiralApproach"
        self.description = "Corkscrew approach toward target"

    def estimate_completion_time(self, ship: ShipState) -> float:
        distance = (self.target_position - ship.position).magnitude
        # Rough estimate assuming net approach at reduced speed
        effective_speed = ship.max_acceleration_ms2() * self.approach_throttle * 0.7  # Account for spiral
        if effective_speed > 0:
            return min(distance / effective_speed, self.max_duration)
        return self.max_duration

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        # Combined forward and lateral thrust
        total_thrust = math.sqrt(1 + self.spiral_amplitude**2) * self.approach_throttle
        accel = ship.max_acceleration_ms2() * total_thrust
        return accel * self.estimate_completion_time(ship)

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        # Calculate direction to target
        to_target = self.target_position - ship.position
        distance = to_target.magnitude

        if distance < 1000:  # Within 1 km
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                throttle=0.0,
                message="Spiral approach complete - target reached"
            )

        forward = to_target.normalized()

        # Calculate perpendicular vectors for spiral
        up = Vector3D.unit_z()
        right = forward.cross(up)
        if right.magnitude < 0.01:
            right = forward.cross(Vector3D.unit_x())
        right = right.normalized()
        actual_up = right.cross(forward).normalized()

        # Update spiral phase
        self._spiral_phase += math.radians(self.spiral_rate_deg_per_s) * dt

        # Calculate lateral offset
        lateral_x = math.cos(self._spiral_phase) * self.spiral_amplitude
        lateral_y = math.sin(self._spiral_phase) * self.spiral_amplitude

        # Combined thrust direction
        thrust_dir = (forward + right * lateral_x + actual_up * lateral_y).normalized()

        self._elapsed_time += dt
        self._progress = min(100.0, (self._elapsed_time / self.max_duration) * 100)

        accel = ship.max_acceleration_ms2() * self.approach_throttle
        self._delta_v_expended += accel * dt

        if self._elapsed_time >= self.max_duration:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Spiral approach timed out"
            )

        if ship.propellant_kg <= 0:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Spiral approach failed - out of propellant"
            )

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=self.approach_throttle,
            thrust_direction=thrust_dir,
            rotation_target=forward,
            message=f"Spiraling toward target, distance: {distance/1000:.1f} km"
        )


@dataclass
class BreakTurn(Maneuver):
    """
    Hard turn to change engagement angle.

    Rapid orientation change followed by thrust to alter trajectory.
    """
    turn_angle_deg: float = 90.0  # Angle to turn
    turn_direction: str = "right"  # "left", "right", "up", "down"
    burn_duration: float = 10.0  # Thrust duration after turn
    throttle: float = 1.0

    _phase: str = field(default="turn", init=False)
    _target_direction: Optional[Vector3D] = field(default=None, init=False)

    def __post_init__(self):
        self.name = "BreakTurn"
        self.description = f"Break {self.turn_direction} {self.turn_angle_deg}deg and burn"

    def _calculate_target_direction(self, ship: ShipState) -> Vector3D:
        """Calculate the direction after the turn."""
        if self._target_direction is not None:
            return self._target_direction

        angle_rad = math.radians(self.turn_angle_deg)

        if self.turn_direction == "right":
            axis = ship.up
            sign = -1
        elif self.turn_direction == "left":
            axis = ship.up
            sign = 1
        elif self.turn_direction == "up":
            axis = ship.right
            sign = 1
        else:  # down
            axis = ship.right
            sign = -1

        self._target_direction = ship.forward.rotate_around_axis(axis, sign * angle_rad)
        return self._target_direction

    def estimate_completion_time(self, ship: ShipState) -> float:
        rotation_time = self.turn_angle_deg / 10.0  # Assume 10 deg/s for hard turn
        return rotation_time + self.burn_duration

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        accel = ship.max_acceleration_ms2() * self.throttle
        return accel * self.burn_duration

    def execute_step(self, ship: ShipState, dt: float) -> ManeuverResult:
        if self._status == ManeuverStatus.PENDING:
            self._status = ManeuverStatus.IN_PROGRESS
            self._phase = "turn"
            self._target_direction = None

        if self._status != ManeuverStatus.IN_PROGRESS:
            return ManeuverResult(status=self._status, progress=self._progress)

        target_dir = self._calculate_target_direction(ship)

        if self._phase == "turn":
            angle = math.degrees(ship.forward.angle_to(target_dir))

            if angle <= 5.0:
                self._phase = "burn"
                self._progress = 50.0
            else:
                self._elapsed_time += dt
                turn_time = self.turn_angle_deg / 10.0
                self._progress = min(50.0, (self._elapsed_time / turn_time) * 50)

                return ManeuverResult(
                    status=self._status,
                    progress=self._progress,
                    throttle=0.0,
                    rotation_target=target_dir,
                    message=f"Break turn: {angle:.1f} degrees remaining"
                )

        # Burn phase
        burn_elapsed = self._elapsed_time - (self.turn_angle_deg / 10.0)
        if burn_elapsed < 0:
            burn_elapsed = 0

        if burn_elapsed >= self.burn_duration:
            self._status = ManeuverStatus.COMPLETED
            self._progress = 100.0
            return ManeuverResult(
                status=self._status,
                progress=100.0,
                throttle=0.0,
                message="Break turn complete"
            )

        self._elapsed_time += dt

        accel = ship.max_acceleration_ms2() * self.throttle
        self._delta_v_expended += accel * dt

        self._progress = min(100.0, 50.0 + (burn_elapsed / self.burn_duration) * 50)

        if ship.propellant_kg <= 0:
            self._status = ManeuverStatus.FAILED
            return ManeuverResult(
                status=self._status,
                progress=self._progress,
                message="Break turn failed - out of propellant"
            )

        return ManeuverResult(
            status=self._status,
            progress=self._progress,
            throttle=self.throttle,
            thrust_direction=target_dir,
            rotation_target=target_dir,
            message=f"Break burn: {self.burn_duration - burn_elapsed:.1f}s remaining"
        )


# =============================================================================
# MANEUVER EXECUTOR
# =============================================================================

@dataclass
class ManeuverExecutor:
    """
    Executes maneuvers step by step.

    Takes a ship state and maneuver command, calculates thrust/rotation
    needed each timestep, tracks progress toward completion.
    """

    current_maneuver: Optional[Maneuver] = None

    def set_maneuver(self, maneuver: Maneuver) -> None:
        """
        Set a new maneuver to execute.

        Aborts any current maneuver in progress.
        """
        if self.current_maneuver and not self.current_maneuver.is_complete:
            self.current_maneuver.abort()

        self.current_maneuver = maneuver
        maneuver.reset()

    def abort_current(self) -> bool:
        """
        Abort the current maneuver.

        Returns:
            True if a maneuver was aborted, False if none was active
        """
        if self.current_maneuver and not self.current_maneuver.is_complete:
            self.current_maneuver.abort()
            return True
        return False

    def update(self, ship: ShipState, dt: float) -> ManeuverResult:
        """
        Execute one timestep of the current maneuver.

        Args:
            ship: Current ship state
            dt: Time step in seconds

        Returns:
            ManeuverResult with control outputs, or idle result if no maneuver
        """
        if self.current_maneuver is None:
            return ManeuverResult(
                status=ManeuverStatus.COMPLETED,
                progress=100.0,
                message="No active maneuver"
            )

        result = self.current_maneuver.execute_step(ship, dt)

        return result

    def estimate_delta_v_cost(self, ship: ShipState) -> float:
        """
        Estimate delta-v cost before execution.

        Args:
            ship: Current ship state

        Returns:
            Estimated delta-v in m/s, or 0 if no maneuver set
        """
        if self.current_maneuver is None:
            return 0.0
        return self.current_maneuver.estimate_delta_v_cost(ship)

    def estimate_completion_time(self, ship: ShipState) -> float:
        """
        Estimate time to complete the maneuver.

        Args:
            ship: Current ship state

        Returns:
            Estimated time in seconds, or 0 if no maneuver set
        """
        if self.current_maneuver is None:
            return 0.0
        return self.current_maneuver.estimate_completion_time(ship)

    @property
    def progress(self) -> float:
        """Current maneuver progress (0-100%)."""
        if self.current_maneuver is None:
            return 100.0
        return self.current_maneuver.progress

    @property
    def is_idle(self) -> bool:
        """Check if executor has no active maneuver."""
        return (self.current_maneuver is None or
                self.current_maneuver.is_complete)


# =============================================================================
# MANEUVER PLANNER
# =============================================================================

@dataclass
class ManeuverPlanner:
    """
    Suggests optimal maneuvers for tactical situations.

    Provides methods to calculate intercept courses and plan
    fuel-efficient approaches.
    """

    @staticmethod
    def calculate_intercept_time(
        pursuer: ShipState,
        target_position: Vector3D,
        target_velocity: Vector3D
    ) -> float:
        """
        Calculate time to intercept a moving target.

        Uses constant acceleration assumption.

        Args:
            pursuer: Pursuing ship's state
            target_position: Target's current position
            target_velocity: Target's velocity

        Returns:
            Estimated intercept time in seconds, or inf if impossible
        """
        # Relative position and velocity
        rel_pos = target_position - pursuer.position
        rel_vel = target_velocity - pursuer.velocity

        distance = rel_pos.magnitude
        closing_rate = -rel_pos.normalized().dot(rel_vel)

        if closing_rate <= 0:
            # Not closing - need to accelerate
            accel = pursuer.max_acceleration_ms2()
            if accel <= 0:
                return float('inf')

            # Time to close using kinematic equation
            # d = v*t + 0.5*a*t^2
            # Solve for t when d = 0
            # Using quadratic formula
            v = -closing_rate
            a = accel
            d = distance

            discriminant = v**2 + 2*a*d
            if discriminant < 0:
                return float('inf')

            t = (-v + math.sqrt(discriminant)) / a
            return t
        else:
            # Already closing
            return distance / closing_rate

    @staticmethod
    def plan_intercept_burn(
        ship: ShipState,
        target_position: Vector3D,
        target_velocity: Vector3D,
        desired_approach_speed: float = 100.0
    ) -> Optional[Maneuver]:
        """
        Plan an intercept burn to reach a moving target.

        Args:
            ship: Ship's current state
            target_position: Target's current position
            target_velocity: Target's velocity
            desired_approach_speed: Desired relative speed at intercept (m/s)

        Returns:
            Appropriate maneuver, or None if intercept is impractical
        """
        rel_pos = target_position - ship.position
        rel_vel = target_velocity - ship.velocity
        distance = rel_pos.magnitude

        if distance < 1000:  # Already close
            return MatchVelocity(target_velocity=target_velocity)

        # Calculate lead position
        intercept_time = ManeuverPlanner.calculate_intercept_time(
            ship, target_position, target_velocity
        )

        if intercept_time == float('inf') or intercept_time > 3600:
            # Intercept impractical - just burn toward target
            return BurnToward(target_position=target_position, max_duration=60.0)

        # Lead the target
        lead_position = target_position + target_velocity * intercept_time

        # Use spiral approach for unpredictability
        return SpiralApproach(
            target_position=lead_position,
            max_duration=intercept_time * 1.2
        )

    @staticmethod
    def suggest_evasion(
        ship: ShipState,
        threat_position: Vector3D,
        threat_velocity: Vector3D
    ) -> Maneuver:
        """
        Suggest an evasive maneuver against an incoming threat.

        Args:
            ship: Ship's current state
            threat_position: Threat's position
            threat_velocity: Threat's velocity

        Returns:
            Appropriate evasive maneuver
        """
        # Calculate threat approach direction
        rel_pos = threat_position - ship.position
        distance = rel_pos.magnitude

        if distance < 1000:
            # Very close - immediate jink
            return EvasiveJink(duration=10.0, jink_interval=1.0, throttle=1.0)

        # Calculate time to impact
        closing_rate = -rel_pos.normalized().dot(threat_velocity - ship.velocity)

        if closing_rate > 0:
            time_to_impact = distance / closing_rate
        else:
            # Not closing - minimal threat
            return EvasiveJink(duration=5.0, jink_interval=3.0, throttle=0.5)

        if time_to_impact < 10:
            # Critical - hard evasive
            return EvasiveJink(duration=15.0, jink_interval=1.0, throttle=1.0)
        elif time_to_impact < 30:
            # Urgent - break turn
            return BreakTurn(turn_angle_deg=90.0, burn_duration=10.0, throttle=1.0)
        else:
            # Time for calculated evasion
            # Turn perpendicular to threat approach
            threat_dir = threat_velocity.normalized() if threat_velocity.magnitude > 0 else rel_pos.normalized()
            perp = threat_dir.cross(Vector3D.unit_z())
            if perp.magnitude < 0.01:
                perp = threat_dir.cross(Vector3D.unit_x())

            return BurnAway(
                target_position=threat_position,
                throttle=0.8,
                max_duration=20.0
            )

    @staticmethod
    def suggest_engagement_approach(
        ship: ShipState,
        target: ShipState,
        preferred_range: float = 100_000  # 100 km
    ) -> Maneuver:
        """
        Suggest maneuver to reach optimal engagement range.

        Args:
            ship: Ship's current state
            target: Target ship's state
            preferred_range: Desired engagement range in meters

        Returns:
            Appropriate approach/positioning maneuver
        """
        rel_pos = target.position - ship.position
        distance = rel_pos.magnitude

        if distance < preferred_range * 0.8:
            # Too close - back off
            return BurnAway(
                target_position=target.position,
                throttle=0.5,
                max_duration=15.0
            )
        elif distance > preferred_range * 1.5:
            # Too far - close in with evasive pattern
            return SpiralApproach(
                target_position=target.position,
                approach_throttle=0.7,
                max_duration=120.0
            )
        else:
            # Good range - match velocity
            return MatchVelocity(target_velocity=target.velocity)

    @staticmethod
    def calculate_fuel_efficient_transfer(
        ship: ShipState,
        target_position: Vector3D,
        target_velocity: Vector3D
    ) -> list[Maneuver]:
        """
        Plan a fuel-efficient transfer to match position/velocity.

        Uses Hohmann-like approach with two burns.

        Args:
            ship: Ship's current state
            target_position: Desired final position
            target_velocity: Desired final velocity

        Returns:
            List of maneuvers for the transfer
        """
        maneuvers = []

        rel_pos = target_position - ship.position
        distance = rel_pos.magnitude

        if distance < 1000:
            # Already there - just match velocity
            return [MatchVelocity(target_velocity=target_velocity)]

        # First burn: accelerate toward target
        maneuvers.append(BurnToward(
            target_position=target_position,
            throttle=1.0,
            max_duration=30.0
        ))

        # Coast phase would happen naturally

        # Second burn: match velocity at arrival
        maneuvers.append(MatchVelocity(
            target_velocity=target_velocity,
            max_duration=120.0
        ))

        return maneuvers


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MANEUVER SYSTEM - SELF TEST")
    print("=" * 70)

    # Create a test ship
    from physics import create_ship_state_from_specs

    ship = create_ship_state_from_specs(
        wet_mass_tons=1990,
        dry_mass_tons=1895,
        length_m=65,
        position=Vector3D.zero(),
        velocity=Vector3D(1000, 0, 0),  # Moving at 1 km/s in +X
        forward=Vector3D.unit_x()
    )

    print("\nTest Ship:")
    print(f"  Position: {ship.position}")
    print(f"  Velocity: {ship.velocity}")
    print(f"  Max accel: {ship.max_acceleration_g():.2f} g")
    print(f"  Delta-v: {ship.remaining_delta_v_kps():.1f} km/s")

    # Test BurnToward
    print("\n--- BurnToward Test ---")
    target = Vector3D(100_000, 0, 0)
    maneuver = BurnToward(target_position=target, throttle=0.5, max_duration=10.0)
    print(f"Maneuver: {maneuver.name}")
    print(f"  Estimated time: {maneuver.estimate_completion_time(ship):.1f}s")
    print(f"  Estimated delta-v: {maneuver.estimate_delta_v_cost(ship):.1f} m/s")

    executor = ManeuverExecutor()
    executor.set_maneuver(maneuver)

    result = executor.update(ship, dt=1.0)
    print(f"  After 1s: progress={result.progress:.1f}%, throttle={result.throttle}")

    # Test MatchVelocity
    print("\n--- MatchVelocity Test ---")
    target_vel = Vector3D(500, 0, 0)
    maneuver = MatchVelocity(target_velocity=target_vel)
    print(f"Maneuver: {maneuver.name}")
    print(f"  Target velocity: {target_vel}")
    print(f"  Estimated delta-v: {maneuver.estimate_delta_v_cost(ship):.1f} m/s")

    # Test FlipAndBurn
    print("\n--- FlipAndBurn Test ---")
    maneuver = FlipAndBurn(target_speed_ms=100.0, throttle=1.0)
    print(f"Maneuver: {maneuver.name}")
    print(f"  Current speed: {ship.velocity.magnitude:.1f} m/s")
    print(f"  Target speed: {maneuver.target_speed_ms} m/s")
    print(f"  Estimated delta-v: {maneuver.estimate_delta_v_cost(ship):.1f} m/s")

    # Test EvasiveJink
    print("\n--- EvasiveJink Test ---")
    maneuver = EvasiveJink(duration=10.0, jink_interval=2.0)
    print(f"Maneuver: {maneuver.name}")
    print(f"  Duration: {maneuver.duration}s")
    print(f"  Jink interval: {maneuver.jink_interval}s")

    # Test ManeuverPlanner
    print("\n--- ManeuverPlanner Test ---")
    planner = ManeuverPlanner()

    # Test intercept calculation
    target_ship = create_ship_state_from_specs(
        wet_mass_tons=1990,
        dry_mass_tons=1895,
        length_m=65,
        position=Vector3D(50_000, 0, 0),
        velocity=Vector3D(0, 500, 0),
        forward=Vector3D.unit_y()
    )

    intercept_time = planner.calculate_intercept_time(
        ship,
        target_ship.position,
        target_ship.velocity
    )
    print(f"Intercept time to target: {intercept_time:.1f}s")

    # Test evasion suggestion
    threat_pos = Vector3D(10_000, 0, 0)
    threat_vel = Vector3D(-1000, 0, 0)  # Coming toward us
    evasion = planner.suggest_evasion(ship, threat_pos, threat_vel)
    print(f"Suggested evasion: {evasion.name} - {evasion.description}")

    # Test engagement approach
    approach = planner.suggest_engagement_approach(ship, target_ship, preferred_range=50_000)
    print(f"Suggested approach: {approach.name} - {approach.description}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
