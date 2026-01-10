#!/usr/bin/env python3
"""
Fire Control System for AI Commanders Space Battle Simulator.

This module implements:
- Hit probability calculation based on geometry, distance, and relative motion
- Command schemas for helm and weapons orders
- Weapons officer AI for optimal fire timing
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple

try:
    from .physics import Vector3D
    from .geometry import ShipGeometry
    from .combat import HitLocation
except ImportError:
    from physics import Vector3D
    from geometry import ShipGeometry
    from combat import HitLocation


# =============================================================================
# HIT PROBABILITY CALCULATION
# =============================================================================

@dataclass
class FiringSolution:
    """
    Complete firing solution for a weapon against a target.

    Attributes:
        can_fire: Whether the weapon can engage this target.
        hit_probability: Estimated probability of hitting (0.0 to 1.0).
        time_of_flight_s: Estimated projectile flight time in seconds.
        predicted_range_km: Distance when projectile reaches target.
        target_aspect: Which face of target is exposed.
        recommendation: Human-readable firing recommendation.
    """
    can_fire: bool
    hit_probability: float
    time_of_flight_s: float
    predicted_range_km: float
    target_aspect: HitLocation
    recommendation: str

    @property
    def is_optimal(self) -> bool:
        """Returns True if hit probability >= 30%."""
        return self.hit_probability >= 0.30

    @property
    def is_good(self) -> bool:
        """Returns True if hit probability >= 50%."""
        return self.hit_probability >= 0.50


def calculate_hit_probability(
    shooter_position: Vector3D,
    shooter_velocity: Vector3D,
    target_position: Vector3D,
    target_velocity: Vector3D,
    target_geometry: ShipGeometry,
    target_forward: Vector3D,
    muzzle_velocity_kps: float,
    weapon_base_accuracy: float = 0.9,
    target_is_evading: bool = False
) -> FiringSolution:
    """
    Calculate the probability of hitting a target.

    Uses an angular size model where hit probability depends on:
    - Target's apparent angular size from shooter's perspective
    - Projectile flight time (uncertainty grows with time)
    - Crossing angle (perpendicular shots are harder)
    - Target evasion state

    Args:
        shooter_position: Shooter's current position (meters).
        shooter_velocity: Shooter's velocity vector (m/s).
        target_position: Target's current position (meters).
        target_velocity: Target's velocity vector (m/s).
        target_geometry: Target ship's geometry for cross-section.
        target_forward: Target's facing direction (normalized).
        muzzle_velocity_kps: Weapon's muzzle velocity in km/s.
        weapon_base_accuracy: Weapon's base accuracy factor (0.0-1.0).
        target_is_evading: Whether target is actively evading.

    Returns:
        FiringSolution with hit probability and recommendations.
    """
    # Calculate relative position and velocity
    rel_pos = target_position - shooter_position
    distance_m = rel_pos.magnitude
    distance_km = distance_m / 1000

    if distance_m < 1.0:
        # Point blank - guaranteed hit
        return FiringSolution(
            can_fire=True,
            hit_probability=1.0,
            time_of_flight_s=0.0,
            predicted_range_km=0.0,
            target_aspect=HitLocation.NOSE,
            recommendation="POINT BLANK - FIRE"
        )

    direction_to_target = rel_pos.normalized()
    rel_vel = target_velocity - shooter_velocity

    # Calculate closing rate (positive = closing)
    closing_rate_mps = -rel_vel.dot(direction_to_target)
    closing_rate_kps = closing_rate_mps / 1000

    # Projectile velocity (relative to shooter, in direction of target)
    muzzle_velocity_mps = muzzle_velocity_kps * 1000

    # Time of flight estimation
    # Projectile speed toward target = muzzle_velocity + closing_rate
    effective_projectile_speed = muzzle_velocity_mps + closing_rate_mps

    if effective_projectile_speed <= 0:
        # Projectile can't catch target
        return FiringSolution(
            can_fire=False,
            hit_probability=0.0,
            time_of_flight_s=float('inf'),
            predicted_range_km=distance_km,
            target_aspect=HitLocation.TAIL,
            recommendation="TARGET OUTRUNNING PROJECTILE - DO NOT FIRE"
        )

    time_of_flight_s = distance_m / effective_projectile_speed

    # Predict target position at impact
    predicted_target_pos = target_position + target_velocity * time_of_flight_s
    predicted_shooter_pos = shooter_position + shooter_velocity * time_of_flight_s
    predicted_distance = (predicted_target_pos - predicted_shooter_pos).magnitude
    predicted_range_km = predicted_distance / 1000

    # Determine which face of target is exposed
    # Based on angle between projectile approach and target forward
    projectile_approach = (shooter_position - target_position).normalized()
    approach_angle = math.degrees(target_forward.angle_to(projectile_approach))

    if approach_angle < 30:
        target_aspect = HitLocation.NOSE
        cross_section_m2 = target_geometry.nose_cross_section_m2
    elif approach_angle > 150:
        target_aspect = HitLocation.TAIL
        cross_section_m2 = target_geometry.tail_cross_section_m2
    else:
        target_aspect = HitLocation.LATERAL
        cross_section_m2 = target_geometry.lateral_cross_section_m2

    # Calculate target angular size (steradians approximation)
    # Angular size = cross_section / distance^2
    target_radius_m = math.sqrt(cross_section_m2 / math.pi)  # Effective radius
    angular_size_rad = target_radius_m / predicted_distance if predicted_distance > 0 else 1.0

    # Base hit probability from angular size
    # At 1 km with 60m effective radius, angular size = 0.06 rad = 3.4°
    # We scale so that reasonable combat ranges give reasonable probabilities
    # Reference: at 100km, a 60m radius target has angular size 0.0006 rad
    # We want ~50% hit chance at "optimal" range

    # Probability scales with angular size, normalized to give good gameplay
    # Using sigmoid-like scaling: prob = 1 - exp(-k * angular_size)
    # k chosen so that at 10km range with 60m target, prob ≈ 0.5
    angular_factor = 10000  # Tuning constant
    base_prob = 1.0 - math.exp(-angular_factor * angular_size_rad)

    # Apply weapon base accuracy
    prob = base_prob * weapon_base_accuracy

    # Flight time penalty - uncertainty grows with time
    # Each 10 seconds of flight time reduces probability by 10%
    flight_time_penalty = max(0.5, 1.0 - (time_of_flight_s / 100))
    prob *= flight_time_penalty

    # Crossing angle penalty - perpendicular shots are harder
    # Closing or receding shots are easier (target moving toward/away from line of fire)
    lateral_velocity = rel_vel - direction_to_target * rel_vel.dot(direction_to_target)
    lateral_speed = lateral_velocity.magnitude
    # Penalty based on how much lateral motion there is vs range
    if predicted_distance > 0:
        lateral_angle = math.atan2(lateral_speed * time_of_flight_s, predicted_distance)
        crossing_penalty = max(0.3, 1.0 - (lateral_angle / math.pi) * 2)
        prob *= crossing_penalty

    # Evasion penalty - actively maneuvering targets are harder to hit
    if target_is_evading:
        prob *= 0.6  # 40% reduction for evading targets

    # Clamp to valid range
    prob = max(0.01, min(0.99, prob))

    # Generate recommendation
    if prob >= 0.7:
        recommendation = "EXCELLENT SHOT - FIRE AT WILL"
    elif prob >= 0.5:
        recommendation = "GOOD SHOT - RECOMMEND FIRE"
    elif prob >= 0.3:
        recommendation = "MARGINAL - FIRE IF AMMO AVAILABLE"
    elif prob >= 0.1:
        recommendation = "POOR - CLOSE RANGE OR WAIT"
    else:
        recommendation = "VERY POOR - HOLD FIRE"

    return FiringSolution(
        can_fire=True,
        hit_probability=prob,
        time_of_flight_s=time_of_flight_s,
        predicted_range_km=predicted_range_km,
        target_aspect=target_aspect,
        recommendation=recommendation
    )


def calculate_engagement_envelope(
    weapon_range_km: float,
    muzzle_velocity_kps: float,
    target_size_m: float = 60.0
) -> dict:
    """
    Calculate engagement envelope for a weapon.

    Returns range brackets with expected hit probabilities.

    Args:
        weapon_range_km: Maximum weapon range in km.
        muzzle_velocity_kps: Muzzle velocity in km/s.
        target_size_m: Typical target radius in meters.

    Returns:
        Dictionary with range brackets and probabilities.
    """
    envelope = {
        'optimal_range_km': 0.0,
        'good_range_km': 0.0,
        'max_range_km': weapon_range_km,
        'brackets': []
    }

    # Calculate probability at various ranges
    for range_km in [10, 25, 50, 100, 200, 500, weapon_range_km]:
        if range_km > weapon_range_km:
            continue

        distance_m = range_km * 1000
        angular_size = target_size_m / distance_m
        prob = 1.0 - math.exp(-10000 * angular_size)
        prob = max(0.01, min(0.99, prob))

        envelope['brackets'].append({
            'range_km': range_km,
            'hit_probability': prob,
            'time_of_flight_s': range_km / muzzle_velocity_kps
        })

        if prob >= 0.5 and envelope['good_range_km'] == 0:
            envelope['good_range_km'] = range_km
        if prob >= 0.7 and envelope['optimal_range_km'] == 0:
            envelope['optimal_range_km'] = range_km

    return envelope


# =============================================================================
# COMMAND SCHEMA
# =============================================================================

class HelmCommand(Enum):
    """Commands for ship navigation/maneuvering."""
    INTERCEPT = auto()      # Close with target, lead for intercept
    FACE_TARGET = auto()    # Point nose directly at target (no lead)
    EVADE = auto()          # Random evasive jinking pattern
    BRAKE = auto()          # Flip and burn to decelerate
    HOLD_COURSE = auto()    # Maintain current heading and velocity
    ROTATE_TO = auto()      # Rotate to specific heading
    PURSUIT = auto()        # Chase target (nose toward target, full thrust)


class WeaponsCommand(Enum):
    """Commands for weapons systems."""
    FIRE_IMMEDIATE = auto()     # Fire as soon as weapon ready
    FIRE_WHEN_OPTIMAL = auto()  # Fire when hit_probability >= threshold
    FIRE_AT_RANGE = auto()      # Fire when target enters specific range
    HOLD_FIRE = auto()          # Don't fire, conserve ammunition
    FREE_FIRE = auto()          # Fire at any valid target when ready


class TacticalPosture(Enum):
    """Overall tactical stance."""
    AGGRESSIVE = auto()     # Close range, maximize damage output
    BALANCED = auto()       # Standard engagement
    DEFENSIVE = auto()      # Prioritize evasion and survival
    WITHDRAW = auto()       # Fighting retreat, break contact


@dataclass
class HelmOrder:
    """
    A navigation order for the helm.

    Attributes:
        command: The type of maneuver to execute.
        target_id: Target ship ID (for INTERCEPT, FACE_TARGET, PURSUIT).
        direction: Specific direction vector (for ROTATE_TO).
        throttle: Throttle setting 0.0-1.0.
        duration: How long to execute (0 = until changed).
        evasion_intensity: For EVADE, how aggressive (0.0-1.0).
    """
    command: HelmCommand
    target_id: Optional[str] = None
    direction: Optional[Vector3D] = None
    throttle: float = 1.0
    duration: float = 0.0
    evasion_intensity: float = 0.5


@dataclass
class WeaponsOrder:
    """
    A fire control order for a weapon.

    Attributes:
        command: The fire control mode.
        weapon_slot: Which weapon this order applies to ('all' for all weapons).
        target_id: Target ship ID.
        min_hit_probability: For FIRE_WHEN_OPTIMAL, minimum probability to fire.
        max_range_km: For FIRE_AT_RANGE, maximum engagement range.
        conserve_ammo: If True, be more conservative with ammunition.
    """
    command: WeaponsCommand
    weapon_slot: str = "all"
    target_id: Optional[str] = None
    min_hit_probability: float = 0.3
    max_range_km: float = 0.0
    conserve_ammo: bool = False


@dataclass
class TacticalOrder:
    """
    High-level tactical order combining helm and weapons.

    Attributes:
        posture: Overall tactical stance.
        primary_target: Main target to engage.
        helm_order: Navigation orders.
        weapons_orders: List of weapons orders.
        priority: Order priority (higher = more important).
    """
    posture: TacticalPosture = TacticalPosture.BALANCED
    primary_target: Optional[str] = None
    helm_order: Optional[HelmOrder] = None
    weapons_orders: List[WeaponsOrder] = field(default_factory=list)
    priority: int = 0


# =============================================================================
# WEAPONS OFFICER AI
# =============================================================================

@dataclass
class WeaponStatus:
    """Status of a single weapon for the weapons officer."""
    slot: str
    weapon_type: str
    ammo_remaining: int
    cooldown_remaining: float
    is_ready: bool
    last_solution: Optional[FiringSolution] = None


class WeaponsOfficer:
    """
    AI-controlled weapons officer that manages fire control.

    Decides when to fire based on:
    - Hit probability calculations
    - Ammunition conservation
    - Target priority
    - Fire control orders
    """

    def __init__(
        self,
        min_probability_threshold: float = 0.3,
        conserve_ammo_threshold: float = 0.5,
        max_ammo_reserve_percent: float = 0.2
    ):
        """
        Initialize weapons officer.

        Args:
            min_probability_threshold: Don't fire below this probability.
            conserve_ammo_threshold: When ammo below this %, be more selective.
            max_ammo_reserve_percent: Keep this much ammo in reserve.
        """
        self.min_probability = min_probability_threshold
        self.conserve_threshold = conserve_ammo_threshold
        self.ammo_reserve = max_ammo_reserve_percent

        # Current orders per weapon slot
        self.orders: dict[str, WeaponsOrder] = {}

        # Last firing solutions per target
        self.solutions: dict[str, dict[str, FiringSolution]] = {}  # target_id -> {weapon_slot -> solution}

    def set_order(self, order: WeaponsOrder) -> None:
        """Set fire control order for a weapon slot."""
        if order.weapon_slot == "all":
            # Apply to all weapons - handled at evaluation time
            self.orders["_default"] = order
        else:
            self.orders[order.weapon_slot] = order

    def get_order(self, weapon_slot: str) -> Optional[WeaponsOrder]:
        """Get current order for a weapon slot."""
        return self.orders.get(weapon_slot, self.orders.get("_default"))

    def evaluate_shot(
        self,
        weapon_slot: str,
        weapon_ammo: int,
        weapon_magazine: int,
        solution: FiringSolution,
        order: Optional[WeaponsOrder] = None
    ) -> Tuple[bool, str]:
        """
        Evaluate whether to take a shot.

        Args:
            weapon_slot: Weapon slot identifier.
            weapon_ammo: Current ammunition count.
            weapon_magazine: Maximum ammunition capacity.
            solution: Firing solution for this shot.
            order: Current fire control order.

        Returns:
            Tuple of (should_fire, reason).
        """
        if order is None:
            order = self.get_order(weapon_slot)

        if order is None:
            # No orders - use default FIRE_WHEN_OPTIMAL
            order = WeaponsOrder(
                command=WeaponsCommand.FIRE_WHEN_OPTIMAL,
                min_hit_probability=self.min_probability
            )

        if not solution.can_fire:
            return False, solution.recommendation

        # Check order type
        if order.command == WeaponsCommand.HOLD_FIRE:
            return False, "HOLD FIRE order active"

        if order.command == WeaponsCommand.FIRE_IMMEDIATE:
            return True, "FIRE IMMEDIATE - engaging"

        if order.command == WeaponsCommand.FREE_FIRE:
            if solution.hit_probability >= 0.1:  # Very low threshold for free fire
                return True, f"FREE FIRE - {solution.hit_probability*100:.0f}% hit chance"
            return False, "FREE FIRE - target out of effective range"

        if order.command == WeaponsCommand.FIRE_AT_RANGE:
            if solution.predicted_range_km <= order.max_range_km:
                return True, f"Target in range ({solution.predicted_range_km:.1f} km <= {order.max_range_km:.1f} km)"
            return False, f"Target out of range ({solution.predicted_range_km:.1f} km > {order.max_range_km:.1f} km)"

        # FIRE_WHEN_OPTIMAL logic
        min_prob = order.min_hit_probability

        # Adjust threshold based on ammo state
        ammo_percent = weapon_ammo / weapon_magazine if weapon_magazine > 0 else 0
        if order.conserve_ammo or ammo_percent < self.conserve_threshold:
            # Be more selective when low on ammo
            min_prob = max(min_prob, 0.5)

            # Don't fire if we're at reserve level
            if ammo_percent <= self.ammo_reserve:
                return False, f"AMMO CRITICAL ({ammo_percent*100:.0f}%) - reserving for emergencies"

        if solution.hit_probability >= min_prob:
            return True, f"{solution.recommendation} ({solution.hit_probability*100:.0f}% >= {min_prob*100:.0f}%)"

        return False, f"Hit probability too low ({solution.hit_probability*100:.0f}% < {min_prob*100:.0f}%)"

    def get_fire_commands(
        self,
        ship_position: Vector3D,
        ship_velocity: Vector3D,
        weapons: dict,  # slot -> WeaponState
        targets: List,  # List of potential targets (ShipCombatState)
        primary_target_id: Optional[str] = None
    ) -> List[dict]:
        """
        Generate fire commands for all ready weapons.

        Args:
            ship_position: Own ship position.
            ship_velocity: Own ship velocity.
            weapons: Dictionary of weapon slots to WeaponState.
            targets: List of valid target ships.
            primary_target_id: Preferred target.

        Returns:
            List of fire command dictionaries.
        """
        commands = []

        if not targets:
            return commands

        # Sort targets - primary first, then by distance
        sorted_targets = sorted(
            targets,
            key=lambda t: (
                0 if t.ship_id == primary_target_id else 1,
                ship_position.distance_to(t.position)
            )
        )

        for slot, weapon_state in weapons.items():
            if not weapon_state.can_fire():
                continue

            # Skip point defense weapons
            if 'pd' in slot.lower():
                continue

            order = self.get_order(slot)
            target_id = order.target_id if order and order.target_id else None

            # Find best target for this weapon
            best_solution = None
            best_target = None

            for target in sorted_targets:
                # Skip if order specifies different target
                if target_id and target.ship_id != target_id:
                    continue

                # Check range
                distance_km = ship_position.distance_to(target.position) / 1000
                if distance_km > weapon_state.weapon.range_km:
                    continue

                # Calculate firing solution
                solution = calculate_hit_probability(
                    shooter_position=ship_position,
                    shooter_velocity=ship_velocity,
                    target_position=target.position,
                    target_velocity=target.velocity,
                    target_geometry=target.geometry if target.geometry else ShipGeometry(
                        length_m=100, beam_m=20, height_m=15
                    ),
                    target_forward=target.forward,
                    muzzle_velocity_kps=weapon_state.weapon.muzzle_velocity_kps,
                    weapon_base_accuracy=0.9,
                    target_is_evading=getattr(target, 'is_evading', False)
                )

                # Store solution
                if target.ship_id not in self.solutions:
                    self.solutions[target.ship_id] = {}
                self.solutions[target.ship_id][slot] = solution

                # Evaluate if this is the best option
                if best_solution is None or solution.hit_probability > best_solution.hit_probability:
                    best_solution = solution
                    best_target = target

            if best_solution and best_target:
                should_fire, reason = self.evaluate_shot(
                    weapon_slot=slot,
                    weapon_ammo=weapon_state.ammo_remaining,
                    weapon_magazine=weapon_state.weapon.magazine,
                    solution=best_solution,
                    order=order
                )

                if should_fire:
                    commands.append({
                        'type': 'fire_at',
                        'weapon_slot': slot,
                        'target_id': best_target.ship_id,
                        'reason': reason,
                        'hit_probability': best_solution.hit_probability
                    })

        return commands
