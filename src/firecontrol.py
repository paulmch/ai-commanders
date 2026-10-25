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
    from .physics import Vector3D, intercept_time
    from .geometry import ShipGeometry
    from .combat import HitLocation
except ImportError:
    from physics import Vector3D, intercept_time
    from geometry import ShipGeometry
    from combat import HitLocation


# =============================================================================
# HIT PROBABILITY CALCULATION
# =============================================================================
#
# How the engine decides whether a slug hits (CombatSimulation, projectile
# flight + _resolve_projectile_hit_geometric):
#
#   1. GATE: the round flies its real ballistic path against the target's real
#      (accelerating, dodging) path. It can only connect if it passes within
#      HIT_TOLERANCE_M of the hull.
#   2. CONVERSION ROLL: a round inside the gate lands with probability
#      conversion x exp(-deflection^2 / 2 TERMINAL_DISPERSION_SIGMA_M^2), where
#      `conversion` is the legacy fire-control number below (angular size,
#      flight time, crossing, EVASIVE) and `deflection` is how much farther
#      from the hull the round passes than it was aimed at launch.
#
# The per-shot P(hit) shown to commanders and used by FIRE_WHEN_OPTIMAL used
# to be the conversion number alone, so it ignored step 1 entirely - and step 1
# is where nearly every long-range round was lost: over 0.85 of the misses in
# the 2026-10 flash mirror battles were rounds that passed 0.5-25 km wide
# because the target moved (dodged, re-tasked, stopped burning) during a
# 30-70 s flight. P(hit) is now conversion x track, where `track` is the
# expectation of steps 1 and 2 under a model of the target's unforeseen
# displacement (see TargetMotion / displacement_sigma_m / track_probability).
# The conversion roll itself is unchanged, so engine hit rates are unchanged;
# only the forecast became honest. Calibration: scripts/fire_control_calibration.py,
# pinned by tests/test_fire_control_calibration.py.

# Extra radius added to the target hull for the swept-geometry hit test.
# A destroyer is ~15 m radius, so 500 m is ~33x the hull: this is not a
# physical hull size, it is the envelope inside which fire control COULD still
# convert the shot given weapon spread and tracking error. Passing the gate is
# therefore "close enough to try", not "hit" - see
# CombatSimulation._resolve_projectile_hit_geometric.
#
# Do not shrink this to make deflection matter: it is the same envelope for
# every gun in the game, so changing it rebalances all gun combat at once.
# The geometry is made to matter inside the envelope instead, via
# TERMINAL_DISPERSION_SIGMA_M below.
HIT_TOLERANCE_M = 500.0

# 1-sigma of the unmodelled terminal dispersion (barrel jitter, sensor noise,
# last-instant target motion the fire-control solution never saw), expressed as
# a lateral offset at the target.
#
# The simulation propagates the round's real trajectory, so it already knows
# how far from the hull the round actually passed. That realized margin is a
# deterministic offset; this sigma is the random part that decides whether the
# round still clips the hull from that offset. For a 2-D Gaussian dispersion
# and a hull much smaller than sigma, P(hit) falls off as exp(-margin^2/2sigma^2).
#
# Anchored to the tolerance envelope rather than tuned: the envelope is 2 sigma,
# i.e. a round passing at the very edge of what fire control could convert is a
# 2-sigma event. A round that passes THROUGH the hull (margin 0) is unaffected,
# so a clean firing solution behaves exactly as it did before this factor
# existed and the launch-time P(hit) keeps its full meaning.
TERMINAL_DISPERSION_SIGMA_M = HIT_TOLERANCE_M / 2.0

# Unforeseen-maneuver dispersion of a target that does NOT react to rounds:
# 1-sigma (per axis, across the line of fire) of the acceleration the firing
# solution could not see coming - an order changed at the next checkpoint,
# range keeping throttled, a burn ended - as a fraction of the target's
# full-thrust acceleration. Its displacement at impact is
# UNFORESEEN_ACCEL_FRACTION * a * T^2 / 2 (T = time of flight), the same
# k * a_max * tau^2 / 2 form trajectory_prediction.reactive_sigma_m uses.
# Calibrated (maximum likelihood) on ~4,600 rounds fired at non-evading
# targets in 12 recorded battles (LLM and heuristic fleets, 2026-09..10);
# predicted/observed per 100 km bucket is 0.81-1.12x out to 600 km
# (scripts/fire_control_calibration.py).
UNFORESEEN_ACCEL_FRACTION = 0.018

# Dodge dispersion of a target whose evasion reacts to inbound rounds (EVASIVE,
# or any maneuver flown with the evade overlay). From the moment a round is
# inside its threat horizon the ship swings across the line of fire with its
# drive lit and burns at its maneuver throttle until the round's predicted
# miss clears ~500 m; it picks its side at random each 8 s jink cycle and
# weighs every inbound round, so where it ends up is a spread, not a fixed
# offset. Modelled as an extra 1-sigma displacement
# k * a * throttle * min(T, horizon)^2 / 2 (the swing builds displacement
# from the start - fitting a reaction delay drove it to zero), with k:
#   DODGE_ACCEL_FRACTION            the evade overlay on another maneuver;
#   EVASIVE_DODGE_ACCEL_FRACTION    the EVASIVE maneuver. Its lead
#       (CombatSimulation._estimate_target_acceleration) already assumes half
#       a burn along the nose, which tracks part of every jink, and between
#       threats it weaves around its course anyway: rounds lose less to the
#       dodge itself (the conversion roll charges EVASIVE x0.6 on top).
# Calibrated together with JINK_ACCEL_FRACTION (maximum likelihood) on the
# live engine: 85k rounds from 1, 2, 4 and 8 shooters at frigates..
# dreadnoughts flying PADLOCK + dodge, HOLD_RANGE + dodge and EVASIVE,
# 40-400 km (scripts/fire_control_calibration.py harness); per maneuver class,
# shooter count and 100 km bucket (30+ hits) the forecast is 0.65-1.17x of
# what landed.
DODGE_ACCEL_FRACTION = 0.10
EVASIVE_DODGE_ACCEL_FRACTION = 0.06

# A dodger whose evasion is flying it RIGHT NOW (mid-jink) is already burning
# across the line of fire, and its lead does not extrapolate that burn (a
# PADLOCK dodger is led as if it coasts). The part of its drive output the
# lead misses (CombatSimulation.target_motion: drive output minus the lead's
# assumed acceleration, across the line of fire) keeps pushing it while the
# jink holds: JINK_ACCEL_FRACTION * a_missed * min(T, horizon)^2 / 2, no
# minimum time to go (it is already moving). Live engine, PADLOCK / HOLD_RANGE
# dodgers at 100-300 km: rounds that found one mid-jink landed 0.15 per round,
# rounds between jinks 0.18 (forecast 0.12 / 0.20). This is how fleet fire
# shows up: other ships' rounds keep a dodger jinking; a separate fleet-fire
# factor (dodge x f under 2+ shooters) fitted to 0.7-1.0 next to this term,
# so it is not modelled. Not for the EVASIVE maneuver: its lead and
# EVASIVE_DODGE_ACCEL_FRACTION cover it.
#
# Known residual: in the 2026-10 flash mirror (LLM fleets, every ship PADLOCK /
# HOLD_RANGE + evade 'dodge' under a whole fleet's fire, new orders every
# checkpoint) the dodgers moved 2-4x farther across the line of fire during a
# flight (median 0.45 of a*T^2/2, vs 0.10-0.24 in seeded engine runs of the
# same shape), and 0 of 188 rounds at 100-300 km landed where this model
# forecasts ~16. Seeded fleet-vs-fleet runs with fixed orders do not
# reproduce it (there the forecast is 0.74-0.96x of what landed), so nothing
# visible at fire time predicts it; the doctrine tells commanders instead
# (prompts.GUN_PHYSICS_REFERENCE), and tests/test_fire_control_calibration.py
# pins it.
JINK_ACCEL_FRACTION = 0.39

# A HEADING target still turning onto its ordered heading (or holding a
# synchronised burn start) burns at the ordered throttle once inside
# HEADING_BURN_CONE_DEG, and its lead (the drive's actual output: a token
# throttle while turning) does not see that burn coming. Its displacement
# across the line of fire grows from the moment the burn lights:
# PENDING_BURN_FRACTION * a_burn * (T - t_light)^2 / 2 (the same difference
# - ordered burn minus drive output - covers the last few degrees of swing
# inside the cone). Fitted on HEADING rounds from 32 seeded battles and the
# live-engine harness: close to the full burn (a deterministic offset of
# a T^2/2 in one direction is a ~0.7 sigma spread; the rest is orders that
# change at the next checkpoint).
PENDING_BURN_FRACTION = 0.9


@dataclass(frozen=True)
class TargetMotion:
    """
    What a target can do to a round while it flies.

    Attributes:
        accel_ms2: Full-throttle acceleration available now (m/s^2, after
            engine damage).
        dodges: Whether its evasion reacts to inbound rounds (EVASIVE, or the
            evade overlay on any other maneuver).
        evasive: It flies the EVASIVE maneuver (EVASIVE_DODGE_ACCEL_FRACTION)
            rather than the evade overlay (DODGE_ACCEL_FRACTION).
        dodge_throttle: Throttle its evasion burns at (the maneuver's).
        dodge_horizon_s: Evasion only sees rounds this close in time to go
            (simulation.BALLISTIC_THREAT_HORIZON_S).
        dodge_min_time_s: Evasion needs at least this much time to go to
            jink at all (simulation EVASION_TURN_ALLOWANCE_S +
            EVASION_MIN_JINK_S); closer rounds are taken or tanked.
        jink_accel_ms2: The overlay's evasion is flying it now: the part of
            its drive output, across the line of fire, that the lead does
            not extrapolate (0 when not jinking).
        pending_burn_ms2: An ordered HEADING burn the lead does not see yet
            (ordered burn minus drive output, across the line of fire).
        pending_burn_in_s: When that burn lights (rest of the turn onto the
            heading, or a held formation start; 0 when already burning).
    """
    accel_ms2: float
    dodges: bool = False
    evasive: bool = False
    dodge_throttle: float = 1.0
    dodge_horizon_s: float = 60.0
    dodge_min_time_s: float = 7.0
    jink_accel_ms2: float = 0.0
    pending_burn_ms2: float = 0.0
    pending_burn_in_s: float = 0.0


def displacement_sigma_m(time_of_flight_s: float, motion: TargetMotion) -> float:
    """
    1-sigma (per axis, across the line of fire) of where the target will be,
    relative to the firing solution's aim point, when a round fired now
    arrives: unforeseen maneuvering, plus - for a target that dodges - its
    dodge and the jink it is already flying, plus an ordered burn about to
    light. Grows as T^2: a 1 g target moves 0.5 * a * t^2 = 4.4 km in 30 s.
    """
    t = max(0.0, float(time_of_flight_s))
    a = max(0.0, float(motion.accel_ms2))
    sigma_free = UNFORESEEN_ACCEL_FRACTION * a * t * t / 2.0
    sigma_dodge = sigma_jink = sigma_pending = 0.0
    if motion.dodges:
        td = min(t, motion.dodge_horizon_s)
        if t >= motion.dodge_min_time_s:
            k = EVASIVE_DODGE_ACCEL_FRACTION if motion.evasive else DODGE_ACCEL_FRACTION
            throttle = max(0.0, min(1.0, motion.dodge_throttle))
            sigma_dodge = k * a * throttle * td * td / 2.0
        jink = max(0.0, float(motion.jink_accel_ms2))
        if jink > 0.0 and not motion.evasive:
            sigma_jink = JINK_ACCEL_FRACTION * jink * td * td / 2.0
    pending = max(0.0, float(motion.pending_burn_ms2))
    if pending > 0.0:
        tp = t - max(0.0, float(motion.pending_burn_in_s))
        if tp > 0.0:
            sigma_pending = PENDING_BURN_FRACTION * pending * tp * tp / 2.0
    return math.sqrt(sigma_free ** 2 + sigma_dodge ** 2 + sigma_jink ** 2 + sigma_pending ** 2)


def track_probability(
    sigma_m: float,
    envelope_m: float,
    dispersion_sigma_m: float = TERMINAL_DISPERSION_SIGMA_M,
) -> float:
    """
    Expected value of the engine's gate x geometry factor for a round aimed
    at the target's predicted position, when the target actually ends up a
    2-D Gaussian (per-axis sigma_m) away from it:

        E[ 1{r < R} * exp(-r^2 / 2 s_t^2) ],  r ~ Rayleigh(sigma_m)
          = s_t^2 / (s_t^2 + sigma^2) * (1 - exp(-R^2 / 2 s_c^2)),
        1/s_c^2 = 1/sigma^2 + 1/s_t^2

    with R the gate radius (HIT_TOLERANCE_M + hull radius) and s_t the
    terminal dispersion sigma. Exactly 1 for a target that cannot surprise the
    solution (sigma 0); falls as (s_t / sigma)^2 once sigma dwarfs the
    envelope.
    """
    sigma = max(0.0, float(sigma_m))
    s_t = float(dispersion_sigma_m)
    if sigma < 1e-6:
        return 1.0
    inv_c = 1.0 / (sigma * sigma) + 1.0 / (s_t * s_t)
    gate = 1.0 - math.exp(-(envelope_m * envelope_m) * inv_c / 2.0)
    return (s_t * s_t) / (s_t * s_t + sigma * sigma) * gate


@dataclass
class FiringSolution:
    """
    Complete firing solution for a weapon against a target.

    Attributes:
        can_fire: Whether the weapon can engage this target.
        hit_probability: Probability this shot hits: conversion_probability x
            track_probability. What commanders see and FIRE_WHEN_OPTIMAL gates.
        time_of_flight_s: Estimated projectile flight time in seconds.
        predicted_range_km: Distance when projectile reaches target.
        target_aspect: Which face of target is exposed.
        recommendation: Human-readable firing recommendation.
        conversion_probability: P(the round lands | it arrives where it was
            aimed) - the engine's dispersion roll consumes THIS, never
            hit_probability (that would charge the target's motion twice).
            Defaults to hit_probability.
        track_probability: P(the target is still where the round was aimed,
            within the gate) x the expected geometry factor; 1.0 when no
            target motion was supplied.
        displacement_sigma_m: The target's 1-sigma unforeseen displacement at
            impact behind track_probability (0 when none was modelled).
        target_motion: The TargetMotion the forecast used (None = none).
    """
    can_fire: bool
    hit_probability: float
    time_of_flight_s: float
    predicted_range_km: float
    target_aspect: HitLocation
    recommendation: str
    conversion_probability: Optional[float] = None
    track_probability: float = 1.0
    displacement_sigma_m: float = 0.0
    target_motion: Optional[TargetMotion] = None

    def __post_init__(self) -> None:
        if self.conversion_probability is None:
            self.conversion_probability = self.hit_probability

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
    target_is_evading: bool = False,
    target_motion: Optional[TargetMotion] = None,
) -> FiringSolution:
    """
    Calculate the probability of hitting a target.

    hit_probability = conversion x track:

    - conversion (the engine's dispersion roll; unchanged legacy model):
      target angular size, flight-time and crossing-angle penalties, and the
      EVASIVE x0.6.
    - track: the round only gets to roll if the target is still inside the
      HIT_TOLERANCE_M gate around the aim point when it arrives. With
      target_motion, the target's unforeseen displacement over the time of
      flight (displacement_sigma_m) is turned into the engine-exact
      expectation of gate x geometry factor (track_probability). Without it
      (unknown target) track is 1 - the bare conversion number, which is an
      upper bound, not a forecast.

    Args:
        shooter_position: Shooter's current position (meters).
        shooter_velocity: Shooter's velocity vector (m/s).
        target_position: Target's current position (meters).
        target_velocity: Target's velocity vector (m/s).
        target_geometry: Target ship's geometry for cross-section.
        target_forward: Target's facing direction (normalized).
        muzzle_velocity_kps: Weapon's muzzle velocity in km/s.
        weapon_base_accuracy: Weapon's base accuracy factor (0.0-1.0).
        target_is_evading: Target flies the EVASIVE maneuver (the engine's
            conversion roll charges it x0.6).
        target_motion: What the target can do during the flight
            (CombatSimulation.target_motion builds it from the live ship).

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

    # Projectile velocity (relative to shooter, in direction of target)
    muzzle_velocity_mps = muzzle_velocity_kps * 1000

    # A radial closing-speed estimate ignores crossing motion and can promise
    # hits on a target moving sideways faster than the round can travel.
    time_of_flight_s = intercept_time(rel_pos, rel_vel, muzzle_velocity_mps)
    if time_of_flight_s is None:
        # Projectile can't catch target
        return FiringSolution(
            can_fire=False,
            hit_probability=0.0,
            time_of_flight_s=float('inf'),
            predicted_range_km=distance_km,
            target_aspect=HitLocation.TAIL,
            recommendation="TARGET OUTRUNNING PROJECTILE - DO NOT FIRE",
            track_probability=0.0,
        )

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

    # ---- Conversion: the engine's dispersion roll (legacy model) ----------
    # Calculate target angular size (steradians approximation)
    # Angular size = cross_section / distance^2
    target_radius_m = math.sqrt(cross_section_m2 / math.pi)  # Effective radius
    angular_size_rad = target_radius_m / predicted_distance if predicted_distance > 0 else 1.0

    # Probability scales with angular size: prob = 1 - exp(-k * angular_size).
    # With k = 10000 and a 60 m effective radius, base_prob crosses 0.5 only
    # at ~866 km (angular size = ln(2)/k ~= 6.9e-5 rad) and saturates near
    # 1.0 for all ranges under ~300 km. At typical combat ranges the visible
    # range falloff of the conversion therefore comes almost entirely from
    # the flight-time and crossing-angle penalties applied below, not from
    # this angular term.
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
    conversion = max(0.01, min(0.99, prob))

    # ---- Track: is the target still where the round was aimed? -------------
    sigma_m = 0.0
    track = 1.0
    if target_motion is not None:
        sigma_m = displacement_sigma_m(time_of_flight_s, target_motion)
        envelope_m = HIT_TOLERANCE_M + max(0.0, float(getattr(target_geometry, 'radius_m', 0.0) or 0.0))
        track = track_probability(sigma_m, envelope_m)
    prob = max(0.0, min(0.99, conversion * track))

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
        recommendation=recommendation,
        conversion_probability=conversion,
        track_probability=track,
        displacement_sigma_m=sigma_m,
        target_motion=target_motion,
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
                    # ShipGeometry takes radius/nose/engine lengths - there are no
                    # beam_m/height_m fields, so this fallback raised TypeError on
                    # any geometry-less target instead of degrading gracefully.
                    target_geometry=target.geometry if target.geometry else ShipGeometry(
                        length_m=100.0,
                        radius_m=12.5,
                        nose_cone_length_m=20.0,
                        engine_section_length_m=20.0,
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
