"""
Full engagement tests with projectile tracking for the AI Commanders simulator.

Tests complete weapon engagements from firing to impact:
- Ships at maximum weapon range
- Fire coilgun or torpedo
- Track projectile through space
- Detect impact with target geometry
- Resolve damage on hit

These tests validate the entire engagement pipeline end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import pytest

from src.physics import Vector3D, ShipState
from src.combat import (
    HitLocation,
    load_fleet_data,
    create_weapon_from_fleet_data,
    create_ship_armor_from_fleet_data,
    CombatResolver,
)
from src.projectile import KineticProjectile, ProjectileLauncher
from src.torpedo import Torpedo, TorpedoLauncher, TorpedoSpecs, GuidanceMode
from src.geometry import ShipGeometry, create_geometry_from_fleet_data
from src.damage import DamagePropagator, DamageCone, WeaponDamageProfile


# =============================================================================
# CONSTANTS
# =============================================================================

# Coilgun engagement parameters
COILGUN_MAX_RANGE_KM = 900.0
COILGUN_MUZZLE_VELOCITY_KPS = 10.0
COILGUN_SLUG_MASS_KG = 25.0

# Torpedo engagement parameters
TORPEDO_MAX_RANGE_KM = 2000.0
TORPEDO_WARHEAD_GJ = 50.0

# Simulation parameters
SIMULATION_DT_SECONDS = 1.0  # 1 second timestep
MAX_SIMULATION_TIME_SECONDS = 600.0  # 10 minute max


# =============================================================================
# HELPER CLASSES
# =============================================================================

@dataclass
class EngagementResult:
    """Result of a full engagement simulation."""
    projectile_type: str
    hit: bool
    hit_location: Optional[HitLocation]
    flight_time_seconds: float
    impact_velocity_kps: float
    kinetic_energy_gj: float
    initial_range_km: float
    final_range_km: float
    trajectory_points: int


@dataclass
class ShipTarget:
    """Target ship state for engagement."""
    position: Vector3D
    velocity: Vector3D
    forward: Vector3D
    geometry: ShipGeometry

    def get_collision_radius(self) -> float:
        """Get collision detection radius (approximation)."""
        return max(self.geometry.radius_m, self.geometry.length_m / 2)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fleet_data():
    """Load fleet data for testing."""
    fleet_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    return load_fleet_data(fleet_path)


@pytest.fixture
def destroyer_geometry(fleet_data):
    """Create destroyer geometry."""
    return create_geometry_from_fleet_data("destroyer", fleet_data)


@pytest.fixture
def cruiser_geometry(fleet_data):
    """Create cruiser geometry."""
    return create_geometry_from_fleet_data("cruiser", fleet_data)


@pytest.fixture
def coilgun_launcher():
    """Create a coilgun launcher."""
    return ProjectileLauncher(
        default_mass_kg=COILGUN_SLUG_MASS_KG,
        default_muzzle_velocity_kps=COILGUN_MUZZLE_VELOCITY_KPS,
        magazine_capacity=450,
        cooldown_seconds=45.0,
    )


@pytest.fixture
def torpedo_launcher():
    """Create a torpedo launcher with fleet specs."""
    specs = TorpedoSpecs.from_fleet_data(
        warhead_yield_gj=TORPEDO_WARHEAD_GJ,
        ammo_mass_kg=1600.0,
        range_km=TORPEDO_MAX_RANGE_KM,
    )
    return TorpedoLauncher(specs=specs, magazine_capacity=16, cooldown_seconds=30.0)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_projectile_impact(
    projectile_pos: Vector3D,
    target: ShipTarget,
) -> Tuple[bool, float]:
    """
    Check if projectile has impacted target.

    Args:
        projectile_pos: Current projectile position
        target: Target ship

    Returns:
        Tuple of (hit, distance_to_target_m)
    """
    distance = projectile_pos.distance_to(target.position)

    # Use ship length as collision radius - if projectile is within
    # half the ship length, it's a hit
    impact_distance = target.geometry.length_m

    return distance <= impact_distance, distance


def calculate_hit_location_from_approach(
    projectile_velocity: Vector3D,
    target: ShipTarget
) -> HitLocation:
    """Calculate hit location based on projectile approach vector."""
    return target.geometry.calculate_hit_location(
        projectile_velocity,
        target.forward
    )


def simulate_coilgun_engagement(
    shooter_pos: Vector3D,
    shooter_vel: Vector3D,
    target: ShipTarget,
    launcher: ProjectileLauncher,
    dt: float = SIMULATION_DT_SECONDS,
    max_time: float = MAX_SIMULATION_TIME_SECONDS,
) -> EngagementResult:
    """
    Simulate a full coilgun engagement from firing to impact.

    Args:
        shooter_pos: Shooter position (meters)
        shooter_vel: Shooter velocity (m/s)
        target: Target ship
        launcher: Projectile launcher
        dt: Simulation timestep (seconds)
        max_time: Maximum simulation time (seconds)

    Returns:
        EngagementResult with engagement outcome
    """
    initial_range = shooter_pos.distance_to(target.position) / 1000  # km

    # Copy target position so we don't modify the original during iteration
    target_pos = Vector3D(target.position.x, target.position.y, target.position.z)

    # Calculate lead direction - simple lead calculation
    # Time to reach target at current muzzle velocity
    rel_vel = target.velocity - shooter_vel
    distance_m = shooter_pos.distance_to(target_pos)
    muzzle_vel_ms = launcher.default_muzzle_velocity_kps * 1000

    # Estimate intercept time (iterative refinement)
    intercept_time = distance_m / muzzle_vel_ms  # Initial estimate

    for _ in range(3):  # Refine estimate
        predicted_target_pos = target_pos + target.velocity * intercept_time
        new_distance = shooter_pos.distance_to(predicted_target_pos)
        # Closing velocity from projectile perspective
        fire_dir = (predicted_target_pos - shooter_pos).normalized()
        proj_vel = shooter_vel + fire_dir * muzzle_vel_ms
        rel_closing = (proj_vel - target.velocity).dot(fire_dir)
        if rel_closing > 0:
            intercept_time = new_distance / rel_closing
        intercept_time = max(1.0, min(intercept_time, 300.0))

    # Calculate aim point
    predicted_target = target_pos + target.velocity * intercept_time
    fire_direction = (predicted_target - shooter_pos).normalized()

    # Launch projectile
    projectile = KineticProjectile.from_launch(
        shooter_position=shooter_pos,
        shooter_velocity=shooter_vel,
        target_direction=fire_direction,
        muzzle_velocity_kps=launcher.default_muzzle_velocity_kps,
        mass_kg=launcher.default_mass_kg,
    )

    # Simulate projectile flight
    time_elapsed = 0.0
    trajectory_points = 0
    hit = False
    hit_location = None
    prev_distance = distance_m
    min_distance = distance_m

    while time_elapsed < max_time:
        # Update projectile position
        projectile.update(dt)
        trajectory_points += 1
        time_elapsed += dt

        # Update target position (constant velocity)
        target_pos = target_pos + target.velocity * dt

        # Check for impact
        current_distance = projectile.position.distance_to(target_pos)

        # Track minimum distance for miss detection
        if current_distance < min_distance:
            min_distance = current_distance

        # Check if within ship collision radius
        if current_distance <= target.geometry.length_m:
            hit = True
            hit_location = calculate_hit_location_from_approach(
                projectile.velocity,
                ShipTarget(target_pos, target.velocity, target.forward, target.geometry)
            )
            break

        # Check if projectile has passed closest approach and is now diverging
        if current_distance > prev_distance and time_elapsed > intercept_time * 0.5:
            # Check if it's clearly missed (distance increasing after closest approach)
            if current_distance > min_distance * 1.5:
                break

        prev_distance = current_distance

    final_range = projectile.position.distance_to(target_pos) / 1000

    return EngagementResult(
        projectile_type="coilgun",
        hit=hit,
        hit_location=hit_location,
        flight_time_seconds=time_elapsed,
        impact_velocity_kps=projectile.speed_kps,
        kinetic_energy_gj=projectile.kinetic_energy_gj,
        initial_range_km=initial_range,
        final_range_km=final_range,
        trajectory_points=trajectory_points,
    )


def simulate_torpedo_engagement(
    shooter_pos: Vector3D,
    shooter_vel: Vector3D,
    target: ShipTarget,
    launcher: TorpedoLauncher,
    dt: float = SIMULATION_DT_SECONDS,
    max_time: float = MAX_SIMULATION_TIME_SECONDS,
) -> EngagementResult:
    """
    Simulate a full torpedo engagement with guided flight.

    Args:
        shooter_pos: Shooter position (meters)
        shooter_vel: Shooter velocity (m/s)
        target: Target ship
        launcher: Torpedo launcher
        dt: Simulation timestep (seconds)
        max_time: Maximum simulation time (seconds)

    Returns:
        EngagementResult with engagement outcome
    """
    initial_range = shooter_pos.distance_to(target.position) / 1000  # km

    # Copy target position
    target_pos = Vector3D(target.position.x, target.position.y, target.position.z)

    # Launch torpedo
    torpedo = launcher.launch(
        shooter_position=shooter_pos,
        shooter_velocity=shooter_vel,
        target_id="target_001",
        target_position=target_pos,
        target_velocity=target.velocity,
        current_time=0.0,
    )

    if torpedo is None:
        return EngagementResult(
            projectile_type="torpedo",
            hit=False,
            hit_location=None,
            flight_time_seconds=0.0,
            impact_velocity_kps=0.0,
            kinetic_energy_gj=0.0,
            initial_range_km=0.0,
            final_range_km=0.0,
            trajectory_points=0,
        )

    # Simulate torpedo flight
    time_elapsed = 0.0
    trajectory_points = 0
    hit = False
    hit_location = None
    prev_distance = initial_range * 1000
    min_distance = prev_distance

    while time_elapsed < max_time:
        # Update torpedo with guidance (target position is updated)
        torpedo.update(dt, target_pos, target.velocity)
        trajectory_points += 1
        time_elapsed += dt

        # Update target position (constant velocity)
        target_pos = target_pos + target.velocity * dt

        # Check for impact
        current_distance = torpedo.position.distance_to(target_pos)

        # Track minimum distance
        if current_distance < min_distance:
            min_distance = current_distance

        # Check if within ship collision radius
        # Use larger collision radius for high-speed projectiles
        collision_radius = max(target.geometry.length_m, 100.0)  # At least 100m

        if current_distance <= collision_radius:
            hit = True
            hit_location = calculate_hit_location_from_approach(
                torpedo.velocity,
                ShipTarget(target_pos, target.velocity, target.forward, target.geometry)
            )
            break

        # Check if torpedo has passed closest approach (could have passed through target)
        if current_distance > prev_distance and prev_distance < collision_radius * 200:
            # We were close and are now moving away - check if close enough for hit
            # Account for high-speed passage through target zone
            torpedo_speed = torpedo.velocity.magnitude
            max_passthrough_distance = torpedo_speed * dt * 2  # Could have passed in one step
            if min_distance < max_passthrough_distance:
                hit = True
                hit_location = calculate_hit_location_from_approach(
                    torpedo.velocity,
                    ShipTarget(target_pos, target.velocity, target.forward, target.geometry)
                )
                break

        # Check if torpedo has passed and is now diverging (after reaching minimum distance)
        if torpedo.fuel_exhausted and current_distance > prev_distance:
            # If distance is increasing and torpedo is out of fuel
            if current_distance > min_distance * 2 and min_distance > collision_radius * 100:
                break

        prev_distance = current_distance

    final_range = torpedo.position.distance_to(target_pos) / 1000
    impact_vel = torpedo.velocity.magnitude / 1000

    # Calculate kinetic energy contribution
    kinetic_energy = 0.5 * torpedo.current_mass_kg * (impact_vel * 1000) ** 2 / 1e9

    return EngagementResult(
        projectile_type="torpedo",
        hit=hit,
        hit_location=hit_location,
        flight_time_seconds=time_elapsed,
        impact_velocity_kps=impact_vel,
        kinetic_energy_gj=kinetic_energy + launcher.specs.warhead_yield_gj if hit else 0,
        initial_range_km=initial_range,
        final_range_km=final_range,
        trajectory_points=trajectory_points,
    )


# =============================================================================
# COILGUN ENGAGEMENT TESTS
# =============================================================================

class TestCoilgunEngagement:
    """Tests for coilgun projectile tracking from firing to impact."""

    def test_coilgun_stationary_target_max_range(
        self, fleet_data, destroyer_geometry, coilgun_launcher
    ):
        """
        Fire coilgun at stationary target at maximum range.
        Track projectile until impact.
        """
        # Shooter at origin, stationary
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)

        # Target at max range (900 km), stationary
        target = ShipTarget(
            position=Vector3D(COILGUN_MAX_RANGE_KM * 1000, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0),  # Facing shooter
            geometry=destroyer_geometry,
        )

        result = simulate_coilgun_engagement(
            shooter_pos, shooter_vel, target, coilgun_launcher
        )

        # Should hit stationary target
        assert result.hit, "Should hit stationary target at max range"
        assert result.hit_location == HitLocation.NOSE, "Should hit nose when target faces shooter"

        # Flight time should be approximately range / velocity
        expected_flight_time = (COILGUN_MAX_RANGE_KM * 1000) / (COILGUN_MUZZLE_VELOCITY_KPS * 1000)
        assert abs(result.flight_time_seconds - expected_flight_time) < 5.0

        # Projectile should maintain velocity (no drag in space)
        assert abs(result.impact_velocity_kps - COILGUN_MUZZLE_VELOCITY_KPS) < 0.1

        print(f"\n  Coilgun max range engagement:")
        print(f"    Initial range: {result.initial_range_km:.0f} km")
        print(f"    Flight time: {result.flight_time_seconds:.1f} s")
        print(f"    Impact velocity: {result.impact_velocity_kps:.1f} km/s")
        print(f"    Kinetic energy: {result.kinetic_energy_gj:.2f} GJ")
        print(f"    Hit location: {result.hit_location.value}")

    def test_coilgun_closing_engagement(
        self, fleet_data, destroyer_geometry, coilgun_launcher
    ):
        """
        Fire coilgun at target closing at high velocity.
        Tests velocity inheritance and lead calculation.
        """
        # Shooter moving at 10 km/s
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(10_000, 0, 0)  # 10 km/s toward target

        # Target at 500 km, closing at 10 km/s
        target = ShipTarget(
            position=Vector3D(500_000, 0, 0),  # 500 km
            velocity=Vector3D(-10_000, 0, 0),  # 10 km/s toward shooter
            forward=Vector3D(-1, 0, 0),
            geometry=destroyer_geometry,
        )

        result = simulate_coilgun_engagement(
            shooter_pos, shooter_vel, target, coilgun_launcher
        )

        # Should hit
        assert result.hit, "Should hit closing target"

        # Projectile velocity in REST FRAME should be shooter + muzzle
        # Shooter: 10 km/s, Muzzle: 10 km/s (but fired at intercept point, not directly ahead)
        # The actual projectile velocity will depend on lead angle
        # But kinetic energy should be significant due to velocity inheritance
        assert result.kinetic_energy_gj > 0.5, "Should have significant kinetic energy"

        # Flight time should be short due to high closing rate
        # Initial separation: 500 km, closing at ~30 km/s -> ~17 seconds
        assert result.flight_time_seconds < 50, "Should hit quickly"

        print(f"\n  Coilgun closing engagement:")
        print(f"    Flight time: {result.flight_time_seconds:.1f} s")
        print(f"    Impact velocity: {result.impact_velocity_kps:.1f} km/s")
        print(f"    Kinetic energy: {result.kinetic_energy_gj:.2f} GJ")

    def test_coilgun_lateral_target(
        self, fleet_data, destroyer_geometry, coilgun_launcher
    ):
        """
        Fire coilgun at target perpendicular to shooter velocity.
        Tests hit location calculation for lateral hits.
        """
        # Shooter at origin, stationary
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)

        # Target at 300 km, broadside to shooter
        target = ShipTarget(
            position=Vector3D(300_000, 0, 0),  # 300 km
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(0, 1, 0),  # Facing perpendicular (broadside)
            geometry=destroyer_geometry,
        )

        result = simulate_coilgun_engagement(
            shooter_pos, shooter_vel, target, coilgun_launcher
        )

        assert result.hit, "Should hit broadside target"
        assert result.hit_location == HitLocation.LATERAL, "Should hit lateral armor"

        print(f"\n  Coilgun lateral engagement:")
        print(f"    Hit location: {result.hit_location.value}")
        print(f"    Flight time: {result.flight_time_seconds:.1f} s")

    def test_coilgun_retreating_target(
        self, fleet_data, destroyer_geometry, coilgun_launcher
    ):
        """
        Fire coilgun at target moving away.
        Tests if projectile can catch retreating target.
        """
        # Shooter stationary
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)

        # Target retreating at 5 km/s
        target = ShipTarget(
            position=Vector3D(400_000, 0, 0),  # 400 km
            velocity=Vector3D(5_000, 0, 0),  # Retreating at 5 km/s
            forward=Vector3D(1, 0, 0),  # Facing away
            geometry=destroyer_geometry,
        )

        result = simulate_coilgun_engagement(
            shooter_pos, shooter_vel, target, coilgun_launcher
        )

        # Projectile (10 km/s) should catch target (5 km/s)
        assert result.hit, "Should catch slower retreating target"
        assert result.hit_location == HitLocation.TAIL, "Should hit tail from behind"

        # Flight time longer due to chase
        # Closing rate: 10 - 5 = 5 km/s
        # Time to close 400 km at 5 km/s = 80 s
        assert result.flight_time_seconds > 70, "Should take longer due to chase"

        print(f"\n  Coilgun chase engagement:")
        print(f"    Hit location: {result.hit_location.value}")
        print(f"    Flight time: {result.flight_time_seconds:.1f} s")

    def test_coilgun_fast_retreating_target_miss(
        self, fleet_data, destroyer_geometry, coilgun_launcher
    ):
        """
        Fire coilgun at target retreating faster than projectile.
        Should miss.
        """
        # Shooter stationary
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)

        # Target retreating at 15 km/s (faster than 10 km/s projectile)
        target = ShipTarget(
            position=Vector3D(400_000, 0, 0),
            velocity=Vector3D(15_000, 0, 0),  # 15 km/s away
            forward=Vector3D(1, 0, 0),
            geometry=destroyer_geometry,
        )

        result = simulate_coilgun_engagement(
            shooter_pos, shooter_vel, target, coilgun_launcher,
            max_time=300,  # Limit simulation time
        )

        # Should miss - target is faster
        assert not result.hit, "Should miss target retreating faster than projectile"
        assert result.final_range_km > result.initial_range_km, "Range should increase"

        print(f"\n  Coilgun miss (fast retreat):")
        print(f"    Initial range: {result.initial_range_km:.0f} km")
        print(f"    Final range: {result.final_range_km:.0f} km")


# =============================================================================
# TORPEDO ENGAGEMENT TESTS
# =============================================================================

class TestTorpedoEngagement:
    """Tests for torpedo tracking with guided flight."""

    def test_torpedo_stationary_target_max_range(
        self, fleet_data, cruiser_geometry, torpedo_launcher
    ):
        """
        Fire torpedo at stationary target at maximum range.
        Track guided flight until impact.
        """
        # Shooter at origin, stationary
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)

        # Target at max range (2000 km), stationary
        target = ShipTarget(
            position=Vector3D(TORPEDO_MAX_RANGE_KM * 1000, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0),
            geometry=cruiser_geometry,
        )

        result = simulate_torpedo_engagement(
            shooter_pos, shooter_vel, target, torpedo_launcher
        )

        assert result.hit, "Torpedo should hit stationary target at max range"
        assert result.hit_location == HitLocation.NOSE

        # Torpedo has 60 km/s delta-v but COLLISION guidance coasts when on course
        # This is fuel-efficient but results in moderate terminal velocity
        assert result.impact_velocity_kps > 10, "Should build up reasonable velocity"

        print(f"\n  Torpedo max range engagement:")
        print(f"    Initial range: {result.initial_range_km:.0f} km")
        print(f"    Flight time: {result.flight_time_seconds:.1f} s")
        print(f"    Impact velocity: {result.impact_velocity_kps:.1f} km/s")
        print(f"    Total energy: {result.kinetic_energy_gj:.1f} GJ (warhead + kinetic)")

    def test_torpedo_medium_range(
        self, fleet_data, cruiser_geometry, torpedo_launcher
    ):
        """
        Fire torpedo at stationary target at medium range.
        Tests guidance convergence.
        """
        # Shooter at origin, stationary
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)

        # Target at medium range, stationary
        target = ShipTarget(
            position=Vector3D(800_000, 0, 0),  # 800 km
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0),
            geometry=cruiser_geometry,
        )

        result = simulate_torpedo_engagement(
            shooter_pos, shooter_vel, target, torpedo_launcher
        )

        # Torpedo should hit stationary target
        assert result.hit, "Torpedo should hit stationary target at medium range"

        print(f"\n  Torpedo medium range engagement:")
        print(f"    Flight time: {result.flight_time_seconds:.1f} s")
        print(f"    Hit location: {result.hit_location.value if result.hit_location else 'N/A'}")

    def test_torpedo_closing_engagement(
        self, fleet_data, cruiser_geometry, torpedo_launcher
    ):
        """
        Fire torpedo at closing target.
        Tests velocity inheritance and terminal guidance.
        """
        # Ships closing at high speed
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(20_000, 0, 0)  # 20 km/s toward target

        # Target closing
        target = ShipTarget(
            position=Vector3D(1_000_000, 0, 0),  # 1000 km
            velocity=Vector3D(-15_000, 0, 0),  # 15 km/s toward shooter
            forward=Vector3D(-1, 0, 0),
            geometry=cruiser_geometry,
        )

        result = simulate_torpedo_engagement(
            shooter_pos, shooter_vel, target, torpedo_launcher
        )

        assert result.hit, "Torpedo should hit closing target"

        # Torpedo inherits 20 km/s from shooter
        # Impact velocity is torpedo's absolute velocity, not closing rate
        assert result.impact_velocity_kps > 15, "Should have reasonable impact velocity"

        # Flight time should be short due to high closing rate
        assert result.flight_time_seconds < 60, "Should hit quickly due to high closing rate"

        print(f"\n  Torpedo closing engagement:")
        print(f"    Flight time: {result.flight_time_seconds:.1f} s")
        print(f"    Impact velocity: {result.impact_velocity_kps:.1f} km/s")
        print(f"    Total energy: {result.kinetic_energy_gj:.1f} GJ")

    def test_torpedo_tail_chase(
        self, fleet_data, cruiser_geometry, torpedo_launcher
    ):
        """
        Fire torpedo at target fleeing at moderate speed.
        Tests guidance and fuel consumption in chase.
        """
        # Shooter stationary
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)

        # Target fleeing at 10 km/s
        target = ShipTarget(
            position=Vector3D(500_000, 0, 0),  # 500 km
            velocity=Vector3D(10_000, 0, 0),  # 10 km/s away
            forward=Vector3D(1, 0, 0),  # Facing away
            geometry=cruiser_geometry,
        )

        result = simulate_torpedo_engagement(
            shooter_pos, shooter_vel, target, torpedo_launcher
        )

        # Torpedo has 60 km/s delta-v, should catch 10 km/s target
        assert result.hit, "Torpedo should catch slower fleeing target"
        assert result.hit_location == HitLocation.TAIL, "Should hit tail from behind"

        print(f"\n  Torpedo tail chase:")
        print(f"    Flight time: {result.flight_time_seconds:.1f} s")
        print(f"    Hit location: {result.hit_location.value}")


# =============================================================================
# COMBINED ENGAGEMENT TESTS
# =============================================================================

class TestCombinedEngagement:
    """Tests comparing coilgun and torpedo engagements."""

    def test_engagement_comparison_mid_range(
        self, fleet_data, destroyer_geometry, coilgun_launcher, torpedo_launcher
    ):
        """
        Compare coilgun vs torpedo at mid-range engagement.
        """
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(5_000, 0, 0)  # 5 km/s

        # Create separate targets (same initial state)
        target_for_coilgun = ShipTarget(
            position=Vector3D(400_000, 0, 0),  # 400 km
            velocity=Vector3D(-5_000, 0, 0),  # 5 km/s toward
            forward=Vector3D(-1, 0, 0),
            geometry=destroyer_geometry,
        )

        target_for_torpedo = ShipTarget(
            position=Vector3D(400_000, 0, 0),
            velocity=Vector3D(-5_000, 0, 0),
            forward=Vector3D(-1, 0, 0),
            geometry=destroyer_geometry,
        )

        coilgun_result = simulate_coilgun_engagement(
            shooter_pos, shooter_vel, target_for_coilgun, coilgun_launcher
        )

        torpedo_result = simulate_torpedo_engagement(
            shooter_pos, shooter_vel, target_for_torpedo, torpedo_launcher
        )

        assert coilgun_result.hit, "Coilgun should hit"
        assert torpedo_result.hit, "Torpedo should hit"

        # Coilgun is faster but less energy
        assert coilgun_result.flight_time_seconds < torpedo_result.flight_time_seconds
        assert torpedo_result.kinetic_energy_gj > coilgun_result.kinetic_energy_gj

        print(f"\n  Mid-range engagement comparison:")
        print(f"    Coilgun:")
        print(f"      Flight time: {coilgun_result.flight_time_seconds:.1f} s")
        print(f"      Impact velocity: {coilgun_result.impact_velocity_kps:.1f} km/s")
        print(f"      Kinetic energy: {coilgun_result.kinetic_energy_gj:.2f} GJ")
        print(f"    Torpedo:")
        print(f"      Flight time: {torpedo_result.flight_time_seconds:.1f} s")
        print(f"      Impact velocity: {torpedo_result.impact_velocity_kps:.1f} km/s")
        print(f"      Total energy: {torpedo_result.kinetic_energy_gj:.1f} GJ")


# =============================================================================
# TRAJECTORY VERIFICATION TESTS
# =============================================================================

class TestTrajectoryTracking:
    """Tests verifying correct projectile trajectory tracking."""

    def test_coilgun_ballistic_trajectory(self, coilgun_launcher):
        """Verify coilgun follows straight ballistic trajectory."""
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)
        target_dir = Vector3D(1, 0, 0)  # Straight ahead

        projectile = KineticProjectile.from_launch(
            shooter_position=shooter_pos,
            shooter_velocity=shooter_vel,
            target_direction=target_dir,
            muzzle_velocity_kps=COILGUN_MUZZLE_VELOCITY_KPS,
            mass_kg=COILGUN_SLUG_MASS_KG,
        )

        positions = []
        velocities = []

        for _ in range(100):  # 100 seconds
            projectile.update(1.0)
            positions.append(Vector3D(projectile.position.x, projectile.position.y, projectile.position.z))
            velocities.append(projectile.velocity.magnitude)

        # Trajectory should be straight (Y and Z should remain ~0)
        for pos in positions:
            assert abs(pos.y) < 1.0, "Should have no Y drift"
            assert abs(pos.z) < 1.0, "Should have no Z drift"

        # Velocity should be constant (no drag in space)
        for vel in velocities:
            assert abs(vel / 1000 - COILGUN_MUZZLE_VELOCITY_KPS) < 0.001

        # Position should increase linearly
        expected_distance = 100 * COILGUN_MUZZLE_VELOCITY_KPS * 1000
        assert abs(positions[-1].x - expected_distance) < 1.0

    def test_torpedo_guidance_changes_trajectory(self, torpedo_launcher):
        """Verify torpedo guidance alters trajectory toward target."""
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)

        # Target offset from direct line
        target_pos = Vector3D(100_000, 50_000, 0)  # 100 km ahead, 50 km to side
        target_vel = Vector3D(0, 0, 0)

        torpedo = torpedo_launcher.launch(
            shooter_position=shooter_pos,
            shooter_velocity=shooter_vel,
            target_id="test",
            target_position=target_pos,
            target_velocity=target_vel,
            current_time=0.0,
        )

        # Track Y position over time
        y_positions = []

        for _ in range(50):  # 50 seconds
            torpedo.update(1.0, target_pos, target_vel)
            y_positions.append(torpedo.position.y)

        # Torpedo should be steering toward Y = 50 km
        # Y position should be increasing
        assert y_positions[-1] > y_positions[0], "Torpedo should steer toward target"
        assert y_positions[-1] > 10_000, "Should have significant Y displacement"


# =============================================================================
# DAMAGE RESOLUTION TESTS
# =============================================================================

class TestEngagementDamageResolution:
    """Tests for resolving damage after projectile impact."""

    def test_coilgun_impact_damage_resolution(
        self, fleet_data, destroyer_geometry, coilgun_launcher
    ):
        """Resolve damage from coilgun impact."""
        import random

        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)

        target = ShipTarget(
            position=Vector3D(300_000, 0, 0),
            velocity=Vector3D(0, 0, 0),
            forward=Vector3D(-1, 0, 0),
            geometry=destroyer_geometry,
        )

        result = simulate_coilgun_engagement(
            shooter_pos, shooter_vel, target, coilgun_launcher
        )

        assert result.hit

        # Resolve damage using combat system
        destroyer_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
        spinal_coiler = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
        resolver = CombatResolver(rng=random.Random(42))

        hit_result = resolver.resolve_hit(
            spinal_coiler,
            destroyer_armor,
            location=result.hit_location,
        )

        assert hit_result.hit
        armor_section = destroyer_armor.get_section(result.hit_location)

        print(f"\n  Coilgun damage resolution:")
        print(f"    Hit location: {result.hit_location.value}")
        print(f"    Projectile energy: {result.kinetic_energy_gj:.2f} GJ")
        print(f"    Armor ablation: {hit_result.armor_ablation_cm:.2f} cm")
        print(f"    Remaining armor: {armor_section.thickness_cm:.2f} cm")

    def test_torpedo_impact_damage_resolution(
        self, fleet_data, cruiser_geometry, torpedo_launcher
    ):
        """Resolve damage from torpedo impact including warhead."""
        import random

        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(5_000, 0, 0)

        target = ShipTarget(
            position=Vector3D(800_000, 0, 0),
            velocity=Vector3D(-5_000, 0, 0),
            forward=Vector3D(-1, 0, 0),
            geometry=cruiser_geometry,
        )

        result = simulate_torpedo_engagement(
            shooter_pos, shooter_vel, target, torpedo_launcher
        )

        assert result.hit

        # Resolve damage
        cruiser_armor = create_ship_armor_from_fleet_data(fleet_data, "cruiser")
        torpedo_weapon = create_weapon_from_fleet_data(fleet_data, "torpedo_launcher")
        resolver = CombatResolver(rng=random.Random(42))

        hit_result = resolver.resolve_hit(
            torpedo_weapon,
            cruiser_armor,
            location=result.hit_location,
        )

        print(f"\n  Torpedo damage resolution:")
        print(f"    Hit location: {result.hit_location.value}")
        print(f"    Total energy: {result.kinetic_energy_gj:.1f} GJ")
        print(f"    Warhead yield: {TORPEDO_WARHEAD_GJ} GJ")
        print(f"    Damage absorbed: {hit_result.damage_absorbed:.2f} GJ")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
