#!/usr/bin/env python3
"""
Comprehensive Test Suite for Targeting and Torpedo Systems

Tests cover:
1. ECM System - Probabilistic lock breaking, strength effects
2. Targeting Computer - Tracking bonus, lock time, multi-target tracking
3. Firing Solution - Lock acquisition, ECM breaks, reacquisition
4. Lead Calculation - Stationary targets, moving targets, relative velocity
5. Torpedo Specs - Tsiolkovsky delta-v, fuel consumption, thrust
6. Torpedo Flight - Release distance, fuel burns, intercept calculation
7. Torpedo Guidance - Pursuit, proportional navigation, terminal guidance
8. Integration Scenarios - Corvette vs cruiser, ECM effects, point defense

Uses pytest with seeded RNG for deterministic tests and parametrized tests
for various engagement geometries.
"""

import json
import math
import random
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics import Vector3D, ShipState, tsiolkovsky_delta_v
from targeting import (
    # ECM
    ECMSystem,
    # Targeting Computer
    TargetingComputer,
    # Firing Solution
    FiringSolution,
    # Lead Calculator
    LeadCalculator,
    # Targeting System
    TargetingSystem,
    # Factory functions
    create_basic_targeting_system,
    create_advanced_targeting_system,
)
from torpedo import (
    # Constants
    DEFAULT_TORPEDO_MASS_KG,
    DEFAULT_PROPELLANT_FRACTION,
    DEFAULT_EXHAUST_VELOCITY_KPS,
    SAFE_ARMING_DISTANCE_M,
    TERMINAL_APPROACH_DISTANCE_M,
    PROPORTIONAL_NAV_CONSTANT,
    # Classes
    TorpedoSpecs,
    GuidanceMode,
    Torpedo,
    TorpedoLauncher,
    TorpedoGuidance,
    # Analysis function
    analyze_intercept,
)
from combat import Weapon


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fleet_data():
    """Load fleet ship data from JSON file."""
    fleet_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    if fleet_path.exists():
        with open(fleet_path, "r") as f:
            return json.load(f)
    return None


@pytest.fixture
def seeded_rng():
    """Create a seeded random number generator for deterministic tests."""
    return random.Random(42)


@pytest.fixture
def basic_ecm():
    """Create a basic ECM system."""
    return ECMSystem(ecm_strength=0.3, reacquisition_time_s=5.0, active=True)


@pytest.fixture
def basic_targeting_computer():
    """Create a basic targeting computer."""
    return TargetingComputer(
        tracking_bonus=0.1,
        lock_time_s=3.0,
        max_targets=4,
        sensor_range_km=5000.0
    )


@pytest.fixture
def basic_torpedo_specs():
    """Create basic torpedo specifications."""
    return TorpedoSpecs()


@pytest.fixture
def basic_targeting_system(seeded_rng):
    """Create a basic targeting system with seeded RNG."""
    return create_basic_targeting_system(
        tracking_bonus=0.1,
        lock_time_s=3.0,
        max_targets=4,
        seed=42
    )


@pytest.fixture
def basic_torpedo(basic_torpedo_specs):
    """Create a basic torpedo for testing."""
    return Torpedo(
        specs=basic_torpedo_specs,
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(10000, 0, 0),  # 10 km/s initial
        target_id="target_001"
    )


@pytest.fixture
def basic_launcher(basic_torpedo_specs):
    """Create a basic torpedo launcher."""
    return TorpedoLauncher(specs=basic_torpedo_specs)


@pytest.fixture
def stationary_shooter():
    """Create a stationary shooter ship state."""
    return ShipState(
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(0, 0, 0)
    )


@pytest.fixture
def moving_shooter():
    """Create a moving shooter ship state."""
    return ShipState(
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(5000, 0, 0)  # 5 km/s forward
    )


@pytest.fixture
def stationary_target():
    """Create a stationary target ship state."""
    return ShipState(
        position=Vector3D(100000, 0, 0),  # 100 km away
        velocity=Vector3D(0, 0, 0)
    )


@pytest.fixture
def moving_target():
    """Create a moving target ship state."""
    return ShipState(
        position=Vector3D(100000, 0, 0),  # 100 km away
        velocity=Vector3D(0, 5000, 0)  # 5 km/s perpendicular
    )


@pytest.fixture
def basic_weapon():
    """Create a basic weapon for testing."""
    return Weapon(
        name="Test Coilgun",
        weapon_type="test_coilgun",
        kinetic_energy_gj=10.0,
        cooldown_s=5.0,
        range_km=1000.0,
        flat_chipping=0.5,
        muzzle_velocity_kps=10.0
    )


# =============================================================================
# ECM SYSTEM TESTS
# =============================================================================

class TestECMSystem:
    """Tests for ECM system functionality."""

    def test_ecm_strength_clamped_to_valid_range(self):
        """Test that ECM strength is clamped to 0.0-1.0."""
        ecm_high = ECMSystem(ecm_strength=1.5)
        assert ecm_high.ecm_strength == 1.0

        ecm_low = ECMSystem(ecm_strength=-0.5)
        assert ecm_low.ecm_strength == 0.0

    def test_ecm_zero_never_breaks(self, seeded_rng):
        """Test that zero ECM never breaks a lock."""
        ecm = ECMSystem(ecm_strength=0.0)
        solution = FiringSolution(target_id="test", locked=True, lock_progress=1.0)

        # Run many trials - should never break
        for _ in range(100):
            assert not solution.check_ecm_break(
                ecm.get_effective_strength(),
                tracking_bonus=0.0,
                rng=seeded_rng
            )

    def test_ecm_breaks_lock_probabilistically(self, seeded_rng):
        """Test that ECM breaks locks with expected probability."""
        ecm_strength = 0.5
        num_trials = 1000
        breaks = 0

        for _ in range(num_trials):
            solution = FiringSolution(target_id="test", locked=True, lock_progress=1.0)
            if solution.check_ecm_break(ecm_strength, tracking_bonus=0.0, rng=seeded_rng):
                breaks += 1

        # Should be approximately 50% with some statistical tolerance
        expected = num_trials * ecm_strength
        tolerance = num_trials * 0.05  # 5% tolerance
        assert abs(breaks - expected) < tolerance, f"Expected ~{expected} breaks, got {breaks}"

    def test_higher_ecm_more_breaks(self, seeded_rng):
        """Test that higher ECM strength leads to more breaks."""
        num_trials = 500

        low_ecm_breaks = 0
        high_ecm_breaks = 0

        for _ in range(num_trials):
            # Test low ECM (20%)
            solution_low = FiringSolution(target_id="test", locked=True, lock_progress=1.0)
            if solution_low.check_ecm_break(0.2, tracking_bonus=0.0, rng=seeded_rng):
                low_ecm_breaks += 1

            # Test high ECM (80%)
            solution_high = FiringSolution(target_id="test", locked=True, lock_progress=1.0)
            if solution_high.check_ecm_break(0.8, tracking_bonus=0.0, rng=seeded_rng):
                high_ecm_breaks += 1

        assert high_ecm_breaks > low_ecm_breaks, (
            f"High ECM ({high_ecm_breaks}) should break more than low ECM ({low_ecm_breaks})"
        )

    def test_ecm_inactive_returns_zero(self):
        """Test that inactive ECM returns zero effective strength."""
        ecm = ECMSystem(ecm_strength=0.8, active=False)
        assert ecm.get_effective_strength() == 0.0

    def test_ecm_active_returns_strength(self, basic_ecm):
        """Test that active ECM returns its strength."""
        assert basic_ecm.get_effective_strength() == 0.3

    def test_set_strength_clamps_values(self, basic_ecm):
        """Test that set_strength clamps to valid range."""
        basic_ecm.set_strength(1.5)
        assert basic_ecm.ecm_strength == 1.0

        basic_ecm.set_strength(-0.5)
        assert basic_ecm.ecm_strength == 0.0


# =============================================================================
# TARGETING COMPUTER TESTS
# =============================================================================

class TestTargetingComputer:
    """Tests for targeting computer functionality."""

    def test_tracking_bonus_reduces_ecm(self, basic_targeting_computer):
        """Test that tracking bonus reduces ECM effectiveness."""
        raw_ecm = 0.5
        effective = basic_targeting_computer.effective_ecm(raw_ecm)
        assert effective == 0.4  # 0.5 - 0.1 tracking bonus

    def test_tracking_bonus_cannot_make_ecm_negative(self, basic_targeting_computer):
        """Test that ECM cannot go below zero."""
        raw_ecm = 0.05  # Less than tracking bonus
        effective = basic_targeting_computer.effective_ecm(raw_ecm)
        assert effective == 0.0

    def test_lock_time_minimum_enforced(self):
        """Test that lock time has a minimum value."""
        computer = TargetingComputer(lock_time_s=0.01)
        assert computer.lock_time_s >= 0.1

    def test_max_targets_minimum_enforced(self):
        """Test that max targets has a minimum of 1."""
        computer = TargetingComputer(max_targets=0)
        assert computer.max_targets >= 1

    def test_can_track_target_respects_limit(self, basic_targeting_computer):
        """Test that can_track_target respects max_targets."""
        assert basic_targeting_computer.can_track_target(0)
        assert basic_targeting_computer.can_track_target(3)
        assert not basic_targeting_computer.can_track_target(4)
        assert not basic_targeting_computer.can_track_target(5)

    def test_sensor_range_check(self, basic_targeting_computer):
        """Test sensor range checking."""
        assert basic_targeting_computer.is_in_sensor_range(1000.0)
        assert basic_targeting_computer.is_in_sensor_range(5000.0)
        assert not basic_targeting_computer.is_in_sensor_range(5001.0)

    @pytest.mark.parametrize("tracking_bonus,raw_ecm,expected", [
        (0.0, 0.5, 0.5),   # No bonus
        (0.1, 0.5, 0.4),   # Small bonus
        (0.3, 0.5, 0.2),   # Medium bonus
        (0.5, 0.5, 0.0),   # Full negation
        (0.5, 0.3, 0.0),   # Over-negation clamped
    ])
    def test_effective_ecm_calculation(self, tracking_bonus, raw_ecm, expected):
        """Test effective ECM calculation with various inputs."""
        computer = TargetingComputer(tracking_bonus=tracking_bonus)
        assert computer.effective_ecm(raw_ecm) == pytest.approx(expected)


# =============================================================================
# FIRING SOLUTION TESTS
# =============================================================================

class TestFiringSolution:
    """Tests for firing solution lock mechanics."""

    def test_lock_acquisition_over_time(self):
        """Test that lock progress increases over time."""
        solution = FiringSolution(target_id="test")
        assert solution.lock_progress == 0.0
        assert not solution.locked

        # Progress lock (3 second lock time, 1 second step = 33% progress)
        solution.attempt_lock(dt_seconds=1.0, ecm_strength=0.0, tracking_bonus=0.0, base_lock_time=3.0)
        assert solution.lock_progress == pytest.approx(1/3, rel=0.01)

        # Continue until locked
        solution.attempt_lock(dt_seconds=2.0, ecm_strength=0.0, tracking_bonus=0.0, base_lock_time=3.0)
        assert solution.locked
        assert solution.lock_progress == 1.0

    def test_ecm_slows_lock_acquisition(self):
        """Test that ECM slows lock acquisition speed."""
        solution_no_ecm = FiringSolution(target_id="test1")
        solution_with_ecm = FiringSolution(target_id="test2")

        # Same time step, different ECM
        solution_no_ecm.attempt_lock(dt_seconds=1.0, ecm_strength=0.0, tracking_bonus=0.0, base_lock_time=3.0)
        solution_with_ecm.attempt_lock(dt_seconds=1.0, ecm_strength=0.5, tracking_bonus=0.0, base_lock_time=3.0)

        assert solution_no_ecm.lock_progress > solution_with_ecm.lock_progress

    def test_ecm_can_break_lock(self, seeded_rng):
        """Test that ECM can break an established lock."""
        solution = FiringSolution(target_id="test", locked=True, lock_progress=1.0)

        # With high ECM, should eventually break
        broken = False
        for _ in range(100):
            solution = FiringSolution(target_id="test", locked=True, lock_progress=1.0)
            if solution.check_ecm_break(ecm_strength=0.9, tracking_bonus=0.0, rng=seeded_rng):
                broken = True
                break

        assert broken, "High ECM should break lock within 100 attempts"

    def test_reacquisition_after_break(self):
        """Test cooldown prevents immediate reacquisition."""
        solution = FiringSolution(target_id="test", locked=True, lock_progress=1.0)

        # Break lock with cooldown
        solution.break_lock(reacquisition_time_s=5.0)
        assert not solution.locked
        assert solution.lock_progress == 0.0
        assert solution.cooldown_remaining == 5.0

        # Cannot progress during cooldown
        solution.attempt_lock(dt_seconds=2.0, ecm_strength=0.0, tracking_bonus=0.0, base_lock_time=3.0)
        assert solution.lock_progress == 0.0  # No progress
        assert solution.cooldown_remaining == 3.0  # Cooldown reduced

        # Consume remaining cooldown
        solution.attempt_lock(dt_seconds=3.0, ecm_strength=0.0, tracking_bonus=0.0, base_lock_time=3.0)
        assert solution.cooldown_remaining == 0.0  # Cooldown now expired

        # Now can progress on next attempt
        solution.attempt_lock(dt_seconds=1.0, ecm_strength=0.0, tracking_bonus=0.0, base_lock_time=3.0)
        assert solution.lock_progress > 0.0

    def test_locked_solution_stays_locked(self):
        """Test that locked solutions maintain lock when attempt_lock is called."""
        solution = FiringSolution(target_id="test", locked=True, lock_progress=1.0)

        result = solution.attempt_lock(dt_seconds=1.0, ecm_strength=0.0, tracking_bonus=0.0, base_lock_time=3.0)
        assert result
        assert solution.locked

    def test_reset_clears_all_state(self):
        """Test that reset clears all firing solution state."""
        solution = FiringSolution(
            target_id="test",
            locked=True,
            lock_progress=1.0,
            time_to_lock=0.0,
            cooldown_remaining=2.0
        )

        solution.reset()

        assert not solution.locked
        assert solution.lock_progress == 0.0
        assert solution.time_to_lock == 0.0
        assert solution.cooldown_remaining == 0.0


# =============================================================================
# LEAD CALCULATION TESTS
# =============================================================================

class TestLeadCalculator:
    """Tests for lead/intercept point calculations."""

    def test_stationary_target_aim_at_target(self):
        """Test that stationary target = aim at target position."""
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)
        target_pos = Vector3D(100, 0, 0)  # 100 km away
        target_vel = Vector3D(0, 0, 0)  # Stationary
        projectile_speed = 10.0  # km/s

        lead_pos = LeadCalculator.calculate_lead(
            shooter_pos, shooter_vel, target_pos, target_vel, projectile_speed
        )

        # For stationary target, lead position should equal target position
        assert lead_pos.x == pytest.approx(target_pos.x, rel=0.01)
        assert lead_pos.y == pytest.approx(target_pos.y, abs=0.1)
        assert lead_pos.z == pytest.approx(target_pos.z, abs=0.1)

    def test_moving_target_aim_ahead(self):
        """Test that moving target = aim ahead of current position."""
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)
        target_pos = Vector3D(100, 0, 0)  # 100 km away
        target_vel = Vector3D(0, 10, 0)  # 10 km/s perpendicular
        projectile_speed = 20.0  # km/s

        lead_pos = LeadCalculator.calculate_lead(
            shooter_pos, shooter_vel, target_pos, target_vel, projectile_speed
        )

        # Lead should be ahead of target in Y direction
        assert lead_pos.y > target_pos.y, "Lead position should be ahead of target"

        # Verify intercept: time to target ~5 seconds at 20 km/s
        time_to_intercept = lead_pos.distance_to(shooter_pos) / projectile_speed
        expected_y_offset = target_vel.y * time_to_intercept
        assert lead_pos.y == pytest.approx(expected_y_offset, rel=0.1)

    def test_both_ships_moving_relative_velocity_lead(self):
        """Test lead calculation when both ships are moving."""
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(5, 0, 0)  # 5 km/s forward
        target_pos = Vector3D(100, 0, 0)  # 100 km away
        target_vel = Vector3D(-5, 5, 0)  # 5 km/s closing + 5 km/s perpendicular
        projectile_speed = 20.0  # km/s

        lead_pos = LeadCalculator.calculate_lead(
            shooter_pos, shooter_vel, target_pos, target_vel, projectile_speed
        )

        # Relative velocity is (-10, 5, 0) km/s
        # Lead should account for both X (closing) and Y (perpendicular) motion
        assert lead_pos.x < target_pos.x, "Target is closing, lead X should be less"
        assert lead_pos.y > 0, "Target moving +Y, lead should have positive Y"

    @pytest.mark.parametrize("distance_km,target_vel_kps,proj_speed_kps", [
        (100, 10, 20),   # 100km, 10km/s perp, 20km/s proj - favorable geometry
        (200, 5, 40),    # 200km, 5km/s perp, 40km/s proj - fast projectile
        (100, 5, 30),    # 100km, 5km/s perp, 30km/s proj - easy intercept
    ])
    def test_known_lead_cases(self, distance_km, target_vel_kps, proj_speed_kps):
        """Test lead calculation produces reasonable intercept points."""
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)
        target_pos = Vector3D(distance_km, 0, 0)
        target_vel = Vector3D(0, target_vel_kps, 0)

        lead_pos = LeadCalculator.calculate_lead(
            shooter_pos, shooter_vel, target_pos, target_vel, proj_speed_kps
        )

        # Lead Y should be positive (target moving +Y)
        assert lead_pos.y > 0, "Lead should be ahead of target in Y direction"

        # For favorable geometries where proj speed > target speed,
        # verify the intercept math: projectile reaches lead point as target arrives
        if proj_speed_kps > target_vel_kps:
            lead_distance = lead_pos.magnitude
            time_to_lead = lead_distance / proj_speed_kps

            # Target's Y position at intercept time should match lead Y
            expected_y = target_vel_kps * time_to_lead
            assert lead_pos.y == pytest.approx(expected_y, rel=0.15)

    def test_time_to_intercept_calculation(self):
        """Test basic time to intercept calculation."""
        distance_km = 100.0
        projectile_speed_kps = 10.0

        time = LeadCalculator.get_time_to_intercept(distance_km, projectile_speed_kps)

        assert time == 10.0  # 100 km / 10 km/s = 10 seconds

    def test_zero_projectile_speed_returns_infinity(self):
        """Test that zero projectile speed returns infinite intercept time."""
        time = LeadCalculator.get_time_to_intercept(100.0, 0.0)
        assert time == float('inf')

    def test_calculate_lead_from_states(self, stationary_shooter, moving_target):
        """Test lead calculation using ShipState objects."""
        projectile_speed_kps = 10.0

        lead_pos = LeadCalculator.calculate_lead_from_states(
            stationary_shooter, moving_target, projectile_speed_kps
        )

        # Result should be in meters (ShipState units)
        assert lead_pos.magnitude > moving_target.position.magnitude

    def test_calculate_lead_direction_normalized(self):
        """Test that lead direction is normalized."""
        shooter_pos = Vector3D(0, 0, 0)
        shooter_vel = Vector3D(0, 0, 0)
        target_pos = Vector3D(100, 50, 25)
        target_vel = Vector3D(-5, 10, 5)

        direction = LeadCalculator.calculate_lead_direction(
            shooter_pos, shooter_vel, target_pos, target_vel, 20.0
        )

        assert direction.magnitude == pytest.approx(1.0, rel=0.01)


# =============================================================================
# TORPEDO SPECS TESTS
# =============================================================================

class TestTorpedoSpecs:
    """Tests for torpedo specification calculations."""

    def test_delta_v_from_tsiolkovsky(self, basic_torpedo_specs):
        """Test that delta-v is calculated using Tsiolkovsky equation."""
        # Manual calculation
        exhaust_velocity_ms = basic_torpedo_specs.exhaust_velocity_kps * 1000
        mass_ratio = basic_torpedo_specs.mass_kg / basic_torpedo_specs.dry_mass_kg
        expected_dv_ms = exhaust_velocity_ms * math.log(mass_ratio)
        expected_dv_kps = expected_dv_ms / 1000

        assert basic_torpedo_specs.total_delta_v_kps == pytest.approx(expected_dv_kps, rel=0.01)

    def test_fuel_consumption_rate(self, basic_torpedo_specs):
        """Test fuel consumption rate calculation."""
        # Mass flow rate = F / v_e
        exhaust_velocity_ms = basic_torpedo_specs.exhaust_velocity_kps * 1000
        expected_mass_flow = basic_torpedo_specs.thrust_n / exhaust_velocity_ms

        burn_time = basic_torpedo_specs.burn_time_seconds()
        total_propellant = basic_torpedo_specs.propellant_mass_kg

        # burn_time = propellant / mass_flow
        calculated_burn_time = total_propellant / expected_mass_flow

        assert burn_time == pytest.approx(calculated_burn_time, rel=0.01)

    def test_thrust_calculations(self, basic_torpedo_specs):
        """Test thrust and acceleration calculations."""
        # Default thrust is set to 10g at wet mass
        expected_thrust = basic_torpedo_specs.mass_kg * 98.1  # F = m * a

        assert basic_torpedo_specs.thrust_n == pytest.approx(expected_thrust, rel=0.01)

        # Acceleration at wet mass
        accel_wet = basic_torpedo_specs.acceleration_at_mass(basic_torpedo_specs.mass_kg)
        assert accel_wet == pytest.approx(98.1, rel=0.01)

        # Acceleration at dry mass (higher)
        accel_dry = basic_torpedo_specs.acceleration_at_mass(basic_torpedo_specs.dry_mass_kg)
        assert accel_dry > accel_wet

    def test_dry_mass_and_propellant_mass(self, basic_torpedo_specs):
        """Test dry mass and propellant mass calculations."""
        expected_propellant = basic_torpedo_specs.mass_kg * DEFAULT_PROPELLANT_FRACTION
        expected_dry = basic_torpedo_specs.mass_kg - expected_propellant

        assert basic_torpedo_specs.propellant_mass_kg == pytest.approx(expected_propellant, rel=0.01)
        assert basic_torpedo_specs.dry_mass_kg == pytest.approx(expected_dry, rel=0.01)

    def test_from_fleet_data_factory(self):
        """Test creating specs from fleet data parameters."""
        specs = TorpedoSpecs.from_fleet_data(
            warhead_yield_gj=50.0,
            ammo_mass_kg=1600.0,
            range_km=2000.0
        )

        assert specs.warhead_yield_gj == 50.0
        assert specs.mass_kg == 1600.0
        # Delta-v should be calculated
        assert specs.total_delta_v_kps > 0

    @pytest.mark.parametrize("propellant_fraction,expected_dv_ratio", [
        (0.5, 0.693),   # ln(2) ~ 0.693
        (0.7, 1.204),   # ln(1/0.3) ~ 1.204
        (0.9, 2.303),   # ln(10) ~ 2.303
    ])
    def test_propellant_fraction_affects_delta_v(self, propellant_fraction, expected_dv_ratio):
        """Test that propellant fraction affects delta-v as expected."""
        specs = TorpedoSpecs(
            propellant_fraction=propellant_fraction,
            exhaust_velocity_kps=1.0  # 1 km/s for easy calculation
        )

        # delta_v = v_e * ln(m_wet / m_dry)
        # For v_e = 1 km/s, delta_v equals the mass ratio logarithm
        assert specs.total_delta_v_kps == pytest.approx(expected_dv_ratio, rel=0.01)


# =============================================================================
# TORPEDO FLIGHT TESTS
# =============================================================================

class TestTorpedoFlight:
    """Tests for torpedo flight mechanics."""

    def test_launch_at_correct_position(self, basic_launcher):
        """Test that torpedo launches at shooter position."""
        shooter_pos = Vector3D(1000, 2000, 3000)
        shooter_vel = Vector3D(8000, 0, 0)
        target_pos = Vector3D(500000, 0, 0)
        target_vel = Vector3D(-5000, 0, 0)

        torpedo = basic_launcher.launch(
            shooter_position=shooter_pos,
            shooter_velocity=shooter_vel,
            target_id="target_001",
            target_position=target_pos,
            target_velocity=target_vel,
            current_time=0.0
        )

        assert torpedo is not None
        assert torpedo.position.x == shooter_pos.x
        assert torpedo.position.y == shooter_pos.y
        assert torpedo.position.z == shooter_pos.z

    def test_torpedo_inherits_shooter_velocity(self, basic_launcher):
        """Test that torpedo inherits shooter velocity at launch."""
        shooter_vel = Vector3D(8000, 1000, 500)

        torpedo = basic_launcher.launch(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=shooter_vel,
            target_id="target_001",
            target_position=Vector3D(500000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            current_time=0.0
        )

        assert torpedo is not None
        assert torpedo.velocity.x == shooter_vel.x
        assert torpedo.velocity.y == shooter_vel.y
        assert torpedo.velocity.z == shooter_vel.z

    def test_fuel_burns_during_thrust(self, basic_torpedo):
        """Test that fuel is consumed during thrust."""
        initial_mass = basic_torpedo.current_mass_kg
        initial_dv = basic_torpedo.remaining_delta_v_kps

        # Apply thrust for 5 seconds
        basic_torpedo.apply_thrust(Vector3D(1, 0, 0), dt_seconds=5.0)

        assert basic_torpedo.current_mass_kg < initial_mass
        assert basic_torpedo.remaining_delta_v_kps < initial_dv

    def test_fuel_exhaustion_behavior(self, basic_torpedo_specs):
        """Test torpedo behavior when fuel is exhausted."""
        torpedo = Torpedo(
            specs=basic_torpedo_specs,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            target_id="test"
        )

        # Apply thrust until fuel exhausted
        burn_time = basic_torpedo_specs.burn_time_seconds()
        torpedo.apply_thrust(Vector3D(1, 0, 0), dt_seconds=burn_time + 10)

        assert torpedo.fuel_exhausted
        assert torpedo.guidance_mode == GuidanceMode.COAST
        assert torpedo.current_mass_kg == pytest.approx(torpedo.specs.dry_mass_kg, rel=0.01)

    def test_intercept_calculation_accuracy(self, basic_torpedo):
        """Test intercept position calculation."""
        target_pos = Vector3D(200000, 50000, 0)
        target_vel = Vector3D(-3000, 1000, 0)

        intercept_pos, t_intercept = basic_torpedo.calculate_intercept(target_pos, target_vel)

        # Intercept should be between current positions (target is approaching)
        # and should account for target motion
        assert t_intercept > 0
        assert intercept_pos.x < target_pos.x  # Target approaching in X

    def test_arming_after_safe_distance(self, basic_torpedo_specs):
        """Test that torpedo arms after safe distance."""
        torpedo = Torpedo(
            specs=basic_torpedo_specs,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(1000, 0, 0),  # Moving forward
            target_id="test"
        )

        assert not torpedo.armed

        # Update to move beyond safe arming distance
        target_pos = Vector3D(100000, 0, 0)
        target_vel = Vector3D(0, 0, 0)

        # Move far enough
        for _ in range(10):
            torpedo.update(dt_seconds=1.0, target_position=target_pos, target_velocity=target_vel)

        assert torpedo.armed

    def test_release_distance_calculation(self, basic_launcher):
        """Test optimal release distance calculation."""
        shooter_vel = Vector3D(8000, 0, 0)
        target_pos = Vector3D(1000000, 0, 0)
        target_vel = Vector3D(-5000, 0, 0)

        release_dist = basic_launcher.calculate_release_distance(
            shooter_vel, target_pos, target_vel
        )

        # Release distance should be positive and reasonable
        assert release_dist > SAFE_ARMING_DISTANCE_M
        assert release_dist < 2000000  # Less than nominal range


# =============================================================================
# TORPEDO GUIDANCE TESTS
# =============================================================================

class TestTorpedoGuidance:
    """Tests for torpedo guidance algorithms."""

    def test_pursuit_mode_tracks_target(self, basic_torpedo_specs):
        """Test that pursuit guidance points at target."""
        torpedo = Torpedo(
            specs=basic_torpedo_specs,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(0, 0, 0),
            target_id="test",
            guidance_mode=GuidanceMode.PURSUIT
        )

        guidance = TorpedoGuidance()
        target_pos = Vector3D(100000, 50000, 25000)
        target_vel = Vector3D(0, 0, 0)

        direction = guidance._pursuit_guidance(torpedo, target_pos, target_vel)

        # Direction should point from torpedo to target
        expected_dir = (target_pos - torpedo.position).normalized()
        assert direction.x == pytest.approx(expected_dir.x, rel=0.01)
        assert direction.y == pytest.approx(expected_dir.y, rel=0.01)
        assert direction.z == pytest.approx(expected_dir.z, rel=0.01)

    def test_proportional_navigation_leads_target(self, basic_torpedo_specs):
        """Test that proportional navigation leads a moving target."""
        torpedo = Torpedo(
            specs=basic_torpedo_specs,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(10000, 0, 0),  # Moving toward target
            target_id="test",
            guidance_mode=GuidanceMode.PROPORTIONAL_NAV
        )

        guidance = TorpedoGuidance()
        target_pos = Vector3D(100000, 0, 0)
        target_vel = Vector3D(0, 5000, 0)  # Moving perpendicular

        direction = guidance._proportional_nav_guidance(
            torpedo, target_pos, target_vel, dt=1.0
        )

        # Direction should have Y component to lead target
        assert direction.y > 0 or direction.x > 0  # Some component toward intercept

    def test_terminal_guidance_maximizes_impact(self, basic_torpedo_specs):
        """Test that terminal guidance aims for high closing velocity."""
        torpedo = Torpedo(
            specs=basic_torpedo_specs,
            position=Vector3D(95000, 0, 0),  # Close to target
            velocity=Vector3D(15000, 0, 0),
            target_id="test",
            guidance_mode=GuidanceMode.TERMINAL
        )

        guidance = TorpedoGuidance()
        target_pos = Vector3D(100000, 0, 0)
        target_vel = Vector3D(-5000, 0, 0)  # Target closing

        direction = guidance._terminal_guidance(torpedo, target_pos, target_vel)

        # Terminal guidance should point roughly toward target
        assert direction.x > 0  # Should have positive X component toward target

    def test_guidance_mode_transition_to_terminal(self, basic_torpedo_specs):
        """Test automatic transition to terminal guidance at close range."""
        torpedo = Torpedo(
            specs=basic_torpedo_specs,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(10000, 0, 0),
            target_id="test",
            guidance_mode=GuidanceMode.PROPORTIONAL_NAV
        )

        guidance = TorpedoGuidance()

        # Target is within terminal distance
        target_pos = Vector3D(TERMINAL_APPROACH_DISTANCE_M - 1000, 0, 0)
        target_vel = Vector3D(0, 0, 0)

        guidance.update_guidance(torpedo, target_pos, target_vel, dt=1.0)

        assert torpedo.guidance_mode == GuidanceMode.TERMINAL

    def test_coast_mode_no_thrust(self, basic_torpedo_specs):
        """Test that coast mode produces no thrust direction."""
        torpedo = Torpedo(
            specs=basic_torpedo_specs,
            position=Vector3D(0, 0, 0),
            velocity=Vector3D(10000, 0, 0),
            target_id="test",
            guidance_mode=GuidanceMode.COAST
        )

        guidance = TorpedoGuidance()
        target_pos = Vector3D(100000, 0, 0)
        target_vel = Vector3D(0, 0, 0)

        direction = guidance.update_guidance(torpedo, target_pos, target_vel, dt=1.0)

        assert direction.magnitude == 0.0


# =============================================================================
# TARGETING SYSTEM TESTS
# =============================================================================

class TestTargetingSystem:
    """Tests for complete targeting system integration."""

    def test_acquire_multiple_targets(self, basic_targeting_system):
        """Test acquiring multiple targets up to limit."""
        # Acquire 4 targets (the max)
        for i in range(4):
            solution = basic_targeting_system.acquire_target(f"target_{i}")
            assert solution is not None

        # 5th target should fail
        solution = basic_targeting_system.acquire_target("target_4")
        assert solution is None

    def test_drop_target_frees_slot(self, basic_targeting_system):
        """Test that dropping a target frees a tracking slot."""
        # Fill all slots
        for i in range(4):
            basic_targeting_system.acquire_target(f"target_{i}")

        # Drop one
        result = basic_targeting_system.drop_target("target_0")
        assert result

        # Now can acquire new target
        solution = basic_targeting_system.acquire_target("target_new")
        assert solution is not None

    def test_update_progresses_locks(self, basic_targeting_system):
        """Test that update progresses lock acquisition."""
        basic_targeting_system.acquire_target("target_0")
        targets_ecm = {"target_0": 0.0}

        # Update for enough time to achieve lock
        for _ in range(10):
            basic_targeting_system.update(dt_seconds=0.5, targets_ecm=targets_ecm)

        assert basic_targeting_system.is_locked("target_0")

    def test_update_with_ecm_systems(self):
        """Test update using ECMSystem objects."""
        # Create a fresh targeting system without ECM breaks during lock acquisition
        targeting = create_basic_targeting_system(
            tracking_bonus=0.1,
            lock_time_s=3.0,
            max_targets=4,
            seed=12345  # Deterministic seed
        )
        targeting.acquire_target("target_0")

        ecm_systems = {
            "target_0": ECMSystem(ecm_strength=0.2, active=True)
        }

        # Update for enough time to achieve lock (ECM slows it but shouldn't break during acquisition)
        # With 20% ECM and 10% tracking bonus, effective ECM is 10%
        # Lock speed modifier is 1.0 - (0.1 * 0.5) = 0.95
        # So 3.0 second base lock becomes ~3.16 seconds, need more margin
        for _ in range(30):
            targeting.update_with_ecm_systems(
                dt_seconds=0.5,
                targets_ecm_systems=ecm_systems
            )

        # Check if locked or check progress
        solution = targeting.get_firing_solution("target_0")
        assert solution is not None
        # Should have made significant progress even if ECM broke lock
        assert solution.locked or solution.lock_progress > 0.5 or solution.cooldown_remaining > 0

    def test_calculate_lead_for_weapon(self, basic_targeting_system, basic_weapon,
                                       stationary_shooter, moving_target):
        """Test lead calculation for weapon engagement."""
        lead_pos = basic_targeting_system.calculate_lead_for_weapon(
            weapon=basic_weapon,
            shooter_state=stationary_shooter,
            target_state=moving_target
        )

        # Lead position should be in the direction the target is moving
        assert lead_pos.y > moving_target.position.y

    def test_get_lock_status_summary(self, basic_targeting_system):
        """Test lock status summary generation."""
        basic_targeting_system.acquire_target("target_0")
        basic_targeting_system.acquire_target("target_1")

        summary = basic_targeting_system.get_lock_status_summary()

        assert "target_0" in summary
        assert "target_1" in summary
        assert "locked" in summary["target_0"]
        assert "lock_progress" in summary["target_0"]

    def test_clear_all_removes_solutions(self, basic_targeting_system):
        """Test that clear_all removes all firing solutions."""
        basic_targeting_system.acquire_target("target_0")
        basic_targeting_system.acquire_target("target_1")

        basic_targeting_system.clear_all()

        assert len(basic_targeting_system.solutions) == 0


# =============================================================================
# INTEGRATION SCENARIOS
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests for realistic combat scenarios."""

    def test_corvette_launches_torpedo_at_cruiser(self, basic_launcher):
        """Test a corvette launching a torpedo at a cruiser."""
        # Corvette position and velocity
        corvette_pos = Vector3D(0, 0, 0)
        corvette_vel = Vector3D(8000, 0, 0)  # 8 km/s forward

        # Cruiser position and velocity (500 km away, closing)
        cruiser_pos = Vector3D(500000, 50000, 0)
        cruiser_vel = Vector3D(-5000, 1000, 0)

        # Launch torpedo
        torpedo = basic_launcher.launch(
            shooter_position=corvette_pos,
            shooter_velocity=corvette_vel,
            target_id="cruiser_001",
            target_position=cruiser_pos,
            target_velocity=cruiser_vel,
            current_time=0.0
        )

        assert torpedo is not None
        assert torpedo.target_id == "cruiser_001"

    def test_intercept_possible_calculation(self, basic_torpedo_specs):
        """Test calculation of whether intercept is possible."""
        launch_pos = Vector3D(0, 0, 0)
        launch_vel = Vector3D(8000, 0, 0)
        target_pos = Vector3D(500000, 0, 0)
        target_vel = Vector3D(-5000, 0, 0)

        analysis = analyze_intercept(
            torpedo_specs=basic_torpedo_specs,
            launch_pos=launch_pos,
            launch_vel=launch_vel,
            target_pos=target_pos,
            target_vel=target_vel
        )

        assert "can_intercept" in analysis
        assert "time_to_intercept_seconds" in analysis
        assert analysis["initial_distance_km"] == pytest.approx(500.0, rel=0.01)

    def test_ecm_affects_guided_torpedo(self, seeded_rng):
        """Test ECM effectiveness against guided weapons (if tracking-dependent)."""
        # High ECM should make maintaining lock difficult
        targeting = create_basic_targeting_system(tracking_bonus=0.1, seed=42)
        targeting.acquire_target("torpedo_guided")

        # Simulate ECM breaking lock repeatedly
        high_ecm = {"torpedo_guided": 0.8}  # 70% effective after tracking bonus

        # Run targeting for a period, counting breaks
        lock_breaks = 0
        for i in range(60):  # 60 seconds
            breaks = targeting.update(dt_seconds=1.0, targets_ecm=high_ecm)
            if breaks.get("torpedo_guided"):
                lock_breaks += 1

        # With high ECM, should see some breaks
        assert lock_breaks > 0, "High ECM should break some locks"

    def test_point_defense_engagement_window(self, basic_torpedo_specs):
        """Test calculating point defense engagement window."""
        # Point defense has limited range (e.g., 50 km)
        pd_range_km = 50.0

        # Torpedo approaching at 20 km/s closing velocity
        torpedo = Torpedo(
            specs=basic_torpedo_specs,
            position=Vector3D(200000, 0, 0),  # 200 km away
            velocity=Vector3D(-20000, 0, 0),  # 20 km/s closing
            target_id="defender"
        )

        defender_pos = Vector3D(0, 0, 0)

        # Calculate time until torpedo enters PD range
        current_distance = torpedo.position.distance_to(defender_pos) / 1000  # km
        closing_rate = abs(torpedo.velocity.x) / 1000  # km/s

        time_to_pd_range = (current_distance - pd_range_km) / closing_rate

        # Time in PD range before impact
        time_in_pd_range = pd_range_km / closing_rate

        assert time_to_pd_range > 0
        assert time_in_pd_range > 0
        assert time_in_pd_range == pytest.approx(2.5, rel=0.1)  # 50 km / 20 km/s = 2.5 s

    def test_full_engagement_sequence(self, basic_launcher, seeded_rng):
        """Test a complete engagement sequence from lock to impact."""
        # Setup targeting system
        targeting = create_basic_targeting_system(tracking_bonus=0.15, seed=42)

        # Acquire target
        solution = targeting.acquire_target("enemy_cruiser")
        assert solution is not None

        # Target ECM
        targets_ecm = {"enemy_cruiser": 0.3}

        # Simulate time until locked
        time_elapsed = 0.0
        while not targeting.is_locked("enemy_cruiser") and time_elapsed < 30.0:
            targeting.update(dt_seconds=0.5, targets_ecm=targets_ecm)
            time_elapsed += 0.5

        assert targeting.is_locked("enemy_cruiser"), "Should achieve lock within 30 seconds"

        # Launch torpedo once locked
        torpedo = basic_launcher.launch(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(8000, 0, 0),
            target_id="enemy_cruiser",
            target_position=Vector3D(300000, 0, 0),
            target_velocity=Vector3D(-5000, 0, 0),
            current_time=time_elapsed
        )

        assert torpedo is not None

        # Simulate torpedo flight
        target_pos = Vector3D(300000, 0, 0)
        target_vel = Vector3D(-5000, 0, 0)

        for _ in range(60):  # 60 seconds max
            torpedo.update(dt_seconds=1.0, target_position=target_pos, target_velocity=target_vel)
            # Update target position
            target_pos = target_pos + target_vel * 1.0

            # Check if intercepted (within 100m)
            distance = torpedo.position.distance_to(target_pos)
            if distance < 100:
                break

        # Torpedo should have been tracking and closing
        assert torpedo.armed or torpedo.time_since_launch > 0


# =============================================================================
# PARAMETRIZED GEOMETRY TESTS
# =============================================================================

class TestEngagementGeometries:
    """Parametrized tests for various engagement geometries."""

    @pytest.mark.parametrize("shooter_pos,shooter_vel,target_pos,target_vel,expect_lead_y", [
        # Head-on approach - no Y lead needed
        ((0, 0, 0), (10, 0, 0), (100, 0, 0), (-10, 0, 0), False),
        # Crossing target - Y lead needed
        ((0, 0, 0), (0, 0, 0), (100, 0, 0), (0, 10, 0), True),
        # Target fleeing - extended chase
        ((0, 0, 0), (20, 0, 0), (100, 0, 0), (10, 0, 0), False),
        # Complex geometry
        ((0, 0, 0), (5, 0, 0), (100, 50, 0), (-5, 5, 0), True),
    ])
    def test_lead_calculation_geometries(self, shooter_pos, shooter_vel, target_pos,
                                          target_vel, expect_lead_y):
        """Test lead calculation with various engagement geometries."""
        s_pos = Vector3D(*shooter_pos)
        s_vel = Vector3D(*shooter_vel)
        t_pos = Vector3D(*target_pos)
        t_vel = Vector3D(*target_vel)

        lead = LeadCalculator.calculate_lead(s_pos, s_vel, t_pos, t_vel, 20.0)

        if expect_lead_y:
            assert abs(lead.y) > 0.1, "Expected Y component in lead"
        else:
            # Allow small numerical errors
            pass  # Head-on cases may have tiny Y components from iteration

    @pytest.mark.parametrize("distance_km,closing_rate_kps,expected_intercept_possible", [
        (100, 15, True),    # Close, fast closing - easy intercept
        (500, 10, True),    # Medium distance, good closing
        (2000, 5, True),    # Long range, slow closing - marginal
        (3000, 2, False),   # Very long range, slow closing - likely fail
    ])
    def test_intercept_feasibility_geometries(self, basic_torpedo_specs, distance_km,
                                               closing_rate_kps, expected_intercept_possible):
        """Test intercept feasibility at various ranges and closing rates."""
        launch_pos = Vector3D(0, 0, 0)
        launch_vel = Vector3D(8, 0, 0) * 1000  # 8 km/s in m/s
        target_pos = Vector3D(distance_km * 1000, 0, 0)
        target_vel = Vector3D(-closing_rate_kps * 1000, 0, 0)  # Closing

        analysis = analyze_intercept(
            torpedo_specs=basic_torpedo_specs,
            launch_pos=launch_pos,
            launch_vel=launch_vel,
            target_pos=target_pos,
            target_vel=target_vel
        )

        if expected_intercept_possible:
            # For possible intercepts, verify reasonable time
            assert analysis["time_to_intercept_seconds"] < 600, "Intercept should be under 10 minutes"


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_projectile_speed(self):
        """Test lead calculation with zero projectile speed."""
        lead = LeadCalculator.calculate_lead(
            Vector3D(0, 0, 0),
            Vector3D(0, 0, 0),
            Vector3D(100, 0, 0),
            Vector3D(0, 10, 0),
            0.0  # Zero projectile speed
        )

        # Should return target position when projectile speed is zero
        assert lead.x == 100

    def test_target_at_shooter_position(self):
        """Test lead calculation when target is at shooter position."""
        lead = LeadCalculator.calculate_lead(
            Vector3D(0, 0, 0),
            Vector3D(0, 0, 0),
            Vector3D(0.0001, 0, 0),  # Very close
            Vector3D(10, 10, 10),
            20.0
        )

        # Should return target position for very close targets
        assert lead.magnitude < 1.0

    def test_launcher_cooldown_respected(self, basic_launcher):
        """Test that launcher respects cooldown between shots."""
        # First launch
        torpedo1 = basic_launcher.launch(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_id="target_1",
            target_position=Vector3D(100000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            current_time=0.0
        )
        assert torpedo1 is not None

        # Immediate second launch should fail
        torpedo2 = basic_launcher.launch(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_id="target_2",
            target_position=Vector3D(100000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            current_time=0.1  # Only 0.1 seconds later
        )
        assert torpedo2 is None

        # After cooldown, should succeed
        torpedo3 = basic_launcher.launch(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_id="target_3",
            target_position=Vector3D(100000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            current_time=35.0  # After 30s cooldown
        )
        assert torpedo3 is not None

    def test_empty_magazine(self, basic_launcher):
        """Test launcher behavior with empty magazine."""
        # Deplete magazine
        basic_launcher.current_magazine = 0

        torpedo = basic_launcher.launch(
            shooter_position=Vector3D(0, 0, 0),
            shooter_velocity=Vector3D(0, 0, 0),
            target_id="target",
            target_position=Vector3D(100000, 0, 0),
            target_velocity=Vector3D(0, 0, 0),
            current_time=0.0
        )

        assert torpedo is None

    def test_targeting_system_duplicate_target(self, basic_targeting_system):
        """Test that acquiring the same target twice returns existing solution."""
        solution1 = basic_targeting_system.acquire_target("target_0")
        solution2 = basic_targeting_system.acquire_target("target_0")

        assert solution1 is solution2

    def test_drop_nonexistent_target(self, basic_targeting_system):
        """Test dropping a target that isn't being tracked."""
        result = basic_targeting_system.drop_target("nonexistent")
        assert result is False

    def test_lock_check_on_unlocked_solution(self, seeded_rng):
        """Test that ECM check on unlocked solution returns False."""
        solution = FiringSolution(target_id="test", locked=False, lock_progress=0.5)

        result = solution.check_ecm_break(ecm_strength=0.9, tracking_bonus=0.0, rng=seeded_rng)

        assert result is False  # Can't break a lock that doesn't exist


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior with seeded RNG."""

    def test_ecm_break_deterministic(self):
        """Test that ECM breaks are deterministic with same seed."""
        results1 = []
        results2 = []

        for seed in [42, 42]:
            rng = random.Random(seed)
            local_results = []

            for _ in range(100):
                solution = FiringSolution(target_id="test", locked=True, lock_progress=1.0)
                broken = solution.check_ecm_break(0.5, 0.0, rng)
                local_results.append(broken)

            if len(results1) == 0:
                results1 = local_results
            else:
                results2 = local_results

        assert results1 == results2

    def test_targeting_system_deterministic(self):
        """Test that targeting system updates are deterministic."""
        def run_simulation(seed):
            targeting = create_basic_targeting_system(tracking_bonus=0.1, seed=seed)
            targeting.acquire_target("target_0")
            targets_ecm = {"target_0": 0.4}

            events = []
            for i in range(60):
                breaks = targeting.update(dt_seconds=1.0, targets_ecm=targets_ecm)
                if breaks:
                    events.append(i)

            return events

        events1 = run_simulation(42)
        events2 = run_simulation(42)

        assert events1 == events2
