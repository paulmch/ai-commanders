#!/usr/bin/env python3
"""
Comprehensive Test Suite for Physics Module

Tests cover:
1. Vector3D operations (add, subtract, multiply, divide, dot, cross, magnitude, normalization, rotation)
2. Delta-v calculations (Tsiolkovsky equation)
3. Thrust application (F=ma, thrust vectoring, propellant consumption)
4. Trajectory propagation (constant thrust, zero-g coast, position/velocity updates)
5. Rotation dynamics (moment of inertia, angular acceleration, rotation times)

Uses pytest with parametrized tests and validates against fleet data from fleet_ships.json.
"""

import json
import math
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics import (
    # Constants
    G_STANDARD,
    EXHAUST_VELOCITY_KPS,
    EXHAUST_VELOCITY_MS,
    MAIN_THRUST_MN,
    MAIN_THRUST_N,
    MAX_GIMBAL_ANGLE_DEG,
    # Classes
    Vector3D,
    ShipState,
    # Delta-v functions
    tsiolkovsky_delta_v,
    propellant_for_delta_v,
    mass_after_burn,
    # Thrust functions
    apply_thrust,
    calculate_torque_from_thrust,
    # Rotation functions
    calculate_moment_of_inertia,
    angular_acceleration_from_torque,
    time_to_rotate,
    max_angular_velocity,
    # Trajectory functions
    propagate_state,
    propagate_trajectory,
    # Utility functions
    create_ship_state_from_specs,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fleet_data():
    """Load fleet ship data from JSON file."""
    fleet_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    with open(fleet_path, "r") as f:
        return json.load(f)


@pytest.fixture
def corvette_data(fleet_data):
    """Get corvette ship data."""
    return fleet_data["ships"]["corvette"]


@pytest.fixture
def corvette_state(corvette_data):
    """Create a ShipState for the corvette."""
    perf = corvette_data["performance"]
    mass = corvette_data["mass_breakdown"]
    hull = corvette_data["hull"]
    prop = corvette_data["propulsion"]["drive"]

    return create_ship_state_from_specs(
        wet_mass_tons=perf["max_wet_mass_tons"],
        dry_mass_tons=perf["max_dry_mass_tons"],
        length_m=hull["length_m"],
        thrust_mn=prop["thrust_mn"],
        exhaust_velocity_kps=prop["exhaust_velocity_kps"],
    )


@pytest.fixture
def zero_vector():
    """Zero vector."""
    return Vector3D(0, 0, 0)


@pytest.fixture
def unit_x():
    """Unit vector in X direction."""
    return Vector3D(1, 0, 0)


@pytest.fixture
def unit_y():
    """Unit vector in Y direction."""
    return Vector3D(0, 1, 0)


@pytest.fixture
def unit_z():
    """Unit vector in Z direction."""
    return Vector3D(0, 0, 1)


# =============================================================================
# VECTOR3D TESTS
# =============================================================================

class TestVector3DBasicOperations:
    """Tests for basic Vector3D arithmetic operations."""

    @pytest.mark.parametrize("v1,v2,expected", [
        ((1, 2, 3), (4, 5, 6), (5, 7, 9)),
        ((0, 0, 0), (1, 1, 1), (1, 1, 1)),
        ((-1, -2, -3), (1, 2, 3), (0, 0, 0)),
        ((1.5, 2.5, 3.5), (0.5, 0.5, 0.5), (2, 3, 4)),
    ])
    def test_vector_addition(self, v1, v2, expected):
        """Test vector addition."""
        vec1 = Vector3D(*v1)
        vec2 = Vector3D(*v2)
        result = vec1 + vec2
        expected_vec = Vector3D(*expected)
        assert result == expected_vec

    @pytest.mark.parametrize("v1,v2,expected", [
        ((5, 7, 9), (4, 5, 6), (1, 2, 3)),
        ((1, 1, 1), (1, 1, 1), (0, 0, 0)),
        ((0, 0, 0), (1, 2, 3), (-1, -2, -3)),
    ])
    def test_vector_subtraction(self, v1, v2, expected):
        """Test vector subtraction."""
        vec1 = Vector3D(*v1)
        vec2 = Vector3D(*v2)
        result = vec1 - vec2
        expected_vec = Vector3D(*expected)
        assert result == expected_vec

    @pytest.mark.parametrize("v,scalar,expected", [
        ((1, 2, 3), 2, (2, 4, 6)),
        ((1, 2, 3), 0, (0, 0, 0)),
        ((1, 2, 3), -1, (-1, -2, -3)),
        ((2, 4, 6), 0.5, (1, 2, 3)),
    ])
    def test_scalar_multiplication(self, v, scalar, expected):
        """Test scalar multiplication (both left and right)."""
        vec = Vector3D(*v)
        expected_vec = Vector3D(*expected)

        # Test right multiplication
        result = vec * scalar
        assert result == expected_vec

        # Test left multiplication
        result = scalar * vec
        assert result == expected_vec

    @pytest.mark.parametrize("v,scalar,expected", [
        ((2, 4, 6), 2, (1, 2, 3)),
        ((1, 2, 3), 1, (1, 2, 3)),
        ((10, 20, 30), 10, (1, 2, 3)),
    ])
    def test_scalar_division(self, v, scalar, expected):
        """Test scalar division."""
        vec = Vector3D(*v)
        result = vec / scalar
        expected_vec = Vector3D(*expected)
        assert result == expected_vec

    def test_division_by_zero_raises_error(self):
        """Test that division by zero raises ValueError."""
        vec = Vector3D(1, 2, 3)
        with pytest.raises(ValueError, match="Cannot divide vector by zero"):
            vec / 0

    def test_negation(self):
        """Test vector negation."""
        vec = Vector3D(1, -2, 3)
        result = -vec
        assert result == Vector3D(-1, 2, -3)


class TestVector3DDotProduct:
    """Tests for Vector3D dot product."""

    @pytest.mark.parametrize("v1,v2,expected", [
        ((1, 0, 0), (0, 1, 0), 0),  # Perpendicular
        ((1, 0, 0), (1, 0, 0), 1),  # Parallel same
        ((1, 0, 0), (-1, 0, 0), -1),  # Parallel opposite
        ((1, 2, 3), (4, 5, 6), 32),  # General case: 1*4 + 2*5 + 3*6 = 32
        ((3, 4, 0), (3, 4, 0), 25),  # Magnitude squared
    ])
    def test_dot_product(self, v1, v2, expected):
        """Test dot product calculations."""
        vec1 = Vector3D(*v1)
        vec2 = Vector3D(*v2)
        result = vec1.dot(vec2)
        assert abs(result - expected) < 1e-10

    def test_dot_product_commutative(self):
        """Test that dot product is commutative."""
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        assert abs(v1.dot(v2) - v2.dot(v1)) < 1e-10


class TestVector3DCrossProduct:
    """Tests for Vector3D cross product."""

    @pytest.mark.parametrize("v1,v2,expected", [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),  # i x j = k
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),  # j x k = i
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),  # k x i = j
        ((1, 0, 0), (0, 0, 1), (0, -1, 0)),  # i x k = -j
        ((1, 0, 0), (1, 0, 0), (0, 0, 0)),  # Parallel vectors
    ])
    def test_cross_product(self, v1, v2, expected):
        """Test cross product calculations."""
        vec1 = Vector3D(*v1)
        vec2 = Vector3D(*v2)
        result = vec1.cross(vec2)
        expected_vec = Vector3D(*expected)
        assert result == expected_vec

    def test_cross_product_anticommutative(self):
        """Test that cross product is anticommutative."""
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        result1 = v1.cross(v2)
        result2 = v2.cross(v1)
        assert result1 == -result2

    def test_cross_product_perpendicular_to_inputs(self):
        """Test that cross product is perpendicular to both input vectors."""
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        result = v1.cross(v2)
        assert abs(result.dot(v1)) < 1e-10
        assert abs(result.dot(v2)) < 1e-10


class TestVector3DMagnitude:
    """Tests for Vector3D magnitude and normalization."""

    @pytest.mark.parametrize("v,expected_mag", [
        ((3, 4, 0), 5),  # 3-4-5 triangle
        ((1, 0, 0), 1),
        ((0, 0, 0), 0),
        ((1, 1, 1), math.sqrt(3)),
        ((2, 3, 6), 7),  # 2^2 + 3^2 + 6^2 = 49
    ])
    def test_magnitude(self, v, expected_mag):
        """Test magnitude calculation."""
        vec = Vector3D(*v)
        assert abs(vec.magnitude - expected_mag) < 1e-10

    def test_magnitude_squared(self):
        """Test magnitude squared calculation."""
        vec = Vector3D(3, 4, 0)
        assert abs(vec.magnitude_squared - 25) < 1e-10

    @pytest.mark.parametrize("v", [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (3, 4, 0),
        (1, 2, 3),
    ])
    def test_normalized_has_unit_magnitude(self, v):
        """Test that normalized vector has magnitude 1."""
        vec = Vector3D(*v)
        normalized = vec.normalized()
        assert abs(normalized.magnitude - 1.0) < 1e-10

    def test_zero_vector_normalization(self):
        """Test that normalizing zero vector returns zero vector."""
        vec = Vector3D(0, 0, 0)
        normalized = vec.normalized()
        assert normalized == Vector3D(0, 0, 0)

    def test_normalized_preserves_direction(self):
        """Test that normalization preserves direction."""
        vec = Vector3D(3, 4, 0)
        normalized = vec.normalized()
        # Original direction is (3/5, 4/5, 0)
        assert abs(normalized.x - 0.6) < 1e-10
        assert abs(normalized.y - 0.8) < 1e-10
        assert abs(normalized.z - 0.0) < 1e-10


class TestVector3DRotation:
    """Tests for Vector3D rotation using Rodrigues' formula."""

    def test_rotate_around_z_axis_90_degrees(self):
        """Test 90 degree rotation around Z axis."""
        vec = Vector3D(1, 0, 0)
        axis = Vector3D(0, 0, 1)
        rotated = vec.rotate_around_axis(axis, math.pi / 2)

        # X axis rotated 90 degrees around Z should give Y axis
        assert abs(rotated.x - 0) < 1e-10
        assert abs(rotated.y - 1) < 1e-10
        assert abs(rotated.z - 0) < 1e-10

    def test_rotate_around_y_axis_90_degrees(self):
        """Test 90 degree rotation around Y axis."""
        vec = Vector3D(1, 0, 0)
        axis = Vector3D(0, 1, 0)
        rotated = vec.rotate_around_axis(axis, math.pi / 2)

        # X axis rotated 90 degrees around Y should give -Z axis
        assert abs(rotated.x - 0) < 1e-10
        assert abs(rotated.y - 0) < 1e-10
        assert abs(rotated.z - (-1)) < 1e-10

    def test_rotate_around_x_axis_90_degrees(self):
        """Test 90 degree rotation around X axis."""
        vec = Vector3D(0, 1, 0)
        axis = Vector3D(1, 0, 0)
        rotated = vec.rotate_around_axis(axis, math.pi / 2)

        # Y axis rotated 90 degrees around X should give Z axis
        assert abs(rotated.x - 0) < 1e-10
        assert abs(rotated.y - 0) < 1e-10
        assert abs(rotated.z - 1) < 1e-10

    def test_rotate_360_degrees_returns_original(self):
        """Test that 360 degree rotation returns original vector."""
        vec = Vector3D(1, 2, 3)
        axis = Vector3D(1, 1, 1)
        rotated = vec.rotate_around_axis(axis, 2 * math.pi)

        assert abs(rotated.x - vec.x) < 1e-10
        assert abs(rotated.y - vec.y) < 1e-10
        assert abs(rotated.z - vec.z) < 1e-10

    def test_rotation_preserves_magnitude(self):
        """Test that rotation preserves vector magnitude."""
        vec = Vector3D(3, 4, 5)
        axis = Vector3D(1, 1, 0)
        original_mag = vec.magnitude

        for angle in [0.1, 0.5, 1.0, math.pi, 2 * math.pi]:
            rotated = vec.rotate_around_axis(axis, angle)
            assert abs(rotated.magnitude - original_mag) < 1e-10


class TestVector3DUtilities:
    """Tests for Vector3D utility methods."""

    def test_distance_to(self):
        """Test distance calculation between points."""
        p1 = Vector3D(0, 0, 0)
        p2 = Vector3D(3, 4, 0)
        assert abs(p1.distance_to(p2) - 5) < 1e-10

    def test_angle_to(self):
        """Test angle calculation between vectors."""
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(0, 1, 0)
        angle = v1.angle_to(v2)
        assert abs(angle - math.pi / 2) < 1e-10

    def test_angle_to_parallel(self):
        """Test angle to parallel vector is zero."""
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(2, 0, 0)
        angle = v1.angle_to(v2)
        assert abs(angle) < 1e-10

    def test_angle_to_opposite(self):
        """Test angle to opposite vector is pi."""
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(-1, 0, 0)
        angle = v1.angle_to(v2)
        assert abs(angle - math.pi) < 1e-10

    def test_to_tuple_and_from_tuple(self):
        """Test tuple conversion round-trip."""
        vec = Vector3D(1.5, 2.5, 3.5)
        t = vec.to_tuple()
        vec2 = Vector3D.from_tuple(t)
        assert vec == vec2

    def test_factory_methods(self):
        """Test factory methods for common vectors."""
        assert Vector3D.zero() == Vector3D(0, 0, 0)
        assert Vector3D.unit_x() == Vector3D(1, 0, 0)
        assert Vector3D.unit_y() == Vector3D(0, 1, 0)
        assert Vector3D.unit_z() == Vector3D(0, 0, 1)


# =============================================================================
# DELTA-V TESTS
# =============================================================================

class TestTsiolkovskyEquation:
    """Tests for delta-v calculations using Tsiolkovsky rocket equation."""

    def test_known_delta_v_calculation(self):
        """Test delta-v calculation with known values."""
        # Using simple values: v_e = 3000 m/s, mass ratio = e (Euler's number)
        # delta_v = v_e * ln(e) = v_e * 1 = 3000 m/s
        exhaust_v = 3000  # m/s
        wet_mass = math.e * 1000  # kg
        dry_mass = 1000  # kg

        dv = tsiolkovsky_delta_v(exhaust_v, wet_mass, dry_mass)
        assert abs(dv - 3000) < 1e-6

    def test_delta_v_mass_ratio_2(self):
        """Test delta-v with mass ratio of 2."""
        # delta_v = v_e * ln(2) ~ 0.693 * v_e
        exhaust_v = 10000  # m/s
        wet_mass = 2000  # kg
        dry_mass = 1000  # kg

        dv = tsiolkovsky_delta_v(exhaust_v, wet_mass, dry_mass)
        expected = exhaust_v * math.log(2)
        assert abs(dv - expected) < 1e-6

    def test_delta_v_no_propellant(self):
        """Test delta-v when wet mass equals dry mass (no propellant)."""
        dv = tsiolkovsky_delta_v(3000, 1000, 1000)
        assert abs(dv) < 1e-10

    def test_delta_v_invalid_masses(self):
        """Test delta-v with invalid mass configurations."""
        # Dry mass greater than wet mass
        assert tsiolkovsky_delta_v(3000, 500, 1000) == 0.0
        # Zero dry mass
        assert tsiolkovsky_delta_v(3000, 1000, 0) == 0.0
        # Negative dry mass
        assert tsiolkovsky_delta_v(3000, 1000, -100) == 0.0

    def test_corvette_delta_v_matches_fleet_data(self, fleet_data, corvette_data):
        """Test that calculated delta-v matches fleet data (~500 km/s for corvette)."""
        perf = corvette_data["performance"]
        constants = fleet_data["constants"]

        wet_mass_kg = perf["max_wet_mass_tons"] * 1000
        dry_mass_kg = perf["max_dry_mass_tons"] * 1000
        exhaust_v_ms = constants["exhaust_velocity_kps"] * 1000

        dv_ms = tsiolkovsky_delta_v(exhaust_v_ms, wet_mass_kg, dry_mass_kg)
        dv_kps = dv_ms / 1000

        expected_dv_kps = perf["delta_v_kps"]

        # Allow 1% tolerance for floating point differences
        assert abs(dv_kps - expected_dv_kps) / expected_dv_kps < 0.01, \
            f"Corvette delta-v {dv_kps:.1f} km/s does not match expected {expected_dv_kps} km/s"

    @pytest.mark.parametrize("ship_type", [
        "corvette", "frigate", "destroyer", "cruiser", "battlecruiser", "battleship", "dreadnought"
    ])
    def test_all_ships_delta_v_approximately_500_kps(self, fleet_data, ship_type):
        """Test that all ship types achieve approximately 500 km/s delta-v."""
        ship = fleet_data["ships"][ship_type]
        perf = ship["performance"]
        constants = fleet_data["constants"]

        wet_mass_kg = perf["max_wet_mass_tons"] * 1000
        dry_mass_kg = perf["max_dry_mass_tons"] * 1000
        exhaust_v_ms = constants["exhaust_velocity_kps"] * 1000

        dv_ms = tsiolkovsky_delta_v(exhaust_v_ms, wet_mass_kg, dry_mass_kg)
        dv_kps = dv_ms / 1000

        # All ships should have delta-v close to 500 km/s (target design goal)
        assert 495 < dv_kps < 505, \
            f"{ship_type} delta-v {dv_kps:.1f} km/s outside expected range"


class TestPropellantCalculations:
    """Tests for propellant requirement calculations."""

    def test_propellant_for_delta_v_known_values(self):
        """Test propellant calculation with known values."""
        # If delta_v = v_e * ln(2), then mass_ratio = 2
        # propellant = wet - dry = 2 * dry - dry = dry
        exhaust_v = 3000  # m/s
        delta_v = exhaust_v * math.log(2)  # m/s
        dry_mass = 1000  # kg

        propellant = propellant_for_delta_v(delta_v, exhaust_v, dry_mass)
        assert abs(propellant - 1000) < 1e-6

    def test_propellant_for_zero_delta_v(self):
        """Test propellant for zero delta-v is zero."""
        propellant = propellant_for_delta_v(0, 3000, 1000)
        assert abs(propellant) < 1e-10

    def test_propellant_for_invalid_inputs(self):
        """Test propellant calculation with invalid inputs."""
        assert propellant_for_delta_v(1000, 0, 1000) == 0.0  # Zero exhaust velocity
        assert propellant_for_delta_v(1000, 3000, 0) == 0.0  # Zero dry mass


class TestMassAfterBurn:
    """Tests for mass after burn calculations."""

    def test_mass_after_burn_known_values(self):
        """Test mass after burn with known values."""
        # If delta_v = v_e * ln(2), then m_final = m_initial / 2
        exhaust_v = 3000  # m/s
        delta_v = exhaust_v * math.log(2)  # m/s
        initial_mass = 2000  # kg

        final_mass = mass_after_burn(initial_mass, delta_v, exhaust_v)
        assert abs(final_mass - 1000) < 1e-6

    def test_mass_after_zero_burn(self):
        """Test mass after zero delta-v burn equals initial mass."""
        initial = 1000
        final = mass_after_burn(initial, 0, 3000)
        assert abs(final - initial) < 1e-10

    def test_mass_after_burn_invalid_exhaust_velocity(self):
        """Test mass after burn with invalid exhaust velocity."""
        initial = 1000
        final = mass_after_burn(initial, 1000, 0)
        assert final == initial


# =============================================================================
# THRUST APPLICATION TESTS
# =============================================================================

class TestThrustApplication:
    """Tests for thrust application (F=ma)."""

    def test_thrust_at_rest(self, corvette_state):
        """Test thrust application on stationary ship."""
        accel, prop_used = apply_thrust(corvette_state, throttle=1.0, dt=1.0)

        # F = ma -> a = F/m
        expected_accel = corvette_state.thrust_n / corvette_state.mass_kg

        # Acceleration should be in forward direction
        assert abs(accel.x - expected_accel) < 1  # Allow small error from avg mass calc
        assert abs(accel.y) < 1e-10
        assert abs(accel.z) < 1e-10

    def test_thrust_propellant_consumption(self, corvette_state):
        """Test propellant consumption rate."""
        _, prop_used = apply_thrust(corvette_state, throttle=1.0, dt=1.0)

        # Mass flow rate = F / v_e
        expected_flow = corvette_state.thrust_n / corvette_state.exhaust_velocity_ms

        assert abs(prop_used - expected_flow) < 1e-6

    def test_thrust_throttle_scaling(self, corvette_state):
        """Test that thrust scales linearly with throttle."""
        accel_full, prop_full = apply_thrust(corvette_state, throttle=1.0, dt=1.0)
        accel_half, prop_half = apply_thrust(corvette_state, throttle=0.5, dt=1.0)

        # Half throttle should give approximately half acceleration and propellant
        assert abs(accel_half.magnitude / accel_full.magnitude - 0.5) < 0.01
        assert abs(prop_half / prop_full - 0.5) < 0.01

    def test_zero_throttle_no_thrust(self, corvette_state):
        """Test that zero throttle produces no thrust."""
        accel, prop_used = apply_thrust(corvette_state, throttle=0.0, dt=1.0)

        assert accel == Vector3D.zero()
        assert prop_used == 0.0

    def test_no_propellant_no_thrust(self, corvette_state):
        """Test that empty propellant produces no thrust."""
        corvette_state.propellant_kg = 0
        accel, prop_used = apply_thrust(corvette_state, throttle=1.0, dt=1.0)

        assert accel == Vector3D.zero()
        assert prop_used == 0.0

    def test_corvette_acceleration_matches_fleet_data(self, corvette_data):
        """Test that corvette acceleration matches fleet data (~3g)."""
        perf = corvette_data["performance"]
        prop = corvette_data["propulsion"]["drive"]

        wet_mass_kg = perf["max_wet_mass_tons"] * 1000
        thrust_n = prop["thrust_mn"] * 1e6

        accel_ms2 = thrust_n / wet_mass_kg
        accel_g = accel_ms2 / G_STANDARD

        expected_g = perf["combat_acceleration_g"]

        assert abs(accel_g - expected_g) < 0.1, \
            f"Corvette acceleration {accel_g:.2f}g does not match expected {expected_g}g"


class TestThrustVectoring:
    """Tests for thrust vectoring at different angles."""

    def test_thrust_vectoring_pitch(self, corvette_state):
        """Test thrust vectoring with pitch deflection."""
        accel, _ = apply_thrust(corvette_state, throttle=1.0, gimbal_pitch_deg=1.0, dt=1.0)

        # With pitch deflection, thrust should have Z component
        assert accel.x > 0  # Still primarily forward
        assert abs(accel.y) < 1e-6  # No yaw component
        # Z component from pitch rotation
        assert abs(accel.z) > 0 or abs(accel.z) < 1e-6  # May be small but present

    def test_thrust_vectoring_yaw(self, corvette_state):
        """Test thrust vectoring with yaw deflection."""
        accel, _ = apply_thrust(corvette_state, throttle=1.0, gimbal_yaw_deg=1.0, dt=1.0)

        # With yaw deflection, thrust should have Y component
        assert accel.x > 0  # Still primarily forward
        # Y component from yaw rotation

    def test_gimbal_angle_clamping(self, corvette_state):
        """Test that gimbal angles are clamped to limits."""
        # Request 10 degree gimbal (beyond max 3 degrees)
        accel_max, _ = apply_thrust(corvette_state, throttle=1.0, gimbal_pitch_deg=10.0, dt=1.0)
        accel_3, _ = apply_thrust(corvette_state, throttle=1.0, gimbal_pitch_deg=3.0, dt=1.0)

        # Should produce same result due to clamping
        assert abs(accel_max.x - accel_3.x) < 1e-6
        assert abs(accel_max.y - accel_3.y) < 1e-6
        assert abs(accel_max.z - accel_3.z) < 1e-6

    def test_thrust_vectoring_efficiency(self, corvette_state):
        """Test that thrust vectoring reduces forward efficiency."""
        accel_straight, _ = apply_thrust(corvette_state, throttle=1.0, dt=1.0)
        accel_gimbal, _ = apply_thrust(corvette_state, throttle=1.0, gimbal_pitch_deg=3.0, dt=1.0)

        # Forward component should be slightly less with gimbal
        # cos(3 degrees) ~ 0.9986
        efficiency = accel_gimbal.x / accel_straight.x
        expected_efficiency = math.cos(math.radians(3.0))

        assert abs(efficiency - expected_efficiency) < 0.001


class TestPropellantConsumption:
    """Tests for propellant consumption accuracy."""

    def test_propellant_not_consumed_beyond_available(self, corvette_state):
        """Test that propellant consumption is limited to available amount."""
        corvette_state.propellant_kg = 1.0  # Very little propellant

        _, prop_used = apply_thrust(corvette_state, throttle=1.0, dt=100.0)  # Long burn

        assert prop_used <= 1.0

    def test_propellant_consumption_scales_with_time(self, corvette_state):
        """Test that propellant consumption scales linearly with time."""
        _, prop_1s = apply_thrust(corvette_state, throttle=1.0, dt=1.0)
        _, prop_2s = apply_thrust(corvette_state, throttle=1.0, dt=2.0)

        assert abs(prop_2s / prop_1s - 2.0) < 0.01


class TestTorqueFromThrust:
    """Tests for torque calculation from thrust vectoring."""

    def test_torque_from_pitch_gimbal(self, corvette_state):
        """Test torque generation from pitch gimbal."""
        torque = calculate_torque_from_thrust(corvette_state, gimbal_pitch_deg=1.0, throttle=1.0)

        # Pitch gimbal should produce pitch torque (y component)
        assert abs(torque.y) > 0

    def test_torque_from_yaw_gimbal(self, corvette_state):
        """Test torque generation from yaw gimbal."""
        torque = calculate_torque_from_thrust(corvette_state, gimbal_yaw_deg=1.0, throttle=1.0)

        # Yaw gimbal should produce yaw torque (z component)
        assert abs(torque.z) > 0

    def test_no_gimbal_no_torque(self, corvette_state):
        """Test that zero gimbal produces zero torque."""
        torque = calculate_torque_from_thrust(corvette_state, gimbal_pitch_deg=0, gimbal_yaw_deg=0, throttle=1.0)

        assert abs(torque.x) < 1e-6
        assert abs(torque.y) < 1e-6
        assert abs(torque.z) < 1e-6

    def test_zero_throttle_no_torque(self, corvette_state):
        """Test that zero throttle produces zero torque."""
        torque = calculate_torque_from_thrust(corvette_state, gimbal_pitch_deg=1.0, throttle=0.0)

        assert torque == Vector3D.zero()


# =============================================================================
# TRAJECTORY PROPAGATION TESTS
# =============================================================================

class TestTrajectoryPropagation:
    """Tests for trajectory propagation."""

    def test_zero_g_coast(self, corvette_state):
        """Test zero-g coasting (no thrust)."""
        corvette_state.velocity = Vector3D(1000, 0, 0)  # 1 km/s

        new_state = propagate_state(corvette_state, dt=10.0, throttle=0.0)

        # Position should change by v * dt
        expected_position = Vector3D(10000, 0, 0)  # 10 km
        assert abs(new_state.position.x - expected_position.x) < 1e-6

        # Velocity should remain constant
        assert abs(new_state.velocity.x - 1000) < 1e-6

        # No propellant should be used
        assert new_state.propellant_kg == corvette_state.propellant_kg

    def test_constant_thrust_trajectory(self, corvette_state):
        """Test trajectory under constant thrust."""
        initial_velocity = corvette_state.velocity.magnitude

        new_state = propagate_state(corvette_state, dt=1.0, throttle=1.0)

        # Velocity should increase
        assert new_state.velocity.magnitude > initial_velocity

        # Position should change
        assert new_state.position.magnitude > 0

        # Propellant should decrease
        assert new_state.propellant_kg < corvette_state.propellant_kg

    def test_position_velocity_update(self, corvette_state):
        """Test that position and velocity update correctly."""
        corvette_state.velocity = Vector3D(100, 0, 0)  # Initial velocity

        # Propagate with no thrust
        new_state = propagate_state(corvette_state, dt=5.0, throttle=0.0)

        # x = x0 + v*t
        assert abs(new_state.position.x - 500) < 1e-6  # 100 m/s * 5 s = 500 m

    def test_multi_step_trajectory(self, corvette_state):
        """Test multi-step trajectory propagation."""
        trajectory = propagate_trajectory(
            corvette_state,
            total_time=10.0,
            dt=1.0,
            throttle=0.5
        )

        # Should have initial state plus 10 steps
        assert len(trajectory) == 11

        # Each state should have increasing position (with thrust)
        for i in range(1, len(trajectory)):
            assert trajectory[i].position.magnitude >= trajectory[i-1].position.magnitude

    def test_trajectory_conserves_energy_in_coast(self, corvette_state):
        """Test that kinetic energy is conserved during coast."""
        corvette_state.velocity = Vector3D(1000, 500, 250)
        initial_ke = 0.5 * corvette_state.mass_kg * corvette_state.velocity.magnitude_squared

        new_state = propagate_state(corvette_state, dt=100.0, throttle=0.0)

        final_ke = 0.5 * new_state.mass_kg * new_state.velocity.magnitude_squared

        assert abs(final_ke - initial_ke) / initial_ke < 1e-10


class TestTrajectoryWithRotation:
    """Tests for trajectory propagation with rotation."""

    def test_angular_velocity_changes_orientation(self, corvette_state):
        """Test that angular velocity changes ship orientation."""
        corvette_state.angular_velocity = Vector3D(0, 0.1, 0)  # Pitch rate

        initial_forward = corvette_state.forward
        new_state = propagate_state(corvette_state, dt=1.0, throttle=0.0)

        # Forward direction should have changed
        angle = initial_forward.angle_to(new_state.forward)
        assert angle > 0


# =============================================================================
# ROTATION DYNAMICS TESTS
# =============================================================================

class TestMomentOfInertia:
    """Tests for moment of inertia calculations."""

    def test_moment_of_inertia_formula(self):
        """Test moment of inertia calculation for elongated cylinder."""
        mass = 1000  # kg
        length = 10  # m

        moi = calculate_moment_of_inertia(mass, length)

        # I_pitch_yaw = (1/12) * m * L^2
        expected_pitch_yaw = (1/12) * mass * length**2
        assert abs(moi['pitch_yaw_kg_m2'] - expected_pitch_yaw) < 1e-6

    def test_corvette_moi_matches_fleet_data(self, fleet_data, corvette_data):
        """Test that corvette MOI matches fleet data."""
        hull = corvette_data["hull"]
        perf = corvette_data["performance"]
        attitude = corvette_data["attitude_control"]

        wet_mass_kg = perf["max_wet_mass_tons"] * 1000
        length_m = hull["length_m"]

        moi = calculate_moment_of_inertia(wet_mass_kg, length_m)
        expected_moi = attitude["moment_of_inertia"]["pitch_yaw_kg_m2"]

        # Allow 1% tolerance
        assert abs(moi['pitch_yaw_kg_m2'] - expected_moi) / expected_moi < 0.01, \
            f"Corvette MOI {moi['pitch_yaw_kg_m2']:.2e} does not match expected {expected_moi:.2e}"


class TestAngularAcceleration:
    """Tests for angular acceleration from torque."""

    def test_angular_acceleration_formula(self):
        """Test alpha = tau / I formula."""
        torque = 1000  # N*m
        moi = 500  # kg*m^2

        alpha = angular_acceleration_from_torque(torque, moi)

        assert abs(alpha - 2.0) < 1e-10  # 1000 / 500 = 2 rad/s^2

    def test_zero_torque_zero_acceleration(self):
        """Test that zero torque gives zero angular acceleration."""
        alpha = angular_acceleration_from_torque(0, 1000)
        assert alpha == 0.0

    def test_zero_moi_zero_acceleration(self):
        """Test that zero MOI gives zero angular acceleration."""
        alpha = angular_acceleration_from_torque(1000, 0)
        assert alpha == 0.0


class TestRotationTime:
    """Tests for rotation time calculations."""

    def test_rotation_time_formula(self):
        """Test bang-bang rotation time formula."""
        # t = 2 * sqrt(theta / alpha)
        alpha = 1.0  # rad/s^2
        angle = 90  # degrees

        t = time_to_rotate(alpha, angle)

        expected = 2 * math.sqrt(math.radians(90) / 1.0)
        assert abs(t - expected) < 1e-10

    def test_zero_acceleration_infinite_time(self):
        """Test that zero acceleration gives infinite time."""
        t = time_to_rotate(0, 90)
        assert t == float('inf')

    def test_corvette_90_degree_rotation_time(self, fleet_data, corvette_data):
        """Test corvette 90 degree rotation time matches fleet data (~12.1s)."""
        attitude = corvette_data["attitude_control"]["thrust_vectoring"]

        alpha_deg_s2 = attitude["angular_accel_deg_s2"]
        alpha_rad_s2 = math.radians(alpha_deg_s2)

        t90 = time_to_rotate(alpha_rad_s2, 90)
        expected_t90 = attitude["time_to_rotate_90_deg_s"]

        # Allow 5% tolerance
        assert abs(t90 - expected_t90) / expected_t90 < 0.05, \
            f"Corvette 90 deg rotation time {t90:.1f}s does not match expected {expected_t90}s"

    @pytest.mark.parametrize("ship_type,angle,time_key", [
        ("corvette", 45, "time_to_rotate_45_deg_s"),
        ("corvette", 90, "time_to_rotate_90_deg_s"),
        ("corvette", 180, "time_to_rotate_180_deg_s"),
        ("frigate", 90, "time_to_rotate_90_deg_s"),
        ("destroyer", 90, "time_to_rotate_90_deg_s"),
        ("cruiser", 90, "time_to_rotate_90_deg_s"),
        ("battleship", 90, "time_to_rotate_90_deg_s"),
        ("dreadnought", 90, "time_to_rotate_90_deg_s"),
    ])
    def test_all_ships_rotation_times(self, fleet_data, ship_type, angle, time_key):
        """Test rotation times for all ship types."""
        ship = fleet_data["ships"][ship_type]
        attitude = ship["attitude_control"]["thrust_vectoring"]

        alpha_deg_s2 = attitude["angular_accel_deg_s2"]
        alpha_rad_s2 = math.radians(alpha_deg_s2)

        t = time_to_rotate(alpha_rad_s2, angle)
        expected_t = attitude[time_key]

        # Allow 5% tolerance
        assert abs(t - expected_t) / expected_t < 0.05, \
            f"{ship_type} {angle} deg rotation time {t:.1f}s does not match expected {expected_t}s"


class TestMaxAngularVelocity:
    """Tests for maximum angular velocity during rotation."""

    def test_max_angular_velocity_formula(self):
        """Test max angular velocity calculation."""
        alpha = 1.0  # rad/s^2
        angle = 90  # degrees

        omega_max = max_angular_velocity(alpha, angle)

        # omega_max = alpha * (t / 2), where t = 2 * sqrt(theta / alpha)
        # omega_max = alpha * sqrt(theta / alpha) = sqrt(alpha * theta)
        expected = math.sqrt(alpha * math.radians(angle))
        assert abs(omega_max - expected) < 1e-10

    def test_zero_acceleration_zero_velocity(self):
        """Test that zero acceleration gives zero max angular velocity."""
        omega = max_angular_velocity(0, 90)
        assert omega == 0.0


# =============================================================================
# SHIP STATE TESTS
# =============================================================================

class TestShipState:
    """Tests for ShipState class."""

    def test_ship_state_defaults(self):
        """Test ShipState default values."""
        state = ShipState()

        assert state.position == Vector3D.zero()
        assert state.velocity == Vector3D.zero()
        assert state.forward == Vector3D.unit_x()
        assert state.up == Vector3D.unit_z()

    def test_right_vector_calculation(self):
        """Test right vector is cross product of forward and up."""
        state = ShipState()

        # forward x up for right-handed system
        expected_right = state.forward.cross(state.up).normalized()
        assert state.right == expected_right

    def test_wet_mass_calculation(self):
        """Test wet mass equals dry mass plus propellant."""
        state = ShipState(
            dry_mass_kg=1000,
            propellant_kg=500
        )

        assert state.wet_mass_kg == 1500

    def test_mass_ratio_calculation(self):
        """Test mass ratio calculation."""
        state = ShipState(
            dry_mass_kg=1000,
            propellant_kg=1000
        )

        assert state.mass_ratio == 2.0

    def test_remaining_delta_v(self, corvette_state):
        """Test remaining delta-v calculation."""
        dv_kps = corvette_state.remaining_delta_v_kps()

        # Should be approximately 500 km/s for full corvette
        assert 495 < dv_kps < 505

    def test_max_acceleration(self, corvette_state):
        """Test max acceleration calculation."""
        accel_g = corvette_state.max_acceleration_g()

        # Corvette should have ~3g acceleration
        assert 2.9 < accel_g < 3.1

    def test_ship_state_copy(self, corvette_state):
        """Test that copy creates independent state."""
        copy = corvette_state.copy()

        # Modify original
        corvette_state.position = Vector3D(100, 200, 300)

        # Copy should be unchanged
        assert copy.position == Vector3D.zero()


class TestCreateShipStateFromSpecs:
    """Tests for ship state creation from specifications."""

    def test_create_corvette_from_specs(self, corvette_data):
        """Test creating corvette state from fleet data."""
        hull = corvette_data["hull"]
        perf = corvette_data["performance"]
        prop = corvette_data["propulsion"]["drive"]

        state = create_ship_state_from_specs(
            wet_mass_tons=perf["max_wet_mass_tons"],
            dry_mass_tons=perf["max_dry_mass_tons"],
            length_m=hull["length_m"],
            thrust_mn=prop["thrust_mn"],
            exhaust_velocity_kps=prop["exhaust_velocity_kps"],
        )

        assert abs(state.mass_kg - perf["max_wet_mass_tons"] * 1000) < 1
        assert abs(state.dry_mass_kg - perf["max_dry_mass_tons"] * 1000) < 1
        assert abs(state.thrust_n - prop["thrust_mn"] * 1e6) < 1
        assert abs(state.exhaust_velocity_ms - prop["exhaust_velocity_kps"] * 1000) < 1

    def test_create_ship_with_position(self):
        """Test creating ship with initial position."""
        position = Vector3D(1000, 2000, 3000)
        state = create_ship_state_from_specs(
            wet_mass_tons=1000,
            dry_mass_tons=900,
            length_m=50,
            position=position
        )

        assert state.position == position

    def test_create_ship_with_velocity(self):
        """Test creating ship with initial velocity."""
        velocity = Vector3D(100, 200, 300)
        state = create_ship_state_from_specs(
            wet_mass_tons=1000,
            dry_mass_tons=900,
            length_m=50,
            velocity=velocity
        )

        assert state.velocity == velocity


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple physics components."""

    def test_full_burn_delta_v_matches_tsiolkovsky(self, corvette_state):
        """Test that full burn delta-v matches Tsiolkovsky prediction."""
        initial_dv = corvette_state.remaining_delta_v_ms()
        initial_propellant = corvette_state.propellant_kg

        # Burn until propellant exhausted (or close to it)
        state = corvette_state.copy()
        total_burn_time = 0
        dt = 1.0

        while state.propellant_kg > 10:  # Leave small margin
            state = propagate_state(state, dt=dt, throttle=1.0)
            total_burn_time += dt

        # Velocity gained should approximately match initial delta-v
        velocity_gained = state.velocity.magnitude - corvette_state.velocity.magnitude

        # Allow 5% tolerance for numerical integration errors
        assert abs(velocity_gained - initial_dv) / initial_dv < 0.05

    def test_rotation_matches_fleet_data(self, corvette_state, corvette_data):
        """Test that rotation behavior matches fleet data."""
        attitude = corvette_data["attitude_control"]["thrust_vectoring"]

        # Set up for rotation with thrust vectoring
        alpha_deg_s2 = attitude["angular_accel_deg_s2"]
        expected_t90 = attitude["time_to_rotate_90_deg_s"]

        # Verify angular acceleration from torque matches
        torque_nm = attitude["torque_mn_m"] * 1e6
        moi = corvette_data["attitude_control"]["moment_of_inertia"]["pitch_yaw_kg_m2"]

        alpha_rad_s2 = angular_acceleration_from_torque(torque_nm, moi)
        alpha_deg_s2_calc = math.degrees(alpha_rad_s2)

        # Should match within 5%
        assert abs(alpha_deg_s2_calc - alpha_deg_s2) / alpha_deg_s2 < 0.05

    def test_propellant_accounting(self, corvette_state):
        """Test that propellant accounting is accurate over trajectory."""
        initial_propellant = corvette_state.propellant_kg

        # Run trajectory for fixed time
        trajectory = propagate_trajectory(
            corvette_state,
            total_time=10.0,
            dt=1.0,
            throttle=1.0
        )

        final_state = trajectory[-1]
        propellant_used = initial_propellant - final_state.propellant_kg

        # Calculate expected propellant from mass flow rate
        mass_flow_rate = corvette_state.thrust_n / corvette_state.exhaust_velocity_ms
        expected_propellant = mass_flow_rate * 10.0

        # Should match within 5% (integration error)
        assert abs(propellant_used - expected_propellant) / expected_propellant < 0.05


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_dt(self, corvette_state):
        """Test propagation with very small time step."""
        state = propagate_state(corvette_state, dt=1e-6, throttle=1.0)

        # Should still work without errors
        assert state is not None

    def test_very_large_dt(self, corvette_state):
        """Test propagation with large time step."""
        state = propagate_state(corvette_state, dt=1000.0, throttle=0.0)

        # Should still work without errors
        assert state is not None

    def test_very_high_velocity(self):
        """Test with relativistic-ish velocities (0.1c)."""
        c = 3e8  # m/s
        state = ShipState(
            velocity=Vector3D(0.1 * c, 0, 0),  # 10% speed of light
            mass_kg=1000,
            dry_mass_kg=900,
            propellant_kg=100
        )

        # Propagate should work (though physics is Newtonian)
        new_state = propagate_state(state, dt=1.0, throttle=0.0)

        # Position should update correctly
        expected_pos = 0.1 * c * 1.0  # v * t
        assert abs(new_state.position.x - expected_pos) < 1e-3

    def test_negative_throttle_clamped(self, corvette_state):
        """Test that negative throttle is clamped to zero."""
        accel, prop = apply_thrust(corvette_state, throttle=-0.5, dt=1.0)

        assert accel == Vector3D.zero()
        assert prop == 0.0

    def test_throttle_over_one_clamped(self, corvette_state):
        """Test that throttle over 1.0 is clamped."""
        accel_max, _ = apply_thrust(corvette_state, throttle=1.0, dt=1.0)
        accel_over, _ = apply_thrust(corvette_state, throttle=2.0, dt=1.0)

        # Should be the same due to clamping
        assert abs(accel_max.magnitude - accel_over.magnitude) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
