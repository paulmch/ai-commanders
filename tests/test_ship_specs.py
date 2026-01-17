#!/usr/bin/env python3
"""
Test Suite for Ship Specifications

Validates that all ship classes have correct:
1. Maximum acceleration (g)
2. Turn rates (thrust vectoring and RCS)
3. Delta-v calculations

Uses data from fleet_ships.json as ground truth.
"""

import json
import pytest
from pathlib import Path

from src.simulation import create_ship_from_fleet_data
from src.physics import Vector3D, G_STANDARD


# =============================================================================
# EXPECTED VALUES FROM FLEET_SHIPS.JSON
# =============================================================================

# Ship class -> expected combat acceleration in g
EXPECTED_ACCELERATION_G = {
    "corvette": 3.0,
    "frigate": 3.0,
    "destroyer": 2.0,
    "cruiser": 1.5,
    "battlecruiser": 1.5,
    "battleship": 1.0,
    "dreadnought": 0.75,
    "dreadnought_siege": 0.676,
}

# Ship class -> expected 90 degree turn time with thrust vectoring (seconds)
EXPECTED_TV_TURN_90_S = {
    "corvette": 12.1,
    "frigate": 15.1,
    "destroyer": 20.6,
    "cruiser": 28.2,
    "battlecruiser": 28.2,
    "battleship": 36.9,
    "dreadnought": 49.9,
    "dreadnought_siege": 52.6,
}

# Ship class -> expected 90 degree turn time with RCS only (seconds)
EXPECTED_RCS_TURN_90_S = {
    "corvette": 54.2,
    "frigate": 83.3,
    "destroyer": 127.6,
    "cruiser": 206.3,
    "battlecruiser": 206.3,
    "battleship": 288.7,
    "dreadnought": 458.4,
    "dreadnought_siege": 484.0,
}


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fleet_data():
    """Load fleet ship data from JSON file."""
    fleet_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    with open(fleet_path, "r") as f:
        return json.load(f)


# =============================================================================
# ACCELERATION TESTS
# =============================================================================

@pytest.mark.parametrize("ship_type,expected_g", list(EXPECTED_ACCELERATION_G.items()))
def test_ship_acceleration_from_json(ship_type: str, expected_g: float, fleet_data: dict):
    """
    Test that acceleration values in fleet_ships.json match expected values.

    This validates the source data is correct.
    """
    ship_data = fleet_data["ships"][ship_type]
    perf = ship_data["performance"]

    actual_g = perf["combat_acceleration_g"]

    assert actual_g == pytest.approx(expected_g, rel=0.01), \
        f"{ship_type}: expected {expected_g}g, got {actual_g}g in JSON"


@pytest.mark.parametrize("ship_type,expected_g", list(EXPECTED_ACCELERATION_G.items()))
def test_ship_acceleration_from_simulation(ship_type: str, expected_g: float, fleet_data: dict):
    """
    Test that ships created via create_ship_from_fleet_data have correct acceleration.

    This validates the simulation correctly uses the JSON data.
    """
    ship = create_ship_from_fleet_data(
        ship_id=f"test_{ship_type}",
        ship_type=ship_type,
        faction="alpha",
        fleet_data=fleet_data,
        position=Vector3D.zero(),
        velocity=Vector3D.zero(),
        forward=Vector3D.unit_x()
    )

    actual_g = ship.kinematic_state.max_acceleration_g()

    assert actual_g == pytest.approx(expected_g, rel=0.02), \
        f"{ship_type}: expected {expected_g}g, got {actual_g:.3f}g from simulation"


@pytest.mark.parametrize("ship_type,expected_g", list(EXPECTED_ACCELERATION_G.items()))
def test_acceleration_calculation_consistency(ship_type: str, expected_g: float, fleet_data: dict):
    """
    Test that acceleration calculated from thrust/mass matches the JSON value.

    acceleration = thrust / mass
    """
    ship_data = fleet_data["ships"][ship_type]
    perf = ship_data["performance"]
    drive = ship_data["propulsion"]["drive"]

    thrust_n = drive["thrust_mn"] * 1e6
    mass_kg = perf["max_wet_mass_tons"] * 1000

    calculated_accel_ms2 = thrust_n / mass_kg
    calculated_g = calculated_accel_ms2 / G_STANDARD

    # Should match JSON value
    json_g = perf["combat_acceleration_g"]
    json_ms2 = perf["combat_acceleration_ms2"]

    assert calculated_g == pytest.approx(json_g, rel=0.01), \
        f"{ship_type}: calculated {calculated_g:.3f}g, JSON says {json_g}g"

    assert calculated_accel_ms2 == pytest.approx(json_ms2, rel=0.01), \
        f"{ship_type}: calculated {calculated_accel_ms2:.2f} m/s^2, JSON says {json_ms2} m/s^2"


# =============================================================================
# TURN RATE TESTS
# =============================================================================

@pytest.mark.parametrize("ship_type,expected_s", list(EXPECTED_TV_TURN_90_S.items()))
def test_thrust_vectoring_turn_time_from_json(ship_type: str, expected_s: float, fleet_data: dict):
    """
    Test that thrust vectoring 90 degree turn times in JSON match expected values.
    """
    ship_data = fleet_data["ships"][ship_type]
    attitude = ship_data["attitude_control"]
    tv = attitude["thrust_vectoring"]

    actual_s = tv["time_to_rotate_90_deg_s"]

    assert actual_s == pytest.approx(expected_s, rel=0.01), \
        f"{ship_type}: expected {expected_s}s TV turn, got {actual_s}s in JSON"


@pytest.mark.parametrize("ship_type,expected_s", list(EXPECTED_RCS_TURN_90_S.items()))
def test_rcs_turn_time_from_json(ship_type: str, expected_s: float, fleet_data: dict):
    """
    Test that RCS 90 degree turn times in JSON match expected values.
    """
    ship_data = fleet_data["ships"][ship_type]
    attitude = ship_data["attitude_control"]
    rcs = attitude["rcs"]

    actual_s = rcs["time_to_rotate_90_deg_s"]

    assert actual_s == pytest.approx(expected_s, rel=0.01), \
        f"{ship_type}: expected {expected_s}s RCS turn, got {actual_s}s in JSON"


# =============================================================================
# MASS CONSISTENCY TESTS
# =============================================================================

@pytest.mark.parametrize("ship_type", list(EXPECTED_ACCELERATION_G.keys()))
def test_mass_breakdown_consistency(ship_type: str, fleet_data: dict):
    """
    Test that mass breakdown sums match total dry/wet mass.
    """
    ship_data = fleet_data["ships"][ship_type]
    mass = ship_data["mass_breakdown"]
    perf = ship_data["performance"]

    # total_dry should match max_dry_mass_tons
    assert mass["total_dry"] == pytest.approx(perf["max_dry_mass_tons"], rel=0.01), \
        f"{ship_type}: mass_breakdown.total_dry != performance.max_dry_mass_tons"

    # total_wet should match max_wet_mass_tons
    assert mass["total_wet"] == pytest.approx(perf["max_wet_mass_tons"], rel=0.01), \
        f"{ship_type}: mass_breakdown.total_wet != performance.max_wet_mass_tons"


# =============================================================================
# ALL SHIPS SUMMARY TEST
# =============================================================================

def test_all_ships_acceleration_summary(fleet_data: dict):
    """
    Print summary table of all ship accelerations.
    Useful for manual verification.
    """
    print("\n" + "=" * 70)
    print("Ship Acceleration Summary")
    print("=" * 70)
    print(f"{'Ship Type':<20} {'Expected (g)':<15} {'JSON (g)':<15} {'Simulation (g)':<15}")
    print("-" * 70)

    all_pass = True
    for ship_type, expected_g in EXPECTED_ACCELERATION_G.items():
        ship_data = fleet_data["ships"][ship_type]
        json_g = ship_data["performance"]["combat_acceleration_g"]

        ship = create_ship_from_fleet_data(
            ship_id=f"test_{ship_type}",
            ship_type=ship_type,
            faction="alpha",
            fleet_data=fleet_data
        )
        sim_g = ship.kinematic_state.max_acceleration_g()

        status = "OK" if abs(sim_g - expected_g) < 0.05 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{ship_type:<20} {expected_g:<15.3f} {json_g:<15.3f} {sim_g:<15.3f} [{status}]")

    print("=" * 70)
    assert all_pass, "Some ships have incorrect acceleration values"


def test_all_ships_turn_rate_summary(fleet_data: dict):
    """
    Print summary table of all ship turn rates.
    Useful for manual verification.
    """
    print("\n" + "=" * 80)
    print("Ship Turn Rate Summary (90 degree rotation)")
    print("=" * 80)
    print(f"{'Ship Type':<20} {'TV Expected (s)':<18} {'TV JSON (s)':<18} {'RCS Expected (s)':<18} {'RCS JSON (s)':<18}")
    print("-" * 80)

    all_pass = True
    for ship_type in EXPECTED_TV_TURN_90_S.keys():
        ship_data = fleet_data["ships"][ship_type]
        attitude = ship_data["attitude_control"]

        tv_expected = EXPECTED_TV_TURN_90_S[ship_type]
        tv_json = attitude["thrust_vectoring"]["time_to_rotate_90_deg_s"]

        rcs_expected = EXPECTED_RCS_TURN_90_S[ship_type]
        rcs_json = attitude["rcs"]["time_to_rotate_90_deg_s"]

        tv_status = "OK" if abs(tv_json - tv_expected) < 0.5 else "FAIL"
        rcs_status = "OK" if abs(rcs_json - rcs_expected) < 0.5 else "FAIL"

        if tv_status == "FAIL" or rcs_status == "FAIL":
            all_pass = False

        print(f"{ship_type:<20} {tv_expected:<18.1f} {tv_json:<18.1f} {rcs_expected:<18.1f} {rcs_json:<18.1f}")

    print("=" * 80)
    assert all_pass, "Some ships have incorrect turn rate values"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
