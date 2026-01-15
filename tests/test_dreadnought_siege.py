"""
Tests for the Dreadnought Siege variant.

The dreadnought_siege is a modified dreadnought that replaces the standard
spinal_coiler_mk3 with the heavy_siege_coiler_mk3 for devastating alpha strikes.

Key tradeoffs:
- +800 tons weapon mass, +35.4 tons ammo
- Acceleration drops from 0.75g to 0.676g (10% reduction)
- Rotation time increases by ~5% due to higher mass
- Alpha strike: 21.75 GJ (siege) vs 4.29 GJ (standard) = +407%
"""

import pytest
import json
from pathlib import Path


@pytest.fixture
def fleet_data():
    """Load fleet_ships.json data."""
    fleet_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    with open(fleet_path) as f:
        return json.load(f)


@pytest.fixture
def siege_dread(fleet_data):
    """Get the dreadnought siege variant."""
    return fleet_data["ships"]["dreadnought_siege"]


@pytest.fixture
def standard_dread(fleet_data):
    """Get the standard dreadnought."""
    return fleet_data["ships"]["dreadnought"]


# =============================================================================
# EXISTENCE AND BASIC STRUCTURE TESTS
# =============================================================================

class TestDreadnoughtSiegeExists:
    """Verify the siege variant exists and has required fields."""

    def test_ship_exists(self, fleet_data):
        """Verify dreadnought_siege is defined."""
        assert "dreadnought_siege" in fleet_data["ships"]

    def test_has_performance(self, siege_dread):
        """Verify performance section exists."""
        assert "performance" in siege_dread

    def test_has_weapons(self, siege_dread):
        """Verify weapons section exists."""
        assert "weapons" in siege_dread

    def test_has_attitude_control(self, siege_dread):
        """Verify attitude control section exists."""
        assert "attitude_control" in siege_dread

    def test_role_is_siege_platform(self, siege_dread):
        """Verify designated role."""
        assert siege_dread["role"] == "Siege Platform"


# =============================================================================
# WEAPON CONFIGURATION TESTS
# =============================================================================

class TestSiegeWeaponConfig:
    """Verify the siege coiler is properly mounted."""

    def test_primary_weapon_is_siege_coiler(self, siege_dread):
        """Verify main weapon is heavy siege coiler."""
        primary = siege_dread["weapons"][0]
        assert primary["type"] == "heavy_siege_coiler_mk3"

    def test_still_has_heavy_coilguns(self, siege_dread):
        """Verify heavy coilgun turrets are preserved."""
        heavy_turrets = [w for w in siege_dread["weapons"]
                        if w["type"] == "heavy_coilgun_mk3"]
        assert len(heavy_turrets) == 5

    def test_still_has_point_defense(self, siege_dread):
        """Verify PD turrets are preserved."""
        pd_turrets = [w for w in siege_dread["weapons"]
                     if w["type"] == "pd_laser"]
        assert len(pd_turrets) == 4

    def test_weapons_summary(self, siege_dread):
        """Verify weapons summary counts."""
        summary = siege_dread["weapons_summary"]
        assert summary["siege_coilers"] == 1
        assert summary["heavy_coilguns"] == 5
        assert summary["point_defense"] == 4


# =============================================================================
# MASS CALCULATIONS
# =============================================================================

class TestMassCalculations:
    """Verify mass calculations are correct."""

    def test_weapon_mass_increase(self, siege_dread, standard_dread):
        """Verify weapon mass increased by 800 tons."""
        diff = siege_dread["mass_breakdown"]["weapons"] - standard_dread["mass_breakdown"]["weapons"]
        assert diff == 800

    def test_ammo_mass_increase(self, siege_dread, standard_dread):
        """Verify ammo mass increased by ~35 tons."""
        diff = siege_dread["mass_breakdown"]["ammunition"] - standard_dread["mass_breakdown"]["ammunition"]
        assert 35 <= diff <= 36  # 35.4 rounded

    def test_dry_mass_total(self, siege_dread):
        """Verify total dry mass is correct.

        Note: attitude_control is listed separately and not included in total_dry
        (same convention as standard dreadnought).
        """
        mb = siege_dread["mass_breakdown"]
        calculated = (mb["hull"] + mb["weapons"] + mb["ammunition"] +
                     mb["thermal_systems"] + mb["crew_and_misc"] +
                     mb["reactor"] + mb["armor"])
        # Attitude control (25 tons) is listed separately, not in total_dry
        assert abs(calculated - mb["total_dry"]) < 1

    def test_wet_mass_total(self, siege_dread):
        """Verify total wet mass is correct."""
        mb = siege_dread["mass_breakdown"]
        assert abs(mb["total_wet"] - mb["total_dry"] - mb["propellant"]) < 1

    def test_mass_ratio_for_delta_v(self, siege_dread, fleet_data):
        """Verify mass ratio gives 500 km/s delta-v."""
        import math
        v_e = fleet_data["constants"]["exhaust_velocity_kps"]
        wet = siege_dread["mass_breakdown"]["total_wet"]
        dry = siege_dread["mass_breakdown"]["total_dry"]
        mass_ratio = wet / dry
        delta_v = v_e * math.log(mass_ratio)
        assert 495 <= delta_v <= 505  # Within 1%


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Verify performance calculations."""

    def test_acceleration_reduced(self, siege_dread, standard_dread):
        """Verify acceleration is lower than standard."""
        assert siege_dread["performance"]["combat_acceleration_g"] < standard_dread["performance"]["combat_acceleration_g"]

    def test_acceleration_value(self, siege_dread):
        """Verify calculated acceleration matches spec."""
        thrust_n = 58.56e6  # 58.56 MN
        mass_kg = siege_dread["mass_breakdown"]["total_wet"] * 1000
        accel_ms2 = thrust_n / mass_kg
        accel_g = accel_ms2 / 9.81
        assert abs(accel_g - siege_dread["performance"]["combat_acceleration_g"]) < 0.01

    def test_delta_v_preserved(self, siege_dread, standard_dread):
        """Verify delta-v is still 500 km/s."""
        assert siege_dread["performance"]["delta_v_kps"] == standard_dread["performance"]["delta_v_kps"]


# =============================================================================
# ATTITUDE CONTROL TESTS
# =============================================================================

class TestAttitudeControl:
    """Verify attitude control calculations."""

    def test_moi_increased(self, siege_dread, standard_dread):
        """Verify moment of inertia increased with mass."""
        siege_moi = siege_dread["attitude_control"]["moment_of_inertia"]["pitch_yaw_kg_m2"]
        std_moi = standard_dread["attitude_control"]["moment_of_inertia"]["pitch_yaw_kg_m2"]
        assert siege_moi > std_moi

    def test_moi_scales_with_mass(self, siege_dread, standard_dread):
        """Verify MoI scales linearly with mass (same geometry)."""
        siege_moi = siege_dread["attitude_control"]["moment_of_inertia"]["pitch_yaw_kg_m2"]
        std_moi = standard_dread["attitude_control"]["moment_of_inertia"]["pitch_yaw_kg_m2"]
        mass_ratio = siege_dread["mass_breakdown"]["total_wet"] / standard_dread["mass_breakdown"]["total_wet"]
        moi_ratio = siege_moi / std_moi
        assert abs(moi_ratio - mass_ratio) < 0.01

    def test_angular_accel_reduced(self, siege_dread, standard_dread):
        """Verify angular acceleration decreased (same torque, higher mass)."""
        siege_alpha = siege_dread["attitude_control"]["thrust_vectoring"]["angular_accel_deg_s2"]
        std_alpha = standard_dread["attitude_control"]["thrust_vectoring"]["angular_accel_deg_s2"]
        assert siege_alpha < std_alpha

    def test_rotation_time_increased(self, siege_dread, standard_dread):
        """Verify rotation takes longer."""
        siege_t90 = siege_dread["attitude_control"]["thrust_vectoring"]["time_to_rotate_90_deg_s"]
        std_t90 = standard_dread["attitude_control"]["thrust_vectoring"]["time_to_rotate_90_deg_s"]
        assert siege_t90 > std_t90

    def test_rotation_time_formula(self, siege_dread):
        """Verify rotation time follows t = 2*sqrt(theta/alpha)."""
        import math
        alpha = siege_dread["attitude_control"]["thrust_vectoring"]["angular_accel_deg_s2"]
        t_90 = siege_dread["attitude_control"]["thrust_vectoring"]["time_to_rotate_90_deg_s"]
        expected = 2 * math.sqrt(90 / alpha)
        assert abs(t_90 - expected) < 0.5


# =============================================================================
# COMBAT CAPABILITY COMPARISON
# =============================================================================

class TestCombatCapability:
    """Compare combat capabilities between variants."""

    def test_alpha_strike_increase(self, siege_dread, standard_dread, fleet_data):
        """Verify massive alpha strike increase."""
        siege_weapon = fleet_data["weapon_types"]["heavy_siege_coiler_mk3"]
        std_weapon = fleet_data["weapon_types"]["spinal_coiler_mk3"]

        siege_alpha = siege_weapon["kinetic_energy_gj"] * siege_weapon["salvo_size"]
        std_alpha = std_weapon["kinetic_energy_gj"]

        ratio = siege_alpha / std_alpha
        assert ratio > 5  # More than 5x damage per salvo

    def test_sustained_dps_comparison(self, fleet_data):
        """Compare sustained DPS between weapons."""
        siege = fleet_data["weapon_types"]["heavy_siege_coiler_mk3"]
        spinal = fleet_data["weapon_types"]["spinal_coiler_mk3"]

        # Siege: 21.75 GJ per 60s cycle
        siege_cycle = (siege["salvo_size"] - 1) * siege["intra_salvo_cooldown_s"] + siege["cooldown_s"]
        siege_dps = (siege["kinetic_energy_gj"] * siege["salvo_size"]) / siege_cycle

        # Spinal: 4.29 GJ per 15s
        spinal_dps = spinal["kinetic_energy_gj"] / spinal["cooldown_s"]

        # Siege actually has higher sustained DPS
        assert siege_dps > spinal_dps

    def test_armor_unchanged(self, siege_dread, standard_dread):
        """Verify armor is identical between variants."""
        assert siege_dread["armor"] == standard_dread["armor"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
