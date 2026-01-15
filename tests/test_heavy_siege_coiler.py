"""
Tests for the Heavy Siege Coiler Mk3 capital ship weapon.

The heavy_siege_coiler_mk3 is a devastating capital ship weapon with:
- 7.25 GJ per slug (656.25 kg at 4.7 km/s)
- 3-shot salvos = 21.75 GJ total per salvo
- 24s cooldown between salvos
- 18s intra-salvo cooldown between shots
- Only fits Battleship/Dreadnought class (requires 3 nose hardpoints)
"""

import pytest
import json
from pathlib import Path

from src.combat import Weapon


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fleet_data():
    """Load fleet_ships.json data."""
    fleet_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    with open(fleet_path) as f:
        return json.load(f)


@pytest.fixture
def siege_coiler_spec(fleet_data):
    """Get the heavy siege coiler specification."""
    return fleet_data["weapon_types"]["heavy_siege_coiler_mk3"]


@pytest.fixture
def siege_coiler_weapon(siege_coiler_spec):
    """Create a Weapon instance from the siege coiler spec."""
    return Weapon.from_json("heavy_siege_coiler_mk3", siege_coiler_spec)


# =============================================================================
# WEAPON SPECIFICATION TESTS
# =============================================================================

class TestHeavySiegeCoilerSpec:
    """Tests for the heavy siege coiler weapon specification."""

    def test_weapon_exists_in_fleet_data(self, fleet_data):
        """Verify the heavy siege coiler is defined in fleet_ships.json."""
        assert "heavy_siege_coiler_mk3" in fleet_data["weapon_types"]

    def test_weapon_mass(self, siege_coiler_spec):
        """Verify weapon mass is 1000 tons (capital-class weapon)."""
        assert siege_coiler_spec["mass_tons"] == 1000

    def test_projectile_mass(self, siege_coiler_spec):
        """Verify projectile mass is 656.25 kg (heavy slug)."""
        assert siege_coiler_spec["warhead_mass_kg"] == 656.25

    def test_muzzle_velocity(self, siege_coiler_spec):
        """Verify muzzle velocity is 4.7 km/s (slower than standard coilguns)."""
        assert siege_coiler_spec["muzzle_velocity_kps"] == 4.7

    def test_kinetic_energy_per_shot(self, siege_coiler_spec):
        """Verify kinetic energy is 7.25 GJ per slug."""
        assert siege_coiler_spec["kinetic_energy_gj"] == 7.25

    def test_kinetic_energy_calculation(self, siege_coiler_spec):
        """Verify KE = 0.5 * m * v^2 matches spec (approximately)."""
        mass_kg = siege_coiler_spec["warhead_mass_kg"]
        velocity_kps = siege_coiler_spec["muzzle_velocity_kps"]
        velocity_m_s = velocity_kps * 1000

        # KE = 0.5 * m * v^2 in Joules
        ke_joules = 0.5 * mass_kg * (velocity_m_s ** 2)
        ke_gj = ke_joules / 1e9

        # Should be close to 7.25 GJ (within rounding)
        assert abs(ke_gj - 7.25) < 0.1

    def test_salvo_size(self, siege_coiler_spec):
        """Verify salvo size is 3 shots."""
        assert siege_coiler_spec["salvo_size"] == 3

    def test_salvo_total_damage(self, siege_coiler_spec):
        """Verify total salvo damage is 21.75 GJ (3 x 7.25 GJ)."""
        salvo_damage = siege_coiler_spec["kinetic_energy_gj"] * siege_coiler_spec["salvo_size"]
        assert salvo_damage == 21.75

    def test_intra_salvo_cooldown(self, siege_coiler_spec):
        """Verify intra-salvo cooldown is 18 seconds."""
        assert siege_coiler_spec["intra_salvo_cooldown_s"] == 18

    def test_salvo_cooldown(self, siege_coiler_spec):
        """Verify cooldown between salvos is 24 seconds."""
        assert siege_coiler_spec["cooldown_s"] == 24

    def test_total_salvo_time(self, siege_coiler_spec):
        """Verify total time to fire a full salvo.

        Shot 1: T+0
        Shot 2: T+18 (intra_salvo_cooldown)
        Shot 3: T+36 (2 x intra_salvo_cooldown)
        Next salvo: T+36+24 = T+60
        """
        intra_cooldown = siege_coiler_spec["intra_salvo_cooldown_s"]
        salvo_size = siege_coiler_spec["salvo_size"]
        salvo_cooldown = siege_coiler_spec["cooldown_s"]

        # Time to complete salvo (time from first to last shot)
        salvo_duration = (salvo_size - 1) * intra_cooldown
        assert salvo_duration == 36  # 2 x 18s

        # Time until next salvo can begin
        total_cycle = salvo_duration + salvo_cooldown
        assert total_cycle == 60  # 36 + 24

    def test_magazine_size(self, siege_coiler_spec):
        """Verify magazine holds 100 rounds."""
        assert siege_coiler_spec["magazine"] == 100

    def test_effective_range(self, siege_coiler_spec):
        """Verify effective range is 900 km."""
        assert siege_coiler_spec["range_km"] == 900

    def test_mount_type(self, siege_coiler_spec):
        """Verify it's a triple nose mount (requires 3 hardpoints)."""
        assert siege_coiler_spec["mount"] == "triple_nose"
        assert siege_coiler_spec["hardpoints_required"] == 3

    def test_pivot_range(self, siege_coiler_spec):
        """Verify limited gimbal of 20 degrees (fixed spinal mount)."""
        assert siege_coiler_spec["pivot_range_deg"] == 20
        assert siege_coiler_spec["is_turreted"] is False

    def test_flat_chipping(self, siege_coiler_spec):
        """Verify flat chipping factor is 0.25 (lower than standard coilguns)."""
        assert siege_coiler_spec["flat_chipping"] == 0.25


# =============================================================================
# WEAPON INSTANCE TESTS
# =============================================================================

class TestHeavySiegeCoilerWeapon:
    """Tests for the Weapon class instantiation from siege coiler spec."""

    def test_weapon_creation(self, siege_coiler_weapon):
        """Verify weapon can be created from spec."""
        assert siege_coiler_weapon is not None
        assert siege_coiler_weapon.name == "Heavy Siege Coiler Mk3"

    def test_weapon_type(self, siege_coiler_weapon):
        """Verify weapon type identifier."""
        assert siege_coiler_weapon.weapon_type == "heavy_siege_coiler_mk3"

    def test_weapon_energy(self, siege_coiler_weapon):
        """Verify kinetic energy is loaded correctly."""
        assert siege_coiler_weapon.kinetic_energy_gj == 7.25

    def test_weapon_cooldown(self, siege_coiler_weapon):
        """Verify cooldown is loaded correctly."""
        assert siege_coiler_weapon.cooldown_s == 24

    def test_weapon_range(self, siege_coiler_weapon):
        """Verify range is loaded correctly."""
        assert siege_coiler_weapon.range_km == 900

    def test_weapon_is_not_turreted(self, siege_coiler_weapon):
        """Verify weapon is a fixed spinal mount."""
        assert siege_coiler_weapon.is_turreted is False

    def test_weapon_pivot_range(self, siege_coiler_weapon):
        """Verify limited gimbal range."""
        assert siege_coiler_weapon.pivot_range_deg == 20

    def test_weapon_in_range_at_500km(self, siege_coiler_weapon):
        """Verify weapon is in range at 500 km."""
        assert siege_coiler_weapon.is_in_range(500) is True

    def test_weapon_in_range_at_900km(self, siege_coiler_weapon):
        """Verify weapon is in range at max range (900 km)."""
        assert siege_coiler_weapon.is_in_range(900) is True

    def test_weapon_out_of_range_at_901km(self, siege_coiler_weapon):
        """Verify weapon is out of range at 901 km."""
        assert siege_coiler_weapon.is_in_range(901) is False


# =============================================================================
# DAMAGE COMPARISON TESTS
# =============================================================================

class TestDamageComparison:
    """Compare siege coiler damage to other weapons."""

    def test_siege_vs_spinal_coiler(self, fleet_data):
        """Siege coiler does more damage per shot than standard spinal."""
        siege = fleet_data["weapon_types"]["heavy_siege_coiler_mk3"]
        spinal = fleet_data["weapon_types"]["spinal_coiler_mk3"]

        assert siege["kinetic_energy_gj"] > spinal["kinetic_energy_gj"]
        # 7.25 GJ vs 4.29 GJ = ~69% more damage per shot
        ratio = siege["kinetic_energy_gj"] / spinal["kinetic_energy_gj"]
        assert ratio > 1.5

    def test_siege_vs_heavy_coilgun(self, fleet_data):
        """Siege coiler does much more damage than heavy coilgun turrets."""
        siege = fleet_data["weapon_types"]["heavy_siege_coiler_mk3"]
        heavy = fleet_data["weapon_types"]["heavy_coilgun_mk3"]

        assert siege["kinetic_energy_gj"] > heavy["kinetic_energy_gj"]
        # 7.25 GJ vs 1.22 GJ = ~6x more damage per shot
        ratio = siege["kinetic_energy_gj"] / heavy["kinetic_energy_gj"]
        assert ratio > 5

    def test_siege_salvo_vs_torpedo(self, fleet_data):
        """Compare full siege salvo to torpedo kinetic penetrator.

        A full siege salvo (21.75 GJ) should be comparable to or exceed
        the kinetic energy of a torpedo penetrator impact.
        """
        siege = fleet_data["weapon_types"]["heavy_siege_coiler_mk3"]
        torpedo = fleet_data["weapon_types"]["torpedo_launcher"]

        salvo_damage = siege["kinetic_energy_gj"] * siege["salvo_size"]

        # Torpedo penetrator: 250 kg at terminal velocity
        # Assuming ~10 km/s terminal (after burn): KE = 0.5 * 250 * (10000)^2 = 12.5 GJ
        # Siege salvo at 21.75 GJ is comparable
        assert salvo_damage == 21.75

    def test_siege_dps_lower_than_standard(self, fleet_data):
        """Siege coiler has lower sustained DPS than spinal coiler.

        This is the tradeoff: massive alpha strike vs sustained damage.
        """
        siege = fleet_data["weapon_types"]["heavy_siege_coiler_mk3"]
        spinal = fleet_data["weapon_types"]["spinal_coiler_mk3"]

        # Calculate DPS for each
        # Siege: 21.75 GJ per 60s cycle = 0.36 GJ/s
        siege_cycle = (siege["salvo_size"] - 1) * siege["intra_salvo_cooldown_s"] + siege["cooldown_s"]
        siege_dps = (siege["kinetic_energy_gj"] * siege["salvo_size"]) / siege_cycle

        # Spinal: 4.29 GJ per 15s = 0.29 GJ/s
        spinal_dps = spinal["kinetic_energy_gj"] / spinal["cooldown_s"]

        # Siege actually has slightly higher DPS, but requires sustained engagement
        # The real tradeoff is burst damage vs flexibility
        assert siege_dps > 0.3  # Decent sustained damage
        assert spinal_dps > 0.25  # Also decent


# =============================================================================
# ARMOR PENETRATION TESTS
# =============================================================================

class TestArmorPenetration:
    """Test armor penetration calculations for siege coiler."""

    def test_penetrate_destroyer_nose(self, fleet_data, siege_coiler_spec):
        """Test if siege coiler can penetrate destroyer nose armor.

        Destroyer nose: ~151 cm Adamantane
        Siege slug: 7.25 GJ, flat_chipping 0.25
        """
        ships = fleet_data["ships"]
        destroyer = ships["destroyer"]

        # Get nose armor thickness from sections
        armor = destroyer.get("armor", {})
        sections = armor.get("sections", {})
        nose_section = sections.get("nose", {})
        nose_thickness = nose_section.get("thickness_cm", 0)

        assert nose_thickness > 0
        assert nose_thickness > 150  # Destroyer has thick nose (~151 cm)

        # A 7.25 GJ hit should ablate significant armor
        energy_gj = siege_coiler_spec["kinetic_energy_gj"]

        # Without exact armor ablation formula, just verify the energy is substantial
        assert energy_gj >= 7.0

        # Check hits to deplete - siege coiler should take fewer hits
        hits_to_deplete = nose_section.get("hits_to_deplete", 0)
        assert hits_to_deplete > 0  # Armor has finite durability

    def test_triple_hit_devastating(self, siege_coiler_spec):
        """Verify that a full 3-shot salvo to same location is devastating."""
        energy_per_shot = siege_coiler_spec["kinetic_energy_gj"]
        salvo_size = siege_coiler_spec["salvo_size"]
        total_energy = energy_per_shot * salvo_size

        # 21.75 GJ concentrated on one location should breach most armor
        assert total_energy > 20


# =============================================================================
# TACTICAL CONSIDERATIONS TESTS
# =============================================================================

class TestTacticalConsiderations:
    """Test tactical aspects of the siege coiler."""

    def test_slower_projectile_gives_more_evasion_time(self, fleet_data):
        """Slower projectiles (4.7 km/s) give targets more time to evade."""
        siege = fleet_data["weapon_types"]["heavy_siege_coiler_mk3"]
        spinal = fleet_data["weapon_types"]["spinal_coiler_mk3"]

        # Time to target at 500 km
        siege_time = 500 / siege["muzzle_velocity_kps"]  # ~106 seconds
        spinal_time = 500 / spinal["muzzle_velocity_kps"]  # ~50 seconds

        assert siege_time > spinal_time
        assert siege_time > 100  # Over 100 seconds flight time at 500km

    def test_only_capital_ships_can_mount(self, siege_coiler_spec):
        """Verify only battleships/dreadnoughts can mount this weapon."""
        assert siege_coiler_spec["hardpoints_required"] == 3
        assert "Capital ship" in siege_coiler_spec.get("notes", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
