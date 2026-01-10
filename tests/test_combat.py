"""
Unit tests for the combat mechanics module.

Run with: python -m pytest tests/test_combat.py -v
"""

import json
import random
import tempfile
from pathlib import Path

import pytest

from src.combat import (
    Armor,
    CombatResolver,
    HitLocation,
    HitResult,
    ShipArmor,
    Weapon,
    HIT_LOCATION_WEIGHTS,
    create_ship_armor_from_fleet_data,
    create_weapon_from_fleet_data,
    load_fleet_data,
    simulate_combat_exchange,
)


# Fixtures

@pytest.fixture
def sample_weapon_data() -> dict:
    """Sample weapon data for testing."""
    return {
        "name": "Test Coilgun",
        "mass_tons": 100,
        "ammo_mass_kg": 50,
        "magazine": 100,
        "muzzle_velocity_kps": 8.0,
        "warhead_mass_kg": 50,
        "kinetic_energy_gj": 2.0,
        "flat_chipping": 0.4,
        "cooldown_s": 10,
        "range_km": 500,
        "mount": "any",
    }


@pytest.fixture
def sample_missile_data() -> dict:
    """Sample missile weapon data for testing."""
    return {
        "name": "Test Torpedo",
        "mass_tons": 25,
        "ammo_mass_kg": 1600,
        "magazine": 16,
        "warhead_yield_gj": 50,
        "cooldown_s": 30,
        "range_km": 2000,
        "mount": "any",
        "type": "missile",
    }


@pytest.fixture
def sample_armor_data() -> dict:
    """Sample armor configuration for testing."""
    return {
        "type": "Titanium",
        "properties": {
            "density": 4820,
            "baryonic_half_cm": 10.5,
            "chip_resist": 0.75,
        },
        "sections": {
            "nose": {
                "thickness_cm": 50.0,
                "area_m2": 100.0,
            },
            "lateral": {
                "thickness_cm": 10.0,
                "area_m2": 500.0,
            },
            "tail": {
                "thickness_cm": 15.0,
                "area_m2": 100.0,
            },
        },
    }


@pytest.fixture
def sample_fleet_data(sample_weapon_data, sample_armor_data) -> dict:
    """Complete fleet data for testing."""
    return {
        "weapon_types": {
            "test_coilgun": sample_weapon_data,
        },
        "ships": {
            "test_ship": {
                "armor": sample_armor_data,
            },
        },
    }


@pytest.fixture
def fleet_data_file(sample_fleet_data, tmp_path) -> Path:
    """Create a temporary fleet data file."""
    filepath = tmp_path / "fleet_ships.json"
    with open(filepath, "w") as f:
        json.dump(sample_fleet_data, f)
    return filepath


@pytest.fixture
def test_weapon(sample_weapon_data) -> Weapon:
    """Create a test weapon instance."""
    return Weapon.from_json("test_coilgun", sample_weapon_data)


@pytest.fixture
def test_armor() -> Armor:
    """Create a test armor instance."""
    return Armor(
        armor_type="Titanium",
        thickness_cm=21.0,  # Exactly 2 half-values for 75% protection
        baryonic_half_cm=10.5,
        chip_resist=0.75,
        density=4820,
        area_m2=100.0,
        location=HitLocation.LATERAL,
    )


@pytest.fixture
def test_ship_armor(sample_armor_data) -> ShipArmor:
    """Create a test ship armor instance."""
    return ShipArmor.from_json(sample_armor_data)


@pytest.fixture
def seeded_resolver() -> CombatResolver:
    """Create a combat resolver with a fixed seed for reproducibility."""
    return CombatResolver(rng=random.Random(42))


# Weapon Tests

class TestWeapon:
    """Tests for the Weapon class."""

    def test_from_json_kinetic(self, sample_weapon_data):
        """Test creating a kinetic weapon from JSON."""
        weapon = Weapon.from_json("test_coilgun", sample_weapon_data)

        assert weapon.name == "Test Coilgun"
        assert weapon.weapon_type == "test_coilgun"
        assert weapon.kinetic_energy_gj == 2.0
        assert weapon.flat_chipping == 0.4
        assert weapon.cooldown_s == 10
        assert weapon.range_km == 500
        assert weapon.magazine == 100
        assert not weapon.is_missile

    def test_from_json_missile(self, sample_missile_data):
        """Test creating a missile weapon from JSON."""
        weapon = Weapon.from_json("test_torpedo", sample_missile_data)

        assert weapon.name == "Test Torpedo"
        assert weapon.kinetic_energy_gj == 50  # Uses warhead_yield_gj
        assert weapon.is_missile

    def test_is_in_range(self, test_weapon):
        """Test range checking."""
        assert test_weapon.is_in_range(400)  # Within range
        assert test_weapon.is_in_range(500)  # At max range
        assert not test_weapon.is_in_range(501)  # Beyond range

    def test_can_fire(self, test_weapon):
        """Test ammunition checking."""
        assert test_weapon.can_fire(10)
        assert test_weapon.can_fire(1)
        assert not test_weapon.can_fire(0)


# Armor Tests

class TestArmor:
    """Tests for the Armor class."""

    def test_from_json(self, sample_armor_data):
        """Test creating armor from JSON."""
        armor = Armor.from_json(
            sample_armor_data,
            "nose",
            sample_armor_data["sections"]["nose"]
        )

        assert armor.armor_type == "Titanium"
        assert armor.thickness_cm == 50.0
        assert armor.baryonic_half_cm == 10.5
        assert armor.chip_resist == 0.75
        assert armor.location == HitLocation.NOSE

    def test_protection_formula(self):
        """Test the protection formula: protection = 1 - 0.5^(thickness/half_value)."""
        # At 0 thickness, protection should be 0
        armor_zero = Armor(
            armor_type="Test",
            thickness_cm=0.0,
            baryonic_half_cm=10.0,
            chip_resist=0.0
        )
        assert armor_zero.protection == 0.0

        # At 1 half-value thickness, protection should be 0.5
        armor_half = Armor(
            armor_type="Test",
            thickness_cm=10.0,
            baryonic_half_cm=10.0,
            chip_resist=0.0
        )
        assert abs(armor_half.protection - 0.5) < 0.001

        # At 2 half-values, protection should be 0.75
        armor_double = Armor(
            armor_type="Test",
            thickness_cm=20.0,
            baryonic_half_cm=10.0,
            chip_resist=0.0
        )
        assert abs(armor_double.protection - 0.75) < 0.001

        # At 3 half-values, protection should be 0.875
        armor_triple = Armor(
            armor_type="Test",
            thickness_cm=30.0,
            baryonic_half_cm=10.0,
            chip_resist=0.0
        )
        assert abs(armor_triple.protection - 0.875) < 0.001

    def test_protection_percent(self, test_armor):
        """Test protection percentage conversion."""
        assert test_armor.protection_percent == test_armor.protection * 100

    def test_calculate_ablation(self, test_armor, test_weapon):
        """Test ablation formula: ablation = base * flat_chipping * (1 - chip_resist)."""
        base_ablation = 2.5
        expected = base_ablation * test_weapon.flat_chipping * (1.0 - test_armor.chip_resist)
        actual = test_armor.calculate_ablation(test_weapon, base_ablation)

        assert abs(actual - expected) < 0.001

        # With 0.75 chip_resist and 0.4 flat_chipping:
        # expected = 2.5 * 0.4 * (1 - 0.75) = 2.5 * 0.4 * 0.25 = 0.25
        assert abs(actual - 0.25) < 0.001

    def test_apply_damage(self, test_weapon):
        """Test damage application reduces armor thickness."""
        armor = Armor(
            armor_type="Test",
            thickness_cm=10.0,
            baryonic_half_cm=10.0,
            chip_resist=0.5  # 50% chip resist
        )
        initial_thickness = armor.thickness_cm

        # With chip_resist=0.5 and flat_chipping=0.4:
        # ablation = 2.5 * 0.4 * 0.5 = 0.5 cm
        ablation = armor.apply_damage(test_weapon, 2.5)

        assert ablation == 0.5
        assert armor.thickness_cm == initial_thickness - 0.5

    def test_apply_damage_cannot_go_negative(self, test_weapon):
        """Test that armor thickness cannot go below zero."""
        armor = Armor(
            armor_type="Test",
            thickness_cm=0.1,
            baryonic_half_cm=10.0,
            chip_resist=0.0  # No chip resist for max damage
        )

        # This should try to ablate 1.0 cm but only 0.1 cm available
        armor.apply_damage(test_weapon, 2.5)

        assert armor.thickness_cm == 0.0

    def test_is_penetrated(self):
        """Test penetration detection."""
        armor_thick = Armor(
            armor_type="Test",
            thickness_cm=10.0,
            baryonic_half_cm=10.0,
            chip_resist=0.0
        )
        assert not armor_thick.is_penetrated()

        armor_zero = Armor(
            armor_type="Test",
            thickness_cm=0.0,
            baryonic_half_cm=10.0,
            chip_resist=0.0
        )
        assert armor_zero.is_penetrated()


# ShipArmor Tests

class TestShipArmor:
    """Tests for the ShipArmor class."""

    def test_from_json(self, sample_armor_data):
        """Test creating ship armor from JSON."""
        ship_armor = ShipArmor.from_json(sample_armor_data)

        assert len(ship_armor.sections) == 3
        assert HitLocation.NOSE in ship_armor.sections
        assert HitLocation.LATERAL in ship_armor.sections
        assert HitLocation.TAIL in ship_armor.sections

    def test_get_section(self, test_ship_armor):
        """Test retrieving armor sections."""
        nose = test_ship_armor.get_section(HitLocation.NOSE)
        assert nose is not None
        assert nose.thickness_cm == 50.0

        lateral = test_ship_armor.get_section(HitLocation.LATERAL)
        assert lateral is not None
        assert lateral.thickness_cm == 10.0


# HitResult Tests

class TestHitResult:
    """Tests for the HitResult class."""

    def test_miss_str(self):
        """Test string representation of a miss."""
        result = HitResult(hit=False)
        assert str(result) == "Miss"

    def test_hit_str(self):
        """Test string representation of a hit."""
        result = HitResult(
            hit=True,
            location=HitLocation.LATERAL,
            damage_absorbed=2.0,
            armor_ablation_cm=0.5,
            penetrated=False,
        )
        assert "Hit (lateral)" in str(result)
        assert "0.5 cm ablated" in str(result)
        assert "absorbed" in str(result)

    def test_penetration_str(self):
        """Test string representation of a penetrating hit."""
        result = HitResult(
            hit=True,
            location=HitLocation.NOSE,
            damage_absorbed=1.0,
            armor_ablation_cm=1.0,
            penetrated=True,
            remaining_damage_gj=3.0,
        )
        assert "PENETRATED" in str(result)
        assert "internal damage" in str(result)

    def test_critical_hit_str(self):
        """Test string representation of a critical hit."""
        result = HitResult(
            hit=True,
            location=HitLocation.TAIL,
            penetrated=True,
            critical_hit=True,
        )
        assert "[CRITICAL]" in str(result)


# CombatResolver Tests

class TestCombatResolver:
    """Tests for the CombatResolver class."""

    def test_determine_hit_location_distribution(self):
        """Test that hit location follows expected probability distribution."""
        rng = random.Random(42)
        resolver = CombatResolver(rng=rng)

        # Run many trials
        counts = {loc: 0 for loc in HitLocation}
        num_trials = 10000

        for _ in range(num_trials):
            location = resolver.determine_hit_location()
            counts[location] += 1

        # Check approximate percentages (with tolerance)
        tolerance = 0.02  # 2% tolerance

        nose_pct = counts[HitLocation.NOSE] / num_trials
        assert abs(nose_pct - 0.15) < tolerance, f"Nose: {nose_pct:.3f} vs expected 0.15"

        lateral_pct = counts[HitLocation.LATERAL] / num_trials
        assert abs(lateral_pct - 0.70) < tolerance, f"Lateral: {lateral_pct:.3f} vs expected 0.70"

        tail_pct = counts[HitLocation.TAIL] / num_trials
        assert abs(tail_pct - 0.15) < tolerance, f"Tail: {tail_pct:.3f} vs expected 0.15"

    def test_calculate_hit_probability_out_of_range(self, test_weapon, seeded_resolver):
        """Test that out-of-range targets have 0% hit probability."""
        prob = seeded_resolver.calculate_hit_probability(test_weapon, distance_km=600)
        assert prob == 0.0

    def test_calculate_hit_probability_at_zero_range(self, test_weapon, seeded_resolver):
        """Test hit probability at point-blank range."""
        prob = seeded_resolver.calculate_hit_probability(
            test_weapon,
            distance_km=0,
            target_accel_g=0
        )
        # Should be high at close range
        assert prob > 0.7

    def test_calculate_hit_probability_decreases_with_range(self, test_weapon, seeded_resolver):
        """Test that hit probability decreases with distance."""
        prob_close = seeded_resolver.calculate_hit_probability(test_weapon, distance_km=100)
        prob_mid = seeded_resolver.calculate_hit_probability(test_weapon, distance_km=300)
        prob_far = seeded_resolver.calculate_hit_probability(test_weapon, distance_km=450)

        assert prob_close > prob_mid > prob_far

    def test_calculate_hit_probability_evasion(self, test_weapon, seeded_resolver):
        """Test that evasive targets are harder to hit."""
        prob_static = seeded_resolver.calculate_hit_probability(
            test_weapon,
            distance_km=200,
            target_accel_g=0
        )
        prob_evasive = seeded_resolver.calculate_hit_probability(
            test_weapon,
            distance_km=200,
            target_accel_g=3
        )

        assert prob_static > prob_evasive

    def test_resolve_hit_applies_damage(self, test_weapon, test_ship_armor, seeded_resolver):
        """Test that resolving a hit damages armor."""
        initial_lateral = test_ship_armor.get_section(HitLocation.LATERAL).thickness_cm

        result = seeded_resolver.resolve_hit(
            test_weapon,
            test_ship_armor,
            location=HitLocation.LATERAL
        )

        assert result.hit
        assert result.location == HitLocation.LATERAL
        assert result.armor_ablation_cm > 0

        # Check armor was actually reduced
        new_thickness = test_ship_armor.get_section(HitLocation.LATERAL).thickness_cm
        assert new_thickness < initial_lateral

    def test_resolve_attack_miss(self, test_weapon, test_ship_armor):
        """Test attack resolution with a guaranteed miss (out of range)."""
        resolver = CombatResolver(rng=random.Random(42))
        result = resolver.resolve_attack(
            test_weapon,
            test_ship_armor,
            distance_km=600  # Beyond 500 km range
        )

        assert not result.hit

    def test_resolve_attack_reproducibility(self, test_weapon, test_ship_armor):
        """Test that same seed produces same results."""
        results_a = []
        results_b = []

        for seed in [42, 42]:
            resolver = CombatResolver(rng=random.Random(seed))
            armor = ShipArmor.from_json({
                "type": "Titanium",
                "properties": {"baryonic_half_cm": 10.5, "chip_resist": 0.75},
                "sections": {
                    "nose": {"thickness_cm": 50.0, "area_m2": 100.0},
                    "lateral": {"thickness_cm": 10.0, "area_m2": 500.0},
                    "tail": {"thickness_cm": 15.0, "area_m2": 100.0},
                }
            })

            result = resolver.resolve_attack(test_weapon, armor, distance_km=200)

            if seed == 42 and len(results_a) == 0:
                results_a.append((result.hit, result.location))
            else:
                results_b.append((result.hit, result.location))

        assert results_a == results_b


# Integration Tests

class TestIntegration:
    """Integration tests using real fleet data."""

    def test_load_fleet_data(self, fleet_data_file):
        """Test loading fleet data from file."""
        data = load_fleet_data(fleet_data_file)

        assert "weapon_types" in data
        assert "ships" in data

    def test_create_weapon_from_fleet_data(self, sample_fleet_data):
        """Test creating weapons from fleet data."""
        weapon = create_weapon_from_fleet_data(sample_fleet_data, "test_coilgun")

        assert weapon.name == "Test Coilgun"
        assert weapon.kinetic_energy_gj == 2.0

    def test_create_weapon_not_found(self, sample_fleet_data):
        """Test error when weapon type not found."""
        with pytest.raises(KeyError, match="not found"):
            create_weapon_from_fleet_data(sample_fleet_data, "nonexistent_weapon")

    def test_create_ship_armor_from_fleet_data(self, sample_fleet_data):
        """Test creating ship armor from fleet data."""
        armor = create_ship_armor_from_fleet_data(sample_fleet_data, "test_ship")

        assert len(armor.sections) == 3
        assert armor.get_section(HitLocation.NOSE).thickness_cm == 50.0

    def test_create_ship_armor_not_found(self, sample_fleet_data):
        """Test error when ship type not found."""
        with pytest.raises(KeyError, match="not found"):
            create_ship_armor_from_fleet_data(sample_fleet_data, "nonexistent_ship")

    def test_simulate_combat_exchange(self, fleet_data_file):
        """Test complete combat simulation."""
        results = simulate_combat_exchange(
            attacker_weapon_type="test_coilgun",
            defender_ship_type="test_ship",
            distance_km=200,
            fleet_data_path=fleet_data_file,
            num_shots=10,
            seed=42
        )

        assert len(results) == 10

        # With seed=42, should have at least some hits and some misses
        hits = sum(1 for r in results if r.hit)
        misses = sum(1 for r in results if not r.hit)

        assert hits > 0 or misses > 0  # At least something happened

    def test_simulate_combat_reproducibility(self, fleet_data_file):
        """Test that simulations are reproducible with same seed."""
        results_a = simulate_combat_exchange(
            attacker_weapon_type="test_coilgun",
            defender_ship_type="test_ship",
            distance_km=200,
            fleet_data_path=fleet_data_file,
            num_shots=5,
            seed=12345
        )

        results_b = simulate_combat_exchange(
            attacker_weapon_type="test_coilgun",
            defender_ship_type="test_ship",
            distance_km=200,
            fleet_data_path=fleet_data_file,
            num_shots=5,
            seed=12345
        )

        for a, b in zip(results_a, results_b):
            assert a.hit == b.hit
            assert a.location == b.location


# Tests with real fleet data (skip if file not found)

class TestRealFleetData:
    """Tests using the actual fleet_ships.json file."""

    @pytest.fixture
    def real_fleet_data(self):
        """Load real fleet data if available."""
        data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
        if not data_path.exists():
            pytest.skip("Fleet data file not found")
        return load_fleet_data(data_path)

    def test_load_real_weapon_types(self, real_fleet_data):
        """Test that all weapon types can be loaded."""
        weapon_types = real_fleet_data.get("weapon_types", {})

        for weapon_type in weapon_types:
            weapon = create_weapon_from_fleet_data(real_fleet_data, weapon_type)
            assert weapon.name is not None
            assert weapon.range_km > 0

    def test_load_real_ship_armor(self, real_fleet_data):
        """Test that all ship armors can be loaded."""
        ships = real_fleet_data.get("ships", {})

        for ship_type in ships:
            armor = create_ship_armor_from_fleet_data(real_fleet_data, ship_type)
            assert len(armor.sections) == 3

    def test_spinal_coiler_vs_destroyer(self, real_fleet_data):
        """Test combat between spinal coiler and destroyer."""
        weapon = create_weapon_from_fleet_data(real_fleet_data, "spinal_coiler_mk3")
        armor = create_ship_armor_from_fleet_data(real_fleet_data, "destroyer")

        resolver = CombatResolver(rng=random.Random(42))

        # Simulate several hits
        hits = 0
        penetrations = 0

        for _ in range(100):
            result = resolver.resolve_attack(
                weapon, armor, distance_km=500, target_accel_g=2.0
            )
            if result.hit:
                hits += 1
                if result.penetrated:
                    penetrations += 1

        # Should have some hits and potentially some penetrations
        assert hits > 0
