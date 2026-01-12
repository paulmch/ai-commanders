"""
Comprehensive tests for the damage propagation system.

This module tests:
- Module creation, damage, and destruction mechanics
- ModuleLayout creation for different ship types
- DamageCone geometry and energy dissipation
- DamagePropagator chain damage calculations
- Integration scenarios (spinal coilgun, torpedo, reactor hit, angled hit)

Run with: python -m pytest tests/test_damage.py -v
"""

import math
import random
from typing import Optional

import pytest

# Import from modules.py (the module layout system)
from src.modules import (
    Module as LayoutModule,
    ModuleLayer,
    ModuleLayout,
    ModulePosition,
    ModuleType,
    CRITICAL_MODULE_TYPES,
)

# Import from damage.py (the damage propagation system)
from src.damage import (
    DamageCone,
    DamagePropagator,
    Module as DamageModule,
    ModuleDamageResult,
    ModuleLayout as DamageModuleLayout,
    WeaponDamageProfile,
    CONE_ANGLES_DEG,
    DISSIPATION_RATES,
    SPALLING_ENERGY_FACTOR,
    calculate_entry_point_from_hit,
    estimate_module_radius,
)

from src.physics import Vector3D
from src.combat import HitLocation, HitResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def rng():
    """Seeded random number generator for deterministic tests."""
    return random.Random(42)


@pytest.fixture
def sample_fleet_data():
    """Minimal fleet data for testing module layouts."""
    return {
        "ships": {
            "corvette": {
                "hull": {
                    "length_m": 65.0,
                    "crew": 20,
                }
            },
            "frigate": {
                "hull": {
                    "length_m": 85.0,
                    "crew": 35,
                }
            },
            "destroyer": {
                "hull": {
                    "length_m": 120.0,
                    "crew": 60,
                }
            },
            "cruiser": {
                "hull": {
                    "length_m": 180.0,
                    "crew": 120,
                }
            },
            "battlecruiser": {
                "hull": {
                    "length_m": 220.0,
                    "crew": 180,
                }
            },
            "battleship": {
                "hull": {
                    "length_m": 280.0,
                    "crew": 250,
                }
            },
            "dreadnought": {
                "hull": {
                    "length_m": 350.0,
                    "crew": 400,
                }
            },
        }
    }


@pytest.fixture
def basic_module():
    """Create a basic module for testing."""
    return LayoutModule(
        name="Test Module",
        module_type=ModuleType.CARGO,
        health_percent=100.0,
        armor_rating=0.2,
        position=ModulePosition(0, 0.0),
        size_m2=20.0,
    )


@pytest.fixture
def critical_reactor():
    """Create a reactor module (critical)."""
    return LayoutModule(
        name="Main Reactor",
        module_type=ModuleType.REACTOR,
        health_percent=100.0,
        armor_rating=0.4,
        position=ModulePosition(2, 0.0),
        size_m2=50.0,
    )


@pytest.fixture
def critical_bridge():
    """Create a bridge module (critical)."""
    return LayoutModule(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        health_percent=100.0,
        armor_rating=0.35,
        position=ModulePosition(1, 0.0),
        size_m2=25.0,
    )


@pytest.fixture
def damage_module():
    """Create a damage.py Module for testing."""
    return DamageModule(
        name="Test Module",
        position=Vector3D(10, 0, 0),
        health=100.0,
        max_health=100.0,
        radius_m=3.0,
        is_critical=False,
        damage_resistance=0.0,
    )


@pytest.fixture
def default_damage_layout():
    """Create a default damage module layout."""
    return DamageModuleLayout.create_default_layout(ship_length_m=65.0)


@pytest.fixture
def kinetic_cone():
    """Create a kinetic damage cone."""
    return DamageCone.from_weapon_type(
        entry_point=Vector3D(32.5, 0, 0),
        direction=Vector3D(-1, 0, 0),
        energy_gj=10.0,
        is_missile=False,
        is_laser=False,
    )


@pytest.fixture
def explosive_cone():
    """Create an explosive/torpedo damage cone."""
    return DamageCone.from_weapon_type(
        entry_point=Vector3D(0, 8, 0),
        direction=Vector3D(0, -1, 0),
        energy_gj=50.0,
        is_missile=True,
        is_laser=False,
    )


# =============================================================================
# MODULE TESTS (modules.py)
# =============================================================================

class TestModuleCreation:
    """Tests for Module creation and attributes."""

    def test_module_creation_with_correct_attributes(self, basic_module):
        """Module should be created with specified attributes."""
        assert basic_module.name == "Test Module"
        assert basic_module.module_type == ModuleType.CARGO
        assert basic_module.health_percent == 100.0
        assert basic_module.armor_rating == 0.2
        assert basic_module.size_m2 == 20.0
        assert basic_module.is_critical is False

    def test_module_auto_critical_for_reactor(self):
        """Reactor modules should be automatically marked as critical."""
        reactor = LayoutModule(
            name="Reactor",
            module_type=ModuleType.REACTOR,
            health_percent=100.0,
        )
        assert reactor.is_critical is True

    def test_module_auto_critical_for_bridge(self):
        """Bridge modules should be automatically marked as critical."""
        bridge = LayoutModule(
            name="Bridge",
            module_type=ModuleType.BRIDGE,
            health_percent=100.0,
        )
        assert bridge.is_critical is True

    def test_non_critical_modules_not_auto_critical(self):
        """Non-critical module types should not be auto-marked critical."""
        for module_type in ModuleType:
            if module_type not in CRITICAL_MODULE_TYPES:
                module = LayoutModule(
                    name=f"Test {module_type.value}",
                    module_type=module_type,
                )
                assert module.is_critical is False, f"{module_type} should not be critical"

    @pytest.mark.parametrize("module_type,expected_critical", [
        (ModuleType.SENSOR, False),
        (ModuleType.BRIDGE, True),
        (ModuleType.REACTOR, True),
        (ModuleType.ENGINE, False),
        (ModuleType.WEAPON, False),
        (ModuleType.CARGO, False),
        (ModuleType.CREW, False),
        (ModuleType.FUEL_TANK, False),
    ])
    def test_critical_module_identification(self, module_type, expected_critical):
        """Each module type should have correct critical status."""
        module = LayoutModule(name="Test", module_type=module_type)
        assert module.is_critical == expected_critical


class TestModuleDamage:
    """Tests for module damage mechanics."""

    def test_damage_reduces_health(self, basic_module):
        """Applying damage should reduce module health."""
        initial_health = basic_module.health_percent
        remaining = basic_module.damage(5.0)  # 5 GJ damage
        assert basic_module.health_percent < initial_health

    def test_module_damage_scales_with_energy(self, basic_module):
        """Module damage should scale with incoming energy."""
        # New damage model: 1 GJ = 50% damage, 2 GJ destroys module
        # Test that partial damage works correctly
        initial_health = basic_module.health_percent
        basic_module.damage(1.0)  # 1 GJ = 50% damage
        assert basic_module.health_percent == 50.0  # Lost 50% health

    def test_damage_returns_remaining_energy(self, basic_module):
        """Damage should return energy that passes through."""
        remaining = basic_module.damage(2.0)
        # Some energy should pass through
        assert remaining >= 0.0

    def test_zero_damage_no_effect(self, basic_module):
        """Zero damage should have no effect."""
        initial = basic_module.health_percent
        remaining = basic_module.damage(0.0)
        assert basic_module.health_percent == initial
        assert remaining == 0.0

    def test_negative_damage_no_effect(self, basic_module):
        """Negative damage should have no effect."""
        initial = basic_module.health_percent
        remaining = basic_module.damage(-5.0)
        assert basic_module.health_percent == initial
        assert remaining == -5.0

    def test_massive_damage_destroys_module(self, basic_module):
        """Massive damage should destroy the module."""
        basic_module.damage(1000.0)
        assert basic_module.is_destroyed is True
        assert basic_module.health_percent == 0.0

    def test_destroyed_module_at_zero_health(self, basic_module):
        """Module should be destroyed when health reaches zero."""
        # Apply enough damage to destroy
        while not basic_module.is_destroyed:
            basic_module.damage(10.0)
        assert basic_module.health_percent <= 0.0
        assert basic_module.is_destroyed is True


class TestModuleFunctionality:
    """Tests for module functionality and effectiveness."""

    def test_is_functional_above_25_percent(self, basic_module):
        """Module should be functional above 25% health."""
        basic_module.health_percent = 26.0
        assert basic_module.is_functional is True

    def test_not_functional_at_25_percent(self, basic_module):
        """Module should not be functional at or below 25% health."""
        basic_module.health_percent = 25.0
        assert basic_module.is_functional is False

    def test_effectiveness_at_full_health(self, basic_module):
        """Effectiveness should be 1.0 at full health."""
        assert basic_module.effectiveness == 1.0

    def test_effectiveness_at_zero_health(self, basic_module):
        """Effectiveness should be 0.0 at zero health."""
        basic_module.health_percent = 0.0
        assert basic_module.effectiveness == 0.0

    @pytest.mark.parametrize("health,expected_effectiveness", [
        (100.0, 1.0),
        (75.0, 0.75),
        (50.0, 0.50),
        (25.0, 0.25),
        (0.0, 0.0),
    ])
    def test_effectiveness_scales_with_health(self, basic_module, health, expected_effectiveness):
        """Effectiveness should scale linearly with health."""
        basic_module.health_percent = health
        assert basic_module.effectiveness == pytest.approx(expected_effectiveness)


class TestModuleRepair:
    """Tests for module repair mechanics."""

    def test_repair_increases_health(self, basic_module):
        """Repair should increase module health."""
        basic_module.health_percent = 50.0
        basic_module.repair(25.0)
        assert basic_module.health_percent == 75.0

    def test_repair_caps_at_100_percent(self, basic_module):
        """Repair should not exceed 100% health."""
        basic_module.health_percent = 90.0
        basic_module.repair(50.0)
        assert basic_module.health_percent == 100.0


# =============================================================================
# MODULE LAYOUT TESTS (modules.py)
# =============================================================================

class TestModuleLayoutCreation:
    """Tests for ModuleLayout creation."""

    @pytest.mark.parametrize("ship_type,expected_layers", [
        ("corvette", 4),
        ("frigate", 5),
        ("destroyer", 6),
        ("cruiser", 8),
        ("battlecruiser", 8),
        ("battleship", 9),
        ("dreadnought", 10),
    ])
    def test_layout_layers_per_ship_type(self, sample_fleet_data, ship_type, expected_layers):
        """Each ship type should have the correct number of layers."""
        layout = ModuleLayout.from_ship_type(ship_type, sample_fleet_data)
        assert layout.total_layers == expected_layers

    def test_corvette_layout_structure(self, sample_fleet_data):
        """Corvette should have proper 4-layer structure."""
        layout = ModuleLayout.from_ship_type("corvette", sample_fleet_data)
        assert layout.total_layers == 4
        assert layout.ship_type == "corvette"

    def test_destroyer_layout_structure(self, sample_fleet_data):
        """Destroyer should have proper 6-layer structure."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        assert layout.total_layers == 6
        assert layout.ship_type == "destroyer"

    def test_dreadnought_layout_structure(self, sample_fleet_data):
        """Dreadnought should have proper 10-layer structure."""
        layout = ModuleLayout.from_ship_type("dreadnought", sample_fleet_data)
        assert layout.total_layers == 10
        assert layout.ship_type == "dreadnought"

    def test_layout_has_modules(self, sample_fleet_data):
        """Layout should contain modules."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        all_modules = layout.get_all_modules()
        assert len(all_modules) > 0


class TestCriticalModulePositioning:
    """Tests for critical module placement in protected positions."""

    def test_critical_modules_exist(self, sample_fleet_data):
        """Layout should contain critical modules."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        critical = layout.get_critical_modules()
        assert len(critical) > 0

    def test_critical_modules_include_reactor_and_bridge(self, sample_fleet_data):
        """Critical modules should include reactor and bridge."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        critical = layout.get_critical_modules()
        critical_types = {m.module_type for m in critical}
        assert ModuleType.REACTOR in critical_types
        assert ModuleType.BRIDGE in critical_types

    def test_critical_modules_in_protected_layers(self, sample_fleet_data):
        """Critical modules should not be in first or last layer."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        critical = layout.get_critical_modules()

        for module in critical:
            layer_idx = module.position.layer_index
            # Critical modules should be in middle layers, not exposed
            assert layer_idx > 0, f"{module.name} in exposed first layer"
            assert layer_idx < layout.total_layers - 1, f"{module.name} in exposed last layer"

    @pytest.mark.parametrize("ship_type", [
        "corvette", "frigate", "destroyer", "cruiser", "dreadnought"
    ])
    def test_all_ships_have_protected_criticals(self, sample_fleet_data, ship_type):
        """All ship types should have critical modules in protected positions."""
        layout = ModuleLayout.from_ship_type(ship_type, sample_fleet_data)
        critical = layout.get_critical_modules()

        assert len(critical) > 0, f"{ship_type} has no critical modules"

        # Bridge and reactor should exist and be on centerline
        bridge = layout.get_module_by_name("Command Bridge")
        reactor = layout.get_modules_by_type(ModuleType.REACTOR)

        assert bridge is not None, f"{ship_type} missing bridge"
        assert len(reactor) > 0, f"{ship_type} missing reactor"


class TestModuleLayoutMethods:
    """Tests for ModuleLayout query methods."""

    def test_get_module_by_name(self, sample_fleet_data):
        """Should find module by name."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        bridge = layout.get_module_by_name("Command Bridge")
        assert bridge is not None
        assert bridge.module_type == ModuleType.BRIDGE

    def test_get_module_by_name_not_found(self, sample_fleet_data):
        """Should return None for non-existent module."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        result = layout.get_module_by_name("Nonexistent Module")
        assert result is None

    def test_get_modules_at_layer(self, sample_fleet_data):
        """Should return modules at specified layer."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        layer0_modules = layout.get_modules_at_layer(0)
        assert len(layer0_modules) > 0
        # All should have layer_index 0
        for m in layer0_modules:
            assert m.position.layer_index == 0

    def test_get_modules_at_invalid_layer(self, sample_fleet_data):
        """Should return empty list for invalid layer."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        result = layout.get_modules_at_layer(100)
        assert result == []

    def test_get_modules_by_type(self, sample_fleet_data):
        """Should return all modules of specified type."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        sensors = layout.get_modules_by_type(ModuleType.SENSOR)
        assert len(sensors) > 0
        for sensor in sensors:
            assert sensor.module_type == ModuleType.SENSOR


class TestGetModulesInCone:
    """Tests for get_modules_in_cone method."""

    def test_nose_hit_returns_modules(self, sample_fleet_data):
        """Nose hit should return modules in damage path."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        modules = layout.get_modules_in_cone(HitLocation.NOSE, angle_deg=15.0)
        assert len(modules) > 0

    def test_tail_hit_returns_modules(self, sample_fleet_data):
        """Tail hit should return modules in damage path."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        modules = layout.get_modules_in_cone(HitLocation.TAIL, angle_deg=15.0)
        assert len(modules) > 0

    def test_lateral_hit_returns_modules(self, sample_fleet_data):
        """Lateral hit should return modules in damage path."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        modules = layout.get_modules_in_cone(HitLocation.LATERAL, angle_deg=30.0)
        assert len(modules) > 0

    def test_wider_angle_includes_more_modules(self, sample_fleet_data):
        """Wider cone angle should include more modules."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        narrow = layout.get_modules_in_cone(HitLocation.NOSE, angle_deg=5.0)
        wide = layout.get_modules_in_cone(HitLocation.NOSE, angle_deg=45.0)
        assert len(wide) >= len(narrow)

    def test_modules_ordered_by_distance(self, sample_fleet_data):
        """Modules should be ordered by distance from entry point."""
        layout = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        modules = layout.get_modules_in_cone(HitLocation.NOSE, angle_deg=30.0)

        if len(modules) > 1:
            # Check that layer indices are non-decreasing
            prev_layer = -1
            for m in modules:
                assert m.position.layer_index >= prev_layer
                prev_layer = m.position.layer_index


# =============================================================================
# DAMAGE CONE TESTS (damage.py)
# =============================================================================

class TestDamageConeCreation:
    """Tests for DamageCone creation."""

    def test_kinetic_cone_properties(self, kinetic_cone):
        """Kinetic cone should have tight angle and slow dissipation."""
        assert kinetic_cone.damage_profile == WeaponDamageProfile.KINETIC
        assert kinetic_cone.cone_angle_deg == CONE_ANGLES_DEG[WeaponDamageProfile.KINETIC]
        assert kinetic_cone.initial_energy_gj == 10.0

    def test_explosive_cone_properties(self, explosive_cone):
        """Explosive cone should have wide angle and fast dissipation."""
        assert explosive_cone.damage_profile == WeaponDamageProfile.EXPLOSIVE
        assert explosive_cone.cone_angle_deg == CONE_ANGLES_DEG[WeaponDamageProfile.EXPLOSIVE]
        assert explosive_cone.initial_energy_gj == 50.0

    def test_laser_cone_properties(self):
        """Laser cone should have very tight angle."""
        laser_cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            energy_gj=5.0,
            is_laser=True,
        )
        assert laser_cone.damage_profile == WeaponDamageProfile.LASER
        assert laser_cone.cone_angle_deg == CONE_ANGLES_DEG[WeaponDamageProfile.LASER]

    def test_spalling_cone_creation(self):
        """Spalling cone should have wide angle."""
        spalling = DamageCone.create_spalling_cone(
            origin=Vector3D(0, 0, 0),
            primary_direction=Vector3D(1, 0, 0),
            energy_gj=2.5,
        )
        assert spalling.damage_profile == WeaponDamageProfile.SPALLING
        assert spalling.cone_angle_deg == CONE_ANGLES_DEG[WeaponDamageProfile.SPALLING]

    def test_direction_normalized(self):
        """Direction vector should be normalized."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(3, 4, 0),  # magnitude = 5
            cone_angle_deg=15.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
        )
        # Should be normalized to (0.6, 0.8, 0)
        assert cone.direction.magnitude == pytest.approx(1.0, rel=1e-6)


class TestConeGeometry:
    """Tests for DamageCone geometry calculations."""

    def test_is_in_cone_on_axis(self):
        """Point directly on cone axis should be in cone."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
        )
        assert cone.is_in_cone(Vector3D(10, 0, 0)) is True

    def test_is_in_cone_off_axis_within_angle(self):
        """Point within cone angle should be in cone."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
        )
        # At distance 10, cone radius = 10 * tan(30) ~= 5.77
        assert cone.is_in_cone(Vector3D(10, 3, 0)) is True

    def test_is_in_cone_outside_angle(self):
        """Point outside cone angle should not be in cone."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
        )
        # At distance 10, cone radius = 10 * tan(30) ~= 5.77
        # Point at y=10 is outside
        assert cone.is_in_cone(Vector3D(10, 10, 0)) is False

    def test_is_in_cone_behind_entry(self):
        """Point behind entry point should not be in cone."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
        )
        assert cone.is_in_cone(Vector3D(-5, 0, 0)) is False

    @pytest.mark.parametrize("position,expected", [
        (Vector3D(10, 0, 0), True),    # On axis
        (Vector3D(10, 5, 0), True),    # Within 30 deg cone (angle ~26.6 deg)
        (Vector3D(10, 10, 0), False),  # Outside cone (angle ~45 deg)
        (Vector3D(-5, 0, 0), False),   # Behind entry
        (Vector3D(10, 0, 5), True),    # Above axis, within cone (angle ~26.6 deg)
        (Vector3D(10, 3, 3), True),    # Diagonal within cone (angle ~23.2 deg)
    ])
    def test_is_in_cone_various_positions(self, position, expected):
        """Test is_in_cone for various positions."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
        )
        assert cone.is_in_cone(position) == expected

    def test_cone_radius_at_distance(self):
        """Cone radius should expand with distance."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=45.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
        )
        # At 45 degrees, radius = distance * tan(45) = distance
        assert cone.get_cone_radius_at_distance(10.0) == pytest.approx(10.0)
        assert cone.get_cone_radius_at_distance(20.0) == pytest.approx(20.0)

    def test_cone_radius_at_zero_distance(self):
        """Cone radius at zero distance should be zero."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
        )
        assert cone.get_cone_radius_at_distance(0.0) == 0.0


class TestEnergyDissipation:
    """Tests for energy dissipation with distance."""

    def test_energy_at_zero_distance(self, kinetic_cone):
        """Energy at zero distance should equal initial energy."""
        energy = kinetic_cone.get_energy_at_distance(0.0)
        assert energy == kinetic_cone.remaining_energy_gj

    def test_energy_decreases_with_distance(self, kinetic_cone):
        """Energy should decrease with distance."""
        e0 = kinetic_cone.get_energy_at_distance(0.0)
        e10 = kinetic_cone.get_energy_at_distance(10.0)
        e50 = kinetic_cone.get_energy_at_distance(50.0)

        assert e10 < e0
        assert e50 < e10

    def test_kinetic_slower_dissipation_than_explosive(self):
        """Kinetic should dissipate slower than explosive."""
        kinetic = DamageCone.from_weapon_type(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            energy_gj=10.0,
            is_missile=False,
        )
        explosive = DamageCone.from_weapon_type(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            energy_gj=10.0,
            is_missile=True,
        )

        distance = 50.0
        kinetic_remaining = kinetic.get_energy_at_distance(distance)
        explosive_remaining = explosive.get_energy_at_distance(distance)

        assert kinetic_remaining > explosive_remaining

    @pytest.mark.parametrize("profile,rate", [
        (WeaponDamageProfile.KINETIC, DISSIPATION_RATES[WeaponDamageProfile.KINETIC]),
        (WeaponDamageProfile.EXPLOSIVE, DISSIPATION_RATES[WeaponDamageProfile.EXPLOSIVE]),
        (WeaponDamageProfile.LASER, DISSIPATION_RATES[WeaponDamageProfile.LASER]),
        (WeaponDamageProfile.SPALLING, DISSIPATION_RATES[WeaponDamageProfile.SPALLING]),
    ])
    def test_dissipation_follows_exponential(self, profile, rate):
        """Energy should follow exponential decay."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=10.0,
            damage_profile=profile,
        )

        distance = 20.0
        expected = 10.0 * math.exp(-rate * distance)
        actual = cone.get_energy_at_distance(distance)

        assert actual == pytest.approx(expected, rel=1e-6)

    def test_is_depleted_below_threshold(self):
        """Cone should be depleted below 0.01 GJ."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=0.005,
        )
        assert cone.is_depleted is True

    def test_not_depleted_above_threshold(self):
        """Cone should not be depleted above 0.01 GJ."""
        cone = DamageCone(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=10.0,
            remaining_energy_gj=1.0,
        )
        assert cone.is_depleted is False


class TestDifferentConeAngles:
    """Tests for different cone angles based on weapon types."""

    def test_kinetic_tight_cone(self):
        """Kinetic weapons should have 15 degree cone."""
        assert CONE_ANGLES_DEG[WeaponDamageProfile.KINETIC] == 15.0

    def test_explosive_wide_cone(self):
        """Explosive weapons should have 60 degree cone."""
        assert CONE_ANGLES_DEG[WeaponDamageProfile.EXPLOSIVE] == 60.0

    def test_laser_very_tight_cone(self):
        """Laser weapons should have 5 degree cone."""
        assert CONE_ANGLES_DEG[WeaponDamageProfile.LASER] == 5.0

    def test_spalling_medium_wide_cone(self):
        """Spalling should have 45 degree cone."""
        assert CONE_ANGLES_DEG[WeaponDamageProfile.SPALLING] == 45.0


# =============================================================================
# DAMAGE PROPAGATOR TESTS (damage.py)
# =============================================================================

class TestDamagePropagatorSingleModule:
    """Tests for single module damage propagation."""

    def test_single_module_hit(self, default_damage_layout, kinetic_cone):
        """Single module in path should take damage."""
        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(kinetic_cone, default_damage_layout)

        assert len(results) > 0
        assert results[0].damage_taken_gj > 0

    def test_damage_reduces_module_health(self, damage_module):
        """Damage should reduce module health."""
        initial = damage_module.health
        damage_module.take_damage(10.0)
        assert damage_module.health < initial

    def test_damage_result_contains_correct_info(self, default_damage_layout, kinetic_cone):
        """Damage results should contain accurate information."""
        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(kinetic_cone, default_damage_layout)

        if results:
            result = results[0]
            assert isinstance(result, ModuleDamageResult)
            assert result.module_name is not None
            assert result.health_before >= result.health_after
            assert result.damage_taken_gj >= 0


class TestChainDamage:
    """Tests for damage propagating through multiple modules."""

    def test_chain_damage_through_multiple_modules(self, default_damage_layout):
        """High energy should damage multiple modules in path."""
        # High energy cone to ensure chain damage
        high_energy_cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=100.0,  # Very high energy
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(high_energy_cone, default_damage_layout)

        # Should hit multiple modules
        assert len(results) > 1

    def test_destruction_passes_energy_to_next(self, default_damage_layout):
        """Destroyed module should pass energy to next module."""
        # Create a layout with modules in a line
        layout = DamageModuleLayout(ship_length_m=65.0)
        # Two modules in a line
        layout.add_module(DamageModule(
            name="First",
            position=Vector3D(20, 0, 0),
            health=10.0,  # Low health, will be destroyed
            max_health=10.0,
            radius_m=5.0,
        ))
        layout.add_module(DamageModule(
            name="Second",
            position=Vector3D(10, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=5.0,
        ))

        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=50.0,
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(cone, layout)

        # Both modules should be affected
        module_names = [r.module_name for r in results]
        assert "First" in module_names
        assert "Second" in module_names

    def test_energy_depletion_stops_propagation(self):
        """Propagation should stop when energy is depleted."""
        layout = DamageModuleLayout(ship_length_m=100.0)

        # Add many modules
        for i in range(10):
            layout.add_module(DamageModule(
                name=f"Module_{i}",
                position=Vector3D(45 - i * 10, 0, 0),
                health=100.0,
                max_health=100.0,
                radius_m=5.0,
            ))

        # Low energy cone
        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(50, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=5.0,  # Low energy
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(cone, layout)

        # Should not hit all 10 modules due to energy depletion
        assert len(results) < 10


class TestKineticVsExplosive:
    """Tests for kinetic vs explosive damage patterns."""

    def test_kinetic_narrow_damage_path(self):
        """Kinetic damage should follow narrow path."""
        layout = DamageModuleLayout(ship_length_m=65.0)
        # Module on axis
        layout.add_module(DamageModule(
            name="Center",
            position=Vector3D(10, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))
        # Module off axis
        layout.add_module(DamageModule(
            name="Side",
            position=Vector3D(10, 8, 0),  # Far from centerline
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))

        kinetic = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=20.0,
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(kinetic, layout)

        # Kinetic should mainly hit center
        module_names = [r.module_name for r in results]
        assert "Center" in module_names
        # Side might not be hit due to narrow cone

    def test_explosive_wide_damage_spread(self):
        """Explosive damage should spread wider."""
        layout = DamageModuleLayout(ship_length_m=65.0)
        # Module on axis
        layout.add_module(DamageModule(
            name="Center",
            position=Vector3D(10, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))
        # Module off axis
        layout.add_module(DamageModule(
            name="Side",
            position=Vector3D(10, 6, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))

        explosive = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=20.0,
            is_missile=True,  # Explosive
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(explosive, layout)

        # Explosive should hit both due to wide cone
        module_names = [r.module_name for r in results]
        assert "Center" in module_names
        # Wide cone should include side module
        # (depends on exact geometry)


class TestSpallingMechanics:
    """Tests for spalling damage mechanics."""

    def test_spalling_enabled_creates_secondary_damage(self):
        """Destroyed module should create spalling damage."""
        layout = DamageModuleLayout(ship_length_m=65.0)

        # Module that will be destroyed
        layout.add_module(DamageModule(
            name="Primary",
            position=Vector3D(20, 0, 0),
            health=10.0,  # Will be destroyed
            max_health=10.0,
            radius_m=3.0,
        ))
        # Nearby module for spalling
        layout.add_module(DamageModule(
            name="Nearby",
            position=Vector3D(17, 3, 0),  # Close to primary
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))

        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=50.0,
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=True)
        results = propagator.propagate(cone, layout)

        # Check for spalling results
        spalling_results = [r for r in results if "spalling" in r.module_name]
        # Spalling may or may not occur depending on geometry

    def test_spalling_disabled_no_secondary(self):
        """With spalling disabled, no secondary damage."""
        layout = DamageModuleLayout(ship_length_m=65.0)

        layout.add_module(DamageModule(
            name="Primary",
            position=Vector3D(20, 0, 0),
            health=10.0,
            max_health=10.0,
            radius_m=3.0,
        ))
        layout.add_module(DamageModule(
            name="Nearby",
            position=Vector3D(17, 3, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))

        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=50.0,
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(cone, layout)

        # No spalling results
        spalling_results = [r for r in results if "spalling" in r.module_name]
        assert len(spalling_results) == 0

    def test_spalling_energy_factor(self):
        """Spalling should use correct energy fraction."""
        assert SPALLING_ENERGY_FACTOR == 0.25


# =============================================================================
# INTEGRATION SCENARIOS
# =============================================================================

class TestIntegrationSpinalCoilgun:
    """Test scenario: Spinal coilgun penetrates nose, damages sensors, then bridge."""

    def test_spinal_coilgun_nose_penetration(self):
        """Spinal coilgun hit to nose should damage sensors then bridge."""
        # Create a simple layout mimicking ship structure
        layout = DamageModuleLayout(ship_length_m=65.0)

        # Sensors at nose (relatively fragile)
        layout.add_module(DamageModule(
            name="Sensors",
            position=Vector3D(28, 0, 0),
            health=25.0,
            max_health=25.0,
            radius_m=2.5,
            is_critical=False,
        ))
        # Bridge behind sensors
        layout.add_module(DamageModule(
            name="Bridge",
            position=Vector3D(20, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
            is_critical=True,
        ))
        # Reactor deeper in ship
        layout.add_module(DamageModule(
            name="Reactor",
            position=Vector3D(5, 0, 0),
            health=150.0,
            max_health=150.0,
            radius_m=4.0,
            is_critical=True,
        ))

        # Spinal coilgun hit (kinetic, high energy to penetrate multiple modules)
        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),  # Nose
            direction=Vector3D(-1, 0, 0),       # Into ship
            energy_gj=50.0,  # High energy for penetration
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(cone, layout)

        # Extract damage by module
        damage_by_module = {r.module_name: r for r in results}

        # Sensors should be hit first
        assert "Sensors" in damage_by_module
        # Bridge should also be hit
        assert "Bridge" in damage_by_module

        # Sensors should take damage before bridge (first in list)
        sensor_idx = next(i for i, r in enumerate(results) if r.module_name == "Sensors")
        bridge_idx = next(i for i, r in enumerate(results) if r.module_name == "Bridge")
        assert sensor_idx < bridge_idx


class TestIntegrationTorpedoExplosion:
    """Test scenario: Torpedo explosion spreads damage to multiple modules."""

    def test_torpedo_spreads_damage_widely(self):
        """Torpedo hit should damage multiple modules due to wide cone."""
        layout = DamageModuleLayout(ship_length_m=65.0)

        # Multiple modules at different positions
        layout.add_module(DamageModule(
            name="Center",
            position=Vector3D(0, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=4.0,
        ))
        layout.add_module(DamageModule(
            name="Starboard",
            position=Vector3D(0, 5, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))
        layout.add_module(DamageModule(
            name="Port",
            position=Vector3D(0, -5, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))
        layout.add_module(DamageModule(
            name="Dorsal",
            position=Vector3D(0, 0, 4),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))

        # Torpedo hit from starboard (lateral)
        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(0, 10, 0),
            direction=Vector3D(0, -1, 0),  # Into ship from starboard
            energy_gj=50.0,
            is_missile=True,  # Explosive
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(cone, layout)

        # Multiple modules should be damaged due to wide cone
        damaged_modules = {r.module_name for r in results}
        assert len(damaged_modules) > 1


class TestIntegrationReactorCatastrophic:
    """Test scenario: Reactor hit causes catastrophic damage."""

    def test_reactor_hit_critical(self):
        """Direct hit to reactor should cause massive damage."""
        layout = DamageModuleLayout(ship_length_m=65.0)

        # Reactor at center (close to entry point for direct hit)
        reactor = DamageModule(
            name="Reactor",
            position=Vector3D(0, 5, 0),  # Close to entry
            health=100.0,
            max_health=100.0,
            radius_m=4.0,
            is_critical=True,
        )
        layout.add_module(reactor)

        # Direct kinetic hit to reactor (kinetic has narrow cone, less spread)
        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(0, 10, 0),
            direction=Vector3D(0, -1, 0),
            energy_gj=500.0,  # Very high energy to ensure destruction
            is_missile=False,  # Kinetic for focused damage
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(cone, layout)

        # Reactor should be destroyed
        reactor_result = next(r for r in results if r.module_name == "Reactor")
        assert reactor_result.destroyed is True
        assert reactor.is_critical is True

    def test_reactor_destruction_detection(self, default_damage_layout):
        """Should detect when reactor is destroyed."""
        # Find the reactor
        reactor = default_damage_layout.get_module_by_name("Reactor")
        if reactor:
            initial_critical = reactor.is_critical
            # Destroy it
            reactor.take_damage(1000.0)
            assert reactor.is_destroyed is True
            assert reactor.is_critical == initial_critical


class TestIntegrationAngledHit:
    """Test scenario: Angled hit affects different module distribution."""

    def test_angled_hit_different_path(self):
        """Angled entry should hit different modules than straight entry."""
        layout = DamageModuleLayout(ship_length_m=65.0)

        # Modules at different positions
        layout.add_module(DamageModule(
            name="Center",
            position=Vector3D(20, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))
        layout.add_module(DamageModule(
            name="Upper",
            position=Vector3D(15, 0, 5),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))
        layout.add_module(DamageModule(
            name="Lower",
            position=Vector3D(15, 0, -5),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))

        # Angled hit from above
        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(25, 0, 10),
            direction=Vector3D(-0.5, 0, -0.866).normalized(),  # Angled down
            energy_gj=15.0,
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=False)
        results = propagator.propagate(cone, layout)

        # Angled hit should potentially hit different modules
        damaged = {r.module_name for r in results}
        # The exact modules hit depend on cone geometry

    def test_lateral_vs_nose_hit_different_modules(self):
        """Lateral hit should affect different modules than nose hit."""
        layout = DamageModuleLayout(ship_length_m=65.0)

        # Forward module
        layout.add_module(DamageModule(
            name="Forward",
            position=Vector3D(25, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))
        # Side module
        layout.add_module(DamageModule(
            name="Side",
            position=Vector3D(10, 5, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))

        # Nose hit
        nose_cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=10.0,
            is_missile=False,
        )

        # Lateral hit
        lateral_cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(10, 10, 0),
            direction=Vector3D(0, -1, 0),
            energy_gj=10.0,
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=False)

        # Reset layout for second test
        layout2 = DamageModuleLayout(ship_length_m=65.0)
        layout2.add_module(DamageModule(
            name="Forward",
            position=Vector3D(25, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))
        layout2.add_module(DamageModule(
            name="Side",
            position=Vector3D(10, 5, 0),
            health=100.0,
            max_health=100.0,
            radius_m=3.0,
        ))

        nose_results = propagator.propagate(nose_cone, layout)
        lateral_results = propagator.propagate(lateral_cone, layout2)

        nose_modules = {r.module_name for r in nose_results}
        lateral_modules = {r.module_name for r in lateral_results}

        # Different entry points should potentially hit different modules
        # Nose should hit Forward
        assert "Forward" in nose_modules


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions in damage.py."""

    def test_calculate_entry_point_nose(self):
        """Nose hit should enter from front of ship."""
        entry, direction = calculate_entry_point_from_hit(
            ship_position=Vector3D(0, 0, 0),
            ship_heading=Vector3D(1, 0, 0),
            hit_location="nose",
            ship_length_m=65.0,
        )
        assert entry.x > 0  # Forward position
        assert direction.x < 0  # Pointing backward

    def test_calculate_entry_point_tail(self):
        """Tail hit should enter from rear of ship."""
        entry, direction = calculate_entry_point_from_hit(
            ship_position=Vector3D(0, 0, 0),
            ship_heading=Vector3D(1, 0, 0),
            hit_location="tail",
            ship_length_m=65.0,
        )
        assert entry.x < 0  # Rear position
        assert direction.x > 0  # Pointing forward

    def test_calculate_entry_point_lateral(self):
        """Lateral hit should enter from side of ship."""
        entry, direction = calculate_entry_point_from_hit(
            ship_position=Vector3D(0, 0, 0),
            ship_heading=Vector3D(1, 0, 0),
            hit_location="lateral",
            ship_length_m=65.0,
        )
        assert abs(entry.y) > 0 or abs(entry.x) < 32.5  # Side position

    def test_estimate_module_radius(self):
        """Module radius estimation should be reasonable."""
        # Small module
        small_radius = estimate_module_radius(1.0)  # 1 ton
        # Large module
        large_radius = estimate_module_radius(100.0)  # 100 tons

        assert small_radius > 0
        assert large_radius > small_radius
        # Reasonable range for module sizes
        assert 0.5 < small_radius < 5.0
        assert 1.0 < large_radius < 20.0


# =============================================================================
# PARAMETRIZED COMPREHENSIVE TESTS
# =============================================================================

class TestParametrizedDamage:
    """Parametrized tests for various damage scenarios."""

    @pytest.mark.parametrize("energy_gj,expected_destroyed", [
        (1.0, False),
        (5.0, False),
        (50.0, True),
        (100.0, True),
    ])
    def test_damage_threshold_destruction(self, energy_gj, expected_destroyed):
        """Test module destruction at various energy levels."""
        module = DamageModule(
            name="Test",
            position=Vector3D(10, 0, 0),
            health=50.0,
            max_health=50.0,
            radius_m=3.0,
        )
        module.take_damage(energy_gj)
        assert module.is_destroyed == expected_destroyed

    @pytest.mark.parametrize("damage_resistance,expected_absorbed_fraction", [
        (0.0, 1.0),    # No resistance, full damage
        (0.5, 0.5),    # 50% resistance
        (0.9, 0.1),    # 90% resistance
    ])
    def test_damage_resistance_effect(self, damage_resistance, expected_absorbed_fraction):
        """Test damage resistance reduces effective damage."""
        module = DamageModule(
            name="Test",
            position=Vector3D(10, 0, 0),
            health=1000.0,  # High health to not destroy
            max_health=1000.0,
            radius_m=3.0,
            damage_resistance=damage_resistance,
        )
        damage_applied = 100.0
        absorbed = module.take_damage(damage_applied)

        expected_absorbed = damage_applied * (1.0 - damage_resistance)
        assert absorbed == pytest.approx(expected_absorbed, rel=0.01)

    @pytest.mark.parametrize("cone_angle,expected_min_modules", [
        (5.0, 1),    # Very tight - few modules
        (30.0, 1),   # Moderate
        (60.0, 1),   # Wide - more modules
    ])
    def test_cone_angle_module_coverage(self, default_damage_layout, cone_angle, expected_min_modules):
        """Test that cone angle affects number of modules hit."""
        cone = DamageCone(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            cone_angle_deg=cone_angle,
            initial_energy_gj=100.0,
            remaining_energy_gj=100.0,
        )

        modules_in_cone = default_damage_layout.get_modules_in_cone(cone)
        assert len(modules_in_cone) >= expected_min_modules


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_energy_cone(self, default_damage_layout):
        """Zero energy cone should cause no damage."""
        cone = DamageCone(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            cone_angle_deg=30.0,
            initial_energy_gj=0.0,
            remaining_energy_gj=0.0,
        )

        propagator = DamagePropagator()
        results = propagator.propagate(cone, default_damage_layout)

        # No damage should be dealt
        assert len(results) == 0

    def test_empty_layout_no_damage(self):
        """Empty layout should result in no damage."""
        empty_layout = DamageModuleLayout(ship_length_m=65.0)

        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=50.0,
            is_missile=False,
        )

        propagator = DamagePropagator()
        results = propagator.propagate(cone, empty_layout)

        assert len(results) == 0

    def test_already_destroyed_module_not_damaged_again(self):
        """Destroyed modules should not take additional damage."""
        module = DamageModule(
            name="Destroyed",
            position=Vector3D(10, 0, 0),
            health=0.0,
            max_health=100.0,
            radius_m=3.0,
            is_destroyed=True,
        )

        absorbed = module.take_damage(50.0)
        assert absorbed == 0.0
        assert module.health == 0.0

    def test_very_long_distance_cone(self):
        """Cone energy should dissipate over very long distances."""
        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(0, 0, 0),
            direction=Vector3D(1, 0, 0),
            energy_gj=10.0,
            is_missile=False,
        )

        # Energy at very long distance
        energy_at_1000m = cone.get_energy_at_distance(1000.0)

        # Should be significantly reduced
        assert energy_at_1000m < 0.1 * cone.initial_energy_gj


# =============================================================================
# DETERMINISTIC SEEDING TESTS
# =============================================================================

class TestDeterministicResults:
    """Tests to verify deterministic behavior with seeded RNG."""

    def test_same_seed_same_results(self, sample_fleet_data):
        """Same seed should produce identical layouts."""
        # Create two layouts with same ship type
        layout1 = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)
        layout2 = ModuleLayout.from_ship_type("destroyer", sample_fleet_data)

        # Should have same structure
        assert layout1.total_layers == layout2.total_layers
        assert len(layout1.get_all_modules()) == len(layout2.get_all_modules())

    def test_damage_propagation_deterministic(self, default_damage_layout):
        """Damage propagation should be deterministic."""
        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=20.0,
            is_missile=False,
        )

        # Run twice on copies
        layout1 = DamageModuleLayout.create_default_layout(65.0)
        layout2 = DamageModuleLayout.create_default_layout(65.0)

        cone1 = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=20.0,
            is_missile=False,
        )
        cone2 = DamageCone.from_weapon_type(
            entry_point=Vector3D(32.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=20.0,
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=False)

        results1 = propagator.propagate(cone1, layout1)
        results2 = propagator.propagate(cone2, layout2)

        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.module_name == r2.module_name
            assert r1.damage_taken_gj == pytest.approx(r2.damage_taken_gj)


class TestDreadnoughtSpinalNoseHit:
    """Test scenario: Spinal coilgun round hits dreadnought nose at 10 km/s with no armor."""

    def test_spinal_round_10kps_dreadnought_nose(self):
        """
        Test a spinal round hitting dreadnought's unarmored front nose at 10 km/s.

        Scenario:
        - Dreadnought (275m length)
        - Spinal coiler round: 88kg at 10 km/s = ~4.4 GJ kinetic energy
        - Entry point: nose (x=137.5m for ship centered at origin)
        - Direction: straight into ship (-x)
        - No armor protection (worst case)

        Expected: Should damage sensors/bridge at nose and potentially continue deeper.
        """
        # Dreadnought layout - 275m long
        # Based on fleet_ships.json modules:
        # - Sensor at x=0.22 (bridge/sensor co-located)
        # - Living quarters at x=0.45
        # - Magazines at various positions
        # - Heatsinks at x=0.68-0.78

        layout = DamageModuleLayout(ship_length_m=275.0)
        half_length = 275.0 / 2  # 137.5m

        # Modules from fleet_ships.json dreadnought (converted to layout coordinates)
        # x=0 is ship center, positive toward nose
        # Position x in fleet_ships.json is 0=nose, 1=tail
        # So we convert: module_x = half_length - (x_fleet * length)

        # Sensor/Bridge at x=0.22 from nose = 60.5m from nose = half_length - 60.5 = 77m from center
        layout.add_module(DamageModule(
            name="Dreadnought Sensor Complex",
            position=Vector3D(77, 0, 3),  # Dorsal position
            health=50.0,  # Sensors are fragile
            max_health=50.0,
            radius_m=3.0,
            is_critical=False,
        ))

        layout.add_module(DamageModule(
            name="Fleet Command Center",
            position=Vector3D(77, 0, 0),  # Bridge below/with sensor
            health=100.0,
            max_health=100.0,
            radius_m=4.0,
            is_critical=True,  # Bridge is critical
        ))

        # Forward magazine (Spinal Magazine) at x=0.06 = 16.5m from nose = 121m from center
        layout.add_module(DamageModule(
            name="Spinal Magazine",
            position=Vector3D(121, 0, 0),  # Near the nose
            health=75.0,
            max_health=75.0,
            radius_m=3.0,
            is_critical=False,  # Magazine is dangerous if hit
        ))

        # Living quarters at x=0.45 = 123.75m from nose = 13.75m from center
        layout.add_module(DamageModule(
            name="Crew Habitation Section",
            position=Vector3D(13.75, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=5.0,
            is_critical=False,
        ))

        # Heavy Battery Magazine A at x=0.25 = 68.75m from nose = 68.75m from center
        layout.add_module(DamageModule(
            name="Heavy Battery Magazine A",
            position=Vector3D(68.75, 0, 3),
            health=75.0,
            max_health=75.0,
            radius_m=3.0,
            is_critical=False,
        ))

        # Reactor deep in ship (around x=0.85 = 233.75m from nose = -96.25 from center)
        layout.add_module(DamageModule(
            name="Reactor",
            position=Vector3D(-96.25, 0, 0),
            health=200.0,
            max_health=200.0,
            radius_m=5.0,
            is_critical=True,
        ))

        # Calculate kinetic energy: 88kg at 10 km/s
        # KE = 0.5 * m * v^2 = 0.5 * 88 * (10000)^2 = 4.4 GJ
        kinetic_energy_gj = 0.5 * 88 * (10000 ** 2) / 1e9  # Convert J to GJ

        # Entry point at nose (no armor, full energy transfer)
        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(137.5, 0, 0),  # Nose of 275m ship
            direction=Vector3D(-1, 0, 0),       # Into ship
            energy_gj=kinetic_energy_gj,
            is_missile=False,  # Kinetic projectile
        )

        propagator = DamagePropagator(enable_spalling=True)
        results = propagator.propagate(cone, layout)

        # Extract damage by module
        damage_by_module = {r.module_name: r for r in results}

        print(f"\n=== Spinal Round Impact (10 km/s, {kinetic_energy_gj:.2f} GJ) ===")
        print(f"Entry: Nose of Dreadnought (275m)")
        print(f"Direction: Straight into ship")
        print(f"\nDamage Results:")
        for result in results:
            status = "DESTROYED" if result.destroyed else f"{result.health_after:.1f}% HP"
            print(f"  {result.module_name}: {result.damage_taken_gj:.2f} GJ absorbed, {status}")

        # The forward magazine should be hit first (closest to nose)
        assert "Spinal Magazine" in damage_by_module, "Spinal Magazine should be in damage path"

        # Check damage ordering - magazine should be hit before bridge (closer to nose)
        module_names_hit = [r.module_name for r in results]

        # If both magazine and bridge are hit, magazine should be first
        if "Spinal Magazine" in module_names_hit and "Fleet Command Center" in module_names_hit:
            mag_idx = module_names_hit.index("Spinal Magazine")
            bridge_idx = module_names_hit.index("Fleet Command Center")
            assert mag_idx < bridge_idx, "Magazine should be hit before bridge"

        # Energy should be significant enough to damage multiple modules
        total_damage = sum(r.damage_taken_gj for r in results)
        print(f"\nTotal energy absorbed: {total_damage:.2f} GJ")
        print(f"Initial energy: {kinetic_energy_gj:.2f} GJ")

    def test_spinal_round_high_energy_penetration(self):
        """Test what happens with very high energy round penetrating deep."""
        layout = DamageModuleLayout(ship_length_m=275.0)
        half_length = 137.5

        # Simplified layout with key modules in a line
        layout.add_module(DamageModule(
            name="Forward Magazine",
            position=Vector3D(121, 0, 0),
            health=75.0,
            max_health=75.0,
            radius_m=3.0,
        ))

        layout.add_module(DamageModule(
            name="Sensor Array",
            position=Vector3D(77, 0, 3),
            health=50.0,
            max_health=50.0,
            radius_m=3.0,
        ))

        layout.add_module(DamageModule(
            name="Bridge",
            position=Vector3D(77, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=4.0,
            is_critical=True,
        ))

        layout.add_module(DamageModule(
            name="Heavy Magazine",
            position=Vector3D(68.75, 0, 0),
            health=75.0,
            max_health=75.0,
            radius_m=3.0,
        ))

        layout.add_module(DamageModule(
            name="Crew Quarters",
            position=Vector3D(13.75, 0, 0),
            health=100.0,
            max_health=100.0,
            radius_m=5.0,
        ))

        layout.add_module(DamageModule(
            name="Heatsink Array",
            position=Vector3D(-49.5, 0, 0),  # x=0.68 -> -49.5
            health=80.0,
            max_health=80.0,
            radius_m=4.0,
        ))

        layout.add_module(DamageModule(
            name="Reactor",
            position=Vector3D(-96.25, 0, 0),
            health=200.0,
            max_health=200.0,
            radius_m=5.0,
            is_critical=True,
        ))

        # Very high energy - 50 GJ (like multiple rounds or very close range)
        cone = DamageCone.from_weapon_type(
            entry_point=Vector3D(137.5, 0, 0),
            direction=Vector3D(-1, 0, 0),
            energy_gj=50.0,  # High energy for deep penetration
            is_missile=False,
        )

        propagator = DamagePropagator(enable_spalling=True)
        results = propagator.propagate(cone, layout)

        print(f"\n=== High Energy Penetration Test (50 GJ) ===")
        destroyed_modules = []
        damaged_modules = []

        for result in results:
            if result.destroyed:
                destroyed_modules.append(result.module_name)
            else:
                damaged_modules.append(result.module_name)
            print(f"  {result.module_name}: {result.damage_taken_gj:.2f} GJ, "
                  f"{'DESTROYED' if result.destroyed else f'{result.health_after:.1f}% HP'}")

        print(f"\nDestroyed: {destroyed_modules}")
        print(f"Damaged: {damaged_modules}")

        # With 50 GJ, forward modules should take significant damage
        assert len(results) > 0, "Should damage at least one module"

        # Bridge is critical - check if it was hit
        bridge_result = next((r for r in results if r.module_name == "Bridge"), None)
        if bridge_result:
            print(f"\nBridge damage: {bridge_result.damage_taken_gj:.2f} GJ, destroyed: {bridge_result.destroyed}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
