"""
Comprehensive tests for the thermal management system.

Tests heat sinks, radiator arrays, heat generation, and radiator vulnerability.
Radiators are located at tail section, angled 45° backwards when extended.
"""

import json
import random
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermal import (
    RadiatorState,
    RadiatorPosition,
    DropletRadiator,
    RadiatorArray,
    HeatSink,
    HeatSource,
    ThermalSystem,
    HEAT_GENERATION_RATES,
    RADIATOR_EXTENSION_ANGLE_DEG,
)
from combat import HitLocation


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def fleet_data():
    """Load fleet data for testing."""
    fleet_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    with open(fleet_path) as f:
        return json.load(f)


@pytest.fixture
def destroyer_thermal(fleet_data):
    """Create thermal system for destroyer."""
    return ThermalSystem.from_ship_data(fleet_data["ships"]["destroyer"])


@pytest.fixture
def basic_radiator():
    """Create a basic radiator for testing."""
    return DropletRadiator(
        position=RadiatorPosition.TAIL_PORT,
        max_dissipation_kw=32500.0,
        mass_tons=2.5,
        state=RadiatorState.RETRACTED,
        health_percent=100.0,
    )


# ============================================================================
# DropletRadiator Tests
# ============================================================================

class TestDropletRadiator:
    """Tests for individual radiator behavior."""

    def test_radiator_starts_retracted(self, basic_radiator):
        assert basic_radiator.state == RadiatorState.RETRACTED

    def test_radiator_positions_at_tail(self):
        for pos in RadiatorPosition:
            assert "TAIL" in pos.name

    def test_dissipation_zero_when_retracted(self, basic_radiator):
        assert basic_radiator.current_dissipation_kw == 0.0

    def test_dissipation_max_when_extended(self, basic_radiator):
        basic_radiator.extend()
        assert basic_radiator.current_dissipation_kw == basic_radiator.max_dissipation_kw

    def test_extend_retract_cycle(self, basic_radiator):
        assert basic_radiator.extend() is True
        assert basic_radiator.state == RadiatorState.EXTENDED
        assert basic_radiator.retract() is True
        assert basic_radiator.state == RadiatorState.RETRACTED

    def test_damage_reduces_health(self, basic_radiator):
        basic_radiator.extend()
        basic_radiator.damage(1.0)  # 1 GJ = 20% damage
        assert basic_radiator.health_percent == 80.0
        assert basic_radiator.state == RadiatorState.DAMAGED

    def test_damage_reduces_dissipation(self, basic_radiator):
        basic_radiator.extend()
        original = basic_radiator.current_dissipation_kw
        basic_radiator.damage(2.0)  # 40% damage
        assert basic_radiator.current_dissipation_kw == pytest.approx(original * 0.6)

    def test_destruction_at_zero_health(self, basic_radiator):
        basic_radiator.extend()
        basic_radiator.damage(5.0)  # 100% damage
        assert basic_radiator.state == RadiatorState.DESTROYED
        assert basic_radiator.current_dissipation_kw == 0.0

    def test_destroyed_cannot_extend(self, basic_radiator):
        basic_radiator.state = RadiatorState.DESTROYED
        assert basic_radiator.extend() is False


# ============================================================================
# RadiatorArray Tests
# ============================================================================

class TestRadiatorArray:
    """Tests for radiator array operations."""

    def test_array_has_four_radiators(self, fleet_data):
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        assert len(array.radiators) == 4

    def test_extend_all(self, fleet_data):
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        count = array.extend_all()
        assert count == 4

    def test_total_dissipation(self, fleet_data):
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        array.extend_all()
        # 10 tons * 13 kW/kg * 1000 = 130,000 kW
        assert array.total_dissipation_kw == pytest.approx(130_000, rel=0.01)

    def test_radiator_extension_angle(self):
        """Radiators extend at 45° backwards."""
        assert RADIATOR_EXTENSION_ANGLE_DEG == 45.0


class TestRadiatorHitProbability:
    """Tests for radiator vulnerability based on hit location and angle."""

    def test_tail_hit_exposes_all_radiators(self, fleet_data):
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        positions = array._get_vulnerable_positions(HitLocation.TAIL)
        assert len(positions) == 4

    def test_lateral_hit_exposes_two_radiators(self, fleet_data):
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        positions = array._get_vulnerable_positions(HitLocation.LATERAL)
        assert len(positions) == 2
        assert RadiatorPosition.TAIL_PORT in positions
        assert RadiatorPosition.TAIL_STARBOARD in positions

    def test_nose_hit_no_exposure(self, fleet_data):
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        positions = array._get_vulnerable_positions(HitLocation.NOSE)
        assert len(positions) == 0

    def test_retracted_low_probability(self, fleet_data):
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        prob = array.get_hit_probability(HitLocation.TAIL)
        assert prob == pytest.approx(0.05, abs=0.01)

    def test_extended_tail_hit_probability(self, fleet_data):
        """Tail hits see more of 45° angled radiators."""
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        array.extend_all()
        prob = array.get_hit_probability(HitLocation.TAIL)
        # ~0.19 due to angle exposure
        assert prob > 0.15
        assert prob < 0.25

    def test_extended_lateral_reduced_probability(self, fleet_data):
        """Lateral hits see less due to 45° backward angle."""
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        array.extend_all()
        prob = array.get_hit_probability(HitLocation.LATERAL)
        # ~0.11 due to cos(45°) factor
        assert prob < 0.15
        assert prob > 0.05


# ============================================================================
# HeatSink Tests
# ============================================================================

class TestHeatSink:
    """Tests for heat sink operations."""

    def test_heatsink_from_fleet_data(self, fleet_data):
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        heatsink = HeatSink.from_ship_data(thermal_data)
        assert heatsink.capacity_gj == 525.0

    def test_absorb_heat(self):
        sink = HeatSink(capacity_gj=100.0)
        assert sink.absorb(50.0) is True
        assert sink.current_heat_gj == 50.0

    def test_absorb_overflow(self):
        sink = HeatSink(capacity_gj=100.0, current_heat_gj=90.0)
        assert sink.absorb(20.0) is False
        assert sink.current_heat_gj == 100.0

    def test_dump_to_radiators(self, fleet_data):
        thermal_data = fleet_data["ships"]["destroyer"]["thermal"]
        array = RadiatorArray.from_ship_data(thermal_data)
        array.extend_all()
        sink = HeatSink(capacity_gj=100.0, current_heat_gj=50.0)
        dumped = sink.dump_to_radiators(array, dt_seconds=1.0)
        assert dumped > 0
        assert sink.current_heat_gj < 50.0


# ============================================================================
# ThermalSystem Tests
# ============================================================================

class TestThermalSystem:
    """Tests for complete thermal system."""

    def test_creation_from_ship_data(self, destroyer_thermal):
        assert destroyer_thermal.heatsink is not None
        assert destroyer_thermal.radiators is not None
        assert len(destroyer_thermal.heat_sources) > 0

    def test_reactor_always_active(self, destroyer_thermal):
        reactor = next(s for s in destroyer_thermal.heat_sources if s.name == "reactor")
        assert reactor.active is True

    def test_overheating_threshold(self, destroyer_thermal):
        capacity = destroyer_thermal.heatsink.capacity_gj
        destroyer_thermal.heatsink.current_heat_gj = capacity * 0.85
        assert destroyer_thermal.is_overheating is True

    def test_critical_threshold(self, destroyer_thermal):
        capacity = destroyer_thermal.heatsink.capacity_gj
        destroyer_thermal.heatsink.current_heat_gj = capacity * 0.96
        assert destroyer_thermal.is_critical is True

    def test_update_dissipates_heat(self, destroyer_thermal):
        destroyer_thermal.heatsink.current_heat_gj = 100.0
        destroyer_thermal.radiators.extend_all()
        initial = destroyer_thermal.heatsink.current_heat_gj
        destroyer_thermal.update(dt_seconds=10.0)
        assert destroyer_thermal.heatsink.current_heat_gj < initial


class TestThermalScenarios:
    """Tactical thermal scenarios."""

    def test_extended_radiators_cooling(self, destroyer_thermal):
        destroyer_thermal.heatsink.current_heat_gj = 200.0
        destroyer_thermal.radiators.extend_all()
        initial = destroyer_thermal.heatsink.current_heat_gj
        for _ in range(100):
            destroyer_thermal.update(dt_seconds=1.0)
        # 130 MW radiators dissipate ~13 GJ in 100s
        # Reactor adds ~0.1 GJ, net cooling ~12.9 GJ
        assert destroyer_thermal.heatsink.current_heat_gj < initial
        assert destroyer_thermal.heatsink.current_heat_gj < 190.0

    def test_retracted_radiators_heating(self, destroyer_thermal):
        destroyer_thermal.radiators.retract_all()
        for source in destroyer_thermal.heat_sources:
            source.active = True
        initial = destroyer_thermal.heatsink.current_heat_gj
        for _ in range(60):
            destroyer_thermal.update(dt_seconds=1.0)
        assert destroyer_thermal.heatsink.current_heat_gj > initial

    def test_destroyed_radiators_thermal_crisis(self, destroyer_thermal):
        destroyer_thermal.radiators.extend_all()
        initial_dissipation = destroyer_thermal.radiators.total_dissipation_kw
        
        # Destroy all radiators
        for radiator in destroyer_thermal.radiators.radiators.values():
            radiator.damage(10.0)
        
        assert destroyer_thermal.radiators.total_dissipation_kw == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
