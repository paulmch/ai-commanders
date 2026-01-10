"""
Tests for the power system module.

Tests cover:
- Weapon capacitor charging and discharging
- Battery operations
- Reactor power output
- Power system distribution logic
- Heat generation from weapon firing
"""

import pytest
import math

from src.power import (
    WeaponCapacitor,
    Battery,
    Reactor,
    PowerSystem,
    KINETIC_WEAPON_EFFICIENCY,
    LASER_WEAPON_EFFICIENCY,
    TORPEDO_LAUNCHER_EFFICIENCY,
)


class TestWeaponCapacitor:
    """Tests for WeaponCapacitor class."""

    def test_capacitor_creation(self):
        """Test basic capacitor creation."""
        cap = WeaponCapacitor(
            capacity_mj=100.0,
            charge_rate_mw=10.0,
            weapon_slot="main_gun",
            efficiency=0.7,
            current_charge_mj=50.0
        )
        assert cap.capacity_mj == 100.0
        assert cap.charge_rate_mw == 10.0
        assert cap.weapon_slot == "main_gun"
        assert cap.efficiency == 0.7
        assert cap.current_charge_mj == 50.0

    def test_is_charged(self):
        """Test is_charged property."""
        cap = WeaponCapacitor(
            capacity_mj=100.0,
            charge_rate_mw=10.0,
            weapon_slot="gun",
            current_charge_mj=100.0
        )
        assert cap.is_charged is True

        cap.current_charge_mj = 99.9
        assert cap.is_charged is False

    def test_charge_percent(self):
        """Test charge percentage calculation."""
        cap = WeaponCapacitor(
            capacity_mj=100.0,
            charge_rate_mw=10.0,
            weapon_slot="gun",
            current_charge_mj=50.0
        )
        assert cap.charge_percent == 50.0

    def test_charge_method(self):
        """Test charging capacitor."""
        cap = WeaponCapacitor(
            capacity_mj=100.0,
            charge_rate_mw=10.0,
            weapon_slot="gun",
            current_charge_mj=0.0
        )
        cap.charge(50.0)
        assert cap.current_charge_mj == 50.0

        # Should cap at capacity
        cap.charge(100.0)
        assert cap.current_charge_mj == 100.0

    def test_discharge_method(self):
        """Test discharging capacitor."""
        cap = WeaponCapacitor(
            capacity_mj=100.0,
            charge_rate_mw=10.0,
            weapon_slot="gun",
            current_charge_mj=100.0
        )
        energy = cap.discharge()
        assert energy == 100.0
        assert cap.current_charge_mj == 0.0

    def test_discharge_when_not_charged(self):
        """Test discharge returns 0 when not fully charged."""
        cap = WeaponCapacitor(
            capacity_mj=100.0,
            charge_rate_mw=10.0,
            weapon_slot="gun",
            current_charge_mj=50.0
        )
        energy = cap.discharge()
        assert energy == 0.0
        assert cap.current_charge_mj == 50.0  # Unchanged

    def test_heat_calculation(self):
        """Test waste heat calculation."""
        cap = WeaponCapacitor(
            capacity_mj=100.0,
            charge_rate_mw=10.0,
            weapon_slot="gun",
            efficiency=0.7
        )
        heat = cap.calculate_heat_generated()
        assert heat == pytest.approx(30.0)  # 100 * (1 - 0.7)

    def test_from_weapon_data_kinetic(self):
        """Test capacitor creation from kinetic weapon data."""
        weapon_data = {
            "type": "kinetic",
            "cooldown_s": 10.0,
            "kinetic_energy_gj": 0.5  # 500 MJ kinetic energy
        }
        cap = WeaponCapacitor.from_weapon_data(weapon_data, "coilgun")

        # Capacity = kinetic energy * 2 (for coil efficiency)
        assert cap.capacity_mj == 1000.0  # 0.5 GJ * 1000 * 2
        assert cap.charge_rate_mw == 100.0  # 1000 MJ / 10s
        assert cap.efficiency == KINETIC_WEAPON_EFFICIENCY
        assert cap.weapon_slot == "coilgun"

    def test_from_weapon_data_pd_laser(self):
        """Test capacitor creation from PD laser data."""
        weapon_data = {
            "type": "point_defense",
            "cooldown_s": 5.0,
            "power_draw_mw": 5.0,
            "efficiency": 0.25
        }
        cap = WeaponCapacitor.from_weapon_data(weapon_data, "pd_laser")

        # Capacity = power_draw * cooldown
        assert cap.capacity_mj == 25.0  # 5 MW * 5s
        assert cap.charge_rate_mw == 5.0  # 25 MJ / 5s
        assert cap.efficiency == 0.25
        assert cap.is_charged  # Starts fully charged

    def test_from_weapon_data_torpedo(self):
        """Test capacitor creation from torpedo launcher data."""
        weapon_data = {
            "type": "missile",
            "cooldown_s": 7.0
        }
        cap = WeaponCapacitor.from_weapon_data(weapon_data, "torpedo_launcher")

        assert cap.efficiency == TORPEDO_LAUNCHER_EFFICIENCY


class TestBattery:
    """Tests for Battery class."""

    def test_battery_creation(self):
        """Test basic battery creation."""
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0,
            name="Test Battery"
        )
        assert battery.capacity_gj == 100.0
        assert battery.current_charge_gj == 100.0  # Starts full

    def test_charge_percent(self):
        """Test charge percentage calculation."""
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0
        )
        assert battery.charge_percent == 100.0

        battery.current_charge_gj = 50.0
        assert battery.charge_percent == 50.0

    def test_discharge(self):
        """Test battery discharge."""
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0
        )

        # Discharge 5 GW for 1 second
        power = battery.discharge(5.0, 1.0)
        assert power == 5.0
        assert battery.current_charge_gj == 95.0

    def test_discharge_rate_limited(self):
        """Test battery respects max discharge rate."""
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=5.0,  # Max 5 GW
            max_recharge_rate_gw=1.0
        )

        # Try to draw 10 GW - should be capped at 5 GW
        power = battery.discharge(10.0, 1.0)
        assert power == 5.0
        assert battery.current_charge_gj == 95.0

    def test_discharge_depletes(self):
        """Test battery can be depleted."""
        battery = Battery(
            capacity_gj=10.0,
            max_discharge_rate_gw=100.0,  # High rate
            max_recharge_rate_gw=1.0
        )

        # Draw 20 GW for 1 second - should only get 10 GJ
        power = battery.discharge(20.0, 1.0)
        assert power == 10.0
        assert battery.current_charge_gj == 0.0
        assert battery.is_depleted

    def test_recharge(self):
        """Test battery recharge."""
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0
        )
        battery.current_charge_gj = 50.0

        # Recharge 1 GW for 1 second
        battery.recharge(1.0, 1.0)
        assert battery.current_charge_gj == 51.0

    def test_recharge_rate_limited(self):
        """Test battery respects max recharge rate."""
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0  # Max 1 GW
        )
        battery.current_charge_gj = 50.0

        # Try to recharge at 10 GW - should be capped
        battery.recharge(10.0, 1.0)
        assert battery.current_charge_gj == 51.0

    def test_recharge_caps_at_capacity(self):
        """Test battery doesn't overcharge."""
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=100.0  # High rate
        )
        battery.current_charge_gj = 99.0

        battery.recharge(10.0, 1.0)
        assert battery.current_charge_gj == 100.0


class TestReactor:
    """Tests for Reactor class."""

    def test_reactor_creation(self):
        """Test basic reactor creation."""
        reactor = Reactor(
            max_output_gw=100.0,
            efficiency=0.999,
            name="Test Reactor"
        )
        assert reactor.max_output_gw == 100.0
        assert reactor.efficiency == 0.999
        assert reactor.current_output_fraction == 1.0

    def test_current_output(self):
        """Test current output calculation."""
        reactor = Reactor(max_output_gw=100.0)
        assert reactor.current_output_gw == 100.0

        reactor.current_output_fraction = 0.5
        assert reactor.current_output_gw == 50.0

    def test_waste_heat(self):
        """Test waste heat calculation."""
        reactor = Reactor(max_output_gw=1000.0, efficiency=0.99)
        # Waste heat = 1000 * (1 - 0.99) = 10 GW
        assert reactor.calculate_waste_heat_gw() == pytest.approx(10.0)


class TestPowerSystem:
    """Tests for PowerSystem class."""

    @pytest.fixture
    def basic_power_system(self):
        """Create a basic power system for testing."""
        reactor = Reactor(max_output_gw=100.0, efficiency=0.999)
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0
        )
        return PowerSystem(reactor=reactor, battery=battery)

    def test_power_system_creation(self, basic_power_system):
        """Test power system creation."""
        ps = basic_power_system
        assert ps.reactor.max_output_gw == 100.0
        assert ps.battery.capacity_gj == 100.0
        assert len(ps.weapon_capacitors) == 0

    def test_add_weapon_capacitor(self, basic_power_system):
        """Test adding weapon capacitors."""
        ps = basic_power_system

        weapon_data = {
            "type": "kinetic",
            "cooldown_s": 10.0,
            "kinetic_energy_gj": 0.1
        }
        cap = ps.add_weapon_capacitor("main_gun", weapon_data)

        assert "main_gun" in ps.weapon_capacitors
        assert cap.weapon_slot == "main_gun"

    def test_set_drive_throttle(self, basic_power_system):
        """Test setting drive throttle."""
        ps = basic_power_system

        ps.set_drive_throttle(0.5)
        assert ps.drive_power_fraction == 0.5

        ps.set_drive_throttle(1.5)  # Should clamp to 1.0
        assert ps.drive_power_fraction == 1.0

        ps.set_drive_throttle(-0.5)  # Should clamp to 0.0
        assert ps.drive_power_fraction == 0.0

    def test_get_available_power(self, basic_power_system):
        """Test available power calculation."""
        ps = basic_power_system

        # No drive usage - all power available
        ps.set_drive_throttle(0.0)
        assert ps.get_available_power_gw() == 100.0

        # 50% drive - half available
        ps.set_drive_throttle(0.5)
        assert ps.get_available_power_gw() == 50.0

        # 100% drive - no power for weapons
        ps.set_drive_throttle(1.0)
        assert ps.get_available_power_gw() == 0.0

    def test_can_weapon_fire(self, basic_power_system):
        """Test weapon fire capability check."""
        ps = basic_power_system

        # No capacitor registered - always can fire
        assert ps.can_weapon_fire("unknown_weapon") is True

        # Add a capacitor and check
        weapon_data = {"type": "kinetic", "cooldown_s": 10.0}
        cap = ps.add_weapon_capacitor("gun", weapon_data)
        cap.current_charge_mj = cap.capacity_mj

        assert ps.can_weapon_fire("gun") is True

        cap.current_charge_mj = 0.0
        assert ps.can_weapon_fire("gun") is False

    def test_fire_weapon(self, basic_power_system):
        """Test firing weapon generates heat."""
        ps = basic_power_system

        weapon_data = {
            "type": "kinetic",
            "cooldown_s": 10.0,
            "kinetic_energy_gj": 0.1  # 100 MJ * 2 = 200 MJ capacity
        }
        cap = ps.add_weapon_capacitor("gun", weapon_data)
        cap.current_charge_mj = cap.capacity_mj  # Fully charged

        heat_gj = ps.fire_weapon("gun")

        # Heat = capacity * (1 - efficiency) / 1000 (MJ to GJ)
        expected_heat = cap.capacity_mj * (1 - KINETIC_WEAPON_EFFICIENCY) / 1000
        assert heat_gj == pytest.approx(expected_heat)
        assert cap.current_charge_mj == 0.0  # Discharged

    def test_fire_weapon_not_charged(self, basic_power_system):
        """Test firing uncharged weapon returns 0 heat."""
        ps = basic_power_system

        weapon_data = {"type": "kinetic", "cooldown_s": 10.0}
        cap = ps.add_weapon_capacitor("gun", weapon_data)
        cap.current_charge_mj = 0.0  # Empty

        heat_gj = ps.fire_weapon("gun")
        assert heat_gj == 0.0

    def test_update_charges_capacitors(self, basic_power_system):
        """Test power system update charges capacitors."""
        ps = basic_power_system
        ps.set_drive_throttle(0.0)  # All power available

        weapon_data = {
            "type": "kinetic",
            "cooldown_s": 10.0,
            "kinetic_energy_gj": 0.001  # Small for faster charging
        }
        cap = ps.add_weapon_capacitor("gun", weapon_data)
        cap.current_charge_mj = 0.0

        # Update for 10 seconds - should fully charge
        ps.update(10.0)

        assert cap.is_charged

    def test_update_uses_battery_when_reactor_insufficient(self):
        """Test battery supplements reactor power."""
        # Small reactor that can't keep up
        reactor = Reactor(max_output_gw=0.001)  # 1 MW
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0
        )
        ps = PowerSystem(reactor=reactor, battery=battery)
        ps.set_drive_throttle(0.0)

        # Large capacitor that needs more power
        weapon_data = {
            "type": "kinetic",
            "cooldown_s": 1.0,
            "kinetic_energy_gj": 1.0  # 2000 MJ capacity, 2000 MW needed
        }
        cap = ps.add_weapon_capacitor("gun", weapon_data)
        cap.current_charge_mj = 0.0

        initial_battery = battery.current_charge_gj

        # Update - should draw from battery
        ps.update(1.0)

        assert battery.current_charge_gj < initial_battery

    def test_update_recharges_battery_with_surplus(self, basic_power_system):
        """Test surplus power recharges battery."""
        ps = basic_power_system
        ps.set_drive_throttle(0.0)  # All power available
        ps.battery.current_charge_gj = 50.0  # Half depleted

        # No capacitors to charge - all power goes to battery
        ps.update(1.0)

        # Battery should have more charge (up to recharge rate)
        assert ps.battery.current_charge_gj > 50.0

    def test_get_status(self, basic_power_system):
        """Test power system status report."""
        ps = basic_power_system
        ps.set_drive_throttle(0.5)

        weapon_data = {"type": "kinetic", "cooldown_s": 10.0}
        ps.add_weapon_capacitor("gun", weapon_data)

        status = ps.get_status()

        assert "reactor_output_gw" in status
        assert "available_power_gw" in status
        assert "drive_power_fraction" in status
        assert "battery_percent" in status
        assert "weapon_capacitors" in status
        assert "gun" in status["weapon_capacitors"]


class TestPowerSystemIntegration:
    """Integration tests for power system with realistic values."""

    def test_realistic_destroyer_power(self):
        """Test power system with realistic destroyer values."""
        # Based on Terra Invicta destroyer specs
        reactor = Reactor(max_output_gw=306430.0, efficiency=0.999)
        battery = Battery(
            capacity_gj=160.0,
            max_discharge_rate_gw=0.75,
            max_recharge_rate_gw=0.075
        )
        ps = PowerSystem(reactor=reactor, battery=battery)

        # Add coilgun capacitor
        coilgun_data = {
            "type": "kinetic",
            "cooldown_s": 20.0,
            "kinetic_energy_gj": 100.0  # 100 GJ projectile
        }
        cap = ps.add_weapon_capacitor("coilgun", coilgun_data)

        # At 50% thrust, should have plenty of power
        ps.set_drive_throttle(0.5)
        available = ps.get_available_power_gw()
        assert available > 100000  # Still massive power available

        # Coilgun should charge quickly
        cap.current_charge_mj = 0.0
        ps.update(1.0)
        assert cap.charge_percent > 0

    def test_pd_laser_rapid_fire(self):
        """Test PD laser can fire rapidly with proper cooldown."""
        reactor = Reactor(max_output_gw=1000.0)
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0
        )
        ps = PowerSystem(reactor=reactor, battery=battery)
        ps.set_drive_throttle(0.0)

        pd_data = {
            "type": "point_defense",
            "cooldown_s": 5.0,
            "power_draw_mw": 5.0,
            "efficiency": 0.25
        }
        cap = ps.add_weapon_capacitor("pd_laser", pd_data)

        # Should start fully charged
        assert cap.is_charged

        # Fire
        heat = ps.fire_weapon("pd_laser")
        assert heat > 0
        assert not cap.is_charged

        # Wait 5 seconds - should recharge
        ps.update(5.0)
        assert cap.is_charged

    def test_weapon_heat_accumulation(self):
        """Test that weapon firing accumulates heat correctly."""
        reactor = Reactor(max_output_gw=1000.0)
        battery = Battery(
            capacity_gj=100.0,
            max_discharge_rate_gw=10.0,
            max_recharge_rate_gw=1.0
        )
        ps = PowerSystem(reactor=reactor, battery=battery)
        ps.set_drive_throttle(0.0)

        # Add weapon with known heat output
        weapon_data = {
            "type": "kinetic",
            "cooldown_s": 1.0,
            "kinetic_energy_gj": 0.1  # 200 MJ capacity
        }
        cap = ps.add_weapon_capacitor("gun", weapon_data)

        total_heat = 0.0

        # Fire 5 times
        for _ in range(5):
            cap.current_charge_mj = cap.capacity_mj  # Recharge
            heat = ps.fire_weapon("gun")
            total_heat += heat

        # Should have accumulated heat
        expected_heat_per_shot = cap.capacity_mj * (1 - KINETIC_WEAPON_EFFICIENCY) / 1000
        assert total_heat == pytest.approx(expected_heat_per_shot * 5)
