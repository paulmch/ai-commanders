"""
Power System for the AI Commanders space battle simulator.

This module implements spacecraft power management including:
- Weapon capacitors that charge during cooldown
- Reactor power distribution (drives vs weapons)
- Battery backup when reactor power is insufficient
- Heat generation from weapon firing

Power Flow Architecture:
1. Reactor generates power (GW)
2. Drives consume power proportional to throttle
3. Remaining reactor power charges weapon capacitors
4. If insufficient, batteries supplement
5. Weapons can only fire when capacitor is full
6. Firing generates heat sent to thermal system

Based on Terra Invicta power mechanics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .thermal import ThermalSystem


# =============================================================================
# CONSTANTS
# =============================================================================

# Efficiency factors for heat generation
KINETIC_WEAPON_EFFICIENCY = 0.70  # 70% efficient, 30% waste heat
LASER_WEAPON_EFFICIENCY = 0.25    # Terra Invicta PD laser efficiency
TORPEDO_LAUNCHER_EFFICIENCY = 0.90  # Mostly mechanical, low heat

# Minimum power fraction for drives (can't drop below this)
MIN_DRIVE_POWER_FRACTION = 0.1

# Default weapon energy estimates (MJ per shot) if not specified
DEFAULT_KINETIC_ENERGY_MJ = 100.0
DEFAULT_LASER_ENERGY_MJ = 50.0
DEFAULT_TORPEDO_LAUNCH_ENERGY_MJ = 10.0


# =============================================================================
# WEAPON CAPACITOR
# =============================================================================

@dataclass
class WeaponCapacitor:
    """
    Energy storage for a single weapon system.

    Each weapon has its own capacitor that:
    - Stores energy needed for one shot
    - Charges during the weapon's cooldown period
    - Must be fully charged before the weapon can fire

    Attributes:
        capacity_mj: Energy required per shot (megajoules).
        current_charge_mj: Current stored energy (megajoules).
        charge_rate_mw: Power draw during charging (megawatts).
        weapon_slot: Identifier for the weapon this capacitor serves.
        efficiency: Energy conversion efficiency (waste heat = 1 - efficiency).
    """
    capacity_mj: float
    charge_rate_mw: float
    weapon_slot: str
    efficiency: float = KINETIC_WEAPON_EFFICIENCY
    current_charge_mj: float = 0.0

    @classmethod
    def from_weapon_data(
        cls,
        weapon_data: dict,
        weapon_slot: str
    ) -> WeaponCapacitor:
        """
        Create a capacitor from weapon specification.

        Calculates capacity from weapon energy and charge rate from cooldown.

        Args:
            weapon_data: Weapon specification dictionary.
            weapon_slot: Identifier for this weapon slot.

        Returns:
            Configured WeaponCapacitor instance.
        """
        weapon_type = weapon_data.get("type", "kinetic")
        cooldown_s = weapon_data.get("cooldown_s", 10.0)

        # Determine energy per shot based on weapon type
        if weapon_type == "point_defense":
            # Laser weapons use shot power
            shot_power_mj = weapon_data.get("power_draw_mw", 5.0) * cooldown_s
            capacity_mj = shot_power_mj
            efficiency = weapon_data.get("efficiency", LASER_WEAPON_EFFICIENCY)
        elif weapon_type == "missile":
            # Torpedo launchers have low energy requirement
            capacity_mj = DEFAULT_TORPEDO_LAUNCH_ENERGY_MJ
            efficiency = TORPEDO_LAUNCHER_EFFICIENCY
        else:
            # Kinetic weapons - use kinetic energy as proxy
            kinetic_energy_gj = weapon_data.get("kinetic_energy_gj", 0.0)
            if kinetic_energy_gj > 0:
                # Capacitor needs to store energy to accelerate projectile
                # Assume capacitor stores 2x kinetic energy (50% efficiency in coils)
                capacity_mj = kinetic_energy_gj * 1000 * 2  # GJ to MJ, 2x for efficiency
            else:
                capacity_mj = DEFAULT_KINETIC_ENERGY_MJ
            efficiency = KINETIC_WEAPON_EFFICIENCY

        # Charge rate = energy / cooldown time
        charge_rate_mw = capacity_mj / cooldown_s if cooldown_s > 0 else capacity_mj

        return cls(
            capacity_mj=capacity_mj,
            charge_rate_mw=charge_rate_mw,
            weapon_slot=weapon_slot,
            efficiency=efficiency,
            current_charge_mj=capacity_mj  # Start fully charged
        )

    @property
    def is_charged(self) -> bool:
        """Check if capacitor has enough energy to fire."""
        return self.current_charge_mj >= self.capacity_mj

    @property
    def charge_percent(self) -> float:
        """Get current charge as percentage."""
        if self.capacity_mj <= 0:
            return 100.0
        return min(100.0, (self.current_charge_mj / self.capacity_mj) * 100.0)

    def charge(self, energy_mj: float) -> None:
        """
        Add energy to the capacitor.

        Args:
            energy_mj: Energy to add (megajoules).
        """
        self.current_charge_mj = min(
            self.capacity_mj,
            self.current_charge_mj + energy_mj
        )

    def discharge(self) -> float:
        """
        Discharge capacitor for weapon firing.

        Returns:
            Energy discharged (megajoules), or 0 if not fully charged.
        """
        if not self.is_charged:
            return 0.0

        energy = self.current_charge_mj
        self.current_charge_mj = 0.0
        return energy

    def calculate_heat_generated(self) -> float:
        """
        Calculate waste heat generated when firing.

        Returns:
            Heat generated in megajoules.
        """
        # Heat = energy * (1 - efficiency)
        return self.capacity_mj * (1.0 - self.efficiency)


# =============================================================================
# BATTERY SYSTEM
# =============================================================================

@dataclass
class Battery:
    """
    Ship battery for backup power storage.

    Batteries provide power when reactor output is insufficient
    for all systems. They recharge when reactor has surplus power.

    Attributes:
        capacity_gj: Maximum energy storage (gigajoules).
        current_charge_gj: Current stored energy (gigajoules).
        max_discharge_rate_gw: Maximum power output (gigawatts).
        max_recharge_rate_gw: Maximum recharge rate (gigawatts).
        name: Battery type name.
    """
    capacity_gj: float
    max_discharge_rate_gw: float
    max_recharge_rate_gw: float
    name: str = "Ship Battery"
    current_charge_gj: float = field(default=0.0)

    def __post_init__(self):
        """Initialize battery to full charge if not specified."""
        if self.current_charge_gj == 0.0:
            self.current_charge_gj = self.capacity_gj

    @classmethod
    def from_ship_data(cls, battery_data: dict) -> Battery:
        """
        Create a Battery from ship configuration.

        Args:
            battery_data: Battery specification dictionary.

        Returns:
            Configured Battery instance.
        """
        capacity_gj = battery_data.get("capacity_gj", 160.0)
        recharge_rate = battery_data.get("recharge_rate_gj_s", 0.075)

        # Discharge rate is typically higher than recharge
        # Assume 10x recharge rate for discharge
        discharge_rate = recharge_rate * 10

        return cls(
            capacity_gj=capacity_gj,
            max_discharge_rate_gw=discharge_rate,
            max_recharge_rate_gw=recharge_rate,
            name=battery_data.get("name", "Ship Battery"),
            current_charge_gj=capacity_gj  # Start fully charged
        )

    @property
    def charge_percent(self) -> float:
        """Get current charge as percentage."""
        if self.capacity_gj <= 0:
            return 0.0
        return (self.current_charge_gj / self.capacity_gj) * 100.0

    @property
    def is_depleted(self) -> bool:
        """Check if battery is empty."""
        return self.current_charge_gj <= 0.0

    def discharge(self, power_gw: float, dt: float) -> float:
        """
        Draw power from battery.

        Args:
            power_gw: Requested power (gigawatts).
            dt: Time step (seconds).

        Returns:
            Actual power delivered (gigawatts).
        """
        # Limit by discharge rate
        actual_power = min(power_gw, self.max_discharge_rate_gw)

        # Limit by available energy
        energy_needed = actual_power * dt
        if energy_needed > self.current_charge_gj:
            actual_power = self.current_charge_gj / dt if dt > 0 else 0.0
            energy_needed = self.current_charge_gj

        self.current_charge_gj -= energy_needed
        self.current_charge_gj = max(0.0, self.current_charge_gj)

        return actual_power

    def recharge(self, power_gw: float, dt: float) -> float:
        """
        Recharge battery with available power.

        Args:
            power_gw: Available power (gigawatts).
            dt: Time step (seconds).

        Returns:
            Power actually used for recharging (gigawatts).
        """
        # Limit by recharge rate
        actual_power = min(power_gw, self.max_recharge_rate_gw)

        # Limit by remaining capacity
        space_remaining = self.capacity_gj - self.current_charge_gj
        energy_to_add = actual_power * dt

        if energy_to_add > space_remaining:
            energy_to_add = space_remaining
            actual_power = space_remaining / dt if dt > 0 else 0.0

        self.current_charge_gj += energy_to_add
        self.current_charge_gj = min(self.capacity_gj, self.current_charge_gj)

        return actual_power


# =============================================================================
# REACTOR
# =============================================================================

@dataclass
class Reactor:
    """
    Ship reactor for primary power generation.

    Attributes:
        max_output_gw: Maximum power output (gigawatts).
        efficiency: Thermal efficiency (waste heat factor).
        name: Reactor type name.
        current_output_fraction: Current output as fraction of max (0.0 to 1.0).
    """
    max_output_gw: float
    efficiency: float = 0.999
    name: str = "Reactor"
    current_output_fraction: float = 1.0

    @classmethod
    def from_ship_data(cls, reactor_data: dict) -> Reactor:
        """
        Create a Reactor from ship configuration.

        Args:
            reactor_data: Reactor specification dictionary.

        Returns:
            Configured Reactor instance.
        """
        return cls(
            max_output_gw=reactor_data.get("output_gw", 306430.0),
            efficiency=reactor_data.get("efficiency", 0.999),
            name=reactor_data.get("name", "Reactor")
        )

    @property
    def current_output_gw(self) -> float:
        """Get current power output."""
        return self.max_output_gw * self.current_output_fraction

    def calculate_waste_heat_gw(self) -> float:
        """
        Calculate waste heat from reactor operation.

        Returns:
            Waste heat in gigawatts.
        """
        # Heat = power * (1 - efficiency)
        return self.current_output_gw * (1.0 - self.efficiency)


# =============================================================================
# POWER SYSTEM
# =============================================================================

@dataclass
class PowerSystem:
    """
    Complete power management system for a spacecraft.

    Manages power generation, distribution, and storage:
    - Reactor provides primary power
    - Drives consume power proportional to throttle
    - Remaining power charges weapon capacitors
    - Battery provides backup when reactor insufficient

    Attributes:
        reactor: Ship's reactor for power generation.
        battery: Ship's battery for energy storage.
        weapon_capacitors: Dict mapping weapon slot to capacitor.
        drive_power_fraction: Fraction of reactor power going to drives (0.0 to 1.0).
    """
    reactor: Reactor
    battery: Battery
    weapon_capacitors: Dict[str, WeaponCapacitor] = field(default_factory=dict)
    drive_power_fraction: float = 0.0  # Updated based on throttle

    # Statistics
    _total_heat_generated_gj: float = field(default=0.0, repr=False)

    @classmethod
    def from_ship_data(
        cls,
        ship_data: dict,
        weapon_slots: Optional[List[dict]] = None
    ) -> PowerSystem:
        """
        Create a PowerSystem from ship configuration.

        Args:
            ship_data: Complete ship configuration dictionary.
            weapon_slots: List of weapon slot configurations.

        Returns:
            Configured PowerSystem instance.
        """
        propulsion = ship_data.get("propulsion", {})

        # Create reactor
        reactor_data = propulsion.get("reactor", {})
        reactor = Reactor.from_ship_data(reactor_data)

        # Create battery
        battery_data = propulsion.get("battery", {})
        battery = Battery.from_ship_data(battery_data)

        # Create weapon capacitors
        weapon_capacitors = {}
        if weapon_slots:
            for slot in weapon_slots:
                slot_name = slot.get("slot_name", f"weapon_{len(weapon_capacitors)}")
                weapon_data = slot.get("weapon_data", {})
                capacitor = WeaponCapacitor.from_weapon_data(weapon_data, slot_name)
                weapon_capacitors[slot_name] = capacitor

        return cls(
            reactor=reactor,
            battery=battery,
            weapon_capacitors=weapon_capacitors
        )

    def add_weapon_capacitor(
        self,
        weapon_slot: str,
        weapon_data: dict
    ) -> WeaponCapacitor:
        """
        Add a capacitor for a weapon.

        Args:
            weapon_slot: Identifier for the weapon slot.
            weapon_data: Weapon specification dictionary.

        Returns:
            The created WeaponCapacitor.
        """
        capacitor = WeaponCapacitor.from_weapon_data(weapon_data, weapon_slot)
        self.weapon_capacitors[weapon_slot] = capacitor
        return capacitor

    def set_drive_throttle(self, throttle: float) -> None:
        """
        Set drive power consumption based on throttle.

        When drives are at 100%, they consume all reactor power.
        When drives are at 0%, all power goes to weapons/battery.

        Args:
            throttle: Drive throttle (0.0 to 1.0).
        """
        self.drive_power_fraction = max(0.0, min(1.0, throttle))

    def get_available_power_gw(self) -> float:
        """
        Get reactor power available for weapons and charging.

        Returns:
            Available power in gigawatts.
        """
        drive_power = self.reactor.current_output_gw * self.drive_power_fraction
        return self.reactor.current_output_gw - drive_power

    def can_weapon_fire(self, weapon_slot: str) -> bool:
        """
        Check if a weapon's capacitor is charged enough to fire.

        Args:
            weapon_slot: Identifier for the weapon slot.

        Returns:
            True if weapon can fire.
        """
        capacitor = self.weapon_capacitors.get(weapon_slot)
        if capacitor is None:
            return True  # No power tracking for this weapon
        return capacitor.is_charged

    def fire_weapon(self, weapon_slot: str) -> float:
        """
        Discharge weapon capacitor and calculate heat generated.

        Args:
            weapon_slot: Identifier for the weapon slot.

        Returns:
            Heat generated in gigajoules, or 0 if weapon couldn't fire.
        """
        capacitor = self.weapon_capacitors.get(weapon_slot)
        if capacitor is None:
            return 0.0

        if not capacitor.is_charged:
            return 0.0

        # Discharge capacitor
        capacitor.discharge()

        # Calculate heat (convert MJ to GJ)
        heat_mj = capacitor.calculate_heat_generated()
        heat_gj = heat_mj / 1000.0

        self._total_heat_generated_gj += heat_gj

        return heat_gj

    def update(self, dt: float) -> float:
        """
        Update power system for one timestep.

        Distributes available power to weapon capacitors,
        uses battery if needed, and recharges battery with surplus.

        Args:
            dt: Time step in seconds.

        Returns:
            Total heat generated this timestep (gigajoules).
        """
        heat_generated_gj = 0.0

        # Get available reactor power (after drives)
        available_power_gw = self.get_available_power_gw()

        # Convert to MW for capacitor charging
        available_power_mw = available_power_gw * 1000.0

        # Charge weapon capacitors
        for slot_name, capacitor in self.weapon_capacitors.items():
            if capacitor.is_charged:
                continue  # Already full

            needed_power_mw = capacitor.charge_rate_mw

            if available_power_mw >= needed_power_mw:
                # Reactor can supply all needed power
                energy_mj = needed_power_mw * dt
                capacitor.charge(energy_mj)
                available_power_mw -= needed_power_mw
            else:
                # Need battery supplement
                reactor_contribution_mw = available_power_mw
                shortfall_mw = needed_power_mw - reactor_contribution_mw

                # Try to get power from battery
                shortfall_gw = shortfall_mw / 1000.0
                battery_power_gw = self.battery.discharge(shortfall_gw, dt)
                battery_power_mw = battery_power_gw * 1000.0

                total_power_mw = reactor_contribution_mw + battery_power_mw
                energy_mj = total_power_mw * dt
                capacitor.charge(energy_mj)

                available_power_mw = 0.0

        # Recharge battery with surplus power
        if available_power_mw > 0:
            surplus_gw = available_power_mw / 1000.0
            self.battery.recharge(surplus_gw, dt)

        # Note: Reactor waste heat is already tracked by the ThermalSystem
        # which has dedicated heat sources for reactor and drives.
        # The power system only generates heat when weapons fire.

        return heat_generated_gj

    def get_status(self) -> dict:
        """
        Get current power system status.

        Returns:
            Dictionary with power system state.
        """
        capacitor_status = {}
        for slot_name, cap in self.weapon_capacitors.items():
            capacitor_status[slot_name] = {
                "charge_percent": cap.charge_percent,
                "is_charged": cap.is_charged,
                "capacity_mj": cap.capacity_mj,
                "charge_rate_mw": cap.charge_rate_mw
            }

        return {
            "reactor_output_gw": self.reactor.current_output_gw,
            "available_power_gw": self.get_available_power_gw(),
            "drive_power_fraction": self.drive_power_fraction,
            "battery_percent": self.battery.charge_percent,
            "battery_gj": self.battery.current_charge_gj,
            "weapon_capacitors": capacitor_status,
            "total_heat_generated_gj": self._total_heat_generated_gj
        }


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def calculate_weapon_energy_mj(weapon_data: dict) -> float:
    """
    Calculate energy requirement for a weapon from its specs.

    Args:
        weapon_data: Weapon specification dictionary.

    Returns:
        Energy per shot in megajoules.
    """
    weapon_type = weapon_data.get("type", "kinetic")

    if weapon_type == "point_defense":
        # Laser - use power draw * dwell time
        power_mw = weapon_data.get("power_draw_mw", 5.0)
        cooldown_s = weapon_data.get("cooldown_s", 5.0)
        return power_mw * cooldown_s

    elif weapon_type == "missile":
        # Torpedo launcher - minimal power
        return DEFAULT_TORPEDO_LAUNCH_ENERGY_MJ

    else:
        # Kinetic weapon - based on projectile kinetic energy
        kinetic_energy_gj = weapon_data.get("kinetic_energy_gj", 0.0)
        if kinetic_energy_gj > 0:
            # Assume 50% coil efficiency
            return kinetic_energy_gj * 1000 * 2
        return DEFAULT_KINETIC_ENERGY_MJ


def calculate_weapon_heat_mj(weapon_data: dict, energy_mj: float) -> float:
    """
    Calculate heat generated when firing a weapon.

    Args:
        weapon_data: Weapon specification dictionary.
        energy_mj: Energy discharged from capacitor.

    Returns:
        Heat generated in megajoules.
    """
    weapon_type = weapon_data.get("type", "kinetic")

    if weapon_type == "point_defense":
        efficiency = weapon_data.get("efficiency", LASER_WEAPON_EFFICIENCY)
    elif weapon_type == "missile":
        efficiency = TORPEDO_LAUNCHER_EFFICIENCY
    else:
        efficiency = KINETIC_WEAPON_EFFICIENCY

    return energy_mj * (1.0 - efficiency)
