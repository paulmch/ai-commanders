"""
Power System for the AI Commanders space battle simulator.

This module implements spacecraft power management including:
- Weapon capacitors that charge during cooldown
- Reactor power distribution (drives vs weapons)
- Battery backup when reactor power is insufficient
- Heat generation from weapon firing

Power Flow Architecture (Terra Invicta: "ships generate power from a dedicated
reactor that should be sufficient to all non-propulsion needs ... battery
utility modules serve as a further backup"):
1. The drive's power is its own business - drive throttle never starves weapons
2. A ship-systems bus charges the weapon capacitors. It is rated at every
   non-propulsion load at once (each capacitor's capacity / cooldown plus the
   battery recharge rate), scaled by reactor module effectiveness
   (set_bus_scale)
3. When the bus falls short (damaged reactor), the battery covers the deficit
   up to its discharge rate; bus surplus recharges the battery
4. Weapons can only fire when their capacitor is full
5. Firing generates heat (shot energy x (1 - efficiency)) sent to the thermal
   system

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

# Efficiency factors for heat generation.
# Our guns are the Mk3 magnetic guns, and Terra Invicta's Mk3 templates
# (CoilgunBatteryMk3, HeavyCoilgunBatteryMk3, SpinalCoilerMk3,
# HeavySiegeCoilerMk3) all have efficiency 0.7 (0.5 is TI's Mk1 value). A shot
# draws shot energy = KE / efficiency from its capacitor and the difference,
# KE x (1/efficiency - 1), is waste heat: coilgun 0.31 GJ, heavy 0.52 GJ,
# spinal 1.84 GJ, siege 3.11 GJ per shot. Capacitor sizing and waste heat are
# both derived from this constant (WeaponCapacitor.from_weapon_data), so the
# per-shot energy budget stays closed; there is no separate per-shot adder.
KINETIC_WEAPON_EFFICIENCY = 0.70  # TI Mk3 magnetic guns
LASER_WEAPON_EFFICIENCY = 0.25    # Terra Invicta PD laser efficiency
TORPEDO_LAUNCHER_EFFICIENCY = 0.90  # Mostly mechanical, low heat

# Terra Invicta PointDefenseLaserTurret: one 50 MJ shot per 5 s cooldown at
# efficiency 0.25 -> 200 MJ drawn and 150 MJ of waste heat per cycle. Our PD
# turret is a continuous-dwell beam re-armed every cooldown (one burst = one
# TI shot cycle), so the burst draws 200 MJ (40 MW) and dumps 150 MJ into the
# hull: a 30 MW continuous-dwell equivalent while the turret is firing.
TI_PD_SHOT_POWER_MJ = 50.0

# Battery discharge limit as a multiple of its recharge rate. TI batteries have
# no discharge-rate field; 10x (0.75 GW for the SuperconductingCoilBattery) is
# a stated game assumption. It only binds when the systems bus is damaged:
# a healthy bus covers every capacitor's full-cadence draw on its own.
BATTERY_DISCHARGE_TO_RECHARGE_RATIO = 10.0

# Capacitor charging priority on the systems bus: point defense first, then
# torpedo launchers, then guns.
CAPACITOR_PRIORITY = {"point_defense": 0, "missile": 1, "kinetic": 2}

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
    kind: str = "kinetic"  # "kinetic" | "point_defense" | "missile"

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
            kind = "point_defense"
            efficiency = weapon_data.get("efficiency", LASER_WEAPON_EFFICIENCY)
            if "shot_power_mj" in weapon_data:
                # TI laser convention: the capacitor holds shotPower / efficiency
                capacity_mj = weapon_data["shot_power_mj"] / efficiency
            else:
                # Legacy: electrical draw (MW) x burst length
                capacity_mj = weapon_data.get("power_draw_mw", 5.0) * cooldown_s
        elif weapon_type == "missile":
            # Torpedo launchers have low energy requirement
            kind = "missile"
            capacity_mj = DEFAULT_TORPEDO_LAUNCH_ENERGY_MJ
            efficiency = TORPEDO_LAUNCHER_EFFICIENCY
        else:
            kind = "kinetic"
            # Kinetic weapons - use kinetic energy as proxy
            kinetic_energy_gj = weapon_data.get("kinetic_energy_gj", 0.0)
            if kinetic_energy_gj > 0:
                # Capacitor must store the muzzle energy divided by coil
                # efficiency, so that capacity*(1 - efficiency) is exactly the
                # energy that did NOT reach the slug (see
                # calculate_heat_generated). Derived rather than hardcoded at
                # 2x so sizing and waste heat can never disagree again.
                capacity_mj = (
                    kinetic_energy_gj * 1000 / KINETIC_WEAPON_EFFICIENCY
                )
            else:
                capacity_mj = DEFAULT_KINETIC_ENERGY_MJ
            efficiency = KINETIC_WEAPON_EFFICIENCY

        # A salvo still spends one capacitor per shot. Size recharge for the
        # intra-salvo cadence so the longer inter-salvo wait does not gate it.
        recharge_s = cooldown_s
        if weapon_data.get("salvo_size", 1) > 1:
            recharge_s = weapon_data.get("intra_salvo_cooldown_s", cooldown_s)
        charge_rate_mw = capacity_mj / recharge_s if recharge_s > 0 else capacity_mj

        return cls(
            capacity_mj=capacity_mj,
            charge_rate_mw=charge_rate_mw,
            weapon_slot=weapon_slot,
            efficiency=efficiency,
            current_charge_mj=capacity_mj,  # Start fully charged
            kind=kind,
        )

    @property
    def is_charged(self) -> bool:
        """Check if capacitor has enough energy to fire."""
        # Fractional recharge rates (e.g. siege coilers over 18s) can sum to
        # a few ulps below capacity. Do not delay the shot by a whole sim tick.
        return (self.current_charge_mj >= self.capacity_mj or math.isclose(
            self.current_charge_mj, self.capacity_mj, rel_tol=1e-12, abs_tol=0.0))

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
        # Snap float residue: capacity/cooldown summed over a cooldown's worth
        # of ticks can land an ulp short of capacity (e.g. 6128.57 MJ / 15 s),
        # which would hold the shot one extra tick - a dt-dependent delay.
        if self.capacity_mj - self.current_charge_mj <= 1e-9 * self.capacity_mj:
            self.current_charge_mj = self.capacity_mj

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
        current_charge_gj: Current stored energy (gigajoules). Defaults to
            None, meaning "start at full capacity".
        max_discharge_rate_gw: Maximum power output (gigawatts).
        max_recharge_rate_gw: Maximum recharge rate (gigawatts).
        name: Battery type name.
    """
    capacity_gj: float
    max_discharge_rate_gw: float
    max_recharge_rate_gw: float
    name: str = "Ship Battery"
    current_charge_gj: Optional[float] = field(default=None)

    def __post_init__(self):
        """
        Initialize battery to full charge if not specified.

        The sentinel is None, not 0.0: with 0.0 as the sentinel a legitimately
        depleted battery was indistinguishable from "unspecified" and got
        silently refilled, so an empty battery could not be constructed and
        dataclasses.replace() on a drained battery recharged it.
        """
        if self.current_charge_gj is None:
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

        # TI has no discharge-rate field: stated game assumption, see
        # BATTERY_DISCHARGE_TO_RECHARGE_RATIO.
        discharge_rate = recharge_rate * BATTERY_DISCHARGE_TO_RECHARGE_RATIO

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

    def set_output_fraction(self, fraction: float) -> float:
        """
        Set reactor output as a fraction of maximum, clamped to a valid range.

        Note: nothing in the simulation currently throttles the reactor, so it
        runs pinned at 1.0 and unconsumed output is simply discarded (it is not
        turned into waste heat, fuel burn, or a throttle-down). Wiring this to
        actual demand belongs in the simulation loop, not here.

        Args:
            fraction: Requested output fraction.

        Returns:
            The clamped fraction actually applied.
        """
        self.current_output_fraction = max(0.0, min(1.0, fraction))
        return self.current_output_fraction

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
    - The drive runs off its own plant output; throttle never starves weapons
    - A ship-systems bus charges the weapon capacitors at up to its rating
      (by default the sum of every capacitor's full-cadence charge rate),
      scaled by reactor module effectiveness
    - The battery covers any bus deficit (damaged reactor) and recharges
      from bus surplus

    Attributes:
        reactor: Ship's reactor for power generation.
        battery: Ship's battery for energy storage.
        weapon_capacitors: Dict mapping weapon slot to capacitor.
        drive_power_fraction: Applied drive throttle (informational only).
        bus_scale: Reactor module effectiveness, 0.0-1.0 (set_bus_scale).
        bus_rating_mw: Explicit bus rating; None derives it from the capacitors.
    """
    reactor: Reactor
    battery: Battery
    weapon_capacitors: Dict[str, WeaponCapacitor] = field(default_factory=dict)
    drive_power_fraction: float = 0.0  # Informational: applied drive throttle
    bus_scale: float = 1.0
    bus_rating_mw: Optional[float] = None

    # Statistics
    _total_heat_generated_gj: float = field(default=0.0, repr=False)
    last_bus_draw_mw: float = field(default=0.0, repr=False)
    last_battery_draw_mw: float = field(default=0.0, repr=False)

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

        # Create battery. Fleet data keeps it under thermal.battery (with the
        # heat sink and radiators it shares the utility budget with); this
        # used to read only propulsion.battery, so editing the data had no
        # effect and the hard-coded defaults happened to match.
        battery_data = (
            ship_data.get("thermal", {}).get("battery")
            or propulsion.get("battery")
            or {}
        )
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
        Record the applied drive throttle (status only).

        The drive no longer shares the weapons' power: the old coupling gave
        the guns 99.99% of a 306 TW reactor at 99% throttle and nothing but
        the battery at exactly 100% - a cliff with no physical basis.

        Args:
            throttle: Drive throttle (0.0 to 1.0).
        """
        self.drive_power_fraction = max(0.0, min(1.0, throttle))

    def set_bus_scale(self, scale: float) -> None:
        """Scale the systems bus by reactor module effectiveness (0.0-1.0)."""
        self.bus_scale = max(0.0, min(1.0, scale))

    def get_bus_rating_mw(self) -> float:
        """
        Healthy bus rating: explicit, else every non-propulsion load at once -
        each capacitor's full-cadence draw plus the battery's recharge rate
        (TI: the systems reactor covers all non-propulsion needs). A healthy
        ship therefore never waits on power; a damaged reactor does.
        """
        if self.bus_rating_mw is not None:
            return self.bus_rating_mw
        return (sum(c.charge_rate_mw for c in self.weapon_capacitors.values())
                + self.battery.max_recharge_rate_gw * 1000.0)

    def get_available_power_gw(self) -> float:
        """
        Get systems-bus power available for charging capacitors.

        Returns:
            Available power in gigawatts (bus rating x reactor effectiveness).
        """
        return self.get_bus_rating_mw() * self.bus_scale / 1000.0

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

    def _charge_order(self) -> List[WeaponCapacitor]:
        """Capacitors in bus priority order (PD, then launchers, then guns)."""
        return sorted(
            self.weapon_capacitors.values(),
            key=lambda c: CAPACITOR_PRIORITY.get(c.kind, len(CAPACITOR_PRIORITY)),
        )

    def update(self, dt: float) -> float:
        """
        Update power system for one timestep.

        Charges capacitors from the systems bus in priority order, covers any
        shortfall from the battery (within its per-tick discharge budget) and
        recharges the battery with bus surplus.

        Args:
            dt: Time step in seconds.

        Returns:
            Total heat generated this timestep (gigajoules) - always 0: the
            conversion loss is charged when a capacitor discharges.
        """
        if dt <= 0:
            return 0.0

        available_mw = self.get_bus_rating_mw() * self.bus_scale
        bus_used_mw = 0.0
        battery_used_mw = 0.0

        # Battery energy budget for this whole timestep. Battery.discharge()
        # only caps power per invocation, so calling it once per uncharged
        # capacitor previously let N weapons each draw the full rated power -
        # i.e. N x max_discharge_rate_gw out of one battery in one tick.
        battery_budget_gj = max(0.0, self.battery.max_discharge_rate_gw * dt)

        for capacitor in self._charge_order():
            if capacitor.is_charged:
                continue
            # Draw only what this capacitor can still take this tick.
            room_mw = (capacitor.capacity_mj - capacitor.current_charge_mj) / dt
            need_mw = max(0.0, min(capacitor.charge_rate_mw, room_mw))
            from_bus_mw = min(need_mw, available_mw)
            available_mw -= from_bus_mw
            bus_used_mw += from_bus_mw

            shortfall_mw = need_mw - from_bus_mw
            from_battery_mw = 0.0
            if shortfall_mw > 0.0 and battery_budget_gj > 0.0:
                request_gw = min(shortfall_mw / 1000.0, battery_budget_gj / dt)
                battery_power_gw = self.battery.discharge(request_gw, dt)
                battery_budget_gj = max(
                    0.0, battery_budget_gj - battery_power_gw * dt
                )
                from_battery_mw = battery_power_gw * 1000.0
                battery_used_mw += from_battery_mw

            capacitor.charge((from_bus_mw + from_battery_mw) * dt)

        # Recharge battery with bus surplus
        if available_mw > 0:
            self.battery.recharge(available_mw / 1000.0, dt)

        self.last_bus_draw_mw = bus_used_mw
        self.last_battery_draw_mw = battery_used_mw
        # Reactor waste heat is not added here: the thermal system carries the
        # plant's housekeeping load and the drive's heat. Weapon heat is
        # emitted at fire time by fire_weapon(); charging is lossless.
        return 0.0

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

        rating = self.get_bus_rating_mw()
        return {
            "reactor_output_gw": self.reactor.current_output_gw,
            "available_power_gw": self.get_available_power_gw(),
            "bus_rating_mw": rating,
            "bus_percent": self.bus_scale * 100.0,
            "bus_draw_mw": self.last_bus_draw_mw,
            "battery_draw_mw": self.last_battery_draw_mw,
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
        # Same derivation as WeaponCapacitor.from_weapon_data
        if "shot_power_mj" in weapon_data:
            return weapon_data["shot_power_mj"] / weapon_data.get(
                "efficiency", LASER_WEAPON_EFFICIENCY)
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
            # Same derivation as WeaponCapacitor.from_weapon_data
            return kinetic_energy_gj * 1000 / KINETIC_WEAPON_EFFICIENCY
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
