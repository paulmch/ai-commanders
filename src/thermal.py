"""
Thermal management system for the AI Commanders space battle simulator.

This module implements heat sinks, radiator arrays, and thermal management
for space combat simulations. Droplet radiators are vulnerable to damage
and can be retracted during combat at the cost of reduced heat dissipation.

Based on Terra Invicta thermal mechanics with liquid droplet radiator physics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

# Import HitLocation from combat module for integration
try:
    from .combat import HitLocation
except ImportError:
    from combat import HitLocation


class RadiatorState(Enum):
    """Operational state of a droplet radiator."""
    RETRACTED = "retracted"
    EXTENDED = "extended"
    DAMAGED = "damaged"
    DESTROYED = "destroyed"


class RadiatorPosition(Enum):
    """
    Physical position of radiator on ship hull.

    Radiators are located at the TAIL section of the ship, before the drive
    and next to the reactor for efficient heat transfer. They are tucked
    against the hull on all four sides around the tail/engine section.

    When extended, radiators angle 45° backwards toward the engine, reducing
    frontal cross-section and exposure to forward-facing weapons.
    """
    TAIL_PORT = "tail_port"
    TAIL_STARBOARD = "tail_starboard"
    TAIL_DORSAL = "tail_dorsal"
    TAIL_VENTRAL = "tail_ventral"


# Radiator extension angle - angled backwards toward engine
RADIATOR_EXTENSION_ANGLE_DEG = 45.0


@dataclass
class DropletRadiator:
    """
    A single droplet radiator panel that dissipates heat into space.

    Droplet radiators spray hot liquid metal droplets into space, which
    radiate heat before being recaptured. They are efficient but vulnerable
    to damage when extended.

    Attributes:
        position: Physical location on the ship hull.
        max_dissipation_kw: Maximum heat dissipation rate in kilowatts.
        mass_tons: Radiator mass in metric tons.
        state: Current operational state.
        health_percent: Remaining structural integrity (0-100).
    """
    position: RadiatorPosition
    max_dissipation_kw: float
    mass_tons: float
    state: RadiatorState = RadiatorState.RETRACTED
    health_percent: float = 100.0

    @property
    def current_dissipation_kw(self) -> float:
        """
        Calculate current heat dissipation rate based on state and health.

        Returns:
            Current dissipation rate in kilowatts.
            - 0 if retracted or destroyed
            - max_dissipation_kw * health_factor if extended or damaged
        """
        if self.state == RadiatorState.RETRACTED:
            return 0.0
        if self.state == RadiatorState.DESTROYED:
            return 0.0

        # Damaged radiators operate at reduced efficiency
        health_factor = self.health_percent / 100.0
        return self.max_dissipation_kw * health_factor

    def extend(self) -> bool:
        """
        Extend the radiator for maximum heat dissipation.

        Returns:
            True if successfully extended, False if destroyed.
        """
        if self.state == RadiatorState.DESTROYED:
            return False

        if self.health_percent < 100.0:
            self.state = RadiatorState.DAMAGED
        else:
            self.state = RadiatorState.EXTENDED
        return True

    def retract(self) -> bool:
        """
        Retract the radiator for protection during combat.

        Returns:
            True if successfully retracted, False if destroyed.
        """
        if self.state == RadiatorState.DESTROYED:
            return False

        self.state = RadiatorState.RETRACTED
        return True

    def damage(self, energy_gj: float) -> float:
        """
        Apply damage to the radiator from weapon impact.

        Radiators are fragile systems. Each GJ of damage reduces health
        by approximately 20%, making them vulnerable to even glancing hits.

        Args:
            energy_gj: Damage energy in gigajoules.

        Returns:
            Health remaining after damage (0-100).
        """
        # Radiators are fragile - roughly 5 GJ to destroy
        damage_percent = energy_gj * 20.0
        self.health_percent = max(0.0, self.health_percent - damage_percent)

        if self.health_percent <= 0:
            self.state = RadiatorState.DESTROYED
            self.health_percent = 0.0
        elif self.health_percent < 100.0 and self.state != RadiatorState.RETRACTED:
            self.state = RadiatorState.DAMAGED

        return self.health_percent


@dataclass
class RadiatorArray:
    """
    Complete radiator array with panels on all four sides of the ship.

    Ships mount radiator panels on port, starboard, dorsal, and ventral
    surfaces. During combat, radiators can be retracted for protection
    but this limits heat dissipation capacity.

    Attributes:
        radiators: Dictionary mapping positions to DropletRadiator instances.
    """
    radiators: dict[RadiatorPosition, DropletRadiator] = field(default_factory=dict)

    @classmethod
    def from_ship_data(cls, thermal_data: dict) -> RadiatorArray:
        """
        Create a RadiatorArray from ship thermal configuration.

        Args:
            thermal_data: Ship's thermal section from fleet data.

        Returns:
            Configured RadiatorArray with four radiator panels.
        """
        radiator_data = thermal_data.get("radiator", {})
        mass_tons = radiator_data.get("mass_tons", 10.0)
        dissipation_kw_per_kg = radiator_data.get("dissipation_kw_per_kg", 13.0)

        # Total dissipation capacity: mass * dissipation rate
        # Divide equally among 4 radiators
        total_dissipation_kw = mass_tons * 1000 * dissipation_kw_per_kg
        per_radiator_dissipation = total_dissipation_kw / 4.0
        per_radiator_mass = mass_tons / 4.0

        radiators = {}
        for position in RadiatorPosition:
            radiators[position] = DropletRadiator(
                position=position,
                max_dissipation_kw=per_radiator_dissipation,
                mass_tons=per_radiator_mass,
                state=RadiatorState.RETRACTED,
                health_percent=100.0,
            )

        return cls(radiators=radiators)

    @property
    def total_dissipation_kw(self) -> float:
        """
        Calculate total heat dissipation across all radiators.

        Returns:
            Combined dissipation rate in kilowatts.
        """
        return sum(r.current_dissipation_kw for r in self.radiators.values())

    def extend_all(self) -> int:
        """
        Extend all non-destroyed radiators.

        Returns:
            Number of radiators successfully extended.
        """
        count = 0
        for radiator in self.radiators.values():
            if radiator.extend():
                count += 1
        return count

    def retract_all(self) -> int:
        """
        Retract all non-destroyed radiators.

        Returns:
            Number of radiators successfully retracted.
        """
        count = 0
        for radiator in self.radiators.values():
            if radiator.retract():
                count += 1
        return count

    def get_hit_probability(self, hit_location: HitLocation) -> float:
        """
        Calculate probability of radiator being hit based on attack angle.

        Only radiators on the side facing the attack can be hit.
        Retracted radiators are tucked against the hull (~5% hit chance).
        Extended radiators are more exposed (~20% hit chance).

        Args:
            hit_location: The attack vector/location.

        Returns:
            Probability (0.0-1.0) that a radiator is hit.
        """
        # Determine which radiators can be hit based on attack angle
        vulnerable_positions = self._get_vulnerable_positions(hit_location)

        if not vulnerable_positions:
            return 0.0

        # Calculate hit probability based on radiator states
        # We return the probability for any single radiator being hit
        max_prob = 0.0
        for pos in vulnerable_positions:
            radiator = self.radiators.get(pos)
            if radiator is None:
                continue

            if radiator.state == RadiatorState.DESTROYED:
                continue
            elif radiator.state == RadiatorState.RETRACTED:
                # Tucked against hull, minimal exposure
                prob = 0.05
            else:
                # Extended at 45° angle backwards
                # Base exposure ~15% when extended (still tucked in droplet design)
                # Angle affects effective cross-section:
                # - Lateral hits see cos(45°) ≈ 0.707 of radiator
                # - Tail hits see sin(45°) ≈ 0.707 of radiator
                angle_factor = math.cos(math.radians(RADIATOR_EXTENSION_ANGLE_DEG))

                if hit_location == HitLocation.TAIL:
                    # Tail sees more of the angled radiator
                    prob = 0.15 * (1.0 + (1.0 - angle_factor))  # ~0.19
                else:
                    # Lateral sees reduced cross-section
                    prob = 0.15 * angle_factor  # ~0.11

            max_prob = max(max_prob, prob)

        return max_prob

    def _get_vulnerable_positions(
        self, hit_location: HitLocation
    ) -> List[RadiatorPosition]:
        """
        Determine which radiator positions are vulnerable to a hit.

        Radiators are located at the TAIL section (before drive, next to reactor).
        - TAIL hits: Direct exposure to all 4 radiators (highest risk)
        - LATERAL hits: Can strike tail_port OR tail_starboard
        - NOSE hits: Must penetrate entire ship - radiators protected (minimal risk)

        Args:
            hit_location: The attack vector.

        Returns:
            List of radiator positions that could be hit.
        """
        if hit_location == HitLocation.TAIL:
            # Tail hits directly expose all radiators around the engine section
            return [
                RadiatorPosition.TAIL_PORT,
                RadiatorPosition.TAIL_STARBOARD,
                RadiatorPosition.TAIL_DORSAL,
                RadiatorPosition.TAIL_VENTRAL,
            ]
        elif hit_location == HitLocation.LATERAL:
            # Lateral hits can strike port OR starboard tail radiators
            return [RadiatorPosition.TAIL_PORT, RadiatorPosition.TAIL_STARBOARD]
        elif hit_location == HitLocation.NOSE:
            # Nose hits rarely reach tail radiators - would need full penetration
            # Return empty - radiators protected by entire ship length
            return []

        return []

    def get_random_vulnerable_radiator(
        self, hit_location: HitLocation
    ) -> Optional[DropletRadiator]:
        """
        Get a random non-destroyed radiator vulnerable to the given hit.

        Args:
            hit_location: The attack vector.

        Returns:
            A vulnerable radiator or None if none available.
        """
        import random

        vulnerable_positions = self._get_vulnerable_positions(hit_location)
        vulnerable_radiators = [
            self.radiators[pos]
            for pos in vulnerable_positions
            if pos in self.radiators
            and self.radiators[pos].state != RadiatorState.DESTROYED
        ]

        if not vulnerable_radiators:
            return None

        return random.choice(vulnerable_radiators)


@dataclass
class HeatSink:
    """
    Thermal mass for absorbing waste heat before radiating to space.

    Heat sinks use materials like lithium to store large amounts of
    thermal energy. This allows ships to temporarily exceed their
    radiator capacity during intense combat operations.

    Attributes:
        capacity_gj: Maximum heat storage in gigajoules.
        current_heat_gj: Current stored heat in gigajoules.
    """
    capacity_gj: float
    current_heat_gj: float = 0.0

    @classmethod
    def from_ship_data(cls, thermal_data: dict) -> HeatSink:
        """
        Create a HeatSink from ship thermal configuration.

        Args:
            thermal_data: Ship's thermal section from fleet data.

        Returns:
            Configured HeatSink instance.
        """
        heatsink_data = thermal_data.get("heatsink", {})
        capacity = heatsink_data.get("capacity_gj", 525.0)

        return cls(capacity_gj=capacity, current_heat_gj=0.0)

    @property
    def heat_percent(self) -> float:
        """
        Get current heat level as percentage of capacity.

        Returns:
            Heat percentage from 0.0 to 100.0.
        """
        if self.capacity_gj <= 0:
            return 100.0
        return (self.current_heat_gj / self.capacity_gj) * 100.0

    @property
    def available_capacity_gj(self) -> float:
        """
        Get remaining heat absorption capacity.

        Returns:
            Available capacity in gigajoules.
        """
        return max(0.0, self.capacity_gj - self.current_heat_gj)

    def absorb(self, heat_gj: float) -> bool:
        """
        Absorb heat into the sink.

        Args:
            heat_gj: Amount of heat to absorb in gigajoules.

        Returns:
            True if all heat was absorbed, False if sink is full.
        """
        if heat_gj <= 0:
            return True

        available = self.available_capacity_gj

        if heat_gj <= available:
            self.current_heat_gj += heat_gj
            return True
        else:
            # Absorb what we can, but report failure
            self.current_heat_gj = self.capacity_gj
            return False

    def dump_to_radiators(
        self, radiator_array: RadiatorArray, dt_seconds: float
    ) -> float:
        """
        Transfer heat from sink to radiators for dissipation.

        Args:
            radiator_array: The ship's radiator array.
            dt_seconds: Time step in seconds.

        Returns:
            Amount of heat dumped in gigajoules.
        """
        if self.current_heat_gj <= 0:
            return 0.0

        # Calculate heat that can be radiated this timestep
        # dissipation_kw = kJ/s, so we convert to GJ
        dissipation_gj_per_s = radiator_array.total_dissipation_kw / 1_000_000.0
        max_dump_gj = dissipation_gj_per_s * dt_seconds

        # Can't dump more than we have
        actual_dump = min(max_dump_gj, self.current_heat_gj)
        self.current_heat_gj -= actual_dump

        return actual_dump


@dataclass
class HeatSource:
    """
    A system that generates waste heat during operation.

    Attributes:
        name: Identifier for the heat source.
        heat_generation_kw: Heat output rate in kilowatts when active.
        active: Whether the source is currently generating heat.
    """
    name: str
    heat_generation_kw: float
    active: bool = False


# Standard heat generation rates (kW)
HEAT_GENERATION_RATES = {
    "reactor_idle": 1000.0,          # Reactor at idle
    "reactor_full_power": 50000.0,   # Reactor at full power
    "engine_per_mn": 10000.0,        # Per MN of thrust
    "coilgun_per_shot": 500.0,       # Per coilgun shot
    "laser_pd_continuous": 100.0,    # Laser PD continuous operation
}


@dataclass
class ThermalSystem:
    """
    Complete thermal management system for a spacecraft.

    Manages heat generation from ship systems, heat absorption into
    sinks, and heat dissipation through radiators. Critical for
    managing ship operations during extended combat.

    Attributes:
        heatsink: The ship's heat sink for thermal storage.
        radiators: The ship's radiator array for heat dissipation.
        heat_sources: List of heat-generating systems.
    """
    heatsink: HeatSink
    radiators: RadiatorArray
    heat_sources: List[HeatSource] = field(default_factory=list)

    # Warning thresholds
    OVERHEAT_THRESHOLD: float = 80.0
    CRITICAL_THRESHOLD: float = 95.0

    @classmethod
    def from_ship_data(cls, ship_data: dict) -> ThermalSystem:
        """
        Create a ThermalSystem from ship configuration data.

        Initializes heat sink and radiators from fleet data, and creates
        standard heat sources based on ship systems.

        Args:
            ship_data: Complete ship configuration dictionary.

        Returns:
            Configured ThermalSystem instance.
        """
        thermal_data = ship_data.get("thermal", {})

        heatsink = HeatSink.from_ship_data(thermal_data)
        radiators = RadiatorArray.from_ship_data(thermal_data)

        # Create standard heat sources based on ship systems
        heat_sources = []

        # Reactor (always present)
        heat_sources.append(HeatSource(
            name="reactor",
            heat_generation_kw=HEAT_GENERATION_RATES["reactor_idle"],
            active=True,  # Reactor always running
        ))

        # Engine (based on propulsion)
        propulsion = ship_data.get("propulsion", {})
        drive = propulsion.get("drive", {})
        thrust_mn = drive.get("thrust_mn", 58.56)
        engine_heat = thrust_mn * HEAT_GENERATION_RATES["engine_per_mn"]
        heat_sources.append(HeatSource(
            name="engines",
            heat_generation_kw=engine_heat,
            active=False,  # Engines off by default
        ))

        # Weapons
        weapons = ship_data.get("weapons", [])
        coilgun_count = 0
        pd_count = 0

        for weapon in weapons:
            weapon_type = weapon.get("type", "")
            if "coilgun" in weapon_type or "coiler" in weapon_type:
                coilgun_count += 1
            elif "pd" in weapon_type or "laser" in weapon_type:
                pd_count += 1

        if coilgun_count > 0:
            heat_sources.append(HeatSource(
                name="coilguns",
                heat_generation_kw=coilgun_count * HEAT_GENERATION_RATES["coilgun_per_shot"],
                active=False,
            ))

        if pd_count > 0:
            heat_sources.append(HeatSource(
                name="point_defense",
                heat_generation_kw=pd_count * HEAT_GENERATION_RATES["laser_pd_continuous"],
                active=False,
            ))

        return cls(
            heatsink=heatsink,
            radiators=radiators,
            heat_sources=heat_sources,
        )

    @property
    def heat_percent(self) -> float:
        """
        Get current heat level as percentage of capacity.

        Returns:
            Heat percentage from 0.0 to 100.0.
        """
        return self.heatsink.heat_percent

    @property
    def is_overheating(self) -> bool:
        """
        Check if ship is in overheating warning state.

        Returns:
            True if heat exceeds 80% of capacity.
        """
        return self.heat_percent >= self.OVERHEAT_THRESHOLD

    @property
    def is_critical(self) -> bool:
        """
        Check if ship is in critical thermal state.

        Returns:
            True if heat exceeds 95% of capacity.
        """
        return self.heat_percent >= self.CRITICAL_THRESHOLD

    def add_heat(self, source_name: str, heat_gj: float) -> bool:
        """
        Add heat from a specific source to the system.

        Args:
            source_name: Name of the heat source (for logging).
            heat_gj: Amount of heat to add in gigajoules.

        Returns:
            True if heat was absorbed, False if sink is full.
        """
        return self.heatsink.absorb(heat_gj)

    def set_source_active(self, source_name: str, active: bool) -> bool:
        """
        Activate or deactivate a heat source.

        Args:
            source_name: Name of the heat source.
            active: Whether to activate or deactivate.

        Returns:
            True if source was found and updated.
        """
        for source in self.heat_sources:
            if source.name == source_name:
                source.active = active
                return True
        return False

    def get_total_heat_generation_kw(self) -> float:
        """
        Calculate total heat generation from all active sources.

        Returns:
            Total heat generation in kilowatts.
        """
        return sum(
            source.heat_generation_kw
            for source in self.heat_sources
            if source.active
        )

    def update(self, dt_seconds: float) -> dict:
        """
        Update thermal system for a time step.

        Generates heat from active sources, then dissipates through
        radiators. Returns status information about the update.

        Args:
            dt_seconds: Time step in seconds.

        Returns:
            Dictionary with update statistics:
            - heat_generated_gj: Heat added this step
            - heat_dissipated_gj: Heat radiated this step
            - net_heat_gj: Net heat change (positive = heating up)
            - heat_percent: Current heat level
            - is_overheating: Whether in warning state
            - is_critical: Whether in critical state
        """
        # Generate heat from active sources
        heat_gen_kw = self.get_total_heat_generation_kw()
        heat_gen_gj = (heat_gen_kw / 1_000_000.0) * dt_seconds

        # Add generated heat to sink
        self.heatsink.absorb(heat_gen_gj)

        # Dissipate heat through radiators
        heat_dissipated = self.heatsink.dump_to_radiators(
            self.radiators, dt_seconds
        )

        return {
            "heat_generated_gj": heat_gen_gj,
            "heat_dissipated_gj": heat_dissipated,
            "net_heat_gj": heat_gen_gj - heat_dissipated,
            "heat_percent": self.heat_percent,
            "is_overheating": self.is_overheating,
            "is_critical": self.is_critical,
        }

    def get_status(self) -> dict:
        """
        Get comprehensive thermal system status.

        Returns:
            Dictionary with full system status.
        """
        radiator_status = {}
        for pos, radiator in self.radiators.radiators.items():
            radiator_status[pos.value] = {
                "state": radiator.state.value,
                "health_percent": radiator.health_percent,
                "dissipation_kw": radiator.current_dissipation_kw,
            }

        source_status = {}
        for source in self.heat_sources:
            source_status[source.name] = {
                "heat_generation_kw": source.heat_generation_kw,
                "active": source.active,
            }

        return {
            "heatsink": {
                "current_heat_gj": self.heatsink.current_heat_gj,
                "capacity_gj": self.heatsink.capacity_gj,
                "heat_percent": self.heatsink.heat_percent,
            },
            "radiators": {
                "total_dissipation_kw": self.radiators.total_dissipation_kw,
                "panels": radiator_status,
            },
            "heat_sources": source_status,
            "is_overheating": self.is_overheating,
            "is_critical": self.is_critical,
        }

    def is_radiator_extended(self, position: RadiatorPosition) -> bool:
        """
        Check if a radiator at the given position is extended.

        Args:
            position: The radiator position to check.

        Returns:
            True if the radiator is extended (or damaged but still active),
            False if retracted or destroyed.
        """
        radiator = self.radiators.radiators.get(position)
        if radiator is None:
            return False

        return radiator.state in (RadiatorState.EXTENDED, RadiatorState.DAMAGED)

    def get_radiator_dissipation(self, position: RadiatorPosition) -> float:
        """
        Get the current dissipation capacity of a radiator in kW.

        Args:
            position: The radiator position to query.

        Returns:
            Current dissipation capacity in kW (0.0 if destroyed or not found).
        """
        radiator = self.radiators.radiators.get(position)
        if radiator is None:
            return 0.0

        return radiator.current_dissipation_kw

    def damage_radiator(self, position: RadiatorPosition, energy_gj: float) -> float:
        """
        Apply damage to a radiator at the given position.

        Args:
            position: The radiator position to damage.
            energy_gj: Damage energy in gigajoules.

        Returns:
            Remaining health of the radiator (0-100), or -1 if position invalid.
        """
        radiator = self.radiators.radiators.get(position)
        if radiator is None:
            return -1.0

        return radiator.damage(energy_gj)

    def apply_radiator_damage(
        self,
        position: RadiatorPosition,
        damage_gj: float
    ) -> tuple[float, bool, float]:
        """
        Apply damage to a radiator and return the results.

        This method implements the ThermalSystemProtocol interface expected
        by the combat system for radiator damage resolution.

        Args:
            position: The radiator position to damage.
            damage_gj: Amount of damage in gigajoules.

        Returns:
            Tuple of (damage_taken_gj, destroyed, dissipation_lost_kw).
        """
        radiator = self.radiators.radiators.get(position)
        if radiator is None:
            return (0.0, False, 0.0)

        # Get current dissipation before damage
        dissipation_before = radiator.current_dissipation_kw

        # Check if already destroyed
        if radiator.state == RadiatorState.DESTROYED:
            return (0.0, False, 0.0)

        # Apply damage
        prev_health = radiator.health_percent
        radiator.damage(damage_gj)

        # Calculate results
        damage_taken = min(damage_gj, prev_health / 20.0)  # ~20% per GJ
        destroyed = radiator.state == RadiatorState.DESTROYED
        dissipation_after = radiator.current_dissipation_kw
        dissipation_lost = dissipation_before - dissipation_after

        return (damage_taken, destroyed, dissipation_lost)


def load_thermal_system(
    fleet_data_path: str | Path,
    ship_type: str
) -> ThermalSystem:
    """
    Load a thermal system for a specific ship type from fleet data.

    Args:
        fleet_data_path: Path to the fleet_ships.json file.
        ship_type: Type of ship (e.g., 'destroyer', 'cruiser').

    Returns:
        Configured ThermalSystem for the ship.

    Raises:
        KeyError: If ship type not found in fleet data.
        FileNotFoundError: If fleet data file not found.
    """
    with open(fleet_data_path, "r") as f:
        fleet_data = json.load(f)

    ships = fleet_data.get("ships", {})
    if ship_type not in ships:
        raise KeyError(f"Ship type '{ship_type}' not found in fleet data")

    return ThermalSystem.from_ship_data(ships[ship_type])


if __name__ == "__main__":
    # Example usage and basic validation
    import sys

    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"

    if not data_path.exists():
        print(f"Fleet data not found at {data_path}")
        sys.exit(1)

    print("AI Commanders Thermal Management System")
    print("=" * 50)

    # Load thermal system for a destroyer
    thermal = load_thermal_system(data_path, "destroyer")

    print("\nDestroyer Thermal System Status:")
    print(f"  Heat Sink Capacity: {thermal.heatsink.capacity_gj} GJ")
    print(f"  Current Heat: {thermal.heatsink.current_heat_gj:.2f} GJ ({thermal.heat_percent:.1f}%)")

    # Extend radiators
    thermal.radiators.extend_all()
    print(f"\n  Radiator Dissipation (extended): {thermal.radiators.total_dissipation_kw:.0f} kW")

    # Show radiator status
    print("\n  Radiator Panels:")
    for pos, radiator in thermal.radiators.radiators.items():
        print(f"    {pos.value}: {radiator.state.value}, "
              f"{radiator.current_dissipation_kw:.0f} kW, "
              f"{radiator.health_percent:.0f}% health")

    # Show heat sources
    print("\n  Heat Sources:")
    for source in thermal.heat_sources:
        status = "ACTIVE" if source.active else "inactive"
        print(f"    {source.name}: {source.heat_generation_kw:.0f} kW ({status})")

    # Simulate some combat operations
    print("\n" + "=" * 50)
    print("Simulating 60 seconds of combat...")

    # Activate engines and weapons
    thermal.set_source_active("engines", True)
    thermal.set_source_active("coilguns", True)
    thermal.set_source_active("point_defense", True)

    # Simulate for 60 seconds
    for t in range(60):
        result = thermal.update(1.0)

        if t % 10 == 0:
            print(f"\n  t={t}s: Heat {result['heat_percent']:.1f}%, "
                  f"+{result['heat_generated_gj']*1000:.1f} MJ, "
                  f"-{result['heat_dissipated_gj']*1000:.1f} MJ")

            if result['is_overheating']:
                print("    WARNING: OVERHEATING!")
            if result['is_critical']:
                print("    CRITICAL: THERMAL EMERGENCY!")

    # Test radiator damage
    print("\n" + "=" * 50)
    print("Simulating radiator damage...")

    port_radiator = thermal.radiators.radiators[RadiatorPosition.PORT]
    print(f"\n  Port radiator before damage: {port_radiator.health_percent:.0f}% health")
    port_radiator.damage(2.5)  # 2.5 GJ hit
    print(f"  Port radiator after 2.5 GJ hit: {port_radiator.health_percent:.0f}% health, "
          f"state: {port_radiator.state.value}")

    print(f"\n  Total dissipation after damage: {thermal.radiators.total_dissipation_kw:.0f} kW")

    # Test hit probability
    print("\n" + "=" * 50)
    print("Radiator hit probabilities:")

    from combat import HitLocation
    for loc in HitLocation:
        prob = thermal.radiators.get_hit_probability(loc)
        print(f"  {loc.value}: {prob*100:.0f}%")
