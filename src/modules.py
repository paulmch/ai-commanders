"""
Module layout system for the AI Commanders space battle simulator.

This module implements internal ship module systems, their positioning within
the ship structure, and damage propagation mechanics for space combat simulations.

Modules are organized in layers from nose to tail, with critical systems
(reactor, bridge) positioned in protected center locations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .combat import HitLocation, HitResult


class ModuleType(Enum):
    """Types of ship modules with their functional roles."""
    SENSOR = "sensor"
    BRIDGE = "bridge"
    REACTOR = "reactor"
    ENGINE = "engine"
    WEAPON = "weapon"
    CARGO = "cargo"
    CREW = "crew"
    FUEL_TANK = "fuel_tank"


# Critical module types that can cause catastrophic damage if destroyed
CRITICAL_MODULE_TYPES: set[ModuleType] = {ModuleType.REACTOR, ModuleType.BRIDGE}


@dataclass
class ModulePosition:
    """
    Position of a module within the ship structure.

    Attributes:
        layer_index: Depth layer from nose (0 = nose, higher = toward tail).
        lateral_offset: Lateral offset from ship centerline in meters.
            Positive = starboard, negative = port, 0 = centerline.
    """
    layer_index: int
    lateral_offset: float = 0.0

    def distance_from_center(self) -> float:
        """
        Calculate distance from the ship's central axis.

        Returns:
            Absolute lateral distance in meters.
        """
        return abs(self.lateral_offset)


@dataclass
class Module:
    """
    Represents an internal ship module with health and damage mechanics.

    Attributes:
        name: Display name of the module.
        module_type: Type of module (sensor, bridge, reactor, etc.).
        health_percent: Current health from 0 to 100.
        armor_rating: Internal armor/shielding protection factor (0.0 to 1.0).
        position: Position within the ship (layer and lateral offset).
        size_m2: Cross-sectional area in square meters.
        is_critical: Whether destruction causes catastrophic ship damage.
    """
    name: str
    module_type: ModuleType
    health_percent: float = 100.0
    armor_rating: float = 0.0
    position: ModulePosition = field(default_factory=lambda: ModulePosition(0))
    size_m2: float = 10.0
    is_critical: bool = False

    def __post_init__(self) -> None:
        """Set is_critical based on module type if not explicitly set."""
        if self.module_type in CRITICAL_MODULE_TYPES:
            self.is_critical = True

    @property
    def is_destroyed(self) -> bool:
        """Check if the module has been destroyed."""
        return self.health_percent <= 0.0

    @property
    def is_functional(self) -> bool:
        """Check if the module is still functional (health > 25%)."""
        return self.health_percent > 25.0

    @property
    def effectiveness(self) -> float:
        """
        Calculate module effectiveness based on health.

        Returns:
            Effectiveness multiplier from 0.0 to 1.0.
        """
        if self.health_percent <= 0:
            return 0.0
        if self.health_percent >= 100:
            return 1.0
        # Linear degradation below 100%
        return self.health_percent / 100.0

    def damage(self, energy_gj: float) -> float:
        """
        Apply damage to this module.

        Args:
            energy_gj: Incoming damage energy in gigajoules.

        Returns:
            Remaining energy that passes through to the next layer (GJ).
        """
        if energy_gj <= 0 or self.is_destroyed:
            return energy_gj

        # Internal armor absorbs some damage
        absorbed_energy = energy_gj * self.armor_rating
        penetrating_energy = energy_gj - absorbed_energy

        # Calculate health damage (1 GJ = approximately 10% health damage for standard modules)
        # This scales with module size - larger modules can absorb more damage
        base_damage_per_gj = 10.0 / (self.size_m2 / 10.0)  # Normalized to 10 m2 standard
        health_damage = penetrating_energy * base_damage_per_gj

        # Apply damage
        old_health = self.health_percent
        self.health_percent = max(0.0, self.health_percent - health_damage)

        # Calculate energy absorbed by health loss
        actual_health_lost = old_health - self.health_percent
        energy_absorbed_by_module = actual_health_lost / base_damage_per_gj

        # Remaining energy passes through (diminished by absorption)
        remaining_energy = max(0.0, penetrating_energy - energy_absorbed_by_module)

        return remaining_energy

    def repair(self, amount_percent: float) -> None:
        """
        Repair the module by a given percentage.

        Args:
            amount_percent: Amount to repair (0 to 100).
        """
        self.health_percent = min(100.0, self.health_percent + amount_percent)

    def __str__(self) -> str:
        status = "DESTROYED" if self.is_destroyed else f"{self.health_percent:.0f}%"
        crit_marker = " [CRITICAL]" if self.is_critical else ""
        return f"{self.name} ({self.module_type.value}): {status}{crit_marker}"


@dataclass
class ModuleLayer:
    """
    A layer of modules at a specific depth in the ship.

    Attributes:
        layer_index: Depth index from nose (0 = nose).
        modules: List of modules at this layer.
        depth_m: Physical depth of this layer in meters.
    """
    layer_index: int
    modules: list[Module] = field(default_factory=list)
    depth_m: float = 10.0

    def add_module(self, module: Module) -> None:
        """Add a module to this layer."""
        module.position = ModulePosition(self.layer_index, module.position.lateral_offset)
        self.modules.append(module)

    def get_modules_at_offset(self, lateral_offset: float, tolerance: float = 5.0) -> list[Module]:
        """
        Get modules near a specific lateral offset.

        Args:
            lateral_offset: Target lateral position in meters.
            tolerance: How far from target to include modules.

        Returns:
            List of modules within tolerance of the lateral offset.
        """
        return [
            m for m in self.modules
            if abs(m.position.lateral_offset - lateral_offset) <= tolerance
        ]

    def get_centerline_modules(self) -> list[Module]:
        """Get modules on or near the ship's centerline."""
        return self.get_modules_at_offset(0.0)

    @property
    def total_cross_section_m2(self) -> float:
        """Calculate total cross-sectional area of all modules in this layer."""
        return sum(m.size_m2 for m in self.modules)


class ModuleLayout:
    """
    Complete module layout for a ship, organized by layers from nose to tail.

    This class manages the internal structure of a ship, handling damage
    propagation through layers and providing methods to query module positions.
    """

    def __init__(self, ship_type: str = "unknown", ship_length_m: float = 100.0):
        """
        Initialize an empty module layout.

        Args:
            ship_type: Type identifier for this ship class.
            ship_length_m: Total ship length in meters.
        """
        self.ship_type = ship_type
        self.ship_length_m = ship_length_m
        self.layers: list[ModuleLayer] = []
        self._module_cache: dict[str, Module] = {}

    def add_layer(self, layer: ModuleLayer) -> None:
        """
        Add a layer to the ship layout.

        Args:
            layer: The module layer to add.
        """
        self.layers.append(layer)
        # Update cache with new modules
        for module in layer.modules:
            self._module_cache[module.name] = module

    def get_module_by_name(self, name: str) -> Optional[Module]:
        """
        Find a module by its name.

        Args:
            name: The module name to search for.

        Returns:
            The module if found, None otherwise.
        """
        return self._module_cache.get(name)

    def get_modules_at_layer(self, layer_index: int) -> list[Module]:
        """
        Get all modules at a specific layer depth.

        Args:
            layer_index: The layer index (0 = nose).

        Returns:
            List of modules at that layer, empty if layer doesn't exist.
        """
        if 0 <= layer_index < len(self.layers):
            return list(self.layers[layer_index].modules)
        return []

    def get_all_modules(self) -> list[Module]:
        """
        Get all modules in the ship.

        Returns:
            List of all modules across all layers.
        """
        return list(self._module_cache.values())

    def get_critical_modules(self) -> list[Module]:
        """
        Get all critical modules (reactor, bridge).

        Returns:
            List of critical modules.
        """
        return [m for m in self._module_cache.values() if m.is_critical]

    def get_modules_by_type(self, module_type: ModuleType) -> list[Module]:
        """
        Get all modules of a specific type.

        Args:
            module_type: The type of modules to find.

        Returns:
            List of matching modules.
        """
        return [m for m in self._module_cache.values() if m.module_type == module_type]

    def get_modules_in_cone(
        self,
        entry_point: HitLocation,
        angle_deg: float,
        direction_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> list[Module]:
        """
        Get modules that would be hit by a damage cone from an entry point.

        This simulates projectile fragmentation or explosion damage spreading
        through the ship structure.

        Args:
            entry_point: Where the damage enters (nose, lateral, tail).
            angle_deg: Cone half-angle in degrees (spread of damage).
            direction_vector: Normalized direction of damage propagation (x, y, z).
                Default is (0, 0, 1) = nose-to-tail direction.

        Returns:
            List of modules in the damage cone, ordered by distance from entry.
        """
        affected_modules: list[tuple[float, Module]] = []

        # Convert angle to radians for calculations
        angle_rad = math.radians(angle_deg)
        cone_spread_factor = math.tan(angle_rad)

        # Determine starting layer and direction based on entry point
        if entry_point == HitLocation.NOSE:
            layer_range = range(len(self.layers))
            base_lateral = 0.0
        elif entry_point == HitLocation.TAIL:
            layer_range = range(len(self.layers) - 1, -1, -1)
            base_lateral = 0.0
        else:  # LATERAL
            layer_range = range(len(self.layers))
            # Lateral hits affect modules based on their lateral position
            base_lateral = direction_vector[0] * 10.0  # Scale by direction

        # Calculate which modules are in the cone
        for i, layer_idx in enumerate(layer_range):
            layer = self.layers[layer_idx]
            distance_from_entry = (i + 1) * layer.depth_m

            # Cone widens with distance
            cone_radius = distance_from_entry * cone_spread_factor

            for module in layer.modules:
                # Check if module is within the cone
                if entry_point == HitLocation.LATERAL:
                    # For lateral hits, check if module's lateral position is near the hit
                    lateral_distance = abs(module.position.lateral_offset - base_lateral)
                    if lateral_distance <= cone_radius + (module.size_m2 ** 0.5):
                        affected_modules.append((distance_from_entry, module))
                else:
                    # For nose/tail hits, all centerline and nearby modules can be hit
                    lateral_distance = module.position.distance_from_center()
                    if lateral_distance <= cone_radius + (module.size_m2 ** 0.5):
                        affected_modules.append((distance_from_entry, module))

        # Sort by distance and return just the modules
        affected_modules.sort(key=lambda x: x[0])
        return [m for _, m in affected_modules]

    def apply_penetrating_damage(
        self,
        hit_result: HitResult,
        spread_angle_deg: float = 15.0
    ) -> list[tuple[Module, float]]:
        """
        Apply penetrating damage from a HitResult to internal modules.

        Args:
            hit_result: The combat hit result with penetration information.
            spread_angle_deg: Damage spread angle in degrees.

        Returns:
            List of (module, damage_dealt) tuples for affected modules.
        """
        if not hit_result.penetrated or hit_result.remaining_damage_gj <= 0:
            return []

        if hit_result.location is None:
            return []

        # Get modules in the damage cone
        affected_modules = self.get_modules_in_cone(
            hit_result.location,
            spread_angle_deg
        )

        damage_results: list[tuple[Module, float]] = []
        remaining_energy = hit_result.remaining_damage_gj

        for module in affected_modules:
            if remaining_energy <= 0:
                break

            old_health = module.health_percent
            remaining_energy = module.damage(remaining_energy)
            damage_dealt = old_health - module.health_percent

            if damage_dealt > 0:
                damage_results.append((module, damage_dealt))

        return damage_results

    @property
    def total_layers(self) -> int:
        """Get the total number of layers in the ship."""
        return len(self.layers)

    @property
    def ship_integrity_percent(self) -> float:
        """
        Calculate overall ship structural integrity based on module health.

        Returns:
            Overall integrity percentage (0-100).
        """
        if not self._module_cache:
            return 100.0

        total_health = sum(m.health_percent for m in self._module_cache.values())
        return total_health / len(self._module_cache)

    @property
    def has_critical_damage(self) -> bool:
        """Check if any critical modules are destroyed."""
        return any(m.is_destroyed for m in self.get_critical_modules())

    def __str__(self) -> str:
        lines = [f"ModuleLayout: {self.ship_type} ({self.total_layers} layers)"]
        for layer in self.layers:
            lines.append(f"  Layer {layer.layer_index}:")
            for module in layer.modules:
                lines.append(f"    - {module}")
        return "\n".join(lines)

    @classmethod
    def from_ship_type(cls, ship_type: str, fleet_data: dict) -> ModuleLayout:
        """
        Create a module layout from ship type using fleet data.

        This factory method creates an appropriate default layout based on
        the ship class, positioning critical modules in protected locations.

        Args:
            ship_type: Ship type identifier (e.g., 'corvette', 'destroyer').
            fleet_data: The loaded fleet data dictionary.

        Returns:
            A configured ModuleLayout for the ship type.

        Raises:
            KeyError: If ship_type is not found in fleet_data.
        """
        ships = fleet_data.get("ships", {})
        if ship_type not in ships:
            raise KeyError(f"Ship type '{ship_type}' not found in fleet data")

        ship_data = ships[ship_type]
        hull_data = ship_data.get("hull", {})
        ship_length = hull_data.get("length_m", 100.0)
        crew_count = hull_data.get("crew", 20)

        # Determine layout based on ship class
        layout_factory = _SHIP_LAYOUT_FACTORIES.get(
            ship_type,
            _create_default_layout
        )

        return layout_factory(ship_type, ship_data, ship_length, crew_count)


def _create_corvette_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create a 4-layer corvette layout.

    Corvette layout (nose to tail):
    - Layer 0: Sensors
    - Layer 1: Bridge (critical, protected)
    - Layer 2: Reactor (critical, protected)
    - Layer 3: Engine
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 4

    # Layer 0: Sensors (nose)
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Primary Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.1,
        position=ModulePosition(0, 0.0),
        size_m2=15.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Bridge (protected by sensors in front)
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        armor_rating=0.3,
        position=ModulePosition(1, 0.0),
        size_m2=20.0,
        is_critical=True
    ))
    layer1.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.1,
        position=ModulePosition(1, 3.0),
        size_m2=15.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Reactor (center protected)
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Main Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.4,
        position=ModulePosition(2, 0.0),
        size_m2=30.0,
        is_critical=True
    ))
    layout.add_layer(layer2)

    # Layer 3: Engine (tail)
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.2,
        position=ModulePosition(3, 0.0),
        size_m2=40.0
    ))
    layer3.add_module(Module(
        name="Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(3, 4.0),
        size_m2=20.0
    ))
    layout.add_layer(layer3)

    return layout


def _create_destroyer_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create a 6-layer destroyer layout.

    Destroyer layout (nose to tail):
    - Layer 0: Sensors
    - Layer 1: Forward Weapons
    - Layer 2: Bridge (critical, center protected)
    - Layer 3: Reactor (critical, center protected)
    - Layer 4: Fuel Storage
    - Layer 5: Engine
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 6

    # Layer 0: Sensors
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Primary Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.1,
        position=ModulePosition(0, 0.0),
        size_m2=20.0
    ))
    layer0.add_module(Module(
        name="Targeting Computer",
        module_type=ModuleType.SENSOR,
        armor_rating=0.15,
        position=ModulePosition(0, 3.0),
        size_m2=10.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Forward Weapons
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Spinal Coiler Mount",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(1, 0.0),
        size_m2=35.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Bridge (protected by weapons and sensors)
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        armor_rating=0.35,
        position=ModulePosition(2, 0.0),
        size_m2=25.0,
        is_critical=True
    ))
    layer2.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.1,
        position=ModulePosition(2, 5.0),
        size_m2=25.0
    ))
    layer2.add_module(Module(
        name="Crew Quarters Port",
        module_type=ModuleType.CREW,
        armor_rating=0.1,
        position=ModulePosition(2, -5.0),
        size_m2=25.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Reactor (center protected)
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Main Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.45,
        position=ModulePosition(3, 0.0),
        size_m2=50.0,
        is_critical=True
    ))
    layer3.add_module(Module(
        name="Auxiliary Power",
        module_type=ModuleType.REACTOR,
        armor_rating=0.3,
        position=ModulePosition(3, 6.0),
        size_m2=15.0,
        is_critical=False  # Auxiliary is not critical
    ))
    layout.add_layer(layer3)

    # Layer 4: Fuel Storage
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Main Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.2,
        position=ModulePosition(4, 0.0),
        size_m2=40.0
    ))
    layer4.add_module(Module(
        name="Reserve Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(4, 5.0),
        size_m2=20.0
    ))
    layout.add_layer(layer4)

    # Layer 5: Engine
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.25,
        position=ModulePosition(5, 0.0),
        size_m2=60.0
    ))
    layout.add_layer(layer5)

    return layout


def _create_dreadnought_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create a 10-layer dreadnought layout.

    Dreadnought layout (nose to tail):
    - Layer 0: Forward Sensors
    - Layer 1: Spinal Weapon
    - Layer 2: Forward Weapons Bay
    - Layer 3: Crew/Bridge (critical, heavily protected)
    - Layer 4: Main Reactor (critical, heavily protected)
    - Layer 5: Ammunition Storage
    - Layer 6: Cargo Bay
    - Layer 7: Secondary Weapons
    - Layer 8: Fuel Storage
    - Layer 9: Engine Room
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 10

    # Layer 0: Forward Sensors
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Long Range Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.15,
        position=ModulePosition(0, 0.0),
        size_m2=30.0
    ))
    layer0.add_module(Module(
        name="Fire Control Radar",
        module_type=ModuleType.SENSOR,
        armor_rating=0.2,
        position=ModulePosition(0, 8.0),
        size_m2=15.0
    ))
    layer0.add_module(Module(
        name="Fire Control Radar Port",
        module_type=ModuleType.SENSOR,
        armor_rating=0.2,
        position=ModulePosition(0, -8.0),
        size_m2=15.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Spinal Weapon
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Spinal Coiler Mount",
        module_type=ModuleType.WEAPON,
        armor_rating=0.3,
        position=ModulePosition(1, 0.0),
        size_m2=60.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Forward Weapons Bay
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Heavy Coilgun Battery A",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(2, 6.0),
        size_m2=35.0
    ))
    layer2.add_module(Module(
        name="Heavy Coilgun Battery B",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(2, -6.0),
        size_m2=35.0
    ))
    layer2.add_module(Module(
        name="Magazine Forward",
        module_type=ModuleType.CARGO,
        armor_rating=0.2,
        position=ModulePosition(2, 0.0),
        size_m2=25.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Bridge (heavily protected by forward layers)
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        armor_rating=0.5,
        position=ModulePosition(3, 0.0),
        size_m2=40.0,
        is_critical=True
    ))
    layer3.add_module(Module(
        name="Combat Information Center",
        module_type=ModuleType.SENSOR,
        armor_rating=0.4,
        position=ModulePosition(3, 4.0),
        size_m2=20.0
    ))
    layout.add_layer(layer3)

    # Layer 4: Main Reactor (center protected)
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Main Fusion Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.55,
        position=ModulePosition(4, 0.0),
        size_m2=80.0,
        is_critical=True
    ))
    layer4.add_module(Module(
        name="Secondary Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.4,
        position=ModulePosition(4, 10.0),
        size_m2=30.0,
        is_critical=False
    ))
    layout.add_layer(layer4)

    # Layer 5: Ammunition Storage
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Main Magazine",
        module_type=ModuleType.CARGO,
        armor_rating=0.3,
        position=ModulePosition(5, 0.0),
        size_m2=50.0
    ))
    layer5.add_module(Module(
        name="Missile Magazine",
        module_type=ModuleType.CARGO,
        armor_rating=0.25,
        position=ModulePosition(5, 8.0),
        size_m2=30.0
    ))
    layout.add_layer(layer5)

    # Layer 6: Cargo and Crew
    layer6 = ModuleLayer(6, depth_m=layer_depth)
    layer6.add_module(Module(
        name="Main Cargo Bay",
        module_type=ModuleType.CARGO,
        armor_rating=0.15,
        position=ModulePosition(6, 0.0),
        size_m2=60.0
    ))
    layer6.add_module(Module(
        name="Crew Quarters A",
        module_type=ModuleType.CREW,
        armor_rating=0.15,
        position=ModulePosition(6, 8.0),
        size_m2=40.0
    ))
    layer6.add_module(Module(
        name="Crew Quarters B",
        module_type=ModuleType.CREW,
        armor_rating=0.15,
        position=ModulePosition(6, -8.0),
        size_m2=40.0
    ))
    layout.add_layer(layer6)

    # Layer 7: Secondary Weapons
    layer7 = ModuleLayer(7, depth_m=layer_depth)
    layer7.add_module(Module(
        name="Heavy Coilgun Battery C",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(7, 6.0),
        size_m2=35.0
    ))
    layer7.add_module(Module(
        name="Heavy Coilgun Battery D",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(7, -6.0),
        size_m2=35.0
    ))
    layer7.add_module(Module(
        name="Point Defense Array",
        module_type=ModuleType.WEAPON,
        armor_rating=0.2,
        position=ModulePosition(7, 0.0),
        size_m2=20.0
    ))
    layout.add_layer(layer7)

    # Layer 8: Fuel Storage
    layer8 = ModuleLayer(8, depth_m=layer_depth)
    layer8.add_module(Module(
        name="Main Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.2,
        position=ModulePosition(8, 0.0),
        size_m2=70.0
    ))
    layer8.add_module(Module(
        name="Reserve Fuel Tank A",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(8, 10.0),
        size_m2=30.0
    ))
    layer8.add_module(Module(
        name="Reserve Fuel Tank B",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(8, -10.0),
        size_m2=30.0
    ))
    layout.add_layer(layer8)

    # Layer 9: Engine Room
    layer9 = ModuleLayer(9, depth_m=layer_depth)
    layer9.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.3,
        position=ModulePosition(9, 0.0),
        size_m2=100.0
    ))
    layer9.add_module(Module(
        name="Maneuvering Thrusters",
        module_type=ModuleType.ENGINE,
        armor_rating=0.2,
        position=ModulePosition(9, 12.0),
        size_m2=25.0
    ))
    layout.add_layer(layer9)

    return layout


def _create_default_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create a default layout for unspecified ship types.

    Default is a 6-layer layout similar to a destroyer.
    """
    return _create_destroyer_layout(ship_type, ship_data, ship_length, crew_count)


def _create_frigate_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create a 5-layer frigate layout.

    Frigate layout (nose to tail):
    - Layer 0: Sensors
    - Layer 1: Bridge (critical)
    - Layer 2: Reactor (critical) + Weapons
    - Layer 3: Fuel Storage
    - Layer 4: Engine
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 5

    # Layer 0: Sensors
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Primary Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.1,
        position=ModulePosition(0, 0.0),
        size_m2=18.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Bridge
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        armor_rating=0.3,
        position=ModulePosition(1, 0.0),
        size_m2=22.0,
        is_critical=True
    ))
    layer1.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.1,
        position=ModulePosition(1, 4.0),
        size_m2=20.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Reactor + Weapons
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Main Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.4,
        position=ModulePosition(2, 0.0),
        size_m2=35.0,
        is_critical=True
    ))
    layer2.add_module(Module(
        name="Coilgun Battery",
        module_type=ModuleType.WEAPON,
        armor_rating=0.2,
        position=ModulePosition(2, 5.0),
        size_m2=20.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Fuel Storage
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(3, 0.0),
        size_m2=30.0
    ))
    layout.add_layer(layer3)

    # Layer 4: Engine
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.2,
        position=ModulePosition(4, 0.0),
        size_m2=45.0
    ))
    layout.add_layer(layer4)

    return layout


def _create_cruiser_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create an 8-layer cruiser layout.

    Cruiser layout (nose to tail):
    - Layer 0: Forward Sensors
    - Layer 1: Spinal Weapon
    - Layer 2: Bridge (critical)
    - Layer 3: Reactor (critical)
    - Layer 4: Weapons Bay
    - Layer 5: Cargo/Crew
    - Layer 6: Fuel Storage
    - Layer 7: Engine
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 8

    # Layer 0: Sensors
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Primary Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.15,
        position=ModulePosition(0, 0.0),
        size_m2=25.0
    ))
    layer0.add_module(Module(
        name="Targeting Sensors",
        module_type=ModuleType.SENSOR,
        armor_rating=0.15,
        position=ModulePosition(0, 5.0),
        size_m2=12.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Spinal Weapon
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Spinal Coiler Mount",
        module_type=ModuleType.WEAPON,
        armor_rating=0.28,
        position=ModulePosition(1, 0.0),
        size_m2=45.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Bridge
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        armor_rating=0.4,
        position=ModulePosition(2, 0.0),
        size_m2=30.0,
        is_critical=True
    ))
    layer2.add_module(Module(
        name="Officer Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.15,
        position=ModulePosition(2, 6.0),
        size_m2=25.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Reactor
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Main Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.5,
        position=ModulePosition(3, 0.0),
        size_m2=60.0,
        is_critical=True
    ))
    layout.add_layer(layer3)

    # Layer 4: Weapons Bay
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Coilgun Battery A",
        module_type=ModuleType.WEAPON,
        armor_rating=0.22,
        position=ModulePosition(4, 5.0),
        size_m2=25.0
    ))
    layer4.add_module(Module(
        name="Coilgun Battery B",
        module_type=ModuleType.WEAPON,
        armor_rating=0.22,
        position=ModulePosition(4, -5.0),
        size_m2=25.0
    ))
    layout.add_layer(layer4)

    # Layer 5: Cargo/Crew
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Cargo Bay",
        module_type=ModuleType.CARGO,
        armor_rating=0.1,
        position=ModulePosition(5, 0.0),
        size_m2=35.0
    ))
    layer5.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.12,
        position=ModulePosition(5, 7.0),
        size_m2=35.0
    ))
    layout.add_layer(layer5)

    # Layer 6: Fuel Storage
    layer6 = ModuleLayer(6, depth_m=layer_depth)
    layer6.add_module(Module(
        name="Main Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.18,
        position=ModulePosition(6, 0.0),
        size_m2=50.0
    ))
    layout.add_layer(layer6)

    # Layer 7: Engine
    layer7 = ModuleLayer(7, depth_m=layer_depth)
    layer7.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.25,
        position=ModulePosition(7, 0.0),
        size_m2=70.0
    ))
    layout.add_layer(layer7)

    return layout


def _create_battlecruiser_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """Create an 8-layer battlecruiser layout (similar to cruiser with more weapons)."""
    # Use cruiser as base with modifications for more weapon focus
    layout = _create_cruiser_layout(ship_type, ship_data, ship_length, crew_count)
    return layout


def _create_battleship_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create a 9-layer battleship layout.

    Similar to dreadnought but slightly smaller.
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 9

    # Layer 0: Sensors
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Long Range Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.15,
        position=ModulePosition(0, 0.0),
        size_m2=28.0
    ))
    layer0.add_module(Module(
        name="Fire Control Radar",
        module_type=ModuleType.SENSOR,
        armor_rating=0.18,
        position=ModulePosition(0, 6.0),
        size_m2=14.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Spinal Weapon
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Spinal Coiler Mount",
        module_type=ModuleType.WEAPON,
        armor_rating=0.3,
        position=ModulePosition(1, 0.0),
        size_m2=55.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Forward Weapons
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Heavy Coilgun Battery A",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(2, 5.0),
        size_m2=32.0
    ))
    layer2.add_module(Module(
        name="Heavy Coilgun Battery B",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(2, -5.0),
        size_m2=32.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Bridge
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        armor_rating=0.48,
        position=ModulePosition(3, 0.0),
        size_m2=38.0,
        is_critical=True
    ))
    layer3.add_module(Module(
        name="Combat Information Center",
        module_type=ModuleType.SENSOR,
        armor_rating=0.35,
        position=ModulePosition(3, 5.0),
        size_m2=18.0
    ))
    layout.add_layer(layer3)

    # Layer 4: Reactor
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Main Fusion Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.52,
        position=ModulePosition(4, 0.0),
        size_m2=70.0,
        is_critical=True
    ))
    layout.add_layer(layer4)

    # Layer 5: Ammunition/Cargo
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Main Magazine",
        module_type=ModuleType.CARGO,
        armor_rating=0.28,
        position=ModulePosition(5, 0.0),
        size_m2=45.0
    ))
    layer5.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.15,
        position=ModulePosition(5, 8.0),
        size_m2=35.0
    ))
    layout.add_layer(layer5)

    # Layer 6: Secondary Weapons
    layer6 = ModuleLayer(6, depth_m=layer_depth)
    layer6.add_module(Module(
        name="Heavy Coilgun Battery C",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(6, 5.0),
        size_m2=32.0
    ))
    layer6.add_module(Module(
        name="Point Defense Array",
        module_type=ModuleType.WEAPON,
        armor_rating=0.2,
        position=ModulePosition(6, 0.0),
        size_m2=18.0
    ))
    layout.add_layer(layer6)

    # Layer 7: Fuel Storage
    layer7 = ModuleLayer(7, depth_m=layer_depth)
    layer7.add_module(Module(
        name="Main Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.2,
        position=ModulePosition(7, 0.0),
        size_m2=60.0
    ))
    layer7.add_module(Module(
        name="Reserve Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(7, 8.0),
        size_m2=25.0
    ))
    layout.add_layer(layer7)

    # Layer 8: Engine
    layer8 = ModuleLayer(8, depth_m=layer_depth)
    layer8.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.28,
        position=ModulePosition(8, 0.0),
        size_m2=85.0
    ))
    layout.add_layer(layer8)

    return layout


# Factory mapping for ship types
_SHIP_LAYOUT_FACTORIES = {
    "corvette": _create_corvette_layout,
    "frigate": _create_frigate_layout,
    "destroyer": _create_destroyer_layout,
    "cruiser": _create_cruiser_layout,
    "battlecruiser": _create_battlecruiser_layout,
    "battleship": _create_battleship_layout,
    "dreadnought": _create_dreadnought_layout,
}


def load_fleet_data(filepath: str | Path) -> dict:
    """
    Load fleet ship data from a JSON file.

    Args:
        filepath: Path to the fleet_ships.json file.

    Returns:
        Dictionary containing all fleet data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(filepath, "r") as f:
        return json.load(f)


def create_module_layout(ship_type: str, fleet_data_path: str | Path) -> ModuleLayout:
    """
    Convenience function to create a module layout for a ship type.

    Args:
        ship_type: Ship type identifier (e.g., 'destroyer', 'dreadnought').
        fleet_data_path: Path to the fleet data JSON file.

    Returns:
        A configured ModuleLayout for the ship type.
    """
    fleet_data = load_fleet_data(fleet_data_path)
    return ModuleLayout.from_ship_type(ship_type, fleet_data)


if __name__ == "__main__":
    # Example usage and validation
    import sys

    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"

    if not data_path.exists():
        print(f"Fleet data not found at {data_path}")
        sys.exit(1)

    print("AI Commanders Module Layout System")
    print("=" * 50)

    fleet_data = load_fleet_data(data_path)

    # Create and display layouts for different ship types
    for ship_type in ["corvette", "destroyer", "dreadnought"]:
        print(f"\n{ship_type.upper()} Layout:")
        print("-" * 40)

        layout = ModuleLayout.from_ship_type(ship_type, fleet_data)
        print(f"Total Layers: {layout.total_layers}")
        print(f"Ship Length: {layout.ship_length_m} m")
        print(f"Critical Modules: {[m.name for m in layout.get_critical_modules()]}")

        print("\nLayers:")
        for layer in layout.layers:
            modules_str = ", ".join(m.name for m in layer.modules)
            print(f"  Layer {layer.layer_index}: {modules_str}")

    # Demonstrate damage propagation
    print("\n" + "=" * 50)
    print("Damage Propagation Test (Destroyer)")
    print("-" * 40)

    destroyer_layout = ModuleLayout.from_ship_type("destroyer", fleet_data)
    print(f"Initial integrity: {destroyer_layout.ship_integrity_percent:.1f}%")

    # Simulate a nose hit with penetrating damage
    from .combat import HitResult, HitLocation

    test_hit = HitResult(
        hit=True,
        location=HitLocation.NOSE,
        penetrated=True,
        remaining_damage_gj=5.0
    )

    damage_results = destroyer_layout.apply_penetrating_damage(test_hit, spread_angle_deg=20.0)

    print(f"\nSimulated nose penetration with 5.0 GJ:")
    for module, damage in damage_results:
        print(f"  {module.name}: -{damage:.1f}% health (now {module.health_percent:.1f}%)")

    print(f"\nFinal integrity: {destroyer_layout.ship_integrity_percent:.1f}%")
    print(f"Critical damage: {destroyer_layout.has_critical_damage}")
