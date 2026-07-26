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
    MAGAZINE = "magazine"
    ARMOR = "armor"  # Armored bulkheads that protect critical modules
    HULL = "hull"  # Hull compartments - structural sections that absorb damage


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
    is_critical: bool | None = None  # None = auto-detect from module type

    def __post_init__(self) -> None:
        """Set is_critical based on module type if not explicitly set."""
        if self.is_critical is None:
            # Auto-detect based on module type
            self.is_critical = self.module_type in CRITICAL_MODULE_TYPES

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

        # Internal armor/shielding attenuates the energy that actually couples
        # into the module. Without this, armor_rating was dead data: the armored
        # bulkheads the layouts place around the bridge and reactor were exactly
        # as fragile as an unarmored cargo bay. The blocked fraction is not
        # destroyed - it keeps travelling to the next module, which mirrors
        # src.damage.Module.take_damage / damage_resistance.
        armor_rating = min(max(self.armor_rating, 0.0), 1.0)
        effective_energy = energy_gj * (1.0 - armor_rating)

        # Calculate health damage
        # 1 GJ can vaporize ~90cm of armor - it should wreck a module
        # 2 GJ destroys a module completely
        damage_per_gj = 50.0  # 50% per GJ - 2 GJ kills a module
        health_damage = effective_energy * damage_per_gj

        # Apply damage - module absorbs energy until destroyed
        old_health = self.health_percent
        self.health_percent = max(0.0, self.health_percent - health_damage)

        # Calculate energy absorbed (proportional to health lost)
        actual_health_lost = old_health - self.health_percent
        energy_absorbed = actual_health_lost / damage_per_gj

        # Remaining energy passes through to next module
        remaining_energy = max(0.0, energy_gj - energy_absorbed)

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
        direction_vector: tuple[float, float, float] = (0.0, 0.0, 1.0),
        ship_right: Optional[tuple[float, float, float]] = None,
        entry_layer_index: Optional[int] = None
    ) -> list[Module]:
        """
        Get modules that would be hit by a damage cone from an entry point.

        This simulates projectile fragmentation or explosion damage spreading
        through the ship structure. Damage is physically constrained:
        - Nose hits: Can reach from nose toward tail
        - Tail hits: Can reach from tail toward nose (engine first)
        - Lateral hits: Only affect middle sections (cannot reach nose/tail extremes)

        Args:
            entry_point: Where the damage enters (nose, lateral, tail).
            angle_deg: Cone half-angle in degrees (spread of damage).
            direction_vector: Direction of damage propagation (x, y, z), in the
                same (world) frame as ``ship_right``.
            ship_right: The ship's local starboard axis expressed in the same
                frame as ``direction_vector``. Supplying it lets a lateral hit
                resolve which SIDE the projectile came in on, so near-side
                modules are damaged before far-side ones. Without it the lateral
                cone is treated as side-agnostic (outer modules first, then
                inward) - which is symmetric and frame-independent rather than
                keyed off an arbitrary world axis.
            entry_layer_index: Axial layer the lateral hit entered at. Defaults
                to the middle of the laterally-reachable layers.

        Returns:
            List of modules in the damage cone, ordered by distance from entry.
        """
        num_layers = len(self.layers)

        # Convert angle to radians for calculations
        angle_rad = math.radians(angle_deg)
        cone_spread_factor = math.tan(angle_rad)

        if entry_point == HitLocation.LATERAL:
            return self._lateral_modules_in_cone(
                cone_spread_factor, direction_vector, ship_right, entry_layer_index
            )

        affected_modules: list[tuple[float, Module]] = []

        if entry_point == HitLocation.NOSE:
            # Nose hits travel from front to back
            layer_range = range(num_layers)
        else:  # TAIL
            # Tail hits travel from back to front - engine first, then forward
            layer_range = range(num_layers - 1, -1, -1)

        for i, layer_idx in enumerate(layer_range):
            layer = self.layers[layer_idx]
            distance_from_entry = (i + 1) * layer.depth_m

            # Cone widens with distance
            cone_radius = distance_from_entry * cone_spread_factor

            for module in layer.modules:
                # For nose/tail hits, all centerline and nearby modules can be hit
                lateral_distance = module.position.distance_from_center()
                if lateral_distance <= cone_radius + (module.size_m2 ** 0.5):
                    affected_modules.append((distance_from_entry, module))

        # Sort by depth into the ship (closer layers first)
        affected_modules.sort(key=lambda x: x[0])
        return [m for _, m in affected_modules]

    def _lateral_modules_in_cone(
        self,
        cone_spread_factor: float,
        direction_vector: tuple[float, float, float],
        ship_right: Optional[tuple[float, float, float]],
        entry_layer_index: Optional[int]
    ) -> list[Module]:
        """
        Resolve the module path for a hit entering through the ship's flank.

        The projectile crosses the hull at one axial station and then travels
        *laterally* through the ship, so the natural ordering is a 2-D path
        distance in the ship's own frame: how far it has burrowed sideways from
        the entry side, combined with how far the widening cone has spread fore
        and aft of the entry station.

        The previous implementation centred the cone on
        ``direction_vector[0] * 10.0`` - the WORLD-frame X component of the
        projectile velocity - and compared it against ship-local lateral offsets
        in metres. Two identical hits on an identical ship returned different
        module sets purely because the engagement happened along a different
        world axis (e.g. a +X shot hit only port modules and missed the bridge
        entirely, a +Z shot hit both sides). It also sorted by lateral offset
        alone, so the axial layer played no role whatsoever.
        """
        num_layers = len(self.layers)
        if num_layers == 0:
            return []

        # Lateral hits can only affect the middle ~50% of the ship: a projectile
        # entering from the side cannot physically reach the extreme nose
        # (spinal weapon) or the extreme tail (engine nozzles).
        skip_nose_layers = max(1, num_layers // 4)
        skip_tail_layers = max(1, num_layers // 4)
        start_layer = skip_nose_layers
        end_layer = num_layers - skip_tail_layers
        if end_layer <= start_layer:
            start_layer, end_layer = 0, num_layers

        # If the caller does not know WHERE along the flank the hit landed, we
        # must not invent a station: pinning it to the middle layer would mean
        # lateral fire could never reach the bridge or reactor of any ship. With
        # the station unknown every laterally-reachable layer stays a candidate
        # and ordering is purely by how deep the projectile has to burrow.
        entry_layer: Optional[int] = None
        if entry_layer_index is not None:
            entry_layer = min(max(entry_layer_index, start_layer), end_layer - 1)

        # Hull half-width, inferred from how far out the outermost module sits.
        half_width = 0.0
        for layer_idx in range(start_layer, end_layer):
            for module in self.layers[layer_idx].modules:
                half_width = max(half_width, abs(module.position.lateral_offset))

        # Which flank did it come in on? Project the impact direction onto the
        # ship's starboard axis: travelling toward starboard means it entered
        # through the port (negative-offset) side.
        entry_offset: Optional[float] = None
        if ship_right is not None and half_width > 0.0:
            projection = sum(d * r for d, r in zip(direction_vector, ship_right))
            if abs(projection) > 1e-6:
                entry_offset = -math.copysign(half_width, projection)

        affected_modules: list[tuple[float, Module]] = []
        for layer_idx in range(start_layer, end_layer):
            layer = self.layers[layer_idx]
            if entry_layer is None:
                axial_distance = 0.0
            else:
                axial_distance = abs(layer_idx - entry_layer) * layer.depth_m

            for module in layer.modules:
                if entry_offset is not None:
                    # Distance the projectile must burrow to reach this module
                    lateral_path = abs(module.position.lateral_offset - entry_offset)
                else:
                    # Side unknown: treat both flanks as equally near, so outer
                    # modules shield the centreline (which is what the armored
                    # bulkheads in the layouts are there to do).
                    lateral_path = half_width - abs(module.position.lateral_offset)
                lateral_path = max(lateral_path, 0.0)

                # The cone widens as it penetrates, so it reaches further
                # fore/aft the deeper into the ship it gets.
                cone_radius = lateral_path * cone_spread_factor
                if axial_distance <= cone_radius + (module.size_m2 ** 0.5):
                    path_distance = math.hypot(lateral_path, axial_distance)
                    affected_modules.append((path_distance, module))

        # Near-side modules at the entry station first; far-side and off-station
        # modules later.
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
    Create a 6-layer corvette layout with hull compartments.

    Corvette layout (nose to tail):
    - Layer 0: Sensors (nose)
    - Layer 1: Forward Hull (protects bridge from nose)
    - Layer 2: Bridge + Crew
    - Layer 3: Reactor
    - Layer 4: Aft Hull (protects reactor from tail)
    - Layer 5: Engine + Fuel
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 6

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

    # Layer 1: Forward Hull (protects bridge from nose hits)
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Forward Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.25,
        position=ModulePosition(1, 0.0),
        size_m2=25.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Bridge + Crew
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        armor_rating=0.3,
        position=ModulePosition(2, 0.0),
        size_m2=20.0,
        is_critical=True
    ))
    layer2.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.1,
        position=ModulePosition(2, 3.0),
        size_m2=15.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Reactor
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Main Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.4,
        position=ModulePosition(3, 0.0),
        size_m2=30.0,
        is_critical=True
    ))
    layout.add_layer(layer3)

    # Layer 4: Aft Hull (protects reactor from tail hits)
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Aft Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.25,
        position=ModulePosition(4, 0.0),
        size_m2=25.0
    ))
    layout.add_layer(layer4)

    # Layer 5: Engine + Fuel (tail)
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.2,
        position=ModulePosition(5, 0.0),
        size_m2=40.0
    ))
    layer5.add_module(Module(
        name="Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(5, 4.0),
        size_m2=20.0
    ))
    layout.add_layer(layer5)

    return layout


def _create_destroyer_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create an 8-layer destroyer layout with hull compartments protecting critical modules.

    Destroyer layout (nose to tail):
    - Layer 0: Spinal Weapon (nose-mounted main gun)
    - Layer 1: Forward Hull + Magazine (protects bridge from nose)
    - Layer 2: Bridge + Sensors + Lateral Armor
    - Layer 3: Central Hull (between bridge and reactor)
    - Layer 4: Reactor + Crew + Lateral Armor
    - Layer 5: Aft Hull (protects reactor from tail)
    - Layer 6: Fuel Storage + PD
    - Layer 7: Engine
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 8

    # Layer 0: Spinal Weapon (nose-mounted main gun)
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Spinal Coiler Mount",
        module_type=ModuleType.WEAPON,
        position=ModulePosition(0, 0.0),
        size_m2=35.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Forward Hull Section + Magazine (protects bridge from nose hits)
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Forward Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.3,
        position=ModulePosition(1, 0.0),
        size_m2=40.0
    ))
    layer1.add_module(Module(
        name="Main Magazine",
        module_type=ModuleType.MAGAZINE,
        position=ModulePosition(1, 5.0),
        size_m2=30.0
    ))
    layer1.add_module(Module(
        name="Dorsal Turret Mount",
        module_type=ModuleType.WEAPON,
        position=ModulePosition(1, -5.0),
        size_m2=15.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Bridge + Sensors (armored bulkheads on sides)
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
        name="Primary Sensor Array",
        module_type=ModuleType.SENSOR,
        position=ModulePosition(2, 4.0),
        size_m2=20.0
    ))
    layer2.add_module(Module(
        name="Targeting Computer",
        module_type=ModuleType.SENSOR,
        position=ModulePosition(2, -4.0),
        size_m2=10.0
    ))
    # Armored bulkheads protect bridge from lateral hits
    layer2.add_module(Module(
        name="Bridge Armored Bulkhead Port",
        module_type=ModuleType.ARMOR,
        armor_rating=0.5,
        position=ModulePosition(2, 8.0),
        size_m2=15.0
    ))
    layer2.add_module(Module(
        name="Bridge Armored Bulkhead Starboard",
        module_type=ModuleType.ARMOR,
        armor_rating=0.5,
        position=ModulePosition(2, -8.0),
        size_m2=15.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Central Hull Section (between bridge and reactor)
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Central Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.3,
        position=ModulePosition(3, 0.0),
        size_m2=45.0
    ))
    layer3.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        position=ModulePosition(3, 5.0),
        size_m2=25.0
    ))
    layout.add_layer(layer3)

    # Layer 4: Reactor (armored bulkheads on sides)
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Main Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.45,
        position=ModulePosition(4, 0.0),
        size_m2=50.0,
        is_critical=True
    ))
    # Armored bulkheads protect reactor from lateral hits
    layer4.add_module(Module(
        name="Reactor Armored Bulkhead Port",
        module_type=ModuleType.ARMOR,
        armor_rating=0.5,
        position=ModulePosition(4, 8.0),
        size_m2=18.0
    ))
    layer4.add_module(Module(
        name="Reactor Armored Bulkhead Starboard",
        module_type=ModuleType.ARMOR,
        armor_rating=0.5,
        position=ModulePosition(4, -8.0),
        size_m2=18.0
    ))
    layout.add_layer(layer4)

    # Layer 5: Aft Hull Section (protects reactor from tail hits)
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Aft Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.3,
        position=ModulePosition(5, 0.0),
        size_m2=40.0
    ))
    layer5.add_module(Module(
        name="Secondary Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.3,
        position=ModulePosition(5, 5.0),
        size_m2=20.0,
        is_critical=False
    ))
    layout.add_layer(layer5)

    # Layer 6: Fuel Storage + PD Lasers
    layer6 = ModuleLayer(6, depth_m=layer_depth)
    layer6.add_module(Module(
        name="Main Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        position=ModulePosition(6, 0.0),
        size_m2=40.0
    ))
    layer6.add_module(Module(
        name="Reserve Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        position=ModulePosition(6, -5.0),
        size_m2=20.0
    ))
    layer6.add_module(Module(
        name="PD Laser Dorsal",
        module_type=ModuleType.WEAPON,
        position=ModulePosition(6, 6.0),
        size_m2=8.0
    ))
    layer6.add_module(Module(
        name="PD Laser Ventral",
        module_type=ModuleType.WEAPON,
        position=ModulePosition(6, -6.0),
        size_m2=8.0
    ))
    layout.add_layer(layer6)

    # Layer 7: Engine
    layer7 = ModuleLayer(7, depth_m=layer_depth)
    layer7.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.25,
        position=ModulePosition(7, 0.0),
        size_m2=60.0
    ))
    layout.add_layer(layer7)

    return layout


def _create_dreadnought_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create a 12-layer dreadnought layout with hull compartments.

    Dreadnought layout (nose to tail):
    - Layer 0: Spinal Weapon (nose-mounted)
    - Layer 1: Forward Hull (protects bridge from nose)
    - Layer 2: Magazine + Forward Weapons Bay
    - Layer 3: Bridge + Sensors + Lateral Armor
    - Layer 4: Central Hull (between bridge and reactor)
    - Layer 5: Main Reactor + Lateral Armor
    - Layer 6: Aft Hull (protects reactor from tail)
    - Layer 7: Ammunition Storage + Cargo
    - Layer 8: Crew Quarters
    - Layer 9: Secondary Weapons
    - Layer 10: Fuel Storage
    - Layer 11: Engine Room
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 12

    # Layer 0: Spinal Weapon (nose-mounted main gun)
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Spinal Coiler Mount",
        module_type=ModuleType.WEAPON,
        armor_rating=0.3,
        position=ModulePosition(0, 0.0),
        size_m2=60.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Forward Hull (protects bridge from nose)
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Forward Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.45,
        position=ModulePosition(1, 0.0),
        size_m2=80.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Magazine + Forward Weapons Bay
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Magazine Forward",
        module_type=ModuleType.MAGAZINE,
        armor_rating=0.25,
        position=ModulePosition(2, 0.0),
        size_m2=35.0
    ))
    layer2.add_module(Module(
        name="Heavy Coilgun Battery A",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(2, 8.0),
        size_m2=35.0
    ))
    layer2.add_module(Module(
        name="Heavy Coilgun Battery B",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(2, -8.0),
        size_m2=35.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Bridge + Sensors (armored bulkheads on sides)
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
        name="Long Range Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.15,
        position=ModulePosition(3, 6.0),
        size_m2=30.0
    ))
    layer3.add_module(Module(
        name="Fire Control Radar",
        module_type=ModuleType.SENSOR,
        armor_rating=0.2,
        position=ModulePosition(3, -6.0),
        size_m2=15.0
    ))
    layer3.add_module(Module(
        name="Combat Information Center",
        module_type=ModuleType.SENSOR,
        armor_rating=0.4,
        position=ModulePosition(3, 4.0),
        size_m2=20.0
    ))
    layer3.add_module(Module(
        name="Bridge Armored Bulkhead Port",
        module_type=ModuleType.ARMOR,
        armor_rating=0.65,
        position=ModulePosition(3, 15.0),
        size_m2=30.0
    ))
    layer3.add_module(Module(
        name="Bridge Armored Bulkhead Starboard",
        module_type=ModuleType.ARMOR,
        armor_rating=0.65,
        position=ModulePosition(3, -15.0),
        size_m2=30.0
    ))
    layout.add_layer(layer3)

    # Layer 4: Central Hull (between bridge and reactor)
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Central Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.45,
        position=ModulePosition(4, 0.0),
        size_m2=85.0
    ))
    layout.add_layer(layer4)

    # Layer 5: Main Reactor (armored bulkheads on sides)
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Main Fusion Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.55,
        position=ModulePosition(5, 0.0),
        size_m2=80.0,
        is_critical=True
    ))
    layer5.add_module(Module(
        name="Secondary Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.4,
        position=ModulePosition(5, 8.0),
        size_m2=30.0,
        is_critical=False
    ))
    layer5.add_module(Module(
        name="Reactor Armored Bulkhead Port",
        module_type=ModuleType.ARMOR,
        armor_rating=0.65,
        position=ModulePosition(5, 15.0),
        size_m2=35.0
    ))
    layer5.add_module(Module(
        name="Reactor Armored Bulkhead Starboard",
        module_type=ModuleType.ARMOR,
        armor_rating=0.65,
        position=ModulePosition(5, -15.0),
        size_m2=35.0
    ))
    layout.add_layer(layer5)

    # Layer 6: Aft Hull (protects reactor from tail)
    layer6 = ModuleLayer(6, depth_m=layer_depth)
    layer6.add_module(Module(
        name="Aft Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.45,
        position=ModulePosition(6, 0.0),
        size_m2=80.0
    ))
    layout.add_layer(layer6)

    # Layer 7: Ammunition Storage + Cargo
    layer7 = ModuleLayer(7, depth_m=layer_depth)
    layer7.add_module(Module(
        name="Main Magazine",
        module_type=ModuleType.CARGO,
        armor_rating=0.3,
        position=ModulePosition(7, 0.0),
        size_m2=50.0
    ))
    layer7.add_module(Module(
        name="Missile Magazine",
        module_type=ModuleType.CARGO,
        armor_rating=0.25,
        position=ModulePosition(7, 8.0),
        size_m2=30.0
    ))
    layer7.add_module(Module(
        name="Main Cargo Bay",
        module_type=ModuleType.CARGO,
        armor_rating=0.15,
        position=ModulePosition(7, -8.0),
        size_m2=60.0
    ))
    layout.add_layer(layer7)

    # Layer 8: Crew Quarters
    layer8 = ModuleLayer(8, depth_m=layer_depth)
    layer8.add_module(Module(
        name="Crew Quarters A",
        module_type=ModuleType.CREW,
        armor_rating=0.15,
        position=ModulePosition(8, 5.0),
        size_m2=40.0
    ))
    layer8.add_module(Module(
        name="Crew Quarters B",
        module_type=ModuleType.CREW,
        armor_rating=0.15,
        position=ModulePosition(8, -5.0),
        size_m2=40.0
    ))
    layout.add_layer(layer8)

    # Layer 9: Secondary Weapons
    layer9 = ModuleLayer(9, depth_m=layer_depth)
    layer9.add_module(Module(
        name="Heavy Coilgun Battery C",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(9, 6.0),
        size_m2=35.0
    ))
    layer9.add_module(Module(
        name="Heavy Coilgun Battery D",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(9, -6.0),
        size_m2=35.0
    ))
    layer9.add_module(Module(
        name="Point Defense Array",
        module_type=ModuleType.WEAPON,
        armor_rating=0.2,
        position=ModulePosition(9, 0.0),
        size_m2=20.0
    ))
    layout.add_layer(layer9)

    # Layer 10: Fuel Storage
    layer10 = ModuleLayer(10, depth_m=layer_depth)
    layer10.add_module(Module(
        name="Main Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.2,
        position=ModulePosition(10, 0.0),
        size_m2=70.0
    ))
    layer10.add_module(Module(
        name="Reserve Fuel Tank A",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(10, 10.0),
        size_m2=30.0
    ))
    layer10.add_module(Module(
        name="Reserve Fuel Tank B",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(10, -10.0),
        size_m2=30.0
    ))
    layout.add_layer(layer10)

    # Layer 11: Engine Room
    layer11 = ModuleLayer(11, depth_m=layer_depth)
    layer11.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.3,
        position=ModulePosition(11, 0.0),
        size_m2=100.0
    ))
    layer11.add_module(Module(
        name="Maneuvering Thrusters",
        module_type=ModuleType.ENGINE,
        armor_rating=0.2,
        position=ModulePosition(11, 12.0),
        size_m2=25.0
    ))
    layout.add_layer(layer11)

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
    Create a 7-layer frigate layout with hull compartments.

    Frigate layout (nose to tail):
    - Layer 0: Sensors (nose)
    - Layer 1: Forward Hull (protects bridge from nose)
    - Layer 2: Bridge + Crew
    - Layer 3: Central Hull (between bridge and reactor)
    - Layer 4: Reactor + Weapons
    - Layer 5: Aft Hull (protects reactor from tail)
    - Layer 6: Engine + Fuel
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 7

    # Layer 0: Sensors (nose)
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Primary Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.1,
        position=ModulePosition(0, 0.0),
        size_m2=18.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Forward Hull (protects bridge from nose)
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Forward Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.25,
        position=ModulePosition(1, 0.0),
        size_m2=30.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Bridge + Crew
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        armor_rating=0.3,
        position=ModulePosition(2, 0.0),
        size_m2=22.0,
        is_critical=True
    ))
    layer2.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.1,
        position=ModulePosition(2, 4.0),
        size_m2=20.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Central Hull (between bridge and reactor)
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Central Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.25,
        position=ModulePosition(3, 0.0),
        size_m2=30.0
    ))
    layout.add_layer(layer3)

    # Layer 4: Reactor + Weapons
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Main Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.4,
        position=ModulePosition(4, 0.0),
        size_m2=35.0,
        is_critical=True
    ))
    layer4.add_module(Module(
        name="Coilgun Battery",
        module_type=ModuleType.WEAPON,
        armor_rating=0.2,
        position=ModulePosition(4, 5.0),
        size_m2=20.0
    ))
    layout.add_layer(layer4)

    # Layer 5: Aft Hull (protects reactor from tail)
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Aft Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.25,
        position=ModulePosition(5, 0.0),
        size_m2=30.0
    ))
    layer5.add_module(Module(
        name="Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(5, 4.0),
        size_m2=30.0
    ))
    layout.add_layer(layer5)

    # Layer 6: Engine
    layer6 = ModuleLayer(6, depth_m=layer_depth)
    layer6.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.2,
        position=ModulePosition(6, 0.0),
        size_m2=45.0
    ))
    layout.add_layer(layer6)

    return layout


def _create_cruiser_layout(
    ship_type: str,
    ship_data: dict,
    ship_length: float,
    crew_count: int
) -> ModuleLayout:
    """
    Create a 10-layer cruiser layout with hull compartments.

    Cruiser layout (nose to tail):
    - Layer 0: Spinal Weapon (nose-mounted)
    - Layer 1: Forward Hull (protects bridge from nose)
    - Layer 2: Magazine
    - Layer 3: Bridge + Sensors + Lateral Armor
    - Layer 4: Central Hull (between bridge and reactor)
    - Layer 5: Reactor + Lateral Armor
    - Layer 6: Aft Hull (protects reactor from tail)
    - Layer 7: Weapons Bay + Cargo
    - Layer 8: Crew + Fuel
    - Layer 9: Engine
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 10

    # Layer 0: Spinal Weapon (nose-mounted main gun)
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Spinal Coiler Mount",
        module_type=ModuleType.WEAPON,
        armor_rating=0.28,
        position=ModulePosition(0, 0.0),
        size_m2=45.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Forward Hull (protects bridge from nose)
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Forward Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.35,
        position=ModulePosition(1, 0.0),
        size_m2=50.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Magazine
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Main Magazine",
        module_type=ModuleType.MAGAZINE,
        armor_rating=0.25,
        position=ModulePosition(2, 0.0),
        size_m2=30.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Bridge + Sensors (armored bulkheads on sides)
    layer3 = ModuleLayer(3, depth_m=layer_depth)
    layer3.add_module(Module(
        name="Command Bridge",
        module_type=ModuleType.BRIDGE,
        armor_rating=0.4,
        position=ModulePosition(3, 0.0),
        size_m2=30.0,
        is_critical=True
    ))
    layer3.add_module(Module(
        name="Primary Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.15,
        position=ModulePosition(3, 4.0),
        size_m2=25.0
    ))
    layer3.add_module(Module(
        name="Targeting Sensors",
        module_type=ModuleType.SENSOR,
        armor_rating=0.15,
        position=ModulePosition(3, -4.0),
        size_m2=12.0
    ))
    layer3.add_module(Module(
        name="Bridge Armored Bulkhead Port",
        module_type=ModuleType.ARMOR,
        armor_rating=0.55,
        position=ModulePosition(3, 10.0),
        size_m2=20.0
    ))
    layer3.add_module(Module(
        name="Bridge Armored Bulkhead Starboard",
        module_type=ModuleType.ARMOR,
        armor_rating=0.55,
        position=ModulePosition(3, -10.0),
        size_m2=20.0
    ))
    layout.add_layer(layer3)

    # Layer 4: Central Hull (between bridge and reactor)
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Central Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.35,
        position=ModulePosition(4, 0.0),
        size_m2=55.0
    ))
    layout.add_layer(layer4)

    # Layer 5: Reactor (armored bulkheads on sides)
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Main Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.5,
        position=ModulePosition(5, 0.0),
        size_m2=60.0,
        is_critical=True
    ))
    layer5.add_module(Module(
        name="Reactor Armored Bulkhead Port",
        module_type=ModuleType.ARMOR,
        armor_rating=0.55,
        position=ModulePosition(5, 10.0),
        size_m2=22.0
    ))
    layer5.add_module(Module(
        name="Reactor Armored Bulkhead Starboard",
        module_type=ModuleType.ARMOR,
        armor_rating=0.55,
        position=ModulePosition(5, -10.0),
        size_m2=22.0
    ))
    layout.add_layer(layer5)

    # Layer 6: Aft Hull (protects reactor from tail)
    layer6 = ModuleLayer(6, depth_m=layer_depth)
    layer6.add_module(Module(
        name="Aft Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.35,
        position=ModulePosition(6, 0.0),
        size_m2=50.0
    ))
    layout.add_layer(layer6)

    # Layer 7: Weapons Bay + Cargo
    layer7 = ModuleLayer(7, depth_m=layer_depth)
    layer7.add_module(Module(
        name="Coilgun Battery A",
        module_type=ModuleType.WEAPON,
        armor_rating=0.22,
        position=ModulePosition(7, 5.0),
        size_m2=25.0
    ))
    layer7.add_module(Module(
        name="Coilgun Battery B",
        module_type=ModuleType.WEAPON,
        armor_rating=0.22,
        position=ModulePosition(7, -5.0),
        size_m2=25.0
    ))
    layer7.add_module(Module(
        name="Cargo Bay",
        module_type=ModuleType.CARGO,
        armor_rating=0.1,
        position=ModulePosition(7, 0.0),
        size_m2=35.0
    ))
    layout.add_layer(layer7)

    # Layer 8: Crew + Fuel
    layer8 = ModuleLayer(8, depth_m=layer_depth)
    layer8.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.12,
        position=ModulePosition(8, 5.0),
        size_m2=35.0
    ))
    layer8.add_module(Module(
        name="Main Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.18,
        position=ModulePosition(8, -5.0),
        size_m2=50.0
    ))
    layout.add_layer(layer8)

    # Layer 9: Engine
    layer9 = ModuleLayer(9, depth_m=layer_depth)
    layer9.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.25,
        position=ModulePosition(9, 0.0),
        size_m2=70.0
    ))
    layout.add_layer(layer9)

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
    Create an 11-layer battleship layout with hull compartments.

    Battleship layout (nose to tail):
    - Layer 0: Spinal Weapon (nose-mounted)
    - Layer 1: Forward Hull (protects bridge from nose)
    - Layer 2: Magazine + Forward Weapons
    - Layer 3: Bridge + Sensors + Lateral Armor
    - Layer 4: Central Hull (between bridge and reactor)
    - Layer 5: Reactor + Lateral Armor
    - Layer 6: Aft Hull (protects reactor from tail)
    - Layer 7: Secondary Weapons + Cargo
    - Layer 8: Crew Quarters
    - Layer 9: Fuel Storage
    - Layer 10: Engine
    """
    layout = ModuleLayout(ship_type, ship_length)
    layer_depth = ship_length / 11

    # Layer 0: Spinal Weapon (nose-mounted main gun)
    layer0 = ModuleLayer(0, depth_m=layer_depth)
    layer0.add_module(Module(
        name="Spinal Coiler Mount",
        module_type=ModuleType.WEAPON,
        armor_rating=0.3,
        position=ModulePosition(0, 0.0),
        size_m2=55.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Forward Hull (protects bridge from nose)
    layer1 = ModuleLayer(1, depth_m=layer_depth)
    layer1.add_module(Module(
        name="Forward Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.4,
        position=ModulePosition(1, 0.0),
        size_m2=65.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Magazine + Forward Weapons
    layer2 = ModuleLayer(2, depth_m=layer_depth)
    layer2.add_module(Module(
        name="Forward Magazine",
        module_type=ModuleType.MAGAZINE,
        armor_rating=0.28,
        position=ModulePosition(2, 0.0),
        size_m2=35.0
    ))
    layer2.add_module(Module(
        name="Heavy Coilgun Battery A",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(2, 6.0),
        size_m2=32.0
    ))
    layer2.add_module(Module(
        name="Heavy Coilgun Battery B",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(2, -6.0),
        size_m2=32.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Bridge + Sensors (armored bulkheads on sides)
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
        name="Long Range Sensor Array",
        module_type=ModuleType.SENSOR,
        armor_rating=0.15,
        position=ModulePosition(3, 5.0),
        size_m2=28.0
    ))
    layer3.add_module(Module(
        name="Fire Control Radar",
        module_type=ModuleType.SENSOR,
        armor_rating=0.18,
        position=ModulePosition(3, -5.0),
        size_m2=14.0
    ))
    layer3.add_module(Module(
        name="Bridge Armored Bulkhead Port",
        module_type=ModuleType.ARMOR,
        armor_rating=0.6,
        position=ModulePosition(3, 12.0),
        size_m2=25.0
    ))
    layer3.add_module(Module(
        name="Bridge Armored Bulkhead Starboard",
        module_type=ModuleType.ARMOR,
        armor_rating=0.6,
        position=ModulePosition(3, -12.0),
        size_m2=25.0
    ))
    layout.add_layer(layer3)

    # Layer 4: Central Hull (between bridge and reactor)
    layer4 = ModuleLayer(4, depth_m=layer_depth)
    layer4.add_module(Module(
        name="Central Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.4,
        position=ModulePosition(4, 0.0),
        size_m2=70.0
    ))
    layout.add_layer(layer4)

    # Layer 5: Reactor (armored bulkheads on sides)
    layer5 = ModuleLayer(5, depth_m=layer_depth)
    layer5.add_module(Module(
        name="Main Fusion Reactor",
        module_type=ModuleType.REACTOR,
        armor_rating=0.52,
        position=ModulePosition(5, 0.0),
        size_m2=70.0,
        is_critical=True
    ))
    layer5.add_module(Module(
        name="Reactor Armored Bulkhead Port",
        module_type=ModuleType.ARMOR,
        armor_rating=0.6,
        position=ModulePosition(5, 12.0),
        size_m2=28.0
    ))
    layer5.add_module(Module(
        name="Reactor Armored Bulkhead Starboard",
        module_type=ModuleType.ARMOR,
        armor_rating=0.6,
        position=ModulePosition(5, -12.0),
        size_m2=28.0
    ))
    layout.add_layer(layer5)

    # Layer 6: Aft Hull (protects reactor from tail)
    layer6 = ModuleLayer(6, depth_m=layer_depth)
    layer6.add_module(Module(
        name="Aft Hull Section",
        module_type=ModuleType.HULL,
        armor_rating=0.4,
        position=ModulePosition(6, 0.0),
        size_m2=65.0
    ))
    layout.add_layer(layer6)

    # Layer 7: Secondary Weapons + Cargo
    layer7 = ModuleLayer(7, depth_m=layer_depth)
    layer7.add_module(Module(
        name="Heavy Coilgun Battery C",
        module_type=ModuleType.WEAPON,
        armor_rating=0.25,
        position=ModulePosition(7, 5.0),
        size_m2=32.0
    ))
    layer7.add_module(Module(
        name="Point Defense Array",
        module_type=ModuleType.WEAPON,
        armor_rating=0.2,
        position=ModulePosition(7, -5.0),
        size_m2=18.0
    ))
    layer7.add_module(Module(
        name="Main Magazine",
        module_type=ModuleType.CARGO,
        armor_rating=0.28,
        position=ModulePosition(7, 0.0),
        size_m2=45.0
    ))
    layout.add_layer(layer7)

    # Layer 8: Crew Quarters
    layer8 = ModuleLayer(8, depth_m=layer_depth)
    layer8.add_module(Module(
        name="Crew Quarters",
        module_type=ModuleType.CREW,
        armor_rating=0.15,
        position=ModulePosition(8, 0.0),
        size_m2=45.0
    ))
    layout.add_layer(layer8)

    # Layer 9: Fuel Storage
    layer9 = ModuleLayer(9, depth_m=layer_depth)
    layer9.add_module(Module(
        name="Main Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.2,
        position=ModulePosition(9, 0.0),
        size_m2=60.0
    ))
    layer9.add_module(Module(
        name="Reserve Fuel Tank",
        module_type=ModuleType.FUEL_TANK,
        armor_rating=0.15,
        position=ModulePosition(9, 8.0),
        size_m2=25.0
    ))
    layout.add_layer(layer9)

    # Layer 10: Engine
    layer10 = ModuleLayer(10, depth_m=layer_depth)
    layer10.add_module(Module(
        name="Main Engine Assembly",
        module_type=ModuleType.ENGINE,
        armor_rating=0.28,
        position=ModulePosition(10, 0.0),
        size_m2=85.0
    ))
    layout.add_layer(layer10)

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
