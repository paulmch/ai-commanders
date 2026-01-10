"""
Damage Propagation System for AI Commanders Space Battle Simulator.

This module implements damage propagation through ships after armor penetration.
When a projectile penetrates armor, it creates a damage cone that propagates
through internal modules, damaging or destroying them based on energy and geometry.

Key concepts:
- DamageCone: Represents the path and spread of damage through the ship
- ModuleLayout: Spatial arrangement of modules within a ship hull
- DamagePropagator: Calculates module damage from penetrating hits

Integrates with combat.py when HitResult.penetrated is True.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .physics import Vector3D


# =============================================================================
# WEAPON CONE ANGLE CONSTANTS
# =============================================================================

class WeaponDamageProfile(Enum):
    """Damage profile categories for different weapon types."""
    KINETIC = "kinetic"      # Coilguns, railguns - tight cone, slow dissipation
    EXPLOSIVE = "explosive"  # Torpedoes, missiles - wide cone, fast dissipation
    LASER = "laser"          # Directed energy - very tight cone, no dissipation
    SPALLING = "spalling"    # Secondary cone from armor penetration


# Weapon cone angles in degrees
CONE_ANGLES_DEG: dict[WeaponDamageProfile, float] = {
    WeaponDamageProfile.KINETIC: 15.0,     # Tight cone - focused damage path
    WeaponDamageProfile.EXPLOSIVE: 60.0,   # Wide cone - blast fragmentation
    WeaponDamageProfile.LASER: 5.0,        # Very tight - coherent beam
    WeaponDamageProfile.SPALLING: 45.0,    # Secondary debris spread
}

# Energy dissipation rates (fraction of energy lost per meter)
# Lower values = slower dissipation = damage travels further
DISSIPATION_RATES: dict[WeaponDamageProfile, float] = {
    WeaponDamageProfile.KINETIC: 0.005,    # Slow dissipation - 0.5% per meter
    WeaponDamageProfile.EXPLOSIVE: 0.02,   # Fast dissipation - 2% per meter
    WeaponDamageProfile.LASER: 0.001,      # Very slow - 0.1% per meter
    WeaponDamageProfile.SPALLING: 0.03,    # Very fast - 3% per meter
}

# Spalling factor - multiplier for creating secondary damage cones
# when primary damage penetrates armor or destroys a module
SPALLING_ENERGY_FACTOR: float = 0.25  # Secondary cone gets 25% of remaining energy


# =============================================================================
# MODULE DAMAGE RESULT
# =============================================================================

@dataclass
class ModuleDamageResult:
    """
    Result of damage applied to a single module.

    Attributes:
        module_name: Name/identifier of the damaged module.
        damage_taken_gj: Amount of energy absorbed by this module (gigajoules).
        health_before: Module health before damage was applied.
        health_after: Module health after damage was applied.
        destroyed: Whether the module was destroyed by this damage.
        remaining_energy_gj: Energy remaining after passing through this module.
    """
    module_name: str
    damage_taken_gj: float
    health_before: float
    health_after: float
    destroyed: bool
    remaining_energy_gj: float

    def __str__(self) -> str:
        status = "DESTROYED" if self.destroyed else f"{self.health_after:.1f} HP remaining"
        return (
            f"{self.module_name}: {self.damage_taken_gj:.2f} GJ absorbed "
            f"({self.health_before:.1f} -> {self.health_after:.1f}), {status}"
        )


# =============================================================================
# DAMAGE CONE CLASS
# =============================================================================

@dataclass
class DamageCone:
    """
    Represents a cone of damage propagating through a ship's interior.

    When a projectile penetrates armor, it creates a conical damage zone
    that expands as it travels through the ship. The cone angle depends
    on weapon type (kinetic vs explosive), and energy dissipates with
    distance traveled.

    Coordinate system assumes the entry point is at the ship's surface
    and direction points inward toward the ship's interior.

    Attributes:
        entry_point: Vector3D position where projectile entered the ship (meters).
        direction: Vector3D unit vector indicating projectile travel direction.
        cone_angle_deg: Half-angle of the damage cone in degrees.
        initial_energy_gj: Starting energy of the damage cone (gigajoules).
        remaining_energy_gj: Current energy after propagation (gigajoules).
        damage_profile: The weapon damage profile for dissipation calculations.
        distance_traveled_m: Total distance the cone has propagated (meters).
    """
    entry_point: Vector3D
    direction: Vector3D
    cone_angle_deg: float
    initial_energy_gj: float
    remaining_energy_gj: float
    damage_profile: WeaponDamageProfile = WeaponDamageProfile.KINETIC
    distance_traveled_m: float = 0.0

    def __post_init__(self) -> None:
        """Normalize direction vector if not already normalized."""
        if self.direction.magnitude > 0:
            mag = self.direction.magnitude
            if abs(mag - 1.0) > 1e-6:
                # Normalize the direction
                normalized = self.direction.normalized()
                object.__setattr__(self, 'direction', normalized)

    @classmethod
    def from_weapon_type(
        cls,
        entry_point: Vector3D,
        direction: Vector3D,
        energy_gj: float,
        is_missile: bool = False,
        is_laser: bool = False,
    ) -> DamageCone:
        """
        Create a DamageCone with appropriate parameters for weapon type.

        Args:
            entry_point: Position where damage enters the ship.
            direction: Direction of damage propagation.
            energy_gj: Initial damage energy in gigajoules.
            is_missile: True if damage is from explosive/missile weapon.
            is_laser: True if damage is from directed energy weapon.

        Returns:
            Configured DamageCone instance.
        """
        if is_laser:
            profile = WeaponDamageProfile.LASER
        elif is_missile:
            profile = WeaponDamageProfile.EXPLOSIVE
        else:
            profile = WeaponDamageProfile.KINETIC

        cone_angle = CONE_ANGLES_DEG[profile]

        return cls(
            entry_point=entry_point,
            direction=direction.normalized(),
            cone_angle_deg=cone_angle,
            initial_energy_gj=energy_gj,
            remaining_energy_gj=energy_gj,
            damage_profile=profile,
        )

    @classmethod
    def create_spalling_cone(
        cls,
        origin: Vector3D,
        primary_direction: Vector3D,
        energy_gj: float,
    ) -> DamageCone:
        """
        Create a secondary spalling cone from armor penetration.

        Spalling occurs when armor is penetrated, creating secondary
        fragments that spread at a wider angle than the primary damage.

        Args:
            origin: Position where spalling originates.
            primary_direction: Direction of the original projectile.
            energy_gj: Energy available for spalling (typically 25% of remaining).

        Returns:
            Spalling DamageCone with wider angle and faster dissipation.
        """
        return cls(
            entry_point=origin,
            direction=primary_direction.normalized(),
            cone_angle_deg=CONE_ANGLES_DEG[WeaponDamageProfile.SPALLING],
            initial_energy_gj=energy_gj,
            remaining_energy_gj=energy_gj,
            damage_profile=WeaponDamageProfile.SPALLING,
        )

    @property
    def cone_angle_rad(self) -> float:
        """Cone half-angle in radians."""
        return math.radians(self.cone_angle_deg)

    @property
    def energy_fraction_remaining(self) -> float:
        """Fraction of initial energy still remaining."""
        if self.initial_energy_gj <= 0:
            return 0.0
        return self.remaining_energy_gj / self.initial_energy_gj

    @property
    def is_depleted(self) -> bool:
        """Check if the damage cone has dissipated below useful threshold."""
        return self.remaining_energy_gj < 0.01  # Less than 10 MJ remaining

    def is_in_cone(self, position: Vector3D) -> bool:
        """
        Check if a position falls within the damage cone.

        The cone originates at entry_point and expands in the direction
        of travel. A position is in the cone if:
        1. It is in front of the entry point (positive dot product with direction)
        2. The angle between the direction and position vector is less than cone_angle

        Args:
            position: 3D position to test.

        Returns:
            True if position is within the damage cone.
        """
        # Vector from entry point to test position
        to_position = position - self.entry_point

        # Distance along the cone axis
        distance_along_axis = to_position.dot(self.direction)

        # Must be in front of entry point
        if distance_along_axis <= 0:
            return False

        # Calculate angle between direction and position vector
        to_position_mag = to_position.magnitude
        if to_position_mag == 0:
            return True  # Position is at entry point

        cos_angle = distance_along_axis / to_position_mag
        # Clamp to avoid floating point errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)

        return angle <= self.cone_angle_rad

    def get_distance_to_point(self, position: Vector3D) -> float:
        """
        Calculate distance from entry point to a position along the cone axis.

        Args:
            position: 3D position to measure to.

        Returns:
            Distance in meters along the cone axis.
        """
        to_position = position - self.entry_point
        return to_position.dot(self.direction)

    def get_energy_at_distance(self, distance_m: float) -> float:
        """
        Calculate remaining energy at a given distance from entry point.

        Energy dissipates exponentially based on weapon type:
        E(d) = E_0 * exp(-dissipation_rate * d)

        Args:
            distance_m: Distance from entry point in meters.

        Returns:
            Energy remaining at that distance in gigajoules.
        """
        if distance_m <= 0:
            return self.remaining_energy_gj

        dissipation_rate = DISSIPATION_RATES[self.damage_profile]

        # Exponential decay
        energy = self.remaining_energy_gj * math.exp(-dissipation_rate * distance_m)

        return max(0.0, energy)

    def get_cone_radius_at_distance(self, distance_m: float) -> float:
        """
        Calculate the radius of the damage cone at a given distance.

        The cone expands linearly with distance based on the cone angle:
        radius = distance * tan(cone_angle)

        Args:
            distance_m: Distance from entry point in meters.

        Returns:
            Radius of the cone at that distance in meters.
        """
        if distance_m <= 0:
            return 0.0

        return distance_m * math.tan(self.cone_angle_rad)

    def calculate_hit_fraction(self, position: Vector3D, module_radius_m: float) -> float:
        """
        Calculate what fraction of damage cone energy hits a module.

        Based on the overlap between the module's cross-section and the
        cone's cross-section at that distance. Simplified as ratio of
        module area to cone area, capped at 1.0.

        Args:
            position: Center position of the module.
            module_radius_m: Approximate radius of the module.

        Returns:
            Fraction of cone energy hitting the module (0.0 to 1.0).
        """
        if not self.is_in_cone(position):
            return 0.0

        distance = self.get_distance_to_point(position)
        if distance <= 0:
            return 1.0

        cone_radius = self.get_cone_radius_at_distance(distance)
        if cone_radius <= 0:
            return 1.0

        # Simplified: ratio of areas (circular approximation)
        module_area = math.pi * module_radius_m ** 2
        cone_area = math.pi * cone_radius ** 2

        if cone_area <= 0:
            return 1.0

        return min(1.0, module_area / cone_area)

    def advance(self, distance_m: float) -> None:
        """
        Advance the damage cone by a distance, updating energy.

        This is called after the cone passes through or past a module
        to update the remaining energy based on distance traveled.

        Args:
            distance_m: Distance to advance in meters.
        """
        if distance_m <= 0:
            return

        self.remaining_energy_gj = self.get_energy_at_distance(distance_m)
        self.distance_traveled_m += distance_m

        # Move the entry point forward
        advance_vector = self.direction * distance_m
        self.entry_point = self.entry_point + advance_vector


# =============================================================================
# MODULE CLASS (for layout)
# =============================================================================

@dataclass
class Module:
    """
    Represents an internal ship module that can be damaged.

    Attributes:
        name: Unique identifier for this module.
        position: 3D position relative to ship center (meters).
        health: Current health points.
        max_health: Maximum health points.
        radius_m: Approximate radius for hit calculations (meters).
        is_critical: Whether destroying this module causes ship-wide effects.
        damage_resistance: Multiplier for incoming damage (0.0 to 1.0 reduces damage).
        is_destroyed: Whether the module has been destroyed.
    """
    name: str
    position: Vector3D
    health: float
    max_health: float
    radius_m: float = 2.0
    is_critical: bool = False
    damage_resistance: float = 0.0
    is_destroyed: bool = False

    @property
    def health_fraction(self) -> float:
        """Current health as fraction of maximum."""
        if self.max_health <= 0:
            return 0.0
        return self.health / self.max_health

    def take_damage(self, damage_gj: float) -> float:
        """
        Apply damage to this module.

        Args:
            damage_gj: Incoming damage in gigajoules.

        Returns:
            Actual damage absorbed by the module.
        """
        if self.is_destroyed:
            return 0.0

        # Apply damage resistance
        effective_damage = damage_gj * (1.0 - self.damage_resistance)

        # Calculate actual damage absorbed (can't absorb more than current health)
        absorbed = min(effective_damage, self.health)

        self.health -= absorbed

        if self.health <= 0:
            self.health = 0.0
            self.is_destroyed = True

        return absorbed


# =============================================================================
# MODULE LAYOUT CLASS
# =============================================================================

@dataclass
class ModuleLayout:
    """
    Spatial arrangement of modules within a ship hull.

    The layout defines where each module is located relative to the ship's
    center, allowing damage propagation to determine which modules are hit
    by a penetrating projectile.

    The coordinate system is:
    - X: Forward (positive toward nose)
    - Y: Right (positive toward starboard)
    - Z: Up (positive toward dorsal)

    Attributes:
        modules: List of modules in the layout.
        ship_length_m: Overall length of the ship (meters).
        ship_radius_m: Approximate radius of the ship hull (meters).
    """
    modules: list[Module] = field(default_factory=list)
    ship_length_m: float = 65.0  # Default corvette length
    ship_radius_m: float = 8.0   # Default approximate radius

    def add_module(self, module: Module) -> None:
        """Add a module to the layout."""
        self.modules.append(module)

    def get_module_by_name(self, name: str) -> Optional[Module]:
        """Find a module by name."""
        for module in self.modules:
            if module.name == name:
                return module
        return None

    def get_modules_in_cone(self, cone: DamageCone) -> list[Module]:
        """
        Get all modules that fall within a damage cone.

        Args:
            cone: The damage cone to test against.

        Returns:
            List of modules within the cone, sorted by distance from entry.
        """
        modules_in_cone = [
            module for module in self.modules
            if not module.is_destroyed and cone.is_in_cone(module.position)
        ]

        # Sort by distance from cone entry point (closest first)
        modules_in_cone.sort(
            key=lambda m: cone.get_distance_to_point(m.position)
        )

        return modules_in_cone

    def get_modules_at_layer(
        self,
        cone: DamageCone,
        layer_depth_m: float,
        layer_thickness_m: float = 5.0,
    ) -> list[Module]:
        """
        Get modules within a specific depth layer of the damage cone.

        Used for iterating through modules layer by layer.

        Args:
            cone: The damage cone.
            layer_depth_m: Distance from entry point to layer center.
            layer_thickness_m: Thickness of the layer.

        Returns:
            List of modules in this layer.
        """
        min_depth = layer_depth_m - layer_thickness_m / 2
        max_depth = layer_depth_m + layer_thickness_m / 2

        modules_in_layer = []
        for module in self.modules:
            if module.is_destroyed:
                continue
            if not cone.is_in_cone(module.position):
                continue

            distance = cone.get_distance_to_point(module.position)
            if min_depth <= distance <= max_depth:
                modules_in_layer.append(module)

        return modules_in_layer

    @classmethod
    def create_default_layout(cls, ship_length_m: float = 65.0) -> ModuleLayout:
        """
        Create a default module layout for testing.

        Creates a simple layout with typical ship modules arranged
        along the ship's length.

        Args:
            ship_length_m: Overall ship length in meters.

        Returns:
            ModuleLayout with default modules.
        """
        layout = cls(ship_length_m=ship_length_m, ship_radius_m=ship_length_m / 8)

        # Module positions are fractions of ship length from center
        # Positive X is forward (toward nose)
        module_specs = [
            # (name, x_fraction, y_offset, z_offset, health, radius, critical)
            ("Bridge", 0.35, 0, 1, 50, 3.0, True),
            ("Reactor", -0.1, 0, 0, 100, 4.0, True),
            ("Engine", -0.4, 0, 0, 80, 5.0, True),
            ("Magazine", 0.0, 2, 0, 60, 2.5, True),
            ("Heat_Sink", 0.0, -2, 0, 40, 2.0, False),
            ("Life_Support", 0.1, 0, 2, 30, 2.0, True),
            ("Sensors", 0.4, 0, 0.5, 25, 1.5, False),
            ("Battery", -0.2, 0, -1, 35, 2.0, False),
            ("Turret_Dorsal", 0.15, 0, 3, 45, 2.5, False),
            ("Turret_Ventral", 0.15, 0, -3, 45, 2.5, False),
        ]

        half_length = ship_length_m / 2

        for name, x_frac, y_off, z_off, health, radius, critical in module_specs:
            position = Vector3D(
                x_frac * half_length,
                y_off,
                z_off
            )
            module = Module(
                name=name,
                position=position,
                health=float(health),
                max_health=float(health),
                radius_m=radius,
                is_critical=critical,
            )
            layout.add_module(module)

        return layout


# =============================================================================
# DAMAGE PROPAGATOR CLASS
# =============================================================================

class DamagePropagator:
    """
    Propagates damage through a ship's module layout.

    When armor is penetrated, the DamagePropagator takes the resulting
    damage cone and calculates which modules are hit and how much damage
    each takes. It handles:

    - Energy distribution based on cone geometry
    - Energy dissipation with distance
    - Module destruction and continuing damage
    - Spalling (secondary damage cones) from destroyed modules

    Usage:
        propagator = DamagePropagator()
        results = propagator.propagate(damage_cone, module_layout)
    """

    def __init__(
        self,
        enable_spalling: bool = True,
        min_energy_threshold_gj: float = 0.01,
        layer_thickness_m: float = 5.0,
    ) -> None:
        """
        Initialize the damage propagator.

        Args:
            enable_spalling: Whether to generate secondary spalling cones.
            min_energy_threshold_gj: Minimum energy to continue propagation.
            layer_thickness_m: Thickness of each processing layer.
        """
        self.enable_spalling = enable_spalling
        self.min_energy_threshold_gj = min_energy_threshold_gj
        self.layer_thickness_m = layer_thickness_m

    def propagate(
        self,
        damage_cone: DamageCone,
        module_layout: ModuleLayout,
    ) -> list[ModuleDamageResult]:
        """
        Propagate damage through a module layout.

        Iterates through module layers from the entry point, calculating
        damage to each module in the cone's path. Continues until energy
        is depleted or the cone exits the ship.

        Args:
            damage_cone: The damage cone from armor penetration.
            module_layout: The ship's module layout.

        Returns:
            List of ModuleDamageResult for each affected module.
        """
        results: list[ModuleDamageResult] = []

        # Get all modules in the cone, sorted by distance
        modules_in_path = module_layout.get_modules_in_cone(damage_cone)

        if not modules_in_path:
            return results

        # Track current energy and position along the cone
        current_energy = damage_cone.remaining_energy_gj
        last_distance = 0.0

        for module in modules_in_path:
            # Check if we still have energy to deal damage
            if current_energy < self.min_energy_threshold_gj:
                break

            # Calculate distance to this module
            module_distance = damage_cone.get_distance_to_point(module.position)

            # Account for energy loss traveling to this module
            distance_traveled = module_distance - last_distance
            if distance_traveled > 0:
                current_energy = damage_cone.get_energy_at_distance(distance_traveled)

            if current_energy < self.min_energy_threshold_gj:
                break

            # Calculate what fraction of the cone hits this module
            hit_fraction = damage_cone.calculate_hit_fraction(
                module.position,
                module.radius_m
            )

            if hit_fraction <= 0:
                continue

            # Calculate damage to this module
            damage_to_module = current_energy * hit_fraction

            # Record state before damage
            health_before = module.health

            # Apply damage
            absorbed = module.take_damage(damage_to_module)

            # Reduce cone energy by absorbed amount
            current_energy -= absorbed

            # Calculate remaining energy after this module
            remaining = max(0.0, current_energy)

            # Create result
            result = ModuleDamageResult(
                module_name=module.name,
                damage_taken_gj=absorbed,
                health_before=health_before,
                health_after=module.health,
                destroyed=module.is_destroyed,
                remaining_energy_gj=remaining,
            )
            results.append(result)

            # Handle spalling from destroyed modules
            if module.is_destroyed and self.enable_spalling:
                spalling_results = self._handle_spalling(
                    module,
                    damage_cone,
                    module_layout,
                    current_energy,
                )
                results.extend(spalling_results)

            # Update tracking
            last_distance = module_distance

            # Stop if we've exited the ship
            if module_distance > module_layout.ship_length_m:
                break

        return results

    def _handle_spalling(
        self,
        destroyed_module: Module,
        original_cone: DamageCone,
        module_layout: ModuleLayout,
        remaining_energy: float,
    ) -> list[ModuleDamageResult]:
        """
        Generate and propagate secondary spalling damage.

        When a module is destroyed, it may create secondary fragments
        that damage nearby modules.

        Args:
            destroyed_module: The module that was just destroyed.
            original_cone: The original damage cone.
            module_layout: The ship's module layout.
            remaining_energy: Energy remaining after module destruction.

        Returns:
            List of ModuleDamageResult from spalling damage.
        """
        results: list[ModuleDamageResult] = []

        # Calculate spalling energy
        spalling_energy = remaining_energy * SPALLING_ENERGY_FACTOR

        if spalling_energy < self.min_energy_threshold_gj:
            return results

        # Create spalling cone from module position
        spalling_cone = DamageCone.create_spalling_cone(
            origin=destroyed_module.position,
            primary_direction=original_cone.direction,
            energy_gj=spalling_energy,
        )

        # Get modules that could be hit by spalling
        # Spalling doesn't propagate through the whole ship, just nearby modules
        for module in module_layout.modules:
            if module.is_destroyed or module.name == destroyed_module.name:
                continue

            # Check if module is in spalling cone and close enough
            distance = (module.position - destroyed_module.position).magnitude
            if distance > 15.0:  # Spalling effective range ~15 meters
                continue

            if not spalling_cone.is_in_cone(module.position):
                continue

            # Calculate spalling damage
            hit_fraction = spalling_cone.calculate_hit_fraction(
                module.position,
                module.radius_m
            )

            if hit_fraction <= 0:
                continue

            energy_at_module = spalling_cone.get_energy_at_distance(distance)
            spalling_damage = energy_at_module * hit_fraction

            if spalling_damage < self.min_energy_threshold_gj:
                continue

            health_before = module.health
            absorbed = module.take_damage(spalling_damage)

            result = ModuleDamageResult(
                module_name=f"{module.name} (spalling)",
                damage_taken_gj=absorbed,
                health_before=health_before,
                health_after=module.health,
                destroyed=module.is_destroyed,
                remaining_energy_gj=0.0,  # Spalling doesn't chain
            )
            results.append(result)

        return results

    def propagate_from_hit_result(
        self,
        hit_result,  # combat.HitResult - avoiding circular import
        entry_point: Vector3D,
        direction: Vector3D,
        module_layout: ModuleLayout,
        is_missile: bool = False,
    ) -> list[ModuleDamageResult]:
        """
        Propagate damage from a combat HitResult.

        Convenience method that creates a damage cone from a HitResult
        when penetration has occurred.

        Args:
            hit_result: HitResult from combat.py with penetrated=True.
            entry_point: Position where projectile entered the ship.
            direction: Direction of projectile travel.
            module_layout: The ship's module layout.
            is_missile: Whether the weapon was a missile/torpedo.

        Returns:
            List of ModuleDamageResult, empty if not penetrated.
        """
        if not hit_result.penetrated or hit_result.remaining_damage_gj <= 0:
            return []

        # Create damage cone from hit result
        damage_cone = DamageCone.from_weapon_type(
            entry_point=entry_point,
            direction=direction,
            energy_gj=hit_result.remaining_damage_gj,
            is_missile=is_missile,
        )

        return self.propagate(damage_cone, module_layout)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_entry_point_from_hit(
    ship_position: Vector3D,
    ship_heading: Vector3D,
    hit_location: str,
    ship_length_m: float = 65.0,
    ship_radius_m: float = 8.0,
) -> tuple[Vector3D, Vector3D]:
    """
    Calculate entry point and direction for a hit on a ship.

    Based on hit location (nose/lateral/tail), determines where on the
    ship's hull the projectile entered and its direction of travel.

    Args:
        ship_position: Ship's center position in world coordinates.
        ship_heading: Ship's forward direction (unit vector).
        hit_location: One of 'nose', 'lateral', 'tail'.
        ship_length_m: Ship length in meters.
        ship_radius_m: Ship radius in meters.

    Returns:
        Tuple of (entry_point, direction) in ship-local coordinates.
    """
    half_length = ship_length_m / 2

    if hit_location == "nose":
        # Hit from the front
        entry_point = Vector3D(half_length, 0, 0)
        direction = Vector3D(-1, 0, 0)  # Traveling backward into ship
    elif hit_location == "tail":
        # Hit from behind
        entry_point = Vector3D(-half_length, 0, 0)
        direction = Vector3D(1, 0, 0)  # Traveling forward into ship
    else:  # lateral
        # Hit from the side (simplified as starboard hit)
        entry_point = Vector3D(0, ship_radius_m, 0)
        direction = Vector3D(0, -1, 0)  # Traveling toward port side

    return entry_point, direction


def estimate_module_radius(mass_tons: float) -> float:
    """
    Estimate module radius from mass for hit calculations.

    Assumes roughly spherical module with density similar to
    spacecraft equipment (~500 kg/m^3).

    Args:
        mass_tons: Module mass in metric tons.

    Returns:
        Estimated radius in meters.
    """
    mass_kg = mass_tons * 1000
    density_kg_m3 = 500.0  # Spacecraft equipment density estimate

    # Volume = mass / density
    # For sphere: V = (4/3) * pi * r^3
    # r = (3V / 4pi)^(1/3)

    volume_m3 = mass_kg / density_kg_m3
    radius_m = (3 * volume_m3 / (4 * math.pi)) ** (1/3)

    return radius_m


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI COMMANDERS DAMAGE PROPAGATION SYSTEM - SELF TEST")
    print("=" * 70)

    # Create a default module layout
    layout = ModuleLayout.create_default_layout(ship_length_m=65.0)

    print(f"\nShip Layout ({layout.ship_length_m}m long):")
    print("-" * 40)
    for module in layout.modules:
        print(f"  {module.name}: pos={module.position}, HP={module.health}, "
              f"radius={module.radius_m}m, critical={module.is_critical}")

    # Test 1: Kinetic penetration from nose
    print("\n" + "=" * 70)
    print("TEST 1: Kinetic projectile hit to nose (15 GJ)")
    print("=" * 70)

    entry_point = Vector3D(32.5, 0, 0)  # Nose of 65m ship
    direction = Vector3D(-1, 0, 0)       # Traveling backward into ship

    kinetic_cone = DamageCone.from_weapon_type(
        entry_point=entry_point,
        direction=direction,
        energy_gj=15.0,
        is_missile=False,
    )

    print(f"Damage Cone:")
    print(f"  Profile: {kinetic_cone.damage_profile.value}")
    print(f"  Cone angle: {kinetic_cone.cone_angle_deg} deg")
    print(f"  Initial energy: {kinetic_cone.initial_energy_gj} GJ")

    propagator = DamagePropagator(enable_spalling=True)
    results = propagator.propagate(kinetic_cone, layout)

    print(f"\nDamage Results:")
    for result in results:
        print(f"  {result}")

    # Reset modules for next test
    for module in layout.modules:
        module.health = module.max_health
        module.is_destroyed = False

    # Test 2: Torpedo hit to lateral (wider cone, more damage)
    print("\n" + "=" * 70)
    print("TEST 2: Torpedo hit to lateral section (50 GJ)")
    print("=" * 70)

    entry_point = Vector3D(0, 8, 0)   # Starboard side
    direction = Vector3D(0, -1, 0)     # Traveling toward port

    torpedo_cone = DamageCone.from_weapon_type(
        entry_point=entry_point,
        direction=direction,
        energy_gj=50.0,
        is_missile=True,
    )

    print(f"Damage Cone:")
    print(f"  Profile: {torpedo_cone.damage_profile.value}")
    print(f"  Cone angle: {torpedo_cone.cone_angle_deg} deg")
    print(f"  Initial energy: {torpedo_cone.initial_energy_gj} GJ")

    results = propagator.propagate(torpedo_cone, layout)

    print(f"\nDamage Results:")
    for result in results:
        print(f"  {result}")

    # Test 3: Check cone geometry
    print("\n" + "=" * 70)
    print("TEST 3: Cone geometry checks")
    print("=" * 70)

    test_cone = DamageCone(
        entry_point=Vector3D(0, 0, 0),
        direction=Vector3D(1, 0, 0),
        cone_angle_deg=30.0,
        initial_energy_gj=10.0,
        remaining_energy_gj=10.0,
    )

    test_points = [
        Vector3D(10, 0, 0),    # On axis
        Vector3D(10, 5, 0),    # Off axis but in cone
        Vector3D(10, 10, 0),   # Outside cone
        Vector3D(-5, 0, 0),    # Behind entry point
    ]

    print(f"Testing cone with entry at origin, direction (+X), angle 30 deg:")
    for point in test_points:
        in_cone = test_cone.is_in_cone(point)
        energy = test_cone.get_energy_at_distance(point.x) if in_cone else 0
        print(f"  Point {point}: in_cone={in_cone}, energy={energy:.2f} GJ")

    # Test energy at various distances
    print(f"\nEnergy dissipation (kinetic profile, 10 GJ initial):")
    test_cone.damage_profile = WeaponDamageProfile.KINETIC
    for dist in [0, 10, 50, 100, 200]:
        energy = test_cone.get_energy_at_distance(dist)
        print(f"  At {dist}m: {energy:.3f} GJ ({energy/10*100:.1f}%)")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
