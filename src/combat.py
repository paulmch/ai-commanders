"""
Combat mechanics module for the AI Commanders space battle simulator.

This module implements weapon systems, armor mechanics, damage calculations,
and hit resolution for space combat simulations.

Based on Terra Invicta game mechanics with baryonic projectile physics.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


# Radiator vulnerability constants
RADIATOR_HIT_CHANCE_RETRACTED = 0.05  # 5% chance to hit retracted radiators
RADIATOR_HIT_CHANCE_EXTENDED = 0.20   # 20% chance to hit extended radiators
RADIATOR_ARMOR_RATING = 0.1           # Very fragile, only 10% damage reduction


class HitLocation(Enum):
    """Ship hit locations with associated targeting probabilities."""
    NOSE = "nose"
    LATERAL = "lateral"
    TAIL = "tail"


class RadiatorPosition(Enum):
    """Radiator mounting positions on the ship hull."""
    PORT = "PORT"
    STARBOARD = "STARBOARD"
    DORSAL = "DORSAL"
    VENTRAL = "VENTRAL"


# Hit location probabilities based on ship geometry
HIT_LOCATION_WEIGHTS: dict[HitLocation, float] = {
    HitLocation.NOSE: 0.15,
    HitLocation.LATERAL: 0.70,
    HitLocation.TAIL: 0.15,
}


@runtime_checkable
class ThermalSystemProtocol(Protocol):
    """
    Protocol defining the interface for thermal systems that support radiator damage.

    This protocol allows the combat system to interact with thermal systems
    without creating circular dependencies.
    """

    def is_radiator_extended(self, position: RadiatorPosition) -> bool:
        """
        Check if a radiator at the given position is extended.

        Args:
            position: The radiator position to check.

        Returns:
            True if the radiator is extended, False if retracted.
        """
        ...

    def get_radiator_dissipation(self, position: RadiatorPosition) -> float:
        """
        Get the current dissipation capacity of a radiator in kW.

        Args:
            position: The radiator position to query.

        Returns:
            Current dissipation capacity in kW (0.0 if destroyed).
        """
        ...

    def apply_radiator_damage(
        self,
        position: RadiatorPosition,
        damage_gj: float
    ) -> tuple[float, bool, float]:
        """
        Apply damage to a radiator and return the results.

        Args:
            position: The radiator position to damage.
            damage_gj: Amount of damage in gigajoules.

        Returns:
            Tuple of (damage_taken_gj, destroyed, dissipation_lost_kw).
        """
        ...


@dataclass
class RadiatorHitResult:
    """
    Result of a radiator hit resolution.

    Attributes:
        hit: Whether the radiator was hit.
        radiator_position: Which radiator was hit (PORT/STARBOARD/DORSAL/VENTRAL).
        damage_taken_gj: Amount of damage absorbed by the radiator.
        radiator_destroyed: Whether the radiator was destroyed by this hit.
        dissipation_lost_kw: Heat dissipation capacity lost in kW.
    """
    hit: bool
    radiator_position: str = ""
    damage_taken_gj: float = 0.0
    radiator_destroyed: bool = False
    dissipation_lost_kw: float = 0.0

    def __str__(self) -> str:
        if not self.hit:
            return "Radiator miss"

        status = "DESTROYED" if self.radiator_destroyed else "damaged"
        return (
            f"Radiator hit ({self.radiator_position}): {self.damage_taken_gj:.2f} GJ "
            f"{status}, {self.dissipation_lost_kw:.1f} kW dissipation lost"
        )


@dataclass
class Weapon:
    """
    Represents a weapon system with its combat statistics.

    Attributes:
        name: Display name of the weapon.
        weapon_type: Internal type identifier (e.g., 'spinal_coiler_mk3').
        kinetic_energy_gj: Kinetic energy of projectile in gigajoules.
        cooldown_s: Time between shots in seconds.
        range_km: Maximum effective range in kilometers.
        flat_chipping: Armor ablation factor (0.0 to 1.0).
        mass_tons: Weapon system mass in metric tons.
        magazine: Number of rounds available.
        muzzle_velocity_kps: Projectile velocity in km/s (optional for missiles).
        warhead_mass_kg: Projectile mass in kilograms.
        mount: Mount type ('nose_spinal', 'hull_turret').
        is_missile: True if this is a missile/torpedo weapon.
        pivot_range_deg: Maximum traverse/gimbal angle in degrees.
        is_turreted: True if weapon has independent turret aiming.
        facing: Hemisphere the weapon covers ('forward' or 'rear').
    """
    name: str
    weapon_type: str
    kinetic_energy_gj: float
    cooldown_s: float
    range_km: float
    flat_chipping: float
    mass_tons: float = 0.0
    magazine: int = 0
    muzzle_velocity_kps: float = 0.0
    warhead_mass_kg: float = 0.0
    mount: str = "hull_turret"
    is_missile: bool = False
    pivot_range_deg: float = 180.0
    is_turreted: bool = True
    facing: str = "forward"

    @classmethod
    def from_json(cls, weapon_type: str, data: dict) -> Weapon:
        """
        Create a Weapon instance from JSON data.

        Args:
            weapon_type: The weapon type identifier.
            data: Dictionary containing weapon statistics.

        Returns:
            A configured Weapon instance.
        """
        # Missiles and kinetic penetrators are both self-propelled guided munitions
        weapon_type_value = data.get("type", "")
        is_missile = weapon_type_value in ("missile", "kinetic_penetrator")

        # Missiles use warhead_yield_gj, kinetic weapons use kinetic_energy_gj
        energy = data.get("kinetic_energy_gj", 0.0)
        if weapon_type_value == "missile":
            energy = data.get("warhead_yield_gj", 0.0)
        # kinetic_penetrator damage is calculated at impact, not stored here

        return cls(
            name=data.get("name", weapon_type),
            weapon_type=weapon_type,
            kinetic_energy_gj=energy,
            cooldown_s=data.get("cooldown_s", 0.0),
            range_km=data.get("range_km", 0.0),
            flat_chipping=data.get("flat_chipping", 0.0),
            mass_tons=data.get("mass_tons", 0.0),
            magazine=data.get("magazine", 0),
            muzzle_velocity_kps=data.get("muzzle_velocity_kps", 0.0),
            warhead_mass_kg=data.get("warhead_mass_kg", data.get("ammo_mass_kg", 0.0)),
            mount=data.get("mount", "hull_turret"),
            is_missile=is_missile,
            pivot_range_deg=data.get("pivot_range_deg", 180.0),
            is_turreted=data.get("is_turreted", True),
            facing=data.get("facing", "forward"),
        )

    def is_in_range(self, distance_km: float) -> bool:
        """
        Check if a target is within weapon range.

        Args:
            distance_km: Distance to target in kilometers.

        Returns:
            True if target is in range, False otherwise.
        """
        return distance_km <= self.range_km

    def can_fire(self, current_ammo: int) -> bool:
        """
        Check if the weapon can fire.

        Args:
            current_ammo: Current ammunition count.

        Returns:
            True if weapon has ammunition and can fire.
        """
        return current_ammo > 0


@dataclass
class Armor:
    """
    Represents a ship's armor section with protection and ablation mechanics.

    Uses Terra Invicta physics:
    - Kinetic damage = 50% of kinetic energy (the other 50% is lost to ejecta/heat)
    - Kinetics resistance reduces incoming kinetic damage
    - Chipping resistance reduces flat armor penetration per hit
    - Heat of vaporization determines how much armor is ablated per MJ

    Attributes:
        armor_type: Type of armor material (e.g., 'Titanium', 'Steel').
        thickness_cm: Current armor thickness in centimeters.
        baryonic_half_cm: Half-value thickness for protection calculation.
        chip_resist: Resistance to flat chipping (0.0-1.0, e.g., 0.75 = 75% reduction).
        kinetics_resist: Resistance to kinetic damage (0.0-1.0, e.g., 0.75 = 25% damage taken).
        density: Armor material density in kg/m^3.
        area_m2: Surface area of this armor section in m^2.
        location: Which part of the ship this armor protects.
        heat_of_vaporization_mj_kg: Energy to vaporize 1 kg of armor (MJ).
        chipping_fraction: Cumulative chipping damage (0.0 to 1.0).
    """
    armor_type: str
    thickness_cm: float
    baryonic_half_cm: float
    chip_resist: float
    kinetics_resist: float = 0.0  # KineticsResistance from Terra Invicta (0.75 = 25% damage)
    density: float = 4820.0  # Default titanium density
    area_m2: float = 100.0
    location: HitLocation = HitLocation.LATERAL
    heat_of_vaporization_mj_kg: float = 8.77  # Default titanium (from Terra Invicta)
    chipping_fraction: float = 0.0  # Accumulated chipping damage
    original_thickness_cm: float = field(default=0.0, init=False)  # Set in __post_init__

    def __post_init__(self):
        """Store the original thickness for damage percentage calculations."""
        self.original_thickness_cm = self.thickness_cm

    @property
    def current_thickness_cm(self) -> float:
        """Alias for thickness_cm for clarity."""
        return self.thickness_cm

    @property
    def damage_percent(self) -> float:
        """Percentage of armor that has been ablated."""
        if self.original_thickness_cm <= 0:
            return 0.0
        return 100.0 * (1.0 - self.thickness_cm / self.original_thickness_cm)

    @classmethod
    def from_json(
        cls,
        armor_data: dict,
        section_name: str,
        section_data: dict
    ) -> Armor:
        """
        Create an Armor instance from JSON ship data.

        Args:
            armor_data: The ship's armor configuration dict.
            section_name: Name of the section ('nose', 'lateral', 'tail').
            section_data: Section-specific armor statistics.

        Returns:
            A configured Armor instance.
        """
        properties = armor_data.get("properties", {})
        location = HitLocation(section_name)

        return cls(
            armor_type=armor_data.get("type", "Unknown"),
            thickness_cm=section_data.get("thickness_cm", 0.0),
            baryonic_half_cm=properties.get("baryonic_half_cm", 10.5),
            chip_resist=properties.get("chip_resist", 0.0),
            kinetics_resist=properties.get("kinetics_resist", 0.0),
            density=properties.get("density", 4820.0),
            area_m2=section_data.get("area_m2", 100.0),
            location=location,
            heat_of_vaporization_mj_kg=properties.get("heat_of_vaporization_mj_kg", 8.77),
        )

    def effective_thickness(self, impact_angle_deg: float = 0.0) -> float:
        """
        Calculate effective armor thickness based on impact angle.

        When a projectile hits at an oblique angle, it must travel through
        more armor material. This is the Line-Of-Sight (LOS) thickness.

        Formula: effective = actual / cos(angle)
        - 0° (perpendicular): effective = actual
        - 45°: effective = actual * 1.41
        - 60°: effective = actual * 2.0
        - 80°: effective = actual * 5.76

        Args:
            impact_angle_deg: Angle from surface normal in degrees (0-90).
                             0° = perpendicular hit, 90° = grazing hit.

        Returns:
            Effective thickness in centimeters.
        """
        import math
        # Clamp angle to valid range (0-89 degrees to avoid division by zero)
        angle_deg = max(0.0, min(89.0, abs(impact_angle_deg)))
        angle_rad = math.radians(angle_deg)
        cos_angle = math.cos(angle_rad)

        # Minimum cos to avoid extreme values (caps at ~5.76x multiplier at 80°)
        cos_angle = max(0.1736, cos_angle)  # cos(80°) ≈ 0.1736

        return self.thickness_cm / cos_angle

    @property
    def protection(self) -> float:
        """
        Calculate the protection factor based on armor thickness.

        Uses the formula: protection = 1 - 0.5^(thickness / half_value)

        Returns:
            Protection factor from 0.0 (no protection) to 1.0 (full protection).
        """
        return self.protection_at_angle(0.0)

    def protection_at_angle(self, impact_angle_deg: float = 0.0) -> float:
        """
        Calculate protection factor accounting for impact angle.

        Oblique hits must travel through more armor, increasing protection.

        Args:
            impact_angle_deg: Angle from surface normal (0° = perpendicular).

        Returns:
            Protection factor from 0.0 to 1.0.
        """
        effective = self.effective_thickness(impact_angle_deg)
        if effective <= 0 or self.baryonic_half_cm <= 0:
            return 0.0
        return 1.0 - (0.5 ** (effective / self.baryonic_half_cm))

    @property
    def protection_percent(self) -> float:
        """
        Get protection as a percentage.

        Returns:
            Protection factor as a percentage (0.0 to 100.0).
        """
        return self.protection * 100.0

    def calculate_ablation(self, weapon: Weapon, base_ablation_cm: float = 2.5) -> float:
        """
        Calculate armor ablation from a weapon hit (legacy method).

        Uses the formula: ablation = base * flat_chipping * (1 - chip_resist)

        Args:
            weapon: The weapon that hit this armor section.
            base_ablation_cm: Base ablation amount in cm (default 2.5).

        Returns:
            Armor thickness removed in centimeters.
        """
        return base_ablation_cm * weapon.flat_chipping * (1.0 - self.chip_resist)

    def calculate_energy_ablation(
        self,
        energy_gj: float,
        impact_area_m2: float = 0.01
    ) -> float:
        """
        Calculate armor ablation from kinetic energy using physics.

        Based on Terra Invicta: kinetic energy vaporizes armor material.
        Energy required = mass * heat_of_vaporization
        Mass = volume * density = (thickness * area) * density

        Solving for thickness ablated:
            thickness_m = energy_j / (density * area * heat_of_vaporization)
            thickness_cm = energy_gj * 1e9 / (density * area * heat_of_vaporization * 1e6) / 100

        Args:
            energy_gj: Kinetic energy in gigajoules.
            impact_area_m2: Area of impact in square meters (default 0.01 = 10cm x 10cm slug).

        Returns:
            Armor thickness that would be ablated in centimeters.
        """
        if self.heat_of_vaporization_mj_kg <= 0 or impact_area_m2 <= 0:
            return 0.0

        # Convert: energy_gj * 1e9 J / (kg/m³ * m² * MJ/kg * 1e6 J/MJ) = meters
        # Then multiply by 100 to get cm
        energy_j = energy_gj * 1e9
        heat_j_kg = self.heat_of_vaporization_mj_kg * 1e6
        thickness_m = energy_j / (self.density * impact_area_m2 * heat_j_kg)
        return thickness_m * 100  # Convert to cm

    def apply_damage(self, weapon: Weapon, base_ablation_cm: float = 2.5) -> float:
        """
        Apply weapon damage to this armor section (legacy method).

        Args:
            weapon: The weapon that hit this armor section.
            base_ablation_cm: Base ablation amount in cm.

        Returns:
            Actual thickness removed in centimeters.
        """
        ablation = self.calculate_ablation(weapon, base_ablation_cm)
        actual_ablation = min(ablation, self.thickness_cm)
        self.thickness_cm = max(0.0, self.thickness_cm - ablation)
        return actual_ablation

    def apply_energy_damage(
        self,
        energy_gj: float,
        flat_chipping: float = 0.3,
        impact_area_m2: float = 0.01
    ) -> tuple[float, float, float]:
        """
        Apply kinetic energy damage using Terra Invicta physics.

        Terra Invicta damage formula:
        1. Kinetic damage = 50% of kinetic energy (other 50% lost to ejecta/heat)
        2. Apply kinetics_resist: effective_damage = damage * (1 - kinetics_resist)
        3. Damage vaporizes armor based on heat_of_vaporization
        4. flat_chipping = guaranteed armor penetration per hit (reduced by chip_resist)

        Args:
            energy_gj: Kinetic energy in gigajoules.
            flat_chipping: Guaranteed armor penetration factor (0.0-0.8 typically).
            impact_area_m2: Area of impact in square meters.

        Returns:
            Tuple of (ablation_cm, energy_absorbed_by_hull_gj, chipping_added).
        """
        # Step 1: 50% of kinetic energy converts to damage (Terra Invicta formula)
        damage_gj = energy_gj * 0.5

        # Step 2: Apply kinetics resistance (e.g., 0.75 = only 25% damage taken)
        effective_damage_gj = damage_gj * (1.0 - self.kinetics_resist)

        # Step 3: Calculate ablation from effective damage
        base_ablation = self.calculate_energy_ablation(effective_damage_gj, impact_area_m2)

        # Step 4: Apply flat_chipping as guaranteed armor penetration
        # flat_chipping is reduced by chip_resist
        effective_chipping = flat_chipping * (1.0 - self.chip_resist)
        chipping_ablation = self.thickness_cm * effective_chipping * 0.1  # Scale factor

        # Total ablation is energy-based + chipping-based
        total_ablation = base_ablation + chipping_ablation

        # Apply ablation to armor
        actual_ablation = min(total_ablation, self.thickness_cm)
        self.thickness_cm = max(0.0, self.thickness_cm - total_ablation)

        # Energy absorbed by hull (damage that gets through armor absorption)
        # Armor protection reduces what reaches the hull
        protection = self.protection
        energy_to_hull_gj = effective_damage_gj * (1.0 - protection)

        # Update chipping fraction (creates weak spots for future critical hits)
        if self.thickness_cm > 0:
            chipping_added = actual_ablation / (self.thickness_cm + actual_ablation) * 0.1
        else:
            chipping_added = 0.5  # Penetrated armor has high chipping
        self.chipping_fraction = min(1.0, self.chipping_fraction + chipping_added)

        return actual_ablation, energy_to_hull_gj, chipping_added

    def roll_critical_through_chipping(self, rng: random.Random = None) -> bool:
        """
        Roll for a critical hit through chipped armor.

        From Terra Invicta: hits on chipped armor have a chance equal to
        the chipping percentage to bypass armor entirely.

        Args:
            rng: Random number generator for reproducibility.

        Returns:
            True if the hit penetrates through a chip in the armor.
        """
        if rng is None:
            rng = random.Random()
        return rng.random() < self.chipping_fraction

    def is_penetrated(self) -> bool:
        """
        Check if armor has been fully penetrated.

        Returns:
            True if armor thickness is zero or negative.
        """
        return self.thickness_cm <= 0


@dataclass
class HitResult:
    """
    Result of a combat hit resolution.

    Attributes:
        hit: Whether the attack hit the target.
        location: Where on the ship the hit landed.
        damage_absorbed: Amount of damage absorbed by armor.
        armor_ablation_cm: Armor thickness removed.
        penetrated: Whether the armor was penetrated.
        remaining_damage_gj: Damage that penetrated to internal systems.
        critical_hit: Whether a critical hit occurred.
        radiator_hit: Whether a radiator was hit during this attack.
        radiator_position: Position of the hit radiator (PORT/STARBOARD/DORSAL/VENTRAL).
        radiator_damage_gj: Amount of damage dealt to the radiator.
        impact_angle_deg: Angle from surface normal (0 = perpendicular).
        effective_armor_cm: Effective armor thickness after angle adjustment.
    """
    hit: bool
    location: Optional[HitLocation] = None
    damage_absorbed: float = 0.0
    armor_ablation_cm: float = 0.0
    penetrated: bool = False
    remaining_damage_gj: float = 0.0
    critical_hit: bool = False
    radiator_hit: bool = False
    radiator_position: Optional[str] = None
    radiator_damage_gj: float = 0.0
    impact_angle_deg: float = 0.0
    effective_armor_cm: float = 0.0

    def __str__(self) -> str:
        if not self.hit:
            return "Miss"

        loc_name = self.location.value if self.location else "unknown"
        status = "PENETRATED" if self.penetrated else "absorbed"

        result = (
            f"Hit ({loc_name}): {self.armor_ablation_cm:.1f} cm ablated, "
            f"{self.damage_absorbed:.2f} GJ {status}"
        )
        if self.penetrated:
            result += f", {self.remaining_damage_gj:.2f} GJ internal damage"
        if self.critical_hit:
            result += " [CRITICAL]"
        if self.radiator_hit:
            result += f" [RADIATOR HIT: {self.radiator_position}, {self.radiator_damage_gj:.2f} GJ]"

        return result


@dataclass
class ShipArmor:
    """
    Complete armor configuration for a ship with all sections.

    Attributes:
        sections: Dictionary mapping hit locations to Armor instances.
    """
    sections: dict[HitLocation, Armor] = field(default_factory=dict)

    @classmethod
    def from_json(cls, armor_data: dict) -> ShipArmor:
        """
        Create a ShipArmor from JSON ship armor configuration.

        Args:
            armor_data: The ship's armor configuration dict.

        Returns:
            A configured ShipArmor instance with all sections.
        """
        sections = {}
        section_data = armor_data.get("sections", {})

        for section_name in ["nose", "lateral", "tail"]:
            if section_name in section_data:
                armor = Armor.from_json(armor_data, section_name, section_data[section_name])
                sections[HitLocation(section_name)] = armor

        return cls(sections=sections)

    def get_section(self, location: HitLocation) -> Optional[Armor]:
        """
        Get the armor section for a specific location.

        Args:
            location: The hit location to retrieve.

        Returns:
            The Armor instance for that location, or None if not found.
        """
        return self.sections.get(location)

    def total_mass_tons(self) -> float:
        """
        Calculate total armor mass in tons.

        Returns:
            Total mass of all armor sections in metric tons.
        """
        total = 0.0
        for armor in self.sections.values():
            # Volume in m^3: area_m2 * thickness_cm / 100
            volume_m3 = armor.area_m2 * (armor.thickness_cm / 100.0)
            mass_kg = volume_m3 * armor.density
            total += mass_kg / 1000.0  # Convert to tons
        return total


class RadiatorHitResolver:
    """
    Resolves radiator hit checks during combat.

    Radiators are fragile components that extend from the ship hull to dissipate
    heat. When extended, they present a larger target. Droplet radiators are
    especially vulnerable due to their low armor rating.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        """
        Initialize the radiator hit resolver.

        Args:
            rng: Optional random number generator for reproducible results.
        """
        self.rng = rng or random.Random()

    def _determine_radiator_position(
        self,
        hit_location: HitLocation
    ) -> RadiatorPosition:
        """
        Determine which radiator could be hit based on the hit location.

        Lateral hits can strike PORT or STARBOARD radiators (50/50).
        Nose/tail hits can strike DORSAL or VENTRAL radiators (50/50).

        Args:
            hit_location: The hull location that was hit.

        Returns:
            The radiator position that could be affected.
        """
        if hit_location == HitLocation.LATERAL:
            # Lateral hits can hit port or starboard radiators
            return self.rng.choice([RadiatorPosition.PORT, RadiatorPosition.STARBOARD])
        else:
            # Nose or tail hits can hit dorsal or ventral radiators
            return self.rng.choice([RadiatorPosition.DORSAL, RadiatorPosition.VENTRAL])

    def resolve_radiator_hit(
        self,
        weapon: Weapon,
        thermal_system: ThermalSystemProtocol,
        hit_location: HitLocation
    ) -> Optional[RadiatorHitResult]:
        """
        Check for and resolve a radiator hit during combat.

        This method determines if a weapon hit also strikes a radiator based on
        the hit location and radiator deployment state. Radiators are fragile
        and easily damaged, with only minimal armor protection.

        Args:
            weapon: The weapon that scored the hit.
            thermal_system: The target ship's thermal management system.
            hit_location: Where on the hull the weapon hit.

        Returns:
            RadiatorHitResult if a radiator was targeted, None if no check needed.
            Note: The result may have hit=False if the radiator check failed.
        """
        # Determine which radiator could potentially be hit
        radiator_position = self._determine_radiator_position(hit_location)

        # Check if this radiator is extended or retracted
        is_extended = thermal_system.is_radiator_extended(radiator_position)

        # Determine hit chance based on radiator state
        hit_chance = (
            RADIATOR_HIT_CHANCE_EXTENDED if is_extended
            else RADIATOR_HIT_CHANCE_RETRACTED
        )

        # Roll for radiator hit
        if self.rng.random() > hit_chance:
            return RadiatorHitResult(
                hit=False,
                radiator_position=radiator_position.value
            )

        # Radiator was hit - calculate damage
        # Apply radiator armor rating (very fragile)
        effective_damage = weapon.kinetic_energy_gj * (1.0 - RADIATOR_ARMOR_RATING)

        # Apply damage to the thermal system and get results
        damage_taken, destroyed, dissipation_lost = thermal_system.apply_radiator_damage(
            radiator_position,
            effective_damage
        )

        return RadiatorHitResult(
            hit=True,
            radiator_position=radiator_position.value,
            damage_taken_gj=damage_taken,
            radiator_destroyed=destroyed,
            dissipation_lost_kw=dissipation_lost
        )


class CombatResolver:
    """
    Resolves combat interactions between weapons and armor.

    This class handles hit determination, damage calculation, armor ablation,
    and penetration checks for space combat encounters.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        """
        Initialize the combat resolver.

        Args:
            rng: Optional random number generator for reproducible results.
        """
        self.rng = rng or random.Random()
        self.radiator_resolver = RadiatorHitResolver(rng=self.rng)

    def determine_hit_location(self) -> HitLocation:
        """
        Randomly determine which part of the ship is hit.

        Uses weighted probabilities: nose 15%, lateral 70%, tail 15%.

        Returns:
            The hit location.
        """
        locations = list(HIT_LOCATION_WEIGHTS.keys())
        weights = list(HIT_LOCATION_WEIGHTS.values())
        return self.rng.choices(locations, weights=weights, k=1)[0]

    def calculate_hit_probability(
        self,
        weapon: Weapon,
        distance_km: float,
        target_accel_g: float = 1.0,
        tracking_modifier: float = 1.0
    ) -> float:
        """
        Calculate probability of hitting a target.

        Args:
            weapon: The weapon being fired.
            distance_km: Distance to target in kilometers.
            target_accel_g: Target's evasion acceleration in g's.
            tracking_modifier: Firing platform's tracking capability.

        Returns:
            Hit probability from 0.0 to 1.0.
        """
        if distance_km > weapon.range_km:
            return 0.0

        # Base hit chance decreases with range
        range_factor = 1.0 - (distance_km / weapon.range_km) ** 2

        # Higher velocity projectiles are harder to evade
        if weapon.muzzle_velocity_kps > 0:
            velocity_factor = min(1.0, weapon.muzzle_velocity_kps / 10.0)
        else:
            velocity_factor = 0.8  # Missiles have good tracking

        # Evasive targets are harder to hit
        evasion_factor = 1.0 / (1.0 + target_accel_g * 0.1)

        # Combine factors
        hit_prob = range_factor * velocity_factor * evasion_factor * tracking_modifier

        return max(0.0, min(1.0, hit_prob))

    def resolve_hit(
        self,
        weapon: Weapon,
        target_armor: ShipArmor,
        location: Optional[HitLocation] = None,
        base_ablation_cm: float = 2.5,
        thermal_system: Optional[ThermalSystemProtocol] = None,
        impact_angle_deg: float = 0.0
    ) -> HitResult:
        """
        Resolve a weapon hit against ship armor, including potential radiator damage.

        After armor resolution, this method checks for radiator hits if a thermal
        system is provided. Radiators are vulnerable external components that can
        be damaged even when the main armor absorbs the hit.

        The impact_angle_deg affects effective armor thickness - oblique hits
        must travel through more armor material, increasing protection.

        Args:
            weapon: The weapon that hit.
            target_armor: The target ship's armor.
            location: Specific hit location (random if None).
            base_ablation_cm: Base ablation amount in cm.
            thermal_system: Optional thermal system for radiator hit resolution.
            impact_angle_deg: Angle from surface normal (0° = perpendicular).
                             Affects effective armor thickness.

        Returns:
            HitResult with damage, penetration, and radiator hit details.
        """
        # Determine hit location
        if location is None:
            location = self.determine_hit_location()

        # Initialize radiator hit info
        radiator_hit = False
        radiator_position: Optional[str] = None
        radiator_damage_gj = 0.0

        armor = target_armor.get_section(location)
        if armor is None:
            # No armor at this location - full penetration
            # Still check for radiator hit even on unarmored sections
            if thermal_system is not None:
                radiator_result = self.radiator_resolver.resolve_radiator_hit(
                    weapon, thermal_system, location
                )
                if radiator_result and radiator_result.hit:
                    radiator_hit = True
                    radiator_position = radiator_result.radiator_position
                    radiator_damage_gj = radiator_result.damage_taken_gj

            return HitResult(
                hit=True,
                location=location,
                damage_absorbed=0.0,
                armor_ablation_cm=0.0,
                penetrated=True,
                remaining_damage_gj=weapon.kinetic_energy_gj,
                critical_hit=True,
                radiator_hit=radiator_hit,
                radiator_position=radiator_position,
                radiator_damage_gj=radiator_damage_gj,
            )

        # Calculate effective armor thickness at this angle
        effective_armor = armor.effective_thickness(impact_angle_deg)

        # Calculate damage absorption using angle-adjusted protection
        protection = armor.protection_at_angle(impact_angle_deg)
        damage_absorbed = weapon.kinetic_energy_gj * protection

        # Calculate armor ablation (reduced at oblique angles)
        # Oblique hits spread damage over larger area, reducing ablation
        import math
        angle_rad = math.radians(min(89.0, abs(impact_angle_deg)))
        ablation_factor = math.cos(angle_rad)  # Less ablation at steeper angles
        ablation = armor.calculate_ablation(weapon, base_ablation_cm) * ablation_factor

        # Apply damage to armor
        actual_ablation = min(ablation, armor.thickness_cm)
        armor.thickness_cm = max(0.0, armor.thickness_cm - ablation)

        # Check for penetration
        penetrated = armor.is_penetrated()
        if penetrated:
            # Armor breached - 90% of energy reaches hull
            remaining_damage = weapon.kinetic_energy_gj * 0.9
        else:
            # Armor intact - only bleed-through based on protection factor
            remaining_damage = weapon.kinetic_energy_gj * (1.0 - protection)

        # Critical hit check (10% base chance, higher if penetrated)
        crit_chance = 0.1 if not penetrated else 0.5
        critical_hit = penetrated and self.rng.random() < crit_chance

        # Check for radiator hit (happens after armor resolution)
        # Radiators can be hit regardless of armor penetration status
        if thermal_system is not None:
            radiator_result = self.radiator_resolver.resolve_radiator_hit(
                weapon, thermal_system, location
            )
            if radiator_result and radiator_result.hit:
                radiator_hit = True
                radiator_position = radiator_result.radiator_position
                radiator_damage_gj = radiator_result.damage_taken_gj

        return HitResult(
            hit=True,
            location=location,
            damage_absorbed=damage_absorbed,
            armor_ablation_cm=actual_ablation,
            penetrated=penetrated,
            remaining_damage_gj=remaining_damage if penetrated else 0.0,
            critical_hit=critical_hit,
            radiator_hit=radiator_hit,
            radiator_position=radiator_position,
            radiator_damage_gj=radiator_damage_gj,
            impact_angle_deg=impact_angle_deg,
            effective_armor_cm=effective_armor,
        )

    def resolve_attack(
        self,
        weapon: Weapon,
        target_armor: ShipArmor,
        distance_km: float,
        target_accel_g: float = 1.0,
        tracking_modifier: float = 1.0,
        base_ablation_cm: float = 2.5,
        thermal_system: Optional[ThermalSystemProtocol] = None
    ) -> HitResult:
        """
        Resolve a complete attack sequence from firing to damage.

        This method handles the full attack sequence: range check, hit probability,
        armor resolution, and optionally radiator hit resolution if a thermal
        system is provided.

        Args:
            weapon: The weapon being fired.
            target_armor: The target ship's armor.
            distance_km: Distance to target in kilometers.
            target_accel_g: Target's evasion acceleration in g's.
            tracking_modifier: Firing platform's tracking capability.
            base_ablation_cm: Base ablation amount in cm.
            thermal_system: Optional thermal system for radiator hit resolution.

        Returns:
            HitResult with complete attack resolution including radiator damage.
        """
        # Check range
        if not weapon.is_in_range(distance_km):
            return HitResult(hit=False)

        # Calculate and check hit probability
        hit_prob = self.calculate_hit_probability(
            weapon, distance_km, target_accel_g, tracking_modifier
        )

        if self.rng.random() > hit_prob:
            return HitResult(hit=False)

        # Resolve the hit (including potential radiator damage)
        return self.resolve_hit(
            weapon,
            target_armor,
            base_ablation_cm=base_ablation_cm,
            thermal_system=thermal_system
        )


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


def create_weapon_from_fleet_data(fleet_data: dict, weapon_type: str) -> Weapon:
    """
    Create a Weapon instance from fleet data.

    Args:
        fleet_data: The loaded fleet data dictionary.
        weapon_type: The weapon type identifier.

    Returns:
        A configured Weapon instance.

    Raises:
        KeyError: If the weapon type is not found in fleet data.
    """
    weapon_types = fleet_data.get("weapon_types", {})
    if weapon_type not in weapon_types:
        raise KeyError(f"Weapon type '{weapon_type}' not found in fleet data")

    return Weapon.from_json(weapon_type, weapon_types[weapon_type])


def create_ship_armor_from_fleet_data(fleet_data: dict, ship_type: str) -> ShipArmor:
    """
    Create a ShipArmor instance from fleet data for a specific ship type.

    Args:
        fleet_data: The loaded fleet data dictionary.
        ship_type: The ship type identifier (e.g., 'destroyer', 'cruiser').

    Returns:
        A configured ShipArmor instance.

    Raises:
        KeyError: If the ship type is not found in fleet data.
    """
    ships = fleet_data.get("ships", {})
    if ship_type not in ships:
        raise KeyError(f"Ship type '{ship_type}' not found in fleet data")

    armor_data = ships[ship_type].get("armor", {})
    return ShipArmor.from_json(armor_data)


# Convenience function for quick combat simulations
def simulate_combat_exchange(
    attacker_weapon_type: str,
    defender_ship_type: str,
    distance_km: float,
    fleet_data_path: str | Path = "data/fleet_ships.json",
    num_shots: int = 1,
    seed: Optional[int] = None
) -> list[HitResult]:
    """
    Simulate a combat exchange between a weapon and a ship.

    Args:
        attacker_weapon_type: Type of weapon firing.
        defender_ship_type: Type of ship being targeted.
        distance_km: Combat distance in kilometers.
        fleet_data_path: Path to fleet data JSON file.
        num_shots: Number of shots to simulate.
        seed: Random seed for reproducibility.

    Returns:
        List of HitResult for each shot.
    """
    fleet_data = load_fleet_data(fleet_data_path)
    weapon = create_weapon_from_fleet_data(fleet_data, attacker_weapon_type)
    ship_armor = create_ship_armor_from_fleet_data(fleet_data, defender_ship_type)

    rng = random.Random(seed) if seed is not None else random.Random()
    resolver = CombatResolver(rng=rng)

    results = []
    for _ in range(num_shots):
        result = resolver.resolve_attack(
            weapon=weapon,
            target_armor=ship_armor,
            distance_km=distance_km,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    # Example usage and basic validation
    import sys

    # Default to the expected data path
    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"

    if not data_path.exists():
        print(f"Fleet data not found at {data_path}")
        sys.exit(1)

    print("AI Commanders Combat System")
    print("=" * 40)

    # Load fleet data
    fleet_data = load_fleet_data(data_path)

    # Create example weapon
    spinal_coiler = create_weapon_from_fleet_data(fleet_data, "spinal_coiler_mk3")
    print(f"\nWeapon: {spinal_coiler.name}")
    print(f"  Energy: {spinal_coiler.kinetic_energy_gj} GJ")
    print(f"  Range: {spinal_coiler.range_km} km")
    print(f"  Flat Chipping: {spinal_coiler.flat_chipping}")

    # Create example ship armor
    destroyer_armor = create_ship_armor_from_fleet_data(fleet_data, "destroyer")
    print(f"\nDestroyer Armor Sections:")
    for loc, armor in destroyer_armor.sections.items():
        print(f"  {loc.value}: {armor.thickness_cm:.1f} cm ({armor.protection_percent:.1f}% protection)")

    # Simulate combat
    print("\nSimulating 5 shots at 500 km:")
    results = simulate_combat_exchange(
        attacker_weapon_type="spinal_coiler_mk3",
        defender_ship_type="destroyer",
        distance_km=500,
        fleet_data_path=data_path,
        num_shots=5,
        seed=42
    )

    for i, result in enumerate(results, 1):
        print(f"  Shot {i}: {result}")
